"""Child process that drives the robot through ``hackjunction2026_intel.run_model`` and reports events.

    python -m operator_gui.runner --events-fd N (--export PATH | --dry-run) cycles
    python -m operator_gui.runner --events-fd N (--export PATH | --dry-run) action "Move any brick to the center" [--duration S]
    python -m operator_gui.runner --events-fd N (--export PATH | --dry-run) reset [--home FILE]

``cycles`` builds one ``PolicyRunner`` and calls ``run_model.run_cycle`` again and again until it
receives SIGUSR1 (finish the current cycle, then exit) or a cycle fails. ``action`` runs one
free-text instruction with ``run_model.run_task``; ``reset`` moves the arm to its saved home pose
with ``run_model.reset_robot``. SIGINT stops at once, the way run_model handles
Ctrl-C: the runtime is interrupted and the arm is told to hold its measured position.

The workflow is only observed:

* a logging handler forwards log records and spots the cycle's task instructions in them to
  report which stage is running;
* ``observer.install`` swaps in an InspectionStation subclass that publishes camera frames,
  scored views and the decision. ``--dry-run`` uses the stand-ins in ``dryrun.py`` instead.

Publishing never raises into the workflow: if the dashboard goes away, the robot code runs on unaffected.
"""

import argparse
import logging
import os
import signal
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hackjunction2026_intel import run_model

from . import events

logger = logging.getLogger(__name__)

FINISH_SIGNAL = signal.SIGUSR1  # end the auto-cycle loop once the current cycle is done


class Reporter:
    """Thread-safe event writer. Swallows write errors: observing must never break the robot code."""

    def __init__(self, fd: int):
        self._file = os.fdopen(fd, "w", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()
        self._stage: str | None = None

    def send(self, kind: str, **fields: Any) -> None:
        line = events.encode(kind, **fields)
        with self._lock:
            try:
                self._file.write(line)
                self._file.flush()
            except (OSError, ValueError):
                pass  # dashboard closed its end of the pipe

    def stage(self, stage: str) -> None:
        """Report ``stage`` when it begins (once, however many times it is called in a row)."""
        with self._lock:
            if stage == self._stage:
                return
            self._stage = stage
        self.send(events.STAGE, stage=stage)


class EventLogHandler(logging.Handler):
    """Forwards log records; with ``stages``, a record naming one of the cycle's tasks starts that stage."""

    def __init__(self, reporter: Reporter, stages: bool):
        super().__init__(level=logging.INFO)
        self._reporter = reporter
        self._stages = {
            run_model.PRESENT_TASK: events.PRESENT,
            run_model.GOOD_TASK: events.PLACE_GOOD,
            run_model.BAD_TASK: events.PLACE_BAD,
        } if stages else {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            args = record.args if isinstance(record.args, tuple) else ()
            for arg in args:
                if isinstance(arg, str) and arg in self._stages:
                    self._reporter.stage(self._stages[arg])
                    break
            self._reporter.send(events.LOG, level=record.levelname, message=record.getMessage())
        except Exception:
            self.handleError(record)


@contextmanager
def policy_runner(export: Path | None, reporter: Reporter) -> Iterator[Any]:
    """A run_model.PolicyRunner for ``export`` with the inspection station observed, or dry-run stand-ins."""
    if export is None:
        from .dryrun import DryRunner, install as install_dry_station

        install_dry_station(reporter)
        yield DryRunner()
        return
    try:
        from .observer import install
    except ImportError as error:
        logger.warning("Live inspection view unavailable: %s", error)
    else:
        install(reporter)
    with run_model.open_export(export) as export_root:
        yield run_model.PolicyRunner(export_root)


def run_cycles(runner: Any, finish: threading.Event, once: bool = False) -> int:
    """Run sorting cycles until ``finish`` is set between two cycles; returns how many ran.

    With ``once`` a single cycle runs and the loop ends, so one brick is sorted per press.
    """
    count = 0
    while not finish.is_set():
        count += 1
        logger.info("Cycle %s" if once else "Auto cycle %s", count)
        run_model.run_cycle(runner)
        if once:
            break
    return count


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--events-fd", type=int, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--export", type=Path, help="Studio OpenVINO runtime export (.zip or folder)")
    source.add_argument("--dry-run", action="store_true", help="No robot, camera or model")
    modes = parser.add_subparsers(dest="mode", required=True)
    cycles = modes.add_parser("cycles", help="Sort bricks until SIGUSR1")
    cycles.add_argument("--once", action="store_true", help="Run a single cycle, then exit")
    action = modes.add_parser("action", help="Run one free-text instruction")
    action.add_argument("task")
    action.add_argument("--duration", type=float, default=run_model.DEFAULT_DURATION_S)
    reset = modes.add_parser("reset", help="Move the arm to its saved home pose")
    reset.add_argument("--home", type=Path, default=run_model.DEFAULT_HOME_FILE)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    reporter = Reporter(args.events_fd)
    # An action's text may equal a cycle task: never count it as a cycle stage.
    logging.getLogger().addHandler(EventLogHandler(reporter, stages=args.mode == "cycles"))
    finish = threading.Event()
    signal.signal(FINISH_SIGNAL, lambda *_: finish.set())

    try:
        with policy_runner(args.export, reporter) as runner:
            if args.mode == "action":
                reporter.stage(events.ACTION)
                run_model.run_task(runner, args.task, duration_s=args.duration)
                message = "Action finished"
            elif args.mode == "reset":
                reporter.stage(events.RESET)
                run_model.reset_robot(runner, home_file=args.home)
                message = "Arm is at its home pose"
            else:
                count = run_cycles(runner, finish, once=args.once)
                if args.once:
                    message = "Cycle finished"
                else:
                    message = f"Auto cycle ended after {count} cycle{'s' if count != 1 else ''}"
    except KeyboardInterrupt:
        logger.warning("Stopped by the operator")
        reporter.send(events.RESULT, outcome=events.STOPPED, message="Stopped by the operator")
        return 130
    except Exception as error:
        logger.exception("Run failed")
        reporter.send(events.RESULT, outcome=events.ERROR, message=str(error) or type(error).__name__)
        return 1
    reporter.send(events.RESULT, outcome=events.DONE, message=message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

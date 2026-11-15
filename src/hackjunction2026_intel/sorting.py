"""Demonstration of one pick, inspection, and sorting cycle."""

import argparse
import logging
import threading
import time
import zipfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing, contextmanager
from math import isfinite
from pathlib import Path
from random import random
from tempfile import TemporaryDirectory
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# The single multi-task π0.5 model always runs on OpenVINO, ignoring the export's device.
BACKEND = "openvino"
# Pinned to the iGPU: it is the fastest local device for a ~3B VLA, and it keeps the policy off
# the NPU, where the anomaly model runs, and off the CPU, which drives the control loop and the
# camera decode. Use "AUTO" to let OpenVINO choose, or "CPU"/"NPU" to pin another device.
OPENVINO_DEVICE = "GPU"
OPENVINO_DEVICES = frozenset({"AUTO", "CPU", "GPU", "NPU"})

# Task instructions match the relabelled demonstration dataset.
PRESENT_TASK = "Move any brick to the center"
GOOD_TASK = "Move the brick to the yellow dish"
BAD_TASK = "Move the brick to the orange dish"

# Presentation ends once inspection has been anything but EMPTY (a brick is in the inspection
# area) continuously for PRESENT_CONFIRM_S; it aborts if that has not happened within PRESENT_DURATION_S.
PRESENT_CONFIRM_S = 3.0
PRESENCE_POLL_S = 0.1

# TODO: Set supervised, validated durations in seconds; zero disables execution.
PRESENT_DURATION_S = 30.0
GOOD_DURATION_S = 15.0
BAD_DURATION_S = 15.0


class StopSignal(Protocol):
    def is_set(self) -> bool: ...


class PresenceWatch:
    """Polls ``is_present`` in a thread and stops a policy run once it stays True for ``hold_s``.

    The sorting cycle passes "inspection is not EMPTY". Pass the watch as the runtime's ``stop_event``.
    A polling error also stops the run, and is kept in ``error``.
    ``after_first_tick`` defers polling until the runtime has completed its first control tick.
    """

    def __init__(
        self,
        is_present: Callable[[], bool],
        hold_s: float,
        poll_s: float = PRESENCE_POLL_S,
        clock: Callable[[], float] = time.monotonic,
        *,
        after_first_tick: bool = False,
    ):
        self._is_present = is_present
        self._hold_s = hold_s
        self._poll_s = poll_s
        self._clock = clock
        self._after_first_tick = after_first_tick
        self._first_runtime_poll = True
        self._started = False
        self._halt = threading.Event()
        self.confirmed = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name="presence-watch", daemon=True)

    def is_set(self) -> bool:
        if self._after_first_tick and not self._started and not self._halt.is_set():
            # RobotRuntime polls stop_event once before each control tick.
            if self._first_runtime_poll:
                self._first_runtime_poll = False
                return False
            self._thread.start()
            self._started = True
            logger.info("First presentation control tick completed; starting inspection presence checks")
        return self.confirmed.is_set() or self.error is not None

    def _run(self) -> None:
        since: float | None = None
        try:
            while not self._halt.is_set():
                if self._is_present():
                    now = self._clock()
                    since = now if since is None else since
                    if now - since >= self._hold_s:
                        self.confirmed.set()
                        return
                else:
                    since = None
                self._halt.wait(self._poll_s)
        except BaseException as error:
            self.error = error

    def __enter__(self) -> "PresenceWatch":
        if not self._after_first_tick:
            self._thread.start()
            self._started = True
        return self

    def __exit__(self, *exc: object) -> None:
        self._halt.set()
        if self._started:
            self._thread.join()


@contextmanager
def open_export(path: Path) -> Iterator[Path]:
    """Yield the root of a Studio runtime export, extracting it first if it is a zip."""
    if path.is_file() and zipfile.is_zipfile(path):
        with TemporaryDirectory(prefix="studio-export-") as extracted, zipfile.ZipFile(path) as archive:
            archive.extractall(extracted)
            root = Path(extracted)
            # Accept archives that wrap everything in one top-level folder.
            entries = list(root.iterdir())
            if not (root / "runtime.yaml").is_file() and len(entries) == 1 and entries[0].is_dir():
                root = entries[0]
            yield root
    elif path.is_dir():
        yield path
    else:
        raise FileNotFoundError(f"Studio runtime export not found (expected a folder or .zip): {path}")


def _validate_export(export_root: Path) -> None:
    if OPENVINO_DEVICE not in OPENVINO_DEVICES:
        raise ValueError(f"OPENVINO_DEVICE must be one of {sorted(OPENVINO_DEVICES)}, got {OPENVINO_DEVICE!r}")
    if not (export_root / "runtime.yaml").is_file():
        raise FileNotFoundError(f"runtime.yaml not found in Studio export: {export_root}")
    if not (export_root / "exports" / BACKEND).is_dir():
        raise FileNotFoundError(f"{BACKEND} weights not found in Studio export: {export_root / 'exports' / BACKEND}")


def _validate_duration(duration_s: float) -> None:
    if not isfinite(duration_s) or duration_s <= 0:
        raise ValueError(f"Set a finite, positive stage duration, got {duration_s}")


def run_policy(export_root: Path, task: str, *, duration_s: float, stop_event: StopSignal | None = None) -> None:
    """Run a stage for ``duration_s``, or until ``stop_event`` is set, which then counts as completion.

    Without ``stop_event`` this is a timed stage that does not claim the manipulation succeeded.
    """
    _validate_duration(duration_s)
    _validate_export(export_root)

    from physicalai.config import Config
    from physicalai.runtime import RobotRuntime

    config = Config.load(export_root / "runtime.yaml")
    if config.class_path != "physicalai.runtime.RobotRuntime":
        raise ValueError("Expected a Studio-exported RobotRuntime configuration")
    source = config.init_args.get("action_source")
    if not isinstance(source, dict) or source.get("class_path") != "physicalai.runtime.PolicySource":
        raise ValueError("Runtime configuration must use PolicySource, not teleoperation")
    source_args = source.get("init_args")
    if not isinstance(source_args, dict):
        raise ValueError("PolicySource configuration is missing init_args")
    model = source_args.get("model")
    if not isinstance(model, dict) or model.get("class_path") != "physicalai.inference.InferenceModel":
        raise ValueError("PolicySource configuration must contain an InferenceModel")
    model_args = model.get("init_args")
    if not isinstance(model_args, dict):
        raise ValueError("InferenceModel configuration is missing init_args")

    # Resolve weights against the export, not the working directory, and pin OpenVINO.
    model_args["export_dir"] = str((export_root / "exports" / BACKEND).resolve())
    model_args["backend"] = BACKEND
    model_args["device"] = OPENVINO_DEVICE
    model_args.pop("policy_name", None)
    source_args["task"] = task
    # A fresh runtime per stage is required: each run() closes its InferenceModel on exit.
    runtime = config.instantiate(expected_type=RobotRuntime)
    if not isinstance(runtime, RobotRuntime):
        raise TypeError("Runtime configuration did not instantiate a RobotRuntime")

    if stop_event is None:
        logger.warning(
            "Running timed demo task %r for %s control seconds; task success is NOT verified",
            task,
            duration_s,
        )
    else:
        logger.info("Running task %r until its completion check passes (timeout %s control seconds)", task, duration_s)
    with runtime:
        try:
            ticks = runtime.run(duration_s=duration_s, stop_event=stop_event, return_to_initial_state=False)
            reason = runtime.last_run_reason
            if reason == "interrupted":
                raise KeyboardInterrupt("Policy stage interrupted; aborting the sorting cycle")
            if stop_event is None and (reason != "duration_elapsed" or ticks <= 0):
                raise RuntimeError(f"Policy stage did not finish its duration: {reason}")
            # Zero ticks is fine here: the completion check may already pass before the first tick.
            if stop_event is not None and reason != "stop_requested":
                raise RuntimeError(f"Policy stage {task!r} did not complete within {duration_s} s: {reason}")
        finally:
            # Cancel any remaining position target before releasing the driver.
            measured = runtime.robot.get_observation()
            runtime.robot.send_action(measured.joint_positions)


def present_block(export_root: Path, station: Any) -> None:
    """Run the presentation policy until inspection has not been EMPTY for PRESENT_CONFIRM_S."""
    if station.inspector.presence is None:
        raise RuntimeError(
            "The anomaly model has no empty-ROI reference, so it never reports EMPTY and presentation "
            "cannot be confirmed (collect empty frames, then `brick-ad prepare` + `brick-ad fit-empty`)"
        )
    with PresenceWatch(lambda: not station.is_empty(), PRESENT_CONFIRM_S, after_first_tick=True) as watch:
        run_policy(export_root, PRESENT_TASK, duration_s=PRESENT_DURATION_S, stop_event=watch)
    if watch.error is not None:
        raise RuntimeError("EMPTY check failed during presentation") from watch.error
    logger.info("Inspection not EMPTY for %s seconds; presentation complete", PRESENT_CONFIRM_S)


def inspect_block(station: Any) -> bool:
    """Return True for defective and False for good; raise when there is no brick or the result is uncertain."""
    from .anomaly_detection import Verdict

    decision = station.inspect()
    logger.info("Inspection: %s", decision.message)
    if decision.verdict is Verdict.DEFECTIVE:
        return True
    if decision.verdict is Verdict.GOOD:
        return False
    raise RuntimeError(f"Inspection inconclusive; not sorting the brick: {decision.message}")


def run_cycle(export_root: Path | None, *, dry_run: bool = False) -> None:
    if not dry_run:
        for duration in (PRESENT_DURATION_S, GOOD_DURATION_S, BAD_DURATION_S):
            _validate_duration(duration)
        if export_root is None:
            raise ValueError("A Studio runtime export is required for robot execution")
    if export_root is not None:
        _validate_export(export_root)

    if not dry_run:
        from .run_model import PolicyRunner, run_cycle as run_model_cycle

        if export_root is None:
            raise ValueError("A Studio runtime export is required for robot execution")
        with closing(PolicyRunner(export_root)) as runner:
            run_model_cycle(runner, inconclusive_as_defective=True)
        return

    def run_stage(task: str, duration_s: float) -> None:
        logger.info("DRY RUN: task %r; configured duration %s seconds; no robot commands", task, duration_s)

    logger.warning(
        "Dry run only: no SDK loading, hardware connections, or inference; "
        "runtime configuration and stage durations are not validated"
    )
    logger.info("Moving the block to the inspection stage")
    run_stage(PRESENT_TASK, PRESENT_DURATION_S)
    logger.warning("Dry-run inspection uses randomness, not the anomaly model")
    is_defective = random() < 0.5

    if is_defective:
        logger.info("Inspection decision: defective; placing in the bad dish")
        run_stage(BAD_TASK, BAD_DURATION_S)
    else:
        logger.info("Inspection decision: good; placing in the good dish")
        run_stage(GOOD_TASK, GOOD_DURATION_S)

    logger.info("Dry-run routing finished; hardware and model readiness remain unverified")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        type=Path,
        help="Studio OpenVINO runtime export: the downloaded .zip or its extracted folder",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log task routing without loading the SDK or connecting hardware; no export required",
    )
    args = parser.parse_args(argv)
    if args.export is None:
        if not args.dry_run:
            parser.error("--export is required unless --dry-run is used")
        run_cycle(None, dry_run=True)
    else:
        with open_export(args.export) as export_root:
            run_cycle(export_root, dry_run=args.dry_run)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

"""Run the exported policy: one arbitrary task, or the full inspect-and-sort cycle.

Both modes go through one PolicyRunner, which validates the export and the Studio runtime
configuration once, up front, and then builds a runtime per stage.

    python -m hackjunction2026_intel.run_model --export ./export task "Move any brick to the center"
    python -m hackjunction2026_intel.run_model --export ./export cycle
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .sorting import (
    BACKEND,
    BAD_DURATION_S,
    BAD_TASK,
    GOOD_DURATION_S,
    GOOD_TASK,
    OPENVINO_DEVICE,
    PRESENT_CONFIRM_S,
    PRESENT_DURATION_S,
    PRESENT_TASK,
    PresenceWatch,
    StopSignal,
    _validate_duration,
    _validate_export,
    open_export,
)

logger = logging.getLogger(__name__)

DEFAULT_DURATION_S = 15.0


class PolicyRunner:
    """Holds a validated Studio runtime configuration and runs stages against it.

    The export and the configuration are checked once in ``__init__``, so a bad export or a
    teleoperation configuration fails before any motion. Each ``run_stage`` call still builds a
    fresh runtime: ``run()`` closes its InferenceModel on exit, so a runtime serves one stage,
    and the task is written into the configuration before instantiation rather than passed to
    ``run()``, so changing task requires rebuilding anyway.
    """

    def __init__(self, export_root: Path):
        _validate_export(export_root)
        self.export_root = export_root

        from physicalai.config import Config

        self._config = Config.load(export_root / "runtime.yaml")
        if self._config.class_path != "physicalai.runtime.RobotRuntime":
            raise ValueError("Expected a Studio-exported RobotRuntime configuration")
        source = self._config.init_args.get("action_source")
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
        self._source_args = source_args

        # Resolve weights against the export, not the working directory, and pin OpenVINO.
        model_args["export_dir"] = str((export_root / "exports" / BACKEND).resolve())
        model_args["backend"] = BACKEND
        model_args["device"] = OPENVINO_DEVICE
        model_args.pop("policy_name", None)
        logger.info("Studio runtime configuration validated; %s on %s", BACKEND, OPENVINO_DEVICE)

    def run_stage(self, task: str, *, duration_s: float, stop_event: StopSignal | None = None) -> None:
        """Run ``task`` for ``duration_s``, or until ``stop_event`` is set, which counts as completion.

        Without ``stop_event`` this is a timed stage that does not claim the manipulation succeeded.
        """
        _validate_duration(duration_s)

        from physicalai.runtime import RobotRuntime

        self._source_args["task"] = task
        runtime = self._config.instantiate(expected_type=RobotRuntime)
        if not isinstance(runtime, RobotRuntime):
            raise TypeError("Runtime configuration did not instantiate a RobotRuntime")

        if stop_event is None:
            logger.warning(
                "Running timed task %r for %s control seconds; task success is NOT verified", task, duration_s
            )
        else:
            logger.info(
                "Running task %r until its completion check passes (timeout %s control seconds)", task, duration_s
            )
        with runtime:
            try:
                ticks = runtime.run(duration_s=duration_s, stop_event=stop_event, return_to_initial_state=False)
                reason = runtime.last_run_reason
                if reason == "interrupted":
                    raise KeyboardInterrupt("Policy stage interrupted; aborting")
                if stop_event is None and (reason != "duration_elapsed" or ticks <= 0):
                    raise RuntimeError(f"Policy stage did not finish its duration: {reason}")
                # Zero ticks is fine here: the completion check may already pass before the first tick.
                if stop_event is not None and reason != "stop_requested":
                    raise RuntimeError(f"Policy stage {task!r} did not complete within {duration_s} s: {reason}")
            finally:
                # Cancel any remaining position target before releasing the driver.
                measured = runtime.robot.get_observation()
                runtime.robot.send_action(measured.joint_positions)

    def present(self, station: Any) -> None:
        """Run the presentation task until inspection has not been EMPTY for PRESENT_CONFIRM_S."""
        if station.inspector.presence is None:
            raise RuntimeError(
                "The anomaly model has no empty-ROI reference, so it never reports EMPTY and presentation "
                "cannot be confirmed (collect empty frames, then `brick-ad prepare` + `brick-ad fit-empty`)"
            )
        with PresenceWatch(lambda: not station.is_empty(), PRESENT_CONFIRM_S) as watch:
            self.run_stage(PRESENT_TASK, duration_s=PRESENT_DURATION_S, stop_event=watch)
        if watch.error is not None:
            raise RuntimeError("EMPTY check failed during presentation") from watch.error
        logger.info("Inspection not EMPTY for %s seconds; presentation complete", PRESENT_CONFIRM_S)


def run_task(runner: PolicyRunner, task: str, *, duration_s: float = DEFAULT_DURATION_S, repeat: int = 1) -> None:
    """Run one arbitrary task ``repeat`` times, each for ``duration_s`` control seconds."""
    if repeat < 1:
        raise ValueError(f"Repeat count must be at least 1, got {repeat}")
    for attempt in range(1, repeat + 1):
        if repeat > 1:
            logger.info("Run %s of %s", attempt, repeat)
        runner.run_stage(task, duration_s=duration_s)
    logger.warning("Task %r ran for its configured duration; its outcome was NOT verified", task)


def run_cycle(runner: PolicyRunner) -> bool:
    """Present a brick, inspect it, and place it in the matching dish. Returns True if defective."""
    for duration in (PRESENT_DURATION_S, GOOD_DURATION_S, BAD_DURATION_S):
        _validate_duration(duration)

    from .anomaly_detection import InspectionStation, Verdict

    # Opened before any motion so a missing model or camera aborts the cycle up front.
    with InspectionStation() as station:
        logger.info("Moving the block to the inspection stage")
        runner.present(station)
        decision = station.inspect()
        logger.info("Inspection: %s", decision.message)
        if decision.verdict is Verdict.DEFECTIVE:
            is_defective = True
        elif decision.verdict is Verdict.GOOD:
            is_defective = False
        else:
            raise RuntimeError(f"Inspection inconclusive; not sorting the brick: {decision.message}")

    if is_defective:
        logger.info("Inspection decision: defective; placing in the bad dish")
        runner.run_stage(BAD_TASK, duration_s=BAD_DURATION_S)
    else:
        logger.info("Inspection decision: good; placing in the good dish")
        runner.run_stage(GOOD_TASK, duration_s=GOOD_DURATION_S)

    logger.warning("Cycle finished; the placement stage is timed and its outcome was not verified")
    return is_defective


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--export",
        type=Path,
        required=True,
        help="Studio OpenVINO runtime export: the downloaded .zip or its extracted folder",
    )
    modes = parser.add_subparsers(dest="mode", required=True)

    one = modes.add_parser("task", help="Run a single arbitrary task for a fixed duration")
    one.add_argument("task", help='Task instruction, e.g. "Move the brick to the yellow dish"')
    one.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"Control seconds to run the task for (default: {DEFAULT_DURATION_S})",
    )
    one.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the same task this many times (default: 1)",
    )

    cycle = modes.add_parser("cycle", help="Present, inspect, and sort into the matching dish")
    cycle.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run this many full sorting cycles (default: 1)",
    )

    args = parser.parse_args(argv)
    with open_export(args.export) as export_root:
        runner = PolicyRunner(export_root)
        if args.mode == "task":
            run_task(runner, args.task, duration_s=args.duration, repeat=args.repeat)
        else:
            if args.repeat < 1:
                raise ValueError(f"Repeat count must be at least 1, got {args.repeat}")
            for attempt in range(1, args.repeat + 1):
                if args.repeat > 1:
                    logger.info("Cycle %s of %s", attempt, args.repeat)
                run_cycle(runner)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

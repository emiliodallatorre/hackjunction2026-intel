"""Run the exported policy: one arbitrary task, or the full inspect-and-sort cycle.

Both modes go through one PolicyRunner, which validates the export and the Studio runtime
configuration once, up front, and then builds a runtime per stage.

    python -m hackjunction2026_intel.run_model --export ./export task "Move any brick to the center"
    python -m hackjunction2026_intel.run_model --export ./export cycle
"""

import argparse
import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from math import isfinite
from pathlib import Path
from typing import Any, Callable

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

# Reset moves the arm back to the pose it held when PolicyRunner was built, interpolating over
# RESET_DURATION_S at RESET_FPS. These mirror physicalai's own return_to_initial_state move.
RESET_DURATION_S = 2.5
RESET_FPS = 30.0
# Skip the move when every joint is already within this fraction of the pose's magnitude.
RESET_SKIP_TOLERANCE = 0.1
# Abort and hold if a joint lags this fraction of the largest joint travel (a likely collision).
RESET_MAX_TRACKING_ERROR = 0.15
# SO-101 gripper positions are normalized to [0, 100], including in degree mode.
GRIPPER_RELEASE_DELTA = 30.0
GRIPPER_RELEASE_HOLD_S = 0.5
GRIPPER_POLL_S = 0.1
# Where `reset --save-home` stores the pose, so a later process can reset to it.
DEFAULT_HOME_FILE = Path("home_pose.json")


class PolicyRunner:
    """Holds a validated Studio runtime configuration and runs stages against it.

    The export and the configuration are checked once in ``__init__``, so a bad export or a
    teleoperation configuration fails before any motion. Each ``run_stage`` call still builds a
    fresh runtime: ``run()`` closes its InferenceModel on exit, so a runtime serves one stage,
    and the task is written into the configuration before instantiation rather than passed to
    ``run()``, so changing task requires rebuilding anyway.

    ``home_pose`` is read from the robot at construction and is what ``reset`` returns to.
    """

    def __init__(self, export_root: Path):
        _validate_export(export_root)
        self.export_root = export_root
        self._load_config()
        self.home_pose: Any | None = None
        logger.info("Studio runtime configuration validated; %s on %s", BACKEND, OPENVINO_DEVICE)

    def _load_config(self, task: str | None = None) -> Any:
        """Load and validate ``runtime.yaml`` afresh.

        Typed instantiation converts nested mappings into Namespaces in place, so a loaded
        configuration cannot be instantiated a second time.
        """
        from physicalai.config import Config

        config = Config.load(self.export_root / "runtime.yaml")
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
        model_args["export_dir"] = str((self.export_root / "exports" / BACKEND).resolve())
        model_args["backend"] = BACKEND
        model_args["device"] = OPENVINO_DEVICE
        model_args.pop("policy_name", None)
        if not isinstance(config.init_args.get("robot"), dict):
            raise ValueError("Runtime configuration is missing a robot")
        if task is not None:
            source_args["task"] = task
        return config

    @contextmanager
    def _connected_robot(self) -> Iterator[Any]:
        """Connect only the robot from the configuration, without loading the policy.

        InferenceModel loads its weights in its constructor, so instantiating the whole runtime
        to move a joint would read the model off disk for nothing. Only the robot is built here.
        """
        from physicalai.config import instantiate
        from physicalai.robot import connect

        robot = instantiate(self._load_config().init_args["robot"])
        with connect(robot) as connected:
            yield connected

    def capture_home_pose(self) -> Any:
        """Read and remember the current pose as the one ``reset`` returns to."""
        with self._connected_robot() as robot:
            self.home_pose = robot.get_observation().joint_positions
            joint_names = list(robot.joint_names)
        logger.info("Home pose captured: %s", dict(zip(joint_names, self.home_pose.tolist(), strict=False)))
        return self.home_pose

    def reset(self, *, pose: Any | None = None) -> None:
        """Interpolate the arm back to ``pose`` (default: ``home_pose``) without running the policy.

        Aborts holding the measured pose if a joint lags badly, which usually means a collision.
        Raises RuntimeError if no pose has been captured yet.
        """
        import numpy as np

        target = self.home_pose if pose is None else pose
        if target is None:
            raise RuntimeError("No home pose captured yet; call capture_home_pose() first")
        target = np.asarray(target, dtype=float)

        steps = max(int(RESET_DURATION_S * RESET_FPS), 1)
        goal_time = RESET_DURATION_S / steps
        with self._connected_robot() as robot:
            current = robot.get_observation().joint_positions
            if target.shape != current.shape:
                raise ValueError(f"Pose has {target.shape} joints, robot reports {current.shape}")
            travel = float(np.max(np.abs(target - current)))
            relative = float(np.linalg.norm(target - current) / max(np.linalg.norm(target), 1e-8))
            if relative <= RESET_SKIP_TOLERANCE:
                robot.send_action(current)
                logger.info("Already within %.0f%% of the home pose; nothing to do", relative * 100)
                return
            logger.info("Resetting the arm to its home pose over %.1f s", RESET_DURATION_S)
            for step in range(1, steps + 1):
                started = time.perf_counter()
                fraction = step / steps
                command = current * (1.0 - fraction) + target * fraction
                robot.send_action(command, goal_time=goal_time)
                measured = robot.get_observation().joint_positions
                error = np.abs(measured - command) / max(travel, 1e-8)
                worst = int(np.argmax(error))
                if error[worst] > RESET_MAX_TRACKING_ERROR:
                    robot.send_action(measured)
                    raise RuntimeError(
                        f"Reset aborted at step {step}/{steps}: joint "
                        f"{robot.joint_names[worst]!r} is off by {error[worst] * 100:.0f}% of the move "
                        "(possible collision); holding the measured pose"
                    )
                remaining = goal_time - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
        logger.info("Reset complete; the arm is at its home pose")

    def run_stage(
        self,
        task: str,
        *,
        duration_s: float,
        stop_event: StopSignal | None = None,
        stop_event_factory: Callable[[Any], StopSignal] | None = None,
    ) -> None:
        """Run ``task`` for ``duration_s``, or until ``stop_event`` is set, which counts as completion.

        Without either stop argument this is a timed stage that does not claim the manipulation succeeded.
        """
        _validate_duration(duration_s)
        if stop_event is not None and stop_event_factory is not None:
            raise ValueError("Pass either stop_event or stop_event_factory, not both")

        from physicalai.runtime import RobotRuntime

        runtime = self._load_config(task).instantiate(expected_type=RobotRuntime)
        if not isinstance(runtime, RobotRuntime):
            raise TypeError("Runtime configuration did not instantiate a RobotRuntime")

        if stop_event is None and stop_event_factory is None:
            logger.warning(
                "Running timed task %r for %s control seconds; task success is NOT verified", task, duration_s
            )
        else:
            logger.info(
                "Running task %r until its completion check passes (timeout %s control seconds)", task, duration_s
            )
        with runtime:
            try:
                if stop_event_factory is not None:
                    stop_event = stop_event_factory(runtime.robot)
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


class GripperReleaseWatch:
    """Stops a placement stage after a sustained, substantial SO-101 gripper movement."""

    def __init__(
        self,
        robot: Any,
        *,
        delta: float = GRIPPER_RELEASE_DELTA,
        hold_s: float = GRIPPER_RELEASE_HOLD_S,
        poll_s: float = GRIPPER_POLL_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        if delta <= 0 or not isfinite(delta):
            raise ValueError(f"Gripper release delta must be finite and positive, got {delta}")
        if hold_s < 0 or not isfinite(hold_s):
            raise ValueError(f"Gripper release hold must be finite and nonnegative, got {hold_s}")
        if poll_s <= 0 or not isfinite(poll_s):
            raise ValueError(f"Gripper poll interval must be finite and positive, got {poll_s}")
        joint_names = robot.joint_names
        if "gripper" not in joint_names:
            raise RuntimeError(f"Placement release check requires an SO-101 'gripper' joint, got {joint_names!r}")
        self._robot = robot
        self._index = joint_names.index("gripper")
        self._delta = delta
        self._hold_s = hold_s
        self._poll_s = poll_s
        self._clock = clock
        self._initial = self._position()
        self._next_poll = clock()
        self._release_since: float | None = None
        self.confirmed = False

    def _position(self) -> float:
        positions = self._robot.get_observation().joint_positions
        if len(positions) <= self._index:
            raise RuntimeError(f"SO-101 observation has no gripper position: expected index {self._index}")
        position = float(positions[self._index])
        if not isfinite(position):
            raise RuntimeError(f"SO-101 gripper position is not finite: {position}")
        return position

    def is_set(self) -> bool:
        if self.confirmed:
            return True
        now = self._clock()
        if now < self._next_poll:
            return False
        self._next_poll = now + self._poll_s
        if abs(self._position() - self._initial) >= self._delta:
            self._release_since = now if self._release_since is None else self._release_since
            if now - self._release_since >= self._hold_s:
                self.confirmed = True
        else:
            self._release_since = None
        return self.confirmed


def run_task(runner: PolicyRunner, task: str, *, duration_s: float = DEFAULT_DURATION_S, repeat: int = 1) -> None:
    """Run one arbitrary task ``repeat`` times, each for ``duration_s`` control seconds."""
    if repeat < 1:
        raise ValueError(f"Repeat count must be at least 1, got {repeat}")
    for attempt in range(1, repeat + 1):
        if repeat > 1:
            logger.info("Run %s of %s", attempt, repeat)
        runner.run_stage(task, duration_s=duration_s)
    logger.warning("Task %r ran for its configured duration; its outcome was NOT verified", task)


def save_home_pose(runner: PolicyRunner, path: Path = DEFAULT_HOME_FILE) -> None:
    """Read the arm's current pose and write it to ``path`` as the home pose to reset to later."""
    import json

    pose = runner.capture_home_pose()
    path.write_text(json.dumps({"joint_positions": [float(value) for value in pose]}, indent=2))
    logger.info("Home pose saved to %s", path)


def load_home_pose(path: Path = DEFAULT_HOME_FILE) -> Any:
    """Read a pose written by ``save_home_pose``."""
    import json

    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(
            f"No home pose at {path}; teach one first with "
            f"`run_model --export ... reset --save-home` while the arm is where you want home to be"
        )
    stored = json.loads(path.read_text())
    positions = stored.get("joint_positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError(f"{path} does not contain a joint_positions list")
    return np.asarray(positions, dtype=float)


def reset_robot(runner: PolicyRunner, *, home_file: Path | None = DEFAULT_HOME_FILE) -> None:
    """Move the arm to its home pose: the runner's captured pose, or the one saved in ``home_file``.

    The GUI calls this. An in-process runner that has already run a stage has a captured pose;
    a fresh process has none, so the saved file is used.
    """
    if runner.home_pose is not None:
        runner.reset()
        return
    if home_file is None:
        raise RuntimeError("No home pose captured and no home file given")
    runner.reset(pose=load_home_pose(home_file))


def run_cycle(runner: PolicyRunner) -> bool:
    """Present, inspect, place, and reset. Returns True if defective."""
    for duration in (PRESENT_DURATION_S, GOOD_DURATION_S, BAD_DURATION_S):
        _validate_duration(duration)

    from .anomaly_detection import InspectionStation, Verdict

    # Opened before any motion so a missing model or camera aborts the cycle up front.
    with InspectionStation() as station:
        if runner.home_pose is None:
            runner.capture_home_pose()
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
        task, duration_s = BAD_TASK, BAD_DURATION_S
    else:
        logger.info("Inspection decision: good; placing in the good dish")
        task, duration_s = GOOD_TASK, GOOD_DURATION_S

    runner.run_stage(task, duration_s=duration_s, stop_event_factory=GripperReleaseWatch)
    runner.reset()
    logger.warning(
        "Cycle finished; gripper release was detected and the arm returned to its captured home pose. "
        "Dish placement itself was not visually verified"
    )
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

    reset = modes.add_parser("reset", help="Move the arm to its saved home pose")
    reset.add_argument(
        "--save-home",
        action="store_true",
        help="Store the arm's current pose as home instead of moving, then exit",
    )
    reset.add_argument(
        "--home",
        type=Path,
        default=DEFAULT_HOME_FILE,
        help=f"File holding the home pose (default: {DEFAULT_HOME_FILE})",
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
        if args.mode == "reset":
            if args.save_home:
                save_home_pose(runner, args.home)
            else:
                reset_robot(runner, home_file=args.home)
            return
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

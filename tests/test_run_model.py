import enum
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from hackjunction2026_intel import run_model


class Verdict(enum.Enum):
    GOOD = "good"
    DEFECTIVE = "defective"
    EMPTY = "empty"
    ROTATE = "rotate"


class ReleaseWatchTests(unittest.TestCase):
    def test_opening_to_pick_up_does_not_confirm_until_grasp_and_release(self) -> None:
        now = [0.0]
        position = [25.0]
        robot = SimpleNamespace(
            joint_names=["shoulder_pan", "gripper"],
            get_observation=lambda: SimpleNamespace(joint_positions=[0.0, position[0]]),
        )
        watch = run_model.GripperReleaseWatch(robot, clock=lambda: now[0])

        for timestamp, value in ((0.0, 25.0), (0.5, 60.0), (1.0, 60.0), (1.5, 60.0)):
            now[0], position[0] = timestamp, value
            self.assertFalse(watch.is_set())
        for timestamp, value in ((2.0, 30.0), (2.5, 28.0), (3.0, 42.9), (3.5, 43.0), (3.75, 43.0)):
            now[0], position[0] = timestamp, value
            self.assertFalse(watch.is_set())
        now[0] = 4.0
        self.assertTrue(watch.is_set())
        self.assertTrue(watch.is_set())

    def test_requires_named_gripper_and_finite_observation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "SO-101 'gripper' joint"):
            run_model.GripperReleaseWatch(SimpleNamespace(joint_names=["finger"], get_observation=Mock()))

        robot = SimpleNamespace(
            joint_names=["gripper"],
            get_observation=lambda: SimpleNamespace(joint_positions=[float("nan")]),
        )
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            run_model.GripperReleaseWatch(robot)


class CycleTests(unittest.TestCase):
    def test_cycle_captures_home_stops_after_release_and_resets(self) -> None:
        events: list[str] = []
        station = SimpleNamespace(
            __enter__=lambda self: self,
            __exit__=lambda *args: None,
            inspect=lambda: SimpleNamespace(verdict=Verdict.GOOD, message="good"),
        )

        class InspectionStation:
            def __enter__(self):
                return station

            def __exit__(self, *args):
                pass

        anomaly = ModuleType("hackjunction2026_intel.anomaly_detection")
        anomaly.InspectionStation = InspectionStation
        anomaly.Verdict = Verdict
        runner = SimpleNamespace(home_pose=None)

        def capture_home_pose():
            events.append("capture home")
            runner.home_pose = [0.0]

        def present(_station):
            events.append("present")

        def run_stage(task, *, duration_s, stop_event_factory):
            events.append("place")
            self.assertEqual(task, run_model.GOOD_TASK)
            self.assertEqual(duration_s, run_model.GOOD_DURATION_S)
            self.assertIs(stop_event_factory, run_model.GripperReleaseWatch)

        def reset():
            events.append("reset")

        runner.capture_home_pose = capture_home_pose
        runner.present = present
        runner.run_stage = run_stage
        runner.reset = reset

        with patch.dict(sys.modules, {"hackjunction2026_intel.anomaly_detection": anomaly}):
            is_defective = run_model.run_cycle(runner)

        self.assertFalse(is_defective)
        self.assertEqual(events, ["capture home", "present", "place", "reset"])


    def _run_cycle(self, runner: Mock, station: Mock, *, inconclusive_as_defective: bool = False) -> bool:
        inspection = Mock(__enter__=Mock(return_value=station), __exit__=Mock(return_value=None))
        anomaly = ModuleType("hackjunction2026_intel.anomaly_detection")
        anomaly.InspectionStation = Mock(return_value=inspection)
        anomaly.Verdict = Verdict
        with patch.dict(sys.modules, {"hackjunction2026_intel.anomaly_detection": anomaly}):
            return run_model.run_cycle(runner, inconclusive_as_defective=inconclusive_as_defective)

    def test_each_outcome_runs_its_placement(self) -> None:
        for verdict, task, duration in (
            (Verdict.DEFECTIVE, run_model.BAD_TASK, run_model.BAD_DURATION_S),
            (Verdict.GOOD, run_model.GOOD_TASK, run_model.GOOD_DURATION_S),
        ):
            with self.subTest(verdict=verdict):
                runner = Mock(home_pose=[0.0])
                station = Mock()
                station.inspect.return_value = SimpleNamespace(verdict=verdict, message=verdict.value)
                self.assertEqual(self._run_cycle(runner, station), verdict is Verdict.DEFECTIVE)
                runner.run_stage.assert_called_once_with(
                    task, duration_s=duration, stop_event_factory=run_model.GripperReleaseWatch
                )

    def test_presentation_failure_prevents_inspection_or_placement(self) -> None:
        runner = Mock(home_pose=[0.0])
        runner.present.side_effect = RuntimeError("aborted")
        station = Mock()
        with self.assertRaisesRegex(RuntimeError, "aborted"):
            self._run_cycle(runner, station)
        station.inspect.assert_not_called()
        runner.run_stage.assert_not_called()

    def test_inconclusive_inspection_prevents_placement(self) -> None:
        for verdict in (Verdict.EMPTY, Verdict.ROTATE):
            with self.subTest(verdict=verdict):
                runner = Mock(home_pose=[0.0])
                station = Mock()
                station.inspect.return_value = SimpleNamespace(verdict=verdict, message=verdict.value)
                with self.assertRaisesRegex(RuntimeError, "inconclusive"):
                    self._run_cycle(runner, station)
                runner.run_stage.assert_not_called()

    def test_inconclusive_inspection_can_default_to_defective(self) -> None:
        for verdict in (Verdict.EMPTY, Verdict.ROTATE):
            with self.subTest(verdict=verdict):
                runner = Mock(home_pose=[0.0])
                station = Mock()
                station.inspect.return_value = SimpleNamespace(verdict=verdict, message=verdict.value)
                with self.assertLogs(run_model.logger, level="WARNING") as logs:
                    self.assertTrue(self._run_cycle(runner, station, inconclusive_as_defective=True))
                self.assertIn("Inspection inconclusive", logs.output[0])
                runner.run_stage.assert_called_once_with(
                    run_model.BAD_TASK, duration_s=run_model.BAD_DURATION_S,
                    stop_event_factory=run_model.GripperReleaseWatch,
                )
                runner.reset.assert_called_once()


class RunStageTests(unittest.TestCase):
    def test_builds_completion_signal_from_connected_robot(self) -> None:
        events: list[str] = []
        signal = SimpleNamespace(is_set=lambda: True)

        class FakeRuntime:
            robot = SimpleNamespace(
                get_observation=lambda: SimpleNamespace(joint_positions=[0.0]),
                send_action=Mock(),
            )
            action_source = SimpleNamespace(set_task=lambda task: events.append(f"task {task}"))
            last_run_reason = "stop_requested"

            def run(self, **kwargs):
                events.append("run")
                self.assert_signal = kwargs["stop_event"]
                return 0

        runtime = FakeRuntime()
        runner = run_model.PolicyRunner.__new__(run_model.PolicyRunner)
        runner._runtime = runtime

        def make_signal(robot):
            self.assertEqual(events, ["task place"])
            self.assertIs(robot, runtime.robot)
            return signal

        runner.run_stage("place", duration_s=1, stop_event_factory=make_signal)

        self.assertEqual(events, ["task place", "run"])
        self.assertIs(runtime.assert_signal, signal)
        runtime.robot.send_action.assert_called_once_with(runtime.robot.get_observation().joint_positions)

    def test_release_diagnostics_on_interrupt_use_existing_observations(self) -> None:
        now = [0.0]
        position = [20.0]
        robot = SimpleNamespace(
            joint_names=["gripper"],
            get_observation=Mock(side_effect=lambda: SimpleNamespace(joint_positions=[position[0]])),
            send_action=Mock(),
        )

        def run(**kwargs):
            watch = kwargs["stop_event"]
            for timestamp, value in (
                (0.0, 20.0), (0.2, 45.0), (0.5, 65.0), (0.8, 35.0), (1.1, 49.0), (1.9, 30.0),
            ):
                now[0], position[0] = timestamp, value
                self.assertFalse(watch.is_set())
            now[0] = 2.5
            return 6

        runner = run_model.PolicyRunner.__new__(run_model.PolicyRunner)
        runner._runtime = SimpleNamespace(
            robot=robot, action_source=SimpleNamespace(set_task=Mock()), run=run, last_run_reason="interrupted",
        )
        with self.assertLogs(run_model.logger, level="INFO") as logs:
            with self.assertRaises(KeyboardInterrupt):
                runner.run_stage(
                    "place", duration_s=1,
                    stop_event_factory=lambda connected: run_model.GripperReleaseWatch(
                        connected, delta=30, clock=lambda: now[0]
                    ),
                )

        output = "\n".join(logs.output)
        self.assertIn("initial=20.00 threshold=30.00", output)
        self.assertIn("Gripper grasp detected: position=35.00 peak=65.00", output)
        self.assertIn("position=49.00 delta=+29.00 recent_min=20.00 recent_max=65.00", output)
        self.assertIn("runtime_reason=interrupted", output)
        self.assertIn("min=20.00 max=65.00 max_abs_delta=45.00 polls=6 max_poll_gap=0.80s", output)
        self.assertIn("since_last_poll=0.60s confirmed=False grasped=True", output)
        self.assertEqual(robot.get_observation.call_count, 8)
        robot.send_action.assert_called_once_with([30.0])

if __name__ == "__main__":
    unittest.main()

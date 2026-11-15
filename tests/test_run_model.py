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
    def test_confirms_only_after_gripper_movement_is_sustained(self) -> None:
        now = [0.0]
        position = [20.0]
        robot = SimpleNamespace(
            joint_names=["shoulder_pan", "gripper"],
            get_observation=lambda: SimpleNamespace(joint_positions=[0.0, position[0]]),
        )
        watch = run_model.GripperReleaseWatch(
            robot,
            delta=30,
            hold_s=0.2,
            poll_s=0.1,
            clock=lambda: now[0],
        )

        self.assertFalse(watch.is_set())
        position[0] = 55.0
        now[0] = 0.1
        self.assertFalse(watch.is_set())
        now[0] = 0.2
        self.assertFalse(watch.is_set())
        now[0] = 0.31
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


    def _run_cycle(self, runner: Mock, station: Mock) -> bool:
        inspection = Mock(__enter__=Mock(return_value=station), __exit__=Mock(return_value=None))
        anomaly = ModuleType("hackjunction2026_intel.anomaly_detection")
        anomaly.InspectionStation = Mock(return_value=inspection)
        anomaly.Verdict = Verdict
        with patch.dict(sys.modules, {"hackjunction2026_intel.anomaly_detection": anomaly}):
            return run_model.run_cycle(runner)

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
        runner = Mock(home_pose=[0.0])
        station = Mock()
        station.inspect.return_value = SimpleNamespace(verdict=Verdict.EMPTY, message="empty")
        with self.assertRaisesRegex(RuntimeError, "inconclusive"):
            self._run_cycle(runner, station)
        runner.run_stage.assert_not_called()


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

if __name__ == "__main__":
    unittest.main()

import copy
import enum
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

from hackjunction2026_intel import run_model, sorting


class RunPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.export = self.root / "studio-export"
        self.weights = self.export / "exports" / "openvino"
        self.weights.mkdir(parents=True)
        self.config_path = self.export / "runtime.yaml"
        self.config_path.write_text("# Mocked Studio configuration\n")

        self.runtime = MagicMock()
        self.runtime.__exit__.return_value = False
        self.runtime.run.return_value = 30
        self.runtime.last_run_reason = "duration_elapsed"
        self.recipe = SimpleNamespace(
            class_path="physicalai.runtime.RobotRuntime",
            init_args={
                "robot": {"port": "/dev/test-follower"},
                "cameras": {"overhead": {"device": "/dev/test-camera"}},
                "fps": 30,
                "action_source": {
                    "class_path": "physicalai.runtime.PolicySource",
                    "init_args": {
                        "task": "previous task",
                        "execution": {"class_path": "physicalai.runtime.SyncExecution"},
                        "model": {
                            "class_path": "physicalai.inference.InferenceModel",
                            "init_args": {
                                "export_dir": "./exports/openvino",
                                "policy_name": "previous-policy-name",
                                "backend": "onnx",
                                "device": "GPU",
                            },
                        },
                    },
                },
            },
            instantiate=Mock(return_value=self.runtime),
        )
        self.original_args = copy.deepcopy(self.recipe.init_args)
        self.config_api = Mock()
        self.config_api.load.return_value = self.recipe

        config_module = ModuleType("physicalai.config")
        config_module.Config = self.config_api
        runtime_module = ModuleType("physicalai.runtime")
        runtime_module.RobotRuntime = MagicMock
        modules = patch.dict(sys.modules, {
            "physicalai": ModuleType("physicalai"),
            "physicalai.config": config_module,
            "physicalai.runtime": runtime_module,
        })
        modules.start()
        self.addCleanup(modules.stop)

    def test_runs_export_on_openvino_preserving_hardware_and_execution_settings(self) -> None:
        sorting.run_policy(self.export, "present block", duration_s=2.5)

        expected = copy.deepcopy(self.original_args)
        source_args = expected["action_source"]["init_args"]
        source_args["task"] = "present block"
        source_args["model"]["init_args"]["export_dir"] = str(self.weights.resolve())
        source_args["model"]["init_args"]["backend"] = "openvino"
        source_args["model"]["init_args"]["device"] = sorting.OPENVINO_DEVICE
        del source_args["model"]["init_args"]["policy_name"]
        self.assertEqual(self.recipe.init_args, expected)
        self.assertEqual(self.config_path.read_text(), "# Mocked Studio configuration\n")
        self.config_api.load.assert_called_once_with(self.config_path)
        self.runtime.run.assert_called_once_with(duration_s=2.5, stop_event=None, return_to_initial_state=False)
        measured = self.runtime.robot.get_observation.return_value
        self.runtime.robot.send_action.assert_called_once_with(measured.joint_positions)
        calls = self.runtime.mock_calls
        self.assertLess(calls.index(call.run(duration_s=2.5, stop_event=None, return_to_initial_state=False)),
                        calls.index(call.robot.get_observation()))
        self.assertLess(calls.index(call.robot.send_action(measured.joint_positions)),
                        calls.index(call.__exit__(None, None, None)))

    def test_interruption_and_early_stops_abort_and_release_hardware(self) -> None:
        for reason, ticks, error in (
            ("interrupted", 10, KeyboardInterrupt),
            ("stop_requested", 10, RuntimeError),
            ("error", 10, RuntimeError),
            ("duration_elapsed", 0, RuntimeError),
        ):
            with self.subTest(reason=reason, ticks=ticks):
                self.runtime.reset_mock()
                self.runtime.last_run_reason = reason
                self.runtime.run.return_value = ticks
                with self.assertRaises(error):
                    sorting.run_policy(self.export, "present", duration_s=1)
                self.runtime.robot.send_action.assert_called_once()
                self.runtime.__exit__.assert_called_once()

    def test_stop_event_completes_the_stage_and_its_absence_is_a_timeout(self) -> None:
        signal = SimpleNamespace(is_set=lambda: True)
        self.runtime.last_run_reason = "stop_requested"
        self.runtime.run.return_value = 0
        sorting.run_policy(self.export, "present", duration_s=1, stop_event=signal)
        self.runtime.run.assert_called_once_with(duration_s=1, stop_event=signal, return_to_initial_state=False)
        for reason in ("duration_elapsed", "error"):
            with self.subTest(reason=reason):
                self.runtime.reset_mock()
                self.runtime.last_run_reason = reason
                with self.assertRaisesRegex(RuntimeError, "did not complete"):
                    sorting.run_policy(self.export, "present", duration_s=1, stop_event=signal)
                self.runtime.robot.send_action.assert_called_once()
                self.runtime.__exit__.assert_called_once()

    def test_inference_failure_still_requests_hold_and_disconnects(self) -> None:
        self.runtime.run.side_effect = ConnectionError("camera lost")
        with self.assertRaisesRegex(ConnectionError, "camera lost"):
            sorting.run_policy(self.export, "present", duration_s=1)
        self.runtime.robot.send_action.assert_called_once()
        self.runtime.__exit__.assert_called_once()

    def test_hold_failure_aborts_and_disconnects(self) -> None:
        self.runtime.robot.send_action.side_effect = OSError("hold failed")
        with self.assertRaisesRegex(OSError, "hold failed"):
            sorting.run_policy(self.export, "present", duration_s=1)
        self.runtime.__exit__.assert_called_once()

    def test_invalid_duration_or_missing_artifacts_never_load_runtime(self) -> None:
        for duration in (0, -1, float("nan"), float("inf")):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                sorting.run_policy(self.export, "present", duration_s=duration)
        with self.assertRaises(FileNotFoundError):
            sorting.run_policy(self.root / "missing", "present", duration_s=1)
        with patch.object(sorting, "OPENVINO_DEVICE", "cuda"), self.assertRaisesRegex(ValueError, "OPENVINO_DEVICE"):
            sorting.run_policy(self.export, "present", duration_s=1)
        self.weights.rmdir()
        with self.assertRaisesRegex(FileNotFoundError, "openvino"):
            sorting.run_policy(self.export, "present", duration_s=1)
        self.weights.mkdir()
        self.config_path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "runtime.yaml"):
            sorting.run_policy(self.export, "present", duration_s=1)
        self.config_api.load.assert_not_called()

    def test_rejects_non_inference_runtime_recipes_before_instantiation(self) -> None:
        self.recipe.class_path = "other.Runtime"
        with self.assertRaisesRegex(ValueError, "Studio-exported"):
            sorting.run_policy(self.export, "present", duration_s=1)
        self.recipe.class_path = "physicalai.runtime.RobotRuntime"
        self.recipe.init_args["action_source"]["class_path"] = "physicalai.runtime.TeleopSource"
        with self.assertRaisesRegex(ValueError, "teleoperation"):
            sorting.run_policy(self.export, "present", duration_s=1)
        self.recipe.init_args = copy.deepcopy(self.original_args)
        self.recipe.init_args["action_source"]["init_args"]["model"] = None
        with self.assertRaisesRegex(ValueError, "InferenceModel"):
            sorting.run_policy(self.export, "present", duration_s=1)
        self.recipe.instantiate.assert_not_called()


class Verdict(enum.Enum):
    EMPTY = "empty"
    ROTATE = "rotate"
    GOOD = "good"
    DEFECTIVE = "defective"


def fake_anomaly_detection(station: object | None = None) -> ModuleType:
    module = ModuleType("hackjunction2026_intel.anomaly_detection")
    module.Verdict = Verdict
    module.InspectionStation = MagicMock(return_value=station or MagicMock())
    return module


class PresenceWatchTests(unittest.TestCase):
    def test_confirms_only_after_continuous_presence(self) -> None:
        # One clock second per present reading; the absent reading restarts the count.
        readings = iter([True, True, True, False, True, True, True, True])
        clock = iter(range(100))
        watch = sorting.PresenceWatch(lambda: next(readings), 3, poll_s=0, clock=lambda: next(clock))
        with watch:
            watch._thread.join(timeout=5)
        self.assertEqual(list(readings), [], "confirmed before the brick stayed in view long enough")
        self.assertTrue(watch.confirmed.is_set())
        self.assertTrue(watch.is_set())
        self.assertIsNone(watch.error)

    def test_not_confirmed_while_presence_keeps_dropping(self) -> None:
        watch = sorting.PresenceWatch(lambda: False, 0, poll_s=0.001)
        with watch:
            time.sleep(0.05)
        self.assertFalse(watch.is_set())

    def test_check_error_stops_the_run_and_is_kept(self) -> None:
        def broken() -> bool:
            raise RuntimeError("camera lost")

        watch = sorting.PresenceWatch(broken, 3, poll_s=0)
        with watch:
            watch._thread.join(timeout=5)
        self.assertTrue(watch.is_set())
        self.assertFalse(watch.confirmed.is_set())
        self.assertIsInstance(watch.error, RuntimeError)


class InspectionTests(unittest.TestCase):
    def setUp(self) -> None:
        modules = patch.dict(sys.modules, {"hackjunction2026_intel.anomaly_detection": fake_anomaly_detection()})
        modules.start()
        self.addCleanup(modules.stop)
        self.station = MagicMock()
        self.export = Path("/unused-export")

    def test_verdicts_map_to_defective_or_raise(self) -> None:
        for verdict, expected in ((Verdict.DEFECTIVE, True), (Verdict.GOOD, False)):
            self.station.inspect.return_value = SimpleNamespace(verdict=verdict, message=verdict.name)
            self.assertIs(sorting.inspect_block(self.station), expected)
        for verdict in (Verdict.EMPTY, Verdict.ROTATE):
            with self.subTest(verdict=verdict):
                self.station.inspect.return_value = SimpleNamespace(verdict=verdict, message=verdict.name)
                with self.assertRaisesRegex(RuntimeError, "inconclusive"):
                    sorting.inspect_block(self.station)

    def test_presentation_completes_only_when_not_empty(self) -> None:
        for empty, completes in ((False, True), (True, False)):
            with self.subTest(empty=empty):
                self.station.is_empty.return_value = empty
                seen = {}

                def run_until_stop(export, task, *, duration_s, stop_event) -> None:
                    seen.update(export=export, task=task, duration_s=duration_s)
                    stop_event._thread.join(timeout=0.2)
                    seen["completed"] = stop_event.is_set()

                with patch.object(sorting, "PRESENT_CONFIRM_S", 0), \
                        patch.object(sorting, "run_policy", side_effect=run_until_stop):
                    sorting.present_block(self.export, self.station)
                self.assertEqual(seen["completed"], completes)
                self.assertEqual((seen["export"], seen["task"], seen["duration_s"]),
                                 (self.export, sorting.PRESENT_TASK, sorting.PRESENT_DURATION_S))

    def test_presentation_requires_presence_calibration_and_a_working_check(self) -> None:
        self.station.inspector.presence = None
        with patch.object(sorting, "run_policy") as run, self.assertRaisesRegex(RuntimeError, "empty-ROI"):
            sorting.present_block(self.export, self.station)
        run.assert_not_called()

        self.station.inspector.presence = object()
        self.station.is_empty.side_effect = OSError("camera lost")

        def wait_for_watch(*args, stop_event, **kwargs) -> None:
            stop_event._thread.join(timeout=5)

        with patch.object(sorting, "run_policy", side_effect=wait_for_watch):
            with self.assertRaisesRegex(RuntimeError, "EMPTY check failed"):
                sorting.present_block(self.export, self.station)


class SortingCycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.station = MagicMock()
        modules = patch.dict(
            sys.modules, {"hackjunction2026_intel.anomaly_detection": fake_anomaly_detection(self.station)}
        )
        modules.start()
        self.addCleanup(modules.stop)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.export = self.root / "studio-export"
        (self.export / "exports" / "openvino").mkdir(parents=True)
        (self.export / "runtime.yaml").write_text("# Mocked Studio configuration\n")
        settings = patch.multiple(
            sorting, PRESENT_DURATION_S=1.0, GOOD_DURATION_S=2.0, BAD_DURATION_S=3.0,
        )
        settings.start()
        self.addCleanup(settings.stop)

    def test_live_sorting_uses_policy_runner_cycle(self) -> None:
        with patch.object(run_model, "PolicyRunner") as create_runner, \
                patch.object(run_model, "run_cycle") as run:
            sorting.main(["--export", str(self.export)])
        create_runner.assert_called_once_with(self.export)
        run.assert_called_once_with(create_runner.return_value, inconclusive_as_defective=True)

    def test_zip_export_is_extracted_including_a_wrapping_folder(self) -> None:
        archive = self.root / "export.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("my-model/runtime.yaml", "# zipped\n")
            zipped.writestr("my-model/exports/openvino/manifest.json", "{}")
        with patch.object(run_model, "PolicyRunner") as create_runner, \
                patch.object(run_model, "run_cycle") as run:
            sorting.main(["--export", str(archive)])
        create_runner.assert_called_once()
        export_root = create_runner.call_args.args[0]
        self.assertEqual(export_root.name, "my-model")
        run.assert_called_once_with(create_runner.return_value, inconclusive_as_defective=True)
        self.assertFalse(export_root.exists(), "extracted export should be cleaned up")

    def test_missing_export_prevents_any_motion(self) -> None:
        with patch.object(run_model, "PolicyRunner") as create_runner:
            with self.assertRaises(FileNotFoundError):
                sorting.main(["--export", str(self.root / "missing.zip")])
            (self.export / "exports" / "openvino").rmdir()
            with self.assertRaises(FileNotFoundError):
                sorting.main(["--export", str(self.export)])
            create_runner.assert_not_called()

    def test_unset_placement_duration_prevents_any_motion(self) -> None:
        with patch.object(sorting, "BAD_DURATION_S", 0), patch.object(run_model, "PolicyRunner") as create_runner:
            with self.assertRaises(ValueError):
                sorting.main(["--export", str(self.export)])
            create_runner.assert_not_called()

    def test_dry_run_without_export_routes_both_outcomes_without_loading_sdk(self) -> None:
        for defective, placement in ((True, sorting.BAD_TASK), (False, sorting.GOOD_TASK)):
            with self.subTest(defective=defective), patch.object(sorting, "run_policy") as run:
                with patch.object(sorting, "random", return_value=0.25 if defective else 0.75):
                    with patch.multiple(
                        sorting, PRESENT_DURATION_S=0.0, GOOD_DURATION_S=0.0, BAD_DURATION_S=0.0,
                    ), patch.dict(sys.modules, {"physicalai": None}):
                        with self.assertLogs(sorting.logger, level="INFO") as logs:
                            sorting.main(["--dry-run"])
                stages = [line for line in logs.output if "DRY RUN: task" in line]
                self.assertEqual(len(stages), 2)
                self.assertIn(sorting.PRESENT_TASK, stages[0])
                self.assertIn(placement, stages[1])
                run.assert_not_called()

    def test_dry_run_with_export_still_rejects_missing_artifacts(self) -> None:
        with patch.object(sorting, "run_policy") as run:
            with self.assertRaises(FileNotFoundError):
                sorting.main(["--dry-run", "--export", str(self.root / "missing.zip")])
            (self.export / "exports" / "openvino").rmdir()
            with self.assertRaisesRegex(FileNotFoundError, "openvino"):
                sorting.main(["--dry-run", "--export", str(self.export)])
            run.assert_not_called()

    def test_normal_execution_requires_export_even_when_durations_are_configured(self) -> None:
        with patch.object(sorting, "run_policy") as run:
            with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                sorting.main([])
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

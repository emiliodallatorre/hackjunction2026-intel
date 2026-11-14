import copy
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock, call, patch

from hackjunction2026_intel import sorting


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
        self.runtime.run.assert_called_once_with(duration_s=2.5, return_to_initial_state=False)
        measured = self.runtime.robot.get_observation.return_value
        self.runtime.robot.send_action.assert_called_once_with(measured.joint_positions)
        calls = self.runtime.mock_calls
        self.assertLess(calls.index(call.run(duration_s=2.5, return_to_initial_state=False)),
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


class SortingCycleTests(unittest.TestCase):
    def setUp(self) -> None:
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

    def test_each_outcome_runs_presentation_then_its_placement_on_the_same_export(self) -> None:
        for defective, task, duration in (
            (True, "Move the brick to the orange dish", sorting.BAD_DURATION_S),
            (False, "Move the brick to the yellow dish", sorting.GOOD_DURATION_S),
        ):
            with self.subTest(defective=defective), patch.object(sorting, "run_policy") as run:
                def inspect() -> bool:
                    self.assertEqual(run.call_count, 1)
                    return defective

                with patch.object(sorting, "inspect_block", side_effect=inspect):
                    sorting.main(["--export", str(self.export)])
                self.assertEqual(run.call_args_list, [
                    call(self.export, "Move any brick to the center", duration_s=1.0),
                    call(self.export, task, duration_s=duration),
                ])

    def test_zip_export_is_extracted_including_a_wrapping_folder(self) -> None:
        archive = self.root / "export.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("my-model/runtime.yaml", "# zipped\n")
            zipped.writestr("my-model/exports/openvino/manifest.json", "{}")
        seen: list[Path] = []
        with patch.object(sorting, "run_policy", side_effect=lambda root, *a, **k: seen.append(root)):
            with patch.object(sorting, "inspect_block", return_value=False):
                sorting.main(["--export", str(archive)])
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0], seen[1])
        self.assertEqual(seen[0].name, "my-model")
        self.assertFalse(seen[0].exists(), "extracted export should be cleaned up")

    def test_missing_export_prevents_any_motion(self) -> None:
        with patch.object(sorting, "run_policy") as run:
            with self.assertRaises(FileNotFoundError):
                sorting.main(["--export", str(self.root / "missing.zip")])
            (self.export / "exports" / "openvino").rmdir()
            with self.assertRaises(FileNotFoundError):
                sorting.main(["--export", str(self.export)])
            run.assert_not_called()

    def test_unset_placement_duration_prevents_any_motion(self) -> None:
        with patch.object(sorting, "BAD_DURATION_S", 0), patch.object(sorting, "run_policy") as run:
            with self.assertRaises(ValueError):
                sorting.main(["--export", str(self.export)])
            run.assert_not_called()

    def test_presentation_failure_prevents_inspection_or_placement(self) -> None:
        with patch.object(sorting, "run_policy", side_effect=RuntimeError("aborted")) as run:
            with patch.object(sorting, "inspect_block") as inspect:
                with self.assertRaisesRegex(RuntimeError, "aborted"):
                    sorting.main(["--export", str(self.export)])
                run.assert_called_once()
                inspect.assert_not_called()


if __name__ == "__main__":
    unittest.main()

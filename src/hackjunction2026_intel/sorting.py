"""Timed demonstration of one pick, inspection, and sorting cycle."""

import argparse
import logging
import zipfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from math import isfinite
from pathlib import Path
from random import random
from tempfile import TemporaryDirectory

logger = logging.getLogger(__name__)

# The single multi-task π0.5 model always runs on OpenVINO, ignoring the export's device.
BACKEND = "openvino"
# AUTO lets OpenVINO pick the best local accelerator; pin "CPU", "GPU" or "NPU" if needed.
OPENVINO_DEVICE = "AUTO"
OPENVINO_DEVICES = frozenset({"AUTO", "CPU", "GPU", "NPU"})

# Task instructions match the relabelled demonstration dataset.
PRESENT_TASK = "Move any brick to the center"
GOOD_TASK = "Move the brick to the yellow dish"
BAD_TASK = "Move the brick to the orange dish"

# TODO: Set supervised, validated durations in seconds; zero disables execution.
PRESENT_DURATION_S = 0.0
GOOD_DURATION_S = 0.0
BAD_DURATION_S = 0.0


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


def run_policy(export_root: Path, task: str, *, duration_s: float) -> None:
    """Run a timed stage, without claiming that the manipulation succeeded."""
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

    logger.warning(
        "Running timed demo task %r for %s control seconds; task success is NOT verified",
        task,
        duration_s,
    )
    with runtime:
        try:
            ticks = runtime.run(duration_s=duration_s, return_to_initial_state=False)
            if runtime.last_run_reason == "interrupted":
                raise KeyboardInterrupt("Policy stage interrupted; aborting the sorting cycle")
            if runtime.last_run_reason != "duration_elapsed" or ticks <= 0:
                raise RuntimeError(f"Policy stage did not finish its duration: {runtime.last_run_reason}")
        finally:
            # Cancel any remaining position target before releasing the driver.
            measured = runtime.robot.get_observation()
            runtime.robot.send_action(measured.joint_positions)


def inspect_block() -> bool:
    """Return True for defective and False for good."""
    # TODO: Replace this demo-only decision with camera capture and anomaly inference.
    # Raise on uncertain results or inspection failure rather than assuming good.
    return random() < 0.5


def run_cycle(export_root: Path) -> None:
    for duration in (PRESENT_DURATION_S, GOOD_DURATION_S, BAD_DURATION_S):
        _validate_duration(duration)
    _validate_export(export_root)
    logger.warning("Demo inspection uses randomness, not real defect detection")

    logger.info("Moving the block to the inspection stage")
    run_policy(export_root, PRESENT_TASK, duration_s=PRESENT_DURATION_S)

    is_defective = inspect_block()
    if is_defective:
        logger.info("Inspection decision: defective; placing in the bad dish")
        run_policy(export_root, BAD_TASK, duration_s=BAD_DURATION_S)
    else:
        logger.info("Inspection decision: good; placing in the good dish")
        run_policy(export_root, GOOD_TASK, duration_s=GOOD_DURATION_S)

    logger.warning("Timed sorting demo finished; manipulation outcomes were not verified")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        type=Path,
        required=True,
        help="Studio OpenVINO runtime export: the downloaded .zip or its extracted folder",
    )
    args = parser.parse_args(argv)
    with open_export(args.export) as export_root:
        run_cycle(export_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

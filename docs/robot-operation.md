# Robot operation and safety

The sorting script runs one presentation, inspection, and placement cycle. The operator dashboard
can run repeated cycles; see the [dashboard guide](../operator_gui/README.md).

**Supervise the arm, keep the workspace clear, and keep a physical emergency stop within reach.**
Software hold and stop requests are not a physical emergency stop.

## Hardware-free dry run

From the repository root:

```bash
PYTHONPATH=src python3 -m hackjunction2026_intel.sorting --dry-run
```

This logs the stages and chooses a simulated inspection result. It does not load the policy, run
the anomaly model, connect cameras, or move the robot.

## Prerequisites

- An SO-101 arm with its serial port and calibration configured, plus policy and inspection cameras
- A compatible Physical AI Studio OpenVINO runtime export, supplied separately from this repository
- The PatchCore model files, downloaded with Git LFS
- A calibrated inspection camera and empty-scene reference, as described in the
  [anomaly detection guide](../anomaly_detection/README.md)

Install the robot dependencies from the repository root:

```bash
git lfs pull
uv sync --extra robot
```

Before operating the robot:

- Check the export's `runtime.yaml` for the robot port, calibration, camera identities, control
  rate, and safety callbacks
- Set the inspection camera source and controls in
  [`anomaly_detection/config.yaml`](../anomaly_detection/config.yaml), and keep that camera out of
  the policy export so only the inspection station opens it
- Stop any competing Studio sessions or camera previews
- Confirm that the task instructions and starting poses match the demonstrations used to train
  the policy
- Set positive, supervised-test values for `PRESENT_DURATION_S`, `GOOD_DURATION_S`, and
  `BAD_DURATION_S` in [`sorting.py`](../src/hackjunction2026_intel/sorting.py)

The runtime export must contain `runtime.yaml` and `exports/openvino/`. The script selects the
OpenVINO backend and the device specified by `OPENVINO_DEVICE` in `sorting.py`; the current value
is `GPU`. It applies these settings in memory without rewriting the export.

## Run one cycle

Set `MODEL_EXPORT` to the location of the Studio runtime export ZIP or extracted directory, then run:

```bash
uv run --extra robot python -m hackjunction2026_intel.sorting --export "$MODEL_EXPORT"
```

The script opens the inspection station before motion and performs the following stages:

1. **Present** — move a brick into the inspection area. Presence polling starts after the first
   policy control tick; the brick must remain present for `PRESENT_CONFIRM_S` seconds before the
   arm holds it for inspection
2. **Inspect** — score the image and select the placement. In the standalone sorting script,
   `EMPTY` and uncertain results default to the defective dish; the dashboard and `run_model cycle`
   abort instead
3. **Place and reset** — run the selected placement instruction until the gripper-motion check
   detects a grasp followed by release, then return to the pose captured before the first cycle

The placement check is a gripper-motion heuristic, not visual confirmation that the brick reached
the dish. A re-grasp during transport can trigger it early. Stage durations use control-loop timing
rather than a hard wall-clock watchdog, so slow inference can extend elapsed time. Model loading is
outside the timed run.

The anomaly model must include an empty-ROI reference. If it is missing, collect empty images and
run `brick-ad prepare` followed by `brick-ad fit-empty` before using the arm.

## Tests

The hardware-free tests require only the Python standard library:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

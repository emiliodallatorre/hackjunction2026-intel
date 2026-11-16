# System architecture

Ricotta Pi combines a vision-language-action policy, a visual anomaly detector, and an operator
dashboard. OpenVINO runs the models locally on the Intel laptop.

## Sorting pipeline

1. The π0.5 policy receives the robot state, camera images, and a text instruction, then produces
   an action chunk for the SO-101 arm
2. The arm presents a brick to a separate inspection camera
3. The inspection station checks for a brick and uses PatchCore to score its appearance
4. The resulting decision selects a placement instruction for the policy

The policy uses the integrated GPU, while the inspection model is configured for the NPU. The
dashboard displays camera frames, inspection results, sorting counts, and device telemetry.

## Physical AI runtime

The robot runtime uses Intel's [`physicalai`](https://github.com/openvinotoolkit/physicalai)
library and SO-101 plugin. `Config` loads the Studio runtime export, and `RobotRuntime` handles
camera capture, observations, policy inference, and robot commands.

[`PolicyRunner`](../src/hackjunction2026_intel/run_model.py) validates the export before motion,
sets the OpenVINO backend and device in memory, and reuses the same runtime across the stages of a
cycle. [`PersistentInferenceModel`](../src/hackjunction2026_intel/persistent_model.py) keeps the
policy loaded between runtime calls and releases it when the session closes.

The inspection camera is managed separately by
[`InspectionStation`](../src/hackjunction2026_intel/anomaly_detection/station.py). Its presence
check compares the camera crop with an empty-scene reference before the anomaly model is run.
Inspection camera settings and the region of interest are defined in
[`config.yaml`](../anomaly_detection/config.yaml).

## Repository layout

| Path | Purpose |
|---|---|
| [`src/hackjunction2026_intel/`](https://github.com/emiliodallatorre/hackjunction2026-intel/tree/main/src/hackjunction2026_intel/) | Robot control, policy runtime, and inspection inference |
| [`anomaly_detection/`](https://github.com/emiliodallatorre/hackjunction2026-intel/tree/main/anomaly_detection/) | Camera calibration, image collection, PatchCore training, and exported models |
| [`operator_gui/`](https://github.com/emiliodallatorre/hackjunction2026-intel/tree/main/operator_gui/) | NiceGUI dashboard and simulated operation |
| [`benchmark/`](https://github.com/emiliodallatorre/hackjunction2026-intel/tree/main/benchmark/) | Model benchmarking scripts, measurements, and plots |
| [`.github/skills/`](https://github.com/emiliodallatorre/hackjunction2026-intel/tree/main/.github/skills/) | Physical AI Studio resource inspection and training automation |
| [`infra/`](https://github.com/emiliodallatorre/hackjunction2026-intel/tree/main/infra/) | Physical AI Studio container configuration |
| [`videos/`](https://github.com/emiliodallatorre/hackjunction2026-intel/tree/main/videos/) | Demonstration recordings and presentation materials |

The anomaly-detection training tools and robot runtime use separate Python environments because
their dependencies conflict. The dashboard operates the robot through `PolicyRunner` and installs
an inspection observer to publish frames and decisions.

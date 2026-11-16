<div align="center">

# Ricotta Pi

**Second place in the** [**Intel Physical AI Challenge**](guidelines/CHALLENGE.md) **at** [**Junction 2026**](https://hackjunction.app/hackathons/junction-2026)

Team Sfogliatelle

[![Watch the demonstration](https://img.youtube.com/vi/i87zoaiSM1U/maxresdefault.jpg)](https://youtu.be/i87zoaiSM1U)

[Watch the demonstration](https://youtu.be/i87zoaiSM1U)

</div>

Ricotta Pi is an autonomous inspection and sorting prototype built during Junction 2026. An SO-101
robot arm presents LEGO DUPLO bricks to a camera, identifies visual defects, and places each brick
in the corresponding dish.

The system combines a fine-tuned **π0.5 vision-language-action model** for robot control with
**PatchCore** for visual inspection. Both models run locally on an Intel Core Ultra X9 388H laptop
using OpenVINO. The robot policy was fine-tuned separately on NVIDIA B200 GPUs through Modal.

## How it works

1. **Present** — π0.5 moves a brick into the inspection area
2. **Inspect** — PatchCore scores the camera image for visual anomalies
3. **Sort** — π0.5 places the brick in the yellow dish if it is good, or the orange dish if it is defective

The same policy handles presentation and placement using different language instructions. During
demonstration runs, the arm also picked bricks from different positions, selected from clutter, and
re-grasped a dropped brick.

<p align="center">
  <a href="videos/04-full-run.mp4">
    <img src="docs/images/full-run-preview.gif" alt="The robot completing a full sorting run, shown at twice normal speed" width="360">
  </a>
</p>

<p align="center"><em>A complete sorting run at twice normal speed. <a href="videos/04-full-run.mp4">Watch the full-resolution video</a></em></p>

![Inspection camera image beside the PatchCore anomaly heatmap](videos/05-anomaly-sidebyside.png)

*The inspection camera image and the corresponding anomaly heatmap*

## Training and data

We recorded **316 teleoperated demonstrations** with Intel Physical AI Studio and used them to
fine-tune π0.5 on NVIDIA B200 GPUs through Modal. The demonstrations covered three task intents:

| Task | Episodes |
|---|---:|
| Present a brick for inspection | 192 |
| Place a brick in the yellow dish | 65 |
| Place a brick in the orange dish | 59 |
| **Total** | **316** |

[Watch the dataset recording video](videos/06-training.mp4). The raw demonstrations are not included
in this repository.

PatchCore was fitted on **181 images of good bricks**. Anomalib reported image AUROC **1.000** and
image F1 **0.991** on 45 good and 57 anomalous images; the same images were used for threshold
calibration. The inspection image data is not included in the repository.

## Local inference

We exported the models to OpenVINO and distributed the workloads across the laptop's integrated
accelerators: π0.5 on the **integrated GPU** and PatchCore on the **NPU**. Inference and robot
control run locally; the cloud GPUs were used for training.

The following model-only benchmark measured a 50-action π0.5 chunk with synthetic inputs:

| Device | Precision | Median inference time |
|---|---|---:|
| Intel integrated GPU | FP16 | **105 ms** |
| CPU | FP32 | **8,104 ms** |

The integrated GPU was approximately **77 times faster** in this comparison. The plot below
compares latency, throughput, package energy per inference, and package temperature for the
same benchmark run.

![π0.5 GPU and CPU comparison showing inference latency, throughput, energy, and temperature](benchmark/results/plots_vla_20261115_055046/device_comparison.png)

### Latency and control-rate headroom

Across 60 measured inferences per device, the integrated GPU's p99 latency was **106.76 ms**,
compared with **8,126.21 ms** on the CPU. The distributions below show the individual inference
times; the CPU and GPU panels use different horizontal scales.

![π0.5 inference latency distributions, with the CPU near 8,104 ms and the GPU near 105 ms](benchmark/results/plots_vla_20261115_055046/latency_distribution.png)

At a 30 Hz control rate, a 50-action chunk represents approximately **1.67 seconds of robot
motion**. GPU inference occupied about **6.4%** of that budget at p99, while CPU inference
exceeded it. This left time for camera capture, preprocessing, and robot communication on the
GPU configuration. These figures describe model inference, not a measured end-to-end control rate.

The [benchmark results](benchmark/RESULTS.md) include the measurement details, additional device
comparisons, sustained-load measurements, and links to the recorded data.

## Operator dashboard

The [operator dashboard](operator_gui/README.md) provides the inspection camera feed, anomaly
heatmap, sorting counts, device telemetry, and controls for running the robot. It also has a
hardware-free simulation mode.

## Team

Ricotta Pi was developed by **Team Sfogliatelle**: [Emilio Dalla Torre](https://github.com/emiliodallatorre), [Chiara Marangoni](https://github.com/chiamara02), and [Dennis Orlando](https://github.com/dennisorlando).

![Team Sfogliatelle with the second-place award at Junction 2026](docs/images/winners.jpeg)

## Explore the project

The sorting sequence can be previewed without a robot or model export:

```bash
PYTHONPATH=src python3 -m hackjunction2026_intel.sorting --dry-run
```

The dry run simulates inspection decisions and does not perform model inference. Robot operation
requires a compatible model export and calibrated hardware. **Supervise the arm and keep a physical
emergency stop within reach.**

- [Robot operation and safety](docs/robot-operation.md)
- [System architecture](docs/architecture.md)
- [Anomaly detection](anomaly_detection/README.md)
- [Benchmark results](benchmark/RESULTS.md) and [benchmark setup](benchmark/README.md)
- [Operator dashboard](operator_gui/README.md)

The PatchCore model files and demonstration videos are stored with Git LFS.

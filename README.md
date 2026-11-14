# Brick inspection and sorting

## Timed sorting demo

Install the robot dependencies and run from the project root on the robot host:

```bash
uv sync --extra robot
uv run --extra robot python -m hackjunction2026_intel.sorting --export ~/Downloads/model_openvino.zip
```

`--export` takes the zip downloaded from Studio (**Models** → model row → **OpenVINO** card → **Runtime export**) or its extracted folder. To swap models, pass a different export; nothing else changes. Zips are extracted to a temporary folder that is removed afterwards.

The optional robot dependencies match the runtime revision and SO-101/Studio plugin versions used by the Studio image in [docker-compose.yml](infra/docker-compose.yml).

The script runs the single multi-task π0.5 model with the presentation instruction for a configured duration, uses `random() < 0.5` to mark the block defective, then runs the same model with the corresponding placement instruction for its configured duration. This is a timed demonstration: elapsed duration does not establish that the block was picked, presented, or released correctly.

### Configuration

The task instructions in [sorting.py](src/hackjunction2026_intel/sorting.py) match the relabelled demonstration dataset:

- Presentation: `Move any brick to the center`
- Good placement: `Move the brick to the yellow dish`
- Defective placement: `Move the brick to the orange dish`

Before using hardware:

- If using a different dataset, update the task instructions to match its demonstrations
- Set `PRESENT_DURATION_S`, `GOOD_DURATION_S`, and `BAD_DURATION_S` to positive, supervised-test durations; all default to zero and prevent motion until configured
- Check the export's `runtime.yaml` for the serial port, robot calibration, camera identities/names/settings, control rate, and any safety callbacks; lines marked `# CHANGE_ME` contain machine-specific device paths
- Ensure presentation ends in a state compatible with both placement instructions

The export must contain `runtime.yaml` (generated for model inference, not teleoperation) and `exports/openvino/`. The script always runs on OpenVINO: it points the model at `exports/openvino` inside the export, regardless of the working directory, and forces `backend: openvino` with the OpenVINO device set by `OPENVINO_DEVICE` in `sorting.py` (default `AUTO`, which lets OpenVINO pick the best local accelerator; set `CPU`, `GPU`, or `NPU` to pin one). The device chosen in the Studio export dialog is ignored. It also sets the task and removes any fixed policy name so the export's manifest determines it. All other runtime settings are preserved, and `runtime.yaml` is not rewritten. Other relative paths in the configuration resolve from the working directory; use absolute hardware/calibration paths when necessary.

Each stage creates a fresh runtime from the same export, whose policy source resets the model and action queue. The model is reloaded per stage because the SDK closes the inference model at the end of every `RobotRuntime.run()`. The runtime manages camera capture, observations, inference, and robot commands. On exit, the script requests a hold at the measured joint positions before the runtime disconnects. The calibrated SO-101 follower driver normally keeps torque enabled on disconnect; verify this behavior on your actual setup.

Runtime errors, early stops, and Ctrl+C abort the cycle rather than advancing to inspection or placement. Hold and cleanup are software requests, not guaranteed physical stopping or an emergency stop. Stop competing Studio sessions before running the standalone script.

Durations use the SDK's control-loop timing, not a hard wall-clock watchdog; slow inference can extend real elapsed time. Model loading is also outside the timed run. Supervise the demo and retain a physical emergency stop. Replace timing with verified stage-completion checks, and random classification with an anomaly detector, before autonomous use.

### Tests

Run the hardware-free tests from a Python environment using the standard library; robot dependencies are not required:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
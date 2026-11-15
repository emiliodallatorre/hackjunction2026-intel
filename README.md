# Brick inspection and sorting

## Sorting demo

### Hardware-free dry run

From the project root, no `uv`, SDK, model export, or robot is needed:

```bash
PYTHONPATH=src python3 -m hackjunction2026_intel.sorting --dry-run
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The dry run logs presentation, a random inspection decision (the anomaly model is not used), and the selected placement without loading the SDK, the inspection camera, or connecting hardware. It allows zero durations. The tests exercise both sorting outcomes and failure handling. This does not validate model inference, calibration, camera access, serial permissions, or safe stage durations. Once downloaded, add `--export` to check the export's required file/folder layout; runtime configuration contents and model weights are still not validated.

On a host with the existing Studio image, a separate disposable container can reuse its Python and SDK without installing `uv` or rebuilding anything:

```bash
docker run --rm --network none --cap-drop ALL --security-opt no-new-privileges \
  --mount "type=bind,src=$PWD,dst=/workspace,readonly" \
  --workdir /workspace \
  --env PYTHONPATH=/workspace/src --env PYTHONDONTWRITEBYTECODE=1 \
  --entrypoint /app/application/backend/.venv/bin/python \
  physical-ai-studio-xpu:junction2026 \
  -m hackjunction2026_intel.sorting --dry-run
```

Replace the final module invocation with `-m unittest discover -s tests -v` to run the tests in that environment. This container has no host devices or Studio storage mounted and does not affect the running Studio service. It is deliberately unsuitable for real robot execution; hardware access and calibration mounts must be configured separately before supervised use.

### Robot execution

Install the robot dependencies and run from the project root on the robot host:

```bash
uv sync --extra robot
uv run --extra robot python -m hackjunction2026_intel.sorting --export ~/Downloads/model_openvino.zip
```

`--export` takes the zip downloaded from Studio (**Models** → model row → **OpenVINO** card → **Runtime export**) or its extracted folder. To swap models, pass a different export; nothing else changes. Zips are extracted to a temporary folder that is removed afterwards.

The optional robot dependencies match the runtime revision and SO-101/Studio plugin versions used by the Studio image in [docker-compose.yml](infra/docker-compose.yml).

The script opens the inspection camera and anomaly model (`InspectionStation`, see [anomaly_detection/README.md](anomaly_detection/README.md)) before any motion, then:

1. Runs the single multi-task π0.5 model with the presentation instruction until inspection has been anything other than `EMPTY` continuously for `PRESENT_CONFIRM_S` (3 seconds); the robot then holds the brick in place. The check uses the same `EMPTY` rule as the inspection but skips the anomaly model, so it does not slow the policy. If that does not happen within `PRESENT_DURATION_S`, the cycle aborts
2. Inspects the brick with the anomaly model. `DEFECTIVE` and `GOOD` pick the placement; no brick (`EMPTY`) or an uncertain score (`ROTATE`) aborts the cycle, since there is no rotate stage
3. Closes the inspection camera and runs the same model with the corresponding placement instruction for its configured duration. Placement is still timed: elapsed duration does not establish that the brick was released correctly

The anomaly model must include the empty-ROI reference (`brick-ad fit-empty`): without it the model never reports `EMPTY`, so the cycle aborts before moving. The config comes from `$BRICK_AD_CONFIG`, else `anomaly_detection/config.yaml`, with the model at `anomaly_detection/models/patchcore/`. The model is versioned with Git LFS: run `git lfs install` before cloning, or `git lfs pull` afterwards. [docker-compose.yml](infra/docker-compose.yml) mounts both into the Studio container at `/brick` and sets `BRICK_AD_CONFIG`.

### Remote inspection demo

From the Mac, [run-inspection-demo.sh](scripts/run-inspection-demo.sh) connects to roberta and runs only the presentation and inspection stages. It uses `~/Downloads/pi05-ricotta-runtime` on roberta by default and saves the report to `~/Downloads/inspection.png`. Set `ROBERTA_HOST` or `EXPORT` to override the host or a directory/zip under roberta's Downloads; pass `--present-timeout` to override the default 45 seconds.

For the full presentation, inspection, and dish placement cycle, run [run-sorting.sh](scripts/run-sorting.sh) from the Mac. It uses the same host and export defaults and sets the presentation timeout to 45 seconds; the selected placement runs for 45 control seconds without verifying completion. `scripts/run-sorting.sh --dry-run` logs random routing without connecting to hardware.

Both scripts command the robot during a normal run. Before running them, stop competing Studio sessions, put the brick at its demonstrated starting position, clear the inspection area, and supervise the arm.

### Configuration

The task instructions in [sorting.py](src/hackjunction2026_intel/sorting.py) match the relabelled demonstration dataset:

- Presentation: `Move any brick to the center`
- Good placement: `Move the brick to the yellow dish`
- Defective placement: `Move the brick to the orange dish`

Before using hardware:

- If using a different dataset, update the task instructions to match its demonstrations
- Set `PRESENT_DURATION_S` (presentation timeout), `GOOD_DURATION_S`, and `BAD_DURATION_S` (placement durations) to positive, supervised-test values; zero prevents motion
- Set the inspection camera source in `anomaly_detection/config.yaml`, and keep that camera out of the export's `runtime.yaml`: only the inspection code may open it
- Check the export's `runtime.yaml` for the serial port, robot calibration, camera identities/names/settings, control rate, and any safety callbacks; lines marked `# CHANGE_ME` contain machine-specific device paths
- Ensure presentation ends in a state compatible with both placement instructions

The export must contain `runtime.yaml` (generated for model inference, not teleoperation) and `exports/openvino/`. The script always runs on OpenVINO: it points the model at `exports/openvino` inside the export, regardless of the working directory, and forces `backend: openvino` with the OpenVINO device set by `OPENVINO_DEVICE` in `sorting.py` (default `AUTO`, which lets OpenVINO pick the best local accelerator; set `CPU`, `GPU`, or `NPU` to pin one). The device chosen in the Studio export dialog is ignored. It also sets the task and removes any fixed policy name so the export's manifest determines it. All other runtime settings are preserved, and `runtime.yaml` is not rewritten. Other relative paths in the configuration resolve from the working directory; use absolute hardware/calibration paths when necessary.

Each stage creates a fresh runtime from the same export, whose policy source resets the model and action queue. The model is reloaded per stage because the SDK closes the inference model at the end of every `RobotRuntime.run()`. The runtime manages camera capture, observations, inference, and robot commands. On exit, the script requests a hold at the measured joint positions before the runtime disconnects. The calibrated SO-101 follower driver normally keeps torque enabled on disconnect; verify this behavior on your actual setup.

Runtime errors, early stops, and Ctrl+C abort the cycle rather than advancing to inspection or placement. Hold and cleanup are software requests, not guaranteed physical stopping or an emergency stop. Stop competing Studio sessions before running the standalone script.

Durations and the presentation timeout use the SDK's control-loop timing, not a hard wall-clock watchdog; slow inference can extend real elapsed time. Model loading is also outside the timed run, while the 3-second `EMPTY` check runs on wall-clock time from just before the policy loads. Supervise the demo and retain a physical emergency stop. Replace the timed placement with a verified completion check before autonomous use.

### Tests

Run the hardware-free tests from a Python environment using the standard library; robot dependencies are not required:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```
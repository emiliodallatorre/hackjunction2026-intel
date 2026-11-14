# Brick anomaly detection

PatchCore (Anomalib 2.7) anomaly detection for Duplo bricks that the SO-101 arm presents to the inspection camera. PatchCore learns only from **good** bricks; anomaly images are used just to test it and to calibrate the threshold.

```
camera ──collect──▶ data/raw/{good,anomaly,empty}/        full frames, never cropped
                         │ prepare (ROI crop + resize, train/test split)
                         ▼
                    data/dataset/train/good                PatchCore memory bank
                    data/dataset/train/empty               empty-ROI background reference
                    data/dataset/test/{good,anomaly,empty} held out for stats
                         │ train (fit, test, export, calibrate empty detection)
                         ▼
                    models/patchcore/                      model.xml/.bin + metadata.json (ROI, size, threshold)
                                                           + empty_reference.png, presence.json
                         │
                         ▼
                    BrickInspector  ◀── robot code (src/hackjunction2026_intel/anomaly_detection)
```

Each inspection first checks whether there is a brick in the ROI at all: the crop is compared against the median of the empty-ROI images, under 1 ms. Only if a brick is present does PatchCore score it.

## Setup

```bash
cd anomaly_detection
uv sync                      # CPU torch + anomalib + OpenVINO
```

Close Physical AI Studio's camera preview first: only one process can open a camera at a time.

## Workflow

All commands read [`config.yaml`](config.yaml); pass `--config other.yaml` to use another file.

| Step | Command | What it does |
|---|---|---|
| 1. Find the camera | `uv run brick-ad cameras` | Lists `/dev/video*` nodes that deliver frames. The RealSense exposes several nodes; pick the colour one and put its stable `/dev/v4l/by-id/...` path in `camera.source`. |
| 2. Test the stream | `uv run brick-ad stream` | Live preview with FPS and the ROI box. `s` saves a snapshot, `q` quits. |
| 3. Set the crop region | `uv run brick-ad roi` | Have the arm present a brick, press Space to freeze, drag a box (with some margin for pose variation), Enter. Saved to `roi:` in the config. |
| 3b. Lock the camera | `uv run brick-ad lock` | With a brick presented, let auto settle and press Space. It finds fixed focus (sharpest inside the ROI), white balance and exposure/gain that reproduce the auto image, and saves them to `camera:` in the config (about 30 s). `lock --off` returns to automatic. |
| 4. Collect | `uv run brick-ad collect` | `g`/`a`/`e` switch label good/anomaly/empty, Space saves a frame, `b` toggles burst capture (every `collect.burst_interval` s) while the arm rotates the brick. A second window shows the exact crop the model will see. For **empty**, capture ~20-30 frames of the ROI with no brick, in the same setup as inspection (e.g. the open gripper in the inspection pose, if the gripper is in view). |
| 5. Prepare | `uv run brick-ad prepare` | Crops and resizes raw frames into the train/test split. Rerun it after changing the ROI or `image_size`. `--shuffle [--seed N]` holds out a random sample instead of the most recent captures. |
| 6. Train + stats | `uv run brick-ad train` | Fits PatchCore, exports an OpenVINO model to `models/patchcore/`, calibrates empty detection, then prints stats on the test split. |
| 6a. Train on Modal | `uv run --with modal modal run modal_train.py [--gpu L4] [--no-evaluate]` | Same as step 6, but on a Modal GPU: uploads `data/dataset/`, trains there, downloads the model into `models/patchcore/`, then evaluates locally. Log in once with `uv run --with modal modal setup`. |
| 6b. Empty only | `uv run brick-ad fit-empty` | Recalibrates only empty detection (after `prepare`), in seconds, without retraining PatchCore. |
| 7. Validate | `uv run brick-ad evaluate --data <dir> --crop --save heatmaps/` | Stats on any folder with `good/`, `anomaly/` and optionally `empty/` subfolders of raw frames; `--save` writes heatmap overlays named by score. |
| 8. Live check | `uv run brick-ad stream --infer` | Status (EMPTY / GOOD / ANOMALY, uncertain), score and heatmap on every frame. |
| 9. Pipeline demo | `uv run python -m hackjunction2026_intel.anomaly_detection [--show]` | The full decision loop (`decide`): capture, empty → error, uncertain → ask to rotate, else good/defective. |
| Ad-hoc | `uv run brick-ad predict img.png folder/` | Prints status and score per image (`--no-crop` for already-cropped images). |

`evaluate` and `predict` accept `--device CPU|GPU|NPU|AUTO` and `--threshold`.

### Stats

`train` and `evaluate` print:
- The final empty/good/anomaly confusion matrix, as the pipeline would decide.
- Anomaly score ranges per class, the margin between the worst anomaly and the worst good brick, precision/recall/specificity/F1, threshold-free AUROC and the best-F1 threshold.
- How many frames fall in the uncertain band (they would be re-inspected), and how many mistakes fall outside it (confident mistakes).
- Latency, and every misclassified file.

### Data tips

- **Good images are what matters.** Collect 100+ good frames covering every brick colour, size, pose and arm position you expect. Anything the model has not seen as good can be flagged as an anomaly.
- 20-50 anomaly frames are enough to check the model and set the threshold.
- Lock the camera (step 3b) **before** collecting and keep the same lock for live inspection. Auto exposure, white balance and focus react to each brick's colour and to the gripper, so the same brick can look different from frame to frame. PatchCore would treat that as an anomaly. If you re-lock or change the lighting, re-collect the good images.
- Why measured and not just frozen: this camera reports stale values while auto is on (e.g. exposure 166 when it really uses about 2000), so "freeze what auto picked" cannot be saved. `lock` searches for values that reproduce the auto image instead.
- If `lock` warns that the image is too dark, add light at the inspection point. Raising `camera.max_exposure` also works, but costs frame rate and adds motion blur.
- The locked values live on the camera until it is unplugged or another program changes them. Any `brick-ad` command re-applies the config when it opens the camera. If your pipeline grabs frames another way, open the camera with `hackjunction2026_intel.anomaly_detection.InspectionStation` so it gets the same settings.
- The test split holds out the **most recent** good captures (not a random sample), so near-duplicate burst frames don't inflate the stats. Collecting the last few good bricks in a separate session gives an honest unseen-brick test.
- The threshold is calibrated on the test split itself (there is too little data for a separate validation split), so confirm it on a fresh capture session with `evaluate --data ... --crop`.

## Using it in the pipeline

`BrickInspector` only needs OpenVINO, OpenCV and NumPy. It applies the same ROI crop and resize used for training, which are stored in `models/patchcore/metadata.json`.

```python
from hackjunction2026_intel.anomaly_detection import BrickInspector

inspector = BrickInspector("anomaly_detection/models/patchcore", device="CPU")  # or "GPU", "NPU"

result = inspector.predict(frame)        # full BGR frame from the inspection camera
result.status                            # "empty" | "good" | "anomaly"
result.uncertain                         # score within inference.uncertain_margin of the threshold
result.score                             # 0..1, trained threshold = 0.5 (None when empty)
result.presence                          # fraction of ROI pixels that differ from the empty background
result.anomaly_map, result.patch         # for display/logging
```

The frame must be what the training images were: a full, uncropped frame at the configured resolution, in BGR order (convert RGB frames from other libraries with `cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)`), from the camera with the same locked settings, and taken once the brick is still.

`decide()` turns the views of one brick into the robot action:

```python
from hackjunction2026_intel.anomaly_detection import BrickInspector, Verdict, decide

looks = [inspector.predict(frame)]
decision = decide(looks, inspector.threshold, max_looks=3)
while decision.verdict is Verdict.ROTATE:
    rotate_wrist()                                   # your robot code; wait until still
    looks.append(inspector.predict(grab_frame()))
    decision = decide(looks, inspector.threshold, max_looks=3)
# decision.verdict: EMPTY -> re-grasp, GOOD -> good bowl, DEFECTIVE -> defective bowl
```

| Situation | Verdict | Message |
|---|---|---|
| Nothing in the ROI | `EMPTY` | `ERROR: no brick in the inspection area (failed pick?)` |
| Score within the uncertain band, fewer than `max_looks` views | `ROTATE` | `UNSURE (score 0.48): rotate the brick and inspect again` |
| Otherwise | `GOOD` / `DEFECTIVE` | Decided on the **highest score over all views**: a defect only needs to be visible from one side. |

`python -m hackjunction2026_intel.anomaly_detection` runs this loop through `InspectionStation` (below), with keyboard prompts in place of the robot.

Tune `inference.uncertain_margin` in the config from the "Uncertain" section of `evaluate`: aim for most mistakes inside the band and few good bricks asked to rotate.

### In the robot code: `InspectionStation`

`InspectionStation` opens the inspection camera itself, with the resolution, pixel format and locked focus / exposure / gain / white balance from `config.yaml` (the settings the model was trained with), and loads the model. It refuses to start if the camera does not deliver the configured resolution, since the ROI is in camera pixels.

```python
from hackjunction2026_intel.anomaly_detection import InspectionStation, Verdict

station = InspectionStation()                     # once at startup; config: $BRICK_AD_CONFIG or anomaly_detection/config.yaml

def rotate_wrist(decision):                        # your robot code: show another side, return once still
    ...

decision = station.inspect(rotate=rotate_wrist)   # per brick, once it is presented and still
# decision.verdict: EMPTY -> re-grasp, GOOD -> good bowl, DEFECTIVE -> defective bowl
```

Without `rotate`, `inspect()` returns the `ROTATE` decision so the caller can rotate and continue with `station.decide(decision.looks + [station.look()])`. `station.grab()` returns a raw frame for logging.

The inspection camera must be opened **only** by this code: do not add it as a camera in Physical AI Studio (a V4L2 camera streams to one process, and opening it elsewhere resets its controls).

### In a robot container

The package's required dependencies are only OpenVINO, OpenCV, NumPy and PyYAML; torch and anomalib come with the `brick-ad` tools in the dev group, which `uv sync` installs but `pip` skips.

```dockerfile
COPY pyproject.toml README.md /opt/hackjunction2026_intel/
COPY src /opt/hackjunction2026_intel/src
RUN pip install /opt/hackjunction2026_intel          # ~400 MB (OpenVINO + OpenCV), no torch
# If the image already has OpenVINO and an OpenCV (e.g. opencv-python-headless), use
#   pip install --no-deps /opt/hackjunction2026_intel
# to keep its versions and avoid two packages providing cv2.
```

Mount the config and the model rather than baking them in, so `brick-ad roi`/`lock`/`train` on the host take effect on the next restart. Paths in the config resolve against its folder, so keep the same layout:

```yaml
environment:
  BRICK_AD_CONFIG: /brick/config.yaml
volumes:
  - ./anomaly_detection/config.yaml:/brick/config.yaml:ro
  - ./anomaly_detection/models/patchcore:/brick/models/patchcore:ro
  - /dev/v4l:/dev/v4l:ro          # the /dev/v4l/by-id/... camera path in the config
devices:
  - /dev/video0                   # the node that by-id path points to (`ls -l /dev/v4l/by-id/`); not needed if privileged
group_add:
  - "${VIDEO_GID:-video}"
```

A read-only model mount is fine (the OpenVINO compile cache is then skipped). For `inference.device: GPU` or `NPU` the container also needs `/dev/dri` or `/dev/accel` and the matching group, as the Physical AI Studio service in `infra/docker-compose.yml` already has.

## Notes

- Torch is installed as a CPU build. PatchCore only extracts features (no backprop), so training takes about 2 minutes for ~50 images on CPU. To try the Intel GPU for training, switch the torch index to `https://download.pytorch.org/whl/xpu` and set `train.accelerator: xpu`.
- The model is exported with a static `[1, 3, image_size, image_size]` input: anomalib 2.7's dynamic-batch export of PatchCore is broken, and the NPU needs static shapes anyway.
- `anomalib` depends on `opencv-python-headless`; `pyproject.toml` replaces it with `opencv-python` so the preview windows work.
- Each training run's anomalib output (including prediction images for the test split) is under `runs/Patchcore/duplo/`.

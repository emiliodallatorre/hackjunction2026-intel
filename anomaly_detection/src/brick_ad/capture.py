"""Interactive camera tools: live preview, ROI selection and dataset collection."""

import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from hackjunction2026_intel.anomaly_detection.camera import Camera
from .config import ANOMALY, EMPTY, GOOD, LABELS, Config, save_roi
from .imaging import draw_roi, list_images, overlay_heatmap, preprocess, put_text, read_image

WINDOW, CROP_WINDOW = "camera", "crop"
QUIT_KEYS = {ord("q"), 27}  # q, Esc


def _save(image: np.ndarray, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{datetime.now():%Y%m%d-%H%M%S-%f}.png"
    cv2.imwrite(str(path), image)
    return path


def stream(cfg: Config, model_dir: Path | None = None, heatmap: bool = False) -> None:
    """Live preview with FPS and ROI. With a model, also shows status, score and heatmap per frame;
    ``heatmap`` also overlays it on the ROI in the main view.

    Keys: s = save snapshot to data/snapshots, q/Esc = quit.
    """
    inspector = None
    if model_dir is not None:
        from hackjunction2026_intel.anomaly_detection.inspector import BrickInspector

        inspector = BrickInspector(model_dir, device=cfg.inference.device,
                                   uncertain_margin=cfg.inference.uncertain_margin)
        roi = inspector.roi  # always show the ROI the model was trained with
        if inspector.presence is None:
            print("Note: no empty-ROI reference in this model, so EMPTY is never shown.")
    else:
        roi = cfg.roi

    with Camera(cfg.camera) as cam:
        print(f"Camera {cfg.camera.source!r} at {cam.resolution[0]}x{cam.resolution[1]}. s = snapshot, q = quit")
        print(f"Controls: {cam.describe_controls()}")
        last, fps = time.perf_counter(), 0.0
        while True:
            frame = cam.read()
            view = frame.copy()
            draw_roi(view, roi)
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now
            put_text(view, f"{fps:4.1f} FPS  {frame.shape[1]}x{frame.shape[0]}  {cam.describe_controls()}")

            if inspector is not None:
                t0 = time.perf_counter()
                pred = inspector.predict(frame)
                ms = (time.perf_counter() - t0) * 1000
                presence = "" if pred.presence is None else f"  presence {pred.presence:.0%}"
                if pred.is_empty:
                    color, label = (200, 200, 200), "EMPTY"
                    text = f"EMPTY (no brick){presence}"
                    crop_view = pred.patch.copy()
                else:
                    color = (0, 165, 255) if pred.uncertain else (0, 0, 255) if pred.is_anomaly else (0, 255, 0)
                    label = f"{pred.status.upper()}{'?' if pred.uncertain else ''} {pred.score:.2f}"
                    text = f"{pred.status.upper()}{' (uncertain)' if pred.uncertain else ''}  score {pred.score:.3f}"
                    text += f"{presence}  ({ms:.0f} ms)"
                    crop_view = overlay_heatmap(pred.patch, pred.anomaly_map)
                    if heatmap and roi is not None:
                        x, y, w, h = roi
                        view[y : y + h, x : x + w] = overlay_heatmap(view[y : y + h, x : x + w], pred.anomaly_map)
                put_text(crop_view, label, 0, color)
                cv2.imshow(CROP_WINDOW, crop_view)
                put_text(view, text, 1, color)
                draw_roi(view, roi, color)
            elif roi is not None:
                cv2.imshow(CROP_WINDOW, preprocess(frame, roi, cfg.model.image_size))

            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEYS:
                break
            if key == ord("s"):
                print("saved", _save(frame, cfg.paths.raw.parent / "snapshots"))
    cv2.destroyAllWindows()


def select_roi(cfg: Config, image_path: Path | None = None) -> None:
    """Draw the crop region on a frozen frame and write it to config.yaml.

    Live mode: position a brick with the arm, press Space to freeze, drag a box,
    Enter to confirm (c to cancel). Use ``image_path`` to draw on a saved frame instead.
    """
    if image_path is not None:
        frame = read_image(image_path)
    else:
        with Camera(cfg.camera) as cam:
            print("Present a brick, then press Space to freeze the frame (q to abort)")
            while True:
                frame = cam.read()
                view = frame.copy()
                draw_roi(view, cfg.roi, (0, 200, 255))
                put_text(view, "Space = freeze and draw ROI, q = abort")
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(1) & 0xFF
                if key in QUIT_KEYS:
                    cv2.destroyAllWindows()
                    return
                if key == ord(" "):
                    break

    print("Drag the crop box around the brick (leave some margin), Enter to confirm, c to cancel")
    x, y, w, h = cv2.selectROI(WINDOW, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()
    if w == 0 or h == 0:
        print("No ROI selected, config unchanged")
        return
    save_roi(cfg, (int(x), int(y), int(w), int(h)))
    print(f"Saved roi: [{x}, {y}, {w}, {h}] to {cfg.file}")
    if abs(w - h) > 0.1 * max(w, h):
        print("Note: the crop is resized to a square, so a roughly square ROI avoids distortion.")
    print("Raw frames are kept uncropped: rerun `brick-ad prepare` after changing the ROI.")


def collect(cfg: Config, label: str = GOOD) -> None:
    """Save full camera frames into data/raw/<label>/.

    Keys: g / a / e = switch label to good / anomaly / empty (nothing in the ROI),
    Space = save one frame, b = toggle burst mode (saves every ``collect.burst_interval`` s
    while the arm rotates the brick), q/Esc = quit.
    """
    colors = {GOOD: (0, 255, 0), ANOMALY: (0, 0, 255), EMPTY: (200, 200, 200)}
    keys = {ord("g"): GOOD, ord("a"): ANOMALY, ord("e"): EMPTY}
    counts = {lbl: len(list_images(cfg.paths.raw / lbl)) for lbl in LABELS}
    burst, last_save = False, 0.0
    with Camera(cfg.camera) as cam:
        print(collect.__doc__)
        if all(v is None for v in cfg.camera.locked.values()):
            print("Warning: camera controls are automatic. Run `brick-ad lock` first so every image "
                  "(and later the live inspection) uses the same focus, exposure and white balance.")
        while True:
            frame = cam.read()
            now = time.monotonic()
            save = False
            key = cv2.waitKey(1) & 0xFF
            if key in QUIT_KEYS:
                break
            if key in keys:
                label, burst = keys[key], False
            elif key == ord("b"):
                burst = not burst
            elif key == ord(" "):
                save = True
            if burst and now - last_save >= cfg.collect.burst_interval:
                save = True
            if save:
                _save(frame, cfg.paths.raw / label)
                counts[label] += 1
                last_save = now

            view = frame.copy()
            color = colors[label]
            draw_roi(view, cfg.roi, color)
            put_text(view, f"label: {label.upper()}   {'BURST ON' if burst else ''}", 0, color)
            put_text(view, "   ".join(f"{lbl}: {n}" for lbl, n in counts.items()), 1)
            put_text(view, "g/a/e = good/anomaly/empty  space = save  b = burst  q = quit", 2)
            put_text(view, cam.describe_controls(), 3)
            if save:
                cv2.rectangle(view, (0, 0), (view.shape[1] - 1, view.shape[0] - 1), (255, 255, 255), 8)
            cv2.imshow(WINDOW, view)
            if cfg.roi is not None:
                cv2.imshow(CROP_WINDOW, preprocess(frame, cfg.roi, cfg.model.image_size))
    cv2.destroyAllWindows()
    print(f"Raw images in {cfg.paths.raw}: " + ", ".join(f"{lbl}={n}" for lbl, n in counts.items()))

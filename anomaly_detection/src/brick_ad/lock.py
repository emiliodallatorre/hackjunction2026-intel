"""Lock focus, exposure and white balance to fixed values and save them in the config.

UVC cameras do not reliably report the values their auto modes pick (this webcam
reports the defaults while auto is on). So instead of reading them, this lets auto
settle on the scene and then searches for fixed values that reproduce that image:

    focus          sweep, keep the sharpest setting inside the ROI (the brick)
    white balance  bisect the temperature to match the auto colour balance (blue/red)
    exposure       bisect to match the auto brightness, capped at camera.max_exposure;
                   gain makes up any remaining difference
"""

import dataclasses
import time
from collections.abc import Callable

import cv2
import numpy as np

from hackjunction2026_intel.anomaly_detection.camera import Camera
from .config import CAMERA_CONTROLS, Config, save_camera_controls
from .imaging import crop, draw_roi, put_text

WINDOW = "camera"


def _frame(cam: Camera, n: int = 4) -> np.ndarray:
    """Drop a few frames so the new setting is applied, then return the latest."""
    for _ in range(n):
        frame = cam.read()
    cv2.imshow(WINDOW, frame)
    cv2.waitKey(1)
    return frame


def _brightness(frame: np.ndarray) -> float:
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def _blue_red(frame: np.ndarray) -> float:
    b, _, r = frame.reshape(-1, 3).mean(axis=0)
    return float(b / max(r, 1e-6))


def _sharpness(frame: np.ndarray, roi) -> float:
    gray = cv2.cvtColor(crop(frame, roi), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _bisect(cam: Camera, name: str, lo: int, hi: int, measure: Callable[[np.ndarray], float],
            target: float, increasing: bool) -> tuple[int, float]:
    """Find the value of a monotonic control whose measurement is closest to target."""
    best = (float("inf"), lo, 0.0)
    while lo <= hi:
        mid = (lo + hi) // 2
        cam.set_value(name, mid)
        value = measure(_frame(cam))
        best = min(best, (abs(value - target), mid, value))
        if (value < target) == increasing:
            lo = mid + 1
        else:
            hi = mid - 1
    return best[1], best[2]


def _sweep_focus(cam: Camera, roi, lo: int, hi: int) -> int:
    def score(value: int) -> float:
        cam.set_value("focus", value)
        time.sleep(0.1)  # let the lens move
        return _sharpness(_frame(cam, 3), roi)

    coarse = max(1, (hi - lo) // 32)
    best = max(range(lo, hi + 1, coarse), key=score)
    fine = max(1, coarse // 8)
    return max(range(max(lo, best - coarse), min(hi, best + coarse) + 1, fine), key=score)


def lock(cfg: Config) -> None:
    auto_cfg = dataclasses.replace(cfg.camera, exposure=None, white_balance=None, focus=None)
    with Camera(auto_cfg) as cam:
        print("Auto mode. Have the arm present a brick in the ROI, wait for the image to settle,\n"
              "then press Space to lock (q to abort).")
        while True:
            frame = cam.read()
            view = frame.copy()
            draw_roi(view, cfg.roi)
            put_text(view, "AUTO - Space = lock focus/exposure/white balance, q = abort")
            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
            if key == ord(" "):
                break

        frames = [cam.read() for _ in range(5)]
        target_brightness = float(np.mean([_brightness(f) for f in frames]))
        target_blue_red = float(np.mean([_blue_red(f) for f in frames]))
        print(f"Auto image: brightness {target_brightness:.1f}, blue/red {target_blue_red:.3f}. Searching...")

        # Freeze everything first so each search runs on a stable image.
        supported = {name: cam.set_auto(name, False) for name in ("focus", "white_balance", "exposure")}
        values: dict[str, int | None] = {}

        if supported["focus"]:
            lo, hi = cam.value_range("focus")
            values["focus"] = _sweep_focus(cam, cfg.roi, lo, hi)
            cam.set_value("focus", values["focus"])
            print(f"  focus         {values['focus']}  (range {lo}-{hi}, sharpest in {'ROI' if cfg.roi else 'frame'})")
        else:
            print("  focus         not adjustable on this camera")

        if supported["white_balance"]:
            lo, hi = cam.value_range("white_balance")
            values["white_balance"], got = _bisect(cam, "white_balance", lo, hi, _blue_red,
                                                   target_blue_red, increasing=False)
            cam.set_value("white_balance", values["white_balance"])
            print(f"  white_balance {values['white_balance']} K  (blue/red {got:.3f})")
        else:
            print("  white_balance not adjustable on this camera")

        if supported["exposure"]:
            lo, hi = cam.value_range("exposure")
            hi = max(lo, min(hi, cfg.camera.max_exposure))
            has_gain = cam.set_value("gain", 0)
            values["exposure"], got = _bisect(cam, "exposure", lo, hi, _brightness,
                                              target_brightness, increasing=True)
            cam.set_value("exposure", values["exposure"])
            values["gain"] = 0 if has_gain else None
            if has_gain and got < 0.95 * target_brightness:
                g_lo, g_hi = cam.value_range("gain")
                values["gain"], got = _bisect(cam, "gain", g_lo, g_hi, _brightness,
                                              target_brightness, increasing=True)
            print(f"  exposure      {values['exposure']} ({values['exposure'] / 10:.1f} ms), gain {values['gain']}"
                  f"  (brightness {got:.1f} vs auto {target_brightness:.1f})")
            if got < 0.85 * target_brightness:
                print("  Warning: too dark even at max exposure and gain. Add light, or raise camera.max_exposure "
                      "(lower fps, more motion blur).")
        else:
            print("  exposure      not adjustable on this camera")

        save_camera_controls(cfg, {k: v for k, v in values.items() if k in CAMERA_CONTROLS})
        cam.cfg = cfg.camera
        print(f"Saved to {cfg.file}: {cam.describe_controls()}")
        print("Showing the locked image. Any key to close.")
        while True:
            view = cam.read()
            draw_roi(view, cfg.roi)
            put_text(view, f"LOCKED  {cam.describe_controls()}")
            cv2.imshow(WINDOW, view)
            if cv2.waitKey(1) != -1:
                break
    cv2.destroyAllWindows()


def unlock(cfg: Config) -> None:
    save_camera_controls(cfg, dict.fromkeys(CAMERA_CONTROLS))
    with Camera(cfg.camera) as cam:  # opening applies the config: auto modes back on
        cam.set_value("gain", cam.value_range("gain")[0])  # gain has no auto mode: reset to minimum
    print(f"Camera controls set back to automatic in {cfg.file}")

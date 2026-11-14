"""Thin OpenCV camera wrapper that applies the config's resolution and locked controls.

The RealSense D415 colour stream is a plain V4L2 device.
"""

import os
import platform

import cv2
import numpy as np

from .config import CameraConfig

# control -> (auto switch property, auto on, auto off, value property). V4L2 auto_exposure
# menu: 1 = manual, 3 = aperture priority (auto). Gain has no auto switch.
AUTO_CONTROLS = {
    "exposure": (cv2.CAP_PROP_AUTO_EXPOSURE, 3, 1, cv2.CAP_PROP_EXPOSURE),
    "white_balance": (cv2.CAP_PROP_AUTO_WB, 1, 0, cv2.CAP_PROP_WB_TEMPERATURE),
    "focus": (cv2.CAP_PROP_AUTOFOCUS, 1, 0, cv2.CAP_PROP_FOCUS),
}
VALUE_PROPS = {name: c[3] for name, c in AUTO_CONTROLS.items()} | {"gain": cv2.CAP_PROP_GAIN}


class Camera:
    def __init__(self, cfg: CameraConfig):
        self.cfg = cfg
        source = int(cfg.source) if str(cfg.source).isdigit() else str(cfg.source)
        backend = cv2.CAP_V4L2 if platform.system() == "Linux" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(source, backend)
        if not self.cap.isOpened():
            if isinstance(source, str) and not os.path.exists(source):
                raise RuntimeError(f"Camera {source!r} is not connected. Plug it in, or run `brick-ad cameras`.")
            raise RuntimeError(
                f"Cannot open camera {cfg.source!r}. Is it in use by another process "
                "(e.g. Physical AI Studio)? Try `brick-ad cameras`."
            )
        if cfg.fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always hand out the freshest frame
        self.read()  # start streaming before touching controls (some UVC cameras reset them)
        self.apply_controls()

    def apply_controls(self) -> None:
        """Lock the controls set in the config; put the others back on automatic."""
        for name in AUTO_CONTROLS:
            value = getattr(self.cfg, name)
            if value is None:
                self.set_auto(name, True)
            elif not self.set_manual(name, value):
                print(f"Warning: camera {self.cfg.source!r} does not support manual {name}")
        if self.cfg.gain is not None and not self.set_value("gain", self.cfg.gain):
            print(f"Warning: camera {self.cfg.source!r} does not support gain")

    def set_auto(self, name: str, on: bool) -> bool:
        """Switch a control's automatic mode. Turning auto off freezes the current setting."""
        prop, auto_on, auto_off, _ = AUTO_CONTROLS[name]
        return self.cap.set(prop, auto_on if on else auto_off)

    def set_value(self, name: str, value: float) -> bool:
        return self.cap.set(VALUE_PROPS[name], value)

    def get_value(self, name: str) -> int:
        """Only reliable in manual mode: this driver reports stale values while auto is on."""
        return int(self.cap.get(VALUE_PROPS[name]))

    def set_manual(self, name: str, value: float) -> bool:
        return self.set_auto(name, False) and self.set_value(name, value)

    def value_range(self, name: str) -> tuple[int, int]:
        """Find a control's range: V4L2 drivers clamp out-of-range values (manual mode only).

        Probes the minimum with 0 because OpenCV wraps negative values to the maximum;
        fine for exposure, gain, white balance and focus, which are never negative.
        """
        self.set_value(name, 0)
        low = self.get_value(name)
        self.set_value(name, 1_000_000)
        return low, self.get_value(name)

    def describe_controls(self) -> str:
        return "  ".join(f"{k}={'auto' if v is None else v}" for k, v in self.cfg.locked.items())

    @property
    def resolution(self) -> tuple[int, int]:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Camera {self.cfg.source!r} returned no frame")
        return frame

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


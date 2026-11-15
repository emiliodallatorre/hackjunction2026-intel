"""Read-only taps on the inspection station, installed in the runner process only.

``run_model.run_cycle`` imports ``InspectionStation`` from ``hackjunction2026_intel.anomaly_detection``
when it runs, so replacing that attribute with ``ObservedStation`` lets the dashboard see what the
station sees without changing the workflow: every method returns exactly what the original returns.
"""

import base64
import logging
import time
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from hackjunction2026_intel import anomaly_detection
from hackjunction2026_intel.anomaly_detection.imaging import overlay_heatmap

from . import events

logger = logging.getLogger(__name__)

FRAME_FPS = 10  # live frames sent per second, at most
FRAME_WIDTH = 640  # live frames are downscaled to this width
VIEW_SIZE = 320  # side of the crop / heatmap images
JPEG_QUALITY = 75
ROI_COLOR = (44, 199, 255)  # BGR of the dashboard's accent yellow


def jpeg(image: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return base64.b64encode(buffer).decode("ascii") if ok else ""


def _safely(publish: Callable[..., None], *args: Any) -> None:
    try:
        publish(*args)
    except Exception:
        logger.debug("Dashboard publish failed", exc_info=True)


def _views(prediction: Any) -> tuple[str, str]:
    """(crop, crop with heatmap) as JPEGs; the heatmap is absent when the ROI was empty."""
    crop = cv2.resize(prediction.patch, (VIEW_SIZE, VIEW_SIZE))
    if prediction.anomaly_map is None:
        return jpeg(crop), ""
    return jpeg(crop), jpeg(overlay_heatmap(crop, prediction.anomaly_map))


class FrameTap:
    """Stands in for the station's camera and publishes a downscaled copy of the frames read."""

    def __init__(self, camera: Any, reporter: Any, roi: tuple[int, int, int, int] | None):
        self._camera = camera
        self._reporter = reporter
        self._roi = roi
        self._last = 0.0

    def read(self) -> np.ndarray:
        frame = self._camera.read()
        now = time.monotonic()
        if now - self._last >= 1 / FRAME_FPS:
            self._last = now
            _safely(self._publish, frame)
        return frame

    def _publish(self, frame: np.ndarray) -> None:
        scale = FRAME_WIDTH / frame.shape[1]
        view = cv2.resize(frame, (FRAME_WIDTH, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        if self._roi is not None:
            x, y, w, h = (round(v * scale) for v in self._roi)
            cv2.rectangle(view, (x, y), (x + w, y + h), ROI_COLOR, 2, cv2.LINE_AA)
        self._reporter.send(events.FRAME, jpeg=jpeg(view))

    def __getattr__(self, name: str) -> Any:  # resolution, close, ...
        return getattr(self._camera, name)


class ObservedStation(anomaly_detection.InspectionStation):
    reporter: Any = None  # set by install()

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.camera = FrameTap(self.camera, self.reporter, self.inspector.roi)

    def look(self) -> Any:
        self.reporter.stage(events.INSPECT)
        prediction = super().look()
        _safely(self._publish_look, prediction)
        return prediction

    def inspect(self, rotate: Any = None) -> Any:
        decision = super().inspect(rotate)
        _safely(self._publish_decision, decision)
        return decision

    def _publish_look(self, prediction: Any) -> None:
        _, overlay = _views(prediction)
        self.reporter.send(
            events.LOOK, status=prediction.status, score=prediction.score,
            uncertain=prediction.uncertain, overlay=overlay,
        )

    def _publish_decision(self, decision: Any) -> None:
        crop, overlay = _views(decision.looks[-1])
        self.reporter.send(
            events.DECISION, verdict=decision.verdict.value, message=decision.message,
            score=decision.score, looks=len(decision.looks), thumbnail=crop, overlay=overlay,
        )


def install(reporter: Any) -> None:
    ObservedStation.reporter = reporter
    anomaly_detection.InspectionStation = ObservedStation

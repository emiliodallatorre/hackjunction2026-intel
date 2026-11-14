"""`brick-ad cameras`: find the camera device to put in camera.source."""

import glob
import os

import cv2

from hackjunction2026_intel.anomaly_detection.camera import Camera

from .config import CameraConfig


def probe_cameras() -> list[tuple[str, tuple[int, int], str | None]]:
    """List video nodes that deliver frames: (node, resolution, stable /dev/v4l/by-id path).

    UVC cameras also create metadata nodes that cannot capture; those are skipped.
    Probing opens each camera, which resets its controls to automatic until a
    ``brick-ad`` command opens it again with the config.
    """
    by_id = {os.path.realpath(p): p for p in glob.glob("/dev/v4l/by-id/*")}
    sources = sorted(glob.glob("/dev/video*"), key=lambda s: int(s.removeprefix("/dev/video")))
    sources = sources or [str(i) for i in range(10)]
    level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)  # hide warnings from metadata nodes
    results = []
    try:
        for source in sources:
            try:
                with Camera(CameraConfig(source=source, fourcc=None)) as cam:
                    frame = cam.read()
            except RuntimeError:
                continue
            results.append((source, (frame.shape[1], frame.shape[0]), by_id.get(source)))
    finally:
        cv2.utils.logging.setLogLevel(level)
    return results

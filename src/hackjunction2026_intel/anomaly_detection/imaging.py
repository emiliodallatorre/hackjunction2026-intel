"""Image helpers shared by dataset preparation and inference.

``preprocess`` is the single place where a camera frame becomes model input, so the
training crops (`brick-ad prepare`) and the live crops always go through exactly the same steps.
"""

from pathlib import Path

import cv2
import numpy as np

from .config import Roi


def crop(image: np.ndarray, roi: Roi | None) -> np.ndarray:
    if roi is None:
        return image
    x, y, w, h = roi
    height, width = image.shape[:2]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError(f"ROI {roi} does not fit in a {width}x{height} image")
    return image[y : y + h, x : x + w]


def preprocess(image_bgr: np.ndarray, roi: Roi | None, size: int) -> np.ndarray:
    """Crop to the ROI and resize to ``size`` x ``size``. Returns a BGR uint8 image."""
    patch = crop(image_bgr, roi)
    if patch.shape[:2] != (size, size):
        patch = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)
    return patch


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image {path}")
    return image


def overlay_heatmap(image_bgr: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] anomaly map onto the image as a JET heatmap (for display/logging)."""
    amap = cv2.resize(anomaly_map.astype(np.float32), image_bgr.shape[1::-1])
    heat = cv2.applyColorMap((np.clip(amap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 1 - alpha, heat, alpha, 0)


def put_text(image: np.ndarray, text: str, line: int = 0, color=(255, 255, 255)) -> None:
    org = (10, 28 + 28 * line)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

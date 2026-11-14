"""File and drawing helpers for the tools; the frame -> model input steps come from the runtime package."""

from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from hackjunction2026_intel.anomaly_detection.imaging import (  # noqa: F401  (re-exported for the tools)
    crop,
    overlay_heatmap,
    preprocess,
    put_text,
    read_image,
)

from .config import Roi

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def expand_paths(paths: Iterable[str | Path]) -> list[Path]:
    """Expand folders into the images they contain (recursively)."""
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            out += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
        else:
            out.append(p)
    return out


def draw_roi(image: np.ndarray, roi: Roi | None, color=(0, 255, 0)) -> None:
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

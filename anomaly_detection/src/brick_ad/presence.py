"""Calibrate the runtime empty-ROI detector from labelled crops and save it next to the model."""

import json
from pathlib import Path

import cv2
import numpy as np

from hackjunction2026_intel.anomaly_detection.presence import (  # noqa: F401  (re-exported for the tools)
    PRESENCE_FILE,
    REFERENCE_FILE,
    PresenceDetector,
)


def fit_presence(empty: list[np.ndarray], bricks: list[np.ndarray],
                 pixel_delta: int = 25, normalize: bool = True) -> tuple[PresenceDetector, dict]:
    """Build the reference from empty crops and put the threshold between the two classes."""
    if not empty or not bricks:
        raise ValueError("Need both empty and brick crops to calibrate presence detection")
    reference = np.median(np.stack(empty), axis=0).astype(np.uint8)
    detector = PresenceDetector(reference, 0.0, pixel_delta, normalize)
    empty_scores = np.array([detector.score(p) for p in empty])
    brick_scores = np.array([detector.score(p) for p in bricks])
    if brick_scores.min() > empty_scores.max():
        detector.threshold = float((empty_scores.max() + brick_scores.min()) / 2)
    else:  # overlap: take the threshold with the fewest mistakes
        candidates = np.unique(np.concatenate([empty_scores, brick_scores]))
        errors = [(empty_scores >= t).sum() + (brick_scores < t).sum() for t in candidates]
        detector.threshold = float(candidates[int(np.argmin(errors))])
    stats = {
        "threshold": detector.threshold,
        "pixel_delta": pixel_delta,
        "normalize": normalize,
        "n_empty": len(empty),
        "n_bricks": len(bricks),
        "empty_max": float(empty_scores.max()),
        "brick_min": float(brick_scores.min()),
    }
    return detector, stats


def save_presence(detector: PresenceDetector, model_dir: Path, stats: dict) -> None:
    """Write the files PresenceDetector.load() reads."""
    cv2.imwrite(str(model_dir / REFERENCE_FILE), detector.reference)
    (model_dir / PRESENCE_FILE).write_text(json.dumps(stats, indent=2))

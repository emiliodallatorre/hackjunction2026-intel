"""Is there a brick in the ROI? Compare the crop with a reference image of the empty ROI.

With the camera locked and a fixed background, an empty ROI looks nearly identical
every time, so the fraction of pixels that clearly differ from the median empty crop
separates "empty" from "brick present" in well under a millisecond. Colour is not used
on its own because light bricks on a wooden table have little colour contrast.
The reference and threshold are calibrated by `brick-ad train` / `brick-ad fit-empty`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REFERENCE_FILE = "empty_reference.png"
PRESENCE_FILE = "presence.json"


def _blur(patch: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(patch, (5, 5), 0).astype(np.int16)


@dataclass
class PresenceDetector:
    reference: np.ndarray  # BGR uint8 crop of the empty ROI, at model resolution
    threshold: float  # presence score at or above which a brick is present
    pixel_delta: int = 25  # per-pixel change (0-255, any channel) counted as "different"
    # Match the crop's overall colour to the reference first (per-channel gain from the medians),
    # so a change of light (e.g. warmer lamp) is not mistaken for a brick. A brick is a local
    # change and covers well under half the ROI, so the medians stay on the background.
    normalize: bool = False

    def score(self, patch: np.ndarray) -> float:
        """Fraction of ROI pixels that differ from the empty background (0 = identical)."""
        if self.normalize:
            ref_median = np.median(self.reference.reshape(-1, 3), axis=0)
            gain = ref_median / np.maximum(np.median(patch.reshape(-1, 3), axis=0), 1)
            patch = np.clip(patch.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        diff = np.abs(_blur(patch) - _blur(self.reference)).max(axis=2)
        return float((diff > self.pixel_delta).mean())

    def is_present(self, patch: np.ndarray) -> bool:
        return self.score(patch) >= self.threshold

    @classmethod
    def load(cls, model_dir: Path) -> "PresenceDetector | None":
        """Returns None for models trained without empty images."""
        ref_file, meta_file = model_dir / REFERENCE_FILE, model_dir / PRESENCE_FILE
        if not (ref_file.exists() and meta_file.exists()):
            return None
        meta = json.loads(meta_file.read_text())
        # Models calibrated before colour matching existed have no "normalize" key.
        return cls(cv2.imread(str(ref_file)), meta["threshold"], meta["pixel_delta"], meta.get("normalize", False))

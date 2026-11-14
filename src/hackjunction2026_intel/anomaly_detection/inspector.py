"""Runtime anomaly inspector: the class to import from the robot pipeline.

Only needs OpenVINO, OpenCV and NumPy (no torch / anomalib), so it loads fast and runs
on the laptop CPU, GPU or NPU.

    from hackjunction2026_intel.anomaly_detection import BrickInspector

    inspector = BrickInspector("anomaly_detection/models/patchcore", device="CPU")
    result = inspector.predict(frame)          # full BGR camera frame, ROI crop applied
    result.status                              # "empty" | "good" | "anomaly"
    result.uncertain                           # score close to the threshold

See decision.py for turning predictions into an action, and station.py / __main__.py
for a complete capture -> decision loop.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import openvino as ov

from .imaging import Roi, preprocess, read_image
from .presence import PresenceDetector

EMPTY, GOOD, ANOMALY = "empty", "good", "anomaly"


@dataclass(frozen=True)
class Prediction:
    status: str  # "empty" | "good" | "anomaly"
    score: float | None  # normalised anomaly score in [0, 1], threshold at 0.5; None when empty
    uncertain: bool  # brick present and score within uncertain_margin of the threshold
    presence: float | None  # fraction of ROI pixels differing from the empty background; None if not calibrated
    anomaly_map: np.ndarray | None  # (H, W) float32 in [0, 1] at model resolution; None when empty
    patch: np.ndarray  # the BGR crop that was analysed, for display/logging

    @property
    def is_empty(self) -> bool:
        return self.status == EMPTY

    @property
    def is_anomaly(self) -> bool:
        return self.status == ANOMALY


class BrickInspector:
    def __init__(
        self,
        model_dir: str | Path,
        device: str = "CPU",
        threshold: float | None = None,
        uncertain_margin: float = 0.1,
    ):
        """
        Args:
            model_dir: Folder written by ``brick-ad train`` (model.xml/.bin + metadata.json,
                plus the empty-ROI reference when empty images were collected).
            device: OpenVINO device: "CPU", "GPU", "NPU" or "AUTO".
            threshold: Decision threshold on the normalised score. ``None`` uses the
                threshold learned during training (0.5 on the normalised scale).
            uncertain_margin: Scores within this distance of the threshold are flagged uncertain.
        """
        self.model_dir = Path(model_dir)
        meta = json.loads((self.model_dir / "metadata.json").read_text())
        self.image_size: int = meta["image_size"]
        self.roi: Roi | None = tuple(meta["roi"]) if meta["roi"] else None
        self.threshold = meta["threshold"] if threshold is None else threshold
        self.uncertain_margin = uncertain_margin
        self.metadata = meta
        self.presence = PresenceDetector.load(self.model_dir)  # None: empty detection disabled

        core = ov.Core()
        if os.access(self.model_dir, os.W_OK):  # skip the cache when the model is mounted read-only
            core.set_property({"CACHE_DIR": str(self.model_dir / "ov_cache")})  # faster GPU/NPU reloads
        model = core.read_model(self.model_dir / "model.xml")
        if model.input().get_partial_shape().is_dynamic:  # NPU needs static shapes
            model.reshape([1, 3, self.image_size, self.image_size])
        self._model = core.compile_model(model, device)
        self.device = device

    def predict(self, image: np.ndarray | str | Path, crop: bool = True, check_presence: bool = True) -> Prediction:
        """Inspect one image.

        Args:
            image: BGR uint8 array (as returned by OpenCV) or an image path.
            crop: Apply the training ROI. Set ``False`` for images that are already crops
                (e.g. files from the prepared dataset).
            check_presence: Return ``status="empty"`` without running the anomaly model when
                the ROI looks empty. ``False`` always runs the model (``presence`` is still reported).
        """
        if not isinstance(image, np.ndarray):
            image = read_image(image)
        patch = preprocess(image, self.roi if crop else None, self.image_size)

        presence = self.presence.score(patch) if self.presence else None
        if check_presence and self._is_empty(presence):
            return Prediction(EMPTY, None, False, presence, None, patch)

        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        outputs = self._model(rgb.transpose(2, 0, 1)[None])
        out = {port.get_any_name(): value for port, value in outputs.items()}
        score = float(np.asarray(out["pred_score"]).reshape(-1)[0])
        anomaly_map = np.asarray(out["anomaly_map"], dtype=np.float32).reshape(self.image_size, self.image_size)
        status = ANOMALY if score >= self.threshold else GOOD
        uncertain = abs(score - self.threshold) < self.uncertain_margin
        return Prediction(status, score, uncertain, presence, anomaly_map, patch)

    def _is_empty(self, presence: float | None) -> bool:
        return presence is not None and presence < self.presence.threshold

    def is_empty(self, image: np.ndarray | str | Path, crop: bool = True) -> bool:
        """True exactly when ``predict(image)`` would return status "empty" (``Verdict.EMPTY``).

        Uses the same rule as ``predict`` but skips the anomaly model, so it is cheap enough to poll.
        Always False when the model has no empty-ROI reference, as ``predict`` then never reports empty.
        """
        if not isinstance(image, np.ndarray):
            image = read_image(image)
        if self.presence is None:
            return False
        patch = preprocess(image, self.roi if crop else None, self.image_size)
        return self._is_empty(self.presence.score(patch))

    def predict_many(self, images, crop: bool = True) -> list[Prediction]:
        return [self.predict(image, crop=crop) for image in images]

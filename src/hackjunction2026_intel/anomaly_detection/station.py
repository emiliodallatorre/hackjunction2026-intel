"""Inspection camera + model in one object, configured from config.yaml: the robot-side entry point.

The camera is opened with the config's resolution, pixel format and locked focus / exposure /
gain / white balance (the settings the model was trained with), so nothing else should open or
configure that camera.

    from hackjunction2026_intel.anomaly_detection import InspectionStation, Verdict

    with InspectionStation() as station:              # config: $BRICK_AD_CONFIG or anomaly_detection/config.yaml
        decision = station.inspect(rotate=rotate_wrist)   # rotate_wrist(decision): your robot code, returns once still
        if decision.verdict is Verdict.EMPTY: ...         # re-grasp
        elif decision.verdict is Verdict.DEFECTIVE: ...   # defective bowl
        else: ...                                         # good bowl
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from .camera import Camera
from .config import Config, load_config
from .decision import Decision, Verdict, decide
from .inspector import BrickInspector, Prediction


class InspectionStation:
    def __init__(
        self,
        config: str | Path | Config | None = None,
        model_dir: str | Path | None = None,
        device: str | None = None,
        threshold: float | None = None,
        max_looks: int = 3,
    ):
        """
        Args:
            config: Config file or loaded Config. ``None``: $BRICK_AD_CONFIG, else anomaly_detection/config.yaml.
            model_dir: Exported model folder (default: paths.model in the config).
            device: OpenVINO device (default: inference.device in the config).
            threshold: Decision threshold (default: inference.threshold in the config, else the trained 0.5).
            max_looks: Views per brick before deciding even if still uncertain.
        """
        self.cfg = config if isinstance(config, Config) else load_config(config)
        self.max_looks = max_looks
        # Load the model first: if it is missing, fail before grabbing the camera.
        self.inspector = BrickInspector(
            model_dir or self.cfg.paths.model,
            device=device or self.cfg.inference.device,
            threshold=threshold if threshold is not None else self.cfg.inference.threshold,
            uncertain_margin=self.cfg.inference.uncertain_margin,
        )
        self.camera = Camera(self.cfg.camera)  # applies resolution, format and locked controls
        expected = (self.cfg.camera.width, self.cfg.camera.height)
        if self.camera.resolution != expected:
            got = self.camera.resolution
            self.camera.close()
            raise RuntimeError(
                f"Camera {self.cfg.camera.source!r} delivers {got[0]}x{got[1]}, config says "
                f"{expected[0]}x{expected[1]}: the ROI would crop the wrong area."
            )

    def grab(self, drop: int = 3) -> np.ndarray:
        """Fresh BGR frame. Call once the brick is still; dropped frames may predate it."""
        for _ in range(drop):
            self.camera.read()
        return self.camera.read()

    def is_empty(self) -> bool:
        """Whether inspecting the latest frame would give ``Verdict.EMPTY``. Cheap: skips the anomaly model."""
        if self.inspector.presence is None:
            raise RuntimeError(
                "Model has no empty-ROI reference, so it never reports EMPTY "
                "(collect empty frames, then `brick-ad prepare` + `brick-ad fit-empty`)."
            )
        return self.inspector.is_empty(self.camera.read())

    def look(self) -> Prediction:
        """Grab a frame and score it."""
        return self.inspector.predict(self.grab())

    def inspect(self, rotate: Callable[[Decision], None] | None = None) -> Decision:
        """Inspect the presented brick until a decision is reached.

        Args:
            rotate: Called with the ROTATE decision when the view is uncertain; should show
                another side of the brick and return once it is still. ``None``: return the
                ROTATE decision to the caller instead (continue with ``self.decide(decision.looks + [self.look()])``).
        """
        looks = [self.look()]
        while True:
            decision = self.decide(looks)
            if decision.verdict is not Verdict.ROTATE or rotate is None:
                return decision
            rotate(decision)
            looks.append(self.look())

    def decide(self, looks: list[Prediction]) -> Decision:
        return decide(looks, self.inspector.threshold, self.max_looks)

    def close(self) -> None:
        self.camera.close()

    def __enter__(self) -> "InspectionStation":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

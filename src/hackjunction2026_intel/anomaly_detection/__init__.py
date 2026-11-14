"""Brick anomaly detection at runtime: inspection camera + OpenVINO PatchCore model -> diagnosis.

Only needs OpenVINO, OpenCV, NumPy and PyYAML. The model and config.yaml are produced by the
`brick-ad` tools in anomaly_detection/ (collect, roi, lock, prepare, train).

    from hackjunction2026_intel.anomaly_detection import InspectionStation, Verdict

    station = InspectionStation()                     # opens the camera with config.yaml's locked settings
    decision = station.inspect(rotate=rotate_wrist)   # rotate_wrist(decision): robot code, returns once still
    decision.verdict                                  # Verdict.EMPTY | GOOD | DEFECTIVE
    decision.message, decision.score, decision.looks

Demo with keyboard prompts in place of the robot: python -m hackjunction2026_intel.anomaly_detection --show
"""

from .config import Config, load_config
from .decision import Decision, Verdict, decide
from .inspector import BrickInspector, Prediction
from .station import InspectionStation

__all__ = [
    "BrickInspector",
    "Config",
    "Decision",
    "InspectionStation",
    "Prediction",
    "Verdict",
    "decide",
    "load_config",
]

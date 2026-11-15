"""Typed access to ``anomaly_detection/config.yaml`` (camera settings, ROI, model paths).

The file is written by the `brick-ad` tools (roi, lock) and read here by the robot code.
"""

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

# <repo>/anomaly_detection/config.yaml when running from the source tree. After a non-editable
# install (e.g. in a robot container) that path does not exist: set BRICK_AD_CONFIG instead.
DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "anomaly_detection" / "config.yaml"
CONFIG_ENV = "BRICK_AD_CONFIG"

GOOD, ANOMALY, EMPTY = "good", "anomaly", "empty"
LABELS = (GOOD, ANOMALY, EMPTY)

Roi = tuple[int, int, int, int]  # x, y, w, h


@dataclass
class CameraConfig:
    source: int | str = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    fourcc: str | None = "MJPG"
    # Locked controls in the camera's native units; None = automatic.
    exposure: int | None = None
    gain: int | None = None
    white_balance: int | None = None
    focus: int | None = None
    max_exposure: int = 1000  # cap used by `brick-ad lock` (V4L2 units of 100 us: 333 = 33 ms = 30 fps)

    @property
    def locked(self) -> dict[str, int | None]:
        return {k: getattr(self, k) for k in CAMERA_CONTROLS}


CAMERA_CONTROLS = ("exposure", "gain", "white_balance", "focus")


# The sections below are only used by the `brick-ad` training tools, but they are part of the
# same file, so the schema lives here.
@dataclass
class CollectConfig:
    burst_interval: float = 0.3


@dataclass
class DatasetConfig:
    test_split: float = 0.2


@dataclass
class ModelConfig:
    image_size: int = 256
    backbone: str = "wide_resnet50_2"
    layers: list[str] = field(default_factory=lambda: ["layer2", "layer3"])
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9


@dataclass
class TrainConfig:
    accelerator: str = "auto"
    batch_size: int = 16
    num_workers: int = 4


@dataclass
class InferenceConfig:
    device: str = "CPU"
    threshold: float | None = None  # score >= threshold -> anomaly; None = the trained threshold (0.5)
    uncertain_margin: float = 0.1


@dataclass
class PathsConfig:
    raw: Path = Path("data/raw")
    dataset: Path = Path("data/dataset")
    runs: Path = Path("runs")
    model: Path = Path("models/patchcore")


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    roi: Roi | None = None
    collect: CollectConfig = field(default_factory=CollectConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    file: Path = DEFAULT_CONFIG


def _build(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys in config: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        default = getattr(cls(), name)
        if is_dataclass(default) and isinstance(value, dict):
            value = _build(type(default), value)
        kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> Config:
    """Load ``path``, else $BRICK_AD_CONFIG, else anomaly_detection/config.yaml."""
    path = Path(path or os.environ.get(CONFIG_ENV) or DEFAULT_CONFIG).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config {path} not found. Pass its path or set {CONFIG_ENV}.")
    values = yaml.safe_load(path.read_text()) or {}
    cfg: Config = _build(Config, values)
    cfg.file = path
    if cfg.roi is not None:
        cfg.roi = tuple(int(v) for v in cfg.roi)
    # Resolve relative paths against the config file's folder.
    for f in fields(PathsConfig):
        p = Path(getattr(cfg.paths, f.name))
        setattr(cfg.paths, f.name, p if p.is_absolute() else path.parent / p)
    return cfg

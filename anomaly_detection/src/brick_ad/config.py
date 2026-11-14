"""Config schema and loader come from the runtime package; this adds the writers used by `roi` and `lock`."""

import re
from pathlib import Path
from typing import Any

from hackjunction2026_intel.anomaly_detection.config import (  # noqa: F401  (re-exported for the tools)
    ANOMALY,
    CAMERA_CONTROLS,
    EMPTY,
    GOOD,
    LABELS,
    CameraConfig,
    Config,
    Roi,
    load_config,
)


def _yaml_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(str(int(v)) for v in value) + "]"
    return str(int(value))


def _set_yaml_line(path: Path, key: str, value: Any) -> None:
    """Rewrite only the ``key:`` line (keys used here are unique) so comments are preserved."""
    text = path.read_text()
    pattern = rf"^([ \t]*){key}:[^\n#]*?([ \t]*#.*)?$"
    text, n = re.subn(pattern, lambda m: f"{m[1]}{key}: {_yaml_value(value)}{m[2] or ''}", text,
                      count=1, flags=re.MULTILINE)
    if n == 0:
        raise ValueError(f"Key {key!r} not found in {path}; add it to the file first")
    path.write_text(text)


def save_roi(cfg: Config, roi: Roi | None) -> None:
    _set_yaml_line(cfg.file, "roi", roi)
    cfg.roi = roi


def save_camera_controls(cfg: Config, values: dict[str, int | None]) -> None:
    for key, value in values.items():
        if key not in CAMERA_CONTROLS:
            raise ValueError(f"Unknown camera control {key!r}")
        _set_yaml_line(cfg.file, key, value)
        setattr(cfg.camera, key, value)

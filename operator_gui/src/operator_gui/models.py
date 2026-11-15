"""Find the Studio runtime exports available on this machine, for the dashboard's model picker.

A usable export is a folder with ``runtime.yaml`` and ``exports/openvino/``, or a .zip holding the
same. Only the folder form can be validated cheaply, so a zip is accepted on its central directory
listing alone and fully checked when it is actually opened.

Discovery is deliberately filesystem-based rather than Studio's REST API: an export is already on
disk by the time it can be run, so this works with no server, no auth and no network.
"""

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Searched in order; the first export found in the first directory becomes the default.
SEARCH_DIRS = (Path.home() / "Downloads", Path.home(), Path.cwd())
SEARCH_DEPTH = 2  # subdirectories below each search dir
BACKEND = "openvino"  # must match hackjunction2026_intel.sorting.BACKEND


@dataclass(frozen=True)
class Model:
    """One runnable export."""

    path: Path
    name: str
    kind: str  # "folder" or "zip"

    @property
    def label(self) -> str:
        return f"{self.name}  ({self.kind})"


def _is_export_folder(path: Path) -> bool:
    return (path / "runtime.yaml").is_file() and (path / "exports" / BACKEND).is_dir()


def _is_export_zip(path: Path) -> bool:
    if path.suffix.lower() != ".zip" or not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except (zipfile.BadZipFile, OSError):
        return False
    # Accept both a flat export and one wrapped in a single top-level folder.
    return any(n.endswith("runtime.yaml") for n in names) and any(f"exports/{BACKEND}/" in n for n in names)


def discover(extra: Path | None = None) -> list[Model]:
    """Every runnable export found, best candidate first. ``extra`` is always included if valid."""
    found: list[Model] = []
    seen: set[Path] = set()

    def add(path: Path, kind: str) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        found.append(Model(resolved, path.name, kind))

    if extra is not None:
        if extra.is_dir() and _is_export_folder(extra):
            add(extra, "folder")
        elif _is_export_zip(extra):
            add(extra, "zip")
        else:
            logger.warning("Ignoring --export %s: no %s/runtime.yaml inside", extra, BACKEND)

    for directory in SEARCH_DIRS:
        if not directory.is_dir():
            continue
        for candidate in sorted(directory.iterdir()):
            try:
                if candidate.is_dir() and _is_export_folder(candidate):
                    add(candidate, "folder")
                elif _is_export_zip(candidate):
                    add(candidate, "zip")
            except OSError:
                continue
        # One level deeper, so an export inside e.g. ~/Downloads/models/ is still found.
        if SEARCH_DEPTH > 1:
            for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
                try:
                    if _is_export_folder(sub):
                        continue  # already added above
                    for candidate in sorted(sub.iterdir()):
                        if candidate.is_dir() and _is_export_folder(candidate):
                            add(candidate, "folder")
                        elif _is_export_zip(candidate):
                            add(candidate, "zip")
                except OSError:
                    continue
    return found

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
    has_yaml = (path / "runtime.yaml").is_file()
    has_weights = (path / "exports" / BACKEND).is_dir()
    if has_yaml or has_weights:  # looks like an attempt at an export: say why it was rejected
        logger.info("  %s: runtime.yaml=%s exports/%s/=%s -> %s",
                    path, has_yaml, BACKEND, has_weights,
                    "ACCEPTED" if (has_yaml and has_weights) else "rejected")
    return has_yaml and has_weights


def _is_export_zip(path: Path) -> bool:
    if path.suffix.lower() != ".zip" or not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except (zipfile.BadZipFile, OSError) as error:
        logger.info("  %s: not readable as a zip (%s)", path, error)
        return False
    # Accept both a flat export and one wrapped in a single top-level folder.
    has_yaml = any(n.endswith("runtime.yaml") for n in names)
    has_weights = any(f"exports/{BACKEND}/" in n for n in names)
    logger.info("  %s: runtime.yaml=%s exports/%s/=%s -> %s",
                path, has_yaml, BACKEND, has_weights,
                "ACCEPTED" if (has_yaml and has_weights) else "rejected")
    return has_yaml and has_weights


def discover(extra: Path | None = None) -> list[Model]:
    """Every runnable export found, best candidate first. ``extra`` is always included if valid."""
    found: list[Model] = []
    seen: set[Path] = set()
    logger.info("Export discovery starting")
    logger.info("  home=%s cwd=%s", Path.home(), Path.cwd())
    logger.info("  search dirs: %s", [str(d) for d in SEARCH_DIRS])
    for directory in SEARCH_DIRS:
        logger.info("  %s exists=%s", directory, directory.is_dir())

    def add(path: Path, kind: str) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        found.append(Model(resolved, path.name, kind))

    extra_dirs: list[Path] = []
    if extra is not None:
        if extra.is_dir() and _is_export_folder(extra):
            add(extra, "folder")
        elif _is_export_zip(extra):
            add(extra, "zip")
        elif extra.is_dir():
            # A directory holding exports (e.g. ~/Downloads): search it first, rather than refuse.
            logger.info("--export %s is not an export itself; searching inside it", extra)
            extra_dirs.append(extra)
        else:
            logger.warning("Ignoring --export %s: no runtime.yaml and exports/%s/ inside", extra, BACKEND)

    scanned: set[Path] = set()
    for directory in [*extra_dirs, *SEARCH_DIRS]:
        if not directory.is_dir():
            logger.info("Skipping %s: not a directory", directory)
            continue
        resolved_dir = directory.resolve()
        if resolved_dir in scanned:
            logger.info("Skipping %s: already scanned", directory)
            continue
        scanned.add(resolved_dir)
        entries = sorted(directory.iterdir())
        logger.info("Scanning %s (%d entries)", directory, len(entries))
        for candidate in entries:
            try:
                if candidate.is_dir() and _is_export_folder(candidate):
                    add(candidate, "folder")
                elif _is_export_zip(candidate):
                    add(candidate, "zip")
            except OSError as error:
                logger.info("  %s: unreadable (%s)", candidate, error)
                continue
        # One level deeper, so an export inside e.g. ~/Downloads/models/ is still found.
        if SEARCH_DEPTH > 1:
            for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
                try:
                    if sub.resolve() in scanned or _is_export_folder(sub):
                        continue  # already scanned, or already added above
                    scanned.add(sub.resolve())
                    for candidate in sorted(sub.iterdir()):
                        if candidate.is_dir() and _is_export_folder(candidate):
                            add(candidate, "folder")
                        elif _is_export_zip(candidate):
                            add(candidate, "zip")
                except OSError as error:
                    logger.info("  %s: unreadable (%s)", sub, error)
                    continue
    logger.info("Export discovery finished: %d found %s", len(found), [m.name for m in found])
    return found

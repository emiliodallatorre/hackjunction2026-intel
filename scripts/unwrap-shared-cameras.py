#!/usr/bin/env python3
"""Replace SharedCamera with the UVCCamera it wraps in a Studio runtime export.

    scripts/unwrap-shared-cameras.py ~/Downloads/pi05-ricotta-runtime [more.yaml ...]
    scripts/unwrap-shared-cameras.py --check ~/Downloads        # report, change nothing

SharedCamera runs its capture in a publisher subprocess that shares frames over iceoryx2. On
hardware where that node cannot be created the runtime never connects its cameras:

    failed to start camera publisher: NodeCreationFailure: InternalError

Nothing here reads one camera from two processes, so the shared transport buys nothing. This
rewrites each camera to the plain UVCCamera already nested inside it, keeping the arguments
UVCCamera accepts and dropping the SharedCamera-only ones.

Studio always exports the SharedCamera form, so re-run this after every export. A .bak backup is
written next to each file it changes.
"""

import argparse
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger("unwrap-shared-cameras")

SHARED = "physicalai.capture.SharedCamera"
# Accepted by SharedCamera but not by a bare camera: they must be dropped, not carried over.
SHARED_ONLY = ("validate_on_connect", "overwrite_settings", "name", "max_subscribers")


def unwrap(document: dict[str, Any]) -> list[str]:
    """Unwrap every SharedCamera in place; returns the names changed."""
    cameras = document.get("init_args", {}).get("cameras")
    if not isinstance(cameras, dict):
        return []
    changed = []
    for name, camera in list(cameras.items()):
        if not isinstance(camera, dict) or camera.get("class_path") != SHARED:
            continue
        outer = camera.get("init_args")
        if not isinstance(outer, dict) or not isinstance(outer.get("camera"), dict):
            logger.warning("  %r is a SharedCamera with no nested camera; left alone", name)
            continue
        inner = dict(outer["camera"])
        inner_args = dict(inner.get("init_args") or {})
        # Carry over outer arguments the inner camera accepts (color_mode), drop the rest.
        for key, value in outer.items():
            if key == "camera" or key in SHARED_ONLY:
                continue
            inner_args.setdefault(key, value)
        inner["init_args"] = inner_args
        cameras[name] = inner
        changed.append(name)
    return changed


def yaml_files(target: Path) -> list[Path]:
    """The runtime.yaml files under ``target``: itself, its export, or every export inside it."""
    if target.is_file():
        return [target]
    if (target / "runtime.yaml").is_file():
        return [target / "runtime.yaml"]
    if target.is_dir():
        return sorted(target.glob("*/runtime.yaml"))
    return []


def process(path: Path, *, check: bool) -> bool:
    """Rewrite ``path`` if it holds a SharedCamera; returns True when one was found."""
    import yaml

    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        logger.warning("%s: not a runtime configuration", path)
        return False
    changed = unwrap(document)
    if not changed:
        logger.info("%s: already direct cameras, nothing to do", path)
        return False
    if check:
        logger.warning("%s: would unwrap %s", path, ", ".join(repr(c) for c in changed))
        return True
    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    path.write_text(yaml.safe_dump(document, sort_keys=False, default_flow_style=False, allow_unicode=True))
    logger.info("%s: unwrapped %s (backup: %s)", path, ", ".join(repr(c) for c in changed), backup.name)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", type=Path, nargs="+",
                        help="A runtime.yaml, an export folder, or a folder holding exports")
    parser.add_argument("--check", action="store_true", help="Report what would change, write nothing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    files = [f for target in args.targets for f in yaml_files(target)]
    if not files:
        logger.error("No runtime.yaml found in %s", ", ".join(str(t) for t in args.targets))
        return 1
    found = sum(process(f, check=args.check) for f in files)
    logger.info("%d of %d file(s) %s", found, len(files), "need unwrapping" if args.check else "rewritten")
    return 1 if (args.check and found) else 0


if __name__ == "__main__":
    sys.exit(main())

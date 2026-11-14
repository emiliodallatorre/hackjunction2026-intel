"""Present one brick to the inspection camera and report whether it is defective; no placement."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import sorting
from .sorting import open_export, present_block

logger = logging.getLogger(__name__)

DEFAULT_REPORT = Path("inspection.png")
DEFAULT_PRESENT_TIMEOUT_S = 45.0


def save_report(decision: Any, path: Path) -> None:
    """Write the analysed crop, with the anomaly heatmap when there is one, labelled with the verdict."""
    import cv2

    from .anomaly_detection.imaging import overlay_heatmap, put_text

    look = decision.looks[-1]
    view = look.patch if look.anomaly_map is None else overlay_heatmap(look.patch, look.anomaly_map)
    view = cv2.resize(view, (512, 512))
    put_text(view, decision.verdict.name)
    if decision.score is not None:
        put_text(view, f"score {decision.score:.2f}", line=1)
    if not cv2.imwrite(str(path), view):
        raise OSError(f"Could not write inspection report {path}")
    logger.info("Inspection report saved to %s", path)


def run(export_root: Path, report: Path | None) -> Any:
    """Present the brick, inspect it once, and return the decision without acting on it."""
    from .anomaly_detection import InspectionStation

    # Opened before any motion so a missing model or camera aborts up front.
    with InspectionStation() as station:
        logger.info("Moving the block to the inspection stage")
        present_block(export_root, station)
        decision = station.inspect()
    logger.info("Inspection: %s", decision.message)
    if report is not None:
        save_report(decision, report)
    return decision


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export",
        type=Path,
        required=True,
        help="Studio OpenVINO runtime export: the downloaded .zip or its extracted folder",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"Where to save the analysed crop with its heatmap (default: {DEFAULT_REPORT})",
    )
    parser.add_argument("--no-report", action="store_true", help="Do not save the inspection image")
    parser.add_argument(
        "--present-timeout",
        type=float,
        default=DEFAULT_PRESENT_TIMEOUT_S,
        help=f"Seconds the arm may take to bring the brick into view (default: {DEFAULT_PRESENT_TIMEOUT_S})",
    )
    args = parser.parse_args(argv)
    sorting.PRESENT_DURATION_S = args.present_timeout
    with open_export(args.export) as export_root:
        run(export_root, None if args.no_report else args.report)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

"""Operator dashboard for the brick sorting robot (hackjunction2026_intel.run_model).

    operator-gui                              # pick from the exports found in ~/Downloads
    operator-gui --export ~/Downloads/an-export.zip   # start on this one
    operator-gui --dry-run                    # no robot, camera or model: random verdicts

Without --export the dashboard discovers every Studio OpenVINO export on this machine and starts
on the first one; the header's picker switches between them while idle.
"""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .controller import ACTION_DURATION_S

DEFAULT_HISTORY = Path(__file__).resolve().parents[2] / "history"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--export", type=Path,
                        help="Studio OpenVINO runtime export to start on: the .zip or its folder "
                             "(default: the first one discovered)")
    source.add_argument("--dry-run", action="store_true", help="Simulate the robot, camera and model")
    parser.add_argument("--action-duration", type=float, default=ACTION_DURATION_S,
                        help=f"Control seconds a typed action runs for (default: {ACTION_DURATION_S})")
    parser.add_argument("--home", type=Path,
                        help="Home pose file for Reset, written by `run_model ... reset --save-home` "
                             "(default: home_pose.json in the working directory)")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to serve on (default: all)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY,
                        help=f"Folder for bricks.csv and thumbnails (default: {DEFAULT_HISTORY})")
    args = parser.parse_args(argv)
    if args.export is not None and not args.export.exists():
        parser.error(f"export not found: {args.export}")

    # Dashboard-side logging: the runner child configures its own.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from .models import discover

    models = [] if args.dry_run else discover(args.export)
    if not args.dry_run and not models:
        parser.error(
            "no Studio OpenVINO export found in ~/Downloads, your home folder or here.\n"
            "An export is a folder (or .zip) holding runtime.yaml and exports/openvino/.\n"
            "Pass one with --export PATH, or use --dry-run to try the dashboard without hardware."
        )
    source_args = ["--dry-run"] if args.dry_run else ["--export", str(models[0].path)]

    from nicegui import app, ui

    from .controller import CycleController
    from .history import History
    from .ui.dashboard import register

    controller = CycleController(source_args, History(args.history), action_duration_s=args.action_duration,
                                 home_file=args.home.resolve() if args.home else None)
    if models:
        controller.set_models(models)  # fills the header's model picker; keeps models[0] selected
    register(controller)
    # Stream the inspection camera while idle; a run takes the camera over and gives it back.
    app.on_startup(controller.start_preview)
    app.on_shutdown(controller.shutdown)
    ui.run(host=args.host, port=args.port, title="Brick Sorter", dark=True, reload=False, show=False,
           favicon="🧱")


if __name__ == "__main__":
    main()

"""Operator dashboard for the brick sorting robot (hackjunction2026_intel.run_model).

    operator-gui --export ~/Downloads/model_openvino.zip
    operator-gui --dry-run          # no robot, camera or model: random verdicts
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

from .controller import ACTION_DURATION_S

DEFAULT_HISTORY = Path(__file__).resolve().parents[2] / "history"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--export", type=Path, help="Studio OpenVINO runtime export: the .zip or its folder")
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
    source_args = ["--dry-run"] if args.dry_run else ["--export", str(args.export.resolve())]

    from nicegui import app, ui

    from .controller import CycleController
    from .history import History
    from .ui.dashboard import register

    controller = CycleController(source_args, History(args.history), action_duration_s=args.action_duration,
                                 home_file=args.home.resolve() if args.home else None)
    register(controller)
    # Stream the inspection camera while idle; a run takes the camera over and gives it back.
    app.on_startup(controller.start_preview)
    app.on_shutdown(controller.shutdown)
    ui.run(host=args.host, port=args.port, title="Brick Sorter", dark=True, reload=False, show=False,
           favicon="🧱")


if __name__ == "__main__":
    main()

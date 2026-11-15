"""Idle camera preview: a child process that streams the inspection camera when nothing is running.

The station's camera is an exclusive open, so this process must not overlap a cycle. The
controller starts it only while idle and stops it before launching a runner, which is why the
preview lives here rather than inside the runner: one holder of the camera at a time.

Frames go out as ``events.FRAME`` on the same pipe protocol the runner uses, so the dashboard's
LiveView needs no changes.
"""

import argparse
import logging
import signal
import sys
import time
from collections.abc import Sequence

logger = logging.getLogger(__name__)

PREVIEW_FPS = 10.0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-fd", type=int, required=True)
    parser.add_argument("--fps", type=float, default=PREVIEW_FPS)
    parser.add_argument("--dry-run", action="store_true", help="No camera: publish nothing, just idle")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    # SIGTERM is how the controller stops us; exit quietly so the camera is released at once.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    if args.dry_run:
        signal.pause()
        return 0

    from .runner import Reporter

    reporter = Reporter(args.events_fd)
    # install() wraps the station's camera in a FrameTap that publishes what it reads.
    from .observer import install

    install(reporter)

    from hackjunction2026_intel import anomaly_detection

    period = 1.0 / max(args.fps, 1.0)
    try:
        with anomaly_detection.InspectionStation() as station:
            while True:
                started = time.monotonic()
                station.camera.read()
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
    except SystemExit:
        return 0
    except Exception as error:
        # A missing camera or model must not spam the dashboard: log once and stop.
        logger.warning("Preview stopped: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

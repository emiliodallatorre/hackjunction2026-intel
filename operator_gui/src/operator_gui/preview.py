"""Idle camera preview: a child process that streams the inspection camera when nothing is running.

The station's camera is an exclusive open, so this process must not overlap a cycle. The
controller starts it only while idle and stops it before launching a runner, which is why the
preview lives here rather than inside the runner: one holder of the camera at a time.

Frames go out as ``events.FRAME`` on the same pipe protocol the runner uses, so the dashboard's
LiveView needs no changes.

SIGUSR1 asks for one inspection of the current view: the station is already open here, so an
on-demand analysis needs no second camera holder. The verdict is published as ``events.LOOK`` and
``events.DECISION``, exactly as a cycle's inspection is, so the dashboard renders it unchanged.
"""

import argparse
import contextlib
import logging
import signal
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

PREVIEW_FPS = 10.0
ANALYZE_SIGNAL = signal.SIGUSR1  # must match controller.ANALYZE_SIGNAL
# A V4L2 device is not immediately reopenable after the previous holder closes it, so the first
# open (or first frame) right after a run can fail. Retry for a while before giving up.
OPEN_RETRY_S = 15.0
OPEN_RETRY_DELAY_S = 1.0


@contextmanager
def _opened_station(anomaly_detection: Any) -> Iterator[Any]:
    """Open the inspection station, retrying while the camera is still held by a finished run.

    Closing a V4L2 device does not make it instantly reopenable, so a preview starting right after
    a policy stage can open the camera and get no frame. Both failures are transient here.
    """
    deadline = time.monotonic() + OPEN_RETRY_S
    attempt = 0
    while True:
        attempt += 1
        station = None
        try:
            station = anomaly_detection.InspectionStation()
            station.__enter__()
            station.camera.read()  # prove it delivers frames before the dashboard trusts it
            break
        except Exception as error:  # noqa: BLE001
            if station is not None:
                with contextlib.suppress(Exception):
                    station.__exit__(None, None, None)
            if time.monotonic() >= deadline:
                raise
            logger.info("Camera not ready yet (attempt %d): %s", attempt, error)
            time.sleep(OPEN_RETRY_DELAY_S)
    try:
        yield station
    finally:
        with contextlib.suppress(Exception):
            station.__exit__(None, None, None)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events-fd", type=int, required=True)
    parser.add_argument("--fps", type=float, default=PREVIEW_FPS)
    parser.add_argument("--dry-run", action="store_true", help="No camera: publish nothing, just idle")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # SIGTERM is how the controller stops us; exit quietly so the camera is released at once.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    # Set by the signal handler, acted on between frames: inspect() must not run in a handler.
    analyze = threading.Event()
    signal.signal(ANALYZE_SIGNAL, lambda *_: analyze.set())

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
        with _opened_station(anomaly_detection) as station:
            while True:
                started = time.monotonic()
                station.camera.read()
                if analyze.is_set():
                    analyze.clear()
                    # ObservedStation publishes the look and the decision as it inspects.
                    try:
                        decision = station.inspect()
                        logger.info("Manual analysis: %s", decision.message)
                    except Exception as error:  # noqa: BLE001
                        logger.warning("Manual analysis failed: %s", error)
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

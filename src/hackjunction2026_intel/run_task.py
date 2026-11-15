"""Run one arbitrary task on the exported policy for a fixed duration; no inspection, no routing."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .sorting import open_export, run_policy

logger = logging.getLogger(__name__)

DEFAULT_DURATION_S = 15.0


def run(export_root: Path, task: str, *, duration_s: float = DEFAULT_DURATION_S, repeat: int = 1) -> None:
    """Run ``task`` ``repeat`` times, each for ``duration_s`` control seconds.

    This is a timed run: it never claims the manipulation succeeded, only that the policy
    was driven for the full duration. A fresh runtime is built per run, because each
    ``run()`` closes its InferenceModel on exit.
    """
    if repeat < 1:
        raise ValueError(f"Repeat count must be at least 1, got {repeat}")
    for attempt in range(1, repeat + 1):
        if repeat > 1:
            logger.info("Run %s of %s", attempt, repeat)
        run_policy(export_root, task, duration_s=duration_s)
    logger.warning("Task %r ran for its configured duration; its outcome was NOT verified", task)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "task",
        help='Task instruction for the multi-task policy, e.g. "Move the brick to the yellow dish"',
    )
    parser.add_argument(
        "--export",
        type=Path,
        required=True,
        help="Studio OpenVINO runtime export: the downloaded .zip or its extracted folder",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"Control seconds to run the task for (default: {DEFAULT_DURATION_S})",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the same task this many times, with a fresh runtime each time (default: 1)",
    )
    args = parser.parse_args(argv)
    with open_export(args.export) as export_root:
        run(export_root, args.task, duration_s=args.duration, repeat=args.repeat)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()

"""Hardware-free stand-ins so the dashboard can be tried without a robot, camera or model.

``DryRunner`` takes the place of ``run_model.PolicyRunner``: each stage only logs and waits a
moment. ``DryStation`` takes the place of ``InspectionStation``: the brick is "present" at once and
the verdict is random. ``run_model.run_cycle`` and ``run_model.run_task`` run unchanged on top.
"""

import logging
import random
import time
from types import SimpleNamespace
from typing import Any

from hackjunction2026_intel import anomaly_detection, run_model
from hackjunction2026_intel.anomaly_detection import Decision, Verdict

from . import events

logger = logging.getLogger(__name__)

STAGE_S = 1.5  # simulated time per stage, so the dashboard has something to show


class DryRunner:
    home_pose = "dry-run home"  # set, so run_model.reset_robot uses reset() rather than a home file

    def reset(self, *, pose: Any | None = None) -> None:
        logger.info("DRY RUN: resetting the arm to its home pose; no robot commands")
        time.sleep(STAGE_S)

    def run_stage(
        self,
        task: str,
        *,
        duration_s: float,
        stop_event: Any = None,
        stop_event_factory: Any = None,
    ) -> None:
        logger.info("DRY RUN: task %r (configured %s s); no robot commands", task, duration_s)
        time.sleep(STAGE_S)

    def present(self, station: Any) -> None:
        self.run_stage(run_model.PRESENT_TASK, duration_s=run_model.PRESENT_DURATION_S)


class DryStation:
    reporter: Any = None  # set by install()

    def __init__(self, *args: Any, **kwargs: Any):
        self.inspector = SimpleNamespace(presence=object(), threshold=0.5)

    def __enter__(self) -> "DryStation":
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    def is_empty(self) -> bool:
        return False

    def inspect(self, rotate: Any = None) -> Decision:
        self.reporter.stage(events.INSPECT)
        time.sleep(STAGE_S)
        score = random.random()
        verdict = Verdict.DEFECTIVE if score >= self.inspector.threshold else Verdict.GOOD
        message = f"{verdict.name} brick (score {score:.2f}; random, dry run)"
        self.reporter.send(events.DECISION, verdict=verdict.value, message=message, score=score, looks=1,
                           thumbnail="", overlay="")
        return Decision(verdict, message, [], score)


def install(reporter: Any) -> None:
    DryStation.reporter = reporter
    anomaly_detection.InspectionStation = DryStation

"""Turn one or more views of a brick into a single robot action.

    no brick?        -> EMPTY      (failed pick: re-grasp)
    score near 0.5?  -> ROTATE     (show another side and look again)
    otherwise        -> GOOD or DEFECTIVE

    from hackjunction2026_intel.anomaly_detection import BrickInspector, Verdict, decide

    looks = [inspector.predict(frame)]
    decision = decide(looks, inspector.threshold)
    while decision.verdict is Verdict.ROTATE:
        rotate_wrist()                                # robot action, wait until still
        looks.append(inspector.predict(grab_frame()))
        decision = decide(looks, inspector.threshold)
"""

from dataclasses import dataclass, field
from enum import Enum

from .inspector import Prediction


class Verdict(Enum):
    EMPTY = "empty"  # no brick in the inspection area
    ROTATE = "rotate"  # unsure: show another side of the brick
    GOOD = "good"
    DEFECTIVE = "defective"


@dataclass
class Decision:
    verdict: Verdict
    message: str
    looks: list[Prediction] = field(default_factory=list)  # every view used for this decision
    score: float | None = None  # highest anomaly score over the views; None when empty


def decide(looks: list[Prediction], threshold: float = 0.5, max_looks: int = 3) -> Decision:
    """Decide from the views of one brick seen so far (oldest first)."""
    if not looks:
        raise ValueError("decide() needs at least one prediction")
    last = looks[-1]
    if last.is_empty:
        return Decision(Verdict.EMPTY, "ERROR: no brick in the inspection area (failed pick?)", looks)
    if last.uncertain and len(looks) < max_looks:
        return Decision(Verdict.ROTATE, f"UNSURE (score {last.score:.2f}): rotate the brick and inspect again",
                        looks, last.score)

    # A defect only needs to be visible from one side: use the highest score over all views.
    score = max(p.score for p in looks if p.score is not None)
    detail = f"score {score:.2f}" if len(looks) == 1 else f"highest score over {len(looks)} views {score:.2f}"
    if last.uncertain:
        detail += ", still close to the threshold"
    if score >= threshold:
        return Decision(Verdict.DEFECTIVE, f"DEFECTIVE brick ({detail})", looks, score)
    return Decision(Verdict.GOOD, f"GOOD brick ({detail})", looks, score)

"""Wire protocol between the cycle runner (child process) and the dashboard.

The runner writes one JSON object per line to a dedicated pipe, so library output on
stdout/stderr never mixes with it. Every object has a ``type`` key, one of the kinds below.
"""

import json
from typing import Any

# Event kinds and their fields.
LOG = "log"  # level, message: a log record from the cycle
STAGE = "stage"  # stage: one of the stages below
FRAME = "frame"  # jpeg: base64 inspection-camera frame with the ROI drawn
LOOK = "look"  # status, score, uncertain, overlay: one view scored by the anomaly model
DECISION = "decision"  # verdict, message, score, looks, thumbnail, overlay
RESULT = "result"  # outcome, message: the last event of a cycle

# Stages, in cycle order.
PRESENT, INSPECT, PLACE_GOOD, PLACE_BAD = "present", "inspect", "place_good", "place_bad"
PLACEMENTS = (PLACE_GOOD, PLACE_BAD)
ACTION = "action"  # a free-text instruction run on the policy, outside the sorting cycle

# Outcomes of a cycle.
DONE, STOPPED, ERROR = "done", "stopped", "error"

Event = dict[str, Any]


def encode(kind: str, **fields: Any) -> str:
    return json.dumps({"type": kind, **fields}, separators=(",", ":")) + "\n"


def decode(line: str) -> Event:
    event = json.loads(line)
    if not isinstance(event, dict) or "type" not in event:
        raise ValueError(f"Not an event: {line[:80]!r}")
    return event

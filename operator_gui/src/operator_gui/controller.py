"""Runs the robot in a child process and keeps the state the dashboard shows. No UI code here.

* ``start_auto()`` starts ``python -m operator_gui.runner ... cycles``: one process that repeats
  ``run_model.run_cycle`` until told to finish, or until a cycle fails.
* ``finish_auto()`` sends SIGUSR1: the current cycle completes, then the loop ends. Until it does,
  ``Status.draining`` is True and nothing else can be started.
* ``run_action(text)`` starts ``... action <text>``: one ``run_model.run_task``.
* ``stop()`` sends SIGINT, which run_model handles like Ctrl-C (the policy stops and the arm is
  told to hold its measured position); if the runner has not exited after STOP_GRACE_S it is
  terminated, then killed. ``stop(force=True)`` while stopping kills it at once.

This is a software stop only. Keep the robot's physical emergency stop within reach.
"""

import os
import signal
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum

from . import events
from .history import DEFECTIVE, GOOD, History

STOP_GRACE_S = 5.0  # time for the runtime to stop the policy, hold the arm and disconnect
TERMINATE_GRACE_S = 2.0
FINISH_SIGNAL = signal.SIGUSR1  # must match runner.FINISH_SIGNAL
ACTION_DURATION_S = 15.0  # how long a free-text action runs on the policy
LOG_LINES = 300

STAGE_MESSAGES = {
    events.PRESENT: "Presenting the brick to the inspection camera",
    events.INSPECT: "Inspecting the brick",
    events.PLACE_GOOD: "Placing the brick in the good dish",
    events.PLACE_BAD: "Placing the brick in the defective dish",
    events.ACTION: "Running action",
}


class Phase(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    STOPPING = "stopping"


class Kind(str, Enum):
    AUTO = "auto"  # the auto-cycle loop
    ACTION = "action"  # one free-text instruction on the policy


@dataclass
class Status:
    phase: Phase = Phase.IDLE
    kind: Kind = Kind.AUTO  # what the running (or last) run is
    draining: bool = False  # auto cycle asked to finish: the current cycle is completing
    stage: str | None = None  # current stage of the running (or last) run
    outcome: str | None = None  # events.DONE / STOPPED / ERROR once the run has ended
    message: str = "Ready"
    frame: str | None = None  # latest inspection-camera frame, base64 JPEG
    frame_seq: int = 0
    camera_live: bool = False  # the station holds the camera (presentation and inspection)
    look: events.Event | None = None  # latest scored view of the current cycle
    decision: events.Event | None = None  # the current cycle's decision, once made
    cycle: int = 0  # cycles started in the current (or last) auto run

    @property
    def auto(self) -> bool:
        """The auto-cycle loop is running (possibly finishing its last cycle)."""
        return self.phase is not Phase.IDLE and self.kind is Kind.AUTO


class CycleController:
    def __init__(self, source_args: Sequence[str], history: History, python: str = sys.executable,
                 action_duration_s: float = ACTION_DURATION_S):
        """
        Args:
            source_args: ``["--export", PATH]`` or ``["--dry-run"]``, passed to every runner.
            history: Where sorted bricks are recorded.
            python: Interpreter for the runner (default: this one, so the same environment).
            action_duration_s: Control seconds a free-text action runs for.
        """
        self.source_args = list(source_args)
        self.history = history
        self.action_duration_s = action_duration_s
        self._python = python
        self._lock = threading.Lock()
        self._status = Status()
        self._process: subprocess.Popen | None = None
        self._logs: deque[tuple[int, str, str]] = deque(maxlen=LOG_LINES)
        self._log_seq = 0

    @property
    def dry_run(self) -> bool:
        return "--dry-run" in self.source_args

    @property
    def status(self) -> Status:
        with self._lock:
            return replace(self._status)

    def logs_after(self, seq: int) -> list[tuple[int, str, str]]:
        """Log lines (seq, level, message) newer than ``seq``."""
        with self._lock:
            return [line for line in self._logs if line[0] > seq]

    def start_auto(self) -> None:
        self._launch(Kind.AUTO, ["cycles"], "Starting auto cycle")

    def finish_auto(self) -> None:
        """End the auto-cycle loop once the current cycle has completed."""
        with self._lock:
            status, process = self._status, self._process
            if process is None or not status.auto or status.phase is not Phase.RUNNING or status.draining:
                return
            status.draining = True
            status.message = "Shutting down: finishing the current cycle"
        process.send_signal(FINISH_SIGNAL)

    def run_action(self, task: str) -> None:
        task = task.strip()
        if not task:
            raise ValueError("Type an action first")
        self._launch(Kind.ACTION, ["action", task, "--duration", str(self.action_duration_s)],
                     f"Running action: {task}")

    def _launch(self, kind: Kind, mode_args: Sequence[str], message: str) -> None:
        with self._lock:
            if self._process is not None:
                raise RuntimeError("Something is already running")
            read_fd, write_fd = os.pipe()
            try:
                process = subprocess.Popen(
                    [self._python, "-m", "operator_gui.runner", "--events-fd", str(write_fd),
                     *self.source_args, *mode_args],
                    pass_fds=(write_fd,),
                )
            except BaseException:
                os.close(read_fd)
                raise
            finally:
                os.close(write_fd)
            self._process = process
            previous = self._status
            self._status = Status(phase=Phase.RUNNING, kind=kind, message=message,
                                  frame=previous.frame, frame_seq=previous.frame_seq)
        threading.Thread(target=self._follow, args=(process, read_fd), name="runner-events", daemon=True).start()

    def stop(self, force: bool = False) -> None:
        """Interrupt now; ``force`` kills the runner at once, skipping the hold (only while stopping)."""
        with self._lock:
            process = self._process
            if process is None:
                return
            stopping = self._status.phase is Phase.STOPPING
            if stopping and not force:
                return
            self._status.phase = Phase.STOPPING
            self._status.message = "Killing the runner" if stopping else "Stopping: holding the arm"
        if stopping:
            process.kill()
            return
        process.send_signal(signal.SIGINT)
        threading.Thread(target=self._escalate, args=(process,), name="runner-stop", daemon=True).start()

    def shutdown(self) -> None:
        """Stop anything running and wait for the runner to exit (dashboard shutdown)."""
        process = self._process
        self.stop()
        if process is not None:
            process.wait()

    @staticmethod
    def _escalate(process: subprocess.Popen) -> None:
        for wait_s, force in ((STOP_GRACE_S, process.terminate), (TERMINATE_GRACE_S, process.kill)):
            try:
                process.wait(wait_s)
                return
            except subprocess.TimeoutExpired:
                force()

    def _follow(self, process: subprocess.Popen, read_fd: int) -> None:
        with os.fdopen(read_fd, encoding="utf-8") as pipe:
            for line in pipe:
                try:
                    self._apply(events.decode(line))
                except ValueError:
                    continue
        code = process.wait()
        with self._lock:
            self._process = None
            status = self._status
            status.phase = Phase.IDLE
            status.draining = False
            status.camera_live = False
            if status.outcome is None:  # killed, or crashed before reporting
                status.outcome = events.STOPPED if code < 0 else events.ERROR
                status.message = f"Runner exited unexpectedly (code {code})"

    def _apply(self, event: events.Event) -> None:
        kind = event["type"]
        placed: tuple[str, events.Event | None] | None = None
        with self._lock:
            status = self._status
            if kind == events.FRAME:
                status.frame = event["jpeg"]
                status.frame_seq += 1
                status.camera_live = True
            elif kind == events.STAGE:
                status.stage = event["stage"]
                if status.stage == events.PRESENT:  # a new cycle begins
                    status.cycle += 1
                    status.look = status.decision = None
                if status.phase is Phase.RUNNING:
                    text = STAGE_MESSAGES.get(status.stage, status.stage)
                    status.message = f"Shutting down after this cycle · {text}" if status.draining else text
                if status.stage in events.PLACEMENTS and status.kind is Kind.AUTO:
                    # The station releases the camera before placement.
                    status.camera_live = False
                    placed = (DEFECTIVE if status.stage == events.PLACE_BAD else GOOD, status.decision)
            elif kind == events.LOOK:
                status.look = event
            elif kind == events.DECISION:
                status.decision = event
                if not status.draining:
                    status.message = event["message"]
            elif kind == events.LOG:
                self._log_seq += 1
                self._logs.append((self._log_seq, event["level"], event["message"]))
            elif kind == events.RESULT:
                status.outcome = event["outcome"]
                status.message = event["message"]
        if placed is not None:
            # A brick counts once the arm is sent to a dish; dry runs have no crop thumbnail.
            verdict, decision = placed
            decision = decision or {}
            self.history.add(verdict, decision.get("score"), decision.get("looks"), decision.get("thumbnail"))

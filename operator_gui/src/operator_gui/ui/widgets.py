"""Dashboard panels. Each builds itself in the current NiceGUI context and has ``refresh(status)``."""

from collections.abc import Callable

from nicegui import ui

from .. import events
from ..controller import CycleController, Kind, Phase, Status
from ..history import DEFECTIVE, GOOD, History
from . import theme

STEPS = (("Present", (events.PRESENT,)), ("Inspect", (events.INSPECT,)), ("Place", events.PLACEMENTS))
VERDICTS = {  # decision verdict -> (label, colour)
    "good": ("GOOD", theme.GOOD),
    "defective": ("DEFECTIVE", theme.DEFECTIVE),
    "rotate": ("INCONCLUSIVE", theme.ACCENT),
    "empty": ("NO BRICK", theme.DANGER),
}


def jpeg_url(b64: str) -> str:
    return f"data:image/jpeg;base64,{b64}"


class StatusPill:
    def __init__(self):
        self.label = ui.label().classes("pill")

    def refresh(self, status: Status) -> None:
        if status.phase is Phase.RUNNING:
            text, color = "RUNNING", theme.GOOD
        elif status.phase is Phase.STOPPING:
            text, color = "STOPPING", theme.DANGER
        elif status.outcome == events.ERROR:
            text, color = "ERROR", theme.DANGER
        elif status.outcome == events.STOPPED:
            text, color = "STOPPED", theme.DEFECTIVE
        else:
            text, color = "READY", theme.MUTED
        self.label.set_text(text)
        self.label.style(replace=theme.pill_style(color))


class Stepper:
    def __init__(self):
        with ui.row().classes("items-center gap-2"):
            self.steps = []
            for i, (name, _) in enumerate(STEPS):
                if i:
                    ui.icon("chevron_right").classes("muted")
                self.steps.append(ui.label(name).classes("step"))

    def refresh(self, status: Status) -> None:
        current = next((i for i, (_, stages) in enumerate(STEPS) if status.stage in stages), -1)
        finished = status.outcome == events.DONE
        for i, step in enumerate(self.steps):
            step.classes(remove="active done")
            if finished or i < current:
                step.classes("done")
            elif i == current and status.phase is not Phase.IDLE:
                step.classes("active")


class LiveView:
    def __init__(self, dry_run: bool):
        self._seq = -1
        self._dry_run = dry_run
        with ui.element("div").classes("panel flex flex-col gap-3 w-full"):
            with ui.row().classes("items-center justify-between w-full"):
                ui.label("Inspection camera").classes("caption")
                self.badge = ui.label().classes("pill")
            with ui.element("div").classes("relative w-full aspect-video rounded-xl overflow-hidden bg-black"):
                self.image = ui.interactive_image().classes("w-full h-full")
                self.note = ui.label().classes(
                    "absolute inset-x-0 bottom-0 px-4 py-3 mono text-sm bg-black/70 text-center")

    def refresh(self, status: Status) -> None:
        if status.frame and status.frame_seq != self._seq:
            self._seq = status.frame_seq
            self.image.set_source(jpeg_url(status.frame))
        live = status.camera_live and status.phase is not Phase.IDLE
        self.badge.set_text("LIVE" if live else "IDLE")
        self.badge.style(replace=theme.pill_style(theme.DANGER if live else theme.MUTED))
        if live:
            note = ""
        elif not status.frame:
            note = "Dry run: no camera, the verdict is random" if self._dry_run else "Press Auto cycle to start sorting"
        elif status.phase is not Phase.IDLE:
            note = "Camera released while the arm places the brick"
        else:
            note = "Last frame of the previous cycle"
        self.note.set_text(note)
        self.note.set_visibility(bool(note))


class VerdictPanel:
    def __init__(self):
        with ui.element("div").classes("panel flex flex-col gap-4 w-full"):
            ui.label("Latest verdict").classes("caption")
            with ui.row().classes("w-full items-start gap-4 no-wrap"):
                with ui.column().classes("gap-2 grow"):
                    self.verdict = ui.label("—").classes("text-4xl font-bold leading-none")
                    self.score = ui.label("").classes("mono muted")
                self.heatmap = ui.image().classes("w-28 h-28 rounded-lg shrink-0")
            with ui.element("div").classes("track w-full"):
                self.fill = ui.element("div").classes("fill")
                ui.element("div").classes("threshold")
            with ui.row().classes("w-full justify-between mono text-xs muted"):
                ui.label("0")
                ui.label("threshold 0.5")
                ui.label("1")
            self.message = ui.label("").classes("mono text-sm muted")

    def refresh(self, status: Status) -> None:
        decision, look = status.decision, status.look
        source = decision or look
        if source is None:
            self.verdict.set_text("—")
            self.verdict.style(replace=f"color: {theme.MUTED}")
            self.score.set_text("waiting for inspection")
            self.fill.style(replace="width: 0")
            self.heatmap.set_visibility(False)
            self.message.set_text("")
            return
        if decision:
            label, color = VERDICTS.get(decision["verdict"], (decision["verdict"].upper(), theme.TEXT))
            views = decision.get("looks") or 1
            detail = f"{views} view{'s' if views > 1 else ''}"
        else:
            label, color = "SCORING…", theme.ACCENT
            detail = "uncertain, rotating" if look.get("uncertain") else look["status"]
        score = source.get("score")
        self.verdict.set_text(label)
        self.verdict.style(replace=f"color: {color}")
        self.score.set_text(f"score {score:.2f} · {detail}" if score is not None else detail)
        self.fill.style(replace=f"width: {100 * min(max(score or 0, 0), 1):.0f}%; background: {color}")
        self.heatmap.set_visibility(bool(source.get("overlay")))
        if source.get("overlay"):
            self.heatmap.set_source(jpeg_url(source["overlay"]))
        self.message.set_text(decision["message"] if decision else "")


class Kpis:
    def __init__(self, history: History):
        self._history = history
        self._count = -1
        with ui.element("div").classes("grid grid-cols-2 gap-4 w-full"):
            self.total = self._card("Sorted", theme.TEXT)
            self.good = self._card("Intact", theme.GOOD)
            self.bad = self._card("Defective", theme.DEFECTIVE)
            self.rate = self._card("Defect rate", theme.ACCENT)

    @staticmethod
    def _card(title: str, color: str) -> ui.label:
        with ui.element("div").classes("panel flex flex-col gap-1"):
            ui.label(title).classes("caption")
            return ui.label("0").classes("text-5xl font-bold leading-tight").style(f"color: {color}")

    def refresh(self, status: Status) -> None:
        counts = self._history.counts()
        total = counts[GOOD] + counts[DEFECTIVE]
        if total == self._count:
            return
        self._count = total
        self.total.set_text(str(total))
        self.good.set_text(str(counts[GOOD]))
        self.bad.set_text(str(counts[DEFECTIVE]))
        self.rate.set_text(f"{100 * counts[DEFECTIVE] / total:.0f}%" if total else "—")


class BrickTable:
    COLUMNS = [
        {"name": "thumbnail", "label": "Brick", "field": "thumbnail", "align": "left"},
        {"name": "id", "label": "#", "field": "id", "align": "left", "sortable": True},
        {"name": "time", "label": "Time", "field": "time", "align": "left"},
        {"name": "verdict", "label": "Verdict", "field": "verdict", "align": "left"},
        {"name": "score", "label": "Score", "field": "score", "align": "right"},
        {"name": "looks", "label": "Views", "field": "looks", "align": "right"},
    ]

    def __init__(self, history: History, thumbnails_url: str):
        self._history = history
        self._url = thumbnails_url
        self._count = -1
        with ui.element("div").classes("panel w-full flex flex-col gap-2"):
            ui.label("Sorted bricks").classes("caption")
            self.table = ui.table(
                rows=[], columns=self.COLUMNS, row_key="id",
                pagination={"rowsPerPage": 8, "sortBy": "id", "descending": True},
            ).classes("w-full").props("flat dark")
        self.table.add_slot("body-cell-thumbnail", """
            <q-td :props="props">
              <img v-if="props.value" :src="props.value" style="width:44px;height:44px;object-fit:cover;border-radius:8px">
              <span v-else class="muted">—</span>
            </q-td>""")
        self.table.add_slot("body-cell-verdict", f"""
            <q-td :props="props">
              <span class="pill" :style="props.value === '{DEFECTIVE}'
                ? '{theme.pill_style(theme.DEFECTIVE)}' : '{theme.pill_style(theme.GOOD)}'">{{{{ props.value }}}}</span>
            </q-td>""")

    def refresh(self, status: Status) -> None:
        records = self._history.records
        if len(records) == self._count:
            return
        self._count = len(records)
        self.table.update_rows([
            {
                "id": r.id,
                "time": r.time[11:],
                "verdict": r.verdict,
                "score": f"{r.score:.2f}" if r.score is not None else "—",
                "looks": r.looks or "—",
                "thumbnail": f"{self._url}/{r.thumbnail}" if r.thumbnail else "",
            }
            for r in records
        ])


class LogPanel:
    def __init__(self, controller: CycleController):
        self._controller = controller
        self._seq = 0
        with ui.element("div").classes("panel w-full flex flex-col gap-2"):
            ui.label("Cycle log").classes("caption")
            self.log = ui.log(max_lines=300).classes("mono text-xs h-44 w-full")

    def refresh(self, status: Status) -> None:
        for seq, level, message in self._controller.logs_after(self._seq):
            self._seq = seq
            self.log.push(f"{level[:4]}  {message}")


class Controls:
    """Auto cycle toggle and a free-text action prompt.

    Auto cycle toggles the loop; switching it off lets the current cycle finish ("Shutting down...").
    The send button becomes an immediate stop while an action runs. Esc / Space stop anything at once.
    """

    PROMPT = "Run action..."
    SHUTTING_DOWN = "Shutting down..."

    def __init__(self, controller: CycleController, on_error: Callable[[str], None]):
        self._controller = controller
        self._on_error = on_error
        with ui.row().classes("w-full items-center gap-4 no-wrap"):
            with ui.button(on_click=self._toggle_auto).props("unelevated size=lg text-color=black") \
                    .classes("px-6 font-bold") as self.auto:
                self.auto_icon = ui.icon("autorenew").classes("mr-2")
                ui.label("Auto cycle")
            with ui.row().classes("grow items-center gap-2 no-wrap"):
                self.prompt = ui.input(placeholder=self.PROMPT).props("outlined dark dense clearable") \
                    .classes("grow mono").on("keydown.enter", self._send)
                self.send = ui.button(icon="send", on_click=self._send_or_stop) \
                    .props("round unelevated size=md text-color=black")
                self.reset = ui.button("Reset", icon="restart_alt", on_click=self._reset, color="grey-8") \
                    .props("unelevated").tooltip("Move the arm back to its home pose")
            self.kill = ui.button("Force kill", icon="dangerous", on_click=lambda: controller.stop(force=True),
                                  color="negative").props("unelevated")
            ui.label("Esc / Space = STOP · software stop only: keep the robot's physical E-stop in reach") \
                .classes("mono text-xs muted max-w-xs")

    def _toggle_auto(self) -> None:
        if self._controller.status.auto:
            self._controller.finish_auto()
            return
        try:
            self._controller.start_auto()
        except Exception as error:
            self._on_error(f"Could not start: {error}")

    def _send(self) -> None:
        if self._controller.status.phase is not Phase.IDLE:
            return
        try:
            self._controller.run_action(self.prompt.value or "")
        except Exception as error:
            self._on_error(str(error))

    def _reset(self) -> None:
        try:
            self._controller.reset()
        except Exception as error:
            self._on_error(f"Could not reset: {error}")

    def _send_or_stop(self) -> None:
        if self._controller.status.phase is Phase.RUNNING:
            self._controller.stop()
        else:
            self._send()

    def refresh(self, status: Status) -> None:
        idle = status.phase is Phase.IDLE
        running = status.phase is Phase.RUNNING
        action = status.kind is Kind.ACTION
        # Auto cycle: starts when idle, asks the loop to finish while it runs; the icon spins until it ends.
        self.auto.set_enabled(idle or (status.auto and running and not status.draining))
        if status.auto:
            self.auto_icon.classes(add="animate-spin")
        else:
            self.auto_icon.classes(remove="animate-spin")
        # Prompt: free only while idle; shows "Shutting down..." until the last auto cycle ends.
        self.prompt.set_enabled(idle)
        self.reset.set_enabled(idle)
        if status.auto and status.draining:
            self.prompt.set_value("")
            self.prompt.props(f'placeholder="{self.SHUTTING_DOWN}"')
        else:
            self.prompt.props(f'placeholder="{self.PROMPT}"')
        # The send button becomes the stop for a running action.
        self.send.set_enabled(idle or (running and action))
        stop_action = running and action
        self.send.props(f"icon={'stop' if stop_action else 'send'} color={'negative' if stop_action else 'primary'}")
        self.kill.set_visibility(status.phase is Phase.STOPPING)

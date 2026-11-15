"""Dashboard panels. Each builds itself in the current NiceGUI context and has ``refresh(status)``."""

from collections.abc import Callable
from typing import Any

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


class DeviceMeters:
    """CPU / iGPU / NPU usage sparklines with a temperature readout, for the top bar.

    Each device keeps the last HISTORY samples and draws them as an inline SVG polyline, so there
    is no charting dependency and the markup stays small enough to refresh a few times a second.
    A device whose counters the kernel does not expose shows "n/a" instead of a misleading 0%.
    """

    HISTORY = 120  # samples kept per device; at SAMPLE_S this is a few seconds of detail
    WIDTH = 96
    HEIGHT = 14

    def __init__(self) -> None:
        from ..telemetry import Telemetry

        self._telemetry = Telemetry()
        self._telemetry.start()
        self._history: dict[str, list[float]] = {}
        self._rows: dict[str, tuple[Any, Any, Any]] = {}
        # One narrow row per device: label, graph, then the number, so the block stays slim.
        with ui.column().classes("gap-0"):
            for name in ("CPU", "iGPU", "NPU"):
                with ui.row().classes("items-center gap-2 no-wrap"):
                    label = ui.label(name).classes("caption w-9 text-right")
                    graph = ui.html("").classes("leading-none")
                    value = ui.label("--").classes("mono text-xs w-24")
                    self._rows[name] = (graph, value, label)

    IDLE_PERCENT = 1.0  # at or below this a device counts as unused and is drawn grey

    def _spark(self, samples: list[float], colour: str) -> str:
        """The samples as an SVG polyline in ``colour``, newest on the right."""
        w, h = self.WIDTH, self.HEIGHT
        if not samples:
            return (f'<svg width="{w}" height="{h}"><rect width="{w}" height="{h}" rx="3" '
                    f'fill="{colour}14"/></svg>')
        step = w / max(len(samples) - 1, 1)
        points = " ".join(
            f"{i * step:.1f},{h - 1 - (min(max(v, 0.0), 100.0) / 100.0) * (h - 2):.1f}"
            for i, v in enumerate(samples)
        )
        # The filled area closes down to the baseline, so a flat trace still reads as a graph.
        area = f"0,{h} {points} {(len(samples) - 1) * step:.1f},{h}"
        return (
            f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<rect width="{w}" height="{h}" rx="3" fill="{colour}14"/>'
            f'<polygon points="{area}" fill="{colour}33"/>'
            f'<polyline points="{points}" fill="none" stroke="{colour}" '
            f'stroke-width="1.5" stroke-linejoin="round"/>'
            f"</svg>"
        )

    def refresh(self, status: Status) -> None:
        for reading in self._telemetry.read():
            graph, value, label = self._rows[reading.name]
            samples = self._history.setdefault(reading.name, [])
            # Append every background sample, so a burst shorter than this refresh still shows.
            if reading.trace:
                samples.extend(reading.trace)
            elif reading.percent is not None:
                samples.append(reading.percent)
            del samples[: max(0, len(samples) - self.HISTORY)]
            # Grey while the device is doing nothing: an unused accelerator should not read as live.
            recent = samples[-20:]
            busy = bool(reading.active) or (max(recent) > self.IDLE_PERCENT if recent else False)
            colour = theme.TELEMETRY if busy else theme.MUTED
            graph.set_content(self._spark(samples, colour))
            if reading.percent is None:
                percent = "n/a"
            else:
                # "~" marks a proxy reading (GPU clock ratio) rather than a true busy counter.
                percent = f"{'~' if reading.approximate else ''}{reading.percent:.0f}%"
            # A device can be powered and working between two samples that both read 0%.
            if reading.percent is not None and reading.percent < 1 and reading.active:
                percent = "busy"
            temperature = "" if reading.celsius is None else f" {reading.celsius:.0f}°C"
            value.set_text(f"{percent}{temperature}")
            label.style(f"color: {colour}")
            value.style(f"color: {colour if busy else theme.MUTED}")


class ModelPicker:
    """Chooses which Studio export the next run uses. Disabled while anything is running."""

    def __init__(self, controller: CycleController, on_error: Callable[[str], None]):
        self._controller = controller
        self._on_error = on_error
        self._by_label = {m.label: m for m in controller.models}
        current = controller.model
        with ui.row().classes("items-center gap-2 no-wrap"):
            ui.icon("smart_toy", size="20px").classes("muted")
            ui.label("Model").classes("caption")
            if controller.dry_run:
                self.select = None
                ui.label("dry run").classes("mono muted")
            elif not self._by_label:
                self.select = None
                ui.label("no export found").classes("mono").style(f"color: {theme.DANGER}")
            else:
                self.select = ui.select(
                    list(self._by_label), value=current.label if current else None,
                    on_change=self._choose,
                ).props("outlined dark dense options-dense").classes("mono grow max-w-[520px]") \
                    .tooltip("Studio OpenVINO export the next run uses")

    def _choose(self, event: Any) -> None:
        model = self._by_label.get(event.value)
        if model is None or model is self._controller.model:
            return
        try:
            self._controller.select_model(model)
        except Exception as error:
            self._on_error(str(error))
            current = self._controller.model
            if self.select is not None and current is not None:
                self.select.set_value(current.label)

    def refresh(self, status: Status) -> None:
        if self.select is not None:
            self.select.set_enabled(status.phase is Phase.IDLE)


class LiveView:
    def __init__(self, dry_run: bool, controller: CycleController | None = None,
                 on_error: Callable[[str], None] | None = None):
        self._seq = -1
        self._dry_run = dry_run
        self._controller = controller
        self._on_error = on_error or (lambda _text: None)
        with ui.element("div").classes("panel flex flex-col gap-3 w-full"):
            with ui.row().classes("items-center justify-between w-full"):
                ui.label("Inspection camera").classes("caption")
                self.badge = ui.label().classes("pill")
            with ui.element("div").classes("relative w-full aspect-video rounded-xl overflow-hidden bg-black"):
                self.image = ui.interactive_image().classes("w-full h-full")
                self.note = ui.label().classes(
                    "absolute inset-x-0 bottom-0 px-4 py-3 mono text-sm bg-black/70 text-center")
                # Inspect the current view on demand; the idle preview already holds the camera.
                self.analyze = ui.button("Analyze", icon="biotech", on_click=self._analyze) \
                    .props("unelevated size=md text-color=black") \
                    .classes("absolute bottom-3 right-3 px-4 font-bold z-10") \
                    .tooltip("Inspect what the camera sees now, without moving the robot")

    def _analyze(self) -> None:
        if self._controller is None:
            return
        try:
            self._controller.analyze()
        except Exception as error:
            self._on_error(str(error))

    def refresh(self, status: Status) -> None:
        if status.frame and status.frame_seq != self._seq:
            self._seq = status.frame_seq
            self.image.set_source(jpeg_url(status.frame))
        # Only possible while the preview holds the camera, i.e. while nothing else runs.
        self.analyze.set_enabled(status.phase is Phase.IDLE and status.preview_live)
        live = (status.camera_live and status.phase is not Phase.IDLE) or status.preview_live
        self.badge.set_text("LIVE" if live else "IDLE")
        self.badge.style(replace=theme.pill_style(theme.DANGER if live else theme.MUTED))
        if live:
            note = ""
        elif not status.frame:
            note = "Dry run: no camera, the verdict is random" if self._dry_run else "Waiting for the camera"
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
    """Auto-cycle toggle, a one-cycle button, an E-stop, and a free-text action prompt.

    The circular icon toggles the auto loop; switching it off lets the current cycle finish
    ("Shutting down..."). "Run" sorts exactly one brick and stops. The send button becomes an
    immediate stop while an action runs. STOP halts everything and is never disabled.
    Esc / Space do the same. All of this is a software stop: the physical E-stop is the real one.
    """

    PROMPT = "Run action..."
    SHUTTING_DOWN = "Shutting down..."

    def __init__(self, controller: CycleController, on_error: Callable[[str], None]):
        self._controller = controller
        self._on_error = on_error
        with ui.row().classes("w-full items-center gap-4 no-wrap"):
            with ui.button(on_click=self._toggle_auto) \
                    .props("round unelevated size=lg text-color=black") \
                    .tooltip("Auto cycle: sort bricks until stopped") as self.auto:
                self.auto_icon = ui.icon("autorenew")
            with ui.button(on_click=self._run_once) \
                    .props("unelevated size=lg text-color=black").classes("px-5 font-bold") as self.once:
                ui.icon("play_arrow").classes("mr-2")
                ui.label("Run")
            self.once.tooltip("Sort one brick, then stop")
            with ui.row().classes("grow items-center gap-2 no-wrap"):
                self.prompt = ui.input(placeholder=self.PROMPT).props("outlined dark dense clearable") \
                    .classes("grow mono").on("keydown.enter", self._send)
                self.send = ui.button(icon="send", on_click=self._send_or_stop) \
                    .props("round unelevated size=md text-color=black")
                self.reset = ui.button("Reset", icon="restart_alt", on_click=self._reset, color="grey-8") \
                    .props("unelevated").tooltip("Move the arm back to its home pose")
                # Always visible and always enabled: an E-stop the operator never has to hunt for.
                self.estop = ui.button(icon="pan_tool", on_click=self._estop, color="negative") \
                    .props("unelevated").tooltip("Stop everything now and hold the arm (software stop)")
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

    def _run_once(self) -> None:
        """Run one cycle, or stop the single cycle that is already running."""
        status = self._controller.status
        if status.phase is Phase.RUNNING and status.kind is Kind.ONCE:
            self._controller.stop()
            return
        try:
            self._controller.start_once()
        except Exception as error:
            self._on_error(f"Could not start: {error}")

    def _send(self) -> None:
        if self._controller.status.phase is not Phase.IDLE:
            return
        try:
            self._controller.run_action(self.prompt.value or "")
        except Exception as error:
            self._on_error(str(error))

    def _estop(self) -> None:
        """Stop whatever is running at once. Safe to press when idle: stop() is then a no-op."""
        self._controller.stop()

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
        once = status.kind is Kind.ONCE
        # Auto cycle: starts when idle, asks the loop to finish while it runs; the icon spins until it ends.
        self.auto.set_enabled(idle or (status.auto and running and not status.draining))
        if status.auto:
            self.auto_icon.classes(add="animate-spin")
        else:
            self.auto_icon.classes(remove="animate-spin")
        # Run: one cycle from idle, and an immediate stop while that cycle runs.
        self.once.set_enabled(idle or (running and once))
        stop_once = running and once
        self.once.props(f"color={'negative' if stop_once else 'primary'}")
        # Prompt: free only while idle; shows "Shutting down..." until the last auto cycle ends.
        self.prompt.set_enabled(idle)
        self.reset.set_enabled(idle)
        self.estop.set_enabled(True)
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

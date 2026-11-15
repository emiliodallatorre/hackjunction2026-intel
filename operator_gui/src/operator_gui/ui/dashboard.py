"""The operator page: composes the panels and refreshes them from the controller's status."""

from nicegui import app, ui
from nicegui.events import KeyEventArguments

from .. import events
from ..controller import CycleController, Phase, Status
from . import theme
from .widgets import (BrickTable, Controls, Kpis, LiveView, LogPanel, ModelPicker, Stepper,
                      StatusPill, VerdictPanel)

REFRESH_S = 0.1
THUMBNAILS_URL = "/thumbnails"
STOP_KEYS = ("Escape", " ")


class Dashboard:
    def __init__(self, controller: CycleController):
        self.controller = controller
        self._last: Status | None = None
        theme.apply()

        with ui.header().classes("items-center justify-between px-6 py-3").style(
                f"background: {theme.BG}; border-bottom: 1px solid {theme.BORDER}"):
            with ui.row().classes("items-center gap-3"):
                ui.icon("view_in_ar", size="28px").style(f"color: {theme.ACCENT}")
                ui.label("Brick Sorter").classes("text-xl font-bold")
                ui.label("operator").classes("mono muted")
            self.stepper = Stepper()
            with ui.row().classes("items-center gap-3"):
                self.picker = ModelPicker(controller, on_error=lambda text: ui.notify(text, type="negative"))
                mode = "DRY RUN" if controller.dry_run else "ROBOT"
                ui.label(mode).classes("pill").style(theme.pill_style(theme.ACCENT if controller.dry_run else theme.GOOD))
                self.pill = StatusPill()

        with ui.element("div").classes("w-full max-w-[1600px] mx-auto p-6 grid grid-cols-1 lg:grid-cols-3 gap-6"):
            with ui.column().classes("lg:col-span-2 gap-6"):
                self.live = LiveView(controller.dry_run)
                self.message = ui.label().classes("text-lg font-medium")
                self.log = LogPanel(controller)
            with ui.column().classes("gap-6"):
                self.verdict = VerdictPanel()
                self.kpis = Kpis(controller.history)
            with ui.element("div").classes("lg:col-span-3"):
                self.table = BrickTable(controller.history, THUMBNAILS_URL)

        with ui.footer().classes("px-6 py-4").style(
                f"background: {theme.PANEL}; border-top: 1px solid {theme.BORDER}"):
            self.controls = Controls(controller, on_error=lambda text: ui.notify(text, type="negative"))

        self.panels = (self.stepper, self.pill, self.live, self.verdict, self.kpis, self.table, self.log,
                       self.controls, self.picker)
        ui.keyboard(on_key=self._on_key)
        ui.timer(REFRESH_S, self.refresh)

    def refresh(self) -> None:
        status = self.controller.status
        if status == self._last:
            return
        self._notify(self._last, status)
        self._last = status
        self.message.set_text(status.message)
        for panel in self.panels:
            panel.refresh(status)

    @staticmethod
    def _notify(before: Status | None, now: Status) -> None:
        if before is None:
            return
        if now.decision and now.decision != before.decision:
            if now.decision["verdict"] == "defective":
                ui.notify(now.decision["message"], type="warning", position="top", timeout=4000)
        if now.outcome != before.outcome and now.outcome in (events.ERROR, events.STOPPED):
            ui.notify(now.message, type="negative", position="top", timeout=0, close_button="OK")

    def _on_key(self, e: KeyEventArguments) -> None:
        # Never escalates to a kill: that needs the on-screen "Force kill" button.
        if e.action.keydown and not e.action.repeat and e.key.name in STOP_KEYS:
            if self.controller.status.phase is Phase.RUNNING:
                self.controller.stop()


def register(controller: CycleController) -> None:
    """Serve the dashboard at / for every browser that connects; all share the one controller."""
    app.add_static_files(THUMBNAILS_URL, controller.history.thumbnails, max_cache_age=0)
    ui.page("/")(lambda: Dashboard(controller))

"""Colours, fonts and the few CSS classes the dashboard uses; layout itself is Tailwind."""

from string import Template

from nicegui import ui

BG = "#0E0F12"
PANEL = "#16181D"
BORDER = "#262A31"
TEXT = "#F2F0EA"
MUTED = "#8A9099"
ACCENT = "#FFC72C"
GOOD = "#5AB4F0"
DEFECTIVE = "#FF7A1A"
DANGER = "#E5484D"

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
    'family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap">'
)

CSS = Template("""
body { background: $BG; color: $TEXT; font-family: 'Space Grotesk', system-ui, sans-serif; }
.mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }
.muted { color: $MUTED; }
.panel { background: $PANEL; border: 1px solid $BORDER; border-radius: 16px; padding: 20px; }
.caption { color: $MUTED; font-family: 'JetBrains Mono', monospace; font-size: 12px;
           letter-spacing: .12em; text-transform: uppercase; }
.pill { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; padding: 4px 12px;
        font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 600; }
.pill::before { content: ''; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.step { font-family: 'JetBrains Mono', monospace; font-size: 13px; padding: 4px 12px; border-radius: 8px;
        color: $MUTED; border: 1px solid $BORDER; }
.step.done { color: $TEXT; }
.step.active { color: $ACCENT; border-color: $ACCENT; box-shadow: 0 0 12px rgba(255,199,44,.25); }
.track { position: relative; height: 10px; border-radius: 999px; background: $BORDER; overflow: visible; }
.track .fill { position: absolute; inset: 0 auto 0 0; border-radius: 999px; transition: width .3s; }
.track .threshold { position: absolute; top: -4px; bottom: -4px; width: 2px; background: $TEXT; left: 50%; }
.q-table__card, .q-table { background: transparent !important; color: $TEXT; }
.q-table th { color: $MUTED; font-family: 'JetBrains Mono', monospace; text-transform: uppercase; }
.q-table tbody td { border-color: $BORDER !important; }
.q-log { background: #0A0B0D; border-radius: 12px; color: #C9CCD1; }
""").substitute(BG=BG, PANEL=PANEL, BORDER=BORDER, TEXT=TEXT, MUTED=MUTED, ACCENT=ACCENT)


def pill_style(color: str) -> str:
    return f"color: {color}; background: {color}22;"


def apply() -> None:
    """Call once per page, before building it."""
    ui.add_head_html(FONTS)
    ui.add_css(CSS)
    ui.colors(primary=ACCENT, positive=GOOD, warning=DEFECTIVE, negative=DANGER, dark=PANEL, dark_page=BG)
    ui.dark_mode(True)

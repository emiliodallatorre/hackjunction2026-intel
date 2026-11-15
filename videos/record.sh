#!/usr/bin/env bash
# Record presentation.html to a narrated video file.
#
# HOW IT WORKS
# ------------
# The deck narrates with the browser's speechSynthesis API. That audio exists
# only as sound coming out of the browser, so the only way to capture it is to
# record real playback: ffmpeg grabs the X11 screen for picture and the
# PulseAudio/PipeWire *monitor* source (what the speakers are playing) for sound.
#
# Runtime is computed from the deck itself (sum of data-dur + the 600 ms gap
# show() inserts between slides), so the recording stops on its own.
#
# Usage:
#   ./record.sh                # record fullscreen deck -> deck.mp4
#   OUT=foo.mp4 ./record.sh    # choose the output name
#   ./record.sh --list-sources # show audio sources if the default is wrong
#
# Requires: chromium, ffmpeg, X11, pipewire/pulse. Run it from a graphical
# session -- it opens a real browser window, so do not touch the machine while
# it records.

set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "--list-sources" ]]; then
  pactl list short sources
  exit 0
fi

OUT="${OUT:-deck.mp4}"
FPS="${FPS:-30}"

command -v chromium >/dev/null || { echo "need chromium"; exit 1; }
[[ -n "${DISPLAY:-}" ]] || { echo "need a graphical session (DISPLAY unset)"; exit 1; }

# The monitor source carries whatever the speakers play, narration included.
AUDIO="${AUDIO:-$(pactl get-default-sink 2>/dev/null).monitor}"
pactl list short sources | grep -q "^[0-9]*	${AUDIO}	" \
  || AUDIO="$(pactl list short sources | awk '/\.monitor\t/{print $2; exit}')"
[[ -n "$AUDIO" ]] || { echo "no monitor source; run --list-sources"; exit 1; }

# Deck length, straight from the markup, so edits to data-dur stay in sync.
# Deck length from the markup (sum of data-dur + the 600 ms gap show() adds),
# plus a 3 s tail. MAXLEN caps it -- the deck is paced for ~2 min.
TOTAL=$(grep -o 'data-dur="[0-9.]*"' presentation.html | sed 's/[^0-9.]//g' \
        | awk '{s+=$1; n++} END {printf "%.0f", s + n*0.6 + 3}')
if [[ -n "${MAXLEN:-}" && "$TOTAL" -gt "$MAXLEN" ]]; then
  echo "note: deck runs ${TOTAL}s, capping recording at ${MAXLEN}s (MAXLEN)"
  TOTAL="$MAXLEN"
fi
# xdpyinfo is not always installed; fall back to xrandr, then to 1920x1080.
SIZE="${SIZE:-}"
if [[ -z "$SIZE" ]]; then
  SIZE=$(xdpyinfo 2>/dev/null | awk '/dimensions:/{print $2; exit}') \
    || SIZE=$(xrandr 2>/dev/null | awk '/\*/{print $1; exit}') || true
fi
SIZE="${SIZE:-1920x1080}"

echo "recording ${TOTAL}s at ${SIZE} from ${AUDIO} -> ${OUT}"

# ?autoplay=1&audio=1&clean=1 makes the run deterministic: the deck advances by
# itself, narrates, and hides the HUD pills -- no key presses, same every time.
# --autoplay-policy is a separate thing: it lets the background <video>s start
# without a click. A throwaway profile keeps your real tabs out of the capture.
PROFILE="$(mktemp -d)"
trap 'rm -rf "$PROFILE"' EXIT
chromium \
  --user-data-dir="$PROFILE" \
  --new-window --start-fullscreen \
  --autoplay-policy=no-user-gesture-required \
  --disable-session-crashed-bubble --no-first-run --disable-infobars \
  "file://$PWD/presentation.html?autoplay=1&audio=1&clean=1#0" &
CHROME=$!

sleep 4   # window map + fullscreen transition + first slide paint

ffmpeg -y -hide_banner -loglevel error -stats \
  -f x11grab -framerate "$FPS" -video_size "$SIZE" -i "$DISPLAY" \
  -f pulse -i "$AUDIO" \
  -t "$TOTAL" \
  -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p \
  -c:a aac -b:a 192k \
  -movflags +faststart \
  "$OUT"

kill "$CHROME" 2>/dev/null || true
wait "$CHROME" 2>/dev/null || true

echo
echo "done -> $OUT  ($(ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" | cut -c1-5)s)"
echo "if the narration is missing, the monitor source was wrong: ./record.sh --list-sources"

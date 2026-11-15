#!/usr/bin/env bash
# Transcode recorded clips into browser-playable H.264 for presentation.html.
#
# Phone/camera footage is usually 4K HEVC (H.265). Browsers do NOT reliably
# decode HEVC in a <video> tag, and a 3840x2160 file is far more than a 1080p
# slide needs. This downscales to 1080p H.264, strips audio (slides are muted
# and loop), and writes web/<name>.mp4.
#
# Speed: each clip has a playback multiplier below. 2 means twice as fast, so a
# 35 s take plays in 17.5 s. This costs nothing extra -- setpts rewrites frame
# timestamps inside the same encode pass that was already happening.
#
# Usage:
#   ./prepare.sh                     # convert every source clip listed below
#   ./prepare.sh 02-pick.mp4         # convert just one, at its table speed
#   ./prepare.sh 04-full-run.mp4 3   # convert one, overriding the speed to 3x
#   FORCE=1 ./prepare.sh             # re-encode even if outputs look current
#
# Source clips live in this folder; outputs go to videos/web/.

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p web

# clip -> playback speed multiplier. Tune these, re-run, reload the deck.
speed_for() {
  case "$1" in
    01-intro.mp4)              echo 1.5 ;;
    02-pick.mp4)               echo 2   ;;
    02-place.mp4)              echo 2.5 ;;
    03-orange-tray.mp4)        echo 2.5 ;;
    04-full-run.mp4)           echo 3   ;;  # 35 s one-take -> ~12 s
    06-training.mp4)           echo 2   ;;  # screen capture: slow enough to read the UI
    *)                         echo 1   ;;
  esac
}

# The dashboard take is one 112 s screen recording. Keep 0-23, 27-40, 40-50,
# 55-71 and 76-82; the 27-40 stretch (the anomaly firing) runs at 2x so it stays
# readable, everything else at 1.5x. Result: 43 s.
dashboard_cut() {
  local src="${1:?usage: dashboard_cut <recording.mp4>}"
  ffmpeg -y -loglevel error -stats -i "$src" -filter_complex "
    [0:v]trim=0:23,setpts=PTS-STARTPTS,setpts=PTS/1.5[a];
    [0:v]trim=27:40,setpts=PTS-STARTPTS,setpts=PTS/2[b];
    [0:v]trim=40:50,setpts=PTS-STARTPTS,setpts=PTS/1.5[c];
    [0:v]trim=55:71,setpts=PTS-STARTPTS,setpts=PTS/1.5[d];
    [0:v]trim=76:82,setpts=PTS-STARTPTS,setpts=PTS/1.5[e];
    [a][b][c][d][e]concat=n=5:v=1:a=0,fps=30,scale=-2:1080[out]" \
    -map "[out]" -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
    -movflags +faststart -an web/07-dashboard.mp4
}

convert() {
  local src="$1"
  local speed="${2:-$(speed_for "$src")}"
  local out="web/${src%.*}.mp4"

  if [[ ! -f "$src" ]]; then
    echo "skip   $src (not recorded yet)"
    return
  fi
  if [[ -z "${FORCE:-}" && -f "$out" && "$out" -nt "$src" ]]; then
    echo "ok     $out (already up to date; FORCE=1 to redo)"
    return
  fi

  local filter="scale=-2:1080"
  if [[ "$speed" != "1" ]]; then
    # PTS/speed moves each frame earlier; fps= resamples to a clean 30.
    filter="setpts=PTS/${speed},${filter},fps=30"
  fi

  echo "encode $src -> $out  (${speed}x)"
  ffmpeg -y -loglevel error -stats -i "$src" \
    -vf "$filter" \
    -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p \
    -movflags +faststart -an \
    "$out"

  local dur
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$out" 2>/dev/null || echo "?")
  printf '       -> %.1fs\n' "$dur" 2>/dev/null || echo "       -> ${dur}s"
}

if [[ $# -gt 0 ]]; then
  convert "$@"
else
  # 05 is a screenshot: cp 05-anomaly-sidebyside.png web/
  # 07 is cut + speed-ramped from the 112 s screen recording; to redo it:
  #   see dashboard_cut() above -- run it with the recording's filename.
  for f in 01-intro.mp4 02-pick.mp4 02-place.mp4 03-orange-tray.mp4 \
           04-full-run.mp4 06-training.mp4; do
    convert "$f"
  done
fi

echo
echo "done. reload presentation.html"

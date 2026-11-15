#!/usr/bin/env bash
# Run the present-and-inspect demo on roberta over SSH, inside the Physical AI Studio image.
# Moves the robot: supervise it. Extra arguments go to inspection_demo, e.g. --present-timeout 60.
set -euo pipefail

HOST="${ROBERTA_HOST:-intel@roberta}"
EXPORT="${EXPORT:-act-single-block-66-runtime.zip}"  # in ~/Downloads on roberta

ssh -o ConnectTimeout=30 "$HOST" bash -s -- "$EXPORT" "$@" <<'EOF'
set -euo pipefail
export_zip="$1"; shift
img=$(docker inspect -f '{{.Config.Image}}' physical-ai-studio)
docker run --rm --user 0 --privileged \
  -v /dev:/dev -v /run/udev:/run/udev:ro \
  -v ~/Desktop/hackjunction/hackjunction2026-intel:/repo -v ~/Downloads:/downloads \
  -w /repo -e PYTHONPATH=/repo/src \
  --entrypoint /app/application/backend/.venv/bin/python "$img" \
  -m hackjunction2026_intel.inspection_demo \
  --export "/downloads/$export_zip" --report /downloads/inspection.png "$@"
EOF

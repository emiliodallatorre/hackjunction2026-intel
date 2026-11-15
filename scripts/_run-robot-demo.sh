#!/usr/bin/env bash
set -euo pipefail

mode="${1:?Expected inspection or sorting}"; shift
case "$mode" in
  inspection|sorting) ;;
  *) echo "Unknown robot demo: $mode" >&2; exit 2 ;;
esac

HOST="${ROBERTA_HOST:-intel@roberta}"
EXPORT="${EXPORT:-pi05-ricotta-runtime}"  # directory or zip in ~/Downloads on roberta

ssh -o ConnectTimeout=30 "$HOST" bash -s -- "$EXPORT" "$mode" "$@" <<'EOF'
set -euo pipefail
export_name="$1"; mode="$2"; shift 2
img=$(docker inspect -f '{{.Config.Image}}' physical-ai-studio)
case "$mode" in
  inspection)
    python_args=(-m hackjunction2026_intel.inspection_demo)
    extra_args=(--report /downloads/inspection.png)
    ;;
  sorting)
    python_args=(-c '
import logging
import sys
from hackjunction2026_intel import inspection_demo, sorting

sorting.PRESENT_DURATION_S = inspection_demo.DEFAULT_PRESENT_TIMEOUT_S
sorting.GOOD_DURATION_S = 45.0
sorting.BAD_DURATION_S = 45.0
sys.argv[0] = "sorting.py"
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
sorting.main()
')
    extra_args=()
    ;;
esac
docker run --rm --user 0 --privileged \
  -v /dev:/dev -v /run/udev:/run/udev:ro \
  -v ~/Desktop/hackjunction/hackjunction2026-intel:/repo -v ~/Downloads:/downloads \
  -w /repo -e PYTHONPATH=/repo/src \
  --entrypoint /app/application/backend/.venv/bin/python "$img" \
  "${python_args[@]}" --export "/downloads/$export_name" "${extra_args[@]}" "$@"
EOF

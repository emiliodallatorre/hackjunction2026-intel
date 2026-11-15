#!/usr/bin/env bash
# Start the operator dashboard.
#
#   ./start-gui.sh                      discover exports and let the GUI's picker choose
#   ./start-gui.sh path/to/export       start on this export (folder or .zip)
#   ./start-gui.sh --dry-run            no robot, camera or model
#
# An export is a folder (or .zip) holding runtime.yaml and exports/openvino/. With no argument the
# dashboard searches ~/Downloads, your home folder and here, and starts on the first it finds.
# Extra flags are passed straight to operator-gui, e.g. ./start-gui.sh --port 9000
set -euo pipefail
cd "$(dirname "$0")"

EXPORT=""
if [[ $# -gt 0 && $1 != -* ]]; then
    EXPORT=$1
    shift
fi

if [[ -n $EXPORT && ! -e $EXPORT ]]; then
    echo "Export not found: $EXPORT" >&2
    exit 1
fi

# --dry-run needs no robot extra, so it starts without installing the heavy dependencies.
if [[ " $* " == *" --dry-run "* ]]; then
    exec uv run --with-editable ./operator_gui operator-gui "$@"
fi

if [[ -n $EXPORT ]]; then
    echo "Starting on export: $EXPORT"
    exec uv run --extra robot --with-editable ./operator_gui operator-gui --export "$EXPORT" "$@"
fi

echo "Discovering Studio exports (~/Downloads, home, here)..."
exec uv run --extra robot --with-editable ./operator_gui operator-gui "$@"

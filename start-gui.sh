#!/usr/bin/env bash
# Start the operator dashboard.
#
#   ./start-gui.sh                      dry run: no robot, camera or model
#   ./start-gui.sh path/to/export.zip   real robot, using that Studio OpenVINO export
#   ./start-gui.sh ~/Downloads          real robot, picking the newest *openvino*.zip in there
#
# Extra flags are passed straight to operator-gui, e.g. ./start-gui.sh --port 9000
set -euo pipefail
cd "$(dirname "$0")"

EXPORT=""
if [[ $# -gt 0 && $1 != -* ]]; then
    EXPORT=$1
    shift
fi

# A directory that is not itself an export: take the newest OpenVINO zip inside it.
if [[ -n $EXPORT && -d $EXPORT && ! -f $EXPORT/runtime.yaml ]]; then
    FOUND=$(find "$EXPORT" -maxdepth 1 -iname '*openvino*.zip' -printf '%T@ %p\n' 2>/dev/null |
        sort -rn | head -1 | cut -d' ' -f2-)
    if [[ -z $FOUND ]]; then
        echo "No *openvino*.zip found in $EXPORT" >&2
        echo "Export one from Physical AI Studio (OpenVINO backend), or run without arguments for a dry run." >&2
        exit 1
    fi
    echo "Using newest export: $FOUND"
    EXPORT=$FOUND
fi

if [[ -z $EXPORT ]]; then
    echo "No export given: starting in dry-run mode (no robot, camera or model)."
    exec uv run --with-editable ./operator_gui operator-gui --dry-run "$@"
fi

if [[ ! -e $EXPORT ]]; then
    echo "Export not found: $EXPORT" >&2
    exit 1
fi

echo "Starting with export: $EXPORT"
exec uv run --extra robot --with-editable ./operator_gui operator-gui --export "$EXPORT" "$@"

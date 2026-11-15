#!/usr/bin/env bash
# Run the full sorting cycle on roberta. Moves the robot: supervise it.
set -euo pipefail

exec bash "$(dirname "$0")/_run-robot-demo.sh" sorting "$@"

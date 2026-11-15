#!/usr/bin/env bash
# Run the present-and-inspect demo on roberta. Moves the robot: supervise it.
set -euo pipefail

exec bash "$(dirname "$0")/_run-robot-demo.sh" inspection "$@"

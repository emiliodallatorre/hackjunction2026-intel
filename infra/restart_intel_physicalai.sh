#!/usr/bin/env bash
set -euo pipefail

infra_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$infra_dir/run_intel_physicalai.sh" --force-recreate

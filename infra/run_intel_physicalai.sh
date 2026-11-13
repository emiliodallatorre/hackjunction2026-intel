#!/usr/bin/env bash
set -euo pipefail

infra_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$infra_dir/.." && pwd)"

exec docker compose --project-directory "$repo_dir" \
    -f "$infra_dir/docker-compose.yml" up -d

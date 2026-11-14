---
name: studio-inspect-resources
description: List Physical AI Studio projects, datasets (with per-task episode counts), trained models, exports, remote trainers, policies and inference/training devices through the Studio REST API. Use before launching training, choosing a model to run, or downloading an export.
---

# Inspect Studio resources

Studio API: `http://roberta:7860/api` (OpenAPI spec at `/api/openapi.json`). Project "Sandbox": `70f6602e-65d0-4f54-b778-7fb6e155fb62`; `GET /api/projects` lists all projects.

## Datasets with task breakdown

There is no global dataset list endpoint; datasets are nested in the project.

```bash
python3 - <<'EOF'
import collections, json, urllib.request
API = "http://roberta:7860/api"
get = lambda path: json.load(urllib.request.urlopen(API + path, timeout=60))
project = get("/projects/70f6602e-65d0-4f54-b778-7fb6e155fb62")
for d in project["datasets"]:
    episodes = get(f"/dataset/{d['id']}/episodes")
    tasks = collections.Counter(t for e in episodes for t in e.get("tasks") or [])
    print(f"{d['id']}  {d['name']}  ({len(episodes)} episodes)")
    for task, n in tasks.most_common():
        print(f"    {n:4d}  {task}")
EOF
```

- `GET /api/dataset/{id}` includes `default_task` and `environment_id`
- Before training, check that task labels match the instructions in `src/hackjunction2026_intel/sorting.py`; differently worded duplicates (for example "dish" vs "plate") split the same behavior across instructions

## Models

```bash
curl -s http://roberta:7860/api/projects/70f6602e-65d0-4f54-b778-7fb6e155fb62/models | python3 -c "$(cat <<'EOF'
import json, sys
for m in json.load(sys.stdin):
    print(m["id"], m["name"], m["policy"], m["available_backends"], "snapflow" if m["snapflow_enabled"] else "", m["created_at"])
EOF
)"
```

- `GET /api/models/{id}` returns details and `GET /api/models/{id}/tasks` the tasks the model knows
- `GET /api/models/{id}/exports/openvino/download` downloads the OpenVINO export and `GET /api/models/{id}/download` the model
- The runtime export zip used by `sorting.py --export` comes from the UI (Models → model row → OpenVINO card → Runtime export)

## Trainers, policies and devices

- `GET /api/remote-trainers` lists trainers (`id`, `name`, `connection_mode`, `url`)
- `GET /api/remote-trainers/{id}/health` checks reachability
- `GET /api/policies/available` lists policy IDs with minimum VRAM; `GET /api/policies/backends` lists inference backends per policy
- `GET /api/system/devices/inference` lists local inference devices (CPU and the Arc B390 iGPU, for both `torch` and `openvino`); `GET /api/system/devices/training` lists local training devices
- `GET /api/projects/{project_id}/robots`, `/cameras` and `/environments` show the configured hardware; `GET /api/projects/{project_id}/environments/{id}/runtime-config` returns an environment's runtime configuration

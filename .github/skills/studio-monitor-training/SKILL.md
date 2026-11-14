---
name: studio-monitor-training
description: Inspect, monitor, detect stalls in, cancel or delete Physical AI Studio training jobs (including remote Modal runs) through the Studio REST API, Docker logs and the Modal CLI. Use when the user asks for training status, whether a run is stale, or to stop or clean up jobs.
---

# Monitor Studio training jobs

Studio API: `http://roberta:7860/api`. Host: `intel@roberta`, container `physical-ai-studio`.

## List active jobs

```bash
curl -s http://roberta:7860/api/jobs | python3 -c "$(cat <<'EOF'
import json, sys
for j in json.load(sys.stdin):
    if j["status"] in ("pending", "running"):
        p = j.get("payload") or {}
        print(j["id"], j["status"], str(j["progress"]) + "%", j["message"], "|", p.get("model_name"),
              "|", p.get("policy"), "|", p.get("remote_trainer_name"), p.get("modal_gpu") or "", "| started", j["start_time"])
EOF
)"
```

`GET /api/jobs` returns a plain list. Each job has `id`, `type` (`training`, `dataset_import`), `status` (`pending`, `running`, `completed`, `failed`, `canceled`), `progress`, `message`, `payload`, `extra_info`, `issues`, `start_time` and `end_time`. Drop the status filter to see recent failures and their messages.

- `GET /api/jobs/{job_id}` returns one job; its payload includes `remote_job_id`, the Modal function call ID (`fc-…`)
- `GET /api/jobs/ws` is a WebSocket with live updates
- `GET /api/jobs/{job_id}/model_metrics` is a server-sent events stream that never ends, so always use `curl -m <seconds>`

## Typical progress messages

`Uploading dataset` → `Dataset ready` (10%) → `Starting Modal GPU function` → `Training model` (grows with steps) → `Downloading trained model from Modal` → `Training finished` (100%). Exporting and downloading can take a long time for π0.5 because of model size.

## Is it stale or working?

A job is likely stale if progress and message stop changing for much longer than one epoch would take. Check in this order:

1. Compare two `GET /api/jobs/{id}` snapshots a few minutes apart
2. Read Studio's view of the remote run; training step lines come from `services.training_backends.modal`:
   ```bash
   ssh intel@roberta 'docker logs --since 15m physical-ai-studio 2>&1 | grep -E "Training progress|modal|ERROR" | tail -30'
   ```
3. Check Modal itself; the `modal` CLI is inside the Studio image and uses environment `junction`:
   ```bash
   ssh intel@roberta 'docker exec physical-ai-studio /app/application/backend/.venv/bin/modal container list --env junction'
   ssh intel@roberta 'docker exec physical-ai-studio timeout 60 /app/application/backend/.venv/bin/modal app logs physicalai-b200-76aff21fbcdf434e9c1f8fcf1c593a6d --env junction'
   ```
   No container for a pending B200 job usually means it is waiting for GPU capacity or building the image
4. Check trainer reachability: `GET /api/remote-trainers/{remote_trainer_id}/health` should return `"status": "healthy"`

Network drops on the robot host can stall the dataset upload or the result download; if so, cancel and resubmit.

## Cancel and delete

Always confirm with the user first.

```bash
curl -s -X POST http://roberta:7860/api/jobs/<job_id>:interrupt
curl -s -X DELETE http://roberta:7860/api/jobs/<job_id>
```

- Deletion only works once the job is `canceled` or `failed`; a later `GET` returns 404
- If a job ignores the interrupt, its worker is an `mp.Process` child of the Studio server that catches SIGTERM. As a last resort and only with the user's approval, SIGKILL the worker. The container has no `kill` binary, so use `docker exec physical-ai-studio python3 -c "import os, signal; os.kill(<pid>, signal.SIGKILL)"`; Studio then marks the job failed
- Interrupting a Studio job may not stop the Modal function; check `modal container list` and stop it with `modal container stop <container_id>` if needed

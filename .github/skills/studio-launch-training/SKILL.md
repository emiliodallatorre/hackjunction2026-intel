---
name: studio-launch-training
description: Launch a Physical AI Studio training job (π0.5, SmolVLA, ACT…) on a remote Modal trainer such as the B200 through the Studio REST API instead of the UI. Use when the user asks to start, resubmit or clone a training run from the terminal.
---

# Launch a Studio training via the API

Studio runs on `intel@roberta` (Docker container `physical-ai-studio`) and serves its API at `http://roberta:7860/api`. The OpenAPI spec is at `http://roberta:7860/api/openapi.json`. Run `curl` from any machine that can reach roberta, or through `ssh intel@roberta`.

## Known identifiers

| What | Value |
|---|---|
| Project "Sandbox" | `70f6602e-65d0-4f54-b778-7fb6e155fb62` |
| Remote trainer "Modal B200" | `22f86c1e-e234-4068-9819-2d34727adfe7` |
| Remote trainer "Modal" (GPU selectable via `modal_gpu`) | `e02ec617-6454-45aa-b8ef-2ee292622169` |

IDs can change; confirm them with the `studio-inspect-resources` skill before submitting.

## Steps

1. Confirm the dataset with the user (list datasets with the `studio-inspect-resources` skill). Prefer datasets whose task labels match the instructions in `src/hackjunction2026_intel/sorting.py`
2. Show the user the settings and get confirmation before submitting: submission starts paid GPU time immediately and there is no preview
3. Avoid duplicate runs: check `GET /api/jobs` for an active job with the same model name or dataset/policy pair
4. Submit with `POST /api/jobs:train`
5. Report the job ID, then monitor with the `studio-monitor-training` skill only if the user wants it

## Payload template (remote B200)

Prefer copying the payload of a previous successful job and changing only what is needed:

```bash
curl -s http://roberta:7860/api/jobs/<job_id> | python3 -c 'import json,sys; print(json.dumps(json.load(sys.stdin)["payload"], indent=1))'
```

Then drop the server-filled fields `remote_job_id`, `snapshot_id`, `remote_trainer_url`, `remote_trainer_name` and `cancel_requested`. A full payload:

```bash
cat > /tmp/train.json <<'EOF'
{
  "project_id": "70f6602e-65d0-4f54-b778-7fb6e155fb62",
  "dataset_id": "<dataset_id>",
  "policy": "pi05",
  "model_name": "<model name shown in the UI>",
  "max_epochs": 5,
  "max_steps": null,
  "batch_size": 8,
  "auto_scale_batch_size": true,
  "num_workers": "auto",
  "base_model_id": null,
  "val_split": 0.1,
  "device": null,
  "precision": "bf16-mixed",
  "compile_model": true,
  "image_key_reorder_map": {},
  "num_cameras": 0,
  "export_backends": ["openvino"],
  "augment_images": true,
  "lora_enabled": false,
  "lora_rank": 32,
  "lora_alpha": null,
  "lora_dropout": 0.05,
  "lora_use_dora": false,
  "snapflow_enabled": true,
  "snapflow_distill_epochs": 3,
  "training_target": "remote",
  "remote_trainer_id": "22f86c1e-e234-4068-9819-2d34727adfe7",
  "modal_gpu": null
}
EOF
curl -s -X POST http://roberta:7860/api/jobs:train -H 'Content-Type: application/json' -d @/tmp/train.json
rm /tmp/train.json
```

The response contains the new job `id` with `status: "pending"`.

## Field notes

- `policy`: one of `act`, `smolvla`, `pi05`, `molmoact2`, `rldx1`, `xr0`; `GET /api/policies/available` lists them with minimum VRAM
- `batch_size`: integer 1–256; there is no `"auto"` value. With `auto_scale_batch_size: true` the trainer runs Lightning's power-scaling batch-size finder before training, starting from `batch_size`. This also applies to Modal runs because Studio sends the same training spec as a local run
- `modal_gpu`: must be `null` for the B200 trainer because Studio's GPU enum does not accept `B200`; the trainer itself pins the GPU. On the generic "Modal" trainer it selects the GPU (for example `H100`, `A100-80GB`)
- `num_cameras: 0` and `image_key_reorder_map: {}` use every camera in the dataset. The previous SmolVLA run used `num_cameras: 3` and `{"realsense rgb": 0, "gripper cam": 1}`
- Previous runs: π0.5 used batch size 8 and SmolVLA 32; both used 5 epochs, bf16-mixed, compile, OpenVINO export, augmentation and 3 SnapFlow distillation epochs
- π0.5 needs Hugging Face access to the gated `google/paligemma-3b-pt-224`, using the `HF_TOKEN` from `infra/.env`. "Gated repo" failures mean the token is missing or lacks access

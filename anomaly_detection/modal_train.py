"""Run `brick-ad train` on a Modal GPU instead of the local CPU.

Prepare locally (it needs the ROI and the raw frames), train remotely, evaluate locally:

    uv run brick-ad prepare [--shuffle]
    uv run --with modal modal run modal_train.py                 # T4, then evaluates locally
    uv run --with modal modal run modal_train.py --gpu L4 --no-evaluate

The prepared dataset (data/dataset) is uploaded, the unchanged `brick_ad.train.train()` runs in a
CUDA container, and the exported OpenVINO model comes back into paths.model (models/patchcore),
exactly where `brick-ad train` would put it. Run `uv run --with modal modal setup` once to log in.
"""

import io
import shutil
import tarfile
import tempfile
from pathlib import Path

import modal

app = modal.App("brick-anomaly-train")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("libglib2.0-0")  # needed by opencv
    # Default PyPI torch wheels are CUDA builds (the local project pins the CPU index instead).
    # Keep these versions in step with ../uv.lock so training and local inference match.
    .pip_install("anomalib[openvino]==2.7.0", "torch==2.15.0", "torchvision==0.29.2", "pyyaml>=6.0")
    .add_local_python_source("brick_ad", "hackjunction2026_intel")
)


def _tar(folder: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(folder, arcname=".")
    return buf.getvalue()


def _untar(data: bytes, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")


@app.function(image=image, gpu="T4", timeout=30 * 60)
def train_remote(config_yaml: str, dataset_tar: bytes) -> bytes:
    """Train on the uploaded dataset with the given config; return the model folder as a .tar.gz."""
    import yaml

    from brick_ad.config import load_config
    from brick_ad.train import train

    work = Path(tempfile.mkdtemp())
    values = yaml.safe_load(config_yaml) or {}
    # Point all paths inside the container's work dir and train on the GPU.
    values["paths"] = {"raw": "data/raw", "dataset": "data/dataset", "runs": "runs", "model": "models/patchcore"}
    values.setdefault("train", {})["accelerator"] = "gpu"
    (work / "config.yaml").write_text(yaml.safe_dump(values))
    _untar(dataset_tar, work / "data" / "dataset")

    model_dir = train(load_config(work / "config.yaml"))
    return _tar(model_dir)


@app.local_entrypoint()
def main(config: str = "", gpu: str = "T4", evaluate: bool = True):
    from brick_ad.config import load_config

    cfg = load_config(config or None)
    dataset = cfg.paths.dataset
    if not (dataset / "meta.json").exists():
        raise SystemExit(f"No prepared dataset in {dataset}. Run `uv run brick-ad prepare` first.")

    print(f"Uploading {dataset} and training on a Modal {gpu}...")
    model_tar = train_remote.with_options(gpu=gpu).remote(cfg.file.read_text(), _tar(dataset))

    # Unpack next to the target first so a failed download never leaves a half-written model.
    staging = cfg.paths.model.with_name(cfg.paths.model.name + ".incoming")
    shutil.rmtree(staging, ignore_errors=True)
    _untar(model_tar, staging)
    shutil.rmtree(cfg.paths.model, ignore_errors=True)
    staging.rename(cfg.paths.model)
    print(f"Model downloaded to {cfg.paths.model}")

    if evaluate:
        from brick_ad.evaluate import evaluate as run_evaluate

        run_evaluate(cfg.paths.model, dataset / "test", device=cfg.inference.device,
                     uncertain_margin=cfg.inference.uncertain_margin)

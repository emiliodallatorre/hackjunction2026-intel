"""Train PatchCore on the prepared dataset and export it to OpenVINO."""

import json
import shutil
from datetime import datetime
from pathlib import Path

from .config import Config


def train(cfg: Config) -> Path:
    # Heavy imports kept local so the camera tools and the inspector start fast.
    from anomalib.data import Folder
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    root = cfg.paths.dataset
    meta_file = root / "meta.json"
    if not meta_file.exists():
        raise SystemExit(f"No prepared dataset in {root}. Run `brick-ad prepare` first.")
    ds_meta = json.loads(meta_file.read_text())
    size = ds_meta["image_size"]
    if size != cfg.model.image_size:
        raise SystemExit(
            f"Dataset was prepared at {size}px but config says {cfg.model.image_size}px. "
            "Rerun `brick-ad prepare`."
        )
    if tuple(ds_meta["roi"] or ()) != tuple(cfg.roi or ()):
        print(f"Warning: dataset ROI {ds_meta['roi']} differs from config ROI {cfg.roi}; using the dataset's.")
    counts = ds_meta["counts"]
    if counts["test/good"] + counts["test/anomaly"] == 0:
        raise SystemExit("The test split is empty: set dataset.test_split > 0 or collect anomaly images.")

    datamodule = Folder(
        name="duplo",
        root=root,
        normal_dir="train/good",
        normal_test_dir="test/good" if counts["test/good"] else None,
        abnormal_dir="test/anomaly" if counts["test/anomaly"] else None,
        train_batch_size=cfg.train.batch_size,
        eval_batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        # Too little data for a separate validation set: the threshold is fit on the test split.
        val_split_mode="same_as_test",
        seed=0,
    )
    model = Patchcore(
        backbone=cfg.model.backbone,
        layers=cfg.model.layers,
        coreset_sampling_ratio=cfg.model.coreset_sampling_ratio,
        num_neighbors=cfg.model.num_neighbors,
        pre_processor=Patchcore.configure_pre_processor(image_size=(size, size)),
    )
    engine = Engine(default_root_dir=cfg.paths.runs, accelerator=cfg.train.accelerator, devices=1)
    engine.fit(model=model, datamodule=datamodule)
    test_metrics = engine.test(model=model, datamodule=datamodule)[0]

    # Export pre-processing, PatchCore and score normalisation as one OpenVINO graph with a
    # fully static [1, 3, size, size] input: anomalib 2.7's dynamic-batch export of PatchCore
    # bakes the example batch size into the graph, and the NPU needs static shapes anyway.
    export_root = cfg.paths.runs / "export"
    xml = Path(
        engine.export(
            model,
            ExportType.OPENVINO,
            export_root=export_root,
            input_size=(size, size),
            onnx_kwargs={"dynamic_axes": None, "opset_version": 18},
        )
    )
    model_dir = cfg.paths.model
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    shutil.copy(xml, model_dir / "model.xml")
    shutil.copy(xml.with_suffix(".bin"), model_dir / "model.bin")

    metadata = {
        "model": "Patchcore",
        "backbone": cfg.model.backbone,
        "layers": list(cfg.model.layers),
        "coreset_sampling_ratio": cfg.model.coreset_sampling_ratio,
        "image_size": size,
        "roi": ds_meta["roi"],
        "threshold": 0.5,  # the exported scores are normalised so the learned threshold is 0.5
        "dataset_counts": counts,
        "anomalib_test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nExported OpenVINO model to {model_dir}")
    fit_empty(cfg, model_dir)
    return model_dir


def fit_empty(cfg: Config, model_dir: Path | None = None) -> None:
    """Calibrate empty-ROI detection from train/empty vs train/good. Fast, no retraining."""
    from .imaging import list_images, read_image
    from .presence import PRESENCE_FILE, REFERENCE_FILE, fit_presence, save_presence

    model_dir = model_dir or cfg.paths.model
    if not (model_dir / "metadata.json").exists():
        raise SystemExit(f"No trained model in {model_dir}. Run `brick-ad train` first.")
    size = json.loads((model_dir / "metadata.json").read_text())["image_size"]
    empty = [read_image(p) for p in list_images(cfg.paths.dataset / "train" / "empty")]
    bricks = [read_image(p) for p in list_images(cfg.paths.dataset / "train" / "good")]
    if not empty:
        for name in (REFERENCE_FILE, PRESENCE_FILE):  # don't keep a stale reference
            (model_dir / name).unlink(missing_ok=True)
        print("Empty-ROI detection off: no empty images (collect some with `e`, then prepare + fit-empty).")
        return
    if any(p.shape[:2] != (size, size) for p in empty + bricks):
        raise SystemExit("Dataset crops do not match the model size. Rerun `brick-ad prepare`.")

    detector, stats = fit_presence(empty, bricks)
    save_presence(detector, model_dir, stats)
    print(f"\nEmpty-ROI detection: {stats['n_empty']} empty vs {stats['n_bricks']} brick crops")
    print(f"  changed pixels: empty max {stats['empty_max']:.1%}, brick min {stats['brick_min']:.1%}"
          f" -> threshold {stats['threshold']:.1%}")
    if stats["brick_min"] <= stats["empty_max"]:
        print("  Warning: empty and brick crops overlap. Check the ROI and lighting, or collect more empty frames.")

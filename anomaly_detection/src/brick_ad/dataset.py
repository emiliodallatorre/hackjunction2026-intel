"""Turn raw camera frames into the cropped PatchCore dataset.

    data/raw/{good,anomaly,empty}/*.png                 (full frames, from `collect`)
        -> data/dataset/train/good       good crops PatchCore learns from
           data/dataset/test/good        held-out good crops (most recent captures, or random with --shuffle)
           data/dataset/test/anomaly     all anomaly crops (PatchCore never trains on them)
           data/dataset/train/empty      empty-ROI crops: background reference for presence detection
           data/dataset/test/empty       held-out empty crops (most recent captures, or random with --shuffle)
           data/dataset/meta.json        ROI and image size used, read by `train`
"""

import json
import random
import shutil

import cv2

from .config import ANOMALY, EMPTY, GOOD, Config
from .imaging import list_images, preprocess, read_image


def prepare(cfg: Config, shuffle: bool = False, seed: int = 0) -> dict[str, int]:
    raw, out = cfg.paths.raw, cfg.paths.dataset
    good, anomalies = list_images(raw / GOOD), list_images(raw / ANOMALY)
    if not good:
        raise SystemExit(f"No good images in {raw / GOOD}. Run `brick-ad collect` first.")
    if cfg.roi is None:
        print("Warning: no ROI set, using whole frames. Run `brick-ad roi` to crop to the brick.")

    empty = list_images(raw / EMPTY)

    rng = random.Random(seed)

    def holdout(files: list) -> tuple[list, list]:
        """Files are named by capture time, so by default this holds out the latest captures.
        With ``shuffle`` the held-out files are a random (seeded) sample across all sessions instead."""
        n_test = round(len(files) * cfg.dataset.test_split)
        if cfg.dataset.test_split > 0 and len(files) > 1:
            n_test = max(n_test, 1)
        if shuffle:
            files = files.copy()
            rng.shuffle(files)
        return files[: len(files) - n_test], files[len(files) - n_test :]

    train_good, test_good = holdout(good)
    train_empty, test_empty = holdout(empty)
    splits = {
        ("train", GOOD): train_good,
        ("test", GOOD): test_good,
        ("test", ANOMALY): anomalies,
        ("train", EMPTY): train_empty,  # empty-ROI reference for presence detection
        ("test", EMPTY): test_empty,
    }

    if out.exists():
        shutil.rmtree(out)
    counts = {}
    for (split, label), files in splits.items():
        folder = out / split / label
        folder.mkdir(parents=True, exist_ok=True)
        for f in files:
            patch = preprocess(read_image(f), cfg.roi, cfg.model.image_size)
            cv2.imwrite(str(folder / f"{f.stem}.png"), patch)
        counts[f"{split}/{label}"] = len(files)

    meta = {"roi": cfg.roi, "image_size": cfg.model.image_size, "counts": counts,
            "split": {"shuffle": shuffle, "seed": seed if shuffle else None}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Dataset written to {out}  (roi={cfg.roi}, size={cfg.model.image_size})")
    for name, n in counts.items():
        print(f"  {name:<14} {n:5d}")
    if not anomalies:
        print("Warning: no anomaly images; the threshold can only be calibrated on good images.")
    if not train_empty:
        print("Note: no empty images, so empty-ROI detection is off. Collect some with `e` in `brick-ad collect`.")
    return counts

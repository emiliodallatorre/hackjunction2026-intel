#!/usr/bin/env python3
"""Merge two or more LeRobot v3 (Physical AI Studio) datasets into one.

    scripts/merge-datasets.py OUT IN1 IN2 [IN3 ...]

Each input is a dataset folder or the .zip Studio downloads. The output is a folder with the
same layout, so Studio and `physicalai train` read it like any other dataset.

What a merge has to rewrite, beyond copying files:

* ``episode_index`` is per-dataset and restarts at 0, so later datasets are offset.
* ``index`` is a global frame counter across the whole dataset, so it is renumbered end to end.
* ``task_index`` points into ``meta/tasks.parquet``. Two datasets can use the same index for
  different task strings, so tasks are matched by *string*: shared ones keep one slot, new ones
  are appended.
* ``data/`` and ``videos/`` files are renumbered, and every episode's recorded chunk/file index
  is updated to where its data actually landed.
* ``dataset_from_index`` / ``dataset_to_index`` are absolute frame ranges, so they shift with the
  renumbered ``index``.
* ``meta/stats.json`` holds the dataset-wide normalization values. ``mean``, ``std``, ``min``,
  ``max`` and ``count`` combine exactly from the inputs (pooled variance for ``std``), and the
  rewritten ``index`` / ``episode_index`` columns get stats recomputed from the merged frames.
  Quantiles cannot be pooled from aggregates, so for the numeric columns they are recomputed from
  the merged data and for image columns they are interpolated by ``count`` (approximate: the
  exact values would need a pass over every video frame).

Video files are copied, never re-encoded: the per-episode ``from_timestamp`` / ``to_timestamp``
offsets stay valid only if the .mp4 they point into is byte-identical.
"""

import argparse
import json
import logging
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

logger = logging.getLogger("merge-datasets")

# info.json keys that must agree across inputs: a merge of mismatched ones is not a dataset.
MUST_MATCH = ("codebase_version", "fps", "robot_type")
# Summed across inputs.
TOTALS = ("total_episodes", "total_frames")


@contextmanager
def open_dataset(path: Path) -> Iterator[Path]:
    """Yield a dataset root, extracting it first if ``path`` is a zip."""
    if path.is_dir():
        if not (path / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Not a LeRobot dataset (no meta/info.json): {path}")
        yield path
        return
    if not (path.is_file() and zipfile.is_zipfile(path)):
        raise FileNotFoundError(f"Dataset not found (expected a folder or .zip): {path}")
    with tempfile.TemporaryDirectory(prefix="merge-dataset-") as tmp:
        logger.info("Extracting %s", path.name)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(tmp)
        root = Path(tmp)
        if not (root / "meta" / "info.json").is_file():
            entries = [e for e in root.iterdir() if e.is_dir()]
            if len(entries) == 1 and (entries[0] / "meta" / "info.json").is_file():
                root = entries[0]
            else:
                raise FileNotFoundError(f"No meta/info.json inside {path}")
        yield root


def read_tasks(root: Path) -> list[str]:
    """Task strings, ordered by their task_index in this dataset."""
    import pandas as pd

    table = pd.read_parquet(root / "meta" / "tasks.parquet")
    # Studio writes `task` as the index and task_index as the only column.
    if "task" in table.columns:
        table = table.set_index("task")
    return list(table.sort_values("task_index").index)


def read_episodes(root: Path) -> Any:
    import pandas as pd

    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata in {root}/meta/episodes")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).sort_values("episode_index")


def pool_stats(parts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Combine per-dataset stats for one feature.

    ``mean``/``std`` use the exact pooled formulas weighted by ``count``; ``min``/``max`` take the
    extremes. Quantiles have no exact pooled form, so they are averaged by weight, which is an
    approximation (callers recompute them from data where that is affordable).
    """
    import numpy as np

    counts = np.array([np.asarray(part["count"], dtype=np.float64).ravel()[0] for part in parts])
    total = counts.sum()
    weights = counts / total
    out: dict[str, Any] = {"count": [int(total)]}

    means = [np.asarray(part["mean"], dtype=np.float64) for part in parts]
    shape = means[0].shape
    mean = sum(w * m for w, m in zip(weights, means, strict=True))
    out["mean"] = mean.tolist()

    if all("std" in part for part in parts):
        # Pooled variance: the mean of each part's variance plus the spread of their means.
        variances = [np.asarray(part["std"], dtype=np.float64) ** 2 for part in parts]
        pooled = sum(w * (v + (m - mean) ** 2) for w, v, m in zip(weights, variances, means, strict=True))
        out["std"] = np.sqrt(pooled).tolist()

    for key, reduce in (("min", np.minimum), ("max", np.maximum)):
        if all(key in part for part in parts):
            values = [np.asarray(part[key], dtype=np.float64) for part in parts]
            combined = values[0]
            for value in values[1:]:
                combined = reduce(combined, value)
            out[key] = combined.tolist()

    for key in ("q01", "q10", "q50", "q90", "q99"):
        if all(key in part for part in parts):
            quantiles = [np.asarray(part[key], dtype=np.float64) for part in parts]
            out[key] = sum(w * q for w, q in zip(weights, quantiles, strict=True)).tolist()
    assert np.asarray(out["mean"]).shape == shape
    return out


def exact_column_stats(values: Any, template: dict[str, Any]) -> dict[str, Any]:
    """Stats computed directly from merged values, for the columns a merge rewrites."""
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    out = {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [int(array.shape[0])],
    }
    for key, q in (("q01", 1), ("q10", 10), ("q50", 50), ("q90", 90), ("q99", 99)):
        if key in template:
            out[key] = np.percentile(array, q, axis=0).tolist()
    return out


def video_keys(info: dict[str, Any]) -> list[str]:
    return [k for k, v in info["features"].items() if v.get("dtype") == "video"]


def merge(output: Path, inputs: Sequence[Path], *, chunk_size: int | None = None) -> None:
    import numpy as np
    import pandas as pd

    with ExitStack() as stack:
        roots = [stack.enter_context(open_dataset(p)) for p in inputs]
        infos = [json.loads((r / "meta" / "info.json").read_text()) for r in roots]

        for key in MUST_MATCH:
            values = {json.dumps(i.get(key)) for i in infos}
            if len(values) > 1:
                raise ValueError(f"Inputs disagree on {key!r}: {sorted(values)}; refusing to merge")
        base_features = json.dumps(infos[0]["features"], sort_keys=True)
        for path, info in zip(inputs, infos, strict=True):
            if json.dumps(info["features"], sort_keys=True) != base_features:
                raise ValueError(f"{path} has different features than the first dataset; refusing to merge")

        keys = video_keys(infos[0])
        chunk = chunk_size or infos[0].get("chunks_size", 1000)

        # One global task table, matched by string so shared tasks keep a single index.
        tasks: list[str] = []
        task_maps: list[dict[int, int]] = []
        for root in roots:
            mapping = {}
            for local_index, task in enumerate(read_tasks(root)):
                if task not in tasks:
                    tasks.append(task)
                mapping[local_index] = tasks.index(task)
            task_maps.append(mapping)

        (output / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for key in keys:
            (output / "videos" / key / "chunk-000").mkdir(parents=True, exist_ok=True)
        (output / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)

        episode_offset = 0
        frame_offset = 0
        data_slot = 0
        video_slots = {key: 0 for key in keys}
        merged_episodes = []

        for root, info, task_map, source in zip(roots, infos, task_maps, inputs, strict=True):
            episodes = read_episodes(root)
            logger.info("%s: %d episodes, %d frames", Path(source).name,
                        len(episodes), info.get("total_frames", 0))

            # --- data/: one output parquet per input parquet, indices rewritten.
            data_map: dict[tuple[int, int], tuple[int, int]] = {}
            seen = episodes[["data/chunk_index", "data/file_index"]].drop_duplicates()
            for _, row in seen.sort_values(["data/chunk_index", "data/file_index"]).iterrows():
                src = root / infos[0]["data_path"].format(
                    chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]))
                frame = pd.read_parquet(src)
                frame["episode_index"] = frame["episode_index"] + episode_offset
                frame["index"] = frame["index"] + frame_offset
                frame["task_index"] = frame["task_index"].map(task_map).astype(frame["task_index"].dtype)
                target = (data_slot // chunk, data_slot % chunk)
                out = output / infos[0]["data_path"].format(chunk_index=target[0], file_index=target[1])
                out.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(out, index=False)
                data_map[(int(row["data/chunk_index"]), int(row["data/file_index"]))] = target
                data_slot += 1

            # --- videos/: copied verbatim, only renamed.
            video_maps: dict[str, dict[tuple[int, int], tuple[int, int]]] = {}
            for key in keys:
                mapping = {}
                pairs = episodes[[f"videos/{key}/chunk_index", f"videos/{key}/file_index"]].drop_duplicates()
                for _, row in pairs.sort_values(list(pairs.columns)).iterrows():
                    source_pair = (int(row.iloc[0]), int(row.iloc[1]))
                    src = root / infos[0]["video_path"].format(
                        video_key=key, chunk_index=source_pair[0], file_index=source_pair[1])
                    slot = video_slots[key]
                    target = (slot // chunk, slot % chunk)
                    dst = output / infos[0]["video_path"].format(
                        video_key=key, chunk_index=target[0], file_index=target[1])
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)  # never re-encode: timestamps index into these bytes
                    mapping[source_pair] = target
                    video_slots[key] = slot + 1
                video_maps[key] = mapping

            # --- meta/episodes/: rewrite every index that moved.
            episodes = episodes.copy()
            episodes["episode_index"] = episodes["episode_index"] + episode_offset
            episodes["dataset_from_index"] = episodes["dataset_from_index"] + frame_offset
            episodes["dataset_to_index"] = episodes["dataset_to_index"] + frame_offset
            episodes["data/chunk_index"], episodes["data/file_index"] = zip(
                *[data_map[(int(c), int(f))] for c, f in
                  zip(episodes["data/chunk_index"], episodes["data/file_index"], strict=True)], strict=True)
            for key in keys:
                pairs = [video_maps[key][(int(c), int(f))] for c, f in
                         zip(episodes[f"videos/{key}/chunk_index"],
                             episodes[f"videos/{key}/file_index"], strict=True)]
                episodes[f"videos/{key}/chunk_index"], episodes[f"videos/{key}/file_index"] = zip(*pairs, strict=True)
            merged_episodes.append(episodes)

            episode_offset += len(episodes)
            frame_offset += int(info["total_frames"])

        # --- meta/: one episodes table, one task table, one info.json.
        all_episodes = pd.concat(merged_episodes, ignore_index=True)
        all_episodes["meta/episodes/chunk_index"] = 0
        all_episodes["meta/episodes/file_index"] = 0
        all_episodes.to_parquet(output / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

        task_table = pd.DataFrame({"task": tasks, "task_index": range(len(tasks))}).set_index("task")
        task_table.to_parquet(output / "meta" / "tasks.parquet")

        info = dict(infos[0])
        for key in TOTALS:
            info[key] = sum(int(i[key]) for i in infos)
        info["total_tasks"] = len(tasks)
        info["splits"] = {"train": f"0:{episode_offset}"}
        if "total_videos" in info:
            info["total_videos"] = sum(int(i.get("total_videos", 0)) for i in infos)
        (output / "meta" / "info.json").write_text(json.dumps(info, indent=4))

        # --- meta/stats.json: the dataset-wide normalization values the policy needs.
        stats_files = [r / "meta" / "stats.json" for r in roots]
        if all(f.is_file() for f in stats_files):
            parts = [json.loads(f.read_text()) for f in stats_files]
            shared = set(parts[0])
            for part in parts[1:]:
                shared &= set(part)
            merged_stats = {key: pool_stats([part[key] for part in parts]) for key in sorted(shared)}
            # Columns the merge rewrote, plus the numeric columns we can afford to read: exact values.
            all_data = pd.concat(
                [pd.read_parquet(f) for f in sorted((output / "data").rglob("*.parquet"))], ignore_index=True)
            for column in ("index", "episode_index", "frame_index", "task_index", "timestamp",
                           "action", "observation.state"):
                if column in merged_stats and column in all_data.columns:
                    values = all_data[column]
                    stacked = np.stack(values.to_numpy()) if values.dtype == object else values.to_numpy()
                    merged_stats[column] = exact_column_stats(stacked, merged_stats[column])
            (output / "meta" / "stats.json").write_text(json.dumps(merged_stats, indent=4))
            approximate = sorted(k for k in merged_stats if k in keys)
            logger.info("meta/stats.json merged (exact for the numeric columns)")
            if approximate:
                logger.info(
                    "Image quantiles in stats.json are interpolated by frame count for %s; mean, std, "
                    "min and max are exact. Recompute with LeRobot if your policy needs exact image "
                    "quantiles (normalization uses mean/std by default).", ", ".join(approximate))
        else:
            logger.warning("Not all inputs have meta/stats.json; the merged dataset has none")

        logger.info("Merged %d episodes, %d frames, %d tasks -> %s",
                    info["total_episodes"], info["total_frames"], info["total_tasks"], output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="Folder to write the merged dataset to")
    parser.add_argument("inputs", type=Path, nargs="+", help="Dataset folders or .zip files (two or more)")
    parser.add_argument("--force", action="store_true", help="Overwrite the output folder if it exists")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(args.inputs) < 2:
        parser.error("give at least two datasets to merge")
    if args.output.exists():
        if not args.force:
            parser.error(f"{args.output} already exists; pass --force to overwrite")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    try:
        merge(args.output, args.inputs)
    except (ValueError, FileNotFoundError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

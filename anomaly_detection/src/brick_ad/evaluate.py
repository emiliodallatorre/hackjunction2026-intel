"""Score a labelled folder (good/, anomaly/ and optionally empty/) with the exported model and print stats."""

import time
from pathlib import Path

import cv2
import numpy as np

from .config import ANOMALY, EMPTY, GOOD, LABELS
from .imaging import list_images, overlay_heatmap, put_text
from hackjunction2026_intel.anomaly_detection.inspector import BrickInspector


def _confusion(labels: np.ndarray, preds: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum(preds & labels))
    fp = int(np.sum(preds & ~labels))
    tn = int(np.sum(~preds & ~labels))
    fn = int(np.sum(~preds & labels))
    return tp, fp, tn, fn


def _ratio(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _f1(tp: int, fp: int, fn: int) -> float:
    return _ratio(2 * tp, 2 * tp + fp + fn)


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Probability that a random anomaly scores above a random good image (ties count half)."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(np.mean((diff > 0) + 0.5 * (diff == 0)))


def evaluate(
    model_dir: Path,
    data_dir: Path,
    device: str = "CPU",
    threshold: float | None = None,
    uncertain_margin: float = 0.1,
    crop: bool = False,
    save_dir: Path | None = None,
) -> dict:
    inspector = BrickInspector(model_dir, device=device, threshold=threshold, uncertain_margin=uncertain_margin)
    samples = [(p, label) for label in LABELS for p in list_images(data_dir / label)]
    if not any(label != EMPTY for _, label in samples):
        raise SystemExit(f"No images in {data_dir}/{GOOD} or {data_dir}/{ANOMALY}")
    if inspector.presence is None and any(label == EMPTY for _, label in samples):
        print("Note: this model has no empty-ROI reference; skipping empty images (run `brick-ad fit-empty`).")
        samples = [s for s in samples if s[1] != EMPTY]

    inspector.predict(samples[0][0], crop=crop)  # warm-up, excluded from latency
    rows, times = [], []
    for path, label in samples:
        t0 = time.perf_counter()
        # Always run the model so every frame gets an anomaly score; the empty decision is applied below.
        pred = inspector.predict(path, crop=crop, check_presence=False)
        times.append((time.perf_counter() - t0) * 1000)
        empty = inspector.presence is not None and pred.presence < inspector.presence.threshold
        status = EMPTY if empty else pred.status
        rows.append((path, label, status, pred.score, pred.uncertain and not empty))
        if save_dir is not None:
            view = overlay_heatmap(pred.patch, pred.anomaly_map)
            text = "EMPTY" if empty else f"{pred.score:.2f}{' ?' if pred.uncertain else ''}"
            put_text(view, text, 0, (0, 0, 255) if status == ANOMALY else (0, 255, 0))
            out = save_dir / label
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{pred.score:.3f}_{path.stem}.png"), view)

    thr = inspector.threshold
    print(f"\nEvaluated {len(rows)} images from {data_dir}  (model {model_dir}, {device})")

    # 1. Final decision as the pipeline sees it: empty / good / anomaly.
    present = [lbl for lbl in LABELS if any(r[1] == lbl for r in rows)]
    statuses = [lbl for lbl in LABELS if lbl != EMPTY or inspector.presence is not None]
    print(f"\nFinal status (threshold {thr:.3f})")
    print(f"  {'actual / predicted':20}" + "".join(f"{s:>10}" for s in statuses))
    for actual in present:
        counts = [sum(1 for r in rows if r[1] == actual and r[2] == s) for s in statuses]
        print(f"  {actual:20}" + "".join(f"{c:10d}" for c in counts))
    accuracy = _ratio(sum(r[1] == r[2] for r in rows), len(rows))
    print(f"  overall accuracy {accuracy:.1%}")

    # 2. Anomaly model on frames with a brick (threshold-free view of the scores).
    bricks = [r for r in rows if r[1] != EMPTY]
    labels = np.array([r[1] == ANOMALY for r in bricks])
    scores = np.array([r[3] for r in bricks])
    preds = scores >= thr
    tp, fp, tn, fn = _confusion(labels, preds)
    best_thr, best_f1 = thr, _f1(tp, fp, fn)
    for t in np.unique(scores):  # optimistic: chosen on the same images
        t_tp, t_fp, _, t_fn = _confusion(labels, scores >= t)
        if (f := _f1(t_tp, t_fp, t_fn)) > best_f1:
            best_thr, best_f1 = float(t), f

    print(f"\nAnomaly scores on brick frames   {'':4} {'n':>5}   min / mean / max")
    for name, mask in ((GOOD, ~labels), (ANOMALY, labels)):
        if mask.any():
            s = scores[mask]
            print(f"  {name:36} {len(s):5d}   {s.min():.3f} / {s.mean():.3f} / {s.max():.3f}")
    if labels.any() and (~labels).any():
        margin = scores[labels].min() - scores[~labels].max()
        print(f"  margin (min anomaly - max good): {margin:+.3f}  {'(perfectly separable)' if margin > 0 else ''}")
    stats = {
        "accuracy": accuracy,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": _ratio(tp, tp + fp),
        "recall": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "f1": _f1(tp, fp, fn),
        "auroc": _auroc(labels, scores),
        "best_f1_threshold": best_thr,
        "latency_ms_mean": float(np.mean(times)),
        "latency_ms_p95": float(np.percentile(times, 95)),
    }
    print(
        f"  precision {stats['precision']:.1%}  recall {stats['recall']:.1%}"
        f"  specificity {stats['specificity']:.1%}  F1 {stats['f1']:.3f}  AUROC {stats['auroc']:.3f}"
    )
    print(f"  best-F1 threshold on this data: {best_thr:.4f} (F1 {best_f1:.3f})")

    # 3. How often the pipeline would ask for another look.
    lo, hi = thr - inspector.uncertain_margin, thr + inspector.uncertain_margin
    print(f"\nUncertain (score {lo:.2f}-{hi:.2f}, would be re-inspected)")
    for name in (GOOD, ANOMALY):
        group = [r for r in bricks if r[1] == name]
        if group:
            n = sum(r[4] for r in group)
            print(f"  {name:10} {n:4d} / {len(group):<4d} ({n / len(group):.0%})")
    wrong_sure = sum(1 for r in bricks if r[1] != r[2] and r[2] != EMPTY and not r[4])
    print(f"  confident mistakes (wrong and outside the band): {wrong_sure}")
    print(f"\nLatency (anomaly model) {stats['latency_ms_mean']:.1f} ms mean, {stats['latency_ms_p95']:.1f} ms p95")

    wrong = [r for r in rows if r[1] != r[2]]
    if wrong:
        print("\nMisclassified (actual -> predicted):")
        for path, label, status, score, uncertain in sorted(wrong, key=lambda r: r[3]):
            print(f"  {label:>7} -> {status:7} {score:.3f}{' uncertain' if uncertain else ''}  {path}")
    if save_dir is not None:
        print(f"\nHeatmaps saved to {save_dir} (file names start with the score)")
    return stats

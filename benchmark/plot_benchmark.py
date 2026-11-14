#!/usr/bin/env python3
"""Turn benchmark output into graphs.

    python3 plot_benchmark.py results/result_20261114_140117.json
    python3 plot_benchmark.py results/result_*.json --outdir plots

Takes the JSON written by bench_core.py and reads its CSV sidecars
(_traces.csv, _latencies.csv, _sustained.csv, _summary.csv) from the same
folder. Writes PNGs next to the JSON unless --outdir is given.

Needs matplotlib only (no pandas, no seaborn).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")                     # headless: write files, never open a window
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is required.  pip install matplotlib", file=sys.stderr)
    sys.exit(1)

# One consistent colour per phase across every figure.
PHASE_COLOR = {
    "idle":         "#8c8c8c",
    "steady_state": "#3b7dd8",
    "throughput":   "#d9693f",
    "end_to_end":   "#4f9d6b",
    "sustained":    "#8e5bb5",
}
PHASE_LABEL = {
    "idle":         "idle baseline",
    "steady_state": "steady-state (batch 1)",
    "throughput":   "peak throughput (async)",
    "end_to_end":   "end-to-end predict()",
    "sustained":    "sustained load",
}

def style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 150,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.autolayout": True,
    })

def read_csv(path):
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))

def fnum(row, key):
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None

def pct(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)

def rapl_keys(rows):
    return sorted({k for r in rows for k in r if k.startswith("rapl_w[")})

def save(fig, outdir, name, made):
    path = outdir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    made.append(path)
    print(f"  {path}")

# ---------------------------------------------------------------- figures
def plot_latency_distribution(lat_rows, outdir, made):
    """Histogram + CDF of per-inference latency: where the tail actually is."""
    groups = defaultdict(list)
    for r in lat_rows:
        ms = fnum(r, "ms")
        if ms is not None:
            groups[(r["device"], r["phase"])].append(ms)
    if not groups:
        return
    n = len(groups)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 7), squeeze=False)
    for col, ((dev, phase), vals) in enumerate(sorted(groups.items())):
        vals_s = sorted(vals)
        color = PHASE_COLOR.get(phase, "#3b7dd8")
        ax = axes[0][col]
        ax.hist(vals, bins=40, color=color, alpha=0.8, edgecolor="white", linewidth=0.5)
        p50, p95, p99 = pct(vals_s, 50), pct(vals_s, 95), pct(vals_s, 99)
        for v, lbl, ls in ((p50, "p50", "-"), (p95, "p95", "--"), (p99, "p99", ":")):
            ax.axvline(v, color="#222", linestyle=ls, linewidth=1.2)
            ax.text(v, ax.get_ylim()[1] * 0.96, f" {lbl} {v:.1f}", fontsize=8,
                    rotation=90, va="top", ha="left")
        ax.set_title(f"{dev} · {PHASE_LABEL.get(phase, phase)}")
        ax.set_xlabel("latency (ms)")
        ax.set_ylabel("count")

        ax2 = axes[1][col]
        ys = [100.0 * (i + 1) / len(vals_s) for i in range(len(vals_s))]
        ax2.plot(vals_s, ys, color=color, linewidth=1.8)
        for q, ls in ((50, "-"), (95, "--"), (99, ":")):
            ax2.axhline(q, color="#bbb", linestyle=ls, linewidth=0.9)
        ax2.set_xlabel("latency (ms)")
        ax2.set_ylabel("percentile (%)")
        ax2.set_ylim(0, 100)
        ax2.set_title("cumulative distribution")
    fig.suptitle("Per-inference latency distribution", y=1.01, fontsize=13, fontweight="bold")
    save(fig, outdir, "latency_distribution.png", made)

def plot_latency_over_time(lat_rows, outdir, made):
    """Latency vs iteration: shows drift and periodic spikes a histogram hides."""
    groups = defaultdict(list)
    for r in lat_rows:
        ms, i = fnum(r, "ms"), fnum(r, "i")
        if ms is not None and i is not None:
            groups[(r["device"], r["phase"])].append((i, ms))
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for (dev, phase), pts in sorted(groups.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, linewidth=0.9, alpha=0.85,
                color=PHASE_COLOR.get(phase, None),
                label=f"{dev} · {PHASE_LABEL.get(phase, phase)}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Latency per iteration (drift and spikes)")
    ax.legend()
    save(fig, outdir, "latency_over_time.png", made)

def plot_power_temp(trace_rows, outdir, made):
    """Power and temperature vs time, one panel per phase."""
    phases = defaultdict(list)
    for r in trace_rows:
        phases[(r["device"], r["phase"])].append(r)
    if not phases:
        return
    rk = rapl_keys(trace_rows)
    order = sorted(phases.items(), key=lambda kv: list(PHASE_LABEL).index(kv[0][1])
                   if kv[0][1] in PHASE_LABEL else 99)
    n = len(order)
    fig, axes = plt.subplots(n, 1, figsize=(11, 3.1 * n), squeeze=False, sharex=False)
    any_power = False
    for i, ((dev, phase), rows) in enumerate(order):
        ax = axes[i][0]
        ts = [fnum(r, "t_s") or 0.0 for r in rows]
        # Power: RAPL domains if present, else the battery counter.
        plotted = False
        for k in rk:
            ys = [fnum(r, k) for r in rows]
            if any(y is not None for y in ys):
                xs = [t for t, y in zip(ts, ys) if y is not None]
                vs = [y for y in ys if y is not None]
                dom = k[len("rapl_w["):-1].split("@")[0]
                ax.plot(xs, vs, linewidth=1.5, label=f"{dom} (W)")
                plotted = any_power = True
        if not plotted:
            ys = [fnum(r, "battery_w") for r in rows]
            if any(y is not None for y in ys):
                xs = [t for t, y in zip(ts, ys) if y is not None]
                vs = [y for y in ys if y is not None]
                ax.plot(xs, vs, linewidth=1.5, color="#d9693f", label="battery (W)")
                plotted = any_power = True
        ax.set_xlabel("time (s)")
        ax.set_title(f"{dev or 'system'} · {PHASE_LABEL.get(phase, phase)}")
        if plotted:
            ax.set_ylabel("power (W)")
        else:
            # No power counter: an empty 0-1 W axis reads as "power was ~0".
            # Hide it so the panel is honestly a temperature-only plot.
            ax.set_yticks([])
            for sp in ("left", "bottom"):
                ax.spines[sp].set_visible(sp == "bottom")

        # Temperature on a twin axis.
        temps = [(t, fnum(r, "temp_c")) for t, r in zip(ts, rows) if fnum(r, "temp_c") is not None]
        if temps:
            axt = ax.twinx()
            axt.plot([t for t, _ in temps], [v for _, v in temps],
                     color="#b5452f", linewidth=1.4, linestyle="--", label="temp (°C)")
            axt.set_ylabel("temperature (°C)", color="#b5452f")
            axt.tick_params(axis="y", colors="#b5452f")
            axt.grid(False)
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = axt.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
        elif plotted:
            ax.legend(loc="upper right", fontsize=8)
        if not plotted and not temps:
            ax.text(0.5, 0.5, "no power or temperature samples",
                    ha="center", va="center", transform=ax.transAxes, color="#999")
    title = "Power and temperature over time"
    if not any_power:
        title += "  (no usable power counter — see notes)"
    fig.suptitle(title, y=1.005, fontsize=13, fontweight="bold")
    save(fig, outdir, "power_temperature.png", made)

def plot_sustained(sus_rows, outdir, made):
    """Throughput and temperature vs time: burst vs sustained (thermal throttling)."""
    devs = defaultdict(list)
    for r in sus_rows:
        devs[r["device"]].append(r)
    if not devs:
        return
    # Three stacked panels, each with ONE y-axis. A twin axis per device would
    # overlay several independent scales on one panel and hide all but the last.
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)
    for dev, rows in sorted(devs.items()):
        rows.sort(key=lambda r: fnum(r, "t_s") or 0)
        ts = [fnum(r, "t_s") for r in rows]
        fps = [fnum(r, "fps") for r in rows]
        p99 = [fnum(r, "p99_ms") for r in rows]
        tc = [fnum(r, "temp_c") for r in rows]
        line, = ax1.plot(ts, fps, marker="o", markersize=3.5, linewidth=1.8, label=dev)
        first = next((v for v in fps if v), None)
        if first:
            # Burst reference: the first bucket, so the decay is readable as a drop.
            ax1.axhline(first, linestyle=":", linewidth=1, color=line.get_color(), alpha=0.6)
            last = next((v for v in reversed(fps) if v), None)
            if last:
                # Negative = throughput fell from the burst level (throttling).
                ax1.annotate(f"{dev}: {(last / first - 1) * 100:+.1f}%",
                             xy=(ts[-1], last), xytext=(4, 0), textcoords="offset points",
                             fontsize=8, va="center", color=line.get_color())
        ax2.plot(ts, p99, marker="s", markersize=3, linewidth=1.5,
                 color=line.get_color(), label=dev)
        pts = [(t, v) for t, v in zip(ts, tc) if v is not None]
        if pts:
            ax3.plot([t for t, _ in pts], [v for _, v in pts], marker="^",
                     markersize=3, linewidth=1.5, linestyle="--",
                     color=line.get_color(), label=dev)
    ax1.set_ylabel("throughput (inf/s)")
    ax1.set_title("Sustained load: throughput decay (thermal throttling)")
    ax1.legend(loc="center right", fontsize=9)
    ax1.set_ylim(bottom=0)
    ax2.set_ylabel("p99 latency (ms)")
    ax2.set_title("Tail latency over the same run")
    ax2.legend(loc="center right", fontsize=9)
    ax3.set_ylabel("temperature (°C)")
    ax3.set_xlabel("time (s)")
    ax3.set_title("Package temperature over the same run")
    if ax3.get_legend_handles_labels()[0]:
        ax3.legend(loc="center right", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "no temperature samples", ha="center", va="center",
                 transform=ax3.transAxes, color="#999")
    save(fig, outdir, "sustained_throughput.png", made)

def plot_cold_start(data, outdir, made):
    """Stacked read/compile/first-infer per cache state, per device."""
    bars, labels = [], []
    for dev, D in (data.get("devices") or {}).items():
        cs = D.get("cold_start")
        if not cs:
            continue
        for state in ("cold_cache", "warm_cache", "no_cache"):
            if state in cs:
                bars.append(cs[state])
                labels.append(f"{dev}\n{state.replace('_', ' ')}")
    if not bars:
        return
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(bars)), 4.8))
    xs = range(len(bars))
    read = [b.get("read_ms", 0) for b in bars]
    comp = [b.get("compile_ms", 0) for b in bars]
    first = [b.get("first_infer_ms", 0) for b in bars]
    ax.bar(xs, read, label="read_model", color="#9bb7e0")
    ax.bar(xs, comp, bottom=read, label="compile_model", color="#3b7dd8")
    ax.bar(xs, first, bottom=[r + c for r, c in zip(read, comp)],
           label="first inference", color="#d9693f")
    for i, b in enumerate(bars):
        tot = b.get("read_ms", 0) + b.get("compile_ms", 0) + b.get("first_infer_ms", 0)
        ax.text(i, tot, f"{tot:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("time (ms)")
    ax.set_title("Cold start: what the first inference actually costs")
    ax.legend()
    save(fig, outdir, "cold_start.png", made)

def plot_device_comparison(data, outdir, made):
    """Side-by-side device comparison; only meaningful with >1 device."""
    rows = data.get("summary_rows") or []
    if len(rows) < 1:
        return
    devs = [r["device"] for r in rows]
    panels = [
        ("latency (ms, lower better)", [("p50_ms", "p50"), ("p95_ms", "p95"), ("p99_ms", "p99")]),
        ("throughput (fps, higher better)", [("seq_fps", "batch 1"), ("peak_fps", "peak async")]),
        ("energy (mJ/inference)", [("mj_per_inference", "mJ/inf")]),
        ("peak temperature (°C)", [("peak_temp_c", "°C")]),
    ]
    live = [(t, ks) for t, ks in panels
            if any(isinstance(r.get(k), (int, float)) for r in rows for k, _ in ks)]
    if not live:
        return
    fig, axes = plt.subplots(1, len(live), figsize=(4.4 * len(live), 4.4), squeeze=False)
    for ax, (title, keys) in zip(axes[0], live):
        width = 0.8 / len(keys)
        for j, (key, lbl) in enumerate(keys):
            vals = [r.get(key) if isinstance(r.get(key), (int, float)) else 0 for r in rows]
            xs = [i + j * width - 0.4 + width / 2 for i in range(len(devs))]
            ax.bar(xs, vals, width=width, label=lbl)
            for x, v in zip(xs, vals):
                if v:
                    ax.text(x, v, f"{v:.0f}" if v >= 10 else f"{v:.2f}",
                            ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(devs)))
        ax.set_xticklabels(devs)
        ax.set_title(title, fontsize=11)
        if len(keys) > 1:
            ax.legend(fontsize=8)
    fig.suptitle("Device comparison", y=1.02, fontsize=13, fontweight="bold")
    save(fig, outdir, "device_comparison.png", made)

# ------------------------------------------------------------------- main
def process(json_path, outdir, made):
    data = json.loads(json_path.read_text())
    stem = json_path.with_suffix("")
    traces = read_csv(Path(f"{stem}_traces.csv"))
    lats   = read_csv(Path(f"{stem}_latencies.csv"))
    sus    = read_csv(Path(f"{stem}_sustained.csv"))

    env = data.get("environment", {})
    print(f"\n{json_path.name}")
    print(f"  host {env.get('hostname')}  cpu {env.get('cpu_model')}")
    print(f"  devices benchmarked: {list((data.get('devices') or {}).keys())}")
    sen = data.get("sensors", {})
    if not sen.get("energy_valid"):
        print(f"  NOTE energy source '{sen.get('energy_source')}' was unusable, "
              f"so power plots may be empty or show charging, not consumption.")
    print(f"  trace samples {len(traces)}, latency samples {len(lats)}, "
          f"sustained buckets {len(sus)}")
    print("  writing:")

    if lats:
        plot_latency_distribution(lats, outdir, made)
        plot_latency_over_time(lats, outdir, made)
    if traces:
        plot_power_temp(traces, outdir, made)
    if sus:
        plot_sustained(sus, outdir, made)
    plot_cold_start(data, outdir, made)
    plot_device_comparison(data, outdir, made)
    if not lats and not traces:
        print("  (no CSV sidecars found next to the JSON - re-run the benchmark to get them)")

def main():
    ap = argparse.ArgumentParser(description="Plot PatchCore benchmark results.")
    ap.add_argument("json", nargs="+", type=Path, help="result_*.json from bench_core.py")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="where to write PNGs (default: next to the JSON)")
    args = ap.parse_args()
    style()
    made = []
    for jp in args.json:
        if not jp.exists():
            print(f"skip {jp}: not found", file=sys.stderr)
            continue
        outdir = args.outdir or jp.parent / f"plots_{jp.stem}"
        outdir.mkdir(parents=True, exist_ok=True)
        process(jp, outdir, made)
    print(f"\n{len(made)} figure(s) written." if made else "\nNo figures written.")

if __name__ == "__main__":
    main()

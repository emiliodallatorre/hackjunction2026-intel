#!/usr/bin/env python3
"""VLA policy inference benchmark (ACT, pi0.5) - performance only, no task success.

Runs the OpenVINO IR(s) inside a Physical AI Studio export's exports/openvino/
folder and reports latency percentiles, throughput, cold-start breakdown, energy
per inference, thermals, sustained throughput and memory.

Unlike the PatchCore benchmark, one inference here produces a whole ACTION CHUNK
(ACT: 100 actions of 6 joints), so raw latency alone is misleading: the policy
only has to run once every chunk_size/control_fps seconds. Every section
therefore also reports actions/s and the control-rate headroom at --fps.

Input is synthetic by default: random tensors at each IR port's declared shape
and dtype. A transformer policy's cost is data-independent (fixed shapes, no
early exit, no data-dependent branching), so noise times identically to a real
camera frame, and the numbers measure silicon rather than camera bandwidth.

Env vars (all optional): BENCH_ITERS BENCH_WARMUP BENCH_SUSTAIN BENCH_DEVICES
BENCH_EXPORT BENCH_REPO_SRC BENCH_FPS BENCH_IDLE BENCH_OUT BENCH_POLICY
BENCH_SKIP_E2E
"""
from __future__ import annotations

import glob
import json
import os
import platform
import re
import statistics
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------- configuration
def _env_int(name, default):
    try:
        return int(float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default

def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

# A VLA forward pass costs 10-1000x a PatchCore one, so the defaults are far
# smaller: 300 iters of pi0.5 on an iGPU would run for hours.
ITERS     = _env_int("BENCH_ITERS", 60)
WARMUP    = _env_int("BENCH_WARMUP", 10)
SUSTAIN   = _env_int("BENCH_SUSTAIN", 300)
IDLE_SECS = _env_int("BENCH_IDLE", 30)
# Control rate the robot runs at; used to turn chunk latency into headroom.
CONTROL_FPS = _env_float("BENCH_FPS", 30.0)
EXPORT_DIR = Path(os.environ.get("BENCH_EXPORT", ".")).expanduser().resolve()
POLICY_NAME = os.environ.get("BENCH_POLICY", "").strip()
SKIP_E2E  = os.environ.get("BENCH_SKIP_E2E", "").strip() not in ("", "0", "no", "false")
# Friendly names -> OpenVINO device names. "igpu" is what people call the
# integrated GPU; OpenVINO just calls it "GPU" (GPU.0 is the integrated one when
# no discrete card is present).
DEVICE_ALIAS = {"CPU": "CPU", "IGPU": "GPU", "GPU": "GPU", "NPU": "NPU",
                "AUTO": "AUTO", "DGPU": "GPU.1"}
DEVICES = []
for _d in os.environ.get("BENCH_DEVICES", "GPU").split(","):
    _d = _d.strip().upper()
    if _d:
        DEVICES.append(DEVICE_ALIAS.get(_d, _d))
OUT_PATH  = os.environ.get("BENCH_OUT", "").strip()

# OpenVINO's model cache must be writable, and must not collide with the
# PatchCore benchmark's cache (different models, same machine).
CACHE_DIR = Path(f"/tmp/vla_bench_ovcache_{os.getuid()}")

BAR = "=" * 74

# Collected for the plotting script: power/temperature traces and raw latencies.
TRACES: list[dict] = []
LATENCIES: list[dict] = []

def collect_trace(sampler, phase, device=""):
    if sampler.trace:
        TRACES.append({"phase": phase, "device": device, "rows": sampler.trace})
    return sampler

def collect_latency(ms, phase, device=""):
    if ms:
        LATENCIES.append({"phase": phase, "device": device, "ms": [round(x, 4) for x in ms]})

def hr(title):
    print(f"\n{BAR}\n{title}\n{BAR}", flush=True)

def note(msg):
    print(f"  {msg}", flush=True)

# ============================================================ sensor discovery
def read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError, TypeError):
        return None

def discover_rapl():
    """Readable Intel RAPL energy counters: {label: path to energy_uj}.

    energy_uj is a monotonic micro-joule counter, so energy over a window is an
    exact difference rather than an average of instantaneous samples. Since the
    Platypus mitigation these are usually mode 0400 (root-only), hence the probe.
    """
    out = {}
    for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:*/")):
        name_f, energy_f = Path(d) / "name", Path(d) / "energy_uj"
        if not (name_f.exists() and energy_f.exists()):
            continue
        try:
            label = name_f.read_text().strip()
            energy_f.read_text()
            out[f"{label}@{Path(d).name}"] = str(energy_f)
        except (PermissionError, OSError):
            continue
    return out

def count_rapl_unreadable():
    n = 0
    for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:*/")):
        f = Path(d) / "energy_uj"
        if f.exists():
            try:
                f.read_text()
            except (PermissionError, OSError):
                n += 1
    return n

def discover_battery():
    for b in sorted(glob.glob("/sys/class/power_supply/BAT*/")):
        p = Path(b)
        if (p / "power_now").exists():
            return {"power_now": str(p / "power_now"), "status": str(p / "status")}
        if (p / "current_now").exists() and (p / "voltage_now").exists():
            return {"current_now": str(p / "current_now"),
                    "voltage_now": str(p / "voltage_now"),
                    "status": str(p / "status")}
    return {}

def discover_temps():
    temps = {}
    for hw in sorted(glob.glob("/sys/class/hwmon/hwmon*/")):
        h = Path(hw)
        try:
            chip = (h / "name").read_text().strip()
        except OSError:
            continue
        for tin in sorted(h.glob("temp*_input")):
            try:
                lbl = tin.with_name(tin.name.replace("_input", "_label"))
                label = lbl.read_text().strip() if lbl.exists() else tin.stem
                tin.read_text()
                temps[f"{chip}/{label}"] = str(tin)
            except OSError:
                continue
    return temps

def ac_online():
    for p in glob.glob("/sys/class/power_supply/A*/online"):
        v = read_int(p)
        if v is not None:
            return bool(v)
    return None

RAPL      = discover_rapl()
RAPL_BAD  = count_rapl_unreadable()
BATTERY   = discover_battery()
TEMPS     = discover_temps()
ON_AC     = ac_online()

def pick_primary_temp(temps):
    for pref in ("Package id 0", "Tctl", "Tdie", "Composite"):
        for k in temps:
            if pref.lower() in k.lower():
                return k
    for k in temps:
        if "core" in k.lower():
            return k
    return next(iter(temps), None)

PRIMARY_TEMP = pick_primary_temp(TEMPS)

# The iGPU has its own temperature sensor, which is the one that throttles when
# the policy runs on GPU. The package sensor can stay cool while the GPU tile
# throttles, so both are tracked.
def pick_gpu_temp(temps):
    for k in temps:
        kl = k.lower()
        if any(t in kl for t in ("i915", "xe", "gpu", "acpitz")) and "pch" not in kl:
            return k
    return None

GPU_TEMP = pick_gpu_temp(TEMPS)

# Energy source. The battery counter only reports real draw while DISCHARGING:
# on AC it reports charge flow, which is unrelated to what inference costs. In
# that case energy is marked unavailable rather than reported as a wrong number.
if RAPL:
    ENERGY_SOURCE, ENERGY_VALID = "rapl", True
elif BATTERY:
    ENERGY_SOURCE = "battery"
    ENERGY_VALID = not ON_AC
else:
    ENERGY_SOURCE, ENERGY_VALID = "none", False

# =================================================================== sampler
class Sampler(threading.Thread):
    """Background power/temperature sampler (100 ms period)."""

    def __init__(self, period=0.1):
        super().__init__(daemon=True)
        self.period = period
        self._stop = threading.Event()
        self.temps: list[float] = []
        self.gpu_temps: list[float] = []
        self.watts: list[float] = []
        self.trace: list[dict] = []
        self.energy_j = 0.0
        self.label = ""
        self.rapl_start = {k: read_int(v) for k, v in RAPL.items()}
        self.rapl_end = dict(self.rapl_start)
        self.rapl_max = {}
        for k, v in RAPL.items():
            mx = Path(v).with_name("max_energy_range_uj")
            self.rapl_max[k] = read_int(str(mx)) if mx.exists() else None
        self.elapsed = 0.0
        self.t0 = time.monotonic()

    def _battery_watts(self):
        if "power_now" in BATTERY:
            uw = read_int(BATTERY["power_now"])
            return uw / 1e6 if uw else None
        if "current_now" in BATTERY:
            c, v = read_int(BATTERY["current_now"]), read_int(BATTERY["voltage_now"])
            if c and v:
                return (c / 1e6) * (v / 1e6)
        return None

    def run(self):
        last = time.monotonic()
        prev_rapl = dict(self.rapl_start)
        while not self._stop.is_set():
            now = time.monotonic()
            rel = now - self.t0
            dt = now - last
            row = {"t_s": round(rel, 3)}
            if PRIMARY_TEMP:
                raw = read_int(TEMPS[PRIMARY_TEMP])
                if raw is not None:
                    c = raw / 1000.0
                    self.temps.append(c)
                    row["temp_c"] = round(c, 2)
            if GPU_TEMP:
                raw = read_int(TEMPS[GPU_TEMP])
                if raw is not None:
                    c = raw / 1000.0
                    self.gpu_temps.append(c)
                    row["gpu_temp_c"] = round(c, 2)
            w = self._battery_watts()
            if w:
                self.watts.append(w)
                self.energy_j += w * dt
                row["battery_w"] = round(w, 3)
            # Instantaneous RAPL power = energy delta / time delta, per domain.
            for k, v in RAPL.items():
                e = read_int(v)
                if e is None:
                    continue
                self.rapl_end[k] = e
                pe = prev_rapl.get(k)
                if pe is not None and dt > 0:
                    d = e - pe
                    if d < 0:
                        d += self.rapl_max.get(k) or 2**32
                    row[f"rapl_w[{k}]"] = round(d / 1e6 / dt, 3)
                prev_rapl[k] = e
            if len(row) > 1:
                self.trace.append(row)
            last = now
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        self.join(timeout=2)
        self.elapsed = time.monotonic() - self.t0
        return self

    def rapl_joules(self):
        out = {}
        for k in RAPL:
            s, e = self.rapl_start.get(k), self.rapl_end.get(k)
            if s is None or e is None:
                continue
            d = e - s
            if d < 0:                                  # counter wrapped
                d += self.rapl_max.get(k) or 2**32
            out[k] = d / 1e6
        return out

    def summary(self):
        d = {"elapsed_s": round(self.elapsed, 3)}
        if self.temps:
            d.update(temp_sensor=PRIMARY_TEMP,
                     temp_start_c=round(self.temps[0], 1),
                     temp_mean_c=round(statistics.fmean(self.temps), 1),
                     temp_max_c=round(max(self.temps), 1),
                     temp_end_c=round(self.temps[-1], 1))
        if self.gpu_temps:
            d.update(gpu_temp_sensor=GPU_TEMP,
                     gpu_temp_mean_c=round(statistics.fmean(self.gpu_temps), 1),
                     gpu_temp_max_c=round(max(self.gpu_temps), 1))
        rj = self.rapl_joules()
        if rj and self.elapsed > 0:
            d["rapl_joules"] = {k: round(v, 2) for k, v in rj.items()}
            d["rapl_watts"] = {k: round(v / self.elapsed, 2) for k, v in rj.items()}
        if self.watts:
            d.update(battery_watts_mean=round(statistics.fmean(self.watts), 2),
                     battery_watts_max=round(max(self.watts), 2),
                     battery_joules=round(self.energy_j, 2))
        return d

def window_joules(sm):
    """Best energy figure for a sampling window, or (None, reason) if untrustworthy."""
    if "rapl_joules" in sm:
        pkg = [v for k, v in sm["rapl_joules"].items() if k.lower().startswith("package")]
        return (sum(pkg) if pkg else max(sm["rapl_joules"].values())), "rapl-package"
    if "battery_joules" in sm and ENERGY_VALID:
        return sm["battery_joules"], "battery-system"
    if "battery_joules" in sm:
        return None, "battery-on-ac-invalid"
    return None, "no-counter"

def fmt_energy(sm, n_inferences, wall):
    e, src = window_joules(sm)
    if e is None or not wall:
        return None, src
    return {"energy_j": round(e, 2), "avg_watts": round(e / wall, 2),
            "mj_per_inference": round(e / n_inferences * 1e3, 1), "source": src}, src

# ============================================================== percentiles
def percentiles(xs):
    s = sorted(xs)
    def p(q):
        if not s:
            return float("nan")
        k = (len(s) - 1) * q / 100.0
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (k - lo)
    return {"n": len(s), "min_ms": s[0], "p50_ms": p(50), "p90_ms": p(90),
            "p95_ms": p(95), "p99_ms": p(99), "max_ms": s[-1],
            "mean_ms": statistics.fmean(s), "stdev_ms": statistics.pstdev(s)}

def print_latency(st, wall, chunk=None):
    print(f"  latency (ms)   min={st['min_ms']:8.2f}  p50={st['p50_ms']:8.2f}  p90={st['p90_ms']:8.2f}")
    print(f"                 p95={st['p95_ms']:8.2f}  p99={st['p99_ms']:8.2f}  max={st['max_ms']:8.2f}")
    print(f"                 mean={st['mean_ms']:7.2f}  stdev={st['stdev_ms']:7.2f}")
    if wall:
        print(f"  throughput     {st['n'] / wall:.2f} inferences/s (sequential, batch 1)")
    if st["p50_ms"]:
        j = st["p99_ms"] / st["p50_ms"]
        print(f"  jitter         p99/p50 = {j:.2f}x "
              f"({'noisy - scheduling or thermal' if j > 1.5 else 'tight'})")
    if chunk:
        print_chunk_budget(st, chunk)

def chunk_budget(st, chunk):
    """Turn chunk latency into the control-rate numbers that actually matter.

    One inference emits `chunk` actions, which the runtime plays out at
    CONTROL_FPS. So the policy only has to produce a chunk every
    chunk/CONTROL_FPS seconds; p99 latency divided by that budget is the
    fraction of real time inference consumes.
    """
    if not chunk or not CONTROL_FPS:
        return {}
    budget_ms = chunk / CONTROL_FPS * 1e3
    return {
        "chunk_actions": chunk,
        "control_fps": CONTROL_FPS,
        "chunk_budget_ms": round(budget_ms, 1),
        "actions_per_s_p50": round(chunk / (st["p50_ms"] / 1e3), 1),
        "duty_p50_pct": round(st["p50_ms"] / budget_ms * 100, 2),
        "duty_p99_pct": round(st["p99_ms"] / budget_ms * 100, 2),
        "max_control_fps_p99": round(chunk / (st["p99_ms"] / 1e3), 1),
        "realtime_p99": st["p99_ms"] <= budget_ms,
    }

def print_chunk_budget(st, chunk):
    b = chunk_budget(st, chunk)
    if not b:
        return
    print(f"  action chunk   {chunk} actions/inference -> {b['actions_per_s_p50']:.1f} actions/s at p50")
    print(f"  control budget {b['chunk_budget_ms']:.0f} ms per chunk at {CONTROL_FPS:g} Hz; "
          f"p50 uses {b['duty_p50_pct']:.1f}%, p99 uses {b['duty_p99_pct']:.1f}%")
    verdict = ("REAL-TIME: p99 fits the budget" if b["realtime_p99"]
               else "TOO SLOW: p99 exceeds the budget, the queue will starve")
    print(f"  -> {verdict}  (ceiling {b['max_control_fps_p99']:.1f} Hz at p99)")

# ============================================================== environment
hr("0. ENVIRONMENT")
try:
    import numpy as np
    import openvino as ov
except ImportError as exc:
    print(f"FATAL: missing dependency: {exc}")
    print("Activate the project venv first (.venv/bin/activate).")
    sys.exit(1)

core = ov.Core()

def cpu_model():
    m = re.search(r"model name\s*:\s*(.+)", Path("/proc/cpuinfo").read_text())
    return m.group(1).strip() if m else platform.processor()

def uniq_sysfs(pattern):
    vals = sorted({Path(p).read_text().strip() for p in glob.glob(pattern)})
    return ",".join(vals) or "n/a"

ENV = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "hostname": platform.node(),
    "kernel": platform.release(),
    "cpu_model": cpu_model(),
    "cpu_count": os.cpu_count(),
    "openvino_version": ov.__version__,
    "numpy_version": np.__version__,
    "python": sys.version.split()[0],
    "ov_available_devices": core.available_devices,
    "requested_devices": DEVICES,
    "cpu_governor": uniq_sysfs("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"),
    "energy_perf_pref": uniq_sysfs("/sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference"),
    "ac_online": ON_AC,
    "battery_status": (Path(BATTERY["status"]).read_text().strip() if BATTERY.get("status") else "n/a"),
    "running_as_root": os.geteuid() == 0,
    "export_dir": str(EXPORT_DIR),
    "control_fps": CONTROL_FPS,
    "iters": ITERS, "warmup": WARMUP, "sustain_s": SUSTAIN,
}
for k, v in ENV.items():
    print(f"  {k:26s} {v}")

print("\n  Device details:")
for dev in core.available_devices:
    try:
        print(f"    {dev:6s} {core.get_property(dev, 'FULL_DEVICE_NAME')}")
    except Exception:
        print(f"    {dev:6s} (name unavailable)")
    if dev.startswith("GPU"):
        # The iGPU shares system DRAM with the CPU. A 3B-parameter VLA can
        # exceed what the driver will allocate for one buffer, which is the
        # usual cause of a compile failure on GPU rather than raw slowness.
        for prop in ("GPU_DEVICE_TOTAL_MEM_SIZE", "MAX_MEM_ALLOC_SIZE",
                     "GPU_EXECUTION_UNITS_COUNT", "DEVICE_ARCHITECTURE"):
            try:
                v = core.get_property(dev, prop)
                if prop.endswith("MEM_SIZE") or prop.endswith("ALLOC_SIZE"):
                    v = f"{int(v) / 1e9:.2f} GB"
                print(f"           {prop:30s} {v}")
            except Exception:
                pass

print("\n  Sensors:")
print(f"    energy source            {ENERGY_SOURCE} ({'usable' if ENERGY_VALID else 'NOT usable'})")
print(f"    RAPL readable            {list(RAPL) or 'none'}")
if RAPL_BAD:
    print(f"    RAPL root-only           {RAPL_BAD} counters (re-run with sudo for package watts)")
print(f"    temp sensors             {len(TEMPS)} found, primary = {PRIMARY_TEMP}, gpu = {GPU_TEMP}")
if ENERGY_SOURCE == "battery" and ON_AC:
    print("    NOTE: on AC the battery counter measures charging, not consumption.")
    print("          Energy is reported as unavailable. For mJ/inference either")
    print("          unplug the charger, or re-run with sudo to read RAPL.")

RESULTS = {"environment": ENV,
           "sensors": {"energy_source": ENERGY_SOURCE, "energy_valid": ENERGY_VALID,
                       "rapl_readable": list(RAPL), "rapl_root_only": RAPL_BAD,
                       "primary_temp": PRIMARY_TEMP, "gpu_temp": GPU_TEMP,
                       "temp_count": len(TEMPS)},
           "devices": {}}

# ====================================================== export discovery
# A Studio export is a folder holding runtime.yaml and exports/openvino/. The
# benchmark accepts either the export root or the openvino folder itself, so
# --export can point at whatever the user already has unzipped.
def find_openvino_dir(root: Path) -> Path:
    if (root / "manifest.json").exists() or list(root.glob("*.xml")):
        return root
    cand = root / "exports" / "openvino"
    if cand.is_dir():
        return cand
    hits = sorted(root.glob("*/exports/openvino"))
    if hits:
        return hits[0]
    return root

OV_DIR = find_openvino_dir(EXPORT_DIR)
MANIFEST_PATH = OV_DIR / "manifest.json"
MANIFEST = {}
if MANIFEST_PATH.exists():
    try:
        MANIFEST = json.loads(MANIFEST_PATH.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"\n  WARNING: manifest.json unreadable ({exc}); falling back to IR ports only")

def manifest_policy(man) -> str:
    return (((man.get("policy") or {}).get("name")) or "unknown")

def manifest_artifacts(man, ov_dir: Path) -> list[Path]:
    """IR files named by the manifest, else every .xml in the folder.

    ACT ships one act.xml. A pi0.5 export may ship several (vision-language
    encoder plus action expert), either as a dict of role -> file or a list.
    """
    art = ((man.get("model") or {}).get("artifacts") or {}).get("openvino")
    out = []
    if isinstance(art, str):
        out = [ov_dir / art]
    elif isinstance(art, dict):
        out = [ov_dir / v for v in art.values() if isinstance(v, str)]
    elif isinstance(art, list):
        out = [ov_dir / v for v in art if isinstance(v, str)]
    out = [p for p in out if p.suffix == ".xml" and p.exists()]
    if not out:
        out = sorted(ov_dir.rglob("*.xml"))
    return out

def artifact_roles(man, ov_dir: Path) -> dict:
    art = ((man.get("model") or {}).get("artifacts") or {}).get("openvino")
    if isinstance(art, dict):
        return {str(ov_dir / v): k for k, v in art.items() if isinstance(v, str)}
    return {}

def manifest_features(man, kind: str) -> list[dict]:
    key = "input_features" if kind == "in" else "output_features"
    feats = (man.get("model") or {}).get(key) or []
    out = []
    for f in feats:
        a = f.get("init_args") if isinstance(f, dict) else None
        if isinstance(a, dict):
            out.append(a)
    return out

def chunk_size_from_manifest(man) -> int | None:
    """Actions per inference, read off the ACTION output feature's shape.

    ACT declares action shape [100, 6]: 100 timesteps of 6 joints. The leading
    dimension is the chunk length, which is what the action queue plays out.
    """
    for a in manifest_features(man, "out"):
        if str(a.get("ftype", "")).upper() == "ACTION":
            shape = a.get("shape") or []
            if len(shape) >= 2:
                return int(shape[0])
    return None

POLICY = POLICY_NAME or manifest_policy(MANIFEST)
IRS = manifest_artifacts(MANIFEST, OV_DIR)
ROLES = artifact_roles(MANIFEST, OV_DIR)
CHUNK = chunk_size_from_manifest(MANIFEST)

if not IRS:
    print(f"\nFATAL: no OpenVINO IR (*.xml) found under {OV_DIR}")
    print("  Point BENCH_EXPORT at the export root (holding exports/openvino/)")
    print("  or at the exports/openvino folder itself.")
    sys.exit(1)

hr("MODEL")
print(f"  policy           {POLICY}")
print(f"  export           {EXPORT_DIR}")
print(f"  openvino dir     {OV_DIR}")
runner = (((MANIFEST.get("model") or {}).get("runner") or {}).get("class_path") or "unknown")
print(f"  runner           {runner}")
print(f"  IR files         {len(IRS)}")
total_bin = 0
for p in IRS:
    b = p.with_suffix(".bin")
    sz = b.stat().st_size if b.exists() else 0
    total_bin += sz
    role = ROLES.get(str(p), "")
    print(f"    {p.name:28s} weights {sz / 1e6:9.1f} MB  {('role: ' + role) if role else ''}")
print(f"  total weights    {total_bin / 1e6:.1f} MB on disk")
if CHUNK:
    print(f"  action chunk     {CHUNK} actions per inference "
          f"-> one inference covers {CHUNK / CONTROL_FPS:.2f} s at {CONTROL_FPS:g} Hz control")
else:
    print("  action chunk     unknown (no ACTION output feature in the manifest);")
    print("                   control-rate headroom will not be reported")
pre = (MANIFEST.get("model") or {}).get("preprocessors") or []
if pre:
    print(f"  preprocessors    {', '.join(str(p.get('type', '?')) for p in pre)}")
    for p in pre:
        if p.get("type") == "resize":
            print(f"    resize to {p.get('image_resolution')} mode={p.get('mode')} "
                  f"(the IR takes the RESIZED image; raw camera frames are larger)")
print(f"  ov cache dir     {CACHE_DIR}")

feats_in = manifest_features(MANIFEST, "in")
if feats_in:
    print("\n  Declared input features (pre-preprocessing shapes):")
    for a in feats_in:
        print(f"    {str(a.get('name')):26s} {str(a.get('ftype')):8s} "
              f"shape={a.get('shape')} dtype={a.get('dtype')}")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

RESULTS["model"] = {
    "policy": POLICY, "runner": runner, "openvino_dir": str(OV_DIR),
    "ir_files": [p.name for p in IRS], "roles": {Path(k).name: v for k, v in ROLES.items()},
    "weights_mb": round(total_bin / 1e6, 1), "chunk_actions": CHUNK,
    "preprocessors": pre, "input_features": feats_in,
    "output_features": manifest_features(MANIFEST, "out"),
}

# ====================================================== synthetic inputs
OV_TO_NP = {
    "f32": np.float32, "f16": np.float16, "f64": np.float64,
    "i64": np.int64, "i32": np.int32, "i16": np.int16, "i8": np.int8,
    "u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64,
    "boolean": bool, "bf16": np.float32,
}

def np_dtype(port):
    return OV_TO_NP.get(port.get_element_type().get_type_name(), np.float32)

def static_shape(port, name=""):
    """Concrete shape for one IR port, resolving any dynamic dimension.

    ACT's ports are fully static. A pi0.5 export can have a dynamic sequence
    length on its token inputs; those are pinned to the dimension's minimum when
    bounded, else to 1 for batch and a small default for the rest, so the model
    compiles and times a realistic single-observation pass.
    """
    ps = port.get_partial_shape()
    dims = []
    for i in range(len(ps)):
        d = ps[i]
        if d.is_static:
            dims.append(d.get_length())
        elif d.get_max_length() not in (-1, 0) and d.get_max_length() < 1 << 30:
            dims.append(d.get_max_length())
        elif d.get_min_length() > 0:
            dims.append(d.get_min_length())
        else:
            dims.append(1 if i == 0 else 1)
    return dims

def fill(shape, dtype, name=""):
    """One synthetic tensor. Values are plausible for the port, not just zeros.

    Zeros can hit faster paths in some kernels and hide real cost, so float
    ports get uniform noise. Integer ports are token ids or masks: small
    non-negative values keep any embedding lookup in range.
    """
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    if np.issubdtype(dtype, np.floating):
        return rng.random(shape).astype(dtype)
    if dtype is bool:
        return np.ones(shape, dtype=bool)
    return rng.integers(0, 2, size=shape, dtype=dtype)

def build_inputs(model, pool=4):
    """A pool of synthetic input dicts keyed by port name, one entry per IR."""
    specs = []
    for port in model.inputs:
        name = port.get_any_name()
        specs.append((name, static_shape(port, name), np_dtype(port)))
    pools = []
    for k in range(pool):
        pools.append({n: fill(s, d, f"{n}#{k}") for n, s, d in specs})
    return pools, specs

# ============================================================= idle baseline
# Measured FIRST, while the machine is cold. Measuring it after the load runs
# captures fans still spinning down and a hot package, which is what produces
# nonsensical negative marginal power.
hr(f"1. IDLE BASELINE ({IDLE_SECS} s, before any load)")
IDLE = {}
if IDLE_SECS > 0:
    print("  idling so load numbers can be reported as a delta ...", flush=True)
    s = Sampler(); s.start(); time.sleep(IDLE_SECS); s.stop()
    collect_trace(s, "idle")
    sm = s.summary()
    e, src = window_joules(sm)
    if e is not None:
        IDLE["idle_watts"] = e / sm["elapsed_s"]
        print(f"  idle power     {IDLE['idle_watts']:.2f} W  [{src}]")
    else:
        print(f"  idle power     unavailable [{src}]")
    if "temp_mean_c" in sm:
        print(f"  idle temp      mean={sm['temp_mean_c']}C max={sm['temp_max_c']}C "
              f"start={sm['temp_start_c']}C  [{sm['temp_sensor']}]")
        IDLE["idle_temp_mean_c"] = sm["temp_mean_c"]
    if "gpu_temp_mean_c" in sm:
        print(f"  idle gpu temp  mean={sm['gpu_temp_mean_c']}C  [{sm['gpu_temp_sensor']}]")
    IDLE["sampler"] = sm
else:
    print("  skipped (BENCH_IDLE=0)")
RESULTS["idle"] = IDLE

# ========================================================== per-IR helpers
def make_core(cache=True):
    c = ov.Core()
    if cache:
        c.set_property({"CACHE_DIR": str(CACHE_DIR)})
    return c

def load_ir(xml, device, cache=True, config=None):
    c = make_core(cache)
    m = c.read_model(xml)
    return c.compile_model(m, device, config or {})

def report_properties(compiled, device):
    props = ["EXECUTION_DEVICES", "NUM_STREAMS", "OPTIMAL_NUMBER_OF_INFER_REQUESTS"]
    if device.startswith("CPU"):
        props += ["INFERENCE_NUM_THREADS", "INFERENCE_PRECISION_HINT"]
    elif device.startswith("GPU"):
        props += ["INFERENCE_PRECISION_HINT", "GPU_HOST_TASK_PRIORITY", "GPU_QUEUE_THROTTLE"]
    elif device.startswith("NPU"):
        props += ["NPU_COMPILATION_MODE_PARAMS", "NPU_TILES", "INFERENCE_PRECISION_HINT"]
    for prop in props:
        try:
            print(f"     {prop:34s} {compiled.get_property(prop)}")
        except Exception:
            pass
    # EXECUTION_DEVICES is the fallback check: a transformer op the GPU plugin
    # cannot take runs on CPU instead, which silently caps the speedup.
    ex_list = None
    try:
        ex = compiled.get_property("EXECUTION_DEVICES")
        ex_list = [ex] if isinstance(ex, str) else [str(e) for e in ex]
        if not any(str(e).upper().startswith(device.split(".")[0]) for e in ex_list):
            print(f"     !! graph did NOT run on {device}: actually on {ex_list}")
        elif len(ex_list) > 1:
            print(f"     !! graph SPLIT across {ex_list} - partial fallback, speedup is capped")
    except Exception:
        pass
    return ex_list

# ========================================================== per-device suite
def bench_ir(xml, device, label, is_primary):
    """Latency / throughput / cold start for a single IR on one device."""
    R = {"ir": xml.name, "role": ROLES.get(str(xml), "")}
    chunk = CHUNK if is_primary else None

    # --- cold start: read / compile / first infer, separately -------------
    print(f"\n  2. COLD START [{label}]")
    def cold_start(tag, cache, clear):
        if clear and CACHE_DIR.exists():
            for f in CACHE_DIR.rglob("*"):
                if f.is_file():
                    try:
                        f.unlink()
                    except OSError:
                        pass
        c = make_core(cache)
        t0 = time.perf_counter(); m = c.read_model(xml); t1 = time.perf_counter()
        t2 = time.perf_counter(); cm = c.compile_model(m, device); t3 = time.perf_counter()
        pools, _ = build_inputs(m, pool=1)
        t4 = time.perf_counter(); cm(pools[0]); t5 = time.perf_counter()
        d = {"read_ms": (t1 - t0) * 1e3, "compile_ms": (t3 - t2) * 1e3,
             "first_infer_ms": (t5 - t4) * 1e3}
        d["total_ms"] = sum(d.values())
        print(f"     {tag:26s} read={d['read_ms']:8.1f}  compile={d['compile_ms']:9.1f}  "
              f"first_infer={d['first_infer_ms']:8.1f}  total={d['total_ms']:9.1f}  (ms)")
        del cm
        return d

    R["cold_start"] = {
        "cold_cache": cold_start("cold cache (first ever)", True, True),
        "warm_cache": cold_start("warm cache (cached blob)", True, False),
        "no_cache":   cold_start("no cache", False, False),
    }
    saved = R["cold_start"]["cold_cache"]["compile_ms"] - R["cold_start"]["warm_cache"]["compile_ms"]
    print(f"     -> blob cache saves {saved:.0f} ms of compile time on {device}")
    print(f"     (a cold compile is paid once per process start, so it sets how long")
    print(f"      the robot waits before its first action, not the control rate)")

    # --- steady-state latency, batch 1 -----------------------------------
    print(f"\n  3. STEADY-STATE LATENCY [{label}] (batch 1, LATENCY hint)")
    compiled = load_ir(xml, device, config={"PERFORMANCE_HINT": "LATENCY"})
    ex_list = report_properties(compiled, device)
    R["execution_devices"] = ex_list
    try:
        R["inference_precision"] = str(compiled.get_property("INFERENCE_PRECISION_HINT"))
    except Exception:
        R["inference_precision"] = "unknown"

    model_for_shapes = make_core().read_model(xml)
    pools, specs = build_inputs(model_for_shapes, pool=4)
    print("     synthetic input ports:")
    for n, s, d in specs:
        print(f"       {n:28s} {tuple(s)} {np.dtype(d).name}")

    req = compiled.create_infer_request()
    for i in range(WARMUP):
        req.infer(pools[i % len(pools)])
    s = Sampler(); s.start()
    lat = []
    t0 = time.perf_counter()
    for i in range(ITERS):
        t = time.perf_counter()
        req.infer(pools[i % len(pools)])
        lat.append((time.perf_counter() - t) * 1e3)
    wall = time.perf_counter() - t0
    s.stop(); collect_trace(s, "steady_state", f"{device}/{label}"); sm = s.summary()
    collect_latency(lat, "steady_state", f"{device}/{label}")
    st = percentiles(lat); st["wall_s"] = round(wall, 3); st["throughput_fps"] = len(lat) / wall
    print_latency(st, wall, chunk)
    if chunk:
        st["chunk"] = chunk_budget(st, chunk)
    en, src = fmt_energy(sm, len(lat), wall)
    if en:
        st["energy"] = en
        print(f"  energy         {en['avg_watts']:.2f} W avg, {en['mj_per_inference']:.1f} mJ/inference  [{src}]")
        if chunk:
            print(f"                 {en['mj_per_inference'] / chunk:.2f} mJ per action "
                  f"({chunk} actions per inference)")
        if "idle_watts" in IDLE:
            marg = en["avg_watts"] - IDLE["idle_watts"]
            st["marginal_watts"] = round(marg, 2)
            st["marginal_mj_per_inference"] = round(marg / st["throughput_fps"] * 1e3, 1)
            print(f"  marginal       {marg:+.2f} W over idle -> "
                  f"{st['marginal_mj_per_inference']:.1f} mJ/inference attributable to inference")
    else:
        print(f"  energy         unavailable [{src}]")
    if "temp_max_c" in sm:
        print(f"  temperature    start={sm['temp_start_c']}C mean={sm['temp_mean_c']}C max={sm['temp_max_c']}C")
    if "gpu_temp_max_c" in sm:
        print(f"  gpu temp       mean={sm['gpu_temp_mean_c']}C max={sm['gpu_temp_max_c']}C")
    R["steady_state"] = {"latency": st, "sampler": sm}
    seq_fps = st["throughput_fps"]

    # --- peak throughput, async ------------------------------------------
    # Not the control-loop number (the robot needs one chunk at a time), but it
    # shows whether the device is latency-bound or occupancy-bound.
    print(f"\n  4. PEAK THROUGHPUT [{label}] (async queue, THROUGHPUT hint)")
    try:
        tc = load_ir(xml, device, config={"PERFORMANCE_HINT": "THROUGHPUT"})
        njobs = tc.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        try:
            nstreams = tc.get_property("NUM_STREAMS")
        except Exception:
            nstreams = "?"
        print(f"     parallel requests {njobs}, streams {nstreams}")
        q = ov.AsyncInferQueue(tc, njobs)
        for i in range(min(WARMUP, 8)):
            q.start_async(pools[i % len(pools)])
        q.wait_all()
        s = Sampler(); s.start()
        t0 = time.perf_counter()
        for i in range(ITERS):
            q.start_async(pools[i % len(pools)])
        q.wait_all()
        w = time.perf_counter() - t0
        s.stop(); collect_trace(s, "throughput", f"{device}/{label}"); sm2 = s.summary()
        TP = {"n": ITERS, "wall_s": round(w, 3), "throughput_fps": ITERS / w,
              "n_requests": njobs}
        print(f"  throughput     {TP['throughput_fps']:.2f} inferences/s "
              f"({TP['throughput_fps'] / seq_fps:.2f}x the batch-1 rate)")
        if TP["throughput_fps"] / seq_fps < 1.15:
            print("     -> no headroom from concurrency: the device is already saturated")
            print("        by one request, which is typical for a large transformer.")
        en2, src2 = fmt_energy(sm2, ITERS, w)
        if en2:
            TP["energy"] = en2
            print(f"  energy         {en2['avg_watts']:.2f} W avg, {en2['mj_per_inference']:.1f} mJ/inference  [{src2}]")
        if "temp_max_c" in sm2:
            print(f"  temperature    max={sm2['temp_max_c']}C")
        R["throughput"] = {"stats": TP, "sampler": sm2}
        del q, tc
    except Exception as exc:
        print(f"     SKIPPED: {exc}")
        R["throughput"] = {"error": str(exc)}

    # --- sustained load ---------------------------------------------------
    print(f"\n  6. SUSTAINED LOAD [{label}] ({SUSTAIN} s) - thermal throttling")
    if SUSTAIN <= 0:
        print("     skipped (BENCH_SUSTAIN=0)")
        R["sustained"] = {"skipped": True}
    else:
        print(f"     running {SUSTAIN} s of continuous batch-1 inference, 10 s buckets ...", flush=True)
        s = Sampler(); s.start()
        buckets, bucket = [], []
        t0 = bstart = time.perf_counter()
        i = 0
        while time.perf_counter() - t0 < SUSTAIN:
            t = time.perf_counter()
            req.infer(pools[i % len(pools)])
            bucket.append((time.perf_counter() - t) * 1e3)
            i += 1
            if time.perf_counter() - bstart >= 10.0:
                dur = time.perf_counter() - bstart
                bs = sorted(bucket)
                tc_now = (read_int(TEMPS[PRIMARY_TEMP]) / 1000.0) if PRIMARY_TEMP else None
                gt_now = (read_int(TEMPS[GPU_TEMP]) / 1000.0) if GPU_TEMP else None
                buckets.append({"t_s": round(time.perf_counter() - t0, 1), "n": len(bucket),
                                "fps": len(bucket) / dur,
                                "p50_ms": statistics.median(bs),
                                "p99_ms": bs[min(len(bs) - 1, int(len(bs) * 0.99))],
                                "temp_c": tc_now, "gpu_temp_c": gt_now})
                bucket, bstart = [], time.perf_counter()
        w = time.perf_counter() - t0
        s.stop(); collect_trace(s, "sustained", f"{device}/{label}"); sm4 = s.summary()
        if buckets:
            print(f"\n     {'t(s)':>6} {'inf/s':>8} {'p50(ms)':>10} {'p99(ms)':>10} "
                  f"{'temp(C)':>8} {'gpu(C)':>7}")
            for b in buckets:
                tstr = f"{b['temp_c']:.1f}" if b["temp_c"] is not None else "n/a"
                gstr = f"{b['gpu_temp_c']:.1f}" if b["gpu_temp_c"] is not None else "n/a"
                print(f"     {b['t_s']:>6.0f} {b['fps']:>8.2f} {b['p50_ms']:>10.2f} "
                      f"{b['p99_ms']:>10.2f} {tstr:>8} {gstr:>7}")
        else:
            print(f"     (no complete 10 s bucket: one inference takes "
                  f"{st['p50_ms']:.0f} ms, raise --sustain)")
        SUS = {"wall_s": round(w, 1), "total_inferences": i, "buckets": buckets}
        if len(buckets) >= 2:
            f0, fl = buckets[0], buckets[-1]
            drop = (1 - fl["fps"] / f0["fps"]) * 100
            SUS.update(first_fps=f0["fps"], last_fps=fl["fps"], throughput_drop_pct=drop)
            print(f"\n     burst    (0-10s)  {f0['fps']:.2f} inf/s @ {f0['temp_c']}C")
            print(f"     sustained (last)  {fl['fps']:.2f} inf/s @ {fl['temp_c']}C")
            print(f"     -> drop {drop:.1f}%  {'THROTTLING' if drop > 5 else 'stable'}")
            if chunk:
                b = chunk_budget({"p50_ms": 1e3 / fl["fps"], "p99_ms": fl["p99_ms"]}, chunk)
                print(f"     -> sustained ceiling {b['max_control_fps_p99']:.1f} Hz control "
                      f"(budget {b['duty_p99_pct']:.1f}% used at p99)")
        en4, src4 = fmt_energy(sm4, max(i, 1), w)
        if en4:
            SUS["energy"] = en4
            print(f"     energy   {en4['avg_watts']:.2f} W avg, {en4['mj_per_inference']:.1f} mJ/inference  [{src4}]")
        if "temp_max_c" in sm4:
            print(f"     temps    start={sm4['temp_start_c']}C max={sm4['temp_max_c']}C end={sm4['temp_end_c']}C")
        if "gpu_temp_max_c" in sm4:
            print(f"     gpu      max={sm4['gpu_temp_max_c']}C")
        R["sustained"] = {"stats": SUS, "sampler": sm4}

    del compiled, req
    return R

def bench_end_to_end(device):
    """Whole-policy InferenceModel.infer(): preprocessing + every IR + postprocessing.

    This is the number that decides the control rate, because the runtime calls
    infer() with a full observation (raw camera frames at their native size, the
    joint state, and the task string) and gets an action chunk back. The per-IR
    sections above exclude the resize and the normalisation, which run on CPU.
    """
    print(f"\n  5. END-TO-END InferenceModel.infer() on {device}")
    if SKIP_E2E:
        print("     skipped (BENCH_SKIP_E2E set)")
        return {"skipped": True}
    try:
        repo_src = os.environ.get("BENCH_REPO_SRC", "")
        if repo_src:
            p = str(Path(repo_src).expanduser().resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
        from physicalai.inference import InferenceModel
    except ImportError as exc:
        print(f"     SKIPPED: physicalai not importable ({exc})")
        print("     The per-IR numbers above are unaffected; they are pure OpenVINO.")
        return {"error": f"physicalai not importable: {exc}"}

    try:
        model = InferenceModel(export_dir=str(OV_DIR), backend="openvino", device=device)
    except Exception as exc:
        print(f"     SKIPPED: InferenceModel({device}) failed: {type(exc).__name__}: {exc}")
        return {"error": f"{type(exc).__name__}: {exc}"}

    try:
        # One synthetic observation at the manifest's PRE-preprocessing shapes:
        # raw camera frames, so the resize cost lands inside the measurement.
        rng = np.random.default_rng(0)
        obs = {}
        for a in feats_in:
            name, ftype = a.get("name"), str(a.get("ftype", "")).upper()
            shape = tuple(int(x) for x in (a.get("shape") or []))
            dt = np.dtype(a.get("dtype") or "float32")
            if ftype == "VISUAL":
                obs[name] = rng.random((1, *shape)).astype(dt)
            else:
                obs[name] = rng.random((1, *shape)).astype(dt)
        task = os.environ.get("BENCH_TASK", "Move any brick to the center")

        def one():
            try:
                return model.infer(obs, task=task)
            except TypeError:
                return model.infer(obs)

        for _ in range(min(WARMUP, 5)):
            one()
        s = Sampler(); s.start()
        e2e = []
        t0 = time.perf_counter()
        for _ in range(ITERS):
            t = time.perf_counter()
            one()
            e2e.append((time.perf_counter() - t) * 1e3)
        w = time.perf_counter() - t0
        s.stop(); collect_trace(s, "end_to_end", device); sm = s.summary()
        collect_latency(e2e, "end_to_end", device)
        E = percentiles(e2e); E["throughput_fps"] = len(e2e) / w
        print(f"     latency (ms)  p50={E['p50_ms']:8.2f}  p95={E['p95_ms']:8.2f}  "
              f"p99={E['p99_ms']:8.2f}  max={E['max_ms']:8.2f}")
        print(f"     throughput    {E['throughput_fps']:.2f} chunks/s")
        if CHUNK:
            E["chunk"] = chunk_budget(E, CHUNK)
            print_chunk_budget(E, CHUNK)
        en, src = fmt_energy(sm, len(e2e), w)
        if en:
            E["energy"] = en
            print(f"     energy        {en['avg_watts']:.2f} W avg, {en['mj_per_inference']:.1f} mJ/chunk  [{src}]")
        return {"stats": E, "sampler": sm}
    except Exception as exc:
        print(f"     SKIPPED: {type(exc).__name__}: {exc}")
        return {"error": f"{type(exc).__name__}: {exc}"}
    finally:
        for closer in ("release", "close"):
            fn = getattr(model, closer, None)
            if fn is not None:
                try:
                    fn()
                    break
                except Exception:
                    pass

def bench_device(device):
    """Full performance suite for one OpenVINO device, over every IR in the export."""
    D = {"irs": {}}
    hr(f"DEVICE: {device}   policy: {POLICY}")
    # With several IRs the LAST one is not necessarily the chunk producer, so the
    # primary is the IR that declares the ACTION output. Single-IR exports (ACT)
    # make that the only IR.
    primary = None
    for xml in IRS:
        try:
            m = make_core().read_model(xml)
            if any("action" in o.get_any_name().lower() for o in m.outputs):
                primary = xml
                break
        except Exception:
            continue
    if primary is None:
        primary = IRS[-1]

    for xml in IRS:
        label = ROLES.get(str(xml)) or xml.stem
        D["irs"][xml.name] = bench_ir(xml, device, label, xml == primary)
    D["primary_ir"] = primary.name
    D["end_to_end"] = bench_end_to_end(device)
    return D

def diagnose(dev, msg):
    """Turn an OpenVINO device failure into the actual fix."""
    m = msg.lower()
    if "libopencl" in m:
        return ("the iGPU plugin needs the OpenCL loader:\n"
                "       sudo apt install -y ocl-icd-libopencl1 intel-opencl-icd")
    if "out of" in m and ("memory" in m or "host memory" in m) or "alloc" in m and "fail" in m:
        return ("the iGPU ran out of memory compiling this model. The iGPU shares\n"
                "       system DRAM, and one buffer cannot exceed MAX_MEM_ALLOC_SIZE\n"
                "       printed above. Options: raise the GPU aperture in BIOS, close\n"
                "       other GPU users, or benchmark this model on CPU instead.")
    if "compiler_loader" in m or "vcl compiler" in m:
        return ("the NPU plugin loaded, but its compiler library is absent.\n"
                "       Install Intel's NPU userspace from:\n"
                "         https://github.com/intel/linux-npu-driver/releases\n"
                "       (intel-driver-compiler-npu + intel-level-zero-npu + level-zero)")
    if "libze" in m or "level zero" in m or "level-zero" in m:
        return ("missing Level Zero runtime:  sudo apt install -y libze1 libze-intel-gpu1")
    if "permission denied" in m and "/dev/" in m:
        return ("no access to the device node. Add yourself to the render/video groups:\n"
                "       sudo usermod -aG render,video $USER   (then log out and back in)")
    if "not available" in m or "cannot find" in m:
        return (f"OpenVINO does not see {dev}. Check drivers and that the plugin "
                f"loads; available = {core.available_devices}")
    return None

for dev in DEVICES:
    try:
        RESULTS["devices"][dev] = bench_device(dev)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        hr(f"DEVICE: {dev} - FAILED")
        print(f"  {msg[:900]}")
        fix = diagnose(dev, msg)
        if fix:
            print(f"\n  FIX: {fix}")
        print(f"\n  OpenVINO available_devices = {core.available_devices}")
        print("  Continuing with the remaining devices.")
        RESULTS["devices"][dev] = {"error": msg, "fix": fix,
                                   "available": core.available_devices}

# ================================================================== memory
hr("7. MEMORY")
try:
    status = Path("/proc/self/status").read_text()
    hwm = re.search(r"VmHWM:\s*(\d+) kB", status)
    rss = re.search(r"VmRSS:\s*(\d+) kB", status)
    MEM = {"peak_rss_mb": round(int(hwm.group(1)) / 1024, 1) if hwm else None,
           "current_rss_mb": round(int(rss.group(1)) / 1024, 1) if rss else None,
           "weights_mb": round(total_bin / 1e6, 1)}
    print(f"  peak RSS (VmHWM)  {MEM['peak_rss_mb']} MB")
    print(f"  current RSS       {MEM['current_rss_mb']} MB")
    print(f"  weights on disk   {MEM['weights_mb']} MB")
    print("  NOTE: peak covers the whole process incl. every device and IR compiled")
    print("        above, so it is an upper bound, not per-device cost. On GPU the")
    print("        weights also occupy shared system DRAM through the driver, which")
    print("        this counter does not separate out.")
    RESULTS["memory"] = MEM
except Exception as exc:
    print(f"  SKIPPED: {exc}")

# ================================================================== summary
hr("SUMMARY (one row per device x IR)")
hdr = (f"  {'device':<7} {'ir':<14} {'p50':>8} {'p95':>8} {'p99':>8} {'inf/s':>7} "
       f"{'act/s':>8} {'duty%':>7} {'cold':>9} {'warm':>9} {'mJ/inf':>9} {'drop%':>6} {'maxC':>5}")
print(hdr)
print(f"  {'':<7} {'':<14} {'ms':>8} {'ms':>8} {'ms':>8} {'':>7} {'':>8} "
      f"{'@' + format(CONTROL_FPS, 'g') + 'Hz':>7} {'ms':>9} {'ms':>9} {'':>9} {'':>6} {'':>5}")
print("  " + "-" * (len(hdr) - 2))
rows = []
for dev, D in RESULTS["devices"].items():
    if "error" in D:
        print(f"  {dev:<7} {'unavailable: ' + D['error'][:60]}")
        continue
    for irname, R in D["irs"].items():
        lat = R["steady_state"]["latency"]
        tp = R.get("throughput", {}).get("stats", {})
        cs = R["cold_start"]
        sus = R.get("sustained", {}).get("stats", {})
        ch = lat.get("chunk") or {}
        row = {
            "device": dev, "ir": irname, "role": R.get("role") or "",
            "primary": irname == D.get("primary_ir"),
            "precision": R.get("inference_precision", "unknown"),
            "execution_devices": ",".join(R.get("execution_devices") or []) or None,
            "p50_ms": round(lat["p50_ms"], 2), "p95_ms": round(lat["p95_ms"], 2),
            "p99_ms": round(lat["p99_ms"], 2),
            "seq_fps": round(lat["throughput_fps"], 3),
            "peak_fps": round(tp.get("throughput_fps", float("nan")), 3),
            "actions_per_s": ch.get("actions_per_s_p50"),
            "duty_p99_pct": ch.get("duty_p99_pct"),
            "max_control_fps_p99": ch.get("max_control_fps_p99"),
            "realtime_p99": ch.get("realtime_p99"),
            "compile_cold_ms": round(cs["cold_cache"]["compile_ms"], 1),
            "compile_warm_ms": round(cs["warm_cache"]["compile_ms"], 1),
            "first_infer_ms": round(cs["cold_cache"]["first_infer_ms"], 1),
            "mj_per_inference": lat.get("energy", {}).get("mj_per_inference"),
            "sustained_drop_pct": round(sus["throughput_drop_pct"], 1) if "throughput_drop_pct" in sus else None,
            "peak_temp_c": R["steady_state"]["sampler"].get("temp_max_c"),
            "peak_gpu_temp_c": R["steady_state"]["sampler"].get("gpu_temp_max_c"),
            "e2e_p50_ms": round(D.get("end_to_end", {}).get("stats", {}).get("p50_ms", float("nan")), 2),
        }
        rows.append(row)
        def f(v, w, p=2):
            return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) and v == v else f"{'n/a':>{w}}"
        print(f"  {dev:<7} {irname[:14]:<14} {f(row['p50_ms'],8)} {f(row['p95_ms'],8)} "
              f"{f(row['p99_ms'],8)} {f(row['seq_fps'],7,2)} {f(row['actions_per_s'],8,1)} "
              f"{f(row['duty_p99_pct'],7,1)} {f(row['compile_cold_ms'],9,1)} "
              f"{f(row['compile_warm_ms'],9,1)} {f(row['mj_per_inference'],9,1)} "
              f"{f(row['sustained_drop_pct'],6,1)} {f(row['peak_temp_c'],5,1)}")
RESULTS["summary_rows"] = rows

# The verdict line: can this policy drive the robot at the configured rate?
print()
for r in rows:
    if not r.get("primary") or r.get("realtime_p99") is None:
        continue
    if r["realtime_p99"]:
        print(f"  {r['device']}: REAL-TIME. One chunk every "
              f"{CHUNK / CONTROL_FPS:.2f} s is needed; p99 is {r['p99_ms'] / 1e3:.2f} s "
              f"({r['duty_p99_pct']:.1f}% of the budget). Control ceiling "
              f"{r['max_control_fps_p99']:.0f} Hz.")
    else:
        print(f"  {r['device']}: TOO SLOW at {CONTROL_FPS:g} Hz. p99 is "
              f"{r['p99_ms'] / 1e3:.2f} s but the budget is {CHUNK / CONTROL_FPS:.2f} s; "
              f"the action queue starves. Highest sustainable control rate is "
              f"{r['max_control_fps_p99']:.1f} Hz.")

e2e_shown = False
for dev, D in RESULTS["devices"].items():
    st = (D.get("end_to_end") or {}).get("stats")
    if st:
        e2e_shown = True
        pr = next((r for r in rows if r["device"] == dev and r.get("primary")), None)
        if pr:
            overhead = st["p50_ms"] - pr["p50_ms"]
            print(f"  {dev}: preprocessing + postprocessing adds {overhead:.1f} ms "
                  f"({overhead / st['p50_ms'] * 100:.1f}% of the end-to-end chunk)")
if not e2e_shown:
    print("  end-to-end InferenceModel numbers unavailable (see section 5); the")
    print("  per-IR numbers exclude the CPU-side resize and normalisation.")

if not ENERGY_VALID:
    print(f"\n  mJ/inf is n/a: energy source '{ENERGY_SOURCE}' unusable "
          f"({'on AC' if ON_AC else 'no counter'}).")
    print("  Unplug the charger, or run with sudo to read RAPL package energy.")

# plot_benchmark.py reads cold_start and the steady-state keys directly at
# devices[<name>]. The nested device -> IR tree moves to "devices_nested" and
# "devices" is rewritten flat, one entry per device x IR, so the existing
# plotting script works on these results unchanged. With a single IR (ACT) the
# flat key is just the device name.
flat = {}
for dev, D in RESULTS["devices"].items():
    if "error" in D:
        flat[dev] = D
        continue
    for irname, R in D["irs"].items():
        key = dev if len(D["irs"]) == 1 else f"{dev}/{Path(irname).stem}"
        flat[key] = dict(R, end_to_end=D.get("end_to_end", {}))
RESULTS["devices_nested"] = RESULTS["devices"]
RESULTS["devices"] = flat

out = Path(OUT_PATH) if OUT_PATH else Path(f"/tmp/vla_bench_{POLICY}_{time.strftime('%Y%m%d_%H%M%S')}.json")
# Keyed by the flat device x IR label, which is also the "device" column the
# sustained CSV and its plot use as the series name.
RESULTS["sustained_buckets_index"] = {
    key: R.get("sustained", {}).get("stats", {}).get("buckets", [])
    for key, R in RESULTS["devices"].items() if "error" not in R
}
out.write_text(json.dumps(RESULTS, indent=2, default=str))

# ------------------------------------------------------------ plot data (CSV)
# Written next to the JSON, same stem and same columns plot_benchmark.py expects,
# so the PatchCore plotting script works on these results unchanged.
import csv

stem = out.with_suffix("")
written = [out]

def write_csv(path, fieldnames, rows_):
    if not rows_:
        return None
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_)
    written.append(path)
    return path

# 1. Power / temperature traces, one row per 100 ms sample.
trace_rows, trace_fields = [], ["phase", "device", "t_s", "temp_c", "gpu_temp_c", "battery_w"]
rapl_cols = set()
for t in TRACES:
    for r in t["rows"]:
        rapl_cols.update(k for k in r if k.startswith("rapl_w["))
        trace_rows.append({"phase": t["phase"], "device": t["device"], **r})
trace_fields += sorted(rapl_cols)
write_csv(f"{stem}_traces.csv", trace_fields, trace_rows)

# 2. Raw per-inference latencies, one row per inference (for histograms / CDFs).
lat_rows = []
for L in LATENCIES:
    for i, ms in enumerate(L["ms"]):
        lat_rows.append({"phase": L["phase"], "device": L["device"], "i": i, "ms": ms})
write_csv(f"{stem}_latencies.csv", ["phase", "device", "i", "ms"], lat_rows)

# 3. Sustained-load buckets (throughput and temperature vs time).
bucket_rows = []
for key, buckets in RESULTS["sustained_buckets_index"].items():
    for b in buckets:
        bucket_rows.append({"device": key, **b})
write_csv(f"{stem}_sustained.csv",
          ["device", "t_s", "n", "fps", "p50_ms", "p99_ms", "temp_c", "gpu_temp_c"], bucket_rows)

# 4. Summary rows, one per device x IR.
if RESULTS.get("summary_rows"):
    write_csv(f"{stem}_summary.csv", list(RESULTS["summary_rows"][0].keys()),
              RESULTS["summary_rows"])

hr("OUTPUT FILES")
for f_ in written:
    pth = Path(f_)
    print(f"  {pth}  ({pth.stat().st_size / 1024:.1f} KB)")
print("\n  Plot them with:")
print(f"    python3 benchmark/plot_benchmark.py {out}")

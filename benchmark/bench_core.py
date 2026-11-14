#!/usr/bin/env python3
"""PatchCore inference benchmark - performance only, no accuracy.

Runs the exported OpenVINO IR in anomaly_detection/models/patchcore and reports
latency percentiles, throughput, cold-start breakdown, energy per inference,
thermals, sustained throughput and memory.

Input is synthetic by default (random noise at the model's 256x256 input), because
the goal is silicon inference speed, not camera bandwidth. --camera adds a separate
section that measures the camera path on its own so the two never get confused.

Env vars (all optional): BENCH_ITERS BENCH_WARMUP BENCH_SUSTAIN BENCH_DEVICES
BENCH_MODEL BENCH_REPO_SRC BENCH_CAMERA BENCH_IDLE BENCH_OUT
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

ITERS     = _env_int("BENCH_ITERS", 300)
WARMUP    = _env_int("BENCH_WARMUP", 50)
SUSTAIN   = _env_int("BENCH_SUSTAIN", 600)
IDLE_SECS = _env_int("BENCH_IDLE", 30)
MODEL_DIR = Path(os.environ.get("BENCH_MODEL", ".")).expanduser().resolve()
# Friendly names -> OpenVINO device names. "igpu" is what people call the
# integrated GPU; OpenVINO just calls it "GPU" (GPU.0 is the integrated one when
# no discrete card is present).
DEVICE_ALIAS = {"CPU": "CPU", "IGPU": "GPU", "GPU": "GPU", "NPU": "NPU",
                "AUTO": "AUTO", "DGPU": "GPU.1"}
DEVICES = []
for _d in os.environ.get("BENCH_DEVICES", "CPU").split(","):
    _d = _d.strip().upper()
    if _d:
        DEVICES.append(DEVICE_ALIAS.get(_d, _d))
CAMERA    = os.environ.get("BENCH_CAMERA", "").strip()
OUT_PATH  = os.environ.get("BENCH_OUT", "").strip()

# OpenVINO's model cache must be writable. The repo's model dir may contain a
# root-owned ov_cache/ from an earlier sudo run, which makes compile_model fail
# with "cannot remove ... Permission denied", so always cache in our own tmp dir.
CACHE_DIR = Path(f"/tmp/patchcore_bench_ovcache_{os.getuid()}")

BAR = "=" * 74

# Collected for the plotting script: power/temperature traces and raw latencies.
TRACES: list[dict] = []   # {phase, device, rows:[{t_s, temp_c, battery_w, rapl_w[..]}]}
LATENCIES: list[dict] = []  # {phase, device, ms:[...]}

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
        self.watts: list[float] = []
        # Timestamped traces (seconds since sampler start) for plotting.
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

def print_latency(st, wall):
    print(f"  latency (ms)   min={st['min_ms']:7.2f}  p50={st['p50_ms']:7.2f}  p90={st['p90_ms']:7.2f}")
    print(f"                 p95={st['p95_ms']:7.2f}  p99={st['p99_ms']:7.2f}  max={st['max_ms']:7.2f}")
    print(f"                 mean={st['mean_ms']:6.2f}  stdev={st['stdev_ms']:6.2f}")
    if wall:
        print(f"  throughput     {st['n'] / wall:.2f} inferences/s (sequential, batch 1)")
    if st["p50_ms"]:
        j = st["p99_ms"] / st["p50_ms"]
        print(f"  jitter         p99/p50 = {j:.2f}x "
              f"({'noisy - scheduling or thermal' if j > 1.5 else 'tight'})")

# ============================================================== environment
hr("0. ENVIRONMENT")
try:
    import numpy as np
    import openvino as ov
except ImportError as exc:
    print(f"FATAL: missing dependency: {exc}")
    print("Activate the project venv first (.venv/bin/activate).")
    sys.exit(1)

try:
    import cv2
except ImportError:
    cv2 = None

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
    "opencv_version": getattr(cv2, "__version__", "missing"),
    "python": sys.version.split()[0],
    "ov_available_devices": core.available_devices,
    "requested_devices": DEVICES,
    "cpu_governor": uniq_sysfs("/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor"),
    "energy_perf_pref": uniq_sysfs("/sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference"),
    "ac_online": ON_AC,
    "battery_status": (Path(BATTERY["status"]).read_text().strip() if BATTERY.get("status") else "n/a"),
    "running_as_root": os.geteuid() == 0,
    "model_dir": str(MODEL_DIR),
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

print("\n  Sensors:")
print(f"    energy source            {ENERGY_SOURCE} ({'usable' if ENERGY_VALID else 'NOT usable'})")
print(f"    RAPL readable            {list(RAPL) or 'none'}")
if RAPL_BAD:
    print(f"    RAPL root-only           {RAPL_BAD} counters (re-run with sudo for package watts)")
print(f"    temp sensors             {len(TEMPS)} found, primary = {PRIMARY_TEMP}")
if ENERGY_SOURCE == "battery" and ON_AC:
    print("    NOTE: on AC the battery counter measures charging, not consumption.")
    print("          Energy is reported as unavailable. For mJ/inference either")
    print("          unplug the charger, or re-run with sudo to read RAPL.")

RESULTS = {"environment": ENV,
           "sensors": {"energy_source": ENERGY_SOURCE, "energy_valid": ENERGY_VALID,
                       "rapl_readable": list(RAPL), "rapl_root_only": RAPL_BAD,
                       "primary_temp": PRIMARY_TEMP, "temp_count": len(TEMPS)},
           "devices": {}}

# ============================================================== model loading
xml = MODEL_DIR / "model.xml"
if not xml.exists():
    print(f"\nFATAL: {xml} not found")
    sys.exit(1)
meta = json.loads((MODEL_DIR / "metadata.json").read_text())
IMG = meta["image_size"]
ROI = meta.get("roi")

hr("MODEL")
print(f"  algorithm        {meta['model']} / {meta['backbone']} (layers {meta['layers']})")
print(f"  input            1x3x{IMG}x{IMG} FP32")
print(f"  model.bin        {(MODEL_DIR / 'model.bin').stat().st_size / 1e6:.1f} MB "
      f"(coreset ratio {meta.get('coreset_sampling_ratio')} - the memory bank dominates this)")
print(f"  training ROI     {ROI}  (x, y, w, h in the full camera frame)")
print(f"  ov cache dir     {CACHE_DIR}")
print("\n  The IR contains normalisation + backbone + coreset nearest-neighbour search,")
print("  so one infer() call is the whole anomaly pipeline. Outputs: pred_score,")
print("  pred_label, anomaly_map (256x256), pred_mask.")

CACHE_DIR.mkdir(parents=True, exist_ok=True)

def make_core(cache=True):
    c = ov.Core()
    if cache:
        c.set_property({"CACHE_DIR": str(CACHE_DIR)})
    return c

def load(device, cache=True, config=None):
    c = make_core(cache)
    m = c.read_model(xml)
    if m.input().get_partial_shape().is_dynamic:
        m.reshape([1, 3, IMG, IMG])
    return c.compile_model(m, device, config or {})

rng = np.random.default_rng(0)
# Synthetic input: random noise in [0,1), the same range/layout the real pipeline
# feeds the model (RGB float32, CHW, /255). PatchCore's cost is data-independent
# (fixed convolutions + a fixed-size coreset search), so noise times identically
# to a real brick. A pool of 16 avoids measuring a single cached buffer.
INPUTS = [rng.random((1, 3, IMG, IMG), dtype=np.float32) for _ in range(16)]

# ============================================================= idle baseline
# Measured FIRST, while the machine is cold. Measuring it after the load runs
# captures fans still spinning down and a hot package, which is what produced the
# nonsensical negative marginal power in the earlier run.
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
    IDLE["sampler"] = sm
else:
    print("  skipped (BENCH_IDLE=0)")
RESULTS["idle"] = IDLE

# ========================================================== per-device suite
def bench_device(device):
    """Full performance suite for one OpenVINO device."""
    D = {}
    hr(f"DEVICE: {device}")

    # --- cold start: read / compile / first infer, separately -------------
    print("  2. COLD START")
    def cold_start(label, cache, clear):
        if clear and CACHE_DIR.exists():
            for f in CACHE_DIR.rglob("*"):
                if f.is_file():
                    try:
                        f.unlink()
                    except OSError:
                        pass
        c = make_core(cache)
        t0 = time.perf_counter(); m = c.read_model(xml); t1 = time.perf_counter()
        if m.input().get_partial_shape().is_dynamic:
            m.reshape([1, 3, IMG, IMG])
        t2 = time.perf_counter(); cm = c.compile_model(m, device); t3 = time.perf_counter()
        dummy = np.zeros((1, 3, IMG, IMG), dtype=np.float32)
        t4 = time.perf_counter(); cm(dummy); t5 = time.perf_counter()
        d = {"read_ms": (t1 - t0) * 1e3, "compile_ms": (t3 - t2) * 1e3,
             "first_infer_ms": (t5 - t4) * 1e3}
        d["total_ms"] = sum(d.values())
        print(f"     {label:26s} read={d['read_ms']:7.1f}  compile={d['compile_ms']:8.1f}  "
              f"first_infer={d['first_infer_ms']:7.1f}  total={d['total_ms']:8.1f}  (ms)")
        del cm
        return d

    D["cold_start"] = {
        "cold_cache": cold_start("cold cache (first ever)", True, True),
        "warm_cache": cold_start("warm cache (cached blob)", True, False),
        "no_cache":   cold_start("no cache", False, False),
    }
    saved = D["cold_start"]["cold_cache"]["compile_ms"] - D["cold_start"]["warm_cache"]["compile_ms"]
    print(f"     -> blob cache saves {saved:.0f} ms of compile time on {device}")

    # --- steady-state latency, batch 1 -----------------------------------
    print("\n  3. STEADY-STATE LATENCY (batch 1, LATENCY hint)")
    compiled = load(device, config={"PERFORMANCE_HINT": "LATENCY"})
    req = compiled.create_infer_request()
    props = ["EXECUTION_DEVICES", "NUM_STREAMS", "OPTIMAL_NUMBER_OF_INFER_REQUESTS"]
    if device.startswith("CPU"):
        props += ["INFERENCE_NUM_THREADS", "INFERENCE_PRECISION_HINT"]
    elif device.startswith("GPU"):
        props += ["INFERENCE_PRECISION_HINT", "GPU_HOST_TASK_PRIORITY"]
    elif device.startswith("NPU"):
        props += ["NPU_COMPILATION_MODE_PARAMS", "NPU_TILES", "INFERENCE_PRECISION_HINT"]
    for prop in props:
        try:
            print(f"     {prop:32s} {compiled.get_property(prop)}")
        except Exception:
            pass
    # EXECUTION_DEVICES is the fallback check: if a graph cannot run entirely on
    # the target (PatchCore's coreset search is not a standard neural op), part
    # of it silently runs on CPU and caps the speedup.
    try:
        ex = compiled.get_property("EXECUTION_DEVICES")
        ex_list = [ex] if isinstance(ex, str) else list(ex)
        if not any(str(e).upper().startswith(device.split(".")[0]) for e in ex_list):
            print(f"     !! graph did NOT run on {device}: actually on {ex_list}")
        elif len(ex_list) > 1:
            print(f"     !! graph SPLIT across {ex_list} - partial fallback, speedup is capped")
    except Exception:
        pass

    for i in range(WARMUP):
        req.infer(INPUTS[i % len(INPUTS)])
    s = Sampler(); s.start()
    lat = []
    t0 = time.perf_counter()
    for i in range(ITERS):
        t = time.perf_counter()
        req.infer(INPUTS[i % len(INPUTS)])
        lat.append((time.perf_counter() - t) * 1e3)
    wall = time.perf_counter() - t0
    s.stop(); collect_trace(s, "steady_state", device); sm = s.summary()
    collect_latency(lat, "steady_state", device)
    st = percentiles(lat); st["wall_s"] = round(wall, 3); st["throughput_fps"] = len(lat) / wall
    print_latency(st, wall)
    en, src = fmt_energy(sm, len(lat), wall)
    if en:
        st["energy"] = en
        print(f"  energy         {en['avg_watts']:.2f} W avg, {en['mj_per_inference']:.1f} mJ/inference  [{src}]")
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
    try:
        D["inference_precision"] = str(compiled.get_property("INFERENCE_PRECISION_HINT"))
    except Exception:
        D["inference_precision"] = "unknown"
    try:
        ex = compiled.get_property("EXECUTION_DEVICES")
        D["execution_devices"] = [ex] if isinstance(ex, str) else [str(e) for e in ex]
    except Exception:
        D["execution_devices"] = None
    D["steady_state"] = {"latency": st, "sampler": sm}
    seq_fps = st["throughput_fps"]

    # --- peak throughput, async ------------------------------------------
    print("\n  4. PEAK THROUGHPUT (async queue, THROUGHPUT hint)")
    try:
        tc = load(device, config={"PERFORMANCE_HINT": "THROUGHPUT"})
        njobs = tc.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        try:
            nstreams = tc.get_property("NUM_STREAMS")
        except Exception:
            nstreams = "?"
        print(f"     parallel requests {njobs}, streams {nstreams}")
        q = ov.AsyncInferQueue(tc, njobs)
        for i in range(min(WARMUP, 32)):
            q.start_async(INPUTS[i % len(INPUTS)])
        q.wait_all()
        s = Sampler(); s.start()
        t0 = time.perf_counter()
        for i in range(ITERS):
            q.start_async(INPUTS[i % len(INPUTS)])
        q.wait_all()
        w = time.perf_counter() - t0
        s.stop(); collect_trace(s, "throughput", device); sm2 = s.summary()
        TP = {"n": ITERS, "wall_s": round(w, 3), "throughput_fps": ITERS / w,
              "n_requests": njobs}
        print(f"  throughput     {TP['throughput_fps']:.2f} inferences/s "
              f"({TP['throughput_fps'] / seq_fps:.2f}x the batch-1 rate)")
        en2, src2 = fmt_energy(sm2, ITERS, w)
        if en2:
            TP["energy"] = en2
            print(f"  energy         {en2['avg_watts']:.2f} W avg, {en2['mj_per_inference']:.1f} mJ/inference  [{src2}]")
        if "temp_max_c" in sm2:
            print(f"  temperature    max={sm2['temp_max_c']}C")
        D["throughput"] = {"stats": TP, "sampler": sm2}
        del q, tc
    except Exception as exc:
        print(f"     SKIPPED: {exc}")
        D["throughput"] = {"error": str(exc)}

    # --- end-to-end predict() --------------------------------------------
    print("\n  5. END-TO-END BrickInspector.predict() (preprocess + presence + infer)")
    try:
        repo_src = os.environ.get("BENCH_REPO_SRC", "")
        if repo_src:
            p = str(Path(repo_src).expanduser().resolve())
            if p not in sys.path:
                sys.path.insert(0, p)
        from hackjunction2026_intel.anomaly_detection import BrickInspector

        # The inspector writes its cache into the model dir when that dir is
        # writable; a root-owned ov_cache/ there breaks compile. Point HOME-free
        # OpenVINO caching at our tmp dir by pre-creating a writable cache and
        # tolerating failure.
        insp = BrickInspector(MODEL_DIR, device=device)
        if ROI:
            x, y, w_, h_ = ROI
            frame = rng.integers(0, 255, (y + h_ + 40, x + w_ + 40, 3), dtype=np.uint8)
        else:
            frame = rng.integers(0, 255, (960, 1280, 3), dtype=np.uint8)
        for _ in range(min(WARMUP, 30)):
            insp.predict(frame, check_presence=False)
        s = Sampler(); s.start()
        e2e = []
        t0 = time.perf_counter()
        for _ in range(ITERS):
            t = time.perf_counter()
            insp.predict(frame, check_presence=False)
            e2e.append((time.perf_counter() - t) * 1e3)
        w = time.perf_counter() - t0
        s.stop(); collect_trace(s, "end_to_end", device); sm3 = s.summary()
        collect_latency(e2e, "end_to_end", device)
        E = percentiles(e2e); E["throughput_fps"] = len(e2e) / w
        print(f"     latency (ms)  p50={E['p50_ms']:7.2f}  p95={E['p95_ms']:7.2f}  "
              f"p99={E['p99_ms']:7.2f}  max={E['max_ms']:7.2f}")
        print(f"     throughput    {E['throughput_fps']:.2f} frames/s")
        ov_head = E["p50_ms"] - st["p50_ms"]
        E["preprocess_overhead_ms"] = round(ov_head, 2)
        print(f"     -> preprocess+presence overhead = {ov_head:.2f} ms "
              f"({ov_head / E['p50_ms'] * 100:.1f}% of the frame)")
        if insp.presence is not None:
            pre = []
            for _ in range(min(ITERS, 200)):
                t = time.perf_counter(); insp.is_empty(frame); pre.append((time.perf_counter() - t) * 1e3)
            E["presence_only_p50_ms"] = round(statistics.median(pre), 3)
            print(f"     presence-only p50={E['presence_only_p50_ms']:.3f} ms "
                  f"(empty-tray fast path, skips the model)")
        D["end_to_end"] = {"stats": E, "sampler": sm3}
    except Exception as exc:
        print(f"     SKIPPED: {type(exc).__name__}: {exc}")
        D["end_to_end"] = {"error": f"{type(exc).__name__}: {exc}"}

    # --- sustained load ---------------------------------------------------
    print(f"\n  6. SUSTAINED LOAD ({SUSTAIN} s) - thermal throttling")
    if SUSTAIN <= 0:
        print("     skipped (BENCH_SUSTAIN=0)")
        D["sustained"] = {"skipped": True}
    else:
        print(f"     running {SUSTAIN} s of continuous batch-1 inference, 10 s buckets ...", flush=True)
        s = Sampler(); s.start()
        buckets, bucket = [], []
        t0 = bstart = time.perf_counter()
        i = 0
        while time.perf_counter() - t0 < SUSTAIN:
            t = time.perf_counter()
            req.infer(INPUTS[i % len(INPUTS)])
            bucket.append((time.perf_counter() - t) * 1e3)
            i += 1
            if time.perf_counter() - bstart >= 10.0:
                dur = time.perf_counter() - bstart
                bs = sorted(bucket)
                tc_now = (read_int(TEMPS[PRIMARY_TEMP]) / 1000.0) if PRIMARY_TEMP else None
                buckets.append({"t_s": round(time.perf_counter() - t0, 1), "n": len(bucket),
                                "fps": len(bucket) / dur,
                                "p50_ms": statistics.median(bs),
                                "p99_ms": bs[min(len(bs) - 1, int(len(bs) * 0.99))],
                                "temp_c": tc_now})
                bucket, bstart = [], time.perf_counter()
        w = time.perf_counter() - t0
        s.stop(); collect_trace(s, "sustained", device); sm4 = s.summary()
        print(f"\n     {'t(s)':>6} {'fps':>8} {'p50(ms)':>9} {'p99(ms)':>9} {'temp(C)':>8}")
        for b in buckets:
            tstr = f"{b['temp_c']:.1f}" if b["temp_c"] is not None else "n/a"
            print(f"     {b['t_s']:>6.0f} {b['fps']:>8.2f} {b['p50_ms']:>9.2f} "
                  f"{b['p99_ms']:>9.2f} {tstr:>8}")
        SUS = {"wall_s": round(w, 1), "total_inferences": i, "buckets": buckets}
        if len(buckets) >= 2:
            f0, fl = buckets[0], buckets[-1]
            drop = (1 - fl["fps"] / f0["fps"]) * 100
            SUS.update(first_fps=f0["fps"], last_fps=fl["fps"], throughput_drop_pct=drop)
            print(f"\n     burst    (0-10s)  {f0['fps']:.2f} fps @ {f0['temp_c']}C")
            print(f"     sustained (last)  {fl['fps']:.2f} fps @ {fl['temp_c']}C")
            print(f"     -> drop {drop:.1f}%  "
                  f"{'THROTTLING' if drop > 5 else 'stable'}")
        en4, src4 = fmt_energy(sm4, max(i, 1), w)
        if en4:
            SUS["energy"] = en4
            print(f"     energy   {en4['avg_watts']:.2f} W avg, {en4['mj_per_inference']:.1f} mJ/inference  [{src4}]")
        if "temp_max_c" in sm4:
            print(f"     temps    start={sm4['temp_start_c']}C max={sm4['temp_max_c']}C end={sm4['temp_end_c']}C")
        D["sustained"] = {"stats": SUS, "sampler": sm4}

    return D

def diagnose(dev, msg):
    """Turn an OpenVINO device failure into the actual fix."""
    m = msg.lower()
    if "libopencl" in m:
        return ("the iGPU plugin needs the OpenCL loader:\n"
                "       sudo apt install -y ocl-icd-libopencl1 intel-opencl-icd")
    if "compiler_loader" in m or "vcl compiler" in m:
        return ("the NPU plugin loaded, but its compiler library is absent.\n"
                "       The wheel ships libopenvino_intel_npu_plugin.so without the\n"
                "       matching compiler_loader, and no Level Zero runtime is installed.\n"
                "       Install Intel's NPU userspace from:\n"
                "         https://github.com/intel/linux-npu-driver/releases\n"
                "       (intel-driver-compiler-npu + intel-level-zero-npu + level-zero)\n"
                "       The ivpu kernel module and /dev/accel/accel0 are already fine.")
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
        print(f"  {msg[:700]}")
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
           "model_bin_mb": round((MODEL_DIR / "model.bin").stat().st_size / 1e6, 1)}
    print(f"  peak RSS (VmHWM)  {MEM['peak_rss_mb']} MB")
    print(f"  current RSS       {MEM['current_rss_mb']} MB")
    print(f"  model.bin on disk {MEM['model_bin_mb']} MB")
    print("  NOTE: peak covers the whole process incl. every device compiled above,")
    print("        so it is an upper bound, not per-device cost.")
    RESULTS["memory"] = MEM
except Exception as exc:
    print(f"  SKIPPED: {exc}")

# ================================================================== camera
hr("8. CAMERA (optional, measured separately from inference)")
if not CAMERA:
    print("  skipped (no --camera). Inference numbers above used synthetic noise,")
    print("  which is what you want for a silicon benchmark: no camera bandwidth in the loop.")
    RESULTS["camera"] = {"skipped": True}
elif cv2 is None:
    print("  skipped: opencv (cv2) not importable")
    RESULTS["camera"] = {"error": "no cv2"}
else:
    print(f"  requested: {CAMERA}")
    print("  V4L2 note: a camera is a /dev/video* node. The long USB string is a")
    print("  stable by-id path; we resolve it to an index/path and open it with V4L2.\n")
    cam = None
    try:
        src = int(CAMERA) if CAMERA.isdigit() else CAMERA
        cam = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not cam.isOpened():
            print(f"  CAMERA UNAVAILABLE: cannot open {CAMERA!r}.")
            print("  Most likely another process holds it (Physical AI Studio, a browser tab,")
            print("  or a previous run). This is fine - inference results above are unaffected.")
            RESULTS["camera"] = {"error": "cannot open", "fatal": False}
        else:
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
            cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cam.read()
            if not ok or frame is None:
                print("  CAMERA UNAVAILABLE: opened but returned no frame. Skipping.")
                RESULTS["camera"] = {"error": "no frame", "fatal": False}
            else:
                h, w = frame.shape[:2]
                fourcc = int(cam.get(cv2.CAP_PROP_FOURCC))
                cc = "".join(chr((fourcc >> (8 * k)) & 0xFF) for k in range(4))
                print(f"  opened: {w}x{h} fourcc={cc!r} driver_fps={cam.get(cv2.CAP_PROP_FPS)}")
                for _ in range(5):
                    cam.read()
                grab = []
                n = min(60, max(20, ITERS // 5))
                for _ in range(n):
                    t = time.perf_counter()
                    ok, _f = cam.read()
                    if ok:
                        grab.append((time.perf_counter() - t) * 1e3)
                if grab:
                    G = percentiles(grab)
                    print(f"  frame grab    p50={G['p50_ms']:.2f} ms  p95={G['p95_ms']:.2f} ms  "
                          f"-> {1000 / G['p50_ms']:.1f} fps ceiling")
                    cpu_p50 = (RESULTS["devices"].get("CPU", {})
                               .get("steady_state", {}).get("latency", {}).get("p50_ms"))
                    if cpu_p50:
                        if G["p50_ms"] > cpu_p50:
                            print(f"  -> CAMERA-BOUND: grabbing ({G['p50_ms']:.1f} ms) costs more "
                                  f"than inference ({cpu_p50:.1f} ms).")
                        else:
                            print(f"  -> COMPUTE-BOUND: inference ({cpu_p50:.1f} ms) dominates "
                                  f"grabbing ({G['p50_ms']:.1f} ms).")
                    RESULTS["camera"] = {"stats": G, "resolution": [w, h], "fourcc": cc}
    except Exception as exc:
        print(f"  CAMERA UNAVAILABLE: {type(exc).__name__}: {exc} (continuing)")
        RESULTS["camera"] = {"error": str(exc), "fatal": False}
    finally:
        if cam is not None:
            cam.release()
            print("  camera released")

# ================================================================== summary
hr("SUMMARY (one row per device)")
hdr = (f"  {'device':<7} {'p50':>7} {'p95':>7} {'p99':>7} {'seq':>7} {'peak':>7} "
       f"{'cold':>8} {'warm':>8} {'mJ/inf':>8} {'drop%':>6} {'maxC':>5}")
print(hdr)
print(f"  {'':<7} {'ms':>7} {'ms':>7} {'ms':>7} {'fps':>7} {'fps':>7} "
      f"{'ms':>8} {'ms':>8} {'':>8} {'':>6} {'':>5}")
print("  " + "-" * (len(hdr) - 2))
rows = []
for dev, D in RESULTS["devices"].items():
    if "error" in D:
        print(f"  {dev:<7} {'unavailable: ' + D['error'][:50]}")
        continue
    lat = D["steady_state"]["latency"]
    tp = D.get("throughput", {}).get("stats", {})
    cs = D["cold_start"]
    sus = D.get("sustained", {}).get("stats", {})
    row = {
        "device": dev, "precision": D.get("inference_precision", "unknown"),
        "execution_devices": ",".join(D.get("execution_devices") or []) or None,
        "p50_ms": round(lat["p50_ms"], 2), "p95_ms": round(lat["p95_ms"], 2),
        "p99_ms": round(lat["p99_ms"], 2),
        "seq_fps": round(lat["throughput_fps"], 2),
        "peak_fps": round(tp.get("throughput_fps", float("nan")), 2),
        "compile_cold_ms": round(cs["cold_cache"]["compile_ms"], 1),
        "compile_warm_ms": round(cs["warm_cache"]["compile_ms"], 1),
        "first_infer_ms": round(cs["cold_cache"]["first_infer_ms"], 1),
        "mj_per_inference": lat.get("energy", {}).get("mj_per_inference"),
        "sustained_drop_pct": round(sus["throughput_drop_pct"], 1) if "throughput_drop_pct" in sus else None,
        "peak_temp_c": D["steady_state"]["sampler"].get("temp_max_c"),
        "e2e_p50_ms": round(D.get("end_to_end", {}).get("stats", {}).get("p50_ms", float("nan")), 2),
    }
    rows.append(row)
    def f(v, w, p=2):
        return f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) and v == v else f"{'n/a':>{w}}"
    print(f"  {dev:<7} {f(row['p50_ms'],7)} {f(row['p95_ms'],7)} {f(row['p99_ms'],7)} "
          f"{f(row['seq_fps'],7)} {f(row['peak_fps'],7)} {f(row['compile_cold_ms'],8,1)} "
          f"{f(row['compile_warm_ms'],8,1)} {f(row['mj_per_inference'],8,1)} "
          f"{f(row['sustained_drop_pct'],6,1)} {f(row['peak_temp_c'],5,1)}")
RESULTS["summary_rows"] = rows

if not ENERGY_VALID:
    print(f"\n  mJ/inf is n/a: energy source '{ENERGY_SOURCE}' unusable "
          f"({'on AC' if ON_AC else 'no counter'}).")
    print("  Unplug the charger, or run with sudo to read RAPL package energy.")

out = Path(OUT_PATH) if OUT_PATH else Path(f"/tmp/patchcore_bench_{time.strftime('%Y%m%d_%H%M%S')}.json")
RESULTS["sustained_buckets_index"] = {
    dev: D.get("sustained", {}).get("stats", {}).get("buckets", [])
    for dev, D in RESULTS["devices"].items() if "error" not in D
}
out.write_text(json.dumps(RESULTS, indent=2, default=str))

# ------------------------------------------------------------ plot data (CSV)
# Written next to the JSON, same stem. plot_benchmark.py reads these.
import csv

stem = out.with_suffix("")
written = [out]

def write_csv(path, fieldnames, rows):
    if not rows:
        return None
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    written.append(path)
    return path

# 1. Power / temperature traces, one row per 100 ms sample.
trace_rows, trace_fields = [], ["phase", "device", "t_s", "temp_c", "battery_w"]
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
for dev, buckets in RESULTS["sustained_buckets_index"].items():
    for b in buckets:
        bucket_rows.append({"device": dev, **b})
write_csv(f"{stem}_sustained.csv",
          ["device", "t_s", "n", "fps", "p50_ms", "p99_ms", "temp_c"], bucket_rows)

# 4. Summary rows, one per device.
if RESULTS.get("summary_rows"):
    write_csv(f"{stem}_summary.csv", list(RESULTS["summary_rows"][0].keys()),
              RESULTS["summary_rows"])

hr("OUTPUT FILES")
for f in written:
    pth = Path(f)
    print(f"  {pth}  ({pth.stat().st_size / 1024:.1f} KB)")
print("\n  Plot them with:")
print(f"    python3 benchmark/plot_benchmark.py {out}")

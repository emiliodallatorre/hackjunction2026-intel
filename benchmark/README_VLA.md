# VLA policy inference benchmark (ACT, π0.5)

Performance only — no task success. Measures what the silicon does with the
OpenVINO IR inside a Physical AI Studio export's `exports/openvino/`.

The sibling of `bench_core.py` (PatchCore), with the same sensors, the same JSON
+ CSV output and the same plotting script.

## Run it

```bash
# iGPU, the default — the interesting device for a VLA
./run_vla_benchmark.sh --export ~/Downloads/pi05-ricotta-runtime

./run_vla_benchmark.sh --export ~/Downloads/ACT.zip --quick      # ~1 min sanity check
./run_vla_benchmark.sh --export DIR --devices igpu,cpu --plot    # compare, with graphs
./run_vla_benchmark.sh --export DIR --sudo                       # RAPL package energy
./run_vla_benchmark.sh --local --export /tmp/actx                # run here, no ssh
```

`--export` takes the export root (the folder holding `runtime.yaml` and
`exports/openvino/`), the `exports/openvino` folder itself, or the `.zip` Studio
downloads — a zip is unpacked on the host, and only the IR is extracted, not the
600 MB `model.ckpt`. `--device` accepts `cpu`, `igpu` or `npu` (OpenVINO calls
the iGPU `GPU`). A device that fails prints the specific fix and the run
continues with the rest.

NPU is accepted but π0.5 is not expected to compile there; use the iGPU.

## The number that actually matters: action chunks

This is the one real difference from the PatchCore benchmark. ACT's manifest
declares its action output as shape `[100, 6]` — **one inference returns 100
actions** of 6 joints, which the runtime's action queue plays out one per control
tick. At 30 Hz control, those 100 actions last 3.33 seconds, so the policy only
has to run once every 3.33 s.

Raw latency alone is therefore misleading. Every section also reports:

- **actions/s** — chunk size divided by latency
- **duty %** — p50 and p99 latency as a fraction of the 3333 ms chunk budget
- **control ceiling** — the highest Hz the policy could feed at p99 latency
- a **REAL-TIME / TOO SLOW** verdict against `--fps` (default 30)

A measured example, ACT on an Arc iGPU: p99 = 27 ms against a 3333 ms budget,
0.8% duty, ceiling ~3700 Hz. The policy is nowhere near the bottleneck at 30 Hz;
π0.5 is far heavier and is where that margin gets spent.

## Why synthetic input

Inputs are random tensors at each IR port's declared shape and dtype, read off
the compiled model. A transformer policy has fixed shapes, no early exit and no
data-dependent branching, so noise times identically to a real camera frame, and
the numbers measure the chip rather than camera bandwidth.

Note the shapes: the manifest's `input_features` are the **pre**-preprocessing
sizes (ACT: 3×720×1280 and 3×600×800, the two cameras), while the IR itself takes
the letterbox-resized 3×384×384. The per-IR sections feed the IR directly, so
they exclude the CPU-side resize and normalisation; section 5 includes them.

## What it measures

| Section | Why it exists |
|---|---|
| Idle baseline | Measured **before** load, so marginal energy is honest |
| Cold start | `read_model` / `compile_model` / first inference, split. Sets how long the robot waits before its first action, not the control rate |
| Steady-state latency | p50/p95/p99 at batch 1, plus the action-chunk budget |
| Peak throughput | Async queue. Not the control-loop number, but shows whether the device is latency-bound or already saturated by one request |
| End-to-end `InferenceModel.infer()` | Whole policy incl. preprocessing — the real control-rate number. Skips with a printed reason if `physicalai` is not importable |
| Sustained load | Burst vs sustained inf/s and the sustained control ceiling — the laptop number |
| Memory | Peak RSS; on GPU the weights also sit in shared system DRAM |

Several IRs in one export (π0.5 may ship an encoder plus an action expert) are
each benchmarked separately, and the IR declaring the ACTION output is the one
the chunk-budget maths uses.

## Energy caveat

Same as the PatchCore benchmark. `mJ/inference` needs a readable counter:

- **RAPL** (`/sys/class/powercap/intel-rapl/`) is exact but usually root-only →
  use `--sudo`.
- **Battery** (`power_now`) works only while **unplugged**. On AC it measures
  charging, so the script reports energy as unavailable instead of printing a
  wrong number.

With a chunk size, energy is also reported per action (mJ/inference ÷ 100).

## Output

JSON + CSVs land in `benchmark/results/`, graphs in
`benchmark/results/plots_<name>/`. The CSVs use the same columns
`plot_benchmark.py` already expects, so the PatchCore plotting script works on
these results unchanged:

```bash
python3 benchmark/plot_benchmark.py benchmark/results/vla_<stamp>.json
```

In the JSON, `devices` is flat (one entry per device × IR, which is what the
plotter reads) and `devices_nested` keeps the device → IR tree.

## Files

- `bench_vla.py` — runs on the laptop, writes JSON + CSVs
- `run_vla_benchmark.sh` — copies it over SSH, runs it in the venv, fetches results
- `plot_benchmark.py` — shared with the PatchCore benchmark

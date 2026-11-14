# PatchCore inference benchmark

Performance only — no accuracy. Measures what the silicon does with the exported
OpenVINO IR in `anomaly_detection/models/patchcore`.

## Run it

```bash
./run_benchmark.sh --quick              # ~1 min sanity check
./run_benchmark.sh                      # full CPU run (~12 min)
./run_benchmark.sh --device igpu        # integrated GPU only
./run_benchmark.sh --all --plot         # cpu + igpu + npu, with graphs
./run_benchmark.sh --sudo               # needed for RAPL package energy
```

`--device` accepts `cpu`, `igpu` or `npu` (OpenVINO calls the iGPU "GPU").
A device that fails prints the specific fix and the run continues with the rest.

## Accelerator setup (Panther Lake)

Kernel drivers (`xe`, `intel_vpu`) ship with the distro, but three userspace
pieces are needed:

```bash
# device node access - both iGPU and NPU
sudo usermod -aG render,video $USER      # then log out / back in

# iGPU
sudo apt install -y ocl-icd-libopencl1 intel-opencl-icd

# NPU: not in the Ubuntu repos, get the .debs from
# https://github.com/intel/linux-npu-driver/releases
```

Verify with `python3 -c "import openvino as ov; print(ov.Core().available_devices)"`.

Results land in `benchmark/results/` (JSON + CSVs), graphs in
`benchmark/results/plots_<name>/`.

## Why synthetic input

The default input is random noise at the model's 256×256 input. PatchCore's cost
is data-independent — a fixed convolution pass plus a fixed-size coreset
nearest-neighbour search — so noise times identically to a real brick, and the
numbers measure the chip rather than camera bandwidth. `--camera N` adds a
separate section timing the camera path on its own, so the two never mix.

## What it measures

| Section | Why it exists |
|---|---|
| Idle baseline | Measured **before** load, so marginal energy is honest |
| Cold start | `read_model` / `compile_model` / first inference, split — different fixes |
| Steady-state latency | p50/p95/p99 at batch 1 — the real-time number |
| Peak throughput | Async queue; a different operating point that can reorder devices |
| End-to-end `predict()` | Adds crop/resize/presence, so you see preprocessing cost |
| Sustained load | Burst vs sustained fps — the laptop number |
| Memory | Peak RSS |
| Camera (optional) | Grab latency; flags camera-bound vs compute-bound |

## Energy caveat

`mJ/inference` needs a readable counter:

- **RAPL** (`/sys/class/powercap/intel-rapl/`) is exact but usually root-only →
  use `--sudo`.
- **Battery** (`power_now`) works only while **unplugged**. On AC it measures
  charging, so the script reports energy as unavailable instead of printing a
  wrong number.

## Files

- `bench_core.py` — runs on the laptop, writes JSON + CSVs
- `run_benchmark.sh` — copies it over SSH, runs it in the venv, fetches results
- `plot_benchmark.py` — JSON + CSVs → PNGs (needs matplotlib locally)

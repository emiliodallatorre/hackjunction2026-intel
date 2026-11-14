#!/usr/bin/env bash
# PatchCore inference benchmark runner: copies bench_core.py to the Intel laptop
# over ssh, runs it inside the project venv, streams the report and saves JSON.
#
# Performance only - no accuracy. Input is synthetic random noise at the model's
# 256x256 input, so the numbers measure silicon, not camera bandwidth.
#
# Usage:
#   ./run_benchmark.sh                          # full run, CPU (~12 min)
#   ./run_benchmark.sh --quick                  # 50 iters, no sustained (~1 min)
#   ./run_benchmark.sh --device npu             # NPU only
#   ./run_benchmark.sh --all --plot             # compare all three
#   ./run_benchmark.sh --sudo                   # read RAPL package energy
#   ./run_benchmark.sh --camera 0               # also measure the camera path
#
#   --host HOST        ssh target        (default intel@roberta)
#   --repo DIR         repo on the host  (default ~/Desktop/hackjunction/hackjunction2026-intel)
#   --venv PATH        activate script   (default <repo>/.venv/bin/activate)
#   --model DIR        model folder      (default <repo>/anomaly_detection/models/patchcore)
#   --device NAME      cpu | igpu | npu  (default cpu)
#   --devices LIST     comma separated, e.g. cpu,igpu,npu
#   --all              shorthand for cpu,igpu,npu
#   --iters N          measured iters    (default 300)
#   --warmup N         discarded iters   (default 50)
#   --sustain SECS     sustained load    (default 600, 0 disables)
#   --idle SECS        idle baseline     (default 30, measured BEFORE load)
#   --camera SPEC      v4l2 index/path   (default off)
#   --quick            iters=50 sustain=0 idle=10
#   --sudo             run remotely under sudo (RAPL energy needs root)
#   --no-fetch         leave results on the host (default: copy them back)
#   --plot             generate graphs locally after fetching
set -euo pipefail

HOST="intel@roberta"
# Absolute remote paths, hardcoded. No "~" anywhere: tilde expansion across the
# ssh command string is what silently blanked BENCH_MODEL before.
HOME_REPO='/home/intel/Desktop/hackjunction/hackjunction2026-intel'
REPO_DIR="$HOME_REPO"
MODEL_DIR="$HOME_REPO/anomaly_detection/models/patchcore"
VENV="$HOME_REPO/.venv/bin/activate"
DEVICES="CPU"
ITERS=300
WARMUP=50
SUSTAIN=600
IDLE=30
CAMERA=""
USE_SUDO=0
FETCH=1
PLOT=0
REMOTE_TMP="/tmp/patchcore_bench"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)    HOST="$2"; shift 2 ;;
    --repo)    REPO_DIR="$2"; shift 2 ;;
    --venv)    VENV="$2"; shift 2 ;;
    --model)   MODEL_DIR="$2"; shift 2 ;;
    --devices) DEVICES="$2"; shift 2 ;;
    --device)  DEVICES="$2"; shift 2 ;;   # singular alias: cpu|igpu|npu
    --all)     DEVICES="CPU,GPU,NPU"; shift ;;
    --iters)   ITERS="$2"; shift 2 ;;
    --warmup)  WARMUP="$2"; shift 2 ;;
    --sustain) SUSTAIN="$2"; shift 2 ;;
    --idle)    IDLE="$2"; shift 2 ;;
    --camera)  CAMERA="$2"; shift 2 ;;
    --quick)   ITERS=50; SUSTAIN=0; IDLE=10; shift ;;
    --sudo)    USE_SUDO=1; shift ;;
    --fetch)   FETCH=1; shift ;;
    --no-fetch) FETCH=0; shift ;;
    --plot)    PLOT=1; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# If --repo moved the root, re-derive the paths that were not set explicitly.
if [ "$REPO_DIR" != "$HOME_REPO" ]; then
  [ "$MODEL_DIR" = "$HOME_REPO/anomaly_detection/models/patchcore" ] \
    && MODEL_DIR="$REPO_DIR/anomaly_detection/models/patchcore"
  [ "$VENV" = "$HOME_REPO/.venv/bin/activate" ] \
    && VENV="$REPO_DIR/.venv/bin/activate"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
REMOTE_JSON="$REMOTE_TMP/result_$STAMP.json"

# One password prompt: share a single ssh connection across every call below.
CTL="/tmp/.pcbench_ssh_%C"
SSH_OPTS=(-o "ControlMaster=auto" -o "ControlPath=$CTL" -o "ControlPersist=20m")
ssh_() { ssh "${SSH_OPTS[@]}" "$@"; }
scp_() { scp -o "ControlMaster=auto" -o "ControlPath=$CTL" -o "ControlPersist=20m" "$@"; }

EST=$(( (IDLE + SUSTAIN + 90) * $(echo "$DEVICES" | tr ',' '\n' | grep -c .) / 60 + 1 ))
echo "=== PatchCore benchmark ==="
echo "  host      $HOST"
echo "  repo      $REPO_DIR"
echo "  model     $MODEL_DIR"
echo "  devices   $DEVICES"
echo "  iters     $ITERS (warmup $WARMUP)   sustain ${SUSTAIN}s   idle ${IDLE}s"
echo "  camera    ${CAMERA:-off (synthetic noise input)}"
echo "  sudo      $([ $USE_SUDO = 1 ] && echo yes || echo no)"
echo "  estimate  ~${EST} min"
echo

if [ ! -f "$HERE/bench_core.py" ]; then
  echo "ERROR: $HERE/bench_core.py not found" >&2; exit 1
fi

echo ">>> connecting to $HOST (password asked once)"
ssh_ -N -f "$HOST" 2>/dev/null || true
ssh_ -o BatchMode=yes "$HOST" true 2>/dev/null \
  || echo "    note: connection sharing unavailable; you may be prompted again"

echo ">>> copying bench_core.py"
ssh_ "$HOST" "mkdir -p $REMOTE_TMP"
scp_ -q "$HERE/bench_core.py" "$HOST:$REMOTE_TMP/bench_core.py"

echo ">>> checking venv: $VENV"
CHECK=$(ssh_ "$HOST" "
  if [ ! -f '$VENV' ]; then echo 'NO_VENV'; exit 0; fi
  if [ ! -f '$MODEL_DIR/model.xml' ]; then echo 'NO_MODEL'; exit 0; fi
  . '$VENV'
  python3 -c 'import openvino,numpy; print(\"OK\",openvino.__version__)' 2>&1 | tail -1
")
echo "    $CHECK"
case "$CHECK" in
  NO_VENV)  echo "ERROR: no activate script at $VENV (use --venv)" >&2; exit 1 ;;
  NO_MODEL) echo "ERROR: model.xml not found in $MODEL_DIR (use --model)" >&2; exit 1 ;;
  OK*) ;;
  *) echo "ERROR: venv imports failed, see above" >&2; exit 1 ;;
esac

# Under sudo the environment is reset on most systems ("-E is ignored"), which
# would silently blank every BENCH_* var and make the benchmark run its defaults.
# So the vars are passed as explicit arguments to `sudo env ...` rather than
# exported, and the venv python is named by absolute path (root's PATH lacks it).
SUDO_PREFIX=""
if [ $USE_SUDO = 1 ]; then
  SUDO_PREFIX="sudo env"
  echo ">>> remote sudo requested (for RAPL package energy) - it may prompt"
fi

echo ">>> running"
echo
# Paths are absolute, so every value is substituted here and the remote shell
# does no expansion of its own. One export per line, each value single-quoted.
# Every value is substituted on this side and single-quoted, so the remote
# shell performs no expansion of its own.
ENVARGS="BENCH_ITERS='$ITERS' BENCH_WARMUP='$WARMUP' BENCH_SUSTAIN='$SUSTAIN'"
ENVARGS="$ENVARGS BENCH_IDLE='$IDLE' BENCH_DEVICES='$DEVICES'"
ENVARGS="$ENVARGS BENCH_MODEL='$MODEL_DIR' BENCH_REPO_SRC='$REPO_DIR/src'"
ENVARGS="$ENVARGS BENCH_CAMERA='$CAMERA' BENCH_OUT='$REMOTE_JSON'"

ssh_ -t "$HOST" "
  set -e
  . '$VENV'
  cd '$REPO_DIR' 2>/dev/null || cd /tmp
  if [ ! -f '$MODEL_DIR/model.xml' ]; then
    echo 'ERROR: model.xml not found in $MODEL_DIR' >&2
    exit 1
  fi
  PY=\$VIRTUAL_ENV/bin/python3
  [ -x \"\$PY\" ] || PY=python3
  echo \"    python: \$PY\"
  echo \"    model:  $MODEL_DIR\"
  $SUDO_PREFIX $ENVARGS \"\$PY\" $REMOTE_TMP/bench_core.py 2>&1 \
    | tee $REMOTE_TMP/last_run.log
"

echo
STEM="${REMOTE_JSON%.json}"
LOCAL_JSON=""
if [ $FETCH = 1 ]; then
  mkdir -p "$HERE/results"
  echo ">>> fetching results into benchmark/results/"
  # The JSON plus its CSV sidecars (_traces, _latencies, _sustained, _summary).
  if scp_ -q "$HOST:$REMOTE_JSON" "$HOST:${STEM}_*.csv" "$HERE/results/" 2>/dev/null      || scp_ -q "$HOST:$REMOTE_JSON" "$HERE/results/" 2>/dev/null; then
    LOCAL_JSON="$HERE/results/$(basename "$REMOTE_JSON")"
    for f in "$HERE/results/$(basename "$STEM")"*; do
      [ -e "$f" ] && echo "    $f"
    done
  else
    echo "    could not fetch (left on host at $REMOTE_JSON)"
  fi
else
  echo "=== results on host: $REMOTE_JSON (and ${STEM}_*.csv)"
fi
echo "=== log: $HOST:$REMOTE_TMP/last_run.log"
ssh_ -O exit "$HOST" 2>/dev/null || true

if [ $PLOT = 1 ] && [ -n "$LOCAL_JSON" ]; then
  echo
  echo ">>> generating graphs"
  python3 "$HERE/plot_benchmark.py" "$LOCAL_JSON" || {
    echo "    plotting failed (matplotlib installed locally?)" >&2
    echo "    run manually: python3 benchmark/plot_benchmark.py $LOCAL_JSON" >&2
  }
elif [ -n "$LOCAL_JSON" ]; then
  echo
  echo "Graphs:  python3 benchmark/plot_benchmark.py $LOCAL_JSON"
fi

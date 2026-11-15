#!/usr/bin/env bash
# VLA policy benchmark runner (ACT, pi0.5): copies bench_vla.py to the Intel
# laptop over ssh, runs it inside the project venv, streams the report and saves
# JSON + CSVs.
#
# Performance only - no task success. Inputs are synthetic tensors at each IR
# port's declared shape, so the numbers measure silicon, not camera bandwidth.
#
# One inference emits a whole action chunk (ACT: 100 actions), so the report also
# gives actions/s and the control-rate headroom at --fps.
#
# Usage:
#   ./run_vla_benchmark.sh --export ~/Downloads/pi05-ricotta-runtime     # iGPU
#   ./run_vla_benchmark.sh --export ~/Downloads/ACT --quick              # ~1 min
#   ./run_vla_benchmark.sh --export DIR --devices igpu,cpu --plot        # compare
#   ./run_vla_benchmark.sh --export DIR --sudo                           # RAPL energy
#   ./run_vla_benchmark.sh --local --export /tmp/actx                    # no ssh
#
#   --export DIR       export root (holding exports/openvino/) or the
#                      exports/openvino folder itself, or a .zip  REQUIRED
#   --host HOST        ssh target        (default intel@roberta)
#   --repo DIR         repo on the host  (default ~/Desktop/hackjunction/hackjunction2026-intel)
#   --venv PATH        activate script   (default <repo>/.venv/bin/activate)
#   --local            run here instead of over ssh
#   --device NAME      cpu | igpu | npu  (default igpu)
#   --devices LIST     comma separated, e.g. igpu,cpu
#   --fps N            robot control rate for the headroom maths (default 30)
#   --iters N          measured iters    (default 60)
#   --warmup N         discarded iters   (default 10)
#   --sustain SECS     sustained load    (default 300, 0 disables)
#   --idle SECS        idle baseline     (default 30, measured BEFORE load)
#   --task TEXT        task string for the end-to-end section
#   --skip-e2e         per-IR OpenVINO numbers only, no physicalai import
#   --quick            iters=10 warmup=3 sustain=0 idle=10
#   --sudo             run remotely under sudo (RAPL energy needs root)
#   --no-fetch         leave results on the host (default: copy them back)
#   --plot             generate graphs locally after fetching
#
# NPU is accepted but is not expected to run pi0.5: the model exceeds what the
# NPU compiler takes in one graph. Use igpu.
set -euo pipefail

HOST="intel@roberta"
HOME_REPO='/home/intel/Desktop/hackjunction/hackjunction2026-intel'
REPO_DIR="$HOME_REPO"
VENV="$HOME_REPO/.venv/bin/activate"
EXPORT_DIR=""
DEVICES="GPU"
FPS=30
ITERS=60
WARMUP=10
SUSTAIN=300
IDLE=30
TASK=""
SKIP_E2E=0
USE_SUDO=0
FETCH=1
PLOT=0
LOCAL=0
REMOTE_TMP="/tmp/vla_bench"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Friendly device names -> what OpenVINO calls them. bench_vla.py accepts both,
# but normalising here keeps the echoed banner honest.
norm_devices() {
  echo "$1" | tr 'a-z' 'A-Z' | sed 's/IGPU/GPU/g'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --export)  EXPORT_DIR="$2"; shift 2 ;;
    --host)    HOST="$2"; shift 2 ;;
    --repo)    REPO_DIR="$2"; shift 2 ;;
    --venv)    VENV="$2"; shift 2 ;;
    --local)   LOCAL=1; shift ;;
    --devices) DEVICES="$(norm_devices "$2")"; shift 2 ;;
    --device)  DEVICES="$(norm_devices "$2")"; shift 2 ;;
    --fps)     FPS="$2"; shift 2 ;;
    --iters)   ITERS="$2"; shift 2 ;;
    --warmup)  WARMUP="$2"; shift 2 ;;
    --sustain) SUSTAIN="$2"; shift 2 ;;
    --idle)    IDLE="$2"; shift 2 ;;
    --task)    TASK="$2"; shift 2 ;;
    --skip-e2e) SKIP_E2E=1; shift ;;
    --quick)   ITERS=10; WARMUP=3; SUSTAIN=0; IDLE=10; shift ;;
    --sudo)    USE_SUDO=1; shift ;;
    --fetch)   FETCH=1; shift ;;
    --no-fetch) FETCH=0; shift ;;
    --plot)    PLOT=1; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [ -z "$EXPORT_DIR" ]; then
  echo "ERROR: --export is required (the Studio export holding exports/openvino/)" >&2
  echo "       e.g. --export ~/Downloads/pi05-ricotta-runtime" >&2
  exit 2
fi

if [ "$REPO_DIR" != "$HOME_REPO" ] && [ "$VENV" = "$HOME_REPO/.venv/bin/activate" ]; then
  VENV="$REPO_DIR/.venv/bin/activate"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"

NDEV=$(echo "$DEVICES" | tr ',' '\n' | grep -c .)
EST=$(( (IDLE + SUSTAIN + 120) * NDEV / 60 + 1 ))
echo "=== VLA policy benchmark ==="
echo "  mode      $([ $LOCAL = 1 ] && echo local || echo "ssh $HOST")"
echo "  export    $EXPORT_DIR"
echo "  devices   $DEVICES"
echo "  control   ${FPS} Hz (used for the action-chunk headroom maths)"
echo "  iters     $ITERS (warmup $WARMUP)   sustain ${SUSTAIN}s   idle ${IDLE}s"
echo "  end2end   $([ $SKIP_E2E = 1 ] && echo "skipped" || echo "yes (needs physicalai importable)")"
echo "  estimate  ~${EST} min (longer for pi0.5: a cold iGPU compile can take minutes)"
echo

if [ ! -f "$HERE/bench_vla.py" ]; then
  echo "ERROR: $HERE/bench_vla.py not found" >&2; exit 1
fi

# Every value is substituted on this side and single-quoted, so the remote shell
# performs no expansion of its own. No "~" is ever passed across the ssh command
# string: tilde expansion there is what silently blanks these variables.
build_envargs() {
  local export_dir="$1" repo="$2" out="$3"
  printf "BENCH_ITERS='%s' BENCH_WARMUP='%s' BENCH_SUSTAIN='%s' BENCH_IDLE='%s' " \
    "$ITERS" "$WARMUP" "$SUSTAIN" "$IDLE"
  printf "BENCH_DEVICES='%s' BENCH_FPS='%s' BENCH_EXPORT='%s' " \
    "$DEVICES" "$FPS" "$export_dir"
  printf "BENCH_REPO_SRC='%s/src' BENCH_OUT='%s' " "$repo" "$out"
  [ -n "$TASK" ] && printf "BENCH_TASK='%s' " "$TASK"
  [ $SKIP_E2E = 1 ] && printf "BENCH_SKIP_E2E=1 "
  printf "\n"
}

# ---------------------------------------------------------------- local mode
if [ $LOCAL = 1 ]; then
  mkdir -p "$HERE/results"
  LOCAL_JSON="$HERE/results/vla_$STAMP.json"
  # Locally the repo is the one this script lives in, not the remote --repo path.
  LOCAL_REPO="$(cd "$HERE/.." && pwd)"
  PY="$LOCAL_REPO/.venv/bin/python3"
  [ -x "$PY" ] || PY="$(command -v python3)"
  echo ">>> running locally with $PY"
  echo
  ENVARGS="$(build_envargs "$(cd "$EXPORT_DIR" && pwd)" "$LOCAL_REPO" "$LOCAL_JSON")"
  if [ $USE_SUDO = 1 ]; then
    # Under sudo the environment is reset on most systems, so the vars go as
    # explicit arguments to `sudo env` rather than exported.
    eval "sudo env $ENVARGS '$PY' '$HERE/bench_vla.py'" 2>&1 | tee "$HERE/results/last_run_$STAMP.log"
  else
    eval "env $ENVARGS '$PY' '$HERE/bench_vla.py'" 2>&1 | tee "$HERE/results/last_run_$STAMP.log"
  fi
  echo
  echo "=== results: $LOCAL_JSON (and _*.csv)"
  if [ $PLOT = 1 ]; then
    echo ">>> generating graphs"
    python3 "$HERE/plot_benchmark.py" "$LOCAL_JSON" || \
      echo "    plotting failed (matplotlib installed?)" >&2
  else
    echo "Graphs:  python3 benchmark/plot_benchmark.py $LOCAL_JSON"
  fi
  exit 0
fi

# ----------------------------------------------------------------- ssh mode
REMOTE_JSON="$REMOTE_TMP/vla_$STAMP.json"

# One password prompt: share a single ssh connection across every call below.
CTL="/tmp/.vlabench_ssh_%C"
SSH_OPTS=(-o "ControlMaster=auto" -o "ControlPath=$CTL" -o "ControlPersist=30m")
ssh_() { ssh "${SSH_OPTS[@]}" "$@"; }
scp_() { scp -o "ControlMaster=auto" -o "ControlPath=$CTL" -o "ControlPersist=30m" "$@"; }

echo ">>> connecting to $HOST (password asked once)"
ssh_ -N -f "$HOST" 2>/dev/null || true
ssh_ -o BatchMode=yes "$HOST" true 2>/dev/null \
  || echo "    note: connection sharing unavailable; you may be prompted again"

echo ">>> copying bench_vla.py"
ssh_ "$HOST" "mkdir -p $REMOTE_TMP"
scp_ -q "$HERE/bench_vla.py" "$HOST:$REMOTE_TMP/bench_vla.py"

# The export may be a .zip (what Studio downloads) or an unpacked folder, and it
# may sit anywhere on the host; resolve and unzip it remotely so --export takes
# whatever the user already has.
echo ">>> resolving export on the host: $EXPORT_DIR"
RESOLVED=$(ssh_ "$HOST" "
  set -e
  E=\$(eval echo '$EXPORT_DIR')
  if [ ! -e \"\$E\" ]; then echo 'NO_EXPORT'; exit 0; fi
  case \"\$E\" in
    *.zip)
      D=$REMOTE_TMP/\$(basename \"\${E%.zip}\")
      mkdir -p \"\$D\"
      # Only the IR is needed; model.ckpt is hundreds of MB of torch weights.
      unzip -o -q \"\$E\" 'exports/openvino/*' -d \"\$D\" 2>/dev/null || unzip -o -q \"\$E\" -d \"\$D\"
      echo \"\$D\" ;;
    *) echo \"\$E\" ;;
  esac
")
if [ "$RESOLVED" = "NO_EXPORT" ] || [ -z "$RESOLVED" ]; then
  echo "ERROR: $EXPORT_DIR does not exist on $HOST" >&2; exit 1
fi
echo "    $RESOLVED"

echo ">>> checking venv: $VENV"
CHECK=$(ssh_ "$HOST" "
  if [ ! -f '$VENV' ]; then echo 'NO_VENV'; exit 0; fi
  . '$VENV'
  if [ -z \"\$(find '$RESOLVED' -name '*.xml' -path '*openvino*' -print -quit 2>/dev/null)\$(find '$RESOLVED' -maxdepth 1 -name '*.xml' -print -quit 2>/dev/null)\" ]; then
    echo 'NO_IR'; exit 0
  fi
  python3 -c 'import openvino,numpy; print(\"OK\",openvino.__version__)' 2>&1 | tail -1
")
echo "    $CHECK"
case "$CHECK" in
  NO_VENV) echo "ERROR: no activate script at $VENV (use --venv)" >&2; exit 1 ;;
  NO_IR)   echo "ERROR: no OpenVINO IR (*.xml) under $RESOLVED" >&2
           echo "       Point --export at the export root or its exports/openvino folder" >&2; exit 1 ;;
  OK*) ;;
  *) echo "ERROR: venv imports failed, see above" >&2; exit 1 ;;
esac

SUDO_PREFIX=""
if [ $USE_SUDO = 1 ]; then
  SUDO_PREFIX="sudo env"
  echo ">>> remote sudo requested (for RAPL package energy) - it may prompt"
fi

echo ">>> running"
echo
ENVARGS="$(build_envargs "$RESOLVED" "$REPO_DIR" "$REMOTE_JSON")"

ssh_ -t "$HOST" "
  set -e
  . '$VENV'
  cd '$REPO_DIR' 2>/dev/null || cd /tmp
  PY=\$VIRTUAL_ENV/bin/python3
  [ -x \"\$PY\" ] || PY=python3
  echo \"    python: \$PY\"
  echo \"    export: $RESOLVED\"
  $SUDO_PREFIX $ENVARGS \"\$PY\" $REMOTE_TMP/bench_vla.py 2>&1 \
    | tee $REMOTE_TMP/last_run.log
"

echo
STEM="${REMOTE_JSON%.json}"
LOCAL_JSON=""
if [ $FETCH = 1 ]; then
  mkdir -p "$HERE/results"
  echo ">>> fetching results into benchmark/results/"
  if scp_ -q "$HOST:$REMOTE_JSON" "$HOST:${STEM}_*.csv" "$HERE/results/" 2>/dev/null \
     || scp_ -q "$HOST:$REMOTE_JSON" "$HERE/results/" 2>/dev/null; then
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

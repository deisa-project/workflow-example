#!/usr/bin/env bash

mkdir -p .logs
# Send stdout to tee, then send stderr to *that* stdout (captures -x too)
exec > >(tee -a .logs/run.log) 2>&1

# (optional) nicer -x lines with timestamps
export PS4='+ $(date "+%F %T") ${BASH_SOURCE##*/}:${LINENO}: '


PROJECT_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
cd $PROJECT_ROOT
VENV="$PROJECT_ROOT/.doreisa-env"
export PYTHONWARNINGS=ignore

# --- Select Python environment ---
if [[ -d "$VENV" ]]; then
  echo "[insitu] Using Python virtualenv at $VENV"
  export VIRTUAL_ENV="$VENV"
  export PATH="$VENV/bin:$PATH"
  PY="$VENV/bin/python"
elif [[ "${USE_NIX:-0}" == "1" ]]; then
  echo "[insitu] Using nix flake"
  PY="python3"
else
  echo "Error: No virtualenv found at $VENV and nix not detected." >&2
  echo "Either create the venv or use the nix flake." >&2
  exit 1
fi


# Stop Ray on exit no matter what
cleanup() {
  set +e
  ray stop || true
}
trap cleanup EXIT

# Start Ray head + a worker (use localhost rather than a hardcoded LAN IP)
ray start --head --port=6379 --node-ip-address=127.0.0.1   > .logs/ray-head.log 2>&1
ray start --address=127.0.0.1:6379                          > .logs/ray-worker.log 2>&1


# Start analytics (uses venv python)
"$PY" -m analytics.avg &
ANALYTICS_PID=$!

# Run the simulation under MPI with the venv python.
# Export env vars to each rank so they inherit the venv PATH/PYTHONPATH.
mpirun --merge-stderr-to-stdout -n 2 \
  -x PATH \
  -x VIRTUAL_ENV \
  "$PY" "$PROJECT_ROOT/python/sim-doreisa.py" \
    --steps 10 --print-every 1 --seed-mode local --periodic --viz-every 1 --viz-gif

wait "$ANALYTICS_PID"


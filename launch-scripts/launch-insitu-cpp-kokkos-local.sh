#!/usr/bin/env bash

mkdir -p .logs
# Send stdout to tee, then send stderr to *that* stdout (captures -x too)
exec > >(tee -a .logs/run.log) 2>&1

# (optional) nicer -x lines with timestamps
export PS4='+ $(date "+%F %T") ${BASH_SOURCE##*/}:${LINENO}: '

PROJECT_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
cd $PROJECT_ROOT
VENV="$PROJECT_ROOT/.doreisa-env"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi

# Use the venv's tools
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
# silence all warnings
export PYTHONWARNINGS=ignore

# Stop Ray on exit no matter what
cleanup() {
  set +e
  ray stop || true
}
trap cleanup EXIT

# Start Ray head + a worker (use localhost rather than a hardcoded LAN IP)
ray start --head --port=6379 --node-ip-address=127.0.0.1 >.logs/ray-head.log 2>&1
ray start --address=127.0.0.1:6379 >.logs/ray-worker.log 2>&1

# Start analytics (uses venv python)
"$PY" -m analytics.avg &
>.logs/analytics.log 2>&1
ANALYTICS_PID=$!

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Run the simulation under MPI
mpirun --merge-stderr-to-stdout -n 2 \
  -x PATH \
  "$PROJECT_ROOT/cpp/gpu/sim-kokkos-doreisa" \
  --steps 10 --print-every 1 --seed-mode local --periodic --viz-every 1 --viz-gif

wait "$ANALYTICS_PID"

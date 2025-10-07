#!/usr/bin/env bash

mkdir -p .logs
# Send stdout to tee, then send stderr to *that* stdout (captures -x too)
exec > >(tee -a .logs/run.log) 2>&1

# (optional) nicer -x lines with timestamps
export PS4='+ $(date "+%F %T") ${BASH_SOURCE##*/}:${LINENO}: '

PROJECT_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
cd $PROJECT_ROOT

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi

export OMP_PROC_BIND=spread
export OMP_PLACES=threads

# Run the simulation under MPI
mpirun --merge-stderr-to-stdout -n 2 \
  -x PATH \
  "$PROJECT_ROOT/cpp/sim" \
  --steps 1000 --print-every 50 --seed-mode local --periodic --viz-every 25 --viz-gif

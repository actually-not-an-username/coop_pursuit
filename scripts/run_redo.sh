#!/usr/bin/env bash
# Redo experiment: 3 pursuers, fleeing prey (p=0.5), 5 seeds x {clean, masked} = 10 training runs
# (settings in config.env), then the evaluation sweep on every saved checkpoint plus the random
# baseline, then figures/tables.
# Usage: bash scripts/run_redo.sh [max_parallel_jobs]
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${1:-5}
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# Force CPU: 10 concurrent CUDA processes crashed the PC (0x133 DPC_WATCHDOG_VIOLATION, NVIDIA driver) 3x on 2026-11-23.
export CUDA_VISIBLE_DEVICES=-1  # "-1", not empty: Windows drops empty env vars
mkdir -p runs/logs

run() { echo "$(date +%H:%M) start: $*"; python main.py "$@" > "runs/logs/$(echo "$*" | tr ' /' '__').log" 2>&1; }
throttle() { while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do sleep 30; done; }

for seed in 1 2 3 4 5; do
  for masked in "" "--masked"; do
    throttle; run train --seed "$seed" $masked &
  done
done
wait
echo "$(date +%H:%M) training done"

for ckpt in runs/*_n3_*_s*/{best,best_noloss,final}_actor.pt; do
  throttle; run sweep --checkpoint "$ckpt" &
done
throttle; run sweep --random &
wait
echo "$(date +%H:%M) sweep done"

python scripts/analyze_sweep.py --runs runs --out runs/analysis
echo "$(date +%H:%M) All done. Results: runs/analysis/, per-episode data: runs/sweep/*.csv"

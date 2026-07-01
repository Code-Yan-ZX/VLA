#!/usr/bin/env bash
# Serially run all runnable jobs in configs/queue.json (1x A40, single-card).
# Picks the next dep-satisfied job, runs it, loops. Stops when no runnable job
# remains (all done, or the only ones left are blocked by a failed dep).
# Usage:  bash scripts/run_queue_all.sh > runs/queue_driver.log 2>&1 &
set -uo pipefail
cd /media/disk2/YZX/research/vla

iter=0
while :; do
  iter=$((iter+1))
  echo "===== driver iter $iter ====="
  out=$(python scripts/queue run 2>&1)
  echo "$out"
  case "$out" in
    *"no runnable job"*) echo "[driver] queue exhausted or blocked; stopping."; break;;
  esac
done
echo "[driver] done after $iter iters"

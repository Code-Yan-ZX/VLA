#!/usr/bin/env bash
# Serially run all runnable jobs in configs/queue.json (1x A40, single-card).
# Picks the next dep-satisfied job, runs it, loops. Stops when no runnable job
# remains (all done, or the only ones left are blocked by a failed dep).
# Between jobs: GPU-settle guard — wait until GPU memory frees (avoids the
# stale-vLLM-process OOM that back-to-back engine inits can hit).
# Usage:  bash scripts/run_queue_all.sh > runs/queue_driver.log 2>&1 &
set -uo pipefail
cd /media/disk2/YZX/research/vla

gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

iter=0
while :; do
  iter=$((iter+1))
  echo "===== driver iter $iter ====="
  # GPU-settle: wait until >=40GB free before launching the next vLLM job.
  for _ in $(seq 1 60); do
    free=$(gpu_free_mib)
    if [ -n "$free" ] && [ "$free" -ge 40000 ]; then break; fi
    echo "[driver] GPU busy (free=${free}MiB), waiting 10s for stale process to release..."; sleep 10
  done
  out=$(python scripts/queue run 2>&1)
  echo "$out"
  case "$out" in
    *"no runnable job"*) echo "[driver] queue exhausted or blocked; stopping."; break;;
  esac
done
echo "[driver] done after $iter iters"

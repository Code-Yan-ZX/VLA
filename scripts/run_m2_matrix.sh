#!/usr/bin/env bash
# M2 concurrency x prune-rate matrix runner (P2 method-D scoping).
# Runs 6 vLLM jobs serially with GPU-settle sleeps between them.
# Usage: bash scripts/run_m2_matrix.sh
set -u
PY=/home/dell/miniconda3/envs/vtc_serve/bin/python
MODEL=runs/models/llava-1.5-7b-hf
SUBSET=eval/subsets/gqa_200.jsonl
OUT=runs/p2_d
mkdir -p "$OUT"

run_one () {
  local name=$1 mns=$2 pr=$3
  local out="$OUT/${name}.json"
  if [ -f "$out" ]; then echo "[m2] SKIP $name (exists)"; return; fi
  echo "[m2] $(date +%H:%M:%S) START $name (max_num_seqs=$mns pruning=$pr)"
  # GPU-settle: wait until GPU memory is mostly free (<4GB used) before launching
  for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -lt 4000 ]; then break; fi
    sleep 15
  done
  sleep 20  # extra settle for stale-process teardown
  $PY -m src.serve_bench --model "$MODEL" --pruning-rate "$pr" --benchmark gqa \
    --subset "$SUBSET" --metrics-out "$out" --max-tokens 16 --max-model-len 4096 \
    --gpu-memory-utilization 0.85 --seed 0 --limit 100 --selector proxy \
    --max-num-seqs "$mns" --batch-submit > "$OUT/${name}.log" 2>&1
  rc=$?
  echo "[m2] $(date +%H:%M:%S) DONE $name rc=$rc"
  if [ $rc -ne 0 ]; then echo "[m2] FAIL $name — tail log:"; tail -8 "$OUT/${name}.log"; fi
}

# Order: c1 first (faster, isolates latency), then c12 (throughput)
run_one m2_c1_r0  1  0.0
run_one m2_c1_r50 1  0.50
run_one m2_c1_r75 1  0.75
run_one m2_c12_r0  12 0.0
run_one m2_c12_r50 12 0.50
run_one m2_c12_r75 12 0.75
echo "[m2] ALL DONE $(date +%H:%M:%S)"

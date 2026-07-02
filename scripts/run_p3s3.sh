#!/usr/bin/env bash
# P3-step-3: n=500 accuracy tightening for the Pareto claims.
#
# 12 jobs: {adaptive, fixed-r25, fixed-r50} x {gqa, mme, mmbench, scienceqa}
# under the SAME settings as the P3-step-2 n=200 runs (clean n=200->n=500
# comparison):  c12 (max_num_seqs=12), bursty profile, num_running controller
# (adaptive: r_min 0.25 / r_max 0.50, conc-lo 0.25 / conc-hi 0.75), seed=0.
#   * gqa          : max-tokens=32 (matches P3-step-1 GQA n=200 anchor)
#   * mme/mmb/sqa  : max-tokens=64 (matches P3-step-2 n=200 anchor)
#
# TextVQA r50 n=500 already exists in runs/p3s2/ (adaptive + r25 + r50 all
# present) -> NOT re-run here; analyze_p3s3 reuses those.
#
# GATE: does adaptive cleanly Pareto-dominate BOTH fixed points at n=500 on
# MME and ScienceQA (the two n=200 Pareto cases)?  acc margins at n=500 with
# stderr ~+-0.022: +0.015 = noise, +0.03 = suggestive, +0.04+ = meaningful.
#
# Usage:  bash scripts/run_p3s3.sh > runs/p3s3_driver.log 2>&1 &
set -uo pipefail
cd /media/disk2/YZX/research/vla

PY=/home/dell/miniconda3/envs/vtc_serve/bin/python
MODEL=runs/models/llava-1.5-7b-hf
OUT=runs/p3s3
MNS=12
mkdir -p "$OUT"

gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

# GPU-settle guard: wait until >=40GB free before the next vLLM init (avoids
# stale-vLLM-process OOM on back-to-back engine starts).
gpu_settle() {
  for _ in $(seq 1 60); do
    free=$(gpu_free_mib)
    if [ -n "$free" ] && [ "$free" -ge 40000 ]; then break; fi
    echo "[p3s3] $(date +%H:%M:%S) GPU busy (free=${free}MiB), waiting 10s..."; sleep 10
  done
  sleep 15
}

# run_serve <name> <benchmark> <subset> <max-tokens> <extra args...>
run_serve () {
  local name=$1 bench=$2 subset=$3 mt=$4; shift 4
  local out="$OUT/${name}.json" log="$OUT/${name}.log"
  if [ -f "$out" ]; then echo "[p3s3] SKIP $name (metrics exist)"; return; fi
  gpu_settle
  echo "[p3s3] $(date +%H:%M:%S) START $name"
  $PY -m src.serve_bench --model "$MODEL" --benchmark "$bench" \
    --subset "$subset" --metrics-out "$out" --max-tokens "$mt" \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0 \
    --selector proxy --max-num-seqs "$MNS" "$@" > "$log" 2>&1
  local rc=$?
  local acc=""; acc=$(grep -oE 'acc=[0-9.]+' "$log" | tail -1)
  echo "[p3s3] $(date +%H:%M:%S) DONE $name rc=$rc $acc"
  if [ $rc -ne 0 ]; then echo "[p3s3] FAIL $name — tail:"; tail -15 "$log"; fi
}

ADAPTIVE_ARGS=(--adaptive --r-min 0.25 --r-max 0.50 --load-signal num_running \
               --conc-lo 0.25 --conc-hi 0.75 --load-profile bursty)

# ===== GQA (mt32, matching the P3-step-1 n=200 anchor) =====
GQA_SUB=eval/subsets/gqa_500.jsonl
run_serve "gqa_adaptive_bursty_n500"   gqa "$GQA_SUB" 32 "${ADAPTIVE_ARGS[@]}"
run_serve "gqa_fixed_r25_bursty_n500"  gqa "$GQA_SUB" 32 --pruning-rate 0.25 --load-profile bursty
run_serve "gqa_fixed_r50_bursty_n500"  gqa "$GQA_SUB" 32 --pruning-rate 0.50 --load-profile bursty

# ===== MME / MMBench / ScienceQA (mt64, matching the P3-step-2 n=200 anchor) =====
for B in mme:mme_500 mmbench:mmbench_500 scienceqa:scienceqa_500; do
  bench=${B%%:*}; sub=eval/subsets/${B##*:}.jsonl
  run_serve "${bench}_adaptive_bursty_n500"  "$bench" "$sub" 64 "${ADAPTIVE_ARGS[@]}"
  run_serve "${bench}_fixed_r25_bursty_n500" "$bench" "$sub" 64 --pruning-rate 0.25 --load-profile bursty
  run_serve "${bench}_fixed_r50_bursty_n500" "$bench" "$sub" 64 --pruning-rate 0.50 --load-profile bursty
done

echo "[p3s3] ALL DONE $(date +%H:%M:%S)"

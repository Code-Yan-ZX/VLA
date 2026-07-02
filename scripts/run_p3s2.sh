#!/usr/bin/env bash
# P3-step-2: breadth Pareto validation across MME / MMBench / ScienceQA, plus
# the n=500 TextVQA acc tightening, the FastV accuracy anchor, and the c4
# concurrency-matrix cells. 1x A40, serial, GPU-settle between jobs.
#
# Each job: {adaptive, fixed-r25, fixed-r50} x {mme, mmbench, scienceqa} under
# the c12 bursty profile (the refined alternating-burst one), max-tokens=64.
# Claim to test: adaptive Pareto-dominates BOTH fixed points on tasks where
# compression costs accuracy (long/dense answers). 2 of 3 = pattern confirmed.
#
# Usage:  bash scripts/run_p3s2.sh > runs/p3s2_driver.log 2>&1 &
set -uo pipefail
cd /media/disk2/YZX/research/vla

PY=/home/dell/miniconda3/envs/vtc_serve/bin/python
FASTV_PY=/home/dell/miniconda3/envs/fastv/bin/python
MODEL=runs/models/llava-1.5-7b-hf
FASTV_MODEL=/media/disk2/YZX/doct/FastV/llava-v1.5-7b
OUT=runs/p3s2
MNS=12          # max_num_seqs: full continuous batching (c12)
MT=64           # max-tokens (longer than GQA's 32 for denser answers)
mkdir -p "$OUT"

gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

# GPU-settle guard: wait until >=40GB free before launching the next vLLM job
# (avoids stale-vLLM-process OOM on back-to-back engine inits).
gpu_settle() {
  for _ in $(seq 1 60); do
    free=$(gpu_free_mib)
    if [ -n "$free" ] && [ "$free" -ge 40000 ]; then break; fi
    echo "[p3s2] $(date +%H:%M:%S) GPU busy (free=${free}MiB), waiting 10s..."; sleep 10
  done
  sleep 15  # extra settle for stale-process teardown
}

# run_serve <name> <benchmark> <subset> <extra args...>
run_serve () {
  local name=$1 bench=$2 subset=$3; shift 3
  local out="$OUT/${name}.json" log="$OUT/${name}.log"
  if [ -f "$out" ]; then echo "[p3s2] SKIP $name (metrics exist)"; return; fi
  gpu_settle
  echo "[p3s2] $(date +%H:%M:%S) START $name"
  $PY -m src.serve_bench --model "$MODEL" --benchmark "$bench" \
    --subset "$subset" --metrics-out "$out" --max-tokens "$MT" \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0 \
    --selector proxy --max-num-seqs "$MNS" "$@" > "$log" 2>&1
  local rc=$?
  local acc=""; acc=$(grep -oE 'acc=[0-9.]+' "$log" | tail -1)
  echo "[p3s2] $(date +%H:%M:%S) DONE $name rc=$rc $acc"
  if [ $rc -ne 0 ]; then echo "[p3s2] FAIL $name — tail:"; tail -15 "$log"; fi
}

# run_fastv <name> <benchmark> <subset> <keep_tokens>
run_fastv () {
  local name=$1 bench=$2 subset=$3 keep=$4
  local out="$OUT/${name}.json" log="$OUT/${name}.log"
  if [ -f "$out" ]; then echo "[p3s2] SKIP $name (metrics exist)"; return; fi
  gpu_settle
  echo "[p3s2] $(date +%H:%M:%S) START $name"
  $FASTV_PY -m src.fastv_bench --model-path "$FASTV_MODEL" --benchmark "$bench" \
    --subset "$subset" --metrics-out "$out" --keep-tokens "$keep" --agg-layer 2 \
    --max-tokens 32 > "$log" 2>&1
  local rc=$?
  echo "[p3s2] $(date +%H:%M:%S) DONE $name rc=$rc"
  if [ $rc -ne 0 ]; then echo "[p3s2] FAIL $name — tail:"; tail -15 "$log"; fi
}

# ===== TASK 2: Pareto comparison on MME / MMBench / ScienceQA (9 jobs) =====
# Adaptive config: num_running signal, conc-lo 0.25 / conc-hi 0.75 (the refined
# P3-step-1 defaults that cleanly traversed [r_min,r_max] under c12 bursty).
ADAPTIVE_ARGS=(--adaptive --r-min 0.25 --r-max 0.50 --load-signal num_running \
               --conc-lo 0.25 --conc-hi 0.75 --load-profile bursty)
for B in mme:mme_200 mmbench:mmbench_200 scienceqa:scienceqa_200; do
  bench=${B%%:*}; sub=eval/subsets/${B##*:}.jsonl
  run_serve "${bench}_adaptive_bursty_mt64" "$bench" "$sub" "${ADAPTIVE_ARGS[@]}"
  run_serve "${bench}_fixed_r25_bursty_mt64" "$bench" "$sub" --pruning-rate 0.25 --load-profile bursty
  run_serve "${bench}_fixed_r50_bursty_mt64" "$bench" "$sub" --pruning-rate 0.50 --load-profile bursty
done

# ===== TASK 3a: n=500 TextVQA to tighten the +0.020 acc margin =====
# Build a 500-sample subset first (shared images dir with the 200 subset).
TVQA500=eval/subsets/textvqa_500.jsonl
if [ ! -f "$TVQA500" ]; then
  echo "[p3s2] building textvqa_500 subset..."
  /home/dell/miniconda3/envs/vtc/bin/python scripts/build_textvqa_500.py \
    > runs/build_textvqa_500.log 2>&1 || echo "[p3s2] WARN: textvqa_500 build failed"
fi
if [ -f "$TVQA500" ]; then
  MT_TVQA=32  # match the n=200 TextVQA mt32 anchor for direct comparability
  for cfg in "adaptive:--adaptive --r-min 0.25 --r-max 0.50 --load-signal num_running --conc-lo 0.25 --conc-hi 0.75 --load-profile bursty" \
             "fixed_r25:--pruning-rate 0.25 --load-profile bursty" \
             "fixed_r50:--pruning-rate 0.50 --load-profile bursty"; do
    name=${cfg%%:*}; args=${cfg##*:}
    out="$OUT/textvqa_${name}_bursty_n500.json"
    if [ -f "$out" ]; then echo "[p3s2] SKIP textvqa_${name}_n500"; continue; fi
    gpu_settle
    echo "[p3s2] $(date +%H:%M:%S) START textvqa_${name}_n500"
    $PY -m src.serve_bench --model "$MODEL" --benchmark textvqa \
      --subset "$TVQA500" --metrics-out "$out" --max-tokens "$MT_TVQA" \
      --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0 \
      --selector proxy --max-num-seqs "$MNS" $args > "$OUT/textvqa_${name}_bursty_n500.log" 2>&1
    rc=$?; acc=$(grep -oE 'acc=[0-9.]+' "$OUT/textvqa_${name}_bursty_n500.log" | tail -1)
    echo "[p3s2] $(date +%H:%M:%S) DONE textvqa_${name}_n500 rc=$rc $acc"
  done
fi

# ===== TASK 3b: FastV accuracy anchor on the new benchmarks (where feasible) =====
# FastV keep=288 (r50) on MME/MMBench/ScienceQA. Accuracy-only comparison row.
# (GQA + TextVQA FastV anchors already exist from P3-step-1.)
for B in mme:mme_200 mmbench:mmbench_200 scienceqa:scienceqa_200; do
  bench=${B%%:*}; sub=eval/subsets/${B##*:}.jsonl
  run_fastv "fastv_${bench}_keep288" "$bench" "$sub" 288
  run_fastv "fastv_${bench}_control576" "$bench" "$sub" 576
done

# ===== TASK 4: concurrency matrix c4 cell (c1/c12 already have from M2) =====
for cfg in "r0:--pruning-rate 0.0 --batch-submit" "r50:--pruning-rate 0.50 --batch-submit"; do
  name=${cfg%%:*}; args=${cfg##*:}
  out="$OUT/gqa_${name}_c4_batch.json"
  if [ -f "$out" ]; then echo "[p3s2] SKIP gqa_${name}_c4"; continue; fi
  gpu_settle
  echo "[p3s2] $(date +%H:%M:%S) START gqa_${name}_c4"
  $PY -m src.serve_bench --model "$MODEL" --benchmark gqa \
    --subset eval/subsets/gqa_200.jsonl --metrics-out "$out" --max-tokens 32 \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0 \
    --selector proxy --max-num-seqs 4 $args > "$OUT/gqa_${name}_c4_batch.log" 2>&1
  rc=$?; acc=$(grep -oE 'acc=[0-9.]+' "$OUT/gqa_${name}_c4_batch.log" | tail -1)
  echo "[p3s2] $(date +%H:%M:%S) DONE gqa_${name}_c4 rc=$rc $acc"
done

echo "[p3s2] ALL DONE $(date +%H:%M:%S)"

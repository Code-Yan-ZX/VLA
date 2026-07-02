#!/usr/bin/env bash
# P2 method-D validation: adaptive vs fixed-r25 vs fixed-r50 under a VARYING
# load profile (bursty), plus a constant-load sanity. Runs on GQA, n=100.
# The adaptive win only appears under varying load (constant high -> just use
# r_max; constant low -> r_min). Output -> runs/p2_d/dval_*.
# Usage: bash scripts/run_d_validation.sh
set -u
PY=/home/dell/miniconda3/envs/vtc_serve/bin/python
MODEL=runs/models/llava-1.5-7b-hf
SUBSET=eval/subsets/gqa_200.jsonl
OUT=runs/p2_d
MNS=12                 # max_num_seqs: full continuous batching
LIMIT=100
mkdir -p "$OUT"

run_one () {
  local name=$1; shift
  local out="$OUT/${name}.json"
  if [ -f "$out" ]; then echo "[dval] SKIP $name (exists)"; return; fi
  echo "[dval] $(date +%H:%M:%S) START $name"
  # GPU-settle: wait until GPU memory is mostly free (<4GB) before launching
  for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -lt 4000 ]; then break; fi
    sleep 15
  done
  sleep 20  # extra settle for stale-process teardown
  $PY -m src.serve_bench --model "$MODEL" --benchmark gqa \
    --subset "$SUBSET" --metrics-out "$out" --max-tokens 16 \
    --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0 \
    --limit "$LIMIT" --selector proxy --max-num-seqs "$MNS" "$@" \
    > "$OUT/${name}.log" 2>&1
  rc=$?
  echo "[dval] $(date +%H:%M:%S) DONE $name rc=$rc"
  if [ $rc -ne 0 ]; then echo "[dval] FAIL $name — tail log:"; tail -15 "$OUT/${name}.log"; fi
}

# 1. ADAPTIVE under bursty (the method's headline case)
# Thresholds CALIBRATED to the engine's achievable occ range at c12/short-seq:
# peak occ at full saturation (num_running=12) is ~0.11, so occ_lo/occ_hi span
# 0.02-0.10 to map the observed load swing across r_min->r_max. (With the
# default 0.40/0.70 the controller never sees enough load to leave r_min.)
run_one dval_adaptive_bursty --adaptive --r-min 0.25 --r-max 0.50 \
    --occ-lo 0.02 --occ-hi 0.10 --load-signal kv_occupancy --load-profile bursty

# 2. FIXED r25 under bursty (accuracy-favoring fixed point)
run_one dval_fixed_r25_bursty --pruning-rate 0.25 --load-profile bursty

# 3. FIXED r50 under bursty (throughput-favoring fixed point)
run_one dval_fixed_r50_bursty --pruning-rate 0.50 --load-profile bursty

# 4. ADAPTIVE under constant (sanity: should ~= fixed-r50 at max load)
run_one dval_adaptive_constant --adaptive --r-min 0.25 --r-max 0.50 \
    --occ-lo 0.02 --occ-hi 0.10 --load-signal kv_occupancy --load-profile constant

echo "[dval] ALL DONE $(date +%H:%M:%S) — analyze with scripts/analyze_d.py"

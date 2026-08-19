#!/bin/bash
# P0-2: Qwen2.5-VL-7B OCRBench FULL matched-configuration rerun.
#
# WHY: the paper Table 1 Qwen2.5-VL OCRBench primary cell is NOT iso-config:
#   old RBM (pre) cell ran at max_pixels=4000000 (mean post-merger tokens 229.6)
#   old post cell ran at max_pixels=0 native (mean tokens 282.0).
# This rerun executes BOTH arms with IDENTICAL processor / pixel cap /
# input resolution: max_pixels=4000000, max_model_len 8192, max_num_seqs 4,
# greedy temp=0, selector=l2, max_tokens=32, same sample IDs (ocrbench.jsonl).
#
# NOTE: native resolution (max_pixels=0) makes the qwen2vl pre path fail with
# skip=1000/1000 (verified: runs/full_matrix/j7_qwen2vl_pre_ocrbench_r0.750_full.json.broken_prev),
# so 4M is the ONLY cap where both arms complete; it also matches the pixel cap
# of the RBM cell already published in Table 1.
#
# κ = 0.25 (r=0.75) runs FIRST (primary); then κ = 0.125 (r=0.875).
#
# Outputs -> results/acmmm_final_controls/p0_2/*.json + *.log
set -u
cd /media/disk2/YZX/research/vla
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       VLLM_NO_USAGE_STATS=1 VLLM_USE_MODELSCOPE=False

OUT=/media/disk2/YZX/research/vla/results/acmmm_final_controls/p0_2
mkdir -p $OUT
FAM=qwen2vl
FS=eval/full_splits/ocrbench.jsonl
N=1000
FLAGS="--max-num-seqs 4 --max-model-len 8192 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9 --max-pixels 4000000"

echo "[P0-2] $(date -u '+%F %T') start; waiting for >= 40000 MiB free"
for i in $(seq 1 240); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$FREE" -gt 40000 ] && { echo "[P0-2] GPU free ${FREE} MiB"; break; }
  sleep 30
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$FREE" -le 40000 ] && { echo "[P0-2][ABORT] GPU busy after wait"; exit 1; }

run_cell(){ # mode r tag
  timeout 7200 $PY src/v3_premerger/v3_premerger_runner.py --model-family $FAM \
    --benchmark ocrbench --subset $FS --n $N --r $2 --mode $1 \
    --selector l2 --max-tokens 32 $FLAGS --out $OUT/$3.json > $OUT/$3.log 2>&1
  echo "[P0-2] $3 exit=$? $(tail -1 $OUT/$3.log)"
}
skip_ratio(){ $PY -c "
import json
try:
    d=json.load(open('$OUT/$1.json')); print((d.get('n_skipped') or 0)/max(len(d.get('per_sample') or [1]),1))
except Exception: print(1.0)
"; }
cell(){ # mode r
  local TAG=$(printf "p0_2_qwen2_%s_ocrbench_r%.3f_full" "$1" "$2")
  if [ -s $OUT/$TAG.json ]; then
    SR=$(skip_ratio $TAG)
    if [ "$($PY -c "print(1 if float('$SR')<=0.10 else 0)")" = "1" ]; then
      echo "=== $TAG EXISTS (skip=$SR), skip ==="; return
    fi
    echo "=== $TAG EXISTS but skip=$SR > 0.10 -> retry ==="
  fi
  echo "=== $TAG n=$N mode=$1 r=$2 ==="
  run_cell $1 $2 $TAG
  echo "=== $TAG done (skip=$(skip_ratio $TAG)) ==="
}

# κ=0.25 primary
cell pre  0.75
cell post 0.75
# κ=0.125
cell pre  0.875
cell post 0.875

echo "=== P0-2 DONE $(date -u '+%F %T') ==="

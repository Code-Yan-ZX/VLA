#!/bin/bash
# P0-1: Qwen3-VL-8B pure-stage control on FULL splits.
#
# PURPOSE: remove the feature-depth confound in the headline RBM (pre ranks at
# ViT layer-8 / deepstack[0] input) vs post (ranks at main-merger output) by
# ranking at the main-merger's OWN input (final ViT-block features, "pre-final").
#   pre-final : rank merge-units by L2 at visual.merger INPUT, select k_units
#               from the visual output (deepstack mergers run FULL/untouched).
#   post      : rank merged tokens by L2 at main-merger OUTPUT.
# The ONLY difference is BEFORE vs AFTER the lossy 2x2 main merger.
#
# ISO-CONFIG with the paper's Table 1 Qwen3-VL cells:
#   model Qwen/Qwen3-VL-8B-Instruct, native pixels (max_pixels=0),
#   greedy temp=0, selector=l2, r=0.75 (keep 25%), max_tokens=32,
#   flags identical to runs/full_matrix/j7_full_matrix_vllm.sh:
#     textvqa/ocrbench/gqa: --max-num-seqs 8 --max-model-len 8192 --gpu-memory-utilization 0.9
#     docvqa:               --max-num-seqs 4 --max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9
#
# none arms: NOT re-run; reuse the verified full-split anchors that Table 1
# cites (j7 none textvqa/ocrbench/gqa + r2_same_scope none docvqa), listed in
# results/acmmm_final_controls/p0_1_none_anchors.json.
#
# Outputs -> results/acmmm_final_controls/p0_1/*.json + *.log
set -u
cd /media/disk2/YZX/research/vla
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       VLLM_NO_USAGE_STATS=1 VLLM_USE_MODELSCOPE=False

OUT=/media/disk2/YZX/research/vla/results/acmmm_final_controls/p0_1
mkdir -p $OUT
FAM=qwen3vl
R=0.75
MAXTOK=32
MODES="pre-final post"
FS=eval/full_splits
declare -A JSONL=( [textvqa]=$FS/textvqa_val.jsonl [docvqa]=$FS/docvqa_val.jsonl \
                   [ocrbench]=$FS/ocrbench.jsonl [gqa]=$FS/gqa_testdev.jsonl )
declare -A N=( [textvqa]=5000 [docvqa]=5349 [ocrbench]=1000 [gqa]=12578 )
STD="--max-num-seqs 8 --max-model-len 8192 --gpu-memory-utilization 0.9"
DOC="--max-num-seqs 4 --max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9"

echo "[P0-1] $(date -u '+%F %T') start; waiting for >= 40000 MiB free"
for i in $(seq 1 240); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$FREE" -gt 40000 ] && { echo "[P0-1] GPU free ${FREE} MiB"; break; }
  sleep 30
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$FREE" -le 40000 ] && { echo "[P0-1][ABORT] GPU busy after wait"; exit 1; }

run_cell(){ # mode bench "flags"
  timeout 21600 $PY src/v3_premerger/v3_premerger_runner.py --model-family $FAM \
    --benchmark $2 --subset ${JSONL[$2]} --n ${N[$2]} --r $R --mode $1 \
    --selector l2 --max-tokens $MAXTOK $3 --out $OUT/$4.json > $OUT/$4.log 2>&1
  echo "[P0-1] $4 exit=$? $(tail -1 $OUT/$4.log)"
}
skip_ratio(){ $PY -c "
import json,sys
try:
    d=json.load(open('$OUT/$1.json')); print((d.get('n_skipped') or 0)/max(len(d.get('per_sample') or [1]),1))
except Exception: print(1.0)
"; }
cell(){ # mode bench
  local MODE=$1 B=$2
  local FLAGS=$STD; [ "$B" = "docvqa" ] && FLAGS=$DOC
  local TAG=$(printf "p0_1_qwen3_%s_%s_r%.3f_full" "$MODE" "$B" "$R")
  if [ -s $OUT/$TAG.json ]; then
    SR=$(skip_ratio $TAG)
    if [ "$($PY -c "print(1 if float('$SR')<=0.10 else 0)")" = "1" ]; then
      echo "=== $TAG EXISTS (skip=$SR), skip ==="; return
    fi
    echo "=== $TAG EXISTS but skip=$SR > 0.10 -> retry ==="
  fi
  echo "=== $TAG n=${N[$B]} mode=$MODE ==="
  run_cell $MODE $B "$FLAGS" $TAG
  echo "=== $TAG done (skip=$(skip_ratio $TAG)) ==="
}

for MODE in $MODES; do
  for B in textvqa docvqa ocrbench gqa; do cell $MODE $B; done
done

echo "=== P0-1 DONE $(date -u '+%F %T') ==="

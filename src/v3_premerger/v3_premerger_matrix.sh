#!/usr/bin/env bash
# V3 go/no-go matrix: PRE-merger (C) vs POST-merger (B) vs baseline (A)
# on Qwen3-VL-8B-Instruct, GQA + TextVQA, n=200.
# keep-ratios {0.5,0.25,0.125} of merge-units  <=>  r (prune) {0.5,0.75,0.875}
#   -> final LLM-input tokens {128,64,32} on TextVQA; {~112,56,28} on GQA.
# One fresh process per cell (enforce_eager, clean hooks). ~1 GPU-h total.
set -e
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
OUT=runs/v3_premerger_cells
mkdir -p $OUT
RUN=runs/v3_premerger_runner.py
RS=(0.5 0.75 0.875)   # prune ratios -> keep 50/25/12.5 %

for BENCH in gqa textvqa; do
  SUB=eval/subsets/${BENCH}_200.jsonl
  # (A) baseline
  python $RUN --mode none --r 0.0 --benchmark $BENCH --subset $SUB \
    --n 200 --max-num-seqs 16 --out $OUT/A_${BENCH}.json 2>&1 | tee $OUT/A_${BENCH}.log
  for R in "${RS[@]}"; do
    RR=$(printf "%.3f" $R)
    # (B) post-merger
    python $RUN --mode post --r $R --benchmark $BENCH --subset $SUB \
      --n 200 --max-num-seqs 16 --out $OUT/B_${BENCH}_r${RR}.json 2>&1 | tee $OUT/B_${BENCH}_r${RR}.log
    # (C) pre-merger
    python $RUN --mode pre --r $R --benchmark $BENCH --subset $SUB \
      --n 200 --max-num-seqs 16 --out $OUT/C_${BENCH}_r${RR}.json 2>&1 | tee $OUT/C_${BENCH}_r${RR}.log
  done
done
echo "=== MATRIX DONE ==="

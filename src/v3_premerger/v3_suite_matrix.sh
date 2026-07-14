#!/usr/bin/env bash
# V3 suite matrix: map the workload-conditional pre-vs-post stage effect across
# the benchmark suite. NEW benchmarks: docvqa, mme, mmbench, scienceqa.
# gqa + textvqa are REUSED from runs/v3_premerger_cells (identical settings:
# n=200, max_num_seqs=16, enforce_eager, Qwen3-VL-8B, same scorers) -- no re-run.
#
# Per new benchmark: (A) baseline + (B post / C pre) at r={0.75,0.875}
#   = keep {25, 12.5}% of merge-units. Deep point = 12.5% (decisive).
# One fresh process per cell (enforce_eager + clean hooks). Offline (model cached).
# Order: cheap MC/yesno benchmarks first (validate runner changes), docvqa last.
# NO `set -e`: one failure must not abort the whole campaign.

cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
OUT=runs/v3_premerger_cells
mkdir -p $OUT
RUN=src/v3_premerger/v3_premerger_runner.py
RS=(0.75 0.875)   # keep 25% / 12.5%

for BENCH in mme scienceqa mmbench docvqa; do
  SUB=eval/subsets/${BENCH}_200.jsonl
  if [ ! -f "$SUB" ]; then echo "[skip] no subset $SUB"; continue; fi
  # (A) baseline -- skip if already present (idempotent re-runs)
  if [ ! -f "$OUT/A_${BENCH}.json" ]; then
    python $RUN --mode none --r 0.0 --benchmark $BENCH --subset $SUB \
      --n 200 --max-num-seqs 16 --out $OUT/A_${BENCH}.json \
      > $OUT/A_${BENCH}.log 2>&1 || echo "[fail] A_${BENCH}"
  fi
  for R in "${RS[@]}"; do
    RR=$(printf "%.3f" $R)
    if [ ! -f "$OUT/B_${BENCH}_r${RR}.json" ]; then
      python $RUN --mode post --r $R --benchmark $BENCH --subset $SUB \
        --n 200 --max-num-seqs 16 --out $OUT/B_${BENCH}_r${RR}.json \
        > $OUT/B_${BENCH}_r${RR}.log 2>&1 || echo "[fail] B_${BENCH}_r${RR}"
    fi
    if [ ! -f "$OUT/C_${BENCH}_r${RR}.json" ]; then
      python $RUN --mode pre --r $R --benchmark $BENCH --subset $SUB \
        --n 200 --max-num-seqs 16 --out $OUT/C_${BENCH}_r${RR}.json \
        > $OUT/C_${BENCH}_r${RR}.log 2>&1 || echo "[fail] C_${BENCH}_r${RR}"
    fi
  done
  echo "=== $BENCH DONE ==="
done
echo "=== SUITE MATRIX DONE ==="

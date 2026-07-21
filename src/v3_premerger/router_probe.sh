#!/usr/bin/env bash
# Task 4 Phase 4a — Collect per-sample data for router probe.
# Run ONLY when GPU is free (needs ~41GB for Qwen3-VL-8B vLLM).
# Serial: ~4 × 20-40s = ~2-3 min total, well under 1 GPU·h.
set -euo pipefail

source /home/dell/miniconda3/etc/profile.d/conda.sh
conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1

OUTDIR=runs/v3_router_probe
mkdir -p "$OUTDIR"

COMMON="--r 0.75 --selector l2 --n 200 --max-num-seqs 16"

for mode in pre post; do
  for bm in textvqa gqa; do
    name="${mode}_${bm}_r0.750_l2_n200"
    echo "====== Running: $name ======"
    python src/v3_premerger/v3_premerger_runner.py \
      --mode "$mode" \
      --benchmark "$bm" \
      --subset "eval/subsets/${bm}_200.jsonl" \
      --out "${OUTDIR}/${name}.json" \
      $COMMON
    echo "====== Done: $name ======"
  done
done

echo ""
echo "All 4 runs complete. Now run analysis:"
echo "  python src/v3_premerger/router_probe.py"

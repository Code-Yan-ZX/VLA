#!/usr/bin/env bash
# Mechanism verification pipeline (drafts/v3_merger_aware_design.md M1-M3).
# Sequential (1x A40, shared box): (1) GPU capture of compact per-image
# pre/post L2 scores + Sobel edge for a deterministic 64-image sample per
# benchmark; (2) M3 ranking-swap control cells (textvqa + docvqa, n=200);
# (3) CPU analysis (M1/M2 aggregation + figures + stats json).
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
mkdir -p runs/v3_merger_aware

echo "=== STEP 1/3: M1/M2 GPU capture (bench=all, n=64, seed=0) ==="
python scripts/mechanism_token_survival.py --mode capture --bench all \
  --n 64 --seed 0 > runs/v3_merger_aware/capture.log 2>&1 \
  && echo "=== STEP 1 DONE ===" || echo "!!! STEP 1 FAILED (capture.log) !!!"

echo "=== STEP 2/3: M3 swap cells (GPU) ==="
bash src/v3_premerger/v3_swap_control.sh > runs/v3_merger_aware/swap_master.log 2>&1
echo "=== STEP 2 DONE ==="

echo "=== STEP 3/3: M1/M2 analyze (CPU) ==="
python scripts/mechanism_token_survival.py --mode analyze \
  > runs/v3_merger_aware/analyze.log 2>&1 \
  && echo "=== STEP 3 DONE ===" || echo "!!! STEP 3 FAILED (analyze.log) !!!"

echo "=== MECHANISM VERIFICATION PIPELINE COMPLETE ==="

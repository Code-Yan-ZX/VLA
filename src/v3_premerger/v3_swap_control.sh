#!/usr/bin/env bash
# M3 ranking-swap causal control (drafts/v3_merger_aware_design.md §2).
#
# --mode post --mask-ranking swap runs the POST forward path (everything
# merged, numerically untouched) but SELECTS the kept 2x2 merge-units with the
# PRE ranking (deepstack[0]-input unit L2 scores, computed exactly as pre mode).
# Because a kept unit's merged token is IDENTICAL at either stage (unit
# equivalence), this must reproduce PRE-standard accuracy almost exactly =>
# the pre>post accuracy gap is 100% a RANKING effect, forward path held
# constant. This is the shared-mask control the mechanism section needs.
#
# Invocation mirrors src/v3_premerger/v3_rescore_rerun.sh (headline cells):
# short-answer subsets, --selector l2, seed 0, enforce_eager (runner default),
# n=200; DocVQA big-document safety: --max-num-batched-tokens 32768 +
# --max-pixels 1500000 + --max-num-seqs 4. One fresh process per cell.
# NO `set -e` (one failure must not abort the rest).
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
OUT=runs/v3_merger_aware/swap
mkdir -p $OUT
RUN=src/v3_premerger/v3_premerger_runner.py
R=0.75

run_cell() {
  local out="$1"; shift
  echo "====== RUN: $(basename "$out") ======"
  python $RUN "$@" --out "$out" > "${out%.json}.log" 2>&1 \
    && echo "====== DONE: $(basename "$out") ======" \
    || echo "!!!!!! FAIL: $(basename "$out") (see ${out%.json}.log) !!!!!!"
}

run_cell $OUT/swap_textvqa_r0.750_l2_n200.json \
  --mode post --mask-ranking swap --r $R --selector l2 --benchmark textvqa \
  --subset eval/subsets/textvqa_200.jsonl --n 200 --seed 0 \
  --max-model-len 32768 --max-num-seqs 16

run_cell $OUT/swap_docvqa_r0.750_l2_n200.json \
  --mode post --mask-ranking swap --r $R --selector l2 --benchmark docvqa \
  --subset eval/subsets/docvqa_200.jsonl --n 200 --seed 0 \
  --max-model-len 32768 --max-num-batched-tokens 32768 --max-pixels 1500000 \
  --max-num-seqs 4

echo "=== M3 SWAP CELLS DONE ==="

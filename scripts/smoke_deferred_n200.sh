#!/usr/bin/env bash
# Deferred-RBM n=200 gate: smoke test (5-10 samples, both arms) BEFORE the full
# runs.  Runs the REAL main() path (no replication) on tiny subsets for:
#   arm A: --mode rankbridge --rb-rho 1.0 (deferred RBM, K=3 keep 25%)
#   arm B: --mode pre --r-pre 0.25        (immediate RBM, same keep set)
# Outputs runs/deferred_rbm/smoke_*.json; then the checker script verifies:
#   (1) layers 0-2 keep the FULL native post-merge visual token count,
#   (2) pruning fires exactly at layer K=3,
#   (3) post-layer-3 visual count == target 25%,
#   (4) deferred kept indices == immediate-RBM top-25% indices per sample,
#   (5) text tokens / ptid / mask / KV alignment (L_after == n_text + kept),
#   (6) keep-25% not drop-25% (kept == 0.25 * full).
# It also re-checks runner INVARIANCE: smoke output == dev n=64 / parent pre
# JSONs on the overlapping samples (proves the diag-only runner change is
# behavior-neutral).
set -euo pipefail
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=python; HF=src/v3_premerger/baselines_hf.py; OUT=runs/deferred_rbm
MODEL=Qwen/Qwen3-VL-8B-Instruct
# textvqa/docvqa n=8, ocrbench/gqa n=8 (small smoke; all share the 200-lock)
mp() { [ "$1" = docvqa ] && echo 600000 || echo 0; }
run_cell() { # run_cell <tag> <bench> <mode-args...>
  local tag=$1 bench=$2; shift 2
  local J=$OUT/smoke_${tag}_${bench}_n8.json
  echo "[smoke] $tag/$bench"
  $PY $HF "$@" --model $MODEL --benchmark $bench \
    --subset eval/subsets/${bench}_200.jsonl --n 8 --seed 0 --fastv-k 3 \
    --max-pixels "$(mp $bench)" --out $J \
    > $OUT/smoke_${tag}_${bench}_n8.log 2>&1 || { echo "[FAIL] $tag/$bench"; tail -8 $OUT/smoke_${tag}_${bench}_n8.log; exit 1; }
}
for B in textvqa docvqa ocrbench gqa; do
  run_cell deferred $B --mode rankbridge --r 0.75 --rb-fuse quota --rb-rho 1.0
  run_cell pre       $B --mode pre       --r-pre 0.25
done
echo "[smoke] all cells done -> checks"
$PY scripts/smoke_deferred_n200_check.py || { echo "[FAIL] smoke checks"; exit 1; }

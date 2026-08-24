#!/usr/bin/env bash
# Deferred-RBM fixed-lifetime K sweep (phase 2, task 2026-08-24).
# Same locked method (rankbridge quota rho=1.0, keep 25%, Qwen3-VL-8B, seed 0,
# n=200, same sample IDs) -- the ONLY variable is the deletion layer K.
#   K=0 : immediate-RBM (pre mode) -- re-run here ONLY for efficiency metrics +
#         parent-reproducibility; accuracy is reused from runs/cascade/gate_pre25_*.
#   K=1,5,8 : new Deferred runs (--fastv-k K).
#   K=3 : re-run ONLY for efficiency + invariance check; accuracy reused from
#         runs/deferred_rbm/locked_deferred_*_n200.json (round-1 gate).
set -euo pipefail
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=python; HF=src/v3_premerger/baselines_hf.py; OUT=runs/deferred_rbm
MODEL=Qwen/Qwen3-VL-8B-Instruct; N=200
mp() { [ "$1" = docvqa ] && echo 600000 || echo 0; }
ready() { $PY -c "import json,sys; d=json.load(open('$1')); sys.exit(0 if len(d.get('per_sample',[]))>=200 and d.get('n_skipped',0)<=50 else 1)" 2>/dev/null; }
cell_k() { # cell_k <K> <bench>
  local K=$1 bench=$2
  local J=$OUT/sweep_deferred_K${K}_${bench}_n200.json
  ready "$J" && { echo "[skip] sweep_deferred_K${K}_${bench}"; return 0; }
  echo "[run ] sweep_deferred_K${K}_${bench}"
  $PY $HF --mode rankbridge --r 0.75 --rb-fuse quota --rb-rho 1.0 \
    --fastv-k $K --model $MODEL --benchmark $bench \
    --subset eval/subsets/${bench}_200.jsonl --n $N --seed 0 \
    --max-pixels "$(mp $bench)" --out $J \
    > $OUT/sweep_deferred_K${K}_${bench}_n200.log 2>&1 \
    && echo "[done] sweep_deferred_K${K}_${bench}" \
    || { echo "[FAIL] sweep_deferred_K${K}_${bench}"; tail -6 $OUT/sweep_deferred_K${K}_${bench}_n200.log; return 1; }
}
cell_pre() { # cell_pre <bench>  (K=0 immediate-RBM, efficiency + repro re-run)
  local bench=$1
  local J=$OUT/sweep_pre_${bench}_n200.json
  ready "$J" && { echo "[skip] sweep_pre_${bench}"; return 0; }
  echo "[run ] sweep_pre_${bench}"
  $PY $HF --mode pre --r-pre 0.25 --model $MODEL --benchmark $bench \
    --subset eval/subsets/${bench}_200.jsonl --n $N --seed 0 \
    --max-pixels "$(mp $bench)" --out $J \
    > $OUT/sweep_pre_${bench}_n200.log 2>&1 \
    && echo "[done] sweep_pre_${bench}" \
    || { echo "[FAIL] sweep_pre_${bench}"; tail -6 $OUT/sweep_pre_${bench}_n200.log; return 1; }
}
# new K
for K in 1 5 8; do for B in textvqa docvqa ocrbench gqa; do cell_k $K $B || exit 1; done; done
# K=3 re-run (efficiency + invariance)
for B in textvqa docvqa ocrbench gqa; do cell_k 3 $B || exit 1; done
# K=0 immediate-RBM re-run (efficiency + parent repro)
for B in textvqa docvqa ocrbench gqa; do cell_pre $B || exit 1; done
echo "################ SWEEP CELLS DONE ################"
echo "git: $(git rev-parse --short HEAD) on $(git branch --show-current)"

#!/usr/bin/env bash
# MALT goal-mode Gate C: locked n=200 confirmation of the best candidate vs
# MALT-1 (task 2026-08-25).
# Candidate (from Phase-1 mechanism finding): H0n = native-coordinate
# immediate-RBM (`--mode pre --r-pre 0.25 --mrope native`) -- the strict
# Pareto improvement: captures the MALT-1 gain at K=0 (minimum) compute.
# Reference (MALT-1): runs/deferred_rbm/sweep_deferred_K1_*_n200.json.
set -euo pipefail
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=python; HF=src/v3_premerger/baselines_hf.py; OUT=runs/malt_goal_mode
MODEL=Qwen/Qwen3-VL-8B-Instruct
mp() { [ "$1" = docvqa ] && echo 600000 || echo 0; }
# symlink the MALT-1 n=200 reference cells into the Gate C namespace
for B in textvqa docvqa ocrbench gqa; do
  ln -sf /media/disk2/YZX/research/vla/runs/deferred_rbm/sweep_deferred_K1_${B}_n200.json \
         $OUT/h1_${B}_n200.json
done
run_cell() { # run_cell <tag> <bench>
  local tag=$1 bench=$2
  local J=$OUT/${tag}_${bench}_n200.json
  if [ -f "$J" ] && [ ! -L "$J" ] && \
     $PY -c "import json,sys; d=json.load(open('$J')); sys.exit(0 if len(d.get('per_sample',[]))>=200 and d.get('n_skipped',0)<=50 else 1)" 2>/dev/null; then
    echo "[skip] ${tag}_${bench}"; return 0; fi
  echo "[run ] ${tag}_${bench}"
  $PY $HF --mode pre --r-pre 0.25 --mrope native --model $MODEL \
    --benchmark $bench --subset eval/subsets/${bench}_200.jsonl \
    --n 200 --seed 0 --max-tokens 32 --max-pixels "$(mp $bench)" --out $J \
    > $OUT/${tag}_${bench}_n200.log 2>&1 \
    && echo "[done] ${tag}_${bench}" \
    || { echo "[FAIL] ${tag}_${bench}"; tail -8 $OUT/${tag}_${bench}_n200.log; return 1; }
}
mkdir -p $OUT
for B in textvqa docvqa ocrbench gqa; do
  run_cell h0n $B || exit 1
done
echo "################ GATE C CELLS DONE ################"
echo "git: $(git rev-parse --short HEAD) on $(git branch --show-current)"

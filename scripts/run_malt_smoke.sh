#!/usr/bin/env bash
# MALT goal-mode Gate A: smoke (n=8 x 4 benches) on ALL causal-ablation arms.
# Arms: H0=pre(immediate), H1=deferred k1, H2=no_text_read, H3=no_anchor_read,
# H4=kv_only, no_both.  Same locked subset/seed as the original deferred smoke so
# H1's fresh output must reproduce smoke_deferred_*_n8.json bit-for-bit (proves
# the ablation plumbing did not perturb the identity path).
set -euo pipefail
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=python; HF=src/v3_premerger/baselines_hf.py; OUT=runs/malt_goal_mode
MODEL=Qwen/Qwen3-VL-8B-Instruct
mp() { [ "$1" = docvqa ] && echo 600000 || echo 0; }
run_cell() { # run_cell <tag> <bench> <extra-args...>
  local tag=$1 bench=$2; shift 2
  local J=$OUT/smoke_${tag}_${bench}_n8.json
  [ -f "$J" ] && { echo "[skip] $tag/$bench"; return 0; }
  echo "[smoke] $tag/$bench"
  $PY $HF "$@" --model $MODEL --benchmark $bench \
    --subset eval/subsets/${bench}_200.jsonl --n 8 --seed 0 --max-tokens 32 \
    --max-pixels "$(mp $bench)" --out $J \
    > $OUT/smoke_${tag}_${bench}_n8.log 2>&1 \
    || { echo "[FAIL] $tag/$bench"; tail -8 $OUT/smoke_${tag}_${bench}_n8.log; exit 1; }
}
mkdir -p $OUT
for B in textvqa docvqa ocrbench gqa; do
  run_cell h0 $B --mode pre --r-pre 0.25
  run_cell h1 $B --mode rankbridge --r 0.75 --rb-fuse quota --rb-rho 1.0 --fastv-k 1
  run_cell h2 $B --mode rankbridge --r 0.75 --rb-fuse quota --rb-rho 1.0 --fastv-k 1 --malt-ablate no_text_read
  run_cell h3 $B --mode rankbridge --r 0.75 --rb-fuse quota --rb-rho 1.0 --fastv-k 1 --malt-ablate no_anchor_read
  run_cell h4 $B --mode rankbridge --r 0.75 --rb-fuse quota --rb-rho 1.0 --fastv-k 1 --malt-ablate kv_only
  run_cell nb $B --mode rankbridge --r 0.75 --rb-fuse quota --rb-rho 1.0 --fastv-k 1 --malt-ablate no_both
done
echo "################ MALT SMOKE CELLS DONE ################"
$PY scripts/malt_smoke_check.py || { echo "[FAIL] malt smoke checks"; exit 1; }
echo "[ok] Gate A smoke: all checks passed"

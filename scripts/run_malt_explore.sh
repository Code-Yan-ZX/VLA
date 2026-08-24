#!/usr/bin/env bash
# MALT goal-mode Phase-1 exploration (Gate B candidates): n=64 x 4 benches on
# the disjoint explore64 manifest, ALL causal-ablation arms.
#   H0=pre(immediate)  H1=deferred k1  H2=no_text_read  H3=no_anchor_read
#   H4=kv_only  no_both
# Same locked method config as MALT-1 (rankbridge rho=1.0, keep 25%, seed 0,
# greedy, max-tokens 32); the ONLY variation is the causal ablation.
set -euo pipefail
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=python; HF=src/v3_premerger/baselines_hf.py; OUT=runs/malt_goal_mode
MODEL=Qwen/Qwen3-VL-8B-Instruct
mp() { [ "$1" = docvqa ] && echo 600000 || echo 0; }
run_cell() { # run_cell <tag> <bench> <extra-args...>
  local tag=$1 bench=$2; shift 2
  local J=$OUT/explore_${tag}_${bench}_n64.json
  if [ -f "$J" ] && $PY -c "import json,sys; d=json.load(open('$J')); sys.exit(0 if len(d.get('per_sample',[]))>=64 and d.get('n_skipped',0)<=4 else 1)" 2>/dev/null; then
    echo "[skip] explore_${tag}_${bench}"; return 0; fi
  echo "[run ] explore_${tag}_${bench}"
  $PY $HF "$@" --model $MODEL --benchmark $bench \
    --subset eval/subsets/${bench}_explore64.jsonl --n 64 --seed 0 --max-tokens 32 \
    --max-pixels "$(mp $bench)" --out $J \
    > $OUT/explore_${tag}_${bench}_n64.log 2>&1 \
    && echo "[done] explore_${tag}_${bench}" \
    || { echo "[FAIL] explore_${tag}_${bench}"; tail -8 $OUT/explore_${tag}_${bench}_n64.log; return 1; }
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
echo "################ MALT EXPLORE CELLS DONE ################"
echo "git: $(git rev-parse --short HEAD) on $(git branch --show-current)"

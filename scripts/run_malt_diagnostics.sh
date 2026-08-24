#!/usr/bin/env bash
# MALT goal-mode Phase-1 follow-up diagnostics (decided from the Phase-1 table).
# So far: h0n = immediate-RBM with NATIVE survivor mrope coords (isolates the
# positional confound -- H0 vllm-mimic vs the deferred arms' native coords).
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
  [ -f "$J" ] && { echo "[skip] explore_${tag}_${bench}"; return 0; }
  echo "[run ] explore_${tag}_${bench}"
  $PY $HF "$@" --model $MODEL --benchmark $bench \
    --subset eval/subsets/${bench}_explore64.jsonl --n 64 --seed 0 --max-tokens 32 \
    --max-pixels "$(mp $bench)" --out $J \
    > $OUT/explore_${tag}_${bench}_n64.log 2>&1 \
    && echo "[done] explore_${tag}_${bench}" \
    || { echo "[FAIL] explore_${tag}_${bench}"; tail -8 $OUT/explore_${tag}_${bench}_n64.log; return 1; }
}
mkdir -p $OUT
# H0-native: immediate-RBM with native survivor coordinates (control for the
# positional confound between pre(vllm-mimic) and the deferred native-coord arms)
for B in textvqa docvqa ocrbench gqa; do
  run_cell h0n $B --mode pre --r-pre 0.25 --mrope native
done
echo "################ MALT DIAGNOSTIC CELLS DONE ################"

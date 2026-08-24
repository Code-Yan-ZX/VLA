#!/usr/bin/env bash
# Deferred-RBM n=200 GATE (life-or-death verification, task 2026-08-24):
# scale the n=64 dev signal to n=200 on the 4 locked benchmarks.
# Method (LOCKED, no parameter search):
#   rankbridge quota rho=1.0 at K=3, keep 25% -- kept identities == plain-RBM
#   pre top-25% (bit-identical), but ALL native-merger visual tokens run
#   through decoder layers 0..2 and are deleted only at layer 3.
# Parents (reused, config/ID parity verified):
#   Full  = runs/rankbridge/locked_none_*_n200.json
#   RBM   = runs/cascade/gate_pre25_*.json
#   FastV = runs/rankbridge/locked_fst3_*_n200.json
set -euo pipefail
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=python; HF=src/v3_premerger/baselines_hf.py; OUT=runs/deferred_rbm
MODEL=Qwen/Qwen3-VL-8B-Instruct; N=200
mp() { [ "$1" = docvqa ] && echo 600000 || echo 0; }
cell() {  # cell <bench>
  local bench=$1
  local J=$OUT/locked_deferred_${bench}_n200.json
  if [ -f "$J" ] && $PY -c "import json,sys; d=json.load(open('$J')); sys.exit(0 if len(d.get('per_sample',[]))>=200 and d.get('n_skipped',0)<=50 else 1)" 2>/dev/null; then
    echo "[skip] locked_deferred_${bench}"; return 0; fi
  echo "[run ] locked_deferred_${bench}"
  $PY $HF --mode rankbridge --r 0.75 --rb-fuse quota --rb-rho 1.0 \
    --fastv-k 3 --model $MODEL --benchmark $bench \
    --subset eval/subsets/${bench}_200.jsonl --n $N --seed 0 \
    --max-pixels "$(mp $bench)" --out $J \
    > $OUT/locked_deferred_${bench}_n200.log 2>&1 \
    && echo "[done] locked_deferred_${bench}" \
    || { echo "[FAIL] locked_deferred_${bench}"; tail -8 $OUT/locked_deferred_${bench}_n200.log; return 1; }
}
for B in textvqa docvqa ocrbench gqa; do cell $B || exit 1; done
echo "################ LOCKED CELLS DONE ################"
echo "git: $(git rev-parse --short HEAD) on $(git branch --show-current)"

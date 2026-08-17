#!/usr/bin/env bash
# D1 pipeline: after capture completes -> train scorers (CPU) -> build held-out
# slices -> launch GPU eval (serial). Guarded: each step only if inputs exist.
set -e
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
BENCHES=(textvqa docvqa gqa)

echo "[d1] 1/3 train scorers (CPU)"
for b in "${BENCHES[@]}"; do
  npz=runs/v3_merger_aware/survival_capture_lrn200/$b.npz
  [[ -f "$npz" ]] || { echo "[d1] MISSING $npz"; exit 1; }
done
$PY scripts/train_learned_scorer.py "${BENCHES[@]}" \
  --out-dir runs/learned_scorer

echo "[d1] 2/3 build held-out slices"
$PY scripts/build_learned_heldout.py 200

echo "[d1] 3/3 launch GPU eval (background)"
bash scripts/run_d1_eval.sh > experiments/d1_eval.log 2>&1 &
echo "[d1] eval launched pid $!"

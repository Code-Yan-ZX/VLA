#!/usr/bin/env bash
# Cross-benchmark sensitivity of --hybrid-text-frac (supplement to the gate
# battery): the gate's tf=0.5 was tuned on textvqa ONLY. To report the sweep
# honestly across all three gate benches (and see whether the gate failure, if
# any, is tf-specific), run the other two fracs on ocrbench + gqa @n=100.
# Same invocations as src/v3_premerger/v3_hybrid_gate.sh stage 3a.
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
OUT=runs/v3_merger_aware/hybrid_gate
RUN=src/v3_premerger/v3_premerger_runner.py
R=0.75

run_cell() {
  local out="$1"; shift
  if [ -s "$out" ]; then
    echo "====== SKIP (exists): $(basename "$out") ======"
    return 0
  fi
  echo "====== RUN: $(basename "$out") ======"
  python $RUN "$@" --out "$out" > "${out%.json}.log" 2>&1 \
    && echo "====== DONE: $(basename "$out") ======" \
    || echo "!!!!!! FAIL: $(basename "$out") (see ${out%.json}.log) !!!!!!"
}

echo "=== SENSITIVITY SWEEP starting $(date) ==="
for TF in 0.0 1.0; do
  run_cell $OUT/hybrid_ocrbench_r0.750_l2_tf${TF}_n100.json \
    --mode hybrid --hybrid-text-frac $TF --save-unit-scores \
    --r $R --selector l2 --benchmark ocrbench \
    --subset eval/subsets/ocrbench_200.jsonl --n 100 --seed 0 \
    --max-model-len 32768 --max-num-batched-tokens 32768 \
    --max-pixels 1500000 --max-num-seqs 4
  run_cell $OUT/hybrid_gqa_r0.750_l2_tf${TF}_n100.json \
    --mode hybrid --hybrid-text-frac $TF --save-unit-scores \
    --r $R --selector l2 --benchmark gqa \
    --subset eval/subsets/gqa_200.jsonl --n 100 --seed 0 \
    --max-model-len 32768 --max-num-seqs 16
done
echo "=== SENSITIVITY SWEEP COMPLETE $(date) ==="

#!/usr/bin/env bash
# Re-run the HEADLINE cells (TextVQA + DocVQA) under the FIXED short-answer
# prompt (subsets now carry "\nAnswer the question using a single word or
# phrase." in each question -- see scripts/fix_shortanswer_subsets.py).
#
# Cells: bench in {textvqa, docvqa} x mode in {none(baseline), post, pre},
#   r=0.75 (keep=25%), n=200, --selector l2, enforce_eager (hardcoded in the
#   runner). seed=0 == the router_probe default (router_probe.sh passed no
#   --seed, so it used the runner default 0) -- kept identical for comparability.
#
# DocVQA big-document safety (canonical config, cf. v3_sota_matrix.sh:56 /
#   v3_sota_matrix_followup.sh:46): --max-num-batched-tokens 32768 (also gates
#   the V1 mm-encoder cache budget) + --max-pixels 1500000 (cap pre-merger
#   tokens) + --max-num-seqs 4 (low concurrency) -> avoids OOM on huge docs.
#
# One fresh process per cell (enforce_eager + clean hooks). Per-sample preds
# are saved by the runner. NO `set -e` (one failure must not abort the rest).
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
OUT=runs/v3_merger_aware/rescore_rerun
mkdir -p $OUT
RUN=src/v3_premerger/v3_premerger_runner.py
R=0.75
RR=$(printf "%.3f" $R)

# run_cell <out_json> <extra runner args...>
run_cell() {
  local out="$1"; shift
  echo "====== RUN: $(basename "$out") ======"
  python $RUN "$@" --out "$out" > "${out%.json}.log" 2>&1 \
    && echo "====== DONE: $(basename "$out") ======" \
    || echo "!!!!!! FAIL: $(basename "$out") (see ${out%.json}.log) !!!!!!"
}

for BENCH in textvqa docvqa; do
  SUB=eval/subsets/${BENCH}_200.jsonl
  if [ "$BENCH" = "docvqa" ]; then
    EXTRA="--max-num-batched-tokens 32768 --max-pixels 1500000 --max-num-seqs 4"
  else
    EXTRA="--max-num-seqs 16"
  fi
  # baseline (mode none; runner forces r=0.0 internally)
  run_cell $OUT/none_${BENCH}_r0.000_l2_n200.json \
    --mode none --r 0.0 --selector l2 --benchmark $BENCH --subset $SUB \
    --n 200 --seed 0 --max-model-len 32768 $EXTRA
  # post-merger
  run_cell $OUT/post_${BENCH}_r${RR}_l2_n200.json \
    --mode post --r $R --selector l2 --benchmark $BENCH --subset $SUB \
    --n 200 --seed 0 --max-model-len 32768 $EXTRA
  # pre-merger
  run_cell $OUT/pre_${BENCH}_r${RR}_l2_n200.json \
    --mode pre --r $R --selector l2 --benchmark $BENCH --subset $SUB \
    --n 200 --seed 0 --max-model-len 32768 $EXTRA
  echo "----- $BENCH complete -----"
done
echo "=== RESCORE RERUN ALL CELLS DONE ==="

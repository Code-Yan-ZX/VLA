#!/usr/bin/env bash
# V3 TIGHTENING CAMPAIGN (self-contained, detached).
# Tightens the three stage-effect claims on the KEY benchmarks (DocVQA,
# TextVQA, GQA) with three tests:
#   (a) n=500 L2 selector, A baseline + B post / C pre @ {0.75, 0.875}
#       -> main n=500 numbers (replaces n=200 suite for these benches).
#   (c) seed=1,2 repeats on the 4 headline cells (DocVQA/TextVQA x {B@0.75,
#       C@0.75}) -> mean+/-std of the C-B gap (seed 0 == the (a) cell).
#   (b) --selector attn (global-centroid-distance proxy) on B/C @0.75 ->
#       robustness: does the pre>post stage effect survive a different selector?
# Priority if GPU-time-tight: (a) > (c) > (b).
#
# DocVQA NOTE: only 200 DocVQA images are downloaded locally and the box is
# offline, so --n 500 is effective n=200 for DocVQA (the runner slices
# [:n]). TextVQA/GQA run the full 500. Also: --max-pixels 1500000 is passed
# for DocVQA to cap pre-merger token count <= ~5859, fixing the encoder-cache
# crash seen on ~16k-token document images.
#
# NO `set -e` (one failure must not abort). Idempotent: skips cells whose
# .json already exists. One fresh process per cell (enforce_eager + hooks).

cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
OUT=runs/v3_tighten_cells
mkdir -p $OUT
RUN=src/v3_premerger/v3_premerger_runner.py
ANZ=src/v3_premerger/v3_tighten_analyze.py
MPX=1500000   # DocVQA max_pixels cap (~5859 pre-merger tokens)

# run_cell <out_json> <extra runner args...>
run_cell() {
  local out="$1"; shift
  if [ -f "$out" ]; then echo "[skip] $out"; return 0; fi
  echo "[run]  $out"
  python $RUN "$@" --out "$out" > "${out%.json}.log" 2>&1 \
    || echo "[fail] $out (see ${out%.json}.log)"
}

# --------------------------------------------------------------------------- #
# SMOKE TESTS (1 image each) -- verify the code changes run end-to-end before
# spending GPU on full cells. Failures are logged, not fatal.
# --------------------------------------------------------------------------- #
echo "=== SMOKE: attn selector (gqa, pre, n=1) ==="
run_cell $OUT/smoke_pre_gqa_attn_n1.json \
  --mode pre --r 0.75 --selector attn --benchmark gqa \
  --subset eval/subsets/gqa_500.jsonl --n 1 --max-num-seqs 4

echo "=== SMOKE: max-pixels DocVQA crash fix (post, r=0.875, n=1) ==="
run_cell $OUT/smoke_post_docvqa_r0875_mpix_n1.json \
  --mode post --r 0.875 --selector l2 --benchmark docvqa \
  --subset eval/subsets/docvqa_200.jsonl --n 1 --max-num-seqs 4 \
  --max-model-len 32768 --max-pixels $MPX

# --------------------------------------------------------------------------- #
# (a) n=500 L2, 3 benchmarks x {A, B@0.75, C@0.75, B@0.875, C@0.875}.
# --------------------------------------------------------------------------- #
echo "=== (a) n=500 L2 key benchmarks ==="
for BENCH in docvqa textvqa gqa; do
  case $BENCH in
    docvqa) SUB=eval/subsets/docvqa_200.jsonl; SEQ=4;  LEN=32768; EXTRA="--max-pixels $MPX" ;;
    *)      SUB=eval/subsets/${BENCH}_500.jsonl; SEQ=16; LEN=32768; EXTRA="" ;;
  esac
  # (A) baseline
  run_cell $OUT/A_${BENCH}_l2_n500.json \
    --mode none --r 0.0 --selector l2 --benchmark $BENCH --subset $SUB \
    --n 500 --max-num-seqs $SEQ --max-model-len $LEN $EXTRA
  for R in 0.75 0.875; do
    RR=$(printf "%.3f" $R)
    # (B) post-merger
    run_cell $OUT/B_${BENCH}_r${RR}_l2_n500.json \
      --mode post --r $R --selector l2 --benchmark $BENCH --subset $SUB \
      --n 500 --max-num-seqs $SEQ --max-model-len $LEN $EXTRA
    # (C) pre-merger
    run_cell $OUT/C_${BENCH}_r${RR}_l2_n500.json \
      --mode pre --r $R --selector l2 --benchmark $BENCH --subset $SUB \
      --n 500 --max-num-seqs $SEQ --max-model-len $LEN $EXTRA
  done
  echo "--- $BENCH (a) done ---"
done

# --------------------------------------------------------------------------- #
# (c) seed=1,2 repeats on the 4 headline cells:
#     DocVQA/TextVQA x {B post@0.75, C pre@0.75}, L2, n=500.
# (seed 0 is the (a) cell: {B,C}_${BENCH}_r0.750_l2_n500.json.)
# --------------------------------------------------------------------------- #
echo "=== (c) seed repeats (mean+/-std) ==="
for BENCH in docvqa textvqa; do
  case $BENCH in
    docvqa) SUB=eval/subsets/docvqa_200.jsonl; SEQ=4;  LEN=32768; EXTRA="--max-pixels $MPX" ;;
    *)      SUB=eval/subsets/${BENCH}_500.jsonl; SEQ=16; LEN=32768; EXTRA="" ;;
  esac
  for SEED in 1 2; do
    run_cell $OUT/B_${BENCH}_r0.750_l2_n500_seed${SEED}.json \
      --mode post --r 0.75 --selector l2 --benchmark $BENCH --subset $SUB \
      --n 500 --max-num-seqs $SEQ --max-model-len $LEN --seed $SEED $EXTRA
    run_cell $OUT/C_${BENCH}_r0.750_l2_n500_seed${SEED}.json \
      --mode pre --r 0.75 --selector l2 --benchmark $BENCH --subset $SUB \
      --n 500 --max-num-seqs $SEQ --max-model-len $LEN --seed $SEED $EXTRA
  done
  echo "--- $BENCH (c) done ---"
done

# --------------------------------------------------------------------------- #
# (b) attn selector on key cells: DocVQA/TextVQA/GQA x {B post, C pre} @0.75.
# --------------------------------------------------------------------------- #
echo "=== (b) attn-selector robustness ==="
for BENCH in docvqa textvqa gqa; do
  case $BENCH in
    docvqa) SUB=eval/subsets/docvqa_200.jsonl; SEQ=4;  LEN=32768; EXTRA="--max-pixels $MPX" ;;
    *)      SUB=eval/subsets/${BENCH}_500.jsonl; SEQ=16; LEN=32768; EXTRA="" ;;
  esac
  run_cell $OUT/B_${BENCH}_r0.750_attn_n500.json \
    --mode post --r 0.75 --selector attn --benchmark $BENCH --subset $SUB \
    --n 500 --max-num-seqs $SEQ --max-model-len $LEN $EXTRA
  run_cell $OUT/C_${BENCH}_r0.750_attn_n500.json \
    --mode pre --r 0.75 --selector attn --benchmark $BENCH --subset $SUB \
    --n 500 --max-num-seqs $SEQ --max-model-len $LEN $EXTRA
  echo "--- $BENCH (b) done ---"
done

# --------------------------------------------------------------------------- #
# ANALYZE + DONE.
# --------------------------------------------------------------------------- #
echo "=== ANALYZE ==="
python $ANZ --dir $OUT > $OUT/_analyze.log 2>&1 || echo "[fail] analyzer (see $OUT/_analyze.log)"
cat $OUT/_analyze.log
echo "=== V3 TIGHTEN DONE ==="

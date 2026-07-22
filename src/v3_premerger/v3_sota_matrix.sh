#!/usr/bin/env bash
# V3 SOTA decision matrix (2026-07-22): NEW OCR-heavy benchmarks ChartQA +
# OCRBench (n=200, subsets built concurrently by builder agent) + GQA
# VisionZip-style fill cells. NO re-run of pre-vs-post known results.
#
# Cells:
#   Phase 1 (subsets ready): GQA VisionZip-style (dom+ctx, post-merger) @25/12.5%
#       -- suite-consistent config (mns 16), matches existing gqa pre/post cells.
#   Phase 2 (waits <=100min for subsets): per bench in {chartqa, ocrbench}:
#       A baseline (mode none) + C pre / B post / VZ post+visionzip-style
#       at r={0.75, 0.875} (keep 25% / 12.5%).
#       Big-image-safe ISO config shared by ALL cells of a bench:
#       --max-num-batched-tokens 32768 --max-pixels 1500000 --max-num-seqs 4
#       (encoder_cache_size guard, cf. DocVQA crash fix; also satisfies the
#        "baseline A same iso-config" TODO in drafts/v3_evidence.md).
#
# One fresh process per cell (enforce_eager + clean hooks). No `set -e`:
# a cell failure must not abort the campaign. Idempotent: skip if JSON exists.
# Echoes sparse [skip]/[done]/[fail] lines -- feed to a Monitor.

cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
OUT=runs/v3_sota_matrix
mkdir -p $OUT
RUN=src/v3_premerger/v3_premerger_runner.py

run_cell() {  # name bench subset mode r [extra flags...]
  local NAME=$1 BENCH=$2 SUB=$3 MODE=$4 R=$5; shift 5
  local JSON=$OUT/$NAME.json LOG=$OUT/$NAME.log
  if [ -f "$JSON" ]; then echo "[skip] $NAME"; return; fi
  local T0=$SECONDS
  python $RUN --mode $MODE --r $R --benchmark $BENCH --subset $SUB --n 200 "$@" --out $JSON > $LOG 2>&1
  local RC=$?
  if [ $RC -eq 0 ]; then
    ACC=$(python -c "import json;d=json.load(open('$JSON'));print('acc=%.4f ptid=%.1f n=%d'%(d['acc'],d['mean_ptid_len'],d['n']))" 2>/dev/null)
    echo "[done] $NAME $ACC t=$((SECONDS-T0))s"
  else
    echo "[fail] $NAME rc=$RC t=$((SECONDS-T0))s (see $LOG)"
  fi
}

# ---------------- Phase 1: GQA VisionZip-style fill ----------------
SUB=eval/subsets/gqa_200.jsonl
run_cell vz_gqa_r0.750  gqa $SUB post 0.75  --max-num-seqs 16 --visionzip-style
run_cell vz_gqa_r0.875  gqa $SUB post 0.875 --max-num-seqs 16 --visionzip-style

# ---------------- Phase 2: ChartQA + OCRBench ----------------
for B in chartqa ocrbench; do
  S=eval/subsets/${B}_200.jsonl
  WAIT=0
  while [ ! -f "$S" ] && [ $WAIT -lt 100 ]; do sleep 60; WAIT=$((WAIT+1)); done
  if [ ! -f "$S" ]; then echo "[skip] $B: subset never appeared after 100min"; continue; fi
  echo "=== $B subset found, starting 7 cells ==="
  BIG="--max-num-batched-tokens 32768 --max-pixels 1500000 --max-num-seqs 4"
  run_cell A_${B}          $B $S none 0.0   $BIG
  run_cell C_${B}_r0.750   $B $S pre  0.75  $BIG
  run_cell B_${B}_r0.750   $B $S post 0.75  $BIG
  run_cell vz_${B}_r0.750  $B $S post 0.75  $BIG --visionzip-style
  run_cell C_${B}_r0.875   $B $S pre  0.875 $BIG
  run_cell B_${B}_r0.875   $B $S post 0.875 $BIG
  run_cell vz_${B}_r0.875  $B $S post 0.875 $BIG --visionzip-style
  echo "=== $B DONE ==="
done
echo "=== SOTA MATRIX CAMPAIGN DONE ==="

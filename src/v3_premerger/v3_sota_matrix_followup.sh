#!/usr/bin/env bash
# SOTA matrix FOLLOW-UP cells (run AFTER v3_sota_matrix.sh campaign finishes):
# align the VisionZip-style column on the OFFICIAL stage (post-merger dom+ctx)
# for the two text-dense benchmarks where the saved VZ cell was pre-mode (textvqa)
# or missing @12.5% (docvqa).
#   textvqa post+VZ @25% / @12.5%  (suite-consistent config, mns 16)
#   docvqa  post+VZ @12.5%         (big-image-safe config: mnbt 32768 / mpix 1.5M / mns 4)
# Idempotent; no set -e; sparse echo lines for monitoring.

cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
OUT=runs/v3_sota_matrix
mkdir -p $OUT
RUN=src/v3_premerger/v3_premerger_runner.py

run_cell() {
  local NAME=$1 BENCH=$2 SUB=$3 R=$4; shift 4
  local JSON=$OUT/$NAME.json LOG=$OUT/$NAME.log
  if [ -f "$JSON" ]; then echo "[skip] $NAME"; return; fi
  local T0=$SECONDS
  python $RUN --mode post --visionzip-style --r $R --benchmark $BENCH --subset $SUB --n 200 "$@" --out $JSON > $LOG 2>&1
  local RC=$?
  if [ $RC -eq 0 ]; then
    ACC=$(python -c "import json;d=json.load(open('$JSON'));print('acc=%.4f ptid=%.1f n=%d'%(d['acc'],d['mean_ptid_len'],d['n']))" 2>/dev/null)
    echo "[done] $NAME $ACC t=$((SECONDS-T0))s"
  else
    echo "[fail] $NAME rc=$RC t=$((SECONDS-T0))s (see $LOG)"
  fi
}

run_cell vz_postmode_textvqa_r0.750  textvqa eval/subsets/textvqa_200.jsonl 0.75  --max-num-seqs 16
run_cell vz_postmode_textvqa_r0.875  textvqa eval/subsets/textvqa_200.jsonl 0.875 --max-num-seqs 16
run_cell vz_postmode_docvqa_r0.875   docvqa  eval/subsets/docvqa_200.jsonl  0.875 --max-num-batched-tokens 32768 --max-pixels 1500000 --max-num-seqs 4
echo "=== FOLLOWUP CAMPAIGN DONE ==="

# NOTE: run_cell forces --visionzip-style; the chartqa @50% budget-probe cells
# below need plain pre/post. Kept as explicit commands (not via run_cell).
for M in pre post; do
  NAME=${M:0:1}_chartqa_r0.500   # C_chartqa_r0.500 / B_chartqa_r0.500
  JSON=runs/v3_sota_matrix/$NAME.json
  if [ -f "$JSON" ]; then echo "[skip] $NAME"; continue; fi
  T0=$SECONDS
  python $RUN --mode $M --r 0.5 --benchmark chartqa --subset eval/subsets/chartqa_200.jsonl \
    --n 200 --max-num-batched-tokens 32768 --max-pixels 1500000 --max-num-seqs 4 \
    --out $JSON > runs/v3_sota_matrix/$NAME.log 2>&1 \
    && echo "[done] $NAME $(python -c "import json;d=json.load(open('$JSON'));print('acc=%.4f ptid=%.1f'%(d['acc'],d['mean_ptid_len']))") t=$((SECONDS-T0))s" \
    || echo "[fail] $NAME rc=$? t=$((SECONDS-T0))s"
done
echo "=== CHARTQA 50% BUDGET PROBE DONE ==="

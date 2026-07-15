#!/usr/bin/env bash
# V3 cross-architecture generality test: Qwen2.5-VL-7B-Instruct.
#
# Reproduces the workload-conditional pre-vs-post stage effect on a SECOND
# architecture (Qwen3-VL -> Qwen2.5-VL). Qwen2.5-VL has the SAME native 2x2
# merger (structurally identical forward: consecutive-4 input tokens = 1 merge-
# unit) but NO deepstack mergers, so the pre-merger hook targets visual.merger
# ONLY (--model-family qwen2vl).
#
# Cells per benchmark (TextVQA + GQA), n=500:
#   A baseline  |  B post-merger @0.75  |  C pre-merger @0.75
# Decisive question (analyzed in v3_crossarch_analyze.py): does the stage-effect
# SIGN reproduce?  TextVQA: pre>post (C-B>0, text-dense).  GQA: post>pre (C-B<0,
# object). Both -> architecture-general.
#
# Outputs: runs/v3_crossarch_cells/{A,B,C}_{bench}_r0.750_qwen2vl.json
# Idempotent (skips existing). One fresh process per cell (enforce_eager + clean
# hooks). NO `set -e`: one failure must not abort the whole campaign.
#
# TUNING: --max-num-seqs starts at 16; if A40 OOMs on a cell, delete its json
# and re-run that cell's command with --max-num-seqs 4 (the runner is identical;
# only the batch size changes).

cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
# Model downloads via the mirror (external to this script). Once cached, vLLM
# loads offline. Fail loudly if the snapshot is not present yet.
export HF_ENDPOINT=https://hf-mirror.com

MODEL_ID="Qwen/Qwen2.5-VL-7B-Instruct"
CACHE_DIR="$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
if [ ! -d "$CACHE_DIR" ]; then
  echo "[FATAL] $MODEL_ID not found in HF cache ($CACHE_DIR)."
  echo "        Download it first (via the mirror):"
  echo "        HF_ENDPOINT=https://hf-mirror.com huggingface-cli download $MODEL_ID"
  exit 1
fi
echo "[crossarch] model cache OK: $CACHE_DIR"

OUT=runs/v3_crossarch_cells
mkdir -p "$OUT"
RUN=src/v3_premerger/v3_premerger_runner.py
FAM=qwen2vl
R=0.75
RR=0.750
MNS=16            # --max-num-seqs; lower to 4 on A40 OOM (delete json, re-run cell)
MML=32768
N=500

for BENCH in textvqa gqa; do
  SUB=eval/subsets/${BENCH}_${N}.jsonl
  NN=$N
  if [ ! -f "$SUB" ]; then
    SUB=eval/subsets/${BENCH}_200.jsonl; NN=200
    echo "[crossarch] $BENCH: _500 subset missing, using _200"
  fi
  echo "[crossarch] $BENCH (n=$NN) subset=$SUB"

  # (A) baseline -- no pruning
  if [ ! -f "$OUT/A_${BENCH}_r${RR}_qwen2vl.json" ]; then
    python "$RUN" --mode none --r 0.0 --model-family $FAM \
      --benchmark $BENCH --subset "$SUB" --n $NN --selector l2 \
      --max-num-seqs $MNS --max-model-len $MML \
      --out "$OUT/A_${BENCH}_r${RR}_qwen2vl.json" \
      > "$OUT/A_${BENCH}_r${RR}_qwen2vl.log" 2>&1 \
      || echo "[fail] A_${BENCH}_r${RR}_qwen2vl (see .log)"
  fi
  # (B) post-merger prune @0.75
  if [ ! -f "$OUT/B_${BENCH}_r${RR}_qwen2vl.json" ]; then
    python "$RUN" --mode post --r $R --model-family $FAM \
      --benchmark $BENCH --subset "$SUB" --n $NN --selector l2 \
      --max-num-seqs $MNS --max-model-len $MML \
      --out "$OUT/B_${BENCH}_r${RR}_qwen2vl.json" \
      > "$OUT/B_${BENCH}_r${RR}_qwen2vl.log" 2>&1 \
      || echo "[fail] B_${BENCH}_r${RR}_qwen2vl (see .log)"
  fi
  # (C) pre-merger prune @0.75
  if [ ! -f "$OUT/C_${BENCH}_r${RR}_qwen2vl.json" ]; then
    python "$RUN" --mode pre --r $R --model-family $FAM \
      --benchmark $BENCH --subset "$SUB" --n $NN --selector l2 \
      --max-num-seqs $MNS --max-model-len $MML \
      --out "$OUT/C_${BENCH}_r${RR}_qwen2vl.json" \
      > "$OUT/C_${BENCH}_r${RR}_qwen2vl.log" 2>&1 \
      || echo "[fail] C_${BENCH}_r${RR}_qwen2vl (see .log)"
  fi
  echo "=== $BENCH DONE ==="
done
echo "=== V3 CROSSARCH DONE ==="

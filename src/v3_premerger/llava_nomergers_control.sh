#!/usr/bin/env bash
# LLaVA-1.5 NO-MERGER negative control: none/pre/post x {textvqa,gqa} + rescore.
#
# Tests the paper's predictive law: a model with NO spatial merger (LLaVA-1.5's
# per-token MLP projector) should show ~0 stage effect (|pre-post|<=1pp on BOTH
# TextVQA + GQA).  See llava_nomergers_control.py for the full rationale.
#
# DO NOT run from the main window -- this is queued for the GPU.  It waits for a
# free GPU (<1GB used), runs 6 cells (3 modes x 2 benchmarks, n=200, greedy,
# max_tokens=32, keep 25%=144/576), then rescores with the official metric +
# paired delta + decision rule.
#
# Usage:  bash src/v3_premerger/llava_nomergers_control.sh
set -euo pipefail

PY=/home/dell/miniconda3/envs/fastv/bin/python
cd "$(dirname "$0")/../.."   # repo root (so src.v3_premerger resolves)

OUT_DIR=runs/llava_nomergers
SUBSET_TEXTVQA=eval/subsets/textvqa_200.jsonl
SUBSET_GQA=eval/subsets/gqa_200.jsonl
N=200
MAX_TOKENS=32
KEEP_FRAC=0.25

mkdir -p "$OUT_DIR"

# ---- wait for a free GPU (>= 40000 MiB free; baseline VRAM ~1.3GB so used-only
#      threshold would never trigger) ----
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
echo "[nomergers] waiting for free GPU (>=40000MB free; now ${free}MB)..."
while [ "${free:-0}" -le 40000 ]; do
  sleep 30
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
done
echo "[nomergers] GPU free (${free}MB free). starting cells."

# ---- 6 cells: none/pre/post x textvqa/gqa ----
for BENCH in textvqa gqa; do
  if [ "$BENCH" = "textvqa" ]; then SUB=$SUBSET_TEXTVQA; else SUB=$SUBSET_GQA; fi
  for MODE in none pre post; do
    OUT="$OUT_DIR/${BENCH}_${MODE}.json"
    if [ -f "$OUT" ]; then
      echo "[nomergers] skip $OUT (exists)"
      continue
    fi
    echo "[nomergers] === $BENCH $MODE (keep ${KEEP_FRAC}, n=${N}) ==="
    "$PY" -m src.v3_premerger.llava_nomergers_control \
      --mode "$MODE" --benchmark "$BENCH" --subset "$SUB" \
      --keep-frac "$KEEP_FRAC" --n "$N" --max-tokens "$MAX_TOKENS" \
      --out "$OUT" 2>&1 | tee "$OUT_DIR/${BENCH}_${MODE}.log"
  done
done

# ---- rescore + decision rule ----
echo "[nomergers] === rescore + decision ==="
"$PY" -m src.v3_premerger.llava_nomergers_control --rescore "$OUT_DIR" 2>&1 | tee "$OUT_DIR/decision.log"
echo "[nomergers] DONE.  decision -> $OUT_DIR/llava_nomergers_decision.json"

#!/usr/bin/env bash
# Q-RBM E1 -- causal-importance gate driver (pre-registered, notes/qrbm_e1_plan.md §6).
#   default : SMOKE -- n=4 x G=8, all 4 benches x both splits (8 cells), with
#             --null-check (drop-nothing -> u=0 verification) and s/pass timing.
#   --full  : full gate -- n=64 (all samples) x G=8, both splits, 8 cells
#             (resumes the smoke cells' already-written samples).
# Resumable/idempotent: cell_done skips cells whose out file already has >= N
# lines; the python driver itself skips samples already in the out file, so an
# OOM-killed cell is re-run and simply continues.
set -u
FULL="${1:-}"
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export HF_HUB_CACHE=/data/models/huggingface/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /media/disk2/YZX/research/vla
PY=src/v3_premerger/e1_causal_import.py
OUT=runs/e1
mkdir -p $OUT

# GPU etiquette: wait for >= 30 GiB free (two consecutive checks, 30 s apart,
# so a co-tenant's model-load ramp cannot race our cell -- cascade lib.sh guard).
wait_gpu() {
  local max_wait=${1:-21600} i=0 free free2
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${free:-0}" -ge 30720 ]; then
      sleep 30
      free2=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
      if [ "${free2:-0}" -ge 30720 ]; then
        echo "[gpu] FREE (${free}->${free2} MiB) -- proceeding"; return 0
      fi
      echo "[gpu] free dropped ${free}->${free2} MiB during stability check -- waiting"
    fi
    i=$((i+30))
    if [ "$i" -ge "$max_wait" ]; then echo "[gpu] TIMEOUT after ${max_wait}s (free=${free} MiB)"; return 1; fi
    [ $((i % 600)) -eq 0 ] && echo "[gpu] busy free=${free} MiB, waited ${i}s ..."
    sleep 30
  done
}

# cell_done <jsonl> <n_min> -> 0 if the cell's out file already has >= n_min
# sample lines (resume / OOM guard).
cell_done() {
  [ -f "$1" ] || return 1
  local n; n=$(wc -l < "$1")
  [ "$n" -ge "$2" ]
}

run_cell() {  # run_cell <split> <bench> <n> [extra args...]
  local SPLIT=$1 B=$2 N=$3; shift 3
  local OUTJ=$OUT/${SPLIT}_${B}.jsonl
  if cell_done "$OUTJ" "$N"; then
    echo "[skip] e1 ${SPLIT}/${B} ($(wc -l < "$OUTJ") lines already)"
    return 0
  fi
  wait_gpu || return 1
  echo "=== e1 ${SPLIT}/${B} n=${N} g=8 fastv-k=3 $* ==="
  timeout 14400 python $PY --benchmark "$B" --split "$SPLIT" --n "$N" --g 8 \
    --fastv-k 3 --seed 0 --out "$OUT" "$@" \
    > "$OUT/${SPLIT}_${B}.log" 2>&1 \
    && echo "[done] e1 ${SPLIT}/${B}" || echo "[FAIL] e1 ${SPLIT}/${B}"
}

# pre-registered: G=8, fastv-k=3 (matches the existing FastV gate cells).
if [ "$FULL" = "--full" ]; then
  N=64; EXTRA=""
  echo "=== E1 FULL GATE (n=64 x G=8, 8 cells) ==="
else
  N=4; EXTRA="--null-check"
  echo "=== E1 SMOKE (n=4 x G=8, 8 cells, --null-check) ==="
fi

for SPLIT in dev heldout; do
  for B in textvqa docvqa gqa ocrbench; do
    run_cell "$SPLIT" "$B" "$N" $EXTRA
  done
done
echo "=== E1 DONE (full=${FULL:-no}) -- outputs in $OUT/*.jsonl ==="

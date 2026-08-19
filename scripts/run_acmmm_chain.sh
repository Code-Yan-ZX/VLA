#!/bin/bash
# Chain runner for the acmmm_final_controls campaign (single A40 -> serial):
#   1. P0-1 (vLLM, Qwen3-VL-8B pure-stage control, full splits)  [may already be running]
#   2. P0-2 (vLLM, Qwen2.5-VL-7B OCRBench matched-config rerun)
#   3. P1   (HF harness, RBM vs FastV-k3 OCRBench full, both models)
#   4. analysis (CPU)
# Each stage waits for the GPU to be free (>= 40000 MiB) so it can be launched
# standalone too.
set -u
cd /media/disk2/YZX/research/vla
ROOT=/media/disk2/YZX/research/vla/results/acmmm_final_controls
wait_gpu(){
  echo "[chain] $(date -u '+%F %T') waiting for >= 40000 MiB free"
  for i in $(seq 1 600); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "$FREE" -gt 40000 ] && { echo "[chain] GPU free ${FREE} MiB"; return 0; }
    sleep 30
  done
  echo "[chain][ABORT] GPU busy after 5h"; exit 1
}

echo "[chain] $(date -u '+%F %T') acmmm_final_controls chain start"

# ---- P0-1 (if not already complete: 8 cells) ----
NEEDED=""
for B in textvqa docvqa ocrbench gqa; do
  for M in pre-final post; do
    [ -s $ROOT/p0_1/p0_1_qwen3_${M}_${B}_r0.750_full.json ] || NEEDED="$NEEDED $M/$B"
  done
done
if [ -n "$NEEDED" ]; then
  echo "[chain] P0-1 pending cells:$NEEDED"
  bash scripts/run_p0_1_full_split.sh
else
  echo "[chain] P0-1 already complete"
fi

# ---- P0-2 ----
NEEDED=""
for R in 0.750 0.875; do
  for M in pre post; do
    [ -s $ROOT/p0_2/p0_2_qwen2_${M}_ocrbench_r${R}_full.json ] || NEEDED="$NEEDED $M@$R"
  done
done
if [ -n "$NEEDED" ]; then
  echo "[chain] P0-2 pending cells:$NEEDED"
  wait_gpu
  bash scripts/run_p0_2_ocrbench_matched.sh
else
  echo "[chain] P0-2 already complete"
fi

# ---- P1 ----
NEEDED=""
for F in qwen3vl qwen2vl; do
  [ -s $ROOT/p1/p1_${F}_pre_ocrbench_r25_full.json ] || NEEDED="$NEEDED $F/rbm"
  [ -s $ROOT/p1/p1_${F}_fastv_ocrbench_k3_full.json ] || NEEDED="$NEEDED $F/fastv"
done
if [ -n "$NEEDED" ]; then
  echo "[chain] P1 pending cells:$NEEDED"
  wait_gpu
  bash scripts/run_p1_fastv_hf_ocrbench.sh
else
  echo "[chain] P1 already complete"
fi

# ---- analysis ----
echo "[chain] analysis"
/home/dell/miniconda3/envs/qwen3vl_clean/bin/python scripts/analyze_acmmm_final_controls.py

echo "[chain] DONE $(date -u '+%F %T')"

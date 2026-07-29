#!/bin/bash
# R1-1 — kept-set Jaccard discriminator: do pre and swap keep the SAME unit set?
#   {qwen3vl,qwen2vl} x {pre, swap} x {docvqa,textvqa}, n=32 seq=1 r=0.75,
#   --save-unit-scores (runner R1-1 patch writes per_sample[i].kept_indices).
# Judge: same set + different answers -> value-level difference (merger batch
# dependence); different sets -> swap ranking-capture misalignment artifact.
# Measurement only; no method-logic change. Resume: existing jsons are skipped.
set -u
cd /media/disk2/YZX/research/vla
PYQ3=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
OUT=runs/r1_1_swap_jaccard
mkdir -p $OUT
echo "[R1-1] $(date -u '+%F %T') waiting for >= 30000 MiB free"
for i in $(seq 1 360); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$FREE" -gt 30000 ]; then echo "[R1-1] GPU free ${FREE} MiB"; break; fi
  sleep 60
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE" -le 30000 ]; then echo "[R1-1][ABORT] GPU busy after 6h"; exit 1; fi
SEQ1="--max-num-seqs 1 --gpu-memory-utilization 0.9 --max-model-len 32768 --max-num-batched-tokens 32768"
for FAM in qwen3vl qwen2vl; do
  for B in docvqa textvqa; do
    for MODE in pre swap; do
      MR=""; M=pre; [ "$MODE" = "swap" ] && { MR="--mask-ranking swap"; M=post; }
      TAG=r1_1_${FAM}_${MODE}_${B}_n32
      if [ -s $OUT/$TAG.json ]; then echo "[skip] $TAG"; continue; fi
      echo "=== $TAG start $(date -u '+%F %T') ==="
      timeout 1800 $PYQ3 src/v3_premerger/v3_premerger_runner.py --model-family $FAM --benchmark $B \
        --subset eval/subsets/${B}_200.jsonl --n 32 --r 0.75 --mode $M $MR --save-unit-scores \
        $SEQ1 --out $OUT/$TAG.json > $OUT/$TAG.log 2>&1
      echo "=== $TAG done rc=$? $(date -u '+%F %T') ==="
      tail -1 $OUT/$TAG.log
    done
  done
done
echo "=== R1-1 Jaccard analysis $(date -u '+%F %T') ==="
$PYQ3 scripts/analyze_r1_1_jaccard.py
echo "[R1-1 done] $(date -u '+%F %T')"

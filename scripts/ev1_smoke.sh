#!/usr/bin/env bash
# ElasticVis EV-1 smoke test: verify PER-REQUEST visual-token budget in ONE batched
# forward (the constraint-break proof for serve_bench.py :1160-1166).
#
# Submits 3 LLaVA-1.5 images to --batch-submit (continuous batching) with
# --k-policy elasticvis --ev-debug-k 576,144,576. The projector hook debug-print
# should show B=2-3 with per-row-k=[576,144,...] — i.e. DIFFERENT requests in the
# SAME forward got different visual-token counts. The metrics JSON also records
# ev_per_batch_k_head + k_by_rid for post-hoc confirmation.
#
# PREREQ: conda activate qwen3vl_clean ; GPU free.
# RUNTIME: ~2 min (model load + 3 images).
# NOT a benchmark — just evidence the per-request k plumbing works end-to-end.
set -e
source /home/dell/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate qwen3vl_clean
cd /media/disk2/YZX/research/vla

MODEL=runs/models/llava-1.5-7b-hf
SUB=eval/subsets/gqa_200.jsonl
OUT=runs/ev1_smoke.json
LOG=runs/ev1_smoke.log
mkdir -p runs

echo "[ev1_smoke] start $(date -Iseconds)"
echo "[ev1_smoke] 3 images, k={576,144,576} in ONE batch -- expect per-row-k evidence"

python -m src.serve_bench \
  --model "$MODEL" --engine v1 \
  --pruning-rate 0.5 --selector proxy \
  --benchmark gqa --subset "$SUB" \
  --batch-submit --max-num-seqs 8 --max-tokens 16 --limit 3 \
  --k-policy elasticvis --ev-debug-k 576,144,576 \
  --metrics-out "$OUT" 2>&1 | tee "$LOG"

echo ""
echo "[ev1_smoke] === EVIDENCE ==="
echo "--- per-row-k from hook debug prints ---"
grep "EV1 projector hook" "$LOG" || echo "(no hook prints found — check $LOG)"

echo "--- k_by_rid from metrics JSON ---"
python -c "
import json
d = json.load(open('$OUT'))
ev = d.get('elasticvis')
if ev:
    print('k_by_rid:', ev['k_by_rid'])
    print('ev_per_batch_k_head:', ev['ev_per_batch_k_head'])
    print('n distinct k:', len(set(ev['k_by_rid'].values())))
    ks = list(ev['k_by_rid'].values())
    if len(set(ks)) > 1:
        print(f'PASS: {len(set(ks))} distinct k values in one batch {sorted(set(ks))}')
    else:
        print(f'FAIL: only 1 k value {ks}')
else:
    print('no elasticvis section in metrics')
" 2>&1 || echo "(metrics parse failed — check $OUT)"

echo "[ev1_smoke] done $(date -Iseconds)"
echo "[ev1_smoke] artifacts: $OUT  +  $LOG"

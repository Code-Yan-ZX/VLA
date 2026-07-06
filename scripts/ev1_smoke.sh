#!/usr/bin/env bash
# ElasticVis EV-1b smoke test: verify PER-REQUEST visual-token budget in ONE batched
# forward using ORDER-INDEPENDENT fingerprint matching (robust to scheduler
# reordering under continuous batching + chunked prefill).
#
# Two test modes:
#   1) SMOKE (default, ~2 min): 3 images, k={576,144,576} round-robin.
#      Verifies different requests in ONE batch get different visual-token counts.
#   2) C64 (EV1_C64=1, ~15 min): 128 TextVQA images, max-num-seqs=64, mixed-SLO.
#      The crash scenario from EV-1a — verifies the fingerprint fix survives
#      continuous batching at scale.
#
# PREREQ: conda activate qwen3vl_clean ; GPU free.
# NOT a benchmark — just evidence the per-request k plumbing works end-to-end.
set -e
source /home/dell/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate qwen3vl_clean
cd /media/disk2/YZX/research/vla

MODEL=runs/models/llava-1.5-7b-hf

if [ "${EV1_C64:-0}" = "1" ]; then
    # ---- C64 STRESS TEST (the EV-1a crash scenario) ----
    SUB=eval/subsets/textvqa_200.jsonl
    OUT=runs/ev1_c64.json
    LOG=runs/ev1_c64.log
    echo "[ev1_c64] start $(date -Iseconds) - 128 TextVQA, c64, mixed-SLO"
    python -m src.serve_bench \
      --model "$MODEL" --engine v1 \
      --pruning-rate 0.5 --selector proxy \
      --benchmark textvqa --subset "$SUB" \
      --batch-submit --max-num-seqs 64 --max-tokens 16 --limit 128 \
      --k-policy elasticvis --ev-mixed-slo 3500,15000 \
      --metrics-out "$OUT" 2>&1 | tee "$LOG"
else
    # ---- 3-IMAGE SMOKE TEST ----
    SUB=eval/subsets/gqa_200.jsonl
    OUT=runs/ev1_smoke.json
    LOG=runs/ev1_smoke.log
    mkdir -p runs
    echo "[ev1_smoke] start $(date -Iseconds)"
    echo "[ev1_smoke] 3 images, k={576,144,576} - expect per-row-k evidence"
    python -m src.serve_bench \
      --model "$MODEL" --engine v1 \
      --pruning-rate 0.5 --selector proxy \
      --benchmark gqa --subset "$SUB" \
      --batch-submit --max-num-seqs 8 --max-tokens 16 --limit 3 \
      --k-policy elasticvis --ev-debug-k 576,144,576 \
      --metrics-out "$OUT" 2>&1 | tee "$LOG"
fi

echo ""
echo "=== EVIDENCE ==="
echo "--- embed_multimodal fingerprint matching ---"
grep "EV embed_multimodal" "$LOG" | head -10 || echo "(none - check $LOG)"
echo "--- projector hook per-row-k ---"
grep "EV1b projector hook" "$LOG" | head -10 || echo "(none - check $LOG)"
echo "--- fingerprint stats from metrics JSON ---"
python -c "
import json
d = json.load(open('$OUT'))
ev = d.get('elasticvis')
if ev:
    print('k_by_rid:', dict(list(ev['k_by_rid'].items())[:6]))
    print('ev_per_batch_k_head:', ev['ev_per_batch_k_head'][:5])
    print(f\"fp_to_k entries: {ev['n_fp_to_k_entries']}\")
    print(f\"embed_multimodal calls: {ev['n_embed_calls']}\")
    print(f\"fp hits: {ev['n_fp_hits']}  misses: {ev['n_fp_miss']}\")
    ks = list(ev['k_by_rid'].values())
    if len(set(ks)) > 1:
        print(f'PASS: {len(set(ks))} distinct k values {sorted(set(ks))}')
    else:
        print(f'WARN: only 1 k value {ks}')
    if ev['n_fp_miss'] > 0:
        print(f'WARN: {ev[\"n_fp_miss\"]} fingerprint misses (cur_k fallback used)')
else:
    print('no elasticvis section in metrics')
" 2>&1 || echo "(metrics parse failed - check $OUT)"
echo "[ev1] done $(date -Iseconds)"
echo "[ev1] artifacts: $OUT + $LOG"
echo ""
echo "To run the c64 stress test: EV1_C64=1 bash scripts/ev1_smoke.sh"

#!/usr/bin/env bash
# D2 content-driven per-image budget (pixel-spectral), FAIR evaluation on the
# D1 held-out slices. Protocol per bench:
#   arm A: uniform r=0.75 l2 @ max-num-seqs=1 (same cache behavior)
#   arm B: per-image budget (frac, iso-mean-total to r=0.75) l2 @ max-num-seqs=1
#   arm C (diagnostic): pixel-frac-only reallocation via runner frac mode
# Comparison: arm B vs arm A on the SAME slice => does content-aware reallocation
# beat uniform at iso-total-budget?
# The budget file is built by scripts/build_d2_budget.py (pixel frac + mean-frac
# normalization to 0.25, i.e. iso-total to r=0.75).
set -e
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
RUNNER=src/v3_premerger/v3_premerger_runner.py
BENCHES=(textvqa docvqa gqa)
R=0.75
OUTDIR=runs/d2_budget
mkdir -p "$OUTDIR"

for b in "${BENCHES[@]}"; do
  # build pixel budget file (offline, CPU)
  "$PY" scripts/stitch_pixel_calib.py --bench "$b" \
    --subset "eval/learned_heldout/${b}_200.jsonl" --n 200 --mode spectral \
    --tau 0.9 --out "$OUTDIR/${b}_pixfrac.json"
  "$PY" scripts/build_d2_budget.py --bench "$b" --r "$R" \
    --frac "$OUTDIR/${b}_pixfrac.json" --subset "eval/learned_heldout/${b}_200.jsonl" \
    --out "$OUTDIR/${b}_budget.json"
  for arm in uniform budget; do
    tag="${b}_r${R}_${arm}_heldout200"
    [[ -f "$OUTDIR/$tag.json" ]] && { echo "[d2] skip $tag"; continue; }
    echo "[d2] RUN $tag $(date +%H:%M)"
    args=(--mode pre --selector l2 --r "$R" --benchmark "$b"
          --subset "eval/learned_heldout/${b}_200.jsonl" --n 200
          --max-num-seqs 1 --out "$OUTDIR/$tag.json")
    [[ "$arm" == "budget" ]] && args+=(--budget-file "$OUTDIR/${b}_budget.json")
    timeout 7200 "$PY" "$RUNNER" "${args[@]}" \
      > "experiments/d2_${tag}.log" 2>&1 && echo "[d2] OK $tag $(date +%H:%M)" \
      || echo "[d2] FAIL $tag (see log)"
  done
done
echo "[d2] done"

#!/usr/bin/env bash
# D1 learned-scorer GPU validation: learned vs PRE-L2 on HELD-OUT slices.
# Cells: bench x {learned, l2} @ r=0.75, n=200 (held-out, disjoint from the
# training subset A). Serial on 1x A40.
# Usage: bash scripts/run_d1_eval.sh [--r 0.875] [benches...]
set -e
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
RUNNER=src/v3_premerger/v3_premerger_runner.py
R=0.75
BENCHES=(textvqa docvqa gqa)
[[ "$1" == "--r" ]] && { R="$2"; shift 2; }
[[ $# -gt 0 ]] && BENCHES=("$@")
OUTDIR=runs/d1_learned
mkdir -p "$OUTDIR"

for b in "${BENCHES[@]}"; do
  for sel in learned l2; do
    tag="${b}_r${R}_${sel}_heldout200"
    if [[ -f "$OUTDIR/$tag.json" ]]; then
      echo "[d1] skip (exists) $tag"; continue
    fi
    echo "[d1] RUN $tag $(date +%H:%M)"
    args=(--mode pre --selector "$sel" --r "$R"
          --benchmark "$b" --subset "eval/learned_heldout/${b}_200.jsonl"
          --n 200 --out "$OUTDIR/$tag.json")
    if [[ "$sel" == "learned" ]]; then
      args+=(--learned-scorer runs/learned_scorer --learned-bench "$b")
    fi
    timeout 7200 "$PY" "$RUNNER" "${args[@]}" \
      > "experiments/d1_${tag}.log" 2>&1 && echo "[d1] OK $tag $(date +%H:%M)" \
      || { echo "[d1] FAIL $tag (see log)"; }
  done
done
echo "[d1] done $(date +%H:%M)"

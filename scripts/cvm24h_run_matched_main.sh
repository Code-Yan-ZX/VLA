#!/usr/bin/env bash
# User-approved extension: uniform-attention full main table, frozen on Oct26.
# Run after the corrected P3 supervisor exits; one A40, always serial.
# Usage: bash scripts/cvm24h_run_matched_main.sh [P3_SUPERVISOR_PID]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
P=${CVM_PYTHON:-/home/dell/miniconda3/envs/qwen3vl_clean/bin/python}
D=results/cvm2027_supplement_20261026/matched_main
Q=results/cvm2027_supplement_20261026/plan
L=results/cvm2027_supplement_20261026/logs
# User's hard delivery limit is 20:00. Protect 90 minutes for final writing/QA.
DEADLINE=${CVM_MATCHED_DEADLINE_EPOCH:-1793010600}  # default: Oct26 18:30 China
SOFT_DEADLINE=$((DEADLINE-60))
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$D" "$L"
if test -n "${1:-}"; then
  echo "[matched] waiting for P3 supervisor $1 at $(date -Iseconds)"
  while test -d "/proc/$1"; do sleep 5; done
fi
"$P" scripts/cvm24h_validate_timing_v2.py "$Q/timing_harness_corrected.json" \
  "$Q/timing_hilo_corrected.json" --out "$Q/timing_corrected_validation.json"
"$P" - "$Q/timing_corrected_validation.json" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))['complete'], 'Corrected P3 must pass before the full matched main'
PY
while test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; do
  test "$(date +%s)" -lt "$SOFT_DEADLINE"
  sleep 5
done
echo "[matched] original eager32 no-op at $(date -Iseconds)"
"$P" scripts/cvm24h_noop_verify.py --out "$Q/p0_noop32_verified.json" \
  --deadline-epoch "$SOFT_DEADLINE" > "$L/p0_noop32_verified.log" 2>&1
"$P" - "$Q/p0_noop32_verified.json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1])); assert x['status']=='pass' and x['n_complete']==32, x['status']
PY
# Order is frozen by priority and runtime risk, never by intermediate scores.
for phase in smoke full; do
for mode in hilo none pre fastv; do
  for bench in textvqa ocrbench; do
    test "$(date +%s)" -lt "$SOFT_DEADLINE"
    if test "$bench" = textvqa; then
      split=eval/full_splits/textvqa_val.jsonl; count=5000
    else
      split=eval/full_splits/ocrbench.jsonl; count=1000
    fi
    if test "$mode" = none; then keep=1.0; else keep=0.25; fi
    "$P" - <<'PY'
import hashlib
from pathlib import Path
for filename, expected in {
    'scripts/cvm24h_matched_quality.py':'67ae3554264dbd780ab98609623dec9111bd9b88f9136d446bbb7c75ae4e5834',
    'scripts/cvm24h_timing_v2.py':'25b526d6cadc809095f28199bfdde754bb9af2dc15d954ab05a05f8f0a3faa1c',
}.items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==expected, f'Frozen source changed: {filename}'
PY
    target_dir="$D"
    if test "$phase" = smoke; then count=3; target_dir="$D/smoke"; fi
    mkdir -p "$target_dir"
    echo "[matched start] $phase $mode $bench $(date -Iseconds)"
    timeout --signal=TERM "$((DEADLINE-$(date +%s)))" "$P" scripts/cvm24h_matched_quality.py \
      --mode "$mode" --benchmark "$bench" --subset "$split" --n "$count" \
      --keep-frac "$keep" --out "$target_dir/${mode}_${bench}.json" \
      --deadline-epoch "$SOFT_DEADLINE" > "$L/matched_${phase}_${mode}_${bench}.log" 2>&1
    echo "[matched end] $phase $mode $bench $(date -Iseconds)"
  done
done
if test "$phase" = smoke; then
  "$P" - "$D/smoke" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); protocol = sources = None
for mode in ('hilo','none','pre','fastv'):
    for bench in ('textvqa','ocrbench'):
        d=json.loads((root/f'{mode}_{bench}.json').read_text())
        assert d['complete'] and d['n_attempted']==d['n_scored']==3 and not d['n_error']
        if protocol is None: protocol=d['protocol']; sources=d['source_hashes']
        assert protocol==d['protocol'] and sources==d['source_hashes']
        assert len(d['blind_p3_check']['warmup'])==3
        assert all(all(v for k,v in r.items() if k!='id') for r in d['blind_p3_check']['warmup'])
        for r in d['per_sample']:
            n,k=r['n_image_full'],r['n_image_kept']
            assert r['actual_k_verified'] and k==(n if mode=='none' else max(1,round(.25*n)))
print('All eight three-request runtime smokes passed; beginning frozen full matrix')
PY
fi
done
"$P" scripts/cvm24h_score_matched.py
"$P" scripts/cvm24h_build_tables.py
echo "[matched] full main and scores complete $(date -Iseconds)"

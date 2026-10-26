#!/usr/bin/env bash
# Reproduce the frozen quality matrix without the historical one-day deadline.
# Requires the preserved model/data cache and fixed subset files; runs serially.
# Usage: bash scripts/cvm24h_reproduce_quality.sh [main|curves|doc|all]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
GROUP=${1:-all}
case "$GROUP" in main|curves|doc|all) ;; *) echo 'Expected main|curves|doc|all' >&2; exit 2;; esac
P=${CVM_PYTHON:-/home/dell/miniconda3/envs/qwen3vl_clean/bin/python}
OUT=${CVM_REPRO_DIR:-results/cvm2027_reproduction}
PLAN=results/cvm2027_supplement_20261026/plan
HF=src/v3_premerger/baselines_hf.py
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$OUT/predictions" "$OUT/logs"
if test -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"; then
  echo 'GPU compute is occupied; wait for the current task instead of starting a second job.' >&2
  exit 2
fi
"$P" scripts/cvm24h_prepare_hilo.py --out "$OUT/hilo_source_manifest.json"
run() {
  local tag=$1 expected=$2
  shift 2
  local file="$OUT/predictions/$tag.json"
  if test -f "$file"; then
    "$P" - "$file" "$expected" <<'PY'
import json, sys
x = json.load(open(sys.argv[1]))
assert len(x['per_sample']) == int(sys.argv[2]), 'Existing output is partial; preserve and inspect it'
print('Complete output already exists:', sys.argv[1])
PY
    return
  fi
  "$@" --out "$file" > "$OUT/logs/$tag.log" 2>&1
}
cell() {
  local tag=$1 arm=$2 bench=$3 split=$4 count=$5 keep=$6 cap=$7
  local common=(--benchmark "$bench" --subset "$split" --n "$count" --max-pixels "$cap" --seed 0 --max-tokens 32)
  case "$arm" in
    none) run "$tag" "$count" "$P" "$HF" --model Qwen/Qwen3-VL-8B-Instruct --mode none "${common[@]}" ;;
    pre) run "$tag" "$count" "$P" "$HF" --model Qwen/Qwen3-VL-8B-Instruct --mode pre --r-pre "$keep" --mrope native "${common[@]}" ;;
    fastv)
      local drop
      drop=$("$P" -c 'import sys; print(1-float(sys.argv[1]))' "$keep")
      run "$tag" "$count" "$P" "$HF" --model Qwen/Qwen3-VL-8B-Instruct --mode fastv --r "$drop" --fastv-k 3 "${common[@]}" ;;
    hilo) run "$tag" "$count" "$P" scripts/cvm24h_hilo_eval.py --keep-frac "$keep" --attn sdpa "${common[@]}" ;;
  esac
}
if test "$GROUP" = main || test "$GROUP" = all; then
  for bench in textvqa ocrbench; do
    if test "$bench" = textvqa; then split=eval/full_splits/textvqa_val.jsonl; count=5000
    else split=eval/full_splits/ocrbench.jsonl; count=1000; fi
    cell "main_${bench}_none" none "$bench" "$split" "$count" 1 4000000
    for arm in pre fastv hilo; do
      cell "main_${bench}_${arm}_0.25" "$arm" "$bench" "$split" "$count" 0.25 4000000
    done
  done
fi
if test "$GROUP" = curves || test "$GROUP" = all; then
  for bench in textvqa ocrbench; do
    if test "$bench" = textvqa; then split="$PLAN/p2_textvqa_1000.jsonl"
    else split=eval/full_splits/ocrbench.jsonl; fi
    cell "curve_${bench}_none" none "$bench" "$split" 1000 1 4000000
    for keep in 0.25 0.50 0.75; do
      for arm in pre fastv hilo; do
        cell "curve_${bench}_${arm}_$keep" "$arm" "$bench" "$split" 1000 "$keep" 4000000
      done
    done
  done
fi
if test "$GROUP" = doc || test "$GROUP" = all; then
  split="$PLAN/p2_docvqa_500.jsonl"
  cell doc_none none docvqa "$split" 500 1 2000000
  for arm in pre fastv hilo; do
    cell "doc_${arm}_0.25" "$arm" docvqa "$split" 500 0.25 2000000
  done
fi
echo "Frozen $GROUP quality matrix complete: $OUT"

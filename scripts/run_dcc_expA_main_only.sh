#!/bin/bash
# DCC pre-submission Experiment A: Qwen3-VL main-only Post-L2 control.
#
# Question: the post-L2 ranking currently scores the FULL concatenated visual
# row [main | ds0 | ds1 | ds2] (vLLM qwen3_vl.py:654 cats main FIRST). Does the
# deepstack-column contribution to the ranking matter? Arm "post-main" scores
# ONLY the main-merger block (s[:, :hidden]) but keeps FULL rows (deepstack
# features of kept units preserved -> identical final token counts).
#
# Protocol = scripts/run_p0_1_full_split.sh (greedy temp=0, selector l2,
# r=0.75 keep 25%, native pixels max_pixels=0, --max-tokens 32):
#   textvqa/ocrbench/gqa:  --max-num-seqs 8 --max-model-len 8192 --gpu-mem 0.9
#   docvqa:                --max-num-seqs 4 --max-model-len 32768 --max-num-batched-tokens 32768
# Comparison anchors (existing, NOT rerun): results/acmmm_final_controls/p0_1/
#   p0_1_qwen3_pre-final_{textvqa,docvqa}_r0.750_full.json
#   (+ n=200 probes: runs/qwen3_prefinal_control/qwen3_pre-final_*.json)
#
# Usage: bash scripts/run_dcc_expA_main_only.sh [smoke|validate|full|extra|all]
# Outputs -> results/dcc_presubmit/expA/  (never overwrites historical dirs)
set -u
cd /media/disk2/YZX/research/vla
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       VLLM_NO_USAGE_STATS=1 VLLM_USE_MODELSCOPE=False

OUT=results/dcc_presubmit/expA
mkdir -p $OUT
FAM=qwen3vl
R=0.75
MAXTOK=32
FS=eval/full_splits
STD="--max-num-seqs 8 --max-model-len 8192 --gpu-memory-utilization 0.9"
DOC="--max-num-seqs 4 --max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9"
STAGE=${1:-all}

wait_gpu(){ # block until >= 40000 MiB free (protocol convention)
  for i in $(seq 1 240); do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "$FREE" -gt 40000 ] && { echo "[expA] GPU free ${FREE} MiB"; return 0; }
    sleep 30
  done
  echo "[expA][ABORT] GPU busy after wait"; return 1
}

run_cell(){ # bench n subset mode extra-flags tag [lowmem]  (skip>10% -> one retry, p0_1 gate)
  local FLAGS=$STD; [ "$1" = "docvqa" ] && FLAGS=$DOC
  # lowmem: desktop/Xorg growth can leave < 40.01 GiB free at vLLM startup;
  # 0.85 shrinks only the KV cache (scheduling), not numerics. Noted in report.
  if [ "$6" = "lowmem" ]; then
    FLAGS="${FLAGS%% --gpu-memory-utilization *} --gpu-memory-utilization 0.85"
  fi
  for ATT in 1 2; do
    timeout 21600 $PY src/v3_premerger/v3_premerger_runner.py --model-family $FAM \
      --benchmark $1 --subset $3 --n $2 --r $R --mode post \
      --post-score-scope main --selector l2 --max-tokens $MAXTOK $4 \
      --out $OUT/$5.json > $OUT/$5.log 2>&1
    echo "[expA] $5 (try $ATT) exit=$? $(tail -1 $OUT/$5.log)"
    SR=$($PY -c "
import json
try:
    d=json.load(open('$OUT/$5.json'))
    print((d.get('n_skipped') or 0)/max(len(d.get('per_sample') or [1]),1))
except Exception: print(1.0)")
    $PY -c "import sys; sys.exit(0 if float('$SR') <= 0.10 else 1)" \
      && return 0
    echo "[expA] $5 skip=$SR > 0.10 -> retry"
  done
  echo "[expA][WARN] $5 still skip=$SR after retry"
}

wait_gpu
if [ "$STAGE" = "smoke" ] || [ "$STAGE" = "all" ]; then
  echo "=== [smoke] textvqa n=8 post-main: branch-execution check ==="
  run_cell textvqa 8 $FS/textvqa_val.jsonl "$STD" smoke_textvqa_n8
  $PY - <<'PYEOF'
import json
d = json.load(open('results/dcc_presubmit/expA/smoke_textvqa_n8.json'))
diag = d.get('diag') or {}
print('[smoke] score_scope =', diag.get('score_scope'),
      '| n_deepstack =', diag.get('n_deepstack'),
      '| main_hidden =', diag.get('main_hidden'),
      '| fires =', diag.get('fires'),
      '| nk =', diag.get('nk'))
print('[smoke] post_score_scope field =', d.get('post_score_scope'),
      '| n_skipped =', d.get('n_skipped'), '/', d.get('n'),
      '| mean_ptid =', d.get('mean_ptid_len'))
assert diag.get('score_scope') == 'main' and diag.get('n_deepstack') == 3
assert diag.get('main_hidden') == 4096, diag.get('main_hidden')  # Qwen3-VL-8B llm hidden
assert diag.get('fires', 0) > 0 and d.get('n_skipped') == 0
print('[smoke] PASS')
PYEOF
  [ $? -ne 0 ] && { echo "[expA][ABORT] smoke failed"; exit 1; }
fi

if [ "$STAGE" = "validate" ] || [ "$STAGE" = "all" ]; then
  echo "=== [validate] n=200 technical check: IDs + per-image final token counts vs pre-final ==="
  run_cell textvqa 200 eval/subsets/textvqa_200.jsonl "$STD" n200_textvqa_main
  run_cell docvqa 200 eval/subsets/docvqa_200.jsonl "$DOC" n200_docvqa_main
  $PY scripts/analyze_dcc_expA.py --stage validate --out-dir $OUT
  [ $? -ne 0 ] && { echo "[expA][ABORT] validate failed"; exit 1; }
fi

if [ "$STAGE" = "full" ] || [ "$STAGE" = "all" ]; then
  wait_gpu
  echo "=== [full] textvqa n=5000 post-main ==="
  run_cell textvqa 5000 $FS/textvqa_val.jsonl "$STD" full_textvqa_main
  wait_gpu
  echo "=== [full] docvqa n=5349 post-main ==="
  run_cell docvqa 5349 $FS/docvqa_val.jsonl "$DOC" full_docvqa_main
  $PY scripts/analyze_dcc_expA.py --stage validate --out-dir $OUT
  $PY scripts/analyze_dcc_expA.py --stage full --out-dir $OUT
fi

if [ "$STAGE" = "extra" ] || [ "$STAGE" = "all" ]; then
  wait_gpu
  echo "=== [extra] ocrbench n=1000 post-main (lowmem) ==="
  run_cell ocrbench 1000 $FS/ocrbench.jsonl "" full_ocrbench_main lowmem
  wait_gpu
  echo "=== [extra] gqa n=12578 post-main (lowmem) ==="
  run_cell gqa 12578 $FS/gqa_testdev.jsonl "" full_gqa_main lowmem
  $PY scripts/analyze_dcc_expA.py --stage full --out-dir $OUT --extra
fi
echo "[expA] stage '$STAGE' done"

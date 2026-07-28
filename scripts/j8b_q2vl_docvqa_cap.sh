#!/bin/bash
# J8b — Qwen2.5-VL DocVQA same-input 600k {none,pre,post} n200 (Table 1 footnote i cross-check)
set -u
cd /media/disk2/YZX/research/vla
PYQ3=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
OUT=runs/full_matrix/ablations
echo "[J8b] waiting for GPU"; for i in $(seq 1 360); do FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "$FREE" -gt 30000 ] && break; sleep 60; done
for MODE in none pre post; do
  R=0.75; [ "$MODE" = "none" ] && R=0.0
  TAG=j8b_qwen2vl_${MODE}_docvqa_cap600k_n200
  [ -s $OUT/$TAG.json ] && { echo "[skip] $TAG"; continue; }
  echo "=== $TAG ==="
  timeout 2400 $PYQ3 src/v3_premerger/v3_premerger_runner.py --model-family qwen2vl --benchmark docvqa \
    --subset eval/subsets/docvqa_200.jsonl --n 200 --r $R --mode $MODE --max-pixels 600000 \
    --max-num-seqs 4 --max-model-len 16384 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.9 \
    --out $OUT/$TAG.json > $OUT/$TAG.log 2>&1
  tail -1 $OUT/$TAG.log
done
python3 - <<'PY'
import json,glob,sys
sys.path.insert(0,'src/v3_premerger')
import official_scorers as S
for f in sorted(glob.glob('runs/full_matrix/ablations/j8b_*.json')):
    d=json.load(open(f)); ps=d.get('per_sample') or []
    preds=[str(p.get('answer','')) for p in ps]; gts=[str(p.get('gt','')) for p in ps]
    off=sum(S.score_docvqa_anls(a,g) for a,g in zip(preds,gts))/max(len(ps),1)
    print(f.split('/')[-1],'| ANLS=',round(off,4),'| ptid=',d.get('mean_ptid_len'),'| skip=',d.get('n_skipped'))
PY
echo "[J8b done] $(date -u '+%F %T')"

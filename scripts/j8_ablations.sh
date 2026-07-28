#!/bin/bash
# J8 — ablation closure (subset n=200, official metrics):
#  (A) budget curve r{0.25,0.5,0.75} pre/post dual model x textvqa/docvqa (r0.75/0.875 exist from J2)
#  (B) selector {l2, attn} pre/post x textvqa/docvqa dual model (qwen2vl l2 exists from J3; fills the rest)
#  (C) mask granularity sanity: unit (2x2, default) vs token-level note (doc only)
set -u
cd /media/disk2/YZX/research/vla
PYQ3=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
OUT=runs/full_matrix/ablations
mkdir -p $OUT
echo "[J8] $(date -u '+%F %T') waiting for >= 30000 MiB free"
for i in $(seq 1 360); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$FREE" -gt 30000 ]; then echo "[J8] GPU free ${FREE} MiB"; break; fi
  sleep 60
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE" -le 30000 ]; then echo "[J8][ABORT] GPU busy after 6h"; exit 1; fi
STD="--max-num-seqs 8 --max-model-len 8192 --gpu-memory-utilization 0.9"
DOC="--max-num-seqs 4 --max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9"
cell(){ # fam mode bench r sel
  local FAM=$1 MODE=$2 B=$3 R=$4 SEL=$5
  local FLAGS=$STD; [ "$B" = "docvqa" ] && FLAGS=$DOC
  local TAG=$(printf "j8_%s_%s_%s_%s_r%.2f_n200" "$FAM" "$SEL" "$MODE" "$B" "$R")
  if [ -s $OUT/$TAG.json ]; then echo "[skip] $TAG"; return; fi
  echo "=== $TAG ==="
  timeout 1700 $PYQ3 src/v3_premerger/v3_premerger_runner.py --model-family $FAM --benchmark $B \
    --subset eval/subsets/${B}_200.jsonl --n 200 --r $R --mode $MODE --selector $SEL $FLAGS \
    --out $OUT/$TAG.json 2>&1 | tail -2
}
# (A) budget curve: r 0.25 (keep 75%) and 0.5 (keep 50%); 0.75/0.875 from J2
for FAM in qwen3vl qwen2vl; do
  for B in textvqa docvqa; do
    for R in 0.25 0.5; do
      cell $FAM pre  $B $R l2
      cell $FAM post $B $R l2
    done
  done
done
# (B) selector invariance fill: attn pre/post x textvqa/docvqa x dual model
for FAM in qwen3vl qwen2vl; do
  for B in textvqa docvqa; do
    cell $FAM pre  $B 0.75 attn
    cell $FAM post $B 0.75 attn
  done
done
echo "=== J8 official rescore ==="
python - <<'EOF'
import json,glob,sys
sys.path.insert(0,'src/v3_premerger')
import official_scorers as S
rows=[]
for f in sorted(glob.glob('runs/full_matrix/ablations/j8_*.json')):
    d=json.load(open(f)); ps=d.get('per_sample') or []; b=d['benchmark']; n=len(ps)
    if not n: print(f.split('/')[-1],'EMPTY'); continue
    preds=[str(p.get('answer','')) for p in ps]; gts=[str(p.get('gt','')) for p in ps]
    off = (sum(S.score_textvqa_vqaacc(a,g) for a,g in zip(preds,gts))/n if b=='textvqa'
           else sum(S.score_docvqa_anls(a,g) for a,g in zip(preds,gts))/n if b=='docvqa' else None)
    rows.append({'file':f.split('/')[-1],'mode':d.get('mode'),'selector':d.get('selector'),'r':d.get('r'),'model':d.get('model_family'),'official':round(off,4),'ptid':d.get('mean_ptid_len')})
    print(f.split('/')[-1],'| official=',round(off,4),'| ptid=',d.get('mean_ptid_len'))
json.dump(rows,open('runs/full_matrix/ablations/j8_summary.json','w'),indent=1,ensure_ascii=False)
EOF
echo "[J8 done] $(date -u '+%F %T')"

#!/bin/bash
# J7 (vLLM part) — official FULL-split main table cells: {none,pre,post} x 2 models x 4 benches.
# HF baseline cells (fastv/pyramid) queued separately after J4 equivalence gate + speed probe.
# Headline cells (none+pre, both models) FIRST to bank the core claim early.
set -u
cd /media/disk2/YZX/research/vla
PYQ3=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
OUT=runs/full_matrix
mkdir -p $OUT
echo "[J7] $(date -u '+%F %T') start; waiting for >= 30000 MiB free"
for i in $(seq 1 360); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$FREE" -gt 30000 ]; then echo "[J7] GPU free ${FREE} MiB"; break; fi
  sleep 60
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE" -le 30000 ]; then echo "[J7][ABORT] GPU busy after 6h"; exit 1; fi
GMU=0.9
STD="--max-num-seqs 8 --max-model-len 8192 --gpu-memory-utilization $GMU"
DOC="--max-num-seqs 4 --max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization $GMU"
FS=eval/full_splits
# bench -> full jsonl + n + flag-set name
declare -A JSONL=( [textvqa]=$FS/textvqa_val.jsonl [docvqa]=$FS/docvqa_val.jsonl [ocrbench]=$FS/ocrbench.jsonl [gqa]=$FS/gqa_testdev.jsonl )
declare -A N=( [textvqa]=5000 [docvqa]=5349 [ocrbench]=1000 [gqa]=12578 )
run_cell(){ # fam bench mode r "flags" tag  (full log kept; tail echoed)
  timeout 21000 $PYQ3 src/v3_premerger/v3_premerger_runner.py --model-family $1 --benchmark $2 \
    --subset ${JSONL[$2]} --n ${N[$2]} --r $4 --mode $3 $5 --out $OUT/$6.json > $OUT/$6.log 2>&1
  tail -2 $OUT/$6.log
}
skip_ratio(){ python3 -c "
import json
try:
    d=json.load(open('$OUT/$1.json')); print((d.get('n_skipped') or 0)/max(len(d.get('per_sample') or [1]),1))
except Exception: print(1.0)
"; }
cell(){ # fam mode bench r  (r=0 for none); self-healing: skip-ratio>25% -> rerun with safe flags
  local FAM=$1 MODE=$2 B=$3 R=$4
  local FLAGS=$STD; [ "$B" = "docvqa" ] && FLAGS=$DOC
  local TAG=$(printf "j7_%s_%s_%s_r%.3f_full" "$FAM" "$MODE" "$B" "$R")
  if [ -s $OUT/$TAG.json ]; then
    SR=$(skip_ratio $TAG)
    if [ "$(python3 -c "print(1 if float('$SR')<=0.25 else 0)")" = "1" ]; then
      echo "=== $TAG EXISTS (skip_ratio=$SR), skip ==="; return
    fi
    echo "=== $TAG EXISTS but skip_ratio=$SR > 0.25 -> safe-flags retry ==="
  fi
  echo "=== $TAG n=${N[$B]} ==="
  run_cell $FAM $B $MODE "$R" "$FLAGS" $TAG
  SR=$(skip_ratio $TAG)
  if [ "$(python3 -c "print(1 if float('$SR')<=0.25 else 0)")" != "1" ]; then
    echo "=== $TAG retry (skip_ratio=$SR): mns4 + chunk + ocr pixel cap ==="
    local SAFE="--max-num-seqs 4 --max-model-len 8192 --gpu-memory-utilization 0.9 --chunk-size 200"
    [ "$B" = "docvqa" ] && SAFE="--max-num-seqs 4 --max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9 --chunk-size 400"
    [ "$B" = "ocrbench" ] && SAFE="$SAFE --max-pixels 4000000"
    run_cell $FAM $B $MODE "$R" "$SAFE" $TAG
    echo "=== $TAG retry done (skip_ratio=$(skip_ratio $TAG)) ==="
  fi
}
# ---- Wave 1: headline (none + pre) both models, all benches ----
for FAM in qwen3vl qwen2vl; do
  for B in textvqa docvqa ocrbench gqa; do
    cell $FAM none $B 0.0
    cell $FAM pre  $B 0.75
  done
done
# ---- Wave 2: post (== VisionZip principle-port) both models ----
for FAM in qwen3vl qwen2vl; do
  for B in textvqa docvqa ocrbench gqa; do
    cell $FAM post $B 0.75
  done
done
# ---- Wave 3: deep budget r=0.875 (12.5% keep) pre/post, text-dense benches ----
for FAM in qwen3vl qwen2vl; do
  for B in textvqa docvqa ocrbench; do
    cell $FAM pre  $B 0.875
    cell $FAM post $B 0.875
  done
done
echo "=== J7-vLLM official rescore (CPU) ==="
python3 - <<'EOF'
import json,glob,sys
sys.path.insert(0,'src/v3_premerger')
import official_scorers as S
qt={}
for src in ['eval/full_splits/ocrbench.jsonl','eval/subsets/ocrbench_200.jsonl']:
    try:
        for ln in open(src):
            o=json.loads(ln); ex=o.get('extras') or {}
            qt.setdefault(str(o['id']),(ex.get('question_type',''), ex.get('category')))
    except Exception as e: print('meta warn',src,e)
rows=[]
for f in sorted(glob.glob('runs/full_matrix/j7_*.json')):
    d=json.load(open(f)); ps=d.get('per_sample') or []; b=d.get('benchmark'); n=len(ps)
    if not n: print(f,'NO per_sample'); continue
    preds=[str(p.get('answer','')) for p in ps]; gts=[str(p.get('gt','')) for p in ps]
    if b=='textvqa': off=sum(S.score_textvqa_vqaacc(a,g) for a,g in zip(preds,gts))/n; m='VQA-acc'
    elif b=='docvqa': off=sum(S.score_docvqa_anls(a,g) for a,g in zip(preds,gts))/n; m='ANLS'
    elif b=='gqa': off=S.score_gqa_batch(preds,gts)['acc']; m='acc-official'
    elif b=='ocrbench':
        items=[(a,g)+tuple(qt.get(str(p.get('id')),('',''))) for a,g,p in zip(preds,gts,ps)]
        r=S.score_ocrbench_batch(items); off=r.get('acc'); m='acc (Final/1000=%s)'%r.get('final_score')
    else: off=None; m='?'
    rows.append({'file':f.split('/')[-1],'bench':b,'mode':d.get('mode'),'r':d.get('r'),'model':d.get('model'),'n':n,'official':round(off,4) if isinstance(off,float) else off,'metric':m,'ptid':d.get('mean_ptid_len')})
    print(f.split('/')[-1],'| official=',round(off,4) if isinstance(off,float) else off,m,'| n=',n,'| ptid=',d.get('mean_ptid_len'))
json.dump(rows,open('runs/full_matrix/j7_official_summary.json','w'),indent=1,ensure_ascii=False)
print('wrote runs/full_matrix/j7_official_summary.json')
EOF
echo "[J7-vLLM done] $(date -u '+%F %T')"

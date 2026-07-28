#!/bin/bash
# J7 HF-baseline part — FastV + Pyramid(canonical) at n=500 (headline n) on
# BOTH models. DocVQA at 600k-pixel cap (same-input; HF attention
# materialization infeasible at native doc resolutions). Qwen2.5-VL HF path
# gets an n=8 smoke gate first (never run on real qwen2vl weights before).
set -u
cd /media/disk2/YZX/research/vla
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
OUT=runs/full_matrix
HF=src/v3_premerger/baselines_hf.py
FS=eval/full_splits
echo "[J7hf] $(date -u '+%F %T') start; waiting for >= 30000 MiB free"
for i in $(seq 1 720); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$FREE" -gt 30000 ]; then echo "[J7hf] GPU free ${FREE} MiB"; break; fi
  sleep 60
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE" -le 30000 ]; then echo "[J7hf][ABORT] GPU busy after 12h"; exit 1; fi
hfrun(){ # model famtag bench mode r ratios tagx npix n
  local MODEL=$1 FAM=$2 B=$3 MODE=$4 R=$5 RATIOS=$6 TAGX=$7 NPIX=$8 N=$9
  local SUB=$FS/${B}_$( [ "$B" = "textvqa" ] && echo val || { [ "$B" = "docvqa" ] && echo val || { [ "$B" = "gqa" ] && echo testdev || echo ""; }; }; ).jsonl
  [ "$B" = "ocrbench" ] && SUB=$FS/ocrbench.jsonl
  local PX="--max-pixels 0"; [ "$NPIX" != "0" ] && PX="--max-pixels $NPIX"
  local RT=""; [ -n "$RATIOS" ] && RT="--pyramid-ratios $RATIOS"
  local FV=""; [ "$MODE" = "fastv" ] && FV="--fastv-k 2"
  local TAG=$(printf "j7hf_%s_%s_%s_r%.3f%s_n%s" "$FAM" "$MODE" "$B" "$R" "$TAGX" "$N")
  if [ -s $OUT/$TAG.json ]; then echo "[skip] $TAG"; return; fi
  echo "=== $TAG ==="
  timeout 21000 python $HF --mode $MODE --model $MODEL --benchmark $B --subset $SUB \
    --n $N --r $R $FV $RT $PX --seed 0 --out $OUT/$TAG.json > $OUT/$TAG.log 2>&1 \
    && echo "[done] $TAG $(python -c "import json;d=json.load(open('$OUT/$TAG.json'));print('acc=%.3f ptid=%.0f skip=%d'%(d['acc'],d['mean_ptid_len'],d['n_skipped']))" 2>/dev/null)" \
    || echo "[fail] $TAG (see log)"
}
Q3=Qwen/Qwen3-VL-8B-Instruct
Q2=Qwen/Qwen2.5-VL-7B-Instruct
# ---- qwen2vl HF smoke gate (n=8 fastv gqa; pass: skip=0 acc>=0.3) ----
SMOKE=$OUT/j7hf_qwen2vl_smoke_fastv_gqa_n8.json
if [ ! -s $SMOKE ]; then
  timeout 900 python $HF --mode fastv --model $Q2 --benchmark gqa --subset $FS/gqa_testdev.jsonl \
    --n 8 --r 0.75 --fastv-k 2 --seed 0 --out $SMOKE > $OUT/j7hf_qwen2vl_smoke.log 2>&1
fi
python -c "import json;d=json.load(open('$SMOKE'));assert d['n_skipped']==0 and d['acc']>=0.3;print('[smoke] qwen2vl HF PASS acc=%.3f'%d['acc'])" 2>/dev/null \
  || { echo "[smoke] qwen2vl HF FAIL -- skipping qwen2vl HF cells (report qwen3-only baselines + note)"; Q2=""; }
# ---- n=500 cells: fastv + pyramid-canon x 4 benches x models ----
for B in textvqa ocrbench gqa docvqa; do
  NPIX=0; [ "$B" = "docvqa" ] && NPIX=600000
  hfrun $Q3 qwen3vl $B fastv  0.75  ""        ""      $NPIX 500
  hfrun $Q3 qwen3vl $B pyramid 0.375 "1.0,0.75,0.5,0.25" "_canon" $NPIX 500
  [ -n "$Q2" ] && hfrun $Q2 qwen2vl $B fastv  0.75  ""        ""      $NPIX 500
  [ -n "$Q2" ] && hfrun $Q2 qwen2vl $B pyramid 0.375 "1.0,0.75,0.5,0.25" "_canon" $NPIX 500
done
echo "=== J7hf official rescore ==="
python - <<'EOF'
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
for f in sorted(glob.glob('runs/full_matrix/j7hf_*.json')):
    d=json.load(open(f)); ps=d.get('per_sample') or []; b=d.get('benchmark'); n=len(ps)
    if not n: print(f.split('/')[-1],'EMPTY'); continue
    preds=[str(p.get('answer','')) for p in ps]; gts=[str(p.get('gt','')) for p in ps]
    if b=='textvqa': off=sum(S.score_textvqa_vqaacc(a,g) for a,g in zip(preds,gts))/n; m='VQA-acc'
    elif b=='docvqa': off=sum(S.score_docvqa_anls(a,g) for a,g in zip(preds,gts))/n; m='ANLS'
    elif b=='gqa': off=S.score_gqa_batch(preds,gts)['acc']; m='acc'
    elif b=='ocrbench':
        items=[(a,g)+tuple(qt.get(str(p.get('id')),('',''))) for a,g,p in zip(preds,gts,ps)]
        r=S.score_ocrbench_batch(items); off=r.get('acc'); m='acc(Final=%s)'%r.get('final_score')
    else: off=None; m='?'
    rows.append({'file':f.split('/')[-1],'official':round(off,4) if isinstance(off,float) else off,'n':n,'ptid':d.get('mean_ptid_len'),'skip':d.get('n_skipped')})
    print(f.split('/')[-1],'| official=',round(off,4) if isinstance(off,float) else off,m,'| n=',n,'| skip=',d.get('n_skipped'))
json.dump(rows,open('runs/full_matrix/j7hf_official_summary.json','w'),indent=1,ensure_ascii=False)
EOF
echo "[J7hf done] $(date -u '+%F %T')"

#!/bin/bash
# Qwen3-VL PRE-FINAL pure-stage confound control (P0-3).
#
# The paper's stage-law claim: ranking BEFORE the lossy 2x2 main merger ("pre")
# beats ranking AFTER it ("post") at iso-token budget. CONFOUND: "pre" ranks at
# deepstack[0]-input (layer-8 features, far upstream) and slices ALL 4 mergers;
# "post" ranks at the main-merger OUTPUT (downstream). They differ in BOTH stage
# AND feature space.
#
# This gate isolates the pure STAGE effect: "pre-final" ranks at visual.merger's
# OWN input (the final ViT-block features = main-merger INPUT, post-deepstack)
# and selects k_units from the visual output. Deepstack mergers run UNTOUCHED.
# By 2x2 merge-unit equivalence, this is bit-identical to slicing the merger
# input. Compared to "post" (ranks at main-merger OUTPUT), the ONLY difference is
# BEFORE vs AFTER the 2x2 merge -- feature space is roughly held constant (both
# are post-deepstack, main-merger vicinity).
#
# Modes (all greedy temp=0; Qwen3-VL is not a thinking model):
#   none       baseline anchor (no pruning)
#   pre-final  rank at main-merger INPUT, select k_units from visual output
#   post       rank at main-merger OUTPUT, select k_units from visual output
#
# Iso-token contract: pre-final and post both keep k_i=round(full_i*(1-r)) units
# per image (r=0.75 -> keep 25%). The rescore step VERIFIES prompt_token_ids
# match between pre-final and post for every sample (iso-token check).
#
# DECISION RULE (P0-3):
#   If TextVQA/DocVQA still show significant pre-final > post  ->  pure-stage
#   control confirmed (the stage effect is real, not a feature-space artifact).
#   Write into paper. If the direction vanishes or shrinks dramatically
#   (pre-final ~= post)  ->  CLAIM-LEVEL EVENT: the pre>post gap was driven by
#   the feature-space confound (deepstack[0]-input vs main-merger-output), not
#   the stage. FLAG, do NOT self-package -- main window escalates.
#
# Usage: bash src/v3_premerger/qwen3_prefinal_control.sh
# Outputs -> runs/qwen3_prefinal_control/
set -u
cd /media/disk2/YZX/research/vla
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_MODELSCOPE=False

OUT=runs/qwen3_prefinal_control
mkdir -p $OUT
FAM=qwen3vl
R=0.75
N=200
MAXTOK=32
MODES="none pre-final post"

# ---- wait for an idle GPU (>= 40000 MiB free) ----
echo "[prefinal] $(date -u '+%F %T') modes='$MODES' waiting >= 40000 MiB free"
for i in $(seq 1 120); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$FREE" -gt 40000 ] && { echo "[prefinal] GPU free ${FREE} MiB"; break; }
  sleep 30
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$FREE" -le 40000 ] && { echo "[prefinal][ABORT] GPU busy after wait"; exit 1; }

STD="--max-num-seqs 16 --chunk-size 250 --max-model-len 8192 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9"
DOC="--max-num-seqs 4 --chunk-size 250 --max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.85"
declare -A JSONL=( [textvqa]=eval/subsets/textvqa_200.jsonl
                   [docvqa]=eval/subsets/docvqa_200.jsonl
                   [gqa]=eval/subsets/gqa_200.jsonl )

run_cell(){ # bench mode r "flags" tag
  timeout 5400 $PY src/v3_premerger/v3_premerger_runner.py --model-family $FAM --benchmark $1 \
    --subset ${JSONL[$1]} --n $N --r $3 --mode $2 --selector l2 --max-tokens $MAXTOK \
    $4 --out $OUT/$5.json > $OUT/$5.log 2>&1
  tail -2 $OUT/$5.log
}
skip_ratio(){ $PY -c "
import json
try:
    d=json.load(open('$OUT/$1.json')); print((d.get('n_skipped') or 0)/max(len(d.get('per_sample') or [1]),1))
except Exception: print(1.0)
"; }
cell(){ # mode bench r
  local MODE=$1 B=$2 R=$3
  local FLAGS=$STD; [ "$B" = "docvqa" ] && FLAGS=$DOC
  local TAG=$(printf "qwen3_%s_%s_r%.3f_n%d" "$MODE" "$B" "$R" "$N")
  if [ -s $OUT/$TAG.json ]; then
    SR=$(skip_ratio $TAG)
    if [ "$($PY -c "print(1 if float('$SR')<=0.10 else 0)")" = "1" ]; then
      echo "=== $TAG EXISTS (skip=$SR), skip ==="; return
    fi
    echo "=== $TAG EXISTS but skip=$SR > 0.10 -> retry ==="
  fi
  echo "=== $TAG n=$N (greedy temp=0) ==="
  run_cell $B $MODE "$R" "$FLAGS" $TAG
  echo "=== $TAG done (skip=$(skip_ratio $TAG)) ==="
}

for MODE in $MODES; do
  for B in textvqa docvqa gqa; do cell $MODE $B $R; done
done

echo "=== prefinal rescore + iso-token verify (CPU) ==="
$PY - <<'PYEOF'
import json, glob, os, sys, statistics as st
sys.path.insert(0,'src')
from v3_premerger.official_scorers import score_textvqa_vqaacc, score_docvqa_anls, score_gqa
OUT='runs/qwen3_prefinal_control'
SCORERS={'textvqa':score_textvqa_vqaacc,'docvqa':score_docvqa_anls,'gqa':score_gqa}
rows=[]
for f in sorted(glob.glob(os.path.join(OUT,'qwen3_*_n200.json'))):
    d=json.load(open(f)); b=d.get('benchmark'); ps=d.get('per_sample') or []
    if b not in SCORERS or not ps: continue
    sc=SCORERS[b]; vals=[]; n_skip=0
    for p in ps:
        if p.get('skipped'): n_skip+=1; continue
        vals.append(float(sc(p.get('answer',''),str(p.get('gt','')))))
    n_ans=len(vals)
    off=sum(vals)/n_ans if n_ans else 0.0
    rows.append({'file':os.path.basename(f),'benchmark':b,'mode':d.get('mode'),
                 'r':d.get('r'),'n':len(ps),'n_skipped':n_skip,'n_answered':n_ans,
                 'official_score':round(off,4),
                 'mean_ptid_len':d.get('mean_ptid_len'),'wall_s':d.get('wall_s')})
    print(f"[rescore] {rows[-1]['file']}: off={off:.4f} skip={n_skip}/{len(ps)} "
          f"mean_ptid={rows[-1]['mean_ptid_len']}")
tab={}
ps_cache={}
for r in rows:
    tab[(r['benchmark'],r['mode'])]=r['official_score']
def load_ps(mode,bench):
    key=(mode,bench)
    if key in ps_cache: return ps_cache[key]
    for r in rows:
        if r['mode']==mode and r['benchmark']==bench:
            d=json.load(open(os.path.join(OUT,r['file'])))
            m={p['id']:p for p in (d.get('per_sample') or [])}
            ps_cache[key]=m; return m
    ps_cache[key]={}; return {}

# ---- iso-token verification: pre-final vs post prompt_token_ids match ----
iso_rows=[]
print('\n=== ISO-TOKEN CHECK (pre-final vs post prompt_token_ids) ===')
for b in ('textvqa','docvqa','gqa'):
    pf=load_ps('pre-final',b); po=load_ps('post',b)
    ids=sorted(set(pf)&set(po))
    if not ids: print(f"{b:8s} no paired samples"); continue
    match=0; mismatch=0; deltas=[]
    for i in ids:
        if pf[i].get('skipped') or po[i].get('skipped'): continue
        pt_pf=pf[i].get('prompt_token_ids',0); pt_po=po[i].get('prompt_token_ids',0)
        if pt_pf==pt_po: match+=1
        else: mismatch+=1; deltas.append((pt_pf,pt_po))
    iso_rows.append({'benchmark':b,'n_paired':match+mismatch,'match':match,'mismatch':mismatch})
    status='PASS' if mismatch==0 else 'FAIL'
    print(f"{b:8s} {status}  match={match} mismatch={mismatch}  "
          f"{'(iso-token confirmed)' if mismatch==0 else '(ISO-TOKEN VIOLATION!)'}")

# ---- paired delta pre-final vs post ----
print('\n=== PAIRED DELTA (pre-final vs post) ===')
delta_rows=[]
for b in ('textvqa','docvqa','gqa'):
    nn=tab.get((b,'none')); pf_s=tab.get((b,'pre-final')); po_s=tab.get((b,'post'))
    line=f"{b:8s} none={nn} pre-final={pf_s} post={po_s}"
    if pf_s is not None and po_s is not None:
        d_pp=round((pf_s-po_s)*100,2)
        pfm=load_ps('pre-final',b); pom=load_ps('post',b)
        ids=sorted(set(pfm)&set(pom)); sc=SCORERS[b]
        diffs=[float(sc(pfm[i].get('answer',''),str(pfm[i].get('gt',''))))
               -float(sc(pom[i].get('answer',''),str(pom[i].get('gt',''))))
               for i in ids if not pfm[i].get('skipped') and not pom[i].get('skipped')]
        mean=st.mean(diffs) if diffs else 0.0
        sd=st.stdev(diffs) if len(diffs)>1 else 0.0
        se=sd/(len(diffs)**0.5) if diffs else 0.0
        delta_rows.append({'benchmark':b,'n_paired':len(diffs),
                           'mean_delta':round(mean,4),'stderr':round(se,4),
                           'd_pp':d_pp})
        line+=f"  d(pre-final-post)={d_pp}pp  paired(n={len(diffs)}) meanD={mean:.4f}+-{se:.4f}"
    print(line)

# ---- decision rule ----
print('\n=== DECISION RULE ===')
for dr in delta_rows:
    d=dr['d_pp']
    if d>1.0:
        verdict=f"pre-final > post by {d}pp -> STAGE EFFECT CONFIRMED (pure-stage control). Write into paper."
    elif d<-1.0:
        verdict=f"pre-final < post by {d}pp -> REVERSED. CLAIM-LEVEL EVENT: flag, do NOT self-package."
    else:
        verdict=f"pre-final ~= post (|d|={abs(d)}pp<=1) -> DIRECTION VANISHED. CLAIM-LEVEL EVENT: flag, escalate."
    print(f"  {dr['benchmark']:8s} {verdict}")

summary={'gate':'qwen3_prefinal_pure_stage_control','protocol':'greedy temp=0',
         'keep_frac':0.25,'selector':'l2','model':'Qwen/Qwen3-VL-8B-Instruct',
         'rows':rows,'paired_delta':delta_rows,'iso_token':iso_rows}
with open(os.path.join(OUT,'prefinal_summary.json'),'w') as fh: json.dump(summary,fh,indent=2)
PYEOF
echo "=== PREFINAL CONTROL DONE ==="

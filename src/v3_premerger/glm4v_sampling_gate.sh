#!/bin/bash
# GLM-4.1V-9B-Thinking OFFICIAL SAMPLING GATE (P0-1).
#   Re-runs the fourth-family stage-law gate with the model's PUBLISHED sampling
#   protocol (generation_config.json: do_sample=True, temperature=0.8, top_p=0.6,
#   top_k=2) instead of greedy. Greedy violated the protocol and floor-collapsed
#   GQA (none EM 0.15 vs official ~0.77; containment anchor 0.775 proved the
#   model is capable). This gate checks whether the none anchor recovers under
#   the official protocol, then seeds 0/1/2 give paired delta + mean+-std.
#
#   Usage: glm4v_sampling_gate.sh SEED [MODES...]   (MODES default: "none pre post")
#     Stage 1 (anchor):  glm4v_sampling_gate.sh 0 none
#     Stage 2 (delta):   glm4v_sampling_gate.sh 0 pre post
#                        glm4v_sampling_gate.sh 1 pre post
#                        glm4v_sampling_gate.sh 2 pre post
#
#   Same sample uses the same --seed across none/pre/post arms (seed is set
#   per-request on SamplingParams) -> paired delta is isolated to pruning.
#   Outputs -> runs/glm4v_gate_sampling/glm4v_{mode}_{bench}_r0.750_s{SEED}_n200.json
#   (greedy runs in runs/glm4v_gate/ are preserved, NOT overwritten.)
set -u
cd /media/disk2/YZX/research/vla
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 VLLM_USE_MODELSCOPE=False

SEED=${1:?usage: $0 SEED [MODES...]}
shift || true
MODES="${*:-none pre post}"

OUT=runs/glm4v_gate_sampling
mkdir -p $OUT
FAM=glm4v
R=0.75
MAXTOK=1024
TEMP=0.8; TOPP=0.6; TOPK=2
SAMP="--temperature $TEMP --top-p $TOPP --top-k $TOPK --seed $SEED"

# ---- wait for an idle GPU (>= 40000 MiB free) ----
echo "[glm4v_sampling] $(date -u '+%F %T') seed=$SEED modes='$MODES' waiting >= 40000 MiB free"
for i in $(seq 1 120); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  [ "$FREE" -gt 40000 ] && { echo "[glm4v_sampling] GPU free ${FREE} MiB"; break; }
  sleep 30
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
[ "$FREE" -le 40000 ] && { echo "[glm4v_sampling][ABORT] GPU busy after wait"; exit 1; }

STD="--max-num-seqs 16 --chunk-size 250 --max-model-len 8192 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.9"
DOC="--max-num-seqs 4 --chunk-size 250 --max-model-len 32768 --max-num-batched-tokens 32768 --gpu-memory-utilization 0.85"
declare -A JSONL=( [textvqa]=eval/subsets/textvqa_200.jsonl
                   [docvqa]=eval/subsets/docvqa_200.jsonl
                   [gqa]=eval/subsets/gqa_200.jsonl )
N=200

run_cell(){ # bench mode r "flags" tag
  timeout 5400 $PY src/v3_premerger/v3_premerger_runner.py --model-family $FAM --benchmark $1 \
    --subset ${JSONL[$1]} --n $N --r $3 --mode $2 --selector l2 --max-tokens $MAXTOK \
    $SAMP $4 --out $OUT/$5.json > $OUT/$5.log 2>&1
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
  local TAG=$(printf "glm4v_%s_%s_r%.3f_s%d_n200" "$MODE" "$B" "$R" "$SEED")
  if [ -s $OUT/$TAG.json ]; then
    SR=$(skip_ratio $TAG)
    if [ "$($PY -c "print(1 if float('$SR')<=0.10 else 0)")" = "1" ]; then
      echo "=== $TAG EXISTS (skip=$SR), skip ==="; return
    fi
    echo "=== $TAG EXISTS but skip=$SR > 0.10 -> retry ==="
  fi
  echo "=== $TAG n=$N (temp=$TEMP top_p=$TOPP top_k=$TOPK seed=$SEED) ==="
  run_cell $B $MODE "$R" "$FLAGS" $TAG
  echo "=== $TAG done (skip=$(skip_ratio $TAG)) ==="
}

for MODE in $MODES; do
  for B in textvqa docvqa gqa; do cell $MODE $B $R; done
done

echo "=== glm4v_sampling official rescore (CPU) seed=$SEED ==="
$PY - <<EOF
import json, glob, os, sys, statistics as st
sys.path.insert(0,'src')
from v3_premerger.official_scorers import score_textvqa_vqaacc, score_docvqa_anls, score_gqa
SEED=$SEED
SCORERS={'textvqa':score_textvqa_vqaacc,'docvqa':score_docvqa_anls,'gqa':score_gqa}
rows=[]
for f in sorted(glob.glob('runs/glm4v_gate_sampling/glm4v_*_s%d_n200.json'%SEED)):
    d=json.load(open(f)); b=d.get('benchmark'); ps=d.get('per_sample') or []
    if b not in SCORERS or not ps: continue
    sc=SCORERS[b]; vals=[]; n_skip=0; gen_lens=[]; boxed=0; fr_len=0
    for p in ps:
        if p.get('skipped'): n_skip+=1; continue
        vals.append(float(sc(p.get('answer',''),str(p.get('gt','')))))
        gl=p.get('gen_len'); gen_lens.append(gl if gl is not None else 0)
        if p.get('boxed'): boxed+=1
        if p.get('finish_reason')=='length': fr_len+=1
    n_ans=len(vals)
    off=sum(vals)/n_ans if n_ans else 0.0
    rows.append({'file':os.path.basename(f),'benchmark':b,'mode':d.get('mode'),'r':d.get('r'),
                 'seed':SEED,'n':len(ps),'n_skipped':n_skip,'n_answered':n_ans,
                 'official_score':round(off,4),
                 'mean_gen_len':round(sum(gen_lens)/len(gen_lens),1) if gen_lens else 0,
                 'boxed_rate':round(boxed/n_ans,3) if n_ans else 0,
                 'length_cutoff_rate':round(fr_len/n_ans,3) if n_ans else 0,
                 'mean_ptid_len':d.get('mean_ptid_len'),'wall_s':d.get('wall_s')})
    print(f"[rescore s{SEED}] {rows[-1]['file']}: off={off:.4f} skip={n_skip}/{len(ps)} "
          f"gen_len={rows[-1]['mean_gen_len']} boxed={rows[-1]['boxed_rate']} len_cutoff={rows[-1]['length_cutoff_rate']}")
tab={}
ps_cache={}
for r in rows:
    tab[(r['benchmark'],r['mode'])]=r['official_score']
def load_ps(mode,bench):
    key=(mode,bench)
    if key in ps_cache: return ps_cache[key]
    for r in rows:
        if r['mode']==mode and r['benchmark']==bench:
            d=json.load(open('runs/glm4v_gate_sampling/'+r['file']))
            m={p['id']:p for p in (d.get('per_sample') or [])}
            ps_cache[key]=m; return m
    ps_cache[key]={}; return {}
print('\\n=== SEED %d OFFICIAL (none/pre/post) + paired d(pre-post) ==='%SEED)
delta_rows=[]
for b in ('textvqa','docvqa','gqa'):
    nn=tab.get((b,'none')); pre=tab.get((b,'pre')); post=tab.get((b,'post'))
    line=f"{b:8s} none={nn} pre={pre} post={post}"
    if pre is not None and post is not None:
        d_pp=round((pre-post)*100,2)
        pp=load_ps('pre',b); po=load_ps('post',b)
        ids=sorted(set(pp)&set(po)); sc=SCORERS[b]
        diffs=[float(sc(pp[i].get('answer',''),str(pp[i].get('gt',''))))-float(sc(po[i].get('answer',''),str(po[i].get('gt',''))))
               for i in ids if not pp[i].get('skipped') and not po[i].get('skipped')]
        mean=st.mean(diffs) if diffs else 0.0
        sd=st.stdev(diffs) if len(diffs)>1 else 0.0
        se=sd/(len(diffs)**0.5) if diffs else 0.0
        delta_rows.append({'benchmark':b,'seed':SEED,'n_paired':len(diffs),
                           'mean_delta':round(mean,4),'stderr':round(se,4),'d_pp':d_pp})
        line+=f"  d(pre-post)={d_pp}pp  paired(n={len(diffs)}) meanD={mean:.4f}+-{se:.4f}"
    print(line)
summary={'gate':'glm4v_sampling_stage_law','protocol':'official sampling temp=0.8/top_p=0.6/top_k=2',
         'keep_frac':0.25,'selector':'l2','seed':SEED,'rows':rows,'paired_delta':delta_rows}
with open('runs/glm4v_gate_sampling/glm4v_sampling_summary_s%d.json'%SEED,'w') as fh: json.dump(summary,fh,indent=2)
EOF
echo "=== GLM4V SAMPLING GATE DONE seed=$SEED ==="

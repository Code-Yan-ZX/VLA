#!/bin/bash
# J6 — efficiency table on Qwen3-VL (A40): req/s + wall + mean prompt tokens
# (runner-provided) + peak GPU memory (nvidia-smi sampler) + single-request
# latency proxy (n=1 x5, offline; note: not serving TTFT under load).
# Cells: mode{none,pre,post} x r{0.25,0.5,0.75} (r=0 for none) on textvqa_200.
set -u
cd /media/disk2/YZX/research/vla
PYQ3=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
OUT=runs/full_matrix/efficiency
mkdir -p $OUT
echo "[J6] $(date -u '+%F %T') waiting for >= 30000 MiB free"
for i in $(seq 1 360); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$FREE" -gt 30000 ]; then echo "[J6] GPU free ${FREE} MiB"; break; fi
  sleep 60
done
STD="--max-num-seqs 8 --max-model-len 8192 --gpu-memory-utilization 0.9"
cell(){ # mode r tag
  local MODE=$1 R=$2 TAG=$3
  if [ -s $OUT/$TAG.json ]; then echo "[skip] $TAG"; return; fi
  echo "=== $TAG ==="
  # peak-mem sampler
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $OUT/$TAG.mem; sleep 1; done ) &
  local SAMP=$!
  timeout 1200 $PYQ3 src/v3_premerger/v3_premerger_runner.py --model-family qwen3vl --benchmark textvqa \
    --subset eval/subsets/textvqa_200.jsonl --n 200 --r $R --mode $MODE $STD --out $OUT/$TAG.json > $OUT/$TAG.log 2>&1
  kill $SAMP 2>/dev/null
  PEAK=$(sort -n $OUT/$TAG.mem 2>/dev/null | tail -1)
  python -c "
import json
d=json.load(open('$OUT/$TAG.json'))
peak=int('${PEAK:-0}')
print('%-28s req/s=%.2f wall=%.1fs ptid=%.0f peak_mem=%dMiB skip=%d'%('$TAG', d.get('req_per_s') or 0, d.get('wall_s') or 0, d.get('mean_ptid_len') or 0, peak, d.get('n_skipped') or 0))
json.dump({'tag':'$TAG','mode':'$MODE','r':$R,'req_per_s':d.get('req_per_s'),'wall_s':d.get('wall_s'),'mean_ptid_len':d.get('mean_ptid_len'),'peak_mem_mib':peak,'n_skipped':d.get('n_skipped')},open('$OUT/$TAG.eff.json','w'),indent=1)
" 2>/dev/null || echo "[fail-parse] $TAG"
}
for R in 0.25 0.5 0.75; do
  cell pre  $R j6_qwen3vl_pre_r$R
  cell post $R j6_qwen3vl_post_r$R
done
cell none 0.0 j6_qwen3vl_none
# ---- single-request latency proxy (n=1 x5, averaged) ----
echo "=== j6 single-request latency (n=1 x5) ==="
for MODE in none pre; do
  for k in 1 2 3 4 5; do
    timeout 600 $PYQ3 src/v3_premerger/v3_premerger_runner.py --model-family qwen3vl --benchmark textvqa \
      --subset eval/subsets/textvqa_200.jsonl --n 1 --r 0.75 --mode $MODE --max-num-seqs 1 \
      --max-model-len 8192 --gpu-memory-utilization 0.9 --out $OUT/j6_lat_${MODE}_$k.json > /dev/null 2>&1
  done
  python -c "
import json,glob
ws=[json.load(open(f)).get('wall_s') for f in sorted(glob.glob('$OUT/j6_lat_${MODE}_*.json'))]
ws=[w for w in ws if w]
print('single-req wall ${MODE}: mean=%.2fs over %d runs'%(sum(ws)/len(ws),len(ws)) if ws else 'no data')
"
done
echo "=== J6 SUMMARY ==="
python -c "
import json,glob
rows=[json.load(open(f)) for f in sorted(glob.glob('$OUT/*.eff.json'))]
print('%-28s %-6s %-5s %-8s %-8s %-7s %-10s'%('tag','mode','r','req/s','wall_s','ptid','peak_MiB'))
for r in rows: print('%-28s %-6s %-5s %-8s %-8s %-7s %-10s'%(r['tag'],r['mode'],r['r'],round(r['req_per_s'] or 0,2),round(r['wall_s'] or 0,1),round(r['mean_ptid_len'] or 0),r['peak_mem_mib']))
"
echo "[J6 done] $(date -u '+%F %T')"

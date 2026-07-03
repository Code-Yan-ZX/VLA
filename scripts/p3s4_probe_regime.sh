#!/usr/bin/env bash
# P3-step-4 STEP 1: find a KV-BOUND regime on 1xA40.
# Sweep (max_num_seqs, max-tokens, gpu-mem) at small n=60 on GQA, fixed r0,
# bursty profile. Read peak_kv_occupancy from the new load_trace field.
# Goal: find a setting where peak KV-occupancy >> 0.5 (KV is the bottleneck,
# not max_num_seqs or compute). Cheap: n=60, ~1-2 min each.
set -uo pipefail
cd /media/disk2/YZX/research/vla

PY=/home/dell/miniconda3/envs/vtc_serve/bin/python
MODEL=runs/models/llava-1.5-7b-hf
OUT=runs/p3s4/probe
mkdir -p "$OUT"

gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }
gpu_settle() {
  for _ in $(seq 1 60); do
    free=$(gpu_free_mib)
    if [ -n "$free" ] && [ "$free" -ge 40000 ]; then break; fi
    echo "[probe] $(date +%H:%M:%S) GPU busy (free=${free}MiB), waiting 10s..."; sleep 10
  done
  sleep 15
}

# probe <name> <mns> <mt> <gmem>
probe () {
  local name=$1 mns=$2 mt=$3 gmem=$4
  local out="$OUT/${name}.json" log="$OUT/${name}.log"
  if [ -f "$out" ]; then echo "[probe] SKIP $name"; return; fi
  gpu_settle
  echo "[probe] $(date +%H:%M:%S) START $name (mns=$mns mt=$mt gmem=$gmem)"
  $PY -m src.serve_bench --model "$MODEL" --benchmark gqa \
    --subset eval/subsets/gqa_500.jsonl --limit 60 \
    --metrics-out "$out" --max-tokens "$mt" \
    --max-model-len 4096 --gpu-memory-utilization "$gmem" --seed 0 \
    --selector proxy --max-num-seqs "$mns" \
    --pruning-rate 0.0 --load-profile bursty > "$log" 2>&1
  local rc=$?
  local trace=""
  trace=$($PY -c "import json;d=json.load(open('$out'));t=d.get('load_trace') or {};print('peak_kv_occ=%.3f peak_nr=%s mns=%s req/s=%.2f'%(t.get('peak_kv_occupancy',-1),t.get('peak_num_running'),t.get('max_num_seqs'),d['agg']['served_req_s']['mean']))" 2>/dev/null)
  echo "[probe] $(date +%H:%M:%S) DONE $name rc=$rc $trace"
  if [ $rc -ne 0 ]; then echo "[probe] FAIL $name tail:"; tail -8 "$log"; fi
}

# Baseline (the n=500 c12/mt32 regime, for contrast) then escalate.
probe base_c12_mt32_g85    12  32  0.85
# raise concurrency
probe c24_mt32_g85         24  32  0.85
probe c32_mt32_g85         32  32  0.85
# raise concurrency + long outputs (grow KV/req)
probe c24_mt256_g85        24 256  0.85
probe c32_mt256_g85        32 256  0.85
probe c48_mt256_g85        48 256  0.85
# shrink KV pool to pressure at lower concurrency
probe c32_mt256_g60        32 256  0.60
probe c48_mt256_g60        48 256  0.60
probe c32_mt512_g60        32 512  0.60
probe c48_mt512_g55        48 512  0.55

echo "[probe] ALL DONE $(date +%H:%M:%S)"

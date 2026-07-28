#!/bin/bash
# InternVL3-8B GPU SMOKE (Part B): validate vLLM load + processor + pre/post hooks
# on tiny slices before the full matrix. Self-serializes on the A40 (waits for an
# idle GPU; R2 campaign + cascade gate run ahead). Uses the COMPLETE HF-cache
# weights (VLLM_USE_MODELSCOPE=False; the modelscope copy is still downloading).
#   none gqa n16         -> sanity acc (InternVL3-8B gqa ~0.6+; gate >=0.45)
#   pre/post gqa n16 r0.75
#   pre/post textvqa n16 r0.75 -> acc>=0.3, no crash; pre>post = mechanism signal
set -u
cd /media/disk2/YZX/research/vla
PY=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
export VLLM_ENABLE_V1_MULTIPROCESSING=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
       VLLM_NO_USAGE_STATS=1 VLLM_USE_MODELSCOPE=False
OUT=runs/internvl3_smoke
mkdir -p $OUT
FS=eval/full_splits

# tiny smoke subsets (first 16 of each full split) -> deterministic n=16
head -16 $FS/gqa_testdev.jsonl > $OUT/gqa_smoke.jsonl
head -16 $FS/textvqa_val.jsonl  > $OUT/textvqa_smoke.jsonl

echo "[smoke] $(date -u '+%F %T') waiting for >= 40000 MiB free GPU"
for i in $(seq 1 360); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$FREE" -gt 40000 ]; then echo "[smoke] GPU free ${FREE} MiB"; break; fi
  sleep 45
done
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$FREE" -le 40000 ]; then echo "[smoke][ABORT] GPU busy after wait"; exit 1; fi

FLAGS="--max-num-seqs 4 --chunk-size 500 --max-model-len 8192 --gpu-memory-utilization 0.9"
run(){ # realbench subsetfile mode r tag
  timeout 3600 $PY src/v3_premerger/v3_premerger_runner.py --model-family internvl3 \
    --benchmark $1 --subset $2 --n 16 --r $4 --mode $3 $FLAGS \
    --out $OUT/$5.json > $OUT/$5.log 2>&1
  echo "--- $5 ---"; tail -1 $OUT/$5.log
}
run gqa    $OUT/gqa_smoke.jsonl    none 0.0  smoke_none_gqa
run gqa    $OUT/gqa_smoke.jsonl    pre  0.75 smoke_pre_gqa
run gqa    $OUT/gqa_smoke.jsonl    post 0.75 smoke_post_gqa
run textvqa $OUT/textvqa_smoke.jsonl pre  0.75 smoke_pre_textvqa
run textvqa $OUT/textvqa_smoke.jsonl post 0.75 smoke_post_textvqa
echo "[smoke] $(date -u '+%F %T') done"

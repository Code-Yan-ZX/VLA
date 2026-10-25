#!/bin/bash
# DocVQA unified pixel-cap probe (directive 12): 4M px fails the None baseline
# (23/32 OOM skips). Walk candidate caps DOWN until None smoke passes (<=2
# OOM skips / 32q); then verify all four methods at the chosen cap.
# Resource-triggered decision, logged; never score-triggered.
set -u
cd "$(git rev-parse --show-toplevel)"
source /home/dell/miniconda3/etc/profile.d/conda.sh && conda activate qwen3vl_clean
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
P=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
HF=src/v3_premerger/baselines_hf.py
D=results/cvm2027_supplement_20261026/predictions/capprobe
L=results/cvm2027_supplement_20261026/logs
Q=results/cvm2027_supplement_20261026/plan
mkdir -p $D
FS=$Q/smoke32_docvqa.jsonl; [ -f $FS ] || head -32 eval/full_splits/docvqa_val.jsonl > $FS
Q3=Qwen/Qwen3-VL-8B-Instruct
res(){ jq -c "{cap:$1,mode:\"$2\",skip:.n_skipped,wall:.wall_s}" "$3" 2>/dev/null | tee -a $Q/doc_cap_probe.jsonl; }

for CAP in 2000000 1500000 1000000 750000; do
  timeout 1800 $P $HF --mode none --model $Q3 --benchmark docvqa --subset $FS \
    --n 32 --max-pixels $CAP --seed 0 --max-tokens 32 \
    --out $D/none_$CAP.json > $L/capnone_$CAP.log 2>&1
  SKIP=$(jq -r '.n_skipped // 99' $D/none_$CAP.json 2>/dev/null)
  res $CAP none $D/none_$CAP.json
  echo "[capprobe] cap=$CAP none skip=$SKIP $(date +%T)"
  if [ "${SKIP:-99}" -le 2 ]; then CHOSEN=$CAP; break; fi
done
[ -z "${CHOSEN:-}" ] && { echo "[capprobe] NO CAP PASSED - manual review"; exit 1; }
for M in "pre --r-pre 0.25 --mrope native" "fastv --r 0.75 --fastv-k 3"; do
  set -- $M; MODE=$1; shift; EXTRA="$*"
  timeout 1800 $P $HF --mode $MODE --model $Q3 --benchmark docvqa --subset $FS \
    --n 32 --max-pixels $CHOSEN --seed 0 --max-tokens 32 $EXTRA \
    --out $D/${MODE}_$CHOSEN.json > $L/cap${MODE}_$CHOSEN.log 2>&1
  res $CHOSEN $MODE $D/${MODE}_$CHOSEN.json
done
timeout 1800 $P scripts/cvm24h_hilo_eval.py --benchmark docvqa --subset $FS --n 32 \
  --keep-frac 0.25 --max-pixels $CHOSEN --seed 0 --out $D/hilo_$CHOSEN.json > $L/caphilo_$CHOSEN.log 2>&1
res $CHOSEN hilo $D/hilo_$CHOSEN.json
echo "[capprobe] CHOSEN_CAP=$CHOSEN" | tee $Q/doc_chosen_cap.txt

# probe_gqa_r0 — GQA control (pruning 0%), n=200

- **status**: done (rc=0) · V0 engine (VLLM_USE_V1=0) · 2026-07-01 16:08 · 172s
- **cmd**: `/home/dell/miniconda3/envs/vtc_serve/bin/python -m src.serve_bench --model runs/models/llava-1.5-7b-hf --pruning-rate 0.0 --benchmark gqa --subset eval/subsets/gqa_200.jsonl --metrics-out runs/p2_probe/gqa_r0_metrics.json --max-tokens 32 --max-model-len 4096 --gpu-memory-utilization 0.90 --seed 0`
- **log**: `runs/probe_gqa_r0.log` · **metrics**: `runs/p2_probe/gqa_r0_metrics.json` (gitignored)

## 结果 / 指标 (the denominator for all GQA speedup ratios)
| metric | mean | stderr |
|---|---|---|
| served_tok_s | **22.81** | 0.23 |
| served_req_s | **1.77** | 0.046 |
| ttft_ms | **631** | 15 |
| peak_kv_mb | ~40484 | — |
| accuracy | 0.01 ⚠️ | — |

prefill_x / e2e_x = None (this IS the r=0 denominator).

## 结论
- **Pipeline VALIDATED at full scale**: V0 engine loads (13.13 GiB weight, 26 GiB KV cache, 13× concurrency), hooks attach (projector + vision_tower), 200 requests served, metrics written. The earlier V1/hook bug is resolved.
- **Throughput denominator captured** — the primary gate signal (≥1.5× prefill / ≥1.2× e2e at r50) is measurable against these numbers and is **unaffected by the scoring bug**.
- ⚠️ **SCORING BUG**: accuracy 0.01 is artificial. Model answers are verbose full sentences ("No, the chair is not in the bottom…") but `score_gqa` does strict exact/fuzzy match — it doesn't extract the lead word or use the GQA answer-set, so correct yes/no & object answers score 0. Does NOT affect the throughput gate, but makes the "≤2% acc drop" secondary criterion meaningless until fixed. **Fix queued for Dev** (after the placeholder patch — same file `serve_bench.py`): GQA official answer-set matching / first-word extraction.

## 下一步
- Dev: placeholder-compact patch (critical path) → then scorer fix.
- Then run probe_gqa_r50 (gate point): compare prefill/e2e vs this r0; re-score accuracy after the scorer fix (may need a re-run of r0 too for a clean acc baseline).

## 产物路径
- metrics: `runs/p2_probe/gqa_r0_metrics.json` (gitignored) · model: `runs/models/llava-1.5-7b-hf` (gitignored)

# probe_gqa_r25

- **status**: done (rc=0)
- **ran**: 2026-07-01 16:27 · 156s
- **cmd**: `/home/dell/miniconda3/envs/vtc_serve/bin/python -m src.serve_bench --model runs/models/llava-1.5-7b-hf --pruning-rate 0.25 --benchmark gqa --subset eval/subsets/gqa_200.jsonl --metrics-out runs/p2_probe/gqa_r25_metrics.json --max-tokens 32 --max-model-len 4096 --gpu-memory-utilization 0.90 --seed 0`
- **est_min**: 22 · **priority**: 80
- **log**: `runs/probe_gqa_r25.log`

## 结果 / 指标
<填: 关键 metrics — 精度/压缩比/延迟/显存；引用 eval/ 表>

## 结论
<填: 是否支撑 claim / 下一步>

## 产物路径
<填: 权重在 runs/... （gitignored）>

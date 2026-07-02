# p3s1_gqa_fixed_r50_bursty_mt32

- **status**: done (rc=0)
- **ran**: 2026-07-02 12:13 · 108s
- **cmd**: `/home/dell/miniconda3/envs/vtc_serve/bin/python -m src.serve_bench --model runs/models/llava-1.5-7b-hf --max-model-len 4096 --gpu-memory-utilization 0.85 --seed 0 --selector proxy --max-num-seqs 12 --benchmark gqa --subset eval/subsets/gqa_200.jsonl --metrics-out runs/p2_d/p3s1_gqa_fixed_r50_bursty_mt32.json --max-tokens 32 --pruning-rate 0.50 --load-profile bursty`
- **est_min**: 5 · **priority**: 95
- **log**: `runs/p3s1_gqa_fixed_r50_bursty_mt32.log`

## 结果 / 指标
<填: 关键 metrics — 精度/压缩比/延迟/显存；引用 eval/ 表>

## 结论
<填: 是否支撑 claim / 下一步>

## 产物路径
<填: 权重在 runs/... （gitignored）>

# gqa_r75 — GQA @ prune 0.75 (keep 144/576), n=200

- **e2e req/s**: 2.53  (1.43x vs r0)
- **prefill TTFT**: 487 ms  (1.30x vs r0)
- **served tok/s**: 26.42
- **accuracy (re-scored, fixed scorer)**: 0.470 (94/200)
- **log**: `runs/gqa_r75.log` · **metrics**: `runs/p2_probe/gqa_r75_metrics.json` (gitignored; raw answers saved)

Full curve + analysis: `eval/p2_probe_summary.md`. Probe n=200 is a seed subset (seed=0) for the go/no-go gate, not the final benchmark number.

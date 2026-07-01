# gqa_r25 — GQA @ prune 0.25 (keep 432/576), n=200

- **e2e req/s**: 2.08  (1.17x vs r0)
- **prefill TTFT**: 553 ms  (1.14x vs r0)
- **served tok/s**: 24.34
- **accuracy (re-scored, fixed scorer)**: 0.555 (111/200)
- **log**: `runs/gqa_r25.log` · **metrics**: `runs/p2_probe/gqa_r25_metrics.json` (gitignored; raw answers saved)

Full curve + analysis: `eval/p2_probe_summary.md`. Probe n=200 is a seed subset (seed=0) for the go/no-go gate, not the final benchmark number.

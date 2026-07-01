# textvqa_r0 — TEXTVQA @ prune 0.0 (keep 576/576), n=200

- **e2e req/s**: 1.59  (— vs r0)
- **prefill TTFT**: 690 ms  (— vs r0)
- **served tok/s**: 23.51
- **accuracy (re-scored, fixed scorer)**: 0.555 (111/200)
- **log**: `runs/textvqa_r0.log` · **metrics**: `runs/p2_probe/textvqa_r0_metrics.json` (gitignored; raw answers saved)

Full curve + analysis: `eval/p2_probe_summary.md`. Probe n=200 is a seed subset (seed=0) for the go/no-go gate, not the final benchmark number.

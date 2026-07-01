# textvqa_r75 — TEXTVQA @ prune 0.75 (keep 144/576), n=200

- **e2e req/s**: 1.85  (1.16x vs r0)
- **prefill TTFT**: 612 ms  (1.13x vs r0)
- **served tok/s**: 26.82
- **accuracy (re-scored, fixed scorer)**: 0.440 (88/200)
- **log**: `runs/textvqa_r75.log` · **metrics**: `runs/p2_probe/textvqa_r75_metrics.json` (gitignored; raw answers saved)

Full curve + analysis: `eval/p2_probe_summary.md`. Probe n=200 is a seed subset (seed=0) for the go/no-go gate, not the final benchmark number.

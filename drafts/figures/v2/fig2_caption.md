## Figure 2 — Concurrency × prune curve, c1–c64, with ceiling-lift (scale headline)

**Served req/s vs prune rate** for LLaVA-1.5-7B / GQA / vLLM V1 (n=200/cell,
closed-loop batch-submit). Four concurrency curves c ∈ {1, 4, 16, 64} (single-hue
blue ramp, light→dark). The prune speedup *grows monotonically and has not
saturated at c64*: the r75/r0 ratio rises **1.19× (c1) → 1.53× (c4) → 1.96× (c16)
→ 2.22× (c64)**.

**Ceiling-lift (the deployment-relevant secondary finding).** The uncompressed
baseline r0 *plateaus* from c16→c64 (8.23→9.18 req/s, **+12%** — r0 is
KV/compute-bound at the single-A40 ceiling), while r75 keeps climbing
(16.17→20.39, **+26%**). Compression **raises the achievable peak throughput of
the hardware**, not just the per-config speedup ratio — an effect invisible to
concurrency-independent FLOPs measurement.

*Source:* `runs/v2_p2/batch_c{1,4,16,64}_r{0,50,75}.json` (paper Table 1 /
`notes/v2_p2_scale.md` Table A). c64 = single-A40 r0 ceiling (peak KV ≈ 41 GB).
*Generator:* `gen_fig2.py`.

## Figure 3 — Goodput-Pareto at c64 (the deployment figure, the paper's lead)

**Main panel (log-log):** served req/s (x) vs p99-TTFT (y, lower is better) at
c=64 on LLaVA-1.5-7B / GQA / V1 (n=200, closed-loop). The three points are r0
(9.2 req/s, p99 18.4 s), r50 (14.6, 10.5 s), r75 (20.4, 6.5 s). **r75 strictly
dominates r0** — it sits at the upper-right of the Pareto (high throughput, low
tail): **2.22× the served req/s AND 2.84× lower p99-TTFT, with no tradeoff**.
This is the pure-win headline: in the saturated regime, pruning relieves the
KV/prefill bottleneck that bounds *both* throughput and tail latency, so it
improves both axes simultaneously.

**Inset:** goodput (req/s meeting the TTFT SLO) vs SLO threshold. At TTFT≤5 s,
r75 delivers **7.4× the goodput** of r0 (13.7 vs 1.8 req/s); tight SLOs (≤1 s)
are unmeetable for any r under c64 closed-loop (the 64-way chunked-prefill floor
is ~3 s).

*Source:* `runs/v2_p2/batch_c64_r{0,50,75}.json` (paper Table 3 /
`notes/v2_p2_scale.md` Tables B, C). Goodput computed from per-request TTFT
(`raw[*].ttft_ms`); *Generator:* `gen_fig3.py`.

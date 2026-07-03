**Figure 4.** Method Pareto frontier at the n=500 noise gate (honest). Served
req/s (x) vs. accuracy (y) for five benchmarks (color) and three configs
(marker: ■ fixed-r25, ● adaptive [ours], ▲ fixed-r50), c12, bursty load,
``num_running`` controller with r ∈ [0.25, 0.50]. Thin lines join each
benchmark's three configs. **Adaptive does not strictly dominate fixed-r50 on
any benchmark at n=500** (clean Pareto-dominate count = 0/5): on GQA, MME,
MMBench, and ScienceQA fixed-r50 has the highest req/s; the +0.014 accuracy
margins on MME/MMBench are within noise (|z| < 0.6). The surviving, robust
claim is a *free throughput gain over fixed-r25 at iso-accuracy-to-r25* on the
dense/MC benchmarks (MMBench +5.3%, ScienceQA +5.2%, TextVQA +3.7%) — a
throughput-optimal-under-guardrail result, not a Pareto win. The n=200
Pareto-dominate verdicts on MME and ScienceQA reversed at n=500 and are
reported openly in §5.2. *Source: Table C-n500; `notes/p3s3_pareto_n500.md`.*

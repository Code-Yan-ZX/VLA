# final_results_v3.md — SINGLE SOURCE OF TRUTH for all numbers in the v3 paper

> Compiled 2026-07-07 by transcribing ONLY values present on disk. Every row
> carries: value · n · #runs · loop mode (closed / open + rate / serial) ·
> source path · evidence-level tag. When the paper needs a number that is NOT on
> disk, it is flagged in §Z (Missing numbers) — do NOT fill from memory.
>
> **Evidence-level tags:** `GPU-measured` (read from a `runs/*.json` agg /
> `goodput_at_slo` field) · `sim` (printed by a `runs/elasticvis_ev0/*.py`
> zero-GPU slot/queue sim) · `fit` (regression / lookup in
> `latpred_coeffs.json` / `predictors_fit_report.md`) · `derived` (ratio or
> goodput@SLO sweep recomputed from a `raw` array; reproduces the curated notes
> tables). #runs = 1 for every row unless stated (no repeat sets exist).
>
> **Environment (all GPU rows):** 1× A40 46GB; conda env `qwen3vl_clean`
> (vLLM 0.19.0 V1 engine, in-process EngineCore via
> `VLLM_ENABLE_V1_MULTIPROCESSING=0`, torch 2.10.0+cu128, py3.10). Each cell =
> fresh process; 8 s GPU-settle between cells. Primary model LLaVA-1.5-7B-hf;
> secondary Qwen3-VL-8B-Instruct (bf16). Selector `proxy` = hidden-state-deviation
> saliency (the v1 selector) unless noted.
>
> **Loop-mode key:** `closed` = `--batch-submit` (all n reqs arrive near-simultaneously
> into max_num_seqs= c); `serial` = one `llm.chat` per request, no queue (c=1);
> `open λ` = open-loop Poisson arrivals at offered rate λ req/s.

---

## §A. V2-P0 — V1 Table A: served-throughput on LLaVA-1.5-7B (F1 on V1)

GQA, n=100, V1 engine. `batch_*` = closed-loop at `max_num_seqs=c`;
`serial_c1` = serial (no queue). Source: `runs/v2_p0/*.json` (`agg`).
Curated narrative: `notes/v2_p0_v1_tableA.md`.

### A.1 Batch (closed-loop) served throughput

| cell | c | r | req/s | tok/s | peak_kv_mb | acc | wall_s | loop | source | tag |
|---|---|---|---|---|---|---|---|---|---|---|
| batch_c1_r0.0  | 1  | 0.00 | 2.036 | 29.60  | 40665 | 0.600 | 49.1 | closed | runs/v2_p0/batch_c1_r0.0.json  | GPU-measured |
| batch_c1_r0.50 | 1  | 0.50 | 2.213 | 32.97  | 40671 | 0.520 | 45.2 | closed | runs/v2_p0/batch_c1_r0.50.json | GPU-measured |
| batch_c1_r0.75 | 1  | 0.75 | 2.469 | 34.05  | 40722 | 0.470 | 40.5 | closed | runs/v2_p0/batch_c1_r0.75.json | GPU-measured |
| batch_c4_r0.0  | 4  | 0.00 | 4.969 | 72.34  | 40451 | 0.600 | 20.1 | closed | runs/v2_p0/batch_c4_r0.0.json  | GPU-measured |
| batch_c4_r0.50 | 4  | 0.50 | 6.208 | 92.68  | 40398 | 0.520 | 16.1 | closed | runs/v2_p0/batch_c4_r0.50.json | GPU-measured |
| batch_c4_r0.75 | 4  | 0.75 | 7.491 | 103.31 | 40503 | 0.470 | 13.3 | closed | runs/v2_p0/batch_c4_r0.75.json | GPU-measured |
| batch_c12_r0.0 | 12 | 0.00 | 7.128 | 103.79 | 40850 | 0.600 | 14.0 | closed | runs/v2_p0/batch_c12_r0.0.json | GPU-measured |
| batch_c12_r0.50| 12 | 0.50 | 10.113| 150.99 | 40527 | 0.520 | 9.9  | closed | runs/v2_p0/batch_c12_r0.50.json| GPU-measured |
| batch_c12_r0.75| 12 | 0.75 | 13.268| 182.97 | 40405 | 0.470 | 7.5  | closed | runs/v2_p0/batch_c12_r0.75.json| GPU-measured |

P0 `agg.ttft_ms` is NaN for all 9 batch cells (P0 harness did not capture
per-request TTFT — `predictors_fit_report.md` §5 "v2_p0 unusable"). p50/p99 and
goodput@SLO for c≤12 are therefore **not available** (see §Z).

### A.2 Serial c1 — prefill breakdown (F2 vision-fixed-cost, LLaVA-1.5)

n=100, mt=32, serial (one `llm.chat` per request, no queue). `prefill_breakdown`
field. `vision_tower_fraction_of_prefill` = vision_tower_ms / ttft_ms_mean.

| r | req/s | acc | ttft_ms_mean | vision_tower_ms | projector_ms | llm_prefill_ms_est | vision_frac | source | tag |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 2.360 | 0.600 | 480.34 | 12.80 | 0.166 | 467.37 | 0.0267 | runs/v2_p0/serial_c1_r0.0.json  | GPU-measured |
| 0.50 | 2.723 | 0.520 | 445.70 | 12.99 | 0.235 | 432.48 | 0.0291 | runs/v2_p0/serial_c1_r0.50.json | GPU-measured |
| 0.75 | 3.024 | 0.470 | 395.85 | 12.71 | 0.228 | 382.92 | 0.0321 | runs/v2_p0/serial_c1_r0.75.json | GPU-measured |

### A.3 P0 derived speedup ratios (paper Table A headlines)

| ratio | value | tag |
|---|---|---|
| r50/r0 @ c1 (req/s) | 2.213/2.036 = 1.09× | derived |
| r75/r0 @ c1 (req/s) | 2.469/2.036 = 1.21× | derived |
| r50/r0 @ c4 (req/s) | 6.208/4.969 = 1.25× | derived |
| r75/r0 @ c4 (req/s) | 7.491/4.969 = 1.51× | derived |
| r50/r0 @ c12 (req/s)| 10.113/7.128 = 1.42× | derived |
| r75/r0 @ c12 (req/s)| 13.268/7.128 = **1.86×** | derived |
| serial r75 e2e× (480.34/395.85) | 1.21× | derived |
| serial r50 e2e× (480.34/445.70) | 1.08× | derived |

---

## §B. V2-P1 — Qwen3-VL-8B generalization (F1/F2/F3)

V1 engine, flat-schema JSON (no `agg` dict). Curated narrative:
`notes/v2_p1_qwen3vl.md`. `mt`=max_tokens.

### B.1 F1 served-throughput matrix (Qwen3-VL-8B, GQA, n=100, mt=32)

| cell | c | r | req/s | acc | mean_ptid | wall_s | loop | source | tag |
|---|---|---|---|---|---|---|---|---|---|
| f1_gqa_r00_c1  | 1  | 0.00 | 0.971 | 0.49 | 279.0 | 103.0 | closed | runs/v2_p1/f1_gqa_r00_c1.json  | GPU-measured |
| f1_gqa_r50_c1  | 1  | 0.50 | 1.030 | 0.47 | 149.1 | 97.1  | closed | runs/v2_p1/f1_gqa_r50_c1.json  | GPU-measured |
| f1_gqa_r75_c1  | 1  | 0.75 | 1.045 | 0.43 | 84.2  | 95.7  | closed | runs/v2_p1/f1_gqa_r75_c1.json  | GPU-measured |
| f1_gqa_r00_c4  | 4  | 0.00 | 3.211 | 0.44 | 279.0 | 31.1  | closed | runs/v2_p1/f1_gqa_r00_c4.json  | GPU-measured |
| f1_gqa_r50_c4  | 4  | 0.50 | 3.417 | 0.46 | 149.1 | 29.3  | closed | runs/v2_p1/f1_gqa_r50_c4.json  | GPU-measured |
| f1_gqa_r75_c4  | 4  | 0.75 | 3.541 | 0.44 | 84.2  | 28.2  | closed | runs/v2_p1/f1_gqa_r75_c4.json  | GPU-measured |
| f1_gqa_r00_c12 | 12 | 0.00 | 6.212 | 0.44 | 279.0 | 16.1  | closed | runs/v2_p1/f1_gqa_r00_c12.json | GPU-measured |
| f1_gqa_r50_c12 | 12 | 0.50 | 7.204 | 0.45 | 149.1 | 13.9  | closed | runs/v2_p1/f1_gqa_r50_c12.json | GPU-measured |
| f1_gqa_r75_c12 | 12 | 0.75 | 8.011 | 0.42 | 84.2  | 12.5  | closed | runs/v2_p1/f1_gqa_r75_c12.json | GPU-measured |

F1 derived ratios: r50/r0 @ c12 = 7.204/6.212 = 1.16×; r75/r0 @ c12 = 8.011/6.212
= **1.29×**; r75 concurrency bonus c1→c12 = 1.08→1.29 = +0.21 (derived).

### B.2 F2 prefill breakdown (Qwen3-VL native merger, GQA c1, n=20, mt=4)

| cell | r | req/s | acc | vision_ms_total | wall_s | mean_ptid | source | tag |
|---|---|---|---|---|---|---|---|---|
| f2_gqa_r0_c1_prefill  | 0.00 | 5.289 | 0.15 | 364.75 | 3.781 | 294.4 | runs/v2_p1/f2_gqa_r0_c1_prefill.json  | GPU-measured |
| f2_gqa_r50_c1_prefill | 0.50 | 6.047 | 0.25 | 357.63 | 3.307 | 156.4 | runs/v2_p1/f2_gqa_r50_c1_prefill.json | GPU-measured |

F2 headline (from notes, n=20 wall≈TTFT): vision tower+merger+deepstack = 0.36 s
= **10% of 3.78 s wall** (derived from vision_ms_total/wall_s × 20 reqs;
vision_ms_total is the per-run hook sum → per-req ≈ 18 ms).

### B.3 F3 workload-dependence (Qwen3-VL, c4, n=100, mt=32)

| workload | cell | r | req/s | acc | mean_ptid | source | tag |
|---|---|---|---|---|---|---|---|
| GQA     | f1_gqa_r00_c4 | 0.00 | 3.211 | 0.44 | 279 | runs/v2_p1/f1_gqa_r00_c4.json | GPU-measured |
| GQA     | f1_gqa_r50_c4 | 0.50 | 3.417 | 0.46 | 149 | runs/v2_p1/f1_gqa_r50_c4.json | GPU-measured |
| TextVQA | f3_textvqa_r00_c4 | 0.00 | 2.161 | 0.77 | 747.7 | runs/v2_p1/f3_textvqa_r00_c4.json | GPU-measured |
| TextVQA | f3_textvqa_r50_c4 | 0.50 | 2.505 | 0.48 | 382.9 | runs/v2_p1/f3_textvqa_r50_c4.json | GPU-measured |

F3 derived: GQA r50/r0 = 3.417/3.211 = 1.06×; TextVQA r50/r0 = 2.505/2.161 =
**1.16×** (more visual tokens → bigger prune speedup).

---

## §C. V2-P2 — Real serving scale (c≥64, tail latency, goodput)

LLaVA-1.5-7B, GQA, n=200, mt=16, proxy selector, V1. Curated narrative:
`notes/v2_p2_scale.md`. `batch_*` = closed-loop; `serial_*` = serial c1.

### C.1 Scale served-throughput + tail-latency matrix (closed-loop)

| cell | c | r | req/s | tok/s | ttft_p50_ms | ttft_p99_ms | e2e_p50_ms | e2e_p99_ms | peak_kv_mb | acc | wall_s | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| batch_c1_r0  | 1  | 0.00 | 2.299  | 29.39  | 42836 | 82985 | 43028 | 83227 | 40665 | 0.595 | 87.0 | runs/v2_p2/batch_c1_r0.json  |
| batch_c1_r50 | 1  | 0.50 | 2.600  | 32.61  | 38225 | 73200 | 38415 | 73415 | 40671 | 0.535 | 76.9 | runs/v2_p2/batch_c1_r50.json |
| batch_c1_r75 | 1  | 0.75 | 2.744  | 34.28  | 35574 | 69199 | 35761 | 69411 | 40722 | 0.475 | 72.9 | runs/v2_p2/batch_c1_r75.json |
| batch_c4_r0  | 4  | 0.00 | 5.453  | 69.80  | 17542 | 33276 | 17881 | 33808 | 40451 | 0.600 | 36.7 | runs/v2_p2/batch_c4_r0.json  |
| batch_c4_r50 | 4  | 0.50 | 7.036  | 88.34  | 13428 | 24712 | 13796 | 25126 | 40411 | 0.530 | 28.4 | runs/v2_p2/batch_c4_r50.json |
| batch_c4_r75 | 4  | 0.75 | 8.360  | 104.96 | 11132 | 20697 | 11562 | 21067 | 40494 | 0.480 | 23.9 | runs/v2_p2/batch_c4_r75.json |
| batch_c16_r0 | 16 | 0.00 | 8.231  | 105.23 | 11530 | 20715 | 12490 | 21312 | 40932 | 0.595 | 24.3 | runs/v2_p2/batch_c16_r0.json |
| batch_c16_r50| 16 | 0.50 | 12.449 | 156.61 | 7403  | 12962 | 8096  | 13377 | 40627 | 0.530 | 16.1 | runs/v2_p2/batch_c16_r50.json|
| batch_c16_r75| 16 | 0.75 | 16.174 | 202.74 | 5502  | 8869  | 6066  | 9287  | 40446 | 0.475 | 12.4 | runs/v2_p2/batch_c16_r75.json|
| batch_c64_r0 | 64 | 0.00 | 9.179  | 117.07 | 10721 | 18369 | 14711 | 19224 | 40987 | 0.590 | 21.8 | runs/v2_p2/batch_c64_r0.json |
| batch_c64_r50| 64 | 0.50 | 14.586 | 183.20 | 6253  | 10526 | 9060  | 11225 | 40992 | 0.535 | 13.7 | runs/v2_p2/batch_c64_r50.json|
| batch_c64_r75| 64 | 0.75 | 20.389 | 255.67 | 4229  | 6471  | 5908  | 7151  | 41049 | 0.475 | 9.8  | runs/v2_p2/batch_c64_r75.json|

All rows: tag `GPU-measured`, loop `closed`, n=200.

### C.2 Serial c1 — queue-free per-request latency (F1 continuity)

n=200, mt=16, serial. tag `GPU-measured`.

| cell | r | req/s | ttft_p50_ms | ttft_p99_ms | e2e_p50_ms | e2e_p99_ms | peak_kv_mb | acc | wall_s | source |
|---|---|---|---|---|---|---|---|---|---|---|
| serial_c1_r0  | 0.00 | 2.514 | 433 | 513 | 433 | 513 | 40664 | 0.595 | 86.0 | runs/v2_p2/serial_c1_r0.json  |
| serial_c1_r50 | 0.50 | 3.007 | 387 | 469 | 387 | 469 | 40668 | 0.535 | 76.0 | runs/v2_p2/serial_c1_r50.json |
| serial_c1_r75 | 0.75 | 3.119 | 367 | 458 | 367 | 458 | 40712 | 0.475 | 71.8 | runs/v2_p2/serial_c1_r75.json |

Derived serial speedups: r50/r0 = 3.007/2.514 = 1.20×; r75/r0 = 3.119/2.514 =
**1.24×**; r75 e2e p50 433→367 ms = 1.18×.

### C.3 c64 goodput@SLO sweep (Pareto) — derived from per-request raw

Recomputed from the `raw` arrays of the 3 c64 cells (tag `derived`; reproduces
P2-notes Table C). goodput = req/s × frac meeting SLO.

| SLO | r0 req/s met | r50 req/s met | r75 req/s met | r75/r0 | source raw |
|---|---|---|---|---|---|
| TTFT ≤ 500 ms  | 0.00 | 0.00 | 0.00  | — (floor) | batch_c64_r{0,50,75}.json |
| TTFT ≤ 1 s     | 0.00 | 0.00 | 0.00  | — | " |
| TTFT ≤ 3 s     | 0.23 | 1.39 | 1.22  | 5.3× | " |
| TTFT ≤ 5 s     | 1.84 | 5.18 | **13.66** | **7.4×** | " |
| TTFT ≤ 10 s    | 4.45 | 13.56 | 20.39 | 4.6× | " |
| e2e ≤ 1 s      | 0.00 | 0.00 | 0.00  | — | " |
| e2e ≤ 4 s      | 0.00 | 0.00 | 1.43  | — | " |
| e2e ≤ 6 s      | 0.00 | 2.48 | 10.40 | — | " |
| e2e ≤ 8 s      | 0.87 | 5.69 | **20.39** | **23.4×** | " |
| e2e ≤ 10 s     | 1.93 | 8.90 | 20.39 | 10.6× | " |

c64 throughput-vs-tail Pareto (derived): r0 9.18 req/s & p99-TTFT 18369 ms;
r50 14.59 & 10526; r75 **20.39 & 6471**. r75 strictly dominates r0:
2.22× throughput (20.39/9.18) AND 2.84× lower p99-TTFT (18369/6471) and 2.69×
lower p99-e2e (19224/7151).

### C.4 c64 feasibility probe (single-wave n=64)

| cell | n | req/s | tok/s | ttft_p50 | ttft_p99 | e2e_p50 | e2e_p99 | peak_kv | acc | wall | source | tag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| probe_c64_r0 | 64 | 8.096 | 104.24 | 3728 | 5482 | 6386 | 6937 | 40987 | 0.594 | 7.9 | runs/v2_p2/probe_c64_r0.json | GPU-measured |
| probe_c4_r0  | 8  | 3.326 | 44.48  | 782  | 1267 | 1030 | 1541 | 40451 | 0.625 | 2.4 | runs/v2_p2/probe_c4_r0.json  | GPU-measured |

### C.5 Qwen3-VL-8B c64 generalization point (flat schema)

GQA, n=200, c64, V1. tag `GPU-measured`. (acc depressed by mt=16 truncation on
the verbose model — a scoring artifact, not a pruning effect; see notes §7.)

| cell | r | req/s | acc | mean_ptid | wall_s | source |
|---|---|---|---|---|---|---|
| qwen3vl_c64_r0  | 0.00 | 12.477 | 0.250 | 284.2 | 16.03 | runs/v2_p2/qwen3vl_c64_r0.json  |
| qwen3vl_c64_r50 | 0.50 | 16.777 | 0.285 | 151.7 | 11.92 | runs/v2_p2/qwen3vl_c64_r50.json |

Derived r50/r0 @ c64 (Qwen3-VL) = 16.777/12.477 = **1.34×** (vs c12 1.16× —
keeps growing past c12).

### C.6 Single-A40 serving ceiling (r0 peak KV, n=200)

| c | peak_kv_mb | wall_s | source |
|---|---|---|---|
| 1  | 40665 | 87.0 | batch_c1_r0.json  |
| 4  | 40451 | 36.7 | batch_c4_r0.json  |
| 16 | 40932 | 24.3 | batch_c16_r0.json |
| 64 | 40987 | 21.8 | batch_c64_r0.json |

tag `GPU-measured`. c128 infeasible at r0 (not measured; notes §6 projection:
128×620≈79k tokens > ~53k KV budget).

---

## §D. V2-P3 — Cross-compressor served-throughput panel (c64)

LLaVA-1.5-7B, GQA, n=200, mt=16, c64, closed-loop, V1. 4 compressors × 3 rates.
Curated narrative: `notes/v2_p3_crosscompressor.md`. tag `GPU-measured`.

### D.1 Cross-compressor served-throughput + accuracy

| compressor | r | req/s | tok/s | ttft_p99_ms | e2e_p99_ms | peak_kv_mb | acc | wall_s | source |
|---|---|---|---|---|---|---|---|---|---|
| proxy (saliency prune)     | 0.00 | 9.202  | 117.37 | 18337 | 19186 | 40987 | 0.590 | 21.7 | runs/v2_p3/proxy_c64_r0.json     |
| proxy                      | 0.50 | 14.471 | 181.76 | 10557 | 11272 | 40992 | 0.535 | 13.8 | runs/v2_p3/proxy_c64_r50.json    |
| proxy                      | 0.75 | 20.020 | 251.05 | 6575  | 7410  | 41049 | 0.475 | 10.0 | runs/v2_p3/proxy_c64_r75.json    |
| true_cls (CLS-attn prune)  | 0.00 | 8.885  | 113.33 | 18801 | 19654 | 40987 | 0.590 | 22.5 | runs/v2_p3/true_cls_c64_r0.json  |
| true_cls                   | 0.50 | 14.150 | 177.02 | 10803 | 11620 | 40992 | 0.525 | 14.1 | runs/v2_p3/true_cls_c64_r50.json |
| true_cls                   | 0.75 | 20.441 | 253.47 | 6608  | 7214  | 41049 | 0.490 | 9.8  | runs/v2_p3/true_cls_c64_r75.json |
| tome_merge (ToMe, merge)   | 0.00 | 8.998  | 114.76 | 18721 | 19596 | 40987 | 0.590 | 22.2 | runs/v2_p3/tome_merge_c64_r0.json  |
| tome_merge                 | 0.50 | 14.161 | 177.15 | 10934 | 11616 | 40992 | 0.550 | 14.1 | runs/v2_p3/tome_merge_c64_r50.json |
| tome_merge                 | 0.75 | 19.729 | 246.41 | 7032  | 7641  | 41173 | 0.540 | 10.1 | runs/v2_p3/tome_merge_c64_r75.json |
| random (uniform prune)     | 0.00 | 9.124  | 116.38 | 18510 | 19357 | 40987 | 0.590 | 21.9 | runs/v2_p3/random_c64_r0.json  |
| random                     | 0.50 | 14.469 | 181.08 | 10580 | 11256 | 40992 | 0.550 | 13.8 | runs/v2_p3/random_c64_r50.json |
| random                     | 0.75 | 20.734 | 258.14 | 6514  | 7125  | 41049 | 0.535 | 9.6  | runs/v2_p3/random_c64_r75.json |

### D.2 r75 vs r0 per compressor (goodput-Pareto generalization) — derived

| compressor | r0 req/s | r75 req/s | r75/r0 | r0 p99-TTFT | r75 p99-TTFT | p99 reduction | source |
|---|---|---|---|---|---|---|---|
| proxy      | 9.202 | 20.020 | 2.18× | 18337 | 6575 | 2.79× | proxy_c64_r{0,75}.json |
| true_cls   | 8.885 | 20.441 | 2.30× | 18801 | 6608 | 2.85× | true_cls_c64_r{0,75}.json |
| tome_merge | 8.998 | 19.729 | 2.19× | 18721 | 7032 | 2.66× | tome_merge_c64_r{0,75}.json |
| random     | 9.124 | 20.734 | 2.27× | 18510 | 6514 | 2.84× | random_c64_r{0,75}.json |

### D.3 c64 goodput@TTFT≤5s per compressor — derived from raw

| compressor | r0 gp@5s | r50 gp@5s | r75 gp@5s | r75/r0 | source raw |
|---|---|---|---|---|---|
| proxy      | 1.84 | 5.14 | 13.01 | 7.1×  | proxy_c64_r*.json |
| true_cls   | 1.38 | 4.95 | 13.70 | 9.9×  | true_cls_c64_r*.json |
| tome_merge | 1.48 | 5.03 | 10.75 | 7.2×  | tome_merge_c64_r*.json |
| random     | 1.82 | 5.21 | 14.41 | 7.9×  | random_c64_r*.json |

### D.4 Prune-vs-merge accuracy at iso-r (derived from D.1)

| metric @ r=0.50 | proxy | true_cls | tome_merge | random |
|---|---|---|---|---|
| req/s            | 14.471 | 14.150 | 14.161 | 14.469 |
| acc              | 0.535  | 0.525  | **0.550** | **0.550** |
| metric @ r=0.75 | | | | |
| req/s            | 20.020 | 20.441 | 19.729 | 20.734 |
| acc              | 0.475  | 0.490  | **0.540** | 0.535 |

ToMe-vs-proxy accuracy delta: +0.015 @ r50, **+0.065 @ r75**; throughput cost
−2.1% @ r50, −1.5% @ r75 (derived). Random beats saliency at r75: random 0.535
vs proxy 0.475 (Δ +0.060) vs true_cls 0.490 (Δ +0.045).

---

## §E. EV-0 — ElasticVis sim stage (zero-GPU)

Slot+queue simulator grounded by the GPU-measured latency predictor
(`runs/elasticvis_ev0/latpred_coeffs.json`) and accuracy(k) curves
(`runs/elasticvis_ev0/accuracy.json`). Curated narrative:
`notes/elasticvis_design.md` §8.

### E.1 Accuracy(k) curves (sim inputs)

**LLaVA-1.5 GQA aggregate** (from `accuracy.json::agg_curve`; derived from v2_p2
per-image majority vote, 200 images × 3 k): tag `GPU-measured` (source).

| k | acc |
|---|---|
| 144 | 0.476 |
| 288 | 0.533 |
| 576 | 0.595 |

acc-range = 0.595 − 0.476 = 0.119. per-image flips r0→r75 (majority vote):
**46/200 = 23.0%** (`accuracy.json::note` + `predictors_fit_report.md` §3).
native_n = 576; k_points = [144, 288, 576]; n per_image = 200.

### E.2 Latency predictor fit (`predictors_fit_report.md`, `latpred_coeffs.json`)

Form: `latency_ms = α + β·num_running + γ·own_k + δ·sum_k + ε·(num_running·sum_k) + ζ/num_running`.
Param fit on 9 batch cells (closed-loop sojourn), n=1800 per-request points.

| term | α | β | γ | δ | ε | ζ |
|---|---|---|---|---|---|---|
| ttft_ms | 1550 | 5.771 | 15.19 | −0.1111 | 0.001381 | 3.055e+04 |
| e2e_ms  | 1811 | 9.013 | 15.15 | −0.001554 | 0.001259 | 3.053e+04 |

Cell-level leave-one-cell-out CV (PRIMARY, 9 batch cells): **ttft R²=0.9962
MAPE=1.93%**; **e2e R²=0.9955 MAPE=1.94%**. tag `fit`. Per-request raw R²≈0.388
(low — within-cell queue-position variance dominates; not allocator-relevant).
p50→p99 multipliers (median across fit cells): ttft 1.7508, e2e 1.6523.

### E.3 Measured (n,k)→p50/p99 lookup table (`predictors_fit_report.md` §4)

tag `GPU-measured` (source cells in `runs/v2_p2/`).

| n | k | ttft_p50 | ttft_p99 | e2e_p50 | e2e_p99 |
|---|---|---|---|---|---|
| 1  | 144 | 367   | 458   | 367   | 458   |
| 1  | 288 | 387   | 469   | 387   | 469   |
| 1  | 576 | 433   | 513   | 433   | 513   |
| 4  | 144 | 11132 | 20697 | 11562 | 21067 |
| 4  | 288 | 13428 | 24712 | 13796 | 25126 |
| 4  | 576 | 17542 | 33276 | 17881 | 33808 |
| 16 | 144 | 5502  | 8869  | 6066  | 9287  |
| 16 | 288 | 7403  | 12962 | 8096  | 13377 |
| 16 | 576 | 11530 | 20715 | 12490 | 21312 |
| 64 | 144 | 4229  | 6471  | 5908  | 7151  |
| 64 | 288 | 6253  | 10526 | 9060  | 11225 |
| 64 | 576 | 10721 | 18369 | 14711 | 19224 |

### E.4 Gating characterization — accuracy(k) steepness gates the win

**5-benchmark gating table** (`notes/elasticvis_design.md` §8; acc-ranges for
the knowledge/ObjectQA/TextVQA groups are curated literature values, NOT measured
in the v2 runs — flag in §Z):

| benchmark group | acc(k) range | ElasticVis H1b outcome |
|---|---|---|
| knowledge (MME / MMBench / ScienceQA) | ~0.01 | flat → no win |
| object QA (GQA, LLaVA-1.5 & Qwen3-VL) | 0.12–0.13 | boundary (<0.15) → NO-GO |
| text-dense (TextVQA, LLaVA-1.5 & Qwen3-VL) | 0.28–0.29 | steep (>0.15) → WIN |

(GQA 0.119 and TextVQA-LLaVA 0.28 ARE measured — §E.1 and §F.1.)

**Synthetic sweep** (`runs/elasticvis_ev0/gating_sweep.py`, linear acc a@144
swept, a@576=0.60; slot+queue sim; k_range=(144,576), max_num_seqs=64, seed=42;
H1 = OpenLoopPoisson(λ=12), e2e SLO 10 s; H1b = MixedSLO over OpenLoopPoisson(λ=
8), 50% tight 3.5 s / 50% slack 15 s, e2e SLO 10 s). tag `sim`, #runs=1.

| claim | value | source |
|---|---|---|
| H1b (mixed-SLO) crossover acc-range | ≈ 0.15 | gating_sweep.py (printed) + design §8 |
| H1 (uniform-SLO) crossover acc-range | ≈ 0.40 | gating_sweep.py (printed) + design §8 |

### E.5 Decisive real-TextVQA sim result (the +35.5% headline)

`runs/elasticvis_ev0/confirm_textvqa.py`. acc curves (inputs): TextVQA-LLaVA
{144:0.275, 288:0.445, 576:0.555} (range 0.28); GQA-LLaVA {144:0.476,
288:0.533, 576:0.595} (range 0.119). Same sim config as E.4. tag `sim`, #runs=1.

| setting | Greedy goodput_rate | best Fixed goodput_rate | ratio | outcome | source |
|---|---|---|---|---|---|
| TextVQA H1b mixed-SLO (poisson8, 50/50 3.5s/15s, e2e@10s) | **2.36** | 1.74 | **1.355 → +35.5%** | WIN | confirm_textvqa.py + design §8 |
| TextVQA H1 uniform-SLO (poisson12, e2e@10s) | — | — | **0.898** | lose | confirm_textvqa.py + design §8 |
| GQA H1b mixed-SLO | — | — | **0.978** | lose (gate boundary) | confirm_textvqa.py + design §8 |

Mechanism (design §8): Greedy gives low-k to deadline-tight (protect SLO) and
high-k to slack (acc 0.555 vs 0.275). Caveat (design §8): sim has fidelity gaps
(slot over-serialization; closed-loop sanity k144 ~2.5× low, biased in
ElasticVis's favor); +35.5% is a *relative* comparison, magnitude needed GPU
confirmation → EV-1 (§G/H). **The GPU confirmation REFUTED the win (§H).**

### E.6 GPU-calibrated TextVQA prediction (`runs/elasticvis_ev0/calibrated_textvqa.py`)

Uses GPU-measured eager-c64 S(k) (service_time {144:4.36, 288:5.75, 576:8.40} s,
prefill_time {144:2.0, 288:2.6, 576:3.6} s, derived from ev1c_fixed wall-times
128 reqs / served_req_s) and GPU-measured acc(k) {144:0.273, 288:0.438,
576:0.555}. tag `sim`. Closed-loop sanity (reproduce GPU): the script prints
expected r0≈7.6, r50≈11.1, r75≈14.7 req/s (target, from `ev1c_fixed_r*`).
**Specific H1b ratio numbers for the tight-deadline sweep (4000/5000/6000/7000
ms) are NOT stored on disk** — they print at runtime only; flagged §Z.

### E.7 TextVQA H1b robustness grid (`runs/elasticvis_ev0/textvqa_robustness.py`)

Sweep: rate ∈ {4,8,12,16} × p_tight ∈ {0.3,0.5,0.7}, TextVQA acc range 0.28,
H1b mixed-SLO, e2e@10 s. tag `sim`. **The 4×3 ratio grid is computed at runtime
and NOT stored on disk** (only the script + the note "base config rate=8,p=0.5
was +35.5%"); flagged §Z. Reproduce: `python runs/elasticvis_ev0/textvqa_robustness.py`.

---

## §F. EV-1c — per-request k integration on GPU (closed-loop)

LLaVA-1.5-7B, TextVQA, c64, V1, `k_policy` ∈ {fixed, elasticvis}. `ev_mixed_slo`
= "5000,15000" (50/50 tight/slack e2e deadlines) unless noted. metric:
`goodput_at_slo.goodput_acc` (acc-weighted met req/s) and `met_rate` (met req/s).
tag `GPU-measured`. Curated setting: STATE.md (EV-1c integration proven).

### F.1 EV-1c TextVQA accuracy(k) — fixed-r closed-loop (GPU-measured)

These are the GPU-grounded TextVQA accuracy(k) points used by §E.6.

| cell | k | n | slo_ms | ev_mixed_slo | acc | req/s | goodput_acc | met_rate | frac_met | wall_s | source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ev1c_fixed_r0.0  | 576 (r=0)   | 128 | 6000  | —            | 0.5547 | 7.615  | 0.000 | 0.000 | 0.000 | 16.81 | runs/ev1c_fixed_r0.0.json  |
| ev1c_fixed_r0.5  | 288 (r=0.5) | 128 | 6000  | —            | 0.4375 | 11.157 | 0.959 | 1.743 | 0.156 | 11.47 | runs/ev1c_fixed_r0.5.json  |
| ev1c_fixed_r75   | 144 (r=0.75)| 128 | 10000 | 3500,15000   | 0.2734 | 14.685 | 2.294 | 7.342 | 0.500 | 8.72  | runs/ev1c_fixed_r75.json   |

### F.2 EV-1c full n=200 fixed-r sweep (closed-loop, ev_mixed_slo 5000,15000, slo 10s)

| cell | r | n | acc | req/s | goodput_acc | met_rate | frac_met | wall_s | source |
|---|---|---|---|---|---|---|---|---|---|
| ev1c_full_fixed_r0.0  | 0.00 | 200 | 0.570 | 7.881  | 0.867 | 1.419 | 0.180 | 25.38 | runs/ev1c_full_fixed_r0.0.json  |
| ev1c_full_fixed_r0.25 | 0.25 | 200 | 0.535 | 9.626  | 1.540 | 3.128 | 0.325 | 20.78 | runs/ev1c_full_fixed_r0.25.json |
| ev1c_full_fixed_r0.5  | 0.50 | 200 | 0.450 | 11.917 | 2.562 | 5.959 | 0.500 | 16.78 | runs/ev1c_full_fixed_r0.5.json  |
| ev1c_full_fixed_r0.75 | 0.75 | 200 | 0.290 | 15.836 | 2.613 | 8.076 | 0.510 | 12.63 | runs/ev1c_full_fixed_r0.75.json |

### F.3 EV-1c n=80 fixed-r + ElasticVis (closed-loop) — the integration proof

| cell | k_policy | r | n | acc | req/s | goodput_acc | met_rate | frac_met | wall_s | source |
|---|---|---|---|---|---|---|---|---|---|---|
| ev1c_n80_fixed_r0.0  | fixed      | 0.00 | 80 | 0.5625 | 7.429  | 2.136 | 3.715  | 0.500 | 10.77 | runs/ev1c_n80_fixed_r0.0.json  |
| ev1c_n80_fixed_r0.25 | fixed      | 0.25 | 80 | 0.5125 | 8.390  | 2.307 | 4.195  | 0.500 | 9.54  | runs/ev1c_n80_fixed_r0.25.json |
| ev1c_n80_fixed_r0.5  | fixed      | 0.50 | 80 | 0.4625 | 10.100 | 3.409 | 7.070  | 0.700 | 7.92  | runs/ev1c_n80_fixed_r0.5.json  |
| ev1c_n80_fixed_r0.75 | fixed      | 0.75 | 80 | 0.3125 | 13.305 | 4.158 | 13.305 | 1.000 | 6.01  | runs/ev1c_n80_fixed_r0.75.json |
| ev1c_n80_ev          | elasticvis | 0.50 | 80 | 0.4125 | 9.262  | 2.663 | 4.747  | 0.5125| 8.64  | runs/ev1c_n80_ev.json          |
| ev1c_ev_c64          | elasticvis | 0.50 | 80 | 0.4125 | 9.705  | 2.912 | 5.095  | 0.525 | 8.24  | runs/ev1c_ev_c64.json          |

ElasticVis allocator realized summary (ev1c_n80_ev / ev1c_ev_c64, identical):
`k_mean=360.0`, k_dist {144: 40, 576: 40} (50/50 tight→k144, slack→k576),
`n_rid_hits=78/80`, `n_embed_calls=8`. `ev_per_batch_k_head` shows alternating
144/576 within each forward — **mechanism evidence that per-request k coexists
with c64 continuous batching** (robust placeholder-shrink integration).

**EV-1c verdict (closed-loop):** best fixed = r75 (goodput_acc 4.158 @ n80 /
2.613 @ n200); ElasticVis (2.663 / 2.912) LOSES to best fixed in closed-loop.
Closed-loop admission load ≈ constant → not the regime where per-request budget
can win (STATE.md).

---

## §G. EV-1d — open-loop GPU + the unfair +7% artifact

LLaVA-1.5-7B, TextVQA, c64, V1, open-loop Poisson, ev_mixed_slo 5000,15000,
e2e@10 s. `ev1d_ol_*` and `ev1d_openloop_*` are duplicate open-loop runs at the
default rate (n/wall ≈ 7.3 → offered λ≈8); `ev1d_r15_*` at offered λ=15.
tag `GPU-measured`.

### G.1 The +7% artifact — closed-loop n=200 (UNFAIR, per STATE.md)

`enforce_eager` asymmetry between EV and fixed arms (EV-1d un-calibrated).
**Superseded by EV-1e (§H).**

| cell | k_policy | n | acc | req/s | goodput_acc | met_rate | wall_s | source |
|---|---|---|---|---|---|---|---|---|
| ev1d_batch200_ev | elasticvis | 200 | 0.410 | 10.547 | 2.742 | 5.168 | 18.96 | runs/ev1d_batch200_ev.json |
| (matched fixed r0.5, §F.2) | fixed | 200 | 0.450 | 11.917 | 2.562 | 5.959 | 16.78 | runs/ev1c_full_fixed_r0.5.json |

Derived artifact ratio: 2.742/2.562 = **1.070 (+7.0%)** — the "+7.5%" of STATE.md
(rounded). Tag `derived` for the ratio; the underlying goodput_acc are
`GPU-measured` but the comparison is UNFAIR (flag for paper: do not cite as a
win — it is the documented artifact).

### G.2 EV-1d open-loop at default rate (λ≈8) — EV vs fixed

| cell | k_policy | r | n | acc | req/s | goodput_acc | met_rate | frac_met | wall_s | source |
|---|---|---|---|---|---|---|---|---|---|---|
| ev1d_openloop_r0 | fixed | 0.00 | 200 | 0.575 | 7.294 | 4.194 | 7.294 | 1.000 | 27.42 | runs/ev1d_openloop_r0.json |
| ev1d_openloop_r50| fixed | 0.50 | 200 | 0.450 | 7.490 | 3.371 | 7.490 | 1.000 | 26.70 | runs/ev1d_openloop_r50.json |
| ev1d_openloop_r75| fixed | 0.75 | 200 | 0.290 | 7.556 | 2.191 | 7.556 | 1.000 | 26.47 | runs/ev1d_openloop_r75.json |
| ev1d_ol_r0       | fixed | 0.00 | 200 | 0.575 | 7.166 | 4.049 | 7.023 | 0.980 | 27.91 | runs/ev1d_ol_r0.json |
| ev1d_ol_r50      | fixed | 0.50 | 200 | 0.450 | 7.491 | 3.371 | 7.491 | 1.000 | 26.70 | runs/ev1d_ol_r50.json |
| ev1d_ol_r75      | fixed | 0.75 | 200 | 0.290 | 7.554 | 2.191 | 7.554 | 1.000 | 26.48 | runs/ev1d_ol_r75.json |
| ev1d_ol_ev       | elasticvis | 0.50 | 200 | 0.410 | 7.394 | 3.032 | 7.394 | 1.000 | 27.05 | runs/ev1d_ol_ev.json |

### G.3 EV-1d open-loop at offered λ=15 (partial: EV + r0 only)

| cell | k_policy | r | n | acc | req/s | goodput_acc | met_rate | frac_met | wall_s | source |
|---|---|---|---|---|---|---|---|---|---|---|
| ev1d_r15_r0 | fixed | 0.00 | 200 | 0.575 | 8.790 | 2.549 | 4.571 | 0.520 | 22.75 | runs/ev1d_r15_r0.json |
| ev1d_r15_ev | elasticvis | 0.50 | 200 | 0.410 | 10.145 | 2.739 | 5.377 | 0.530 | 19.71 | runs/ev1d_r15_ev.json |

(EV-1d lacks fixed r50/r75 at λ=15 → cannot compute a fair best-fixed ratio;
this is why EV-1e re-ran the full grid, §H.)

### G.4 EV-1d debug / uniform-k runs (mechanism diagnostics, not paper headlines)

| cell | n | acc | req/s | goodput_acc | wall_s | note | source |
|---|---|---|---|---|---|---|---|
| ev1d_crash_debug | 30 | 0.333 | 5.095 | 1.698 | 5.88 | n=30 EV smoke (c64 order-independent matching proof) | runs/ev1d_crash_debug.json |
| ev1d_uniform144  | 128 | 0.273 | 14.800 | 4.047 | 8.65 | ev_debug_k=144 (force all-k144) | runs/ev1d_uniform144.json |
| ev1d_uniform576  | 128 | 0.555 | 7.661  | 1.137 | 16.71 | ev_debug_k=576 (force all-k576) | runs/ev1d_uniform576.json |

---

## §H. EV-1e — DECISIVE fair open-loop GPU test (negative verdict)

LLaVA-1.5-7B, TextVQA, c64, V1, **recalibrated load-dependent allocator + EV/fixed
under identical engine settings** (the fair test EV-1d was not). Open-loop Poisson
at offered λ ∈ {8, 12, 15} req/s, ev_mixed_slo 5000,15000, e2e@10 s. Full grid:
EV + fixed r0/r50/r75 at each λ. tag `GPU-measured`. n=200, #runs=1 per cell.

### H.1 EV-1e full open-loop grid (the decisive table)

| λ | cell | k_policy | r | acc | req/s | goodput_acc | met_rate | frac_met | wall_s | source |
|---|---|---|---|---|---|---|---|---|---|---|
| 8  | ev1e_r8_r0  | fixed | 0.00 | 0.575 | 7.229  | 4.157 | 7.157 | 0.990 | 27.66 | runs/ev1e_r8_r0.json  |
| 8  | ev1e_r8_r50 | fixed | 0.50 | 0.450 | 7.492  | 3.372 | 7.492 | 1.000 | 26.69 | runs/ev1e_r8_r50.json |
| 8  | ev1e_r8_r75 | fixed | 0.75 | 0.290 | 7.556  | 2.191 | 7.556 | 1.000 | 26.47 | runs/ev1e_r8_r75.json |
| 8  | ev1e_r8_ev  | elasticvis | 0.50 | 0.560 | 7.354  | 4.118 | 7.354 | 1.000 | 27.20 | runs/ev1e_r8_ev.json  |
| 12 | ev1e_r12_r0 | fixed | 0.00 | 0.570 | 8.778  | 2.677 | 4.828 | 0.550 | 22.78 | runs/ev1e_r12_r0.json |
| 12 | ev1e_r12_r50| fixed | 0.50 | 0.450 | 10.870 | 4.892 | 10.870| 1.000 | 18.40 | runs/ev1e_r12_r50.json|
| 12 | ev1e_r12_r75| fixed | 0.75 | 0.290 | 11.181 | 3.242 | 11.181| 1.000 | 17.89 | runs/ev1e_r12_r75.json|
| 12 | ev1e_r12_ev | elasticvis | 0.50 | 0.495 | 9.954  | 4.330 | 8.859 | 0.890 | 20.09 | runs/ev1e_r12_ev.json |
| 15 | ev1e_r15_r0 | fixed | 0.00 | 0.575 | 8.970  | 2.646 | 4.754 | 0.530 | 22.30 | runs/ev1e_r15_r0.json |
| 15 | ev1e_r15_r50| fixed | 0.50 | 0.450 | 12.221 | 4.950 | 10.999| 0.900 | 16.36 | runs/ev1e_r15_r50.json|
| 15 | ev1e_r15_r75| fixed | 0.75 | 0.290 | 13.569 | 3.935 | 13.569| 1.000 | 14.74 | runs/ev1e_r15_r75.json|
| 15 | ev1e_r15_ev | elasticvis | 0.50 | 0.445 | 11.188 | 3.748 | 7.775 | 0.695 | 17.88 | runs/ev1e_r15_ev.json |

### H.2 EV vs best-fixed decisive ratios (derived from H.1) — the negative verdict

best-fixed = max goodput_acc over r0/r50/r75 at that λ.

| λ | EV goodput_acc | best fixed (which r) | best fixed goodput_acc | EV / best-fixed | outcome | source |
|---|---|---|---|---|---|---|
| 8  | 4.118 | r0  | 4.157 | **0.991×** | tied (lose by 0.9%) | ev1e_r8_*.json |
| 12 | 4.330 | r50 | 4.892 | **0.885×** | LOSE (−11.5%) | ev1e_r12_*.json |
| 15 | 3.748 | r50 | 4.950 | **0.757×** | LOSE (−24.3%) | ev1e_r15_*.json |

**EV-1e verdict (STATE.md): per-request visual-token budget does NOT improve
goodput@SLO under continuous batching; the sim's +35.5% (§E.5) does not migrate
to GPU.** Root cause (STATE.md): in continuous batching every forward shares GPU
compute, so high-k (k576) requests in the batch lengthen prefill for the WHOLE
batch (including tight-deadline requests) → giving slack requests high-k slows
the batch and tight requests miss SLO; the sim's independent-slot model does not
capture this batch interference.

### H.3 EV-1e allocator realized k-distribution (mechanism)

From `elasticvis.allocator_realized_summary`:

| λ | k_mean | k_dist {576, 288, 144} | ema_e2e_ms | ema_k | n_met / n | source |
|---|---|---|---|---|---|---|
| 8  | 532.8  | {576:170, 288:30, 144:0}  | 1691.7 | 463.0 | 200/200 | ev1e_r8_ev.json  |
| 12 | 437.04 | {576:121, 288:44, 144:35} | 3690.7 | 296.6 | 199/200 | ev1e_r12_ev.json |
| 15 | 397.44 | {576:110, 288:22, 144:68} | 4736.6 | 279.4 | 167/200 | ev1e_r15_ev.json |

tag `GPU-measured`. Shows the allocator DOES adapt k downward as load rises
(load-dependent), but the adaptation does not yield a goodput win (§H.2).

---

## §I. Curated qualitative findings (pointer, not numbers)

Each notes file contains the verified findings + mechanism stories that accompany
the above tables. The numbers in those notes are reproduced above; cite the
tables here, not the notes prose.

- `notes/v2_p0_v1_tableA.md` — F1 holds & stronger on V1; V1-vs-V0 crossover.
- `notes/v2_p1_qwen3vl.md` — F1 attenuated (~1/3); merger & pruner are
  SUBSTITUTES; F2 vision=10% of TTFT on Qwen3-VL; F3 holds but accuracy-fragile.
- `notes/v2_p2_scale.md` — amplification grows to c64 (no saturation); ceiling-
  lift; pruning is a tail-latency reducer; r75 strictly dominates r0 @ c64.
- `notes/v2_p3_crosscompressor.md` — goodput win 4/4 compressors; prune-vs-merge
  tradeoff; random beats saliency at r75 (honest red flag).
- `notes/elasticvis_design.md` §8 — gating characterization + the +35.5% sim
  (later refuted on GPU, §H).

---

## §Y. Claims Registry

Each paper claim → the numbers + evidence-levels that back it. "backed" = the
listed evidence is on disk; see §Z for claims whose backing is partial.

### Claim 1 — "c64 served-throughput amplification" (compression × concurrency)

Prune speedup r75/r0 grows monotonically c1→c64; no saturation at c64.
- 1a. LLaVA-1.5 r75/r0: 1.19× (c1) → 1.53× (c4) → 1.96× (c16) → **2.22× (c64)**
  — derived from §C.1 (`GPU-measured` req/s). Source: runs/v2_p2/batch_c*_r*.json.
- 1b. Qwen3-VL-8B r50/r0: 1.06× (c1) → 1.16× (c12) → **1.34× (c64)** — derived
  from §B.1 + §C.5 (`GPU-measured`). Attenuated but same direction.
- 1c. Serial c1 r75 = 1.24× is the FLOOR; grows to 2.22× at c64 — §C.2 + §C.1
  (`GPU-measured`).
- 1d. Ceiling-lift: r0 plateaus c16→c64 (+12%, 8.23→9.18) while r75 keeps climbing
  (+26%, 16.17→20.39) — §C.1 (`GPU-measured`).
- Evidence level: **GPU-measured (primary)** + derived ratios.

### Claim 2 — "r75 strictly dominates r0 at c64" (throughput AND tail, no tradeoff)

- 2a. 2.22× throughput (20.39 vs 9.18 req/s) AND 2.84× lower p99-TTFT (18369→6471
  ms) AND 2.69× lower p99-e2e (19224→7151 ms) — §C.1, §C.3 (`GPU-measured` +
  derived).
- 2b. Goodput@TTFT≤5s = 13.66 vs 1.84 (7.4×); @e2e≤8s = 20.39 vs 0.87 (23.4×) —
  §C.3 (`derived` from `GPU-measured` raw).
- Evidence level: GPU-measured + derived.

### Claim 3 (layer-1) — "iso-k served throughput is compressor-independent"

Throughput is compressor-invariant at iso-k; the goodput-Pareto win generalizes
4/4 compressors.
- 3a. r50 req/s = 14.15–14.47 (3% spread); r75 req/s = 19.73–20.73 (5%) across
  proxy/true_cls/tome_merge/random — §D.1 (`GPU-measured`).
- 3b. Every compressor's r75 strictly dominates its r0: 2.18–2.30× throughput AND
  2.66–2.85× lower p99-TTFT — §D.2 (`GPU-measured` + derived).
- 3c. Goodput@TTFT≤5s r75/r0 = 7.1× (proxy) / 9.9× (true_cls) / 7.2× (tome) /
  7.9× (random) — §D.3 (`derived`).
- 3d. r0 is byte-identical across compressors (acc=0.590, req/s 8.89–9.20) —
  §D.1 (`GPU-measured`).
- Evidence level: GPU-measured + derived.

### Claim 4 — "prune vs merge tradeoff (merge buys accuracy at ~free throughput)"

- 4a. ToMe acc vs proxy: +0.015 @ r50, **+0.065 @ r75** (0.540 vs 0.475) — §D.4
  (`GPU-measured`).
- 4b. Throughput cost −2.1% @ r50, −1.5% @ r75 (19.73 vs 20.02 req/s) — §D.1/D.4
  (`GPU-measured` + derived).
- Evidence level: GPU-measured + derived.

### Claim 5 — "saliency selectors underperform random at high r on GQA" (honest limitation)

- 5a. r75 acc: proxy 0.475 (Δ −0.060 vs random), true_cls 0.490 (Δ −0.045),
  random 0.535, tome 0.540 (+0.005) — §D.4 (`GPU-measured`).
- Evidence level: GPU-measured.

### Claim 6 (gating, sim) — "accuracy(k) steepness gates ElasticVis's benefit ceiling"

- 6a. 5-benchmark gating: knowledge ~0.01 (no win), GQA 0.12–0.13 (boundary
  NO-GO), TextVQA 0.28–0.29 (WIN) — §E.4 (`sim`; GQA 0.119 + TextVQA 0.28 are
  `GPU-measured`, the knowledge range is curated literature — §Z).
- 6b. Synthetic crossover: H1b mixed-SLO ≈ 0.15; H1 uniform-SLO ≈ 0.40 — §E.4
  (`sim`, gating_sweep.py).
- Evidence level: sim (mechanism) + GPU-measured (acc curves).

### Claim 7 (sim, later REFUTED on GPU) — "ElasticVis +35.5% on TextVQA mixed-SLO"

- 7a. TextVQA H1b Greedy 2.36 vs best-Fixed 1.74 = +35.5% — §E.5 (`sim`,
  confirm_textvqa.py).
- 7b. **REFUTED by GPU**: EV-1e open-loop decisive — EV/best-fixed = 0.991×
  (λ=8, tied), 0.885× (λ=12, lose), 0.757× (λ=15, lose) — §H.2 (`GPU-measured`).
- 7c. Root cause: continuous-batching batch interference (high-k slows the whole
  batch incl. tight-deadline requests); sim's independent-slot model misses it —
  §H.2 mechanism (STATE.md).
- Evidence level: sim (the claim) **overturned by** GPU-measured (the refutation).

### Claim 8 (layer-2, decisive negative) — "per-request visual-token budget fails
under batch interference" [open-loop decisive EV-1e]

- 8a. The full λ∈{8,12,15} grid shows EV never beats best-fixed; loses worse as
  load rises (0.991→0.885→0.757×) — §H.2 (`GPU-measured`).
- 8b. Mechanism supported by the allocator's load-dependent k adaptation
  (k_mean 532.8→437.0→397.4 as λ rises) YET no goodput win — §H.3 (`GPU-measured`).
- 8c. The earlier EV-1d "+7%" was an unfair artifact (enforce_eager asymmetry,
  uncalibrated); 2.742 vs 2.562 = +7.0% — §G.1 (`GPU-measured` + derived, do NOT
  cite as a win).
- Evidence level: **GPU-measured (decisive)**.

### Claim 9 (layer-3) — "aggregate budget / load-adaptive controller"

- 9a. The ElasticVis allocator realizes a load-dependent aggregate budget:
  k_mean falls 532.8 → 437.0 → 397.4 as λ rises 8 → 12 → 15 — §H.3
  (`GPU-measured`).
- 9b. BUT under continuous batching this does not improve goodput@SLO (Claim 8).
  The controller is implemented and behaves as designed; the win is gated by the
  architecture (batch interference), not by the controller logic.
- Evidence level: GPU-measured.

### Claim 10 — "per-request k integrates with c64 continuous batching" (mechanism
enabler, the robust placeholder-shrink integration)

- 10a. ev1c_n80_ev / ev1c_ev_c64: k_dist {144:40, 576:40}, n_rid_hits 78/80,
  n_embed_calls 8, ev_per_batch_k_head shows alternating 144/576 within forwards
  — §F.3 (`GPU-measured`).
- 10b. ev1d_crash_debug (n=30) confirmed order-independent req_id→k matching at
  c64 — §G.4 (`GPU-measured`).
- Evidence level: GPU-measured.

### Claim 11 — "vision fixed cost is small / pruning is post-merger" (F2)

- 11a. LLaVA-1.5 serial c1: vision_tower = 2.7–3.2% of TTFT — §A.2
  (`GPU-measured`).
- 11b. Qwen3-VL-8B c1: vision tower + native 2×2 merger + deepstack = 10% of
  wall (≈18 ms/req), irreducible by post-merger pruning — §B.2 (`GPU-measured` +
  derived).
- Evidence level: GPU-measured + derived.

---

## §Z. Missing numbers — flagged (do NOT invent; reproduce or measure)

1. **P0 (v2_p0) per-request TTFT/e2e/p50/p99/goodput@SLO for c≤12.** All
   `agg.ttft_ms` are NaN; no `e2e_ms_*`/`goodput_*` keys. P0 is throughput-only.
   predictors_fit_report.md §5: "v2_p0 unusable". (Paper: do not cite c12
   tail-latency from P0; use P2 c4/c16/c64 instead.)
2. **EV-0 calibrated_textvqa.py H1b ratio sweep (tight ∈ {4,5,6,7}s).** Printed
   at runtime; not stored. Reproduce: `python runs/elasticvis_ev0/calibrated_textvqa.py`.
3. **EV-0 textvqa_robustness.py 4×3 (rate × p_tight) ratio grid.** Printed at
   runtime; not stored. Reproduce: `python runs/elasticvis_ev0/textvqa_robustness.py`.
4. **5-benchmark gating: knowledge-benchmark (MME/MMBench/ScienceQA) acc(k)
   ranges (~0.01) and the Qwen3-VL TextVQA range (0.29).** These appear only in
   `notes/elasticvis_design.md` §8 as curated/literature values — not present as
   measured curves in any `runs/*.json`. If the paper cites them as measured,
   they must be re-measured (only LLaVA-1.5 GQA 0.119 and TextVQA 0.28 are
   GPU-measured here — §E.1, §F.1).
5. **EV-1d fixed r50/r75 at offered λ=15.** Only ev1d_r15_r0 and ev1d_r15_ev
   exist; no r50/r75 at λ=15 → no fair best-fixed ratio at λ=15 for EV-1d. The
   fair λ=15 grid is in EV-1e (§H), use that instead.
6. **EV-1d "+7.5%": on disk the artifact computes to +7.0%** (2.742/2.562 =
   1.070). The "7.5%" in STATE.md is a rounding of the same comparison. Cite
   +7.0% (derived) or describe as "~+7%" with the unfair-caveat.
7. **Open-loop EV-1c/1d/1e offered rate λ is not stored as a JSON field.** It is
   encoded only in the filename (r8/r12/r15) and inferable from the run script
   (default λ=8 for the `ol_`/`openloop_` EV-1d cells; n/wall ≈ 7.3 completed/s).
   If the paper needs the exact offered λ per cell, read `runs/*run*.sh`.
8. **c128 at r0 / c128 at r75.** Not measured (single-A40 KV-infeasible at r0;
   c128@r75 skipped as unfair c-across-r). Only the projection in notes §6.
9. **Qwen3-VL cross-compressor panel / TextVQA cross-compressor panel.** Not
   measured (P3 is GQA-only, LLaVA-1.5-only). Listed as P3 open items.
10. **v2_p1 (Qwen3-VL) per-request TTFT/e2e/goodput.** Flat-schema JSONs store
    only req_per_s/acc/mean_ptid_len/wall_s; no per-request latency. F1 on
    Qwen3-VL is throughput-only.

---

*End of final_results_v3.md. If a number you need is not above, it is not on
disk — add it via a new run, not from memory.*

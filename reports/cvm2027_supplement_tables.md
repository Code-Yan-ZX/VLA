# CVM2027 supplement — auto tables (2026-10-26)

## P1 main table — keep=25%, full splits, official attempted denominator

| Benchmark | None@100% | RBM@25 | FastV-k3@25 | HiLo@25 |
|---|---:|---:|---:|---:|
| textvqa (acc) | 83.24 | 75.50 | — | 66.77 |
| ocrbench (pts/1000) | 707.00 | 559.00 | 413.00 | 613.00 |

Pairs (A−B, paired bootstrap 20k, attempted denom; * = common-answered sensitivity where skip sets differ):
- textvqa rb_none: Δ=-7.74 CI[-8.65,-6.84] p=0.0000 (common-answered Δ=-7.74, n=4995)
- textvqa hl_none: Δ=-16.47 CI[-17.59,-15.36] p=0.0000 (common-answered Δ=-16.56, n=4995)
- textvqa hl_rb: Δ=-8.73 CI[-9.94,-7.51] p=0.0000 (common-answered Δ=-8.82, n=4995)
- textvqa hl_fv: Δ=-10.05 CI[-11.25,-8.87] p=0.0000 (common-answered Δ=-10.14, n=4995)
- ocrbench rb_fv: Δ=+146.00* CI[+118.00,+175.00] p=0.0000 (common-answered Δ=+158.52, n=921)
- ocrbench rb_none: Δ=-148.00* CI[-175.00,-122.00] p=0.0000 (common-answered Δ=-160.69, n=921)
- ocrbench fv_none: Δ=-294.00* CI[-324.00,-264.00] p=0.0000 (common-answered Δ=-319.22, n=921)
- ocrbench hl_none: Δ=-94.00* CI[-126.00,-63.00] p=0.0000 (common-answered Δ=-163.95, n=921)
- ocrbench hl_rb: Δ=+54.00* CI[+23.00,+85.00] p=0.0008 (common-answered Δ=-3.26, n=921)
- ocrbench hl_fv: Δ=+200.00* CI[+168.00,+233.00] p=0.0000 (common-answered Δ=+155.27, n=921)

## Budget curve — textvqa (acc; TextVQA 1000 stratified subset)
| keep | RBM | FastV | HiLo | Δvs None (RBM/FastV/HiLo, attempted*) |
|---|---:|---:|---:|---|
| 0.25 | 75.29 | 76.95 | 66.20 | -7.64 / -5.98 / -16.73 |
| 0.50 | 79.65 | 82.24 | 75.64 | -3.28 / -0.69 / -7.29 |
| 0.75 | 81.73 | 82.57 | 81.24 | -1.20 / -0.36 / -1.69 |

## Budget curve — ocrbench (pts/1000; OCRBench full 1000)
| keep | RBM | FastV | HiLo | Δvs None (RBM/FastV/HiLo, attempted*) |
|---|---:|---:|---:|---|
| 0.25 | 559.00 | 413.00 | 613.00 | -148.00 / -294.00 / -94.00 |
| 0.50 | 659.00 | 569.00 | 709.00 | -48.00 / -138.00 / +2.00 |
| 0.75 | 684.00 | 676.00 | 740.00 | -23.00 / -31.00 / +33.00 |

## DocVQA-500 stress (subset, unified 2M px cap, 0 skips all arms)
| arm | ANLS attempted | Δ vs None | p |
|---|---:|---:|---:|
| RBM@25 | 82.76 | -12.96pp CI[-16.0,-10.0] | 0.0000 |
| None@100% | 95.73 | — | — |
| FastV@25 | 70.45 | -25.28pp CI[-29.0,-21.7] | 0.0000 |
| HiLo@25 | 76.38 | -19.35pp CI[-22.8,-16.0] | 0.0000 |
- direct pair rb_hl: Δ=+6.39pp CI[+3.04,+9.78] p=0.0005
- direct pair rb_fv: Δ=+12.32pp CI[+8.15,+16.47] p=0.0000

## Quality target: >=95% of same-protocol None (attempted). Cheapest MEASURED point meeting it
- textvqa rb: MEETS at keep=0.50
- textvqa fv: MEETS at keep=0.50
- textvqa hl: MEETS at keep=0.75
- ocrbench rb: MEETS at keep=0.75
- ocrbench fv: MEETS at keep=0.75
- ocrbench hl: MEETS at keep=0.50

## timing_harness — per-request cost (200 requests x 3 repeats, bs=1)
| config | p50 total_s | p95 total_s | p50 TTFT_s | out_len med | peak_alloc GB | work_wall_s |
|---|---:|---:|---:|---:|---:|---:|
| none | 0.604 | 1.915 | 0.422 | 4.0 | 42.7 | 489.9 |
| pre_0.25 | 0.466 | 1.914 | 0.298 | 4.0 | 41.5 | 469.5 |
| pre_0.5 | 0.504 | 1.872 | 0.330 | 4.0 | 41.03 | 462.9 |
| pre_0.75 | 0.543 | 1.851 | 0.372 | 4.0 | 42.67 | 460.5 |
| fastv_0.25 | 0.530 | 1.903 | 0.310 | 4.0 | 41.45 | 491.3 |
| fastv_0.5 | 0.550 | 1.838 | 0.338 | 4.0 | 42.44 | 488.2 |
| fastv_0.75 | 0.587 | 1.845 | 0.376 | 4.0 | 42.67 | 484.9 |

## timing_hilo — per-request cost (200 requests x 3 repeats, bs=1)
| config | p50 total_s | p95 total_s | p50 TTFT_s | out_len med | peak_alloc GB | work_wall_s |
|---|---:|---:|---:|---:|---:|---:|
| hilo_0.25 | 0.371 | 1.435 | 0.043 | 4.5 | 17.28 | 364.6 |
| hilo_0.5 | 0.363 | 1.453 | 0.033 | 4.5 | 17.28 | 361.8 |
| hilo_0.75 | 0.361 | 1.468 | 0.032 | 4.5 | 17.28 | 364.7 |

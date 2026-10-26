# Quality and cost on the corrected fixed 200-request workload

Fixed TextVQA-100 and OCRBench-100 requests; subset, same protocol as corrected timing.

Shared vision SDPA / text eager, bf16, native positions, greedy <=32 new tokens. Quality uses the first repeated answer; all three repetitions must agree. Latencies include processor and transfer from an already decoded RGB image; disk/decode/loading/warmup excluded.

| Config | Task subset | Official score | Δ vs None [95% CI] | p50 / p95 total (s) | p50 / p95 TTFT (s) | Answered / attempted |
|---|---|---:|---:|---:|---:|---:|
| none | textvqa n=100 | 81.70 | +0.00 [+0.00, +0.00] | 0.396 / 0.527 | 0.309 / 0.424 | 100 / 100 |
| none | ocrbench n=100 | 780.00 | +0.00 [+0.00, +0.00] | 0.713 / 3.654 | 0.084 / 2.387 | 100 / 100 |
| pre_0.25 | textvqa n=100 | 73.00 | -8.70 [-15.40, -2.30] | 0.254 / 0.414 | 0.172 / 0.224 | 100 / 100 |
| pre_0.25 | ocrbench n=100 | 670.00 | -110.00 [-190.00, -30.00] | 0.572 / 2.033 | 0.077 / 0.997 | 100 / 100 |
| pre_0.5 | textvqa n=100 | 80.50 | -1.20 [-5.40, +2.90] | 0.295 / 0.457 | 0.210 / 0.277 | 100 / 100 |
| pre_0.5 | ocrbench n=100 | 760.00 | -20.00 [-80.00, +30.00] | 0.642 / 2.403 | 0.078 / 1.341 | 100 / 100 |
| pre_0.75 | textvqa n=100 | 80.10 | -1.60 [-5.20, +1.70] | 0.331 / 0.477 | 0.248 / 0.345 | 100 / 100 |
| pre_0.75 | ocrbench n=100 | 770.00 | -10.00 [-50.00, +30.00] | 0.730 / 2.951 | 0.086 / 1.790 | 100 / 100 |
| fastv_0.25 | textvqa n=100 | 79.20 | -2.50 [-8.10, +3.00] | 0.269 / 0.427 | 0.185 / 0.244 | 100 / 100 |
| fastv_0.25 | ocrbench n=100 | 500.00 | -280.00 [-390.00, -170.00] | 0.585 / 2.352 | 0.080 / 1.162 | 100 / 100 |
| fastv_0.5 | textvqa n=100 | 80.40 | -1.30 [-4.70, +1.80] | 0.306 / 0.463 | 0.221 / 0.294 | 100 / 100 |
| fastv_0.5 | ocrbench n=100 | 690.00 | -90.00 [-180.00, -10.00] | 0.656 / 2.536 | 0.082 / 1.477 | 100 / 100 |
| fastv_0.75 | textvqa n=100 | 81.00 | -0.70 [-3.70, +2.10] | 0.340 / 0.498 | 0.256 / 0.359 | 100 / 100 |
| fastv_0.75 | ocrbench n=100 | 790.00 | +10.00 [-30.00, +50.00] | 0.729 / 3.043 | 0.088 / 1.871 | 100 / 100 |
| hilo_0.25 | textvqa n=100 | 65.70 | -16.00 [-23.20, -9.40] | 0.357 / 0.491 | 0.247 / 0.321 | 100 / 100 |
| hilo_0.25 | ocrbench n=100 | 590.00 | -190.00 [-280.00, -100.00] | 0.830 / 2.867 | 0.103 / 1.502 | 100 / 100 |
| hilo_0.5 | textvqa n=100 | 72.50 | -9.20 [-14.60, -4.40] | 0.418 / 0.569 | 0.320 / 0.424 | 100 / 100 |
| hilo_0.5 | ocrbench n=100 | 710.00 | -70.00 [-150.00, +10.00] | 0.923 / 3.190 | 0.110 / 2.021 | 100 / 100 |
| hilo_0.75 | textvqa n=100 | 79.30 | -2.40 [-5.80, +0.00] | 0.494 / 0.660 | 0.398 / 0.543 | 100 / 100 |
| hilo_0.75 | ocrbench n=100 | 750.00 | -30.00 [-80.00, +20.00] | 0.978 / 3.939 | 0.126 / 2.629 | 100 / 100 |

## Cheapest measured point meeting 95% of same-protocol None

- textvqa pre: pre_0.5, p50=0.295s (target=77.61).
- textvqa fastv: fastv_0.25, p50=0.269s (target=77.61).
- textvqa hilo: hilo_0.75, p50=0.494s (target=77.61).
- ocrbench pre: pre_0.5, p50=0.642s (target=741.00).
- ocrbench fastv: fastv_0.75, p50=0.729s (target=741.00).
- ocrbench hilo: hilo_0.75, p50=0.978s (target=741.00).

## Request segments (p50 seconds) and raw peak memory

Each task uses its fixed 100 requests; three repeats are reduced to one median per request before the p50. Selection/fusion can overlap prefill and must not be added again. Total and TTFT come from direct same-request timestamps. Memory uses the maximum over all raw repeats, in GiB.

| Config | Task | CPU processor | Transfer | Full vision | Selection | Fusion | Prefill | TTFT | Decode | Peak allocated / reserved (GiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | textvqa | 0.0170 | 0.0037 | 0.0788 | 0.0000 | 0.0000 | 0.2051 | 0.3086 | 0.0814 | 16.94 / 17.63 |
| none | ocrbench | 0.0054 | 0.0010 | 0.0227 | 0.0000 | 0.0000 | 0.0550 | 0.0843 | 0.4299 | 22.07 / 28.06 |
| pre_0.25 | textvqa | 0.0176 | 0.0037 | 0.0788 | 0.0008 | 0.0000 | 0.0692 | 0.1724 | 0.0804 | 16.59 / 17.06 |
| pre_0.25 | ocrbench | 0.0054 | 0.0010 | 0.0225 | 0.0008 | 0.0000 | 0.0461 | 0.0768 | 0.3444 | 17.27 / 18.52 |
| pre_0.5 | textvqa | 0.0172 | 0.0035 | 0.0789 | 0.0008 | 0.0000 | 0.1071 | 0.2100 | 0.0806 | 16.60 / 17.18 |
| pre_0.5 | ocrbench | 0.0050 | 0.0010 | 0.0224 | 0.0008 | 0.0000 | 0.0469 | 0.0775 | 0.3721 | 18.12 / 20.32 |
| pre_0.75 | textvqa | 0.0163 | 0.0034 | 0.0792 | 0.0008 | 0.0000 | 0.1453 | 0.2481 | 0.0808 | 16.78 / 17.38 |
| pre_0.75 | ocrbench | 0.0053 | 0.0010 | 0.0228 | 0.0008 | 0.0000 | 0.0522 | 0.0858 | 0.4180 | 19.81 / 23.53 |
| fastv_0.25 | textvqa | 0.0164 | 0.0034 | 0.0793 | 0.0003 | 0.0000 | 0.0846 | 0.1854 | 0.0807 | 16.82 / 17.70 |
| fastv_0.25 | ocrbench | 0.0057 | 0.0010 | 0.0227 | 0.0003 | 0.0000 | 0.0500 | 0.0802 | 0.4112 | 21.59 / 28.56 |
| fastv_0.5 | textvqa | 0.0161 | 0.0034 | 0.0793 | 0.0003 | 0.0000 | 0.1191 | 0.2206 | 0.0852 | 16.82 / 17.65 |
| fastv_0.5 | ocrbench | 0.0055 | 0.0009 | 0.0227 | 0.0003 | 0.0000 | 0.0506 | 0.0816 | 0.3835 | 21.59 / 28.49 |
| fastv_0.75 | textvqa | 0.0169 | 0.0036 | 0.0789 | 0.0003 | 0.0000 | 0.1528 | 0.2563 | 0.0811 | 16.82 / 17.47 |
| fastv_0.75 | ocrbench | 0.0061 | 0.0009 | 0.0225 | 0.0003 | 0.0000 | 0.0547 | 0.0880 | 0.4405 | 21.58 / 28.13 |
| hilo_0.25 | textvqa | 0.0306 | 0.0056 | 0.0783 | 0.0476 | 0.0016 | 0.1330 | 0.2465 | 0.0912 | 16.82 / 17.68 |
| hilo_0.25 | ocrbench | 0.0116 | 0.0013 | 0.0229 | 0.0107 | 0.0016 | 0.0637 | 0.1030 | 0.6098 | 21.61 / 28.55 |
| hilo_0.5 | textvqa | 0.0295 | 0.0056 | 0.0785 | 0.0862 | 0.0000 | 0.2047 | 0.3199 | 0.0915 | 16.82 / 17.72 |
| hilo_0.5 | ocrbench | 0.0112 | 0.0013 | 0.0230 | 0.0190 | 0.0000 | 0.0709 | 0.1099 | 0.6166 | 21.61 / 28.56 |
| hilo_0.75 | textvqa | 0.0295 | 0.0055 | 0.0784 | 0.1289 | 0.0000 | 0.2811 | 0.3979 | 0.0904 | 16.82 / 17.72 |
| hilo_0.75 | ocrbench | 0.0118 | 0.0013 | 0.0229 | 0.0278 | 0.0000 | 0.0885 | 0.1257 | 0.5808 | 21.61 / 28.56 |

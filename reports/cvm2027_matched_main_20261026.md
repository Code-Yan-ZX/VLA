# Attention-matched modern-baseline main comparison (2026-10-26)

Attention-matched Qwen3-VL full TextVQA-5000 and OCRBench-1000; compressed retention .25 and uncompressed None.

This is a new protocol: shared vision SDPA / decoder eager, bf16, native positions, four-million-pixel cap, greedy decoding up to 32 new tokens. Both full benchmark splits include all attempted IDs and assign zero to failures. Historical operational comparisons are retained separately. The old 50/75% curves use their original attention protocol and are not relabelled as matched curves.

| Benchmark | None@100% | RBM@25% | FastV-k3@25% | Hi-Lo@25% | Answered None / RBM / FastV / Hi-Lo |
|---|---:|---:|---:|---:|---|
| textvqa n=5000 | 83.45 | 75.58 | 76.90 | 66.75 | 5000 / 5000 / 5000 / 5000 |
| ocrbench n=1000 | 765.00 | 610.00 | 453.00 | 612.00 | 1000 / 1000 / 1000 / 1000 |

## textvqa: all six planned pairs

| A − B | Attempted Δ [95% CI] | p | Common-answered Δ [95% CI] | Common n |
|---|---:|---:|---:|---:|
| RBM − FastV-k3 | -1.32 [-2.32, -0.34] | 0.0093 | -1.32 [-2.29, -0.32] | 5000 |
| RBM − Hi-Lo | +8.83 [+7.63, +10.04] | <0.0001 | +8.83 [+7.65, +10.02] | 5000 |
| RBM − None | -7.87 [-8.78, -6.95] | <0.0001 | -7.87 [-8.80, -6.97] | 5000 |
| FastV-k3 − Hi-Lo | +10.15 [+8.93, +11.37] | <0.0001 | +10.15 [+8.92, +11.36] | 5000 |
| FastV-k3 − None | -6.55 [-7.37, -5.74] | <0.0001 | -6.55 [-7.37, -5.75] | 5000 |
| Hi-Lo − None | -16.70 [-17.83, -15.56] | <0.0001 | -16.70 [-17.83, -15.59] | 5000 |

Input hashes, original visual N, full prompt lengths and image source match on all 5000 commonly answered IDs. All compressed methods retain exactly round(.25N) per ID; mean N=744.71, K=186.18. Failure IDs and reasons are retained in the JSON.

## ocrbench: all six planned pairs

| A − B | Attempted Δ [95% CI] | p | Common-answered Δ [95% CI] | Common n |
|---|---:|---:|---:|---:|
| RBM − FastV-k3 | +157.00 [+128.00, +187.00] | <0.0001 | +157.00 [+128.00, +187.00] | 1000 |
| RBM − Hi-Lo | -2.00 [-31.00, +27.00] | 0.9429 | -2.00 [-31.00, +27.00] | 1000 |
| RBM − None | -155.00 [-182.00, -128.00] | <0.0001 | -155.00 [-182.00, -128.00] | 1000 |
| FastV-k3 − Hi-Lo | -159.00 [-191.00, -127.00] | <0.0001 | -159.00 [-191.00, -127.00] | 1000 |
| FastV-k3 − None | -312.00 [-343.00, -281.00] | <0.0001 | -312.00 [-343.00, -282.00] | 1000 |
| Hi-Lo − None | -153.00 [-181.00, -126.00] | <0.0001 | -153.00 [-181.00, -125.00] | 1000 |

Input hashes, original visual N, full prompt lengths and image source match on all 1000 commonly answered IDs. All compressed methods retain exactly round(.25N) per ID; mean N=621.98, K=155.51. Failure IDs and reasons are retained in the JSON.

All paired intervals use 20,000 resamples of paired question scores and fixed seeds. Common-answered sensitivity does not drop failures from the primary denominator. Attention, preprocessing, weights and final budgets match. Hi-Lo still uses the official adapted model implementation; no-compression and P3 overlap checks bound this implementation difference.

Reproduce from the worktree root:

```bash
/home/dell/miniconda3/envs/qwen3vl_clean/bin/python scripts/cvm24h_score_matched.py
```

Original predictions are read only. The JSON records every input SHA256, pinned scoring references, configuration, failure IDs, all twelve planned pair comparisons and validation evidence.

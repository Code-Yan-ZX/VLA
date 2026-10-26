# CVM2027 supplement — audited tables (2026-10-26)

Scores use all attempted IDs, with failures scored zero. Hi-Lo uses the official vendored adaptation with SDPA; RBM/FastV use the eager harness. Common-answered analyses control the failure set, but do not remove implementation differences.

## Main comparison at 25% retention

| Benchmark | None@100% | RBM | FastV-k3 | Hi-Lo | Attempted / answered |
|---|---:|---:|---:|---:|---|
| textvqa | 83.24 | 75.50 | 76.83 | 66.77 | 5000; 4995 / 4995 / 5000 |
| ocrbench | 707.00 | 559.00 | 413.00 | 613.00 | 1000; 921 / 921 / 1000 |

Answered counts are shown in RBM/FastV/Hi-Lo order; None answers 4,995 TextVQA and 921 OCRBench requests.

| Pair A − B | Task | Attempted Δ [95% CI] | Common-answered Δ [95% CI] | Common n |
|---|---|---:|---:|---:|
| rb_fv | textvqa | -1.32 [-2.33, -0.34] | -1.32 [-2.33, -0.32] | 4995 |
| rb_none | textvqa | -7.74 [-8.65, -6.84] | -7.74 [-8.66, -6.83] | 4995 |
| fv_none | textvqa | -6.41 [-7.23, -5.60] | -6.42 [-7.24, -5.61] | 4995 |
| hl_none | textvqa | -16.47 [-17.59, -15.36] | -16.56 [-17.67, -15.45] | 4995 |
| hl_rb | textvqa | -8.73 [-9.94, -7.51] | -8.82 [-10.01, -7.59] | 4995 |
| hl_fv | textvqa | -10.05 [-11.25, -8.87] | -10.14 [-11.36, -8.94] | 4995 |
| rb_fv | ocrbench | +146.00 [+118.00, +175.00] | +158.52 [+128.12, +187.84] | 921 |
| rb_none | ocrbench | -148.00 [-175.00, -122.00] | -160.69 [-188.93, -132.46] | 921 |
| fv_none | ocrbench | -294.00 [-324.00, -264.00] | -319.22 [-350.73, -287.73] | 921 |
| hl_none | ocrbench | -94.00 [-126.00, -63.00] | -163.95 [-193.27, -134.64] | 921 |
| hl_rb | ocrbench | +54.00 [+23.00, +85.00] | -3.26 [-33.66, +26.09] | 921 |
| hl_fv | ocrbench | +200.00 [+168.00, +233.00] | +155.27 [+122.69, +187.84] | 921 |

Optional full TextVQA grid: partial; subset curves remain the complete comparison. Completed extension cells are retained in scores/full_q10_*.json.

## TextVQA fixed 1,000-question subset

| Keep | RBM | FastV-k3 | Hi-Lo | None |
|---|---:|---:|---:|---:|
| 0.25 | 75.29 | 76.95 | 66.20 | 82.93 |
| 0.50 | 79.65 | 82.24 | 75.64 | 82.93 |
| 0.75 | 81.73 | 82.57 | 81.24 | 82.93 |

## OCRBench full 1,000 attempts

| Keep | RBM | FastV-k3 | Hi-Lo | None |
|---|---:|---:|---:|---:|
| 0.25 | 559.00 | 413.00 | 613.00 | 707.00 |
| 0.50 | 659.00 | 569.00 | 709.00 | 707.00 |
| 0.75 | 684.00 | 676.00 | 740.00 | 707.00 |

## OCRBench common 921 answered IDs, normalized ×1,000

| Keep | RBM | FastV-k3 | Hi-Lo | None |
|---|---:|---:|---:|---:|
| 0.25 | 606.95 | 448.43 | 603.69 | 767.64 |
| 0.50 | 715.53 | 617.81 | 711.18 | 767.64 |
| 0.75 | 742.67 | 733.98 | 741.59 | 767.64 |

## DocVQA fixed 500-question subset, unified 2M pixel cap

| Method | ANLS (%) | Δ vs None [95% CI] | p |
|---|---:|---:|---:|
| RBM@25% | 82.76 | -12.96 [-15.96, -10.05] | <0.0001 |
| FastV-k3@25% | 70.45 | -25.28 [-28.97, -21.67] | <0.0001 |
| Hi-Lo@25% | 76.38 | -19.35 [-22.76, -16.03] | <0.0001 |
| None@100% | 95.73 | — | — |

rb_hl: Δ=+6.39pp, CI[+3.04, +9.78], p=0.0005.

rb_fv: Δ=+12.32pp, CI[+8.15, +16.47], p=<0.0001.

## 95% of None quality target

This table gives the minimum **tested retention** meeting the target. It does not assume retention orders measured latency. The Hi-Lo OCRBench attempted result includes its higher completion rate.

| Scope | None | 95% target | RBM keep | FastV keep | Hi-Lo keep |
|---|---:|---:|---:|---:|---:|
| TextVQA fixed 1,000-question subset | 82.93 | 78.78 | 0.50 | 0.50 | 0.75 |
| OCRBench full 1,000 attempts | 707.00 | 671.65 | 0.75 | 0.75 | 0.50 |
| OCRBench common 921 answered IDs, normalized ×1,000 | 767.64 | 729.26 | 0.75 | 0.75 | 0.75 |

## Actual visual token budgets

Statistics use the same answered IDs across the three methods at each budget. Hi-Lo fusion boosts attention before removal; its fusion candidates are not additional retained tokens. RBM retains K before decoder layer 0; FastV-k3 processes N through layers 0–3, then K; Hi-Lo processes N through layers 0–2, then K. All methods execute the full vision encoder and merger.

| Scope | Keep | Common n | Mean input N | Mean K (RBM / FastV / Hi-Lo) | K equal on every common ID |
|---|---:|---:|---:|---:|---|
| textvqa 1k subset | 0.25 | 999 | 739.67 | 184.92 / 184.92 / 184.92 | yes |
| textvqa 1k subset | 0.50 | 999 | 739.67 | 369.83 / 369.83 / 369.83 | yes |
| textvqa 1k subset | 0.75 | 999 | 739.67 | 554.75 / 554.75 / 554.75 | yes |
| ocrbench full | 0.25 | 921 | 352.93 | 88.26 / 88.26 / 88.26 | yes |
| ocrbench full | 0.50 | 921 | 352.93 | 176.43 / 176.43 / 176.43 | yes |
| ocrbench full | 0.75 | 921 | 352.93 | 264.66 / 264.66 / 264.66 | yes |

## Timing validation

The original timing artifacts are excluded from comparison: the harness enabled gradient recording, FastV omitted deepstack features, and the Hi-Lo model reused its initial retention configuration at later budgets. Corrected timing requires a new artifact with explicit inference, parity, budget, and boundary validation.

Validated corrected timing available: `timing_harness_corrected.json`; quality-cost statistics use its own saved answers.

Validated corrected timing available: `timing_hilo_corrected.json`; quality-cost statistics use its own saved answers.

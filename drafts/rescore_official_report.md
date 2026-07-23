# Rescore with OFFICIAL metrics — claim-overturn gate

_Generated 2026-07-23 · offline · CPU-only · scorer: `src/v3_premerger/official_scorers.py`_

**Metrics applied:** TextVQA → official VQA accuracy; DocVQA → official ANLS. GQA/ChartQA/OCRBench/MME/MMBench/ScienceQA keep their existing stored metric (unchanged).

**Important caveat:** VQA-acc and ANLS are computed on the RAW stored generations (`per_sample[].answer`, often verbose multi-sentence text). No answer extraction is applied, so these are an *honest lower bound* of the official metric for these runs.

### Root cause of the near-zero official scores

The router-probe runs were generated with a free-form prompt (no short-answer constraint; `max_tokens=32` but outputs are full sentences, median ~133 chars). Official VQA-acc requires the normalized prediction to EXACTLY equal a short gt answer, and official ANLS requires high string overlap with a short gt — both fail against verbose sentences. Consequently textvqa VQA-acc is **exactly 0.0** for every pre AND post sample, and docvqa ANLS collapses to ~0.02 (pre) / 0.0 (post), driven by a handful of coincidentally short 'yes'/'no' answers. The OLD containment rule passed whenever the short gt appeared *anywhere* inside the verbose answer, which is why it credited 0.695 / 0.725 where the official metric sees ~0. **These runs are therefore not comparable on the official scale; the containment numbers over-state absolute performance by ~100×.** Re-running with a short-answer prompt would be needed to get meaningful official-metric values; that is a data-generation change, out of scope here.

## ⚑ DIRECTION-FLIP CHECK (top priority)

> **⚠️ AT LEAST ONE DIRECTION FLIPPED under the official metrics — review the affected conclusion(s) below before reporting.**

- 🔄 **FLIP** **textvqa** (VQA-acc): OLD containment pre>post (0.695 vs 0.255) → NEW tie (0.000 vs 0.000)
- ✅ hold **docvqa** (ANLS): OLD containment pre>post (0.725 vs 0.390) → NEW pre>post (0.019 vs 0.000)
- ✅ hold **gqa** (stored): OLD containment post>pre (0.320 vs 0.380) → NEW post>pre (0.320 vs 0.380)
- ✅ hold **ocrbench** (stored): OLD containment pre>post (0.580 vs 0.165) → NEW pre>post (0.580 vs 0.165)
- ✅ hold **chartqa** (stored): OLD containment tie (0.190 vs 0.190) → NEW tie (0.190 vs 0.190)

## VERDICT at keep=25% (r=0.75)

| Benchmark | Metric | OLD pre | OLD post | NEW pre | NEW post | OLD dir | NEW dir | NEW gap (pre−post, pp) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| textvqa | VQA-acc | 0.695 | 0.255 | 0.000 | 0.000 | pre>post | tie | +0.0 | **FLIP** ⚠near-floor |
| docvqa | ANLS | 0.725 | 0.390 | 0.019 | 0.000 | pre>post | pre>post | +1.9 | HOLD ⚠near-floor |
| gqa | stored | 0.320 | 0.380 | 0.320 | 0.380 | post>pre | post>pre | -6.0 | HOLD |
| ocrbench | stored | 0.580 | 0.165 | 0.580 | 0.165 | pre>post | pre>post | +41.5 | HOLD |
| chartqa | stored | 0.190 | 0.190 | 0.190 | 0.190 | tie | tie | +0.0 | HOLD |

Interpretation: **textvqa/docvqa** NEW pre/post use the official metric; **gqa/chartqa/ocrbench** NEW == OLD (existing metric, so direction cannot change).

## Full per-cell table (OLD containment vs NEW official)

| Cell | Bench | Mode | keep% | sel | OLD acc | NEW metric | NEW value | nonzero/n | n | note |
|---|---|---|---|---|---|---|---|---|---|---|
| B_chartqa_r0.875.json | chartqa | post | 12 | l2 | 0.095 | kept | — | — | 200 |  |
| vz_chartqa_r0.875.json | chartqa | post | 12 | l2 | 0.095 | kept | — | — | 200 |  |
| C_chartqa_r0.875.json | chartqa | pre | 12 | l2 | 0.150 | kept | — | — | 200 |  |
| B_chartqa_r0.750_attn.json | chartqa | post | 25 | attn | 0.165 | kept | — | — | 200 |  |
| B_chartqa_r0.750.json | chartqa | post | 25 | l2 | 0.190 | kept | — | — | 200 |  |
| vz_chartqa_r0.750.json | chartqa | post | 25 | l2 | 0.190 | kept | — | — | 200 |  |
| C_chartqa_r0.750_attn.json | chartqa | pre | 25 | attn | 0.190 | kept | — | — | 200 |  |
| C_chartqa_r0.750.json | chartqa | pre | 25 | l2 | 0.190 | kept | — | — | 200 |  |
| B_chartqa_r0.500.json | chartqa | post | 50 | l2 | 0.335 | kept | — | — | 200 |  |
| C_chartqa_r0.500.json | chartqa | pre | 50 | l2 | 0.390 | kept | — | — | 200 |  |
| A_chartqa.json | chartqa | baseline | 100 | l2 | 0.820 | kept | — | — | 200 |  |
| vz_postmode_docvqa_r0.875.json | docvqa | post | 12 | l2 | 0.135 | anls | 0.000 | 0/200 | 200 |  |
| B_docvqa_r0.750_attn.json | docvqa | post | 25 | attn | 0.365 | anls | 0.000 | 0/200 | 200 |  |
| C_vzstyle_docvqa_r0.750_l2_n200.json | docvqa | post | 25 | l2 | 0.390 | anls | 0.000 | 0/200 | 200 |  |
| post_docvqa_r0.750_l2_n200.json | docvqa | post | 25 | l2 | 0.390 | anls | 0.000 | 0/200 | 200 |  |
| pre_docvqa_r0.750_l2_n200.json | docvqa | pre | 25 | l2 | 0.725 | anls | 0.019 | 5/200 | 200 |  |
| vz_gqa_r0.875.json | gqa | post | 12 | l2 | 0.305 | kept | — | — | 200 |  |
| post_gqa_r0.750_l2_n200.json | gqa | post | 25 | l2 | 0.380 | kept | — | — | 200 |  |
| vz_gqa_r0.750.json | gqa | post | 25 | l2 | 0.380 | kept | — | — | 200 |  |
| pre_gqa_r0.750_l2_n200.json | gqa | pre | 25 | l2 | 0.320 | kept | — | — | 200 |  |
| post_mme_r0.750_l2_n200.json | mme | post | 25 | l2 | 0.820 | kept | — | — | 200 |  |
| pre_mme_r0.750_l2_n200.json | mme | pre | 25 | l2 | 0.815 | kept | — | — | 200 |  |
| B_ocrbench_r0.875.json | ocrbench | post | 12 | l2 | 0.075 | kept | — | — | 200 |  |
| vz_ocrbench_r0.875.json | ocrbench | post | 12 | l2 | 0.075 | kept | — | — | 200 |  |
| C_ocrbench_r0.875.json | ocrbench | pre | 12 | l2 | 0.380 | kept | — | — | 200 |  |
| B_ocrbench_r0.750_attn.json | ocrbench | post | 25 | attn | 0.170 | kept | — | — | 200 |  |
| B_ocrbench_r0.750.json | ocrbench | post | 25 | l2 | 0.165 | kept | — | — | 200 |  |
| vz_ocrbench_r0.750.json | ocrbench | post | 25 | l2 | 0.165 | kept | — | — | 200 |  |
| C_ocrbench_r0.750_attn.json | ocrbench | pre | 25 | attn | 0.480 | kept | — | — | 200 |  |
| C_ocrbench_r0.750.json | ocrbench | pre | 25 | l2 | 0.580 | kept | — | — | 200 |  |
| A_ocrbench.json | ocrbench | baseline | 100 | l2 | 0.760 | kept | — | — | 200 |  |
| vz_postmode_textvqa_r0.875.json | textvqa | post | 12 | l2 | 0.175 | vqa_accuracy | 0.000 | 0/200 | 200 |  |
| post_textvqa_r0.750_l2_n200.json | textvqa | post | 25 | l2 | 0.255 | vqa_accuracy | 0.000 | 0/200 | 200 |  |
| vz_postmode_textvqa_r0.750.json | textvqa | post | 25 | l2 | 0.255 | vqa_accuracy | 0.000 | 0/200 | 200 |  |
| pre_textvqa_r0.750_l2_n200.json | textvqa | pre | 25 | l2 | 0.695 | vqa_accuracy | 0.000 | 0/200 | 200 |  |

## Cells skipped (missing per_sample preds)

- None.

## Disk artifacts

- Summary JSON: `runs/rescore_official/summary.json`
- Scorer module: `src/v3_premerger/official_scorers.py`
- This report: `drafts/rescore_official_report.md`

# Cross-Architecture Predictive Law Test

**Hypothesis:** the pre>post stage-effect magnitude is predicted by the
degree of pre/post unit-saliency ranking divergence (M1). Where the 2x2
merger rewrites the ranking (pre and post select DIFFERENT units -> low
Jaccard / low Spearman), the stage effect is large. Where pre and post
agree (high Jaccard), the stage effect is small.

**CPU-only analysis.** M1 recomputed from `runs/v3_merger_aware/survival_capture/*.npz`
(Qwen3-VL, 64 images/bench, L2 both stages, r=0.75). Verified against
`drafts/figures/token_survival_stats.json`. Stage-effect magnitudes from
`experiments/paired_metric_statistics.md` (official scorers, seed=0).

---

## Data availability (M1 = pre/post ranking divergence)

| Family | M1 available? | Source | Benchmarks |
|--------|---------------|--------|------------|
| Qwen3-VL-8B | YES | survival_capture npz (pre/post L2 + Sobel edge, 64 imgs) | textvqa, docvqa, gqa |
| Qwen2.5-VL-7B | **NO** | kept_indices never saved; `--save-unit-scores` does not write kept indices (j3_mechanism_crossarch.md) | - |
| InternVL3-8B | **NO** | no unit-score capture, no kept_indices | - |
| GLM-4.1V-9B | **NO** | n=200, no mechanism capture | - |

**Critical gap:** the cross-architecture quantitative test CANNOT be performed.
M1 exists for only 1 of 4 families (Qwen3-VL). The law is tested WITHIN
Qwen3-VL only (3 benchmarks, 64 images each = 192 per-image points).

---

## Test A: Benchmark-level (3 Qwen3-VL points)

| Benchmark | Jaccard@k | Spearman(pre,post) | Edge-rankshift | Stage-effect (pp) |
|-----------|-----------|---------------------|----------------|-------------------|
| textvqa | 0.2433 | 0.3319 | 0.1546 | +38.37 |
| docvqa | 0.1798 | 0.1370 | 0.4385 | +24.30 |
| gqa | 0.2776 | 0.3601 | 0.0360 | -2.83 |

| Metric | Pearson r (pred sign) | Spearman r |
|--------|-----------------------|------------|
| Jaccard ~ effect | -0.509 (p=0.660) [pred: -] | -0.500 (p=0.667) |
| Spearman(p,p) ~ effect | -0.293 (p=0.811) [pred: -] | -0.500 (p=0.667) |
| Edge-rankshift ~ effect | +0.454 (p=0.700) [pred: +] | +0.500 (p=0.667) |

n=3 points: direction-only, no statistical significance possible.

---

## Test B: Per-image within Qwen3-VL (192 images)

Per-image stage effect = pre_correct - post_correct in {-1, 0, +1}.
n=192 images | pre>post: 50 | post>pre: 12 | tie: 130 (pre>post rate = 26.0%).

| Metric | Pearson r (pred sign) | p | Spearman r | p |
|--------|-----------------------|------|-----------|------|
| Jaccard ~ effect | -0.0239 [pred: -] | 0.7421 | -0.0285 | 0.6948 |
| Spearman(p,p) ~ effect | -0.0056 [pred: -] | 0.9391 | -0.0009 | 0.9901 |
| Edge-rankshift ~ effect | -0.0068 [pred: +] | 0.9252 | +0.0187 | 0.7969 |

### Group comparison: pre>post vs pre<=post

| Metric | pre>post mean (n) | rest mean (n) | diff |
|--------|-------------------|---------------|------|
| Jaccard | 0.2297 (50) | 0.2349 (142) | -0.0052 (MWU p=0.7037) |
| Spearman | 0.2788 (50) | 0.2755 (142) | +0.0033 (MWU p=0.7258) |
| EdgeRS | 0.2056 (50) | 0.2111 (142) | -0.0055 (MWU p=0.8672) |

### Per-bench per-image correlations (Pearson, Jaccard ~ effect)

| Benchmark | n | pre>post | Jaccard~eff r (p) | Spearman(p,p)~eff r (p) |
|-----------|---|----------|--------------------|------------------------|
| textvqa | 64 | 28 | +0.140 (0.269) | +0.193 (0.126) |
| docvqa | 64 | 17 | +0.014 (0.915) | -0.044 (0.730) |
| gqa | 64 | 5 | -0.031 (0.806) | -0.039 (0.757) |

---

## Cross-architecture context (qualitative)

Stage-effect magnitudes (delta pre-post, pp) @ 25% retention:

| Family | textvqa | docvqa | ocrbench | gqa |
|--------|---------|--------|----------|-----|
| qwen3vl | +38.37 | +24.30 | +36.30 | -2.83 |
| qwen2vl | +26.05 | +11.04 | +29.30 | -2.58 |
| internvl3 | +37.42 | +34.64 | +43.20 | -0.38 |
| glm4v | +16.83 | +9.84 | - | +4.50 |

All 3 non-thinking families show the same **pattern**: large pre>post on
text-dense benchmarks (textvqa/docvqa/ocrbench: +11 to +43 pp) and ~zero
on GQA (-0.4 to -2.8 pp). The Qwen3-VL M1 shows more ranking divergence
on text-dense (Jaccard 0.18-0.24, Spearman 0.14-0.33) than on GQA
(Jaccard 0.28, Spearman 0.36). This is **consistent** with the law but
NOT a quantitative cross-architecture test (M1 missing for 3 of 4 families).

---

## Verdict

**Does M1 predict the stage-effect magnitude?**

1. **Benchmark-level (n=3, Qwen3-VL only):** Jaccard~effect Pearson r=-0.509.
   Direction is CORRECT (negative, as predicted),
   but n=3 precludes any significance claim.

2. **Per-image (n=192, Qwen3-VL):** Jaccard~effect Pearson r=-0.0239 (p=0.7421);
   Spearman(pre,post)~effect r=-0.0056 (p=0.9391);
   Edge-rankshift~effect r=-0.0068 (p=0.9252).
   No M1 metric reaches p<0.05 in the predicted direction.

3. **Cross-architecture: UNTESTED.** M1 (ranking divergence) is available for
   Qwen3-VL only. Qwen2.5-VL, InternVL3, and GLM-4.1V have no kept_indices
   or unit scores, so their M1 cannot be computed (confirmed in
   experiments/j3_mechanism_crossarch.md).

**Bottom line:** The law shows the **correct direction** within Qwen3-VL but does
NOT reach significance per-image, and is **NOT cross-architecture validated**.
The predictive-law claim is **not supported** by the current data. Flag honestly.

---

## What would be needed to test the cross-architecture law properly

1. Capture pre/post per-unit L2 scores (like the Qwen3-VL survival_capture)
   for Qwen2.5-VL and InternVL3 on textvqa/docvqa/gqa/ocrbench. This requires
   a GPU forward pass (~1h/family) with `mechanism_token_survival.py --mode capture`
   extended to those architectures.
2. Alternatively, add `kept_indices` output to the runner (`attach_kept_indices`
   exists but is off by default; `--save-unit-scores` was confirmed to NOT write
   kept indices for Qwen2.5-VL). Then re-run pre/post cells with the flag on.
3. With M1 for 3 families x 4 benchmarks = 12 points, the cross-architecture
   correlation (M1 vs stage-effect) becomes testable.

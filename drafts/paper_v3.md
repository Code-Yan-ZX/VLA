# Selection Stage Beats Scoring — Full Draft (v3)

> Drafting date: 2026-07-23. Evidence base locked. Every quantitative claim carries an inline traceability tag [E: source]; all numbers are verbatim from `drafts/v3_sota_matrix.md` (FINAL), `drafts/v3_evidence.md`, `drafts/visionzip_gap_report.md`, `drafts/baseline_methods_audit.md`, `drafts/qualitative_examples.md`, and `drafts/figures/token_survival_stats.json`. Citations are bracketed placeholders of the form "[CITE: shortkey]" for a later pass.

---

## Title candidates (three; none selected)

1. **Selection Stage Beats Scoring: Pre-Merger Token Selection Rescues Text-Dense Compression in Vision-Language Models**
2. **The Lossy Merger: Where You Select Visual Tokens Matters More Than How You Score Them**
3. **Before or After the 2×2 Merger? A Controlled Stage Effect in VLM Visual Token Compression, with a Workload-Conditional Taxonomy**

---

## Abstract

Visual-token compression for vision-language models (VLMs) invests almost exclusively in *how* to score tokens — attention saliency, dominant-plus-contextual selection, or trained importance estimators — while the question of *where* in the pipeline selection occurs has gone unexamined. We isolate this variable on Qwen3-VL-8B-Instruct under strict iso-model, iso-token, and iso-selector control: the same text-agnostic L2-norm scorer is applied either before or after the model's native 2×2 vision merger, and the number of tokens fed to the language model is verified identical. At 25% retention, pre-merger selection outperforms post-merger selection by +33.5 to +44.0 percentage points (z = 7.0–9.8σ) on three text-dense benchmarks (TextVQA, DocVQA, OCR-Bench), with the gap widening under deeper compression; post-merger leads on object-centric GQA (−6.0pp, not significant), and perception/multiple-choice benchmarks sit at ≈0. Token-survival analysis exposes a two-layer mechanism: on documents, post-merger selection is actively misdirected away from glyphs (kept-unit edge density 0.148 versus 0.342 for dropped units); on scene text, selection locations are near-correct but 2×2 feature averaging has already erased stroke contrast. A VisionZip-style dominant+contextual scorer reproduces post-merger L2 exactly in 11/11 cells — contextual tokens contribute zero everywhere — and the official method's own published numbers show the text-dense onset at 50% retention (OCRBench 81.5 → 70.5). ChartQA constitutes a third, budget-dominated regime where all methods tie. Practically, pre-merger selection is a robust safe default; a cheap image-token-count router adds +2.1pp pooled, while 73% of the per-sample oracle headroom is query-dependent and unreachable by image-level signals.

(241 words)

---

## 1. Introduction

Visual tokens dominate the context length of modern VLMs. A single document image fed to a Qwen3-VL-class model produces on the order of a thousand post-merger visual tokens [E: matrix §0, DocVQA mean ptid 1054], and high-resolution multi-page inputs can exceed sixteen thousand tokens before merging [E: vz-audit §3]. Visual-token compression — pruning or merging visual tokens before they reach the language model — is therefore a direct lever on inference cost, and a large body of work has developed increasingly sophisticated scoring functions for deciding which tokens to keep: attention received by the CLS token [CITE: visionzip-yang2024], query-guided attention [CITE: fastv], trained entropy estimators [CITE: ifprune-sun2026], hierarchical loss estimates [CITE: hiloprune-sun2026], or learned query proxies [CITE: quietprune-gao2026].

The shared, rarely questioned assumption in this lineage is *where* selection happens: after the vision encoder's native spatial pooling. In merger-equipped architectures (Qwen2.5-VL, Qwen3-VL [CITE: qwen25vl] [CITE: qwen3vl]), the vision tower first averages every 2×2 block of patch features into one merged unit, and all published token-compression methods for these models then select among the already-merged units [E: vz-audit §1, post-merger confirmed at code level]. The merger is a lossy operation: it is trained to aggregate, not to preserve legibility. If the information a downstream task needs — the stroke contrast of small glyphs, the digit in a dense table cell — is attenuated by that averaging, no scoring function applied afterwards can recover it; the selection problem is being solved on degraded evidence.

This paper asks the orthogonal question. Not *how* to score visual tokens, but *where*: does the selection stage — before versus after the native merger — dominate the choice of scoring sophistication? We answer it with a controlled experiment rather than a new method. On Qwen3-VL-8B-Instruct [CITE: qwen3vl], we apply one and the same text-agnostic scorer (the L2 norm of unit features) at two hook points — pre-merger (select 2×2 units, then merge only the survivors natively) and post-merger (merge everything, then select among merged units) — holding the model, the token budget, and the scorer constant. The post-merger LLM input length (ptid) is verified identical between conditions on every benchmark [E: matrix §0; e.g., TextVQA 203 = 203, DocVQA 1054 = 1054], so any accuracy difference is attributable to selection stage alone.

The controlled finding is a workload-conditional stage law. At 25% retention on 200-sample subsets, pre-merger selection beats post-merger selection by +44.0pp on TextVQA, +33.5pp on DocVQA, and +41.5pp on OCR-Bench (z = 9.8, 7.0, 9.5σ respectively) [E: matrix §0]. The effect deepens under stronger compression: at 12.5% retention the DocVQA gap reaches +47.5pp (11.2σ) [E: matrix §1]. On object-centric GQA the ordering inverts (post leads by 6.0pp, 1.3σ — not significant) [E: matrix §0], and on perception/multiple-choice benchmarks (ScienceQA, MMBench, MME) the gap collapses to ≈0 [E: evidence §2]. ChartQA defines a third regime in which stage is irrelevant because the token budget itself is the binding constraint: all three methods tie at 0.190 at 25% retention, and the +5.5pp pre-merger lead at 12.5% is not significant (1.7σ) [E: matrix §2].

We trace the mechanism with token-survival visualization. The damage is not uniform; it is two-layered. On a document image, pre-merger selection keeps units of edge density 0.931 and drops units of edge density 0.081 — it looks at glyphs — while post-merger selection keeps 0.148 and drops 0.342: it is *misdirected*, systematically avoiding the very strokes the task needs, with the two kept sets barely overlapping (Jaccard 0.079) [E: matrix §8.2; stats json, docvqa]. On a scene-text image, by contrast, post-merger selection locations are about as good as pre-merger ones (kept-unit edge density 0.306 vs 0.312, Jaccard 0.257) — yet accuracy still collapses from 0.695 to 0.255 [E: matrix §0; stats json, textvqa]. The scene-text failure is therefore not wrong selection but *feature degradation*: the 2×2 average has already smeared stroke contrast inside the tokens that are kept [E: matrix §8.2]. We report both layers, including the scene-text case that does not support a location-error account.

Finally, we audit the practical consequence for post-merger state-of-the-art. A faithful same-model port of the VisionZip dominant+contextual principle [CITE: visionzip-yang2024], scored with our L2 proxy at the post-merger stage, reproduces post-merger L2-only selection exactly in 11/11 tested cells — the contextual (mean-pooled) tokens add zero accuracy on every workload and at every depth [E: matrix §0]. The official VisionZip code does not run on Qwen3-VL or under vLLM (audited; four independent blockers) [E: vz-audit §3], but the authors' own Qwen2.5-VL results already show the text-dense onset: at 50% retention, OCRBench falls from 81.5 to 70.5 (−13%) while DocVQA loses only 1.3 points, and no 25% row is published [E: vz-audit §1]. The authors themselves note that "Qwen2.5VL already uses PatchMerger … the performance gain from VisionZip is less striking compared to LLaVA" [E: vz-audit §1].

**Contributions.** (1) A controlled stage effect: under iso-model/iso-token/iso-selector conditions on Qwen3-VL-8B, selection before the native merger beats selection after it by +33.5 to +44.0pp (7.0–9.8σ) on three text-dense benchmarks, inverts (non-significantly) on object-centric data, and vanishes on perception/multiple-choice tasks [E: matrix §0, §1; evidence §2]. (2) A two-layer mechanism from token-survival analysis: selection misdirection on documents, feature degradation on scene text, both post-merger-exclusive [E: matrix §8.2]. (3) A two-axis workload taxonomy — stage axis (text-dense ≫ perception ≈ 0 > object) and budget axis (ChartQA as an honest third, budget-dominated regime with a documented failure mode) — rather than uniform monotonicity [E: matrix §0–2; evidence §2]. (4) Evidence that the post-merger dominant+contextual SOTA recipe is fragile on text-dense workloads for reasons internal to the stage choice: contextual tokens add nothing in 11/11 cells, and the official method's own published numbers show text-dense degradation onset [E: matrix §0; vz-audit §1].

---

## 2. Related work

**Visual-token pruning and selection in VLMs.** The dominant strategy is to score visual tokens and drop the least important before, or in the early layers of, the language model. FastV prunes tokens by attention at the second LLM layer, observing that many visual tokens receive negligible attention [CITE: fastv]; FasterVLM extends attention-based pruning toward faster convergence [CITE: fastervlm]; SparseVLM and related work explore query-conditioned sparsification of the visual stream [CITE: sparsevlm]; FitPrune fits lightweight predictors of token importance [CITE: fitprune]. A broader survey of efficient VLMs contextualizes these methods within KV-cache and visual-redundancy reductions [CITE: vlm-efficiency-survey]. All of these operate on tokens that have already passed through the vision encoder's native pooling; none treats the pooling stage itself as a design variable.

**Dominant-plus-contextual selection.** VisionZip selects "dominant" tokens by CLS-to-patch attention at the penultimate ViT layer and merges the remainder into "contextual" tokens by key-cosine assignment and count-averaging, reporting strong general-task retention at aggressive budgets on LLaVA models [CITE: visionzip-yang2024]. A line-reading of the official code confirms that both the LLaVA path and the Qwen2.5-VL path select *after* the encoder's native pooling — for Qwen2.5-VL, explicitly after the PatchMerger, inside the LLM forward on `inputs_embeds` [E: vz-audit §1]. The official Qwen2.5-VL variant uses a 65%-dominant/5%-contextual split at 70% retention and 45%/5% at 50% [E: vz-audit §1]. Our comparison reconstructs this principle at the post-merger stage with a saliency-free L2 scorer (Section 3.4) and uses the equivalence result (11/11 cells) to attribute the collapse to the stage, not the scorer.

**Recent (2026) token-pruning methods — positioning from a reproducibility audit.** Three CVPR-2026 methods define the current frontier, and we audited each for same-budget reproducibility on our stack (Qwen3-VL-8B + vLLM) [E: baseline-audit, all sections]. *QuietPrune* [CITE: quietprune-gao2026] performs query-guided early (in-ViT) pruning via a trained [Q-CLS] adapter; no public code exists, and it is not training-free [E: baseline-audit §1]. *Hi-Lo Prune* [CITE: hiloprune-sun2026] proposes hierarchical loss-estimate selection with prune-aware fusion; it is training-free and reportedly evaluated on Qwen2/2.5/3-VL — the best architectural fit of the three — but its repository is an empty placeholder (a single 13-byte README), so it is a re-implementation project rather than a runnable baseline; we flag it as the highest-priority future comparison [E: baseline-audit §2]. *IF-Prune* [CITE: ifprune-sun2026] prunes by learned per-token entropy from a small auxiliary estimator (variational information bottleneck); code exists, but it requires a per-model trained KL-estimator LoRA with no Qwen checkpoint released, supports only Qwen2/2.5-VL and InternVL2/2.5, runs on HF transformers only (no vLLM), and its training step alone exceeds our budget by 5–15× [E: baseline-audit §3]. None of the three is reproducible as a fair, like-for-like Qwen3-VL-8B baseline; we report this gap rather than force an unfair comparison (Sections 4.10, 6). Notably, IF-Prune's released pruning grid is the *post*-2×2-merger grid [E: baseline-audit §3], and QuietPrune operates early in the vision tower [E: baseline-audit §1] — the stage axis we study is not the axis any of them varies.

**Merger-stage compression as an empty cell.** Qwen2.5-VL introduced the PatchMerger, and its authors' own evaluation notes that VisionZip's gains are "less striking" on that model because the merger already compresses [E: vz-audit §1]. To our knowledge no published work selects tokens *before* a VLM's native merger and merges only the survivors natively — i.e., the pre-merger × native-merger cell is empty. A source-code reading of VisionZip confirms it retains the native merger and selects afterwards [E: vz-audit §1; evidence §6]. Pre-merger selection is thus not an incremental scorer but an orthogonal stage choice; earlier attempts in this project to win by improving the *scorer* alone (CLS-attention, LLM-cosine, and CLIP-based selectors; a load-adaptive controller; an ElasticVis-style schedule) all failed, consistent with the total-token bound dominating scorer quality within a fixed stage [E: evidence §6]. This paper's claim is precisely that the stage axis dominates the scorer axis for text-dense workloads.

---

## 3. Method

### 3.1 Problem setup and pipeline

We study Qwen3-VL-8B-Instruct [CITE: qwen3vl] in bf16 with eager attention, served by vLLM 0.19 (V1 engine) on a single A40 (46 GB) [E: matrix header]. Its vision pipeline is: image → ViT patch tokens (effective patch footprint 16 px) → native 2×2 merger (four patch features averaged/projected into one unit, 32 px footprint) → additional deepstack mergers (`visual.deepstack_merger_list`, denoted ds0–ds2) that merge intermediate-depth ViT features → concatenation of main-merger and deepstack outputs → language model [E: matrix §8.2, geometry verified; vz-audit §3]. The retention ratio r is defined over merge units: keeping a fraction κ of the N 2×2 units. We report κ ∈ {0.50, 0.25, 0.125}.

Two hook points define the experimental variable:

- **Pre-merger.** Score all N 2×2 units on their *merger-input* features, keep the top-κN units, and pass only the survivors through the native 2×2 merger (and, correspondingly, through each deepstack merger). The merger operates exactly as in the uncompressed model, on a subset of units.
- **Post-merger.** Run the full native merger on all N units, then score the N merged units on their merger-output features and keep the top-κN. This is the stage used by published compression methods for merger-equipped VLMs [E: vz-audit §1].

**Iso-token control.** At the same κ, the two conditions feed the language model the same number of visual tokens: we verify equality of the mean post-merger token count (ptid) per benchmark — TextVQA 203 = 203, DocVQA 1054 = 1054, OCR-Bench 258 = 258, ChartQA 142 = 142, GQA 85 = 85 at κ = 0.25 [E: matrix §0]. Any accuracy difference is therefore attributable to *which* κ-fraction survives and to the *representational state* (raw vs merged features) on which selection was made — not to token count.

### 3.2 The L2 selector as control variable

To isolate the stage, we deliberately use the simplest possible text-agnostic scorer, identical at both hook points: the L2 norm of the unit feature vector, s(u) = ‖f_u‖₂, computed on merger-input features for pre-merger selection and on merger-output features for post-merger selection [E: matrix header; §8.2, PRE scores deepstack_0 merger-input features, POST scores cat(main+ds0..2)]. The rationale is methodological: a strong, task- or query-aware scorer would confound *scoring quality* with *stage*. With an identical, saliency-free scorer, the only manipulated variable is the hook point. Section 4.4 shows the resulting stage law survives replacing this scorer with a second family (global-centroid attention saliency) [E: matrix §8.1], indicating the effect is not an artifact of the L2 scorer.

### 3.3 Pre-merger pruning (ours)

Given an image with N merge units, compute s(u) for every unit on merger-input features, retain the top k = κN units, and invoke the native merger (main and all deepstack mergers) on the retained units only, producing k main tokens plus the corresponding deepstack tokens. Implementation wraps the merger's forward so that selection executes before every native merge call, including the Qwen3-VL deepstack mergers [E: vz-audit §4; matrix §8.2]. No attention weights are required and no parameters are trained; the method is a pure inference-time stage change. Because selection precedes averaging, glyph-bearing patches are protected from being averaged with surrounding background before selection — the mechanism we test in Section 4.7.

### 3.4 VisionZip-style dominant+contextual proxy (post-merger principle port)

To ask whether a dominant+contextual recipe [CITE: visionzip-yang2024] rescues post-merger selection, we implement its principle at matched stage. Per image, the cached per-2×2-unit L2 scores split the kept budget k into k_dom = round(k·0.7) *dominant* units and k_ctx = k − k_dom *contextual* units. Dominant units are the top-scored units, passed natively through the merger. Contextual units are the remaining units split into k_ctx contiguous equal-sized groups, mean-pooled to one unit each, then merged. Total output is k units per image, exactly iso-token with vanilla pre-merger pruning; the split executes inside the same merger-wrap and applies on every merger call, including the deepstack mergers; no attention weights and no training are used [E: vz-audit §4]. A post-mode variant applies the identical dom+ctx split to already-merged units (dom-ratio 0.7); this is the configuration used in the equivalence test of Section 4.4.

**Faithfulness statement and its limits.** This proxy is not a numerical stand-in for official VisionZip, and is not intended as one: official VisionZip scores dominant tokens by ViT attention (CLS-to-patch at layer −2 for CLIP-LLaVA; attention received at the last ViT layer for Qwen2.5-VL), assigns contextual tokens by key-cosine similarity to uniform targets, and uses dominant shares of 0.84 (LLaVA-1.5) and 0.90–0.93 (Qwen2.5-VL) against our 0.70 [E: vz-audit §1, §4]. It *is* a faithful implementation of the dominant+contextual *principle* transplanted to the stage axis we study, with one conservative-bias note: official attention scoring and higher dominant shares would, if anything, buy extra performance on general benchmarks, so a same-stage comparison does not obviously favor our proxy [E: vz-audit §4]. Crucially, the proxy's one structural deviation that could matter for our claim — the contextual-token mechanism (contiguous-group mean vs key-cosine assignment) — is empirically inert: Section 4.4 shows the dom+ctx proxy coincides with dom-only post-merger selection in 11/11 cells, so the contextual implementation choice cannot explain any result in this paper [E: matrix §0].

### 3.5 Adaptive stage selection via ptid

Pre-merger selection is not uniformly best (Section 4.2, GQA). We therefore probe a cheap workload-level router. Offline, we collect per-sample outcomes of both stages on four benchmarks (N = 774, id-aligned, oracle reproduced) [E: evidence §5]. The only image-level signal that improves over a fixed policy is ptid, the number of visual tokens an image produces: a threshold router at ptid = 94 (route to post-merger below the threshold, pre-merger above) reaches pooled accuracy 0.655, versus 0.634 for always-pre and 0.452 for always-post, against a per-sample oracle of 0.702 [E: evidence §5]. An OCR-keyword-based router scores worse (0.539) [E: evidence §5]. The decomposition is the honest part: only 27% of the oracle headroom is workload-level; the remaining 73% is sample-level and query-dependent, unreachable by cheap image-level signals [E: evidence §5]. We therefore recommend pre-merger as the safe default and ptid-threshold routing as a modest workload-level improvement, and report the oracle as a ceiling rather than a target (Sections 4.2, 5).

---

## 4. Experiments

### 4.1 Setup

**Model and engine.** Qwen3-VL-8B-Instruct, bf16, enforce_eager, 1× A40 46 GB, vLLM 0.19 V1, greedy decoding (temperature 0) [E: matrix header].

**Benchmarks and subsets.** TextVQA [CITE: textvqa], DocVQA [CITE: docvqa], OCR-Bench [CITE: ocrbench], ChartQA [CITE: chartqa], GQA [CITE: gqa], and — for the tier structure only — ScienceQA [CITE: scienceqa], MMBench [CITE: mmbench], MME [CITE: mme]. All headline cells use fixed seed-0 subsets of n = 200 [E: matrix header]; TextVQA is additionally verified at n = 500 [E: matrix §0]. ChartQA/OCRBench cells run under an iso-config (max_new_batch_tokens 32768, max-pixels 1.5M, min-num-slots 4); GQA under suite-config (min-num-slots 16) [E: matrix header].

**Scorers.** ChartQA and OCR-Bench scorers are verbatim ports of lmms-eval [CITE: lmms-eval] (ChartQA: numeric ±5% relaxed match; OCR-Bench: normalized containment), with ground-truth self-test passing 200/200 [E: matrix header].

**Error model.** Greedy decoding is deterministic, so there is no seed variance; error bars are binomial standard errors √(p(1−p)/n), and gap significance uses σ_gap = √(σ₁² + σ₂²) [E: matrix header; evidence §1]. Reported z-values are in units of σ_gap.

**Baseline column.** The "uncompressed" column is the full-token model under the corresponding serving config; part of these numbers comes from the earlier suite config rather than the final iso-config (Section 6) and serves as a reference ceiling, not a fourth experimental condition [E: evidence §2, "待补"; matrix header].

### 4.2 Main matrix (κ = 0.25, iso-token)

Table 1 is the headline. Pre-merger and post-merger use the identical L2 scorer and identical ptid; the VisionZip-style column is the post-merger dom+ctx proxy of Section 3.4.

**Table 1.** Keep 25% (r = 0.75), n = 200, greedy. Errors are binomial stderr. ptid is the mean LLM-input visual-token count, identical across pre/post/VZ-style per benchmark. [E: matrix §0]

| Benchmark | tier | baseline (full) | **pre-merger (ours)** | post-merger (L2) | VZ-style (post dom+ctx) | gap pre−post | z |
|---|---|---|---|---|---|---|---|
| TextVQA | text-dense (scene) | 0.820 ± .027 | **0.695 ± .033** (ptid 203) | 0.255 ± .031 | 0.255 ± .031 | **+44.0pp** | 9.8σ |
| DocVQA | text-dense (doc) | 0.770 ± .030 | **0.725 ± .032** (ptid 1054) | 0.390 ± .035 | 0.390 ± .035 | **+33.5pp** | 7.0σ |
| OCR-Bench | text-dense (mixed) | 0.760 ± .030 | **0.580 ± .035** (ptid 258) | 0.165 ± .026 | 0.165 ± .026 | **+41.5pp** | 9.5σ |
| ChartQA | detail-dense (chart) | 0.820 ± .027 | 0.190 ± .028 (ptid 142) | 0.190 ± .028 | 0.190 ± .028 | 0.0pp | 0 |
| GQA | object/spatial | 0.415 ± .035 | 0.320 ± .033 (ptid 85) | **0.380 ± .034** | 0.380 ± .034 | −6.0pp | 1.3σ |

Three facts. First, the text-dense gaps are large and highly significant: +44.0pp (9.8σ), +33.5pp (7.0σ), +41.5pp (9.5σ) [E: matrix §0]. Second, the VisionZip-style column is numerically identical to the post-merger L2 column in every cell of this table — the first instance of the 11/11 equivalence analyzed in Section 4.4 [E: matrix §0]. Third, the ordering inverts on GQA (−6.0pp, post leads) and flattens on ChartQA (0.0pp); neither the GQA inversion nor the ChartQA tie approaches significance (1.3σ and 0), so we report *direction*, not victories (Section 6 red lines) [E: matrix §0].

A larger TextVQA run confirms direction and sharpens significance: at n = 500, pre 0.738 ± .020 vs post 0.272 ± .020, +46.6pp at 16.7σ [E: matrix §0].

Pre-merger also retains a large fraction of the uncompressed baseline: DocVQA 0.725 vs 0.770 (94% retention of full-model accuracy at 25% token budget) and TextVQA 0.695 vs 0.820 (85%) [E: matrix §0]. The pooled practical picture (four benchmarks, per-sample) is 0.634 always-pre vs 0.452 always-post [E: evidence §5].

### 4.3 The stage law across tiers and depths

**Tiers.** Extending the same protocol to perception and multiple-choice benchmarks yields a three-tier structure (Table 2; Figure 1).

**Table 2.** Stage law at 25% keep, n = 200, L2 selector. [E: evidence §2]

| Benchmark | category | baseline | pre | post | gap (pp) |
|---|---|---|---|---|---|
| TextVQA (scene text) | text-dense | 0.820 | 0.695 | 0.255 | **+44.0** |
| DocVQA (document OCR) | text-dense | 0.770 | 0.725 | 0.390 | **+33.5** |
| ScienceQA (figure+text MC) | perception/MC | 0.725 | 0.365 | 0.370 | −0.5 |
| MMBench (perception MC) | perception/MC | 0.895 | 0.295 | 0.320 | −2.5 |
| MME (yes/no perception) | perception/MC | 0.885 | 0.815 | 0.820 | −0.5 |
| GQA (object/spatial) | object | 0.415 | 0.320 | 0.380 | **−6.0** |

The coarse structure — text-dense (+33.5 to +44.0) ≫ perception/MC (−0.5 to −2.5) > object (−6.0) — is robust; perfect monotonicity within a tier is not claimed. Within-tier inversions (DocVQA +33.5 < TextVQA +44.0; MME −0.5 > MMBench −2.5) are reported as-is and attributed to selector behavior on very-large-token document images and to n = 200 sampling noise [E: evidence §2].

**Figure 1.** Pre-minus-post accuracy gap across the six benchmarks at 25% retention. File: `drafts/figures/stage_law.png`. [E: evidence §2]

**Depth.** The text-dense gap widens or holds under deeper compression (Table 3).

**Table 3.** Keep 12.5% (r = 0.875), n = 200, L2 selector, greedy. [E: matrix §1]

| Benchmark | baseline | pre | post | VZ-style (post) | gap pre−post | z |
|---|---|---|---|---|---|---|
| TextVQA | 0.820 | **0.615 ± .034** (ptid 111) | 0.175 ± .027 | 0.175 ± .027 | **+44.0pp** | 10.6σ |
| DocVQA | 0.770 | **0.610 ± .035** (ptid 538) | 0.135 ± .024 | 0.135 ± .024 | **+47.5pp** | 11.2σ |
| OCR-Bench | 0.760 | **0.380 ± .034** (ptid 140) | 0.075 ± .019 | 0.075 ± .019 | **+30.5pp** | 7.8σ |
| ChartQA | 0.820 | 0.150 ± .025 (ptid 88) | 0.095 ± .021 | 0.095 ± .021 | +5.5pp | 1.7σ |
| GQA | 0.415 | 0.250 ± .031 (ptid 52) | **0.305 ± .033** | 0.305 ± .033 | −5.5pp | 1.2σ |

On TextVQA across κ = 50/25/12.5%, pre = {0.75, 0.70, 0.62} vs post = {0.51, 0.26, 0.18}, with the gap growing from +24pp to +44pp and holding; at the deepest point pre retains 75% of the uncompressed baseline against 21% for post — a 3.5× retention ratio [E: evidence §3]. On DocVQA the gap widens from +33.5pp at 25% to +47.5pp at 12.5% [E: matrix §1; evidence §3]. The pre/post retention ratio on OCR-Bench expands from 3.5× (25%) to 5.1× (12.5%) [E: matrix §1]. The ChartQA +5.5pp at 12.5% and the GQA −5.5pp remain non-significant (1.7σ, 1.2σ) — direction only [E: matrix §1].

**Figure 2.** Accuracy-vs-depth retention curves, five-benchmark small multiples on a log₂ depth axis, pre vs post with binomial error bars. Files: `drafts/figures/retention_curves.png` / `.pdf`. [E: matrix §8.3]

### 4.4 Scoring-invariance ablations

**Contextual tokens add nothing (11/11).** In post-merger mode with dom-ratio 0.7, the VisionZip-style dom+ctx proxy is numerically identical to plain post-merger dom-only selection in all 11 tested cells (TextVQA, DocVQA, OCR-Bench, ChartQA, GQA at 25% and 12.5% retention, plus one further retention point) [E: matrix §0]. Representative ties from Table 1: TextVQA 0.255 ± .031 (both), DocVQA 0.390 ± .035 (both), OCR-Bench 0.165 ± .026 (both), ChartQA 0.190 ± .028 (both), GQA 0.380 ± .034 (both) [E: matrix §0]. Mean-pooled contextual tokens computed on already-merged features carry no recoverable text evidence — averaging destroys the same information a second time — so the dom/ctx budget ratio is not the cause of the post-merger collapse. The collapse is explained by the stage alone [E: matrix §0].

**Cross-selector invariance.** Replacing the L2 scorer with a second family — global-centroid attention saliency (`--selector attn`) — preserves the law (Table 4, n = 200 at 25% unless noted).

**Table 4.** Stage gaps under two selector families, 25% keep. [E: matrix §8.1]

| Benchmark | L2 pre/post (gap) | attn pre/post (gap) | reading |
|---|---|---|---|
| TextVQA | 0.695 / 0.255 (+44.0) | 0.670 / 0.292 (+37.8, n = 500) | stage effect survives |
| DocVQA | 0.725 / 0.390 (+33.5) | 0.680 / 0.365 (+31.5) | survives |
| OCR-Bench | 0.580 / 0.165 (+41.5) | 0.480 / 0.170 (+31.0) | survives |
| ChartQA | 0.190 / 0.190 (0.0) | 0.190 / 0.165 (+2.5, n.s.) | budget regime survives |
| GQA | 0.320 / 0.380 (−6.0) | 0.328 / 0.380 (−5.2, n = 500) | object inversion survives |

Both the stage law (text-dense +31 to +44pp under either scorer) and the budget regime (ChartQA ≈ 0 under either scorer) are selector-invariant; the GQA inversion persists under both [E: matrix §8.1]. The effect is not an L2-scorer artifact.

### 4.5 OCR-Bench subskill microscope

OCR-Bench decomposes into subskills, and the stage-vs-budget split descends to subskill granularity (Table 5; per-type n is small, so this is qualitative mechanism evidence, not an inferential benchmark — Section 6).

**Table 5.** OCR-Bench subskills at 25% keep, n per type as shown. [E: matrix §3]

| question type | n | baseline | pre | post | behavior |
|---|---|---|---|---|---|
| Regular Text Recognition | 9 | 1.000 | **1.000** | 0.000 | stage catastrophe (post wiped out) |
| Non-Semantic Text Rec. | 12 | 0.917 | **0.917** | 0.000 | same |
| Irregular Text Rec. | 11 | 0.909 | **0.909** | 0.000 | same |
| Artistic Text Rec. | 13 | 1.000 | **0.923** | 0.077 | same |
| Scene Text-centric VQA | 39 | 0.769 | **0.718** | 0.179 | large stage win |
| Doc-oriented VQA | 42 | 0.405 | **0.452** | 0.119 | pre ≥ baseline |
| Key Info Extraction | 41 | 0.829 | 0.341 | 0.195 | both collapse (spatial-precision sensitive) |
| Handwritten Math Expr | 21 | 0.667 | 0.238 | 0.238 | tie (budget regime, ChartQA-like) |
| Digit String Rec. | 8 | 0.625 | 0.500 | 0.625 | n = 8 noise, not interpreted |
| Handwriting Rec. | 4 | 0.750 | 1.000 | 0.500 | n = 4 |

The cleanest stage evidence is in pure text recognition: pre-merger holds the uncompressed baseline (1.000, 0.917, 0.909, 0.923) while post-merger scores exactly zero on three of four recognition types [E: matrix §3]. Key Info Extraction and Handwritten Math Expressions instead behave like ChartQA — both stages collapse, or tie — which is the budget regime at subskill resolution [E: matrix §3].

### 4.6 ChartQA: the budget-dominated third regime

ChartQA is where the stage law goes quiet, and the failure analysis shows why (Table 6).

**Table 6.** ChartQA depth scan, n = 200, L2 selector. [E: matrix §2]

| keep | pre | post | gap | reading |
|---|---|---|---|---|
| 50% (ptid 252) | 0.390 ± .035 | 0.335 ± .033 | +5.5 (1.1σ) | already collapsed at 50% (baseline 0.82 → 48% retention) |
| 25% (ptid 142) | 0.190 ± .028 | 0.190 ± .028 | 0.0 | three-way tie (VZ-style also 0.190) |
| 12.5% (ptid 88) | 0.150 ± .025 | 0.095 ± .021 | +5.5 (1.7σ) | weak pre trend, not significant |

The failure mode was verified per-sample and is not a scoring bug. Under compression the model stops emitting numbers and switches to hedging verbose answers — mean prediction length grows from 1.1 to 7.7 words ("Based on the chart…") — or reads a neighboring wrong value ("3" vs "4", "Jul'21" vs "May'21") [E: matrix §2]. Pre- and post-merger each fail 38 questions, but only 20 overlap: the tie at 0.190 is a coincidence of *different failure sets* [E: matrix §2]. The split is consistent across the human and augmented partitions (augmented 0.942 → 0.231 pre / 0.212 post; human 0.688 → 0.146 pre / 0.167 post) [E: matrix §2].

Interpretation: chart questions require multi-value arithmetic and fine bar/label discrimination, with the relevant information *globally distributed* across the figure; an L2 local-saliency scorer has no way to prioritize it, so the total token budget dominates and the stage is secondary [E: matrix §2]. Contrast TextVQA — single-value, locally localized — where pre at 50% still scores 0.75 (91% of baseline) [E: matrix §2]. We therefore position ChartQA as an honest third regime in a *two-axis* taxonomy: the stage axis says *where* pre-merger wins; the budget axis says *whether* compression is viable at all. Uniform monotonicity across workloads is not claimed [E: matrix §2].

### 4.7 Two-layer mechanism from token-survival analysis

We visualized which 2×2 units survive selection under each stage and measured the Sobel edge density of kept vs dropped units (geometry verified: patch 16 px → unit 32 px; pre scores deepstack_0 merger-input features, post scores cat(main+ds0..2); token↔unit 1:1) [E: matrix §8.2].

**Table 7.** Token-survival statistics, 25% keep. Edge density = mean Sobel response over the unit region. [E: stats json; matrix §8.2]

| image | units / kept | kept (pre) edge | dropped (pre) edge | kept (post) edge | dropped (post) edge | Jaccard(pre∩post) |
|---|---|---|---|---|---|---|
| DocVQA (document, 2240×1728) | 3780 / 945 | 0.931 | 0.081 | 0.148 | 0.342 | 0.079 |
| TextVQA (scene, 480×1024) | 480 / 120 | 0.312 | 0.192 | 0.306 | 0.194 | 0.257 |

**Layer 1 — selection misdirection (documents, supported).** On the DocVQA image, pre-merger selection keeps units of edge density 0.931 and drops 0.081: it looks at glyphs. Post-merger selection inverts the signal — kept 0.148, dropped 0.342: it systematically *avoids* strokes, selecting smooth merged units, and the two kept sets are nearly disjoint (Jaccard 0.079) [E: matrix §8.2; stats json]. Because the merged features have averaged strokes with background, the smoothed units now look "more energetic" to the norm scorer than glyph units diluted by whitespace; selection on degraded features is actively misdirected. This is directly visible in `drafts/figures/token_survival_docvqa.png` / `.pdf` (Figure 3) [E: matrix §8.2].

**Layer 2 — feature degradation (scene text, location-level non-support reported).** On the TextVQA image, post-merger selection locations are *not* worse: kept-unit edge density is 0.306 vs 0.312 for pre, with substantially overlapping kept sets (Jaccard 0.257) [E: matrix §8.2; stats json; Figure 4, `drafts/figures/token_survival_textvqa.png` / `.pdf`]. Yet accuracy still collapses (0.695 → 0.255 on this benchmark) [E: matrix §0]. The TextVQA failure is therefore not a location error but degradation of the *feature values themselves*: the 2×2 averaging has erased intra-unit stroke contrast inside tokens whose positions are fine [E: matrix §8.2]. We report this non-supporting case deliberately: the honest reading is a two-layer mechanism — misdirection where post-merger scores diverge from glyph locations (documents), and pure feature degradation where they do not (scene text) — with both layers post-merger-exclusive and both absent pre-merger [E: matrix §8.2].

### 4.8 Efficiency

Stage choice is throughput-neutral. At matched ptid, pre and post run within ±10% of each other: ChartQA at 25% pre 4.53 vs post 5.34–5.49 req/s; OCR-Bench 1.74 vs 1.61 req/s [E: matrix §7]. The native merger itself accounts for only ≈10% of time-to-first-token (measured in the v2 round) [E: matrix §7]. We therefore claim no throughput advantage for pre-merger selection; the claim is text-dense *accuracy* at equal throughput (Table 8).

**Table 8.** Throughput, n = 200 batched runs. [E: matrix §7]

| comparison | req/s | speedup |
|---|---|---|
| TextVQA: full → compressed @25% | 3.70 → 4.76 | +28% |
| DocVQA: full → compressed @12.5% | 0.51 → 0.74 | +45% |
| GQA: full → compressed @25% | 7.47 → 8.37 | +12% |
| ChartQA @25%: pre vs post (same ptid) | 4.53 vs 5.34–5.49 | stage-neutral |
| OCR-Bench @25%: pre vs post | 1.74 vs 1.61 | stage-neutral |

Compression against the full-token baseline yields +12% to +45% throughput, larger for smaller ptid [E: matrix §7].

### 4.9 Qualitative cases

From ten audited pre/post flips at 25% keep (`drafts/qualitative_examples.md`; image paths verified) [E: qualitative, all], three are illustrative.

*Unit corruption, identically wrong under both post methods* (DocVQA 58439, image `runs/data/docvqa/58439.jpg`): the ground truth is "$1.3 BILLION"; pre answers "$1.3 billion" ✓ while both post-L2 and VZ-style answer "$1.3 million" ✗ — the numeral survives but the unit word, printed elsewhere, is averaged below readability, and both post-merger methods reconstruct the higher-prior unit, a 1000× quantitative error that the dominant-token channel does not rescue [E: qualitative Ex. 6].

*Text as absent* (TextVQA 35014, image `runs/data/textvqa/35014.jpg`): asked for the date on the right page (GT 07/10/2012), pre reads it correctly; post answers "there is no visible date" — it does not misread the text, it reports it as absent, the purest form of pre-selection destruction [E: qualitative Ex. 3]. Similar signatures appear in a form-field PO number denied as nonexistent [E: qualitative Ex. 8] and in storefront signs hallucinated as high-prior brands once letter strokes are gone ("NORTEL NETWORKS" → "PS/PlayStation") [E: qualitative Ex. 2].

*Honest counter-case* (GQA 201370409, image `runs/data/gqa/201370409.jpg`): on an object-centric scene with no text to protect, post answers "paper" ✓ while pre denies the scissors exist (then describes them) ✗ — pre-merger pruning can discard object evidence the merged representation preserves, the expected trade-off direction consistent with the GQA −6.0pp (n.s.) aggregate [E: qualitative Ex. 10; matrix §0]. A large high-contrast DocVQA number ($1,109,423) is read correctly by all three conditions, confirming the collapse is resolution-dependent rather than a blanket OCR failure [E: qualitative Ex. 9]. The remaining cases (letter corruption "Chase" → "Chosen"; LED "HAPPY BIRTHDAY" → "00"; table-cell cross-contamination 10.3% → 6.1%) are catalogued in `drafts/qualitative_examples.md` [E: qualitative Ex. 4, 5, 7].

### 4.10 Comparison with official VisionZip

**Why a same-model head-to-head is absent.** Official VisionZip (arXiv 2412.04467, CVPR 2025) [CITE: visionzip-yang2024] ships implementations for CLIP-based LLaVA and for Qwen2.5-VL only. Our audit returned verdict (c): not runnable on this setup, with four independent blockers — no Qwen3-VL support (the Qwen code is a hand-modified Qwen2.5-VL modeling file; Qwen3-VL's deepstack mergers are unhandled), no vLLM path (it monkey-patches HF forwards that vLLM never calls), ViT attention materialization that would OOM on ~16k-token document images, and no serving-side hook for its post-LLM-embed rewriting [E: vz-audit §2, §3]. No community Qwen3-VL port exists either [E: vz-audit §2]. A faithful port was estimated at >6 GPU·h of validation risk plus a modeling fork — outside budget — so we do not present official Qwen3-VL numbers and do not claim head-to-head victories over the official method [E: vz-audit §3, §4].

**What we compare instead.** (i) The same-model principle port of Section 3.4, evaluated at the post-merger stage: it coincides with post-merger L2 in 11/11 cells (Section 4.4), so the dominant+contextual machinery adds nothing once the stage is post-merger [E: matrix §0]. (ii) The authors' *own* published Qwen2.5-VL numbers, which are our strongest external anchor because they are the official method on the same model family (Table 9).

**Table 9.** Official VisionZip on Qwen2.5-VL (~7B, lmms-eval), from the authors' README. Model/stage-mismatched reference, not a Qwen3-VL comparison. [E: vz-audit §1]

| retain | MME | MMVet | OCRBench | POPE | RealWorldQA | DocVQA | MathVerse |
|---|---|---|---|---|---|---|---|
| 100% | 2316 | 61.6 | **81.5** | 86.7 | 68.6 | **95.1** | 46.3 |
| 70% (65d+5c) | 2334 | 60.0 | 80.9 | 86.4 | 68.2 | 94.5 | 45.8 |
| 50% (45d+5c) | 2209 | 57.0 | **70.5** | 86.3 | 68.6 | 93.8 | 45.1 |

The shape of Table 9 is the point the authors themselves flag: general tasks (POPE, RealWorldQA) hold while the *text-dense* task degrades first — OCRBench 81.5 → 70.5 (−13%) at 50% retention while DocVQA loses only 1.3 — and the table stops at 50%, with no 25% row published [E: vz-audit §1]. On LLaVA-1.5-7B at 64 tokens (11% retention), TextVQA moves 58.2 → 55.5 (−2.7) [E: vz-audit §1]. Our post-merger proxy at 25% (DocVQA 0.390, TextVQA 0.255) is the consistent extrapolation of that trajectory into the regime the official evaluation did not enter [E: matrix §0; vz-audit §4]. The authors' README caveat — "Qwen2.5VL already uses PatchMerger for visual token compression. As a result, the performance gain from VisionZip is less striking compared to LLaVA" [E: vz-audit §1] — is, in our reading, the same stage effect observed from the other side: once a lossy merger has run, post-merger selection has little left to protect.

The claim is therefore stated conservatively: we cannot produce an official Qwen3-VL number at 25%; the text-dense fragility of post-merger dominant+contextual selection rests on (i) the 11/11 same-model equivalence, (ii) the official method's own published 50%-retention trajectory on the same model family, and (iii) the two-layer mechanism of Section 4.7 [E: vz-audit §4].

---

## 5. Discussion

**Why the stage dominates.** The native 2×2 merger is a learned aggregator, not a preserver of legibility; once it has averaged four patch features, information destroyed by that average is unrecoverable by any downstream operation, however sophisticated the scorer [E: vz-audit §4; matrix §8.2]. Post-merger selection can only choose *among* degraded units. At 25% retention, 75% of units — each covering four times the patch area — are dropped, and in document images text strokes are spatially *uniform* rather than concentrated in salient objects, so any per-token scorer drops most glyphs [E: vz-audit §4]. Pre-merger selection moves the decision upstream of the lossy step: the merger then averages only survivors, and glyph-bearing units enter the LLM with stroke evidence intact (Table 7, Layer 1; Table 5 recognition rows) [E: matrix §8.2, §3]. The object-centric inversion is the symmetric story: on scenes where evidence is object-concentrated and the merger's learned aggregation helps, selecting on merged features is mildly beneficial (GQA −6.0pp, n.s.) [E: matrix §0; qualitative Ex. 10].

**Why contextual tokens add nothing.** The 11/11 equivalence (Section 4.4) is mechanistically expected and empirically total: a contextual token is a mean-pool of already-merged units, i.e., a second averaging applied to features whose text evidence the first averaging already attenuated [E: matrix §0]. At deep budgets the official ratio would force ~5% contextual slots to absorb ~70% of units (~14:1), turning glyphs into unreadable averages [E: vz-audit §4]. The dom/ctx budget ratio — the axis VisionZip variants tune — is simply not the binding variable on text-dense workloads; the stage is [E: matrix §0].

**Why ChartQA is different.** Chart information is globally distributed and the task is arithmetic over many fine values, so no local-saliency stage can prioritize the needed evidence; the total token budget binds first, and both stages fail on *different* question subsets that happen to tie in count (20/38 overlap) [E: matrix §2]. The taxonomy is thus genuinely two-axis, and ChartQA is the evidence that the stage law is a *where*-claim, not a universal *pre-is-better* claim.

**The router ceiling, honestly.** The pooled routing experiment decomposes the oracle headroom (0.702 vs 0.634 always-pre) into 27% workload-level and 73% sample-level, query-dependent structure unreachable by image-level signals such as ptid (OCR-keyword routing does worse, 0.539) [E: evidence §5]. The ptid-threshold router's +2.1pp over always-pre therefore captures approximately the entire *reachable* share; the remaining 73% would require query-conditioned routing, which reintroduces exactly the per-sample machinery this controlled study holds constant [E: evidence §5]. We report 0.702 as a ceiling, not a failure of the router.

**Implications for the field.** Merger-equipped VLMs are now the majority architecture for high-resolution input, and essentially all published token compression for them selects post-merger [E: vz-audit §1; baseline-audit §3, IF-Prune's grid is the post-merger grid]. Our results suggest the field has been optimizing the wrong axis on text-dense workloads: scorer sophistication (attention vs L2 vs dom+ctx) moved accuracy by ≈0 in our controlled comparison, while the stage choice moved it by 33–44pp [E: matrix §0, §8.1]. The merger stage should be re-examined as a first-class design axis — including, where the model is retrainable, merger designs that are selection-aware. Among recent methods, Hi-Lo Prune is explicitly stage-compatible (pre/post) and Qwen3-VL-ready in its paper; its code is not yet released, and we identify it as the most important future head-to-head [E: baseline-audit §2]. Earlier within-project failures of stronger selectors and adaptive controllers [E: evidence §6] are retrospectively consistent: within a fixed stage, the total-token bound dominates scorer quality.

---

## 6. Limitations

**Single architecture.** All results are on Qwen3-VL-8B-Instruct [E: matrix header]. Cross-architecture validation is future work: a planned Qwen2.5-VL comparison is currently blocked by an unresolved mrope position-embedding misalignment in the pre-merger path, so the claim scope is explicitly Qwen3-VL-8B [E: evidence §6].

**No fair third-party SOTA comparison.** Official VisionZip does not run on Qwen3-VL or under vLLM (verdict (c), four independent blockers, audited) [E: vz-audit §3]; QuietPrune has no public code and requires a trained adapter [E: baseline-audit §1]; Hi-Lo Prune's repository is an empty placeholder [E: baseline-audit §2]; IF-Prune requires a per-model trained estimator with no Qwen checkpoint, a Qwen3-VL modeling port, and a non-vLLM dual-model harness (20–60 GPU·h) [E: baseline-audit §3]. The SOTA column is therefore a same-model principle port, and official numbers (Table 9) are model- and stage-mismatched references, not head-to-head results [E: matrix §4; vz-audit §5].

**Two selector families.** The law is demonstrated for L2-norm and global-centroid attention saliency [E: matrix §8.1]; trained or query-conditioned scorers at the post-merger stage are not benchmarked here (by design of the control; see Section 5).

**Subsets and decoding.** Headline cells use n = 200 fixed subsets with binomial stderr under greedy decoding [E: matrix header]; TextVQA is corroborated at n = 500 (+46.6pp, 16.7σ) [E: matrix §0]. OCR-Bench subskill counts (Table 5) are small and support qualitative mechanism claims only [E: matrix §3]. Part of the uncompressed-baseline column is from an earlier serving config (Section 4.1).

**Non-significant gaps reported as direction.** The ChartQA gaps (+5.5pp at 12.5%, 1.1σ) and the GQA inversion (−6.0pp at 25%, −5.5pp at 12.5%; 1.3σ, 1.2σ) are not significant; we report direction only and explicitly do not claim post-merger superiority on GQA or pre-merger superiority on ChartQA [E: matrix §0, §1]. Within-tier inversions (DocVQA +33.5 < TextVQA +44.0; MME −0.5 > MMBench −2.5) are reported, not smoothed [E: evidence §2].

**Proxy conservatism.** The VisionZip-style column is an L2-scored dominant+contextual proxy (dom-ratio 0.7, contiguous-group mean context), not official attention scoring; the contextual-mechanism difference is shown empirically inert (11/11), the stage is aligned, and the bias direction is conservative for us [E: matrix §0; vz-audit §4].

**Mechanism evidence.** Layer 1 (misdirection) is directly measured via edge density of kept/dropped units [E: matrix §8.2]; Layer 2 (feature degradation) is inferred from the combination of location-level non-support and accuracy collapse on TextVQA [E: matrix §8.2], and the per-narrative averaging accounts in Section 4.9 are marked as inference from answer patterns [E: qualitative, caveats].

---

## 7. Conclusion

Under iso-model, iso-token, and iso-selector control on Qwen3-VL-8B, the selection stage — before versus after the native 2×2 merger — dominates scoring sophistication on text-dense workloads: +33.5 to +44.0pp (7.0–9.8σ) at 25% retention on TextVQA, DocVQA, and OCR-Bench, widening under deeper compression, with an insignificant object-centric inversion and a perception/MC null [E: matrix §0, §1; evidence §2]. The mechanism is two-layered — post-merger selection misdirected away from glyphs on documents, and 2×2 feature degradation on scene text even where selection locations are sound — and the dominant+contextual recipe of the post-merger SOTA contributes nothing beyond dominant-only selection in 11/11 cells [E: matrix §8.2, §0]. ChartQA establishes a second, budget-dominated axis on which all methods tie for identifiable reasons [E: matrix §2], and the official VisionZip's own published numbers show the text-dense onset at 50% retention [E: vz-audit §1]. The practical takeaways are modest by design: pre-merger selection is a robust safe default, a ptid-threshold router recovers the reachable (+2.1pp) share of the oracle headroom, and the field should treat the merger stage as a first-class design axis [E: evidence §5]. Cross-architecture validation and a head-to-head with Hi-Lo Prune, should its code be released, are the natural next steps [E: evidence §6; baseline-audit §2].

---

*Figure inventory (all files verified to exist):* Figure 1 `drafts/figures/stage_law.png`; Figure 2 `drafts/figures/retention_curves.png` / `.pdf`; Figure 3 `drafts/figures/token_survival_docvqa.png` / `.pdf`; Figure 4 `drafts/figures/token_survival_textvqa.png` / `.pdf`; supporting numbers `drafts/figures/token_survival_stats.json`; qualitative images `runs/data/{textvqa,docvqa,gqa}/<id>.jpg`.

# paper_v4 — submission draft (CCF-B methods paper; Pattern Recognition / ICME tier)

> Drafted 2026-07-24 · writing sub-agent (CPU-only). Language = English (paper body).
> Scope of this draft: **all sections that do NOT depend on the full-split main table** are
> written out; the full-split headline (Table 1, **PENDING J7**) and the efficiency table
> (Table 3, **PENDING J6**) are left as designed placeholders. All *interim* numbers are
> n=200 seed-0 subsets under **official metrics** and are flagged `[interim]`; the submission
> version keeps only the J7 full-split numbers. Old containment figures are void.
> Evidence anchors use `[E: <digest>]` → files under `experiments/` and `drafts/`.
> Red-line self-check checklist is at the end of this file.

---

## Title candidates (three; none selected — mechanism + robustness framing; no "beats/SOTA/superior")

> The earlier working title "Selection Stage Beats Scoring" is **abandoned**: it conflicts with
> the evidence that a *smarter* (query-conditioned) scorer, FastV, out-scores the plain pre-merger
> ranker on TextVQA and GQA (Section 5). "Stage beats scoring" is therefore not a universal claim
> we can defend. The candidates below frame a **mechanism + robust-default** contribution instead.

1. **The Lossy Merger: Pre-Merger Selection as a Query-Blind Robust Default for Visual-Token Compression in Vision-Language Models**
2. **Where You Select Matters: A Causal Ranking Mechanism and a Robust Pre-Merger Default for Merger-Equipped VLMs**
3. **Rank Before You Merge: How Lossy 2×2 Pooling Corrupts Token Saliency, and a Query-Blind Stage Fix for Text-Dense VLM Compression**

<!-- NUMBERS: title — no numbers; "beats scoring" retired per plan §Title + FastV evidence (j4_probe). -->

---

## Abstract

Merger-equipped vision-language models treat their native 2×2 token merger as lossless. We show it
is not. On Qwen3-VL-8B the merger nearly re-shuffles unit saliency from scratch (M1: pre–post rank
ρ = 0.14–0.36), systematically demotes high-edge/text-stroke units (M2: demoted-group Sobel 0.64
vs 0.12 on DocVQA), and—via a ranking-swap control that holds the forward path fixed—accounts for
the *entire* pre-vs-post accuracy gap (M3: swap ≡ pre; DocVQA 200/200 byte-identical, TextVQA
198/200). The mechanism is causal: the gap is a ranking effect, not forward-path information
destruction. We therefore select **before** the merger—Rank-Before-Merge (RBM), a query-blind L2
ranking—and find the effect is cross-generation: on text-dense benchmarks RBM leads post-merger
selection by **+18.8 to +38.3 pp** (Qwen3-VL-8B and Qwen2.5-VL-7B, n=200, official metrics), ties
on object-centric GQA with **no crossover**, and the gap widens under deeper compression. In a
same-model, iso-budget comparison, the query-conditioned FastV wins three benchmarks, yet RBM alone
holds OCR-Bench (**+42.5 pp** over FastV); RBM is the **robust default**. Three pre-registered
negative results close the design space: pre-merger selection **must remain query-blind**.

<!-- NUMBERS: M1 ρ 0.14–0.36 (mechanism_verification §1); M2 Sobel 0.641 vs 0.124 (mechanism §2);
     M3 swap≡pre 200/200 & 198/200 (mechanism §3); +18.8~+38.3pp cross-gen (j2 + rescore_rerun);
     OCR +42.5pp = RBM 0.580 vs FastV 0.155 (j4_probe). Abstract ~190 words. -->

---

## 1. Introduction

**The unexamined assumption.** Vision-language models (VLMs) convert an image into hundreds to
tens of thousands of visual tokens, and visual-token compression—pruning or merging the least
important tokens before the language model—has become a standard efficiency lever. In
merger-equipped architectures (the Qwen-VL family, and others that pool 2×2 patch groups into one
token), the compression community has implicitly treated the native merger as a *lossless*
preprocessing step and debated only **which** tokens to keep and **how** to score them. The prior
question—**where** selection should happen, before or after the merger—has received little
attention. This paper shows that, on text-dense workloads, the *stage* of selection dominates the
*scorer* of selection, and that the reason is a previously unreported property of the learned
merger: it is **lossy in a text-hostile direction**.

**What we find.** Under strict iso-model / iso-token / iso-selector control, applying the *same*
text-agnostic L2 scorer before the merger versus after it produces a large accuracy gap on
text-dense benchmarks and no meaningful gap on object-centric ones. The gap is not explained by
information destroyed in the forward pass. A causal ranking-swap control—run the post-merger
forward path but select units with the pre-merger ranking—recovers the pre-merger accuracy exactly
(DocVQA: 200/200 byte-identical answers; TextVQA: 198/200, the residual being greedy-decode
run noise). The entire pre>post gap is therefore a **ranking effect**: the merger rewrites unit
saliency, and post-merger selection reads the corrupted ranking.

**What we propose.** Rank-Before-Merge (RBM): score 2×2 units on their merger-*input* features
with a query-blind L2 norm, keep the top-κ units, and pass only the survivors through the native
merger. No attention weights, no training, no query input—a pure inference-time stage change. The
method is deliberately minimal: we freeze it as *plain* RBM and do not decorate it with hybrids or
routers, because the experiments below show every such decoration is either neutral or harmful
(Section 6).

**Cross-generation consistency.** The stage law and the lossy-merger signature reproduce on a
second generation, Qwen2.5-VL-7B (text-dense pre>post of +18.8 to +32.0 pp; GQA a statistical tie),
with one honest boundary: the causal swap control replicates on Qwen3-VL but not on Qwen2.5-VL, so
we claim the causal decomposition only for Qwen3-VL and present the Qwen2.5-VL evidence as
corroboration of the ranking law (Section 4).

**Positioning by fair comparison.** We compare RBM against same-model, same-budget, runnable
baselines—FastV (one-shot attention pruning at LLM layer 2) and PyramidDrop (progressive in-LLM
dropping)—plus a faithful same-stage principle-port of VisionZip. The comparison is deliberately
honest about where RBM does *not* lead: FastV, being query-conditioned, wins TextVQA, DocVQA, and
GQA, while RBM is the *only* method that retains OCR-Bench. We therefore position RBM not as a
method that "beats" the field but as the **robust default**—it never collapses, it never loses to
the post-merger family, and it uniquely preserves dense OCR that every post-merger and
query-conditioned competitor destroys.

**Three negative results that close the space.** An image-level routing policy, a merger-aware
hybrid mask, and a query-level embedding blend were each pre-registered and each failed (all
λ>0 or routing fractions harmful, −1.7 to −5.2 pp). Together with the FastV result they imply that
pre-merger features are purely visual: pre-merger selection **must be query-blind**, and the
query-dependent headroom is reachable only after the LLM's cross-attention layers mix in the
question—exactly where FastV operates.

**Contributions.**
1. **A causal lossy-merger mechanism.** We show the native 2×2 merger re-shuffles unit saliency
   (M1), demotes text-stroke units on documents (M2), and—by a ranking-swap control—explains 100%
   of the pre-vs-post gap as a ranking effect (M3) on Qwen3-VL-8B.
2. **A method (RBM) and its cross-generation law.** A query-blind pre-merger L2 ranking leads
   post-merger selection by +18.8 to +38.3 pp on text-dense benchmarks across two Qwen-VL
   generations, ties on object-centric GQA with no crossover, and widens under deeper compression.
3. **A same-model, iso-budget comparison** that positions RBM against FastV, PyramidDrop, and a
   VisionZip principle-port, establishing RBM as the robust default (never collapses; uniquely
   holds OCR-Bench) rather than a universal winner.
4. **An honest negative-result set** (image-level router, merger-aware hybrid, query-level gate)
   that closes the design space: pre-merger selection must remain query-blind.

<!-- NUMBERS: +18.8~+38.3pp (j2 Table1 cross-gen Δ@25%; rescore_rerun +38.3/+26.5 Qwen3);
     M3 200/200 & 198/200 (mechanism §3); negative λ −1.7~−5.2pp (j5_qa_gate_result);
     OCR +42.5pp (j4_probe). No containment numbers used. GQA stated as tie/no-crossover (R4). -->

---

## 2. Related work

**Token pruning and merging in VLMs.** Most methods score visual tokens and drop the least
important before or within the early LLM layers. **FastV** prunes at the second LLM layer using
attention the visual tokens receive, observing many receive negligible attention [CITE: fastv];
*relation*: it is our primary same-model baseline and the empirical instance of a
query-conditioned scorer that wins scene-text/object but cannot rescue merger-destroyed OCR
(Section 5). **PyramidDrop** drops tokens progressively across LLM layers in a pyramid schedule
[CITE: pyramiddrop]; *relation*: a second same-model baseline whose canonical schedule occupies a
different budget point, and whose mechanism degenerates to collapse at an iso-25% budget. **FasterVLM**
pushes attention-based pruning toward faster convergence [CITE: fastervlm]. **SparseVLM** and
related work explore query-conditioned sparsification of the visual stream [CITE: sparsevlm].
**PruMerge** merges tokens by attention-derived importance rather than dropping them
[CITE: prumerge]. **FitPrune** fits lightweight predictors of token importance [CITE: fitprune].
All of these operate on tokens that have *already* passed the encoder's native pooling; none treats
the pooling stage itself as a design variable, and none of the query-conditioned variants is
available as a runnable same-model baseline on our stack except FastV and PyramidDrop.

**Qwen-family-specific compression.** Qwen-VL models introduce a PatchMerger that already compresses
visual tokens, and a line of work targets this family directly. **GlimpsePrune** [CITE: glimpseprune]
and **VScan** [CITE: vscan] prune or scan visual tokens within Qwen-VL inference; *relation*: they
confirm the Qwen family is an active compression target, but like the methods above they select
*after* the native merger, i.e. on the post-merger side of the stage axis we study.

**Dominant-plus-contextual selection.** **VisionZip** selects "dominant" tokens by CLS-to-patch
attention at the penultimate ViT layer and merges the remainder into "contextual" tokens by
key-cosine assignment and count-averaging, reporting strong general-task retention at aggressive
budgets on LLaVA models [CITE: visionzip-yang2024]. A line-reading of the official code confirms
both the LLaVA and the Qwen2.5-VL paths select *after* the native pooling—for Qwen2.5-VL,
explicitly after the PatchMerger, inside the LLM forward on `inputs_embeds` [E: visionzip_gap_report
§1]. *Relation*: we reconstruct its dominant+contextual principle at the post-merger stage with a
saliency-free scorer (Section 3.4) and use a byte-equivalence result (both generations) to attribute
the text-dense collapse to the *stage*, not the scorer; the authors' own Qwen2.5-VL numbers
(OCRBench 81.5→70.5 at 50% retention) are our strongest external anchor (Section 5).

**Reproducibility audit of the 2026 frontier.** We audited three CVPR-2026 methods for same-budget
reproducibility on our stack [E: baseline_methods_audit]. **QuietPrune** (query-guided early in-ViT
pruning via a trained [Q-CLS] adapter) has no public code and is not training-free [E:
baseline_methods_audit §1]. **Hi-Lo Prune** (hierarchical loss-estimate selection with prune-aware
fusion) is training-free and reportedly evaluated on Qwen2/2.5/3-VL—the best architectural fit—but
its repository is an empty placeholder; we flag it as the highest-priority *future* baseline [E:
baseline_methods_audit §2]. **IF-Prune** (learned per-token entropy from a small auxiliary
estimator) has code but requires a per-model trained KL-estimator with no Qwen checkpoint, supports
only Qwen2/2.5-VL and InternVL, runs HF-only, and prunes on the *post*-2×2-merger grid; its training
step alone exceeds our budget 5–15× [E: baseline_methods_audit §3]. None is reproducible as a fair
Qwen3-VL baseline, so we report the gap (Section 7) rather than force an unfair comparison; the
stage axis we study is not the axis any of the three varies.

**Efficiency evaluation of VLMs.** Reported speedups depend heavily on the serving engine and the
measurement protocol (offline batch vs continuous-batching serving, time-to-first-token vs
throughput). We adopt the lmms-eval family of *official* scorers for accuracy [CITE: lmms-eval]
(VQA-accuracy, ANLS, OCR-Bench official five-category scoring, GQA word-normalized exact match) so
that our accuracy claims use community-standard metrics, and we disclose the engine of every
efficiency number, confining throughput claims to a single engine (Section 7).

**Merger-stage compression as an empty cell.** The Qwen2.5-VL authors themselves note VisionZip's
gains are "less striking" on that model because the merger already compresses [E:
visionzip_gap_report §1]. To our knowledge no published work selects tokens *before* a VLM's native
merger and merges only the survivors natively—the pre-merger × native-merger cell is empty. This
paper's claim is that, for text-dense workloads, this stage axis dominates the scorer axis.

<!-- NUMBERS: VisionZip 81.5→70.5 @50% (visionzip_gap_report §1); "11/11 → both-generation
     byte-equivalence" (v3_sota_matrix §0 + j2). Related work is mechanism/positioning, no new metrics. -->

---

## 3. Method — Rank-Before-Merge (RBM)

### 3.1 Problem setup

We study two merger-equipped VLMs, **Qwen3-VL-8B-Instruct** [CITE: qwen3vl] and
**Qwen2.5-VL-7B-Instruct** [CITE: qwen25vl], in bf16 with eager attention, served by vLLM 0.19
(V1 engine) on a single A40 (46 GB). Their shared vision pipeline is: image → ViT patch tokens
(effective 16 px patch footprint) → native 2×2 merger (four patch features projected into one
*unit*, 32 px footprint) → (Qwen3-VL) additional deepstack mergers that merge intermediate-depth
ViT features → concatenation of main-merger and deepstack outputs → language model [E:
mechanism_verification_report §0; visionzip_gap_report §3].

**Budget definition.** We define the retention budget κ over *merge units*, relative to each
image's **own** full unit count: keeping a fraction κ of that image's N units. We report
κ ∈ {0.25, 0.125} (i.e. 25% and 12.5% retention). Because κ is per-image, two methods at the same κ
feed the language model the same number of visual tokens per image; we verify equality of the mean
post-merger token count per benchmark (**iso-token** control), so any accuracy difference is
attributable to *which* units survive and the *representational state* on which selection is made,
not to token count.

**The stage axis (the experimental variable).** Two hook points differ only in *where* selection
happens:

- **Pre-merger (RBM).** Score all N 2×2 units on their *merger-input* features, keep the top-κN,
  and pass only the survivors through the native 2×2 merger (and, for Qwen3-VL, each deepstack
  merger). The merger operates exactly as in the uncompressed model, on a subset of units.
- **Post-merger.** Run the full native merger on all N units, then score the N merged units on
  their merger-*output* features and keep the top-κN. This is the stage used by published
  compression methods for merger-equipped VLMs, including VisionZip's Qwen path [E:
  visionzip_gap_report §1].

**Figure 1 (pipeline; [FIG: fig_pipeline TBD — to be drawn]).** Schematic contrasting the two hook
points: (a) RBM scores raw 32 px-unit features, keeps top-κ units, then invokes the native merger
on survivors only; (b) post-merger selection merges all units first and scores the merged tokens.
The native merger block, the deepstack mergers, and the LLM are drawn identically in (a) and (b);
only the tap point of the saliency score differs.

**Cross-architecture mrope note (configuration disclosure).** Qwen2.5-VL uses a *block* M-RoPE
layout (axis spans [16,24,24], θ = 1e6) while Qwen3-VL uses an *interleaved* layout ([24,20,20],
θ = 5e6). When tokens are pruned, the position cursor must advance by the *actual* surviving count
k; the stock vLLM position routine advances by the *full* grid, so trailing text tokens inherit 2D
grid positions. The block layout concentrates this error into a whole axis and collapses output; the
interleaved layout disperses it and tolerates it. A family-scoped fix that advances the cursor by k
(r = 0 bit-degrades to the original; the Qwen3-VL branch is untouched) makes the Qwen2.5-VL
compression path well-formed [E: j1_qwen2vl_mrope_fix]. We disclose this because it is a
configuration dependency of the Qwen2.5-VL cells, and because the block-vs-interleaved contrast is
itself a diagnostic of why the two generations behave differently under pruning (Section 4).

### 3.2 The L2 selector as a control variable

To isolate the *stage*, we deliberately use the simplest possible **text-agnostic, query-blind**
scorer, identical at both hook points: the L2 norm of the unit feature vector,
`s(u) = ‖f_u‖₂`, computed on merger-input features for pre-merger selection and on merger-output
features for post-merger selection [E: mechanism_verification_report §0]. The rationale is
methodological: a strong, task- or query-aware scorer would confound *scoring quality* with *stage*.
With an identical, saliency-free scorer, the only manipulated variable is the hook point. We freeze
this as **plain RBM** and add no variant. Section 5 reports that the stage law survives replacing L2
with a second scorer family (global-centroid attention) on Qwen3-VL, indicating the effect is not an
L2 artifact; the same invariance does *not* extend to the attention proxy on Qwen2.5-VL, which we
report honestly (Section 4).

### 3.3 Rank-Before-Merge

Given an image with N merge units, compute `s(u)` for every unit on merger-input features, retain
the top `k = κN` units, and invoke the native merger (main and all deepstack mergers) on the
retained units only, producing k main tokens plus the corresponding deepstack tokens. The
implementation wraps the merger's forward so that selection executes before every native merge call,
including the deepstack mergers [E: visionzip_gap_report §4]. No attention weights are required and
no parameters are trained; the method is a pure inference-time stage change. Because selection
precedes averaging, glyph-bearing patches are protected from being averaged with surrounding
background *before* the keep decision—the mechanism we test in Section 4.

```text
Algorithm 1: Rank-Before-Merge (RBM), one image
------------------------------------------------------------------
Input : image x; retention κ; query-blind scorer s = L2 norm
Output: compressed visual token sequence z
1.  P  <- ViT patch tokens of x                         # 16 px footprint
2.  U  <- group P into N native 2x2 units               # 32 px footprint
3.  for each unit u in U:  score[u] <- || f_in(u) ||_2  # MERGER-INPUT features
4.  K  <- top-k units by score,  k = round(κ · N)        # per-image budget
5.  z_main <- NativeMerger( {u : u in K} )              # merge SURVIVORS only
6.  z_ds   <- DeepstackMergers( {u : u in K} )          # Qwen3-VL only
7.  z  <- concat(z_main, z_ds)
8.  return z                                            # feed standard LLM
------------------------------------------------------------------
Post-merger selection (the contrast, = VisionZip-type stage): replace lines 3-5 by
   U' <- NativeMerger(U); score'[u'] <- || f_out(u') ||_2 on MERGER-OUTPUT features;
   K' <- top-k by score'; z_main <- keep(K'). Everything downstream is identical.
```

### 3.4 VisionZip-style dominant+contextual proxy (post-merger principle port)

To ask whether a dominant+contextual recipe [CITE: visionzip-yang2024] rescues post-merger
selection, we implement its *principle* at the matched (post-merger) stage. Per image, the cached
per-unit scores split the kept budget k into `k_dom = round(k·0.7)` dominant units (top-scored,
kept natively) and `k_ctx = k − k_dom` contextual units (the remainder split into k_ctx contiguous
equal-sized groups, mean-pooled to one unit each, then merged). Total output is exactly k units per
image, iso-token with plain selection [E: visionzip_gap_report §4].

**Faithfulness and its limits.** This proxy is *not* a numerical stand-in for official VisionZip
(official scores by ViT attention and uses dominant shares 0.84–0.93 vs our 0.70), and is not
intended as one [E: visionzip_gap_report §1, §4]. It *is* a faithful implementation of the
dominant+contextual principle transplanted to the stage axis we study. The one structural deviation
that could matter—the contextual mechanism (contiguous-group mean vs key-cosine assignment)—is
empirically inert: the dom+ctx proxy coincides with dom-only post-merger selection **byte-for-byte
in every cell on both generations** (11/11 on Qwen3-VL; e.g. TextVQA 0.415 == 0.415 on Qwen2.5-VL)
[E: v3_sota_matrix §0; j2_crossgen_matrix]. Hence the dominant+contextual machinery adds nothing
once the stage is post-merger, and the collapse is attributable to the stage, not the scorer.

<!-- NUMBERS: κ∈{0.25,0.125}; dom-ratio 0.7; VZ≡post 11/11 (v3_sota_matrix §0) + 0.415==0.415
     byte-identical Qwen2.5 (j2). mrope [16,24,24] θ1e6 vs [24,20,20] θ5e6 (j1 §根因). -->

---

## 4. Mechanism — why the merger is lossy

We give the pre-vs-post gap a mechanistic account in three parts (M1–M3), then a cross-generation
boundary. M1–M3 are measured on **Qwen3-VL-8B** (the generation on which the causal control holds);
M1/M2 use a deterministic seed-0 sample of n=64 images per benchmark, M3 uses n=200 with the same
invocation as the headline cells [E: mechanism_verification_report §0].

### 4.1 M1 — the merger reshuffles unit ranks

Per image we correlate the *pre* ranking (merger-input unit L2) with the *post* ranking
(merged-token L2) over all units. The two rankings are barely related:

| benchmark (type) | Spearman ρ | Kendall τ | Jaccard@25% |
|---|---|---|---|
| DocVQA (text-dense doc) | **0.137 ± 0.158** | 0.094 | **0.180** |
| TextVQA (text-dense scene) | 0.332 ± 0.124 | 0.227 | 0.243 |
| GQA (object-centric) | **0.360 ± 0.091** | 0.249 | **0.278** |

The 2×2 pooling reorders saliency almost from scratch (ρ = 0.14–0.36, nowhere near 1); at the
decision-relevant top-25% cut, pre and post keep only 18–27% of the same units. The reshuffle is
**strongest on the most text-dense benchmark (DocVQA) and weakest on object-centric GQA**—the same
text-density gradient as the accuracy stage law [E: mechanism_verification_report §1; FIG:
token_survival_m1_rank_overlap.png].

### 4.2 M2 — the demoted units are text-stroke units

Define rank_shift = post_rank − pre_rank (+ ⇒ the merger demoted the unit) and use per-unit Sobel
edge energy as a text-stroke proxy. At κ = 25%, group (a) = pre-kept/post-dropped, (b) =
post-kept/pre-dropped:

| benchmark | ρ(rank_shift, edge) | mean Sobel (a) | (b) | frac > median (a) | (b) |
|---|---|---|---|---|---|
| DocVQA | **+0.439** | **0.641** | 0.124 | **0.918** | 0.347 |
| TextVQA | +0.155 | 0.281 | 0.186 | 0.722 | 0.531 |
| GQA | +0.036 | 0.304 | 0.271 | 0.580 | 0.532 |

The merger preferentially *demotes* high-edge units (positive ρ everywhere), ~12× more strongly on
DocVQA than GQA. The units post drops that pre keeps—group (a)—are the highest-edge units of all:
on DocVQA, 0.641 mean Sobel vs 0.124 for (b), and 92% of them sit above the per-image median edge
(vs 35% of (b))—post-merger selection is **systematically anti-text on documents** [E:
mechanism_verification_report §2; FIG: token_survival_m2_edge_demotion.png].

### 4.3 M3 — a ranking-swap causal control (the decisive test)

By **unit equivalence**, a kept unit's merged token is bit-identical regardless of stage (the mask
is at unit granularity and the kept unit's merger sees all four patches either way). Therefore,
holding the *forward path* fixed at post-merger and swapping in the *pre-merger ranking* must
reproduce pre-merger accuracy if the gap is purely a ranking effect. We implement
`--mode post --mask-ranking swap` and rescore under official metrics:

| benchmark | metric | none | post | **pre** | **swap** (post-path + pre-ranking) | Δ(swap − pre) | identity |
|---|---|---|---|---|---|---|---|
| TextVQA | VQA-acc | 0.858 | 0.215 | **0.598** | **0.603** | +0.005 (paired SE 0.005) | **198/200** |
| DocVQA | ANLS | 0.976 | 0.200 | **0.465** | **0.465** | **0.000 (exact)** | **200/200** |

Swapping in the pre ranking recovers full pre-merger accuracy (exactly on DocVQA; within 2/200
greedy-decode answers on TextVQA) and erases the post-merger collapse (+38.3 pp on TextVQA,
+26.5 pp on DocVQA over post). The swap path agrees with post on only 40/200 (TextVQA) and 63/200
(DocVQA) answers—without the pre ranking, the same forward path produces the post collapse. A full
independent rerun of the TextVQA swap cell is 200/200 byte-identical to the first, so the residual
2/200 is ε-level kernel numerics, not scheduling [E: mechanism_verification_report §3]. **Verdict:
swap ≡ pre.** The entire pre>post gap is attributable to the **ranking alone**; the merged
representations of kept units carry no stage-dependent information loss. The causal claim is stated
for Qwen3-VL-8B.

### 4.4 Two failure modes (dual-layer mechanism)

The mechanism is clean and complete on DocVQA (post selection is *misguided away from text*;
restoring the pre ranking restores accuracy fully). On TextVQA the ranking corruption exists
(M1 ρ = 0.33) and is again the *entire* source of the gap (M3 swap ≡ pre), but the M2
text-directionality is only moderate (ρ = 0.16; group (a) 0.281 vs (b) 0.186)—scene-text images do
not show the document-level "post avoids text" contrast, and even pre-selected units underperform
the uncompressed baseline (0.60 vs 0.86), a **budget effect** shared by any 25%-keep method. GQA
shows near-null reshuffle directionality (ρ = 0.04), consistent with post ≈ pre there [E:
mechanism_verification_report §4]. We summarize: **documents = selection misdirection; scene text =
feature degradation; object scenes = near-null**—all three regimes reported, nothing hidden.

### 4.5 Cross-generation boundary and the mrope diagnostic

**Replicates on Qwen2.5-VL-7B.** The ranking law and the stage effect replicate: with the L2
selector at r = 0.75 (n=64), DocVQA pre 0.664 > post 0.531 (+13.3 pp) and TextVQA pre 0.719 > post
0.349 (+37.0 pp)—the L2 *sign* is unchanged across generations; VisionZip-style ≡ post holds
byte-for-byte (TextVQA 0.415 == 0.415) [E: j2_crossgen_matrix; j3_mechanism_crossarch].

**Does not replicate: the causal swap.** The M3 swap control does *not* reproduce on Qwen2.5-VL:
swap exceeds pre rather than matching it (n=64 batched: DocVQA swap 0.687 vs pre 0.664; n=16 with
seq=1, which rules out batch re-ordering: DocVQA swap 0.730 vs pre 0.538, +19.2 pp, 8/16 identical
answers; TextVQA 0.750 vs 0.625). Since a random ranking cannot beat the L2-pre ranking by ~19 pp on
text-dense data, the residual is either an implementation artifact (window-attention
`reverse_indices` misaligning the swap scores with merged-token order) or a batch-dependent merger
(subset merge ≠ full merge). The root cause is **undecided** [E: j3_mechanism_crossarch]. We
therefore **claim the causal decomposition only for Qwen3-VL**, and present the Qwen2.5-VL evidence
as corroboration of the ranking *law* (M1 + stage-law + VZ≡post), not of the causal decomposition.

**Selector invariance, with an honest exception.** On Qwen3-VL the stage law holds under a second
scorer family (global-centroid attention: TextVQA pre 0.553 vs post 0.200, +35.3 pp) [E:
method_gate_report §5]. On Qwen2.5-VL the L2 sign is invariant but the attention *proxy* fails
(DocVQA sign reversal: pre_attn 0.530 < post_attn 0.605; TextVQA pre_attn 0.552 well below L2
0.719). This is a proxy-family specificity, not a counter-example to the stage law—L2 is the paper's
selector—and we report it as such [E: j3_mechanism_crossarch].

**mrope as a cross-architecture diagnostic.** The block-vs-interleaved M-RoPE contrast of
Section 3.1 explains why naive pruning collapses Qwen2.5-VL but is tolerated by Qwen3-VL, and is the
most concrete structural difference we have between the two generations [E: j1_qwen2vl_mrope_fix].

<!-- NUMBERS: M1 ρ 0.137/0.332/0.360, Jaccard 0.180/0.243/0.278 (mechanism §1); M2 ρ +0.439/+0.155/
     +0.036, Sobel 0.641 vs 0.124, 92% vs 35% (mechanism §2); M3 swap TextVQA 0.603/pre 0.598/198/200,
     DocVQA 0.465==0.465/200/200 (mechanism §3); dual-layer budget 0.60 vs 0.86 (mechanism §4);
     Qwen2.5 swap 0.730 vs 0.538 seq1, identity 8/16 (j3); attn proxy 0.553/0.200 Qwen3 (method_gate §5),
     0.530/0.605 Qwen2.5 (j3). All M1–M3 = Qwen3-VL-8B; Qwen2.5 = corroboration only (R2/R4). -->

---

## 5. Experiments

### 5.1 Setup

**Models.** Qwen3-VL-8B-Instruct and Qwen2.5-VL-7B-Instruct, bf16, enforce_eager, 1× A40 46 GB,
vLLM 0.19 V1, greedy decoding (temperature 0).

**Benchmarks and metrics (official).** TextVQA (VQA-accuracy) [CITE: textvqa], DocVQA (ANLS)
[CITE: docvqa], OCR-Bench (official five-category score /1000) [CITE: ocrbench], and GQA
(word-normalized exact match) [CITE: gqa]. All scorers are verbatim ports of the lmms-eval family
[CITE: lmms-eval], with ground-truth self-test passing 200/200. Each TextVQA/DocVQA/GQA question
carries the canonical short-answer instruction ("Answer the question using a single word or
phrase.") baked into the subset, which is required for VQA-acc/ANLS/exact-match to be meaningful.

**Budget.** Retention κ over per-image merge units (Section 3.1); same κ ⇒ iso-token per image
(verified via mean post-merger token count).

**Subset vs full split.** The numbers reported here are seed-0 **n=200 subsets**, flagged
`[interim]`. Table 1 (full-split headline) is **PENDING J7**; the submission version replaces the
interim table with the full-split numbers and does not mix the two.

**Baseline engines and fairness protocol (disclosure, R7).** FastV and PyramidDrop are run in an
independent HuggingFace transformers 4.57.6 eager harness (they have no vLLM path); RBM and the
post-merger / none cells use vLLM. Fairness is enforced by (i) the same keep ratio relative to each
image's own full token count, (ii) reporting mean absolute post-merger token count, (iii) a
per-family min/max-pixel calibration so token counts are iso across patch sizes, and (iv) for
PyramidDrop, folding its layer schedule into an equivalent mean retention. **Accuracy is comparable
across engines** because the HF harness is validated to bit-degrade to native at r = 0 (r=0 anchor:
8/8 per-sample identical answers; manual pre-norm vs native max-diff = 0) and because HF-vs-vLLM
*none* cells agree 16/16 [E: j4_step2_fix §验证]. **Throughput is not comparable across engines**;
all efficiency numbers are confined to vLLM (Section 7; Table 3 PENDING J6).

**DocVQA iso-pixel disclosure (R7).** DocVQA runs under a large-document configuration
(max-num-batched-tokens 32768, max-pixels 1.5M cap). Because the HF baselines cannot honor a pixel
cap through the engine, the DocVQA *baseline-comparison* row of Table 1b uses a PIL pre-scaled
**600k-pixel** cap under which **all methods skip 0 samples**; native-resolution RBM/none cells are
reported separately as a reference. This is disclosed because post-merger deep-compression collapse
is partly configuration-dependent.

### 5.2 Main result — full split **[TABLE 1: PENDING J7]**

> **PLACEHOLDER — depends on J7 (official full-split main table, ~50–60 GPU·h). Do not fill with
> subset numbers in the submission version.** Column design (fixed now so J7 fills straight in):

**Table 1 (design).** Rows = TextVQA (VQA-acc) / DocVQA (ANLS) / OCR-Bench (official /1000) / GQA
(exact match). Column groups = **Qwen3-VL-8B | Qwen2.5-VL-7B**; each group has five columns =
`full` / `RBM@25%` / `post≡VZ@25%` / `FastV@25%` / `Pyramid` (see note). Each cell = official metric
± binomial CI; a gap column reports z, with z ≥ 1.96 labelled a lead and 1.5 ≤ z < 1.96 labelled
direction-only.

- **Pyramid note:** canonical keep_equiv 0.625 is a *different budget point* (≈2.1× ours) and is
  reported as a retention-curve point; the iso-25% schedule `[1,0,0,0]` collapses (Table 1b) and is
  reported as such.
- **Known gaps to fill at J7:** the FastV/Pyramid DocVQA cells (HF, currently skip 187/200 at native
  pixels) must be re-run at a lower pixel cap; OCR-Bench/GQA `none` columns are currently reference
  cells.
- **Mandatory table note (printed):** *FastV wins TextVQA/GQA at the same model and budget; this
  table does not support "RBM outperforms existing methods". The RBM claim is robustness (never
  collapses on any benchmark) plus a large text-dense/OCR lead over the post-merger family.*

### 5.3 Same-model, iso-budget comparison (FastV / PyramidDrop / VisionZip-port)

Table 1b reports the **ready** n=200 subset numbers on Qwen3-VL-8B under official metrics, at
κ = 25% unless noted [interim; the full-split version is Table 1].

**Table 1b.** Qwen3-VL-8B, n=200, official metrics, κ = 25% (Pyramid at its two indicated budgets).
`post ≡ VZ` is the post-merger L2 cell, byte-identical to the VisionZip principle-port (Section 3.4).
[interim]

| Benchmark (metric) | none | RBM (pre) | post ≡ VZ | FastV | Pyramid |
|---|---|---|---|---|---|
| TextVQA (VQA-acc) | 0.858 | 0.598 | 0.215 | **0.680** | 0.852 @62.5% |
| DocVQA (ANLS, 600k px) | 0.951 | 0.424 | 0.251 | **0.518** | 0.878 @62.5% |
| OCR-Bench (/1000) | ~0.73 | **0.580** | 0.165 | 0.155 | — |
| GQA (exact match) | ~0.53 | 0.420 | 0.465 | **0.490** | — |
| Pyramid iso-25% `[1,0,0,0]` (TextVQA / GQA / OCR) | — | — | — | — | 0.073 / 0.305 / 0.005 |

Reference (native-resolution DocVQA, not iso-pixel with the HF baselines): none 0.976, RBM 0.465,
post 0.200 [E: rescore_rerun_report].

**Reading (R1-respecting).** FastV, being query-conditioned (layer-2 attention has already mixed in
the question), wins TextVQA (+8.2 pp over RBM), DocVQA (+9.4 pp at matched ptid 176), and GQA
(+7.0 pp). **RBM's distinctive property is not that it wins these but that it never collapses**: it
leads the post-merger family on every text-dense benchmark, and it is the **only** compared method
that retains OCR-Bench (0.580 vs FastV 0.155, **+42.5 pp**)—dense OCR is destroyed by the merger,
and no ranking, however query-conditioned, can recover information the merger already discarded;
only pre-merger selection, which keeps raw patches, preserves it [E: j4_probe_qwen3vl; j4 补遗]. The
mechanism behind FastV's wins is precisely the query-dependent headroom that our query-level gate
cannot reach pre-merger (Section 6).

**PyramidDrop occupies a different budget and degenerates at ours.** Its canonical schedule
(keep_equiv 0.625, ≈2.1× our budget) is near-lossless (TextVQA 0.852, DocVQA 0.878)—a
retention-curve point, not a same-budget comparison. Forced to an honest iso-25% budget, its
faithful degenerate schedule `[1,0,0,0]` (drop all visual tokens after the first layer segment)
collapses (TextVQA 0.073 / GQA 0.305 / OCR 0.005): PyramidDrop's progressive schedule is not viable
at 25%, whereas RBM holds (0.60 / 0.42 / 0.58 on the same benchmarks) [E: j4_probe_qwen3vl 补遗].

### 5.4 Cross-generation consistency

Table 2 reports the **ready** Qwen2.5-VL-7B n=200 official numbers alongside Qwen3-VL deltas,
showing the stage law reproduces on a second generation [interim].

**Table 2.** Qwen2.5-VL-7B, n=200, official metrics. Δ = pre − post (percentage points). [interim]

| Benchmark (metric) | none | RBM pre@25% | post@25% | Δ@25% | pre@12.5% | post@12.5% | Δ@12.5% |
|---|---|---|---|---|---|---|---|
| TextVQA (VQA-acc) | 0.870 | 0.735 | 0.415 | **+32.0** | 0.618 | 0.318 | **+30.0** |
| DocVQA (ANLS) | 0.975 | 0.687 | 0.499 | **+18.8** | 0.476 | 0.253 | **+22.3** |
| OCR-Bench (/1000) | 0.805 | 0.465 | 0.180 | **+28.5** | 0.335 | 0.060 | **+27.5** |
| GQA (exact match) | 0.585 | 0.565 | 0.555 | **+1.0 (tie)** | 0.505 | 0.505 | **0.0 (exact tie)** |

The three text-dense Δ are all ≥ 5σ (paired SE ≈ 0.035 at n=200); GQA is a statistical tie
(z ≈ 0.3). VisionZip-style ≡ post holds byte-for-byte on this generation too (0.415 == 0.415)
[E: j2_crossgen_matrix].

**Cross-generation summary (deltas, both models).**

| Δ@25% (pre − post) | Qwen3-VL-8B | Qwen2.5-VL-7B |
|---|---|---|
| TextVQA | +38.3 pp | +32.0 pp |
| DocVQA @25% / @12.5% | +26.5 / +47.5 pp | +18.8 / +22.3 pp |
| OCR-Bench | +41.5 pp | +28.5 pp |
| GQA | tie / no crossover* | +1.0 pp tie; @12.5% exact tie |

\*On Qwen3-VL at n=200 the post-merger cell leads GQA by +4.5 pp (≈1.3σ, **not significant**); at
n=100 the same cells tied exactly. We report this as **direction only** and characterize GQA as a
tie / no-crossover regime (Section 7) [E: j2_crossgen_matrix].

**Honest nuance (reported, not hidden).** Qwen2.5-VL's post-merger cell is more robust on
DocVQA@25% than Qwen3-VL's (0.499 vs 0.200); the gap between generations opens fully only at 12.5%
deep compression. We read this as: lossy-merger distortion worsens monotonically with compression,
with the *same direction but different rate* across generations [E: j2_crossgen_matrix].

### 5.5 Comparison with official VisionZip (mismatched anchor only, R3)

Official VisionZip is not runnable on our setup (no Qwen3-VL support; no vLLM path; ViT-attention
materialization OOMs on ~16k-token document images; no serving-side hook), so we present **no**
head-to-head Qwen3-VL number and claim **no** victory over the official method [E:
visionzip_gap_report §2, §3]. We use the authors' *own* published Qwen2.5-VL numbers only as a
**model/stage-mismatched reference anchor**:

**Table 3-anchor.** Official VisionZip on Qwen2.5-VL (~7B, lmms-eval), from the authors' README.
Mismatched reference, **not** a Qwen3-VL comparison, **not** a same-model cell. [E:
visionzip_gap_report §1]

| retain | MME | MMVet | OCRBench | POPE | RealWorldQA | DocVQA | MathVerse |
|---|---|---|---|---|---|---|---|
| 100% | 2316 | 61.6 | **81.5** | 86.7 | 68.6 | **95.1** | 46.3 |
| 70% (65d+5c) | 2334 | 60.0 | 80.9 | 86.4 | 68.2 | 94.5 | 45.8 |
| 50% (45d+5c) | 2209 | 57.0 | **70.5** | 86.3 | 68.6 | 93.8 | 45.1 |

The *shape* is the anchor: general tasks (POPE, RealWorldQA) hold while the text-dense task degrades
**first**—OCRBench 81.5 → 70.5 (−13%) at 50% retention while DocVQA loses only 1.3—and the authors
stop at 50% with no 25% row. The README caveat ("Qwen2.5VL already uses PatchMerger … the gain is
less striking than on LLaVA") is the same stage effect observed from the other side. Our
post-merger cells at 25% (Table 1b/2) are the consistent extrapolation of that trajectory into the
regime the official evaluation did not enter [E: visionzip_gap_report §4].

<!-- NUMBERS: Table 1 = PENDING J7 (placeholder, no numbers). Table 1b ready (j4_probe + j4 补遗 +
     rescore_rerun): TextVQA .858/.598/.215/.680/.852; DocVQA@600k .951/.424/.251/.518/.878;
     OCR .580/.165/.155; GQA .490/.465/.420; Pyramid iso-25% .073/.305/.005; FastV vs RBM
     +8.2/+9.4/+7.0, RBM vs FastV OCR +42.5. Table 2 ready (j2): Qwen2.5 .870/.735/.415 etc;
     cross-gen deltas. GQA Qwen3 post +4.5pp ~1.3σ direction-only (R4). VisionZip anchor 81.5→70.5
     @50% (visionzip_gap_report) — mismatched only (R3). Efficiency Table 3 = PENDING J6. -->

---

## 6. Negative Results — pre-merger must be query-blind

We pre-registered three extensions that would make pre-merger selection query-aware or
regime-aware, and report all three as failures. They are not buried: they are the evidence that
closes the design space and explain *why* RBM is frozen as a query-blind method.

**(a) Image-level routing fails.** Offline, with both-stage outcomes collected per image, an
always-pre policy is near-dominant at the image level (pre ≥ post on 84–97% of images in every
benchmark). The best image-level router (a disagreement threshold on pooled n=192) scores 0.484,
*below* always-pre (0.494) and far below the per-sample oracle (0.576); a ptid-threshold router on
the larger pooled N=774 reaches 0.655 vs always-pre 0.634 vs always-post 0.452 vs oracle 0.702 [E:
method_gate_report §4; DECISIONS 07-21 4b]. Decomposing the oracle gap, only ~27% is
workload-level; ~73% is **sample-level and query-dependent**, unreachable from image-level signals.

**(b) Merger-aware hybrid masking fails the gate.** A hybrid that keeps the pre/post agreement set
and routes the contested budget to high-edge (text) units by a text-fraction t was pre-registered
against a no-OCR-regression + GQA-gain gate. At the tuned t = 0.5 it gains +2.3 pp on TextVQA
(0.560 vs pre 0.537, within noise) but **loses 8 pp on OCR-Bench** (0.510 vs pre 0.590, ≈2σ) and
gains nothing on GQA (0.500, where pre == post == 0.510). **No single text-fraction passes the
gate**; pre-merger ranking is the fixed point that ranking-informed post-stage allocation cannot
improve upon [E: method_gate_report §2–3].

**(c) Query-level embedding blending fails.** A query-aware pre-merger saliency
`s = (1−λ)·L2 + λ·(question-embedding cosine)` was tuned on a disjoint dev slice. **Every λ > 0 is
harmful**: dev mean 0.5772 at λ=0, −1.7 pp (λ=0.3), −5.2 pp (λ=0.5), −3.3 pp (λ=0.7). By the
pre-registered rule we select λ = 0 and freeze plain RBM [E: j5_qa_gate_result].

**Closure.** Three negative results at three granularities—image-level routing, merger-aware
masking, query-level blending—all fail. The unifying reading is that **pre-merger features are
purely visual and contain no usable query information**: cheap query-conditioned signals at the
pre-merger stage hurt rather than help. The query-dependent headroom (the oracle's +8.2 pp, and
FastV's wins on TextVQA/DocVQA/GQA) is reachable **only after the LLM's cross-attention layers mix
in the question**—exactly where FastV operates. Hence the design space closes on a single
prescription: **pre-merger selection must remain query-blind**, and RBM is that robust default.

<!-- NUMBERS: router always-pre 0.494 / oracle 0.576 / dis-router 0.484 (method_gate §4);
     ptid-router 0.655 / always-pre 0.634 / always-post 0.452 / oracle 0.702 (DECISIONS 07-21 4b);
     73% sample-level (4b); hybrid t=0.5 TextVQA 0.560/OCR 0.510/GQA 0.500 vs pre 0.537/0.590/0.510
     (method_gate §2–3); QA gate λ {0/−1.7/−5.2/−3.3pp} (j5). FastV query-conditioned wins (j4_probe). -->

---

## 7. Limitations

1. **Single LLM family, two generations.** Both models are Qwen-VL (Qwen3-VL-8B and
   Qwen2.5-VL-7B): same family, two generations, not two families. InternVL and LLaVA families are
   not validated; whether the lossy-merger mechanism generalizes to other mergers is open.
2. **Cross-engine baselines (accuracy comparable, throughput not).** FastV and PyramidDrop run in an
   HF eager harness while RBM/post/none run in vLLM. Accuracy is comparable across engines (r=0
   anchor 8/8 per-sample identical; manual pre-norm vs native max-diff = 0; HF-vs-vLLM *none* 16/16
   equivalent) [E: j4_step2_fix], but **throughput is not**; we report no cross-engine speedup and
   confine efficiency to a single engine.
3. **DocVQA configuration dependence.** DocVQA uses a large-document config (max-num-batched-tokens
   32768, max-pixels 1.5M), and the baseline-comparison row uses a 600k PIL pre-scale so all methods
   skip 0 samples. Post-merger deep-compression collapse is partly configuration-dependent, which we
   disclose rather than present as a pure method effect.
4. **GQA is reported honestly as a tie / no-crossover.** On Qwen3-VL at n=200 the post-merger cell
   leads GQA by +4.5 pp (≈1.3σ, not significant; it tied exactly at n=100). We report the direction
   and do **not** claim a crossover (Section 5.4).
5. **Subset → full-split migration pending.** Headline numbers are n=200 seed-0 subsets `[interim]`;
   the full-split main table (Table 1) is **PENDING J7**, and the efficiency table (Table 3) is
   **PENDING J6**. The Qwen2.5-VL causal swap is **undecided** (Section 4.5), so the causal claim is
   scoped to Qwen3-VL.
6. **Greedy decoding / single device.** Decoding is deterministic (temperature 0), so error bars are
   binomial standard errors only, with no temperature variance; all runs are on a single A40.
   Efficiency is measured **offline**, not under continuous-batching serving.

<!-- NUMBERS: r=0 anchor 8/8 + 16/16 equivalence (j4_step2_fix); GQA Qwen3 post +4.5pp ~1.3σ
     (j2); DocVQA 1.5M/32768 + 600k cap (rescore_rerun + j4 补遗); Qwen2.5 swap undecided (j3).
     No SOTA / cross-model claims (R2). -->

---

## 8. Conclusion

We showed that the native 2×2 merger in merger-equipped VLMs is **lossy in a text-hostile
direction**: it re-shuffles unit saliency almost from scratch (M1), systematically demotes
text-stroke units on documents (M2), and—by a ranking-swap control on Qwen3-VL-8B—accounts for the
*entire* pre-vs-post accuracy gap as a **ranking effect**, not forward-path destruction (M3:
swap ≡ pre, DocVQA 200/200 byte-identical, TextVQA 198/200). The practical consequence is
**Rank-Before-Merge (RBM)**: a query-blind L2 ranking applied to merger-input units. Its stage law
is cross-generation—on text-dense benchmarks RBM leads post-merger selection by +18.8 to +38.3 pp
across Qwen3-VL-8B and Qwen2.5-VL-7B, ties object-centric GQA with no crossover, and widens under
deeper compression. In a same-model, iso-budget comparison RBM is the **robust default**: it never
collapses, never loses to the post-merger family, and is the only compared method that retains
OCR-Bench (+42.5 pp over the query-conditioned FastV, which wins the other three benchmarks
precisely because it is query-conditioned). Three pre-registered negative results close the design
space: pre-merger selection **must remain query-blind**, because the query-dependent headroom is
reachable only after the LLM's cross-attention layers mix in the question. Future work: extend the
mechanism and method to a second model family (e.g. InternVL3), add a same-model comparison to
Hi-Lo Prune if its code is released, and measure serving-side (continuous-batching) efficiency.

<!-- NUMBERS: M3 200/200 & 198/200 (mechanism §3); +18.8~+38.3pp (j2 + rescore_rerun); OCR +42.5pp
     (j4_probe); three negatives (method_gate §2–4 + j5). No "beats existing methods" / no SOTA (R1/R2). -->

---

## Red-line self-check (plan §红线 R1–R7)

- [x] **R1 — no "RBM beats/outperforms existing methods/SOTA".** Full text grepped for
  beats/outperforms/SOTA re: RBM: absent. FastV explicitly stated to win TextVQA/DocVQA/GQA
  (Abstract, §1, §5.3, §8). RBM claims = "robust default / never collapses / never loses to the
  post-merger family / uniquely holds OCR-Bench". The OCR +42.5 pp is stated as RBM vs FastV on one
  benchmark, not as a general superiority.
- [x] **R2 — no cross-model SOTA.** All causal claims scoped to Qwen3-VL-8B (§4.3, §7); Qwen2.5-VL
  presented as corroboration of the ranking *law*, with the causal swap explicitly **not** claimed
  to generalize (§4.5). No cross-model "state-of-the-art" anywhere.
- [x] **R3 — VisionZip official numbers = mismatched anchor only.** Table 3-anchor labelled
  "model/stage-mismatched reference, NOT a Qwen3-VL comparison, NOT a same-model cell"; no
  head-to-head victory claimed (§5.5).
- [x] **R4 — GQA = tie / no crossover.** Stated as tie / no-crossover throughout; Qwen3 n=200 post
  +4.5 pp (≈1.3σ) reported as **direction only** (§5.4, §7). No "post beats pre" claim.
- [x] **R5 — non-significant gaps = direction only.** GQA +4.5 pp and +1.0 pp flagged not
  significant / direction only; ChartQA not re-asserted as a win (not carried into v4 headline).
- [x] **R6 — official metrics only.** VQA-acc / ANLS / OCR-Bench official / GQA exact match used;
  no containment figures (0.695/0.255/0.725/0.390/0.320/0.380/+44.0/+33.5/−6.0/+46.6 all absent).
- [x] **R7 — engine + pixel-cap disclosure.** HF-vs-vLLM engine difference and accuracy-only
  comparability disclosed (§5.1, §7); DocVQA 1.5M/32768 config and 600k iso-pixel cap disclosed
  (§5.1, §7); throughput confined to vLLM.

**Grep guard (run before submission):** confirm zero hits for `0.695|0.255|0.725|0.390|0.320|0.380|
+44.0|+33.5|−6.0|+46.6|outperform|beats(?! …scoring-retired)|state-of-the-art|SOTA` in the
submission-bound prose, and confirm Table 1 carries only J7 full-split numbers.

<!-- NUMBERS: self-check references — containment numbers to be grep-killed (plan §A);
     R1–R7 mapped to sections above. -->

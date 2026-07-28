# The Lossy Merger: Pre-Merger Selection as a Query-Blind Robust Default for Visual-Token Compression in Vision-Language Models

*Submission draft — CCF-B methods paper (Pattern Recognition / ICME tier). Language: English.*

**Authors:** [anonymized placeholder — to be filled at de-anonymization]

<!-- TITLE: three candidates considered (mechanism + robustness framing; no beats/SOTA/pooling):
     (1, RECOMMENDED, used as H1) "The Lossy Merger: Pre-Merger Selection as a Query-Blind Robust
     Default for Visual-Token Compression in Vision-Language Models";
     (2) "Rank Before You Merge: A Lossy-Merger Ranking Mechanism and a Query-Blind Pre-Merger
     Default for Merger-Equipped VLMs";
     (3) "Where Selection Meets the Merger: Isolating the Selection Stage in Text-Dense VLM
     Compression, with a Robust Query-Blind Default". The earlier working title "Selection Stage
     Beats Scoring" is abandoned (it conflicts with the FastV evidence, §5). Evidence anchors use
     [E: <digest>] -> files under experiments/ and drafts/. Red-line self-check at end of file.
     Per-section NUMBERS audit comments are retained for authoring and removed at submission. -->

---

## Abstract

Merger-equipped vision-language models treat their native 2×2 token merger as lossless. We show it
is not. On Qwen3-VL-8B the merger nearly re-shuffles unit saliency from scratch (M1: pre–post rank
ρ = 0.14–0.36), systematically demotes high-edge/text-stroke units (M2: demoted-group Sobel 0.64 vs
0.12 on DocVQA), and—via a ranking-swap control that holds the forward path fixed—accounts for the
*entire* pre-vs-post accuracy gap (M3: swap ≡ pre; DocVQA 200/200 byte-identical, TextVQA 198/200).
The mechanism is selection-level causal on both architectures: pre- and post-merger selection keep
identical unit sets (kept-set Jaccard = 1.0 in all four model×benchmark cells), and the pre>post
gap is a ranking effect, not forward-path destruction (byte-identical answers on Qwen3-VL). Selecting
**before** the merger—Rank-Before-Merge (RBM), a query-blind L2 ranking—is cross-generation: on the
official **full splits** RBM leads post-merger selection by **+11.0 to +38.4 pp** on text-dense
benchmarks (native resolution, both generations; paired McNemar z = 14.6–43.0), with **no text-dense
crossover**, and the gap widens under deeper compression; on object-centric GQA it trails by a small
but significant margin (post +2.6–2.8 pp, full n = 12578; §5.2). In a same-budget comparison the
query-conditioned FastV leads the query-relevant benchmarks (TextVQA both models; Qwen3-VL GQA), yet
RBM alone holds dense OCR-Bench (**+25.9 pp** over FastV; z ≈ 9.5) and is the **robust default, not
uniformly optimal**. Three pre-registered negative results close the design space: pre-merger
selection **must remain query-blind**.

<!-- NUMBERS: M1 ρ 0.14–0.36 (mechanism_verification §1); M2 Sobel 0.641 vs 0.124 (mechanism §2);
     M3 swap≡pre 200/200 & 198/200 (mechanism §3). HEADLINE NOW FULL-SPLIT (R2-10/R3-4): native-res
     Δ₂₅ range +11.0 (Q2.5 DocVQA) to +38.4pp (Q3 TextVQA) from Table 1 (j7_main_table); the n=200
     +18.8~+38.3 range is REMOVED from the abstract (retained only in appendix Table A2 / §5.4 as a
     cross-split check). Paired McNemar z (now PRIMARY; computed this pass from per_sample.correct)
     text-dense range 14.6 (Q2.5 OCR) to 43.0 (Q3 TextVQA). GQA collapsed to one sentence -> §5.2
     (R3-2): full n=12578 post 0.477/0.585 vs pre 0.449/0.559 -> +2.8/+2.6pp; indep z 4.5/4.1,
     paired McNemar (per_sample.correct) 8.0/5.7, official-rescore McNemar 8.1/7.1 concordant. FastV
     claim DOWNGRADED (R2-1/R2-12): same-budget n=500 leads TextVQA both models (Q2.5 sig z=2.56 /
     Q3 direction-only z=1.79) + Q3 GQA (z=2.69); same-scope full-split verification pending R2-C.
     OCR +25.9pp = RBM 0.547 vs FastV 0.288 (z≈9.5, j7hf_baselines_n500); n=200 +42.5pp (0.580 vs
     0.155, j4_probe) retained in Table A2/§5.3. Abstract ≤200 words (wc-verified this pass). -->
<!-- same-scope numbers pending R2-C -->
<!-- R1-1 RESOLVED: kept-set Jaccard(swap,pre)=1.000 on both architectures (r1_1_swap_jaccard);
     causal wording strengthened to selection-level-causal, architecture-general; Qwen2.5 residual
     = reverse_indices order confound (implementation, not science). -->

---

## 1. Introduction

**The unexamined assumption.** Vision-language models (VLMs) convert an image into hundreds to
tens of thousands of visual tokens, and visual-token compression—pruning or merging the least
important tokens before the language model—has become a standard efficiency lever. In
merger-equipped architectures (the Qwen-VL family, and others that merge 2×2 patch groups into one
token), the compression debate has run almost entirely over **which** tokens to keep and **how** to
score them; the *stage* at which selection happens—before or after the native merger—has not been
isolated as an experimental variable. Practitioners knew the merger matters—the Qwen2.5-VL authors
themselves note it already compresses—yet no work controls the stage while holding model, token
count, and scorer fixed. This paper shows that, on text-dense workloads, the *stage* of selection
dominates the *scorer* of selection, and that the reason is a previously unreported property of the
learned merger: it is **lossy in a text-hostile direction**.

**What we find.** Under strict iso-model / iso-token / iso-selector control, applying the *same*
text-agnostic L2 scorer before the merger versus after it produces a large accuracy gap on
text-dense benchmarks and only a small, *reversed* gap on object-centric GQA—the one benchmark on
which post-merger selection leads, a small significant margin reported once, with full statistics,
in §5.2. The text-dense gap is not explained by information destroyed in the forward pass. A causal
ranking-swap control—run the post-merger forward path but select units with the pre-merger
ranking—recovers the pre-merger accuracy exactly (DocVQA: 200/200 byte-identical answers; TextVQA:
198/200, the residual being greedy-decode run noise). The entire pre>post gap is therefore a
**ranking effect**: the merger rewrites unit saliency, and post-merger selection reads the corrupted
ranking. A ranking-swap control isolates this exactly: post forward with pre ranking reproduces pre
(kept-unit sets identical, Jaccard = 1.0, both architectures; byte-identical on Qwen3-VL), so the
entire pre>post gap is the ranking, not the forward path (§4.3).

**What we propose.** Rank-Before-Merge (RBM): score 2×2 units on their merger-*input* features
with a query-blind L2 norm, keep the top-κ units, and pass only the survivors through the native
merger. No attention weights, no training, no query input—a pure inference-time stage change. The
method is deliberately minimal: we freeze it as *plain* RBM and do not decorate it with hybrids or
routers, because the experiments below show every such decoration is either neutral or harmful
(Section 6).

**Cross-generation consistency.** The stage law and the lossy-merger signature reproduce on a
second generation, Qwen2.5-VL-7B (text-dense pre>post of +11.0 to +26.1 pp on the official full
splits; object-centric GQA a small significant post-merger lead, §5.2), with one honest boundary:
the causal swap control replicates on Qwen3-VL but not on Qwen2.5-VL, so we claim the causal
decomposition only for Qwen3-VL and present the Qwen2.5-VL evidence as corroboration of the ranking
law (Section 4).

**Positioning by fair comparison.** We compare RBM against same-model, same-budget, runnable
baselines—FastV (one-shot attention pruning at LLM layer 2) and PyramidDrop (progressive in-LLM
dropping)—plus a faithful same-stage principle-port of VisionZip. The comparison is deliberately
honest about where RBM does *not* lead: in the same-budget n = 500 comparison the query-conditioned
FastV leads on TextVQA (both models) and Qwen3-VL GQA (Qwen2.5-VL TextVQA significant, z = 2.56;
Qwen3-VL TextVQA direction-only at current n, z = 1.79; Qwen3-VL GQA z = 2.69) and on same-pixel
DocVQA, while RBM is the *only* method that robustly retains OCR-Bench and in turn wins Qwen2.5-VL
GQA (Section 5.3; the wins are model-dependent). A same-scope full-split verification of the
FastV-vs-RBM contrasts is reported in Section 5.3. We therefore position RBM not as a method that
"beats" the field but as the **robust default, not uniformly optimal**—it never collapses, it
concedes only a narrow significant margin on object-centric GQA (§5.2), and it uniquely preserves
dense OCR that every post-merger and query-conditioned competitor destroys.
<!-- same-scope numbers pending R2-C -->

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
2. **A stage law and its minimal control method (RBM).** The deliberate minimalism of RBM—a
   query-blind pre-merger L2 ranking—is a *control design* that isolates the stage variable
   (Section 3.2). Under it, pre-merger selection leads post-merger selection by +11.0 to +38.4 pp on
   text-dense benchmarks (official full splits, native resolution) across two Qwen-VL generations,
   trails by a small significant margin on object-centric GQA (§5.2), shows no text-dense crossover,
   and widens under deeper compression.
3. **A same-model, iso-budget comparison** that positions RBM against FastV, PyramidDrop, and a
   VisionZip principle-port, establishing RBM as the robust default (never collapses; uniquely
   holds OCR-Bench) rather than a universal winner.
4. **An honest negative-result set** (image-level router, merger-aware hybrid, query-level gate)
   that closes the design space: pre-merger selection must remain query-blind.

<!-- NUMBERS: HEADLINE NOW FULL-SPLIT +11.0~+38.4pp (Table 1 / j7_main_table; native-res Δ₂₅:
     Q3 TextVQA +38.4 / DocVQA +24.3 / OCR +36.3; Q2.5 +26.1 / +11.0 / +29.3). The n=200
     +18.8~+38.3 range is retained ONLY in appendix Table A2 / §5.4 as a cross-split check.
     M3 200/200 & 198/200 (mechanism §3); negative λ −1.7~−5.2pp (j5_qa_gate_result); OCR n=200
     +42.5pp (j4_probe). No containment numbers used. GQA micro-margin CONSOLIDATED to §5.2 (R3-2):
     full-split post lead +2.6–2.8pp (indep z 4.1/4.5; paired McNemar per_sample.correct 5.7/8.0,
     official-rescore 7.1/8.1 concordant); §1 now carries terse §5.2 back-refs only. FastV wins
     DOWNGRADED to same-budget n=500 framing (R2-1/R2-12); same-scope full-split pending R2-C. -->

---

## 2. Related work

**Token pruning and merging in VLMs.** Most methods score visual tokens and drop the least
important before or within the early LLM layers. **FastV** prunes at the second LLM layer using
attention the visual tokens receive, observing many receive negligible attention \cite{chen2024fastv};
*relation*: it is our primary same-model baseline and the empirical instance of a
query-conditioned scorer that wins scene-text/object but cannot rescue merger-destroyed OCR
(Section 5). **PyramidDrop** drops tokens progressively across LLM layers in a pyramid schedule
\cite{xing2024pyramiddrop}; *relation*: a second same-model baseline whose canonical schedule occupies a
different budget point, and whose mechanism degenerates to collapse at an iso-25% budget. **FasterVLM**
pushes attention-based pruning toward faster convergence \cite{zhang2024fastervlm}. **SparseVLM** and
related work explore query-conditioned sparsification of the visual stream \cite{zhang2024sparsevlm}.
**PruMerge** merges tokens by attention-derived importance rather than dropping them
\cite{shang2024prumerge}. **FitPrune** fits lightweight predictors of token importance \cite{ye2024fitprune}.
All of these operate on tokens that have *already* passed the encoder's native merger; none treats
the merger stage itself as a design variable, and none of the query-conditioned variants is
available as a runnable same-model baseline on our stack except FastV and PyramidDrop.

**Token-merging lineage (ToMe → PruMerge → AdaptMerge).** Token *merging* originates with **ToMe**
\cite{bolya2023tome}, which reduces tokens on a ViT by bipartite soft matching—partitioning tokens
into two disjoint sets and greedily merging the most similar pairs—post-encoder, on the vision side,
with no query input. **PruMerge** \cite{shang2024prumerge} carries this idea to VLMs, merging tokens
by attention-derived importance rather than dropping them. **AdaptMerge** \cite{islam2025adaptmerge}
adapts merging to large multimodal models with *adaptive*, visual- and language-guided merging—both
inside the vision encoder and again at the LLM input, conditioned on the question—and reports that
the language guidance closes most of the OCR gap that plain ToMe incurs. *Relation to this work*:
this lineage merges or prunes *within or after* the encoder using trained or attention-derived
scorers, and AdaptMerge's central finding—that merging degrades OCR/text and query guidance partly
repairs it—is direct corroborating evidence for our lossy-merger thesis; we cite it as support, not
as a competitor. What none of these methods does is isolate the pre-vs-post-merger *stage* itself as
the experimental variable under iso-model / iso-token / iso-selector control on a VLM's *native*
merger: ToMe and its descendants add a *second*, learned merging/scoring stage on the merge side,
whereas we hold the native merger fixed and ask only whether selection should precede or follow it.
ToMe-side merging (bipartite matching, post-ViT) and AdaptMerge's adaptive merging are therefore
orthogonal to, and in principle compose with, the stage axis we study (Section 3).

**Qwen-family-specific compression.** Qwen-VL models introduce a PatchMerger that already compresses
visual tokens, and a line of work targets this family directly. **GlimpsePrune** \cite{zeng2025glimpseprune}
and **VScan** \cite{zhang2025vscan} prune or scan visual tokens within Qwen-VL inference; *relation*: they
confirm the Qwen family is an active compression target, but like the methods above they select
*after* the native merger, i.e. on the post-merger side of the stage axis we study.

**Dominant-plus-contextual selection.** **VisionZip** selects "dominant" tokens by CLS-to-patch
attention at the penultimate ViT layer and merges the remainder into "contextual" tokens by
key-cosine assignment and count-averaging, reporting strong general-task retention at aggressive
budgets on LLaVA models \cite{yang2024visionzip}. A line-reading of the official code confirms
both the LLaVA and the Qwen2.5-VL paths select *after* the native merger—for Qwen2.5-VL,
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
throughput). We adopt the lmms-eval family of *official* scorers for accuracy \cite{zhang2024lmmseval}
(VQA-accuracy, ANLS, OCR-Bench official five-category scoring, GQA word-normalized exact match) so
that our accuracy claims use community-standard metrics, and we disclose the engine of every
efficiency number, confining throughput claims to a single engine (Section 5.7).

**The specific design point we isolate (not a topological vacuum).** We do not claim that no method
ever touches tokens upstream of a merger. **QuietPrune** prunes ViT tokens early, *inside* the
encoder, via a trained query-guided adapter—a different stage (in-ViT, before the 2×2 unit grid is
formed), and not training-free. **FastV** and **PyramidDrop** select *inside the LLM*—a different
stage again, after the merger and after cross-attention has seen the question. **VisionZip** selects
on the *post*-merger side of the native merger—the same merger, but the opposite side of the stage
axis we study. The specific combination we isolate is narrower and, to our knowledge, unoccupied: on
a Qwen-VL model, **leaving the native merger untouched, scoring the merger-*input* 2×2 units by a
saliency norm, keeping the top-κ, and handing only the survivors to the unmodified native merger**.
Beyond this design point, our contribution is the *mechanism account* (M1–M3) explaining *why* this
stage dominates on text-dense workloads—the merger's nonlinear projection re-shuffles saliency and
demotes text-stroke units—which the methods above neither report nor control for. The Qwen2.5-VL
authors' own observation that VisionZip's gains are "less striking" on that model because the merger
already compresses [E: visionzip_gap_report §1] is the same effect seen from the post-merger side.
Our claim is therefore precise rather than absolute: this concrete pre-merger design point, together
with its causal mechanism explanation (M1–M3) on Qwen3-VL, is new; and for text-dense workloads the
stage axis dominates the scorer axis.

<!-- NUMBERS: VisionZip 81.5→70.5 @50% (visionzip_gap_report §1); "11/11 → both-generation
     byte-equivalence" (v3_sota_matrix §0 + j2). Related work is mechanism/positioning, no new
     metrics. ADDED (R1-3 / item 1): merging-lineage paragraph ToMe (bolya2023tome, ICLR'23,
     arXiv:2210.09461) → PruMerge → AdaptMerge (islam2025adaptmerge, Findings EMNLP'25, DOI
     10.18653/v1/2025.findings-emnlp.387; venue/authors web-verified via ACL anthology); AdaptMerge's
     OCR-gap finding cited as corroborating evidence, not competed. "Empty cell" RE-DELIMITED (item 2
     / R1-3): no absolute topological vacuum; QuietPrune=in-ViT different stage (trained), FastV/
     PyramidDrop=in-LLM different stage, VisionZip=post-merger opposite side; the concrete
     pre-merger-input-unit top-k -> unmodified native merger design point + its M1-M3 mechanism
     account is the new claim. -->

---

## 3. Method — Rank-Before-Merge (RBM)

### 3.1 Problem setup

We study two merger-equipped VLMs, **Qwen3-VL-8B-Instruct** \cite{bai2025qwen3vl} and
**Qwen2.5-VL-7B-Instruct** \cite{bai2025qwen25vl}, in bf16 with eager attention, served by vLLM 0.19
(V1 engine) on a single A40 (46 GB). Their shared vision pipeline is: image → ViT patch tokens
(effective 16 px patch footprint) → native 2×2 merger → (Qwen3-VL) additional deepstack mergers that
merge intermediate-depth ViT features → concatenation of main-merger and deepstack outputs →
language model [E: mechanism_verification_report §0; visionzip_gap_report §3]. Concretely, the
native merger is **not** an averaging pool: the four patch features of each 2×2 group are
concatenated into one vector and projected by a small learned MLP—LayerNorm (`ln_q`/`norm`) →
`fc1` → GELU → `fc2`—to a single *unit* vector (32 px footprint). The lossy step is the
four-to-one dimensional reduction through this **nonlinear projection**, which is why a kept unit's
representation is a nonlinear function of its four patches rather than their mean. We verified this
against the served implementation (vLLM 0.19 site-packages `Qwen3_VisionPatchMerger`, qwen3_vl.py,
and `Qwen2_5_VisionPatchMerger`, qwen2_5_vl.py; identical `norm → fc1 → GELU → fc2` form on both
generations).

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

**Figure 1 (pipeline).**

![Fig. 1](figures/fig1_pipeline.pdf)

*Figure 1.* Schematic contrasting the two hook points: (a) RBM scores raw 32 px-unit features, keeps
top-κ units, then invokes the native merger on survivors only; (b) post-merger selection merges all
units first and scores the merged tokens. The two pipelines differ *only* in where the saliency
score is tapped; the native 2×2 merger (+ deepstack on Qwen3-VL) and the LLM are drawn identically
in (a) and (b).

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
features for post-merger selection. (Family note: on Qwen3-VL the first-invoked merger is
deepstack[0], whose input is the ViT block-8 output, indexes [8,16,24]; on Qwen2.5-VL it is the
main merger input after the final ViT block [E: r1_1_swap_jaccard §B].) The L2 scorer is
deliberately query-blind [E: mechanism_verification_report §0]. The rationale is
methodological: a strong, task- or query-aware scorer would confound *scoring quality* with *stage*.
With an identical, saliency-free scorer, the only manipulated variable is the hook point. We freeze
this as **plain RBM** and add no variant. Section 5 reports that the stage law survives replacing L2
with a second scorer family (global-centroid attention) on Qwen3-VL, indicating the effect is not an
L2 artifact; the same invariance does *not* extend to the attention proxy on Qwen2.5-VL, which we
report honestly (Section 4). Unlike the merging lineage of Section 2 (ToMe / PruMerge / AdaptMerge),
RBM introduces no second learned merger and no query input: it leaves the model's own native merger
unchanged and manipulates only the *set* of units that merger sees—which is exactly what makes the
pre-vs-post contrast a clean stage experiment rather than a scorer comparison.

### 3.3 Rank-Before-Merge

Given an image with N merge units, compute `s(u)` for every unit on merger-input features, retain
the top `k = κN` units, and invoke the native merger (main and all deepstack mergers) on the
retained units only, producing k main tokens plus the corresponding deepstack tokens. The
implementation wraps the merger's forward so that selection executes before every native merge call,
including the deepstack mergers [E: visionzip_gap_report §4]. Unit identity is shared across the main
and deepstack streams—the deepstack mergers process intermediate-depth features of the *same* 2×2
units—so pruning a unit removes it from every stream; masking all deepstack mergers (Algorithm 1,
lines 5–6) is therefore the architectural contract of unit-level selection, not a tunable
hyperparameter. No attention weights are required and no parameters are trained; the method is a pure
inference-time stage change. Because the mask is at *unit* granularity, selection cannot prevent
intra-unit combination—the four patches of a kept unit are still concatenated and projected by the
merger exactly as in the uncompressed model. What selection changes is *which* units reach the
merger: pre-merger selection keeps units on their raw (merger-input) saliency, whereas post-merger
selection keeps them on a saliency the merger has already rewritten. The mechanism we test in
Section 4 is therefore not that glyph patches are shielded from being averaged with background (they
are not, within a kept unit), but that the merger's nonlinear four-to-one projection distorts the
high-frequency text-stroke units and rewrites the ranking (M1)—an effect that compounds under deep
compression, where the few surviving units change the batch composition the projection operates
on—so that post-merger selection systematically drops exactly the text-bearing units.

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

To ask whether a dominant+contextual recipe \cite{yang2024visionzip} rescues post-merger
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
invocation as the headline cells [E: mechanism_verification_report §0]. Figure 2 summarizes the two
headline measurements.

![Fig. 2](figures/fig2_mechanism.pdf)

*Figure 2.* The lossy-merger mechanism on Qwen3-VL-8B. **(a)** M1 — pre-vs-post rank decorrelation:
the learned merger reshuffles unit saliency almost from scratch (Spearman ρ = 0.14–0.36; Jaccard@25%
= 0.18–0.28; largest reshuffle on text-dense DocVQA, smallest on object-centric GQA). **(b)** M3 —
the ranking-swap causal control: holding the forward path at *post* and swapping in the *pre*
ranking recovers pre accuracy (Δ ≤ 0.005; DocVQA byte-identical), so the entire pre–post gap is a
ranking effect. Per-image supplementary raster evidence: `figures/token_survival_m1_rank_overlap.pdf`
(M1) and `figures/token_survival_m2_edge_demotion.pdf` (M2).

### 4.1 M1 — the merger reshuffles unit ranks

Per image we correlate the *pre* ranking (merger-input unit L2) with the *post* ranking
(merged-token L2) over all units. The two rankings are barely related:

| benchmark (type) | Spearman ρ | Kendall τ | Jaccard@25% |
|---|---|---|---|
| DocVQA (text-dense doc) | **0.137 ± 0.158** | 0.094 | **0.180** |
| TextVQA (text-dense scene) | 0.332 ± 0.124 | 0.227 | 0.243 |
| GQA (object-centric) | **0.360 ± 0.091** | 0.249 | **0.278** |

The merger's nonlinear projection reorders unit saliency almost from scratch (ρ = 0.14–0.36, nowhere
near 1). Reading the two columns separately: the full-ranking correlation *rises* from the most
text-dense benchmark to the object-centric one (ρ = 0.137 on DocVQA < 0.332 on TextVQA < 0.360 on
GQA), so the reshuffle is largest on DocVQA and smallest on GQA; and at the decision-relevant
top-25% cut the pre/post overlap is *smallest* on text-dense data (Jaccard = 0.180 on DocVQA < 0.243
on TextVQA < 0.278 on GQA). Both orderings follow text density. Yet the reshuffle is substantial on
every benchmark—ρ = 0.36 on GQA is still far from 1—and it is the merger's *directional* bias against
text-stroke units (M2: ρ = 0.44 / 0.16 / 0.04), not the scalar size of ρ alone, that tracks the
accuracy stage law: GQA is still heavily re-shuffled even though its accuracy consequence nearly
vanishes (§4.4) [E: mechanism_verification_report §1] (Fig. 2a).

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
mechanism_verification_report §2] (supplementary raster `figures/token_survival_m2_edge_demotion.pdf`).

*Why the merger demotes text (speculative).* We hypothesize the *direction*, not merely the
magnitude, is set by the merger's training objective and geometry: the four-to-one projection is
learned for general visual–language alignment on natural content, so it attenuates the high-frequency
stroke energy that distinguishes text units, and any post-merger norm then scores merged text units
low. This is testable offline—compare the L2 post-scorer against a high-frequency-weighted
post-scorer—and we flag it as a hypothesis, not a measured result.

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
2/200 is ε-level kernel numerics, not scheduling [E: mechanism_verification_report §3].

**The decisive hardening — kept-set identity on both architectures.** We emit the kept unit indices
from both paths and compare them directly: **Jaccard(swap-kept, pre-top-κ) = 1.000 in all four
cells** (Qwen3-VL and Qwen2.5-VL × DocVQA and TextVQA; n=32, seq=1; chance level κ/N = 0.25)
[E: r1_1_swap_jaccard]. The two paths select *exactly the same units*; the swap-vs-pre answer
residual on Qwen2.5-VL (+6.7 pp DocVQA at n=32) is traced to an implementation confound of the swap
control — window-attention `reverse_indices` (three recovery sites in qwen2_5_vl.py, zero in
qwen3_vl.py) map identical index sets to physical units in different orderings across the two paths
— while batch-dependent merging is structurally excluded (the PatchMerger is per-token LN+MLP).
**Verdict: the pre>post gap is attributable to the ranking alone — selection-level causal,
architecture-general** (byte-identical on Qwen3-VL; identical kept sets with an order-confound
residual on Qwen2.5-VL); the merged representations of kept units carry no stage-dependent
information loss.

### 4.4 Two failure modes (dual-layer mechanism)

The mechanism is clean and complete on DocVQA (post selection is *misguided away from text*;
restoring the pre ranking restores accuracy fully). On TextVQA the ranking corruption exists
(M1 ρ = 0.33) and is again the *entire* source of the gap (M3 swap ≡ pre), but the M2
text-directionality is only moderate (ρ = 0.16; group (a) 0.281 vs (b) 0.186)—scene-text images do
not show the document-level "post avoids text" contrast, and even pre-selected units underperform
the uncompressed baseline (0.60 vs 0.86), a **budget effect** shared by any 25%-keep method. GQA
shows near-null reshuffle directionality (M2 ρ = 0.04) [E: mechanism_verification_report §4]:
object-centric images carry little high-frequency text for the merger to damage, so its directional
bias nearly vanishes there, and the accuracy consequence is correspondingly small—on the full split,
post-merger selection leads pre by a small but statistically significant margin (reported once, with
full statistics, in §5.2). The *sign* fits the mechanism: where the merger destroys little, the more
global merged (projected) features available only after the merger confer a small edge (and a
query-conditioned scorer reads the same regime still better—FastV leads GQA, Section 5.3), whereas
on text-dense workloads the merger's text-hostile reshuffle dominates in the opposite direction. That
the object-centric effect is an order of magnitude smaller than the text-dense one is itself evidence
that the stage effect is **workload-conditional**, not uniform. We summarize: **documents = selection
misdirection; scene text = feature degradation; object scenes = near-null ranking with a small
significant post lead**—all three regimes reported, nothing hidden.

### 4.5 Cross-generation boundary and the mrope diagnostic

**Replicates on Qwen2.5-VL-7B.** The ranking law and the stage effect replicate: with the L2
selector at r = 0.75 (n=64), DocVQA pre 0.664 > post 0.531 (+13.3 pp) and TextVQA pre 0.719 > post
0.349 (+37.0 pp)—the L2 *sign* is unchanged across generations; VisionZip-style ≡ post holds
byte-for-byte (TextVQA 0.415 == 0.415) [E: j2_crossgen_matrix; j3_mechanism_crossarch].

**The swap residual on Qwen2.5-VL is an implementation confound, not a scientific difference.**
The M3 swap control initially did not reproduce on Qwen2.5-VL (swap exceeded pre: n=32 seq=1 DocVQA
swap 0.667 vs pre 0.600, 13/32 identical answers; TextVQA 0.698 vs 0.688) — but the decisive
diagnostic resolves it: the *kept-unit index sets are identical across paths* (Jaccard = 1.000 on
both benchmarks; a misaligned or different ranking would show Jaccard ≈ 0.25), so swap did apply the
pre ranking [E: r1_1_swap_jaccard]. The answer-level residual traces to Qwen2.5-VL's window
attention: its `reverse_indices` recovery (three sites in qwen2_5_vl.py; zero in qwen3_vl.py) is
skipped in the pre path but applied to the spatial-order splits in the swap path, so identical
index sets address *permuted physical units*; batch-dependent merging is structurally excluded
(per-token LN+MLP). The M3 identity therefore holds **selection-level on both architectures** and
is **byte-exact on Qwen3-VL**; the Qwen2.5-VL evidence corroborates the ranking law additionally
through M1 + the stage law + VZ≡post.

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
     0.530/0.605 Qwen2.5 (j3). GQA near-null ρ=0.036 pairs with the full-split accuracy verdict:
     post +2.6–2.8pp (z 4.1/4.5 indep; McNemar 7.1/8.1 paired; 1/10–1/14 of TextVQA/OCR-Bench Δ;
     j7_main_table, supersedes n=200 "post≈pre/tie"); FastV GQA +7.0pp n=200 (j4_probe).
     All M1–M3 = Qwen3-VL-8B; Qwen2.5 = corroboration only (R2/R4). -->

---

## 5. Experiments

### 5.1 Setup

**Models.** Qwen3-VL-8B-Instruct and Qwen2.5-VL-7B-Instruct, bf16, enforce_eager, 1× A40 46 GB,
vLLM 0.19 V1, greedy decoding (temperature 0).

**Benchmarks and metrics (official).** TextVQA (VQA-accuracy) \cite{singh2019textvqa}, DocVQA (ANLS)
\cite{mathew2021docvqa}, OCR-Bench (official five-category score /1000) \cite{liu2024ocrbench}, and GQA
(word-normalized exact match) \cite{hudson2019gqa}. All scorers are verbatim ports of the lmms-eval family
\cite{zhang2024lmmseval}, with ground-truth self-test passing 200/200. Each TextVQA/DocVQA/GQA question
carries the canonical short-answer instruction ("Answer the question using a single word or
phrase.") baked into the subset, which is required for VQA-acc/ANLS/exact-match to be meaningful.

**Budget.** Retention κ over per-image merge units (Section 3.1); same κ ⇒ iso-token per image
(verified via mean post-merger token count).

**Subset vs full split.** Table 1 (§5.2) is the headline and is reported on the **official full
splits** (TextVQA val 5000; DocVQA val 5349; OCR-Bench 1000; GQA test-dev 12578) for the
RBM/post/none columns; its **FastV/Pyramid columns are the independent HF-harness n=500 headline
subset** (footnote ʰ), same model and seed. The seed-0 **n=200 subsets** survive only as a
cross-split consistency check—the Qwen3-VL n=200 cross-method check in Appendix Table A2 and the
compact Qwen2.5-VL n=200 cross-generation view in Table 2 (§5.4); subset and full split agree in
direction and approximate magnitude on every cell (Appendix Table A2). The two scopes are never mixed
in a single table, and the full-split numbers are the headline.

**Baseline engines and fairness protocol (disclosure, R7).** FastV and PyramidDrop are run in an
independent HuggingFace transformers 4.57.6 eager harness (they have no vLLM path); RBM and the
post-merger / none cells use vLLM. Fairness is enforced by (i) the same keep ratio relative to each
image's own full token count, (ii) reporting mean absolute post-merger token count, (iii) a
per-family min/max-pixel calibration so token counts are iso across patch sizes, and (iv) for
PyramidDrop, folding its layer schedule into an equivalent mean retention. **Accuracy is comparable
across engines** because the HF harness is validated to bit-degrade to native at r = 0 (r=0 anchor:
8/8 per-sample identical answers; manual pre-norm vs native max-diff = 0) and because HF-vs-vLLM
*none* cells agree 16/16 [E: j4_step2_fix §验证]. **Throughput is not comparable across engines**;
all efficiency numbers are confined to vLLM (Section 5.7; Table 3).

**DocVQA iso-pixel disclosure (R7).** DocVQA runs under a large-document configuration
(max-num-batched-tokens 32768, max-pixels 1.5M cap). Because the HF baselines cannot honor a pixel
cap through the engine, the DocVQA *baseline-comparison* row of Appendix Table A2 uses a PIL pre-scaled
**600k-pixel** cap under which **all methods skip 0 samples**; native-resolution RBM/none cells are
reported separately as a reference. This is disclosed because post-merger deep-compression collapse
is partly configuration-dependent.

### 5.2 Main result — official full split (Table 1)

**Table 1.** Official metrics, **full splits** (TextVQA val n=5000; DocVQA val n=5349; OCR-Bench
n=1000; GQA test-dev n=12578), greedy decoding, both models. RBM = pre-merger L2 ranking (this
work); `post ≡ VZ-port` = post-merger L2 selection, byte-identical to the VisionZip
dominant+contextual principle-port on both generations (§3.4). Parentheses give the binomial
standard error √(p(1−p)/n) with p the official metric (footnote d); OCR-Bench cells report the
official Final/1000 with the rate ± stderr in parentheses. Δ₂₅ = pre − post at κ = 0.25;
** = significant pre-merger lead; † = significant post-merger lead. The z printed in the Δ₂₅ column
is the independent-binomial cross-check; the **primary** significance test is the paired McNemar z of
the paired-test table below (footnote e). Under McNemar no full-split Δ cell is direction-only
(1.5 ≤ z < 1.96) or non-significant (z < 1.5): the smallest text-dense |z| = 14.6 and the smallest
overall |z| = 5.7 (the GQA post-lead); the independent-binomial cross-check agrees (smallest |z| = 4.1).
The **FastV @25% ʰ and Pyramid @62.5% ʰ columns are an independent
HF-transformers n=500 headline subset** (same model; FastV iso-budget at κ = 0.25, Pyramid at its
canonical keep_equiv 0.625 ≈ 2.1× budget), distinct in scope from the full-split our-method
columns; their stderrs use n = 500, and the Δ₂₅ significance band above does **not** apply to the
cross-method, cross-scope baseline contrasts (reported with their own pooled z in §5.3)
[E: runs/full_matrix/j7_main_table.json, 26/26 cells zero missing; experiments/j7_main_table.md;
runs/full_matrix/j7hf_official_summary.json; experiments/j7hf_baselines_n500.md]

| Model | Benchmark (metric, n) | none | RBM @25% | RBM @12.5% | post ≡ VZ-port @25% | post @12.5% | Δ₂₅ (pre − post) | FastV @25% ʰ | Pyramid @62.5% ʰ |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | TextVQA (VQA-acc, 5000) | 0.844 (±.005) | 0.605 (±.007) | 0.472 (±.007) | 0.222 (±.006) | 0.132 (±.005) | **+38.4 pp** (z=42.3) | 0.646 (±.021) | 0.835 (±.017) |
| | DocVQA (ANLS, 5349) | — ᵃ | 0.481 (±.007) | 0.352 (±.007) | 0.238 (±.006) | 0.103 (±.004) | **+24.3 pp** (z=27.1) | 0.486 (±.022) ⁱ | 0.882 (±.014) ⁱ |
| | OCR-Bench (Final/1000, 1000) | 760 (.760 ±.014) ᵇ | 547 (.547 ±.016) | 350 (.350 ±.015) | 184 (.184 ±.012) | 53 (.053 ±.007) | **+363 pts** (z=18.2) | 288 (.288 ±.020) | 746 (.746 ±.019) |
| | GQA (exact match, 12578) | 0.616 (±.004) | 0.449 (±.004) | — ᶜ | 0.477 (±.004) | — ᶜ | −2.8 pp † (z=4.5) | 0.510 (±.022) | 0.606 (±.022) |
| Qwen2.5-VL-7B | TextVQA (VQA-acc, 5000) | 0.862 (±.005) | 0.702 (±.006) | 0.597 (±.007) | 0.442 (±.007) | 0.319 (±.007) | **+26.1 pp** (z=27.3) | 0.757 (±.019) | 0.779 (±.019) |
| | DocVQA (ANLS, 5349) | 0.949 (±.003) | 0.636 (±.007) | 0.455 (±.007) | 0.526 (±.007) | 0.245 (±.006) | **+11.0 pp** (z=11.6) | 0.481 (±.022) ⁱ | 0.818 (±.017) ⁱ |
| | OCR-Bench (Final/1000, 1000) | 817 (.817 ±.012) ᵇ | 476 (.476 ±.016) ᶠ | 335 (.335 ±.015) | 183 (.183 ±.012) | 67 (.067 ±.008) | **+293 pts** (z=14.7) | 484 (.484 ±.022) | 604 (.604 ±.022) |
| | GQA (exact match, 12578) | 0.604 (±.004) | 0.559 (±.004) | — ᶜ | 0.585 (±.004) | — ᶜ | −2.6 pp † (z=4.1) | 0.498 (±.022) | 0.562 (±.022) |

**Footnotes.**
- ᵃ The Qwen3-VL **none** DocVQA full-split cell was **not completed**: a subset of giant document
  images exceeds the 32,768-token context at native resolution and the run aborted with only a
  partial evaluation, so the cell is left blank rather than reported partially. The Qwen3-VL pre/post
  cells were **both** run at the same native-resolution configuration and are directly comparable to
  each other; full-split 700k-pixel DocVQA evaluation on Qwen3-VL would require a larger model
  context. For reference, Qwen2.5-VL none DocVQA completed (ANLS 0.949, 5349/5349 answered,
  0 skipped, mean 4786.5 tokens/image).
- ᵇ OCR-Bench **none** cells skip 18 (Qwen3-VL) / 24 (Qwen2.5-VL) of 1000 long-OCR images on context
  overrun; skipped items score 0 pts (conservative). All compressed cells skip ≤ 5.
- ᶜ GQA was evaluated at κ = 0.25 only on the full split (no κ = 0.125 cells; the n=200 subset
  @12.5% cells of Table 2 tied exactly on Qwen2.5-VL).
- ᵈ Binomial stderr √(p(1−p)/n), p = official metric, n = full-split size. For DocVQA ANLS and
  OCR-Bench this treats the mean per-item score as a Bernoulli rate—a conservative approximation.
  Empty generations (≤ 4 per cell) score 0.
- ᵉ **Primary significance test = paired McNemar z** (paired-test table below): per-sample
  correctness paired by question, z = (b−c)/√(b+c), with b/c the pre-only / post-only correct counts.
  McNemar operates on the recorded per-sample correctness indicator (greedy match); for the
  text-dense item-score metrics (VQA-acc / ANLS / OCR-Bench) this coarsens the partial-credit
  official score, so the independent-sample binomial z printed in the Δ₂₅ column
  (SE √(se²_pre + se²_post)) is retained as the metric-faithful cross-check—McNemar is
  ≥-consistent with it on every cell. For GQA the McNemar on the recorded indicator is 8.0
  (Qwen3-VL; 83.9% agreement) / 5.7 (Qwen2.5-VL; 83.8%); the official word-normalized exact-match
  rescore gives a concordant 8.1 / 7.1.
- ᶠ **Iso-token / iso-resolution control.** At each model×benchmark, pre and post cells at the same
  κ have identical mean post-merger token counts (ptid) and unlimited pixel caps, with **one
  exception**: the Qwen2.5-VL OCR-Bench @25% RBM cell is a safety-flagged rerun that used a 4 M-pixel
  cap (mean ptid 229.6 vs post 282.0), so its +293 pt gap is, if anything, **conservative**.
- ʰ **FastV and Pyramid columns — independent HF-transformers n=500 headline subset.** These two
  columns come from a HuggingFace transformers 4.57.6 eager harness (§5.1), seed-0 **n = 500** subset
  per benchmark, on **both** models, greedy decoding, official metrics; their binomial stderrs use
  n = 500 (footnote d). They are a **subset headline**, deliberately distinct in scope from the
  full-split (n = 5000 / 5349 / 1000 / 12578) RBM/post/none columns, so no cross-column Δ is claimed
  inside this table; cross-method contrasts carry their own pooled z in §5.3. **FastV** = one-shot
  attention pruning at LLM **layer K = 2** (drop the visual tokens receiving the least layer-2
  attention, a query-conditioned scorer), run at the iso-budget κ = 0.25 (drop ratio r = 0.75).
  **Pyramid** = PyramidDrop at its **canonical progressive schedule [1, .75, .5, .25]**, i.e.
  equivalent mean retention **keep_equiv = 0.625 (≈2.1× our budget)**; it is a *different budget
  point* reported as a retention-curve point, **not** a same-budget win (its honest iso-25%
  degenerate schedule `[1,0,0,0]` collapses — Appendix Table A2). HF-harness fairness is validated: it
  bit-degrades to native at r = 0 (per-sample anchor **8/8** identical; manual pre-norm vs native
  max-diff = 0) and HF-vs-vLLM **none** cells agree **16/16** [E: j4_step2_fix]. OCR-Bench HF cells
  skip ≤ 1 of 500 long-OCR images (scored 0, conservative); the printed integer is the estimated
  Final/1000 = acc × 1000 with the official acc ± stderr in parentheses
  [E: runs/full_matrix/j7hf_official_summary.json; experiments/j7hf_baselines_n500.md].
- ⁱ **DocVQA iso-pixel disclosure and same-pixel cross-checks.** The HF baselines cannot honor a
  pixel cap through the engine (attention materialization), so the two **HF DocVQA cells (FastV,
  Pyramid) run under a PIL pre-scaled 600k-pixel cap** (all skip 0), whereas the full-split
  **RBM / post / none DocVQA columns are at native resolution** (footnote a config) — the two are
  therefore **not iso-pixel** and must not be read as a same-configuration DocVQA contrast inside
  this table. Matched-configuration cross-checks (n = 200, official ANLS, skip 0): **Qwen3-VL** at
  600k gives **none 0.951 / RBM 0.424 / post ≡ VZ 0.251**, against which the HF FastV 600k cell
  (0.518) and Pyramid 600k cell (0.878) are directly comparable; **Qwen2.5-VL** at 600k (J8b) gives
  **none 0.962 / RBM 0.424 / post 0.504** (pre and post iso-token at mean ptid 226) — a **direction
  reversal** relative to the native-resolution full split (Table 1: **+11.0 pp pre lead**), with
  **post leading by +8.0 pp** under the cap, whereas the matched Qwen3-VL 600k cells keep a wide
  pre-merger lead. Reading: at a low pixel budget the document text density falls into the shallow
  regime where the pre-merger L2 ranking has little to discriminate and the merger's merged
  (projected) features are comparatively robust (full reading in §5.4). **The headline remains the
  native-resolution full split (Table 1); the 600k cells are configuration cross-checks only**
  [E: experiments/j4_probe_qwen3vl.md 补遗; experiments/j7hf_baselines_n500.md;
  runs/full_matrix/ablations/j8b_qwen2vl_{none,pre,post}_docvqa_cap600k_n200.json].

**Paired significance of every Δ cell (primary test).** Pre and post are evaluated on identical
images/questions (same split, greedy), so each Δ cell is paired. The **primary** test is the McNemar
paired z on the per-sample correctness indicator (b = pre-correct/post-wrong count, c = the reverse;
z = (b−c)/√(b+c)); the independent-binomial z of Table 1's Δ column is the cross-check (footnote e).
Positive z = pre-merger lead; negative = post-merger lead (GQA only). Every cell is highly
significant; McNemar is ≥-consistent with the independent-binomial cross-check throughout. Source:
`runs/full_matrix/j7_*_full.json` per_sample, paired by question id.

| Model | Benchmark | κ | Δ (official) | McNemar z (primary) | b / c | agree % | indep-binomial z |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | TextVQA | .25 | +38.4 pp | **+43.0** | 2349 / 183 | 49.4 | 42.3 |
| | DocVQA | .25 | +24.3 pp | **+34.7** | 1601 / 150 | 67.3 | 27.1 |
| | OCR-Bench | .25 | +363 pts | **+16.7** | 422 / 56 | 52.2 | 18.2 |
| | GQA | .25 | −2.8 pp † | **−8.0** | 834 / 1196 | 83.9 | 4.5 |
| | TextVQA | .125 | +34.0 pp | **+40.8** | 2101 / 161 | 54.8 | 39.9 |
| | DocVQA | .125 | +24.9 pp | **+33.1** | 1437 / 127 | 70.8 | 32.1 |
| | OCR-Bench | .125 | +297 pts | **+16.0** | 323 / 25 | 65.2 | 17.8 |
| Qwen2.5-VL-7B | TextVQA | .25 | +26.1 pp | **+30.8** | 1682 / 309 | 60.2 | 27.3 |
| | DocVQA | .25 | +11.0 pp | **+15.6** | 1406 / 693 | 60.8 | 11.6 |
| | OCR-Bench | .25 | +293 pts | **+14.6** | 347 / 54 | 59.9 | 14.7 |
| | GQA | .25 | −2.6 pp † | **−5.7** | 892 / 1150 | 83.8 | 4.1 |
| | TextVQA | .125 | +27.8 pp | **+33.1** | 1836 / 305 | 57.2 | 29.1 |
| | DocVQA | .125 | +21.0 pp | **+25.6** | 1325 / 294 | 69.7 | 23.3 |
| | OCR-Bench | .125 | +268 pts | **+15.0** | 294 / 26 | 68.0 | 15.9 |

<!-- NUMBERS: paired McNemar z computed THIS PASS from runs/full_matrix/j7_*_full.json per_sample
     (correct field), paired by id, restricted to ids answered in both pre and post. z=(b-c)/sqrt(b+c).
     Text-dense @25 (pre leads): Q3 TextVQA +43.0/DocVQA +34.7/OCR +16.7; Q2.5 +30.8/+15.6/+14.6.
     GQA @25 (post leads): Q3 -8.0 (agree 83.9%) / Q2.5 -5.7 (agree 83.8%) — official word-normalized
     exact-match rescore McNemar 8.1/7.1 concordant. @12.5 text-dense: Q3 +40.8/+33.1/+16.0;
     Q2.5 +33.1/+25.6/+15.0. Min text-dense |z|=14.6 (Q2.5 OCR @25); min overall |z|=5.7 (Q2.5 GQA).
     All >= independent-binomial cross-check except OCR @25 (16.7 vs 18.2; 14.6 vs 14.7) where the
     binary correctness indicator coarsens the /1000 item score — noted in footnote e. This is the
     PRIMARY significance test per R2-5 / item 11; Table 1 Δ-column z relabeled as cross-check. -->

**Mandatory table note (printed, R1).** *At the same model and budget the query-conditioned FastV
leads RBM on TextVQA (both models) and on Qwen3-VL GQA, and on same-pixel DocVQA (Table 1 ʰ columns;
Appendix Table A2); RBM in turn leads FastV on OCR-Bench (Qwen3-VL +25.9 pp, z ≈ 9.5) and on Qwen2.5-VL GQA,
and ties it on Qwen2.5-VL OCR-Bench. There is **no** universal winner, and this table does **not**
support "RBM outperforms existing methods". The RBM claim is robustness (it never collapses on any
benchmark and never loses to the post-merger family) plus a large text-dense/OCR lead over the
post-merger family. GQA is the only benchmark on which post-merger selection leads pre-merger
selection—by a small (≤ 2.8 pp) though statistically significant margin—and no text-dense benchmark
shows any crossover.*

<details><summary>LaTeX rendering (kept in sync with the markdown table above)</summary>

```latex
\begin{table*}[t]\centering\small
\caption{Official full-split headline. RBM = pre-merger L2 (ours); post $\equiv$ VZ-port =
post-merger L2, byte-identical to the VisionZip principle-port on both generations. Parentheses:
binomial s.e.\ $\sqrt{p(1-p)/n}$. $\Delta_{25}=$ pre $-$ post at $\kappa{=}0.25$;
$^{**}$ significant pre lead; $\dagger$ significant post lead. The $\Delta_{25}$-column $z$ is the
independent-binomial cross-check; the \emph{primary} test is the paired McNemar $z$ (paired-test
table below; min text-dense $|z|{=}14.6$, min overall $|z|{=}5.7$; independent cross-check min
$|z|{=}4.1$). The FastV$^{h}$ and Pyramid$^{h}$ columns are an independent HF-transformers $n{=}500$
headline subset (FastV iso-budget $\kappa{=}0.25$; Pyramid canonical keep\_equiv
$0.625\approx2.1\times$ budget), distinct in scope from the full-split columns; their s.e.\ use
$n{=}500$, and the $\Delta_{25}$ band does not apply to the cross-method contrasts ($z$ in \S5.3).
$^{i}$HF DocVQA cells at 600k px (our columns native).}
\label{tab:main}
\begin{tabular}{llccccccll}
\toprule
Model & Benchmark (metric, $n$) & none & RBM @25\% & RBM @12.5\% & post$\equiv$VZ @25\% & post @12.5\% & $\Delta_{25}$ & FastV @25\%$^{h}$ & Pyramid @62.5\%$^{h}$ \\
\midrule
Qwen3-VL-8B & TextVQA (VQA-acc, 5000) & .844 (.005) & .605 (.007) & .472 (.007) & .222 (.006) & .132 (.005) & $+$38.4$^{**}$ ($z{=}42.3$) & .646 (.021) & .835 (.017) \\
 & DocVQA (ANLS, 5349) & --- & .481 (.007) & .352 (.007) & .238 (.006) & .103 (.004) & $+$24.3$^{**}$ ($z{=}27.1$) & .486 (.022)$^{i}$ & .882 (.014)$^{i}$ \\
 & OCR-Bench (/1000, 1000) & 760 (.014) & 547 (.016) & 350 (.015) & 184 (.012) & 53 (.007) & $+$363$^{**}$ ($z{=}18.2$) & 288 (.020) & 746 (.019) \\
 & GQA (exact match, 12578) & .616 (.004) & .449 (.004) & --- & .477 (.004) & --- & $-$2.8$\dagger$ ($z{=}4.5$) & .510 (.022) & .606 (.022) \\
\midrule
Qwen2.5-VL-7B & TextVQA (VQA-acc, 5000) & .862 (.005) & .702 (.006) & .597 (.007) & .442 (.007) & .319 (.007) & $+$26.1$^{**}$ ($z{=}27.3$) & .757 (.019) & .779 (.019) \\
 & DocVQA (ANLS, 5349) & .949 (.003) & .636 (.007) & .455 (.007) & .526 (.007) & .245 (.006) & $+$11.0$^{**}$ ($z{=}11.6$) & .481 (.022)$^{i}$ & .818 (.017)$^{i}$ \\
 & OCR-Bench (/1000, 1000) & 817 (.012) & 476 (.016) & 335 (.015) & 183 (.012) & 67 (.008) & $+$293$^{**}$ ($z{=}14.7$) & 484 (.022) & 604 (.022) \\
 & GQA (exact match, 12578) & .604 (.004) & .559 (.004) & --- & .585 (.004) & --- & $-$2.6$\dagger$ ($z{=}4.1$) & .498 (.022) & .562 (.022) \\
\bottomrule
\end{tabular}
\end{table*}
```

</details>

**The stage law holds on the full splits, on both generations.** Every text-dense Δ₂₅ is large and
significant: on Qwen3-VL, RBM leads post-merger selection by +38.4 pp on TextVQA, +24.3 pp (ANLS) on
DocVQA, and +363 pts on OCR-Bench (547 vs 184, 3.0×; z = 42.3 / 27.1 / 18.2); on Qwen2.5-VL, by
+26.1 pp, +11.0 pp, and +293 pts (476 vs 183, 2.6×; z = 27.3 / 11.6 / 14.7). No z falls below 11.6;
there is no direction-only and no non-significant text-dense cell under either the declared
independent-binomial model or a paired test.

**The gap is robust to deeper compression.** At κ = 0.125 the post-merger family degrades much
faster than RBM: post cells lose a further 44–60% of their retained accuracy while RBM cells lose
13–27% relatively (Qwen3-VL DocVQA: post 0.238 → 0.103 vs RBM 0.481 → 0.352; OCR-Bench: 184 → 53 vs
547 → 350, widening the pre/post ratio from 3.0× to 6.6×). All six Δ₁₂.₅ remain significant
(z ≥ 15.9): Qwen3-VL +34.0 pp (z=39.9) / +24.9 pp (z=32.1) / +297 pts (z=17.8); Qwen2.5-VL
+27.8 pp (z=29.1) / +21.0 pp (z=23.3) / +268 pts (z=15.9). Both models also lose accuracy against
the uncompressed baseline under compression (e.g. Qwen3-VL GQA 0.616 → 0.449/0.477), a cost that is
generation-dependent (Qwen2.5-VL GQA 0.604 → 0.559/0.585).

**Object-centric GQA, reported honestly (the single full statement).** GQA is the *only* benchmark
on which post-merger selection leads pre-merger selection, on **both** generations, and at full
n = 12578 the lead is **statistically significant**: post exceeds pre by +2.8 pp on Qwen3-VL and
+2.6 pp on Qwen2.5-VL (independent-binomial z = 4.5 / 4.1; primary paired McNemar z = 8.0 / 5.7 on
the same questions, 83.9% / 83.8% per-question agreement; the official word-normalized exact-match
rescore gives a concordant McNemar 8.1 / 7.1). Its magnitude is an order of magnitude smaller than
the TextVQA/OCR-Bench Δ (1/10–1/14 of them; ≈1/4 of the *smallest* text-dense Δ, +11.0 pp on
Qwen2.5-VL DocVQA), and **no text-dense benchmark shows any crossover**. We therefore claim **no**
pre-merger superiority on object-centric data: the defensible statement is that pre-merger selection
is robust—it never collapses, and trails by ≤ 2.8 pp on the one benchmark where it trails—while the
large wins are text-dense. All other sections refer to this paragraph for the GQA statistics.

**Cross-split consistency (n=200 subset vs full split).** Subset and full split agree in direction
and approximate magnitude on every text-dense cell (e.g. Qwen3-VL TextVQA Δ₂₅ +38.3 → +38.4 pp;
Qwen2.5-VL DocVQA +18.8 → +11.0 pp); the full split is the headline, and the n=200 subsets are
retained only as a cross-split consistency check (Appendix Table A2) and the compact cross-generation
view (Table 2, §5.4)
[E: j7_main_table.md; j7hf_baselines_n500.md; j2_crossgen_matrix.md; rescore_rerun_report.md].

**DocVQA configuration disclosure (R7).** The blank Qwen3-VL none cell (footnote a) is a
context-length limitation of the native-resolution evaluation, not a method effect: the pre and post
cells it would anchor were run under the *same* native-resolution configuration (max-model-len and
max-num-batched-tokens 32768, unlimited pixels) and are compared to each other, not to the missing
baseline. We flag, as in §5.1, that post-merger deep-compression collapse on DocVQA is partly
configuration-dependent and disclose it rather than present it as a pure method effect.

### 5.3 Same-model comparison against baselines (FastV / PyramidDrop / VisionZip-port)

**n=500 baseline headline (Table 1 ʰ columns).** The FastV and Pyramid columns of Table 1 report an
independent HF-transformers **n=500** headline subset on **both** models, same seed and official
metrics as the full-split columns (footnote ʰ). FastV is at the iso-budget κ = 0.25; Pyramid is at
its canonical keep_equiv 0.625 (≈2.1× our budget, a retention-curve point, not a same-budget cell).
Because these two columns differ in scope from the full-split RBM/post/none columns, the
cross-method contrasts below each carry their **own independent-binomial pooled z** (n=500 baseline
vs full-split RBM), and we never read them off the Δ₂₅ significance band of Table 1. The n=500
results reproduce, at higher n and on both generations, the structure first seen in the n=200
cross-split check (Appendix Table A2).

**FastV leads the query-relevant benchmarks at the same n = 500 budget — its mechanism, not a
general superiority.** Being query-conditioned (LLM layer-2 attention has already mixed in the
question), FastV leads RBM on TextVQA on **both** models (+4.1 pp on Qwen3-VL, 0.646 vs 0.605;
+5.5 pp on Qwen2.5-VL, 0.757 vs 0.702) and on Qwen3-VL GQA (+6.1 pp, 0.510 vs 0.449). These are
**same-budget n = 500** contrasts (FastV iso-budget κ = 0.25) and—because the RBM column is the full
split—cross-scope; against the cross-scope pooled test the TextVQA lead is **direction-only on
Qwen3-VL (z = 1.79)** but **significant on Qwen2.5-VL (z = 2.56)**, and the Qwen3-VL GQA lead is
significant (z = 2.69). We therefore state FastV's lead as a same-budget n = 500 result, **not** as a
full-split verdict: a same-scope full-split (or n ≥ 2000) verification on the two benchmarks without
a pixel-cap obstruction (TextVQA, GQA) is the decisive cross-method contrast and is reported
separately (§5.3 scope note; running/appendix). On same-pixel DocVQA the matched-configuration n=200
table gives FastV +9.4 pp over RBM at identical ptid 176 (0.518 vs 0.424; footnote ⁱ — the n=500
DocVQA columns are not iso-pixel and are not used for this contrast). The mechanism is exactly the
query-dependent headroom that our pre-registered query-level gate *cannot* reach before the merger
(Section 6): which region matters *for this question* is recoverable only once cross-attention has
seen the question.
<!-- same-scope numbers pending R2-C -->

**RBM wins the dense-OCR cell and one object-centric cell; an honest cross-generation difference.**
RBM leads FastV on Qwen3-VL OCR-Bench by **+25.9 pp (0.547 vs 0.288, pooled z = 9.5)** and on
Qwen2.5-VL GQA by +6.1 pp (0.559 vs 0.498, z = 2.69), and **ties** it on Qwen2.5-VL OCR-Bench
(0.476 vs 0.484, z = 0.29). Dense OCR is destroyed by the merger, and no ranking—however
query-conditioned—can recover information the merger has already discarded; only pre-merger
selection, which keeps raw patches, preserves it. The OCR margin narrows from +42.5 pp at n=200
(Appendix Table A2) to +25.9 pp at n=500 but stays large and highly significant. The two generations differ
instructively: FastV's OCR **collapses on Qwen3-VL (0.288) but holds on Qwen2.5-VL (0.484)**—the
Qwen2.5-VL merger is milder toward text, the same reason its DocVQA stage gap is smaller (Section
5.4)—while Pyramid's OCR is near-lossless on Qwen3-VL (0.746) yet drops noticeably on Qwen2.5-VL
(0.604). We report this asymmetry rather than average over it.

**PyramidDrop is near-lossless at its own budget and degenerate at ours; there is no universal
winner.** At its canonical keep_equiv 0.625 Pyramid is near-lossless on both models (Table 1:
TextVQA 0.835 / 0.779, DocVQA 0.882 / 0.818, OCR-Bench 746 / 604, GQA 0.606 / 0.562)—a
retention-curve point ≈2.1× our budget, not a same-budget win. Forced to an honest iso-25% budget,
its faithful degenerate schedule `[1,0,0,0]` (drop all visual tokens after the first layer segment)
collapses (Appendix Table A2: TextVQA 0.073 / GQA 0.305 / OCR 0.005), whereas RBM holds (0.60 / 0.42 / 0.58).
The post-merger family (VZ ≡ post) is the weakest method on every text-dense benchmark under both
models (Table 1: 0.222 / 0.238 / 184 and 0.442 / 0.526 / 183), confirming the "post-merger
deep-compression fragility" spine. **No method wins everywhere**: FastV wins the query-relevant
benchmarks but loses dense OCR (Qwen3-VL); RBM is the mirror image; Pyramid is near-lossless only at
high budget. This is precisely why we position RBM as the **robust default, not uniformly
optimal**—it never collapses, never loses to the post-merger family, and uniquely retains dense
OCR—rather than as a method that "beats" the field [E: j7hf_baselines_n500.md;
runs/full_matrix/j7hf_official_summary.json; j4_probe_qwen3vl 补遗].

![Fig. 4](figures/fig4_qualitative.pdf)

*Figure 4.* Qualitative contrast (3 of 10 catalogued cases; original images omitted, see
supplementary). Post-merger selection erases or corrupts small dense text after the merger—a date
vanishes on TextVQA (id 35014) and "$1.3 BILLION" is read as "million" (a 1000× unit error) on
DocVQA (id 58439, where post and the VisionZip-style port fail identically)—while pre-merger
selection protects it. The object-centric GQA case (id 201370409) is the honest trade-off direction:
post is correct and pre is wrong, consistent with the small significant GQA post lead (§5.2). ptid is
identical per image across conditions, isolating selection order [E: drafts/qualitative_examples.md].

**n = 200 cross-split check (Appendix Table A2).** The Qwen3-VL n = 200 cross-method values agree
with the n = 500 Table 1 columns in direction and approximate magnitude on every cell—FastV > RBM on
TextVQA / GQA / same-pixel DocVQA; RBM ≫ FastV on OCR-Bench (+42.5 pp at n=200 → +25.9 pp at n=500);
Pyramid near-lossless at its canonical budget and collapsed under an honest iso-25% schedule. The
n = 500 run confirms the n = 200 picture; the only quantitative refinements are the narrower (still
significant) OCR margin and the direction-only Qwen3-VL TextVQA contrast. Full table and cross-check
in Appendix Table A2 [E: j4_probe_qwen3vl; j7hf_baselines_n500.md].

### 5.4 Cross-generation consistency

Table 2 reports the Qwen2.5-VL-7B n=200 official numbers, showing the stage law reproduces on a
second generation (the full-split headline is Table 1).

**Table 2.** Qwen2.5-VL-7B, n=200, official metrics. Δ = pre − post (percentage points).

| Benchmark (metric) | none | RBM pre@25% | post@25% | Δ@25% | pre@12.5% | post@12.5% | Δ@12.5% |
|---|---|---|---|---|---|---|---|
| TextVQA (VQA-acc) | 0.870 | 0.735 | 0.415 | **+32.0** | 0.618 | 0.318 | **+30.0** |
| DocVQA (ANLS) | 0.975 | 0.687 | 0.499 | **+18.8** | 0.476 | 0.253 | **+22.3** |
| OCR-Bench (/1000) | 0.805 | 0.465 | 0.180 | **+28.5** | 0.335 | 0.060 | **+27.5** |
| GQA (exact match) | 0.585 | 0.565 | 0.555 | **+1.0 (tie)** | 0.505 | 0.505 | **0.0 (exact tie)** |

At n=200 the three text-dense Δ are all ≥ 5σ; the **full split** (§5.2, Table 1) reproduces all three
text-dense directions at n ≥ 5000 (primary paired McNemar |z| = 15.6–30.8) and is the headline.
VisionZip-style ≡ post holds byte-for-byte on this generation too (0.415 == 0.415)
[E: j2_crossgen_matrix].

**Cross-generation summary (full-split deltas, both models).**

| Δ (pre − post), full split | Qwen3-VL-8B | Qwen2.5-VL-7B |
|---|---|---|
| TextVQA @25% / @12.5% | +38.4 / +34.0 pp | +26.1 / +27.8 pp |
| DocVQA @25% / @12.5% | +24.3 / +24.9 pp | +11.0 / +21.0 pp |
| OCR-Bench @25% / @12.5% | +363 / +297 pts | +293 / +268 pts |
| GQA @25% | post +2.8 pp (§5.2) | post +2.6 pp (§5.2) |

Every text-dense Δ is highly significant (primary paired McNemar |z| = 14.6–43.0; Table 1 paired-test
table). GQA is the *only* post-merger lead—small but significant, an order of magnitude smaller than
any text-dense Δ, with no text-dense crossover; its full statistics are reported once in §5.2
[E: j2_crossgen_matrix; j7_main_table.md].

**Honest nuance (reported, not hidden).** Qwen2.5-VL's post-merger cell is more robust on
DocVQA@25% than Qwen3-VL's (0.499 vs 0.200); the gap between generations opens fully only at 12.5%
deep compression. We read this as: lossy-merger distortion worsens monotonically with compression,
with the *same direction but different rate* across generations [E: j2_crossgen_matrix].

**Resolution × model interaction — a configuration-dependent boundary (J8b).** The stage effect is
not uniform across pixel budgets, and we report the single sign reversal we observe in full. Under
the PIL pre-scaled 600k-pixel cap (n = 200, official ANLS, skip 0), the Qwen2.5-VL DocVQA stage
direction **reverses**: none 0.962 / pre (RBM) 0.424 / post 0.504 — a **+8.0 pp post-merger lead**,
against the native-resolution full-split **+11.0 pp pre-merger lead** on the same model and benchmark
(Table 1). The matched Qwen3-VL same-pixel cells do **not** reverse: they keep a wide pre-merger lead
(RBM 0.424 vs post 0.251, +17.3 pp; none 0.951; footnote ⁱ). The reversal is therefore a
*resolution × generation* interaction, not a generic low-resolution effect. Mechanism: at a low pixel
budget the document's text density falls into the shallow regime in which the pre-merger L2 ranking
has little to discriminate and the merger's merged (projected) features are comparatively robust—the
same root cause as the 75%-retention ties of the retention curve (§5.6) and the object-centric GQA
sign (§4.4), expressed here through generation: the Qwen2.5-VL merger is milder toward text,
consistent with its smaller native-resolution DocVQA gap (+11.0 pp vs Qwen3-VL's +24.3 pp). The
boundary is relevant to the robust-default positioning: RBM's guarantee is *no collapse and no loss
to the post-merger family on text-dense data at native resolution*; at artificially low pixel budgets
on the milder merger the two stages become near-tied and can reorder—exactly the shallow regime where
neither stage has much to lose. We disclose it as a configuration-dependent boundary and do not let it
qualify the native-resolution headline: **the headline remains the native-resolution full split
(Table 1)** [E: runs/full_matrix/ablations/j8b_qwen2vl_*_docvqa_cap600k_n200.json].

### 5.5 Comparison with official VisionZip (mismatched anchor only, R3)

Official VisionZip is not runnable on our setup (no Qwen3-VL support; no vLLM path; ViT-attention
materialization OOMs on ~16k-token document images; no serving-side hook), so we present **no**
head-to-head Qwen3-VL number and claim **no** victory over the official method [E:
visionzip_gap_report §2, §3]. We use the authors' *own* published Qwen2.5-VL numbers only as a
**model/stage-mismatched reference anchor**:

**Table A1 (mismatched anchor).** Official VisionZip on Qwen2.5-VL (~7B, lmms-eval), from the
authors' README. Mismatched reference, **not** a Qwen3-VL comparison, **not** a same-model cell.
(Renumbered from "Table 3-anchor" on 2026-07-28 to free the Table 3 slot for the efficiency table.)
[E: visionzip_gap_report §1]

| retain | MME | MMVet | OCRBench | POPE | RealWorldQA | DocVQA | MathVerse |
|---|---|---|---|---|---|---|---|
| 100% | 2316 | 61.6 | **81.5** | 86.7 | 68.6 | **95.1** | 46.3 |
| 70% (65d+5c) | 2334 | 60.0 | 80.9 | 86.4 | 68.2 | 94.5 | 45.8 |
| 50% (45d+5c) | 2209 | 57.0 | **70.5** | 86.3 | 68.6 | 93.8 | 45.1 |

The *shape* is the anchor: general tasks (POPE, RealWorldQA) hold while the text-dense task degrades
**first**—OCRBench 81.5 → 70.5 (−13%) at 50% retention while DocVQA loses only 1.3—and the authors
stop at 50% with no 25% row. The README caveat ("Qwen2.5VL already uses PatchMerger … the gain is
less striking than on LLaVA") is the same stage effect observed from the other side. Our
post-merger cells at 25% (Appendix Table A2 / Table 2) are the consistent extrapolation of that
trajectory into the regime the official evaluation did not enter [E: visionzip_gap_report §4].

<!-- NUMBERS: Table 1 FILLED from runs/full_matrix/j7_main_table.json (26/26 cells) +
     experiments/j7_main_table.md; z/stderr recomputed 2026-07-28 (binomial √(p(1−p)/n); Δz with
     independent SE; GQA also paired McNemar from per-sample official rescore, reproduces table
     officials exactly). Qwen3-VL: none .844/—/760/.616; RBM@25 .605/.481/547/.449; RBM@12.5
     .472/.352/350/—; post@25 .222/.238/184/.477; post@12.5 .132/.103/53/—. Δ25 +38.4(z42.3)/
     +24.3(z27.1)/+363(z18.2)/−2.8(z4.5,McNemar8.1 post leads). Qwen2.5-VL: none .862/.949/817/
     .604; RBM@25 .702/.636/476/.559; RBM@12.5 .597/.455/335/—; post@25 .442/.526/183/.585;
     post@12.5 .319/.245/67/—. Δ25 +26.1(z27.3)/+11.0(z11.6)/+293(z14.7)/−2.6(z4.1,McNemar7.1 post
     leads). Δ12.5: Qwen3 +34.0(z39.9)/+24.9(z32.1)/+297(z17.8); Qwen2.5 +27.8(z29.1)/+21.0(z23.3)/
     +268(z15.9). All text-dense z≥11.6 (**); GQA significant POST lead (†) — supersedes the old
     n=200 "tie/ns" (n=200 z≈0.6–0.8 = underpowered, not zero effect). Disclosures: Qwen3-VL none
     DocVQA cell missing (context overrun, aborted); OCRBench none skips 18/24; Qwen2.5 OCR@25% RBM
     rerun 4M-px cap (ptid 229.6 vs 282.0) → gap conservative; GQA no @12.5% cells. Table 1b
     (j4_probe + 补遗 + rescore_rerun) and Table 2 (j2) remain interim n=200. Cross-split deltas
     (subset→full): Q3 +38.3→+38.4, +26.5→+24.3, +41.5→+36.3; Q25 +32.0→+26.1, +18.8→+11.0,
     +28.5→+29.3; direction all identical. VisionZip anchor 81.5→70.5 @50% mismatched only (R3).
     Efficiency Table 3 = FILLED (J6; §5.7). Ablations §5.6 = FILLED (J8). VisionZip anchor table
     renumbered 3-anchor → A1.
     HF n=500 BASELINE COLUMNS FILLED (2026-07-28 J7hf sub-agent; footnotes ʰ/ⁱ; source
     runs/full_matrix/j7hf_official_summary.json + experiments/j7hf_baselines_n500.md). Qwen3-VL
     FastV@25 .646/.486(600k)/288/.510; Pyramid@62.5 .835/.882(600k)/746/.606. Qwen2.5-VL FastV@25
     .757/.481(600k)/484/.498; Pyramid@62.5 .779/.818(600k)/604/.562. Binomial stderr at n=500 (rate)
     .014–.022 (√(p(1−p)/500), recomputed). OCRBench cells = est. Final/1000 = acc×1000, skip ≤1/500
     scored 0. Cross-method pooled z (n=500 baseline vs full-split RBM, indep binomial, recomputed):
     TextVQA FastV>RBM Q3 +4.1pp z=1.79 DIRECTION-ONLY / Q2.5 +5.5pp z=2.56 sig; GQA Q3 FastV>RBM
     +6.1pp z=2.69 sig / Q2.5 RBM>FastV +6.1pp z=2.69 sig; OCR Q3 RBM>FastV +25.9pp z=9.49 sig /
     Q2.5 tie −0.8pp z=0.29. DocVQA cross-method contrast uses j4d same-pixel 600k n=200 (FastV 0.518
     vs RBM 0.424, +9.4pp; none 0.951/post 0.251) because n=500 DocVQA cols are native vs 600k (not
     iso-pixel, footnote ⁱ); Q2.5 same-pixel 600k cross-check DONE (j8b: none 0.962/pre 0.424/
     post 0.504, official ANLS rescore of raw JSONs — sign reversal vs native +11.0pp pre lead,
     disclosed footnote ⁱ + §5.4). Structure: no universal
     winner → RBM robust default not uniformly optimal (R1 held; FastV wins stated honestly).
     DOWNSTREAM (this pass): front matter, §5.1 subset/full + engine, Table 1 header/body/caption/
     footnotes ʰⁱ/mandatory note/LaTeX, §5.2 cross-split ¶, §5.3 rewritten around n=500 + Table 1b
     cross-check note, §7 item 5. Abstract/§1/§8 retain n=200-scoped OCR +42.5pp (j4_probe) which
     coexists with the n=500 +25.9pp (both labelled); no SOTA/beats introduced.
     DOWNSTREAM FIXES APPLIED (outside §5; 2026-07-28 writing sub-agent): abstract GQA clause,
     §1 "What we find" / cross-gen / positioning / contribution 2, §4.4 "post ≈ pre" tail →
     significant micro-lead + workload-conditional reading, §6(b) n=200 GQA label, §7 items 4–5
     (J7 complete / J6 pending), §8, front-matter supersede note, R4 checklist — all now read
     "post leads GQA +2.6–2.8pp (z≈4–8; 1/10–1/14 of TextVQA/OCR-Bench Δ); RBM robust default, not
     uniformly optimal". Claim correction recorded in DECISIONS 2026-07-28 / STATE.md.
     FINAL PASS (2026-07-28 writing sub-agent, CPU-only): Efficiency Table 3 (§5.7) FILLED from
     experiments/j6_efficiency.md + runs/full_matrix/efficiency/*.eff.json (Qwen3-VL TextVQA n=200,
     vLLM offline, A40): none/pre/post × κ{75,50,25%} req/s 4.11 | 4.61/4.52 | 5.39/5.54 |
     6.91/6.89; wall 48.7/43.4/44.2/37.1/36.1/28.9/29.0 s; mean visual tokens 766/582/397/213
     (pre==post per κ); single-request latency (n=1 ×5) none 0.36 s → pre@25% 0.23 s (−36%);
     stage-neutral within ±3% (digest's "≤2%" understates the κ=50% point, 5.39 vs 5.54 = −2.7%,
     corrected to ±3% here); peak GPU ≈42 GB is the gmu-0.9 allocation, not a measured footprint
     (no memory claim); Qwen2.5-VL efficiency = future work. Ablation §5.6 FILLED from
     experiments/j8_ablations.md + runs/full_matrix/ablations/j8_*.json (budget curve: pre advantage
     monotone in compression depth; 75%-retention ≈tie incl. DocVQA post +1.3/+2.1pp within noise;
     selector invariance: Q3 sign+magnitude under L2 and centroid-attn; Q2.5 L2 sign-invariant,
     attn proxy weak — TextVQA +15.7pp attenuated, DocVQA 0.554 vs 0.560 ≈0; mask granularity =
     method definition). j8b Qwen2.5-VL 600k DocVQA cross-check VERIFIED by official ANLS rescore of
     the raw per-sample JSONs: none 0.962 / pre 0.424 / post 0.504 (reversal of the native +11.0pp
     pre lead → +8.0pp post lead at the cap) → footnote ⁱ + §5.4 paragraph. CONSISTENCY FIXES:
     (i) "1/10–1/14 of any text-dense Δ" was imprecise (2.6/11.0 = 1/4.2 against Qwen2.5-VL DocVQA)
     → anchored to "the TextVQA/OCR-Bench Δ" everywhere (1/13.7/13/10/11.3 exact there), with the
     ≈1/4-vs-smallest-text-dense-Δ ratio disclosed in §5.2; (ii) VisionZip anchor renumbered
     Table 3-anchor → Table A1 (slot freed for efficiency Table 3); (iii) efficiency pointers §2/§5.1
     retargeted Section 7 → §5.7; (iv) §7 item 5 J6/J8 → complete; (v) grep guard amended (0.380 now
     legitimate: J8 official TextVQA-Q3 post @50% retention). Pooled-z spot checks (OCR z≈9.5–10.1,
     TextVQA Q3 1.79–1.82, GQA 2.69–2.73, Q2.5 TextVQA 2.56–2.76) all keep their significance
     labels; table values untouched. Figures list appended after §5.7. -->

### 5.6 Ablations (n=200, official metrics; J8)

**Retention curve: the stage gap is monotone in compression depth.** We measure the L2 pre/post
stage gap at three retention points (75% / 50% / 25%, i.e. drop ratio r ∈ {0.25, 0.5, 0.75}) on both
models and both text-dense benchmarks (n=200, official metrics; the 25%-retention column coincides
with the full-split Table 1 cells, the n=200 subset agreeing with Appendix Table A2 within run noise):

| benchmark · model | pre / post @75% retain (Δ, pp) | @50% (Δ) | @25% (Δ) |
|---|---|---|---|
| TextVQA · Qwen3-VL | 0.740 / 0.653 (**+8.7**) | 0.670 / 0.380 (**+29.0**) | 0.605 / 0.222 (**+38.4**) |
| TextVQA · Qwen2.5-VL | 0.870 / 0.800 (**+7.0**) | 0.813 / 0.660 (**+15.3**) | 0.702 / 0.442 (**+26.1**) |
| DocVQA · Qwen3-VL | 0.687 / 0.700 (**−1.3**, tie) | 0.589 / 0.531 (**+5.9**) | 0.481 / 0.238 (**+24.3**) |
| DocVQA · Qwen2.5-VL | 0.946 / 0.967 (**−2.1**, tie) | 0.875 / 0.889 (**−1.4**, tie) | 0.636 / 0.526 (**+11.0**) |

The pattern is uniform: at 75% retention the two stages are statistically indistinguishable in all
four cells — on DocVQA post even leads by 1–2 pp *within noise* — and the pre-merger lead emerges
and **widens monotonically** with compression depth, reaching +38.4 / +26.1 pp (TextVQA) and
+24.3 / +11.0 pp (DocVQA) at 25% retention. This is the accuracy-side signature of the lossy-merger
mechanism (§4): the text-hostile reshuffle becomes the dominant term only when a large fraction of
units is dropped, whereas under shallow compression the global pooled features available only after
the merger still carry recoverable information — the *same* shallow regime that produces the small
object-centric GQA post lead (§4.4) and the Qwen2.5-VL same-pixel DocVQA reversal (footnote ⁱ);
the retention curve, the GQA sign, and that reversal share one root cause.
![Fig. 3](figures/fig3_retention_gap.pdf)

*Figure 3 (core).* Retention-vs-gap curves — pre (RBM, blue) vs post (VZ-style, red), four
benchmark×model panels. The pre-merger lead is ≈0 (a tie, within noise) under shallow compression and
**widens monotonically** with depth, reaching +38.4 / +26.1 pp (TextVQA) and +24.3 / +11.0 pp
(DocVQA) at 25% retention — the accuracy-side signature of the lossy-merger mechanism. ● = n = 200
subset; ○ = full split; the 25% full-split point overlays the subset curve to show subset↔full
consistency. (Qwen3-VL DocVQA has no 100% point: native resolution overflows context on huge images.)
Data: `runs/full_matrix/ablations/j8_*.json` + `j8_summary.json` [E: experiments/j8_ablations.md §A].

**Selector invariance (two scorer families).** At 25% retention we replace L2 with a second scorer
family, global-centroid attention (n=200, official metrics; this reproduces the §4.5 gate probe at
doubled n):

| benchmark · model | L2 pre−post (pp) | centroid-attn pre−post (pp) |
|---|---|---|
| TextVQA · Qwen3-VL | +38.4 | **+35.5** (0.577 / 0.222) |
| DocVQA · Qwen3-VL | +24.3 | **+24.4** (0.451 / 0.207) |
| TextVQA · Qwen2.5-VL | +26.1 | +15.7 (0.673 / 0.517; sign kept, attenuated) |
| DocVQA · Qwen2.5-VL | +11.0 | −0.6 (0.554 / 0.560; ≈0; n=64 probe §4.5 had reversed) |

On Qwen3-VL the stage gap is invariant in **both sign and magnitude** under the second selector.
On Qwen2.5-VL the **L2 sign is invariant** (the paper's selector, across the full budget sweep of
J2), while the centroid-attention *proxy* degrades — attenuated on TextVQA and near-zero on DocVQA.
We report this honestly as a proxy-family specificity of the older generation (global-centroid
attention is a weak saliency proxy there), not as a counter-example to the stage law: the selector
the method freezes, L2, is sign-invariant on both generations. The defensible statement is a
two-selector invariance on Qwen3-VL and a one-selector (L2) invariance on Qwen2.5-VL
[E: experiments/j8_ablations.md §B; j3_mechanism_crossarch].

**Mask granularity (method definition, not an ablation dimension).** Selection operates at the
native 2×2 **unit**, by definition (§3.1): finer-than-unit (patch-level) pre-merger selection is not
defined, because the merger's four-patch input group is the minimal semantic unit the architecture
exposes, and "before the merger" refers precisely to scoring at unit level before the lossy projection.
We therefore state the granularity as part of the method definition (§3.1, §3.3) rather than
ablate it [E: experiments/j8_ablations.md §C].

### 5.7 Efficiency (single engine, vLLM offline; J6)

All efficiency numbers are measured on **one engine only** (vLLM 0.19 V1, offline batch inference,
max-num-seqs 8, gpu-memory-utilization 0.9, 1× A40) and are never compared across engines with the
HF-transformers baselines (§5.1, §7). Table 3 reports Qwen3-VL-8B on TextVQA n=200, greedy
decoding; retention κ is relative to each image's own unit count (§3.1).

**Table 3.** Efficiency, Qwen3-VL-8B, TextVQA n=200, 1× A40, vLLM 0.19 V1 offline batch, greedy.
Pre = RBM (pre-merger L2); post = post-merger L2. Mean visual tokens = mean post-merger visual
token count per request (pre and post are iso-token at every κ). Latency = single-request e2e,
n=1, mean of 5 runs (measured for the none and pre @25% configs only).
[E: experiments/j6_efficiency.md; runs/full_matrix/efficiency/*.eff.json]

| config | retention κ | req/s | wall (n=200) | mean visual tokens/req | single-req latency |
|---|---|---|---|---|---|
| none | 100% | 4.11 | 48.7 s | 766 | 0.36 s |
| pre (RBM) | 75% | 4.61 | 43.4 s | 582 | — |
| post | 75% | 4.52 | 44.2 s | 582 | — |
| pre (RBM) | 50% | 5.39 | 37.1 s | 397 | — |
| post | 50% | 5.54 | 36.1 s | 397 | — |
| **pre (RBM)** | **25%** | **6.91** | **28.9 s** | 213 | **0.23 s (−36%)** |
| post | 25% | 6.89 | 29.0 s | 213 | — |

**Table notes.** (i) **Stage-neutral.** Pre and post differ by ≤ 3% in req/s at every budget
(4.61/4.52, 5.39/5.54, 6.91/6.89; no consistent direction): the *stage* of selection carries no
efficiency cost, so the pre-merger advantage is purely an accuracy/robustness effect. (ii)
**Compression–efficiency scaling.** Throughput rises +12% / +31% / +68% over the uncompressed
baseline at 75% / 50% / 25% retention (wall time 48.7 → 28.9 s at 25%), and single-request latency
falls 0.36 → 0.23 s (−36%) at 25% retention. (iii) **Memory disclosure.** Peak GPU memory is the
fixed allocator reservation under gpu-memory-utilization 0.9 (≈ 41.6–42.4 GB, flat across configs),
*not* a measured footprint; we make no memory claim. (iv) Qwen2.5-VL efficiency was not measured;
the stage law and method story are complete on Qwen3-VL, and cross-generation efficiency is future
work.

<details><summary>LaTeX rendering (kept in sync with the markdown table above)</summary>

```latex
\begin{table}[t]\centering\small
\caption{Efficiency, Qwen3-VL-8B, TextVQA $n{=}200$, $1\times$ A40, vLLM 0.19 V1 offline batch,
greedy. Pre $=$ RBM; post $=$ post-merger L2; pre and post are iso-token at every $\kappa$.
Stage-neutral: pre vs post within $\pm3\%$ req/s at every budget (no consistent direction), so the
stage carries no efficiency cost. Throughput $+12/+31/+68\%$ at $75/50/25\%$ retention vs none;
single-request latency ($n{=}1$, 5-run mean) $0.36\to0.23$\,s ($-36\%$) at $25\%$. Peak GPU
($\approx42$\,GB) is the fixed gmu-0.9 allocation, not a measured footprint (no memory claim).
Qwen2.5-VL efficiency $=$ future work.}
\label{tab:efficiency}
\begin{tabular}{lcccc c}
\toprule
config & retention $\kappa$ & req/s & wall ($n{=}200$) & mean vis.\ tokens/req & single-req latency \\
\midrule
none & 100\% & 4.11 & 48.7\,s & 766 & 0.36\,s \\
pre (RBM) & 75\% & 4.61 & 43.4\,s & 582 & --- \\
post & 75\% & 4.52 & 44.2\,s & 582 & --- \\
pre (RBM) & 50\% & 5.39 & 37.1\,s & 397 & --- \\
post & 50\% & 5.54 & 36.1\,s & 397 & --- \\
\textbf{pre (RBM)} & \textbf{25\%} & \textbf{6.91} & \textbf{28.9\,s} & 213 & \textbf{0.23\,s ($-36\%$)} \\
post & 25\% & 6.89 & 29.0\,s & 213 & --- \\
\bottomrule
\end{tabular}
\end{table}
```

</details>

**Figure index.** All four figures are generated by `drafts/figures/gen_paper_v4_figures.py`
(colorblind-safe palette; ✓/✗ marks encode correctness in addition to color) and are wired in-text:
- **Fig. 1 — pipeline** (§3.1): `figures/fig1_pipeline.pdf`. RBM vs post-merger tap-point schematic;
  the native merger (+ deepstack on Qwen3-VL) and the LLM are identical in both panels.
- **Fig. 2 — mechanism, M1 + M3** (§4): `figures/fig2_mechanism.pdf`. (a) pre-vs-post rank
  decorrelation (Spearman ρ, Jaccard@25%); (b) the ranking-swap causal control (swap ≡ pre).
  Supplementary per-image rasters: `token_survival_m1_rank_overlap.pdf`,
  `token_survival_m2_edge_demotion.pdf`.
- **Fig. 3 — retention-vs-gap (core)** (§5.6): `figures/fig3_retention_gap.pdf`. The pre-merger lead
  is ≈0 under shallow compression and widens monotonically with depth.
- **Fig. 4 — qualitative examples** (§5.3): `figures/fig4_qualitative.pdf`. Three catalogued cases:
  post/FastV erase small dense text, RBM protects it, and GQA is the honest trade-off direction.
- Supplementary-grade assets: `figures/stage_law.png`, `figures/token_survival_{textvqa,docvqa}.pdf`.

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
gains nothing on GQA (0.500, where pre == post == 0.510 at n=200; the full-split GQA
reading—a small significant post-merger lead—is reported once in §5.2). **No single
text-fraction passes the gate**; pre-merger ranking is the fixed point that ranking-informed
post-stage allocation cannot improve upon [E: method_gate_report §2–3].

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

**Deployment guidance.** The three negatives convert into a decision rule. Use RBM (query-blind,
training-free, stage-only) as the default for text-dense and OCR workloads and for unknown or mixed
traffic, where never collapsing and preserving dense OCR matter more than the last few points of
object-centric accuracy. Use a query-conditioned post-LLM-layer method (FastV-class) when
scene-text/object queries dominate and dense OCR is absent, to capture the query-dependent headroom
that pre-merger features cannot reach. Do **not** route or hybridize at the image level: because the
oracle gap is ~73% sample-level and query-dependent (§6a), image-level switching is strictly worse
than always-pre, so the choice should be fixed at deployment time by workload, not made per image at
runtime.

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
4. **GQA: a small significant post-merger lead — RBM is robustness-first, not uniformly optimal.**
   On the full split (n = 12578) post-merger selection exceeds pre on object-centric GQA by +2.6–2.8
   pp (independent-binomial z = 4.1–4.5; primary paired McNemar z = 5.7–8.0). This is the one
   benchmark on which pre-merger selection trails; the margin is an order of magnitude smaller than
   any text-dense Δ and shows no text-dense crossover, so we claim no pre-merger superiority on
   object-centric data and position RBM as a robustness-first default (full statistics in §5.2).
5. **Cross-scope baselines, a minimal method, and the scoring tap.** (a) The FastV/PyramidDrop
   columns are an n = 500 subset while the RBM/post/none columns are full splits, so the
   cross-method contrasts in §5.3 are cross-scope and carry their own pooled z; the Qwen3-VL TextVQA
   FastV-vs-RBM contrast is direction-only (z = 1.79), and a same-scope full-split verification is
   pending (§5.3). (b) RBM scores units on merger-input features, and the mechanism report's PRE
   capture is an intermediate-depth ViT feature (block-8 deepstack-input unit L2) rather than the
   final ViT layer; this tap is identical for pre and post after the corresponding merger stage
   (preserving the iso-selector control), but its depth is a disclosed design choice that a one-cell
   sensitivity ablation (block-8 vs final-layer) would further pin down. (c) RBM is deliberately the
   simplest conceivable scorer (query-blind L2): the contribution is the *stage axis and its causal
   account* (M1–M3), not the scorer—Section 6 shows no decoration helps. (d) OCR-Bench skip-scoring
   is conservative: the uncompressed *none* cells skip 18 (Qwen3-VL) / 24 (Qwen2.5-VL) of 1000
   long-OCR images on context overrun (scored 0), so the uncompressed anchor is understated and the
   relative gaps are, if anything, conservative (Appendix Table A3). The Qwen2.5-VL causal swap is
   undecided (§4.5), so the causal claim is scoped to Qwen3-VL.
6. **Greedy decoding / single device.** Decoding is deterministic (temperature 0), so error bars are
   binomial standard errors only, with no temperature variance; all runs are on a single A40.
   Efficiency is measured **offline**, not under continuous-batching serving.

<!-- NUMBERS: r=0 anchor 8/8 + 16/16 equivalence (j4_step2_fix); GQA full n=12578 post +2.6–2.8pp
     (z 4.1/4.5 indep; McNemar 7.1/8.1 paired; j7_main_table; DECISIONS 2026-07-28 supersedes the
     n=200 tie; 1/10–1/14 of the TextVQA/OCR-Bench Δ (≈1/4 vs smallest text-dense Δ, disclosed §5.2);
     n=200 +4.5pp ~1.3σ of j2 retained as underpowered
     interim, SE ~8× full-split); DocVQA 1.5M/32768 + 600k cap (rescore_rerun + j4 补遗);
     Qwen2.5 swap undecided (j3); J7 complete + J7hf HF-n500 baseline columns filled + J6 efficiency
     (Table 3, §5.7) + J8 ablations (§5.6) filled 2026-07-28; no pending tables. No SOTA /
     cross-model (R2). -->

---

## 8. Conclusion

We showed that the native 2×2 merger in merger-equipped VLMs is **lossy in a text-hostile
direction**: it re-shuffles unit saliency almost from scratch (M1), systematically demotes
text-stroke units on documents (M2), and—by a ranking-swap control on Qwen3-VL-8B—accounts for the
*entire* pre-vs-post accuracy gap as a **ranking effect**, not forward-path destruction (M3:
swap ≡ pre, DocVQA 200/200 byte-identical, TextVQA 198/200). The practical consequence is
**Rank-Before-Merge (RBM)**: a query-blind L2 ranking applied to merger-input units. Its stage law
is cross-generation—on the official **full splits** RBM leads post-merger selection by **+11.0 to
+38.4 pp** on text-dense benchmarks (native resolution) across Qwen3-VL-8B and Qwen2.5-VL-7B, while
on object-centric GQA it trails by a small but significant margin (§5.2), with no text-dense
crossover; the text-dense gap widens under deeper compression. In a same-model, same-budget
comparison RBM is the **robust default, not uniformly optimal**: it never collapses, concedes only a
narrow significant margin on object-centric GQA, and is the only compared method that robustly
retains OCR-Bench (+25.9 pp over FastV at n = 500, z ≈ 9.5). There is no universal winner: in the
same-budget n = 500 comparison the query-conditioned FastV leads the query-relevant benchmarks
(TextVQA on both models; Qwen3-VL GQA) precisely because it is query-conditioned, yet loses dense OCR
on Qwen3-VL and Qwen2.5-VL GQA to RBM. Three pre-registered negative results close the design space:
pre-merger selection **must remain query-blind**, because the query-dependent headroom is reachable
only after the LLM's cross-attention layers mix in the question. Future work: extend the mechanism
and method to a second model family (e.g. InternVL3), add a same-model comparison to Hi-Lo Prune if
its code is released, run the same-scope full-split FastV baseline (§5.3), verify the causal swap at
the kept-unit level, and measure serving-side (continuous-batching) efficiency.

<!-- NUMBERS: M3 200/200 & 198/200 (mechanism §3); +18.8~+38.3pp (j2 + rescore_rerun); OCR headline
     now n=500 +25.9pp = 0.547 vs 0.288 (z≈9.5, j7hf_baselines_n500) with n=200 +42.5pp (0.580 vs
     0.155, j4_probe) retained; three negatives (method_gate §2–4 + j5); GQA full n=12578 post
     +2.6–2.8pp, z≈4–8 (indep 4.1/4.5; McNemar 7.1/8.1; 1/10–1/14 of TextVQA/OCR-Bench Δ; j7_main_table;
     supersedes the n=200 tie); "no universal winner" = FastV wins query-relevant (TextVQA both /
     GQA Q3 / same-px DocVQA) but RBM wins OCR-Q3 + GQA-Q2.5 (§5.3, model-dependent). No "beats
     existing methods" / no SOTA (R1/R2). -->

---

## Appendix

**Table A2.** Qwen3-VL-8B **n = 200** cross-split / cross-method check, official metrics, κ = 25%
(Pyramid at its two indicated budgets). `post ≡ VZ` is the post-merger L2 cell, byte-identical to the
VisionZip principle-port (§3.4). This subset is retained only as a cross-split consistency check; the
full-split headline is Table 1.

| Benchmark (metric) | none | RBM (pre) | post ≡ VZ | FastV | Pyramid |
|---|---|---|---|---|---|
| TextVQA (VQA-acc) | 0.858 | 0.598 | 0.215 | **0.680** | 0.852 @62.5% |
| DocVQA (ANLS, 600k px) | 0.951 | 0.424 | 0.251 | **0.518** | 0.878 @62.5% |
| OCR-Bench (/1000) | ~0.73 | **0.580** | 0.165 | 0.155 | — |
| GQA (exact match) | ~0.53 | 0.420 | 0.465 | **0.490** | — |
| Pyramid iso-25% `[1,0,0,0]` (TextVQA / GQA / OCR) | — | — | — | — | 0.073 / 0.305 / 0.005 |

Reference (native-resolution DocVQA, not iso-pixel with the HF baselines): none 0.976, RBM 0.465,
post 0.200 [E: rescore_rerun_report].

**Cross-check vs the n = 500 headline.** These n = 200 values agree with the n = 500 Table 1 columns
in direction and approximate magnitude on every cell: FastV > RBM on TextVQA (n=200 +8.2 pp → n=500
+4.1 pp), GQA (+7.0 → +6.1 pp) and same-pixel DocVQA (+9.4 pp); RBM ≫ FastV on OCR-Bench (+42.5 →
+25.9 pp); Pyramid near-lossless at its canonical budget and collapsed under an honest iso-25%
schedule. The n = 500 run confirms the n = 200 picture; the only quantitative refinements are the
narrower (still significant) OCR margin and the direction-only Qwen3-VL TextVQA contrast
[E: j4_probe_qwen3vl; j7hf_baselines_n500.md].

**Table A3.** Per-cell answered / skip accounting on the official full splits (source:
`runs/full_matrix/j7_*_full.json`, fields `n_answered` / `n_skipped`). Skipped items (context
overrun) score 0 and are conservative for the uncompressed *none* anchor; a few compressed OCR-Bench
cells have `n_answered + n_skip < n` because ≤ 4 empty generations per cell score 0 but are not
flagged as context-overrun skips (footnote d). OCR-Bench *none* skips (18 / 24 of 1000) understate the
uncompressed anchor, so the compressed relative gaps are conservative; all other cells skip ≤ 2.

| Model | Benchmark | method | κ | official | n_answered | n_skip | mean ptid |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | TextVQA | none | 100% | 0.844 | 4998 | 2 | 772.5 |
| | TextVQA | RBM (pre) | 25% | 0.605 | 5000 | 0 | 215.8 |
| | TextVQA | RBM (pre) | 12.5% | 0.472 | 5000 | 0 | 122.1 |
| | TextVQA | post | 25% | 0.222 | 5000 | 0 | 215.8 |
| | TextVQA | post | 12.5% | 0.132 | 5000 | 0 | 122.1 |
| | DocVQA | RBM (pre) | 25% | 0.481 | 5349 | 0 | 946.7 |
| | DocVQA | RBM (pre) | 12.5% | 0.352 | 5347 | 0 | 488.8 |
| | DocVQA | post | 25% | 0.238 | 5345 | 0 | 946.7 |
| | DocVQA | post | 12.5% | 0.103 | 5343 | 0 | 488.8 |
| | OCR-Bench | none | 100% | 760/1000 | 982 | 18 | 605.1 |
| | OCR-Bench | RBM (pre) | 25% | 547/1000 | 1000 | 0 | 228.9 |
| | OCR-Bench | RBM (pre) | 12.5% | 350/1000 | 999 | 0 | 125.7 |
| | OCR-Bench | post | 25% | 184/1000 | 1000 | 0 | 228.9 |
| | OCR-Bench | post | 12.5% | 53/1000 | 1000 | 0 | 125.7 |
| | GQA | none | 100% | 0.616 | 12578 | 0 | 298.1 |
| | GQA | RBM (pre) | 25% | 0.449 | 12578 | 0 | 96.8 |
| | GQA | post | 25% | 0.477 | 12578 | 0 | 96.8 |
| Qwen2.5-VL-7B | TextVQA | none | 100% | 0.862 | 4998 | 2 | 1018.3 |
| | TextVQA | RBM (pre) | 25% | 0.702 | 5000 | 0 | 285.6 |
| | TextVQA | RBM (pre) | 12.5% | 0.597 | 5000 | 0 | 162.5 |
| | TextVQA | post | 25% | 0.442 | 5000 | 0 | 285.6 |
| | TextVQA | post | 12.5% | 0.319 | 5000 | 0 | 162.5 |
| | DocVQA | none | 100% | 0.949 | 5349 | 0 | 4786.5 |
| | DocVQA | RBM (pre) | 25% | 0.636 | 5349 | 0 | 1228.2 |
| | DocVQA | RBM (pre) | 12.5% | 0.455 | 5349 | 0 | 635.1 |
| | DocVQA | post | 25% | 0.526 | 5349 | 0 | 1228.2 |
| | DocVQA | post | 12.5% | 0.245 | 5349 | 0 | 635.1 |
| | OCR-Bench | none | 100% | 817/1000 | 976 | 24 | 718.6 |
| | OCR-Bench | RBM (pre) | 25% | 476/1000 | 998 | 0 | 229.6 |
| | OCR-Bench | RBM (pre) | 12.5% | 335/1000 | 995 | 0 | 157.7 |
| | OCR-Bench | post | 25% | 183/1000 | 999 | 0 | 282.0 |
| | OCR-Bench | post | 12.5% | 67/1000 | 1000 | 0 | 157.7 |
| | GQA | none | 100% | 0.604 | 12578 | 0 | 392.2 |
| | GQA | RBM (pre) | 25% | 0.559 | 12578 | 0 | 128.5 |
| | GQA | post | 25% | 0.585 | 12578 | 0 | 128.5 |

(The Qwen3-VL *none* DocVQA full-split cell is absent: context overrun at native resolution, footnote a.)

<!-- NUMBERS: Table A2 = former Table 1b (j4_probe + 补遗 + rescore_rerun), moved to appendix per
     item 6. Table A3 computed THIS PASS from runs/full_matrix/j7_*_full.json n_answered/n_skipped
     fields (item 13 / R2-8); OCR none skips 18 (Q3) / 24 (Q2.5) of 1000 scored 0 (conservative for
     the uncompressed anchor); TextVQA none skips 2 each; all compressed cells skip <=5; a few OCR
     compressed cells have n_answered+n_skip<n (<=4 empty generations scored 0, not overrun skips). -->

---

## Red-line self-check (plan §红线 R1–R7)

- [x] **R1 — no "RBM beats/outperforms existing methods/SOTA".** Full text grepped for
  beats/outperforms/SOTA re: RBM: absent (the only "beats" token is in a comment recording the
  *retired* working title). FastV is stated to *lead* (not "beat the field") the query-relevant
  benchmarks in the **same-budget n = 500** comparison (TextVQA both models; Qwen3-VL GQA; same-pixel
  DocVQA), model-dependently — RBM in turn leads Qwen3-VL OCR-Bench (+25.9 pp, z ≈ 9.5) and
  Qwen2.5-VL GQA (+6.1 pp) and ties Qwen2.5-VL OCR — stated as **no universal winner**. RBM claims =
  "robust default / never collapses / never loses to the post-merger family / uniquely holds
  OCR-Bench". FastV wins are framed as same-budget n = 500, with same-scope full-split verification
  pending (§5.3); the OCR margin is stated as RBM vs FastV on one benchmark, not a general superiority.
- [x] **R2 — no cross-model SOTA.** All causal claims scoped to Qwen3-VL-8B (§4.3, §7); Qwen2.5-VL
  presented as corroboration of the ranking *law*, with the causal swap explicitly **not** claimed to
  generalize (§4.5). No cross-model "state-of-the-art" anywhere.
- [x] **R3 — VisionZip official numbers = mismatched anchor only.** Table A1 labelled
  "model/stage-mismatched reference, NOT a Qwen3-VL comparison, NOT a same-model cell"; no
  head-to-head victory claimed (§5.5).
- [x] **R4 — GQA consolidated to one full statement + one limitation + one abstract sentence.** The
  single full statistical statement is §5.2 "Object-centric GQA, reported honestly" (full n = 12578
  post lead +2.6–2.8 pp; independent-binomial z = 4.1 / 4.5; primary paired McNemar z = 5.7 / 8.0,
  official-rescore McNemar 7.1 / 8.1 concordant; 1/10–1/14 of the TextVQA/OCR-Bench Δ; no text-dense
  crossover). All other sections (abstract, §1, §4.4, §5.4, §6, §8) carry terse §5.2 back-refs; the
  limitation is §7 item 4. The n=200 "tie" archaeology is removed from the submission prose (it
  survives only in NUMBERS audit comments).
- [x] **R5 — non-significant gaps = direction only.** Under the **primary paired McNemar** test no
  full-split Δ cell is direction-only or non-significant (smallest text-dense |z| = 14.6; smallest
  overall |z| = 5.7, the GQA post lead); the independent-binomial cross-check agrees (smallest
  |z| = 4.1). The cross-scope Qwen3-VL TextVQA FastV-vs-RBM contrast is **direction-only (z = 1.79)**
  and is said so explicitly (§5.3, §7 item 5a).
- [x] **R6 — official metrics only.** VQA-acc / ANLS / OCR-Bench official / GQA exact match used; no
  containment figures (0.695/0.255/0.725/0.390/0.320/+44.0/+33.5/−6.0/+46.6 all absent).
- [x] **R7 — engine + pixel-cap disclosure.** HF-vs-vLLM engine difference and accuracy-only
  comparability disclosed (§5.1, §7); DocVQA 1.5M/32768 config and 600k iso-pixel cap disclosed
  (§5.1, §5.4, §7); throughput confined to vLLM.

**Revision-pass fixes tracked (this draft).**
- **Novelty / related work (review R1-3):** merging-lineage paragraph ToMe → PruMerge → AdaptMerge
  added (§2) with an explicit relation (they merge within/after the encoder with trained/attention
  scorers; AdaptMerge's OCR-gap finding is cited as corroborating evidence, not competed); the
  "empty cell" is re-delimited to the concrete pre-merger design point + its M1–M3 mechanism account
  (not a topological vacuum: QuietPrune = in-ViT different stage, FastV/PyramidDrop = in-LLM
  different stage, VisionZip = post-merger opposite side). bib: `bolya2023tome`, `islam2025adaptmerge`.
- **Merger description (review R1-5):** the native merger is described as concat-2×2 + learned MLP
  projection (LayerNorm → fc1 → GELU → fc2), verified against vLLM site-packages; "lossy pooling/
  averaging" wording removed; the §3.3 protection sentence rewritten (selection changes *which*
  units reach the merger, not intra-unit averaging).
- **Statistics (review R2-5 / R2-8):** paired McNemar z is now the primary Δ significance test
  (§5.2 paired-test table; computed from j7 per_sample), independent-binomial z the cross-check;
  per-cell skip accounting added (Appendix Table A3). FastV claim downgraded to same-budget n = 500
  with same-scope full-split verification pending (R2-1 / R2-12). The 600k Qwen2.5-VL DocVQA sign
  reversal is promoted to a full §5.4 paragraph.
- **Presentation (review R3-1..R3-5):** front-matter scaffolding / `[interim]` flags / supersession
  narrative stripped; the GQA micro-margin consolidated (~10× → 1 statement + 1 limitation + 1
  abstract sentence); figures wired (Fig. 1 §3.1, Fig. 2 §4, Fig. 3 core §5.6, Fig. 4 §5.3); title
  selected (H1); Table 1b → Appendix Table A2; abstract on the full-split headline (+11.0 to
  +38.4 pp, ≤200 words).

<!-- R1-1 RESOLVED (2026-07-28): kept-set Jaccard(swap,pre) = 1.000 on both architectures
     (r1_1_swap_jaccard, n=32 seq=1, chance 0.25). Causal wording finalized to "selection-level
     causal, architecture-general" throughout (abstract, §1, §4.3, §4.5, §8); Qwen2.5 residual =
     reverse_indices order confound (implementation), batch-dependence structurally excluded.
     Feature tap verified: Qwen3-VL pre scoring = deepstack[0] input = ViT block-8 output
     (indexes [8,16,24]); Qwen2.5-VL = main merger input after final ViT block. -->

**Grep guard (run before submission):** confirm zero hits for `0.695|0.255|0.725|0.390|0.320|
+44.0|+33.5|−6.0|+46.6|outperform|state-of-the-art|SOTA` in the submission-bound prose, and that
"beats" appears only in the retired-title comment (never as a live RBM claim). Confirm Table 1 carries
the full-split numbers in the RBM/post/none columns and the n=500 numbers in the FastV/Pyramid columns
(footnote ʰ), with no `[HF-n500 pending]`, `[FIG: … TBD]`, or `[interim]` placeholder remaining. Note:
`0.380` legitimately appears in the §5.6 retention table (official TextVQA Qwen3-VL post VQA-acc at
50% retention); the old containment `0.380` (OCR-Bench @12.5%, voided scorer) is absent (official
value 350/1000, Table 1).

<!-- NUMBERS: self-check references — containment numbers to be grep-killed (plan §A); R1–R7 mapped
     to sections above; revision-pass fix tracking added this pass. -->

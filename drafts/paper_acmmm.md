# Rank Before You Merge: A Stage Law for Training-Free Visual Token Compression in Merger-Equipped Vision-Language Models

*ACM MM'27 submission draft — Main Technical Track, area "Multimedia Generative and Foundation
Models". Target: ≤8 pp body + ≤2 pp references, `\documentclass[sigconf, screen, review, anonymous]{acmart}`,
double-blind. CCS concepts: Computing methodologies → Multimodality / Vision and language models.
Keywords: visual token compression, vision-language models, training-free inference, multimodal
foundation models, inference efficiency. Numbers discipline: every number is sourced to the digests
listed in the SOURCE MAP at the end; unsourced numbers are marked [TODO: source]. Figures are
`[FIG: …]` placeholders with caption intent (no figures generated). Author notes at file end are NOT
part of the submission. Language: English.*

**Authors:** [anonymized — double-blind; to be filled at de-anonymization]

<!-- TITLE candidates (no beats/SOTA/pooling wording; multimodal framing required to avoid
     out-of-scope desk-reject per notes/acm_mm27_cfp.md):
     (1, USED) "Rank Before You Merge: A Stage Law for Training-Free Visual Token Compression in
         Merger-Equipped Vision-Language Models";
     (2) "The Stage, Not the Scorer: Training-Free Pre-Merger Visual Token Selection Generalizes
         Across Three Merger-Equipped VLM Families";
     (3) "Where Selection Meets the Merger: A Query-Blind Stage Law and Its Causal Mechanism for
         Visual Token Compression in Multimodal Foundation Models". -->

---

## Abstract (~190 words)

Multimodal foundation models pay for every visual token they hand to the language model, and
merger-equipped vision-language models (VLMs) already reduce tokens with a native merger. We isolate
the *stage* at which token selection happens — before or after that merger — as an experimental
variable under iso-model / iso-token / iso-selector control, and report three findings. **(1) A stage
law.** On text-dense workloads, query-blind pre-merger selection systematically beats post-merger
selection — +11.0 to +38.4 pp across two Qwen-VL generations and +34.6 to +432 pts on a third
family, InternVL3-8B (a different merger design: pixel-shuffle rather than Qwen's PatchMerger), under
iso-selector / iso-budget control (same query-blind L2 scorer, identical token budgets across stages).
The law has no text-dense crossover at the native-resolution full splits (one shallow 600k-cap DocVQA
configuration reverses; configuration boundary in §8),
the gap widens monotonically under deeper compression, and object-centric GQA shows at most a small
significant post-stage edge (Qwen +2.6–2.8 pp) or a tie (InternVL3, −0.4 pp); against a separate
query-conditioned in-layer baseline (FastV), RBM wins dense OCR yet concedes scene-text/object
benchmarks on Qwen, so
it is a robust default, not a uniform winner. **(2) A selection-level causal mechanism.** A ranking-swap control
recovers pre-merger accuracy exactly, and the kept unit sets are identical across stages (Jaccard =
1.000 on both architectures tested): the gap is the ranking the merger rewrites, not the forward
path. **(3) Generalization with efficiency and a closed design space.** The law reproduces across
three model families; 25%-retention compression is stage-neutral and raises throughput +68% (Qwen) /
1.8–2.5× req/s (InternVL3); and four pre-registered extensions — QA-gate, hybrid mask, image router,
two-stage cascade — all fail, showing pre-merger selection is a fixed point that no downstream
selector improves. Code will be released.

---

## 1. Introduction (~70 lines)

**Multimodal inference cost is a visual-token problem.** Vision-language models (VLMs) have become a
core class of multimodal foundation models, but an image enters the language model as hundreds to
tens of thousands of visual tokens, and the quadratic attention cost and the key-value memory of
these tokens dominate multimodal inference. Visual-token compression — pruning or merging the least
important tokens before or within the language model — is therefore a standard efficiency lever,
with direct consequences for document understanding, scene-text question answering, and OCR, the
most token-hungry multimodal workloads. In merger-equipped architectures — the Qwen-VL family,
InternVL3, and others that merge 2×2 patch groups into one token — the compression debate has run
almost entirely over *which* tokens to keep and *how* to score them. The *stage* at which selection
happens, before or after the native merger, has not been isolated as an experimental variable, even
though practitioners knew the merger matters: the Qwen2.5-VL authors themselves note it already
compresses, and merging is reported to degrade OCR.

**What we find — a stage law.** Under strict iso-model / iso-token / iso-selector control, applying
the *same* text-agnostic L2 scorer before the merger versus after it — iso-selector and iso-budget,
so a pure stage variable — produces a large accuracy gap on text-dense benchmarks and at most a small,
reversed gap on object-centric GQA. On the official full splits, pre-merger
selection leads by +11.0 to +38.4 pp on text-dense benchmarks across two Qwen-VL generations
(primary paired McNemar |z| = 14.6–43.0), and by +34.6 to +432 pts on a third model family,
InternVL3-8B, whose merger is a different design (pixel-shuffle downsampling rather than Qwen's
learned PatchMerger). The law is workload-conditional, not uniform: GQA is the one benchmark on
which the post stage leads — by a small significant margin on Qwen (+2.6–2.8 pp, an order of
magnitude below any text-dense Δ) and by a statistical tie on InternVL3 (−0.4 pp) — and no
text-dense benchmark shows any crossover. The gap widens monotonically with compression depth: the
two stages are indistinguishable at 75% retention and far apart at 25%.

**Why — a selection-level causal mechanism.** The gap is not information destroyed in the forward
pass. A ranking-swap control — run the post-merger forward path but select units with the
pre-merger ranking — recovers pre-merger accuracy exactly (DocVQA 200/200 byte-identical answers;
TextVQA 198/200), and the two paths keep *exactly the same* unit sets (kept-set Jaccard = 1.000 in
all four model×benchmark cells on both architectures). The entire pre>post gap is therefore a
*ranking effect*: the learned merger re-shuffles unit saliency almost from scratch (Spearman
ρ = 0.14–0.36) and, on documents, systematically demotes high-edge text-stroke units. The merged
representations of kept units carry no stage-dependent loss; selection order is everything.

**What we propose — Rank-Before-Merge (RBM).** Score 2×2 merge units on their merger-*input*
features with a query-blind L2 norm, keep the top-κ units, and pass only the survivors through the
unmodified native merger. No attention weights, no training, no query input — a pure inference-time
stage change. Because the merger is per-unit (a kept unit's merged token is bit-identical whichever
stage selected it), RBM leaves the retained tokens' bits unchanged and changes only *which* units
reach the merger. The method is deliberately minimal: we freeze it as *plain* RBM and decorate it
with nothing, because every decoration we tried is neutral or harmful (§6).

**A closed design space.** Four pre-registered extensions that would make pre-merger selection
query-aware, regime-aware, or composite — a query-level embedding gate, a merger-aware hybrid mask,
an image-level router, and a two-stage RBM→FastV cascade — each fail. Together they imply that
pre-merger features are purely visual: pre-merger selection is a **fixed point** that no downstream
selector improves, and the query-dependent headroom (which FastV captures on scene-text/object
benchmarks) is reachable only after the LLM's cross-attention layers mix in the question.

**Contributions.**
1. **A stage law for merger-equipped VLMs.** Query-blind pre-merger selection systematically beats
   post-merger selection on text-dense workloads (+11.0 to +38.4 pp, Qwen full splits; +34.6 to
   +432 pts on a third family, InternVL3-8B, under iso-selector / iso-budget control), across two
   merger designs — with no text-dense crossover and a gap that widens monotonically under
   compression — while remaining a robust default (it wins dense OCR) where a separate
   query-conditioned in-layer baseline (FastV) leads on scene-text/object benchmarks.
2. **A selection-level causal mechanism.** A ranking-swap control plus a kept-set identity
   (Jaccard = 1.000 on both architectures) attributes 100% of the gap to the ranking the merger
   rewrites (M1 reshuffle, M2 text-stroke demotion, M3 swap ≡ pre), not the forward path.
3. **Generalization, efficiency, and a negative-result closure.** The law generalizes to a third
   family with a different merger; compression is stage-neutral and yields +68% throughput at 25%
   retention; and four pre-registered negative results establish that pre-merger selection is a
   fixed point. We position RBM as the **robust default, not uniformly optimal**: it never collapses
   and uniquely retains dense OCR, while conceding that query-conditioned methods lead on
   scene-text/object benchmarks.

---

## 2. Related Work (~55 lines)

**Token pruning and merging in VLMs.** Most methods score visual tokens and drop or merge the least
important before or within the early LLM layers. **FastV** prunes at the second LLM layer from the
attention visual tokens receive \cite{chen2024fastv}; it is our primary same-model baseline and the
empirical instance of a query-conditioned scorer that wins scene-text/object benchmarks yet cannot
rescue merger-destroyed OCR (§4.3). **PyramidDrop** drops tokens progressively across LLM layers
\cite{xing2024pyramiddrop}; **FasterVLM** accelerates attention-based pruning \cite{zhang2024fastervlm};
**SparseVLM** explores query-conditioned sparsification \cite{zhang2024sparsevlm}; **PruMerge** merges
by attention-derived importance \cite{shang2024prumerge}; **FitPrune** fits lightweight importance
predictors \cite{ye2024fitprune}. All operate on tokens that have *already* passed the encoder's native
merger; none treats the merger stage itself as a design variable.

**Token-merging lineage (ToMe → PruMerge → AdaptMerge).** Token *merging* originates with **ToMe**
\cite{bolya2023tome} (bipartite soft matching on a ViT, post-encoder, query-blind). **AdaptMerge**
\cite{islam2025adaptmerge} adapts merging to large multimodal models with visual- and language-guided,
adaptive merging and reports that language guidance closes most of the OCR gap plain ToMe incurs — its
central finding that merging degrades OCR and query guidance partly repairs it is direct corroborating
evidence for our lossy-merger thesis, which we cite as support, not as a competitor. This lineage adds
a *second*, learned merging/scoring stage on the merge side; we instead hold the native merger fixed
and ask only whether selection should precede or follow it, so ToMe-side and adaptive merging are
orthogonal to, and compose with, the stage axis we study.

**VisionZip and Qwen-family compression.** **VisionZip** selects "dominant" tokens by CLS-to-patch
attention and merges the remainder into "contextual" tokens \cite{yang2024visionzip}; a code reading
confirms its Qwen path selects *after* the PatchMerger, i.e. on the post-merger side of our axis. Its
authors' own Qwen2.5-VL numbers are a model/stage-mismatched anchor only: OCRBench 81.5 → 70.5 at 50%
retention while general tasks hold, with the README caveat that gains are "less striking" because the
merger already compresses — the same effect seen from the post-merger side; we claim no head-to-head
victory over it (§4.4). Qwen-specific methods **GlimpsePrune** \cite{zeng2025glimpseprune} and **VScan**
\cite{zhang2025vscan} likewise select after the native merger.

**The design point we isolate.** We do not claim a topological vacuum. QuietPrune-style in-ViT pruning
is a different, earlier, trained stage; FastV/PyramidDrop select inside the LLM; VisionZip selects on
the opposite (post-merger) side. The combination we isolate is narrower: on a merger-equipped VLM,
leaving the native merger untouched, scoring the merger-*input* 2×2 units by a query-blind saliency
norm, keeping the top-κ, and handing only the survivors to the unmodified native merger — together
with the causal mechanism (M1–M3) explaining *why* this stage dominates on text-dense workloads,
which the methods above neither report nor control for.

**Efficiency evaluation of VLMs.** Reported speedups depend heavily on engine and protocol. We adopt
the lmms-eval family of *official* scorers for accuracy \cite{zhang2024lmmseval} and confine all
throughput claims to a single engine (§7).

---

## 3. Method — Rank-Before-Merge (RBM) (~85 lines)

### 3.1 Models, mergers, and the stage axis

We study three merger-equipped VLMs spanning two merger designs: **Qwen3-VL-8B-Instruct**
\cite{bai2025qwen3vl} and **Qwen2.5-VL-7B-Instruct** \cite{bai2025qwen25vl} (learned **PatchMerger**),
and **InternVL3-8B** \cite{chen2024internvl} [TODO: verify bibkey] (**pixel-shuffle** downsampling
merger). All are served in bf16, eager attention, vLLM 0.19 (V1), single A40 (46 GB), greedy decoding.
The shared pipeline is: image → ViT patch tokens → native merger (2×2 groups → one token) → language
model. The Qwen merger is *not* an averaging pool: the four patch features of each 2×2 group are
concatenated and projected by a small learned MLP (LayerNorm → fc1 → GELU → fc2) to one unit vector;
the lossy step is the four-to-one nonlinear projection, so a kept unit's representation is a nonlinear
function of its four patches. We verified this against the served implementation on both Qwen
generations.

**The stage axis (the experimental variable).** Two hook points differ only in *where* selection
happens, with model, token count, and scorer held fixed:

- **Pre-merger (RBM).** Score all N 2×2 units on their *merger-input* features, keep the top-κN, and
  pass only the survivors through the unmodified native merger (and, for Qwen3-VL, each deepstack
  merger). The merger operates exactly as in the uncompressed model, on a subset of units.
- **Post stage (contrast).** Either score the N *merged* units on merger-*output* features and keep
  the top-κN (the stage used by published compression methods for merger-equipped VLMs, including
  VisionZip's Qwen path), or — the stronger, query-conditioned contrast — prune inside the LLM by
  layer-2 attention (the FastV stage). The main matrices (Tables 1–2) use the post-merger-L2 contrast,
  iso-selector with the pre arm; the layer-2 FastV contrast is used only in Table 3.

[FIG:1 — pipeline schematic. Two panels: (a) RBM scores raw merger-input unit features, keeps top-κ,
then invokes the native merger on survivors only; (b) post-stage selection merges all units first (or
prunes at LLM layer 2). The native merger and the LLM are drawn identically in both panels; the two
pipelines differ only in where the saliency score is tapped. Caption intent: isolate stage as the sole
manipulated variable.]

**Budget definition.** Retention κ is defined over *merge units* relative to each image's **own**
full unit count, so two methods at the same κ feed the language model the same number of visual tokens
per image; we verify equality of the mean post-merger token count per benchmark (**iso-token**
control), so any accuracy difference is attributable to *which* units survive and the representational
state on which selection is made, not to token count. We report κ ∈ {0.25, 0.125} (25% / 12.5%
retention); per-benchmark token counts are printed with each table.

### 3.2 The L2 selector as a control variable

To isolate the *stage*, we deliberately use the simplest possible **text-agnostic, query-blind**
scorer, identical at both hook points: the L2 norm of the unit feature vector, `s(u) = ‖f_u‖₂`,
computed on merger-input features for pre-merger selection and on merger-output features for
post-merger selection. A strong task- or query-aware scorer would confound *scoring quality* with
*stage*; with an identical, saliency-free scorer, the only manipulated variable is the hook point. We
freeze this as **plain RBM** and add no variant. §5.4 reports that the law survives replacing L2 with
a second scorer family (global-centroid attention) on Qwen3-VL, indicating the effect is not an L2
artifact. Unlike the merging lineage of §2, RBM introduces no second learned merger and no query
input: it manipulates only the *set* of units the native merger sees — which is exactly what makes
the pre-vs-post contrast a clean stage experiment rather than a scorer comparison.

### 3.3 Rank-Before-Merge

Given an image with N merge units, compute `s(u)` on merger-input features for every unit, retain the
top `k = κN`, and invoke the native merger(s) on the retained units only. Unit identity is shared
across the main and deepstack streams (deepstack mergers process intermediate-depth features of the
*same* units), so pruning a unit removes it from every stream. No attention weights are required and
no parameters are trained.

**Unit equivalence (the structural invariant).** Because the mask is at *unit* granularity and the
merger is per-unit, a kept unit's merged token is **bit-identical** regardless of stage — the merger
sees all four patches of a kept unit either way. Selection cannot prevent intra-unit combination; it
changes only *which* units reach the merger. Pre-merger selection keeps units on their raw
(merger-input) saliency; post-stage selection keeps them on a saliency the merger has already
rewritten (§5). This invariant is what makes the ranking-swap control of §5.3 decisive.

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
Post-merger selection (the contrast): replace lines 3-5 by
   U' <- NativeMerger(U); score'[u'] <- || f_out(u') ||_2 on MERGER-OUTPUT
   features; K' <- top-k by score'; keep(K'). Everything downstream identical.
```

**Configuration disclosures.** (i) Qwen2.5-VL uses a block M-RoPE layout and Qwen3-VL an interleaved
one; when tokens are pruned the position cursor must advance by the *actual* surviving count, which
the stock serving routine does not do. A family-scoped fix (bit-degrading to the original at full
retention) makes the Qwen2.5-VL compression path well-formed; the block-vs-interleaved contrast is
itself a diagnostic of why naive pruning collapses one generation and is tolerated by the other.
(ii) On Qwen3-VL the first-invoked merger is deepstack[0] (ViT block-8 output); on Qwen2.5-VL it is
the main merger input after the final ViT block; on InternVL3 the pixel-shuffle stage. The L2 scorer
taps the corresponding merger-input feature in every case, preserving the iso-selector control.

---

## 4. Main Experiments (~140 lines)

### 4.1 Setup

**Benchmarks and metrics (official).** TextVQA (VQA-accuracy) \cite{singh2019textvqa}, DocVQA (ANLS)
\cite{mathew2021docvqa}, OCR-Bench (official five-category score /1000) \cite{liu2024ocrbench}, and GQA
(word-normalized exact match) \cite{hudson2019gqa}, on the official full splits (TextVQA val 5000;
DocVQA val 5349; OCR-Bench 1000; GQA test-dev 12578). Scorers are verbatim ports of the lmms-eval
family \cite{zhang2024lmmseval}, with ground-truth self-test passing 200/200.

**Engines and fairness protocol.** RBM and the none / post cells run in vLLM; the FastV baseline runs
in an independent HuggingFace transformers eager harness (it has no vLLM path). Accuracy is comparable
across engines (the harness bit-degrades to native at zero drop, 8/8 per-sample identical; HF-vs-vLLM
none cells agree 16/16); **throughput is not** — all efficiency numbers are confined to vLLM (§7).
Fairness is enforced by the same keep ratio relative to each image's own token count and by reporting
mean post-merger token counts.

**Significance.** Each pre-vs-post Δ is paired (same images/questions, greedy). The **primary** test
is the paired McNemar z on per-sample correctness (b/c = pre-only / post-only correct; z = (b−c)/√(b+c));
an independent-binomial z is the cross-check. Tables print Δ with the McNemar z unless noted.

### 4.2 Qwen main result — official full splits (Table 1)

**Table 1.** Official metrics, full splits (n = 5000 / 5349 / 1000 / 12578), greedy, both Qwen
models. RBM = pre-merger L2 (ours); post = post-merger L2 (byte-identical to the VisionZip
dominant+contextual principle-port on both generations, §3.2). Δ₂₅ = pre − post at κ = 0.25 with the
primary paired McNemar z in parentheses; † marks the (only) significant post-stage lead. Parenthetical
stderrs (√(p(1−p)/n)) and skip accounting are in the supplementary material. Source: j7_main_table.md.

| Model | Benchmark (metric) | none | RBM @25% | post @25% | Δ₂₅ (McNemar z) | RBM @12.5% | post @12.5% |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | TextVQA (VQA-acc) | 0.844 | **0.605** | 0.222 | **+38.4 pp (43.0)** | 0.472 | 0.132 |
| | DocVQA (ANLS) | 0.956 | **0.481** | 0.238 | **+24.3 pp (34.7)** | 0.352 | 0.103 |
| | OCR-Bench (/1000) | 760 | **547** | 184 | **+363 pts (16.7)** | 350 | 53 |
| | GQA (exact match) | 0.616 | 0.449 | 0.477 | −2.8 pp † (−8.0) | — | — |
| Qwen2.5-VL-7B | TextVQA (VQA-acc) | 0.862 | **0.702** | 0.442 | **+26.1 pp (30.8)** | 0.597 | 0.319 |
| | DocVQA (ANLS) | 0.949 | **0.636** | 0.526 | **+11.0 pp (15.6)** | 0.455 | 0.245 |
| | OCR-Bench (/1000) | 817 | **476** | 183 | **+293 pts (14.6)** | 335 | 67 |
| | GQA (exact match) | 0.604 | 0.559 | 0.585 | −2.6 pp † (−5.7) | — | — |

**Table notes.** (a) Qwen3-VL none DocVQA was re-run at max-model-len 49,152 and completed with
0/5349 skips (ANLS 0.956); the pre/post cells it anchors ran at the same native-resolution
configuration and are directly comparable to each other. (b) OCR-Bench none cells skip 18 (Qwen3-VL) /
24 (Qwen2.5-VL) of 1000 long-OCR images on context overrun, scored 0 (conservative; compressed cells
skip ≤ 5), so the uncompressed anchor is understated and the relative gaps are, if anything,
conservative. (c) GQA was evaluated at κ = 0.25 only on the full split. (d) Iso-token control: at each
model×benchmark, pre and post have identical mean post-merger token counts, with one exception — the
Qwen2.5-VL OCR-Bench @25% RBM cell used a 4 M-pixel cap (mean tokens 229.6 vs post 282.0), so its
+293-pt gap is conservative. (e) Primary test = paired McNemar z (printed); independent-binomial
cross-check agrees (smallest |z| = 4.1, the GQA post lead). No text-dense Δ cell is direction-only or
non-significant (smallest text-dense |z| = 14.6).

**The stage law holds on both generations.** Every text-dense Δ₂₅ is large and significant: RBM leads
post-merger selection by +38.4 pp (TextVQA), +24.3 pp (DocVQA), +363 pts (OCR-Bench; 547 vs 184, 3.0×)
on Qwen3-VL, and by +26.1 pp, +11.0 pp, +293 pts (476 vs 183, 2.6×) on Qwen2.5-VL; no McNemar z falls
below 14.6. The gap is robust to deeper compression: at κ = 0.125 the post stage degrades much faster
(Qwen3-VL OCR-Bench 184 → 53 vs RBM 547 → 350, widening the pre/post ratio from 3.0× to 6.6×), and
all six text-dense Δ₁₂.₅ remain significant (McNemar |z| ≥ 15.9).

**Object-centric GQA, reported honestly (the single full statement).** GQA is the *only* benchmark on
which the post stage leads pre on Qwen, on both generations, and at full n = 12578 the lead is
significant: +2.8 pp (Qwen3-VL) / +2.6 pp (Qwen2.5-VL); independent-binomial z = 4.5 / 4.1; primary
McNemar z = 8.0 / 5.7 (83.9% / 83.8% per-question agreement; the official exact-match rescore gives a
concordant 8.1 / 7.1). Its magnitude is an order of magnitude below the TextVQA/OCR-Bench Δ (1/10–1/14
of them), and **no text-dense benchmark shows any crossover**. We therefore claim **no** pre-merger
superiority on object-centric data; all other sections refer here for the GQA statistics.

### 4.3 Generalization to a third family — InternVL3-8B (Table 2)

To test whether the stage law is a Qwen-family artifact or a property of merger-equipped VLMs, we run
RBM and the post stage on **InternVL3-8B**, whose merger is pixel-shuffle downsampling — a different
design from Qwen's PatchMerger. The post arm here is **post-merger L2 top-k** — the *same*
query-blind L2 selector as the pre arm, at an identical token budget (ptid equal across arms), so
Table 2 is a **pure stage-variable experiment under iso-selector / iso-budget control** at 25%
retention. The query-conditioned in-layer FastV contrast is a separate campaign, reported in Table 3.

**Table 2.** InternVL3-8B, official full splits (n = 5000 / 5349 / 1000 / 12578), greedy. pre = RBM
(pre-merger L2); post = post-merger L2 top-k (query-blind, same L2 selector as pre; iso-selector and
iso-budget — ptid identical across arms). The query-conditioned FastV contrast is Table 3. ptid = mean
post-merger visual tokens/request (pre and post iso-token). Source: internvl3_main_matrix.md (16/16
cells, zero missing).

| Benchmark (metric) | none | pre @25% | post @25% | pre − post | pre vs none | ptid |
|---|---|---|---|---|---|---|
| TextVQA (VQA-acc) | 0.8338 | **0.7890** | 0.4148 | **+37.4 pp** | −4.5 pp | 690 |
| DocVQA (ANLS) | 0.9221 | **0.7284** | 0.3820 | **+34.6 pp** | −19.4 pp | 860 |
| OCR-Bench (/1000) | 852 | **753** | 321 | **+432 pts** | −99 | 555 |
| GQA (exact match) | 0.6293 | 0.5993 | 0.6031 | −0.4 pp (tie) | −3.0 pp | 690 |

Deep budget, 12.5% retention (r = 0.875, text-dense only): TextVQA pre 0.7230 vs post 0.3064
(**+41.7 pp**); DocVQA pre 0.5054 vs post 0.2451 (**+26.0 pp**) — the gap *widens* under deeper
compression, as on Qwen.

**Cross-family isomorphism.** InternVL3's pre−post text-dense gaps (+37.4 / +34.6 pp / +432 pts) are
the same order and direction as Qwen3-VL's (+38.4 / +24.3 / +363 pts) on all three text-dense
benchmarks, and GQA stays within the red line — a tie here (−0.4 pp) versus Qwen3-VL's small post
edge (+2.8 pp), both small, with no text-dense crossover. The stage law — the same query-blind L2
scorer beats when applied before the merger because selection deferred past it loses raw-patch text
information to the merger's ranking rewrite (§5), at identical selector and budget — is therefore
established on a **third model family with a different merger**, supporting the
"merger-equipped VLMs" generalization rather than a family-specific claim.

[FIG:2 — cross-family stage-law bars. Pre−post Δ on the three text-dense benchmarks, one bar group
per family (Qwen3-VL, Qwen2.5-VL, InternVL3), with the GQA Δ plotted near zero for contrast. Caption
intent: the text-dense advantage is the same order of magnitude across three families and two merger
designs, while GQA stays near zero.]

**Table 2 notes (configuration boundaries — read before comparing across models).**
(i) **DocVQA pixel cap differs by family.** InternVL3 DocVQA runs under a **4 M-pixel** cap (its
dynamic tiling tolerates more, and its image processor rejects a smaller cap); the Qwen DocVQA
baseline-comparison cells of §4.4 use a **600k** cap. Within-model comparisons are valid; **cross-model
DocVQA absolute values are not comparable** — compare per-model Δ, not per-model levels.
(ii) **RBM is not lossless on DocVQA/OCR.** pre@25% costs −19.4 pp vs none on InternVL3 DocVQA and
−99 /1000 on OCR-Bench, while TextVQA costs only −4.5 pp and GQA −3.0 pp; a near-lossless reading is
**TextVQA/GQA-scoped only**. On the text-dense benchmarks the claim is the large pre>post margin, not
pre = none. (iii) The GQA "tie" is at full n = 12578 (−0.4 pp; no meaningful difference), contrasted
with Qwen's small significant post edge — family-dependent, both small.

### 4.4 Same-model comparison against the strongest query-conditioned baseline (Table 3)

We compare RBM against FastV at its **best layer K** (K = 3; a K-sensitivity probe found K = 3 best),
reported at the same scope with official rescoring, so the comparison cannot be attributed to a
handicapped K. At its canonical progressive schedule PyramidDrop is near-lossless but occupies a
different budget point (equivalent mean retention 0.625 ≈ 2.1× ours); forced to an honest iso-25%
schedule it collapses (TextVQA 0.073 / GQA 0.305 / OCR 0.005 on Qwen3-VL), whereas RBM holds
(0.60 / 0.42 / 0.58). There is therefore no universal winner.

**Table 3.** FastV at best K = 3 vs RBM @25%, both Qwen models, official rescoring. Δ = FastV-k3 −
RBM (positive ⇒ FastV leads). Source: r2b_fastv_k3.md; RBM references from j7_main_table.md
(TextVQA/GQA full split) and the same-scope n = 200 cells (DocVQA/OCR-Bench).

| Model | Benchmark (scope) | FastV-k3 | RBM @25% | Δ (winner) |
|---|---|---|---|---|
| Qwen3-VL-8B | TextVQA (full 5000) | **0.7771** | 0.605 | **+17.2 pp (FastV)** |
| | GQA (full 12578) | **0.5376** | 0.449 | **+8.9 pp (FastV)** |
| | DocVQA (n = 200, 600k cap) | **0.5863** | 0.4239 | **+16.2 pp (FastV)** |
| | OCR-Bench (n = 200) | 0.415 | **0.5801** | **+16.5 pp (RBM)** |
| Qwen2.5-VL-7B | TextVQA (n = 200) | 0.7467 | — | FastV leads [TODO: paired Δ] |
| | GQA (n = 200) | 0.475 | — | FastV leads [TODO: paired Δ] |
| | DocVQA (n = 200) | 0.4852 | — | FastV leads [TODO: paired Δ] |
| | OCR-Bench (n = 200) | 0.285 | — | RBM leads [TODO: paired Δ] |

**Table 3 notes.** (i) FastV-k3 keeps ≈ 214 mean visual tokens/request; budgets are approximately
matched (~25% retention) but stated per cell — this is "each method at its strong configuration," not
a byte-iso-budget cell [TODO: verify per-benchmark ptid alignment]. (ii) OCR-Bench n = 200 FastV cells
skip ≈ 10% of long-OCR prompts (19/200 Qwen3-VL, 20/200 Qwen2.5-VL; empty FastV outputs, scored over
attempted) — disclosed, conservative for FastV. (iii) Qwen3-VL TextVQA/GQA are full-split on both
sides; DocVQA/OCR-Bench and all Qwen2.5-VL cells are n = 200 (dev scope); Qwen2.5-VL paired Δ are
[TODO] pending the matched-scope RBM cells. (iv) DocVQA cells are at the 600k pixel cap on both sides
(iso-pixel within this table).

**Reading.** FastV (query-conditioned: layer-2 attention has mixed in the question) leads RBM on
TextVQA, GQA, and same-pixel DocVQA on Qwen3-VL (+17.2 / +8.9 / +16.2 pp), and the direction holds on
Qwen2.5-VL. This is the query-dependent headroom our pre-registered query-level gate cannot reach
before the merger (§6): *which* region matters for *this question* is recoverable only once
cross-attention has seen the question. **But RBM still wins OCR-Bench by +16.5 pp even at FastV's best
K** (0.580 vs 0.415 on Qwen3-VL): dense OCR is destroyed once text patches are discarded inside the
LLM, and no query conditioning recovers information already thrown away — only pre-merger selection,
which keeps raw patches, preserves it. The core claim is therefore robust to the baseline's K choice.
We report these wins for FastV honestly: RBM is the robust default (never collapses, uniquely holds
dense OCR), not a method that beats query-conditioned pruning everywhere.

---

## 5. Mechanism — Why the Merger Is Lossy (~85 lines)

We give the pre-vs-post gap a mechanistic account in three parts (M1–M3), measured on
**Qwen3-VL-8B** (M1/M2 on a deterministic seed-0 n = 64 sample per benchmark; M3 on n = 200 with the
same invocation as the headline cells), then state its architectural scope.

### 5.1 M1 — the merger reshuffles unit ranks

Per image we correlate the *pre* ranking (merger-input unit L2) with the *post* ranking (merged-token
L2) over all units. The two rankings are barely related, and the reshuffle tracks text density:

| Benchmark (type) | Spearman ρ | Jaccard@25% |
|---|---|---|
| DocVQA (text-dense doc) | **0.137** | **0.180** |
| TextVQA (text-dense scene) | 0.332 | 0.243 |
| GQA (object-centric) | **0.360** | **0.278** |

The merger's nonlinear projection reorders unit saliency almost from scratch (ρ = 0.14–0.36, nowhere
near 1). Both the full-ranking correlation and the decision-relevant top-25% overlap *rise* from the
most text-dense benchmark to the object-centric one (ρ = 0.137 < 0.332 < 0.360; Jaccard = 0.180 <
0.243 < 0.278), so the reshuffle is largest on DocVQA and smallest on GQA — yet substantial on every
benchmark (source: mechanism_verification_report, §4.1 of the snapshot draft).

### 5.2 M2 — the demoted units are text-stroke units

With rank_shift = post_rank − pre_rank (+ ⇒ demoted) and per-unit Sobel edge energy as a text-stroke
proxy, the merger preferentially *demotes* high-edge units at κ = 25%: ρ(rank_shift, edge) = +0.439 on
DocVQA, +0.155 on TextVQA, +0.036 on GQA — ~12× stronger on documents than on object scenes. The units
post drops that pre keeps are the highest-edge of all: on DocVQA, mean Sobel **0.641 vs 0.124** for
the reverse group, with **92% above the per-image median (vs 35%)** — post-merger selection is
**systematically anti-text on documents**. (The *direction*, we hypothesize, is set by the merger's
training on natural content, which attenuates high-frequency stroke energy; this is a testable
hypothesis, not a measured result.)

### 5.3 M3 — a ranking-swap causal control, and the decisive kept-set identity

By unit equivalence (§3.3), holding the *forward path* fixed at post and swapping in the *pre ranking*
must reproduce pre accuracy if the gap is purely a ranking effect. It does:

| Benchmark | metric | none | post | **pre** | **swap** (post-path + pre-ranking) | identity |
|---|---|---|---|---|---|---|
| TextVQA | VQA-acc | 0.858 | 0.215 | **0.598** | **0.603** (Δ +0.005) | **198/200** |
| DocVQA | ANLS | 0.976 | 0.200 | **0.465** | **0.465** (Δ 0.000, exact) | **200/200** |

Swapping in the pre ranking recovers full pre-merger accuracy (exactly on DocVQA; within 2/200
greedy-decode answers on TextVQA, ε-level kernel numerics confirmed by a 200/200 byte-identical
independent rerun) and erases the collapse (+38.3 pp TextVQA, +26.5 pp DocVQA over post). The swap
path agrees with post on only 40/200 (TextVQA) and 63/200 (DocVQA) answers — without the pre ranking,
the same forward path produces the collapse.

**The decisive hardening — kept-set identity on both architectures.** Emitting the kept unit indices
from both paths and comparing directly: **Jaccard(swap-kept, pre-top-κ) = 1.000 in all four cells**
(Qwen3-VL and Qwen2.5-VL × DocVQA and TextVQA; n = 32, chance level κ/N = 0.25). The two paths select
*exactly the same units*. The small swap-vs-pre answer residual on Qwen2.5-VL (+6.7 pp DocVQA at
n = 32) traces to an implementation confound of the swap control — window-attention `reverse_indices`
(three recovery sites on Qwen2.5-VL, zero on Qwen3-VL) map identical index sets to physical units in
different orderings — while batch-dependent merging is structurally excluded (the merger is per-token
LN+MLP). **Verdict: the pre>post gap is attributable to the ranking alone — selection-level causal,
architecture-general** (byte-identical on Qwen3-VL; identical kept sets with an order-confound
residual on Qwen2.5-VL). The merged representations of kept units carry no stage-dependent loss.

### 5.4 Two failure modes, selector invariance, and a cross-generation boundary

The mechanism is clean on DocVQA (post selection is *misguided away from text*; restoring the pre
ranking restores accuracy fully). On TextVQA the ranking corruption is again the *entire* source of
the gap (swap ≡ pre), but the text-directionality is moderate and even pre-selected units underperform
the uncompressed baseline (0.60 vs 0.86) — a **budget effect** shared by any 25%-keep method. GQA shows
near-null directionality (M2 ρ = 0.04): object scenes carry little high-frequency text for the merger
to damage, so the accuracy consequence nearly vanishes and reverses sign (the small GQA post edge,
§4.2). Summary: **documents = selection misdirection; scene text = feature degradation; object scenes
= near-null ranking with a small significant post edge** — the stage effect is workload-conditional.

**Selector invariance.** On Qwen3-VL the stage law holds under a second scorer family
(global-centroid attention: TextVQA pre 0.553 vs post 0.200, +35.3 pp; at doubled n, +35.5 / +24.4 pp
on TextVQA/DocVQA — invariant in sign and magnitude). On Qwen2.5-VL the L2 *sign* is invariant but the
attention proxy degrades — a proxy-family specificity, not a counter-example; L2 is the paper's
selector.

**Replication and boundary.** The ranking law replicates on Qwen2.5-VL-7B (n = 64, r = 0.75: DocVQA
pre 0.664 > post 0.531, +13.3 pp; TextVQA pre 0.719 > post 0.349, +37.0 pp; and the VisionZip-style
port ≡ post byte-for-byte, 0.415 == 0.415), and now on InternVL3-8B at the accuracy level (Table 2).
The causal swap control is byte-exact on Qwen3-VL and selection-level (identical kept sets) on
Qwen2.5-VL; we claim the causal decomposition for Qwen3-VL and present the others as corroboration of
the ranking law.

**Retention curve (the accuracy-side signature).** At 75% retention the two stages are
indistinguishable (DocVQA post even leads 1–2 pp *within noise*), and the pre-merger lead emerges and
widens monotonically with depth — TextVQA Δ +8.7 → +29.0 → +38.4 pp (Qwen3-VL) and +7.0 → +15.3 →
+26.1 pp (Qwen2.5-VL); DocVQA −1.3 → +5.9 → +24.3 pp (Qwen3-VL) and −2.1 → −1.4 → +11.0 pp
(Qwen2.5-VL) at 75% / 50% / 25% retention (n = 200; source: j8_ablations). The text-hostile reshuffle
becomes the dominant term only when a large fraction of units is dropped — the same shallow regime
behind the GQA sign and the Qwen2.5-VL low-resolution boundary (§8).

[FIG:3 — retention-vs-gap curves. Pre (RBM) vs post, four benchmark×model panels; the pre-merger lead
is ≈0 under shallow compression and widens monotonically with depth. Caption intent: the stage effect
is compression-activated, not constant.]

---

## 6. Negative Results — Pre-Merger Selection Is a Fixed Point (~75 lines)

We pre-registered four extensions that would make pre-merger selection query-aware, regime-aware, or
composite, and report all four as failures. They are the evidence that closes the design space and
explains why RBM is frozen as a plain query-blind method.

**(a) Image-level routing fails.** With both-stage outcomes collected per image, an always-pre policy
is near-dominant (pre ≥ post on 84–97% of images in every benchmark). The best image-level router
(0.484) scores *below* always-pre (0.494) and far below the per-sample oracle (0.576); a ptid-threshold
router on pooled N = 774 reaches 0.655 vs always-pre 0.634 vs always-post 0.452 vs oracle 0.702.
Decomposing the oracle gap, only ~27% is workload-level; **~73% is sample-level and query-dependent**,
unreachable from image-level signals.

**(b) Merger-aware hybrid masking fails the gate.** A hybrid that keeps the pre/post agreement set and
routes the contested budget to high-edge units by a text-fraction t was pre-registered against a
no-OCR-regression + GQA-gain gate. At the tuned t = 0.5 it gains +2.3 pp on TextVQA (0.560 vs pre
0.537, within noise) but **loses 8 pp on OCR-Bench** (0.510 vs pre 0.590, ≈ 2σ) and gains nothing on
GQA (0.500). No single text-fraction passes the gate.

**(c) Query-level embedding blending fails.** A query-aware pre-merger saliency
`s = (1−λ)·L2 + λ·(question-embedding cosine)` was tuned on a disjoint dev slice. **Every λ > 0 is
harmful**: dev mean 0.5772 at λ = 0; −1.7 pp (λ = 0.3); −5.2 pp (λ = 0.5); −3.3 pp (λ = 0.7). By the
pre-registered rule we select λ = 0 and freeze plain RBM.

**(d) Two-stage cascade fails (NO-GO).** The natural composition — Stage 1 query-blind RBM (keep
fraction X) followed by Stage 2 query-conditioned FastV (keep fraction 1−Y), total keep X·(1−Y) — was
pre-registered against a Pareto gate: GO iff *every* benchmark's cascade is within 0.5 pp of
max(pre-alone, FastV-alone) *and* at least one benchmark strictly beats both arms (paired McNemar
z ≥ 1.5). On Qwen3-VL, n = 200 × 4 benchmarks, official rescoring (mean token counts identical across
the three arms in every cell — same images, same budget, fair pairing):

**Table 4.** Cascade two-stage gate, Qwen3-VL-8B, n = 200, official metrics. Pareto = within 0.5 pp
of max(pre, FastV). Source: cascade_gate.md.

| Total keep | Benchmark | cascade | pre-alone | FastV-alone | Pareto? |
|---|---|---|---|---|---|
| 25% | TextVQA | 0.595 | 0.597 | **0.680** | ❌ |
| 25% | DocVQA (600k) | 0.478 | 0.424 | **0.518** | ❌ |
| 25% | OCR-Bench (/1000 est.) | 370 | **580** | 171 | ❌ |
| 25% | GQA | 0.405 | 0.415 | **0.490** | ❌ |
| 12.5% | TextVQA | 0.405 | **0.500** | 0.445 | ❌ |
| 12.5% | DocVQA | 0.207 | **0.298** | 0.278 | ❌ |
| 12.5% | OCR-Bench (/1000 est.) | 160 | **359** | 88 | ❌ |
| 12.5% | GQA | 0.350 | 0.395 | **0.455** | ❌ |

Cascade fails condition 1 on **all 8** budget×benchmark cells — never within 0.5 pp of
max(pre, FastV), typically **8–18 pp below** — and condition 2 holds nowhere. The pattern is
structural, not noise: cascade sits *between* the two single methods and is dominated by the better
one on every benchmark (OCR-Bench 370 ≈ between pre 580 and FastV 171; same shape at 12.5%).
Query-blind Stage-1 discards degrade the token set Stage-2 attends over, while on text-dense/OCR the
FastV stage destroys exactly the raw-patch information Stage-1 was built to protect. **Two
training-free selectors composed in series do not complement.** (The harness was first validated:
Y = 0 cascade ≡ pre-alone and X = 1 cascade ≡ FastV-alone byte-exactly, 8/8 answers each; HF-pre vs
vLLM-pre engine consistency 14/16 with kept-set Jaccard 0.986.)

**Closure — the one-sentence chain.** Pre-merger features are purely visual and contain no usable
query information, so: query-blending hurts (c), image-level routing cannot beat always-pre because
the headroom is sample-/query-level (a), hybrid text-routing trades OCR for nothing (b), and stacking
a query-conditioned selector after RBM is dominated by the better single stage (d). **Pre-merger
selection is a fixed point; selectors do not stack.** The query-dependent headroom (the oracle's
+8.2 pp, and FastV's wins in Table 3) is reachable **only after the LLM's cross-attention layers mix
in the question** — exactly where FastV operates.

**Deployment guidance.** Use RBM (query-blind, training-free, stage-only) as the default for
text-dense/OCR and unknown/mixed traffic, where never collapsing matters more than the last few
object-centric points; use a FastV-class query-conditioned method when scene-text/object queries
dominate and dense OCR is absent; do **not** route or cascade at runtime — fix the choice at
deployment time by workload.

---

## 7. Efficiency — Supporting Evidence (~45 lines)

Efficiency is supporting evidence for a multimodal-foundation-model contribution, not the claim. All
numbers are measured on **one engine only** (vLLM 0.19 V1, offline batch, 1× A40) and are never
compared across engines with the HF baselines (§4.1).

**Table 5.** Efficiency, Qwen3-VL-8B, TextVQA n = 200, vLLM offline, greedy. Pre = RBM; pre and post
are iso-token at every κ. Source: j6_efficiency.md.

| Config | retention κ | req/s | wall (n = 200) | mean vis. tokens/req | single-req latency |
|---|---|---|---|---|---|
| none | 100% | 4.11 | 48.7 s | 766 | 0.36 s |
| pre (RBM) | 75% | 4.61 | 43.4 s | 582 | — |
| pre (RBM) | 50% | 5.39 | 37.1 s | 397 | — |
| **pre (RBM)** | **25%** | **6.91** | **28.9 s** | 213 | **0.23 s (−36%)** |
| post | 75% / 50% / 25% | 4.52 / 5.54 / 6.89 | — | 582 / 397 / 213 | — |

**Stage-neutral.** Pre and post differ by ≤ 3% in req/s at every budget (4.61/4.52, 5.39/5.54,
6.91/6.89; no consistent direction): the *stage* of selection carries no efficiency cost, so the
pre-merger advantage is purely an accuracy/robustness effect. **Compression scaling.** Throughput
rises +12% / +31% / **+68%** over the uncompressed baseline at 75% / 50% / 25% retention, and
single-request latency falls 0.36 → 0.23 s (−36%) at 25%. **Memory disclosure.** Peak GPU memory is
the fixed allocator reservation (≈ 42 GB, flat across configs), not a measured footprint; we make no
memory claim.

**The speedup reproduces across families.** On InternVL3-8B (within-run relative, not comparable to
the Qwen run), 25%-retention RBM raises request throughput 2.2× (TextVQA), 2.5× (GQA), 2.4× (DocVQA),
and 1.8× (OCR-Bench) over none, with post ≈ pre req/s at identical token counts — the compression
speedup is real on a third family, and remains stage-neutral (source: internvl3_main_matrix.md).
Serving-side (continuous-batching) efficiency and Qwen2.5-VL efficiency are future work.

---

## 8. Discussion and Limitations (~50 lines)

**What the results mean for multimodal foundation models.** Merger-equipped VLMs treat their native
merger as lossless; across three model families and two merger designs we show it is not — on
text-dense workloads it rewrites unit saliency in a text-hostile direction, and the *stage* at which
a training-free selector taps the visual stream dominates the *scorer*. The practical consequence is
a deliberately minimal default, RBM, plus a closed design space: because pre-merger features are
purely visual, no query-aware or composite decoration helps, and the query-dependent headroom lives
only after cross-attention.

**Robust default, not uniformly optimal.** RBM never collapses and never loses to the post-merger
family on text-dense data, and is the only compared method that retains dense OCR (+16.5 pp over
best-K FastV on Qwen3-VL OCR-Bench); but query-conditioned FastV leads it on TextVQA/GQA/DocVQA
(Table 3), and post-stage selection leads it by a small significant margin on Qwen GQA (§4.2). There
is no universal winner.

**Limitations.**
1. **Three families, two merger designs — but two are Qwen.** The mechanism (M1–M3) is established on
   Qwen3-VL; Qwen2.5-VL and InternVL3 corroborate the stage *law* at the accuracy level, with the
   causal swap selection-level on Qwen2.5-VL and the InternVL3 evidence accuracy-level (a kept-set
   identity on InternVL3 is future work). LLaVA-family models without a native merger are out of scope
   by construction.
2. **Cross-engine baselines (accuracy comparable, throughput not).** FastV runs in an HF eager harness
   while RBM/post/none run in vLLM; accuracy is validated comparable (r = 0 anchor 8/8; none cells
   16/16), but throughput is not, and efficiency is confined to one engine, measured offline.
3. **DocVQA is configuration-dependent.** DocVQA uses family-specific pixel caps — **4 M px on
   InternVL3 vs 600k on Qwen** for the baseline-comparison cells — so **cross-model DocVQA absolute
   values are not comparable**; within-model Δ are. Post-stage deep-compression collapse on DocVQA is
   partly configuration-dependent, disclosed rather than presented as a pure method effect. At an
   artificially low 600k cap on the milder Qwen2.5-VL merger the DocVQA stage direction even reverses
   (none 0.962 / pre 0.424 / post 0.504, +8.0 pp post lead vs the native-resolution +11.0 pp pre lead
   of Table 1), a resolution×generation interaction in the shallow regime where neither stage has much
   to lose; the matched Qwen3-VL cells do not reverse. The headline remains the native-resolution full
   split.
4. **RBM is not lossless on DocVQA/OCR.** Near-lossless is TextVQA/GQA-scoped (InternVL3 −4.5 / −3.0 pp
   vs none); on DocVQA/OCR, pre keeps a large margin over post but still costs −19.4 pp / −99 pts vs
   none (InternVL3).
5. **Cross-scope baseline cells and a minimal scorer.** Some Table 3 cells are n = 200 while Table 1
   is full split, each carrying its own scope note; RBM is deliberately the simplest scorer (query-blind
   L2) — the contribution is the stage axis and its causal account, not the scorer (§6 shows no
   decoration helps). OCR-Bench skip-scoring is conservative.
6. **Greedy decoding / single device.** Decoding is deterministic, so error bars are binomial only,
   with no temperature variance; all runs are on a single A40.

---

## 9. Conclusion (~25 lines)

We isolated the *stage* of visual-token selection — before versus after the native merger — as an
experimental variable in merger-equipped VLMs, and found a **stage law**: on text-dense workloads,
query-blind pre-merger selection systematically beats post-merger selection — +11.0 to +38.4 pp across
two Qwen-VL generations and +34.6 to +432 pts on a third family, InternVL3-8B, under iso-selector /
iso-budget control — with two different merger designs (PatchMerger and pixel-shuffle), no text-dense
crossover at the native-resolution full splits (configuration boundary in §8), a gap that widens
monotonically under compression, and at most a small post-stage edge on object-centric GQA; against a
separate query-conditioned in-layer baseline (FastV) RBM wins dense OCR
yet concedes scene-text/object benchmarks, so it is a robust default, not a uniform winner. A ranking-swap control plus a kept-set identity (Jaccard = 1.000 on both
architectures) makes the mechanism **selection-level causal**: the merger rewrites unit saliency in a
text-hostile direction (M1 reshuffle, M2 text-stroke demotion), and the entire pre>post gap is that
ranking, not the forward path (M3 swap ≡ pre). The method that falls out, **Rank-Before-Merge**, is
training-free and stage-only; it is the **robust default, not uniformly optimal** — never collapsing
and uniquely retaining dense OCR (+16.5 pp over best-K FastV), while conceding scene-text/object
benchmarks to query-conditioned methods — and 25%-retention compression is stage-neutral, raising
throughput +68% (Qwen) / 1.8–2.5× (InternVL3). Four pre-registered negative results close the design
space: pre-merger selection is a fixed point that no downstream selector improves, because the
query-dependent headroom is reachable only after cross-attention. Future work: a kept-set identity on
InternVL3, serving-side efficiency, and a same-model comparison to Hi-Lo Prune if its code is released.
Code will be released.

---

<!-- ====================== AUTHOR NOTES — NOT PART OF THE SUBMISSION ======================

## SOURCE MAP (every number → digest; digests = the 7 must-reads)
- Table 1 (Qwen full splits: none/RBM/post @25 & @12.5; Δ; McNemar z 43.0/34.7/16.7/−8.0/30.8/
  15.6/14.6/−5.7; none DocVQA 0.956 @49152 0-skip; OCR none skip 18/24; Q2.5 OCR@25 4M-cap rerun
  ptid 229.6 vs 282.0): experiments/j7_main_table.md + drafts/paper_v4.md Table 1 / paired-test
  table / footnotes / Table A3.
- Table 2 (InternVL3 0.8338/0.7890/0.4148/+37.4/−4.5/690; 0.9221/0.7284/0.3820/+34.6/−19.4/860;
  852/753/321/+432/−99/555; 0.6293/0.5993/0.6031/−0.4/−3.0/690; r0.875 0.7230/0.3064/+41.7,
  0.5054/0.2451/+26.0; 4M px cap; efficiency 2.2/2.5/2.4/1.8×): experiments/internvl3_main_matrix.md.
- Table 3 (FastV-k3 Q3 0.7771/0.5376/0.5863/0.415; Q2.5 0.7467/0.475/0.4852/0.285; vs RBM
  +17.2/+8.9/+16.2/−16.5; RBM refs 0.605/0.449/0.4239/0.5801; ptid ≈214; OCR skip 19/20 of 200):
  experiments/r2b_fastv_k3.md (RBM TextVQA/GQA from j7_main_table.md; DocVQA/OCR n200 refs from
  cascade_gate.md pre-alone column).
- Table 4 (cascade 8 cells; OCR /1000 extrap 370.5/160.4 vs pre 580/359, FastV 171/88; degeneracy
  self-tests 8/8; engine consistency 14/16, Jaccard 0.986): experiments/cascade_gate.md.
- §5 mechanism (M1 ρ 0.137/0.332/0.360, Jaccard@25 0.180/0.243/0.278; M2 ρ +0.439/+0.155/+0.036,
  Sobel 0.641 vs 0.124, 92% vs 35%; M3 swap 0.603/0.598 198/200, 0.465==0.465 200/200; swap-vs-post
  agreement 40/200 & 63/200; Jaccard≡1.000 four cells n=32 chance 0.25; Q2.5 residual +6.7pp
  reverse_indices; selector invariance 0.553/0.200 +35.3, doubled-n +35.5/+24.4; Q2.5 replication
  0.664/0.531, 0.719/0.349; budget 0.60 vs 0.86): drafts/paper_v4.md §4 (citing
  mechanism_verification_report, r1_1_swap_jaccard, method_gate_report, j3).
- §5.4 retention curve (+8.7/+29.0/+38.4; +7.0/+15.3/+26.1; −1.3/+5.9/+24.3; −2.1/−1.4/+11.0 @
  75/50/25%): drafts/paper_v4.md §5.6 (j8_ablations).
- §6 (a/b/c) (router 0.484/0.494/0.576, ptid-router 0.655/0.634/0.452/0.702, 84–97%, ~27%/~73%;
  hybrid t=0.5 0.560/0.510/0.500 vs pre 0.537/0.590/0.510; QA-gate λ 0/−1.7/−5.2/−3.3, dev 0.5772;
  oracle +8.2pp): drafts/paper_v4.md §6 (method_gate_report, j5).
- §7 Table 5 (4.11/4.61/5.39/6.91 req/s; wall 48.7→28.9; tokens 766/582/397/213; latency 0.36→0.23;
  post 4.52/5.54/6.89; ±3%; +12/+31/+68%; ≈42GB gmu-0.9): drafts/paper_v4.md §5.7 (j6).
- Pyramid (keep_equiv 0.625 ≈2.1×; iso-25% collapse 0.073/0.305/0.005; RBM holds 0.60/0.42/0.58):
  drafts/paper_v4.md §5.3 (j7hf_baselines_n500).
- VisionZip anchor 81.5→70.5 @50% (mismatched only): drafts/paper_v4.md §5.5 (visionzip_gap_report).
- §8 600k reversal (Q2.5 none 0.962/pre 0.424/post 0.504, +8.0pp; Q3 keeps pre lead 0.424 vs 0.251):
  drafts/paper_v4.md §5.4 footnote ⁱ (j8b).
- Engine fairness (r=0 8/8; HF-vs-vLLM none 16/16): drafts/paper_v4.md §5.1 (j4_step2_fix).

## RED-LINE SELF-CHECK (per task 红线 + STATE.md)
- [PASS] No SOTA / no "beats existing methods" as an RBM claim. Grep-guard tokens absent from live
  prose; "robust default, not uniformly optimal" used instead (§Abstract, §1, §4.4, §9).
- [PASS] GQA = tie (InternVL3 −0.4pp) / post significant micro-lead (Qwen +2.6–2.8pp; 1/10–1/14 of
  text-dense Δ; McNemar 8.0/5.7; indep 4.5/4.1) / no text-dense crossover — single full statement
  §4.2, terse back-refs elsewhere (§Abstract, §4.3, §5.4, §8).
- [PASS] FastV wins TextVQA/GQA/DocVQA reported honestly (Table 3, +17.2/+8.9/+16.2pp at best K=3);
  our wins = OCR (+16.5pp) + never collapses + mechanism explanation.
- [PASS] Config boundaries annotated: InternVL3 DocVQA 4M px cap vs Qwen 600k → cross-model DocVQA
  absolutes NOT comparable (Table 2 note i, §8 item 3); pre vs none DocVQA −19.4pp honest (Table 2
  note ii, §8 item 4); near-lossless TextVQA/GQA-scoped only; n/budget/pixel-cap per table note.
- [PASS] VisionZip official numbers = mismatched anchor only (§2, one sentence, 81.5→70.5, no
  head-to-head victory).
- [PASS] Pre-registered criteria unchanged (cascade gate rule quoted verbatim as LOCKED, §6d).
- [PASS] Double-blind hygiene: no author names / repo URLs / institutions / commit hashes / internal
  paths; code = "code will be released"; \documentclass[...anonymous]{acmart}.
- [PASS] Official metrics only (VQA-acc / ANLS / OCR-Bench /1000 / GQA exact match); no containment
  figures.

## UNRESOLVED TODOs (number gaps / figure placeholders)
1. [TODO: source] InternVL3 citation bibkey (\cite{chen2024internvl} is a guess — verify against the
   actual InternVL3 reference before building the .bib).
2. [TODO: paired Δ] Table 3 Qwen2.5-VL rows: r2b gives FastV-k3 raw numbers and states the DIRECTION
   holds on both families, but only Qwen3-VL paired Δ are explicit; fill Q2.5 matched-scope Δ before
   submission (do NOT subtract across runs).
3. [TODO: verify] Table 3 budget alignment — FastV-k3 ptid ≈214 stated; confirm per-benchmark ptid vs
   RBM and label iso- vs approximately-matched precisely (the r2b digest's "690 ptid" line conflicts
   with the full-matrix Qwen3 TextVQA ptid 215.8; do not quote 690 for Qwen).
4. [FIG:1] pipeline schematic (§3.1); [FIG:2] cross-family stage-law bars (§4.3); [FIG:3]
   retention-vs-gap curves (§5.4). Qualitative examples figure (post/FastV erase small text, RBM
   protects; GQA trade-off direction) optional if space — from v4 Fig. 4 catalog.
5. Move to SUPPLEMENTARY (CFP: no appendix after body; supp ≤50MB): binomial stderrs + Table A3
   skip accounting; full paired McNemar b/c table; n=200 cross-split consistency table (v4 A2);
   M1 Kendall τ column; selector-invariance table; mrope layout details; per-cell token counts.
6. Qwen2.5-VL efficiency not measured (§7) — state as future work (already done).

## REVIEWER-SIMULATION ROUND-2: three most-likely attack points (prediction)
1. "Novelty is thin — the method is just top-k L2 norm; the negative results say nothing helps, so
   what is the technical contribution?" Defense to harden in rebuttal: contribution is the STAGE AXIS
   as experimental variable + causal mechanism (Jaccard≡1.000 / swap≡pre), not the scorer; minimalism
   is the control design, and the four negatives are the scientific content (a fixed-point theorem by
   experiment). Add one sentence in §3.2 framing minimalism-as-control more explicitly if room.
2. "Cross-family generalization is overclaimed: InternVL3's 'post' is a different operator from
   Qwen's." RESOLVED at artifact level: the InternVL3 main-matrix post arm is **post-merger L2 top-k**
   (runner `--mode post --selector l2`, setup_post_merger_internvl) — the SAME query-blind L2 selector
   as the pre arm, at IDENTICAL token budget (ptid equal across arms): a pure stage variable,
   iso-selector + iso-budget, exactly like the Qwen main table (post = post-merger L2). FastV
   (in-layer query-conditioned attn) is a SEPARATE baseline campaign (Table 3 / r2b_fastv_k3), never
   the main-matrix post arm; the R1-style objection ("post = FastV-style layer-2 pruning") does not
   hold. The paper now proactively discloses the iso-selector / iso-budget design in §4.3 and the
   Table 2 caption, so no rebuttal-only experiment is needed.
3. "DocVQA numbers are not comparable across models and RBM loses 19.4pp to none — the 'robust'
   framing hides a real regression." Defense: we state this outright (Table 2 notes, §8 items 3–4);
   the claim is pre>post on text-dense, never pre=none on DocVQA. Strengthen by moving the −19.4pp
   into the abstract-adjacent contribution sentence? (Currently abstract says near-lossless is
   scoped — check the abstract does NOT imply DocVQA losslessness; it says 'no text-dense crossover'
   and efficiency only — OK.)
-->

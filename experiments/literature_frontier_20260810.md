# 2025--2026 visual-token compression frontier audit

**Date:** 2026-08-10
**Scope:** CVPR / ICCV / NeurIPS / ACL-family papers that suggest mechanisms
outside plain attention ranking, routers, serial cascades, and simple
representation-delta/L2/edge scores. This is a mechanism audit, not a claim that
paper-reported numbers are directly comparable with VLA's Qwen3-VL-8B protocol.

## 1. Verification policy and current-project constraints

Verification levels used below:

- **Venue-verified:** title and abstract occur on the official CVF, NeurIPS, or
  ACL Anthology proceedings page.
- **Code-verified:** the advertised Git repository resolved on 2026-08-10 and
  its HEAD/files were inspected. A repository that only contains a README is
  not counted as usable code.
- **Author claim:** performance numbers in an official abstract or README; not
  independently reproduced here.

The relevant VLA facts are unusually constraining:

1. Plain RBM (pre-merger mean-L2 unit selection) is the frozen method. It is the
   robust text-dense/OCR parent; FastV-k3 is the stronger query-conditioned
   parent on TextVQA/DocVQA/GQA.
2. Serial cascade and pointwise RankBridge fusion are locked NO-GO results.
   RankBridge at `rho=0.2` beat FastV-k3 in all four locked cells but remained
   11.6 pp below RBM on OCRBench. Any new proposal must explain why it is not
   another pointwise rank mixture or irreversible two-stage selector.
3. One A40 46 GB means a useful first gate should fit below 6 GPU-hours and use
   the existing four paired n=64/n=200 subsets and official scorers.
4. A method reported on LLaVA cannot be called a reproduced SOTA baseline after
   a principle-only Qwen port. The port can test a mechanism; a SOTA claim needs
   same-model, same-budget, same-scope evaluation.

## 2. Verified frontier map

| Paper | Venue | Mechanism beyond a scalar attention/L2 rank | Code status checked 2026-08-10 | Relevance to VLA |
|---|---|---|---|---|
| **DivPrune: Diversity-based Visual Token Pruning** | CVPR 2025 | Max-min diversity/coreset selection in feature space | Public LLaVA code is referenced by later verified repos | Establishes that subset geometry, not only importance, matters; query-blind |
| **CDPruner: Beyond Attention or Similarity** | NeurIPS 2025 | Conditional DPP: a determinant objective jointly rewards instruction relevance and subset diversity | Usable LLaVA-1.5/1.6 tree, HEAD `9541616`; 60 Python files | Best direct antidote to RankBridge's pointwise fusion failure |
| **Why 1+1<1 ... Multi-Objective Balanced Covering (MoB)** | NeurIPS 2025 | Bi-objective epsilon-covering with greedy radius trading and a Hausdorff error bound | Official abstract says code “will be made available”; no official repo located | Direct theoretical warning against naive rank integration; implementation unavailable |
| **Balanced Token Pruning (BTP)** | NeurIPS 2025 | Calibration chooses stage policies by balancing local-output and downstream/global discrepancy | Repository resolves at HEAD `9682db0` | Useful calibration principle, but introduces calibration-set dependence and multiple stages |
| **SCOPE** | NeurIPS 2025 | Greedy submodular-style saliency plus marginal set coverage | No code link on official page located | Relevant coverage objective, but still attention-saliency based |
| **Token Pruning in MLLMs: Are We Solving the Right Problem?** | Findings of ACL 2025 | Controlled study questions attention, language conditioning, duplication, and shows that sophisticated selectors can trail random pruning | Official page says code is in supplementary material | Independent warning that selector gains need random and coverage controls, not another method to port |
| **VFlowOpt** | ICCV 2025 | Recycles pruned tokens and optimizes a progressive policy using full-vs-pruned information-flow discrepancy | Official page/paper verified; no official code link on page | Aggregation/calibration direction; too close to staged-policy search for first gate |
| **FEATHER** | ICCV 2025 | Early uniform spatial coverage followed by ensemble criteria; exposes top-of-image pruning bias | Official page/paper verified; no code link on page | Strong reminder to include spatial/token-type retention diagnostics |
| **Representation Shift** | ICCV 2025 | Token representation-change metric compatible with FlashAttention | Public repository HEAD `a528e02` | Same broad family as TransPrune/V2Drop; not a sufficiently distinct VLA bet |
| **MetaCompress (Rethinking Token Reduction...)** | CVPR 2026 | Learned prompt-agnostic compression mapping for multi-turn VQA | Repository resolves at HEAD `70a620e` | Important multi-turn direction, but it is learned compression and not a sub-6-GPU-hour first move |
| **CORE** | CVPR 2026 | External segmentation decoder creates object masks; merge to ordered object-centric tokens | Advertised repo HEAD `1a86030`, but only 2 files/0 Python; README says code is being cleaned | Semantically attractive, currently non-reproducible and risky for tiny text/chart marks |
| **CoIn** | CVPR 2026 | Joint optimal subset selection: intrinsic saliency + cross-modal alignment + volume-based global coverage | No code link on official page | Convergent evidence for set-level coverage; exact Qwen port would be a reimplementation |
| **ZOO-Prune** | CVPR 2026 | Zeroth-order finite-difference estimate of projector sensitivity, then sensitivity-aware diverse selection | Usable LLaVA tree, HEAD `c7d05ca`; implementation and launch script present | Provides a functional-sensitivity signal that can be transplanted specifically to the native merger |
| **When Token Pruning is Worse than Random** | CVPR 2026 | Defines token information by output-probability change on removal; finds a task/model-dependent “information horizon” | Usable LLaVA tree, HEAD `75909b2`; 64 Python files | Strongest evidence for token *lifetime* as an axis separate from token identity |
| **DocPrune** | CVPR 2026 | Document background/question/comprehension-aware pruning and automatic start-layer choice | Official paper verified; no code link on official page | Domain-specific support for late/task-dependent pruning; limited to long-document QA |
| **IF-Prune** | CVPR 2026 | A trained small VLM with a variational information bottleneck predicts non-informative tokens | Official paper verified; code not linked on official page | High reported compression, but auxiliary training/model makes it unsuitable for the immediate gate |
| **HTC-VLM (Hybrid Token Compression)** | CVPR 2026 | Learned continuous-detail and discrete-semantic channels compressed through a one-token bottleneck | Official paper verified; no code link on official page | Architecturally distinct but requires retraining; not a drop-in inference method |

Two boundaries matter. First, MacTok is a continuous tokenizer for **image
generation**, not an inference-time visual-token reducer for a frozen LVLM; its
gFID result is not a comparable VQA SOTA target. Second, Representation Shift
and its representation-delta relatives are already covered by the
TransPrune/V2Drop direction in the user brief, so they are not counted among the
three opportunities below.

## 3. Opportunity 1 (preferred): Deferred RBM / anchor-preserving burn-in

### Mechanism and why it is genuinely new relative to the failed gate

Compute the ordinary pre-merger RBM unit rank, but do **not** delete anything at
the vision/LLM boundary. Let all native merged visual tokens participate in the
first `K=3` LLM layers, then retain exactly the units selected by RBM and delete
the rest. The kept-token identity is therefore RBM's, while the kept hidden
states have already absorbed context from tokens that will be removed.

The existing harness already implements this arm as:

```text
--mode rankbridge --rb-fuse quota --rb-rho 1.0 --fastv-k 3 --r 0.75
```

The current dry check proves `rho=1` has the same **keep set** as pre-rank
top-k. It does *not* prove hidden-state or accuracy equivalence to plain RBM:
RankBridge keeps the full sequence through layer 3. No rho=1 result was found
in `runs/`, `experiments/`, or `output/` on 2026-08-10.

This is not another rank fusion: there is only one selector and one deletion.
It is also not the failed cascade: no candidate is deleted before the final
choice. Relative to AOT, information transfer is implicit through three frozen
self-attention blocks rather than explicit optimal-transport redistribution.
Relative to TransPrune/V2Drop, layerwise representation change does not decide
token identity; depth controls token lifetime only. It operationalizes the
CVPR-2026 Information-Horizon insight in a merger-aware setting.

### Falsifiable prediction and gate

Prediction: because the final identities are RBM-identical, OCRBench should
remain within 1 pp of RBM, while dense-context burn-in should recover at least
part of FastV's TextVQA/DocVQA/GQA advantage. The sharp falsifier is OCRBench:
if the 11.6 pp RankBridge gap was caused by its 80% attention-selected seats,
removing all such seats should close most of that gap.

1. Run n=64 on the four locked subsets, K=3, keep 25%, seed 0, same pixel caps.
2. **GO to n=200 only if** every benchmark is at least
   `max(RBM, FastV-k3)-1 pp`, and one benchmark strictly beats both parents.
3. Log exact kept-index equality with RBM and compare survivor hidden states at
   deletion. If keep identities differ, the experiment is invalid, not a loss.
4. Treat this as a separately pre-registered “deferred-RBM” gate, not a reopened
   search over RankBridge rho. Do not search K unless the K=3 mechanism passes.

Estimated cost: about 0.7 GPU-hour for n=64 and about 2.3 GPU-hours for a locked
n=200 follow-up, using the measured RankBridge wall-time scale. No code change
should be necessary beyond a named run script/result label.

Main risk: extra early-layer interaction can contaminate rather than enrich RBM
anchors, and it forfeits RBM's first-three-layer compute saving. Even if accuracy
wins, efficiency must be reported as a different Pareto point, not “free.”

## 4. Opportunity 2: Pre-feature conditional DPP at the FastV decision layer

### Mechanism

Cache one normalized vector `z_i` per native merger unit from the same
pre-merger tensor RBM uses (flatten or pool the four patch vectors). At FastV
layer 3, use query-conditioned attention only as a positive quality term `q_i`.
Select all K survivors jointly with greedy MAP inference under

```text
L = diag(q) @ (Z @ Z.T) @ diag(q).
```

The determinant rewards a high-quality set that spans distinct pre-merger
directions. This is the cleanest Qwen-specific synthesis of NeurIPS-2025
CDPruner and CVPR-2026 CoIn. It differs from RankBridge because a token's value
depends on the tokens already selected; there is no quota, rank sum, or scalar
top-k fusion. It differs from DivPrune by conditioning set quality on the
question. It differs from AOT because it does not aggregate discarded tokens or
solve transport; it chooses a representative subset. It differs from RBM by
optimizing coverage rather than feature magnitude.

### Falsifiable prediction and gate

Prediction: relative to FastV, determinant coverage reduces duplicate object
tokens and retains minority text/stroke directions; relative to RBM, `q_i`
retains question relevance. Use one fixed quality scaling copied from the
published CDPruner formulation; do not tune theta on all four benchmarks.

Gate sequence:

1. Pure-function dry checks: PSD/symmetry, exact K, deterministic tie handling,
   per-image budgets, native order restoration, and `q=constant` diversity-only
   degeneration.
2. n=64 four-benchmark gate with the same GO rule as Opportunity 1.
3. Add diagnostics before any n=200 run: RBM/FastV/DPP Jaccard, spatial coverage,
   and OCR edge-density retention. A DPP win without distinct keep sets is a bug.

Expected engineering: 1--2 days in `baselines_hf.py`; GPU about 0.7 + 2.3 hours
if it reaches the locked gate. Memory is O(N^2) for the Gram matrix; greedy
fast-MAP should be used rather than enumerating subsets.

Main risks: cached pre-merger vectors and post-merger attention live in different
representational geometries; determinant objectives can overselect orthogonal
noise; CDPruner's released implementation is coupled to CLIP/LLaVA and cannot be
claimed as an official Qwen reproduction. Most importantly, if OCR loss comes
from the merger itself rather than duplicate selection, a set objective cannot
recover information that is already destroyed.

## 5. Opportunity 3: Native-merger Jacobian sensitivity

### Mechanism

Replace RBM's mean feature norm with the local functional sensitivity of each
four-patch unit to Qwen's actual native merger. For unit `x_i` and fixed random
directions `u_j`, estimate

```text
s_i = mean_j || M(x_i + h u_j) - M(x_i - h u_j) ||_2 / (2h),
```

where `M` is only the lightweight native merger/projector. Select top-K units
before the merger exactly as RBM does. This adapts CVPR-2026 ZOO-Prune to the
component VLA has shown to be causally important, rather than copying its
LLaVA MLP projector port. ZOO-Prune's released code uses 64 random directions,
`h=0.01`, and then multiplies sensitivity by a max-min diversity score; for the
first VLA probe, isolate the sensitivity signal before adding another set
objective.

This differs from TransPrune/V2Drop: it measures the derivative of a specific
function at one architectural boundary, not token displacement between network
layers. It differs from RBM because two equal-norm units can have very different
merger gains. It differs from AOT because no dropped content is redistributed.

### Falsifiable prediction and stop rule

Before a benchmark run, measure Spearman correlation and top-25% Jaccard against
RBM on 64 cached samples. **Stop immediately** if Spearman is above 0.95 or
Jaccard above 0.90: the signal is operationally the same selector and cannot
justify a new campaign. Otherwise run n=64 on OCRBench + GQA first.

GO to the other two benchmarks only if it is at least RBM-1 pp on OCRBench and
FastV-k3-1 pp on GQA. A fixed RNG seed, symmetric perturbations, fp32 score
accumulation, and a small direction-count ablation (8 versus 16 on the same
samples) are mandatory; score instability above 5% top-K disagreement is a
NO-GO. Expected GPU cost is under 2 hours for the diagnostic gate, but the exact
overhead depends on whether the Qwen merger can be vectorized over perturbations.

Main risks: local Jacobian norm can measure amplification/noise rather than task
information; the native merger may be close enough to linear that the score is
nearly constant or reduces to a weight-norm artifact; stochastic ranking hurts
reproducibility. The public ZOO code is substantial but is LLaVA-specific and
its reported result includes a diversity term, so a Qwen merger-Jacobian result
must be presented as a new mechanism test, not “ZOO-Prune reproduced.”

## 6. Priority and decision

| Rank | Opportunity | New code | First falsifier | Best case | Principal downside |
|---|---|---:|---|---|---|
| **1** | Deferred RBM | None | OCRBench fails to return to RBM | RBM identities + early implicit context close the OCR/query trade-off | Less prefill saving in first K layers |
| **2** | Conditional pre-feature DPP | Moderate | OCR remains >1 pp below RBM | Set-level objective succeeds where pointwise fusion failed | Qwen port is not official CDPruner reproduction |
| **3** | Merger-Jacobian sensitivity | Moderate | Score collapses to RBM or is unstable | Native-merger causal score beats magnitude | Stochastic/functional sensitivity may be non-semantic |

**Recommendation:** run only Opportunity 1 first. It is the highest-information
experiment per GPU-hour and reuses a path whose keep-set invariant is already
tested. A clean loss would rule out “let dropped tokens write into RBM anchors
before deletion” and strengthen the fixed-point story; a win would yield a
simple, mechanistically distinct method without reopening rho tuning. Only if it
fails narrowly should Opportunity 2 be implemented. Opportunity 3 is a fallback
selector study, not the first SOTA attempt.

## 7. Source ledger

Official proceedings pages (accessed 2026-08-10):

- CVPR 2025 DivPrune:
  https://openaccess.thecvf.com/content/CVPR2025/html/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.html
- ICCV 2025 VFlowOpt, FEATHER, Representation Shift:
  https://openaccess.thecvf.com/content/ICCV2025/html/Yang_VFlowOpt_A_Token_Pruning_Framework_for_LMMs_with_Visual_Information_ICCV_2025_paper.html
  https://openaccess.thecvf.com/content/ICCV2025/html/Endo_Feather_the_Throttle_Revisiting_Visual_Token_Pruning_for_Vision-Language_Model_ICCV_2025_paper.html
  https://openaccess.thecvf.com/content/ICCV2025/html/Choi_Representation_Shift_Unifying_Token_Compression_with_FlashAttention_ICCV_2025_paper.html
- NeurIPS 2025 CDPruner, MoB, BTP, SCOPE:
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/2433fec2144ccf5fea1c9c5ebdbc3924-Abstract-Conference.html
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/6818dcc65fdf3cbd4b05770fb957803e-Abstract-Conference.html
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/5aab3631d0d3131281fb88265db69480-Abstract-Conference.html
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/ec6b4456c2bdfd04002d7984043c4936-Abstract-Conference.html
- Findings of ACL 2025 controlled token-pruning audit:
  https://aclanthology.org/2025.findings-acl.802/
- CVPR 2026 MetaCompress, CORE, CoIn, ZOO-Prune, Information Horizon,
  DocPrune, IF-Prune, HTC-VLM:
  https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Rethinking_Token_Reduction_for_Large_Vision-Language_Models_CVPR_2026_paper.html
  https://openaccess.thecvf.com/content/CVPR2026/html/Lei_CORE_Compact_Object-centric_REpresentations_as_a_New_Paradigm_for_Token_Merging_in_LVLMs_CVPR_2026_paper.html
  https://openaccess.thecvf.com/content/CVPR2026/html/Du_CoIn_Coverage_and_Informativeness-Guided_Token_Reduction_for_Efficient_Large_Multimodal_CVPR_2026_paper.html
  https://openaccess.thecvf.com/content/CVPR2026/html/Kim_ZOO-Prune_Training-Free_Token_Pruning_via_Zeroth-Order_Gradient_Estimation_in_Vision-Language_CVPR_2026_paper.html
  https://openaccess.thecvf.com/content/CVPR2026/html/Wang_When_Token_Pruning_is_Worse_than_Random_Understanding_Visual_Token_CVPR_2026_paper.html
  https://openaccess.thecvf.com/content/CVPR2026/html/Choi_DocPrune_Efficient_Document_Question_Answering_via_Background_Question_and_Comprehension-aware_CVPR_2026_paper.html
  https://openaccess.thecvf.com/content/CVPR2026/html/Sun_IF-Prune_Information-Flow_Guided_Token_Pruning_for_Efficient_Vision-Language_Models_CVPR_2026_paper.html
  https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Hybrid_Token_Compression_for_Vision-Language_Models_CVPR_2026_paper.html

Code repositories inspected at the HEADs recorded in Section 2:

- https://github.com/Theia-4869/CDPruner
- https://github.com/AIM-SKKU/ZOO-Prune
- https://github.com/YahongWang1/Information-Horizon
- https://github.com/jingyulei/CORE
- https://github.com/MArSha1147/MetaCompress
- https://github.com/EmbodiedCity/NeurIPS2025-Balanced-Token-Pruning
- https://github.com/mlvlab/Representation-Shift

Local project evidence: `STATE.md`, `experiments/rankbridge_gate.md`,
`experiments/cascade_gate.md`, and
`src/v3_premerger/baselines_hf.py` (especially the documented `rho=1`
degeneracy and full-sequence-through-K execution semantics).

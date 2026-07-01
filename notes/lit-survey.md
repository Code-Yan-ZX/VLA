# Literature Survey — VLM Visual Token Compression (2023–2026)

> Project: novel visual-token-compression method for VLMs → validate on standard multimodal benchmarks → target Q1/Q2 SCI.
> Hardware constraint: 1× A40 46GB (single card, serial jobs). This biases feasibility toward training-free / light-LoRA methods.
> Author: Lit subagent · Date: 2026-07-01 · Status: P1 deliverable.
> Primary-source verified: every arXiv ID below was confirmed via the arXiv API on 2026-07-01.

---

## 1. Introduction & scope

Visual-token compression has become the dominant inference-efficiency lever for Multimodal/Vision-Language Large Language Models (MLLMs/VLMs), because the visual branch typically emits hundreds-to-thousands of tokens per image while the text branch emits tens, and self-attention is quadratic. The field split into three families after FastV (ECCV'24) popularized post-layer-2 attention-based pruning:

1. **Encoder-side selection** — pick informative ViT tokens *before* the LLM (VisionZip, VTC-CLS, FasterVLM).
2. **Inside-LLM pruning** — discard/merge tokens at an early LLM layer using attention/CLS scores (FastV, SparseVLM, G-Prune, AdaptPrune, DyCoke).
3. **Projector-level compression** — train a projector that emits far fewer tokens (TokenPacker, FlashSloth, LLaVA-PruMerge light-tune).

A second axis is whether the method is **query/text-aware** (compress based on the question, e.g. SparseVLM, Q-Zoom, AdaptMerge, SparseVILA-decode) vs **query-agnostic** (FastV, VisionZip, VTC-CLS). A third axis is whether compression is **fixed-ratio** or **content-/complexity-adaptive** per image (GlimpsePrune, Q-Zoom).

This survey covers 23 methods with verified primary sources, builds a full comparison table, then evaluates 5 candidate research gaps against novelty × feasibility-on-1×A40, and recommends one.

---

## 2. Comparison table

Legend — Training-free: **TF**=fully training-free, **LT**=light-tuning/adapter-only, **FT**=full/multi-stage training. Throughput: **Y**=reports real wall-clock latency/throughput (CUDA latency, prefill time, e2e speedup), **N**=FLOPs/token-count only, **DEPLOY**=measured under a serving engine (vllm/lmdeploy/TRT-LLM/SGLang). "Q2.5/3-VL" = explicit support for Qwen2.5-VL / Qwen3-VL (M-RoPE, native variable resolution).

| # | Method | Year/Venue | Train-free | Base model(s) | Benchmarks reported | Compression ratio | Accuracy (key metric + Δ vs full-token) | Real throughput? | Q2.5/3-VL? | Code | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **FastV** "Image worth 1/2 Tokens" | 2024 / ECCV'24 (Oral) | TF | LLaVA-1.5 (7B/13B), QwenVL-Chat, Video-LLaVA | GQA, MMBench, VQA, TextVQA, ScienceQA, POPE, MME, video-QA | ~50% visual tokens pruned; ~45% FLOPs (13B) | Pareto: 13B <7B-FLOPs, accuracy held | **N** (FLOPs only) | No | Yes (github) | Canonical baseline; prunes after LLM layer 2 by attention. Largest drops on text-dense tasks. |
| 2 | **FasterVLM** "[CLS] Attn" | 2024-12 / arXiv | TF | LLaVA-1.5, LLaVA-NeXT | GQA, TextVQA, ScienceQA, MMBench, POPE, MME | up to ~95% of visual tokens | ~90% of original perf retained at high pruning | **N** (FLOPs) | No | Yes | arXiv 2412.01818 (later retitled VisPruner extension). CLIP [CLS] attention re-ranking. |
| 3 | **SparseVLM** "Visual Token Sparsification" | 2024-10 / ICML'25 | TF | LLaVA-1.5, LLaVA-NeXT, video variants | GQA, TextVQA, ScienceQA, MMBench, POPE, MME | ~54% FLOPs reduction | **97% of original accuracy** retained | **Y** — "37% CUDA latency" (measured, not serving-engine) | No | Yes | Text-guided self-attn sparsification + token recycling. Closest to deployment-level among classic TF methods. |
| 4 | **VisionZip** "Longer is Better but Not Necessary" | 2024-12 / arXiv | TF | LLaVA-NeXT (7B/13B), video-LLaVA | GQA, MMBench, TextVQA, ScienceQA, MMMU, MME, POPE | up to **8× fewer** visual tokens | >5% acc gain over prior SOTA at same compression | **Y** — **8× faster prefilling time** (real wall-clock prefill) | No | Yes | Encoder-side selection before LLM. NeXT-13B infers faster than 7B baseline. |
| 5 | **VTC-CLS** "[CLS] Token Tells Everything" | 2024-12 / arXiv | TF | LLaVA-1.5/NeXT, Qwen2-VL | GQA, TextVQA, ScienceQA, MMBench, POPE, MME, MMMU | high ratios (~89% per citing work) | SOTA among TF token-compression across tasks | **N** (FLOPs) | Qwen2-VL (not 2.5/3) | Yes | Ensembled [CLS] attn across ViT layers. |
| 6 | **TokenPacker** "Efficient Visual Projector" | 2024-07 / **IJCV'25** | FT (multi-stage) | LLaVA-1.5, high-res MLLMs | TextVQA, ChartQA, DocVQA, ScienceQA, GQA, MMBench, MMMU | **75–89% reduction** (576→64/128) | comparable/better than MLP projector at 4–9× fewer tokens | **N** (FLOPs) | No | Yes | Coarse-to-fine projector; needs coarse-pretrain → refine → instruction-tune. |
| 7 | **LLaVA-PruMerge** | 2024-03 / arXiv (v6) | LT (hybrid) | LLaVA-1.5 | VQA, GQA, TextVQA, ScienceQA, POPE, MME | **~14× compression** (576→~40) | comparable perf across VQA/reasoning | **N** (FLOPs) | No | Yes | Pruning training-free (CLIP CLS sparsity) + optional light merging adapter. |
| 8 | **ToMe** (original ViT) | 2023 / ICLR'23 | TF | ViT-L/H (image, video, audio) | ImageNet-1k, Kinetics-400, AudioSet | up to ~2× token reduction (r=16/24) | 0.2–0.3% drop image; 0.4% mAP audio | **Y** — real ~2× wall-clock on ViT | n/a (ViT) | Yes | Bolya et al. Not a VLM but the merging origin. Bipartite soft matching. |
| 9 | **AdaptMerge** (ToMe→LLaVA) | 2025 / EMNLP'25 Findings | TF | LLaVA-family | standard VQA suite | adaptive, language-guided | closes most of OCR gap vs plain ToMe | **N** | No | Yes | Nearest "ToMe for LLaVA"; query-conditioned merging. |
| 10 | **G-Prune** "Graph perspective" | 2025-01 / arXiv | TF | LLaVA-NeXT | incl. TextVQA | 63.57% FLOPs reduced | **TextVQA only −2.34%** (explicitly studies fine-grained/OCR) | **N** (FLOPs) | No | Yes | Graph-based token importance; foreground+background both retained. |
| 11 | **AdaptPrune** (Multi-Cue) | 2025-03 / arXiv | TF | cross-LVLM | standard | moderate–high | competitive at high ratios | **N** (FLOPs) | No | Yes | Attention + spatial + similarity (NMS) cues. |
| 12 | **Q-Zoom** (query-aware) | 2026 / arXiv | LT (gating+SD-RPN self-distill) | **Qwen2.5-VL-7B (primary)**; also Qwen3-VL, LLaVA, RL image-thinking | Document & OCR, High-Resolution | inference-time | **2.52× (Doc/OCR), 4.39× (HR)** speedup, no acc loss | **Y** — reported as inference time | **Yes (2.5-VL primary; 3-VL)** | TBD | Strongest explicit Qwen2.5-VL + query-aware result. |
| 13 | **GlimpsePrune** (content-adaptive) | 2025-08 / arXiv | TF | LLaVA-NeXT, **Qwen2.5-VL-7B-Instruct** | free-form VQA, OCR-heavy | prunes **92.6%** visual tokens | retains baseline; GlimpsePrune+ hits **110%** | **N** (token/FLOPs) | **Yes (2.5-VL)** | Yes (HF ckpt) | Single "glimpse" forward → dynamic per-image budget. |
| 14 | **SparseVILA** (decode query-aware) | 2025 / ICCV'25 | TF | architecture-agnostic (AWQ pipeline) | long-video, doc, reasoning | prefill prune + decode retrieval | accuracy *gains* on doc/reasoning | **Y — 4.0× prefill, 2.5× decode, 2.6× e2e** | partial (Qwen2-VL family via AWQ) | Yes | **Closest to a deployed-engine wall-clock result, but on AWQ pipeline — NOT vllm/lmdeploy.** |
| 15 | **DyCoke** (video) | 2024-11 / CVPR'25 | TF | LLaVA-NeXT-Video, Video-LLaVA | video-QA suite | temporal merging + dynamic KV-cache spatial prune | improves acc vs FastV | **Y — 1.5× inference, 1.4× memory** (research code, not serving engine) | No | Yes | Best-known training-free video baseline. |
| 16 | **DynaTok** (video) | 2026 / arXiv | TF | LLaVA-OneVision, LLaVA-Video | MVBench, LongVideoBench, MLVU, VideoMME | **90% token reduction** | >95% accuracy retained | **N** (token-count) | No | TBD | Temporal+spatial budget allocation (EMA memory). |
| 17 | **ShaRP** (shallow-layer video) | 2025 / arXiv | TF | video LLaVA-family | video-QA | shallow-layer prune | addresses FastV deep-layer degradation | **N** | No | TBD | Complements DyCoke. |
| 18 | **FlashSloth** | 2024-12 / arXiv | LT/FT | builds on Qwen2-VL | efficient high-res MLLM | token reduction | competitive | **N** | Qwen2-VL | Yes | Efficient MLLM line, less cited as a compression method per se. |
| 19 | **LLaVA-UHD v3** | 2025-11 / arXiv | LT (native-res encoder) | LLaVA + native-res | standard + TTFT | native variable resolution | TTFT **1.9× lower** vs Qwen2-VL | **Y** — TTFT reported | compares to Qwen2-VL | Yes | Encoder-resolution, adjacent to compression. |
| 20 | **LLaMA-VID** | 2023-11 / arXiv | FT | LLaMA-based | video QA | 1 token/image-side per frame (context) | trade-off accepted | **N** | No | Yes | Foundational "few tokens" idea (single-context-token). |
| 21 | **HiRED** | NSF PAR | TF | LLaVA-1.5 | fine-grained transcription | **40% budget** (1152 tokens) | preserves fine-grained acc | **N** | No | TBD | Targets the OCR-degradation problem directly. |
| 22 | **VFlowOpt** | 2025 / ICCV'25 | LT/FT | LLaVA-1.5 | fine-grained | optimizes pruning to preserve fine-grained info | better OCR retention | **N** | No | TBD | Flow-based optimization for fine-grained preservation. |
| 23 | **Eval-Framework** "Are We Using the Right Benchmark?" | 2025-10 / arXiv | (meta-eval) | surveys many | proposes dedicated VTC eval | — | argues current benchmarks miss the real cost | **partial** — critiques missing throughput eval | n/a | TBD | **Most directly relevant prior work to the deployment-throughput gap.** arXiv 2510.07143. |

### 2.1 Throughput-reporting summary (the suspected gap)

Of 23 methods, only **6 report any real wall-clock number** (SparseVLM, VisionZip, SparseVILA, DyCoke, Q-Zoom, LLaVA-UHD; plus original ToMe on ViT). Of those:
- **SparseVILA** is the most deployment-flavored (AWQ pipeline, 4.0× prefill / 2.6× e2e) but is **not** on vllm/lmdeploy/TRT-LLM/SGLang.
- **VisionZip / SparseVLM / DyCoke** measure raw CUDA latency or prefill time on research code.
- **None** of the 23 methods measure served throughput (tokens/sec or req/sec) **inside a production serving engine** (vllm/lmdeploy/TRT-LLM/SGLang).

The only serving-engine artifact found is **vLLM RFC #45098** (`--image-pruning-rate` flag), which is **unfinished infrastructure**, not a published method with benchmarks. ⇒ **The deployment-throughput gap is genuinely open.**

---

## 3. Candidate research gaps (evaluated, not pre-judged)

### Gap A — Real deployment-level speedup integrated with vllm/lmdeploy
- **Description:** Take a training-free compressor (FastV/SparseVLM/VisionZip-style) and wire it *inside* a serving engine's multimodal processor (vLLM `--image-pruning-rate`-style hook or a custom lmdeploy pipeline), then report served tokens/sec, req/sec, TTFT, and KV-cache memory — under continuous batching. Pair with a small benchmark suite that measures *deployment* cost, not just FLOPs.
- **Novelty: 5/5** — no published method does this; vLLM RFC #45098 is unfinished; Eval-Framework (2510.07143) explicitly calls out the missing throughput evaluation.
- **Feasibility-on-1×A40: 4/5** — vLLM runs on a single A40 for 7B models; no training needed (training-free compressor); main cost is engineering + eval harness. Risk: vLLM internals + transformers 5.x API drift (P0 noted). Mitigated by starting on LLaVA-1.5 (well-supported) before Qwen2.5-VL.
- **Evidence it exploits:** FastV/VTC-CLS/G-Prune all report FLOPs only; SparseVILA's AWQ result shows real numbers exist but no one ports to vLLM; Eval-Framework paper demands it.
- **Fastest non-trivial result:** port FastV to vLLM, report served tokens/sec at 50% pruning on GQA/TextVQA + a throughput curve. Repeatable in days on 1 A40.

### Gap B — Preserving accuracy on token-sensitive tasks (OCR / charts / docs) under compression
- **Description:** Characterize & fix the disproportionate degradation of FastV-style pruning on TextVQA / ChartQA / DocVQA / OCR. Build a query/content-aware compressor that recovers fine-grained accuracy.
- **Novelty: 3/5** — already crowded: G-Prune, GlimpsePrune, Q-Zoom, HiRED, VFlowOpt all target this. A new entrant must beat SOTA.
- **Feasibility-on-1×A40: 4/5** — eval-only is cheap; but SOTA-beating likely needs light training (Q-Zoom self-distill), still within budget.
- **Evidence:** FastV drops most on text-dense tasks; G-Prune shows −2.34% on TextVQA; Q-Zoom claims no loss. The "open" part is narrower than Gap A.

### Gap C — Test-time / query-aware compression
- **Description:** Compress conditioned on the question (SparseVLM/Q-Zoom/AdaptMerge direction), per-query.
- **Novelty: 2.5/5** — Q-Zoom (2026) and SparseVILA-decode already do strong query-aware compression; crowded.
- **Feasibility-on-1×A40: 3/5** — Q-Zoom needs light self-distillation training; doable but not trivial.
- **Evidence:** Q-Zoom 4.39× HR speedup; SparseVILA 2.6× e2e. Marginal novelty over existing work.

### Gap D — Content-/complexity-adaptive per-image budget
- **Description:** Variable pruning ratio per image based on a cheap complexity signal (GlimpsePrune-style single forward).
- **Novelty: 3/5** — GlimpsePrune (2025) already does this well (92.6% prune, 110% with +). Strong recent prior art.
- **Feasibility-on-1×A40: 4/5** — training-free, eval-cheap.
- **Evidence:** GlimpsePrune shows fixed ratios are suboptimal; but the core idea is taken.

### Gap E — Video-VLM token compression
- **Description:** Extend compression to video (DyCoke, DynaTok, ShaRP territory).
- **Novelty: 3/5** — active but less saturated than image; DyCoke/DynaTok are strong.
- **Feasibility-on-1×A40: 2/5** — video eval (MVBench/LongVideoBench/MLVU) is GPU-time heavy; long videos blow the single-card serial budget. **Lowest feasibility.**

### Gap scorecard
| Gap | Novelty | Feasibility (1×A40) | Product | Verdict |
|---|---|---|---|---|
| **A. Real deploy-throughput (vllm/lmdeploy)** | **5** | **4** | **20** | **★ RECOMMENDED** |
| B. OCR/chart/doc preservation | 3 | 4 | 12 | viable, crowded |
| C. Query-aware compression | 2.5 | 3 | 7.5 | crowded |
| D. Content-adaptive per-image | 3 | 4 | 12 | viable, strong prior (GlimpsePrune) |
| E. Video-VLM | 3 | 2 | 6 | low feasibility |

---

## 4. Recommendation

**Recommended gap: A — Real deployment-level speedup integrated with a serving engine (vLLM primary, lmdeploy secondary).**

**Why:**
1. **Genuinely open** — zero of 23 surveyed methods report served tokens/sec inside vllm/lmdeploy/TRT-LLM/SGLang. The only artifact (vLLM RFC #45098) is unfinished infrastructure. The Eval-Framework paper (2510.07143) explicitly demands this evaluation. This is the strongest novelty claim available.
2. **Highest feasibility on 1×A40** — training-free compressors ported to a serving engine require **no training**; the work is engineering + a deployment-aware eval harness. vLLM runs 7B models on a single A40. Days-to-first-result.
3. **Compelling SCI narrative** — "everyone reports FLOPs/tokens; we are the first to show that FLOPs-reduction does *not* translate to wall-clock under continuous batching (KV-cache, prefill scheduling), and we fix it." Pattern Recognition / Information Sciences / Neurocomputing favor measurement-gap + method papers.
4. **Lowest crowding risk** — B/C/D have 2025-2026 SOTA (G-Prune, Q-Zoom, GlimpsePrune) that a 3-month project is unlikely to beat on accuracy. Gap A's competition is *absence*, not a strong baseline.

**Recommended base model to START: LLaVA-1.5-7B.**
- Best-supported in vLLM, simplest multimodal processor to hook, smallest tokens-per-image (576) → fastest iteration. FastV/SparseVLM/VisionZip all have public LLaVA-1.5-7B checkpoints to reuse as the compression primitive, so P2 can be eval-only at first.
- **Pivot target after first result: Qwen2.5-VL-7B-Instruct.** It is the modern base with variable-length tokens + M-RoPE, supported by GlimpsePrune (HF ckpt) and Q-Zoom; vLLM has first-class Qwen2.5-VL support. The M-RoPE / variable-resolution handling is where the deployment-throughput story gets harder and more publishable (KV-cache behavior differs). Plan: LLaVA-1.5-7B for the method+eval skeleton, then Qwen2.5-VL-7B for the harder, more novel deployment result.

**Success criterion (first non-trivial result):** a training-free compressor (start by re-implementing FastV inside vLLM's processor) reporting a served-throughput curve (tokens/sec vs pruning ratio) on GQA + TextVQA on 1× A40, showing where FLOPs-reduction decouples from wall-clock (the "surprising" finding that anchors the paper).

---

## 5. Surprising / important findings

- **The throughput gap is real and untouched.** Of 23 methods, 6 report *any* wall-clock number; **0 report served throughput inside a production engine** (vllm/lmdeploy/TRT-LLM/SGLang). The Eval-Framework paper (2510.07143) calls this out explicitly. This is the cleanest novelty opening in the field.
- **FLOPs-reduction ≠ wall-clock speedup** is *asserted but never demonstrated under continuous batching*. vLLM's KV-cache + prefill scheduling + variable batching mean a 50% token cut may yield <30% latency cut — and no paper measures this. This is the paper's likely headline finding.
- **SparseVILA (ICCV'25)** is the *only* method with deployment-flavored wall-clock (4.0× prefill / 2.6× e2e) but on an AWQ custom pipeline, **not** a serving engine — confirming the gap is about *engine integration*, not about whether compression works.
- **Qwen2.5/3-VL support is thin.** Only Q-Zoom (2.5-VL primary, 3-VL claimed) and GlimpsePrune (2.5-VL HF ckpt) explicitly support it. M-RoPE + native variable resolution is rarely handled — a secondary novelty hook.
- **OCR/Chart/Doc degradation is the known weak spot** but already addressed by 5+ papers (G-Prune, GlimpsePrune, Q-Zoom, HiRED, VFlowOpt) — so it's crowded, not the best primary gap for a 3-month project.
- **vLLM RFC #45098** (`--image-pruning-rate`) exists as open infrastructure — a natural host / collaborator surface for the recommended method, lowering engineering risk.
- **Training-free dominates recent SOTA** (FastV, SparseVLM, VisionZip, VTC-CLS, FasterVLM, G-Prune, GlimpsePrune) — perfectly aligned with the 1×A40 serial constraint. No need to train.
- **Video-VLM compression (DyCoke/DynaTok) reports better wall-clock than image-VLM** (1.5× e2e), but video eval is GPU-heavy → lowest feasibility for single-card.

---

## 6. Bibliography (primary sources, arXiv-verified 2026-07-01)

1. Chen, L. et al. **An Image is Worth 1/2 Tokens After Layer 2: A Plug-and-play Approach for Fully-Attention-Specific VLMs (FastV).** ECCV 2024 (Oral). arXiv:2403.06764. https://arxiv.org/abs/2403.06764
2. Zhang, Q. et al. **FasterVLM: [CLS] Attention is All You Need for Training-Free Visual Token Pruning.** arXiv:2412.01818. https://arxiv.org/abs/2412.01818
3. Zhang, Y. et al. **SparseVLM: Visual Token Sparsification for Efficient Vision-Language Inference.** ICML 2025. arXiv:2410.04417. https://arxiv.org/abs/2410.04417
4. Yang, S. et al. **VisionZip: Longer is Better but Not Necessary in Vision Language Models.** arXiv:2412.04467. https://arxiv.org/abs/2412.04467
5. Wang, A. et al. **VTC-CLS: [CLS] Token Tells Everything Needed for Training-Free Efficient MLLMs.** arXiv:2412.05819. https://arxiv.org/abs/2412.05819
6. Li, W. et al. **TokenPacker: Efficient Visual Projector for Multimodal LLM.** IJCV 2025. arXiv:2407.02392. https://arxiv.org/abs/2407.02392
7. Shang, Y. et al. **LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models.** arXiv:2403.15388. https://arxiv.org/abs/2403.15388
8. Bolya, D. et al. **Token Merging: Your ViT But Faster (ToMe).** ICLR 2023. arXiv:2210.09461. https://arxiv.org/abs/2210.09461
9. AdaptMerge. **Language-Guided Token Merging for VLMs.** EMNLP 2025 Findings. https://aclanthology.org/2025.findings-emnlp.387.pdf
10. G-Prune. **What Kind of Visual Tokens Do We Need? Training-free Visual Token Pruning … from the Perspective of Graph.** arXiv:2501.02268. https://arxiv.org/abs/2501.02268
11. AdaptPrune. **Multi-Cue Training-Free Visual Token Pruning.** arXiv:2503.08019. https://arxiv.org/abs/2503.08019
12. **Q-Zoom: Query-Aware Adaptive Perception for Efficient Multimodal Large Language Models.** arXiv:2604.06912. https://arxiv.org/abs/2604.06912
13. **A Glimpse to Compress: Dynamic Visual Token Pruning for Large Vision-Language Models (GlimpsePrune).** arXiv:2508.01548. https://arxiv.org/abs/2508.01548
14. Khaki et al. **SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference.** ICCV 2025. arXiv:2510.17777. https://arxiv.org/abs/2510.17777
15. Tao et al. **DyCoke: Dynamic Compression of Tokens for Fast Video Large Language Models.** CVPR 2025. arXiv:2411.15024. https://arxiv.org/abs/2411.15024
16. **DynaTok: Dynamic Token Allocation for Video VLMs.** arXiv:2605.19322. https://arxiv.org/abs/2605.19322
17. **ShaRP: Shallow-Layer Pruning for Video LLMs.** arXiv:2512.05385. https://arxiv.org/abs/2512.05385
18. **FlashSloth: Efficient Multimodal LLM.** arXiv:2412.04317. https://arxiv.org/abs/2412.04317
19. **LLaVA-UHD v3: Native-Resolution Encoding.** arXiv:2511.21150. https://arxiv.org/abs/2511.21150
20. **LLaMA-VID.** arXiv:2311.17043. https://arxiv.org/abs/2311.17043
21. **HiRED: High-Resolution Token-Efficient VLM.** NSF PAR (2025).
22. **VFlowOpt: Flow-Optimized Fine-Grained Token Pruning.** ICCV 2025.
23. **Are We Using the Right Benchmark: An Evaluation Framework for Visual Token Compression Methods.** arXiv:2510.07143. https://arxiv.org/abs/2510.07143 — *most relevant prior work to Gap A.*
24. **vLLM RFC #45098 — Image Token Pruning flag.** https://github.com/vllm-project/vllm/issues/45098 — unfinished infrastructure, natural host for the recommended method.
25. **Efficient LVLM survey.** arXiv:2603.27960. https://arxiv.org/abs/2603.27960

---

*End of survey. Next: main window synthesizes `notes/positioning.md` from this file's §4–§5.*

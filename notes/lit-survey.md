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

This survey covers **37 methods** with verified primary sources (23 original + 14 folded in from the curated ammo list on 2026-07-01: PyramidDrop + 13 new 2025-2026 entries), builds a full comparison table, then evaluates 5 candidate research gaps against novelty × feasibility-on-1×A40, and recommends one. All arXiv IDs below were re-verified against arXiv abs pages on 2026-07-01 (see §8 for the verification log + corrections).

---

## 2. Comparison table

Legend — Training-free: **TF**=fully training-free, **LT**=light-tuning/adapter-only, **FT**=full/multi-stage training. Throughput: **Y**=reports real wall-clock latency/throughput (CUDA latency, prefill time, e2e speedup), **N**=FLOPs/token-count only, **DEPLOY**=measured under a serving engine (vllm/lmdeploy/TRT-LLM/SGLang). "Q2.5/3-VL" = explicit support for Qwen2.5-VL / Qwen3-VL (M-RoPE, native variable resolution).

| # | Method | Year/Venue | Train-free | Base model(s) | Benchmarks reported | Compression ratio | Accuracy (key metric + Δ vs full-token) | Real throughput? | Q2.5/3-VL? | Code | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **FastV** "Image worth 1/2 Tokens" | 2024 / ECCV'24 (Oral) | TF | LLaVA-1.5 (7B/13B), QwenVL-Chat, Video-LLaVA | GQA, MMBench, VQA, TextVQA, ScienceQA, POPE, MME, video-QA | ~50% visual tokens pruned; ~45% FLOPs (13B) | Pareto: 13B <7B-FLOPs, accuracy held | **N** (FLOPs only) | No | Yes (github) | Canonical baseline; prunes after LLM layer 2 by attention. Largest drops on text-dense tasks. **arXiv 2403.06764 VERIFIED.** |
| 2 | **FasterVLM / VisPruner** "[CLS] Attn" | 2024-12 / arXiv (v2 retitled VisPruner) | TF | LLaVA-1.5, LLaVA-NeXT | GQA, TextVQA, ScienceQA, MMBench, POPE, MME | up to ~95% of visual tokens; **91% FLOPs** | ~90% of original perf retained at high pruning | **Y** (partial) — claims **75% inference latency** (abstract) | No | Yes | ⚠ arXiv **2412.01818 VERIFIED but v2 title is now "Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning in VLMs"** (method renamed **VisPruner**); the "[CLS] Attention is All You Need / FasterVLM" title survives only on the project page. CLIP [CLS] attention re-ranking. Latency measured in authors' own harness, NOT inside vLLM/lmdeploy/TRT-LLM/SGLang. |
| 3 | **SparseVLM** "Visual Token Sparsification" | 2024-10 / ICML'25 | TF | LLaVA-1.5, LLaVA-NeXT, video variants | GQA, TextVQA, ScienceQA, MMBench, POPE, MME | ~54% FLOPs reduction | **97% of original accuracy** retained | **Y** — "37% CUDA latency" (measured, not serving-engine) | No | Yes | Text-guided self-attn sparsification + token recycling. Closest to deployment-level among classic TF methods. **arXiv 2410.04417 VERIFIED.** |
| 4 | **VisionZip** "Longer is Better but Not Necessary" | 2024-12 / **CVPR'25** | TF | LLaVA-NeXT (7B/13B), video-LLaVA | GQA, MMBench, TextVQA, ScienceQA, MMMU, MME, POPE | up to **8× fewer** visual tokens | >5% acc gain over prior SOTA at same compression | **Y** — **8× faster prefilling time** (real wall-clock prefill) | No | Yes | Encoder-side selection before LLM. NeXT-13B infers faster than 7B baseline. **arXiv 2412.04467 VERIFIED** (CVPR 2025, not just arXiv). |
| 5 | **PyramidDrop** "Pyramid Visual Redundancy Reduction" | 2024-10 / **CVPR'25** | TF | LLaVA-1.5, LLaVA-NeXT | GQA, TextVQA, ScienceQA, MMBench, POPE, MME | pyramid (increasing-drop) across LLM layers | accuracy held while accelerating training+inference | **N** (FLOPs/token-count, theoretical speedup) | No | Yes | NEW ROW. Layer-by-layer increasing visual-token discard; redundancy grows with depth. **arXiv 2410.17247 VERIFIED.** |
| 6 | **VTC-CLS** "[CLS] Token Tells Everything" | 2024-12 / arXiv | TF | LLaVA-1.5/NeXT, Qwen2-VL | GQA, TextVQA, ScienceQA, MMBench, POPE, MME, MMMU | high ratios (~89% per citing work) | SOTA among TF token-compression across tasks | **N** (FLOPs) | Qwen2-VL (not 2.5/3) | Yes | Ensembled [CLS] attn across ViT layers. |
| 7 | **TokenPacker** "Efficient Visual Projector" | 2024-07 / **IJCV'25** | FT (multi-stage) | LLaVA-1.5, high-res MLLMs | TextVQA, ChartQA, DocVQA, ScienceQA, GQA, MMBench, MMMU | **75–89% reduction** (576→64/128) | comparable/better than MLP projector at 4–9× fewer tokens | **N** (FLOPs) | No | Yes | Coarse-to-fine projector; needs coarse-pretrain → refine → instruction-tune. |
| 8 | **LLaVA-PruMerge** | 2024-03 / arXiv (v6) | LT (hybrid) | LLaVA-1.5 | VQA, GQA, TextVQA, ScienceQA, POPE, MME | **~14× compression** (576→~40) | comparable perf across VQA/reasoning | **N** (FLOPs) | No | Yes | Pruning training-free (CLIP CLS sparsity) + optional light merging adapter. **arXiv 2403.15388 VERIFIED** (no top-tier venue on the page; user-list "ICCV 2025" not confirmed on abs page). |
| 9 | **ToMe** (original ViT) | 2023 / ICLR'23 | TF | ViT-L/H (image, video, audio) | ImageNet-1k, Kinetics-400, AudioSet | up to ~2× token reduction (r=16/24) | 0.2–0.3% drop image; 0.4% mAP audio | **Y** — real ~2× wall-clock on ViT | n/a (ViT) | Yes | Bolya et al. Not a VLM but the merging origin. Bipartite soft matching. |
| 10 | **AdaptMerge** (ToMe→LLaVA) | 2025 / EMNLP'25 Findings | TF | LLaVA-family | standard VQA suite | adaptive, language-guided | closes most of OCR gap vs plain ToMe | **N** | No | Yes | Nearest "ToMe for LLaVA"; query-conditioned merging. |
| 11 | **G-Prune** "Graph perspective" | 2025-01 / arXiv | TF | LLaVA-NeXT | incl. TextVQA | 63.57% FLOPs reduced | **TextVQA only −2.34%** (explicitly studies fine-grained/OCR) | **N** (FLOPs) | No | Yes | Graph-based token importance; foreground+background both retained. |
| 12 | **AdaptPrune** (Multi-Cue) | 2025-03 / arXiv | TF | cross-LVLM | standard | moderate–high | competitive at high ratios | **N** (FLOPs) | No | Yes | Attention + spatial + similarity (NMS) cues. |
| 13 | **Q-Zoom** (query-aware) | 2026 / arXiv | LT (gating+SD-RPN self-distill) | **Qwen2.5-VL-7B (primary)**; also Qwen3-VL, LLaVA, RL image-thinking | Document & OCR, High-Resolution | inference-time | **2.52× (Doc/OCR), 4.39× (HR)** speedup, no acc loss | **Y** — reported as inference time | **Yes (2.5-VL primary; 3-VL)** | TBD | Strongest explicit Qwen2.5-VL + query-aware result. |
| 14 | **GlimpsePrune** (content-adaptive) | 2025-08 / arXiv | TF | LLaVA-NeXT, **Qwen2.5-VL-7B-Instruct** | free-form VQA, OCR-heavy | prunes **92.6%** visual tokens | retains baseline; GlimpsePrune+ hits **110%** | **N** (token/FLOPs) | **Yes (2.5-VL)** | Yes (HF ckpt) | Single "glimpse" forward → dynamic per-image budget. |
| 15 | **SparseVILA** (decode query-aware) | 2025 / ICCV'25 | TF | architecture-agnostic (AWQ pipeline) | long-video, doc, reasoning | prefill prune + decode retrieval | accuracy *gains* on doc/reasoning | **Y — 4.0× prefill, 2.5× decode, 2.6× e2e** | partial (Qwen2-VL family via AWQ) | Yes | **Closest to a deployed-engine wall-clock result, but on AWQ pipeline — NOT vllm/lmdeploy.** |
| 16 | **DyCoke** (video) | 2024-11 / CVPR'25 | TF | LLaVA-NeXT-Video, Video-LLaVA | video-QA suite | temporal merging + dynamic KV-cache spatial prune | improves acc vs FastV | **Y — 1.5× inference, 1.4× memory** (research code, not serving engine) | No | Yes | Best-known training-free video baseline. |
| 17 | **DynaTok** (video) | 2026 / arXiv | TF | LLaVA-OneVision, LLaVA-Video | MVBench, LongVideoBench, MLVU, VideoMME | **90% token reduction** | >95% accuracy retained | **N** (token-count) | No | TBD | Temporal+spatial budget allocation (EMA memory). |
| 18 | **ShaRP** (shallow-layer video) | 2025 / arXiv | TF | video LLaVA-family | video-QA | shallow-layer prune | addresses FastV deep-layer degradation | **N** | No | TBD | Complements DyCoke. |
| 19 | **FlashSloth** | 2024-12 / arXiv | LT/FT | builds on Qwen2-VL | efficient high-res MLLM | token reduction | competitive | **N** | Qwen2-VL | Yes | Efficient MLLM line, less cited as a compression method per se. |
| 20 | **LLaVA-UHD v3** | 2025-11 / arXiv | LT (native-res encoder) | LLaVA + native-res | standard + TTFT | native variable resolution | TTFT **1.9× lower** vs Qwen2-VL | **Y** — TTFT reported | compares to Qwen2-VL | Yes | Encoder-resolution, adjacent to compression. |
| 21 | **LLaMA-VID** | 2023-11 / arXiv | FT | LLaMA-based | video QA | 1 token/image-side per frame (context) | trade-off accepted | **N** | No | Yes | Foundational "few tokens" idea (single-context-token). |
| 22 | **HiRED** | NSF PAR | TF | LLaVA-1.5 | fine-grained transcription | **40% budget** (1152 tokens) | preserves fine-grained acc | **N** | No | TBD | Targets the OCR-degradation problem directly. |
| 23 | **VFlowOpt** | 2025 / ICCV'25 | LT/FT | LLaVA-1.5 | fine-grained | optimizes pruning to preserve fine-grained info | better OCR retention | **N** | No | TBD | Flow-based optimization for fine-grained preservation. |
| 24 | **Eval-Framework** "Are We Using the Right Benchmark?" | 2025-10 / arXiv | (meta-eval) | surveys many | proposes dedicated VTC eval | — | argues current benchmarks miss the real cost | **partial** — critiques missing throughput eval | n/a | TBD | **Most directly relevant prior work to the deployment-throughput gap.** arXiv 2510.07143. |
| 25 | **PRUNESID** "Synergistic Importance-Diversity" | 2026-03 / **ICLR'26** | TF | LLaVA-1.5 (primary) | GQA, MMBench, TextVQA, ScienceQA, POPE, MME | 11.1% tokens retained | **96.3% acc** on LLaVA-1.5 at 11.1% tokens | **Y** (partial) — claims **7.8× faster prefilling** (abstract) | No | Yes | NEW ROW. Two-stage PSCA clustering + intra-group NMS, info-aware dynamic ratio. **arXiv 2603.09480 VERIFIED.** Prefill-speed is a wall-clock-style number but NOT named as inside a serving engine. |
| 26 | **E-AdaPrune** "Energy-Driven Adaptive Pruning" | 2026-03 / arXiv | TF | VLMs (LLaVA-family) | standard VQA suite | adaptive per-image budget from SV-spectrum energy | **+0.6% avg (+5.1% MMVet)** at matched budgets | **Y** (partial) — pruning overhead **~8 ms/image** (randomized-SVD, abstract) | No | Yes | NEW ROW. ⚠ This is the **only method that reports the cost of the pruning step itself as a real latency** — a useful honesty precedent, but still NOT end-to-end served throughput. **arXiv 2603.05950 VERIFIED.** |
| 27 | **AgilePruner** "Empirical Study of Attention & Diversity" | 2026-03 / **ICLR'26** | TF | LLaVA-family | standard VQA + hallucination | adaptive hybrid (erank + attn-entropy) | competitive; ties diversity-preservation to hallucination | **N** (analysis paper; no serving throughput) | No | Yes | NEW ROW. ⭐ **User-flagged direct competitor to a "缝合/systematic study" angle.** Systematic empirical study of attention- vs diversity-based pruning; the accuracy/FLOPs combination-study space is now crowded by this. **arXiv 2603.01236 VERIFIED.** |
| 28 | **VisionTrim** "Unified Training-Free Compression" | 2026-01 / **ICLR'26** | TF | MLLMs (LLaVA-family) | standard VQA suite | unified TF framework | competitive | **N** (FLOPs/token-count, theoretical prefill) | No | Yes | NEW ROW. Two modules: Dominant Vision Token Selection (global-local) + Text-Guided Vision Complement (context-aware merging). **arXiv 2601.22674 VERIFIED.** |
| 29 | **FocusUI** "Position-Preserving Token Selection" | 2026-01 / arXiv (CVPR'26 ext.) | **FT** (UI-domain) | UI-grounding VLMs (FocusUI-7B) | UI grounding (ScreenSpot, etc.) | 30% token retention | FocusUI-7B only −3.2% at 30% retention | **Y** (partial) — **1.44× faster inference + 17% lower peak GPU mem** (authors' harness) | No | Yes | NEW ROW. Domain-specific (UI/GUI); PosPad marker preserves positional continuity. **arXiv 2601.03928 VERIFIED.** Wall-clock measured, NOT inside a serving engine. |
| 30 | **HybridToken-VLM** "Hybrid Token Compression" | 2025-12 / arXiv (CVPR'26 ext.) | **FT** | VLMs (7B) | 7 benchmarks | **580→1 token** (extreme) | **87.2% avg retention** vs 81.0% continuous baseline | **N** (FLOPs/quadratic-cost arguments only) | No | Yes | NEW ROW. Discrete MGVQ semantic anchors (4 tok) + continuous ViT features → single `<voco>` token via disentanglement mask. **arXiv 2512.08240 VERIFIED.** |
| 31 | **PPE** "Positional Preservation Embedding" | 2025-10 / arXiv (ICLR'26 ext.) | **TF** (parameter-free) | MLLMs (image+video) | MMBench, TextVQA, VideoMME | merging w/ position preservation | **+2–5%** over SOTA merging baselines | **N** (accuracy only; efficiency implied) | No | Yes | NEW ROW. Parameter-free operator injecting disentangled 3D-position encodings into merged tokens; cascade clustering. **arXiv 2510.22936 VERIFIED.** |
| 32 | **Fourier-VLM** "Frequency-Domain Compression" | 2025-08 / arXiv | **TF** (parameter-free) | LLaVA-v1.5 | standard VQA suite | up to **83.8% FLOPs** reduction | competitive | **Y** (partial) — **31.2% faster generation** (authors' harness) | No | Yes | NEW ROW. 2D-DCT low-pass filter (FFT, O(n log n)) on vision features. v1 title was "Fourier Compressor". **arXiv 2508.06038 VERIFIED.** Wall-clock measured, NOT inside a serving engine. |
| 33 | **METEOR** "Multi-Encoder Collaborative Pruning" | 2025-07 / **ICCV'25** | TF | multi-encoder MLLMs (EAGLE-based) | 11 benchmarks | **76% fewer visual tokens** vs EAGLE | **−0.3% avg** only | **N** (token-count + FLOPs) | No | Yes | NEW ROW. Progressive 3-stage (encode/fuse/decode): rank-guided intra-encoder assignment + cross-encoder cooperative + prompt-adaptive LLM-decoding pruning. **arXiv 2507.20842 VERIFIED.** |
| 34 | **AdaTP** "Attention-Debiased Token Pruning" (video) | 2025-05 / arXiv (EMNLP'25 Find. ext.) | TF | LLaVA-OneVision-7B | video-QA suite | **≤27.3% FLOPs** | matches vanilla performance | **N** (FLOPs ratio only) | No | Yes | NEW ROW. Corrects two attention biases (global sequence-end; local cross-frame spatial) before pruning. **arXiv 2505.20100 VERIFIED.** |
| 35 | **AdaReTaKe** "Adaptive Redundancy Reduction" (video) | 2025-03 / arXiv (ACL'25 Find. ext.) | TF | LLaVA-OneVision (7B/72B) | LVBench, video-QA | non-uniform spatiotemporal ratio | **+2.3%/2.8%** (7B/72B); **+5.9%/6.0%** on LVBench | **N** (frame-capacity + acc; no wall-clock) | No | Yes | NEW ROW. Temporal-adaptive + layer-adaptive compression-ratio allocation with theoretical guarantees; expands video capacity 256→2048 frames. **arXiv 2503.12559 VERIFIED.** |
| 36 | **PLPHP** "Per-Layer Per-Head Pruning" | 2025-02 / arXiv | TF | LLaVA-family | standard VQA suite | per-layer/per-head fine-grained | **−0.46% avg** | **Y** (partial) — **18% faster decoding + >50% KV-cache** (authors' harness) | No | Yes | NEW ROW. Layer-level retention allocation (Vision Token Re-attention) + head-level pruning within each layer. **arXiv 2502.14504 VERIFIED.** Wall-clock measured, NOT inside a serving engine. |
| 37 | **RedundancyLens** "Redundancy for Decoder-Only MLLMs" | 2025-01 / **ACL'25 Findings** | TF | decoder-only MLLMs | standard VQA suite | FFN+attn computation savings | competitive | **N** (FLOPs/computation savings) | No | Yes | NEW ROW. Diagnostic + acceleration: Probe-Activated Dynamic FFN + Hollow Attention + Layer Ranking; finds structured/clustered redundancy unique to decoder-only archs. **arXiv 2501.19036 VERIFIED.** |

### 2.1 Throughput-reporting summary (the suspected gap)

Of **37 methods**, **13 report *some* real wall-clock-style number** — SparseVLM, VisionZip, SparseVILA, DyCoke, Q-Zoom, LLaVA-UHD, original ToMe (ViT), and the 6 newly folded in: FasterVLM/VisPruner (75% latency), PRUNESID (7.8× prefill), E-AdaPrune (8 ms/img pruning overhead), FocusUI (1.44×), Fourier-VLM (31.2% faster gen), PLPHP (18% faster decoding). Of those:
- **SparseVILA** remains the most deployment-flavored (AWQ pipeline, 4.0× prefill / 2.6× e2e) but is **NOT** on vllm/lmdeploy/TRT-LLM/SGLang.
- **VisionZip / SparseVLM / DyCoke / FasterVLM / PRUNESID / Fourier-VLM / FocusUI / PLPHP** all measure raw CUDA latency, prefill time, or decode speed on **research code / the authors' own harness** — not a serving engine.
- **E-AdaPrune** is notable for honesty: it reports the *cost of the pruning step itself* (8 ms/img SVD) — a useful precedent, but still not end-to-end served throughput.
- **Q-Zoom / LLaVA-UHD** report inference-time / TTFT, again offline.
- **NONE of the 37 methods measure served throughput (tokens/sec or req/sec) INSIDE a production serving engine** (vllm/lmdeploy/TRT-LLM/SGLang).

**The ICLR'26 combination-study cluster (AgilePruner #27, VisionTrim #28, PRUNESID #25) — which the user flagged as the direct "缝合/systematic-study" competitor — confirms this:** all three report **accuracy + FLOPs/token-count on offline research code only**; **none** integrates inside a serving engine or reports served throughput. The accuracy/FLOPs combination-study space is now crowded, but the **serving-engine-throughput space is still empty.**

The only serving-engine artifact found is **vLLM RFC #45098** (`--image-pruning-rate` flag), which is **unfinished infrastructure**, not a published method with benchmarks. ⇒ **The deployment-throughput gap (Gap A) is genuinely open even after folding in 16 new 2025-2026 methods.**

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

- **The throughput gap is real and untouched, even after the 2026-07-01 merge.** Of **37 methods** now surveyed, **13 report some wall-clock-style number** (up from 6), but **0 report served throughput inside a production engine** (vllm/lmdeploy/TRT-LLM/SGLang). The newly folded-in ICLR'26 cluster — **AgilePruner, VisionTrim, PRUNESID** — all crowd the accuracy/FLOPs-combination space and report **offline FLOPs/token-count only**; E-AdaPrune's 8 ms/img is the pruning-step cost, not served throughput. The Eval-Framework paper (2510.07143) and the Westlake survey (2507.20198 §6.5) call this out explicitly. **This is the cleanest novelty opening in the field and it holds after the merge.**
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
26. **PyramidDrop: Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction.** CVPR 2025. arXiv:2410.17247. https://arxiv.org/abs/2410.17247
27. **Prune Redundancy, Preserve Essence: Vision Token Compression in VLMs via Synergistic Importance-Diversity (PRUNESID).** ICLR 2026. arXiv:2603.09480. https://arxiv.org/abs/2603.09480
28. **Energy-Driven Adaptive Visual Token Pruning for Efficient Vision-Language Models (E-AdaPrune).** arXiv:2603.05950. https://arxiv.org/abs/2603.05950
29. **AgilePruner: An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in Large Vision-Language Models.** ICLR 2026. arXiv:2603.01236. https://arxiv.org/abs/2603.01236
30. **VisionTrim: Unified Vision Token Compression for Training-Free MLLM Acceleration.** ICLR 2026. arXiv:2601.22674. https://arxiv.org/abs/2601.22674
31. **FocusUI: Efficient UI Grounding via Position-Preserving Visual Token Selection.** arXiv:2601.03928 (CVPR 2026 externally). https://arxiv.org/abs/2601.03928
32. **HybridToken-VLM: Hybrid Token Compression for Vision-Language Models.** arXiv:2512.08240 (CVPR 2026 externally). https://arxiv.org/abs/2512.08240
33. **Positional Preservation Embedding for Multimodal Large Language Models (PPE).** arXiv:2510.22936 (ICLR 2026 externally). https://arxiv.org/abs/2510.22936
34. **Fourier-VLM: Compressing Vision Tokens in the Frequency Domain for Large Vision-Language Models.** arXiv:2508.06038 (v1 "Fourier Compressor"). https://arxiv.org/abs/2508.06038
35. **METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models.** ICCV 2025. arXiv:2507.20842. https://arxiv.org/abs/2507.20842
36. **AdaTP: Attention-Debiased Token Pruning for Video Large Language Models.** arXiv:2505.20100 (EMNLP 2025 Findings externally). https://arxiv.org/abs/2505.20100
37. **AdaReTaKe: Adaptive Redundancy Reduction to Perceive Longer for Video-language Understanding.** arXiv:2503.12559 (ACL 2025 Findings externally). https://arxiv.org/abs/2503.12559
38. **PLPHP: Per-Layer Per-Head Vision Token Pruning for Efficient Large Vision-Language Models.** arXiv:2502.14504. https://arxiv.org/abs/2502.14504
39. **RedundancyLens: Revealing and Exploiting Visual Token Processing Redundancy for Efficient Decoder-Only MLLMs.** ACL 2025 Findings. arXiv:2501.19036. https://arxiv.org/abs/2501.19036

---

*End of survey. Next: main window synthesizes `notes/positioning.md` from this file's §4–§5.*

---

## 7. Novelty re-check (2026-07-01, P2-step-1)

**Task:** re-scan 2026-H1 for ANY paper that integrates a visual-token compressor INSIDE vLLM / SGLang / lmdeploy / TRT-LLM AND reports served throughput (tok/s, req/s, TTFT). Sources: WebSearch + arXiv. Verdict below.

### Closest hits found (and why none close Gap A)

1. **ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism** — arXiv **2507.10069** (Liu, Cheng, Tan, You, Tao; NUS + ICT-CAS; v1 Jul 2025, v2 Aug 2025). Built on vLLM v0.6.6; reports TTFT↓4.2×, SLO-throughput 3.2–4.5×. **BUT it is NOT a visual-token compressor** — it does modality-aware load balancing, elastic stage parallelism, unified multimodal prefix caching (memoizes *identical* images, token count unchanged), and async encoding. The paper *explicitly disclaims* compression: "These methods operate at the model level, trading off accuracy… Therefore, we do not compare against these optimization methods." **Verdict: adjacent (scheduling/disaggregation), not a competitor. Stackable with — not subsuming — our compression.**

2. **A Survey of Token Compression for Efficient Multimodal LLMs** — arXiv **2507.20198** (Shao, Tao et al.; Westlake/ZJU/NUS; v5 Feb 2026). ~90–100 methods surveyed. **§6.5.3 "Deployment Hurdles" explicitly states** that attention-score pruning "cannot be seamlessly integrated into current optimization frameworks" because FlashAttention fuses matmul+softmax, making scores inaccessible — "this creates a critical gap, as these compression methods cannot leverage the performance gains of state-of-the-art deployment pipelines." **§6.5.4 "Evaluation Challenges"** names **TTFT** and per-token decode latency as "crucial for accurately assessing real-world inference acceleration" but notes they are **missing/unreported**. **No method in the survey** integrates compression inside vLLM/SGLang/lmdeploy/TRT-LLM with served throughput; vLLM/RFC #45098 not mentioned anywhere. **Verdict: this survey CONFIRMS Gap A is open — it is the strongest independent corroboration that the deployment-engine-throughput gap is real and unfilled.**

3. **vLLM RFC #45098** (`--image-pruning-rate`, https://github.com/vllm-project/vllm/issues/45098): still an **RFC/infrastructure** (opt-in flag, "pruned before LLM fusion using a configurable method"), **no published benchmarks, no method paper** as of 2026-07-01. Natural host for our method but not a competitor.

4. Other 2026 arXiv hits checked (DyToK 2512.06866, VisionTrim ICLR'26 2601.22674, Nüwa 2602.02951, Vision Token Reduction 2602.12618, HoloCV NeurIPS'25, Unified Spatiotemporal CVPR'26): **all report FLOPs/token-count only**, none integrate inside a serving engine. SparseVILA (ICCV'25) remains the only deployment-flavored wall-clock result but on its own AWQ pipeline, not vLLM/SGLang/lmdeploy/TRT-LLM.

### Additional 2026-H1 papers folded in (coordinator update 2026-07-01)

5. **EffiVLM-BENCH: Unified Benchmark for Efficient VLMs** — arXiv **2506.00479**. Unifies eval of training-free token-pruning (FastV/VisionZip/PruMerge+) + KV-cache compression on 17 benchmarks; reports **offline** TTFT + decode-time speedup, batch=1, on **lmms-eval + HuggingFace transformers** (single A800, FlashAttention-2). **Does NOT use vLLM/SGLang/lmdeploy/TRT-LLM and does NOT report served throughput (tok/s, req/s under batching) or KV-cache MB.** vLLM RFC #45098 not mentioned. **Verdict: OFFLINE eval harness → leaves our gap open.** Useful as a *positioning anchor* (proves the field still measures offline latency) and as a candidate eval-harness backbone for our accuracy tables (its OP/OG/OL/OE indices are reusable). **Confirmed: served-throughput-inside-engine gap is still unfilled.**

6. **Combination-study competitors (ICLR 2026 cluster) — crowd the *accuracy/FLOPs systematic-study* space, NOT our gap:**
   - **AgilePruner** (arXiv 2603.01236) — user flags it as "exactly the systematic-research paradigm you want to do": systematic empirical combination of scoring-basis × reduction-method. ⇒ A PURE accuracy/FLOPs combination study is no longer novel.
   - **VisionTrim** (arXiv 2601.22674) — unified training-free framework.
   - **PRUNESID** (arXiv 2603.09480) — importance + diversity co-scoring.
   All three report accuracy + FLOPs/token-count on offline research code; **none integrates inside a serving engine or reports served throughput.** They reinforce that our differentiator MUST be **serving-engine real throughput**, not another combination study on accuracy/FLOPs.

### Verdict: **Gap A is still OPEN.**
No competitor landed. The most-recent relevant works all *strengthen* our novelty: ElasticMM is non-overlapping (scheduling, explicitly avoids compression); the 2507.20198 survey independently documents the exact gap (deployment-engine integration + served TTFT/throughput) and diagnoses the FlashAttention-score root cause; EffiVLM-BENCH confirms the field still measures offline latency; AgilePruner/VisionTrim/PRUNESID crowd the accuracy/FLOPs-combination space but leave serving-throughput untouched. **Our served-throughput-inside-engine angle is uncontested. No blocker. Proceed to go/no-go probe.**

---

## 8. arXiv-ID verification log (2026-07-01, Lit subagent merge)

**Task:** verify every arXiv ID in the curated ammo list (`recent_papers/VLM视觉Token压缩_论文汇总.md`) against arXiv abs pages. The user explicitly flagged that "some classic-paper arXiv IDs are from memory — verify before citing" (FastV, LLaVA-PruMerge, SparseVLM, PyramidDrop, FasterVLM, VisionZip). Method: WebSearch on arxiv.org + webReader fetch of each abs page.

### Result: **all 19 IDs RESOLVE. No fabrication. One title/alias correction.**

| Paper | arXiv ID | Status | Note |
|---|---|---|---|
| FastV | 2403.06764 | ✅ resolves | title exact ("An Image is Worth 1/2 Tokens After Layer 2…"); ECCV'24 Oral. |
| LLaVA-PruMerge | 2403.15388 | ✅ resolves | title exact. ⚠ user list tags "ICCV 2025" — **NOT confirmed on the abs page** (arXiv only as of 2026-07-01); left as "arXiv (v6)" in §2/§6. |
| SparseVLM | 2410.04417 | ✅ resolves | title exact; ICML 2025. |
| PyramidDrop | 2410.17247 | ✅ resolves | title exact; **CVPR 2025** (was arXiv-only in old survey — venue upgraded). |
| **FasterVLM** | **2412.01818** | ✅ resolves **but TITLE CHANGED** | **before → after:** v1 title "FasterVLM: [CLS] Attention is All You Need…" → **v2 retitled "Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning in VLMs"**; method renamed **VisPruner**. The "FasterVLM" name now survives only on the project page. §2 row #2 and §6 entry #2 updated to reflect both names. **The ID itself is correct — only the title/alias drifts.** |
| VisionZip | 2412.04467 | ✅ resolves | title exact; **CVPR 2025** (was arXiv-only — venue upgraded). |
| PRUNESID | 2603.09480 | ✅ resolves | title exact ("Prune Redundancy, Preserve Essence…"); ICLR 2026. |
| E-AdaPrune | 2603.05950 | ✅ resolves | title exact ("Energy-Driven Adaptive Visual Token Pruning…"). |
| AgilePruner | 2603.01236 | ✅ resolves | title exact; ICLR 2026. |
| VisionTrim | 2601.22674 | ✅ resolves | title exact; ICLR 2026. |
| FocusUI | 2601.03928 | ✅ resolves | title exact; CVPR 2026 (external). |
| HybridToken-VLM | 2512.08240 | ✅ resolves | title exact; CVPR 2026 (external). |
| PPE | 2510.22936 | ✅ resolves | title exact. |
| Fourier-VLM | 2508.06038 | ✅ resolves | v2 title "Fourier-VLM: Compressing Vision Tokens in the Frequency Domain…" (v1 was "Fourier Compressor"). |
| METEOR | 2507.20842 | ✅ resolves | title exact; ICCV 2025. |
| AdaTP | 2505.20100 | ✅ resolves | title exact; EMNLP 2025 Findings (external). |
| AdaReTaKe | 2503.12559 | ✅ resolves | title exact; ACL 2025 Findings (external). |
| PLPHP | 2502.14504 | ✅ resolves | title exact. |
| RedundancyLens | 2501.19036 | ✅ resolves | title exact; ACL 2025 Findings. |

### Corrections applied to the survey
1. **FasterVLM → "FasterVLM / VisPruner"** (§2 row #2, §6 entry #2): the arXiv ID 2412.01818 is correct, but the live v2 title is now "Beyond Text-Visual Attention…" and the method is renamed **VisPruner**. Both names retained for discoverability; citation should use the current title.
2. **PyramidDrop & VisionZip venue upgrades** (§2 rows #4/#5, §6): both confirmed **CVPR 2025** (were "arXiv" in the prior survey version).
3. **LLaVA-PruMerge venue caveat**: user list tags it "ICCV 2025" but the abs page shows no venue as of 2026-07-01 → kept as "arXiv (v6)" to avoid an unverified venue claim.

### Papers marked `[UNVERIFIED]`: **none.** All 19 IDs resolve to the titled papers; no fabrication occurred.

### Throughput-honesty re-audit (the novelty backbone)
Of the 14 newly folded-in methods (PyramidDrop + 13), **6 report a real wall-clock-style number** (FasterVLM/VisPruner 75% latency, PRUNESID 7.8× prefill, E-AdaPrune 8 ms/img, FocusUI 1.44×, Fourier-VLM 31.2%, PLPHP 18%). **Crucially, every one of these is measured in the authors' own harness / research code — NONE is measured inside vLLM / lmdeploy / TRT-LLM / SGLang.** Combined with the prior 7/23 (SparseVLM, VisionZip, SparseVILA, DyCoke, Q-Zoom, LLaVA-UHD, ToMe), the post-merge tally is **13/37 report some wall-clock, 0/37 report served throughput inside a production engine.** **The "0 methods report served throughput in an engine" narrative HOLDS after the merge.**

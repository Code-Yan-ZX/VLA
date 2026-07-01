# Positioning — Serving-Aware Visual Token Compression for VLMs

> P1 output, Main-synthesized from `notes/lit-survey.md`. Decisions mirrored in `DECISIONS.md`.
> Date: 2026-07-01 · Base(s): LLaVA-1.5-7B (start) → Qwen2.5-VL-7B-Instruct (generalize)

## Positioning (one line)
The first **serving-engine-aware** visual-token compression method for VLMs: a compressor co-designed with continuous-batching / KV-cache realities, evaluated by **served throughput** (tok/s, req/s, TTFT, KV-cache) inside vLLM — not FLOPs.

## Novelty sentence
Of 23 surveyed methods, all report FLOPs/token reduction; **0/23 measure served throughput inside a production serving engine** (vLLM/lmdeploy/TRT-LLM/SGLang). We close that gap with a method that yields wall-clock speedup **where naive compressors (FastV/SparseVLM) do not**, and explain why FLOPs-cut ≠ wall-clock under continuous batching.

## The gap (Gap A) — why open
- Survey 2026-07-01 (23 methods, arXiv-verified): 6/23 report any wall-clock; **0/23 inside a serving engine**. Closest = SparseVILA (ICCV'25, 4× prefill) but on AWQ custom pipeline, not vLLM.
- vLLM RFC #45098 (`--image-pruning-rate`) = unfinished infra, not a method; Eval-Framework (arXiv 2510.07143) explicitly demands this eval.
- B/C/D crowded by 2025-26 SOTA (G-Prune, Q-Zoom, GlimpsePrune) — unbeatable on accuracy in 3 months; A's competition is *absence*, not a strong baseline.

## Method framing — method, not just measurement
**Hypothesis:** existing compressors (FastV prunes intra-LLM at layer k; SparseVLM similar) cut FLOPs but under vLLM continuous batching the wall-clock gain is sub-linear because (i) prefill is parallelized/saturated, (ii) decode is text-token-bandwidth-bound, (iii) KV-cache scheduling dominates. **Serving-aware compressor** = prune at the multimodal-processor boundary (VisionZip/FasterVLM-style) + aware of batch composition / KV-cache budget → recover near-linear speedup. The go/no-go probe below tests this premise *before* we build the method.

## Base choice
- **Start: LLaVA-1.5-7B** — best vLLM support, 576 tok/img, reusable FastV/SparseVLM/VisionZip checkpoints → days-to-first-result, eval-only at first.
- **Generalize (publishable stretch): Qwen2.5-VL-7B-Instruct** — M-RoPE + variable-length tokens = harder, more novel deployment story; vLLM first-class support; GlimpsePrune HF ckpt exists.

## ★ Go / No-Go gate — P2 first milestone (the kill-switch)
Minimal probe on 1× A40, LLaVA-1.5-7B in vLLM: a **boundary-level training-free** compressor at ratios {0,25,50,75}%, measure served tok/s, req/s, TTFT, KV-cache on GQA + TextVQA.
- **GO** if 50% token cut → ≥1.5× prefill speedup AND ≥1.2× e2e served req/s at ≤2% GQA drop → proceed to serving-aware method design.
- **NO-GO** if wall-clock <1.2× e2e even at high pruning → **pivot**: reframe as negative-result characterization paper ("FLOPs≠wall-clock, why") OR fall back to Gap D (content-adaptive) as primary method, throughput as a secondary section.
- This is the project's biggest risk (claim-overturn). Run BEFORE heavy method investment. Per charter §6, a NO-GO = escalate to user.

## Novelty-confirm gate (cheap, run first)
Re-scan last 6 months for ANY vLLM/SGLang/lmdeploy/TRT-LLM integrated compression+throughput paper. Survey is current to 2026-07-01; if a competitor landed, downgrade novelty and reconsider gap.

## Success criteria
- **Milestone:** go/no-go decision (above).
- **Paper-level:** (1) served-throughput curves across ≥3 pruners/bases showing FLOPs↔wall-clock decoupling; (2) our serving-aware method beats best training-free baseline on served-throughput at iso-accuracy; (3) accuracy table GQA/MME/MMBench/TextVQA; (4) Qwen2.5-VL generalization.

## Risks & mitigations
1. **vLLM + transformers 5.x + vtc compat** (vLLM may pin transformers<5) → Dev resolves (possible dedicated serving env).
2. **Negative wall-clock result** → go/no-go gate + pivot options.
3. **Intra-LLM pruners (FastV) hard to hook in vLLM** → use boundary-level pruners for integration; reproduce FastV on `fastv` env as accuracy anchor only.
4. **Novelty erosion** (fast-moving field) → novelty-confirm gate.

## Target venues (working)
Pattern Recognition / Information Sciences / Neurocomputing (measurement-gap + method). Stretch: CVPR/ECCV.

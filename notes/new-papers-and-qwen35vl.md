# New-Papers + Qwen3.5-VL Assessment (2026-07-02)

> Subagent deliverable. Sources: 11-paper mechanism deep-read (Task 1) + 2026 H1/H2 fresh scan (Task 2) + Qwen3-VL feasibility (Task 3). All arXiv IDs verified to resolve (HTTP 200) except where flagged.

---

## TASK 1 — Mechanism deep-read of the 10 + SparseVLM

Decomposition = **打分依据 (scoring signal) → 缩减方式 (reduction)**. Boundary = pre-LLM = vLLM-integrable for us.

| # | Paper (arXiv) | Scoring signal → Reduction | OCR handling | Boundary / Intra-LLM | Train-free? | One-line takeaway |
|---|---|---|---|---|---|---|
| 1 | **PRUNESID / PruneSID** (2603.09480, ICLR26) | PSCA 主成分聚类 (覆盖/diversity) + 组内 NMS (重要性) + 信息感知动态压缩比 → 剪枝 | **无文本对齐**；纯视觉多样性最大化，极端压缩下文字 patch 高危 | Boundary (ViT 特征) | TF | Prefill 7.8×；OCR-unsafe |
| 2 | **E-AdaPrune** (2603.05950, abs verified HTTP200, intro not directly read — UNVERIFIED-detail) | 能量 (随机 SVD 谱能量) → 自适应预算 → 剪枝 | 未提文本对齐（能量=信息密度，非语义） | Boundary (ViT 特征) | TF | 唯一 spectral/energy boundary；可能保高频文本但不保证 |
| 3 | **AgilePruner** ⭐ (2603.01236, ICLR26) | **方法论**: erank (多样性) vs 注意力熵对比。洞察: diversity 实际保留多样性低于宣称且残留多样性→更多幻觉；attention 适合简单/证据集中图，diversity 适合复杂/分散图。→ image-aware 混合剪枝 | 未显式 OCR；但 CHAIR 幻觉分析暗示 attention 保文字焦点区 | Boundary (混合) | TF | **系统研究范式参考**：按图像自适应切换 attention/diversity = OCR-safe 设计起点 |
| 4 | **VisionTrim** (2601.22674, ICLR26) | DVTS (global-local 显著性→选主导) + **TGVC (text-guided merging)** | TGVC 用文本线索引导合并（最接近 text-aware），但属 merging 非 region 保护 | Boundary (两阶段) | TF (plug-and-play) | 唯一显式 text-guided boundary pipeline，OCR 友好但无 OCR benchmark 声明 |
| 5 | **HybridToken-VLM** (2512.08240) | 连续 ViT patch + 离散 MGVQ 量化 4 锚点 → bottleneck → 压成 1 voco token | 纹理走连续通道理论可保字形，但 580:1 压缩，未报 TextVQA | 混合 (改架构, pre-LLM 但非 plug-in) | **Trained** (码本+bottleneck) | 极致压缩靠学习；不适合 serving 直接套 |
| 6 | **PPE** (2510.22936) ★ | **非打分**，token merging 的位置编码增强：注入 disentangled 3D 位置，支持 cascade clustering | **直接 TextVQA 验证 +2~5%**；位置保持对排版/OCR 友好 | Boundary-兼容 (parameter-free, 可叠加任何 merging) | TF, parameter-free | **OCR-safe 关键拼图**：任何 boundary merging 叠加 PPE 保排版 |
| 7 | **Fourier-VLM** (2508.06038) | 频域低通 (2D DCT, FFT O(n log n)) | **低通丢高频 = 文字笔画高频**，OCR 直接冲突；未报 TextVQA | Boundary (ViT 输出) | TF (0 参数) | FLOPs −83.8% 但与 OCR 字形冲突；不宜单独用于 OCR |
| 8 | **METEOR** (2507.20842, ICCV25) | 多编码器: 编码内 rank 协同分配 → 融合 cooperative pruning → **解码内 text-prompt 自适应比例** | 解码阶段 text-prompt 引导可针对 OCR 调比例；但绑定多编码器架构 | Both (编码/融合 boundary, 解码 intra-LLM) | TF (启发式) | text-prompt 解码剪枝是 OCR-safe 可行点，但绑定 multi-encoder |
| 9 | **PLPHP** (2502.14504) | Intra-LLM **逐层逐头**注意力 → Re-attention 现象，强层多留弱层激进剪 | 头级独立性允许"某些头专保文本"，间接利文本但非显式 | **Intra-LLM** (需 LLM 自注意力) | TF | KV cache −50% 解码 +18%，但不可作 boundary 复用 |
| 10 | **RedundancyLens** (2501.19036, ACL25 Findings) ★ | 分析工具+加速：Probe-Activated Dynamic FFN + Hollow Attention (跳过视觉 token self-attn/FFN) + Layer Ranking | 作者 Lianwen Jin (SCUT 顶级 OCR lab)；可能隐含 OCR 友好但未列 OCR benchmark | **Intra-LLM** (减 decoder-only 内部计算) | TF | 与 boundary 剪枝正交可叠加 |
| 11 | **SparseVLM** (2410.04417, ICML25) ★KEY | **Text-guided**: 选相关 text token → 用 **self-attention 矩阵**对视觉 token 评分 → rank-based 自适应层稀疏比 + token recycling | 评分依赖文本↔视觉注意力——OCR prompt 下文字区高分被保留，是文本感知的 | **Definitively INTRA-LLM** (评分用 LLM self-attn, 逐层) | TF (无额外参数) | 文本引导思想是 OCR-safe 正确方向，但实现位置限制 boundary 复用 |

### Q1 — Did ANY training-free BOUNDARY selector crack OCR?
**No clean win.** PPE (training-free, boundary-compatible, **+2~5% TextVQA**) is the strongest evidence but it's a **merging-augmentation operator, not a standalone selector**. VisionTrim (TGVC text-guided merging) and SparseVLM (text-guided attention) are text-aware but: VisionTrim=merging not region-protection; SparseVLM=intra-LLM. **No paper demonstrates a training-free *boundary token-selector* that beats baselines specifically on OCR/TextVQA.** → **OPEN GAP = our entry point.**

### Q2 — SparseVLM: boundary or intra-LLM? Boundary variant feasible?
**Definitively intra-LLM** — scores come from LLM self-attention, applied layer-by-layer.
**Boundary variant is feasible & promising.** ViT (esp. CLIP-style) gives a [CLS]/pooled embedding + patch tokens *before* the LLM. Compute **text↔patch relevance from ViT cross-modal features alone** (CLIP trained for image-text similarity): embed user prompt via text encoder, dot-product with each patch token → OCR/text patches score high when task is reading. This is SparseVLM's *signal* relocated to *boundary*. **No training needed (CLIP alignment is pre-trained). Feasibility HIGH.** Catch: ViT-level relevance coarser than LLM self-attn (no deep linguistic reasoning); pure-ViT boundary variant may need a *small trained* refinement head to match intra-LLM fidelity on OCR — the open question to test at P1.

### Q3 — Most promising mechanism families for OCR-safe boundary pruning
1. **Text-relevance (CLIP text↔patch)** — most directly OCR-aligned; SparseVLM proven in-LLM, relocatable. **TOP PICK.**
2. **Attention-based** (AgilePruner: simple/concentrated-evidence images → text regions spatially concentrated → attention preserves them; lower hallucination than diversity per CHAIR).
3. **PPE positional-preservation as overlay** — orthogonal must-add for any merging boundary method to keep text layout.
4. **Diversity/erank** — RISK for OCR (AgilePruner: retained diversity correlates with hallucination; text is small spatial subset easily dropped by coverage-max selection).
5. **Energy/spectral (E-AdaPrune) + frequency low-pass (Fourier-VLM)** — **OCR-hostile**: text strokes are high-freq/high-local-energy, global spectral energy doesn't isolate text semantics; avoid as sole signal.
6. **[CLS]** — cheap but coarse, no spatial specificity; only as pre-filter.

---

## TASK 2 — Fresh scan 2026 H1/H2

### Sub-problem 1 — Boundary query-aware OCR preservation
- **FlashVLM: Text-Guided Visual Token Selection** — arXiv **2512.20561** (verified HTTP200), 2025-12. Pre-LLM boundary: cross-modal similarity (image tokens × normalized text embeddings) fused with saliency + diversity partition; "beyond-lossless" at 77.8% pruning on LLaVA-1.5. **Mechanism overlap (THREAT) but NO OCR claim** — 14 general benchmarks, no OCRBench/TextVQA headline. Supports our OCR-still-open angle.
- **QuietPrune: Query-Guided Early Token Pruning** — CVPR 2026 (Gao et al., openaccess PDF). Semi-structured 2×2, query-guided, early. Direct SparseVLM descendant. No OCR-specific claim surfaced.
- **RTPrune: Reading-Twice Inspired Token Pruning for DeepSeek-OCR** — arXiv **2605.00392** (verified HTTP200), 2026. TF, OCR-targeted, accelerates DeepSeek-OCR. **Closest to "boundary+OCR" but specific to DeepSeek-OCR's optical-compression pipeline, not general VLM.** Watch.
- **VisionDrop (AAAI26), ZOO-Prune (CVPR26, 94.4% prune, 2.30×)** — TF boundary-ish, none claim OCR crack.

### Sub-problem 2 — Qwen2.5-VL / Qwen3-VL compression
- **GlimpsePrune (TCSVT)** — arXiv **2508.01548**, repo [HVision-NKU/GlimpsePrune](https://github.com/HVision-NKU/GlimpsePrune). The reference Qwen2.5-VL compression: glimpse-token + VIP prunes 92.6% tokens on Qwen2.5-VL-7B, <1h training. **CAVEAT:** Qwen3-VL natively does 2×2 MLP compression already (SigLIP-2 + MLP adapter, tech report 2511.21631) → less headroom; LLaVA-1.5 fixed-576 budgets must be re-tuned against Qwen-VL variable tokens + M-RoPE.
- **DeepSeek-OCR optical compression** — arXiv 2510.18234. Text→visual tokens (7–20× compression) preserving OCR. Not Qwen-VL but relevant OCR-region baseline.

### Sub-problem 3 — Serving-engine / deployment-throughput (OUR CORE NOVELTY)
**No vLLM/SGLang-integrated token-compression paper for image/OCR VLMs exists as of 2026-07.** Verified three candidate threats:
- **GlimpsePrune** uses `vllm==0.9.0.1` only as *lmms-eval measurement backend* + RL reward model server — pruning runs in a custom `transformers` fork, **NOT inside vLLM serving path**. Not a served-throughput paper.
- **CodecSight** — arXiv **2604.06036** (verified HTTP200), 2026. vLLM-compatible but **video-codec-metadata patch pruning for streaming video LLMs only.** Not image/OCR.
- **vLLM RFC #45098** "Image Token Pruning for Multimodal Models" — open RFC (`--image-pruning-rate`), **NOT merged, no paper.** Feature request only.

**→ Gap OPEN.** All image-VLM compression papers report FLOPs/accuracy, not served tokens/sec under batching.

### Sub-problem 4 — SparseVLM follow-ups
- **SparseVLM+** (improved text-visual attention pattern) — RG 398766915.
- **TextScythe** (OpenReview) — finds "vision-critical" text tokens first, then prunes visual. 2026 boundary text-guided variant.
- **"Don't Just Chase Highlighted Tokens"** (NeurIPS25 poster 115059) — critiques attention-first pruning; relevant to our saliency design.
- **Security**: "Less Is More — Until It Breaks" (arXiv **2601.12042**, verified HTTP200, 2026-01) — adversarial robustness of compressed VLMs. Novel eval dimension we could claim.

---

## TASK 3 — Qwen3.5-VL assessment + A40 feasibility

**Naming reality check:** "Qwen3.5-VL" is **NOT a standalone model**. As of 2026-07 two branches exist:
- **Qwen3-VL** — the dedicated VLM series (latest, 2025-11). **This is the correct VLM target from LLaVA-1.5.**
- **Qwen3.5** — native multimodal foundation models (0.8B/2B/4B/9B/27B/35B-A3B/122B-A10B/397B-A17B), NOT branded "VL".

**Qwen3-VL variants** = 4 Dense (**2B, 4B, 8B, 32B**) + 2 MoE (30B-A3B, 235B-A22B), each with Instruct + Thinking. HF: `Qwen/Qwen3-VL-8B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct`, `Qwen/Qwen3-VL-2B-Instruct`, collection `huggingface.co/collections/Qwen/qwen3-vl`. **No Qwen3-VL-7B exists** (Qwen2.5-VL had 3B/7B/72B — older gen).

**arXiv tech report:** **2511.21631** "Qwen3-VL Technical Report" (2025-11). (Qwen2.5-VL = 2502.13923.)

**Token mechanics:** native dynamic resolution + **M-RoPE** (time/height/width decomposed RoPE, default-on in vLLM). Configurable visual token budget **256–16384/image**, context up to 256K (extensible 1M). **CRITICAL for our story:** unlike LLaVA-1.5's fixed-576, Qwen3-VL is already variable/budgeted → "compress a fixed sequence" framing must be REFRAMED (e.g. aggressive pruning at high budgets 4K–16K where redundancy is the real pain).

**vLLM support:** **vLLM ≥ 0.11.1** (`docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html`). Servable on our engine.

### ★ 1× A40 46GB feasibility (HARD constraint)
| Variant | bf16 weights | Verdict |
|---|---|---|
| 32B dense | 64GB | **IMPOSSIBLE** bf16 (and even int4 ~16GB but tight KV-cache) |
| 8B dense | ~16GB | **RECOMMENDED** — ~30GB for KV-cache+activations → big concurrency headroom (our serving win) |
| 4B dense | ~8GB | feasible but weaker accuracy |
| MoE 30B-A3B | ~60GB bf16 | needs int4; non-official ckpts have vLLM compat issues — NOT recommended |
| 27B (Qwen3.5 base, not VL) | ~54GB | bf16 IMPOSSIBLE; user's "27B 挺厉害" likely conflated with Qwen3.5 base |

**★ Recommendation: `Qwen/Qwen3-VL-8B-Instruct` (bf16).** Flags for user:
- "Qwen3.5-VL 27B" **does not exist**; 27B is a Qwen3.5 *text* dense variant. Nearest VL at that size = **Qwen3-VL-32B = 64GB bf16 = INFEASIBLE on A40 even with int4** (KV-cache starved).
- No official AWQ/GPTQ int4 for Qwen3-VL-8B yet (3rd-party `llm-compressor` AWQ w4a16 has vLLM recognition issues). For 8B on A40, **don't quantize** — bf16 keeps the benchmark clean (no quant confound on our throughput/accuracy story).
- **Compression story must be reframed**: Qwen3-VL already variable-token → pivot to "redundancy elimination at high visual budgets (4K–16K) + served-throughput under batching", not "fixed-576 sequence compression".

**UNVERIFIED:** exact Qwen3-VL-8B VQAv2/GQA/OCRBench numbers vs LLaVA-1.5-7B; official AWQ release date.

---

## TASK 4 — Synthesis for the strategic fork (D / A'' / C / B')

| Option | New evidence | Verdict |
|---|---|---|
| **A''** (one more selector: SparseVLM-style boundary variant) | SparseVLM is intra-LLM but its *signal* (text↔patch relevance) relocates to boundary via CLIP features **training-free, feasibility HIGH**. No competitor cracked boundary-OCR training-free (PPE only as overlay). This is a clean, defensible, low-cost probe. | **STRONGEST support. Do this next.** |
| **D** (serving-aware method on proxy) | Served-throughput gap **STILL OPEN post-2026-07** (no image/OCR vLLM-integrated paper). Our differentiator holds. GlimpsePrune uses vLLM as eval-only; CodecSight is video; RFC #45098 unmerged. | **Confirmed viable — but needs a selector (A'') first.** D is the *delivery vehicle*, A'' is the *selector component*. Do A'' → plug into D. |
| **C** (switch base to Qwen-VL) | Qwen3-VL-8B is a strong, A40-feasible LLaVA-1.5-7B replacement; **but reframes the compression story** (variable tokens already). More accurate baseline but harder to show compression win (Qwen3-VL native 2×2 MLP already compresses). GlimpsePrune already owns Qwen2.5-VL. | **Defer.** Land the method on LLaVA-1.5-7B first (apples-to-apples vs FastV/SparseVLM), then port to Qwen3-VL-8B as a generality claim. Don't switch base mid-probe. |
| **B'** (reframe around throughput findings) | Gap open + three serving-specific findings locked. But B' alone = empirical study, lower novelty ceiling than D-with-method. | **Keep B' as the framing layer over D**, not a standalone pivot. |

### ★ Recommended path
**A'' first (boundary text↔patch selector, training-free, via CLIP features + PPE overlay), measured on TextVQA/OCRBench.** If it cracks OCR (≥ FastV intra-LLM parity on TextVQA r50), **fold it into D (serving-aware, vLLM-integrated, load-adaptive budget)** — that's the paper. Keep LLaVA-1.5-7B as primary base; add Qwen3-VL-8B as a generality/robustness row. B' is the narrative spine (served-throughput findings) that D delivers on.

---

## Sources
- Mechanism deep-read: arXiv 2603.09480, 2603.05950, 2603.01236, 2601.22674, 2512.08240, 2510.22936, 2508.06038, 2507.20842, 2502.14504, 2501.19036, 2410.04417
- Fresh scan: 2512.20561 (FlashVLM, HTTP200), 2605.00392 (RTPrune, HTTP200), 2601.12042 (security, HTTP200), 2604.06036 (CodecSight, HTTP200), 2508.01548 (GlimpsePrune), 2510.18234 (DeepSeek-OCR), vLLM RFC #45098, NeurIPS25 poster 115059
- Qwen3-VL: arXiv 2511.21631, HF collection qwen3-vl, vLLM Recipes Qwen3-VL, GitHub qwenlm/qwen3-vl

## UNVERIFIED flags
- E-AdaPrune (2603.05950) abstract verified HTTP200 but intro not directly fetched — mechanism drawn from WebSearch snippet, treat as detail-unverified.
- FlashVLM / RTPrune / CodecSight / security-paper: arXiv IDs resolve (HTTP200) but I did not fetch full abstracts to confirm exact benchmark claims — confirm before citing.
- Qwen3-VL-8B exact OCR/TextVQA numbers vs LLaVA-1.5-7B; official AWQ release date.

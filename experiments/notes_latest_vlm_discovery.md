# Latest-VLM Discovery — Independent Family for Workload-Conditional Stage Law
Date: 2026-08-03 · Host: A40 44GB · Context: Rank-Before-Merge / pre- vs post-merger token selection validation
Existing families in paper: Qwen3-VL-8B, Qwen2-VL, InternVL3-8B → reviewers flagged "Qwen-heavy evidence".

**Verification method**: ModelScope REST API (`/api/v1/models/<owner>/<name>`, PUT `/api/v1/models` search, `/repo/files` listing), raw file resolve (`https://modelscope.cn/models/<id>/resolve/master/<file>`) for config.json + modeling source, plus arXiv/web cross-check. All merger claims below were read from actual code (local transformers 4.57.3 for `glm4v`; downloaded modeling files for the rest, kept at `/tmp/vlm_src/`).

---

## 1. Screening table (all candidates)

| Model | Family | Release (ModelScope) | Params | ModelScope ID (verified) | Merger / compression stage (code evidence) | A40 bf16 fit | Verdict |
|---|---|---|---|---|---|---|---|
| **GLM-4.6V-Flash** | GLM (Zhipu) | 2025-12-07 | ~9B dense | `ZhipuAI/GLM-4.6V-Flash` ✔ | `Glm4vVisionPatchMerger`, pixel-shuffle `spatial_merge_size=2` (4× compress); local transformers `models/glm4v/modeling_glm4v.py` L684-784 | ~18GB ✔✔ | **TOP-1** |
| **Kimi-VL-A3B-Instruct** | Moonshot | 2025-04-10 (2506 upd. 06-22) | 16B MoE, 2.8B active | `moonshotai/Kimi-VL-A3B-Instruct` ✔ | MoonViT `patch_merger()` 2×2 space-to-depth + `KimiVLMultiModalProjector` (modeling_kimi_vl.py L560-590, L2298-2323, L2356-2383) | ~33GB ✔ (tight) | **TOP-2** |
| **Ovis2.5-9B** | Ovis (ATH-MaaS/AIDC) | 2025-08-16 | ~9B | `ATH-MaaS/Ovis2.5-9B` ✔ | `VisualTokenizer`: stride-2 merge (4×) → Linear→softmax over 65536 visual codebook → `VisualEmbedding`(VTE) matmul (modeling_ovis2_5.py L645-775) | ~19GB ✔✔ | **TOP-3** (Qwen3-8B backbone caveat) |
| **Step3-VL-10B** | StepFun | 2026-01-13 (tech rep arXiv 2601.09668, Jan 2026) | ~10.2B dense (20.3GB wts) | `stepfun-ai/Step3-VL-10B` ✔ | PAE ViT (1.8B) + two-stage strided conv downsamplers: `vit_downsampler1/2 = Conv2d(k3,s2)` → 4× compress (vision_encoder.py L387-396) | ~20GB ✔ | **#4 alt** (Qwen3-8B backbone caveat) |
| GLM-4.1V-9B-Thinking | GLM (Zhipu) | 2025-06-28 | ~9B dense | `ZhipuAI/GLM-4.1V-9B-Thinking` ✔ | identical `Glm4vVisionPatchMerger` (spatial_merge_size=2); config targets transformers 4.57.1 = local | ~18GB ✔✔ | **Same-arch fallback** if 4.6V-Flash load issues |
| MiniCPM-V-4.5 | OpenBMB | 2025-08-25 | ~8.7B | `OpenBMB/MiniCPM-V-4_5` ✔ | 2D Perceiver `Resampler`, `query_num=64` (resampler.py L83+) | ~17.4GB ✔ | Benched: 2025 successor of explicitly-deprecated MiniCPM line; Qwen3-style backbone |
| InternVL3.5-8B | InternVL | 2025-08-25 | ~8B | `OpenGVLab/InternVL3_5-8B` ✔ | pixel-shuffle, `downsample_ratio=0.5` | ~16GB ✔ | Fallback only (InternVL family, per task) |
| Kimi-VL-A3B-Thinking-2506 | Moonshot | 2025-06-22 | 16B MoE | `moonshotai/Kimi-VL-A3B-Thinking-2506` ✔ | same as Instruct | ~33GB ✔ | Backup variant of TOP-2 |
| GLM-4.5V / GLM-4.6V (MoE) | GLM | 2025-08 / 2025-12 | 106B (A12B) | `ZhipuAI/GLM-4.5V`, `ZhipuAI/GLM-4.6V` ✔ | pixel-shuffle merger (Glm4vMoe) | ✘ 212GB bf16 | Rejected: size; active 12B > 8B cap |
| MiMo-VL-7B-SFT / -RL / -RL-2508 | Xiaomi | 2025-04 → 2025-08 | ~8B | `XiaomiMiMo/MiMo-VL-7B-RL` ✔ | MLP w/ pixel-shuffle | ~16GB ✔ | **Rejected: arch is literally `Qwen2_5_VLForConditionalGeneration`** (ModelScope API) → zero independence |
| Ovis2-8B / Ovis-U1-3B | Ovis | 2025-02 / 2025-06 | 8B / 3B | `ATH-MaaS/Ovis2-8B` ✔ | codebook clustering | ✔ | Rejected: superseded by Ovis2.5-9B / too small |
| Ovis-U1-8B | Ovis | — | 8B | **NOT on ModelScope** (only U1-3B found) | — | — | Rejected: availability |
| MiniCPM-V-4.6 | OpenBMB | 2026-05-10 | ~1.3B (single 2.6GB safetensors) | `OpenBMB/MiniCPM-V-4.6` ✔ | n/a | ✔ | Rejected: <7B |
| dots.ocr / dots.ocr-1.5 | rednote | 2025-07-31 / 2026-02-16 | ~3B | `rednote-hilab/dots.ocr-1.5` ✔ | n/a | ✔ | Rejected: <7B, OCR-specialized |
| HunyuanOCR | Tencent | 2025-11-18 | ~1B | `Tencent-Hunyuan/HunyuanOCR` ✔ | n/a | ✔ | Rejected: <7B |
| SmolVLM / SmolVLM2 | HF | 2025-01/02 | ≤2.2B | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` ✔ | n/a | ✔ | Rejected: <7B |
| DeepSeek-VL2 | DeepSeek | 2024-12 | ≤4.5B | `deepseek-ai/deepseek-vl2` ✔ | token reduction | ✔ | Rejected: <7B, old |
| Pixtral-12B | Mistral | 2024-09 | 12B | `AI-ModelScope/Pixtral-12B-2409`, `LLM-Research/pixtral-12b` ✔ | **none** — plain MLP connector, no token-compression stage | ~24GB | Rejected: fails merger criterion + age |
| LLaVA-OneVision(-1.5) | LLaVA | 2024-08+ | 7B | present via mirrors | pixel-unshuffle | ✔ | Deprioritized: Qwen2/2.5 backbone → same lineage critique |
| OvisOCR / OvisOCR2 | Ovis | 2026-06 / 2026-07 | small | `ATH-MaaS/OvisOCR2` ✔ | — | ✔ | Rejected: OCR-specialized; OvisOCR2 arch = `Qwen3_5ForConditionalGeneration` |
| MiMo-VL2 | — | — | — | **no results** on ModelScope | — | — | Does not exist (searched) |
| Qwen3-VL-* | Qwen | — | — | — | — | — | Excluded by family constraint |

No newer (2026) independent-family dense 7–12B VLM with a real merger stage was found on ModelScope beyond Step3-VL-10B (2026-01) and MiniCPM-V-4.6 (2026-05, too small). GLM-4.6V-Flash (2025-12) is the newest qualifying dense model.

---

## 2. Top-3 detailed profiles

### TOP-1 — GLM-4.6V-Flash  (`ZhipuAI/GLM-4.6V-Flash`)
- **Release**: ModelScope created 2025-12-07; README: "GLM-4.6V-Flash (9B), a lightweight model optimized for local deployment and low-latency", 128K context. Dense flagship-lite of the GLM-4.6V series (MoE 106B sibling too big for us).
- **Architecture**: `glm4v` (Glm4vForConditionalGeneration). Text: GLM-4 lineage, 40 layers, hidden 4096, GQA 32/2, mrope. Vision: 24-layer ViT, hidden 1536, patch 14, `out_hidden_size 4096`.
- **Merger (code-verified)**: `vision_config.spatial_merge_size = 2` → `Glm4vVisionPatchMerger` in local transformers 4.57.3 (`models/glm4v/modeling_glm4v.py`, merger instantiated L694; pixel-shuffle reshape L713-728/779-784: `x.reshape(h//2, 2, w//2, 2, d)` → concat → LN+MLP to 4096). Deterministic 4× token reduction, same shape semantics as Qwen2-VL/Qwen2.5-VL merger.
- **Independence**: zero Qwen lineage — GLM-4-9B text backbone, AIMv2-derived vision encoder (arXiv 2507.01006). Different attention (partial-rotary GQA), different ViT family from everything currently in the paper.
- **A40 fit**: ~9B dense → ~18GB bf16 weights; large headroom for activations/KV at moderate resolution. ✔✔
- **Loading**: architecture native to installed transformers 4.57.3 (no remote code). **Risk**: config emitted with `transformers_version "5.0.0rc0"` and uses new-style `rope_parameters`; if Glm4vConfig parse hiccups, either upgrade transformers or fall back to GLM-4.1V-9B-Thinking (identical arch, config targets 4.57.1).
- **Hook plan**: pre-merger selector on ViT block outputs (or the pre-merge 1536-d stream); post-merger selector on `model.visual.merger` output (4096-d, N/4 tokens). Identical instrumentation pattern to our existing Qwen2-VL hooks → minimal runner changes; iso-token pre/post control trivial (merge is positional/deterministic, no data-dependent selection).

### TOP-2 — Kimi-VL-A3B-Instruct  (`moonshotai/Kimi-VL-A3B-Instruct`)
- **Release**: 2025-04-10; Thinking-2506 refresh 2025-06-22. Tech report arXiv 2504.07491. 16B total / **2.8B activated** MoE, 128K context.
- **Architecture**: MoonViT (27L, 1152-d, native resolution) → merger → MLP projector → DeepSeek-V3-style MoE LLM (MLA, 64 routed + 2 shared experts, top-6).
- **Merger (code-verified)**: `patch_merger()` (modeling_kimi_vl.py L560-588): per-image 2×2 space-to-depth reshape (`merge_kernel_size=[2,2]` from config), then `KimiVLMultiModalProjector` (LayerNorm(1152) → view(-1, 4608) → Linear → GELU → Linear(→2048)). Called at MoonViT exit (`MoonVitPretrainedModel.forward` L2356-2373) and inside `KimiVLModel` (L2376-2390).
- **Independence**: Moonshot/DeepSeek lineage — no Qwen, no InternVL. Unique design point for the paper: (a) MoE LLM, (b) MLA attention, (c) native-res MoonViT.
- **A40 fit**: 16B×bf16 ≈ 32-33GB (all experts resident) → ~10-11GB left for ViT activations/KV. Fits at moderate resolution; keep visual-token budget sane (<~4-5K/image) and short generations. ✔ (tightest of top-3)
- **Loading**: remote code (`modeling_kimi_vl.py` ships in repo, `AutoModelForCausalLM` with trust_remote_code; also vLLM-supported). Not in stock transformers → runner needs a small model-specific adapter.
- **Hook plan**: pre-hook = MoonViT encoder output (1152-d tokens, per-image grids via `grid_hws`); post-hook = projector output (2048-d, N/4). Merger is deterministic (positional), good for iso-token controls.

### TOP-3 — Ovis2.5-9B  (`ATH-MaaS/Ovis2.5-9B`)
- **Release**: ModelScope created 2025-08-16 (owner ATH-MaaS = AIDC-AI rebrand; README references `AIDC-AI/Ovis2.5-9B`). OpenCompass 78.3 avg (claimed SOTA <40B open MLLM at release).
- **Architecture**: SigLIP2-SO400M NaViT (512, `hidden_stride=2`) → `VisualTokenizer` → `VisualEmbedding` (VTE) → Qwen3-8B LLM.
- **Merger (code-verified)**: `VisualTokenizer` (modeling_ovis2_5.py L645-775): ViT last hidden → reshape `seq_len//4` (stride-2 space-to-depth, 4× compression) → `head = Linear(4608→65530) + LayerNorm` → **softmax over a 65536-entry visual codebook** → soft tokens → VTE = `Embedding(65536, 4096)` looked up by matmul (soft-attention over embedding table). The compression stage is a genuine *discrete/soft visual-token bottleneck* — architecturally unlike anything else in our paper.
- **Independence caveat**: LLM backbone is literally `Qwen/Qwen3-8B` (config `_name_or_path`). Vision side (SigLIP2 + codebook tokenizer) is independent, but reviewers could still call it "Qwen-adjacent". Rank accordingly; best used as a third family, not the headline independent one.
- **A40 fit**: ~9B + VTE (65536×4096 ≈ 0.27B) ≈ 19GB bf16. ✔✔
- **Loading**: remote code (`modeling_ovis2_5.py`, custom `preprocess_inputs`/chat pipeline; AutoModelForCausalLM w/ trust_remote_code). Moderate integration effort (its tokenizer/preprocessor API differs from HF processors).
- **Hook plan**: pre = `visual_tokenizer._encode` output (4608-d merged features, before codebook head); post = `vte(visual_tokens)` embeddings (4096-d). Note: compression here includes a *learned, data-dependent* softmax bottleneck — an interesting but noisier pre/post L2-norm control than positional mergers; keep as third family.

### #4 alternate — Step3-VL-10B (`stepfun-ai/Step3-VL-10B`, 2026-01-13)
Newest qualifying model (tech report arXiv 2601.09668, Jan 2026). ~10.2B dense; PAE 1.8B ViT + **two-stage strided-conv downsamplers** (`vit_downsampler1: Conv2d(1536→3072,k3,s2,p1)`, `vit_downsampler2: Conv2d(3072→6144,k3,s2,p1)` — vision_encoder.py L387-396, 4× total) then projector into LLM. Excellent, explicit hook target (pre = encoder out; post = downsample2 out / projector out). Downsides: text backbone is Qwen3-8B (same lineage caveat as Ovis2.5), fixed-tile 728px input design, remote code. Promote to TOP-3 if we want maximum recency or a convolutional (non-shuffle) merger design point.

### Benched per task instructions
- **MiniCPM-V-4.5** (2025-08, ~8.7B, `OpenBMB/MiniCPM-V-4_5`): has clean Perceiver resampler (`resampler.py`, `query_num=64`) — but task explicitly deprecates MiniCPM-era families as top choice, and its LLM is Qwen3-style. MiniCPM-V-4.6 (2026-05) is only ~1.3B → fails size criterion.
- **InternVL3.5-8B** (2025-08): pixel-shuffle merger, but InternVL family = fallback-only per task; also Qwen3-8B backbone.

---

## 3. Final recommendation
**Primary: GLM-4.6V-Flash.** Newest qualifying dense model (Dec 2025), fully Qwen/InternVL-free lineage, deterministic pixel-shuffle merger with hook semantics identical to our existing Qwen2-VL instrumentation, ~18GB bf16 → comfortable on the A40, and transformers-native loading. **Runner-up for cross-family generality: Kimi-VL-A3B-Instruct** (MoE + MLA + MoonViT merger; tight but feasible VRAM). **Third family: Ovis2.5-9B** (unique codebook bottleneck; Qwen3 backbone disclosed as caveat). GLM-4.1V-9B-Thinking is the drop-in fallback if GLM-4.6V-Flash's transformers-5-style config misbehaves on 4.57.3.

### Runner integration risk notes
1. GLM-4.6V-Flash config uses `rope_parameters` / was built against transformers 5.0.0rc0 → smoke-test `AutoModel.from_pretrained` on 4.57.3 first; fallback = GLM-4.1V-9B-Thinking (same arch, 4.57.1-era config).
2. Kimi-VL: remote code + MoE — all 16B expert weights resident (~33GB); cap image resolution and batch in iso-token runs; needs a thin model-specific loader branch (trust_remote_code).
3. Ovis2.5: custom chat/preprocess pipeline (`preprocess_inputs`, negative-placeholder input_ids in `merge_multimodal`) — do not reuse our HF-processor path; write a dedicated adapter. Its post-merger tokens pass through a softmax codebook → L2-norm selection post-merger operates on VTE embeddings, which is fine but semantically different from positional mergers (report as such).
4. For all three, verify the exact hook module paths at load time (`visual.merger` for glm4v; `vit.encoder` exit + `multi_modal_projector` for Kimi-VL; `visual_tokenizer` + `vte` for Ovis2.5) with a 1-image forward before running the full benchmark sweep.

Sources (main): arXiv 2507.01006 (GLM-4.5V/4.1V-Thinking), arXiv 2504.07491 (Kimi-VL), arXiv 2506.03569 (MiMo-VL, confirms Qwen2.5-VL base), arXiv 2601.09668 (Step3-VL-10B), arXiv 2508.18265 (InternVL3.5), ModelScope model pages/configs/modeling files listed in table above.

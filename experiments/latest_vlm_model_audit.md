# Latest Independent VLM Family Audit — Cross-Family Gate Target Selection

Date: 2026-08-03 · Decision tier: 协调员自主（<6 GPU·h、非 claim 级、不升级）
Purpose: 降低二审"Qwen-only / 常识性发现"风险——为 workload-conditional stage law 补一个**最新独立模型族**验证（text-dense pre≫post、GQA post≥pre 的 workload 条件反转是否跨族成立）。**目标不是证明 RBM 全面赢 FastV。**
完整筛选细节（20+ 模型、ModelScope API 验证、code-level merger 证据、源码路径/arXiv 出处）：`experiments/notes_latest_vlm_discovery.md`

## 筛选判据（锁定）

1. 非 Qwen；InternVL 仅 fallback；MiniCPM-V / GLM-4V 旧世代明确不作首选。
2. 7B–12B dense（或 MoE active ≤8B），bf16 单张 A40 44GB 可跑。
3. ModelScope 可下载（model ID 经 API 实测验证）。
4. 源码中有**清晰 native visual resampler/merger/token-compression stage**，可定义 pre/post iso-token + iso-selector(L2) 对照。
5. 尽量最新世代（2025–2026）。

## Top-3 候选

| Rank | 模型 | ModelScope ID（已验证） | 发布 | 参数 | Merger stage（代码级证据） | 独立性 | 主要风险 |
|---|---|---|---|---|---|---|---|
| 1 | **GLM-4.6V-Flash** | `ZhipuAI/GLM-4.6V-Flash` | 2025-12 | ~9B dense | `Glm4vVisionPatchMerger` pixel-shuffle `spatial_merge_size=2`（确定性 4× 压缩）；本地 transformers 4.57.3 原生 `models/glm4v/modeling_glm4v.py` L684–784 | GLM-4 文本骨干 + AIMv2 系 ViT，**零 Qwen/InternVL 血统** | config 由 transformers 5.0.0rc 生成（新式 `rope_parameters`）→ 需在 4.57.6 上 load 冒烟 |
| 2 | Kimi-VL-A3B-Instruct | `moonshotai/Kimi-VL-A3B-Instruct` | 2025-04（06 更新） | 16B MoE / 2.8B active | MoonViT `patch_merger()` 2×2 space-to-depth + MLP projector（modeling_kimi_vl.py L560–590, L2356–2383） | DeepSeek-V3 系；MoE+MLA 独特设计点 | trust_remote_code；16B 全专家驻留 ~33GB，A40 紧 |
| 3 | Ovis2.5-9B | `ATH-MaaS/Ovis2.5-9B` | 2025-08 | ~9B | `VisualTokenizer`：stride-2 merge(4×) → softmax over 65536 visual codebook → VTE（modeling_ovis2_5.py L645–775） | 视觉侧独立，**LLM=Qwen3-8B（Qwen-adjacent 瑕疵）** | 自定义 preprocess 管线；codebook 为数据依赖 → post L2 对照语义不同于位置型 merger |

- **#4 备选** Step3-VL-10B（2026-01，最新，双 strided-conv downsampler 已验证）——但文本骨干为 Qwen3-8B，独立性出局，降格。
- **关键拒绝**：MiMo-VL-7B 架构即 `Qwen2_5_VLForConditionalGeneration`（零独立性）；Pixtral 无压缩 stage；GLM-4.5V/4.6V MoE 106B 超显存；MiniCPM-V-4.6 仅 ~1.3B；Ovis-U1-8B 不在 ModelScope；dots.ocr/HunyuanOCR/SmolVLM <7B。
- **"新"的边界（记录在案）**：ModelScope 上不存在比 GLM-4.6V-Flash 更新的独立族 dense 7–12B 且带真 merger stage 的模型（Step3-VL-10B 更新但骨干 Qwen3-8B）。GLM-4.6V-Flash（2025-12）即最新合格候选。

## 最终选择：**GLM-4.6V-Flash**（fallback：GLM-4.1V-9B-Thinking，同架构、config 面向 4.57.1）

理由：
1. **零 Qwen/InternVL 血统**（GLM-4 文本 + AIMv2 视觉）→ 直接回应 Qwen-only 批评；
2. pixel-shuffle merger 与既有 Qwen2-VL instrumentation 形状语义相同 → runner 改动最小，merge 为位置确定性 → **iso-token / iso-selector 对照平凡成立**；
3. ~18GB bf16 → A40 余量充足（含激活/KV）；
4. transformers 原生加载（无 remote code）。

## 架构 GO/NO-GO：**GO（条件式）**

- **GO 依据**：merger 已代码级验证（确定性 4× pixel-shuffle）；pre 臂 hook = merger 输入前 ViT 流（1536-d），post 臂 hook = `visual.merger` 输出（4096-d, N/4 tokens）；pre/post iso-token + L2 iso-selector 对照可定义，与 Qwen3-VL/InternVL3 主矩阵同构。
- **条件风险**：5.0rc-style config 在 transformers 4.57.6 的加载兼容性 → 先 load 冒烟；失败即切 fallback 模型（不改 qwen3vl_clean 的 transformers 版本，保护既有族可复现性）。两者皆败 → 升级 user。

## Gate 计划（load 冒烟通过后执行，预算 <6 GPU·h）

- **Cells**：none / pre / post × textvqa_200 + docvqa_200 + gqa_200（9 cells），keep=25%（runner `--r 0.75`，prune-fraction 约定），selector=l2，官方 scorer 离线重打分（`scripts/rescore_official.py`）。
- **成功判据（stage law 迁移）**：两个 text-dense benchmark 上 pre ≫ post；GQA 上 post ≥ pre（tie 或微胜）——与 Qwen3-VL / InternVL3 同构。
- **失败判据**：text-dense Δ(pre−post) < 5pp，或 GQA 出现反向 crossover → claim 诚实降档并升级 user。
- **红线**：不宣 SOTA、不写 beats existing methods；绝对值因像素 cap / token 预算差异不与 Qwen/InternVL 跨模比较，只比**模式**（方向×workload 条件）。
- **产出**：digest `experiments/glm4v_stage_gate.md`。

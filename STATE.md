# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-02

## 当前阶段
**P2 → 方法 A''（CLIP 边界 selector）实现中**；方向定 A''→D

## ★ 已确立（不变）
- **provisional GO**：压缩→serving 真实加速（GQA e2e 1.33×@r50）。核心 claim 成立。
- **3 条 serving 专属发现**：① e2e>prefill（KV-cache/并发）；② prefill 次线性（vision tower 固定→早 prune 杠杆）；③ 加速依赖视觉占比。
- **served-throughput gap 仍开放**（0/37 OCR-VLM 进 vLLM serving）。

## selector 迭代史
- v1 边界 CLS-attn：TextVQA r50 0.445（vision-tower CLS 不关注文本）❌
- v2 边界 LLM-embed cosine：TextVQA r50 ~0.38（嵌入空间非对比对齐）❌
- proxy(hidden-state)：TextVQA r50 0.530（当前最好边界）｜FastV(intra-LLM)：0.555 但 vLLM 不可集成
- **根因**：边界 training-free 廉价信号打不过 intra-LLM OCR；v2 错在用 LLM 词义空间而非对比空间。

## 立即下一步 —— A''（Dev subagent，CLIP 边界 selector）
- **CLIP 对比对齐**：CLIP text encoder(问题) · CLIP ViT patch 特征 → 选 top-k（对比训练，文本/OCR patch 得分高）= v2 的正确修法。
- 复用 vision-tower hook 取 CLIP 特征 + projector-output compaction；selector `--selector clip_query`。
- **关键验证**：limit=50 TextVQA r0.5 → CLIP 对比能否恢复 OCR（目标接近 FastV 0.555 / proxy 0.530，远超 v2 0.38）。
- A'' + 早 prune 协同：pre-projector 选择 → 既 query-aware 又省编码器/投影器开销（finding #2）。

## A'' 通过后 → D（论文方法）
vLLM 集成 + 早 prune（pre-projector）+ 负载自适应 budget（finding #1）。定位"首个 vLLM 集成、保 OCR 的 training-free 压缩"。基座 LLaVA-1.5-7B，**Qwen3-VL-8B 作泛化行**（Qwen3.5-VL 不存在；32B 不可行；不换基座）。

## 关键约束
- 算力 1× A40 46GB 串行；GPU job 走 `scripts/queue`（driver 连跑需 GPU-settle）。
- env：`vtc` / `vtc_serve`(vLLM0.10.2 V0) / `fastv`(accuracy-only)。
- 提交以用户本人名义，禁 AI 署名。
- 升级找人：凭据 / >6GPU·h / claim 被推翻 / 投稿前。

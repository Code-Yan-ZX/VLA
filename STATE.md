# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-02

## 当前阶段
**P2 selector 三连败 → pivot 到 D-on-proxy（serving-aware 方法）。先做 2 个测量定 D 范围。**

## ★ 论文脊梁（已确立，不变）
- **provisional GO**：压缩→serving 真实加速（GQA e2e 1.33×@r50）。核心 claim。
- **3 条 serving 专属发现**：① e2e>prefill（KV-cache/并发收益）；② prefill 次线性（vision tower 固定开销）；③ 加速依赖视觉占比。
- **served-throughput gap 仍开放**（0/37 OCR-VLM 进 vLLM serving）——核心差异判据。

## selector 三连败（停止追逐）
v1 边界 CLS-attn 0.445 → v2 LLM-cosine 0.38 → A'' CLIP-对比 0.18（TextVQA r50）。A'' 实证根因：CLIP 只对齐 [CLS] 非 per-patch（MaskCLIP 已知结果）。proxy(hidden-state) 0.530 = 边界最好；FastV(intra-LLM) 0.555 不可集成。**结论：边界 TF 廉价信号结构性打不过 intra-LLM OCR，接受 proxy 级精度。**

## 立即下一步 —— D-on-proxy（先测量、后实现）
方法 = proxy selector + serving-aware 优化。先跑 2 个测量定范围：
1. **prefill 拆解**（vision tower vs LLM 占比）→ 定"早 prune / mid-encoder prune"值不值得动 ViT 手术。
2. **并发 × prune-rate 权衡**（不同 concurrency 下 r0/r50/r75 的 req/s）→ 验**负载自适应 budget** 概念（finding #1 的 serving 专属新意，0/37 碰过）。
测量结果定 D 实现：若并发自适应显著抬 req/s → D 核心就是 KV-cache/负载自适应 budget（最 novel）；若早 prune 收益大 → 加 mid-encoder prune。

## D 之后 → P3
全基准（MME/MMBench/ScienceQA）× 多压缩比 × {proxy, D-method, FastV(accuracy anchor)}。基座 LLaVA-1.5-7B；Qwen3-VL-8B 泛化行。

## 关键约束
- 算力 1× A40 46GB 串行；GPU job 走 `scripts/queue`（driver 连跑需 GPU-settle）。
- env：`vtc` / `vtc_serve`(vLLM0.10.2 V0) / `fastv`(accuracy-only)。
- 提交以用户本人名义，禁 AI 署名。training-free 优先（1×A40 约束）。
- 升级找人：凭据 / >6GPU·h / claim 被推翻 / 投稿前。

# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-01

## 当前阶段
**P2 方法-v1 对比完成 → 方法重定向到 v2（query-aware boundary selector）**

## ★ 已确立（不变）
- **go/no-go provisional GO**：压缩→serving 真实 wall-clock 加速（proxy probe: GQA e2e 1.33× @ r50）。核心 claim 成立。
- **3 条 serving 专属发现**：① e2e>prefill（KV-cache/并发收益）；② prefill 次线性（vision tower 固定开销→早 prune 杠杆）；③ 加速依赖视觉 token 占比。

## P2 方法-v1 结果（proxy vs v1-真CLS vs FastV，详见 `eval/p2_method_v1_comparison.md`）
- **v1 真 CLS-attn 非赢家**：TextVQA 灾难（r50 0.445 vs proxy 0.530，r75 0.275）—— vision-tower CLS 不关注文本/OCR。GQA 也不优（r50 0.545≤proxy 0.565）。capture 另加吞吐开销。
- **FastV（intra-LLM）OCR/极端压缩占优**（GQA r75 0.515, TextVQA r50 0.555），但 intra-LLM 在 vLLM 不可集成。
- **核心约束**：serving 方法必须**边界 prune**（vLLM 可集成），但边界视觉显著性丢 OCR → **需要边界 query-aware selector**。

## 立即下一步 —— 方法 v2（Dev subagent）
1. **边界 query-aware selector**：问题文本 ↔ patch 相关性选 top-k（边界、vLLM 可集成、保任务/文本 patch）。目标：边界恢复 FastV 级 OCR 精度。
2. **早 prune**（finding #2）：选择点移入 vision tower（V0 eager，可行）→ 编码器少干活 → 更大 prefill 加速。
3. **KV-cache/batch 感知 budget**（finding #1，v2 sketch）。
4. served 吞吐为差异判据（0/37 测过）。

## v2 之后
- P3 全基准（MME/MMBench/ScienceQA 等）× 多压缩比 × {v2, FastV, 早-prune} 基线。
- 复现 1-2 个 serving-throughput baseline 对比（无现成，自建）。

## 关键约束
- 算力 1× A40 46GB 串行；GPU job 走 `scripts/queue`（注意：driver 连跑需 GPU-settle，曾出现 stale-vLLM 进程致 OOM）。
- env：`vtc` / `vtc_serve`(vLLM0.10.2 V0) / `fastv`(FastV accuracy-only)。
- 提交以用户本人名义，禁 AI 署名。
- 升级找人：凭据 / >6GPU·h / claim 被推翻 / 投稿前。

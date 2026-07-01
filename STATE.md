# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-01

## 当前阶段
**P2 — go/no-go 闸门已过（provisional GO）**；TextVQA 曲线运行中 → 接方法设计

## ★ go/no-go 判定（2026-07-01）：provisional GO，非 §6 升级
GQA 曲线（LLaVA-1.5-7B @ vLLM V0, n=200, 连续 compaction, proxy CLS-attn selector）：
| r | e2e req/s 加速 | prefill(TTFT) 加速 | acc Δ |
|---|---|---|---|
| 0.25 | 1.17× | 1.14× | −3.0% |
| 0.50 | **1.33×** | 1.24× | −2.0% |
| 0.75 | **1.43×** | 1.30× | −11.5% |
核心 claim（压缩→serving 真实 wall-clock 加速）**确认**。详见 `eval/p2_probe_summary.md`。
**两条论文级发现**：① e2e>prefill（每档都是）→ 收益主来自 KV-cache/并发，非 prefill FLOPs（serving 专属，离线测不到）；② prefill 次线性 → vision tower 固定开销 → 启示更早 prune。

## 立即下一步
1. **TextVQA 曲线跑**（driver `bclc6511e`，~60min）—— OCR 鲁棒性（gate 次判据 ≤5% 掉点）。已修 score_textvqa 签名。
2. TextVQA 完 → 补 `eval/p2_probe_summary.md` + 各 digest + commit。
3. **P2 方法设计**（`notes/method-design.md` 扩展，Dev subagent）：① 真 CLS-attn selector（替 proxy）提 acc；② **编码器内/后更早 prune**（吃掉固定编码器开销，提 prefill）；③ **KV-cache/batch 感知 budget**（强化 e2e>prefill 这个核心差异点）。复现 FastV 基线（`fastv` env，clone 上游）。
4. → P3 全基准 × 多压缩比 × 基线对比。

## 已完成
- P0 env+scaffold+queue；P1 lit-survey(37 法)+positioning(Gap A)；P2 probe（vtc_serve/vLLM0.10.2/V0 hook+placeholder-compact patch c05ca86、scorer 46f9b9d）。
- novelty 复核 Gap A 仍 OPEN（0/37 在 serving engine 内测吞吐）。

## 关键约束
- 算力 1× A40 46GB 串行；GPU job 走 `scripts/queue`。env：`vtc`/`vtc_serve`(vLLM0.10.2)/`fastv`。
- 提交以用户本人名义，禁 AI 署名。
- 升级找人：凭据 / >6GPU·h / claim 被推翻 / 投稿前。

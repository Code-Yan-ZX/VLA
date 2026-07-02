# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-02

## 当前阶段
**P2 → D 实现（负载自适应 budget）。D 范围已测量锁定。**

## ★ 论文脊梁（已确立）
- **provisional GO**：压缩→serving 真实加速。
- **3 条 serving 发现**：① e2e>prefill（KV-cache/并发）；② prefill 次线性（vision tower 仅 6.6%→不值得早 prune）；③ 加速依赖视觉占比。
- **served-throughput gap 仍开放**（0/37）。
- **headline 数字**：c12/r75 = **1.76× req/s**（加速随并发放大）。

## selector 定论
三连败（CLS-attn 0.445→LLM-cosine 0.38→CLIP 0.18，TextVQA r50）。边界 TF 天花板 = proxy(hidden-state) 0.530。**接受 proxy 级精度，停止 selector 追逐。**

## D 范围（测量锁定，详见 notes/p2_d_measurements.md）
- **BUILD**：负载自适应/KV-cache 感知 budget —— r = f(并发, KV-占用) ∈ [r_min, r_max]，高负载多剪抬 req/s、低负载少剪保精度。这是 D 核心、0/37 碰过的新意。
- **SKIP**：早 prune / mid-encoder ViT 手术（M1: vision tower 仅 6.6% TTFT，不值得）。
- **KEEP**：proxy selector + served-throughput 评测（并发×prune 矩阵作 headline）。accuracy guardrail r_max ≤ 0.50。

## 立即下一步 —— D 实现（Dev subagent）
1. 控制器：读 vLLM V0 引擎状态（KV-占用 / 运行序列数）→ 算 r(load) ∈ [r_min, r_max]。
2. 执行器：proxy selector 用动态 r（per-request，按提交时负载）。
3. 验证：**adaptive vs fixed-r50/r75**，在变化负载 profile 下比 req/s + accuracy。claim：adaptive 在 iso-accuracy 下 req/s 高于 fixed（高负载多剪、低负载少剪）。
4. accuracy guardrail 机制（r_max 按 benchmark 限）。

## D 之后 → P3
全基准（GQA/MME/MMBench/ScienceQA/TextVQA）× 多压缩比 × {adaptive-D, fixed-proxy, FastV(anchor)} × 并发矩阵。基座 LLaVA-1.5-7B；Qwen3-VL-8B 泛化行。→ P4 写作。

## 关键约束
- 算力 1× A40 46GB 串行；GPU job 走 `scripts/queue`（driver 连跑需 GPU-settle）。
- env：`vtc` / `vtc_serve`(vLLM0.10.2 V0) / `fastv`。training-free 优先。
- 提交以用户本人名义，禁 AI 署名。
- 升级找人：凭据 / >6GPU·h / claim 被推翻 / 投稿前。

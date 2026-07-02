# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-02

## 当前阶段
**P2 → D 实现完成 + 验证通过（n=100 快验）。下一步 P3 全基准扫。**

## ★ 论文脊梁（已确立）
- **provisional GO**：压缩→serving 真实加速。
- **3 条 serving 发现**：① e2e>prefill（KV-cache/并发）；② prefill 次线性（vision tower 仅 6.6%→不值得早 prune）；③ 加速依赖视觉占比。
- **served-throughput gap 仍开放**（0/37）。
- **headline 数字**：c12/r75 = **1.76× req/s**（加速随并发放大）。
- **D headline（n=200 已验证）**：GQA bursty **adaptive Pareto-dominates fixed** —— req/s 3.38 > r25 3.32（+0.06），acc 0.546 > r50 0.522（+0.024，也击败 r25）。TextVQA bursty：adaptive acc 0.554 > r50 0.526（+0.028），req/s 与 r25 平（2.60 vs 2.63）。constant 2.06× 快于 bursty（负载跟踪确认）。详见 notes/p2_d_results.md。

## selector 定论
三连败（CLS-attn 0.445→LLM-cosine 0.38→CLIP 0.18，TextVQA r50）。边界 TF 天花板 = proxy(hidden-state) 0.530。**接受 proxy 级精度，停止 selector 追逐。**

## D 实现（完成，详见 notes/p2_d_results.md）
- **引擎负载读取（vLLM V0，可工作）**：`llm.llm_engine.scheduler[0].running`（num_running）+ `.block_manager.get_num_free/total_gpu_blocks`（KV-占用）。无需 fallback。
- **控制器**：分段线性 r = f(occ) ∈ [r_min, r_max]，阈值 occ_lo/hi **calibrate 到部署负载区间**（c12/短序列峰值 occ 仅 ~0.04 → occ_lo=0.02/hi=0.10）。
- **3 个结构坑（已解）**：① sync llm.chat() 排空引擎 → 改 engine streaming loop（add_request+step），逐段 drain；② batched forward + shared-hook-k → per-segment r（非 per-request）+ 段间 reset_mm_cache；③ 变 k 破 CUDA graph → adaptive 用 enforce_eager。one-segment-lag 控制器（段内采样峰值负载→下段决策）。
- **验证**：adaptive vs fixed-r25/r50 under bursty，n=100 GQA。**adaptive Pareto-dominates**（上）。控制器实自适应（r 0.250↔0.305 随 occ 0↔0.04）。

## 立即下一步 —— P3 全基准扫
跑 `notes/d_method_jobs.json`（n=200 GQA adaptive/fixed × {bursty,step,constant} + TextVQA adaptive/fixed），收紧 D 的 Pareto 证据。然后全基准（GQA/MME/MMBench/ScienceQA/TextVQA）× 多压缩比 × {adaptive-D, fixed-proxy, FastV(anchor)} × 并发矩阵。基座 LLaVA-1.5-7B；Qwen3-VL-8B 泛化行。→ P4 写作。

## D 之后 → P3
全基准（GQA/MME/MMBench/ScienceQA/TextVQA）× 多压缩比 × {adaptive-D, fixed-proxy, FastV(anchor)} × 并发矩阵。基座 LLaVA-1.5-7B；Qwen3-VL-8B 泛化行。→ P4 写作。

## 关键约束
- 算力 1× A40 46GB 串行；GPU job 走 `scripts/queue`（driver 连跑需 GPU-settle）。
- env：`vtc` / `vtc_serve`(vLLM0.10.2 V0) / `fastv`。training-free 优先。
- 提交以用户本人名义，禁 AI 署名。
- 升级找人：凭据 / >6GPU·h / claim 被推翻 / 投稿前。

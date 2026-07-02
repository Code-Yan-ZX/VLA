# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-02

## 当前阶段
**P2 方法完成（D=负载自适应 budget，n=200 验证通过）→ P3（先精化控制器，再全基准）**

## ★ 论文脊梁（已确立）
- **provisional GO**：压缩→serving 真实加速。
- **3 条 serving 发现** + **served-throughput gap 仍开放（0/37）**。
- **headline 数字**：c12/r75=**1.76× req/s**（M2）；constant vs bursty=**2.06×**（负载跟踪）。
- **D（n=200 GQA bursty）**：adaptive **Pareto-dominates** fixed（req/s 3.38>r25 3.32，acc 0.546>r25/r50 0.522）。TextVQA：adaptive acc 0.554>r50 0.526，req/s≈r25。详见 notes/p2_d_results.md。

## selector 定论
三连败（CLS-attn/LLM-cosine/CLIP-对比，TextVQA r50）。边界 TF 天花板=proxy(hidden-state) 0.530。**接受 proxy 级精度。**

## D 方法（完成）
控制器 r=f(负载)∈[r_min,r_max]，读 vLLM V0 `scheduler.running`+`block_manager`。proxy selector 动态 r。engine streaming loop + per-segment r + enforce_eager。n=200 GQA bursty Pareto-dominate fixed。

## 立即下一步 —— P3-step-1（Dev，精化控制器 + 干净 Pareto）
1. **负载信号换 num_running/max_num_seqs**（当前 KV-occupancy 在 c12/短序列峰值仅 ~0.04→控制器 barely 离开 r_min，realized r 0.25-0.30）。num_running 0→12 映射 0→1 跨满量程 + 重标定阈值 → r 真正在 [r_min,r_max] 全范围跑。
2. **重验 Pareto @ max-tokens=32**（当前=16 截断答案致 acc 偏低）：GQA+TextVQA adaptive vs fixed-r25/r50，n=200，确认 adaptive 干净 Pareto-dominate（更强 headline）。
**P3-step-2**：扩旗舰基准 MME/MMBench/ScienceQA；FastV accuracy anchor；并发矩阵；消融。→ P4 写作。

## 关键约束
- 算力 1× A40 46GB 串行；GPU job 走 `scripts/queue`（driver 已带 GPU-settle 保护）。
- env：`vtc` / `vtc_serve`(vLLM0.10.2 V0) / `fastv`。training-free 优先。
- 提交以用户本人名义，禁 AI 署名。
- 升级找人：凭据 / >6GPU·h / claim 被推翻 / 投稿前。

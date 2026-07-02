# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-02

## 当前阶段
**P3 评测基本完成（5 基准 + 并发矩阵 + FastV anchor）→ P4 写作。方法 framing 已定。**

## ★ 论文脊梁（measurement-led，已确立）
- **provisional GO**：压缩→serving 真实加速。
- **测量主贡献**：served-throughput（0/37 测过）+ 3 发现（e2e>prefill/KV-cache-并发、prefill 次线性、视觉占比依赖）。headline：c12/r75=**1.76×**、constant-vs-bursty=**2.06×**、并发放大（c1→c12 r50/r0 1.17×→1.42×）。
- **方法（supporting）**：load-adaptive budget = **throughput-optimal under per-benchmark acc guardrail**。robust：全部 5 基准 req/s 胜 r25（+2-7%）；acc 胜 r50 仅在 r50-costly 基准（**MME、ScienceQA** Pareto-dominate；GQA/MMBench/TextVQA 则否）。

## selector 定论
三连败（CLS-attn/LLM-cosine/CLIP-对比）。边界 TF 天花板=proxy(hidden-state)。**接受 proxy 级精度。** FastV(intra-LLM) accuracy 可比但跑不进 vLLM。

## D 方法（完成）
控制器 r=f(num_running/max_num_seqs)∈[r_min,r_max]，conc_lo=0.25/conc_hi=0.75，realized r 全范围跑。proxy selector 动态 r。engine streaming loop + per-segment r + enforce_eager。

## P3 关键数据（详见 notes/p3s2_pareto.md, notes/p3s1_pareto.md, notes/p2_d_results.md）
- MME/ScienceQA：adaptive Pareto-dominate（acc-sensitive 任务）。
- GQA/MMBench：r50 acc-中性→fixed-r50 占优。
- TextVQA n=500：adaptive acc ≈ r50（n=200 的胜是噪声）。
- FastV anchor：accuracy 可比，不可 serving。
- 并发矩阵：压缩收益随并发放大（M2 确认）。

## 立即下一步 —— P4 写作
1. **整合结果** → `eval/final_results.md`（论文表格/图数据：5 基准×configs、并发矩阵、3 发现、controller 跟踪图）。
2. **锁定 method-design**（最终 D + throughput-optimal-under-guardrail framing）。
3. **nature-writing 起草各章**（intro/related/method/exp/discussion），measurement-led 定位。目标期刊：Pattern Recognition / Information Sciences / Neurocomputing。
4. nature-figure 出投稿级图；nature-citation 补引用；nature-polishing 润色。
→ P5 投稿前**强制升级找人**。

## 关键约束
- 算力 1× A40 46GB 串行；env：vtc/vtc_serve(vLLM0.10.2 V0)/fastv。training-free。
- 提交以用户本人名义，禁 AI 署名。升级找人：凭据/>6GPU·h/claim 推翻/投稿前。

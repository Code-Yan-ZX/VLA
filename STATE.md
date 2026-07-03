# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-03

## 当前阶段
**P3-step-4（用户选 B）：KV-admission 方法赌注——先验 regime-thesis，再投实现**

## §6 升级已解决（用户定 B）
n=500 推翻方法 Pareto claim（0/5）。用户选 **B**：投资更狠方法（KV-admission）而非直接写 measurement-led。测量支柱不受影响（solid）。

## 方法 thesis（B 路径核心）
**load-adaptive 的收益是 regime-dependent**：n=500 在 compute-bound/短序列（KV-occ 仅 0.04，admission 不受 KV 约束）→ null。**KV-bound regime**（高并发 + 长输出，KV 是瓶颈）下，pruning 释放 KV → admission 受益 → adaptive 应胜 fixed。方法 = **admission-aware pruning**（prune arriving request 以 fit KV budget，紧耦合于 vLLM scheduler admission，0/37 碰过的杠杆）。

## 立即下一步 —— P3-step-4（Dev，先验 thesis，phased）
1. **确立 KV-bound regime**（1×A40）：高 max_num_seqs(24+) + 长 max-tokens(128-256) + 可能缩 KV pool(gpu_mem 0.7)→让 KV 成瓶颈。验证此处 pruning(r50) 的 req/s 增益 > 短序列 c12（甜区）。
2. **该 regime 跑 adaptive vs fixed-r25/r50**（GQA + 一个长输出基准）→ **adaptive 是否在此胜 fixed**（c12/短序列没胜处）？
3. **thesis 成立 →** 实现 admission-aware pruning + 验证胜 fixed（KV-bound 下）。
4. **仍不胜 →** thesis 死，回 A（measurement-led 写作）。

## ★ 论文脊梁（不变）
- served-throughput 首测（0/37）+ 3 发现（e2e>prefill/KV-cache-并发、prefill 次线性、视觉占比依赖）+ headline（1.76×、并发放大）。**measurement-led，solid。**
- D load-adaptive：n=500 仅 dominate r25 on 3/5（modest）—— B 旨在强化。

## 关键约束
- 算力 1× A40 46GB 串行；env：vtc/vtc_serve(vLLM0.10.2 V0)/fastv。training-free。
- 提交以用户本人名义，禁 AI 署名。升级找人：凭据/>6GPU·h/claim 推翻/投稿前。

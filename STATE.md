# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-03

## 当前阶段
**P4 写作（measurement-led）。B 路径已放弃 → 回 A。**

## 方向定论（A 路径，measurement-led）
- **B（KV-admission）已放弃**：1×A40 上 KV-bound regime 压不出（peak KV-occ 仅 0.148，需 >>0.5）；thesis 未验证，user 中途停。详见 DECISIONS 2026-07-03。
- **回 A**：measurement-led 论文。测量是主轴（0/37 served-throughput + 3 发现 + headline 1.76×/并发放大），方法如实作 supporting（load-adaptive n=500 仅 dominate r25 on 3/5；KV-admission 作 future work）。

## ★ 论文脊梁（solid，reviewer 推不倒）
- **首测**：VLM 视觉 token 压缩的 served-throughput 在 serving engine（vLLM）内首测（0/37 论文做过）。
- **3 发现**：① e2e>prefill（KV-cache/并发收益，非 prefill FLOPs）；② prefill 次线性（vision tower 仅 6.6%）；③ 加速依赖视觉占比。
- **headline**：c12/r75=1.76× req/s；并发放大（c1→c12 r50/r0 1.17×→1.42×）；constant-vs-bursty 2.06×。
- 目标期刊：**Pattern Recognition**（primary）；Information Sciences / Neurocomputing 备选。

## 立即下一步 —— P4 写作
1. **起草各章** → `drafts/paper_v1.md`（按 `drafts/outline.md`，7 章，measurement-led，诚实方法段）。源：eval/final_results.md + notes/{positioning,lit-survey,method-design} + p3s2/p3s3/d measurements。
2. nature-figure 出 4 图（served-throughput gap、并发×prune 曲线、controller 跟踪、Pareto 前沿）。
3. nature-citation 补 CNS/领域引用；nature-polishing 润色。
→ P5 投稿前**强制升级找人**。

## 已完成
P0 env/基建；P1 37 法 survey + Gap A 定位；P2 probe（gate 过）+ selector 三连败（CLS/cosine/CLIP）→ proxy 天花板；D load-adaptive 实现 + n=200/n=500 验证（modest）；P3 5 基准 + 并发矩阵 + FastV anchor。

## 关键约束
- 算力 1× A40 46GB 串行；env：vtc/vtc_serve(vLLM0.10.2 V0)/fastv。training-free。
- 提交以用户本人名义，禁 AI 署名。升级找人：凭据/>6GPU·h/claim 推翻/投稿前。

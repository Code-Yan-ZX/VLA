# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-02

## 当前阶段
**P3-step-1 完成（num_running 控制器 + 干净 mt32 Pareto）→ P3-step-2（扩基准/消融）**

## ★ 论文脊梁（已确立）
- **provisional GO**：压缩→serving 真实加速。
- **3 条 serving 发现** + **served-throughput gap 仍开放（0/37）**。
- **headline 数字**：c12/r75=**1.76× req/s**（M2）；constant vs bursty=**2.06×**（负载跟踪）。
- **D（P3-step-1，mt32，num_running 信号）**：**TextVQA adaptive 干净 Pareto-dominate 两固定点**（req/s +0.044>r25，acc +0.020>r50）。GQA 在 mt32 下 r50 acc-中性（短答案可恢复）→ r50 双轴占优，adaptive 仅 req/s 胜 r25。详见 notes/p3s1_pareto.md。

## selector 定论
三连败（CLS-attn/LLM-cosine/CLIP-对比，TextVQA r50）。边界 TF 天花板=proxy(hidden-state) 0.530。**接受 proxy 级精度。**

## D 方法（P3-step-1 完成）
控制器 r=f(num_running/max_num_seqs)∈[r_min,r_max]，conc_lo=0.25/conc_hi=0.75。bursty profile 交替小/大 burst（2/12@c12）让控制器真在 [r_min,r_max] 全范围跑（realized r 0.25↔0.50，16:14）。step profile 控制器图：r run-length 0.25x31→0.50x1→0.25x109。修了 gap-block 重复行 bug（200→249 行）。

## P3-step-1 诚实结论
- **TextVQA**（OCR，r50 有 acc 代价）：adaptive 干净 Pareto-dominate，且比 mt16 更强（mt16 是 acc-only，mt32 双轴胜）。
- **GQA**（短答案，r50 acc-中性）：adaptive 无法占优 r50；prior mt16 的 Pareto 胜是截断解码假象。**load-adaptive 的收益是 benchmark-conditional**——压缩有代价处（OCR）才有收益。

## 立即下一步 —— P3-step-2（Dev，扩基准 + 消融）
1. 扩旗舰基准：MME/MMBench/ScienceQA（长答案/密集文本→r50 应有 acc 代价→adaptive 应胜）。
2. FastV accuracy anchor（对比 SOTA training-free）。
3. 并发矩阵（c1/c4/c12/c24）+ 消融（r_min/r_max、conc 阈值、信号选择）。
4. n=500 TextVQA 收紧 acc 噪声（当前 +0.020 在 n=200 噪声内）。
→ P4 写作。**核心 framing**：load-adaptive 在压缩有代价的任务上 Pareto-dominate。

## 关键约束
- 算力 1× A40 46GB 串行；GPU job 走 `scripts/queue`（driver 已带 GPU-settle 保护）。
- env：`vtc` / `vtc_serve`(vLLM0.10.2 V0) / `fastv`。training-free 优先。
- 提交以用户本人名义，禁 AI 署名。
- 升级找人：凭据 / >6GPU·h / claim 被推翻 / 投稿前。

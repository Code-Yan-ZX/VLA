# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 详见 ORCHESTRATION.md
> 最近更新：2026-07-13 · **⚠️ v2 头条 claim 被 ERA 推翻（在 vLLM 内测了 served throughput）→ 升级找人定方向；已停 v2 自主改动** · EV-VAR 计划 ~/.claude/plans/warm-tinkering-bird.md（已关闭）

## ★ ⚠️ v2 头条 novelty claim 被推翻（2026-07-13，charter §6 升级）
v2 核心卖点"**0/37 surveyed compressors report served throughput inside a production serving engine**"被 **ERA**(arXiv:2606.31982, Jun 2026) 打破：ERA 是 VLM 视觉 token 压缩器，在 vLLM 内测 served req/s(~42)、prefill 4.3×、KV 5×、peak mem 244→60GB。ERA 不在 37 survey（漏）。Salvage = sharpen claim 到 ERA 漏的：**continuous batching + online arrivals、TTFT、p99、goodput@SLO**（ERA 仅静态 batch）。**re-scan 现为必要**（ERA 被漏=survey 有 gap）。已做：§5.9 negative 节 + 5 图整合 + RTP-LLM 事实错误已修(commit f3449f6)。待 user 定 A(re-scan+sharpen)/B(sharpen now)/C(重评估)。详见 DECISIONS.md。

## ★ 当前主线（待重定向）：v2 测量论文（measurement-led，drafts/paper_v2.md）
v2 = 0/37 served-throughput 测量（survey 37 法无一在 serving engine 内测吞吐）+ 3 发现 + c64 goodput，9表5图47ref。**user 2026-07-13 定：升为主线，目标 Q1/Q2**。下一步：评估当前稿状态 → 整合 negative 节 → 投稿就绪。

## ★ method search 收束（两条 method 线均 GPU 证伪 → 折入 v2 negative-finding 节）
1. **ElasticVis allocator**（EV-1e 终判 negative）：连续批处理下 per-request 视觉 token 预算不提升 goodput@SLO（架构性）。
2. **EV-VAR variance 外部性**（Stage 1 ❌ NO-GO，high conf）：4-cell GPU step-log 回归（2048 steps），`{var_k,max_k}` 零预测力（F=0.18 p=0.84），chunked-ON 残余 F=0.20 p=0.82；max_k 系数反方向。variance 信号被证伪。
**共同根因**：serving-signal-based（per-request/variance）方法在连续批处理下被 **total-token(sum_k)** 主导；chunked-prefill 封顶 sum_k/step → 每 step wallclock 由 n_prefill/n_decode（工作结构）驱动，与 k 方差无关，无信号供 scheduler 榨取。两条 negative 折成 v2 的"方法空间探索与界定"节。产物：src/ev_var/*.py（分析）、serve_bench `--log-step-composition`（工具 commit 14c3645）、runs/ev_var/（gitignored）。详见 DECISIONS.md。

## v2 资产（substrate，全现成）
framework `src/serve_bench.py`(V1 c64 goodput+openloop+per-req k+`--log-step-composition`)+`load_controller.py`+`compressors.py`+`src/elasticvis/`(sim 作 null)；数据 eval/subsets/{gqa,textvqa,mme,mmbench,scienceqa}_{200,500}.jsonl + runs/v2_p*/；模型 LLaVA-1.5-7B(runs/models/)+Qwen3-VL-8B；env `qwen3vl_clean`(V1, vllm0.19)。

## 评测制度（沿用）
open-loop 变载到达为主 + 混合-SLO 为辅。baseline：fixed-{r0,r25,r50,r75}+控制器+oracle。

## 已完成（背景）
P0-P4 全完成（lit+定位｜probe｜selector 三连败→proxy 天花板｜v2 实验(2引擎×2架构×4压缩器+c64+goodput)｜v2 论文）；ElasticVis EV-0..1e + EV-VAR Stage0-1（method search 收束）。详见 DECISIONS.md。

## 关键约束
- 算力 1× A40 46GB 串行；env `qwen3vl_clean`(V1)。
- 提交以用户本人名义，禁 AI 署名。每步 web 核实版本+novelty 监控。
- 升级找人：凭据 / >6GPU·h / claim 推翻 / 投稿前。

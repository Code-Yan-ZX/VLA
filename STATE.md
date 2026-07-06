# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 详见 **ORCHESTRATION.md** + **notes/elasticvis_design.md**（spine）+ **notes/elasticvis_positioning.md**
> 最近更新：2026-07-06 · **新主对话从 notes/elasticvis_design.md 读起**

## ★ 当前方法（ElasticVis）— 定位已重定位（2026-07-06 novelty 扫描）
**准入时、由实时负载+SLO 裕度驱动、优化 goodput@SLO 的 per-request 视觉 token 分配器**，坐在 elastic 压缩器（PARCEL-like）之上，非压缩器。
**❗"0/N 做 per-request 预算"claim 已 RETIRE**：CARES/DyToS/PARCEL 已做 per-request 视觉预算（内容驱动/无系统信号），AdaServe/SLOs-Serve 做了 per-request budget-for-goodput（文本域）。**干净单元 = 系统信号驱动（H1）**。详见 `notes/elasticvis_design.md` §0。

**已批准 spine：A→B**（用户 2026-07-06）。核心 H1 系统信号 allocator（现有数据够），内容维度 H2 作后续扩展（对标 CARES）。

## ★ 立即进行：EV-0 go/no-go = **NO-GO on GQA**（2026-07-06，升级找人定方向）
EV-0 三组件全建好（predictors LatPred R²0.996+hybrid / allocator H1 机制本地验证 / sim slot+queue）。**独立 probe（real vs wide acc）推翻 Z 的 headline**：Z 报 H1b "1.10× win" 是 MockAcc(0.4-0.6) artifact。**真实 GQA acc(k)=0.476-0.595（区间 0.119）下，H1 均匀-SLO=0.79 LOSE、H1b 混合-SLO=0.998 TIE/LOSE。** 机制真实但被 **accuracy(k) 陡度门控**：flat acc→吞吐主导→Fixed 赢；steep acc→ElasticVis 赢（wide-acc 0.30-0.70 → H1b 1.307 WIN）。sim sanity k144 低估 2.5× 且偏差有利于 ElasticVis → GPU 大概率更差 → NO-GO 稳健。**待定方向：**(A)探 steep-acc workload(TextVQA/OCR/doc)找赢家｜(B)GPU确认GQA NO-GO｜(C)重定位为'acc(k)陡度门控'刻画并入v2｜(D)放弃回v2论文。详见 DECISIONS.md + `runs/elasticvis_ev0/probe.py`。

## ★ 评测制度（已批准）
**open-loop 变载到达为主 + 混合-SLO 为辅**。现有 c64 闭环准入负载≈常数（v2 逐段控制器 n=500 null 的原因）→ 不是正确评测。H1 赢点=变载；H1b=混合 deadline。baseline：fixed-{r0,r25,r50,r75}+v2控制器+oracle。

## v2 资产（substrate，全现成）
framework `src/serve_bench.py`(V1 c64 goodput)+`src/load_controller.py`(逐段→ElasticVis 逐请求后继)+`src/compressors.py`；数据 `runs/v2_p{0..3}/`；模型 LLaVA-1.5-7B + Qwen3-VL-8B；env `qwen3vl_clean`(V1)。v2 论文 `drafts/paper_v2.md`(9表5图47ref) 可独立投（fallback/伴生）。

## 已完成（背景）
P0-P4 全完成（lit+定位｜probe｜selector 三连败→proxy 天花板｜v2 实验(2引擎×2架构×4压缩器+c64+goodput)｜v2 论文+图）。ElasticVis 是 P5 方法转向。详见 DECISIONS.md。

## 关键约束
- 算力 1× A40 46GB 串行（c64 是天花板）；env `qwen3vl_clean`(V1)。
- 提交以用户本人名义，禁 AI 署名。每步前 web 核实版本+novelty 监控（关键词见 design §0：elastic visual-token / per-query resolution / SLO-customized token budget + VLM）。
- 升级找人：凭据 / >6GPU·h / claim 推翻 / 投稿前。

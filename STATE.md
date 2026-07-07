# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 详见 **ORCHESTRATION.md** + **notes/elasticvis_design.md**（spine）+ **notes/elasticvis_positioning.md**
> 最近更新：2026-07-06 · **新主对话从 notes/elasticvis_design.md 读起**

## ★ 当前方法（ElasticVis）— 定位已重定位（2026-07-06 novelty 扫描）
**准入时、由实时负载+SLO 裕度驱动、优化 goodput@SLO 的 per-request 视觉 token 分配器**，坐在 elastic 压缩器（PARCEL-like）之上，非压缩器。
**❗"0/N 做 per-request 预算"claim 已 RETIRE**：CARES/DyToS/PARCEL 已做 per-request 视觉预算（内容驱动/无系统信号），AdaServe/SLOs-Serve 做了 per-request budget-for-goodput（文本域）。**干净单元 = 系统信号驱动（H1）**。详见 `notes/elasticvis_design.md` §0。

**已批准 spine：A→B**（用户 2026-07-06）。核心 H1 系统信号 allocator（现有数据够），内容维度 H2 作后续扩展（对标 CARES）。

## ★ 立即进行：EV-0 GO on TextVQA（H1b mixed-SLO +35.5%，零 GPU sim 确认）
**门控刻画被 5 benchmark 验证**：ElasticVis 的 goodput 收益被 accuracy(k) 陡度门控。知识型(MME/MMBench/ScienceQA)~0.01→无 win；GQA 0.12-0.13→边界 NO-GO；**TextVQA 0.28-0.29→WIN**。synthetic sweep：H1b(混合-SLO) crossover≈0.15，H1(均匀-SLO)≈0.40 → **混合-SLO 是稳健 regime**。**真实 TextVQA sim：H1b mixed-SLO Greedy 2.36 vs bestFixed 1.74 = +35.5% WIN**；H1 0.898 lose；GQA H1b 0.978 lose。机制=紧 deadline 给低 k、松给高 k。详见 design §8 + `runs/elasticvis_ev0/{gating_sweep,confirm_textvqa}.py`。
## ★ 立即进行：ElasticVis ❌ 终判 negative（EV-1e 公平 GPU 测试输）→ 收尾方向待 user 定
EV-1e（recalibrated load-dependent allocator + EV/fixed 同引擎设置，公平）：open-loop TextVQA c64 mixed-SLO，rate8 0.991×(tied)、rate12 0.885×输、rate15 0.757×输。EV-1d 的 +7.5% 是不公平 artifact（enforce_eager 不对称+未校准）。**根因 fundamental：continuous batching 同 forward 共享 GPU compute，k576 长 prefill 延迟整批 decode（含紧 deadline）→ 给 slack 高 k 拖慢整批→紧请求 miss SLO；sim 独立 slot 模型不捕获此 batch 干扰→+35.5% 不迁移。结论：per-request 视觉 token 预算在连续批处理下不提升 goodput@SLO（架构性）。** 已得成果：门控刻画(5 benchmark sim)、robust placeholder-shrink 集成(per-request k 在 c64 工作,机制证)、GPU 实测 accuracy(k)+fixed-r baselines、batch-interference 这个新发现。v2 测量论文(drafts/paper_v2.md,9表5图47ref)仍可独立投。**待 user 定**：A 折入 v2 论文 negative-finding 节→投 v2｜B 独立 negative-result 论文｜C pivot SLO-aware batching(按 deadline 分批避免干扰,新方向)。

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

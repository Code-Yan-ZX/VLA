# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 详见 **ORCHESTRATION.md** + **next_plan.md**
> 最近更新：2026-07-03

## 当前阶段
**v2 实验阶段(P0-P3)完成 → P4 全文重写。**

## ★ v2 实验全景（系统拆解 v1 全部致命伤）
- **P0 V1 引擎**（`qwen3vl_clean`=vllm0.19.0+cu128, multiproc=0 in-process 保 V1 scheduler）：F1 在 V1 成立且更强，c12/r75 **1.86×**（V0 1.75×）。§4.3 V1-migration-as-contribution。
- **P1 Qwen3-VL-8B**：F1/F2/F3 **architecture-conditional**（F1 衰减 c12/r75 1.29×、F3 成立、F2 merger 高效仅 10% TTFT）。原生 2×2 merger 与 post-hoc pruner **替代**→边际递减。测量跨 2 引擎×2 架构。
- **P2 真实规模**（c64 + p50/p99 + goodput）：r75/r0 1.19×(c1)→**2.22×(c64)** 未饱和；**压缩抬高硬件吞吐天花板**；**r75@c64 严格 dominate r0**（2.22×吞吐+2.84×更低 p99，无 tradeoff 纯赚）；goodput@TTFT≤5s r75 13.7 vs r0 1.8(7.4×)。
- **P3 跨压缩器**（proxy/cls/ToMe-merge/random @c64）：**goodput-Pareto win 4/4 通用**；throughput iso-k 下与 selector 无关 = **FRAMEWORK 属性**（拆"只 proxy"+"0/37 脆弱"）；**prune-vs-merge tradeoff**（ToMe 补回 55% 掉点仅 −1.5% 吞吐）；诚实 red flag: random>r75 saliency（GQA FastV 失败模式）。
- **novelty 仍成立(0/N)**：RTP-LLM 是引擎非压缩机；vLLM-Omni/DeepSeek-OCR 不威胁。补 EarlyTom/ADSC。

## 立即下一步 —— P4 全文重写（`drafts/paper_v2.md`）
新 framing：**served-throughput measurement FRAMEWORK**（非"0/37 + 3 发现"），跨 2 引擎×2 架构×4 压缩器×生产规模(c64/goodput)验证。
- 主贡献：framework + 部署级 goodput-Pareto（pure-win headline）+ framework-generality(4/4) + architecture-conditional + prune-vs-merge + "lifts ceiling"。
- §4.3 V1-migration(multiproc=0)；§2.3 gap 改写（DeepSeek-OCR/EarlyTom/vLLM-Omni）；加 Qwen3-VL 列 + c64/goodput/跨压缩器表图。
- 方法(load-adaptive)降为 instantiation；诚实。
源：notes/v2_{p0,p1,p2,p3,ecosystem_assessment}.md + eval/final_results.md + drafts/paper_v1.md。

## P4 后
图重渲染（c64 goodput Pareto、跨压缩器、Qwen3-VL 列）→ nature-citation/polishing → **P5 投稿前强制升级找人**。

## 关键约束
- 算力 1× A40 46GB 串行；v2 serving env=`qwen3vl_clean`(V1)；vtc_serve(V0)/fastv 留存。
- 提交以用户本人名义，禁 AI 署名。每步前 web 核实版本+novelty 监控。
- 升级找人：凭据 / >6GPU·h / claim 推翻 / 投稿前。

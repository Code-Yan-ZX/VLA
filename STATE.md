# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 详见 **ORCHESTRATION.md** + **next_plan.md** + **notes/elasticvis_positioning.md**
> 最近更新：2026-07-03 · **新主对话从此读起**

## ★ 当前核心目标（方法转向）
**ElasticVis（工作名）/ TokenSched：把"每请求视觉 token 预算分配"做成 serving 方法。** 所有 37 法（含我们 v2 控制器）都用**全局**剪枝率；ElasticVis 让 serving 引擎在**准入时按请求**分配 k_i，目标直接 **goodput@SLO**。形式化：在线分配 `max Σ accuracy(k_i) s.t. LatencyPred(Σk_i, load) ≤ SLO`。**0/37 做逐请求预算** → 新意干净。详见 `notes/elasticvis_positioning.md`（必读：形式化、组件、第一步、资产）。

**为何是对的方向**：绕开饱和的"选哪些 token"（3 selector 败、proxy 是 TF 天花板、native merger 替代 post-hoc），换成"每请求给多少 token"——scheduling 问题，引擎有独特杠杆（负载/SLO/请求特征）。直接复用 v2 全部资产。

## ★ v2 资产（ElasticVis 的 substrate，全部现成）
- **测量 framework**（V1 引擎 c1-c64 + goodput@SLO + p50/p99）：`src/serve_bench.py`、`src/load_controller.py`（**逐段控制器 → ElasticVis 逐请求后继**，已带 `get_metrics()` V1 信号）。
- **accuracy(k) 曲线 + latency(k,load) 数据**：`runs/v2_p{0..3}/` + `notes/v2_p{0..3}_*.md`（ElasticVis 的目标项 + 约束预测器输入，现成）。
- **跨 2 引擎×2 架构×4 压缩器**：LLaVA-1.5-7B（`runs/models/`）+ Qwen3-VL-8B（HF cache）；env `qwen3vl_clean`(V1)。
- **v2 论文**（`drafts/paper_v2.md` + `drafts/figures/v2/`，9 表 5 图 47 refs）**可独立投稿**（measurement-led）——作 fallback/伴生；后续定是否把 ElasticVis 折成方法节。

## 立即第一步（新主对话）
1. 形式化在线分配目标（objective/constraint/signals）。
2. 建 **latency predictor** `LatencyPred(Σk_i, load)`（拟合 v2 并发×prune×延迟数据）。
3. 建 **per-request accuracy(k) 模型**（probe 数据，按请求特征分桶）。
4. 实现准入时 allocator（greedy/Lagrangian）→ V1 processor-level placeholder-shrink 集成。
5. 验证：ElasticVis vs fixed-{r0..r75} vs v2 逐段控制器，on **goodput@SLO @ c64**（LLaVA-1.5 + Qwen3-VL）。claim：ElasticVis > 任何 fixed rate。

## 已完成（背景）
P0-P1 lit+定位｜P2 probe(gate 过)+selector 三连败→proxy 天花板｜P3 v2 实验全完成(V1+Qwen3-VL+c64+goodput+跨压缩器+architecture-conditional)｜P4 v2 论文+图。novelty 0/N 成立。详见 DECISIONS.md。

## 关键约束
- 算力 1× A40 46GB 串行（c64 是天花板）；env `qwen3vl_clean`(V1)。
- 提交以用户本人名义，禁 AI 署名。每步前 web 核实版本+novelty 监控。
- 升级找人：凭据 / >6GPU·h / claim 推翻 / 投稿前。

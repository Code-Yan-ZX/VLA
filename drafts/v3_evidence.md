# V3 Pre-merger Pruning — Consolidated Evidence（论文 results 骨架）

> Spine（user 定 A，发现主导）：lossy-merger 机制 + workload-conditional stage law + post-merger SOTA(VisionZip类)在 text-dense 深压灾难崩溃、pre-merger 鲁棒修复。方法 = pre-merger pruning（空 cell）+ adaptive stage selection（ptid、workload 级）实用节。
> Model: Qwen3-VL-8B-Instruct, bf16, enforce_eager, 1×A40。Selector: L2-norm text-agnostic（控变量隔离 stage 效应）。updated 2026-07-21。

## 1. Headline — text-dense 深压 pre 大胜（iso-token, keep 25%）

| Benchmark | n | pre-merger | post-merger (L2) | VisionZip-style (dom+ctx) | gap pre−post | 显著性 |
|---|---|---|---|---|---|---|
| TextVQA | 500 | **0.738 ± 0.020** | 0.272 ± 0.020 | 0.390 (n=200) | **+46.6 pp** | z=16.7σ |
| DocVQA  | 200 | **0.725 ± 0.032** | 0.390 ± 0.034 | 0.390 (n=200) | **+33.5 pp** | z=7.2σ |

- **VisionZip-style(dom+ctx) 与 post dom-only 在 DocVQA 上同为 0.390**（iso-token ptid=1054）→ context tokens 在 text-dense 上零增益（avg-pool 同样毁文字）。两类 post-merger 方案同样灾难崩溃，pre 独保。
- pre−VisionZip：TextVQA +34.8pp / DocVQA +33.5pp。
- 误差棒 = binomial stderr √(p(1−p)/n)；gap stderr = √(se_pre²+se_post²)。greedy(temp=0) 确定性 → seed 无方差，用 binomial。

## 2. Workload-conditional stage law（@25% keep, n=200, pre−post gap）

| Benchmark | 类别 | baseline A | pre (C) | post (B) | gap (pp) |
|---|---|---|---|---|---|
| DocVQA (document OCR) | text-dense | 0.77 | 0.725 | 0.39 | **+33.5** |
| TextVQA (scene text) | text-dense | 0.82 | 0.695 | 0.255 | **+44.0** |
| ScienceQA (figure+text MC) | perception/MC | 0.725 | 0.365 | 0.37 | −0.5 |
| MMBench (perception MC) | perception/MC | 0.895 | 0.295 | 0.32 | −2.5 |
| MME (yes/no perception) | perception/MC | 0.885 | 0.815 | 0.82 | −0.5 |
| GQA (object/spatial) | object | 0.415 | 0.32 | 0.38 | **−6.0** |

- **三层结构**：text-dense（+33~44）≫ perception/MC（≈0，−0.5~−2.5）> object（−6）。
- 图：`drafts/figures/stage_law.png`。
- **诚实注记**：within-tier 小 inversion（DocVQA +33.5 < TextVQA +44；MME −0.5 > MMBench −2.5）。假设：L2 selector 在 DocVQA 4k-token 巨图上行为差异 / n=200 噪声。tier 结构稳健，单调性为 coarse（text-dense vs 非）。**写作时如实报，不overclaim 完美单调**。

## 3. 深度效应（越深压 pre 越胜）

- TextVQA：keep 50/25/12.5% → pre {0.75,0.70,0.62} vs post {0.51,0.26,0.18}，Δ{+24,+44,+44}pp。深点 pre 保 75% baseline vs post 21%（**3.5× retention**）。
- DocVQA（mnbt 修复后 clean）：@25% pre 0.725 vs post 0.39（+33.5）；@12.5% pre 0.61 vs post 0.135（**+47.5**）。
- 注：suite_map 的 DocVQA @12.5% post=0.0 是 mnbt 修复**前**的崩溃值（弃用）；clean 深度对比用 Task1 修复后数字。

## 4. 机制（lossy merger）

- 原生 2×2 merger 是 lossy 聚合，破坏 text 高频细节 → post-merger 在已退化特征上选 token → text-dense 灾难掉点。
- pre-merger 在 raw patch 上选 → 保 text；object 任务上 merger 的 learned 聚合 helpful → post 胜。
- throughput pre≈post（merger 仅 ~10% TTFT，F2）→ pre 优势是**精度**（深压 text-dense），非吞吐。

## 5. Adaptive stage selection（Task4，实用方法节）

- 离线全梯度路由（4 bench × pre/post per-sample，id 对齐已核、oracle 复现）。pooled(N=774)：always-pre 0.634 / always-post 0.452 / oracle 0.702 / 廉价 router(ptid≥94) **0.655**（胜两固定、距 oracle 4.65pp）。
- **分解**：workload 级仅占 oracle 增益 **27%**、sample 级 **73%**（query 依赖，ptid 等图像级廉价信号够不到）。OCR 关键词路由更差(0.539)；有效信号 = **ptid（图像 token 数）**。
- **实用结论**：pre 是鲁棒安全默认（从不崩）；用 ptid 廉价信号检测非文字 workload → 切 post 补回 object gap（+~2pp on balanced mix）。per-sample 精细路由 headroom 有限（query 依赖），如实报。
- 代码：`src/v3_premerger/router_probe_full.py`。

## 6. 已有 negative / 对照
- selector 三连败（CLS/LLM-cosine/CLIP）｜load-adaptive controller null｜ElasticVis 负 → boundary TF selector / scheduling 维度被 total-token bound。pre-merger = 正交新维度。
- VisionZip 经 GitHub 源码逐行核实为 POST-merger（保留 native merger）→ pre-merger×native-merger cell 真空（novelty 成立）。
- 跨架构 Qwen2.5-VL：mrope 错位未解，搁置（claim scope 限 Qwen3-VL，跨架构作 future work）。

## 待补（写论文前）
- [ ] DocVQA n=500（可选；n=200 已 7σ，非必需）
- [ ] 更多 text-dense（OCR-Bench/ChartQA）强化 law（可选，强化 tier 结构）
- [ ] baseline A 同 iso-config 复核（部分 A 为旧 config）
- [ ] 论文撰写（intro/method/results/discussion）—— 可调用 nature-writing/nature-figure skills

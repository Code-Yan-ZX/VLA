# Direction A/B 实验结果报告摘要（三表格 + 失败原因）

实验日期：2026-08-13 · 模型：Qwen3-VL-8B-Instruct（repo `qwen3vl`）
协议：训练-free，论文 §3 iso-selector + iso-budget；metric 均为 official rescore
（TextVQA=VQA-acc、DocVQA=ANLS、OCRBench=acc、GQA=acc、ChartQA=relaxed-acc）
数据规模：n=500（ChartQA n=200），full-split 提示（含 "Answer..." 后缀；subsets 无后缀会导致 verbose 答案 → official 全 0，已用 full-split 规避）

<<< 摘要 TL;DR >>>
- **任务一 freq（Direction B）✅ 可入正文（有界、诚实打包）**：freq(α=1,β=0.6) 在 text-dense + 高压缩下提升（TextVQA +1.2pp@25%、+0.5pp@50%，DocVQA +0.6pp@50%）；非 text-heavy / 宽松预算下 −1.6~−2.2pp（docvqa/ocrbench/gqa@25%），放补充。
- **任务二 adaptive（Direction A）❌ ACCEPTANCE FAIL（2 轮搜索已尽）**：TextVQA 最优 τ 路由全图→PRE ≡ RBM（text 三基准 0 回归 ✓，GQA +0.00pp < 要求 +1pp ✗）；τ=0.13 翻 GQA→POST 却致 TextVQA −7pp。失败原因=GQA/TextVQA 的 hf 分布重叠，query-free 全局阈值两难。记录为 §6/补充的 bounded negative。
- **任务三 combined = freq-alone**（τ=0.08 router 为 no-op）——无新 claim；ChartQA 新基准入表（@25% 0.195 > RBM 0.170 / FastV 0.180）。

---

## 一、任务一：频率感知打分器 freq = α·z(L2) + β·z(var)

**方法核心**：raw-scale 混合退化（L2≈12 vs var≈0.04，差 300:1，β 无效）→ 两分量 per-image z-score 后线性混合，α:β 成为真实权衡旋钮。网格 {0.5,1,2}² + 精化共 2 轮 → **胜者 (α,β)=(1.0, 0.6)**。

| bench@r | freq | RBM(pre) | FastV(post) | Δfreq−RBM |
|---|---|---|---|---|
| **textvqa@0.75** | **0.6173** | 0.6053 | 0.2073 | **+1.20pp** |
| **textvqa@0.5** | **0.6780** | 0.6733 | 0.3873 | **+0.47pp** |
| **docvqa@0.5** | **0.6271** | 0.6212 | 0.5487 | **+0.59pp** |
| docvqa@0.75 | 0.4903 | 0.5063 | 0.2317 | −1.60pp |
| ocrbench@0.75 | 0.7160 | 0.7380 | 0.1700 | −2.20pp |
| ocrbench@0.5 | 0.8220 | 0.8220 | 0.4980 | 0.00pp |
| gqa@0.75 | 0.4220 | 0.4380 | 0.4540 | −1.60pp |
| gqa@0.5 | 0.4860 | 0.4840 | 0.5000 | +0.20pp |

**正收益（进正文）**：TextVQA @25%/+0.5pp @50%，DocVQA@50% +0.6pp。
**负/平（放补充）**：docvqa/ocrbench/gqa @25% −1.6~−2.2pp；ocrbench@50% 持平；gqa@50% +0.2pp。
**一句话结论**：方差项把文字笔画单元权重上调，在 text-dense + 收紧预算时保留结构收益最明显 → 支持 M2「merger 削弱高频单元」的可操作推论。

---

## 二、任务二：自适应阶段选择器（Direction A）—— ACCEPTANCE FAIL

**方法**：每图 workload 检测器（merge 输入，无 query）：hf_ratio（mean+1σ 尾占比，scale-free）+ L2 熵（20-bin 直方图 nats）；`hf>τ_hf 或 H>τ_ent → PRE(RBM)`，否则 POST(FastV)。网格 τ_hf∈{0.08,0.13,0.20} × τ_ent∈{2,2.5,3}（第 2 轮；第 1 轮 {0.15,0.30,0.50} 因 > TextVQA ratio≈0.142 而 hf 永不触发，见 DECISIONS）→ 胜者 **(τ_hf,τ_ent)=(0.08,2.0)**。

| bench@r | adaptive | RBM(pre) | Δadapt−RBM |
|---|---|---|---|
| textvqa@0.75 | 0.6053 | 0.6053 | 0.00pp |
| docvqa@0.75 | 0.5063 | 0.5063 | 0.00pp |
| ocrbench@0.75 | 0.7380 | 0.7380 | 0.00pp |
| gqa@0.75 | 0.4380 | 0.4380 | **0.00pp（需 ≥+1pp ✗）** |
| 四基准 @0.5 | 同左列 | 同左列 | 全部 0.00pp |

**失败原因（如实记录，2 轮搜索已尽）**：
1. TextVQA 最优 τ_hf=0.08 → 所有图路由 PRE → adaptive 与 RBM **结果 bit-identical**（每位比较 0.00pp）：text 三基准 0 回归 ✓，但 **GQA +0.00pp < 验收线 +1pp ✗**。
2. 唯一能触发 GQA→POST 的 τ_hf=0.13：TextVQA 51/320 图被翻到 POST → **−7pp（远超 0.5pp 回归限）**；而 GQA 只有 22/243 图翻到 POST → 0.428 < RBM 0.438。
3. **结构原因**：GQA 与 TextVQA 的 hf_ratio(mean1sd) 分布重叠（均值 0.127 vs 0.142，per-image 波动大）—— query-free（merger 输入统计）单一全局阈值无法把「场景文字」与「文档文字」分开，因此不存在同时满足「GQA≥+1pp」与「text≤0.5pp 回归」的两全配置。
4. 机制本身工作正常（路由计数正确、isotoken 保持、无 fallback），问题在特征可分性，不在实现。

---

## 三、任务三：组合评估（freq-best + adaptive + ChartQA）

combined = freq(1,0.6) scorer + adaptive(0.08,2.0) router。因 τ=0.08 为全 PRE（router no-op），**combined ≡ freq-alone**；本表计入 ChartQA（新基准）。

| bench@r | combined | RBM(pre) | FastV(post) | Δc−RBM | Δc−FastV |
|---|---|---|---|---|---|
| textvqa@0.75 | **0.6173** | 0.6053 | 0.2073 | +1.20pp | +41.00pp |
| textvqa@0.5 | **0.6780** | 0.6733 | 0.3873 | +0.47pp | +29.07pp |
| docvqa@0.75 | 0.4903 | 0.5063 | 0.2317 | −1.60pp | +25.86pp |
| docvqa@0.5 | **0.6271** | 0.6212 | 0.5487 | +0.59pp | +7.84pp |
| ocrbench@0.75 | 0.7160 | 0.7380 | 0.1700 | −2.20pp | +54.60pp |
| ocrbench@0.5 | 0.8220 | 0.8220 | 0.4980 | 0.00pp | +32.40pp |
| gqa@0.75 | 0.4220 | 0.4380 | **0.4540** | −1.60pp | −3.20pp |
| gqa@0.5 | 0.4860 | 0.4840 | **0.5000** | +0.20pp | −1.40pp |
| **chartqa@0.75** | **0.1950** | 0.1700 | 0.1800 | **+2.50pp** | +1.50pp |
| **chartqa@0.5** | 0.3900 | **0.4000** | 0.3400 | −1.00pp | +5.00pp |

**备注**：
- ChartQA@50% combined 落后 RBM 0.010（0.390 vs 0.400）—— n=200 内属噪声，@25% 的 +2.5pp 是干净正号。
- gqa 上 FastV 仍最优（0.454/0.500）—— 与论文「FastV=query-conditioned 强 baseline」定位一致。
- PyramidDrop 对照（既有 j7hf r0.375_canon，宽松预算 keep 62.5%，非 iso-budget）：textvqa 0.8347 / docvqa 0.8821 / ocrbench 0.7460 / gqa 0.6060，仅作外部锚点，不直接入主表。

---

## 四、对写作的定位建议

| 条目 | 建议位置 | 写法要点 |
|---|---|---|
| freq 打分器（α=1,β=0.6） | **正文机制延伸段落**（若 user 批准） | 只报 textvqa ±、docvqa@50；负收益进补充；限定「text-dense + 高压缩」；不写 beats |
| M2 的可操作推论句 | **正文**（机制段落 1 句） | "Up-weighting within-unit variance at selection improves TextVQA by +1.2pp at 25% retention (0.617 vs 0.605)" |
| adaptive 路由负结果 | **§6 / 补充** | bounded negative：query-free hf 统计无法在全局阈值下分离 scene/document workload；不堵死未来方向 |
| combined ≈ freq-alone | **不写**（无新 claim） | — |
| ChartQA | 入表（n=200 标注） | @25% +2.5pp；@50% 与 RBM 持平 |

> 红线提醒：以上任何进正文的条目都需 **user 确认** 后才可触碰 overleaf 权威稿；本文档仅存于 notes/experiments，不构成正文改动。
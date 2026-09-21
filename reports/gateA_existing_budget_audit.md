# Gate A — 已有预算(Retention)结果审计

> 状态：**完成**（2026-09-21）
> 分支 `exp/rbm-multibudget-validation`，基 commit `460c7d2`
> 目的：证伪/证实"已有 50% retention 实验且结果明显变弱"，盘点 12.5/25/50/75% 的现成证据。

## 0. 命名约定（全仓审计后统一，重要）

两个 harness 的 `--r` 参数语义**相反**：
- **vLLM runner**（`v3_premerger_runner.py`）与 **j7/j8 结果文件**：`r` = **丢弃比例(prune ratio)**，retention = `round(n*(1-r))`。即 `r=0.75`→keep 25%，`r=0.50`→keep 50%，`r=0.875`→keep 12.5%。
- **HF harness**（`baselines_hf.py`）：`--r-pre` = **保留比例(keep ratio)**；FastV 的 `--r` 仍为丢弃比。
- FastV 仅存在于 HF harness（`baselines_hf.py`）；vLLM runner 无 FastV。

| keep | vLLM `--r`(丢弃) | HF pre `--r-pre`(keep) | HF FastV `--r`(丢弃) |
|---|---|---|---|
| 50% | 0.5 | 0.5 | 0.5 |
| 25% | 0.75 | 0.25 | 0.75 |
| 12.5% | 0.875 | 0.125 | 0.875 |

> 本报告除标注外，数值均取自 **vLLM backend + 官方评分器(official_scorers.py)**，j7 为 full-split，j8 为 n=200 ablation。

## 1. 权威锚点：Qwen3-VL-8B，vLLM full-split，官方评分（`runs/full_matrix/j7_main_table.json`）

| bench | none | **RBM pre** @keep25% (r=.75) | **Post-L2** @keep25% | RBM pre @keep12.5% (r=.875) | Post-L2 @12.5% | mean_tokens(pre25%/12.5%) |
|---|---|---|---|---|---|---|
| TextVQA | 0.844 | **0.6053** | 0.2217 | 0.4718 | 0.1318 | 215.8 / 122.1 |
| DocVQA | 0.9562 | **0.4806** | 0.2377 | 0.3522 | 0.1034 | 946.7 / 488.8 |
| OCRBench | 0.760 | **0.547** | 0.184 | 0.350 | 0.053 | 228.9 / 125.7 |
| GQA | 0.6165 | **0.4488** | 0.4771 | — | — | 96.8 / — |

pre − post @25%：TV **+38.4**，DV **+24.3**，OCR **+36.3**，GQA **−2.8**（GQA 例外，post 更好）。

## 2. 50% / 75% 留存：仅 n=200 ablation（`runs/full_matrix/ablations/j8_summary.json`）

Qwen3-VL，L2 selector，官方评分。`r` 为丢弃比（r=0.25→keep75%，r=0.50→keep50%，r=0.75→keep25%）。

| keep | pre TV | post TV | **Δ TV** | pre DV | post DV | **Δ DV** |
|---|---|---|---|---|---|---|
| 25% (r=.75) | 0.5767(attn)/0.6053(full) | 0.2217(full) | +38.4 | 0.451(attn)/0.4806(full) | 0.2377(full) | +24.3 |
| 50% (r=.50) | **0.67** | **0.38** | **+29.0** | **0.5892** | **0.5307** | **+5.9** |
| 75% (r=.25) | **0.74** | **0.6533** | **+8.7** | **0.6865** | **0.6998** | **−1.3** |

（n=200 值；50%/75% 无 full-split。75% 的 +8.7/−1.3 在 n=200 下噪声较大，但方向性衰减清晰。）
GQA keep50%：`runs/v3_premerger_cells/C_gqa_r0.500.json` raw acc=0.38（非 official，per_sample 空，不作为严格证据）。

## 3. 核心发现 A：**pre 优势随 retention 单调衰减（H1 初步支持）**

| bench | keep 25% | keep 50% | keep 75% |
|---|---|---|---|
| TextVQA Δ(pre−post) | **+38.4** (full) | **+29.0** (n200) | **+8.7** (n200) |
| DocVQA Δ(pre−post) | **+24.3** (full) | **+5.9** (n200) | **−1.3** (n200) |

- RBM pre 的优势在 **aggressive compression（25% 及以下）最大**，随 retention 上升单调缩小。
- DocVQA 在 keep 75% 时 Post **反超**（−1.3 pp），提示存在边际拐点。
- 这一模式与"RBM 主要在 aggressive compression 下有效"的 H1 假说**方向一致**，但 50%/75% 仅 n=200，需 Gate B full-split 确认。

## 4. 用户疑点核实："50% 是否明显变弱？"

**取决于对比对象：**
- "50% 比 25% 弱"？ **错误**。50% retention 绝对分数更高（TV pre 0.67 vs 0.6053 full；DV pre 0.5892 vs 0.4806）——更多 token 更好，符合直觉。
- "50% 下 RBM 相对 Post-L2 的优势消失/缩到很弱"？ **部分正确**。Δ(pre−post) 从 25% 的 +38(TV)/+24(DV) 缩到 50% 的 +29(TV)/+5.9(DV)，DocVQA 优势已接近噪声。
- "50% 下 RBM 大幅低于 strong baseline"？ **未在同 harness 下测过 FastV@50%**（FastV 无 r=0.5 记录）。但 HF TextVQA @25% 下 FastV 已 +1.20pp 超 RBM（DCC expB），50% 下 FastV 优势很可能更大（FastV 是 query-conditioned，retention 高时保留更准）。

## 5. 既有结论速览（其他报告）

- **DCC expB**：HF TextVQA full n=5000 @25%：FastV-k3 0.8494 vs RBM-pre 0.8374，Δ=+1.20pp，CI[−2.18,−0.22]，p=0.018。**FastV 已显著略优。**
- **acmmm controls**：post/pre 阶段控制已大量披露；核心 headline pre 仍高于一切 post 变体（但这是方法内对比，非 vs 强 baseline）。
- 未有 RBM vs FastV 在 12.5% 的同 harness 数据。

## 6. Case 判定（Gate A 停止点）

**Case B（接近 Case A 边界）：50% 下 RBM 仍 ≥ Post-L2，但优势大幅缩小。**

- TV：keep50% pre 0.67 > post 0.38，仍占优（Δ+29）。
- DV：keep50% pre 0.5892 > post 0.5307，占优但 Δ 仅 +5.9（近噪声）。
- 但 keep75% 时 DV post 反超，且 strong baseline(FastV)@25% 已超 RBM。
- 判定依据的 n=200 数据需 Gate B 用 full-split(或固定同 n 子集)确认。

## 7. 结论与下一步（Gate B）

现有证据不支持"50% RBM 明显弱于 Post-L2"；但支持"**RBM 优势集中于 aggressive compression（≤25%）**"。50%/12.5%/25% 的 full-split multi-budget 曲线**尚不存在**（50% 只有 n=200），是 Gate B 必须产出的核心资产。

Gate B 计划：vLLM runner（Qwen3-VL-8B），L2 selector，`--r ∈ {0.875, 0.75, 0.5}`（keep 12.5/25/50%），pre+post 双臂，4 benchmark，全 reuse 25% 的 j7 full 结果，补跑 50%/12.5% 的缺失格子。FastV 对照推迟到 HF harness（Gate E）。

## 产物路径
- j7 主表：`runs/full_matrix/j7_main_table.json`
- j8 ablation：`runs/full_matrix/ablations/j8_summary.json`
- DCC expB：`reports/dcc_presubmit_expB.md` + `results/dcc_presubmit/expB/`
- 本报告：`reports/gateA_existing_budget_audit.md`
# MALT Adaptive-Gate Feasibility Audit（离线可预测性审计）

- **日期**：2026-08-24
- **分支**：`exp/deferred-rbm-n200`
- **上游**：`experiments/deferred_rbm_lifetime_sweep.md`（scenario A，建议进入 adaptive-gate 设计）
- **范围**：只做离线 transition audit + 特征可预测性审计 + policy simulation；**未实现在线 MALT-A，未修改论文正文**。
- **数据/代码**：`experiments/malt_adaptive_gate_feasibility/`（transitions.json / features_*.json / audit.json）；`scripts/audit_malt_transitions.py` / `scripts/extract_malt_features.py` / `scripts/audit_malt_gates.py` / `scripts/audit_malt_summary.py`。
- **Git**：commit（见本文件末尾）。

---

## 0. 结论速览（executive summary）

| 问题 | 结论 |
| --- | --- |
| K0/K1/K8 逐样本转移结构 | 非单调：存在 K0C/K1W（2–16）与 K1C/K8W（2–5）反例；K0→K1 是主跃迁（G0 侧），K1→K8 挽救极少 |
| G0（K0 安全删除）是否可预测 | **基本不可靠**：3/4 数据集 AUROC≈0.47–0.59（接近随机），仅 OCRBench ~0.72–0.77 可学习 |
| G1（K1→K8 延长）是否可预测 | **不可靠（统计上不可用）**：正样本仅 2–10/数据集，rescue precision≈0 |
| adaptive policy 是否通过验收 | **NO-GO**：全部 3 类模型、全部 operating point 均失败 |
| 是否冻结 fixed K=1 | **是**：冻结 fixed K=1 为正式 MALT 方法（MALT-1）；不实现 MALT-A |

**一句话**：基于推理内自然信号（pre-merger L2 统计 / layer-1 注意力动态）无法在无泄漏评估下可靠预测 K0/K1/K8 路由；adaptive policy 在每个 operating point 上都以算力换不来准确率，甚至损失准确率。**停止 adaptive gate，正式方法 = fixed K=1（MALT-1）。**

---

## 1. 转移矩阵（no-GPU，官方 rescore，K0∩K1∩K8 公共样本 = 781）

数据：`sweep_per_sample.json` 的 `correct`（官方 rescore，见 `analyze_deferred_sweep.py::official_per_sample`）。

> **metric 口径说明**：lifetime-sweep 报告 §6.1 的 docvqa 数字是 **mean ANLS**（连续值，docvqa K8=0.6212）。本审计所有 accuracy 一律用 **binary correctness（metric≥0.5）**：docvqa K8 binary=0.690。两套数字不可混用。

| bench | n | K0C,K1C | K0C,K1W | K0W,K1C | K0W,K1W | K1C,K8C | K1C,K8W | K1W,K8C | K1W,K8W |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| textvqa | 200 | 114 | 4 | 34 | 48 | 146 | 2 | 5 | 47 |
| docvqa | 200 | 89 | 5 | 42 | 64 | 128 | 3 | 10 | 59 |
| ocrbench | 181 | 97 | 8 | 18 | 58 | 112 | 3 | 3 | 63 |
| gqa | 200 | 67 | 16 | 41 | 76 | 103 | 5 | 2 | 90 |

**G0 类平衡**（pos=K0 correct；high-risk neg=K0W&K1C；neutral=K0W&K1W 训练剔除）：

| bench | pos | neg | neutral | pos/neg |
| --- | ---: | ---: | ---: | ---: |
| textvqa | 118 | 34 | 48 | 3.47 |
| docvqa | 94 | 42 | 64 | 2.24 |
| ocrbench | 105 | 18 | 58 | 5.83 |
| gqa | 83 | 41 | 76 | 2.02 |

**G1 类平衡**（pos=K1W&K8C rescue；neg=其余；harm=K1C&K8W 单独统计）：

| bench | pos | neg | harm | pos/neg |
| --- | ---: | ---: | ---: | ---: |
| textvqa | 5 | 195 | 2 | 0.026 |
| docvqa | 10 | 190 | 3 | 0.053 |
| ocrbench | 3 | 178 | 3 | 0.017 |
| gqa | 2 | 198 | 5 | 0.010 |

> **⚠️ G1 统计不可靠（必须声明）**：G1 正样本每数据集仅 **2–10 个**（共 20/781 ≈ 2.6%）。K1→K8 的 oracle 可挽救上限 = docvqa 5pp / textvqa 2.5pp / ocrbench 1.7pp / gqa 1pp（binary），远小于 G0 侧（K0→K1 主跃迁 12.5pp macro）。任何 G1 预测器（无论特征多好）都无法在 2–10 个正样本上获得可靠统计；下述 G1 数字仅作完整性报告，不作决策依据。
>
> **G0 的"正确路由无害样本"**：neutral（K0W&K1W）路由到 K0 或 K1 都保持错误，对 accuracy 无影响。G0 分类器只用 pos∪high-risk 训练；policy 中 neutral 路由到 K0 不计为 false-safe 也不计为 correct-route。

---

## 2. 特征定义与提取

### 2.1 G0（pre-decoder，K 无关；来自 merger 输入 / 捕获的 post-merge 行）
`n_img_log`（视觉 token 数）、`n_text_log`/`q_len_log`（文本 token / 问题长度）、`log_aspect`（grid 宽高比）、`l2_mean/std/cv/gini/entropy`（pre-merger 逐 unit L2 = mean per-patch L2 的分布统计）、`top25_mass`（top-25% 质量占比）、`topk_margin_rel`/`kth_gap_rel`（top-k vs 其余 margin / 阈值间隙）、`pre_post_jaccard`（pre-L2 与 post-merge L2 的 top-25% Jaccard）、`group_var_rel`（2×2 merge group 内方差）、`anchor_bbox_cov`（anchor 包围盒覆盖率）、`n_conn_comp`（anchor 8-连通分量数）、`anchor_ctx_cos_pre`（anchor/context 质心余弦）。

### 2.2 G1（layer-1 动态，K≥1 时 K 无关；零额外 forward）
`t2a/t2c/a2c_mass` + 跨 head std/max（text→anchor / text→context / anchor→context 注意力质量）、`ctx_entropy_all/img`（context 查询行注意力熵）、`text_top5_conc`（text 在图像 key 上的 top-5 集中度）、`attnout_norm_{anchor,ctx,text,all}`（layer-1 attention output = 加权 value 贡献的 log 范数）、`cos_anchor_01`/`cos_text_01`（layer0→1 hidden 余弦变化）、`sep_anchor_ctx_{0,1}`（layer0/1 anchor-context 表示分离度）。

> **FlashAttention 说明**：sweep 用 `attn_implementation="eager"`（`output_attentions` 可用）；layer-1 在 K=1 本来就是 prune 层、注意力已物化。本审计只做聚合统计、不保存完整矩阵，峰值内存与 sweep 相同（~18GB）。`attnout_norm_*` 需要 layer-1 attention output，由 self_attn hook 捕获。

### 2.3 提取不变性（强检查通过）
`extract_malt_features.py` 复用 runner 的 tested forward（rankbridge rho=1.0, keep_frac=0.25, fastv-k=1），仅加观察型 forward hooks。rho=1.0 → keep set = 纯 pre-merger L2 top-25%（与 K 无关，K=1 == K=8）。

| bench | n | skip | **keep-set == sweep anchor (100% 要求)** | answer == sweep K1 |
| --- | ---: | ---: | ---: | ---: |
| textvqa | 200 | 0 | **200/200** | 197/200 |
| docvqa | 200 | 0 | **200/200** | 191/200 |
| ocrbench | 181 | 19（=sweep 已知 OOM skip） | **181/181** | 168/181 |
| gqa | 200 | 0 | **200/200** | 198/200 |

### 2.4 已知 decode 非确定性（不影响审计）
27/781（3.5%）样本重生成答案与 sweep K=1 不同，但 **keep set 全部一致（781/781）**。逐 cell 比对证明是模型在这些边界样本上的固有答案不稳定（例：35084 时钟跨 cell 给出 '2:55'/'12:14'/'12:15'/'2:57'；34720 我的输出与 K0/K3 多数一致、sweep K=1 反而是离群）。有/无 hook 双跑复现一致，排除 hook 干扰。**特征全部来自 prefill/layer-1（与 decode 无关且 keep 验证一致）；标签一律取 sweep 官方 rescore**，故对审计无影响。

---

## 3. 审计协议

- **模型**：单特征 threshold / logistic（`class_weight='balanced'`）/ max_depth≤2 decision tree（`class_weight='balanced'`）。无 MLP、无 embedding 网络、无 VLM finetune。
- **标准化 / 特征选择 / 阈值选择**：一律只在训练 fold 内（`impute_train`/`std_train` 用 train 中位数/均值）。
- **评估 1**：每数据集内 **stratified 5-fold CV**（StratifiedKFold, seed 0）→ 严格 out-of-fold（每样本用其所在 fold 的阈值）。
- **评估 2**：leave-one-dataset-out（3 训练 / 1 测试，4 轮轮换）。
- **阈值规则**：G0 在 train 上 max recall 且 precision≥θ₀（默认 0.90）；G1 max rescue-recall 且 rescue-precision≥0.50（K8 昂贵，保守）。另做 θ₀∈{0.85,0.90,0.95,0.97} 敏感性扫描。
- **禁止项全部满足**：无 GT、无 correctness、无 K8 答案、无数据集 ID、无手工 benchmark 规则、无 full-model prediction、无测试泄漏。✓

---

## 4. G0 / G1 审计结果（5-fold OOF）

### 4.1 G0（路由到 K0；precision/recall/FSR 取 θ₀=0.90）

| bench | model | P | R | FSR(routed) | FSR(neg-cond) | **AUROC** | AUPRC | route% | routedK0acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| textvqa | logistic | 0.773 | 0.720 | 0.227 | 0.735 | **0.526** | 0.807 | 55.0 | 0.773 |
| | tree | 0.800 | 0.407 | 0.200 | 0.353 | **0.551** | 0.797 | 30.0 | 0.800 |
| | threshold | 0.809 | 0.322 | 0.191 | 0.265 | **0.586** | 0.835 | 23.5 | 0.809 |
| docvqa | logistic | 0.692 | 0.670 | 0.308 | 0.667 | **0.470** | 0.658 | 45.5 | 0.692 |
| | tree | 0.653 | 0.340 | 0.347 | 0.405 | **0.515** | 0.679 | 24.5 | 0.653 |
| | threshold | 0.864 | 0.202 | 0.136 | 0.071 | **0.585** | 0.763 | 11.0 | 0.864 |
| ocrbench | logistic | 0.896 | 0.981 | 0.104 | 0.667 | **0.737** | 0.908 | 63.5 | 0.896 |
| | tree | 0.929 | 0.752 | 0.071 | 0.333 | **0.719** | 0.922 | 47.0 | 0.929 |
| | threshold | 0.890 | 0.848 | 0.110 | 0.611 | **0.771** | 0.947 | 55.2 | 0.890 |
| gqa | logistic | 0.712 | 0.566 | 0.288 | 0.463 | **0.546** | 0.700 | 33.0 | 0.712 |
| | tree | 0.744 | 0.386 | 0.256 | 0.268 | **0.584** | 0.718 | 21.5 | 0.744 |
| | threshold | 0.636 | 0.084 | 0.364 | 0.098 | **0.592** | 0.715 | 5.5 | 0.636 |

**解读**：G0 在 textvqa/docvqa/gqa 上 AUROC 0.47–0.59（接近随机，signal 弱）；**仅 OCRBench 明显可学（0.72–0.77）**，且 FSR(routed) 低（0.07–0.11）、routedK0acc 高（0.89–0.93）。但把 OCRBench 大比例路由到 K0 会直接丢掉 K0→K1 的 +5.5pp（见 §5）。**G0 的"安全删除"信号在 3/4 数据集中基本不存在。**

### 4.2 G1（延长到 K8；θ₁ 保守阈值）

| bench | model | rescueP | rescueR | **AUROC** | AUPRC | route% | unnec% | harm% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| textvqa | logistic | 0.000 | 0.000 | 0.353 | 0.046 | 6.5 | 1.000 | 0.000 |
| | tree | 0.032 | 0.200 | 0.511 | 0.026 | 15.5 | 0.968 | 0.000 |
| | threshold | 0.000 | 0.000 | 0.686 | 0.045 | 11.5 | 1.000 | 0.043 |
| docvqa | logistic | 0.111 | 0.200 | 0.777 | 0.133 | 9.0 | 0.889 | 0.056 |
| | tree | 0.082 | 0.400 | 0.556 | 0.055 | 24.5 | 0.918 | 0.041 |
| | threshold | 0.143 | 0.100 | 0.796 | 0.175 | 3.5 | 0.857 | 0.143 |
| ocrbench | logistic | 0.000 | 0.000 | 0.356 | 0.016 | 4.4 | 1.000 | 0.125 |
| | tree | 0.000 | 0.000 | 0.444 | 0.017 | 11.0 | 1.000 | 0.000 |
| | threshold | 0.000 | 0.000 | 0.597 | 0.026 | 5.5 | 1.000 | 0.000 |
| gqa | logistic | 0.000 | 0.000 | 0.194 | 0.009 | 3.0 | 1.000 | 0.000 |
| | tree | 0.000 | 0.000 | 0.477 | 0.010 | 4.5 | 1.000 | 0.111 |
| | threshold | 0.000 | 0.000 | 0.294 | 0.011 | 19.0 | 1.000 | 0.053 |

**解读**：**G1 不可预测**。rescueP 几乎处处 ≈0（即便 docvqa logistic/threshold AUROC 0.78–0.80，rescued 样本也只有 1–2 个，AUPRC 0.13–0.18 极低）；unnecessary-extension ≥86%；harm% 0–14%。结合 §1 的 2–10 正样本，**G1 侧统计不可靠，不得作为决策依据**。

---

## 5. 离线三段式策略模拟（严格 OOF）

### 5.1 固定 baseline（binary correctness，K0∩K1∩K8 公共集）

| bench | K0 | K1 | K8 | oracle |
| --- | ---: | ---: | ---: | ---: |
| textvqa | 0.590 | 0.740 | 0.755 | 0.780 |
| docvqa | 0.470 | 0.655 | 0.690 | 0.720 |
| ocrbench | 0.580 | 0.635 | 0.635 | 0.685 |
| gqa | 0.415 | 0.540 | 0.525 | 0.625 |
| **macro** | **0.514** | **0.643** | **0.651** | **0.703** |

compute ΣN_l²：K0=5.63e8，K1=1.03e9，K8=2.68e9，oracle(earliest-correct)=6.91e8。

### 5.2 adaptive policy（θ₀=0.90，5-fold OOF）

| model | macro | Δ vs K1 | ΣN_l² | vs K1 | vs K8 | 路由 K0/K1/K8 | 单数据集最大 drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 0.562 | **−0.081** | 8.51e8 | 0.824 | 0.318 | 382/368/31 | 0.125 (gqa) |
| tree | 0.603 | **−0.040** | 1.09e9 | 1.054 | 0.407 | 237/467/77 | 0.070 (gqa) |
| threshold | 0.618 | **−0.024** | 1.00e9 | 0.971 | 0.375 | 180/540/61 | 0.045 (gqa) |

**全部 3 类模型：macro 均低于 fixed K=1**；没有任何模型通过 accuracy-oriented（需 ≥K1+1pp）或 efficiency-oriented（需 ≥K1−0.5pp **且** ≤0.85×K1）。tree/threshold 甚至把算力升到 ≥K1（G1 把样本送到昂贵的 K=8，rescue 却没换来准确率）。

### 5.3 operating-point 敏感性（θ₀∈{0.85…0.97}）

| θ₀ | model | macro | Δ vs K1 | vs K1 compute | accPass | effPass |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.85 | threshold | 0.586 | −0.056 | 0.872 | ✗ | ✗ |
| 0.90 | threshold | 0.618 | −0.024 | 0.971 | ✗ | ✗ |
| 0.95 | threshold | 0.635 | −0.007 | 1.046 | ✗ | ✗ |
| 0.97 | threshold | **0.638** | **−0.005** | 1.075 | ✗ | ✗ |

θ₀ 越严 → 路由到 K0 越少 → accuracy 越接近 K1，但 **compute 反而升到 ≥K1**（G1 的 K8 延长抵消了 G0 的节省）。**没有任何 operating point 同时满足效率 PASS 的两个条件**；即便 accuracy 最接近的点，单数据集 drop 也达 2pp（>1pp 上限）。

### 5.4 leave-one-dataset-out（θ₀=0.90）

| model | textvqa | docvqa | ocrbench | gqa | LODO macro | vs 5-fold macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| logistic | 0.605 | 0.650 | 0.658 | **0.430** | 0.586 | 0.562 |
| tree | 0.740 | 0.655 | 0.635 | 0.540 | 0.643 | 0.603 |
| threshold | 0.715 | 0.635 | 0.652 | 0.530 | 0.633 | 0.618 |

LODO 未完全崩溃（macro > 0.55），但 **logistic 在 gqa 上 LODO=0.430（比 fixed K=1 低 11pp）**，说明部分 gate 学到了数据集偏差（gqa 是门控最差的数据集）；tree/threshold 更稳但仍不超 K1。

---

## 6. 验收判断

| 条件 | logistic | tree | threshold |
| --- | --- | --- | --- |
| Accuracy-oriented PASS（macro ≥ K1+1pp 且 compute < 0.75×K8） | ✗ | ✗ | ✗ |
| Efficiency-oriented PASS（macro ≥ K1−0.5pp 且 compute ≤ 0.85×K1） | ✗ | ✗ | ✗ |
| 单数据集 drop ≤ 1pp | ✗ (gqa −12.5pp) | ✗ (gqa −7pp) | ✗ (gqa −4.5pp) |
| 方向一致性（threshold/logistic vs tree） | 一致（全部 NO-GO 方向） | — | — |
| OOF / LODO | 满足（严格 OOF；LODO 未完全崩溃但 gqa 明显退化） | — | — |

**VERDICT：NO-GO（明确）**。未满足任何一条 PASS，且**所有单数据集 drop 均超过 1pp**。G0 在 3/4 数据集中接近随机、G1 正样本过少且 rescue precision≈0；adaptive policy 在每个 operating point 上都以算力换不来准确率。按 task 约定：**停止 adaptive gate，将 fixed K=1 作为正式 MALT 方法（MALT-1）。**

> 附带结论：K1→K8 的延长（G1）即便在 oracle 下也只有 ≤5pp 可挽救、且 K8 算力是 K1 的 ~2.6×，**MALT-1（fixed K=1）已是强 Pareto 默认，不建议任何形式的 K8 长尾**。

---

## 7. 附录：复现与代码

```bash
# 1) transition audit（no-GPU）
python3 scripts/audit_malt_transitions.py          # -> transitions.json
# 2) 特征提取（GPU，1×A40，每数据集 ~5–6 min）
python3 scripts/extract_malt_features.py --benchmark <textvqa|docvqa|ocrbench|gqa> --n 200
# 3) 审计 + policy simulation（CPU）
python3 scripts/audit_malt_gates.py --g0-precision 0.90   # -> audit.json（含 3×781 条 OOF 预测）
# 4) 渲染 summary 表
python3 scripts/audit_malt_summary.py
```

- 数据：`experiments/malt_adaptive_gate_feasibility/{transitions.json, features_{bench}.json, audit.json}`
- `audit.json` 含：每模型每数据集 G0/G1 指标、policy、baselines、oracle、LODO、acceptance、以及 **2343 条逐样本 strict OOF 预测**（`results.*.oof_predictions`：g0/g1 score、各自阈值、routed_K、routed_correct）。

### 复现核验
- keep-set 一致性 781/781（100%）；decode answer 27/781 抖动但 keep 全一致（§2.4）。
- compute proxy（ΣN_l、ΣN_l²）由逐样本 `layer_visual_counts` 计算，中位数与 lifetime-sweep 报告逐项吻合（textvqa K0 6768/1.273e6、K1 7896/2.334e6、K8 11844/6.047e6）。

# J5 — query-aware pre-merger gate 裁决：NO-GO（有效测量，方法冻结）

**预注册判据**（DECISIONS 2026-07-23，跑前定稿未挪）：GO = (i) 4 基准官方均值 ≥ plain RBM +1.0pp (ii) ≥3/4 基准 ≥ plain−0.5pp (iii) pooled paired z≥1.5。止损 ≤2 GPU·h。

## 第一次运行（无效，已纠正）

qsim 实现 bug：`_find_embed_tokens` 命中 `visual.pos_embed`（vLLM 词嵌入是 VocabParallelEmbedding 非 nn.Embedding 子类被 isinstance 扫描跳过）→ qsim 维度不符全部 blend 回退 → 全程实测 λ=0 vs λ=0 的空测量。修复（c92e32f）：候选路径+vocab 规模识别，验证 embed=language_model.model.embed_tokens(151936)、blend 真启用。

## 有效运行（2026-07-24）

**Dev slice λ 选择**（textvqa+docvqa 各 32 条，id 与 _200 子集不相交，seed=0 审计落盘，官方指标）：

| λ | dev 均值 | vs λ0 |
|---|---|---|
| 0.0（plain） | 0.5772 | — |
| 0.3 | 0.5599 | **−1.7pp** |
| 0.5 | 0.5250 | **−5.2pp** |
| 0.7 | 0.5443 | **−3.3pp** |

**所有 λ>0 皆负 → 按预注册规则选 λ=0 → gate 判 NO-GO**（McNemar z=0，因 QA≡plain）。GPU 实际 ~0.8h ≤ 2h 止损线。

## 裁决与解读

- **方法冻结 = plain RBM（L2 pre-merger ranking）。不再搜索任何 hybrid/router/query-aware 变体**（延伸 07-23 merger-aware gate FAIL + 07-21 router 探针：图像级信号够不到 oracle + 本次 query 级 embedding 信号有害）。
- **机制强化**：pre-merger 特征是纯视觉（merger 输入），不含 query 信息——廉价的 query-conditioned 信号（词嵌入 cosine）在 pre-merger 阶段有害无益；oracle 的 +8.2pp query 余量只能在 LLM 交叉注意力混合后获得（**FastV layer-2 注意力排序胜 TextVQA/GQA 的机制解释**：J4 探针实证）。入稿作 bounded negative + 机制旁证（"为何 pre-merger 必须是 query-blind 的鲁棒默认"）。
- claim 终态（同模型公平比较）：RBM = 鲁棒默认（任何基准不崩、text-dense/OCR 大胜 post 族 +36~42pp）；FastV = query-conditioned 竞争者（胜 TextVQA +8.2/GQA +7，OCR 崩至 post 水平）；**不写"RBM 超过现有方法"**。

## 产物
runs/v3_merger_aware/j5/{lambda_selection.json, j5_gate_result.json, dev_*, gate_*}；实现 experiments/j5_impl.md；设计 notes/j5_qa_gate_design.md。

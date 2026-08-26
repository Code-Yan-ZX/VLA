# J5 — query-aware pre-merger 单次 gate 设计（预注册，跑前定稿不挪门槛）

> 判据已在 DECISIONS 2026-07-23 预注册。本文件=实现规格（J2 后执行，GPU ≤2h 止损）。

## 方法：question-conditioned pre-merger saliency（QA-pre）

- 对每个 merge-unit i：`qsim_i = max_{t ∈ question tokens} cos( merger(unit_feat_i), LLM_embed(q_t) )`
  - `merger(unit_feat_i)`：原生 merger MLP 单 unit 前向（unit 恰为 spatial_merge_size 输入行）→ LLM 空间嵌入；
  - `LLM_embed(q_t)`：问题 token 经 LLM 词嵌入层（共享空间，merger 输出即入 LLM）；
  - 动机：oracle +8.2pp 是 query 依赖（哪段文字对**该题**关键）；在共享嵌入空间算 unit-问题相关性是最便宜的 query-aware 信号。
- 组合：`s_i = (1−λ)·norm(l2_saliency_i) + λ·norm(qsim_i)`，两组分数各自 min-max 归一后加权。
- λ ∈ {0.3, 0.5, 0.7}，**dev slice 选 λ**（与 gate slice 不相交）：
  - dev = eval/full_splits/{textvqa_val,docvqa_val}.jsonl 中 **id 不在 *_200 子集内**的随机各 32 条（固定 seed=0 取前 32，脚本落盘 dev id 清单供审计）；
  - 选法：dev 上 QA-pre vs plain-pre 的**均值增益最大**的 λ；三者皆 ≤0 则取 λ=0（即 QA 无增益，直接进 gate 作负结果）。

## Gate 协议（预注册判据，DECISIONS 2026-07-23）

- 模型=Qwen3-VL-8B；4 子集 n=200（textvqa_200/docvqa_200/ocrbench_200/gqa_200）；r=0.75；官方指标（VQA-acc/ANLS/OCRBench 官方/GQA official exact-match）。
- **GO** 需同时：
  (i) 4 基准官方均值 ≥ plain RBM +1.0pp；
  (ii) ≥3/4 基准各 ≥ plain RBM −0.5pp（无单项回退）；
  (iii) pooled 增益 z≥1.5（paired binomial SE，按 per_sample correct 配对）。
- **NO-GO** → 方法冻结为 plain RBM；QA 作 bounded negative 如实写入论文；**不再搜索任何 hybrid/router 变体**（延伸 07-23 已 FAIL 的 merger-aware gate）。
- GPU 止损 ≤2 GPU·h（dev λ 选择 + 4×2 cell：QA-pre 与 plain-pre 同池重跑配对）。

## 实现点（runner，J2 完成后动）

- `_score_units` 插拔点（注释 "SELECTOR plug-in point"）加 `--selector qsim`：
  unit 特征 → merger MLP（复用 PreMergerPruner 已有 merger 引用）→ 与缓存的 question token 嵌入算 cos-max；
  question 嵌入在请求级算一次（tokenizer + embed_tokens；vLLM 进程内取模型 embed 层，enforce_eager 可读）。
- 输出 per_sample 附 qa_scores（可选，供分析）。
- 脚本：`scripts/j5_qa_dev_select.py`（dev 上 λ 选择）+ `runs/v3_merger_aware/j5_gate.sh`（4+4 cell + paired 检验 + GO/NO-GO 判定打印）。

## 诚实预期

router 探针已证图像级信号够不到 oracle（0.484 ≤ always-pre 0.494）；qsim 是**查询级**新信号，可能仍是噪声级。NO-GO 是大概率结局——那也是有价值的结果：证明 pre-merger 纯显著性是不动点，query-aware 在原生 merger 前无增益，强化机制 claim。

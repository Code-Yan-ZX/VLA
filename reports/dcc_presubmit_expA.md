# DCC 投稿前实验 A — Qwen3-VL main-only Post-L2 控制（Post-main vs Pre-final）

日期：2026-09-19 ｜ 分支：`exp/qrbm-e1`（代码提交 `5deeaa0`、`bcbdaae`，已同步 main `26253cd`+）
状态：TextVQA / DocVQA **完成**；OCRBench / GQA 见 §5。

## 1. 问题

论文 `tab:stage` 的 "Post-L2" 排序打分对象是 vLLM Qwen3-VL 视觉输出的**完整拼接行**
`[main | ds0 | ds1 | ds2]`（vLLM 0.19.0 `qwen3_vl.py:654-656`：`torch.cat([hidden_states] +
deepstack_feature_lists, dim=1)`，main 块在前，已对安装版源码核实）。Pre-final 则在
main-merger **输入**处对 2×2 unit 特征打 L2。因此 Pre-final vs Post-L2 同时改变了
(i) 排序所处阶段 与 (ii) 排序特征空间（4 patch 输入范数 vs main+deepstack 拼接范数）。

本实验新增控制臂 **Post-main**：仅用 main-merger 列块（`s[:, :4096]`）计算 L2/TopK 排序，
但保留**完整** main+deepstack 行（`index_select` 全行，kept unit 的 deepstack 特征不丢），
每图最终 token 数与其它臂完全一致。它分离出 deepstack 列对 Post 排序的贡献，
回答：论文报告的 pre>post 差值有多少是顺序效应、多少是拼接范数的特征空间效应。

## 2. 命令与环境

- 环境：conda `qwen3vl_clean`（python 3.10.20, torch 2.10.0+cu128, vLLM 0.19.0,
  transformers 4.57.6），`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  VLLM_NO_USAGE_STATS=1 VLLM_USE_MODELSCOPE=False VLLM_ENABLE_V1_MULTIPROCESSING=0`。
- 1× A40 46GB，串行；GPU 空闲门 ≥40000 MiB。
- 代码：`src/v3_premerger/v3_premerger_runner.py` 新增
  `--post-score-scope {full,main}`（默认 `full` = 原行为逐位不变；main 限定
  `--mode post` + stage ranking + qwen3vl，守卫见 main()）；dry check 新增
  d2 步（合成 main/deepstack 排序相反张量，验证 main-only 选择差异与保留全行）。
- Campaign：`scripts/run_dcc_expA_main_only.sh`（日志即原始 stdout）。
  协议与 `scripts/run_p0_1_full_split.sh` 完全一致：
  - 通用：`--model-family qwen3vl --r 0.75 --mode post --post-score-scope main
    --selector l2 --max-tokens 32`，greedy temp=0，native 像素（max_pixels=0），seed=0。
  - textvqa/ocrbench/gqa：`--max-num-seqs 8 --max-model-len 8192 --gpu-memory-utilization 0.9`
  - docvqa：`--max-num-seqs 4 --max-model-len 32768 --max-num-batched-tokens 32768
    --gpu-memory-utilization 0.9`
  - 数据：`eval/full_splits/{textvqa_val,docvqa_val,ocrbench,gqa_testdev}.jsonl`
    （n=5000/5349/1000/12578）。
- 分析：`scripts/analyze_dcc_expA.py`（官方评分器 = `src/v3_premerger/paired_stats.py`
  的 `score_sample`，与论文官方 rescore 同口径；配对 bootstrap 20000 次 95% CI +
  sign-flip 置换 p + McNemar，seed 固定）。
- 对照锚点（**未重跑**，直接复用）：`results/acmmm_final_controls/p0_1/p0_1_qwen3_pre-final_*.json`
  （官方分 0.4985 / 0.2836，与 `reports/acmmm_final_controls.md:38-39` 一致）。

## 3. 技术验证（跑全量前）

- dry check：ALL PASS（含新 d2 main-scope 测试）。
- 冒烟 n=8 textvqa：`score_scope=main, n_deepstack=3, main_hidden=4096, fires=2,
  nk=[[672,168]]`（保 25% ✓），n_skipped=0。
- n=200 两基准 vs 既有 pre-final n=200（`runs/qwen3_prefinal_control/`）：
  ID 集 200/200 完全一致、per-sample `prompt_token_ids` **逐样本相等**（iso-token 达成）、
  双臂 0 skip。VALIDATE: PASS。
- 全量 iso-token：textvqa mean_ptid 215.8 = 215.8；docvqa 946.7 = 946.7；双臂 0 skip。
- 分支执行确认：全量 diag `score_scope=main, main_hidden=4096, fires=2143/1391`。

## 4. 结果（官方评分器，配对 bootstrap 95% CI，A = Post-main）

### TextVQA（n=5000，配对 n=5000）

| 臂 | VQA-acc |
|---|---|
| none（锚点） | 0.8443 |
| Pre-final | 0.4985 |
| **Post-main（A）** | **0.3170** |
| Post-L2 full-scope（论文现值） | 0.2217 |

Δ(Post-main − Pre-final) = **−18.15 pp**，95% CI [−19.66, −16.67]，p=5e-05。
（Post-main − Post-full = +9.5 pp：deepstack 列在 TextVQA 上使 Post 排序**变差**。）

### DocVQA（n=5349，配对 n=5349）

| 臂 | ANLS |
|---|---|
| none（锚点） | 0.9562 |
| Pre-final | 0.2836 |
| **Post-main（A）** | **0.3735** |
| Post-L2 full-scope（论文现值） | 0.2377 |

Δ(Post-main − Pre-final) = **+9.00 pp**，95% CI [+7.84, +10.14]，p=5e-05。**方向反转。**

完整配对统计（含 McNemar/置换）：`results/dcc_presubmit/expA/analysis_full.json`。

### 解读（无论方向如实报告）

1. **论文核心 headline 不倒**：RBM early-tap pre（TV 0.605 / DV 0.481）仍显著高于
   任何 post 变体（full 0.222/0.238 或 main-only 0.317/0.374）。
2. **tab:stage 的 DocVQA "+4.59 pp" 归因被推翻**：Pre-final > Post 的 DocVQA 差值
   完全来自 Post-L2 排序里 deepstack 列的污染；剔除后 Post-main **反超** Pre-final
   +9.00 pp。DocVQA 上不存在"先排序后合并受损"的顺序效应——存在的是反向顺序效应。
3. TextVQA 的顺序效应方向保留但幅度缩减（27.68 → 18.15 pp）：约 1/3 的
   full-scope 差值由 deepstack 列污染解释，其余 18 pp 为真实的阶段/特征空间差。
4. 机制表述需对齐：Pre-final 与 Post-main 都是"main-merger 邻域"特征，两者之差
   是纯粹的 merger 前/后排序差；DocVQA 上 merged-token 范数排序反而更好。

## 5. OCRBench / GQA（补跑）

- [待补：OCRBench n=1000]
- [待补：GQA n=12578]

## 6. 对论文的修改建议（`drafts/dcc2027_submission_20260916/main.tex`，本实验未改动稿件）

1. `:136` "Pre-final gains 27.68 points on TextVQA, **4.59 on DocVQA**, …"：
   DocVQA +4.59 必须加限定——main-only 控制下符号反转（Post-main +9.00 pp，
   CI [+7.84, +10.14]），该符号是拼接范数特征空间效应，不是顺序效应。
2. `:136` "The smaller DocVQA gain … prevents attributing that headline contrast to
   the merger boundary alone"：方向仍成立且现在有直接量化证据，建议升级表述为
   "a main-only scoring control reverses the DocVQA sign, attributing it to the
   concatenated-norm feature space rather than the ordering stage"。
3. `:140`（tab:stage caption）"Post-L2 scores the concatenated main-plus-deepstack
   output"：建议增加 Post-main 行/列或脚注披露本控制；
   `:88` 的 "neither a pure stage intervention" 披露句可保留并引用本实验。

## 7. 产物路径

- 原始结果：`results/dcc_presubmit/expA/{smoke_textvqa_n8, n200_*, full_*}_*.json`（+ `.log`）
  ；失败首跑存档：`full_ocrbench_main.failed_try1.{json,log}`（vLLM 编码器缓存
  99.6% 满 → 批 OOM → 全体 skip，p0_1 同旗标 0 skip，属边缘内存调度波动；已按
  skip≤10% 门重试）。
- 脚本：`scripts/run_dcc_expA_main_only.sh`、`scripts/analyze_dcc_expA.py`；
  代码改动：`src/v3_premerger/v3_premerger_runner.py`（`--post-score-scope`）。
- 日志：`experiments/dcc_expA_{validate,full,extra*}.log`。

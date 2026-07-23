# J0b — 官方评分器：OCRBench + GQA（离线重评分，纯 CPU）

给官方指标库补齐 **OCRBench** 与 **GQA** 的官方评分器，供全量评测离线重评分。
改动仅限 `src/v3_premerger/official_scorers.py`、`scripts/rescore_official.py`，
新增 `tests/test_official_scorers_toy.py`。未用 GPU，未 git commit，无 AI 署名。

---

## 1. 移植来源（URL + 行号）

### GQA — 归一化后 exact match
- **官方 eval**：`https://cs.stanford.edu/people/dorarad/gqa/evaluate.html`
  → 下载 `https://nlp.stanford.edu/data/gqa/eval.zip` 内 `eval.py`（"Evaluation code for GQA"）。
- **计分主循环 L347-351**：`gold = question["answer"]; predicted = predictions[qid]; correct = (predicted == gold)`（**L350**），`toScore` L152-153 = `float(1 if b else 0)`。
  → 严格官方版是**裸字符串 exact match**，假定 predictions 文件已清洗（GQA gold 本就是干净单词）。
- **归一化**：VLM 标准做法（lmms-eval / LLaVA 的 GQA eval）在 exact match 前对两侧做 VQA-eval 归一化
  （`processPunctuation` + `processDigitArticle`：小写 / 去标点 / 去冠词 a-an-the / 数词→数字 / 缩写展开）。
  本库已实现为 `vqa_normalize`（`official_scorers.py` L110-115），直接复用。
- **关键验证**：在本数据上 `normalized-exact == raw-exact == 0.470`（归一化对干净单词输出是 no-op），
  故实现既等于严格官方裸 exact，又能容忍冗长生成。

### OCRBench — 按 question_type 的 containment + 5 类 /1000
- **官方 repo**：`github.com/Yuliang-Liu/OCRBench`（eval 脚本的规范镜像 =
  **VLMEvalKit** `vlmeval/dataset/utils/ocrbench.py:OCRBench_eval`，**L8-60**，逐式移植）。
- **逐样本规则**（`if answer in predict` containment，对 `;`/list 任一命中即对）：
  - **HME**（`question_type == "Handwritten Mathematical Expression Recognition"`，即 HME100k LaTeX）：
    两侧 `strip + "\n"→" " + 去全部空格`，**不 lower（大小写敏感）**（VLMEvalKit **L29-35**）。
  - **其余类型**：两侧 `lower + strip + "\n"→" "`（**L36-42**）。
- **5 类汇总 + 总分**：VLMEvalKit L44-60（`Final Score` = 5 类正确数之和，满基准 = /1000；`Final Score Norm` = /10）。

### 数据侧 5 类口径（本库采用，来自 J0a 落地的 `eval/full_splits/ocrbench.jsonl`）
runner per_sample **不透传 category/extras**，故按 **id** 映射到 `eval/full_splits/ocrbench.jsonl`
（1000 行，含 `category` 字段）。5 类各 200（**TR/HTR/ST-VQA/DT-VQA/KIE**）：

| 数据侧 category (200) | 缩写 | 细 question_type 构成 |
|---|---|---|
| Text Recognition | TR | Regular / Irregular / Artistic / Digit String（各 50）|
| Handwriting Text Recognition | HTR | Handwriting(50) + **HME(100)** + Non-Semantic(50) |
| Scene Text-centric VQA | ST-VQA | Scene Text-centric VQA(200) |
| Document Text-centric VQA | DT-VQA | Doc-oriented VQA(200) |
| Key Information Extraction | KIE | Key Information Extraction(200) |

> **口径差异（已核对）**：VLMEvalKit 官方把 6 个识别类全归 "Text Recognition" 且单列 HMER；
> 数据侧把 Handwriting/HME/Non-Semantic 归 **HTR**、Doc→**DT-VQA**。**逐样本匹配规则两者完全一致**，
> 仅汇报汇总不同。按协调员指示采用**数据侧 category**（`OCRBENCH_CATEGORIES` + `OCRBENCH_QT_TO_CAT`）。
> HME 的逐样本规则仍按细 `question_type` 触发（nospace + 大小写敏感）。

---

## 2. 评分器 API（`official_scorers.py`）

| 函数 | 签名 | 返回 |
|---|---|---|
| `score_gqa` | `(pred, gt_str) -> float` | 归一化 exact match，1.0/0.0（gt 支持 `;` 多答案）|
| `score_ocrbench` | `(pred, gt_str, question_type="", nospace=None) -> int` | 官方逐样本 containment，0/1 |
| `ocrbench_category` | `(question_type="", category=None) -> str` | 数据侧 5 类（优先显式 category）|
| `score_ocrbench_total` | `(per_cat_counts) -> dict` | 5 类 + `Final Score`(/1000) + `Norm`(/10) |
| `score_gqa_batch` / `score_ocrbench_batch` | `(preds,gts)` / `(items)` | `{acc, n, per_item, …}`（含 extrap/1000）|

`scripts/rescore_official.py` 现支持 `--benchmark {textvqa,docvqa,ocrbench,gqa}`（可重复，默认全跑），
自动选 scorer；runner 的 containment 规则在脚本内**纯 CPU 复刻**（`runner_containment_gqa/_ocrbench`），
用于同口径（/n_total）对照，**不 import torch/vllm**。

---

## 3. Self-test 数字表

### 3a. GQA — 复现 merger pre≈post tie（写作红线）
`runs/v3_merger_aware/hybrid_gate/`（n=100，单词作答 prompt；rescore_rerun 无 GQA cell）：

| cell | stored(containment) | **OFFICIAL exact** | containment 重算(/n) | Δ(off−cont) |
|---|---|---|---|---|
| pre_gqa_r0.750_l2_n100 | 0.510 | **0.470** | 0.510 | **−4.0pp** |
| post_gqa_r0.750_l2_n100 | 0.510 | **0.470** | 0.510 | **−4.0pp** |
| none_gqa_r0.000_l2_n100（baseline）| 0.640 | 0.620 | 0.640 | −2.0pp |
| pre/post_gqa_cap64_n64 | 0.516 | 0.484/0.484 | 0.516 | −3.1pp |
| hybrid_gqa tf0.0/0.5/1.0 | 0.48/0.50/0.47 | 0.44/0.46/0.43 | 同左 | −4.0pp |

**结论**：**official pre == post == 0.470（精确 tie）**，containment 0.510==0.510。
官方 exact 比 containment 低 **4pp**（containment 把 "gt 是长答案子串" 误判为对，如 pred="it's a cat" / gt="cat"）。
**tie（pre==post）在官方口径下稳健保持** → merger 不伤 GQA 的写作结论不被推翻。
（router_probe n=200 用 free-form prompt → 官方 exact ≈0.01 近地板，Δ≈−30pp，同 textvqa/docvqa 冗长生成问题，非本红线。）

### 3b. OCRBench — 官方 vs containment 对照
`runs/v3_sota_matrix/`（+ hybrid_gate）全部 ocrbench cell：

| cell | stored | **OFFICIAL** | containment 重算 | Δ(off−cont) | extrap/1000 |
|---|---|---|---|---|---|
| A_ocrbench（baseline）| 0.760* | 0.730 | 0.730 | **+0.0pp** | 732 |
| C_ocrbench_r0.750（pre）| 0.580 | 0.580 | 0.580 | +0.0pp | 581 |
| B_ocrbench_r0.750（post）| 0.165 | 0.165 | 0.165 | +0.0pp | 166 |
| C_ocrbench_r0.875（pre）| 0.380 | 0.380 | 0.380 | +0.0pp | 378 |
| B/vz_ocrbench_r0.875（post）| 0.075 | 0.075 | 0.075 | +0.0pp | 75 |

\* A 的 stored 0.760 = 146/**192**（/n_answered，8 条 skipped 被剔除）；官方与 containment 重算均按
/n_total=200 → 146/200=0.730。**差异纯为分母口径，非评分规则。**

**5 类拆分（correct/total）**：
- A：TR 37/41 · HTR 28/37 · ST 30/39 · DT 17/42 · KIE 34/41
- C_r0.750(pre)：TR 35/41 · HTR 20/37 · ST 28/39 · DT 19/42 · KIE 14/41
- B_r0.750(post)：TR 6/41 · HTR 7/37 · ST 7/39 · DT 5/42 · KIE 8/41

**结论**：**official == containment（12/12 cell，Δ=+0.0pp，0 个逐样本分歧）** —— containment 本就是 OCRBench
官方指标，**不存在高估**；官方 HME 大小写敏感 vs runner 大小写不敏感的唯一差异在本数据 **0 样本触发**。
pre(C) > post(B) 方向在两套口径一致（0.580 vs 0.165，HOLD）。

### 3c. toy 单测（归一化分支）
`tests/test_official_scorers_toy.py`（17 例全过）+ `official_scorers.py __main__`（28 例全过）。
GQA 5 条归一化分支：case(`No`→`no`✓) / article(`the tree`→`tree`✓) / punct(`cat.`→`cat`✓) /
numword(`three cats`→`3 cats`✓) / **no-leak**(`I don't know` vs `no`→0，containment 会误判✓)。

---

## 4. 与 containment 差异总结论
- **GQA**：containment **高估 ~4pp**（merger_aware，子串误判）；官方 exact 0.470，**tie 保持**。
- **OCRBench**：containment **不高估**（== 官方，0pp）；唯一规则差（HME 大小写）本数据无影响。
- 写作影响：GQA "pre≈post" 与 OCRBench "pre>post" 两条结论在官方口径下均**不被推翻**。

## 5. 用法示例命令
```bash
# 单元自检（CPU）
python3 src/v3_premerger/official_scorers.py          # 内嵌 self-test（28 例）
python3 tests/test_official_scorers_toy.py            # toy 归一化单测（17 例）

# 离线重评分（写 runs/rescore_official/summary.json + drafts/rescore_official_report.md）
python3 scripts/rescore_official.py                            # 全 4 项官方指标
python3 scripts/rescore_official.py --benchmark gqa            # 仅 GQA
python3 scripts/rescore_official.py --benchmark ocrbench       # 仅 OCRBench
python3 scripts/rescore_official.py --benchmark textvqa --benchmark docvqa

# 程序内调用
python3 -c "import sys; sys.path.insert(0,'src'); from v3_premerger.official_scorers import score_gqa, score_ocrbench; \
  print(score_gqa('the tree','tree'), score_ocrbench('x ^ 2','x^2','Handwritten Mathematical Expression Recognition'))"
```

## 6. 改动文件
- `src/v3_premerger/official_scorers.py`：+ `score_gqa` / `score_ocrbench` / `ocrbench_category` /
  `score_ocrbench_total` / `score_gqa_batch` / `score_ocrbench_batch` + 数据侧 5 类常量 + self-test。
- `scripts/rescore_official.py`：+ `--benchmark` 选择器、`load_ocrbench_meta`（按 id 映射 category，
  优先 `eval/full_splits/ocrbench.jsonl`）、runner-containment 纯 CPU 复刻、GQA/OCRBench 报告段。
- `tests/test_official_scorers_toy.py`：新增 toy 单测。
- 产物：`runs/rescore_official/summary.json`、`drafts/rescore_official_report.md`。

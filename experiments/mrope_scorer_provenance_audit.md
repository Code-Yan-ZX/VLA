# mRoPE / Scorer Provenance Audit

> 分支 `exp/deferred-rbm-n200`；2026-08-25。触发：MALT-C 命令与论文正式 RBM
> 配置重复 + Gate C 用 raw `correct` 而非 official rescore。本报告只做审计与
> 纠错，不改论文正文、不产生新命名、不运行新方法。
> 复现脚本：`scripts/audit_mrope_gateC_official_rescore.py`（Gate C official
> rescore 重执行）；数据全在 `runs/`（gitignored）。

## 0. 结论（三项选一）

**`MALT-C DUPLICATE / MALT-1 REFUTED`**

- **MALT-C（H0n）不是新方法**：其命令 `--mode pre --r-pre 0.25 --mrope native`
  与论文正式 native-coordinate RBM 配置逐字相同（DECISIONS 2026-07-30 决策 +
  `ef72607` + `runs/r2_same_scope/r2c_qwen3vl_pre_r0.75_ocrbench_n200.json`），
  per-sample 答案 181/181 逐字相同。MALT-C = 论文已要求的 coordinate-correct
  RBM 实现。
- **MALT-1（deferred K1）的“+12.5pp”增益被证伪**：该增益来自 mRoPE 不一致
  （K0 用 `vllm-mimic`、K1 用 native），不是 deferral。official-rescore 下，
  native-immediate（H0n，= 论文 RBM）与 native-deferred（H1）macro 差
  −0.0009（paired bootstrap 95% CI [−0.0021, +0.0016]），≈ 0。
- **Deferred contextualization hypothesis REFUTED**：先前的 K0→K1 增益主要由
  inconsistent positional handling（vllm-mimic→native）造成。

**不得继续使用未经 official-rescore 的“六 Gate PASS”表述。**

---

## 1. 冻结事实与关键机制

1. `baselines_hf.py` 运行时 `correct` 字段由**文件内联的 ad-hoc scorer**
   计算（`_norm_words`/`_singular` 词包含匹配：`score_textvqa` 为任意 gt 变体
   词序列包含、`score_docvqa = score_textvqa` 非 ANLS、`score_gqa` 词成员 +
   yes/no 特判、`score_ocrbench` 子串包含）。文件注释明示：“the AUTHORITATIVE
   metric is the offline re-score by official_scorers.py”。
2. **AUTHORITATIVE** = `src/v3_premerger/official_scorers.py`：TextVQA 官方
   VQA-acc、DocVQA ANLS、GQA normalized EM、OCRBench 官方五类别。
3. `--mrope` 默认 = `vllm-mimic`（pre/cascade 才生效；rankbridge/fastv 恒用
   native 坐标，与 `--mrope` 无关）。
4. sweep 分析（`analyze_deferred_sweep.py`）**用了 official_scorers**；
   MALT Gate C 分析（`analyze_malt_gateC.py`）**读 JSON 的 raw `correct`**。
   → 同一 arm 在两个管道下数字不同（如 H1 textvqa：raw 0.830 vs official
   0.7433）。

## 2. Provenance 全表

manifest hash（`eval/subsets/*_200.jsonl`）：textvqa `ae3e8f56dfe8`、
docvqa `a5a21dad7f98`、ocrbench `71e48812ed1d`、gqa `05a5e806ab43`；
explore64：`085dc342166a` / `eb178c60d9dc` / `1ad0863a9add` / `78a06c85ebc2`。
model 全部 `Qwen/Qwen3-VL-8B-Instruct`，seed 0，greedy，max-tokens 32，
docvqa max-pixels 600000 其余 0。

| cell | raw file | git commit | exact command | mode | mrope | manifest | scorer | stored metric |
| ---- | -------- | ---------- | ------------- | ---- | ----- | -------- | ------ | ------------- |
| 原 deferred n=200 RBM parent（immediate-RBM/K0） | `runs/cascade/gate_pre25_{b}.json` | cascade-gate 时代（先于 9af0682 复用；runs 不入库，命令见下） | `baselines_hf.py --mode pre --r-pre 0.25 ...`（无 `--mrope`→默认） | pre | **vllm-mimic**（diag 200/200） | {b}_200 | raw `correct`（ad-hoc） | 官方重打分在分析侧：textvqa .5967 / docvqa .4239 / ocrbench .5801 / gqa .4150 |
| lifetime sweep K0 | `runs/deferred_rbm/sweep_pre_{b}_n200.json` | ddcd303 | `--mode pre --r-pre 0.25 --model Qwen/Qwen3-VL-8B-Instruct --benchmark {b} --subset eval/subsets/{b}_200.jsonl --n 200 --seed 0 --max-tokens 32 --max-pixels {docvqa?600000:0}` | pre | **vllm-mimic** | {b}_200 | raw `correct`（ad-hoc） | 官方：textvqa .5967 / docvqa .4239 / ocrbench .5801 / gqa .4150 |
| lifetime sweep K1 | `runs/deferred_rbm/sweep_deferred_K1_{b}_n200.json` | ddcd303 | `--mode rankbridge --r 0.75 --rb-fuse quota --rb-rho 1.0 --fastv-k 1 --model ... --benchmark {b} --subset eval/subsets/{b}_200.jsonl --n 200 --seed 0 --max-tokens 32 --max-pixels {docvqa?600000:0}` | rankbridge | **native**（rankbridge 恒 native） | {b}_200 | raw `correct`（ad-hoc） | **官方（本审计复现）**：textvqa **.7433** / docvqa **.5959** / ocrbench **.6354** / gqa **.5400** |
| Goal H0 | `runs/malt_goal_mode/explore_h0_{b}_n64.json` | ff30ce9 / 9945ac0 | `--mode pre --r-pre 0.25 ... --subset eval/subsets/{b}_explore64.jsonl --n 64` | pre | **vllm-mimic**（diag 64/64） | {b}_explore64 | raw `correct`（ad-hoc） | n64 ad-hoc：textvqa .703 / docvqa .406 / ocrbench .550 / gqa .578 |
| Goal H1 | `runs/malt_goal_mode/explore_h1_{b}_n64.json` | 9945ac0 | `--mode rankbridge --r 0.75 --rb-fuse quota --rb-rho 1.0 --fastv-k 1 ... --subset {b}_explore64 --n 64` | rankbridge | **native** | {b}_explore64 | raw `correct`（ad-hoc） | n64 ad-hoc：textvqa .812 / docvqa .500 / ocrbench .567 / gqa .578 |
| Goal H0n | `runs/malt_goal_mode/explore_h0n_{b}_n64.json` | 1eba1b0 | `--mode pre --r-pre 0.25 --mrope native ... --subset {b}_explore64 --n 64` | pre | **native**（diag 64/64） | {b}_explore64 | raw `correct`（ad-hoc） | n64 ad-hoc：textvqa .828 / docvqa .500 / ocrbench .550 / gqa .609 |
| Gate C H0n | `runs/malt_goal_mode/h0n_{b}_n200.json` | 2e97d30 | `--mode pre --r-pre 0.25 --mrope native --model Qwen/Qwen3-VL-8B-Instruct --benchmark {b} --subset eval/subsets/{b}_200.jsonl --n 200 --seed 0 --max-tokens 32 --max-pixels {docvqa?600000:0}` | pre | **native**（diag 200/200） | {b}_200 | raw `correct`（ad-hoc） | **官方（本审计计算）**：textvqa .7433 / docvqa .5924 / ocrbench .6354 / gqa .5400 |
| Gate C H1（symlink→sweep K1） | `runs/malt_goal_mode/h1_{b}_n200.json` | ddcd303（symlink） | 同 sweep K1（见上） | rankbridge | native | {b}_200 | raw `correct`（ad-hoc） | 官方：.7433 / .5959 / .6354 / .5400 |
| 论文 Table 3 RBM（Qwen3，native） | `runs/r2_same_scope/r2c_qwen3vl_pre_r0.75_ocrbench_n200.json` | ef72607（--mrope native 定稿） | `--mode pre --r-pre 0.25 --mrope native --model Qwen/Qwen3-VL-8B-Instruct --benchmark ocrbench --subset eval/subsets/ocrbench_200.jsonl --n 200 --seed 0` | pre | **native** | ocrbench_200 | raw `correct`（ad-hoc）+ 分析侧 official | 官方 ocrbench acc .6354（**与 H0n 逐样本 181/181 相同**） |
| 论文 Table 3 RBM（Qwen2.5，native） | `runs/r2_same_scope/r2c_qwen2vl_pre_r0.75_{b}_n200.json` | ef72607（修复后重跑） | `--mode pre --r-pre 0.25 --mrope native --model Qwen/Qwen2.5-VL-7B-Instruct ... --subset {b}_200.jsonl --n 200 --seed 0` | pre | **native**（修复 qwen2vl×pre×mimic 退化） | {b}_200 | raw `correct`（ad-hoc）+ 分析侧 official | **官方（本审计重算）**：textvqa .6683 / gqa .520 / docvqa .5062 / ocrbench .370（skip-as-wrong；excl-skip .4111）——与论文 Table 2 一致 ✓ |

**Q1 哪些 K0/RBM 用 vllm-mimic**：`gate_pre25_*`（原 deferred n=200 RBM
parent）、`sweep_pre_*`（sweep K0）、Goal H0（n64）、cascade 阶段其余
`--mode pre` 无 `--mrope native` 的 cell。
**Q2 哪些用 native**：sweep K1/K3/K5/K8 与全部 rankbridge/fastv（恒 native）、
H0n（n64+n200）、H1（native）、r2c qwen2vl/qwen3vl pre（--mrope native）、
post 模式（恒 native）、P1/ef72607 之后全部 HF RBM pre cell。
**Q3 论文声称 native 的 cell，原始命令是否含 `--mrope native`**：
r2c 系列 **是**（脚本含该 flag）；Table 2 的 Qwen3 RBM 主矩阵若来自
`gate_pre25_*`（vllm-mimic）则**否**——见 §5 污染清单。
**Q4 报告文字写 native、原始产物实际为 mimic 的情况**：`deferred_rbm
_lifetime_sweep.md` / `deferred_rbm_n200_gate.md` / STATE.md 把
“RBM=pre25”当 native 语义的对照父（未写错 mrope 字符串，但把 mimic-RBM
当作“普通 RBM”参与 K0→K1 归因，等价于把 mimic 当作 native 比较基）。
**Q5 H0n 是否与论文某个正式 RBM arm 命令完全相同**：**是**——r2c qwen3vl
pre native（以及 ef72607 后全部 HF RBM pre cell）。
**Q6 若相同，逐样本答案是否相同**：ocrbench 181/181 **完全相同**（qwen3vl
仅有该 native r2c cell；其余 bench 的论文 Qwen3 RBM 若为 mimic 源则与 H0n
不同——见下）。

## 3. Gate C official rescore 重执行

用 `official_scorers.py`（= sweep 同一评测器版本，`analyze_deferred_sweep.py`
同款调用）对 H0n/H1 原始 answer 重打分，**不读 raw `correct`**。

**H1（deferred K1）官方分数复现 ✓**：
textvqa **0.7433** / docvqa **0.5959** / ocrbench **0.6354** / gqa **0.5400**
——与 sweep_analysis.json（git ea359d8）逐位一致。raw `correct` 同文件读取
0.830 / 0.470 / 0.6354 / 0.565（raw≠official 是预期，见 §4）。

**H0n（native immediate）官方**：textvqa .7433 / docvqa .5924 / ocrbench
.6354 / gqa .5400。

| bench | H0n off | H1 off | Δ(H0n−H1) | paired bootstrap 95% CI | W/T/L(binary) | n |
| ----- | ------- | ------ | --------- | ----------------------- | -------------- | - |
| textvqa | .7433 | .7433 | .0000 | [−.0050, +.0050] | 0/200/0 | 200 |
| docvqa | .5924 | .5959 | **−.0035** | [−.0185, +.0106] | 1/198/1 | 200 |
| ocrbench | .6354 | .6354 | .0000 | [.0000, .0000] | 0/181/0 | 181 |
| gqa | .5400 | .5400 | .0000 | [−.0250, +.0250] | 3/194/3 | 200 |
| **macro** | **.6278** | **.6287** | **−.0009** | **[−.0021, +.0016]** | | 781 |

OCRBench 官方五类别（两臂**逐类相同**）：TextRec 37/41、HTR 22/37、
ST-VQA 28/39、DT-VQA 4/25、KIE 24/39（per-item acc .6354）。
DocVQA 用连续 ANLS per-item（不混 binary）；差值 CI 均含 0。

> 结论：**official 下 H0n ≈ H1**。Gate C 报告的“macro .621 vs .625
> CI[−.012,+.004]”是 raw ad-hoc `correct` 的产物，不是 official 分数，不可
> 作为正式数字引用。

## 4. raw `correct`（ad-hoc）vs official 差异（Gate C scorer-mismatch 尺度）

| arm/bench | raw_correct | official | raw−official |
| --------- | ----------- | -------- | ------------ |
| h0n textvqa | .820 | .7433 | **+.0767** |
| h0n docvqa | .465 | .5924 | −.1274（ANLS 连续>binary 率） |
| h0n ocrbench | .6354 | .6354 | .0000 |
| h0n gqa | .565 | .5400 | +.0250 |
| h1 textvqa | .830 | .7433 | **+.0867** |
| h1 docvqa | .470 | .5959 | −.1259 |
| h1 ocrbench | .6354 | .6354 | .0000 |
| h1 gqa | .565 | .5400 | +.0250 |

raw `correct` 与官方指标偏差可达 ±12.7pp；任何把 raw `correct` 当
official score 写入表格/报告的行为均属污染。`correct` 字段未被特征提取
阶段覆盖（`baselines_hf.py` 运行期写入后无 rescore 回写）。

## 5. 原 Deferred 因果 claim 审计（因果对照重写）

keep-set 在**所有**对照臂 100% 相同（实测：H0n==gate_pre25==H1，
200/200×3 + 181/181），故 keep-set 恒为常量，只 mRoPE 与删除层可变。

| 对照 | keep-set | mRoPE | 删除层 | 能隔离什么 |
| -- | -------- | ----- | --- | ----- |
| sweep K0（pre, mimic）vs sweep K1（rankbridge, native） | 相同 | mimic vs native | K=0 vs K=1 | **不能隔离 deletion timing**（mRoPE 混淆）。官方 macro +12.47pp = 混淆差 |
| Goal H0（pre, mimic）vs H1（rankbridge, native）n64 | 相同 | mimic vs native | K=0 vs K=1 | **不能隔离 deletion timing**（同上） |
| **H0n（pre, native）vs H1（rankbridge, native）n200** | 相同 | **native vs native** | K=0 vs K=1 | **✅ 隔离 deletion timing**。官方 macro Δ −.0009（CI [−.0021,+.0016]）→ 无 deferral 效应 |
| H0n（native）vs gate_pre25（mimic），均 K=0 | 相同 | native vs mimic | 均 K=0 | **✅ 隔离 mRoPE/positional**。答案差异 textvqa 33.0% / docvqa 56.5% / ocrbench 63.0% / gqa 42.0% |

答案差异计数（H0n vs H1，native/native）：textvqa 7/200、docvqa 20/200、
ocrbench 23/181、gqa 6/200（3–13%），official 净效果 ≈ 0 —— 真正的
deferred effect 是噪声级。

**正式结论：**

> **Deferred contextualization hypothesis is refuted; the prior K0→K1 gain
> was caused primarily by inconsistent positional handling (vllm-mimic →
> native mRoPE).**
> 只有 official-rescore 后的 H0n/H1 差值（−.0009, CI[−.0021,+.0016]）才代表
> 真正的 deferred effect；它不显著且 ≈ 0。不得继续把 MALT-1 写成方法。

## 6. MALT-C 是否重复 → **是**

MALT-C/H0n 与以下逐项一致：
- **当前 HF native RBM**（ef72607 后 P1/Table 3 RBM cells）：`--mode pre
  --r-pre 0.25 --mrope native`——命令逐字相同；
- **Qwen2.5 native RBM fix**（DECISIONS 2026-07-30：全部 5 cell 统一
  `--mrope native`，qwen2vl 必需、qwen3vl 可比）——同一 flag；
- **论文方法定义**（pre-merger L2 top-κ，survivor 保留 native 坐标）；
- **`ef72607`**（“P1: use --mrope native for HF RBM (pre) cells”，Table 2
  note iii 明确 “vllm-mimic position layout is a known-degenerate path”）；
- **DECISIONS.md 2026-07-30 native-mRoPE 决策**（决策正文 + 脚本注脚）。

算法与命令相同 → **明确写：**

> **MALT-C is not a new method; it is the already-required coordinate-correct
> implementation of RBM.**

外部 novelty collision（不能以“保留 native coordinates”单独提出新方法 claim）：
- **IVC-Prune**：preserve original position IDs（与被剪 token 共享原始坐标）；
- **Reroute**：physical pruning baselines 用 keep-index positional convention。

（与 MALT-C “immediate prune + 保留 survivor 自身 get_rope_index 坐标”属同一
positional convention 族；无独立算子 novelty。）

## 7. 论文主表污染清单（Table 1/2/3 RBM cells 逐项核对）

**Table 1（main full-split stage-law，Qwen3/Qwen2.5 RBM pre @25%）— CLEAN ✓**
- 来源：`runs/full_matrix/j7_{fam}_pre_{bench}_r0.750_full.json`（vLLM 0.19，
  native `get_mrope_input_positions`，Qwen2.5 cursor fix 已打），
  `scripts/j7_main_table.py` official rescore。native + official，无污染。
- 数字：Qwen3 0.605/0.481/547/0.449；Qwen2.5 0.702/0.636/476/0.559。

**Table 2（regime map，RBM @25% same-scope）— 1 处污染 ⚠️**
| row | 表格值 | 来源 cell | mrope | official | 判定 |
| --- | ------ | --------- | ----- | -------- | ---- |
| Qwen3 TextVQA (full) | 0.605 | j7 full-split（复用 Table 1） | native (vLLM) | ✓ | CLEAN |
| Qwen3 GQA (full) | 0.449 | j7 full-split（复用 Table 1） | native (vLLM) | ✓ | CLEAN |
| Qwen3 OCRBench (n=200) | 0.575 | `r2c_qwen3vl_pre_r0.75_ocrbench_n200.json`（skip-as-wrong 口径 0.6354×181/200） | native | ✓ | CLEAN |
| **Qwen3 DocVQA (n=200)** | **0.4239** | **`runs/cascade/gate_pre25_docvqa.json`（vllm-mimic！）** | **vllm-mimic** | ✓(ANLS) | **⚠️ POLLUTED：native 应为 0.5924** |
| Qwen2.5 TextVQA | 0.6683 | `r2c_qwen2vl_pre_r0.75_textvqa_n200.json` | native | ✓ | CLEAN |
| Qwen2.5 GQA | 0.520 | r2c_qwen2vl_gqa | native | ✓ | CLEAN |
| Qwen2.5 DocVQA | 0.5062 | r2c_qwen2vl_docvqa | native | ✓ | CLEAN |
| Qwen2.5 OCRBench | 0.370 | r2c_qwen2vl_ocrbench（skip-as-wrong 0.4111×180/200） | native | ✓ | CLEAN |

- **污染**：Table 2 Qwen3-VL DocVQA RBM @25% cell（0.4239）来自 vllm-mimic
  `gate_pre25_docvqa`（= gate_pre25 official ANLS 0.4239，逐值相同）；native
  值（H0n/r2c 同款命令 `--mode pre --r-pre 0.25 --mrope native` docvqa 600k）
  为 **0.5924**。footnote (iii)“no reported cell comes from the degenerate
  path / 所有 n=200 cell family-correct native positions”**与此 cell 矛盾**。
  该行 Δ 由“FastV +16.2 F”变为“RBM 约 +0.6pp（≈parity，inconclusive）”，
  削弱正文“FastV 在全部 3 个 Qwen3 non-OCR cell 领先”的表述。
- 其余全部 main-table RBM cells = native + official；**无 raw `correct`
  写入正式表格**（表值均为 official VQA-acc/ANLS/exact-match/OCRBench）。
- 无 “deferred/cascade 阶段 mimic arm 误复用到 RBM 主表” 的其他案例；
  补充 S4b cascade gate 表（`runs/cascade/gate_pre25_*`）是 cascade 方法自身
  的 exploratory 结果，不属 RBM 主表，但其中 pre25 基同为 mimic（同属
  positional-convention 敏感，建议标注）。
- **次要：OCRBench skip 口径不一致（footnote ii vs 实际数字）**：Table 2
  Qwen3 OCRBench RBM 0.575 = skip-as-wrong（0.6354 × 181/200），而 footnote
  (ii) 声称 “scores are over attempted samples”。两臂同 skip 集，Δ 不受影响，
  但口径表述与数字不符，修订表注时应一并修正（over-attempted 应为 0.6354
  或明确 skip-as-wrong）。

## 8. 需要重跑的 cell 与预计 GPU 时间（本轮未自动重跑全量）

| cell | 问题 | 修复 | 预计 GPU |
| ---- | ---- | ---- | -------- |
| Table 2 Qwen3-VL DocVQA RBM @25%（n=200, 600k cap） | vllm-mimic 污染 | 替换为 native：**cell 已存在**（`runs/malt_goal_mode/h0n_docvqa_n200.json`，官方 ANLS **0.5924**；或按 r2c 样式重跑，命令 `--mode pre --r-pre 0.25 --mrope native --benchmark docvqa --subset eval/subsets/docvqa_200.jsonl --n 200 --seed 0 --max-pixels 600000`） | 0（复用现有 cell）；若重跑 ~2.5 min |
| 其余 Table 1/2/3 RBM cells | 无污染 | 无需重跑 | 0 |

即：**全表仅 1 个 cell 需替换/重跑，0–3 min GPU**（复用 H0n docvqa 则 0）。
修订表注 (iii) 的“no degenerate cell”表述以匹配该替换（论文正文修订留给
后续 user 决策轮，本轮不改正文）。

## 8.5 纠错路径核验（H1 无法复现时的排查项）

- **scorer 版本**：`official_scorers.py` 自 2026-07-23（d580508/a3e7780）以来
  未改（仅 chmod 100755，无内容变化）→ H1 官方分数逐位复现（§3）成立；
  r2c 表值（0.6683/0.520/0.5062/0.370/0.575）与本审计重算一致。
- **raw answer 文件**：Gate C `h1_*` 为 symlink → `sweep_deferred_K1_*_n200.json`
  （symlink 指向预期文件，realpath 已核）。H0n 为独立 fresh run（native）。
- **manifest**：全部 n=200 arms 用同一 `eval/subsets/{b}_200.jsonl`
  （hash：ae3e8f56dfe8 / a5a21dad7f98 / 71e48812ed1d / 05a5e806ab43）。
- **`correct` 是否被覆盖**：`extract_malt_features.py` 与 `audit_malt_*.py`
  只写独立输出（features_*.json / transitions.json / audit.json），**不回写**
  cell JSON；`correct` 保持运行期 ad-hoc 值。
- **answer normalization**：本审计与 sweep 均对 raw answer 直接跑
  official_scorers（VQA 内部做官方 normalize），无额外预处理。

## 9. 已同步修正的文件

- `STATE.md`：MALT-C 标记 HOLD（§0），禁止 GO/Pareto/删除时间唯一/raw=official
  表述。
- `experiments/malt_method_discovery.md`：§1 顶部审计修正横幅 + §9 GO→推翻 +
  §8 novelty 修正（含 IVC-Prune/Reroute 外部碰撞）。
- `experiments/malt_goal_mode_log.md`：新增 §2i 审计条目（HOLD + 结论推翻）。
- 新增 `scripts/audit_mrope_gateC_official_rescore.py`（复现脚本）。
- **不改论文正文**（Table 2 docvqa 替换与表注修订留给后续 user 决策轮）。

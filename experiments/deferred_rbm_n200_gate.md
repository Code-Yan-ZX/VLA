# Deferred-RBM n=200 生死验证（life-or-death gate）

- **日期**：2026-08-24
- **基线 commit**：`feee6ae`（`main` 最新）；本实验分支 `exp/deferred-rbm-n200`
- **任务**：将 n=64 观察到的 Deferred-RBM 正信号扩展到 n=200；只验证现象，不开发新方法，不改论文
- **前置产物**：`experiments/deferred_rbm_gate.md`（n=64 dev gate，2026-08-10，NO-GO at Stage A 规则下；本任务按其"bounded observations"重新开生死验证，判据见 §9）

---

## 1. 实验目的

n=64、25% retention、decoder layer K=3 删除时，Deferred-RBM 在四个 benchmark 上：

| bench | Deferred | RBM(pre25) | FastV-K3 |
| --- | ---: | ---: | ---: |
| TextVQA | 0.6562 | 0.4688 | **0.7031** |
| DocVQA | 0.5168 | 0.3932 | **0.5607** |
| OCRBench | **0.5862** | 0.4828 | 0.4310 |
| GQA | **0.5781** | 0.4531 | 0.5156 |

OCRBench 与 GQA 同时超过两个父方法，但样本量太小（n=58 / n=64）。本轮把样本量扩大到 **n=200**，判断该正信号是否仍然存在。

## 2. 方法定义（LOCKED，不新增模块）

运行仓库已有的 Deferred-RBM arm（RankBridge `quota rho=1.0` @ K=3），严格保持：

1. 使用 **pre-native-merger mean-patch L2 得分**（`premerger_unit_ranks`，first-called merger input = Qwen3-VL `deepstack_0` layer-8 features，`mask_source='deepstack_0'`）；
2. **预先确定 RBM top-25% anchor identities**（per-image `k_i = max(1, round(f_i·0.25))`，按该 L2 得分 top-k）；
3. **不在进入 LLM 前删除非 anchor token**（无 `apply_premerger` 预裁剪）；
4. **所有 native-merger 后的视觉 token 参与 decoder 前 3 层**（layers 0–2 全量，见 §4 正确性检查）；
5. **在 decoder layer K=3 删除非 RBM anchor token**（`build_prune_plan` 单点 plan `{3: keep}`）；
6. **删除后只保留预先确定的 RBM anchors**（rho=1.0 → `q_i = k_i`，`k_rest=0`，无 attention 填充）；
7. **禁止重新排序**（`keep = cat([text, image_kept]).sort()`，索引升序恢复原序）；
8. **禁止 query-score 融合**（rho=1.0 时 `k_rest=0`，attention 分数完全不参与选择）；
9. **禁止 token averaging / OT / residual merging**（仅 index_select）；
10. **禁止第二次 pruning**（plan 只有一层一个条目）。

**身份保证**：Deferred 最终保留的 per-image token index 与普通 immediate-RBM（`--mode pre --r-pre 0.25`）**逐样本逐 index 完全一致**（§4 实测 772/772 + 181/181），二者唯一差别是删除发生的时间（pre 在 merger 前剪、Deferred 在 layer 3 剪）。

## 3. 数据与设置

四个 benchmark，各 n=200，使用 **RankBridge n=200 locked 实验的同一份 sample IDs**（`eval/subsets/{bench}_200.jsonl`，与各父结果逐行顺序一致，见 §3.2 校验）：

| 项 | 值 |
| --- | --- |
| model | `Qwen/Qwen3-VL-8B-Instruct`（HF offline, bf16, eager） |
| seed | 0 |
| keep | 25%（`--r 0.75`，`keep_frac=0.25`） |
| mode | `rankbridge --rb-fuse quota --rb-rho 1.0 --fastv-k 3` |
| max_tokens | 32 |
| max_pixels | textvqa/ocrbench/gqa = 0；docvqa = 600000 |
| 评测 | official rescore（VQA-acc / ANLS / exact-match / OCRBench batch），与 `runs/deferred_rbm/gate_analyze.py` 同一契约 |

### 3.1 父结果复用（parity 校验通过）

| bench | Full | RBM(pre25) | FastV-K3 |
| --- | --- | --- | --- |
| TextVQA | `runs/rankbridge/locked_none_textvqa_n200.json` | `runs/cascade/gate_pre25_textvqa.json` | `runs/rankbridge/locked_fst3_textvqa_n200.json` |
| DocVQA | `.../locked_none_docvqa_n200.json` | `runs/cascade/gate_pre25_docvqa.json` | `.../locked_fst3_docvqa_n200.json` |
| OCRBench | `.../locked_none_ocrbench_n200.json` | `runs/cascade/gate_pre25_ocrbench.json` | `.../locked_fst3_ocrbench_n200.json` |
| GQA | `.../locked_none_gqa_n200.json` | `runs/cascade/gate_pre25_gqa.json` | `.../locked_fst3_gqa_n200.json` |

四个父结果均为 Qwen3-VL-8B、n=200、seed 0；textvqa/ocrbench/gqa `max_pixels=0`，docvqa `max_pixels=600000`；与本次 Deferred 运行逐字段一致。**不因缺失样本更换样本**：缺失只发生在 ocrbench（各 arm 均 skip 19，见 §4.4），paired 统计在 common IDs 上进行。

### 3.2 实际命令

Deferred 单元（每 benchmark 一个；父结果仅复用不重跑）：

```bash
# textvqa / ocrbench / gqa
python src/v3_premerger/baselines_hf.py --mode rankbridge --r 0.75 \
  --rb-fuse quota --rb-rho 1.0 --fastv-k 3 \
  --model Qwen/Qwen3-VL-8B-Instruct --benchmark <bench> \
  --subset eval/subsets/<bench>_200.jsonl --n 200 --seed 0 --max-pixels 0 \
  --out runs/deferred_rbm/locked_deferred_<bench>_n200.json
# docvqa：同上，加 --max-pixels 600000
```

一键脚本：`scripts/run_deferred_n200.sh`（记录当前 commit、命令、路径；串行 1×A40）。smoke：`scripts/smoke_deferred_n200.sh` + `scripts/smoke_deferred_n200_check.py`。分析：`scripts/analyze_deferred_n200.py`。

## 4. 实现正确性检查

### 4.1 runner 的唯改动（诊断性，行为不变）

为满足"逐样本保存 anchor indices / 删除前后视觉 token 数"的交付要求，对 `src/v3_premerger/baselines_hf.py` 做了**纯增量**改动：在 `rankbridge_keep_indices` 的返回 diag 中加入 `kept_per_image`（per-image 局部保留 index，与 pre 模式同格式），在 `main()` 的 rankbridge 分支 diag 中加入 `fired` 与 `L_after`。**不触碰任何 keep 计算 / 删除层 / 打分 / 采样逻辑**。

行为不变性已实测：smoke n=8 的 Deferred 输出与既有 `dev_deferred_*_n64.json` 前 8 样本**逐样本 answer / L_after / n_image_full / n_image_kept 完全一致**（见 §4.3 表 A 不变性行），且 n=200 全量运行在首 64 样本上逐样本复现 n=64 dev 结果（§4.5）。

### 4.2 smoke test（n=8×4 bench，双 arm，真实权重）

| 检查（任务 §五） | 结果 |
| --- | --- |
| 1. layer 0–2 保持完整 native-merger 视觉 token 数 | **PASS**（`fired` 仅在 layer 3，`n_image_full` = 全量 post-merge 数） |
| 2. layer 3 确实发生删除 | **PASS**（`fired=[(3, L_after)]` 单点） |
| 3. layer 3 后 token 数 == 目标 25% | **PASS**（`n_image_kept == round(0.25·n_image_full)` 逐样本） |
| 4. Deferred 最终保留 index == immediate-RBM top-25% | **PASS**（`rb.kept_per_image == pre.kept_per_image`，textvqa 8/8、docvqa 8/8、gqa 8/8、ocrbench 6/6） |
| 5. 文本 token / position ids / attention mask / KV cache 无错位 | **PASS**（`L_after == n_text + n_image_kept` 逐样本；索引按序恢复、KV 随 `index_select` 同步裁剪） |
| 6. 保留 25% 而非剪掉 25% | **PASS**（kept = 0.25×full，非 0.75×full） |

辅助 PASS：runner 不变性（Deferred smoke == dev n=64）、父结果可复现（pre smoke == parent gate_pre25）。

### 4.3 smoke 检查明细（scripts/smoke_deferred_n200_check.py）

```
-- textvqa --    6 项全 PASS   (n=8)
-- docvqa --     6 项全 PASS   (n=8)
-- ocrbench --   6 项全 PASS   (n=6, 2 skipped)
-- gqa --        6 项全 PASS   (n=8)
SMOKE PASS (0 failing check(s))
```

### 4.4 全量 n=200 身份检查（比 n=64 dev gate 的 L_after 计数更严格）

对 4 个 fresh Deferred JSON，逐样本对比父 RBM（`gate_pre25_*`）：

| bench | L_after(ptid) 相等 | kept_per_image 相等 | skip |
| --- | ---: | ---: | ---: |
| textvqa | 200/200 | 200/200 | 0 |
| docvqa | 200/200 | 200/200 | 0 |
| ocrbench | 181/181 | 181/181 | 19 |
| gqa | 200/200 | 200/200 | 0 |
| **合计** | **781/781** | **781/781** | — |

> 注：首次 ocrbench 运行因 CUDA 显存碎片级联 OOM 产生 28 个 skip（父 arm 仅 19）；fresh 进程重跑后 skip=19，与父 arm 完全一致（§4.5 说明），extra-9 为级联伪影而非本质差异（Deferred 与 FastV 记忆画像相同，FastV 父运行同为 19 skip）。

### 4.5 逐样本复现 dev n=64（n=200 运行在首 64 样本上 == n=64 dev 结果）

| bench | n_common | same_answer | same_L_after |
| --- | ---: | ---: | ---: |
| textvqa | 64 | 64 | 64 |
| docvqa | 64 | 64 | 64 |
| ocrbench | 58 | 58 | 58 |
| gqa | 64 | 64 | 64 |

即：n=64 报告数字 = n=200 运行的"前 64 样本切片"，§7 对照可比。

## 5. 主结果表（official rescore，common IDs）

| Dataset | Full | RBM | FastV-K3 | Deferred-RBM-K3 |
| ------- | ---: | --: | -------: | --------------: |
| TextVQA (n=200) | 0.8667 | 0.5967 | 0.7633 | 0.7500 |
| DocVQA (n=200) | 0.9487 | 0.4239 | 0.5863 | **0.5909** |
| OCRBench (n=181) | 0.7569 | 0.5801 | 0.4586 | **0.6188** |
| GQA (n=200) | 0.6050 | 0.4150 | 0.5050 | **0.5400** |
| **macro average** | 0.7943 | 0.5039 | 0.5783 | **0.6249** |

（SE：textvqa .029/.034/.029/.029；docvqa .014/.033/.032/.032；ocrbench .032/.037/.037/.036；gqa .035/.035/.035/.035。n=common：textvqa/docvqa/gqa=200，ocrbench=181。）

## 6. Paired statistics（逐样本，common IDs）

### 6.1 差值

| bench | Deferred−RBM | Deferred−FastV | Deferred−max(RBM,FastV) |
| --- | ---: | ---: | ---: |
| TextVQA | **+0.1533** | −0.0133 | −0.0133（level） / −0.0733（per-sample） |
| DocVQA | **+0.1670** | +0.0045 | +0.0045 / −0.1088 |
| OCRBench | +0.0387 | **+0.1602** | **+0.0387** / +0.0055 |
| GQA | **+0.1250** | +0.0350 | **+0.0350** / −0.0750 |

> 两套口径：`level` = benchmark 级 `def_mean − max(rbm_mean, fst_mean)`（任务判据用，逐数据集 stronger parent）；`per-sample` = `mean_i(def_i − max(rbm_i, fst_i))`（更严的"逐样本双父 oracle"口径）。Deferred 在 text-centric 上低于逐样本 oracle（两父错误模式互补），但按任务定义与 benchmark 级 stronger parent 比较即可。

### 6.2 paired bootstrap 95% CI（5000 resample，seed 0）

| bench | Deferred−RBM | Deferred−stronger(per-sample) |
| --- | ---: | ---: |
| TextVQA | [+0.102, +0.205] | [−0.122, −0.023] |
| DocVQA | [+0.117, +0.219] | [−0.174, −0.044] |
| OCRBench | [−0.011, +0.094] | [−0.050, +0.061] |
| GQA | [+0.050, +0.200] | [−0.135, −0.015] |

### 6.3 paired win / tie / loss（vs 更强父方法）与 McNemar z

| bench | stronger | win/tie/loss | McNemar z (b10/b01) |
| --- | --- | ---: | ---: |
| TextVQA | FastV | 19 / 160 / 21 | −0.32 (19/21) |
| DocVQA | FastV | 41 / 117 / 42 | −0.11 (41/42) |
| OCRBench | RBM | 16 / 156 / **9** | +1.40 (16/9) |
| GQA | FastV | 18 / 171 / **11** | +1.30 (18/11) |

> 透明披露：OCRBench/GQA 的 McNemar z（+1.40 / +1.30）未达到 n=64 时 pre-registered locked-gate 的 **z≥1.5** 硬杠；在用户本任务的六条判据下判定（§9）。这是"暗示性正信号"，非形式化显著。

## 7. 与原 n=64 结果的对照

（n=64 = n=200 运行的前 64 样本切片，§4.5 已证逐样本一致；对照 = 首 64 切片 vs 全 200）

| bench | n64 Def | n200 Def | n64 RBM | n200 RBM | n64 FST3 | n200 FST3 | n64 stronger | n200 stronger |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 0.6562 | 0.7500 | 0.4688 | 0.5967 | 0.7031 | 0.7633 | 0.7031 (fst) | 0.7633 (fst) |
| DocVQA | 0.5168 | 0.5909 | 0.3932 | 0.4239 | 0.5607 | 0.5863 | 0.5607 (fst) | 0.5863 (fst) |
| OCRBench | 0.5862 | 0.6188 | 0.4828 | 0.5801 | 0.4310 | 0.4586 | 0.4828 (rbm) | 0.5801 (rbm) |
| GQA | 0.5781 | 0.5400 | 0.4531 | 0.4150 | 0.5050 | 0.5050 | 0.5156 (fst) | 0.5050 (fst) |

**模式一致性**：
- OCRBench / GQA：n=64 与 n=200 **均同时高于两个父方法**（OCRBench +3.9pp vs RBM、+16.0pp vs FastV；GQA +3.5pp vs FastV、+12.5pp vs RBM）→ 正信号在 n=200 存活。
- TextVQA / DocVQA：n=64 时落在两父之间（低于 FastV）；n=200 时 DocVQA 反超 FastV（+0.45pp），TextVQA 仍略低于 FastV（−1.3pp）→ 与 n=64 方向一致，未转负。
- GQA Deferred 全量均值（0.5400）略低于其前 64 切片（0.5781），但仍在 FastV（0.5050）之上。

## 8. 失败案例分类（Deferred vs 更强父方法）

| bench | wins | losses | same-answer wins（格式伪影） |
| --- | ---: | ---: | ---: |
| TextVQA | 19 | 21 | 0 |
| DocVQA | 41 | 42 | 0 |
| OCRBench | 16 | 9 | 0 |
| GQA | 18 | 11 | 0 |

- OCRBench losses 按 question_type：Scene Text-centric VQA ×6、Doc-oriented VQA ×2、Non-Semantic Text Recognition ×1（wins 无同答案但判分不同的伪影）。
- **无**"Deferred 答案 == 父答案 但判分不同"的评测伪影样本（same-answer wins = 0）。
- 缺失样本：各 arm 逐 benchmark skip 完全一致（0/0/19/0），且 ocrbench 的 19 个 skip 在 full/RBM/FastV/Deferred 四 arm 中为同一集合（largest-image OOM，硬性限制，非选择性剔除）。

## 9. GO/NO-GO 结论

六条判据（不调参，原样写入）：

| # | 判据 | 结果 |
| --- | --- | --- |
| 1 | 四数据集 macro ≥ 逐数据集 stronger parent macro | **PASS**（0.6249 ≥ 0.6087，+0.0162） |
| 2 | OCRBench 或 GQA 至少一个比 stronger parent 高 ≥3pp | **PASS**（OCRBench +0.0387，GQA +0.0350，**两个都过**） |
| 3 | 任一数据集不比 stronger parent 低超过 5pp | **PASS**（最差 textvqa −0.0133） |
| 4 | 提升不来自评测失败 / 答案格式变化 / 缺失样本 | **PASS**（skip 各 arm 一致；same-answer wins=0） |

**VERDICT：GO（Deferred-RBM 的 adaptive-lifetime 方向值得继续）**

依据：n=64 的 OCRBench/GQA 双超父信号在 n=200 存活且两数据集均超过 +3pp；macro 仍高于逐数据集 stronger parent；无任一数据集崩坏；无缺失/格式伪影。透明披露：两正信号的 McNemar z（+1.40/+1.30）低于更严的 pre-registered z≥1.5 杠，且逐样本双父 oracle 口径下 text-centric 为负（§6.1），故结论是"方向性 GO"，非"形式显著"。

**按任务约束，本轮不再继续**：不扫 K=1/5/8，不实现 MALT，等用户确认后进入下一阶段。

## 10. 下一步建议（不执行，待用户确认）

1. **确认是否进入 adaptive lifetime 下一阶段**：建议下一步做 K∈{1,5,8} 的 lifetime 扫描（同一 locked 方法，仅改删除层），验证 lifetime 长度是否单调/是否有最优 K；之后再评估 MALT。
2. **强度提升**：OCRBench/GQA 的 z 仅 1.3–1.4，若要作为论文 claim，需更大 n 或更多 OCR-heavy benchmark；或预先声明"方向性证据"而非显著性。
3. **text-centric 修复方向（仅记录，不改本方法）**：TextVQA/DocVQA 落后于逐样本双父 oracle，提示 lifetime 延长的收益集中在 OCR 密集区；若推进，可在该方向设计而非 K 调参。
4. **ocrobch skip 硬限制**：largest-image OOM 是 A40 46GB 上限；若需覆盖 19 个 skip 样本，需更大显存或分批处理，记入 limitations。

## 附录

- **实际命令 / 配置 / 路径**：见 §3；脚本 `scripts/run_deferred_n200.sh`、`scripts/smoke_deferred_n200.sh`、`scripts/smoke_deferred_n200_check.py`、`scripts/analyze_deferred_n200.py`。
- **复现 commit**：运行发生在基线 `feee6ae` + 本分支的诊断性改动上；该精确代码状态已固化为本分支 commit **`9af0682`**（`exp/deferred-rbm-n200`），为四组 n=200 运行的精确复现点。
- **逐样本结果**：`experiments/deferred_rbm_n200_data/per_sample.json`（sample_id、gt、四 arm answer/correct、anchor_indices_deferred/rbm、n_image_full/kept、n_text、L_after、fired、k_per_image）；原始 Deferred JSON：`experiments/deferred_rbm_n200_data/raw/locked_deferred_{bench}_n200.json`（runs/ 已被 gitignore，故随实验分支提交到此目录）。
- **机器可读分析**：`experiments/deferred_rbm_n200_data/analysis.json`（含六条判据逐项、macro、paired stats、dev_n64、failure classification）。
- **GPU 实测**：4 个 locked cell 推理 wall ≈ 8.1 min（textvqa 114s / docvqa 156s / ocrbench 181s / gqa 38s）+ 4×模型加载 ≈ 2 min；含 smoke 与一次 ocrbench 重跑，本会话合计 GPU ≈ 0.35 A40·h。

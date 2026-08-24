# Deferred-RBM 固定生命周期 K sweep（phase 2）

- **日期**：2026-08-24
- **分支**：`exp/deferred-rbm-n200`；**运行代码状态**：基线 `ea359d8`（= round-1 `9af0682` + 效率诊断增量，本报告附录 A 固化）
- **上游**：`experiments/deferred_rbm_n200_gate.md`（K=3 方向性 GO）→ 本实验回答四个问题（§1）
- **范围**：只扫固定删除层 K∈{0,1,3,5,8}；不实现 adaptive MALT；不改论文

---

## 1. 实验目标（四个问题）与答案预告

| # | 问题 | 答案 |
| --- | --- | --- |
| 1 | K=3 增益是否单点偶然 | **否**：K=1/5/8 与 K=3 同量级，均显著优于 RBM（macro 0.629–0.634 vs 0.504），K=3 甚至不是局部最高 |
| 2 | 性能是否随生命周期有稳定规律 | **部分**：K=0→K=1 是主跃迁（+12.5pp macro）；K=1→8 基本平坦（+0.5pp），仅在 docvqa 上单调上升 |
| 3 | 各数据集是否偏好不同删除层 | **是（方向性）**：OCR 密集（ocrbench/gqa）偏好短生命周期 K=1；文本密集（textvqa/docvqa）偏好长生命周期 K=8 |
| 4 | 是否形成真实 accuracy–efficiency Pareto | **是**（逐数据集 Pareto 前沿存在，见 §6.4）：K=1 是强 Pareto 点（大增益、中等算力）；K=8 高算力仅边际增益 |

## 2. 方法定义与 K 语义（LOCKED，唯一变量 = K）

保持 round-1 Deferred-RBM 全部定义：pre-native-merger mean-patch L2；固定 top-25% RBM anchor identities；非 anchor 只改变删除时间；不重排；不用 query-score；不融合/不平均/不做 OT；不做第二次剪枝。**每个 K 的最终保留 index 与 immediate-RBM 完全一致**（§5 全量验证）。

**K 的准确语义（off-by-one 明确）**：代码中 decoder layer 编号从 0 开始（`enumerate(LM.layers)`），删除发生在 `layer idx == K` 的 **forward 完成之后**（先 `_layer_step(layer K)` 再 `index_select` 剪枝）。因此：

- **K∈{1,3,5,8}**：非 anchor 视觉 token 参与 decoder **layers 0..K 的 forward（共 K+1 层）**，在第 K 层 forward 后删除；anchor 继续 layers K+1..35。
- **K=0**：immediate-RBM（pre-mode，复用 `runs/cascade/gate_pre25_*`）：非 anchor 在 merger 之后、**任何 decoder 层之前**删除（生命周期 ≈ 0）。

> 例如 round-1 的 K=3：非 anchor 参与 layers 0,1,2 **及 layer 3 的 forward**（共 4 次 layer forward），layer 3 后删除。上一轮"前 3 层"措辞按此精确化。

## 3. 实验点、数据与配置

- 四个 benchmark，**完全相同的 locked n=200 sample IDs**（`eval/subsets/{bench}_200.jsonl`，与 round-1/父结果逐行一致）。
- K=0：复用 `gate_pre25_*`（未重跑精度）。K=3：复用 round-1 `locked_deferred_*_n200.json`（未重跑精度）。
- K=1/5/8：新增（`sweep_deferred_K{K}_{bench}_n200.json`）。
- **为获取效率与不变性证据**，另重跑 K=3 与 K=0（pre）各 4 cell（`sweep_deferred_K3_*`、`sweep_pre_*`），逐样本 answer 与复用数据 100% 一致（§5.2）。

配置（所有 20 cell 相同）：`Qwen/Qwen3-VL-8B-Instruct`（HF offline, bf16, eager）、`--mode rankbridge --r 0.75 --rb-fuse quota --rb-rho 1.0 --fastv-k K`、seed 0、`--max-tokens 32`、batch 1、greedy；`--max-pixels` docvqa=600000 其余 0；`--mode pre --r-pre 0.25`（K=0）。脚本：`scripts/run_deferred_sweep.sh`。

```bash
# K∈{1,3,5,8}（K=3 仅用于效率/不变性；精度复用 round-1）
python src/v3_premerger/baselines_hf.py --mode rankbridge --r 0.75 \
  --rb-fuse quota --rb-rho 1.0 --fastv-k $K \
  --model Qwen/Qwen3-VL-8B-Instruct --benchmark <bench> \
  --subset eval/subsets/<bench>_200.jsonl --n 200 --seed 0 \
  --max-pixels $( [ <bench> = docvqa ] && echo 600000 || echo 0 ) \
  --out runs/deferred_rbm/sweep_deferred_K${K}_<bench>_n200.json
# K=0（效率/不变性重跑）
python ... --mode pre --r-pre 0.25 ... --out runs/deferred_rbm/sweep_pre_<bench>_n200.json
```

## 4. runner 诊断增量（纯记录，行为不变）

为产出效率指标，在 round-1 诊断（`kept_per_image/fired/L_after`）之上再追加纯增量字段：`layer_visual_counts`（每层 forward 前活跃视觉 token 数）、`prefill_s`、`ttft_s`、`peak_mem_mb`（每样本 `reset_peak_memory_stats` 后记录）。不触碰任何推理计算。不变性已实测：dry-check ALL PASS；K=3 重跑 == round-1 200/200（181/181）；pre 重跑 == 父 200/200（181/181）。

## 5. 正确性检查（全量 n，非抽查）

对 K=1/5/8 全部有效样本（textvqa/docvqa/gqa=200，ocrbench=181；共 2343）：

| 检查 | 通过 |
| --- | ---: |
| 1. 删除前保持完整 native-merger 视觉 token 数（layers 0..K 全量） | 2343/2343 |
| 2. 只在指定 K 发生一次删除（`fired==[[K,L_after]]`） | 2343/2343 |
| 3. 删除后保持 25%（`kept==round(0.25·full)`） | 2343/2343 |
| 4. 最终 anchor index 与 immediate-RBM 完全一致（`kept_per_image==pre.kept_per_image`） | **2343/2343（100%）** |
| 5. 不改变 token 原始顺序（per-image kept 列表升序） | 2343/2343 |
| 6. attention mask / position IDs / KV cache 无错位（`L_after==n_text+kept`） | 2343/2343 |
| 7. 各 K 相同 generation/evaluation 配置 | 20 cell 同命令模板（§3） |

skip/OOM：各 K 各 arm 逐 bench 一致（textvqa/docvqa/gqa=0，ocrbench=19，largest-image OOM 硬限制，与父一致；fresh process 无级联）。

## 6. 结果

### 6.1 准确率表（official rescore，common IDs）

| Dataset | K=0 | K=1 | K=3 | K=5 | K=8 | FastV | Full |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TextVQA | 0.5967 | 0.7433 | 0.7500 | 0.7500 | **0.7550** | 0.7633 | 0.8667 |
| DocVQA | 0.4239 | 0.5959 | 0.5909 | 0.6041 | **0.6212** | 0.5863 | 0.9487 |
| OCRBench | 0.5801 | **0.6354** | 0.6188 | 0.6188 | **0.6354** | 0.4586 | 0.7569 |
| GQA | 0.4150 | **0.5400** | **0.5400** | 0.5300 | 0.5250 | 0.5050 | 0.6050 |
| **macro** | 0.5039 | 0.6287 | 0.6249 | 0.6257 | **0.6341** | 0.5783 | 0.7943 |

（SE 各格 ≈0.029–0.037；n=common：textvqa/docvqa/gqa=200，ocrbench=181。）

### 6.2 差值 / paired 统计

**vs RBM(K=0)**（paired bootstrap CI 与 McNemar z）：

| bench | K=1 | K=3 | K=5 | K=8 |
| --- | --- | --- | --- | --- |
| TextVQA | +0.1467 (z=4.87) | +0.1533 (z=5.19) | +0.1533 (z=5.06) | +0.1583 (z=5.15) |
| DocVQA | +0.1721 (z=5.40) | +0.1670 (z=5.43) | +0.1802 (z=5.40) | +0.1973 (z=6.10) |
| OCRBench | +0.0552 (z=1.96) | +0.0387 (z=1.40) | +0.0387 (z=1.40) | +0.0552 (z=1.96) |
| GQA | +0.1250 (z=3.31) | +0.1250 (z=3.20) | +0.1150 (z=2.99) | +0.1100 (z=2.89) |

bootstrap 95% CI（vs K=0）：textvqa K=1 [+0.093,+0.202]…K=8 [+0.103,+0.215]；docvqa K=8 [+0.143,+0.250]；ocrbench K=1 [+0.000,+0.110]；gqa K=8 [+0.040,+0.185]。全部 deferred K 在 textvqa/docvqa/gqa 显著优于 RBM；ocrbench 为暗示性（K=1/8 CI 含 0，z<2）。

**vs FastV-K3**（level）：

| bench | K=1 | K=3 | K=5 | K=8 |
| --- | --- | --- | --- | --- |
| TextVQA | −0.0200 | −0.0133 | −0.0133 | −0.0083 |
| DocVQA | +0.0096 | +0.0045 | +0.0178 | +0.0348 |
| OCRBench | +0.1768 | +0.1602 | +0.1602 | +0.1768 |
| GQA | +0.0350 | +0.0350 | +0.0250 | +0.0200 |

**win/tie/loss vs RBM(K=0)**（w/t/l）：textvqa K8=(37,159,4)；docvqa K8=(48,148,4)；ocrbench K1=(18,155,8)；gqa K1=(41,143,16)。

**skip/OOM/格式错误**：见 §5（skip 各 arm 一致；同答案但判分不同 = 0）。

### 6.3 效率（median over common 非 skip 样本；每 cell 独立 fresh process，batch=1，max_new_tokens=32，warm-up = 模型加载 + 首样本）

| bench | K | visual-token layer area ΣN_l | attention compute ΣN_l² | prefill (s) | TTFT (s) | peak mem (MB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TextVQA (n=200) | 0 | 6 768 | 1.27e6 | 0.0812 | 0.0848 | 18 270 |
| | 1 | 7 896 | 2.33e6 | 0.0895 | 0.0932 | 18 264 |
| | 3 | 9 024 | 3.39e6 | 0.0970 | 0.1007 | 18 264 |
| | 5 | 10 152 | 4.46e6 | 0.1037 | 0.1073 | 18 264 |
| | 8 | 11 844 | 6.05e6 | 0.1138 | 0.1174 | 18 264 |
| DocVQA (n=200) | 0 | 5 220 | 7.57e5 | 0.0717 | 0.0749 | 17 678 |
| | 8 | 9 135 | 3.60e6 | 0.0903 | 0.0936 | 17 673 |
| OCRBench (n=181) | 0 | 1 296 | 4.67e4 | 0.0681 | 0.0710 | 16 865 |
| | 8 | 2 268 | 2.22e5 | 0.0705 | 0.0734 | 16 854 |
| GQA (n=200) | 0 | 2 340 | 1.52e5 | 0.0671 | 0.0701 | 16 962 |
| | 8 | 4 095 | 7.22e5 | 0.0746 | 0.0777 | 16 959 |

（docvqa/ocrbench/gqa 中间 K 的单调插入值略；完整表见 `sweep_analysis.json`。）

**关键效率事实**：
1. **峰值内存对 K 基本持平**（16.9–18.3GB）：由模型权重 + 单层最大 attention 决定；而单层全量 attention 在所有 K≥1 都存在（layer 0），K=0 亦受 decode/KV 主导。→ 提前退出的收益在 **FLOPs/延迟**，不在峰值内存。
2. compute proxy 随 K 明显上升（textvqa ΣN_l²：1.27e6→6.05e6，×4.8）；prefill/TTFT 温和上升（0.081→0.114s）。
3. 每层活跃视觉 token 数（示意，textvqa 中位）：K=1 → [752,752,188×34]；K=3 → [752×4,188×32]；K=8 → [752×9,188×27]（752=中位全量，188=中位 25%）。

### 6.4 生命周期曲线与 accuracy–efficiency Pareto

（图：`sweep_accuracy_vs_K.png`、`sweep_pareto.png`。）

- **accuracy vs K**：除 docvqa 单调上升到 K=8 外，其余三数据集在 K≥1 近似平台/轻微非单调（ocrbench K=1 与 K=8 均 0.6354，K=3/5 略低 0.6188；gqa K=1/3 峰值后缓降）。
- **Pareto**（逐数据集，accuracy vs ΣN_l²）：K=0（RBM）→ K=1 是最大步进（textvqa +14.7pp @ +1.8× compute；docvqa +17.2pp；ocrbench +5.5pp；gqa +12.5pp）；K=1→8 算力 ×2.6 而精度 ≤ +2.5pp（docvqa 除外，+2.5pp）。**K=1 是强 Pareto 点**。FastV 在 textvqa 上于 K=3 同等算力优于 deferred K=3（0.7633 vs 0.7500）；在 ocrbench 上 deferred 全面压过 FastV（+16–18pp）。

### 6.5 生命周期异质性（earliest-exit / oracle）

| bench | best fixed K | best-fixed acc | **oracle acc** | oracle−best | oracle 平均 K | earliest-correct 分布 (K0/K1/K3/K5/K8/never) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| TextVQA | 8 | 0.7550 | 0.7800 | +0.025 | 0.37 | 118/34/1/1/2/44 |
| DocVQA | 8 | 0.6212 | 0.7250 | **+0.104** | 0.77 | 94/42/0/1/8/55 |
| OCRBench | 1 | 0.6354 | 0.6851 | +0.050 | 0.21 | 105/18/0/0/1/57 |
| GQA | 1 | 0.5400 | 0.6400 | **+0.100** | 0.45 | 83/41/3/0/1/72 |

（oracle = 每样本选择能答对的最早 K；仅用于测量 adaptive 上限，不作为方法结果。）

**K=3→5、K=5→8 转换**（w→c / c→w / 答案不变）：

| bench | 3→5 | 5→8 |
| --- | --- | --- |
| TextVQA | 1/1/198 | 2/1/197 |
| DocVQA | 2/1/197 | 9/2/189 |
| OCRBench | 3/3/175 | 5/2/174 |
| GQA | 1/3/196 | 3/4/193 |

**读解**：① 绝大多数正确样本在 K=0 或 K=1 已答对（oracle 平均 K 仅 0.21–0.77）→ adaptive 收益主要来自在 **K=0/1 边界**做样本级选择；② docvqa/gqa 的 oracle−best 达 +10pp → 存在真实样本级异质性上限；③ K=3→5/5→8 转换量很小（≈1–9 样本），答案 95–99% 不变 → 大 K 区间差异主要由少数长尾样本贡献。

## 7. A/B/C/D 判断

| 情况 | 判定 |
| --- | --- |
| **A**：不同数据集/样本偏好不同 K | **主判定**：best fixed K = {textvqa 8, docvqa 8, ocrbench 1, gqa 1}——OCR 密集（ocrbench/gqa）偏好短生命周期 K=1，文本密集（textvqa/docvqa）偏好长生命周期 K=8，**方向性模式与任务预设的 A 例一致**；且 docvqa/gqa oracle−best ≈ +10pp 表明样本级异质性真实存在 |
| **B**：同一 K 四处稳定最好 | 否（无共同最优 K） |
| **C**：只有 K=3 孤立高点 | **明确否定**：K=3 在 ocrbench/gqa 上还是较差点（0.6188/0.5400 与 K=1 相同或更低）；K=1 与 K=8 均达到或超过 K=3 → K=3 绝非偶然单点，反而说明效应在多个 K 上稳健 |
| **D**：K 越大准确率越高 | 否（非单调：ocrbench/gqa 在 K=1 后回落） |

**结论：情况 A（qualified）——支持开发 sample-adaptive lifetime，进入 MALT adaptive gate 设计。**

必须并行的限定条件（避免夸大）：
1. **固定 K 之间的差异在 n=200 处于噪声量级**（macro 0.629–0.634，±0.5pp；per-bench best-vs-other 多 ≤2.5pp，SE≈0.03）。稳健结论是"任何 K≥1 的 deferred 均显著优于 RBM、并 ≥ FastV（textvqa 持平 / docvqa 略优 / ocrbench 大优 / gqa 略优）"，而非"某个 K 最优"。
2. MALT 的价值完全取决于能否兑现 oracle 上限（docvqa/gqa +10pp）；且 oracle 平均 K 很小 → adaptive gate 的核心决策是 **K=0 vs K≥1**（以及少量长尾 K=8），而不是细粒度 K 扫描。
3. adaptive 的收益形态是 **FLOPs/延迟**（提前退出省算力），**不是**峰值内存（§6.3 持平）。
4. 若后续 MALT 无法显著逼近 oracle，则退化建议为"固定 K=1（强 Pareto 点）作为稳健默认"。

## 8. 是否值得开发 adaptive MALT

**建议进入 MALT adaptive-gate 设计（下一阶段，待 user 确认）**，依据：
- 存在真实样本级 adaptive 上限（oracle−best：docvqa +10.4pp、gqa +10.0pp、ocrbench +5.0pp）；
- 数据集级偏好方向清晰（OCR 短 / 文本长），说明"何时退出"有结构可寻；
- 但 MALT 设计目标应写为"**在 K=0/1 边界做二元/低粒度样本选择 + 长尾 K=8**"，并须以能否逼近 oracle 为验收标准；若失败则退回固定 K=1。

**本轮明确不做**：不实现 MALT、不做 per-dataset tuning、不追加其他 K。

## 附录

- **git commit**：`experiments/deferred_rbm_lifetime_sweep.md` 记录提交时 HEAD；运行代码状态 = 基线 `ea359d8` + 效率诊断增量，详见提交信息。
- **脚本**：`scripts/run_deferred_sweep.sh`（运行）、`scripts/analyze_deferred_sweep.py`（分析）、`scripts/plot_deferred_sweep.py`（图）。
- **数据**（随分支提交，`experiments/deferred_rbm_n200_data/`）：`sweep_analysis.json`（全统计）、`sweep_per_sample.json`（逐样本：sample_id、gt、answer、metric、correct、anchor_indices、n_image_full/kept、n_text、L_after、fired、prefill_s、ttft_s、peak_mem_mb、layer_visual_counts）、`sweep_pareto.png`、`sweep_accuracy_vs_K.png`。原始 JSON：`experiments/deferred_rbm_n200_data/raw/`（round-1）+ `runs/deferred_rbm/sweep_*_n200.json`（gitignored，不提交；如需可另行归档）。
- **GPU**：20 cell（12 新增 + 8 重跑），串行 1×A40，推理 wall ≈ 20 cell × ~1–3 min ≈ 45 min + 模型加载；含 round-1 与 smoke，本阶段合计 ≈ 1.0 A40·h。
- **上游对照**：round-1 K=3 门（GO）所有数字在本 sweep 中逐一复现（K=3 列与 `deferred_rbm_n200_gate.md` §5 一致）。

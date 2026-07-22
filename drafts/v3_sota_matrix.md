# V3 SOTA Decision Matrix — pre-merger vs post-merger vs VisionZip-style（FINAL 2026-07-22）

> Model **Qwen3-VL-8B-Instruct**（bf16, enforce_eager, 1×A40, vLLM 0.19 V1）· selector **L2-norm text-agnostic**（控变量隔离 stage 效应）· greedy temp=0 → 误差棒 = binomial stderr √(p(1−p)/n)，gap σ=√(σ₁²+σ₂²) · n=200（seed=0 固定子集）· retention = keep% of merge-units，pre/post iso-token（同 r 下 mean_ptid 相等已核）。
> 新 cell 全部来自 `runs/v3_sota_matrix/`（ChartQA/OCRBench iso-config: mnbt 32768 / max-pixels 1.5M / mns 4；GQA suite-config: mns 16）。ChartQA/OCRBench scorer = lmms-eval 逐式移植（chartqa: numeric ±5% relaxed；ocrbench: normalized containment），gt 自测 200/200。

## 0. Headline — keep 25%（r=0.75, iso-token）

| Benchmark | tier | n | baseline A | **pre-merger (ours)** | post-merger (L2) | VisionZip-style (post dom+ctx) | gap pre−post | z |
|---|---|---|---|---|---|---|---|---|
| TextVQA | text-dense (scene) | 200 | 0.820 ± .027 | **0.695 ± .033** (ptid 203) | 0.255 ± .031 (ptid 203) | 0.255 ± .031 | **+44.0pp** | 9.8σ |
| DocVQA | text-dense (doc) | 200 | 0.770 ± .030 | **0.725 ± .032** (ptid 1054) | 0.390 ± .035 (ptid 1054) | 0.390 ± .035 | **+33.5pp** | 7.0σ |
| OCR-Bench | text-dense (mixed) | 200 | 0.760 ± .030 | **0.580 ± .035** (ptid 258) | 0.165 ± .026 (ptid 258) | 0.165 ± .026 | **+41.5pp** | 9.5σ |
| ChartQA | detail-dense (chart) | 200 | 0.820 ± .027 | 0.190 ± .028 (ptid 142) | 0.190 ± .028 (ptid 142) | 0.190 ± .028 | 0.0pp | 0 |
| GQA | object/spatial | 200 | 0.415 ± .035 | 0.320 ± .033 (ptid 85) | **0.380 ± .034** (ptid 85) | 0.380 ± .034 | −6.0pp | 1.3σ |

- TextVQA n=500 复核同向：pre 0.738 ± .020 / post 0.272 ± .020（+46.6pp, 16.7σ）。
- **VisionZip-style ≡ post dom-only：11/11 cell 完全一致**（GQA/ChartQA/OCRBench ×{@25,@12.5} + TextVQA/DocVQA ×{@25,@12.5}，post-mode dom-ratio 0.7）→ context tokens（位置连续组均值）在全部 workload/深度零增益；post-merger 崩溃由 **stage（merger 后选择）** 单独解释，与 dom/ctx 配比无关。

## 1. keep 12.5%（r=0.875, 深压）

| Benchmark | baseline A | pre | post | VZ-style (post) | gap pre−post | z |
|---|---|---|---|---|---|---|
| TextVQA | 0.820 | **0.615 ± .034** (ptid 111) | 0.175 ± .027 | 0.175 ± .027 | **+44.0pp** | 10.6σ |
| DocVQA | 0.770 | **0.610 ± .035** (ptid 538) | 0.135 ± .024 | 0.135 ± .024 | **+47.5pp** | 11.2σ |
| OCR-Bench | 0.760 | **0.380 ± .034** (ptid 140) | 0.075 ± .019 | 0.075 ± .019 | **+30.5pp** | 7.8σ |
| ChartQA | 0.820 | 0.150 ± .025 (ptid 88) | 0.095 ± .021 | 0.095 ± .021 | +5.5pp | 1.7σ |
| GQA | 0.415 | 0.250 ± .031 (ptid 52) | **0.305 ± .033** | 0.305 ± .033 | −5.5pp | 1.2σ |

- text-dense 三 bench：越深压 gap 越大或保持（ratio 扩大：OCRBench retention pre/post = 3.5×→5.1×）。DocVQA clean post@12.5 源 = `runs/v3_tighten_cells/B_docvqa_r0.875_l2_n500.json`（文件名 n500 系误标，实 n=200/answered 200/mnbt32768；v3_premerger_cells 内同名 0.0 为修复前崩溃值，弃用）。

## 2. ChartQA 深度扫描 — budget-dominated 第三 regime（新发现）

| keep | pre | post | gap | 解读 |
|---|---|---|---|---|
| 50% (ptid 252) | 0.390 ± .035 | 0.335 ± .033 | +5.5 (1.1σ) | 50% 即已崩（baseline 0.82 → 48% retention） |
| 25% (ptid 142) | 0.190 ± .028 | 0.190 ± .028 | 0.0 | 三方同分（VZ 亦 0.190） |
| 12.5% (ptid 88) | 0.150 ± .025 | 0.095 ± .021 | +5.5 (1.7σ) | 弱 pre 趋势，不显著 |

- **失败模式（per-sample 核实，非 bug）**：压缩下模型停止吐数字、转 hedging 长答（pred 均词数 1.1→7.7，"Based on the chart…"），或读邻近错值（"3"vs"4"、"Jul'21"vs"May'21"）。pre/post 各对 38 题但**仅 20 重叠**——同分巧合、失败集不同。
- 解读：chart = 多数值算术 + 细 bar/label 辨别，信息**全局分布**，L2 局部显著性无从优先 → 总预算主导、stage 次要。对比 TextVQA（单值局部定位）@50% pre 仍 0.75（91% baseline）。
- 论文定位：**诚实第三 regime**（stage law 管"何处 pre 胜"，budget 管"压缩是否可行"），不 overclaim 全 workload 单调。ChartQA human/augmented 分裂一致（aug .942→.231/.212；hum .688→.146/.167）。

## 3. OCR-Bench 子技能分解（@25%；per-type n 小，定性用）

| question_type | n | baseline | pre | post | 行为 |
|---|---|---|---|---|---|
| Regular Text Recognition | 9 | 1.000 | **1.000** | 0.000 | stage 灾难（post 全灭） |
| Non-Semantic Text Rec. | 12 | 0.917 | **0.917** | 0.000 | 同上 |
| Irregular Text Rec. | 11 | 0.909 | **0.909** | 0.000 | 同上 |
| Artistic Text Rec. | 13 | 1.000 | **0.923** | 0.077 | 同上 |
| Scene Text-centric VQA | 39 | 0.769 | **0.718** | 0.179 | stage 大胜 |
| Doc-oriented VQA | 42 | 0.405 | **0.452** | 0.119 | pre ≥ baseline |
| Key Info Extraction | 41 | 0.829 | **0.341** | 0.195 | 两者皆崩（空间精度敏感） |
| Handwritten Math Expr | 21 | 0.667 | 0.238 | 0.238 | **同分**（budget regime，类 ChartQA） |
| Digit String Rec. | 8 | 0.625 | 0.500 | 0.625 | n=8 噪声，勿过度解读 |
| Handwriting Rec. | 4 | 0.750 | 1.000 | 0.500 | n=4 |

- **stage-vs-budget 分裂下沉到子技能级**：纯文字识别 = pre 保住 baseline、post 清零（最强 stage 证据）；公式/KIE = budget regime（类 ChartQA）。写作可作机制节显微镜证据。

## 4. Official / faithful SOTA 列 — audit 结论

**VisionZip**（`JIA-Lab-research/VisionZip`, arXiv 2412.04467, CVPR'25；详 `drafts/visionzip_gap_report.md`）：
- **判定 (c) 本机不可跑**：仅 CLIP-LLaVA 系 + Qwen2.5-VL（HF monkey-patch, bs=1, eager attn；无 Qwen3-VL、无 vLLM；Qwen 变体需 ViT attention 物化 → DocVQA 16k-token 图 OOM 风险）。
- 代码级再确认 **post-merger**（Qwen 变体在 PatchMerger 后对 inputs_embeds 选择；dom = ViT 末层 received-attention，Qwen 配 65%dom+5%ctx）。
- **作者自家 README（Qwen2.5-VL）即是软肋证据**：50% keep 时 **OCRBench 81.5→70.5（−13%）** 而 DocVQA 仅 −1.3 —— text-dense 崩溃在官方数字已 onset；25% 官方未公布。
- 我方 proxy 差异：stage（官方 post；我方 post-mode 对齐）+ L2 scores（vs attention）+ 连续组均值 ctx（vs key-cosine）+ dom-ratio 0.7（vs 0.84–0.93）。**ctx 实现差异经 11/11 cell 证无关**；bias 对我方保守（attention 显著性只会更强）。
- 论文列处理：SOTA 列 = "VisionZip principle, same-model port (ours)"；官方 LLaVA/Qwen2.5 数字作 **model/stage-mismatched reference**（引用其 README OCRBench 行 + "gain less striking on Qwen2.5-VL due to PatchMerger" 自述）。

**QuietPrune / Hi-Lo Prune / IF-Prune**（详 `drafts/baseline_methods_audit.md`）：
- **三者均不可作同 budget 公平 Qwen3-VL-8B baseline**：QuietPrune 无代码；Hi-Lo Prune 空仓（13B README，但论文支持 Qwen2/2.5/3-VL → **最佳未来 baseline，挂 watch**）；IF-Prune 有码但 post-merger + 无 Qwen3-VL + HF-only + 需训估计器（移植 20–60 GPU·h，超 5–15×）。
- 决策：本轮不跑 IF-Prune InternVL2.5-1B recipe（模型失配、非公平）；matrix SOTA 列以 VisionZip 原则 + 我方 same-model port 为准，缺口如实报。

## 5. 定性证据
`drafts/qualitative_examples.md`（10 例，图路径已验）：5 TextVQA + 3 DocVQA flip（pre✓/post✗）+ 1 both-correct 对照 + 1 GQA post✓/pre✗ 诚实对照。典型：DocVQA "$1.3 BILLION" → post **与 VZ 均**读成 "$1.3 million"（1000× 单位错）、pre 读对；TextVQA "NORTEL"→post "PlayStation PS"、"date on right page"→post "no visible date"（text-as-absent 签名）。

## 6. 来源登记 + caveats
| 资产 | 路径 |
|---|---|
| 新 cell（16+3+2=21 json） | `runs/v3_sota_matrix/`（gitignored；每 cell 含 acc/mean_ptid_len/n/diag + per_sample） |
| ChartQA/OCRBench 子集 | `eval/subsets/{chartqa,ocrbench}_200.jsonl`（lmms-lab/ChartQA test 中 96hum/104aug；echo840/OCRBench test）+ scorer in `src/serve_bench.py`/runner（commit f23841a） |
| 旧 bench cell | `runs/v3_premerger_cells/`（6-bench suite）· `runs/v3_tighten_cells/`（docvqa @12.5 clean） |
| 运行脚本 | `src/v3_premerger/v3_sota_matrix.sh` + `v3_sota_matrix_followup.sh` |

**Caveats（写作红线）**：① stage law 报 coarse 三层（text-dense +33~44 ≫ ChartQA ≈0 budget-regime > object −6），不 claim 完美单调；within-tier inversion（DocVQA +33.5 < TextVQA +44）如实报。② ChartQA/GQA gap 不显著（≤1.7σ），只报方向不报胜。③ OCRBench per-type n 小 → 定性机制证据、非推断基准。④ VZ-style = L2-scored dom+ctx proxy（非官方 attention），但 ctx 差异实证无关、stage 已对齐。⑤ claim scope = Qwen3-VL-8B 单架构（跨架构 future work）。

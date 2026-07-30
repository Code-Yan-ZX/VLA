# 三图画图规格（user 手绘版）— 数据源 + 设计建议

> 对应 drafts/paper_acmmm.md 的 [FIG:1/2/3] spec 块；数字全部官方指标、已审计。
> 单位陷阱：OCRBench 是 /1000 pts，其余 pp —— 同图混用必须标注。

## FIG:1 — 方法示意（pipeline，无数据）
**推荐画法**：双面板、共享"stage 轴"。
- (a) RBM：ViT → 对 merger **输入**端 2×2 unit 特征打 L2 分 → top-κ 保留 → 原生 merger 只作用于存活 unit → LLM
- (b) Post：ViT → 原生 merger 全量合并 → 对 merger **输出**打 L2 分 → top-κ → LLM（虚线变体：FastV 在 LLM layer-2 剪）
- 两面板 merger/LLM 图形**完全相同**，只有打分抽头位置不同 → 视觉传达 "stage 是唯一被操纵变量"
- 标注：2×2 unit（32px footprint）、κ 预算箭头
- Takeaway 句（caption）：iso-model / iso-token / iso-selector，仅 stage 不同

## FIG:2 — 三族 pre−post Δ（头条图）
**数据**（25% 保留，pre − post，官方；源：experiments/j7_main_table.md + internvl3_main_matrix.md）：

| family | TextVQA (pp) | DocVQA (pp) | OCRBench (pts) | GQA (pp) |
|---|---|---|---|---|
| Qwen3-VL-8B | **+38.4** | +24.3 | +363 | −2.8 |
| Qwen2.5-VL-7B | +26.1 | +11.0 | +293 | −2.6 |
| InternVL3-8B | +37.4 | +34.6 | **+432** | −0.4 |

**推荐画法**：分组柱状图，x = 4 基准、每组 3 根（族）。OCRBench 单位不同 → 二选一：① 在柱上标 "+NNN pts"、y 轴只放 pp 基准且 OCR 柱用断轴/阴影区分；② 全图改用"相对 none 的标准化差距"（各柱 ÷ 该基准 none 值）—— ① 更忠实、② 更可比，建议 ①。GQA 三根都在 0 附近（−0.4 ~ −2.8），加 y=0 参考线突出"text-dense 大幅正、GQA 近零"。
**Takeaway**：三族 + 两种 merger 设计（Qwen PatchMerger / InternVL pixel-shuffle），text-dense 优势同量级同方向；GQA 无 crossover。

## FIG:3 — 保留率×差距曲线（压缩激活效应）
**数据**（n=200，pre − post Δ pp；源：experiments/j8_ablations.md，明细 runs/j8_summary.json，supp Table S3）：

| bench × model | 75% keep | 50% keep | 25% keep |
|---|---|---|---|
| TextVQA Q3 | +8.7 | +29.0 | +38.4 |
| TextVQA Q2.5 | +7.0 | +15.3 | +26.1 |
| DocVQA Q3 | −1.3 | +5.9 | +24.3 |
| DocVQA Q2.5 | −2.1 | −1.4 | +11.0 |

**推荐画法**：2×2 面板（TextVQA / DocVQA × Q3 / Q2.5），x = 保留率 {75, 50, 25}%，y = Δ pp，y=0 参考线（= none 基线）。DocVQA 浅压负值点真实画出（别截零）—— 那正是"效应由深压缩激活"的证据。
**Takeaway**：stage 效应在浅压缩 ≈0、随深度单调扩大（压缩激活，非恒定偏置）。caption 注 n=200。

## 可选 FIG:4（篇幅允许时）
定性样例：post/FastV 抹掉小字文本、RBM 保护（+ GQA 权衡方向）——素材在 v4 Fig.4 catalog（drafts/paper_v4.md 内引用）。

## 制图注意
1. OCRBench pts vs pp 必须区分标注（审稿人会抓）
2. FIG:3 用 n=200 数、FIG:2 用 full-split 数 —— 两图 caption 各标 n，勿混
3. 双盲：图内不得出现模型仓库名以外的作者/机构信息（模型名可出现）
4. 配色建议：族 = 色相、基准 = 分组；色盲安全（avoid red/green 单靠）

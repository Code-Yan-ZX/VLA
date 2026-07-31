# 三图画图规格（user 手绘版）— 数据源 + 设计建议

> 对应 drafts/paper_acmmm.md 的 [FIG:1/2/3] spec 块；数字全部官方指标、已审计。
> 单位陷阱：OCRBench 是 /1000 pts，其余 pp —— 同图混用必须标注。

## FIG:1 — 方法示意（pipeline，无数据）
**推荐画法**：横向三栏、中央双面板，共享"stage 轴"。整体学习 PyramidDrop Fig.2 的三栏总览，但把中央改成严格对齐的 pre/post 两条流程；右栏学习 SparseVLM Fig.1 的同图 token 保留对照。
- 左栏：同一张输入图 + patch division + ViT，只画一次，避免上下重复
- (a) RBM：ViT → 对 merger **输入**端 2×2 unit 特征打 L2 分 → top-κ 保留 → 原生 merger 只作用于存活 unit → LLM
- (b) Post：ViT → 原生 merger 全量合并 → 对 merger **输出**打 L2 分 → top-κ → LLM（虚线变体：FastV 在 LLM layer-2 剪）
- 两面板 merger/LLM 图形**完全相同**，只有打分抽头位置不同 → 视觉传达 "stage 是唯一被操纵变量"
- 右栏：同一输入的两张简化 token-grid 局部图，分别标 `rank before merger` / `rank after merger`；只显示选择位置与 token 状态，不放答案文本，也不伪造 attention heatmap
- 标注：2×2 unit（32px footprint）、κ 预算箭头
- Takeaway 句（caption）：iso-model / iso-token / iso-selector，仅 stage 不同

## FIG:2 — 三族 pre−post Δ（头条图）
**数据**（25% 保留，pre − post，官方；源：experiments/j7_main_table.md + internvl3_main_matrix.md）：

| family | TextVQA (pp) | DocVQA (pp) | OCRBench (pts) | GQA (pp) |
|---|---|---|---|---|
| Qwen3-VL-8B | **+38.4** | +24.3 | +363 | −2.8 |
| Qwen2.5-VL-7B | +26.1 | +11.0 | +293 | −2.6 |
| InternVL3-8B | +37.4 | +34.6 | **+432** | −0.4 |

**推荐画法**：全宽 1×4 同构小面板，每面板 3 根分组柱（族）。面板依次为 TextVQA (Δ pp)、DocVQA (Δ pp)、OCRBench (Δ pts)、GQA (Δ pp)，**各自使用独立且明示单位的 y 轴**；不要断轴，也不要把 pts 与 pp 放在同一连续轴。建议范围分别为 0–45、0–40、0–480、−3.5–0.5；每柱直接标带符号原值，所有面板都画深灰 y=0 参考线。GQA 三根向下但仍保持各自模型色，方向由零线和柱方向表达，不另用红色编码“负面”。
**Takeaway**：三族 + 两种 merger 设计（Qwen PatchMerger / InternVL pixel-shuffle），text-dense 优势同量级同方向；GQA 无 crossover。

## FIG:3 — 保留率×差距曲线（压缩激活效应）
**数据**（n=200，pre − post Δ pp；源：experiments/j8_ablations.md，明细 runs/j8_summary.json，supp Table S3）：

| bench × model | 75% keep | 50% keep | 25% keep |
|---|---|---|---|
| TextVQA Q3 | +8.7 | +29.0 | +38.4 |
| TextVQA Q2.5 | +7.0 | +15.3 | +26.1 |
| DocVQA Q3 | −1.3 | +5.9 | +24.3 |
| DocVQA Q2.5 | −2.1 | −1.4 | +11.0 |

**推荐画法**：2×2 同构小面板，列 = TextVQA / DocVQA，行 = Qwen3-VL-8B / Qwen2.5-VL-7B；x = 保留率 {75, 50, 25}%（从左到右压缩加深），共享 y = Δ pp，统一范围建议 −5–42，并画深灰 y=0 参考线。Qwen3 全行用蓝，Qwen2.5 全行用橙；线型和圆点相同，颜色只编码模型。DocVQA 浅压负值点真实画出（别截零）—— 那正是"效应由深压缩激活"的证据。
**Takeaway**：stage 效应在浅压缩 ≈0、随深度单调扩大（压缩激活，非恒定偏置）。caption 注 n=200。

## 可选 FIG:4（篇幅允许时）
定性样例：post/FastV 抹掉小字文本、RBM 保护（+ GQA 权衡方向）——素材在 v4 Fig.4 catalog（drafts/paper_v4.md 内引用）。

## 制图注意
1. OCRBench pts vs pp 必须区分标注（审稿人会抓）
2. FIG:3 用 n=200 数、FIG:2 用 full-split 数 —— 两图 caption 各标 n，勿混
3. 双盲：图内不得出现模型仓库名以外的作者/机构信息（模型名可出现）
4. 配色建议：族 = 色相、基准 = 分组；色盲安全（avoid red/green 单靠）

## 顶会配图调研（已实看论文 PDF）

以下不是泛化的“论文风格”描述，而是逐页查看 camera-ready / 论文 PDF 后，为本稿筛出的可迁移元素。详细拆解与近似色值采样见 `drafts/research/method_figure_survey.md`。

| 论文（会议） | 实看图 | 本稿借鉴 | 明确不照搬 |
|---|---|---|---|
| [FastV](https://arxiv.org/pdf/2403.06764)（ECCV 2024 Oral） | Fig.1, p.1；Fig.5, p.7 | 蓝/橙/绿三系列、共享图例；在模型栈中明确标剪枝插入点 | 聚合成含义模糊的 Average Performance；大段答案文本和视频胶片 |
| [TokenPacker](https://doi.org/10.1007/s11263-025-02491-7)（IJCV 2025） | Fig.2（[arXiv PDF](https://arxiv.org/pdf/2407.02392)） | 全局 pipeline + 局部模块放大，减少重复 | 过多内部算子和竖排长文本 |
| [VisionZip](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.pdf)（CVPR 2025） | Fig.1, p.1；Fig.3, p.3；Fig.4, p.7 | token 状态、压缩前后长度、1×N benchmark 小面板 | 雷达图；不同单位共轴；以相近颜色承载不同语义 |
| [Conical Visual Concentration / PyramidDrop](https://openaccess.thecvf.com/content/CVPR2025/papers/Xing_Conical_Visual_Concentration_for_Efficient_Large_Vision-Language_Models_CVPR_2025_paper.pdf)（CVPR 2025） | Fig.1, p.2；Fig.2, p.4；Fig.3, p.8 | 共享 stage 轴、暖色操作条、等宽同构曲线分面 | 每个 panel 随意换主色，使颜色含义漂移 |
| [SparseVLM](https://arxiv.org/pdf/2410.04417)（ICML 2025） | Fig.1, p.2；Fig.2, p.4 | 白底细描边、矩阵/token ID 的紧凑语法、严格列对齐 | 红/蓝只靠色相区分；大面积装饰性圆角底板 |
| [LLaVA-PruMerge](https://openaccess.thecvf.com/content/ICCV2025/papers/Shang_LLaVA-PruMerge_Adaptive_Token_Reduction_for_Efficient_Large_Multimodal_Models_ICCV_2025_paper.pdf)（ICCV 2025） | Fig.2, p.4 | selection → merge 的顺序、信号引导箭头 | 与 RBM 无关的聚类细节 |
| [ToMe](https://arxiv.org/pdf/2210.09461)（ICLR 2023） | Fig.1, p.3 | 用小 token 圆/格和边连接解释 merge，不靠长句 | 五步算法全部塞进主图 |

### 统一视觉系统（3 张主图必须共用）

| 语义 | 色值 / 样式 | 使用规则 |
|---|---|---|
| Qwen3 / 主方法冷色 | `#4C72B0` | FIG:2 的 Qwen3；FIG:3 的 Qwen3 行；FIG:1 共享 backbone 可用其浅阶 `#DCE6F5` |
| Qwen2.5 | `#DD8452` | FIG:2 的 Qwen2.5；FIG:3 的 Qwen2.5 行 |
| InternVL3 | `#55A868` | 只在 FIG:2 编码第三模型族 |
| RBM score/top-k 操作 | `#F2BA02`，浅底 `#FFF2CC` | FIG:1 唯一最高视觉优先级；两面板同一模块同形同色，只移动 stage |
| Post / FastV 辅助标记 | `#EE822F` | 仅作面板标题下划线或虚线 callout，不替代模型色 |
| 保留 / 丢弃 / 合并 token | `#3A7EC7` 实心 / `#D9DEE7` 斜纹或 × / `#FFF2CC` 四格汇一格 | 颜色 + 纹理/形状双编码，灰度打印仍可辨 |
| 文字 / 轴 / 网格 | `#222222` / `#333333` / `#D9D9D9` | 白底；只留水平细网格；零线比网格粗 |

- 画布：ACM 双栏通栏；FIG:1/2 建议约 7.1×2.8 in，FIG:3 约 7.1×4.6 in；优先输出 PDF/SVG 矢量和 300 dpi PNG 预览。
- 字体：Arial / Helvetica / Liberation Sans；最终嵌入论文后正文最小 7 pt，轴标签 7.5–8 pt，panel 标题 9–10 pt semibold；数学符号用 `κ`、`Δ`，不要混入手写体。
- 线条：模块边框 0.9–1.1 pt，数据线 1.6–1.9 pt，正交箭头 1.1 pt；无阴影、无渐变、无 3D、无厚重外框。
- 版式：panel 字母置左上；所有同类模块、坐标轴和标题基线严格对齐；图例全图只出现一次；图内不写完整 takeaway 句，把解释留给 caption。
- 可访问性：颜色含义全局固定，并以形状/纹理/正负方向辅助；做灰度与 deuteranopia 检查。生成模型常会拼错文字和数值，最终必须在矢量软件中人工逐字校正。

## 每张图的生成提示词（追加区）

下面提示词已把本文件中的数据、配色和布局合并进去。图内可见文字统一用英文；所有数值必须逐项核对，禁止生成模型补写或改写。

### FIG:1 prompt — RBM vs post-merger pipeline

```text
Create an editable vector scientific figure for a top-tier computer-vision paper, full-width landscape, about 7.1 x 3.0 inches, white background, flat 2D vector style, ACM two-column print readability. Use a three-column overview inspired by recent VLM compression papers, with no decorative outer card. Left column (about 22% width): one real input image, a clean patch-division inset, and a shared ViT block, shown only once. Middle column (about 53% width): a strict two-row comparison with one shared left-to-right stage axis and perfectly aligned modules. Panel (a) title: "RBM: pre-merger selection (ours)". Panel (b) title: "Post-merger selection". Right column (about 25% width): two aligned token-grid close-ups from the same input, labeled "rank before merger" and "rank after merger", showing only token state and operation order, similar in layout density to SparseVLM Fig.1.

In both rows use exactly the same shapes, sizes, colors, and typography for the shared modules: Image -> ViT -> native 2x2 merger -> LLM. Only the position of one identical amber module labeled "L2 rank + top-k" changes. In panel (a), place it between ViT and native merger: Image -> ViT, subtitle "2x2 units (N)" -> L2 rank + top-k -> "kN kept" -> native 2x2 merger -> LLM. In panel (b), place it after the native merger: Image -> ViT, subtitle "2x2 units (N)" -> native 2x2 merger -> "N merged" -> L2 rank + top-k -> "kN kept" -> LLM. Add a small dashed orange callout from the panel-(b) ranking stage to "FastV: prune at LLM layer 2"; it is a secondary variant, not part of the main lane.

Show token states with compact glyphs: a 2x2 blue grid for a raw 32 px unit, solid blue squares for kept tokens, pale gray hatched squares with x marks for dropped tokens, and four small cells converging into one pale-yellow outlined square for a merged token. In the right-column close-ups, use a lightly faded copy of the same image under an aligned patch grid; kept units remain saturated and dropped units become pale gray with x marks. These are schematic selection masks, not attention heatmaps: do not invent attention values, saliency colors, or question-specific evidence. Add one compact shared legend at the bottom: "kept", "dropped", "merged". Add the labels "32 px footprint" and "retention k" once, without prose paragraphs.

Color system: shared backbone light blue #DCE6F5 with dark blue border #2F5597; identical ranking module amber #F2BA02 with pale fill #FFF2CC; post/FastV secondary dashed accent #EE822F; kept token #3A7EC7; dropped token #D9DEE7 plus hatch/x; text #222222; arrows #5B6573. Use Arial/Helvetica, panel titles 9-10 pt semibold, all other text at least 7 pt at final size, 0.9-1.1 pt borders, orthogonal arrows. The visual message must be that stage is the only manipulated variable: iso-model, iso-token, iso-selector. No gradients, shadows, 3D, clip-art, decorative icons, rounded card background, or extra modules. Do not invent labels.
```

### FIG:2 prompt — three-family pre-minus-post deltas

```text
Create an editable vector grouped-bar figure for a top-tier VLM paper, full-width landscape, about 7.1 x 2.8 inches, white background. Use a 1x4 row of equal-width small-multiple panels with a single shared legend above the panels. Each panel contains exactly three bars in this fixed left-to-right order and fixed colors: Qwen3-VL-8B #4C72B0, Qwen2.5-VL-7B #DD8452, InternVL3-8B #55A868. Use slightly darker same-hue bar outlines. Color always encodes model family, never sign or benchmark.

Panel 1 title "TextVQA (Delta pp)", y range 0 to 45, values +38.4, +26.1, +37.4. Panel 2 title "DocVQA (Delta pp)", y range 0 to 40, values +24.3, +11.0, +34.6. Panel 3 title "OCRBench (Delta pts)", y range 0 to 480, values +363, +293, +432. Panel 4 title "GQA (Delta pp)", y range -3.5 to +0.5, values -2.8, -2.6, -0.4. Print every exact signed value directly above a positive bar or below a negative bar; use "pts" only for OCRBench and "pp" for all other panels. Draw a dark gray #333333 y=0 line in every panel, including the three positive-only panels. Use only subtle horizontal grid lines #D9D9D9. Add a small global annotation: "Delta = pre - post; 25% retained; full split".

Do not use a broken axis, dual axis, radar chart, normalized values, red for negative bars, background tint, 3D, gradients, error bars, or a repeated legend. Make independent y scales unmistakable through each panel title and ticks. Use Arial/Helvetica, 7.5-8 pt axis text, 9 pt semibold panel titles, 0.8 pt axes, and balanced spacing. Preserve the exact decimals and signs; do not average, round, reorder, or invent values.
```

### FIG:3 prompt — compression-activated stage gap

```text
Create an editable vector 2x2 small-multiple line figure for a top-tier VLM paper, full-width, about 7.1 x 4.6 inches, white background. Columns are TextVQA and DocVQA; rows are Qwen3-VL-8B and Qwen2.5-VL-7B. Put column headers at the top and row labels outside the left margin so titles are not repeated inside panels. The x-axis is "Visual-token retention" with positions 75%, 50%, 25% from left to right, plus a small right-pointing note "deeper compression". All panels share the y-axis label "pre - post (pp)" and the same y range -5 to 42.

Use one solid 1.8 pt line with circular markers per panel. Qwen3 row color #4C72B0; Qwen2.5 row color #DD8452. Exact values: Qwen3/TextVQA +8.7, +29.0, +38.4; Qwen3/DocVQA -1.3, +5.9, +24.3; Qwen2.5/TextVQA +7.0, +15.3, +26.1; Qwen2.5/DocVQA -2.1, -1.4, +11.0. Label the 25% endpoint in every panel with its signed value. Plot the negative DocVQA points below zero without clipping. Draw a dark gray #333333 y=0 reference line and subtle horizontal grid #D9D9D9. Add one compact note below the grid: "n = 200 per benchmark".

Keep model color consistent across both columns; do not use a different color for every panel. Use Arial/Helvetica, 7.5-8 pt axes, 9 pt semibold headers, common tick positions, aligned plot areas, and enough space for negative labels. No smoothing, shaded confidence bands, gradients, 3D, decorative background, or extrapolated 100% point. Do not alter the point order or values.
```

### 可选 FIG:4 prompt — qualitative OCR failure and GQA trade-off

```text
Create an editable vector qualitative comparison figure for a top-tier VLM paper, full-width landscape, white background, three equal columns for real benchmark cases. Column 1: TextVQA id 35014, question "What is the date on the right page?", ground truth "07/10/2012". Column 2: DocVQA id 58439, question about promotional meetings and events in 1998, ground truth "$1.3 BILLION". Column 3: GQA id 201370409, question "What are the scissors on?", ground truth "paper".

Within each column, put the original image on top with one thin, precise callout box around the evidence region, then place two aligned answer rows below: "RBM pre" and "Post". For TextVQA show RBM pre "07/10/2012" with a blue check and Post "no visible date" with an orange x. For DocVQA show RBM pre "$1.3 billion" with a blue check and Post "$1.3 million" with an orange x, plus a compact annotation "1000x unit error". For GQA show RBM pre "no scissors visible" with an orange x and Post "paper" with a blue check. Add the same-token-budget label "iso-token per image" once above the answer rows.

Use real uncropped source images from runs/data, preserving aspect ratio; do not synthesize or replace the benchmark images. Palette: correct/evidence #4C72B0, incorrect/callout #EE822F, neutral frames #5B6573, text #222222. Use solid vs dashed outlines in addition to color. Keep all text at least 7 pt at final size, use short answer excerpts only, and align all rows. No heatmap unless it comes from measured token data, no invented saliency, no decorative icons, no shadows, and no fabricated text inside images. This figure must be assembled with the actual assets rather than generated end-to-end by an image model.
```

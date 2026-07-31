# VLM 视觉 Token 压缩方法图调研（FIG:1 参考）

> 日期：2026-07-31。范围：近年 CVPR / ICCV / ECCV / EMNLP 中与视觉 token pruning、merging、projector compression 最相关的方法总览图。下列结论均来自实际查看论文 PDF 中的指定 Figure，而非只读摘要或二手网页。颜色为从 PDF 页渲染图中采样得到的近似 hex；不是作者发布的品牌色规范。

## 1. 结论先行

RBM 的 FIG:1 最适合采用三种顶会图的组合语法，而不是照搬某一篇：

1. **PyramidDrop Fig. 2 的“stage 轴”**：把“发生在哪一阶段”置于画面中心，天然匹配 RBM 的唯一自变量（merger 前/后）。
2. **TokenPacker Fig. 2 的“总览 + 局部放大”**：左侧给共享 VLM pipeline，右侧放大相同 merger 模块，避免上下两条 pipeline 重复导致拥挤。
3. **VisionZip / PruMerge 的 token 状态语法**：用小方块/网格表达完整、保留、丢弃、合并后的 token，并显式标出 `N -> kN`。

FastV Fig. 5 适合借“在模型栈中标出一次插入点”和 token 图例；不宜借其回答文本/视频胶片等叙事内容。最终应是一张**横向双面板、共享 stage 轴、局部放大 merger**的机制图，视觉重点只落在两个不同的 score tap-point，其他模块完全同形同色。

## 2. 实看论文与指定方法图

| 论文 | 会议 | 实看来源 | 指定图 | 与 RBM 的相关度 |
|---|---|---|---|---|
| *An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models* (FastV) | ECCV 2024 Oral | [PDF](https://arxiv.org/pdf/2403.06764) · [项目/代码](https://github.com/pkunlp-icler/FastV) | **Fig. 5, p.7** | 高：明确画出剪枝插入层与过滤前后 token |
| *TokenPacker: Efficient Visual Projector for Multimodal LLM* | EMNLP 2024 | [PDF](https://aclanthology.org/2024.emnlp-main.469.pdf) · [项目/代码](https://github.com/CircleRadon/TokenPacker) | **Fig. 2, p.4** | 高：总览 + 模块放大，且位于 encoder/LLM 之间 |
| *VisionZip: Longer is Better but Not Necessary in Vision Language Models* | CVPR 2025 | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.pdf) · [项目/代码](https://github.com/dvlab-research/VisionZip) | **Fig. 3, p.3** | 很高：selection + contextual merge 与 token 数缩短均可视化 |
| *Conical Visual Concentration for Efficient Large Vision-Language Models*（PyramidDrop） | CVPR 2025 | [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Xing_Conical_Visual_Concentration_for_Efficient_Large_Vision-Language_Models_CVPR_2025_paper.pdf) · [项目/代码](https://github.com/Cooperx521/PyramidDrop) | **Fig. 2, p.4** | 很高：stage 是整图主轴，rank/drop 点非常清楚 |
| *LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models* | ICCV 2025 | [CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Shang_LLaVA-PruMerge_Adaptive_Token_Reduction_for_Efficient_Large_Multimodal_Models_ICCV_2025_paper.pdf) · [项目页](https://llava-prumerge.github.io/) | **Fig. 2, p.4** | 很高：最接近“先选后并”的 token selection/cluster/merge 图 |

## 3. 逐图拆解

### 3.1 FastV — Fig. 5

- **配色（近似）**：Transformer 深蓝 `#2F5597`；剪枝模块黄 `#FFD966`；图像 token 蓝 `#4B8AC2`；文本/正确事实绿 `#548235` / `#A9D18E`；深蓝边框 `#172C51`。
- **排版**：上方输入视频，左侧问题/输出例子，右侧独立圆角框画 Transformer 栈；剪枝模块是一条横向黄色带，夹在 `K+1` 与 `K+2` 层之间。token 用短竖条列表示，过滤 token 另用黄色编码。
- **箭头/图例**：流程箭头较细；右上角三色 token legend；层间用回环箭头表现重复 block；`K`、`R%` 直接写入黄色算子条。
- **视觉层次**：黄色剪枝条是唯一高饱和模块，蓝色 Transformer 是稳定背景；方法名配闪电符号，抓眼但略营销化。
- **可迁移**：在 merger 模块两侧各放 token 条带，并给“kept / dropped / merged”图例；把 `top-k` 的发生点做成唯一暖色横条。
- **不迁移**：不要放长回答文本、视频胶片或机器人图标；RBM 的核心是 stage control，不需要用大量 qualitative 内容争夺视线。

### 3.2 TokenPacker — Fig. 2

- **配色（近似）**：LLM 灰蓝 `#8497B0`；特征图浅蓝 `#9DC3E6` / `#BDD7EE`；视觉 encoder 淡绿（约 `#E2F0D9`）；token 淡黄 `#FFF2CC`；冻结/训练用蓝雪花与橙火焰 `#FD7A14`。
- **排版**：一条竖分隔线切成 `(Left) overall MLLM` 与 `(Right) TokenPacker detail`。左侧只回答“模块在哪里”，右侧回答“模块内部怎样工作”；两侧都自上而下，但主数据流依然清楚。
- **箭头/图例**：视觉流绿箭头、语言流蓝箭头，图例只在左下；右侧虚线圆角框包住 Injection Module，输入尺寸写在张量旁。
- **视觉层次**：深灰蓝 LLM 是最大稳定块；放大模块以浅灰底 + 虚线框隔离；token/feature map 用重复小方格建立视觉语义。
- **可迁移**：FIG:1 可先画一条共享 `ViT -> native merger -> LLM` 总览，再在右侧放大 merger 前后；这样双面板只比较 score tap-point，不重复完整模型。
- **不迁移**：雪花/火焰代表 frozen/trainable，在 training-free RBM 中没有必要；多层 feature map 立体堆叠也会误导为多尺度方法。

### 3.3 VisionZip — Fig. 3

- **配色（近似）**：encoder/projector/LLM 蓝灰 `#729FB4`；system token 蓝 `#3A7EC7`；image token 酒红 `#952B39`；question token 金黄 `#EEBB4E`；未选 token 淡蓝 `#A1BFCD`；强调红 `#C00000`。
- **排版**：严格左到右：image -> Vision Encoder -> 上支路 similarity merge / 下支路 attention select -> `+` 聚合 -> Projector -> LLM；虚线竖隔断后单独显示压缩前后序列长度 `2144 -> 160`。
- **箭头/图例**：统一粗浅蓝箭头；分支分别标 `Based on similarity` 与 `Based on attention`；token 条带宽度本身编码长度，数字写在色块内。
- **视觉层次**：蓝灰 backbone 保持中性，酒红 image-token 长条形成最强的“压缩前很长”信号；实图与 token 热图提供语义锚点。
- **可迁移**：为 RBM/Post 同时画 `N units -> kN units -> kN merged tokens`，让预算同构一眼可见；用上下支路或两面板明确 score 与 native merger 的先后关系。
- **不迁移**：VisionZip 的“dominant + contextual merge”是额外算法；RBM 必须把原生 merger 画成同一冻结模块，不能画成两路新 merge 或让人误解为 token fusion 新方法。

### 3.4 PyramidDrop — Fig. 2

- **配色（近似）**：Transformer 淡蓝 `#91ACE0`；Rank and Drop 桃橙 `#F5B482`；image token 金黄 `#F2BA02`；instruction token 橙 `#EE822F`；system token 绿 `#75BD42`；面板虚线用同族浅色。
- **排版**：左为输入与 token 化，中间是自下而上的 `Stage 1 -> Stage 2 -> Stage S`，右为 sequence-length 小图与两个 rank/drop 局部示例。三列均对齐，中心 stage 栈是主角。
- **箭头/图例**：阶段间粗蓝向上箭头；`Rank and Drop` 固定成每阶段顶部的暖色横条；底部一行 token legend；右侧用“attention 排序柱 + Drop Low 箭头”解释筛选。
- **视觉层次**：大面积低饱和蓝承载 backbone，暖橙只标操作；随着阶段变深 token 方块减少，形成无需文字的漏斗叙事。
- **可迁移**：这是 RBM 最值得借的结构。把中央纵轴改成横向 `ViT output -> [score hook] -> native merger -> LLM`，上下两面板共用精确对齐的 stage 刻度；只移动暖橙 score hook。
- **不迁移**：RBM 不是多阶段递进 drop，不应画成反复剪枝漏斗；红/绿/橙三类 token 在灰度和色盲条件下区分一般，建议改用形状/纹理辅助。

### 3.5 LLaVA-PruMerge — Fig. 2

- **配色（近似）**：Q/K/V 与 FFN 青蓝 `#96DCF8`；三步方法区淡黄 `#FFF6DB` / `#FFDF7F` / `#FFEBB0`；guide 箭头酒红 `#972637`；普通 token 蓝 `#0070C0`；待并 token 橙 `#C04F15`。
- **排版**：单条左到右长带：视觉 token -> Q/K/V 与 attention -> FFN -> Step 1 选 token -> Step 2 cluster -> Final weighted update -> 压缩 token。三步区用相邻圆角框形成明确顺序。
- **箭头/图例**：主流程灰箭头；酒红 guide 线跨越上方/下方，把 attention/self-sim 分别连接到选择与聚类；token 以五边形/纹理区分角色，cluster 用彩色椭圆圈住。
- **视觉层次**：黄色三步区占最大宽度，蓝色 backbone 次之；跨区 guide 箭头明确“哪个信号指导哪个操作”，但线条较多、缩栏后拥挤。
- **可迁移**：对 RBM 的 `L2 score -> top-k -> native merger` 用三段相邻模块；从 merger-input feature 到 scorer 画一条短 guide 线，从 merger-output 到 post scorer 画另一条，突出取数位置。
- **不迁移**：不要画 cluster 椭圆、加权中心更新或跨越全图的长 guide 线；这些会让 RBM 看起来像 PruMerge 式 token merging，而 RBM 明确不引入第二个 merger。

## 4. 对 RBM FIG:1 的具体落地建议

### 4.1 推荐构图

- **画布**：横向全栏，两个等宽面板 `(a) Rank-Before-Merge` 与 `(b) Post-Merge Rank`，中间留 3–4 mm gutter；不使用嵌套卡片。
- **共享 stage 轴**：两面板严格对齐 5 个列位：`ViT units | score | top-k | native 2x2 merger | LLM`。Post 面板把 `score + top-k` 移到 merger 后；其余模块尺寸、颜色、箭头完全相同。
- **局部放大**：每个面板左侧只放一个 4x4 token 网格，框出一个 `2x2 unit (32 px footprint)`；merger 画成 2x2 四格汇成 1 格。无需真实照片，避免引入内容噪声。
- **token 数量**：在箭头下方写 `N units -> kN surviving units -> kN merged tokens`（Post：`N units -> N merged tokens -> kN retained tokens`）；用同一终点长度证明 iso-token。
- **FastV 虚线变体**：只在 Post 面板 LLM 的第二层旁放一条细虚线 hook，标 `FastV (layer 2)`；降低饱和度，不与主对比争主视觉。
- **公平性底条**：图底单行 `same model | same token budget | same L2 selector | only the stage changes`；caption 再完整解释，不在图内放长句。

### 4.2 推荐配色（由上述五图的共同语法合成）

| 语义 | 推荐色 | 来源/理由 |
|---|---|---|
| 共享 backbone（ViT / merger / LLM） | 深蓝 `#2F5597` + 浅蓝 `#91ACE0` | FastV / PyramidDrop：蓝色稳定承载模型模块 |
| RBM 的 score/top-k 操作 | 金黄 `#F2BA02`（浅底可用 `#FFD966`） | FastV / PyramidDrop：暖色只给方法操作，最高视觉优先级 |
| Post 对照的 score/top-k 操作 | 橙 `#EE822F` | 与蓝/黄在色盲模式中仍有亮度差；不要用红/绿二分 |
| 保留 token | 蓝 `#3A7EC7` | VisionZip 的 system-token 蓝，清晰且印刷稳定 |
| 被丢弃 token | 浅灰 `#D9DEE7` + 斜线纹理 | 不靠颜色单独表达；灰度打印仍可辨 |
| 合并后 token | 淡黄 `#FFF2CC` + 深蓝描边 | TokenPacker 的 compact token 语法 |
| 文字/轮廓/箭头 | `#1F2937` / `#5B6573` | 避免纯黑大面积压过流程；主箭头保持中性 |

注意：不要把 VisionZip 的酒红 `#952B39` 与绿色并列作为唯一语义编码；不要用五种同饱和色给模块逐个上色。目标是“共享模块冷色、唯一操作暖色、token 状态靠颜色 + 纹理/形状双编码”。

### 4.3 视觉层次与制图检查

1. 第一眼：暖色 score hook 的位置不同。
2. 第二眼：两条 pipeline 的 native merger 完全相同，输入/输出 token 数的变化顺序不同。
3. 第三眼：`2x2 unit`、`k`、FastV 虚线等实现细节。
4. 缩到 ACM 双栏整页宽度后，最小文字仍 >= 7 pt；模块名控制在 1–2 行，禁竖排长文本。
5. 只使用直线/正交箭头，箭头从模块边界出发；禁止跨文字、回折穿框和无来源的悬浮箭头。
6. 色盲/灰度检查：保留 token=实心，丢弃 token=斜纹或 `x`，合并 token=四格汇一格；颜色失效时语义仍成立。

## 5. 可直接转成绘图提示词的设计摘要

`Top-conference scientific method diagram, full-width horizontal two-panel comparison, shared aligned stage axis, identical ViT/native 2x2 merger/LLM modules, only move the warm-colored L2-score-and-top-k hook from before the merger in panel (a) to after the merger in panel (b), compact token grids with kept/dropped/merged legend, orthogonal arrows, restrained blue backbone palette with amber/orange operation highlights, white background, vector-flat style, ACM paper typography, no decorative gradients, no 3D, no marketing icons, readable at two-column print size.`

该英文摘要只用于后续提示词的共同风格前缀；具体 FIG:1 提示词仍应逐项写入用户 spec，并锁定上述模块次序、数量标注和公平性底条。

## 6. 核验记录

- 实际下载并查看 PDF 页：FastV p.7 Fig.5；TokenPacker p.4 Fig.2；VisionZip p.3 Fig.3；PyramidDrop p.4 Fig.2；LLaVA-PruMerge p.4 Fig.2。
- 会议/年份以 ECCV、ACL Anthology、CVF Open Access 页面及论文 camera-ready 信息为准；VisionZip/PyramidDrop 的 CVPR camera-ready 标题与早期 arXiv 标题可能有细微差异，表中采用正式标题。
- hex 为 160-dpi PDF 渲染后对图形区域的主色采样近似值，适合“学习配色”，不应宣称为原作者官方色板。

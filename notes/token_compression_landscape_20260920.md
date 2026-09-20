# VLM 视觉 token 压缩：顶会代表工作与研究判断

> 核验日期：2026-09-20。范围为视觉 token 数量/计算压缩，不含模型权重量化和纯文本压缩。只把正式主会或 ACL/EMNLP 正式论文列入主表；优势中的数字均是作者在各自设置下报告，不能跨论文直接排名。局限如非论文明确承认，标作「推断」。详细证据见同日 `lit_venue_map`、`lit_methods`、`lit_frontier` 三份笔记。

## 一、顶会代表工作

| 会议、方法 | 要解决的痛点；核心做法 | 优势 | 局限或代价 |
|---|---|---|---|
| [ECCV 2024 FastV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/10478_ECCV_2024_paper.php) | LLM 深层仍处理大量视觉 token；用早层 attention 排名，在指定层后剪枝。 | 免训练、易接入；LLaVA-1.5-13B 报告约 45% FLOPs 下降。 | 早于剪枝层的编码和 prefill 成本已经付出；attention 排名对细字和偏置敏感（后者已有 [DART](https://aclanthology.org/2025.emnlp-main.505/) 与 [VisPruner](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Beyond_Text-Visual_Attention_Exploiting_Visual_Cues_for_Effective_Token_Pruning_ICCV_2025_paper.html) 实证挑战）。 |
| [CVPR 2025 VisionZip](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.html) | 编码器输出冗余；保留 dominant tokens，把其余聚成 contextual tokens，再送 LLM。 | 前置压缩适合省 LLM 首次 prefill；可用于图像、视频、多轮。 | 不按当前问题选择；通用显著性与聚合可能漏掉文字、空间关系（推断）；某些配置需要 projector 调优，不能一律称免训练。 |
| [CVPR 2025 DivPrune](https://openaccess.thecvf.com/content/CVPR2025/html/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.html) | top-attention 留下许多重复 token；用最大最小距离选择多样子集。 | 免训练，兼顾延迟与显存；覆盖更多图像区域。 | 多样性不等于答题证据；在极低预算 OCR 上仍有显著退化：补充材料的 LLaVA-1.5 TextVQA 由 46.08 降至 35.97（[补充材料](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Alvar_DivPrune_Diversity-based_Visual_CVPR_2025_supplemental.pdf)）。 |
| [CVPR 2025 ATP-LLaVA](https://openaccess.thecvf.com/content/CVPR2025/html/Ye_ATP-LLaVA_Adaptive_Token_Pruning_for_Large_Vision_Language_Models_CVPR_2025_paper.html) | 固定比例不适合不同图文输入和不同层；学习逐层、逐样本阈值，加空间增强剪枝。 | 能动态分配，平均删 75% token 时七基准平均约 1.9% 性能下降。 | 需要训练轻量模块，依赖 LLM 内部 attention；高效 attention kernel 和不同架构迁移需额外验证。 |
| [CVPR 2025 ViCo（原 PyramidDrop）](https://openaccess.thecvf.com/content/CVPR2025/html/Xing_Conical_Visual_Concentration_for_Efficient_Large_Vision-Language_Models_CVPR_2025_paper.html) | 早层需要更多视觉信息、深层冗余增加；分阶段逐步删 token。 | 兼顾训练和推理效率；作者报告训练时间下降逾 40%、推理 FLOPs 降 55%。 | 阶段比例预设，未随证据难度自动调；早层开销保留（推断）。CVPR 正式题名为 *Conical Visual Concentration*。 |
| [ICML 2025 SparseVLM](https://openreview.net/forum?id=80faIPZ67S) | 视觉 token 重要性应参考文本问题；用相关文本 token 的 LLM attention 逐层稀疏化，并回收被删信息。 | 免训练；LLaVA 报告 FLOPs 降 54%、CUDA 时间降 37%、准确率保持 97%。 | 要进入 LLM 才能取分，早期成本仍在；文本和 attention 代理可能不稳定（推断）。 |
| [ICCV 2025 LLaVA-PruMerge](https://openaccess.thecvf.com/content/ICCV2025/html/Shang_LLaVA-PruMerge_Adaptive_Token_Reduction_for_Efficient_Large_Multimodal_Models_ICCV_2025_paper.html) | 只剪枝易丢视觉上下文；先按 ViT attention 留关键 token，再把相似的其余 token 合并。 | 编码器后压缩，正式摘要报告约 4× token 减少；可免训练或进一步微调。 | 依赖 CLS/attention 语义；跨无 CLS、多尺度动态编码器迁移和细粒度位置保真需核验（推断）。早版 arXiv 的 14× 不可与正式摘要 4× 混用。 |
| [ICCV 2025 VisPruner](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Beyond_Text-Visual_Attention_Exploiting_Visual_Cues_for_Effective_Token_Pruning_ICCV_2025_paper.html) | 文本到视觉 attention 排名有偏；编码器侧先选显著 token，再去重保留多样 token。 | 免训练，剪枝位置早，兼容高效 attention；论文报告 LLaVA-1.5-7B 推理延迟降 75% 的设置。 | 图像通用视觉信号未必知道当前提问；细小文字/定位任务仍需专门验证（推断）。 |
| [EMNLP 2025 DART](https://aclanthology.org/2025.emnlp-main.505/) | 单 token 重要性分数可能差于随机，还与高效 attention 不兼容；改按与 pivot 的重复度剪枝。 | 免训练；作者报告剪 88.9% token 时总时间加速 1.99×、prefill 2.99×。 | 低重复但无关的 token 可能占预算；高相似的关键字形差异可能被误认为重复（推断）。 |
| [NeurIPS 2025 MoB](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6818dcc65fdf3cbd4b05770fb957803e-Abstract-Conference.html) | 提问相关性与全图覆盖有资源冲突；用双目标覆盖及预算分配求保留集合。 | 给出理论误差界和线性扩展算法；LLaVA-1.5-7B 仅留 11.1% token 时保留 96.4% 表现。 | 覆盖是表征代理，并不直接保证最小字形、数字或表格单元正确；预算分配效果依任务分布（推断）。 |
| [NeurIPS 2025 DyVTE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/f60d4225745fb682bd088b63dcbbe52d-Abstract-Conference.html) | 视觉信息已被文本吸收后，token 还在后层计算；动态决定视觉 token 退出层。 | 与 token 子集剪枝互补；LLaVA-1.5 报告最多 45.7% 计算开销减少且无性能下降。 | 不能省视觉编码和退出前的计算；退出判据对 OCR 等深层依赖任务要重校（结合 [CVPR 2026 信息视界](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_When_Token_Pruning_is_Worse_than_Random_Understanding_Visual_Token_CVPR_2026_paper.html) 推断）。 |
| [NeurIPS 2025 C&C](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2e01083b381b4865919b4915ef32e3d2-Abstract-Conference.html) | 同一张图用于生成与检索，反复在线压缩浪费；预先索引成少量摘要 token，并训练兼顾生成和检索。 | 压缩表征可缓存复用，适合大规模图像库和多轮。 | 需要专门训练、索引与缓存；一次性请求的前置代价不一定划算（推断）。 |
| [ICLR 2026 MMTok](https://proceedings.iclr.cc/paper_files/paper/2026/hash/4f8745c0ca5c951a92844ad7668cfca5-Abstract-Conference.html) | 单看文本或图像都不够；将选 token 定义为同时覆盖文本与视觉集合的问题。 | LLaVA-NeXT-13B 在 POPE 报告 1.87× 加速、98.7% 性能保持。 | 覆盖代理仍不保证细粒度证据；集合优化成本需计入总时延（推断）。 |
| [CVPR 2026 QuietPrune](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_QuietPrune_Query-Guided_Early_Token_Pruning_for_Vision-Language_Models_CVPR_2026_paper.html) | 早期 ViT 剪枝快，但缺少问题语义；将 query 映射到视觉域，在 ViT 内按 2×2 组筛选并聚合。 | 适配 Qwen3-VL、InternVL3 的原生组结构；相较所比晚剪方法，报告最高 19% prefill 延迟降低。 | adapter 与组粒度带来模型耦合；组内小字/细线是否仍保真需要实测（推断）。 |
| [CVPR 2026 DocPrune](https://openaccess.thecvf.com/content/CVPR2026/html/Choi_DocPrune_Efficient_Document_Question_Answering_via_Background_Question_and_Comprehension-aware_CVPR_2026_paper.html) | 文档背景大、证据稀、层位不固定；背景、问题、理解程度三阶段剪枝。 | 直接面向文档 QA；M3DocRAG 上报告编码器/解码器吞吐分别 3.0×/3.3×，F1 +1.0。 | 证据集中于该基准；encoder/decoder 分段吞吐不能等同整个在线服务吞吐。 |
| [CVPR 2026 MetaCompress](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Rethinking_Token_Reduction_for_Large_Vision-Language_Models_CVPR_2026_paper.html) | 多轮后续问题未知，首轮 query 剪枝会删掉未来证据；学习与 prompt 无关的压缩映射。 | 正面处理多轮复用，并统一 pruning/merging 表述。 | 需要训练；提示无关压缩可能为未来任意问题保留更多冗余（推断）。 |
| [CVPR 2026 V2Drop](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Variation-aware_Vision_Token_Dropping_for_Faster_Large_Vision-Language_Models_CVPR_2026_paper.html) | LLM 内 attention 打分存在位置偏置与高效算子冲突；按 token 跨层变化逐步丢弃。 | 图像/视频报告保持 94.0%/98.6% 原性能，生成延迟分别降 31.5%/74.2%。 | 与其他 LLM 内方案一样，编码器及剪枝前成本尚存；特征变化小不必然代表 OCR 细节无用（推断）。 |
| [CVPR 2026 信息视界研究](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_When_Token_Pruning_is_Worse_than_Random_Understanding_Visual_Token_CVPR_2026_paper.html) | 解释为何深层重要性剪枝不如随机；以删 token 对输出概率影响测信息视界。 | 指出 OCR 的信息视界比一般 VQA 更深，提供层位/任务控制的重要基线。 | 是机制诊断，不是通用早剪方法；信息视界随模型、任务改变。 |

## 二、研究问题如何收窄

1. **已被占据的方向**：单独讲早期 query-aware（QuietPrune）、文档三阶段剪枝（DocPrune）、多信号覆盖（MoB/MMTok/DART）、多轮通用压缩（MetaCompress）或“首次报告真实延迟”都不成立。[EarlyTom](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_EarlyTom_Early_Token_Compression_Completes_Fast_Video_Understanding_CVPR_2026_paper.html) 已报告 TTFT 和吞吐；[ApET](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_ApET_Approximation-Error_Guided_Token_Compression_for_Efficient_VLMs_CVPR_2026_paper.html) 已处理 FlashAttention 兼容的一个痛点。
2. **可检验而尚不能宣称空白的问题**：同模型、同后端、同保留 token 数，在 TextVQA/OCRBench/DocVQA/ChartQA 与多轮任务上，比较平均性能和最差切片（小字、密集表格、稀有关键数字）以及完整服务链的 p95 TTFT、请求吞吐、峰值显存。先看现有方法是否出现“平均分相近，稀有证据和真实吞吐差很大”的稳定现象。
3. **方法机会取决于该诊断结果**：若存在上述差距，再研究按证据风险动态分预算、保持局部几何/字形的压缩；同时与 QuietPrune、DocPrune、MMTok、MetaCompress 在同一协议中比较。若无稳定差距，就不应另造一个打分器。

## 三、比较协议

- 同一基座与视觉分辨率、同一评测集和题目、同一输出长度与解码设置；固定 **LLM 实际收到的视觉 token 数**，并另报视觉编码器成本。
- 同时记录 token 数、FLOPs、首次 token 延迟、完整延迟、峰值显存、连续批处理请求吞吐；单请求和多并发分开。
- 固定样本切片报告文字识别、文档、图表、定位、普通 VQA；给出绝对分数与保留率，并用配对样本估计差值，避免只给任务平均值。
- 标记训练免/轻量训练/重训，避免将不同成本、不同基座、不同预算的论文数字直接排队。

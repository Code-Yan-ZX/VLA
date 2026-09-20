# VLM 视觉 token 压缩顶会发表核验（2026-09-20）

范围：以 **进入视觉编码器、投影器或 LLM 的视觉 token 数量缩减** 为主；不把纯图像生成 tokenizer、权重量化或 KV cache 压缩混入。发表场次以会议官方论文集、PMLR、ECVA、ACL Anthology 或 NeurIPS 论文集为准。下列“局限”是根据论文目标和机制的研究判断，不能直接解释成论文已测得的负面结果。此表是代表作地图，并非穷尽检索；nature-academic-search 的专用 MCP 本会话不可用，故以官方网页逐篇核验。

| 年份/正式会议 | 论文/方法（官方链接） | 解决的痛点、机制 | 优点 | 需要验证的局限 |
|---|---|---|---|---|
| 2023 ICLR | [Token Merging: Your ViT But Faster (ToMe)](https://openreview.net/forum?id=JroZRaRw7Eu) | ViT token 冗余；逐层匹配并合并相似 patch。 | 免训练、可用于训练/推理；奠定 merging 基线。 | 原设计面向 ViT，不能把 ImageNet 加速直接算作 MLLM 加速；合并可能稀释小字/小物体。 |
| 2024 ECCV | [An Image is Worth 1/2 Tokens After Layer 2 (FastV)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/10478_ECCV_2024_paper.php) | LLM 深层视觉 token 使用率低；浅层注意力估计重要性后在后层裁剪。 | 免训练、实现简洁；官方报告 LLaVA-1.5-13B 减少 45% FLOPs。 | 浅层 attention 可能有位置偏置；早裁剪对定位/OCR 的风险须专项测。 |
| 2025 CVPR | [VisionZip](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_VisionZip_Longer_is_Better_but_Not_Necessary_in_Vision_Language_CVPR_2025_paper.html) | CLIP/SigLIP 特征冗余；视觉编码器末端选 dominant/contextual tokens。 | LLM 前减长，预填充与多轮复用受益；官方报告最多 8× prefill。 | 以视觉内部信号为主，初始问题的细粒度目标不一定得到保留；encoder 本身仍运行。 |
| 2025 CVPR | [DivPrune](https://openaccess.thecvf.com/content/CVPR2025/html/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.html) | 重要性 Top-K 保留相似冗余 token；最大化保留集合的最小距离。 | 无需微调、兼顾图像/视频、提高覆盖。 | 纯 diversity 可能忽略文本问题；距离度量与选择成本需计入端到端。 |
| 2025 CVPR | [ATP-LLaVA](https://openaccess.thecvf.com/content/CVPR2025/html/Ye_ATP-LLaVA_Adaptive_Token_Pruning_for_Large_Vision_Language_Models_CVPR_2025_paper.html) | 固定层/样本裁剪率不适配难度；动态阈值加空间增强。 | 逐样本逐层调预算；官方报告平均减少 75% token，七基准仅 1.9% 性能下降。 | 模块需适配/训练；跨模型迁移与实际调度开销需测。 |
| 2025 CVPR | [TopV](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_TopV_Compatible_Token_Pruning_with_Inference_Time_Optimization_for_Fast_CVPR_2025_paper.html) | attention 打分与 FlashAttention 不兼容；以特征相似度/空间距离优化选 token。 | 避免提取注意力矩阵，一次 prefill 剪枝并缩小 KV cache。 | 优化器自身开销、空间中心先验和小目标稳定性需测。 |
| 2025 CVPR | [Conical Visual Concentration / ViCo](https://openaccess.thecvf.com/content/CVPR2025/html/Xing_Conical_Visual_Concentration_for_Efficient_Large_Vision-Language_Models_CVPR_2025_paper.html) | 早期裁剪损失信息、深层冗余更多；分阶段形成锥形 token 数。 | 同时面向训练和推理；官方报告训练时间减少 >40%、推理 FLOPs 减少 55%。 | 分阶段比例仍有配置，性能随任务/模型变化；须分开量训练/真实延迟。早期版本称 PyramidDrop，正式会议标题如本行。 |
| 2025 CVPR | [Accelerating MLLMs by Searching Optimal Vision Token Reduction (G-Search/P-Sigmoid)](https://openaccess.thecvf.com/content/CVPR2025/html/Zhao_Accelerating_Multimodal_Large_Language_Models_by_Searching_Optimal_Vision_Token_CVPR_2025_paper.html) | 不同层应保留多少 token 缺乏预算配置；搜索层级策略。 | 优化延迟–准确率曲线；官方报告 LLaVA/InternVL2 >2× 加速。 | 搜索/校准成本、注意力排名稳定假设和跨任务可迁移性需测。 |
| 2025 ICML | [SparseVLM](https://proceedings.mlr.press/v267/zhang25s.html) | 视觉 token 对文本任务的贡献不同；选视觉相关文本作评分、逐层裁剪并回收 token。 | 免训练、文本条件化；官方报告 LLaVA FLOPs -54%、CUDA 时间 -37%，保持 97% 准确率。 | 多次读取 self-attention/回收的实际开销、文本打分偏置和多轮复用需测。 |
| 2025 ICCV | [LLaVA-PruMerge](https://openaccess.thecvf.com/content/ICCV2025/html/Shang_LLaVA-PruMerge_Adaptive_Token_Reduction_for_Efficient_Large_Multimodal_Models_ICCV_2025_paper.html) | 视觉 token 空间冗余；视觉编码器 CLS attention 选主 token，再按相似性合并。 | 同时保留重要主 token 和被裁剪 token 的部分信息；官方报告视觉 token 减为约 1/4。 | 依赖注意力/特征相似度，问题相关性与空间定位仍需专项检查。**正式发表为 ICCV 2025，不能沿用预印本年份当会议年。** |
| 2025 ICCV | [Feather the Throttle (FEATHER)](https://openaccess.thecvf.com/content/ICCV2025/html/Endo_Feather_the_Throttle_Revisiting_Visual_Token_Pruning_for_Vision-Language_Model_ICCV_2025_paper.html) | 早层裁剪有图像上部被系统性删除、常规基准掩盖定位退化；先均匀采样再多阶段剪枝。 | 把定位/细粒度视觉能力纳入评估；官方称相同计算节约下定位基准 >5× 改进。 | 均匀覆盖不保证问题相关性；需兼顾 OCR、小区域与效率。 |
| 2025 ICCV | [Beyond Text-Visual Attention: Exploiting Visual Cues](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Beyond_Text-Visual_Attention_Exploiting_Visual_Cues_for_Effective_Token_Pruning_ICCV_2025_paper.html) | 纯 text–visual attention 难以覆盖视觉关键内容；引入视觉线索。 | 扩展打分信息，缓解文本注意力盲点。 | 多信号权重、实际额外计算和跨模型有效性需核测。 |
| 2025 ICCV | [METEOR](https://openaccess.thecvf.com/content/ICCV2025/html/Liu_METEOR_Multi-Encoder_Collaborative_Token_Pruning_for_Efficient_Vision_Language_Models_ICCV_2025_paper.html) | 多视觉编码器之间重复，预算难分配；编码、融合、解码多阶段协同。 | 官方报告 EAGLE 视觉 token -76%、TFLOPs -49%，OCR 表现尤其改善。 | 对特定多编码器结构更有针对性；融合实现/训练成本需算。 |
| 2025 NeurIPS | [SCOPE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ec6b4456c2bdfd04002d7984043c4936-Abstract-Conference.html) | 高注意力 token 扎堆导致语义覆盖不足；显著性和集合覆盖联合贪心选择。 | 明确把覆盖作为目标，降低语义缺失。 | 迭代选择开销、coverage 代理与真实任务一致性需测。 |
| 2025 NeurIPS | [CDPruner](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2433fec2144ccf5fea1c9c5ebdbc3924-Abstract-Conference.html) | 单纯 attention 重复、单纯 similarity 忽视问题；指令条件化 DPP diversity。 | 免训练；官方报告 LLaVA FLOPs -95%、CUDA latency -78%，保留约 94% 准确率。 | DPP 选集成本与高分辨率规模；指令条件化不适合未知后续问题。 |
| 2025 NeurIPS | [Balanced Token Pruning (BTP)](https://papers.neurips.cc/paper_files/paper/2025/hash/5aab3631d0d3131281fb88265db69480-Abstract-Conference.html) | 局部输出影响不能代表后续层全局影响；用校准集分阶段平衡两者。 | 官方报告 78% 压缩时保留 96.7% 平均性能。 | 需要校准集；跨分布和不同基准适应性需测。 |
| 2025 NeurIPS | [Each Complexity Deserves a Pruning Policy (AutoPrune)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/ab4ccdc1fd3d09c5841bda2a7e6bd783-Abstract-Conference.html) | 固定深度策略不区分输入难度；互信息信号映射成样本专属预算曲线。 | 免训练、可约束预算；官方报告 LLaVA-1.5-7B 删 89% token、FLOPs -76.8%，保留 96.7% 准确率。 | 难度估计和互信息计算本身需计时；最难 OCR/定位样本需分层评估。 |
| 2025 EMNLP 主会 | [DART: Stop Looking for “Important Tokens”](https://aclanthology.org/2025.emnlp-main.505/) | attention-importance 排名常逊于随机；按相对于 pivot 的重复性删 token。 | 免训练、FlashAttention 兼容；官方报告删 88.9% token，端到端 1.99×、prefill 2.99×。 | 去重复不等于保留细小独有目标；pivot 选取和多轮信息覆盖需测。 |
| 2025 ACL Findings | [Token Pruning in MLLMs: Are We Solving the Right Problem?](https://aclanthology.org/2025.findings-acl.802/) | 审计随机基线、attention 打分、文本信息和重要性–重复性关系。 | 对实验设计特别重要：提醒不要用单一平均分声称保真。 | 分析论文不是通用压缩算法。**Findings 与 ACL 主会应区分。** |
| 2025 ACL Findings | [PruneVid](https://aclanthology.org/2025.findings-acl.1024/) | 视频跨帧静态冗余与问题无关 token；先时空合并后 LLM 引导裁剪。 | 无训练，官方报告可删 >80% 视频 token。 | 视频时序事件、短时目标和长视频抽帧策略须细查。 |
| 2025 NAACL Findings | [LVPruning](https://aclanthology.org/2025.findings-naacl.242/) | 需要语言相关的视觉 token 评分；交叉注意力导向裁剪。 | 不修改基础 MLLM 参数；官方报告中层可删 90% token、TFLOPs -62.1%。 | 附加 cross-attention 计算与跨模型迁移需核验。 |
| 2025 COLING 主会 | [TRIM: Less is More](https://aclanthology.org/2025.coling-main.508/) | 无条件视觉选择忽视问题；CLIP 图文相似度指导视觉 token 筛选。 | 简单、可在 LLM 前缩短序列。 | CLIP 文本对齐对 OCR、数字、小物体及复杂问句的可靠性需测。 |
| 2026 CVPR | [When Token Pruning is Worse than Random](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_When_Token_Pruning_is_Worse_than_Random_Understanding_Visual_Token_CVPR_2026_paper.html) | 正面检验“重要性”打分失效和视觉信息分布。 | 为随机裁剪/保真评价提供强基线；官方摘要报告 DivPrune+随机在 Qwen2.5-VL-7B 删 50% 时保持 96.9% 性能。 | 分析结论需按模型、裁剪点和任务复核。 |
| 2026 CVPR | [Rethinking Token Reduction / MetaCompress](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Rethinking_Token_Reduction_for_Large_Vision-Language_Models_CVPR_2026_paper.html) | 多轮问答的后续问题不可预知；学习与提示无关的压缩映射。 | 直接面向多轮复用，统一选择/合并目标。 | 需训练；学习到的先验可能漏掉罕见细节或域外后续问题。 |
| 2026 CVPR | [QuietPrune](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_QuietPrune_Query-Guided_Early_Token_Pruning_for_Vision-Language_Models_CVPR_2026_paper.html) | LLM 内晚裁剪省不了视觉编码器计算；通过逆投影 adapter 注入问题指导 ViT 内裁剪。 | 双阶段节省；官方报告 Qwen3-VL/InternVL3 最多 prefill -19%。 | adapter 与模型/projector 绑定；真实端到端时间及 2×2 分组对细节的影响需测。 |
| 2026 CVPR | [SCoRe](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_SCoRe_Salience-Coverage_Reduction_for_Vision_Token_Pruning_in_Vision-Language_Models_CVPR_2026_paper.html) | 显著性与覆盖两段式选择可塌缩；以 weighted k-center 联合优化代表性。 | 官方报告删 94.4% token 保留 95% 性能；提出清晰集合覆盖目标。 | 贪心搜索成本、单一距离空间表达细粒度语义的能力需核测。 |
| 2026 CVPR | [TransPrune](https://openaccess.thecvf.com/content/CVPR2026/html/Li_TransPrune_Token_Transition_Pruning_for_Efficient_Large_Vision-Language_Model_CVPR_2026_paper.html) | attention/相似度有位置偏置；以 token 表征跨层变化加指令注意力打分。 | 免训练，覆盖 LLaVA/Next/Qwen2.5-VL；官方报告 TFLOPs 减半以上。 | token 变化不一定对应任务必要性；计算评分的开销须计入。 |

## 另列：常见误引及预印本

- [FasterVLM](https://arxiv.org/abs/2412.01818) 是 2024 arXiv 版本；本次未找到上述目标顶会的正式论文集条目，不应写作 CVPR/ICML 正式论文。
- [MustDrop](https://arxiv.org/abs/2411.10803) 是 2024 arXiv 版本；本次未找到上述目标顶会的正式论文集条目。
- [TokenPacker](https://arxiv.org/abs/2407.02392) 先以 2024 预印本发布，作者仓库标注后发表于 **IJCV 2025 期刊**；不应列作顶会。
- LLaVA-PruMerge 的早期预印本常被标作 2024，但官方 [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Shang_LLaVA-PruMerge_Adaptive_Token_Reduction_for_Efficient_Large_Multimodal_Models_ICCV_2025_paper.html) 论文集已确认；正式引用用会议年份。
- [Conical Visual Concentration](https://openaccess.thecvf.com/content/CVPR2025/html/Xing_Conical_Visual_Concentration_for_Efficient_Large_Vision-Language_Models_CVPR_2025_paper.html) 的正文仍称算法 PyramidDrop，正式会议题名是 Conical Visual Concentration；CVF HTML 摘要称 ViCo。可在方法描述中说明别名，不要编造另一篇 CVPR 论文。

## 会议版图读法

1. **2023–2024：** ViT 合并奠基，FastV 建立 LLM 层内视觉 token 剪枝基线。
2. **2025：** 争点从“能否删”转向 *删在哪里、删哪些、预算怎么分配*；预编码器选择、多层锥形预算、任务相关性、集合覆盖、FlashAttention 兼容齐头发展。
3. **2025 下半年–2026：** DART、FEATHER 和 CVPR 2026 审计论文质疑单一 attention 排名和平均 VQA 分数；多轮问答、细粒度定位/OCR、真实端到端 latency 成为更有解释力的验证条件。

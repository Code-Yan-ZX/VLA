# 2023--2025 VLM 视觉 Token 压缩技术谱系

> 范围：2023--2025 年顶会/重要正式论文，并补充少量同期预印本以说明技术演化。本文讨论推理阶段进入多模态系统的视觉 token 数量或活跃计算量如何减少，不展开通用量化、蒸馏、纯文本 KV cache 压缩，也不逐篇复述实验表。
>
> 状态核验口径：会议/期刊归属以 CVF Open Access、PMLR、ACL Anthology、OpenReview、NeurIPS Proceedings、AAAI Proceedings、Springer/IJCV 等正式页面为准；只有 arXiv 页面而无正式 proceedings 记录者标为“预印本”。同一工作有多个名称或版本时，以正式发表版本为主。

## 1. 一句话结论

视觉 token 压缩有两个互补问题：

- **视觉侧压缩回答“哪些视觉信息本身重复或低价值”**：利用视觉显著性、patch 相似度、空间邻域、层间冗余等做 pruning、merging、resampling 或 early reduction，通常与具体问题弱耦合，优点是压缩发生早、节省范围大。
- **语言/查询侧条件化回答“相对于当前问题，哪些视觉信息有用”**：利用文本--视觉 attention、指令、查询 token 或浅层 LLM 状态来选 token，任务适应性更强，但往往要先运行若干语言层，且 attention 并不天然等于因果重要性。

二者不是互斥类别。PuMer、CrossGET、SparseVLM 等同时包含视觉冗余处理和跨模态条件化；BLIP-2、InstructBLIP 的 Q-Former 则属于**学习到的查询瓶颈**，不是通常意义上的逐样本硬剪枝。

## 2. 统一分析框架

任何方法都可以放在四条正交轴上，而不宜只按“pruning/merging”单标签分类。

| 轴 | 主要取值 | 核心问题 |
|---|---|---|
| 压缩算子 | pruning；merging；resampling/projector；early/staged reduction | 是删除、聚合，还是用固定/可变数量的新 token 重表达？ |
| 决策信号 | visual-only；text/query-conditioned；learned-query bottleneck | 重要性相对于图像本身、当前文本，还是一组学习查询定义？ |
| 发生阶段 | ViT 内部；视觉编码器/LLM 边界；浅/中层 LLM；decode/KV 阶段 | 能节省哪些模块，何时开始节省？ |
| 训练方式 | training-free；轻量适配/微调；端到端联合训练 | 能否直接套用已有模型，迁移成本多大？ |

### 2.1 四类压缩算子分别做什么

1. **Pruning（删）**
   给每个视觉 token 计算重要性并硬删除低分项。其优势是 token 数和理论 FLOPs 下降直接；缺点是不可逆，细粒度 OCR、计数、小目标和多轮追问容易受损。

2. **Merging（并）**
   按特征相似度、空间邻域或跨模态相关性，把多个 token 聚合成代表 token。它比纯删除保留更多“总量信息”，但可能把边界、多个实例或不同语义平均掉；合并匹配本身也有开销。

3. **Resampling / projector bottleneck（重采样）**
   用少量 learned queries、卷积/变形注意力或粗到细注入，把可变数量视觉特征映射为较短序列。它从架构入口控制预算，适合训练一个紧凑接口；但固定瓶颈可能与图像复杂度、分辨率和任务难度失配。

4. **Early / staged reduction（早减或分阶段减）**
   在 ViT 内、LLM 前或多个 LLM 深度逐步缩短序列。越早压缩，端到端收益通常越大；分阶段策略则允许模型先建立粗跨模态联系再继续压缩，在效率与选择可靠性间折中。

### 2.2 三类条件信号分别做什么

- **Visual-only**：看视觉 token 的自注意力、范数、相似度、空间密度、聚类中心或冗余，不依赖问题。适合图像特征复用、多轮对话和预先缓存，也能在 LLM 前完成压缩；但可能保留“显眼却无关”的区域。
- **Text/query-conditioned**：用输入指令、文本 token 或跨模态 attention 估计相关性。它会为“读门牌”和“汽车是什么颜色”保留不同 patch；代价是需要已知查询，且浅层相关性可能不稳定。
- **Learned-query bottleneck**：少量可训练 query 主动从视觉特征中读取信息，如 Q-Former。它通过训练学习“该抽取什么”，输出长度天然受控；但 query 数通常是静态超参数，且迁移到新域要依赖训练分布。

## 3. 代表工作总表

下表以正式版本为准。“V”表示视觉侧信号，“T/Q”表示文本或查询条件化；“混合”表示两者共同参与。局限列是机制层面的主要风险，不等同于论文实验结论。

| 年份 | 工作 | 正式状态 | 算子 / 位置 | 条件信号与关键机制 | 主要局限 |
|---|---|---|---|---|---|
| 2023 | ToMe | ICLR 2023 | merging；ViT 层内 | V：二分匹配近似 token 并逐层合并，奠定训练外 token merging 范式 | 原本面向通用 ViT；相似不等于对语言任务冗余，细节可能被平均 |
| 2023 | PuMer | ACL 2023 Long | pruning + merging；多模态 Transformer 内 | 混合：文本知情剪枝先去除低相关视觉 token，再做模态感知合并 | 依赖中间跨模态分数；压缩前仍需部分融合计算 |
| 2023 | BLIP-2 | ICML 2023 | resampling；视觉编码器与 LLM 之间 | learned Q：Q-Former 用少量学习 query 从冻结视觉编码器抽取固定长度表示 | 固定 query 瓶颈；需要两阶段预训练，不是即插即用剪枝 |
| 2023 | InstructBLIP | NeurIPS 2023 | resampling；接口层 | T/Q：instruction-aware Q-Former 让 query 抽取受指令条件化 | 依赖指令调优；固定 query 数对复杂图像仍可能不足 |
| 2024 | MADTP | CVPR 2024 | dynamic pruning；视觉/多模态编码过程 | 混合：按样本和层动态减少冗余视觉 token，使压缩率不必完全静态 | 动态控制和训练增加复杂度；跨架构迁移需验证 |
| 2024 | CrossGET | ICML 2024 | token reduction / merging；跨模态模型层内 | 混合：用跨模态引导 token 选择/聚合，使被压缩表示保留另一模态需要的信息 | 需要跨模态交互后才能可靠引导；早期模块节省有限 |
| 2024 | Honeybee | CVPR 2024 | projector resampling；LLM 前 | V/learned：C-Abstractor、D-Abstractor 强化局部性，以卷积或可变形注意力生成紧凑视觉 token | projector 需训练；输出预算仍主要由架构预设 |
| 2024 | FastV | ECCV 2024 | one-shot pruning；浅层 LLM 后 | T/Q：正式题名 *An Image is Worth 1/2 Tokens After Layer 2*；用浅层文本--视觉 attention 在后续层删除低相关视觉 token | 前两层不省；显式 attention 实现依赖明显，多轮查询改变时不可复用 |
| 2025 | HiRED | AAAI 2025 | early reduction；高分辨率视觉输入端 | V：利用视觉编码器已有注意力/显著性，在进入 LLM 前优先保留高信息 patch | 问题无关；高分辨率小字或非显著目标可能被漏删 |
| 2025 | FitPrune | AAAI 2025 | adaptive pruning；LLM 前/推理期 | V：训练外估计样本可压缩性，并为不同图像拟合合适 token 预算 | 预算估计仍是代理目标；极端任务和域外样本稳定性有限 |
| 2025 | Visual Tokens Withdrawal | AAAI 2025 | token withdrawal；LLM 推理层间 | 混合：让视觉 token 完成信息注入后从后续计算中退出，保留文本侧已吸收的视觉信息 | 退出时机难统一；后层若需重新访问细节则不可逆 |
| 2025 | Recoverable Compression | AAAI 2025 | compression + recovery | 混合：压缩主路径外保留恢复机制，在发现信息需求时补回或重建视觉内容 | 恢复路径和状态管理增加系统复杂度，最坏时收益下降 |
| 2025 | TRIM | COLING 2025 | pruning / information-preserving reduction；LLM 前 | 混合：以任务相关性结合视觉信息量选取紧凑 token 子集 | 排序指标仍可能偏向显著区域；硬删除风险保留 |
| 2025 | VisionZip | CVPR 2025 | retention + merging；LLM 前 | V：保留视觉编码器中的 dominant tokens，并把其余 contextual tokens 合并而非全部丢弃 | dominant 判定与当前问题无关；聚合会弱化局部边界 |
| 2025 | PyramidDrop | CVPR 2025 | progressive pruning；多个 LLM 深度 | T/Q：按金字塔式阶段逐步丢弃视觉 token，早层建联、后层加大压缩 | 前段计算仍存在；各阶段比例和层位对模型敏感 |
| 2025 | DyCoke | CVPR 2025 | dynamic context/token compression；视频 LMM | 混合：针对视频时空冗余动态维护紧凑视觉上下文 | 视频专用状态管理复杂；动态操作未必转化为等比例墙钟收益 |
| 2025 | SparseVLM | ICML 2025 | adaptive sparsification + recycling；LLM 层内 | T/Q：文本引导视觉 token 稀疏化，并以 token recycling 减轻误删信息损失 | 依赖中间 attention/相关性；晚压缩不能节省视觉编码和早层成本 |
| 2025 | LLaVA-PruMerge | ICCV 2025 | pruning + merging；LLM 前 | V：先找显著 pivot tokens，再将相似冗余 token 聚到 pivot 周围，兼顾删减与信息汇聚 | pivot 对查询不敏感；聚类会混合邻近但不同的实例/文字 |
| 2025 | VisPruner | ICCV 2025 | early pruning；LLM 前 | V：强调超越 text-visual attention，联合视觉重要性与视觉冗余做训练外选择 | 查询无关选择对任务特定小区域不占优；评分/排序仍有额外开销 |
| 2025 | SparseVILA | ICCV 2025 | sparse token computation；多层/系统级 | 混合：面向图像与视频 LMM，在多阶段利用 token 稀疏性降低计算 | 稀疏算子、批处理和硬件支持决定实际加速，工程耦合较强 |
| 2025 | Feather the Throttle | ICCV 2025 | budget-aware/dynamic pruning | 混合：把 token 保留率视为可调推理节流器，按效率预算调整视觉计算 | 需要可靠预算--精度校准；不同任务的同一比例不可直接比较 |
| 2025 | AdaptMerge | Findings of EMNLP 2025 | adaptive merging；LLM 前/层间 | 混合：兼顾任务相关性和表示多样性，自适应确定合并对象与粒度 | 匹配/聚合本身有成本；软融合仍可能掩盖细粒度差异 |
| 2025 | DART | EMNLP 2025 Main | duplicate-aware reduction | 混合：显式考虑重复视觉信息与任务相关性，避免只按单一 attention 排序 | 去重质量依赖特征空间；对独特但低分的细节仍存在风险 |
| 2025 | MoB | NeurIPS 2025 | dynamic budget allocation | 混合：将不同输入/阶段的视觉 token 预算动态化，而非固定压缩率 | 控制器训练、预算稳定性和批内不规则性带来额外成本 |
| 2025 | VisionThink | NeurIPS 2025 | adaptive visual computation | T/Q：根据任务需求决定视觉信息保留或进一步处理，使视觉预算与推理难度耦合 | 难度判断错误会过早压缩；动态路径降低部署可预测性 |
| 2025 | TokenPacker | IJCV 2025（arXiv 2024） | coarse-to-fine resampling；projector | learned/V：以粗粒度 query 为骨架，从高分辨率特征做 region-to-point 注入，生成紧凑但细节增强的 token | projector 需训练；压缩长度和局部覆盖仍受架构设置限制 |

## 4. 技术谱系：从“通用冗余”到“任务与预算自适应”

### 4.1 2023：两条根系形成

第一条是 **ToMe 式视觉相似度合并**：先承认 ViT token 在深层大量同质，再用低成本匹配减少序列。这条线后来进入 VLM，演化为“显著 token 保留 + 上下文 token 合并”（VisionZip）、“pivot + 聚类”（LLaVA-PruMerge）和同时考虑相关性/多样性的 AdaptMerge。

第二条是 **查询瓶颈与跨模态条件化**。BLIP-2 用 Q-Former 把大量视觉特征重采样成少量 query 输出，InstructBLIP 进一步让指令参与抽取；PuMer 则直接把文本相关性用于视觉剪枝，并用 merging 缓解纯删除。这两类工作共同说明：视觉序列可以在进入大语言模型前显著缩短，但“固定接口”与“按问题动态选择”是两种不同设计。

### 4.2 2024：压缩位置从编码器扩展到 LLM 浅层

Honeybee、TokenPacker 路线把 projector 从简单 MLP 提升为有空间归纳偏置的压缩器，强调**短序列不等于粗糙池化**。MADTP、CrossGET 让压缩随层或跨模态状态变化。FastV 则给出影响很大的经验范式：先让完整视觉 token 经过少量 LLM 层建立文本--视觉关系，再一次性裁掉后续层不需要的 token。

这形成一个基本张力：

- LLM 前压缩：覆盖的计算最多，但尚不知道语言模型真正需要什么；
- LLM 浅层后压缩：相关性更可信，但视觉编码器和浅层 LLM 成本已付出；
- 多阶段压缩：折中两者，但超参数、实现和动态形状更复杂。

### 4.3 2025：四种收敛趋势

1. **从固定率到样本/层/预算自适应**：FitPrune、PyramidDrop、SparseVILA、Feather the Throttle、MoB、VisionThink 不再假设所有图片和问题都应保留同样数量 token。
2. **从单一 attention 到多证据排序**：VisPruner 强调视觉显著性和冗余，DART 强调重复，AdaptMerge 同时考虑相关性与多样性。这是对“attention 即重要性”的修正。
3. **从纯删除到可恢复或信息汇聚**：VisionZip、LLaVA-PruMerge 用 merging 回收被删 token 的上下文；Recoverable Compression、SparseVLM 的 recovery/recycling 思路则为误删留后路。
4. **从图像单轮推理走向视频、动态推理和系统协同**：DyCoke、SparseVILA 等把时空冗余、动态上下文和硬件实际加速纳入问题，不再只报告静态 token 数。

## 5. 视觉侧与语言/查询侧的直接比较

| 维度 | 视觉侧压缩 | 语言/查询侧条件化 |
|---|---|---|
| 优化对象 | 图像内部显著性、冗余、相似性和空间覆盖 | 对当前问题、指令或生成状态的相关性 |
| 可发生的最早位置 | ViT 内或 LLM 前 | 通常需已有文本 query；若依赖 LLM attention，则在浅层之后 |
| 可复用性 | 同一图像的压缩表示可跨问题、多轮复用 | 查询改变通常要重算选择或维护被删 token |
| 主要优势 | 节省链路长；与文本模板较少耦合；适合缓存 | 任务针对性强；能忽略视觉上显著但与问题无关的区域 |
| 主要盲点 | 不知道用户之后会问什么 | 浅层 attention 噪声、提示词敏感、查询漂移 |
| 最适任务 | 通用描述、视觉缓存、视频冗余去除、固定部署预算 | VQA、指令跟随、目标明确的检索/OCR |
| 理想组合 | 先做保守 visual-only 去冗余，再做 query-aware 精筛；对低置信度 token 合并/旁路保存，而非全部硬删 | 同左 |

由此可得到一个对新方法有用的设计原则：**把“压缩多少”和“保留谁”拆开建模**。前者可由图像复杂度、延迟预算和层深决定；后者可融合视觉覆盖、跨模态相关性和不可恢复风险。只用一个 attention 排名同时承担两件事，通常会造成静态比例脆弱或细节误删。

## 6. 共性局限与实验审计要点

### 6.1 信息损失不是平均发生的

- OCR、文档、图表、计数、小目标和空间关系对少数 patch 极敏感；总体 VQA 均值可能掩盖这些失败。
- pruning 的错误不可逆；merging 虽保留总信息，但平均化会模糊字符边界、实例边界和颜色/位置差异。
- 高分辨率切片或视频中，重复背景很多，但“唯一出现一次”的 token 可能恰好最关键，纯频率/相似度规则容易误伤。

### 6.2 条件化分数并非可靠真值

- attention weight 是路由信号，不必然等于删除该 token 后的因果影响；不同头、层和文本位置给出的排序可能冲突。
- 首轮问题上无关的 token，可能在后续追问中变关键。query-conditioned 压缩若要支持多轮对话，应重算、保留可恢复缓存或采用保守视觉底座。
- 生成过程中问题焦点可能漂移；只在 prefill 时一次性选择，不能保证 decode 后期仍覆盖所需视觉证据。

### 6.3 理论 token/FLOPs 不等于墙钟加速

- 动态 top-k、排序、聚类、matching、gather/scatter 和不规则 batch 会抵消部分节省。
- 某些方法需要导出 attention matrix；在 FlashAttention 等融合内核下，显式返回完整 attention 可能破坏原有速度或显存优势。
- 应同时报告 prefill latency、decode latency、峰值显存、吞吐、batch size、输入分辨率和实际保留率，而不只报告 token reduction 或理论 FLOPs。
- 压缩发生在 LLM 第 \(k\) 层后，只节省 \(k+1\) 之后的视觉相关计算；不能把最终序列缩短比例直接当成端到端加速比例。

### 6.4 静态压缩率的可迁移性有限

- 同一保留率跨模型、分辨率、任务和视觉 tokenization 不可直接比较。
- 固定 learned queries 或固定 token budget 对简单图片可能浪费，对密集文档可能过紧。
- 动态预算虽更合理，但必须报告预算控制器开销、方差、最坏精度和 batch 调度影响。

## 7. 对后续研究设计的启示

一个较有研究价值、也便于与既有工作区分的组合框架可以包含：

1. **早期保守去冗余**：在 LLM 前只合并高置信度重复 token，保证空间覆盖和唯一细节不被硬删。
2. **浅层查询条件化排序**：在少量 LLM 层后，用文本相关性重新分配预算，而非只依赖视觉显著性。
3. **风险感知的三分决策**：保留、合并、暂存可恢复，而不是二元保留/删除。
4. **样本与阶段双动态预算**：图片复杂度决定初始预算，跨模态置信度决定后续层的下降曲线。
5. **系统指标闭环**：以真实 latency/显存约束训练或校准预算，避免只优化 token 数代理指标。

关键消融应至少拆开：视觉分数、文本分数、冗余分数、预算控制器、恢复路径；并按 OCR/计数/空间/知识/VQA/视频分别报告，而非仅看综合均值。

## 8. 发表状态与名称审计

### 8.1 容易误标的正式版本

- **LLaVA-PruMerge**：早期以预印本传播，正式版本为 **ICCV 2025**，不应只标“arXiv 2024/2025”。
- **FasterVLM / VisPruner**：相关早期名称为 FasterVLM；正式发表题名为 **VisPruner: Beyond Text-Visual Attention...，ICCV 2025**。
- **TokenPacker**：2024 年已有 arXiv 版本；正式期刊版本为 **International Journal of Computer Vision (IJCV), 2025**。
- **VisionZip、PyramidDrop**：均已有 **CVPR 2025** 正式版本，不应继续标为仅预印本。
- **FastV**：正式论文题名为 *An Image is Worth 1/2 Tokens After Layer 2*，归属 **ECCV 2024**；FastV 是方法简称。
- **CrossGET**：曾出现 ICLR 投稿/撤稿页面；正式归属按 **ICML 2024（PMLR）** 记录。

### 8.2 截至核验时仍按预印本处理的同期工作

以下工作对谱系有参考价值，但在本调查的 2023--2025 窗口内未核验到对应正式 proceedings/期刊记录，因此不与正式顶会论文混排：

| 工作 | 状态 | 可借鉴点 | 使用时注意 |
|---|---|---|---|
| MustDrop | arXiv 预印本 | 多阶段、训练外视觉 token dropping | 不应写成已被某顶会接收 |
| VTC-CLS | arXiv 预印本 | 利用视觉编码器 CLS/视觉侧信号做早期压缩 | 会议归属需以后续正式页面重新核验 |
| G-Prune | arXiv 预印本 | 图结构/关系感知的视觉 token pruning | 机制和实验结论应注明预印本性质 |
| AdaptPrune | arXiv 预印本 | 输入自适应保留率 | 不与正式发表的 AdaptMerge 混淆 |
| VScan | arXiv 预印本 | 面向视觉 token 扫描/选择的效率方法 | 名称近似工作较多，引用时核对作者与版本 |

## 9. 经核验的主要正式入口

以下链接优先给出正式 proceedings/期刊页面；它们也用于校正论文名称和会议归属。

- [PuMer — ACL 2023](https://aclanthology.org/2023.acl-long.721/)
- [BLIP-2 — ICML 2023 / PMLR](https://proceedings.mlr.press/v202/li23q.html)
- [LLaVA-PruMerge — ICCV 2025 / CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Shang_LLaVA-PruMerge_Adaptive_Token_Reduction_for_Efficient_Large_Multimodal_Models_ICCV_2025_paper.html)
- [VisPruner — ICCV 2025 / CVF](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Beyond_Text-Visual_Attention_Exploiting_Visual_Cues_for_Effective_Token_Pruning_ICCV_2025_paper.html)
- [SparseVLM — ICML 2025 / PMLR](https://proceedings.mlr.press/v267/zhang25v.html)
- [AdaptMerge — Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.387/)
- [DART — EMNLP 2025 Main](https://aclanthology.org/2025.emnlp-main.505/)
- [TokenPacker — IJCV 2025](https://link.springer.com/article/10.1007/s11263-025-02433-z)

## 10. 最简谱系图

```text
通用视觉冗余
  ToMe (相似合并)
    -> VisionZip / LLaVA-PruMerge / AdaptMerge / DART

学习查询瓶颈
  BLIP-2 Q-Former
    -> InstructBLIP (指令条件化)
    -> Honeybee / TokenPacker (空间与粗细粒度 projector)

文本相关剪枝
  PuMer (文本知情剪枝 + 合并)
    -> CrossGET (跨模态引导)
    -> FastV (浅层 LLM attention 后一次剪枝)
    -> SparseVLM / PyramidDrop (自适应或分阶段)

早期视觉侧压缩
  MADTP
    -> HiRED / FitPrune / VisPruner

动态预算与可恢复压缩
  Visual Tokens Withdrawal / Recoverable Compression
    -> DyCoke / SparseVILA / Feather / MoB / VisionThink
```

总体上，2023--2025 的主线不是“谁删得最多”，而是从单一静态规则走向三个联合目标：**任务相关性、视觉覆盖/不可恢复风险、真实系统预算**。下一步方法若只提出新的 attention 排名或固定压缩率，创新空间已较窄；更有价值的是证明在多轮查询、细粒度任务和真实推理内核下，动态压缩仍能形成稳定 Pareto 改善。

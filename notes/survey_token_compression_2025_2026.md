# 2025--2026 VLM 视觉 Token 压缩前沿综述

> 检索截止：2026-08-18
> 范围：视觉 token 剪枝、选择、合并、动态预算、视频/流式 token、KV/稀疏访问及真实部署加速。
> 状态口径：只有在 CVF Open Access、NeurIPS Proceedings、ICLR Proceedings、ACL
> Anthology 等正式 proceedings 中查到的工作才标为“已接收”；只有 arXiv 记录且未在本轮
> 正式索引中找到对应版本者标为“仅 arXiv/观察项”。论文报告的速度均视为作者结果，
> 不等于本项目独立复现。

## 1. 执行摘要

2025--2026 已从“找一个更好的 attention top-k 分数”迅速演化为六个相互耦合的问题：

1. **重要性与覆盖联合**：既保留任务相关 token，又避免 top-k 集中于同一物体或区域。
2. **token 生命周期与压缩阶段**：何时删与删谁同样重要；早删省得多，但信息尚未充分融合。
3. **逐样本、逐层动态预算**：预算开始由图像复杂度、问题和推理阶段共同决定。
4. **视频时空分解**：帧选择、帧内压缩、跨帧合并、流式记忆和 KV 管理被分别建模。
5. **OCR/文档/高分辨率专门化**：通用 VQA 均值无法保证小字、笔画、布局和唯一细节安全。
6. **真实系统闭环**：研究开始进入 vLLM/SGLang，但 continuous batching、goodput、
   tail TTFT、KV 压力和 SLO 证据仍明显不足。

最拥挤的赛道是 LLaVA 类模型上的固定比例、training-free、attention/similarity top-k；
最有空间的方向是 native merger/projector 边界、OCR 风险条件化、面向服务负载的动态预算、
真实视频 serving、跨模型稳定性和严格的系统 break-even 分析。

## 2. 检索与核验方法

### 2.1 纳入规则

- 时间：2025-01-01 至 2026-08-18，少量 2024 方法只用于界定技术边界。
- 对象：减少 VLM/MLLM/LVLM 推理中视觉序列长度、活跃视觉计算或视觉 KV 访问的工作。
- 纳入 venue：CVPR、ICCV、ECCV、NeurIPS、ICLR、ACL、EMNLP、AAAI、ACM MM 等；
  同时纳入少量具有系统实现、批判性评估或新机制的高影响预印本。
- 排除：纯图像/视频生成 tokenizer、通用 ViT 分类剪枝、模型权重量化、数据剪枝，以及与
  VLM 推理视觉 token 无直接关系的“compression”工作。

### 2.2 已核验来源

- CVPR 2025/2026、ICCV 2025：CVF Open Access 全量题名索引与论文页。
- NeurIPS 2025、ICLR 2026：官方 proceedings。
- ACL 2025/2026、EMNLP 2025：ACL Anthology。
- 预印本：arXiv 标识符与版本页。
- 部署：vLLM 稳定文档/发布记录，以及论文公开的 SGLang 集成说明。

本轮未连接 Google Scholar、Web of Science、Scopus、OpenAlex、Semantic Scholar 的完整
检索接口，因此不声称覆盖这些数据库，也不报告拼接的引用次数。2026 年下半年尚未发布正式
proceedings 的会议不能凭作者主页或 arXiv 注释提前标为“已接收”。

## 3. 统一技术框架

| 轴 | 典型取值 | 关键权衡 |
|---|---|---|
| 压缩位置 | ViT 内；native merger/projector 前后；LLM 浅层/中层；decode/KV | 越早节省范围越大，越晚任务相关性通常越可靠 |
| 决策信号 | norm/attention；相似度；query alignment；coverage；重建/扰动；learned policy | scalar score 简单，但容易重复、偏置或缺少因果性 |
| 操作 | pruning；merging；aggregation；resampling；skip/withdraw；sparse access | 删除最便宜但不可逆，合并/回收更稳但有额外开销 |
| 预算 | 固定比例；分层 schedule；逐样本；逐请求/SLO | 动态预算精度更好，但会制造不规则 batch |
| 模态范围 | 单图；多图；视频；流式视频；3D/omnimodal | 视频需同时处理空间、时间和历史记忆 |
| 训练要求 | training-free；校准；轻量 scorer；端到端训练/RL | 可部署性、迁移性与最优性能之间存在张力 |

## 4. 2025 正式接收工作

以下只列与视觉 token 效率直接相关的核心簇；“机制”是用于建立技术地图的压缩概括。

### 4.1 CVPR 2025

| 工作 | 机制主线 | 主要边界 |
|---|---|---|
| VisionZip | dominant token 保留 + contextual token 聚合 | query-blind；合并可能平滑细节 |
| ATP-LLaVA | 输入/层自适应 token pruning | 动态策略与模型耦合 |
| DivPrune | diversity/coreset 选择 | 覆盖强，但不直接建模问题相关性 |
| DyCoke | 视频动态上下文压缩 | 状态管理和墙钟收益需单独验证 |
| EfficientLLaVA | 可泛化 auto-pruning | 学习/迁移成本高于纯训练外方法 |
| FlashSloth | 嵌入式视觉压缩 | 依赖特定压缩模块和训练设置 |
| Libra-Merging | importance-redundancy 与 prune-merge 权衡 | 多目标权重及合并损失 |
| PVC | 图像/视频统一的 progressive compression | 阶段与比例超参数较多 |
| TopV | 兼容推理优化的 token pruning | 系统收益依赖内核和序列形状 |
| Hybrid-Level Instruction Injection | 视频 token 的多层 instruction 条件化 | 视频专用，跨模型迁移待验证 |
| Searching Optimal Vision Token Reduction | 搜索压缩位置/配置 | 搜索成本和配置可迁移性 |

说明：VisionZip 等已存在正式 CVPR 版本，不应继续标为“仅 arXiv”。

### 4.2 ICCV 2025

| 工作 | 机制主线 | 主要边界 |
|---|---|---|
| LLaVA-PruMerge | pivot pruning + 相似 token 聚合 | pivot/query 无关，文字边界可能被合并 |
| SparseVILA | 解耦视觉稀疏性并做系统级优化 | 稀疏内核和批处理决定收益 |
| VisPruner / FasterVLM | 视觉线索 + diversity，反思 text-attention 排名 | query-blind 的任务特定盲点 |
| Dynamic-VLM | 简单动态视频 token 压缩 | 视频数据和预算策略依赖 |
| FEATHER | 早期均匀空间覆盖 + 多准则筛选 | 暴露空间偏置，但仍需任务条件化 |
| FrameFusion | 视频 similarity + importance | 跨帧唯一事件可能被误并 |
| VFlowOpt | 信息流差异驱动的渐进策略与 token 回收 | 校准与多阶段执行复杂 |
| METEOR | 多视觉编码器协同 pruning | 多 encoder 成本可能抵消收益 |
| ZipVL | 动态 token sparsity | 动态形状和硬件适配 |
| Skip-Vision | 自适应 token skipping | 跳过不等于序列实际缩短 |
| Pruning All-Rounder | 统一考察多阶段推理效率 | 广度大，部署结论仍依赖实现 |
| Representation Shift | 表征变化信号，兼容 FlashAttention | shift 不必然等于任务因果重要性 |
| Keyframe-oriented Vision Token Pruning | 长视频 keyframe 导向压缩 | 关键帧错误会造成不可恢复损失 |
| Multi-Granular Spatio-Temporal Token Merging | 多粒度时空合并 | matching/merge 开销和细节平滑 |
| AirCache | 跨模态相关性驱动视觉 KV 压缩 | 主要作用于 KV/后段，不省前端视觉编码 |

### 4.3 NeurIPS 2025

| 工作 | 机制主线 | 对领域的意义 |
|---|---|---|
| CDPruner | conditional DPP 联合 relevance 与 diversity | 从逐 token 分数转向 set-level 目标 |
| Balanced Token Pruning (BTP) | 用局部/全局差异校准分阶段策略 | 预算和阶段需要校准而非拍脑袋 |
| Multi-Objective Balanced Covering (MoB) | 双目标 covering 与理论误差界 | 直接反驳简单分数加权即可兼顾目标 |
| SCOPE | saliency + 边际 coverage | coverage 成为主流第二目标 |
| FlowCut | 信息流/依赖关系驱动压缩 | 压缩对象从静态 token 变为信息流 |
| VisionThink | 按任务难度自适应视觉计算 | 视觉预算与推理过程耦合 |
| FastVID | 高效视频 token/推理路径 | 长视频成为独立效率赛道 |
| FlexSelect | 灵活 token selection/budget | 强调输入条件化和弹性预算 |
| VQToken | 紧凑视觉 token 表达 | 从删除扩展到表示重构 |
| HoloTom / dynamic visual-token exit / token policies | token 退出与策略学习 | token 生命周期和 learned policy 升温 |

### 4.4 ACL/EMNLP 2025

| 工作 | 正式状态 | 机制或贡献 |
|---|---|---|
| PruneVid | ACL 2025 | 视频视觉 token pruning |
| RedundancyLens | ACL 2025 | 分析/利用视觉冗余 |
| TAMP | ACL 2025 | 多阶段或任务感知压缩 |
| Token Pruning in MLLMs: Are We Solving the Right Problem? | Findings of ACL 2025 | 控制实验显示复杂 selector 可能不如随机；要求更强审计 |
| AVG-LLaVA | ACL 2025 | 自适应视觉 token/粒度 |
| VidCom2 | EMNLP 2025 main | 视频 token 压缩；正式归属不是 ACM MM |
| CROP | EMNLP 2025 | 压缩/选择与任务相关性 |
| DCP | EMNLP 2025 | 动态压缩策略 |
| METok | EMNLP 2025 | 高效多模态 token 化/压缩 |
| VisiPruner | EMNLP 2025 | 视觉 token pruning |
| SpecVLM | EMNLP 2025 | speculative/高效 VLM 推理 |
| Query-adaptive video selection | EMNLP 2025 | 查询条件化视频选择 |
| LangDC | EMNLP 2025 | language-guided dynamic compression |
| AdaptMerge | Findings of EMNLP 2025 | 相关性与多样性驱动自适应合并 |
| CoViPAL | EMNLP 2025 | 视频感知/压缩协同 |

## 5. 2026 正式接收工作

### 5.1 CVPR 2026：从 scorer 竞赛转向阶段、覆盖、对象和策略

CVPR 2026 的相关论文数量已形成高度拥挤的独立簇。下表按机制归并，避免把几十个相近方法
误读为完全独立的问题。

| 机制簇 | 已接收代表工作 | 领域信号 |
|---|---|---|
| 早期/query-aware | QuietPrune；Mostly Text, Smart Visuals；VISion On Request | 尝试把任务条件提前到低成本位置 |
| 阶段/生命周期 | TransPrune；Variation-aware Vision Token Dropping；One Layer's Trash is Another Layer's Treasure；DUET-VLM | “何时压缩”成为核心变量 |
| coverage/set objective | CoIn；SCoRe；HAWK；VLM-Pruner | importance-only 已不足以构成新意 |
| 误差/因果/功能敏感度 | ApET；ZOO-Prune；IF-Prune；Information Horizon | 从 attention 代理转向近似输出影响 |
| 学习式压缩 | EvoComp；MetaCompress（Rethinking Token Reduction）；HTC-VLM | learned mapping/policy 换取更强表达 |
| object/structure aware | CORE；Merge3D；Proxy3D | 由 patch token 转向对象/结构表示 |
| 文档/OCR/领域 | DocPrune；Prune2Drive；Token-Efficient VLM | 通用固定策略让位于领域风险建模 |
| 图像/视频统一 | UniCompress；Unified Spatiotemporal Token Compression；Blink | 统一多输入形态与动态分辨率 |
| 视频/流式 | MeToM；GroundVTS；Towards Sparse Video Understanding and Reasoning；OASIS；FluxMem | 从离线短视频扩展到流式记忆 |
| KV/稀疏交互 | Revisiting Multimodal KV Cache Compression；VISion On Request | 压缩延伸到 decode 和稀疏访问 |
| 训练与推理一体 | DUET-VLM；EvoComp | 不再局限于冻结模型的后处理 |
| 反例/审计 | When Token Pruning Is Worse than Random；What Do Visual Tokens Really Encode?；Random Wins All | 领域开始重视随机、阶段和机制控制 |

特别值得关注：

- **When Token Pruning Is Worse than Random** 提出 token information horizon：
  同一个 token 的价值会随层改变，错误阶段的“聪明剪枝”可输给随机。
- **CORE** 以外部分割/对象先验生成 object-centric token，代表从 patch 去重转向语义对象。
- **DocPrune** 说明文档问答需要背景、问题和理解阶段的专门策略，不能由通用 VQA 均值代替。
- **ZOO-Prune/ApET/IF-Prune** 共同显示，扰动敏感度、近似误差和信息流正取代裸 attention。
- **CoIn/SCoRe** 进一步确认“相关性 + 覆盖”已成为标准组合，而非稀缺创新点。

### 5.2 ICLR 2026

| 工作 | 机制主线 |
|---|---|
| VisionTrim | training-free 统一视觉 token 压缩 |
| AgilePruner | attention 与 diversity 的自适应经验研究 |
| PruneSID | 结构/信息驱动 pruning |
| LearnPruner | 早期 relevance、后期 redundancy 的分工 |
| Task-Related Compression | 任务相关压缩 |
| PPE | 渐进/策略化 token efficiency |
| FlashVID | 高效长视频处理 |
| FLoC | coverage/局部结构压缩 |
| MARC | 多阶段自适应压缩 |
| SURGE | 动态视觉计算 |
| QueryStream | query-aware 流式视频 |
| VideoChat-Flash | 长视频快速推理 |
| HiDrop | 分层 token dropping |
| IVC-Prune | 输入条件化视觉 pruning |

### 5.3 ACL 2026

| 工作 | 机制主线 |
|---|---|
| VisPCO | 视觉 token 压缩/优化 |
| Official compression evaluation framework | 标准化视觉 token 压缩评估 |
| Vista-LLM | 高效视觉序列处理 |
| TrimTokenator | token trimming |
| CrisPrune | 细粒度/条件化 pruning |
| HiPrune | 分层 pruning |
| CAPA | 自适应压缩 |
| HybridKV | 多模态 KV 压缩 |
| HERMES | 高效多模态推理 |
| MoPrune | 多目标 pruning |
| EMCompress | 高效多模态压缩 |

ACL 2026 出现专门的 compression evaluation framework，是一个重要成熟信号：方法数量
已经多到需要统一预算、模型、数据、速度和统计口径。

## 6. 截至 2026-08-18 的仅 arXiv 高价值观察项

下表中的“仅 arXiv”意为：本轮在已检查的正式 proceedings 中未找到对应版本；未来若出现
正式发表，应按正式版本更新，不能永久沿用预印本标签。

| 工作/标识符 | 状态 | 值得跟踪的原因 | 当前限制 |
|---|---|---|---|
| EVS, arXiv:2510.14624 | 仅 arXiv；有 SGLang 集成报告，并进入 vLLM 视频 pruning 选项 | 少数把视频 token 选择接到主流 serving engine 的工作 | 公开证据偏 BS=1/online TTFT，缺少连续批处理 goodput/SLO |
| QV-PIC, arXiv:2608.12121 | 仅 arXiv | 2026-08 新近的 query/视觉压缩方向 | 发布时间很近，复现与正式 venue 尚待观察 |
| Deployment break-even study, arXiv:2608.03649 | 仅 arXiv | 直接研究压缩开销何时能被下游节省抵消 | 需在更多模型、GPU、batch 和内核上复核 |
| OCR provenance audit, arXiv:2608.00077 | 仅 arXiv | 追踪 OCR/文字证据在压缩链路中的来源与损失 | 领域专用，结论未必外推到通用 VQA |
| ET-Prune, arXiv:2608.01979 | 仅 arXiv | text-rich 场景与动态预算 | 新近预印本，缺少跨模型长期验证 |
| CRISP, arXiv:2607.16326 | 仅 arXiv | 强调更结构化/稳健的视觉压缩 | 尚无正式 proceedings 核验 |
| SPARE, arXiv:2606.18681 | 仅 arXiv | 用子空间/重建而非纯删除保存信息 | 重建开销可能改变真实 break-even |
| ViCoStream, arXiv:2606.19849 | 仅 arXiv | 流式视频压缩与持续上下文 | 需要真实流量、缓存增长和 tail latency 证据 |

这组预印本的共同变化不是“压得更多”，而是开始问三个过去常被跳过的问题：
压缩器本身花多少钱、被删信息能否重建、以及动态/流式负载下是否仍然产生净收益。

## 7. 真实部署与加速证据审计

### 7.1 证据等级

| 等级 | 最低要求 | 能支持的 claim |
|---|---|---|
| E0 代理指标 | token 数、理论 FLOPs、attention complexity | 只能说“理论计算减少” |
| E1 单请求离线 | 固定硬件、batch=1 latency/显存 | 可说单请求有加速，不可推导吞吐 |
| E2 批量推理 | 多 batch、不同长度、端到端计时 | 可说特定离线服务配置有效 |
| E3 serving engine | vLLM/SGLang/TensorRT-LLM 集成，含 scheduler | 可说具备生产引擎可执行路径 |
| E4 服务负载 | continuous batching、goodput、P50/P95/P99 TTFT、SLO、并发与 KV 压力 | 才能支持生产 serving 改善 |

### 7.2 当前事实

- “没有任何生产 serving 集成”的旧结论已经过时。
- vLLM 稳定文档暴露 `--mm-processor-kwargs.video_pruning_rate`，并列出 `evs` 与
  `vidcom2`；vLLM `v0.27.0` 发布记录增加 VidCom2/视频 pruning 支持。
- EVS 报告了 SGLang 集成和 online TTFT 改善，因此至少达到 E3，并触及部分在线指标。
- 但现有公开结果仍主要集中于单请求或低并发 TTFT。尚缺：
  continuous batching goodput、P95/P99 TTFT、不同请求长度混排、KV cache 压力、
  scheduler 公平性、压缩器 CPU/GPU 开销，以及高并发下的 break-even 曲线。

因此，最准确的结论是：

> 视觉/视频 token 压缩已经开始进入主流推理引擎，但“引擎可用”尚不等于“生产负载下
> 稳定增益”；后者仍是明显开放问题。

### 7.3 为什么 token 变少不一定更快

- 排序、top-k、聚类、DPP、matching、gather/scatter 和重建都有固定开销。
- 动态长度造成 batch 内不规则，可能降低张量核利用率和调度效率。
- 若压缩发生在第 \(k\) 个 LLM 层之后，前 \(k\) 层和视觉编码器成本已经支付。
- 显式导出 attention matrix 可能破坏 FlashAttention 的融合与显存优势。
- 对短输入、小模型或低压缩率，压缩器开销可能超过节省，存在硬件相关的 break-even。
- 视频中帧解码、预处理和 vision encoder 可能占主导，仅减少 LLM token 不保证端到端收益。

## 8. 当前热点、饱和问题与开放空间

### 8.1 当前热点

1. **importance + diversity/coverage**：CDPruner、CoIn、SCoRe、DivPrune、FEATHER。
2. **query-aware early compression**：QuietPrune 等尝试在完整 LLM 交互前获得任务条件。
3. **progressive/layerwise token lifetime**：BTP、VFlowOpt、TransPrune、LearnPruner。
4. **per-input dynamic budget**：ATP-LLaVA、Dynamic-VLM、FlexSelect、VisionThink。
5. **视频空间-时间-记忆分解**：FrameFusion、VidCom2、EVS、QueryStream、ViCoStream。
6. **OCR/文档/高分辨率保护**：DocPrune、ET-Prune、OCR provenance audit。
7. **对象中心与可重建压缩**：CORE、SPARE、aggregation/recycling 路线。
8. **部署 break-even 与内核兼容**：vLLM/SGLang 集成、FlashAttention-compatible 方法。

### 8.2 正在饱和的问题

- 在 LLaVA-1.5/1.6 上提出另一个固定比例 attention top-k。
- 把 attention、norm、similarity 做线性加权，却不证明 set-level 或因果差异。
- 只在单一平均 benchmark 上报告精度，缺少 OCR、计数、小目标、图表和文档分层。
- 只报告 token/FLOPs reduction，把离线 batch=1 latency 称为 deployment。
- 用不同模型、分辨率、pixel cap、tokenizer 和保留率直接比较“SOTA”。
- 只测试一个压缩点，不做 iso-token、iso-selector、stage-only 控制。
- 动态预算只报告平均 token 数，不报告方差、最坏样本和批处理代价。

### 8.3 尚有明显空间的问题

1. **native merger/projector-aware compression**
   现有方法多默认视觉 token 已生成，较少严格研究 native spatial merger 前后同一 selector、
   同一预算为何改变信息保真与 OCR 表现。

2. **不可恢复风险建模**
   文字笔画、唯一小目标、图表刻度和跨区域对应关系不能用一般冗余分数替代。需要保留、
   合并、旁路缓存三分决策，而不是统一 hard drop。

3. **面向服务状态的预算控制**
   预算应同时看请求难度、batch 占用、KV 压力、SLO 和 GPU 饱和度，而非只看图像复杂度。

4. **真实视频 serving**
   需要在长短视频混合、连续请求、帧解码和预处理都计入时报告 goodput 与 tail TTFT。

5. **跨模型/分辨率稳定性**
   selector 在 CLIP-LLaVA、Qwen-VL native merger、InternVL dynamic tiles 等架构间可能改变
   含义；迁移必须区分“原则移植”与“官方方法复现”。

6. **严格评估与统计控制**
   至少需要 random、uniform/spatial coverage、iso-token、pixel-cap、skip-vs-delete、
   多 seed/置信区间和 paired significance；动态方法还需按实际 token 分布匹配。

7. **内核与稀疏格式协同**
   研究 selector 是否能在不导出完整 attention、不产生高成本 gather/scatter 的条件下与
   FlashAttention、paged KV、continuous batching 兼容。

8. **多轮与 query drift**
   query-conditioned 压缩在下一轮问题改变后可能失效；需要重算、保守视觉底座或可恢复缓存。

## 9. 对 VLA/RBM 项目的直接含义

RBM 的合理定位不是“普适最强视觉 token scorer”，而是：

- **native-merger-stage / boundary-aware 的受控研究**；
- **query-blind、training-free、偏 OCR/text-dense 鲁棒的默认方案**；
- 用同一 ranking/budget 隔离 merger 前后信息变化，而不是与所有 learned/query-aware
  方法争夺无条件 SOTA。

最接近的学术边界包括：

- FastV/QuietPrune：query-conditioned relevance，但压缩阶段不同；
- DivPrune/CDPruner/CoIn/SCoRe：coverage/set objective；
- TransPrune/Information Horizon/LearnPruner：token lifetime 与阶段；
- DocPrune/ET-Prune/OCR audit：文字密集任务条件化；
- VisionZip/LLaVA-PruMerge/CORE/SPARE：合并、对象化与信息回收。

建议论文主张保持如下强度：

1. native merger 会改变可用于后续选择的信息，且该变化对 OCR/text-dense 任务尤其重要；
2. pre-merger RBM 是一个简单、可解释、无需 query 的稳健默认，不宣称跨所有任务通用胜出；
3. FastV 等 query-aware 方法在 reference-grounded/general VQA 上可以更强，二者体现
   “早期证据保护”和“后期任务相关性”的结构性张力；
4. 所有速度结论限定在实测配置，不把 token reduction 直接等同于生产吞吐提升。

## 10. 推荐实验与报告清单

- 模型：至少一个 native-merger VLM；跨 family 结果标为 directional replication。
- 预算：按实际送入 LLM 的视觉 token 做 iso-token，而非只对齐名义 pruning ratio。
- 输入：固定 pixel cap、tile/frame 数、预处理版本和最大序列长度。
- 对照：no compression、random、uniform/grid、norm、attention/FastV、coverage/diversity。
- 阶段：同 selector 在 merger 前/后或不同 LLM 层的 stage-only control。
- 任务：通用 VQA、OCRBench/TextVQA/DocVQA、图表、计数、小目标/空间关系、视频。
- 质量：平均分之外报告类别分层、最坏组、paired bootstrap/置信区间。
- 系统：端到端 TTFT、prefill、decode、峰值显存、实际 token、batch、goodput、P95/P99。
- 开销：单列 scorer/top-k/merge/gather 时间，并画保留率/输入长度/batch 的 break-even。
- 可复现性：模型与 engine commit、CUDA/attention backend、精度、GPU、warm-up、测量轮数。

## 11. 结论

截至 2026-08-18，视觉 token 压缩已经从一个“冗余 patch 删除技巧”发展为横跨表示学习、
信息流、动态控制和推理系统的成熟子领域。2026 年的论文密度说明固定比例 scalar scorer
已接近饱和；新的可信贡献需要回答至少一个更深问题：压缩发生在哪个不可逆边界、如何保护
稀有证据、预算如何随请求与系统状态变化，或算法怎样在真实 serving engine 中形成稳定净收益。

对本项目而言，最有辨识度且与现有结果一致的路线，是把 RBM 收紧为 native-merger-stage
与 OCR 鲁棒性的因果/受控证据，并用严格 iso-token、stage-only 和系统指标阻断过度外推。

## 12. 核心来源入口

### 正式 proceedings

- CVPR 2025 all papers: https://openaccess.thecvf.com/CVPR2025?day=all
- ICCV 2025 all papers: https://openaccess.thecvf.com/ICCV2025?day=all
- CVPR 2026 all papers: https://openaccess.thecvf.com/CVPR2026?day=all
- NeurIPS 2025 main proceedings:
  https://proceedings.neurips.cc/paper_files/paper/2025/vol38-main-conference
- ICLR 2026 proceedings: https://proceedings.iclr.cc/paper_files/paper/2026
- ACL Anthology event index: https://aclanthology.org/events/
- VidCom2, EMNLP 2025 main: https://aclanthology.org/2025.emnlp-main.98/
- Controlled pruning audit, Findings of ACL 2025:
  https://aclanthology.org/2025.findings-acl.802/

### 代表性正式论文页

- DivPrune:
  https://openaccess.thecvf.com/content/CVPR2025/html/Alvar_DivPrune_Diversity-based_Visual_Token_Pruning_for_Large_Multimodal_Models_CVPR_2025_paper.html
- ATP-LLaVA:
  https://openaccess.thecvf.com/content/CVPR2025/html/Ye_ATP-LLaVA_Adaptive_Token_Pruning_for_Large_Vision_Language_Models_CVPR_2025_paper.html
- TopV:
  https://openaccess.thecvf.com/content/CVPR2025/html/Yang_TopV_Compatible_Token_Pruning_with_Inference_Time_Optimization_for_Fast_CVPR_2025_paper.html
- LLaVA-PruMerge:
  https://openaccess.thecvf.com/content/ICCV2025/html/Shang_LLaVA-PruMerge_Adaptive_Token_Reduction_for_Efficient_Large_Multimodal_Models_ICCV_2025_paper.html
- SparseVILA:
  https://openaccess.thecvf.com/content/ICCV2025/html/Khaki_SparseVILA_Decoupling_Visual_Sparsity_for_Efficient_VLM_Inference_ICCV_2025_paper.html
- FEATHER:
  https://openaccess.thecvf.com/content/ICCV2025/html/Endo_Feather_the_Throttle_Revisiting_Visual_Token_Pruning_for_Vision-Language_Model_ICCV_2025_paper.html
- FrameFusion:
  https://openaccess.thecvf.com/content/ICCV2025/html/Fu_FrameFusion_Combining_Similarity_and_Importance_for_Video_Token_Reduction_on_ICCV_2025_paper.html
- VFlowOpt:
  https://openaccess.thecvf.com/content/ICCV2025/html/Yang_VFlowOpt_A_Token_Pruning_Framework_for_LMMs_with_Visual_Information_ICCV_2025_paper.html
- CDPruner:
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/2433fec2144ccf5fea1c9c5ebdbc3924-Abstract-Conference.html
- Balanced Token Pruning:
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/5aab3631d0d3131281fb88265db69480-Abstract-Conference.html
- Multi-Objective Balanced Covering:
  https://proceedings.neurips.cc/paper_files/paper/2025/hash/6818dcc65fdf3cbd4b05770fb957803e-Abstract-Conference.html
- MetaCompress / Rethinking Token Reduction:
  https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Rethinking_Token_Reduction_for_Large_Vision-Language_Models_CVPR_2026_paper.html
- CORE:
  https://openaccess.thecvf.com/content/CVPR2026/html/Lei_CORE_Compact_Object-centric_REpresentations_as_a_New_Paradigm_for_Token_Merging_in_LVLMs_CVPR_2026_paper.html
- CoIn:
  https://openaccess.thecvf.com/content/CVPR2026/html/Du_CoIn_Coverage_and_Informativeness-Guided_Token_Reduction_for_Efficient_Large_Multimodal_CVPR_2026_paper.html
- ZOO-Prune:
  https://openaccess.thecvf.com/content/CVPR2026/html/Kim_ZOO-Prune_Training-Free_Token_Pruning_via_Zeroth-Order_Gradient_Estimation_in_Vision-Language_CVPR_2026_paper.html
- Information Horizon:
  https://openaccess.thecvf.com/content/CVPR2026/html/Wang_When_Token_Pruning_is_Worse_than_Random_Understanding_Visual_Token_CVPR_2026_paper.html
- DocPrune:
  https://openaccess.thecvf.com/content/CVPR2026/html/Choi_DocPrune_Efficient_Document_Question_Answering_via_Background_Question_and_Comprehension-aware_CVPR_2026_paper.html
- IF-Prune:
  https://openaccess.thecvf.com/content/CVPR2026/html/Sun_IF-Prune_Information-Flow_Guided_Token_Pruning_for_Efficient_Vision-Language_Models_CVPR_2026_paper.html

### 预印本与部署入口

- EVS: https://arxiv.org/abs/2510.14624
- QV-PIC: https://arxiv.org/abs/2608.12121
- Deployment break-even study: https://arxiv.org/abs/2608.03649
- OCR provenance audit: https://arxiv.org/abs/2608.00077
- ET-Prune: https://arxiv.org/abs/2608.01979
- CRISP: https://arxiv.org/abs/2607.16326
- SPARE: https://arxiv.org/abs/2606.18681
- ViCoStream: https://arxiv.org/abs/2606.19849
- vLLM multimodal processing documentation:
  https://docs.vllm.ai/en/stable/serving/engine_args.html
- vLLM releases: https://github.com/vllm-project/vllm/releases

# VLM 视觉 Token 压缩方向地图（截至 2026-08-18）

## 1. 不应只分成“视觉处理”和“文字处理”

主流工作几乎都在压缩视觉 token；文字通常不是被压缩的对象，而是作为任务条件，
帮助模型判断哪些视觉证据与当前问题有关。更准确的研究地图有四个轴：

1. **在哪里压缩（stage）**
   - 视觉编码器内部：减少后续 vision encoder 计算。
   - 编码器输出 / projector / native merger 前后：减少送入 LLM 的 token。
   - LLM 浅层或中层：经过图文交互后再剪枝，相关性更准但已支付部分 prefill 成本。
   - 解码 / KV cache：按生成步检索或稀疏访问视觉 token。
2. **依据什么决策（signal）**
   - 纯视觉显著性：CLS attention、L2 norm、局部能量、边缘等。
   - 冗余与覆盖：相似度、聚类、diversity、coverage、重建误差。
   - 文本 / query 条件：图文 attention、QK 相关性、问题 embedding、任务标签。
   - 学习式决策：轻量 scorer、gate、蒸馏、RL 或可微预算。
3. **如何缩减（operator）**
   - pruning：直接丢弃。
   - merging / aggregation：把被删 token 的信息并入保留 token。
   - resampling：用固定或动态数量的 learned queries 汇聚视觉信息。
   - sparse attention / retrieval：保留 token，但只让部分层或生成步访问。
4. **保留多少（budget）**
   - 全局固定比例。
   - 随层递减的 progressive schedule。
   - 按图像、问题或任务动态分配。
   - 面向延迟、显存、吞吐或 SLO 的系统预算。

## 2. 近三年的演化

### 2024：证明“视觉 token 大量冗余”

- **FastV（ECCV 2024）**：在 LLM 浅层用图文 attention 给视觉 token 排序并剪枝。
- **PruMerge**：用重要性筛选加相似 token 合并，代表 prune + merge 路线。
- 核心问题仍是：能不能在不训练或少训练的条件下，大幅删 token 而少掉点。

### 2025：从单一显著性转向多信号、动态和渐进压缩

- **VisionZip（CVPR 2025）**：视觉侧保留 dominant tokens，并把其余内容聚合为
  contextual tokens；强调“重要性 + 上下文”。
- **DivPrune（CVPR 2025）**：指出只追逐高 attention token 会保留大量相似内容，
  因而改用 diversity。
- **ATP-LLaVA（CVPR 2025）**：按输入和层动态决定剪枝强度。
- **SparseVLM（ICML 2025）**：利用文本引导的图文关系、逐层稀疏化和 token recycling。
- **VisPruner（ICCV 2025）**：质疑直接使用 text-to-visual attention 排名，转而利用
  visual cues 与 diversity。

这一阶段形成了两个竞争目标：**任务相关性**与**视觉覆盖度**。前者容易漏掉尚未被问题
直接点名但回答所需的证据；后者容易保留“看起来不同”却与问题无关的区域。

### 2026：研究对象从“分数”上升到信息流、阶段和策略

CVPR / ICLR 2026 的密集工作表明，单纯提出另一个 scalar scorer 已趋于饱和：

- **QuietPrune（CVPR 2026）**：query-guided early pruning，尝试更早获得任务条件。
- **IF-Prune（CVPR 2026）**：按 information flow 判断 token 对最终语言输出的贡献。
- **Hi-Lo Prune（CVPR 2026）**：剪前同时评估局部重要性和被删除后损失。
- **SCoRe / CoIn（CVPR 2026）**：显著性与覆盖度联合，而非只取 top-attention。
- **When Token Pruning Is Worse than Random（CVPR 2026）**：研究视觉信息随层演化，
  说明某些“聪明剪枝”在错误阶段会比随机选择更差。
- **MMTok（ICLR 2026）**：把保留集合写成 coverage maximization，强调 set-level
  目标而不是逐 token 独立打分。
- **LearnPruner（ICLR 2026）**：早期 relevance 剪枝与后期 redundancy 剪枝分开，
  明确承认不同阶段应解决不同问题。
- 学习式 / RL 路线开始直接学习 token policy、每样本预算或稀疏模式。

## 3. 当前顶会最活跃的六条方向

1. **早期 query-aware 压缩**
   - 目标：同时得到视觉侧早剪的速度和语言侧 query-aware 的准确性。
   - 难点：在图文深度融合前，视觉特征与问题 embedding 往往没有可靠的逐 patch 对齐。

2. **阶段感知与层级策略**
   - 目标：回答“在哪一层、以什么节奏剪”，而不只是“剪哪些”。
   - 趋势：relevance、redundancy、coverage 在不同层承担不同角色。

3. **显著性、覆盖度与冗余的联合优化**
   - 目标：避免 top-k attention 聚集在同一物体或局部区域。
   - 趋势：set objective、聚类、facility-location、重建误差和多尺度选择。

4. **动态预算与学习式策略**
   - 目标：简单图少留、复杂图和文字密集图多留，并由 query / task 决定预算。
   - 难点：控制器训练成本、跨模型泛化、实际 batch 中变长 token 的调度收益。

5. **OCR、文档、图表与高分辨率专门化**
   - 目标：保护小文字、笔画、布局和跨区域对应关系。
   - 关键事实：通用 VQA 平均分可能掩盖文字证据已经被不可逆删除。

6. **真实系统加速**
   - 目标：从 token/FLOPs 减少走向 TTFT、吞吐、显存、KV cache、连续批处理和 SLO。
   - 难点：算法减 token 不等于 serving engine 中等比例提速，干预位置决定可部署性。

## 4. 我们的 RBM 在地图中的位置

RBM 不是“又一个更好的重要性分数”，而是在研究一个被大量工作混用、但很少被严格控制的
变量：**native spatial merger 前后，选择同一批预算的 token，结果为何不同。**

- 它属于 **boundary / merger-aware stage analysis**。
- 它使用 query-blind L2，是为了隔离 stage，而不是声称 L2 是最优 scorer。
- 其核心贡献是：native merger 会重排显著性并降权文字笔画；在 text-dense/OCR 任务上，
  若先 merge 再选，后续再聪明的排序也可能无法恢复已丢失的细粒度证据。
- FastV 代表另一端：在 LLM 内获得 query-conditioned relevance，适合 reference-grounded
  场景，但位置较晚、系统集成更困难，且无法补救 merger 前已经消失的信息。

因此，RBM 与当前前沿的正确关系不是“视觉法对文字法”，而是：

> **早期视觉证据保护** 与 **后期查询相关性判断** 之间存在结构性张力；stage 决定了
> 一个方法可见什么信息、能够节省多少计算，以及哪些错误已经不可逆。

## 5. 对后续研究判断的直接结论

- “再换一个 training-free scorer”赛道非常拥挤，除非能给出新的机制或严格反例。
- query-aware + dynamic budget 已是主流热点，仅做组合通常不够新。
- OCR-safe 的早期压缩仍有价值，但必须处理图文对齐、空间布局和 native merger。
- stage law、信息不可逆性和严格 iso-model / iso-token / iso-selector 控制，是本项目
  比单纯追 SOTA 数字更清晰、也更难被替代的学术位置。
- 下一代更完整的方法应是分阶段的：早期保护不可恢复的细粒度证据，后期再按 query
  消除任务无关冗余；但本项目已有六项扩展的负结果，不应在投稿前重新开放方法搜索。

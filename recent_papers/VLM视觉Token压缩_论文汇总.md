# VLM 视觉 Token 压缩 — 论文汇总

> 用途：给 agent 做「缝合 + 系统研究」的弹药库。除了你要的 **论文名 / 摘要 / 链接 / 时间**，我额外加了三列：
> - **类型**：免训练（training-free，改推理即可）/ 需训练（training-based，要训模块或 LoRA）。你起步应优先免训练那批。
> - **核心机制（打分依据 → 缩减方式）**：把每个方法拆成可替换的零件——「怎么判断 token 重不重要」→「怎么减（剪枝 / 合并 / 重采样）」。这一列就是给你做 A+B 组合用的。
> - **发表**：会议/期刊或 arXiv。
>
> ⚠️ 说明：摘要列是我**改写的概述**（非原文摘录），用于速览；正式引用前请以原文为准。少数经典论文的 arXiv 号凭记忆填写，投稿前建议再核一遍链接。

---

## 一、经典基线（你已经在用的 + 两个绕不开的地基）

| 论文（方法） | 摘要 | 链接 | 时间 | 类型 | 核心机制（打分 → 缩减） | 发表 |
|---|---|---|---|---|---|---|
| **FastV** — An Image is Worth 1/2 Tokens After Layer 2 | 发现视觉 token 在 LLM 深层高度冗余，在某一浅层（如第 2 层）之后按注意力一次性剪掉大部分视觉 token，即插即用。 | https://arxiv.org/abs/2403.06764 | 2024-03 | 免训练 | LLM 层平均注意力 → 某层后一次性剪枝 | ECCV 2024 |
| **LLaVA-PruMerge** — Adaptive Token Reduction | 用视觉编码器 [CLS]–patch 注意力选关键 token，再把被删 token 按相似度合并回保留 token，自适应压缩率。 | https://arxiv.org/abs/2403.15388 | 2024-03 | 免训练 | ViT [CLS] 注意力 → 剪枝 + 相似度合并 | ICCV 2025 |
| **SparseVLM** — Visual Token Sparsification | 用文本 token 引导，挑出与问题相关的视觉 token 做稀疏化，并回收部分被删 token 的信息。 | https://arxiv.org/abs/2410.04417 | 2024-10 | 免训练 | 文本–视觉相关性 → 稀疏化 + token 回收 | ICML 2025 |
| **PyramidDrop** — Pyramid Visual Redundancy Reduction | 金字塔式在 LLM 不同层逐级递增地丢弃视觉 token，越深保留越少，兼顾精度与开销。 | https://arxiv.org/abs/2410.17247 | 2024-10 | 免训练 | 分层冗余 → 逐层递增丢弃 | CVPR 2025 |
| **FasterVLM** — [CLS] Attention is All You Need | 指出 LLM 内部注意力有偏、不可靠，改用视觉编码器 [CLS] 注意力，在 token 进入 LLM **之前**就剪枝。 | https://arxiv.org/abs/2412.01818 | 2024-12 | 免训练 | ViT [CLS] 注意力（去 LLM 偏置）→ 入 LLM 前剪枝 | arXiv |
| **VisionZip** — Longer is Better but Not Necessary | 从视觉编码器注意力选出少量「主导 token」，其余按相似度合并成 context token，训练/免训练两版。 | https://arxiv.org/abs/2412.04467 | 2024-12 | 免训练 | ViT 注意力选主导 → 剪枝 + 合并为 context | CVPR 2025 |

---

## 二、近期新方法（2025 – 2026，按时间倒序）

| 论文（方法） | 摘要 | 链接 | 时间 | 类型 | 核心机制（打分 → 缩减） | 发表 |
|---|---|---|---|---|---|---|
| **PRUNESID** — Synergistic Importance-Diversity | 同时考虑「重要性」和「多样性」做剪枝：既保留重要 token，又避免保留一堆彼此冗余的 token。 | https://arxiv.org/abs/2603.09480 | 2026-03 | 免训练 | 重要性 + 多样性协同打分 → 剪枝 | ICLR 2026 |
| **E-AdaPrune** — Energy-Driven Adaptive Pruning | 用随机 SVD 估计各 token 的「能量」，据此自适应分配 token 预算并剪枝，额外延迟极低（每图约 8ms）。 | https://arxiv.org/abs/2603.05950 | 2026-03 | 免训练 | 能量（随机 SVD）→ 自适应预算 + 剪枝 | arXiv |
| **AgilePruner** — Empirical Study of Attention & Diversity | ⭐ 一篇「注意力 vs 多样性」的经验研究 + 自适应剪枝方法——**正是你想做的系统研究范式**，值得精读。 | https://arxiv.org/abs/2603.01236 | 2026-03 | 免训练 | 注意力 + 多样性（系统对比）→ 自适应剪枝 | ICLR 2026 |
| **VisionTrim** — Unified Training-Free Compression | 统一多种信号的免训练视觉 token 压缩框架，追求「一套方法通吃多种设置」。 | https://arxiv.org/abs/2601.22674 | 2026-01 | 免训练 | 统一重要性信号 → 免训练压缩 | ICLR 2026 |
| **FocusUI** — Position-Preserving Token Selection (UI) | 面向 UI/GUI grounding 的领域专用方法：选 token 时保留位置信息，避免破坏界面元素定位。 | https://arxiv.org/abs/2601.03928 | 2026-01 | 需训练 | 位置保持的 token 选择（UI 领域） | CVPR 2026 |
| **HybridToken-VLM** — Hybrid Token Compression | 用离散语义锚点 + 连续瓶颈做混合压缩，在极端压缩（压到 1 个视觉 token）下仍稳住语义。 | https://arxiv.org/abs/2512.08240 | 2025-12 | 需训练 | 离散锚点 + 连续瓶颈 → 极端压缩至 1 token | arXiv |
| **PPE** — Positional Preservation Embedding | 针对「合并会丢位置信息」的问题，提出位置保持嵌入，让 token 合并后仍保留空间位置。 | https://arxiv.org/abs/2510.22936 | 2025-10 | 需训练 | 合并 + 位置保持嵌入 | arXiv |
| **Fourier-VLM** — Frequency-Domain Compression | 换个思路：在**频域**压缩视觉 token，用低频成分表示，避免直接在空间域挑 token。 | https://arxiv.org/abs/2508.06038 | 2025-08 | 需训练 | 频域低频成分 → 压缩表示 | arXiv |
| **METEOR** — Multi-Encoder Collaborative Pruning | 针对「多视觉编码器」VLM，让多个编码器协同决定剪哪些 token。 | https://arxiv.org/abs/2507.20842 | 2025-07 | 免训练 | 多编码器协同信号 → 剪枝 | arXiv |
| **AdaTP** — Attention-Debiased Token Pruning (Video) | 视频 LLM：先纠正注意力的系统性偏置，再据此剪枝，缓解「注意力打分不准」的老问题。 | https://arxiv.org/abs/2505.20100 | 2025-05 | 免训练 | 去偏后的注意力 → 剪枝（视频） | arXiv |
| **AdaReTaKe** — Adaptive Redundancy Reduction (Video) | 视频 LLM：跨时间和层自适应分配压缩比，让长视频在同等预算下「看」更多帧。 | https://arxiv.org/abs/2503.12559 | 2025-03 | 免训练 | 时空冗余 → 自适应分配压缩比（视频） | arXiv |
| **PLPHP** — Per-Layer Per-Head Pruning | 不再用全局统一比例，而是**逐层逐头**自适应剪枝，粒度更细。 | https://arxiv.org/abs/2502.14504 | 2025-02 | 免训练 | 逐层逐头注意力 → 细粒度自适应剪枝 | arXiv |
| **RedundancyLens** — Redundancy for Decoder-Only MLLMs | 揭示 decoder-only MLLM 处理视觉 token 的冗余规律，并据此设计高效压缩。 | https://arxiv.org/abs/2501.19036 | 2025-01 | 免训练 | 处理冗余分析 → 剪枝 | arXiv |

---

## 三、综述 & 评测基准（做系统研究必看）

| 论文 | 摘要 | 链接 | 时间 | 类型 | 备注 | 发表 |
|---|---|---|---|---|---|---|
| **A Survey of Token Compression for Efficient MLLMs** | 图/视频/音频 MLLM token 压缩的全面综述，带分类体系和 GitHub 论文库，入门/查漏必备。 | https://arxiv.org/abs/2507.20198 | 2025-07 | — | 综述（配套 awesome-list） | TMLR 2026 |
| **EffiVLM-BENCH** | 统一评测各类**免训练**加速方法（token 剪枝 + KV cache 压缩）的基准，方便公平横比。 | https://arxiv.org/abs/2506.00479 | 2025-06 | — | 评测基准 | arXiv |
| **Are We Using the Right Benchmark** | 指出现有 token 压缩的评测方式有偏差（在某些 benchmark 上"虚高"），提出更合理的评测框架。 | https://arxiv.org/abs/2510.07143 | 2025-10 | — | 评测框架 / 批评性工作 | arXiv |

---

## 给 agent 的两点提示

1. **拆零件做组合**：看「核心机制」列——打分依据大致就几类（LLM 层注意力 / ViT [CLS] 注意力 / 文本-视觉相关性 / token 间冗余相似度 / 能量 / 频域），缩减方式就三类（剪枝 / 合并 / 重采样）。缝合就是在这两维里换着配（如：FasterVLM 的 [CLS] 打分 + PRUNESID 的多样性约束 + PLPHP 的逐层策略）。

2. **别只看单一 benchmark**：`EffiVLM-BENCH` 和 `Are We Using the Right Benchmark` 都在警告——同一方法在不同 benchmark / 不同 token 预算下结论会翻。所以任何"A+B 更好"都要跨 **多 benchmark × 多 backbone × 多预算 × 多种子** 验证，并同时报**精度**和**真实延迟/FLOPs**，否则很可能是噪声。这恰好是 agent 自动化最该发力的地方。

> 需要的话，我可以把这张表补充更多维度（比如每个方法的官方代码仓库链接、在 LLaVA-1.5-7B 上保留 ~128 token 时的精度、是否支持 Qwen2-VL 等），或者按「打分依据 / 缩减方式」重新做一张矩阵图，方便 agent 直接按格子组合。

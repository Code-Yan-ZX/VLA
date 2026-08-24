# MALT Goal Mode — 实时研究日志

> 分支 `exp/deferred-rbm-n200`；开始 2026-08-25。目标：从 MALT-1 已验证机制出发，回答 transient-token 一额外 block 收益的因果来源，并发现更完整的方法结构（如 MALT-Memory/KV）。工作名未锁定。
> 最终报告：`experiments/malt_method_discovery.md`。

## 0. 冻结事实与边界（不重述完整，见任务书）
- MALT-1 精确配置（repo 惯例）：`Qwen/Qwen3-VL-8B-Instruct`, bf16, eager,
  `--mode rankbridge --r 0.75 --rb-fuse quota --rb-rho 1.0 --fastv-k 1`,
  seed 0, greedy, `--max-tokens 32`, docvqa `--max-pixels 600000` 其余 0。
- 层级事实（sweep 报告 §5.3 + 实测 layer_visual_counts）：`--fastv-k 1` 下
  transient tokens 在 **layer 0 与 layer 1 两个 block** 完整参与，layer 1 结束后删除
  （textvqa 中位 [752,752,188×34]）。任务书"H1=参与第一个 decoder block"在 repo
  语义下即 fastv-k 1 = 2 blocks；H2/H3/H4 的屏蔽/回退作用于 transient 存活的两层。
- keep-set 与 immediate-RBM 100% 相同（rho=1.0 → 纯 pre-merger L2 top-25%，query-blind）。
- 因果 mask 为 strict causal（`torch.triu(...,diagonal=1)`）。序列顺序：
  system text → vision_start → image_pads → vision_end → question text。
  故 causality 下：question text 可读 transient K/V；system text 不可；anchor 只可读
  其前面的 transient。

## 1. 设计要点（写入时记录）
- **H4 关键机制发现**：transient 的 layer-1 K/V 由其 layer-0 更新后的 hidden 投影
  （含 deepstack 注入）。H4（K/V-only，跳过 transient 自身 attention/MLP/更新）
  在 layer 1 的 transient K/V 来自 base embedding(+deepstack)。故 H4≠H1 是有意义的
  行为差异，正好回答"transient 自身 hidden/MLP 更新是否必要"。
- **实现策略**：H2/H3 用注意力 mask（把 (text/anchor query, transient key) 置 -inf）；
  H4 用"回退"行为等价（layer 0 后把 transient hidden 恢复到输入嵌入，再让 deepstack
  注入），真实计算节省另作建模，二者严格区分。
- **探针 manifest**：`eval/subsets/{bench}_explore64.jsonl`，与 n=200 锁 0 重叠，
  固定确定性抽样（id 排序 + 等距 64）。

## 2. 进度
- [x] 建 manifest（textvqa/docvqa/gqa/ocrbench × 64，disjoint 已验证）
- [x] runner 实现 H0–H4（`--malt-ablate`，identity 路径零改动；tiny-model 单测全绿）
- [ ] Gate A smoke（n=8×4×6 臂，运行中）
- [ ] n=64×4 全量 + official scorer
- [ ] Phase 2 候选设计
- [ ] Gate B / Gate C
- [ ] 最终报告 + novelty audit + GO/NO-GO

## 2b. 实现要点（已在代码验证）
- H2/H3 = 注意力 mask 屏蔽 (text|anchor query, transient key)，作用于 transient 存活
  的两层（idx ≤ prune_idx=1）；H4 = layer 0 后把 transient hidden 回退到 inputs_embeds
  （deepstack 注入之前），使 layer-1 transient K/V 来自 raw features。
- 验证：`scripts/test_malt_ablations.py` 全绿，含 B4（DynamicCache.update 钩子捕获
  pre-crop 的 layer-1 transient V == v_proj(ln1(raw inputs_embeds))）——证明回退语义
  精确成立；prune 后 cache 被裁剪，transient K/V 已删除（需在写入时捕获）。
- 纯函数：rb_rho1_keep_set（query-blind anchor set）== rankbridge rho=1 keep set。

## 2c. 效率模型（H4 真实计算节省，与行为模拟严格区分）
MALT-1：layer 0-1 全部 L token 全量；layer 2+ 仅 n_text+kept。
H4 真实 ragged 实现：transient 每层跳过 Q-proj、自身 query 的整行 attention
（QK^T 行 + softmax + V 加权行）、o_proj 行、MLP、residual/LN；保留 K/V proj
（供他人读取）+ 作为 keys 参与他人 attention 的不可跳过部分。
FLOPs 估算：transformer 层 per-token 约 MLP 2×H×4H + attn（QKV 3×H×d + 2×H×d +
Oproj H×d）。H4 可省 ~45–55% per-transient-token per-layer FLOPs。准确数字待
layer_visual_counts 到位后按 ΣN_l、ΣN_l² 与真实实现路径计算。

## 3. 提交（每 Gate 一次）
- `ff6e594` Phase-1 setup：explore64 manifest + `--malt-ablate` runner + 单测 + 日志。

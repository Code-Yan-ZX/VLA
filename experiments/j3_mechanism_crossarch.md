# J3/J3b/J3c — 机制跨代复制（Qwen2.5-VL，2026-07-24）

## 结果（官方指标）

**stage law @Qwen2.5（L2，n=64，r=0.75）**：docvqa pre 0.664 > post 0.531（+13.3pp）；textvqa pre 0.719 > post 0.349（+37.0pp）→ **L2 符号不变 ✓**（与 Qwen3 同向）。

**M3 swap 对照 @Qwen2.5**：
- n=64 batched：docvqa swap 0.687 vs pre 0.664（+2.3pp，答案一致 22/64）；textvqa swap 0.734 vs pre 0.719（+1.6pp，28/64）。
- n=16 seq=1（排除 V1 encoder-cache replay 批处理错排假设）：docvqa swap 0.730 vs pre 0.538（**+19.2pp**，8/16）；textvqa swap 0.750 vs pre 0.625（+12.5pp，7/16）。
- **swap≡pre 不复制**；seq=1 后排除了批处理错排；Jaccard 判别未成（--save-unit-scores 不写 kept indices）。swap 显著胜 pre（随机排序不可能在 text-dense 胜 L2-pre 19pp）→ 二选一未决：(a) qwen2vl window-attention reverse_indices 使 swap 捕获分数与 merged token 序错位（swap 实际用另一更优排序=实现 artifact）；(b) merger 输出 batch 依赖（subset merge ≠ full merge）= 真实值差异。

**selector 不变性 @Qwen2.5（n=64）**：L2 符号不变 ✓；**attn（质心代理）失效**——docvqa 符号反转（pre_attn 0.530 < post_attn 0.605）、绝对值低（textvqa pre_attn 0.552 vs l2 0.719）。

## 入稿口径（红线）

- **M3 因果声明限 Qwen3-VL**（swap≡pre 200/200 字节一致，text 198/200）；Qwen2.5 写"swap 对照不精确复制（identity 7–8/16，swap 高于 pre，根因未决：序错位 artifact 或 batch 依赖合并），因果分解不声称推广；Qwen2.5 机制佐证 = M1 ranking 去相关 + stage-law 结果（pre 大胜 post）+ VZ≡post 字节一致"。
- selector 不变性：Qwen3 双 selector 符号不变（既有）；Qwen2.5 = L2 不变、attn 代理失效，如实写（代理信号族特异，非 stage law 反例——L2 是论文 selector）。
- 若审稿质疑 Qwen2.5 swap：rebuttal 补 kept-set Jaccard 诊断（runner 加 kept-indices 输出，~1h 工作量）。

## 产物
runs/v3_crossarch_cells/{j3,j3b,j3c}_*.json + j3_qwen2vl_mechanism.sh + j3b_mechanism_followup.sh + j3c_swap_jaccard.sh。

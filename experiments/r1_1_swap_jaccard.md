# R1-1 — pre-vs-swap kept-set Jaccard 诊断 + R2-2 特征 tap 核实（2026-07-28）

## Part A — Jaccard 结果（n=32, seq=1, r=0.75, L2, 官方指标）

runner 补丁：`--save-unit-scores` 现对 mode=pre（pruner 掩码处）与 mode=post+swap（_process_image_input 处）写 `per_sample[i].kept_indices`（FIFO+n_units guard，warmup 首条丢弃，仿 attach_hybrid_unit_scores）。8 cell：{qwen3vl,qwen2vl}×{pre, post+swap}×{docvqa,textvqa}。attach 率 30–31/32（其余为 encoder-cache replay），swap fallback=0、queue leftover=0。

| cell | Jaccard 均值 | min/med/max | frac=1 | chance k/f | pre(官方) | swap(官方) | Δ(swap−pre) | 答案一致 |
|---|---|---|---|---|---|---|---|---|
| qwen3vl/docvqa | **1.000** | 1/1/1 | 1.00 | 0.25 | 0.467 ANLS | 0.481 | +1.4pp | 31/32 |
| qwen3vl/textvqa | **1.000** | 1/1/1 | 1.00 | 0.25 | 0.458 acc | 0.458 | +0.0pp | 32/32 |
| qwen2vl/docvqa | **1.000** | 1/1/1 | 1.00 | 0.25 | 0.600 ANLS | 0.667 | +6.7pp | 13/32 |
| qwen2vl/textvqa | **1.000** | 1/1/1 | 1.00 | 0.25 | 0.688 acc | 0.698 | +1.0pp | 13/32 |

## 判读（两假设三选一 → 实为结构性定论）

**排序错位（分数层）否证**：swap 与 pre 在**同一 merger 输入**上取分（qwen2vl 同为 window 序），top-k 索引集合逐图恒等（Jaccard≡1，随机基线 0.25）。j3 遗留假设"swap 捕获另一更优排序"被否。

**merger 值 batch 依赖（假设 b）排除**：Qwen2.5-VL PatchMerger = 逐 token LayerNorm+MLP（qwen2_5_vl.py L533-551，无跨 token 运算）→ 子集 merge ≡ 全集 merge，单元值不可能依赖 batch/子集。

**Qwen2.5 残差根因 = reverse_indices 序置换（实现层，结构性）**：Qwen2.5-VL 原生 visual.forward 先 window 置换、merger 后用 `reverse_indices` 恢复空间序（qwen2_5_vl.py 3 处引用；qwen3_vl.py **0 处** → Qwen3 无置换故字节一致 200/200）。由此：
- pre 补丁必须跳过恢复（剪枝后形状不匹配崩溃）→ pre 把 k 个 token 以 **window 序** 配 **raster M-RoPE 位** 送入 LLM（位置-内容配对错位）；
- swap 在 window 序取 top-k 索引、却施加于**空间序**的 _process_image_input splits → 索引集合相同（Jaccard≡1）但**物理保留单元被 reverse_indices 置换**；
- 二者叠加 ⇒ 同集合同值、答案仅 13/32 一致，swap 略胜（+1~7pp）= pre 的位置错位 handicapp 大于 swap 的选择置换。这是**对照实验的实现混杂，非方法/排序/值的科学差异**。

**对 §4 causal 措辞建议**（因果声明可保、且可加强为跨架构 selection 级）：
1. M3 声明改写为："swap 在**两架构**上逐图复现 pre 的保留索引集（Jaccard=1.000，n=30–31/cell，chance=0.25）→ pre>post gap 100% 来自 ranking（selection 级因果、架构通用）；Qwen3-VL 上进一步字节一致（swap≡pre 200/200 + 本实验 31/32、32/32，1 例差异为 temp=0 kernel 非确定性）"。
2. Qwen2.5 段改为："swap≡pre 在输出处不复制（identity 13/32），根因已定：该代 window-attention 置换 + 剪枝兼容补丁跳过 reverse_indices 恢复（序/位置实现混杂；merger 值 batch 依赖已按逐 token LN+MLP 结构排除）。机制佐证以 M1 ranking 去相关 + stage law（pre 大胜 post）+ VZ≡post 为主。"
3. 勿再写"Qwen2.5 根因未决"（已决）；勿声称 Qwen2.5 上 swap 字节复现。

## Part B — 特征 tap 核实（R2-2）

**结论：pre 打分特征 = 首个被调用 merger 的输入。qwen3vl = deepstack_merger_list[0] 的输入 = ViT block-8 输出（中间层，非最终层）；qwen2vl = 主 merger 输入 = ViT 最终 block 输出。** 三证：
1. vllm qwen3_vl.py L637-656：deepstack mergers 在 block 循环内（layer 8/16/24 后）调用，主 merger 在循环后调用 → deepstack_0 最先触发；
2. 模型 config：Qwen3-VL-8B vision depth=27、deepstack_visual_indexes=[8,16,24]；Qwen2.5-VL-7B depth=32、无 deepstack；
3. 全部历史 run diag：qwen3vl `mask_computed_at='deepstack_0'`、qwen2vl `='main'`（本次 8 cell 亦同）。

审稿人所疑"block-8 deepstack 输入"**与实现相符（对 Qwen3-VL 字面正确）**；真正偏松的是 paper §3.1/§3.3 的"merger-input features"（读者会理解为主 merger 输入=最终层特征，对 Qwen3-VL 不成立）。**§3 建议改**：加一句 "On Qwen3-VL the first-called merger is deepstack_merger_list[0] (deepstack_visual_indexes=[8,16,24]), so unit scores are computed on block-8 outputs; the single resulting mask is shared by all mergers, including the main merger that consumes the final-block output. On Qwen2.5-VL (no deepstack) scores are computed on the main merger's input (final-block output)."（runner --mask-ranking help 文本已一致，无需改码）。§5 limitation 可补：Qwen3-VL 的 ranking 取自中间层特征（架构常量、确定性 tap），非最终层——经验上有效，描述上应如实。

## 产物

runs/r1_1_swap_jaccard/{r1_1_{qwen3vl,qwen2vl}_{pre,swap}_{docvqa,textvqa}_n32.json, jaccard_summary.json, *.log, _campaign.log}；补丁 src/v3_premerger/v3_premerger_runner.py（save_kept/kept_log/attach_kept_indices，仅 --save-unit-scores 路径，py_compile 过，λ=0/非 save 行为不变）；scripts/{r1_1_swap_jaccard.sh, analyze_r1_1_jaccard.py}。

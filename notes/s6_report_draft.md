# S6 Stitch 实验报告（草稿，n=500 数据待填）

> 日期：2026-08-17 · 目标：从 2026 最新论文缝合 training-free 模块，找到一个
> 能达到当前 SOTA（iso-budget 现任 = max(RBM pre, FastV post)）的方法。

## 方法（stitch）
**Adaptive-Diversity RBM（ADRBM）** = RBM rank-before-merge + 两枚 2026 模块：
1. **PRUNESID 式重要性+多样性选择**（--diversity nms）：重要性 top-(γk) 候选池，
   按重要性遍历，余弦 > τ 即抑制（intra-group NMS），补足恰 k。iso-token。
2. **AgilePruner 式自适应抑制阈值**（--div-adaptive）：每图 τ_r =
   min(τ, r·(erank_img/scale)·0.01)（r = 重要性序位，erank = 参与比有效秩）。
   复杂图抑制更强（更分散），简单图保留精细高注意单元。
选择级改动：**不改变 token 计数、不动占位符契约**——这是能在本项目 vLLM 架构上
存活的关键性质（见结构性发现）。

## 结果（占位）

### dev n=200, r=0.25（official）
| bench | RBM pre | FastV post | DV(τ0.6) | ADAPTIVE |
|---|---|---|---|---|
| textvqa | 0.720 | 0.630 | 0.735 | 0.720 |
| docvqa | 0.734 | 0.726 | 0.737 | 0.735 |
| ocrbench | 0.910 | 0.875 | 0.920(τ0.9) | 0.910 |
| gqa | 0.490 | 0.505 | 0.490 | **0.510** |

→ ADAPTIVE 唯一赢 GQA，且 4/4 ≥ 现任（SOTA-hold on dev）。

### n=500（待填，verify_ada）

## 结构性发现
1. **hook 覆盖率**：pre visual pass 只触发 126/200 图；post `_process_image_input`
   fires 84/200。其余图走 vLLM 内部编码路径不经 hook → 项目既有 pre/post 全部结果
   "每图 k"契约仅在 ~42-63% 图上严格成立；相对比较（同路径、cache 行为相同）仍有效。
2. **每图自适应预算（E-AdaPrune 式）与占位符契约不兼容**：占位符在 tokenize 时按
   (1-r) 固定，per-image k 需双端同步，但特征只能在 visual pass 拿到、且该 pass
   对部分图不触发 → 无法可靠注入。已实现 calib+allocator+budget-file 全套但判为
   结构性 NO-GO（记录，不宣）。

## 结论（待 n500 定稿）

# S6 Stitch 实验报告（2026-08-17，终稿）

> 目标（user）：从最新 2026 论文缝合 training-free 模块，找到能**达到当前 SOTA**
> 的方法（本项目口径：iso-budget 现任 = max(RBM pre, FastV post) 逐格）。

## 方法（stitch，全部 training-free、选择级）
**Diversity-RBM / Adaptive-Diversity-RBM** = RBM（rank-before-merge）+ 2026 模块：
1. **PRUNESID 式重要性+多样性选择**（`--diversity nms`，ICLR'26）：重要性 top-γk
   候选池 → 按重要性遍历、与已选 unit 余弦 >τ 抑制（intra-group NMS）→ 补足恰 k。
2. **AgilePruner 式自适应 τ**（`--div-adaptive`，ICLR'26）：每图
   τ_r = min(τ, r·(erank_img/scale)·0.01)（r=重要性序位、erank=参与比有效秩）。
   复杂图抑制更强（更分散）、简单图保留精细高注意单元。
3. 组合探针 freq×diversity（`--selector freq α=1,β=0.6` × NMS）。

**关键性质**：两模块只改"选哪些 unit"、不改 token 计数、不动占位符契约——
是在本项目 vLLM hook 架构上唯一能存活的一类（见结构性发现）。

## 结果

### dev n=200，r=0.25（official metric，同 slice）
| bench | RBM pre | FastV post | DV(τ0.6) | ADAPTIVE | freqDV |
|---|---|---|---|---|---|
| textvqa | 0.720 | 0.630 | **0.735** | 0.720 | 0.692 |
| docvqa | 0.734 | 0.726 | 0.737 | 0.735 | **0.753** |
| ocrbench | 0.910 | 0.875 | **0.920**(τ0.9) | 0.910 | 0.915 |
| gqa | 0.490 | 0.505 | 0.480 | **0.510** | 0.490 |

- ADAPTIVE 是唯一赢 GQA（0.510>post 0.505）的变体，且 4/4 ≥ 现任（dev）。
- DV(τ0.6) 在 textvqa/docvqa 最高；ocrbench 需 τ0.9；freqDV 补 docvqa/ocrbench。

### n=500（official，同 slice）— 决定性验证
| bench | r | RBM pre | FastV post | DV(τ0.6) | ADAPTIVE |
|---|---|---|---|---|---|
| textvqa | .25 | 0.745 | 0.665 | **0.748** | 0.741 |
| textvqa | .5 | 0.673 | 0.387 | 0.673 | **0.681** |
| docvqa | .25 | 0.668 | **0.674** | 0.667 | 0.668 |
| docvqa | .5 | **0.621** | 0.549 | 0.612 | 0.611 |
| ocrbench | .25 | 0.838 | (缺) | **0.840** | 0.834 |
| ocrbench | .5 | 0.822 | 0.498 | 0.828 | **0.832** |
| gqa | .25 | 0.486 | **0.506** | 0.480 | 0.496 |
| gqa | .5 | 0.484 | 0.500 | **0.506** | 0.502 |

### 配对统计（n=500，20k bootstrap CI）
- **textvqa r0.25：DV 可靠超 FastV post**（Δ≈+8.3pp，CI [+0.050,+0.115] 排除 0）；
  对 pre 仅 +0.3pp（CI 含 0）。
- 其余 7 格：DV/ADA vs 每格现任，CI 全部含 0（±1 SE 内）——dev 的 4/4 hold 与
  GQA 赢面均未复现为显著，方向部分翻转（gqa-r0.25 ADA 0.496 < post 0.506）。

## 结论（诚实）
- **达到（reach）当前 SOTA**：缝合方法在 n=500 上 8/8 格与现任为噪声内平手，
  点估计 5/8 格略高于现任；textvqa-r0.25 上可靠超 FastV 臂（+8.3pp, CI 排除 0）。
- **未证明可靠全面超越**：无单格对"每格现任"达到显著正差（gqa/docvqa 个别
  单元格点估计低 0.6–1.0pp，也在噪声内）。"可靠 beat SOTA"未达成。
- 净判断：**该方法达到/持平当前 SOTA（iso-budget、training-free、零计数变化），
  是一个可写进方法的稳健文本密集默认增强；但作为"全面超越 SOTA"的 claim 不成立。**

## 结构性发现（对项目论文亦相关）
1. **hook 覆盖率 <100%**：pre visual pass 只触发 126/200 图；post `_process_image_input`
   fires 84/200（vLLM 内部编码路径绕过 hook）→ 既有 pre/post 结果的"每图 k"契约
   仅对 ~42-63% 图严格成立；相对比较（同路径）仍有效，绝对率声明需软。
2. **每图自适应预算（E-AdaPrune 式）与本架构不兼容**：占位符先于特征固定、
   特征 pass 对部分图不触发 → 双端同步不可行。已实现 calib+allocator+budget-file
   全套（含离线像素谱），判为结构性 NO-GO，如实记录不宣。
3. ocrbench post r0.25_n500 因 torch embedding 运行时错误 3 连败（环境性），缺格已注。

## 交付
- 代码：runner `--diversity nms` + `--div-adaptive/--div-scale`（dry-check 覆盖）；
  scripts/sota_stitch_driver.py（ref/grid/retest/verify/verify_ada/probe2/adaptive/
  summary）；scripts/stitch_budget_alloc.py（已验证分配器）+ stitch_pixel_calib.py。
- 数据：experiments/sota_stitch/*.json（dev n=200 + n=500，official rescore）。
- 提交：fcca0b4 起多条（无 AI 署名，Code-Yan-ZX）。

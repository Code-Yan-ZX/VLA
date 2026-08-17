# S6 — 2026-Paper-Module Stitching (goal: beat current SOTA at iso budget)

> 2026-08-14 · 目的：从 2026 最新论文缝合 training-free 模块，做创新方法实验，
> 目标 = 在 4-core 套件（textvqa/docvqa/ocrbench/gqa）× r∈{25%,50%} × 同 token
> 预算下，逐格打败现任基准 = max(RBM pre, FastV post)。论文暂不考虑。

## 已判定的架构铁律（决定哪些模块可缝）
- **占位符契约**：patch_processor 在 tokenize 时把占位符缩放为 round(full·(1−r))；
  pre 路径的 `_patched_pii` 用 `pruner.k_units` 切分。单图请求下**占位符数 == 视觉输出数**
  必须逐图相等，否则 vLLM 对齐错（这也是 Direction A POST 图像的隐患来源）。
- 推论 1：**改"选哪些 unit、数量不变"（多样性）零风险** → 本轮主业。
- 推论 2：**改"每图数量"（自适应预算）必须让占位符也自适应** → 离线预算文件
  （calib pass 先出谱统计，eval pass 两侧同读 per-index k_i）+ max-num-seqs=1。

## 管线
- runner：src/v3_premerger/v3_premerger_runner.py（qwen3vl 族，1×A40，vLLM）。
- driver：scripts/sota_stitch_driver.py（ref/grid/retest/verify，官方 rescore 选型）。
- 成本：n=200 cell ~1–3 min；n=500 ~1.5–2 min。极便宜。

## Phase 1 — Diversity-NMS（PRUNESID 模块，已完成并开工）
- `--diversity nms`：重要性 top-(γ·k) 候选池 → 按重要性遍历,与已选 unit 的余弦
  > τ 则抑制（intra-group NMS）→ 补足至正好 k。iso-token，默认 bit-identical。
- 参考:AgilePruner 的结论（attention 对"集中证据"简单图、diversity 对"分布特征"
  复杂图即 text-dense;diversity 过量会升幻觉) → 高 τ 保 OCR 笔画（笔画互异,
  天然过 NMS）,低 τ 展空间覆盖救 GQA/scene。
- Grid：τ∈{0.6,0.75,0.9}×γ∈{1.25,2.0} @ r=0.25, n=200, 4 bench。
- **门控**：赢 ≥1 个"当前 post 领先"格（textvqa/gqa）+ OCRBench/DocVQA 无 >1pp
  回退 → 升 n=500；全 4 格同时 ≥ 现任才算"当前 SOTA 达阵"。

## Phase 2 — 自适应预算（E-AdaPrune 谱能量 / PRUNESID 动态压缩比 / AgilePruner erank）
- 已写离线分配器 scripts/stitch_budget_alloc.py（spectral τ / erank → iso-token
  budget,clamp 0.5,largest-remainder,穷举验证过）。
- runner 待加：`--budget-calib out.json`（pre r 小时,记录 per-image {f, svd_desc}）、
  `--budget-file alloc.json`（patch_processor 与 begin_pass 同读 per-index k_i,
  max-num-seqs=1,FIFO + fallback 沿用 adaptive 模式防御模式）。
- AgilePruner 升级：NMS 阈值自适应 τ_i = rank_order_i × (erank_img/erank_avg × 0.01)
  (cap),可取代静态 div_tau —— Phase 2 的 diversity 升级。

## 证据纪律
- 每格出官方 rescore;ref 必须同 slice;iso-token 由分配器保证;GPU 串行,driver 带锁。
- 已完成改动：fcca0b4（runner diversity + driver + allocator）。
## Phase-1 结果（2026-08-14，dev n=200, official rescore, r=0.25）
| bench | RBM(pre) | FastV(post) | DV_best(τ) | 结论 |
|---|---|---|---|---|
| textvqa | 0.720 | 0.630 | **0.735**(0.6) | DV +1.5pp vs pre, +10.5pp vs post |
| docvqa | 0.734 | 0.726 | **0.737**(0.6) | DV +0.3pp vs pre |
| ocrbench | 0.910 | 0.875 | **0.920**(0.9) | DV +1.0pp vs pre |
| gqa | 0.490 | **0.505** | 0.490(0.75) | DV 未胜；post 仍领先 |

- **Passpoint**：βg's合计 no >1pp 回归 + 3/4 PRE 领先格 DOMINATE → 多样性-NMS 在
  text-dense 家族把 RBM 再推高（iso-budget、零计数变化）。γ knob 在 r=0.25 退化
  （pool≈全量）；真正旋钮是 τ。
- **缺口**：GQA（scene/object）一格仍由 query-blind 权重+空间覆盖无法闭合。候选补救：
  (a) Phase-2 谱预算给复杂图更多 token（E-AdaPrune 方向——GQA 图若是高复杂度→多 token）；
  (b) AgilePruner 自适应 τ（简单图→低 τ 保 attention 细节）；
  (c) budget × diversity 联合。

## 机制发现（2026-08-14，重要）
- **hook 覆盖率 < 100%**：pre 路径 visual.forward（pre-hook）只触发 n=126/200 图
  （seq=8 和 seq=1 相同），post 路径 `_process_image_input` fires=84/200。其余图由
  vLLM 内部（vision-encoder cache / 分离编码路径）处理，不经任何 hook → 不作剪枝。
  影响：(a) 本项目既有 pre/post 全部结果的"每图 k"契约仅在 ~42-63% 图上严格成立，
  整体是混合压缩率——相对比较（同路径、cache 行为相同）仍有效，绝对率声明要软；
  (b) **预算的自适应（每图 k）必须让占位符与剪枝同步自适应**，这在"单图请求 +
  占位符先于特征确定"契约下不可行（预算文件按 index 消费会在双端错位）。
- 已实测验证（dbg pre seq=8: visual_calls=87, vc_total_imgs=126; calib seq=1 仍 126）。
- AgilePruner 自适应 τ（tau_r = min(tau, r·erank/scale·0.01)）是**选择级**（不改计数、
  不动占位符）→ 无此结构性障碍，可直接探。

## Phase-1b 结果（2026-08-17, dev n=200, official, r=0.25）
| bench | RBM pre | FastV post | DV(τ0.6) | **ADAPTIVE**(τ0.75,scale25) |
|---|---|---|---|---|
| textvqa | 0.720 | 0.630 | **0.735** | 0.720 |
| docvqa | 0.734 | 0.726 | **0.737** | 0.735 |
| ocrbench | 0.910 | 0.875 | **0.920**(τ0.9) | 0.910 |
| gqa | 0.490 | 0.505 | 0.490 | **0.510** ✅ |
- **ADAPTIVE-NMS（AgilePruner tau_r = min(τ, r·erank/scale·0.01)）是唯一赢 GQA 的变体**
  （0.510 > post 0.505），且 textvqa/docvqa/ocrbench 三格 ≥ 现任 → dev 上 4/4 SOTA-hold。
- DV(τ0.6) 在 textvqa/docvqa 更高；ocrbench 需 τ0.9。单一方法取舍：ADAPTIVE 全 hold、
  DV(τ0.6) 更尖。n=500 验证中（verify_ada）。

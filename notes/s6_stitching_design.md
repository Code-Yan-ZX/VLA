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
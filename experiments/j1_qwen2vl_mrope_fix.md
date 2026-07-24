# J1 — Qwen2.5-VL-7B mrope 修复与验证（PASS）

**根因**：vLLM `Qwen2_5_VLForConditionalGeneration.get_mrope_input_positions` 按**完整**图像网格推进位置游标，而 runner `patch_processor` 已把占位符缩到 k=round(full×(1−r)) → 位置数组比 prompt 长，`_calc_mrope_positions` 静默截断 → **尾部文本 token 继承 2D 网格位置**（t 轴冻结）。Qwen2.5-VL block-mrope [16,24,24]、θ=1e6 把错误集中到整块维度 → 输出崩坏（修复前 pre/post acc≈0.004–0.005）；Qwen3-VL interleaved [24,20,20]、θ=5e6 把同样过冲分散 → 容忍（单架构时期未暴露）。

**修复**：`setup_qwen2vl_mrope_fix`（runner 内，仅 family==qwen2vl 且 r>0 时装载）——实例级覆写 `get_mrope_input_positions`，按**实际占位符数 k**（数 image_token_id 连续段）推进游标；图像 token 取前 k 个网格位（与原截断逐位一致），仅修正尾部文本位置。r=0 逐位退化为原版；qwen3vl 分支完全不碰。

**验证（2026-07-24，GPU A40，gmu 0.55，containment 口径，官方口径 J2 重评分）**

| cell | acc | skip | ptid | mrope_fix |
|---|---|---|---|---|
| none GQA n16（卫检，期望 ~0.5） | 0.438 | 0 | 400 | —（r=0 不装载）|
| pre GQA r.75 n8（快验） | 0.875 | 0 | 126 | 9/9 触发 |
| **pre GQA r.75 n50（预注册 ≥0.40）** | **0.680** | 0 | 126 | 51/51 |
| **post GQA r.75 n50（预注册 ≥0.40）** | **0.640** | 0 | 126 | 51/51 |
| pre TextVQA r.75 n50（次级） | 0.780 | 0 | 274 | 51/51 |
| qwen3vl 回归 pre GQA r.75 n8（期望 ~0.3–0.55） | 0.750 | 0 | 94 | —（不变）|

**结论**：PASS。跨架构差异确认 = mrope 布局（block vs interleaved），与原机制 claim 自洽。**早期跨代信号**：Qwen2.5-VL GQA r0.75 pre 0.68 ≥ post 0.64（弱占优，与 Qwen3 同模式；官方口径+官方 GQA exact-match 下预期整体下移 ~4pp，待 J2 n=200 确证 tie/弱占优）。

**产物**：runs/v3_crossarch_cells/j1_*.json、j1_smoke_wait.sh。下一步：J2 跨代矩阵（21 cell + GQA tie n=200 双模型 + 官方 rescore）。

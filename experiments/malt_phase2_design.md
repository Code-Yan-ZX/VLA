# MALT Phase-2 设计（候选方案，待 Phase-1 结果锁定）

> 状态：**DRAFT — 方向由 Phase-1 因果结果决定，未锁定任何候选**。
> 规则：≤3 个机制驱动候选；每个候选先写 hypothesis / 与 MALT-1 唯一差别 /
> 预计 accuracy / 预计 FLOPs-latency / 可证伪条件；禁止网格搜索。
> 候选必须满足：一个 block、training-free、无第二次选择、无动态路由、无额外
> 可训练参数、permanent anchor identities 不变、无 averaging/OT/直接 mixing。

## 0. Phase-1 决策树（结果到来后按此选择）
| Phase-1 结果 | 主通路 | 候选方向 |
|---|---|---|
| H2≈H0（且 H4≈H1）| text←transient | C1 read-only transient visual memory（K/V-only 真实 ragged）|
| H3≈H0（且 H4≈H1）| anchor←transient | C2 purity-preserving anchor read（同 C1 结构，framing 为 anchor 读 write-only buffer）|
| H2≈H3≈H0 | 双通路 | C3 dual-path（text+anchor 都读 raw transient K/V）|
| H4<H1（refinement 必要）| — | C-alt: transient 一层轻更新（attention 无 MLP）后再删（计算介于 H1 与 H4 之间）|
| H2≈H3≈H4≈H1 | 无 K/V 读取通路 | 负结果：增益非读取所致 → 检查 deepstack/softmax/位置效应，冻结 MALT-1 |

## 1. 候选模板（占位，待锁定填充）

### C1：Read-only transient visual memory（K/V-only）
- **hypothesis**：MALT-1 增益来自"transient 作为一次性视觉 K/V memory 被读"，
  transient 自身无需 attention 自更新/MLP（H4≈H1）；真实 ragged 实现省算力。
- **与 MALT-1 唯一差别**：transient 在 layer 0-1 跳过 query-attention 行、
  o_proj 行、MLP、residual/LN，仅计算 K/V 投影并留在 KV cache 供读；隐藏态不更新。
- **预计 accuracy**：= H1（行为等价于 H4；若 H4≈H1 则无损）。
- **预计 FLOPs**：layer 0-1 每 transient 省 ~50%（Q-proj+attn 行+o_proj+MLP）；
  真 ragged 下 ΣN_l 约 H1 的 f_kv 比例（见 analyze_malt_compute.py）。
- **可证伪条件**：H4 ≠ H1（accuracy 掉 >0.5pp macro）→ 方向否决。

### C2：Purity-preserving anchor write
- **hypothesis**：MALT-1 增益主要来自 anchor 读取 transient K/V（H3≈H0）；
  用 read-only buffer 保持 anchor 纯度（anchor 只通过 attention 写，无 feature
  mixing/aggregation，符合"attention-only write"约束）。
- **与 MALT-1 唯一差别**：仅暴露 transient K/V 给 anchor 查询（text 读取是否保留
  待 Phase-1 定）；transient 不更新自身。
- **预计 accuracy**：= H4（若 anchor 是唯一接收者）。
- **预计 FLOPs**：同 C1（transient 省自更新）。
- **可证伪条件**：H3≈H1（anchor 读取无作用）→ 方向否决。

### C3：Dual-path（text+anchor 双接收）
- **hypothesis**：H2≈H0 且 H3≈H0（两通路都重要）→ 双接收原始 transient K/V。
- **与 MALT-1 唯一差别**：transient K/V-only（自更新跳过），text+anchor 均读。
- **预计 accuracy**：= H4（若 H4≈H1）。
- **预计 FLOPs**：同 C1。
- **可证伪条件**：H4≠H1。

## 2. 候选实现路径
- H4 行为等价 = 直接复用 `--malt-ablate kv_only`（已实现、已单测）。
- 真实 ragged 计算节省：需自定义 eager kernel（query-selective）；smoke/explore
  用行为等价版验证 accuracy，计算节省单独建模（analyze_malt_compute.py）。
- Gate B 通过后唯一候选进 Gate C（n=200 paired vs MALT-1，无调参）。

## 3. 禁止项复查
- 不引入 learned scorer/router；不加第二个选择；不加训练参数；不改 anchor
  identities；不 averaging/OT/mixing；不改论文正文；≤3 候选。

## 4. 新增对照（textvqa n=64 预览驱动，2026-08-25）
- textvqa 全 6 臂：H0=0.703 / H1=H2=H3=H4=nb≈0.81–0.83。**所有因果删除均不
  消除增益**；消融生效但答案差异仅 1–4/64。→ 怀疑 positional/mrope 混淆
  （H0 用 vllm-mimic 重编号，deferred 臂全用 native 坐标）。
- **h0n = `--mode pre --r-pre 0.25 --mrope native`**（immediate-RBM + native
  坐标）：隔离位置混淆。若 h0n ≈ MALT-1 → 增益主因是位置布局而非 transient
  K/V 读取 → 候选 = **native-coordinate immediate pruning**（同精度、K=0 算力，
  严格 Pareto 优于 MALT-1）；若 h0n ≈ h0 → 增益确为 transient 机制 → 回退到
  C1/C2/C3（K/V-only memory）方向。
- 注意：textvqa 是文本密集、偏好长生命周期的数据集，结论以 4-bench 数据为准。

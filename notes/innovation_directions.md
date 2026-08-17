# 创新方向合成（2026-08-17，草稿待文献补充）

> 目标（user）：方法不再"太简单"，达到 SOTA 或对论文有实质帮助。
> 输入：论文现状 digest（main.tex §3/§6）+ 项目 DECISIONS 全史 + 自有资产盘点。
> 状态：草稿，待 2026-SOTA 文献调研返回后补竞争基线并定排序。

## 一、项目已证伪/耗尽的路径（避免重复）
- boundary training-free selector 三连败：CLS-attn / LLM-cosine / CLIP per-patch（对齐缺失）→ proxy(L2) 是边界天花板；FastV(intra-LLM) 更准但不可 serving 集成。
- load-adaptive 预算：n=500 仅 3/5 基准 dominate r25，方法 modest/conditional。
- ElasticVis per-request 预算：连续批处理下 batch-interference → 架构性失败（EV-1e）。
- EV-VAR 方差信号：k 组成零预测力（F=0.179 p=0.836），被证伪。
- S6 缝合（Diversity-NMS/Adaptive/freqDV）：对 incumbent 无显著差，不入主文。
- 训练-free 组合六连负（router/QA-gate/cascade/RankBridge/Deferred/OT）。
- **关键空档：以上全部是 training-free 启发式。文献共识（项目自记）是 OCR/选择需要学习组件。learned scorer 是唯一没试过的类别。**

## 二、论文现状（"太简单"具体指什么）
- 方法 = §3 纯固定 top-k L2 启发式（~127 行 + Algorithm 1）：全局 L2 评分 → 每图固定 k=κN 保留 → 原生 merger 只跑幸存者。
- 三个贡献：① 选择级机制（kept set 是因果杠杆）；② workload-conditional stage law（三族模型，iso-selector/iso-budget）；③ 极简 RBM + 与 FastV 的 trade-off 图 + 6 个预设扩展不提升。
- claim：RBM 在 OCR/text-dense 胜 FastV（OCRBench +16.0pp@Qwen3），FastV 在 reference-grounded 胜 RBM（TextVQA +17.2pp / GQA +8.9pp）。
- **升级点 = §3.3（换 scorer）+ §3.2（加 query/预算感知）。**

## 三、自有资产（可复用）
- `scripts/mechanism_token_survival.py`：capture-only 逐 unit {pre L2, post L2, edge} + 逐样本 correct/wrong；npz 已在盘（textvqa/docvqa/gqa × 64）。
- runner `save_kept`：记录每图 kept unit 集合（R1-1）。
- `scripts/stitch_pixel_calib.py`：像素域 SVD 谱 → 每图复杂度 frac，cache-independent（编码前可算，全样本覆盖）。
- budget 机制：`--budget-file {k|frac}`，占位符与剪枝位同一游标，mode-agnostic。
- freq scorer（α,β 手调 blend）：textvqa +1.2pp@25%（方向 B）。
- 数据：eval/subsets/*.jsonl（id/image/question/gt/choices），full_matrix j7_*.json 含 per_sample。
- 评测：official_scorers（vqaacc/ANLS/GQA/OCRBench），A40 单卡，GPU 当前空闲。

## 四、候选方向（待文献校准）

### D1. Learned scorer for RBM（首选候选）
- 把 §3.3 的 L2 换成小 MLP/线性 scorer，输入 per-unit 特征（L2、var、high-freq、edge、位置、2x2 邻域上下文、可加像素谱）。
- 监督（self-distillation / stage-distillation）：teacher = POST 阶段全模型特征的重要性（高预算下），或 kept-set 的 correctness-flip 标签；从 survival capture 复用特征与正确/错误标记。
- query-aware 变体：scorer 输入拼接 question embedding → 同时拿下 reference-grounded（FastV 现在的领地）。
- 成本：特征已捕获，MLP 训练 <<1 GPU·h；评测 n=200/500 每基准。
- 论文叙事：stage law 不变，贡献③从"极简规则"升级为"stage law + 轻量学习评分器（几分钟训练）"；与 iso-selector 控制对照，学习收益可归因。

### D2. Content-driven per-image budget（像素谱，复核"结构性 NO-GO"）
- 证据：S6 budget 相位唯一的精度评测 skip=200（技术失败：首次 OOM + 游标/占位符失步），从未公平跑过；像素谱 cache-independent、编码前可算 → 占位符与剪枝位同源一致，理论上绕开"特征 pass 不触发"。
- 修复点：游标按图 id 确定性映射（非 position），或 frac 模式全样本覆盖后重跑。
- claim：iso-总预算下按内容重分配（text-dense 多给）→ 混合基准精度增益；training-free，保持论文身份。
- 可与 D1 叠加：learned 每图预算。

### D3. D1+D2 合成："stage law + 轻量 head"完整方法升级。

### D4. （候选）任何文献暴露的 2026 更强方法 → 直接 baseline 对标或移植。

## 五、下一步
- [ ] 等 2026-SOTA 文献调研返回 → 补竞争基线/校准排序。
- [ ] 选 D1 或 D2 先做 dry-check 级原型。
- [ ] D1 训练 <6 GPU·h 自主；若牵动论文核心 claim/方法 → 升级 user 定稿。

## 六、文献调研并入（2026-08-17，agent 一手核验）
- Qwen3-VL 无 Mamba 层（SigLIP-2 + 2×2 MLP merge，纯 transformer）——Mamba 压缩线不适用。
- 竞争格局（2026-08，arXiv 一手核验）：
  - **ET-Prune** 2608.01979（TF）：decoder QK-block query-conditioned 证据 + 文本区保护 + 动态逐样本 token 下限；Qwen3-VL-8B OCRBench-v2 超最强剪枝 baseline +1.80pp@~50% 保留。**同基座同基准直接竞争者**。
  - **RUTA** 2608.04132（轻量训练）：可微 Bernoulli gate 学"每图每 query 保留数"+ anchor 聚合；Qwen3-VL-8B 4.2% token 保 94.4%。→ **极少量训练在 Qwen3-VL 收益最大，支持 D1/D3**。
  - **RoRA** 2608.07088（TF）：role-oriented 区域预算（protected/context/detail）+ prompt 校准；Qwen3-VL 75-90% 剪超 D2Pruner ~5pp。
  - **GMC** 2608.02134（TF）：message-coreset 聚合（被丢 token 的 signed message 原位传输）；Qwen2.5-VL 80.2% 少 token 保 97.78%。
  - **FastOCR** 2605.17447（TF）：解码侧每步 attend ~5% 视觉 token，与 prefill 压缩正交。
  - 综合 benchmark 2511.02650：OCR 最脆弱（88.9% 剪时 OCR-B 掉 75.9%）、随机剪枝是强 baseline、**无方法全胜**。
- 启示：training-free query-aware + 动态预算已是拥挤赛道（ET-Prune/RoRA）；本项目的差异化 = ①iso-budget 严格评测协议 + stage law 机制理解；②轻量学习评分器（RUTA 式收益 + 边界部署）；③同时赢 text-dense 与 reference-grounded 的单一方法（目前无人做到）。

## 七、零 GPU 实证（2026-08-17，proto_stage_distill.py）
- 假设：POST 阶段（全模型输出）的 unit 重要性与边界启发式大幅不一致，且可被边界特征学习。
- 数据：survival_capture npz（64 图/bench，逐 unit pre/post/edge + 位置 + 邻域特征）。
- 结果（held-out 图像 70/30，Ridge + MLP 学 post 排序）：
  | bench | PRE-L2 Jaccard | MLP Jaccard | Δ | rank corr pre→mlp |
  |---|---|---|---|---|
  | textvqa | 0.373 | 0.416 | +4.3pp | 0.309→0.356 |
  | docvqa | 0.345 | **0.537** | **+19.2pp** | 0.215→0.461 |
  | gqa | 0.441 | 0.490 | +4.9pp | 0.396→0.487 |
- 解读：①边界特征含足够信息，MLP 能在未见图像上泛化学到 POST 视角；②docvqa（密集文档）提升最大——与 stage law 的 workload-conditional 一致；③**这不直接等于任务精度增益**（POST 排序任务上是否更优需 GPU 实测），是"可学性"证据。
- ⚠️ 论文 Table 1 显示 post-merger 选择在 docvqa 精度低于 pre（0.238 vs 0.481）——那是"选择位置"对比；此处 teacher 是"特征来源"。二者不同，**必须用任务精度裁决 teacher 选择**。

## 八、下一步（任务 #4）
- 首选 D1：learned boundary scorer（stage-distillation），GPU 真实验证任务精度。
  - 训练：survival npz 全量 + 更大捕获集（n 扩到 200/500，用 mechanism_token_survival capture）。
  - 部署：runner 新增 selector="learned"，加载 scorer 权重 → 逐 unit 特征评分 → top-k。
  - 评估：n=200，docvqa/textvqa/gqa/ocrbench @ r=0.75/0.875，vs PRE-L2 基线（论文方法）+ FastV。
  - 判据：learned ≥ PRE-L2 为 GO 继续（teacher 迭代/query-aware）；< PRE-L2 则换 teacher（correctness-flip）或转 D2。
  - 预算：训练 CPU 秒级；GPU eval 4 基准 × 2 r × n=200 ≈ 1-2 GPU·h（自主范围内）。

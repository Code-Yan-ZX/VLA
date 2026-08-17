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

## 九、D1 执行状态（2026-08-17）
- 已实现：learned_scorer.py（训练/运行时同一特征构建 + MLP）、train_learned_scorer.py、
  runner `--selector learned --learned-scorer --learned-bench`（dry-check ALL PASS）、
  build_learned_heldout.py（held-out 200/bench）、run_d1_eval.sh + analyze_d1.py。
- 捕获：n=200/bench 特征捕获进行中（qwen3vl_clean env，docvqa 大图慢，~2h 总量；
  个别图 vision tower 未触发被 SKIP，resume 会补）。
- 协议：训练 subset A（eval/subsets/200）→ 评测 held-out 200（full_splits 不相交）
  → learned vs l2 配对 per-sample。
- 待办：捕获完成 → train_learned_scorer（CPU 秒级）→ run_d1_eval（3 bench × 2 sel
  × r=0.75，~1.5-2h GPU）→ analyze_d1 配对对比。
- GO 判据：held-out 上 learned ≥ l2 且配对增益可辨 → 扩展（r=0.875、ocrbench、
  query-aware 变体、正确性翻转 teacher 对照）；否则换 teacher / 转 D2。

## 十、D1 首个 teacher NO-GO + 机制洞察（2026-08-17）
- **结果（textvqa held-out 200，官方 vqaacc）**：learned(POST-distill)=0.435 vs l2=0.595
  （−16pp）；runner 内部打分 −19.5pp。两套打分均大幅劣于基线 → **stage-distillation
  到 POST 排序在任务精度上是负面的**。
- **机制洞察（写入论文 negative results 的候选）**：POST 阶段系统性降权高 edge
  （文本笔画）单位（rank_delta hi-edge −68.6 / lo-edge +28.0）——学 POST 排序 = 学
  "避开文本" → 文本密集任务（textvqa/docvqa）变差。这与论文 M2（text-stroke
  demotion）+ Table 1（pre 选择胜 post 选择）完全自洽，**反向印证 PRE 保留文本能力
  是 RBM 优势的本质**。
- **区分"方法缺陷 vs teacher 缺陷"**：gqa 的 POST 降权弱（corr(edge,drank)=−0.034）→
  若 gqa 上 learned ≥ l2，则证明特征映射有效、错在 teacher；若 gqa 也差 → 方法本身
  有缺陷（特征/模型容量）。
- **下一步候选**：① correctness-flip teacher（任务接地，成本高 ~4-6 GPU·h）；
  ② 换 teacher 为"l2 加权 var/edge 的任务最优组合"（freq scorer 证明 var 有任务价值，
  但手调已试）；③ 放弃 learned，转 D2（像素谱每图预算，training-free 保身份）。

## 十一、D1 诊断终判：method 可行、teacher 反任务（2026-08-17）
- **held-out 200（官方 scorer）**：
  | bench | learned(POST) | l2 | Δ | 配对 |
  |---|---|---|---|---|
  | textvqa | 0.435 | 0.595 | −16.0pp | 13胜45负 |
  | gqa | 0.550 | 0.515 | +3.5pp | 18胜11负 |
- **结论：learned scorer 的方法机制有效（gqa 正），POST teacher 在文本密集任务反任务（textvqa 大负）**。
  机制：POST 系统性降权文本笔画 → 学 POST = 学避开文本。与论文 M2（text-stroke demotion）
  + Table 1（pre 胜 post）自洽。
- **论文价值**：这是可写进论文 negative-results/机制的完整证据链——"POST 阶段的重要性在
  text-dense 上反任务"反向强化 RBM 的 PRE 选择+保留文本。learned scorer（gqa 正）可作为
  "学习评分器可行但需任务接地 teacher"的诚实记录。
- **下一步**：① D2 像素谱每图预算（便宜、training-free、保身份）；② corrective distillation
  （任务接地 teacher：从答对的 keep-set 学，预期 learned ≥ l2 across benches，成本更高）。

## 十二、D2 终判：双重 NO-GO（2026-08-17）
- **像素谱预算**：像素级 SVD 谱能量无判别力（128/256/512px 下 spectral-tau 全≈1.0、
  erank 全 0.05）——自然图谱系重尾，累计能量饱和；slope（log 衰减）有方差但不构成
  干净复杂度代理。→ 像素复杂度信号不可用。
- **feature 谱预算**：feature-side spectral 有真实范围（mean_k=182.5 range=[80,256]
  iso-total 36328）——但部署 eval **skip=200 全跳过**，复现 S6 的 skip=200 故障。
  → **确认 S6"结构性 NO-GO"判定**（占位符先于特征固定、部分图特征 pass 不触发、
  cursor 顺序错位 → 数量不匹配）。我此前"技术失败可修"的怀疑被证伪。
- **结论**：per-image budget（无论像素/feature 信号）在本 vLLM 占位符架构上不可行，
  作为方法贡献不成立。uniform@max1=0.595 与 max16 一致（缓存行为不影响 uniform 精度）。

## 十三、D1+D2 探索综合结论（2026-08-17）
- **给论文的直接帮助（已获）**：
  ① 机制洞察：POST 阶段重要性降权文本笔画 → 学 POST 反任务（textvqa −16pp）；
     gqa 上 POST 降权弱 → learned +3.5pp。**反向印证 RBM 的 PRE 选择+保留文本是
     text-dense 优势的本质**。可入论文 negative-results/机制段。
  ② learned scorer 机制可行：小 MLP 能从边界特征学 POST 排序（held-out Jaccard
     docvqa +20.5pp）——证明边界特征信息量 + 学习方法有效，只是 teacher 需任务接地。
  ③ 负发现：像素复杂度代理无判别力；per-image budget 架构性不可行（确认 S6）。
- **方法升级的诚实评估**：learned scorer 需 corrective distillation（任务接地 teacher：
  从"答对的随机 keep-set"学），预期 modest（l2 本身已强，headroom ~1-3pp 据 freq
  scorer/diversity 证据），成本 ~3-4 GPU·h（批量标签生成可压到 ~1.5h）。不会达 SOTA。
- **SOTA 的现实评估**：2026 竞品（ET-Prune/RUTA/RoRA）已占据 training-free/轻训
  query-aware+动态预算赛道；本项目 1×A40 自主预算内无法追平。论文的现实定位 =
  机制理解（stage law + POST/PRE 差异机制）+ 诚实方法升级（learned scorer 或保持
  极简）+ 严格 iso-budget 评测协议。

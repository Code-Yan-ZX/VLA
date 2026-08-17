# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：Rank-Before-Merge -> ACM MM'27
> 最近更新：2026-08-17 · **S6 已以探索性扩展写入权威投稿稿；核心方法仍为 plain RBM。**

## ★ 当前论文状态
- 权威入口：`drafts/overleaf_submission/main.tex`；本地 `latexmk` 构建通过。
- `main.pdf` 共 22 页：正文仍为 8 页，参考文献 2 页，补充材料从第 11 页开始。
- 核心结论不变：RBM 是 OCR/text-dense 鲁棒默认；FastV 是 query-conditioned 强 baseline。
- 不宣 RBM beats/SOTA，不修改标题、摘要、贡献列表或 headline method。

## ★ S6 Diversity-RBM（2026-08-17）
- 方法：RBM + importance-ordered cosine NMS；Adaptive 版按图像有效秩调 NMS 阈值。
- 范围：Qwen3-VL-8B，固定 `n=500` slices，仅 75%/50% retention；不是深压缩主设定。
- 结果：两个固定变体均有小幅胜负交换，但未建立对逐格较强 RBM/FastV incumbent 的改进。
- TextVQA@75%：DV 0.748、RBM 0.745、FastV 0.665；只说明相对 FastV 的条件收益，非对 RBM 的可靠提升。
- 对逐格 incumbent 的配对 bootstrap 区间均含 0；不据此宣称等价或 SOTA parity。
- 入稿位置：正文 §6 末尾一段；补充 S4c.3 为 8 格完整表、缺失单元格和范围限制。
- OCRBench FastV@75% 因运行时失败缺失；原始 S6 JSON/Bootstrap 工件未在当前 clone 中，投稿前应恢复到匿名 artifact。
- hook 覆盖率的“混合压缩率”解释未经 cache-disabled 审计，不写入论文、不作为当前结论。
- **user 裁定**：S6 对论文帮助不大、不进主文；上述入稿仅作探索性扩展。评估要点见
  notes/s6_value_assessment.md：`--r`=剪枝率（S6 的 75/50% retention ≠ 论文 25/12.5%），
  无单一方法稳定胜出（DV 4胜1平3负 / ADA 3胜5负），对 incumbent RBM 无显著差（CI 含 0
  非"持平"，需非劣界+非劣检验）；hook 覆盖率解释未经 cache-disabled 审计，不推翻主表。
- 若补强：固定 Diversity-RBM 跑 r=0.75/0.875、用与调参前 200 条不重叠的 slice、非劣性检验。

## ★ 创新 D1：learned boundary scorer（2026-08-17，进行中）
- user 目标：方法不再"太简单"，达 SOTA 或强补论文。调研结论（notes/innovation_directions.md）：
  全部 6+ 扩展均为 training-free 启发式并已证伪；文献共识 OCR 选择需学习组件；2026
  新方法（ET-Prune/RUTA/RoRA/GMC）证明 query-aware+轻量训练+动态预算是当前赢家。
- **D1 = stage-distillation learned scorer**：小 MLP 把边界特征 [l2, edge, var, pos,
  neighbors] 映射到 POST 阶段（全模型）重要性排序；全部特征在 pre-merger hook 实时可算。
- 零 GPU 离线验证（proto_stage_distill.py，held-out 图）：POST 与 PRE top-25% kept-set
  重叠仅 0.295-0.441（docvqa 最低）；MLP 学到 POST 排序 Jaccard docvqa 0.345→0.537、
  textvqa 0.373→0.416、gqa 0.441→0.490（均为 held-out 泛化）。
- 代码已落地 + commit：runner `--selector learned`（dry-check ALL PASS）、learned_scorer.py、
  train_learned_scorer.py、build_learned_heldout.py、run_d1_eval.sh、analyze_d1.py。
- **进行中**：n=200/bench 特征捕获（qwen3vl_clean，docvqa 大图慢 ~2h，个别图 SKIP 由
  resume 补）→ 训练 → GPU 评测（learned vs l2 @ r=0.75 held-out 200，~1.5-2h）→ 配对分析。
- GO 判据：held-out learned ≥ l2 → 扩展（r=0.875、ocrbench、query-aware、correctness-flip
  teacher 对照）；否则换 teacher / 转 D2（像素谱每图预算，复核"结构性 NO-GO"=技术失败）。

## ★ 验证与下一步
- 新增内容编译无 undefined citation/reference；视觉检查正文第 8 页、补充第 14/16 页无裁切或重叠。
- 仅有既存标题 metadata overfull 9.77pt；非 S6 引入。
- 工作树包含本任务前已有的投稿稿修改；提交时只纳入本次 S6 hunks、状态与决策记录。
- 下一步人工项：核验 ACM MM'27 官方日期/地点/模板元数据；实际上传/投稿须 user 确认。

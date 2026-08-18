# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩；目标：Rank-Before-Merge -> ACM MM'27
> 最近更新：2026-08-18；方法创新冻结，进入投稿硬化与评审修改。

## 当前论文
- 权威入口：`drafts/overleaf_submission/main.tex`；正文 8 页，参考文献 2 页，补充材料从第 11 页开始，总计 22 页。
- 核心定位：plain RBM 是 training-free OCR/text-dense 鲁棒默认；FastV 是 query-conditioned 强 baseline；不宣通用胜出或 SOTA。
- 三模型结果按 accuracy-level directional replication 表述；byte-exact 因果归因仅限 Qwen3-VL。
- 2026-08-18 已完成：修复 InternVL `\textbf` 拼写；将 “generalization/isomorphism” 收紧为 third-family replication。

## 创新探索终判
- 六项 prespecified 扩展均未稳定优于 stronger constituent；不再继续 router/cascade/RankBridge/RBM-OT/freq/adaptive 等开放式搜索。
- D1 learned boundary scorer 为事后 held-out 机制探针，不计入六项 prespecified 扩展。
- D1 @25%：TextVQA 0.435 vs L2 0.595（-16.0pp）；GQA 0.550 vs L2 0.515（+3.5pp）。
- 入稿口径：scorer 可学习 POST 排序，但 POST teacher 在 text-dense 上反任务；该失败独立支持 M2 的 text-stroke demotion。
- S6 Diversity-RBM 已从正文/补充删除：无稳定提升、原始工件缺失且不值得重跑。

## 验证
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` 通过；无 undefined citation/reference。
- 页数保持 22；正文第 8 页、补充第 14--15 页视觉终验无裁切/重叠；仅存既有标题 metadata overfull 9.77pt。

## 下一步
- user 将用评审 agent 审稿；按 findings 做定向修改，不启动新方法实验。
- 投稿前恢复匿名 artifact：核心 JSON、逐样本结果、bootstrap、配置、命令、模型版本、checksum，含 D1 audit record。
- 审计 Qwen2.5 OCRBench、FastV 跨引擎、split/pixel-cap/skip/token-budget；决定是否补第二模型轻量 pure-stage control，或仅进一步收紧 claim。
- 核验 ACM MM'27 官方模板/日期/地点并做匿名性、引用、PDF 视觉终验；实际上传仍须 user 明确确认。

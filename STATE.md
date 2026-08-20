# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）
> 项目：VLM 视觉 token 压缩；论文：Rank Before You Merge；更新：2026-08-20。

## 当前阶段
- 投稿前内容更新、压缩与版面重构已在 `paper/acmmm-final-revision` 完成，等待作者审阅；未合并 main。
- 权威源：`drafts/overleaf_submission/main.tex`、`supp.tex`、`references.bib`。
- 最终审阅 PDF：`output/pdf/rank_before_you_merge_acmmm_final_revision_20260820.pdf`（21 页）。
- 页序：正文 1--8，参考文献 9--10，补充材料 11--21；独立补充 PDF 为 11 页。

## 冻结结论
- Operational RBM 使用 Qwen3-VL layer-8 pre-merger score；不得称为 pure stage-only control。
- Full-split matched boundary（pre-final vs post）：TextVQA +27.68 pp、DocVQA +4.59 pp、OCRBench +235、GQA -5.64 pp；四项 Holm 显著。
- Qwen2.5 OCRBench matched 4M cap：25% 480/182（+298），12.5% 328/70（+258），两臂有效 prompt length 相等。
- Full OCRBench HF eager：Qwen3 RBM/FastV 559/413（79 symmetric skips）；Qwen2.5 382/295（88 symmetric skips）。
- 结论限定为 workload-conditioned stage law；RBM 是 OCR/text-heavy 鲁棒默认，不是通用冠军。

## 版面与验证
- p5 顺序为 Table 1、matched-boundary Table 2、Figure 2；Figure 3 为 token-survival maps，Figure 4 为定量机制图。
- Related Work、negative extensions、表注和重复协议已压缩；LLaVA 完整解释移至 Supplementary S7c。
- `latexmk` 主稿/独立补充通过；无 undefined refs/cites、无 `[?]`、无 overfull、无 Type 3 字体。
- 21 页已逐页渲染目检：无裁切/重叠，双栏图可读；`git diff --check` 通过。

## 下一步
- 作者审阅新 PDF；venue 确定后再处理模板、匿名开关、页限与 supplement 上传政策。
- 匿名 artifact 仍未恢复（当前 0/53 run JSON）；恢复、校验 manifest/checksum 后才能解除 artifact gate。
- 实际投稿、合并 main 均需作者明确确认；不得重新跑实验或改动冻结结果。

# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）
> 项目：VLM 视觉 token 压缩；目标：Rank-Before-Merge -> Q1/Q2 venue（待选）
> 最近更新：2026-08-19；方法创新冻结，当前为导师审阅与投稿硬化阶段。

## 当前论文
- 权威入口：`drafts/overleaf_submission/main.tex`；导师审阅 PDF：`output/pdf/rank_before_you_merge_supervisor_review_20260819.pdf`。
- 正文 8 页，参考文献 2 页，补充从第 11 页开始，总计 22 页。
- 定位：RBM 是最小 stage operationalization 与 OCR-oriented 鲁棒默认；FastV 是 query-conditioned 强 baseline；不宣通用胜出/scorer novelty。
- “Stage Law” 仅指 tested text-dense、iso-model、iso-budget 且共享 query-blind L2-magnitude family 的 pre/post cells；byte-exact 仅限 Qwen3-VL。
- 六项 prespecified 扩展均未稳定优于 stronger constituent；learned scorer 仅作 exploratory mechanism probe。

## 2026-08-19 投稿硬化
- 明示精确打分：pre=`mean_j ||h_uj||2`，post=`||M(h_u1:4)||2`；不再称 identical scorer。
- 前置披露 Qwen3 主 pre tap=deepstack[0]/ViT layer 8；matched-depth pre-final control 在主 merger boundary 保留 text-dense 效应。
- 修正 S1 随机 Jaccard 基线 `1/7≈0.143`、S3 n=200/full-split 混用、GQA z 符号、Post-L2/FastV 误标及 partial-credit SE 表述。
- 补 AnchorPrune（Qwen2.5 明确 post-merge）与 PixelPrune（pre-ViT）；未发现 matched native-merger pre/post control。
- 移除 S10 对额外 6 panel/匿名 archive 的未兑现承诺；README 改为 clean allowlist package。

## 验证与 gate
- `latexmk` 通过；0 undefined citation/reference、0 overfull、无 Type 3 字体；页 1/3/8/9/10/12/13/22 无裁切/重叠。
- 当前采用 `sigconf,nonacm` 通用双栏，无会议/期刊/DOI/版权占位；作者显示为 Zhengxing Yan，单位与邮箱待确认。
- S9 run JSON 仍为 0/53；不得声称匿名 artifact 已完成。恢复受服务器凭据/连接阻塞。
- 2026-08-19 官网仅确认 MM'27 在香港、日期待定；正式 CFP/页限/补充政策尚无法核验。

## 下一步
- 恢复并核验 53/53 run JSON、14 类 manifest、逐样本/统计/版本/checksum，再解除 artifact gate。
- 选定 venue 后切换其官方模板并核验页限、匿名规则及 integrated/separate supplement 政策；实际投稿仍须 user 明确确认。

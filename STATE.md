# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）
> 项目：VLM 视觉 token 压缩；目标：Rank-Before-Merge -> IEEE TCSVT。
> 最近更新：2026-08-27；方法创新冻结，当前为投稿硬化阶段。

## 当前论文
- 内容权威入口：`drafts/overleaf_submission/main.tex`；TCSVT 投稿包：`drafts/ieee_tcsvt_submission/`。
- IEEEtran 双栏正文含参考文献 10 页（TCSVT Transactions Paper 上限 14 页）；补充材料独立 11 页。
- 定位：RBM 是最小 stage operationalization 与 OCR-oriented 鲁棒默认；FastV 是 query-conditioned 强 baseline；不宣通用胜出/scorer novelty。
- “Stage Law” 仅指 tested text-dense、iso-model、iso-budget、共享 query-blind L2-magnitude family 的 pre/post cells；byte-exact 仅限 Qwen3-VL。
- 五项干净的 prespecified 扩展未稳定优于 stronger constituent；learned scorer 仅作 exploratory mechanism probe；受污染 cascade 已从正式材料移除。

## 2026-08-25 native-mRoPE/scorer 定点纠错
- 权威审计：`origin/exp/deferred-rbm-n200` commit `2ec9d15`；MALT-C=既有 native RBM，MALT-1/deferred 假说被证伪，均不得入稿。
- Table 2 唯一污染格：Qwen3 DocVQA RBM `0.4239→0.5924`（native + official ANLS）；FastV `0.5863` 不变，结论为 RBM +0.6pp 但 paired inconclusive。
- FastV 仅明确领先 Qwen3 TextVQA/GQA；Qwen3 DocVQA statistically indistinguishable；RBM 继续明确领先 OCRBench。
- OCRBench `0.575`/Qwen2.5 同口径均为 nominal n=200、共同 skip 计 0，不再称 attempted-only。
- 修订报告：`experiments/paper_native_mrope_correction_report.md`；其余正式主表数字未改。

## 验证与 gates
- TCSVT `latexmk main.tex` 与 `latexmk supplement.tex` 通过；0 undefined citation/reference、0 overfull、0 Type 3；全页渲染无裁切/重叠。
- S9 run JSON 仍未恢复；不得声称匿名 artifact 已完成。exact H0n-vs-FastV paired CI 待两份 gitignored raw run 恢复后复算。
- **CLAIM-LEVEL GQA 未解决**：full-split pre-final vs post 为 -5.64pp（CI[-6.42,-4.86]），现稿仍含 n=200 的 0.0pp 表述，投稿前必须修订。

## 下一步
- user 决策 GQA claim 修订及是否采用 Qwen2.5 OCRBench matched-config 新数字（480/182）；恢复 53/53 JSON 与 manifest。
- 补齐作者 affiliation/country/email/ORCID，提交前复核 ScholarOne 匿名/关键词/source-file 规则；实际投稿仍须 user 明确确认。

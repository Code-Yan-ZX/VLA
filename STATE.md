# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）
> 项目：VLM 视觉 token 压缩；目标：Rank-Before-Merge -> IEEE TCSVT。
> 最近更新：2026-09-09；方法创新冻结，当前为投稿硬化阶段。

## 当前论文
- 当前 TCSVT 权威稿：`drafts/ieee_tcsvt_submission_20260904/main.tex`；补充材料：同目录 `supplement.tex`。
- 中文方法稿已按 anti-defensive-writing 重写并校准到权威稿：`drafts/RBM_merger_aware_中文方法包装草稿_反防御性改写版.docx`。
- IEEEtran 双栏正文 11 页；补充材料 9 页；主文 Figure 1 使用 Word 选定图，Figure 2 已简化为纯视觉对比。
- 方法叙事已升级为 representation-dependent selection--merger ordering：含 kept-set 等价条件、阶段分歧、四项 RBM contract、算法复杂度与部署边界；未新增模块或实验 claim。
- 定位：RBM 是最小 stage operationalization 与 OCR-oriented 鲁棒默认；FastV 是 query-conditioned 强 baseline；不宣通用胜出/scorer novelty。
- 标题已改为 “Rank Before You Merge: Stage-Consistent Visual Token Selection for Merger-Equipped Vision-Language Models”；正文避免把结果包装成 universal law。
- 五项干净的 prespecified 扩展未稳定优于 stronger constituent；learned scorer 仅作 exploratory mechanism probe；受污染 cascade 已从正式材料移除。

## 2026-08-25 native-mRoPE/scorer 定点纠错
- 权威审计：`origin/exp/deferred-rbm-n200` commit `2ec9d15`；MALT-C=既有 native RBM，MALT-1/deferred 假说被证伪，均不得入稿。
- Table 2 唯一污染格：Qwen3 DocVQA RBM `0.4239→0.5924`（native + official ANLS）；FastV `0.5863` 不变，结论为 RBM +0.6pp 但 paired inconclusive。
- FastV 仅明确领先 Qwen3 TextVQA/GQA；Qwen3 DocVQA statistically indistinguishable；RBM 继续明确领先 OCRBench。
- OCRBench `0.575`/Qwen2.5 同口径均为 nominal n=200、共同 skip 计 0，不再称 attempted-only。
- 修订报告：`experiments/paper_native_mrope_correction_report.md`；其余正式主表数字未改。

## 验证与 gates
- TCSVT `latexmk main.tex` 与 `latexmk supplement.tex` 通过；0 undefined citation/reference、0 overfull、0 Type 3；全页渲染无裁切/重叠。
- 服务器已恢复 S9 53/53（当前稿权威索引 29 + 历史 gate 24）和 S10 6/6；本地匿名 staging 已清理路径并生成 147-entry manifest。
- P0-2/P1 六个 headline 数字均由 raw → gz → analysis 三级核验，无需 GPU 重跑。
- LaTeX 编译通过：正文/补充材料分别 11/9 页；无 undefined citation/reference、无 overfull、无 Type 3；关键页逐页 QA 通过。

## 下一步
- 最终投稿包：`drafts/ieee_tcsvt_submission_20260904_final.zip`；匿名 artifact：`experiments/artifact_anonymous_tcsvt_20260904.tar.gz`。
- 最终复核已完成：旧 headline 数字无残留；匿名路径扫描 clean；artifact 147 项 checksum 全通过；S9 current/legacy=29/24，S10 PNG=6。
- 补齐作者 affiliation/country/email/ORCID，提交前复核 ScholarOne 匿名/关键词/source-file 规则；实际投稿仍须 user 明确确认。

# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）
> 项目：VLM 视觉 token 压缩；目标：Rank-Before-Merge -> Q1/Q2 venue（待选）
> 最近更新：2026-08-25；方法创新冻结，当前为投稿硬化阶段。

## 当前论文
- 权威入口：`drafts/overleaf_submission/main.tex`；当前编译：`drafts/overleaf_submission/main.pdf`。
- 通用 `sigconf,nonacm` 双栏：正文 8 页、参考文献 2 页、补充从第 11 页开始，总计 21 页；venue/页限尚待选定后核验。
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
- `latexmk main.tex` 与 `latexmk supp.tex` 通过；0 undefined citation/reference、0 overfull、0 Type 3；PDF 旧数字/旧 claim/cascade/MALT 残留为 0。
- S9 run JSON 仍未恢复；不得声称匿名 artifact 已完成。exact H0n-vs-FastV paired CI 待两份 gitignored raw run 恢复后复算。
- **CLAIM-LEVEL GQA 未解决**：full-split pre-final vs post 为 -5.64pp（CI[-6.42,-4.86]），现稿仍含 n=200 的 0.0pp 表述，投稿前必须修订。

## 下一步
- user 决策 GQA claim 修订及是否采用 Qwen2.5 OCRBench matched-config 新数字（480/182）；恢复 53/53 JSON 与 manifest。
- 选定 venue 后迁移模板并核验页限/匿名/supplement；实际投稿仍须 user 明确确认。

## 2026-08-26 Merger-Representation Goal（分支 exp/merger-representation-goal）
- **VERDICT: NO-GO（明确，locked n=200 确认）**。Representation-level 候选
  （native merged base + 独立组内 residual-detail token）全部失败；无方法性
  强于 plain RBM 的 training-free representation 扩展。按任务止损：停止本
  轮与后续 method-variant search，RBM 保持为 finding-driven minimal method，
  返回投稿硬化。
- 机制根因：任务定义 r_a=M(ΔX)-M(0) 被证伪（native merger 内部 LayerNorm 把
  de-mean patch 放大为 2-7× 且与细节无关）；机制推导 r_b=M(X)-M(X̄) 量级正常
  但 cos(r_b,base)≈0.92——native merger（norm+MLP）本身已把组内结构编码进
  base token，residual 冗余；RBM@25% 已保留高细节单元。结果：随 ratio 单调
  退化（Gate B dev64 最佳 +0.42pp 噪声；Gate C locked200 macro **-3.05pp**，
  DocVQA -4.2pp p=0.028 / OCRBench -45.3pts p=0.012 显著为负）。
- 报告 `experiments/merger_representation_method_discovery.md`；日志
  `experiments/merger_representation_goal_log.md`；数据 runs/merger_repr/。
- 顺带验证：native RBM DocVQA locked200 = 0.5924 == 论文冻结 cell 精确一致
  （harness 正确）。位置方案 duplicate ≡ adjacent（dev 全数据集 ≤0.0007）。

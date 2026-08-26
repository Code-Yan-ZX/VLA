# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）
> 项目：VLM 视觉 token 压缩；目标：Rank-Before-Merge -> Q1/Q2 venue（待选）
> 最近更新：2026-08-26；方法创新冻结，论文内容同步完成，artifact 尚阻塞投稿。

## 当前论文
- 权威入口：`drafts/overleaf_submission/main.tex`；PDF：`drafts/overleaf_submission/main.pdf`。
- 分支 `paper/submission-hardening-final`，基线 `cad68f902fa4606d4c41f9d7b64e0dcdd86ccb82`。
- 通用 `sigconf,nonacm` 双栏共 22 页；正文约 8 页（讨论/结论与参考文献共用 p9），补充从 p11 开始。
- 定位：finding-driven、mechanism-centered；RBM 是 OCR-oriented minimal operationalization/robust default，不宣 universal SOTA 或 scorer novelty。
- Stage effect 限定为 tested、iso-model/budget、共享 query-blind L2-magnitude family；byte-exact 机制仅限 Qwen3-VL。

## 已同步事实
- Qwen3 DocVQA：native RBM `0.5924` vs FastV `0.5863`，仅 parity/paired inconclusive；旧 `0.4239` 与 cascade 不在正式稿。
- matched-depth full split：TextVQA `+27.68pp`、DocVQA `+4.59pp`、OCRBench `+235`，GQA `-5.64pp` CI `[-6.42,-4.86]`；旧 n=200 GQA 0.0 仅作 superseded sampling estimate。
- Qwen2.5 OCRBench matched 4M/full1000/iso-token/0-skip 官方结果 `480/182` 已同步至表、图与统计镜像。
- representation/deferred/adaptive NO-GO 不升级方法，不进入 supplement。
- 近邻文献只补直接边界；完整记录见 `experiments/final_paper_sync_report.md`。

## 验证与 blockers
- `latexmk` main/supp PASS；0 undefined citation/reference、0 overfull、0 Type3；风险词和关键数字 PDF 检查 PASS。
- anonymous artifact 未完成：历史合同 0/53，当前 S9 0/29；合同漂移且无匿名 metadata manifest/checksum 包。
- exact DocVQA H0n-vs-FastV paired CI 的两份 raw JSON 均缺失；继续保持 parity/inconclusive。

## 下一步
- 从服务器恢复 raw export，选定并统一 artifact 合同，补 none anchors，重建 manifest/anonymity scan/checksums。
- 作者选定 venue/模板、页限与匿名策略后再做最终格式迁移；任何外部投稿须明确确认。

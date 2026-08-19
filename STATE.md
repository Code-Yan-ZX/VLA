# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩；目标：Rank-Before-Merge -> ACM MM'27
> 最近更新：2026-08-18；方法创新冻结，当前为投稿硬化阶段。

## 当前论文
- 权威入口：`drafts/overleaf_submission/main.tex`。
- 正文 8 页，参考文献 2 页，补充从第 11 页开始，总计 22 页。
- 核心定位：RBM 是最小 stage operationalization 与 OCR/text-dense 鲁棒默认；FastV 是 query-conditioned 强 baseline；不宣通用胜出或 scorer novelty。
- “Stage Law” 仅指 tested same-scorer、iso-model、iso-budget pre/post cells；byte-exact 因果归因仅限 Qwen3-VL。
- 六项 prespecified 扩展均未稳定优于 stronger constituent；D1 learned scorer 仅作后续 held-out 机制探针。

## 2026-08-18 评审修改
- 摘要已桥接 same-scorer +38.4pp 与 FastV 非通用赢家；移除前置机制统计和 GQA 独立 caveat 句。
- 引言改为问题 -> 定义 pre/post 与 RBM -> 结果 -> 机制 -> contributions。
- 主假设族固定为 9 个 25% retention text-dense stage contrasts；paired sign-flip p 值做 Holm 校正，其他分析标为 secondary/exploratory。
- Related Work 排他性措辞收紧；learned scorer/负结果压缩；移除未来 artifact release 声明。
- README/Supp 清除陈旧 `\acmConference`、`paper_v4`、S1--S9 和 McNemar-primary 表述。

## 验证与阻塞
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` 通过；无 undefined citation/reference。
- 页 1/8/9/10/11 已渲染终验，无裁切或重叠；第 9 页短 running title 不再碰撞页眉。
- 仅存标题 metadata 的 9.77pt overfull 与非致命 underfull page-balance 警告。
- S9 要求 53 个唯一 JSON，本地为 0/53；S10 缺 6 个 panel；不得声称匿名 artifact 已完成。
- 已加 `scripts/build_anonymous_artifact.ps1` 与 `experiments/artifact_recovery_plan.md`；服务器恢复受凭据/连接阻塞。

## 下一步
- 恢复并核验 53/53 run JSON、14 类 audit manifest、6 个 S10 panel、逐样本/统计/版本/checksum，再解除 artifact gate。
- 核验 ACM MM'27 官方规则与 public prior art；实际投稿仍须 user 明确确认。

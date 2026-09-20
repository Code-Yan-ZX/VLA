# STATE.md — 当前项目状态（主窗口维护，≤30 行）
> 项目：VLM 视觉 token 压缩；最近更新：2026-09-20。

## 当前方向
- 用户明确要求放下 RBM 作为研究主线，重新对齐视觉 token 压缩领域；当前阶段为 P1 文献版图与选题判断。
- 顶会综述：notes/token_compression_landscape_20260920.md；官方来源核验表：notes/lit_venue_map_20260920.md、notes/lit_methods_20260920.md、notes/lit_frontier_20260920.md。
- 2026 年 QuietPrune、DocPrune、MMTok、MetaCompress 等已占据早期 query 引导、文档压缩、集合覆盖和多轮复用等宽泛命题；不能沿用旧的“无人做”说法。
- 候选研究问题：同模型同实际 token 预算下，细粒度证据保真与完整服务链 p95 TTFT、请求吞吐、峰值显存的联合权衡；须先做差距诊断，再确定新方法。

## 既有结果
- RBM 相关论文、实验和图表保留为历史材料及可能的对照，不作为当前默认投稿方向。
- DCC 2027 证据审计稿：drafts/dcc2027_submission_20260916/main.tex，尚未实际投稿；后续是否使用须基于新方向另行判断。
- 既有 TCSVT 稿：drafts/ieee_tcsvt_submission_20260904/main.tex；部分模型适配和评测经验可复用。

## 下一步
- 从同日综述选 3–4 个强基线和 2 个现代模型，设计同后端、同实际 LLM 视觉 token 预算的诊断协议。
- 先核验 TextVQA/OCRBench/DocVQA/ChartQA 与普通 VQA、多轮下的绝对分数、稀有证据失败切片、端到端延迟及连续批处理吞吐。
- 若诊断差距稳定，再在 notes/positioning.md 写新颖性句、方法假设和成功/停止判据；不要预设 RBM 或任何新打分器一定有效。

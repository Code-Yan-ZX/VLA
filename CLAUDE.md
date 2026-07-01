# CLAUDE.md — VLA 项目（VLM 视觉 Token 压缩）

科研项目：VLM 视觉 token 压缩新方法 → 标准多模态基准验证 → 目标 Q1/Q2 SCI。完整手册与 P0–P5 阶段计划在 **ORCHESTRATION.md**（按需读章节，勿整篇灌入上下文）。

**新会话启动（每次）：** ① 先读 **STATE.md**（当前阶段/下一步，≤30 行）恢复进度 → ② 按 STATE 的"下一步"推进，细节查 ORCHESTRATION.md。

**主窗口铁律（LEAN MAIN，最重要）：** 本会话是指挥不是苦力。
- 绝不整篇读论文/日志/代码 → 派子 agent（`Agent(Explore/general-purpose)`、`deep-research`、`nature-*` skills、后台 bash）。
- 子 agent 只回 **≤20 行 digest + 磁盘路径 + 下一步建议**。
- 状态写 `STATE.md`；自主决策写 `DECISIONS.md`（含理由）。

**执行模式：** 尽量全自动端到端；仅在 **凭据 / 单次>6GPU·h 训练 / claim 被推翻 / 投稿前** 升级找人（ORCHESTRATION.md §6）。
**算力：** 1× A40 46GB（单卡 → 实验串行，§5）。
**Git：** 有进展即 commit + push origin main（**以你本人名义提交，禁止任何 AI/Claude 署名、禁止 `Co-Authored-By` 尾注**），不提交权重/数据/日志。

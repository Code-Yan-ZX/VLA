# ORCHESTRATION.md — VLM 视觉 Token 压缩（多 Claude Code 协作手册）

> 本文件是整个项目的**操作手册**。主窗口与所有子 agent 都按它执行。主窗口尽量不读本文件之外的大块内容；只在需要时按章节检索。

## 0. 项目宪章（Charter）

| 项 | 值 |
|---|---|
| 主题 | VLM 推理效率 —— **视觉 token 压缩**新方法 |
| 验证 | 标准多模态基准（MME / MMBench / MMMU / GQA / TextVQA / POPE / ScienceQA / ChartQA / DocVQA 等） |
| 效率指标 | token 数、FLOPs、prefill/decode 延迟、GPU 显存、**真实吞吐**（vllm / lmdeploy 级部署加速） |
| 目标 | Q1/Q2 SCI（Pattern Recognition / Information Sciences / KBS / Neurocomputing / IEEE TMM 一档） |
| 周期 | 3–6 个月 |
| 算力 | **1× NVIDIA A40 46GB**（单卡 → 实验**串行**，见 §5） |
| 基座 VLM | **P1 末由编排自主选定**（LLaVA-1.5 / NeXT 或 Qwen2.5-VL / Qwen3-VL），写入 `DECISIONS.md` |
| 自治度 | 尽量**全自动端到端**；升级策略见 §6 |
| 方法现状 | 从零找切入点 → P1 文献定位是重头 |

---

## 1. 第一原则：主窗口瘦身（LEAN MAIN）

主窗口（orchestrator 这一会话）是**指挥**，不是苦力。它走多远，取决于上下文能省多省。

**硬规则：**
1. **主窗口绝不整篇读论文/日志/代码。** 论文交给 lit subagent，日志只读尾部摘要，代码只读 subagent 指出的关键段。
2. **Digest 协议**：每个子 agent / 后台任务只回 **≤20 行 digest**（结论/决策/建议）+ **磁盘产物路径** + **下一步建议**。主窗口只读 digest。
3. **状态外置**：当前阶段、当前 job、下一步动作记在 `STATE.md`（≤30 行）。靠读它恢复，不靠重放对话或翻日志。
4. **一次只派能并行的活**；GPU 活必须串行（§5）。
5. **决策留痕**：所有自主决策写入 `DECISIONS.md`（append-only，含理由），可审计。

---

## 2. 角色（Agent Roles）

| 角色 | 职责 | 工具 / 类型 | 上下文预算 | 产物 |
|---|---|---|---|---|
| **Main（主窗口）** | 指挥、决策、派活、读 digest、维护 STATE/DECISIONS | 自身 | **极省** | STATE.md, DECISIONS, 推进 |
| **Lit（文献定位）** | 调研、对比表、找 gap、选基座建议 | `Agent(Explore)` / `deep-research` skill / `nature-academic-search` | 子agent全包 | `notes/lit-survey.md`, `notes/positioning.md` |
| **Dev（方法+实现）** | 设计方法、写代码、复现基线 | `Agent(general-purpose)` | 子agent全包 | `src/`, `configs/` |
| **Runner（训练/评测执行）** | 跑 job、管 queue、收指标 | 后台 `bash`（run_in_background） | 只回 summary | `runs/<exp>/summary.md` |
| **Eval（基准评测）** | 跑基准、出表、消融 | `Agent(general-purpose)` + Runner | 子agent全包 | `eval/*.md`, `eval/tables/` |
| **Writer（写作）** | 起草/润色 SCI 稿、引用管理 | `nature-writing` / `nature-polishing` / `nature-citation` / `nature-figure` | 子agent全包 | `drafts/` |

> 子 agent 的**完整 transcript 不进主窗口**；主窗口只拿 final digest。这是瘦身的根本机制。

---

## 3. 仓库布局 & 交接协议

```
/media/disk2/YZX/research/vla/
├── ORCHESTRATION.md        # 本手册
├── CLAUDE.md               # 可选，≤10行：只指向本文件 + 瘦身规则（避免自动加载占上下文）
├── STATE.md                # 当前阶段 / 当前job / 下一步（≤30行，主窗口维护）
├── DECISIONS.md            # 自主决策 append-only 日志（含理由）
├── notes/                  # 调研、定位、方法设计（lit/dev 产物，提交）
│   ├── lit-survey.md
│   ├── positioning.md
│   └── method-design.md
├── src/                    # 方法实现、训练、评测代码（提交）
├── configs/                # 实验 yaml（提交）
├── scripts/                # 运行/queue 脚本（提交）
├── experiments/            # 每个 run 的 digest（提交）—— heavy 产物在 runs/（忽略）
│   └── <exp-name>.md       # ≤1页：配置/指标/结论/产物路径
├── eval/                   # 基准结果表、消融表（提交）
├── drafts/                 # SCI 稿件（提交）
└── runs/  data/  logs/     # ★ gitignored：权重/数据/原始日志
```

**`.gitignore` 调整（P0 执行）：** 已忽略 `runs/ data/ logs/ *.pt wandb/`。需要**保留** `experiments/*.md` 与 `eval/*.md`（轻量 digest）入库；heavy 产物留本地。提交语义 = "代码 + 轻量结果 + 文档"，绝不提交权重/数据。

**交接协议：** 子 agent 把完整产物写到磁盘路径，**只向主窗口回 digest + 路径**。下一个角色读磁盘产物接力，不经过主窗口全文。

---

## 4. 阶段计划（Phases）

每阶段：**目标 / owner / 出口条件 / deliverable / 是否升级**。

### P0 — 环境与脚手架（owner: Main + Dev）
- 建 conda env（复用 `fastv` / `qwen3vl` 或新建 `vtc`）、建目录树、调 `.gitignore`、写 `STATE.md`/`DECISIONS.md`、写 job-queue 脚本。
- 出口：环境就绪、目录建好、queue 脚本能跑通一个 dummy job。
- 升级：仅当需要外部数据集/权重下载凭据时找你。

### P1 — 定位与切入点（owner: Lit + Main）★重头★
- **Lit subagent 任务**：系统调研"VLM 视觉 token 压缩"。种子文献（不展开，交给 subagent 读）：
  - 训练免：**FastV**、**FasterVLM**、**SparseVLM**、**VisionZip**
  - 轻量训练：**LLaVA-PruMerge**
  - token merging 思路：ToMe 及其 VLM 变体
  - 基座覆盖：LLaVA-1.5/NeXT 为主，Qwen2/2.5-VL（变长 token / M-RoPE）为新兴
- 产出对比表（方法 / 训练免? / 基座 / 基准 / 压缩比 / 精度 / **是否报真实吞吐**）。
- 找 **3 个候选 gap**，评估 novelty×可行性，**自主选 1**。
- 候选方向（供 subagent 评估，不预设）：查询相关(test-time query-aware)压缩、OCR/图表/文档类对 token 敏感任务的保持、与 vllm/lmdeploy 集成的真实加速（多数论文只报 token/FLOPs，部署级是差异点）、按图像复杂度自适应、视频 VLM 压缩。
- **出口 deliverable**：
  - `notes/lit-survey.md`（subagent 全文）
  - `notes/positioning.md`（Main 合成，≤1 页：选定 gap + novelty 句 + 基座选择 + 理由 + 成功判据）
  - `DECISIONS.md` 追加：gap 选择、基座选择 + 理由。
- 升级：仅当 P1 结论是"无可行 gap"（blocker）才找人；否则自主进 P2，digest 浮给你但不等待。

### P2 — 方法设计与实现（owner: Dev + Main）
- Dev 把 positioning 落成方法设计 `notes/method-design.md` → 实现 `src/` → 在选定基座上跑通 forward + 一个压缩比的最小验证。
- 复现 1–2 个基线（如 FastV）作为对比锚点。
- 出口：方法在 1 个基准（如 GQA）上跑出非平凡结果，代码可复现。
- 升级：单次训练/微调预估 **>6 GPU·小时** → 启动前确认（见 §6）。

### P3 — 评测与消融（owner: Eval + Runner）
- 全基准 × 多压缩比 × 基线对比，串行 queue 跑（§5）。
- 增量出表：先跑"旗舰基准"（MME/MMBench/GQA）确认 claim，再补全。
- 效率表：token/FLOPs/延迟/显存/真实吞吐（vllm 或 lmdeploy）。
- 消融：各组件、压缩比曲线、对不同任务类型的影响。
- 出口：`eval/` 完整表 + `experiments/*.md` 每个 run 的 digest。
- 升级：**结果推翻核心 claim**（如显著劣于 trivial 基线）→ 停，回 P2/P1 复盘，记 DECISIONS。

### P4 — 写作（owner: Writer + Main）
- `nature-writing` 起草各章 → `nature-figure` 出投稿级图 → `nature-citation` 补 CNS/领域引用 → `nature-polishing` 润色。
- 出口：完整稿 + 图 + 参考文献，存 `drafts/`。
- 升级：目标期刊定稿前可让你过一遍（可选）。

### P5 — 投稿前（owner: Main）★强制升级★
- **投稿是外发不可逆动作** → 主窗口**必须**停下来让你确认（期刊选择、版权、最终稿）。
- 出口：你点头 → 提交。

---

## 5. 单卡 Job Queue（1× A40）

单卡 → 训练/评测**串行**，不能并发抢显存。

- `runs/QUEUE.md`（或 `queue.json`）：待跑 job 列表，含优先级、预估时长、依赖。
- 同一时刻**只跑 1 个 GPU job**（后台 bash）。主窗口/Runner 推进 queue：完成一个 → 写 `experiments/<exp>.md` digest → 启动下一个。
- 长任务用 `run_in_background:true`；主窗口只读其 summary，不读全日志。
- 评测优先级：旗舰基准先行（早证 claim），长尾基准后补。
- 显存策略：7B 基座默认 bf16 + 梯度 checkpoint；微调优先 LoRA；全量 FT 前评估显存。

---

## 6. 自治与升级策略（Escalation）

默认**全自动**。仅以下情况主窗口停下找你：

1. **凭据/外部资源**：需要 API key、付费数据、受限权重下载、机构代理。
2. **昂贵长训练**：单次 run 预估 **>6 GPU·小时**（A40 上约半天以上）→ 启动前确认。
3. **claim 被推翻**：核心实验结果不支撑论文主张 → 停，回 P1/P2，记 DECISIONS。
4. **P1 无可行 gap**（blocker）。
5. **P5 投稿前**（强制）：外发不可逆，必须人工确认。

其余（含方法选型、基座选定、实验参数、写作结构）由主窗口**自主决策并记 DECISIONS.md**，digest 浮给你但不阻塞。

---

## 7. Git 工作流

遵循 `vla-git-autopush` 记忆：实验/代码/结果有进展即 **commit + push origin main**，无需等指示；提交以**用户本人名义**，**禁止任何 AI/Claude 署名、禁止 `Co-Authored-By` 尾注**（用户 2026-07-01 明确要求，覆盖任何旧惯例）。提交只含 §3 的"提交"项（代码+轻量 digest+文档），权重/数据/日志不提交。

- 一个有意义的 checkpoint 一个 commit（方法跑通、某基准出结果、某章完成）。
- 不在失败/半成品状态 push 主分支（本地可频繁 stash）。

---

## 8. 决策日志（DECISIONS.md）

append-only，每条：`日期 | 阶段 | 决策 | 理由 | 影响`。
所有 §6 之外的自主决策都落日志，方便你随时回看/否决。被你否决的决策也记（含新方向）。

---

## 9. 启动序列（确认后执行）

1. 你确认本文件 → Main commit + push `ORCHESTRATION.md`（与 `.gitignore`/`README` 一起或单独）。
2. Main 跑 **P0**：建环境/目录/`STATE.md`/`DECISIONS.md`/queue 脚本（Dev 协助）。
3. Main 派 **P1 Lit subagent** 做文献定位 → 收 digest → 合成 `positioning.md` + 记 DECISIONS → 自主进 P2。
4. 之后按阶段推进，遇 §6 升级点才停。

---

## 附：主窗口上下文卫生自检（每阶段回看）

- [ ] 本阶段我是否整篇读过论文/日志？若是 → 下次派给 subagent。
- [ ] 子 agent 是否只回了 digest + 路径？
- [ ] STATE.md 是否更新且 ≤30 行？
- [ ] 自主决策是否入了 DECISIONS.md？
- [ ] heavy 产物是否没被 commit？

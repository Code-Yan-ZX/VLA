# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-01

## 当前阶段
**P0 待执行**（环境/脚手架 bootstrap）

## 已完成（setup 会话）
- 仓库 init + 绑定 `origin` (github.com/Code-Yan-ZX/VLA)，`main` 已首次 push；自动 push 已配（凭据在 `~/.git-credentials`）
- `.gitignore`（忽略权重/数据/日志）、`README.md`、`ORCHESTRATION.md`、`CLAUDE.md`、本文件 已落盘
- nature-skills 已装（lit/写作阶段可用 `deep-research`、`nature-writing/citation/polishing/figure`、`nature-academic-search`）

## 立即下一步 —— P0 任务清单
1. 建 conda env（复用 `fastv`/`qwen3vl` 或新建 `vtc`）：torch 2.8+cu128、transformers、accelerate、mmcv/评测库、选定基座依赖。
2. 建目录：`notes/ src/ configs/ scripts/ experiments/ eval/ drafts/`（`runs/ data/ logs/` 已忽略）。
3. 建 `DECISIONS.md`（表头：日期|阶段|决策|理由|影响）。
4. 写 `scripts/queue`（单卡串行：读 `queue.json` → 跑一个 → 写 `experiments/<exp>.md` digest → 推下一个）。
5. dummy job 冒烟测 queue。
6. P0 收尾 → commit+push → 把本文件更新为「P1 进行中」。

## 之后 —— P1（重头）
派 **Lit subagent** 做视觉 token 压缩文献定位（种子：FastV/FasterVLM/SparseVLM/VisionZip/LLaVA-PruMerge）→ 收 digest → 合成 `notes/positioning.md`（选 gap + 选基座 + 理由 + 成功判据）→ 记 `DECISIONS.md` → 自主进 P2。细节见 ORCHESTRATION.md §4 P1。

## 关键约束 / 备注
- 算力 1× A40 46GB，单卡串行；基座 P1 末自主选定（LLaVA-1.5/NeXT 或 Qwen2.5-VL/Qwen3-VL）。
- 本机已有 env：`fastv qwen3vl qwen3vl_clean qwen3vl_serving vllm_* lmdeploy llama_factory`。
- 升级找人：凭据 / 单次>6GPU·h 训练 / claim 被推翻 / 投稿前。

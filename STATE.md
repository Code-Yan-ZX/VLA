# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-01

## 当前阶段
**P1 进行中**（定位与切入点）—— P0 已完成

## P0 已完成（2026-07-01）
- env：新建 `vtc`（python 3.11 / torch 2.11.0+cu128 / transformers 5.12.1 / accelerate 1.14.0）✅ CUDA=A40
- 双 env 策略：`vtc`(新方法+Qwen-VL) / `fastv`(torch2.0·复现 FastV@LLaVA-1.5)
- 目录树建好；`scripts/queue` 串行 runner + `configs/queue.json` + `scripts/_dummy.py`；dummy & env-smoke 冒烟测通过
- `DECISIONS.md` 建好（3 条 P0 决策）；base 专属依赖(mmcv/flash-attn)延后

## 立即下一步 —— P1（Lit subagent 已派）
1. **Lit subagent（后台运行中）**：系统调研"VLM 视觉 token 压缩"。
   种子：FastV / FasterVLM / SparseVLM / VisionZip / LLaVA-PruMerge / ToMe 系列；基座 LLaVA-1.5/NeXT + Qwen2/2.5-VL。
   产出全文 → `notes/lit-survey.md`（含对比表：方法/训练免?/基座/基准/压缩比/精度/**是否报真实吞吐**）+ 3 候选 gap 评估。
2. Main 收 digest → 合成 `notes/positioning.md`（选 gap + novelty 句 + 基座 + 理由 + 成功判据）。
3. `DECISIONS.md` 追加 gap 选择、基座选择。
4. 自主进 P2（digest 浮给用户，不阻塞）。细节见 ORCHESTRATION.md §4 P1。

## 关键约束 / 备注
- 算力 1× A40 46GB，单卡串行；P1 末自主选定基座。
- transformers 5.x API 与 4.x 有变，基座加载/压缩 hook 实现需适配。
- 升级找人：凭据 / 单次>6GPU·h 训练 / claim 被推翻 / P1 无可行 gap / 投稿前。

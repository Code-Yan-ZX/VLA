# DECISIONS.md — 自主决策日志（append-only）

> 格式：`日期 | 阶段 | 决策 | 理由 | 影响`
> 主窗口自主决策落此；你随时可回看/否决。被否决的也记（含新方向）。

---

## 2026-07-01 | P0 | 新建专用 `vtc` conda env（python 3.11 + torch cu128） | 三既有 env（fastv/qwen3vl/qwen3vl_clean）均缺 accelerate/pillow/mmcv 且绑别项目栈；base 待 P1 末定，复用会污染或被绑死；研究项目长期可复现性优先 | env 构建脚本 `scripts/build_env.sh`，~2.10/cu128；base 专属依赖（mmcv/flash-attn/qwen）延后 P1/P2 加 | 后台构建中

## 2026-07-01 | P0 | base 专属依赖（mmcv/flash-attn）不在 P0 安装 | base 尚未选定，mmcv 是 LLaVA 专用且编译重，flash-attn 编译耗时；P1 选定 base 后按需装 | P0 env 为"通用骨架"，P1/P2 再补 base 栈 | —

## 2026-07-01 | P0 | vtc 落地为 torch 2.11.0+cu128 / transformers **5.12.1**（比预期 4.x 新）；采用"双 env"策略 | cu128 index 当前发 torch 2.11；transformers 已进 5.x major，API 较 4.x 有变；既有 legacy `fastv` env（torch 2.0/transformers 4.31）保留用于在 LLaVA-1.5 上复现 FastV 原始基线 | 新方法与 Qwen2.5/3-VL 走 `vtc`；FastV@LLaVA-1.5 复现走 `fastv`；P1 文献定位时确认 transformers 5.x 下基座加载路径 | —

## 2026-07-01 | P1 | **选定 Gap A：serving-engine-aware 视觉 token 压缩**（部署级真实吞吐，集成进 vLLM 内部） | survey 23 法 0/23 在 serving engine 内测吞吐；Novelty5×Feasib4=20 最高；B/C/D 被 2025-26 SOTA 挤占；A 的对手是"空白"非强基线 | 方向锁定；详见 notes/positioning.md | —
## 2026-07-01 | P1 | 贡献框定为"方法 + 测量"，非纯测量 | 纯"塞 FastV 进 vLLM 量吞吐"易被审稿人判为工程 paper；故方法 = 与 serving 协同设计的压缩器（边界级 prune + 感知 KV-cache/batch），测量为支撑 | 决定 P2 方法设计取向 | —
## 2026-07-01 | P1 | 基座：LLaVA-1.5-7B 起步 → Qwen2.5-VL-7B-Instruct 泛化 | LLaVA-1.5 vLLM 支持最成熟、576 tok/img、可复用 FastV/SparseVLM/VisionZip ckpt→最快出结果；Qwen2.5-VL 的 M-RoPE/变长 token 是更难更可发的部署故事 | 双基座策略；先骨架后泛化 | —
## 2026-07-01 | P1 | **设 go/no-go 闸门为 P2 第一里程碑（claim-overturn 风险）**：LLaVA-1.5-7B@vLLM 边界级压缩器 {0,25,50,75}%，测 served tok/s/req/s/TTFT/KV-cache on GQA+TextVQA | 核心 claim 押在"压缩→真实 wall-clock 加速"，但 under continuous batching 极可能 wall-clock 几乎不动；重投入前必须先测 | GO(≥1.5×prefill & ≥1.2×e2e req/s @ ≤2%掉点)→设计方法；NO-GO(<1.2×e2e)→转 negative-result paper 或 Gap D；NO-GO 触发 charter §6 升级找人 | 项目最大风险点
## 2026-07-01 | P1 | 设 novelty 复核闸门（probe 前先跑） | survey 截止 2026-07-01，但领域迭代快，"0/23"是核心 novelty，若有竞品 landed 需重估 | Dev 在 P2 step-1 做 last-6mo 复扫 | —

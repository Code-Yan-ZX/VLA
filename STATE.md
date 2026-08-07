# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：Rank-Before-Merge -> ACM MM'27
> 最近更新：2026-08-07 · **RankBridge gate 进行中**（user /goal 指令，预注册判据锁定）。

## ★ 当前：RankBridge gate（user 指令 2026-08-07，进行中）
- 背景：plain RBM 创新偏弱、stage-law 证据强；J5(07-24)/cascade(07-29) 冻结指串行 hybrid，RankBridge 为 user 令重启的**结构不同**设计（完整候选集上融合，非级联）。
- 方法：全 visual tokens 保留到 FastV layer-K；缓存每 merger unit 的 pre-merger L2 rank；layer-K attention rank 与 pre-rank 融合生成最终 25% keep mask。融合=quota（rho·k_i 保护席位）+ RRF（备选）；K=3。
- 实现：src/v3_premerger/baselines_hf.py `--mode rankbridge`（commit 5fa3933）。dry-check ALL PASS（quota rho=0≡FastV hidden 一致、rho=1≡pre top-k）；真权重 smoke n=4：rho=0 与 FastV 答案/ptid 全同，quota 计数精确。
- Dev gate（跑前预注册）：Qwen3-VL-8B，textvqa+ocrbench n=64，rho={0.1,0.2,0.3}，lock=max(两基准官方分均值)，平手取小 rho。参照臂：pre25 复用切片 + FastV-k3 新跑。
- Locked gate：n=200×4 bench（textvqa/docvqa/gqa/ocrbench）；臂=none(新)/RBM=cascade gate pre25(复用)/FastV-k2(复用参照)/FastV-k3(新)/RankBridge(新)。docvqa 用 HF harness 稳定配置 --max-pixels 600000（对应 vLLM mnbt32768/mns4）。
- GO/NO-GO（锁定）：GO=四基准均 ≥ max(RBM,FastV-k3)−1pp 且 ≥1 基准严格超强 parent（报二项 SE+paired McNemar z）；NO-GO=停止调参，bounded negative 入 digest。
- 产物：experiments/rankbridge_gate.md（≤80 行）；脚本 runs/rankbridge/{smoke,dev_gate,locked_gate}.sh+gate_analyze.py（gitignored）。

## ★ 既有结论（未变）
- P0-1：GLM 集成 blocker（think loop）→ GLM 全撤出论文（user 选 A，commit e772a18）。
- P0-2：paired bootstrap/permutation/McNemar 重算 Table1/3 零 mismatch；stage law 三族（Qwen3-VL/Qwen2.5-VL/InternVL3）text-dense pre≫post 显著；InternVL3 GQA indistinguishable；Qwen2.5VL RBM GQA/DocVQA exploratory。
- P0-3：pre-final pure-stage control 确认 text-dense stage effect（+30.0/+12.5pp）；GQA 无 stage effect（原反转为 layer-8 ranking artifact）——已写入论文（user 选 A）。
- 创新轮（08-06/07）：cross-arch M1 预测律 DEAD；LLaVA no-merger 对照 CONFIRMED（空间 merger 是 stage effect 必要条件）；D2 selector MIXED（benchmark-conditional，非 method）；P1-1 InternVL3 swap GENERALIZES；P1-3 效率 stage-neutral（r=0.50 确认）。
- claim 红线：不写 RBM beats/SOTA；RBM=鲁棒默认+OCR/text-dense 大胜 post 族；FastV=query-conditioned 强 baseline（胜 TextVQA/GQA/DocVQA、OCRBench 崩）。
- 投稿前/claim 推翻/凭据/>6GPU·h → 升级 user。GPU 单卡 A40 串行。

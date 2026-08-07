# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：Rank-Before-Merge -> ACM MM'27
> 最近更新：2026-08-07 · **RankBridge gate 进行中**（user /goal 指令，预注册判据锁定）。

## ★ RankBridge gate = NO-GO（user 指令 2026-08-07，预注册判据终判，方法冻结 plain RBM）
- 方法：全 visual tokens 到 FastV layer-3，pre-merger L2 rank × attention rank 融合（quota rho=0.2 locked），25% keep。实现 baselines_hf.py `--mode rankbridge`（commit 5fa3933；dry-check 全过，rho=0≡FastV）。
- Dev n=64：rho=0.2 lock（textvqa 0.7344/ocrbench 0.4310，mean 0.5827 > fst3 0.5671 > pre 0.4758）。
- Locked n=200 官方指标：textvqa rb **0.7950** > fst3 0.7633（z=+2.11）；docvqa **0.6023** > 0.5863；gqa 0.5100 ≈ 0.5050；ocrbench **0.4641 ≪ RBM 0.5801**（−11.6pp，z=−3.66）→ cond1 挂。
- 判据：GO 需四基准均 ≥max(RBM,FastV-k3)−1pp 且 ≥1 严格超强 parent → **NO-GO**（ocrbench 失败）。停止调参（不搜 rho/RRF/K），bounded negative 入 digest；方法维持 plain RBM。
- 观察：rb 全胜 FastV-k3（quota 有增益、救回部分 OCR 0.171→0.464），但纯 OCR 上全 pre 选择仍是唯一最优——非保护 80% 由 merger-distorted 特征上的 attention 选 = 机制预测的误排。training-free 组合四连负（router/QA/cascade/RankBridge），支持 fixed-point framing。
- 预算：rb≡RBM per-sample ptid 全等（200/200×3+181/181）；wall ≈ FastV。GPU ≈2.3h <6，无训练。
- 产物：experiments/rankbridge_gate.md（79 行 digest）；runs/rankbridge/（gitignored）。论文 spine 不变；§6 可选引用待 user OK。

## ★ 既有结论（未变）
- P0-1：GLM 集成 blocker（think loop）→ GLM 全撤出论文（user 选 A，commit e772a18）。
- P0-2：paired bootstrap/permutation/McNemar 重算 Table1/3 零 mismatch；stage law 三族（Qwen3-VL/Qwen2.5-VL/InternVL3）text-dense pre≫post 显著；InternVL3 GQA indistinguishable；Qwen2.5VL RBM GQA/DocVQA exploratory。
- P0-3：pre-final pure-stage control 确认 text-dense stage effect（+30.0/+12.5pp）；GQA 无 stage effect（原反转为 layer-8 ranking artifact）——已写入论文（user 选 A）。
- 创新轮（08-06/07）：cross-arch M1 预测律 DEAD；LLaVA no-merger 对照 CONFIRMED（空间 merger 是 stage effect 必要条件）；D2 selector MIXED（benchmark-conditional，非 method）；P1-1 InternVL3 swap GENERALIZES；P1-3 效率 stage-neutral（r=0.50 确认）。
- claim 红线：不写 RBM beats/SOTA；RBM=鲁棒默认+OCR/text-dense 大胜 post 族；FastV=query-conditioned 强 baseline（胜 TextVQA/GQA/DocVQA、OCRBench 崩）。
- 投稿前/claim 推翻/凭据/>6GPU·h → 升级 user。GPU 单卡 A40 串行。

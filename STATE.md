# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标升级：**Rank-Before-Merge → ACM MM'27**（user 批准 cascade + InternVL3 强化包）
> 最近更新：2026-07-29 · 会话交接（user 要求全停、换窗口继续；git token 已更新）。**论文不急，实验优先**（user 指示）。

## ★ 实验已完成
- **主表 26/26（官方指标·完整 split·零缺失）**：text-dense pre 全胜（Q3 TextVQA +38.4/DocVQA +24.3/OCRBench +363pts；Q2.5 +26.1/+11.0/+293；z≥12）；GQA post 显著微胜 +2.6–2.8pp（量级 1/10，无 text-dense crossover）；DocVQA none 锚补齐（Q3 0.956 skip0）。digest experiments/j7_main_table.md
- **Baseline 同模型同预算**：FastV 胜 TextVQA/GQA/DocVQA-600k（query-conditioned），RBM 胜 OCRBench +25.9pp；VZ-style≡post 字节一致双代；Pyramid canon@62.5% 近无损；K 敏感性 k3 最优（0.751 vs k2 0.646）。digest j7hf_baselines_n500.md
- **机制**：M1–M3 + **Jaccard≡1.000 双架构 → 因果=选择级、架构通用**（Qwen2.5 残差=reverse_indices 序置换 artifact）；**InternVL3 smoke 通过=模式同构**（TextVQA pre 0.75 vs post 0.375 +37.5pp、GQA tie）。digest r1_1_swap_jaccard.md / internvl3_onboarding.md
- **效率**：stage 中性 ±3%；25% 保留 +68% 吞吐、延迟 −36%。**负结果**：QA-gate λ 全负、hybrid/router FAIL（pre 不动点）。

## ★ 待办（按序，GPU 串行；sunlogin 钉 ~40% 幻影 util——等待逻辑只认 mem<6000，已修）
1. **Cascade gate 续跑**（幂等）：`bash runs/cascade/run_all.sh >> runs/cascade/run_all.out 2>&1 &`（gate 24 cell 已跑部分，续完→`runs/cascade/gate_result.json` 裁决：**GO=每基准 ≥max(两 alone)−0.5pp 且 ≥1 严格胜两者 z≥1.5 → full_splits.sh 自动；NO-GO→入负结果、方法冻结 plain RBM**）。digest experiments/cascade_gate.md（待终表）
2. **InternVL3 主矩阵**：`bash scripts/internvl3_main_matrix.sh`（none/pre/post×4 基准 full split，smoke 已绿）
3. **FastV k3 同 scope 补**：`bash runs/r2_same_scope/r2b_fastv_k3.sh`（双模型 k3；公平性=报最优 K）
4. **论文重构**（ACM MM 版，user 说不急）：方法泛化 merger-equipped VLMs + cascade 节（若 GO）+ InternVL3 泛化表；drafts/paper_v4.md 已 95% 完成（Qwen 双模型版）；审稿模拟一轮在 drafts/pre_submission_review.md
5. **投稿前升级 user**（charter 强制）

## ★ 红线（判据锁定于 DECISIONS.md，勿挪）
不跨模型宣 SOTA、不写 "beats existing methods"；GQA 只报显著微胜/无 crossover；VisionZip 官方数仅 mismatched 锚；cascade/QA gate 判据预注册不改；claim 标配置边界。

## ★ 约束/资产
env qwen3vl_clean（vllm 0.19 V1）；权重 /data/models/huggingface/hub（Qwen3/2.5/InternVL3 齐）；runner=src/v3_premerger/v3_premerger_runner.py（internvl3 family 已接入）；HF harness=src/v3_premerger/baselines_hf.py（pre/cascade/fastv/pyramid 模式）；runs/ gitignore、digest 入 experiments/；commit=Code-Yan-ZX 禁 AI 署名；升级=凭据/>6GPU·h 训练/claim 推翻/投稿前。手册 ORCHESTRATION.md。

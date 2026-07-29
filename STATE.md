# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：**Rank-Before-Merge → ACM MM'27**（user 批准 cascade + InternVL3 强化包）
> 最近更新：2026-07-29 · **cascade gate 终判 NO-GO（预注册止损，方法冻结 plain RBM，cascade 入负结果节）**；InternVL3 主矩阵后台跑中。

## ★ 实验已完成
- **主表 26/26（官方指标·完整 split·零缺失）**：text-dense pre 全胜（Q3 TextVQA +38.4/DocVQA +24.3/OCRBench +363pts；Q2.5 +26.1/+11.0/+293；z≥12）；GQA post 显著微胜 +2.6–2.8pp（量级 1/10，无 crossover）；DocVQA none 锚补齐。digest experiments/j7_main_table.md
- **Baseline 同模型同预算**：FastV 胜 TextVQA/GQA/DocVQA-600k，RBM 胜 OCRBench +25.9pp；VZ≡post 字节一致；Pyramid canon@62.5% 近无损；K 敏感性 k3 最优。digest j7hf_baselines_n500.md
- **机制**：Jaccard≡1.000 双架构 → 因果=选择级、架构通用；InternVL3 smoke 通过（TextVQA pre 0.75 vs post 0.375 +37.5pp、GQA tie）。digest r1_1_swap_jaccard.md / internvl3_onboarding.md
- **Cascade gate = NO-GO**：24/24 cell（n200×4 bench，paired，mean_ptid 三臂全同）。cas 8/8 cell 落后 max(pre,fst) 8–18pp（远超 0.5pp 阈值），无条件2。两 training-free 选择器串联不互补（pre 不动点第四证）。full_splits 不执行。digest experiments/cascade_gate.md（终表）
- **效率**：stage 中性 ±3%；25% 保留 +68% 吞吐、延迟 −36%。**负结果**：QA-gate λ 全负、hybrid/router FAIL、cascade NO-GO。

## ★ 待办（按序，GPU 串行）
1. **InternVL3 主矩阵（后台跑中）**：`scripts/internvl3_main_matrix.sh`（none/pre/post×4 基准 full split，smoke 已绿）→ out: runs/internvl3_main_matrix.out；digest experiments/internvl3_main_matrix.md（agent 产出）
2. **FastV k3 同 scope 补（排 ① 后）**：`runs/r2_same_scope/r2b_fastv_k3.sh`（双模型 k3；公平性=报最优 K）→ digest experiments/r2b_fastv_k3.md
3. **论文重构**（ACM MM 版）：方法泛化 merger-equipped VLMs + §6 负结果节收录 cascade NO-GO（按预注册分支写）+ InternVL3 泛化表；drafts/paper_v4.md（Qwen 双模型版 95%）；审稿模拟二轮（nature-reviewer）
4. **投稿前升级 user**（charter 强制）

## ★ 红线（判据锁定于 DECISIONS.md，勿挪）
不跨模型宣 SOTA、不写 "beats existing methods"；GQA 只报显著微胜/无 crossover；VisionZip 官方数仅 mismatched 锚；cascade/QA gate 判据预注册不改（已兑现 NO-GO）；claim 标配置边界。

## ★ 约束/资产
env qwen3vl_clean（vllm 0.19 V1）；权重 /data/models/huggingface/hub（Qwen3/2.5/InternVL3 齐）；runner=src/v3_premerger/v3_premerger_runner.py；HF harness=src/v3_premerger/baselines_hf.py（pre/cascade/fastv/pyramid）；GPU 空闲只认 mem<6000（sunlogin 幻影 util）；runs/ gitignore、digest 入 experiments/；commit=Code-Yan-ZX 禁 AI 署名；升级=凭据/>6GPU·h/claim 推翻/投稿前。手册 ORCHESTRATION.md。

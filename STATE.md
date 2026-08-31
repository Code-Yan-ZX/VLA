# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）
> 项目：VLM 视觉 token 压缩；目标：Rank-Before-Merge -> IEEE TCSVT。
> 最近更新：2026-09-01；**Q-RBM E1 门 = NO-GO**，回投稿硬化（RBM 稿保持 fallback）。

## 当前论文（fallback，未改动）
- 内容权威入口：`drafts/overleaf_submission/main.tex`；TCSVT 投稿包：`drafts/ieee_tcsvt_submission/`。
- 方法叙事：representation-dependent selection--merger ordering；定位 RBM=robust default、FastV=query-conditioned 强 baseline；不宣通用胜出。
- TCSVT latexmk 通过（0 undefined/overfull/Type3）；S9 run JSON 未恢复、exact H0n-vs-FastV CI 待 raw run 恢复后复算。

## Q-RBM 方向（2026-09-01，用户明确）— E1 gate 结果
- 分支 `exp/qrbm-e1`（基 origin/main 370fb7b，已 push origin/exp/qrbm-e1）；34 项未提交/gitignored 服务器产物全部保留。
- **E1 causal-importance gate = NO-GO（预注册判据兑现）**：held-out n=252，query 特征加入 pre-merger predictor 对 causal-utility 排序 ΔnDCG@25% **+0.0003（CI[−0.0000,+0.0008]，p=0.74）**→ 核心 G1 失败，不进 scorer 训练。G2/G3 亦 NO-GO；G4 仅 textvqa 通过。
- 结论：pre-merger 特征对 causal（answer-dependent）unit utility 是 query-blind —— 与论文机制、J5/D1 负结果一致；Q-RBM 假设被证伪。报告 `experiments/qrbm_e1_gate.md`；计划 `notes/qrbm_e1_plan.md`（FINAL）；数据 runs/e1/（gitignored）。
- 运行成本 ≈0.7 A40·h（8/8 cells，OCRBench 14 巨图 native OOM 跳过，与论文 ~10% skip 口径一致）。

## 下一步
1. 回投稿硬化（E1 NO-GO）：**先修 GQA claim 冲突**（见下），恢复 53/53 JSON 与 manifest，Qwen2.5 OCRBench matched-config 决策。
2. 论文不引入任何 Q-RBM 材料（E1 为干净 bounded negative，是否入稿 §negative 待 user 定）。
3. 实际投稿仍须 user 明确确认。

## GQA claim 待修（user 第 9 项，记录不改，投稿前必修）
- main.tex:557 n=200 "pre-final==post 0.0pp" vs full-split n=12578 Δ=−5.64pp CI[−6.42,−4.86] p=5e-05（reports/acmmm_final_controls_artifacts.md:83）→ 0.0pp 是采样假象，必须修订。

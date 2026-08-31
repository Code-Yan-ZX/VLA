# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）
> 项目：VLM 视觉 token 压缩；目标：Rank-Before-Merge -> IEEE TCSVT。
> 最近更新：2026-09-01；**P2 方法创新重启：Q-RBM**（覆盖 08-30 冻结），RBM 投稿稿保持 fallback。

## 当前论文（fallback，未改动）
- 内容权威入口：`drafts/overleaf_submission/main.tex`；TCSVT 投稿包：`drafts/ieee_tcsvt_submission/`。
- 方法叙事：representation-dependent selection--merger ordering；定位 RBM=robust default、FastV=query-conditioned 强 baseline；不宣通用胜出。
- TCSVT latexmk 通过（0 undefined/overfull/Type3）；S9 run JSON 未恢复、exact H0n-vs-FastV CI 待 raw run 恢复后复算。

## 当前分支/基座（Q-RBM 方向）
- 分支 `exp/qrbm-e1`（基 origin/main 370fb7b）；34 项未提交/gitignored 服务器产物全部保留，禁 reset/clean。
- env：qwen3vl_clean（torch 2.10 / transformers 4.57.6）；Qwen3-VL-8B-Instruct 本地 17G；A40 空闲。

## 新方向：Q-RBM（2026-09-01，用户明确，见 DECISIONS.md）
- 重启 P2，聚焦 **query-conditioned pre-merger 排序**；不重做 frequency/edge/diversity/router/cascade/OT/RankBridge 等 training-free 拼装。
- 边界审计：唯一未证伪单元 = **learned query-conditioned pre-merger ranker + causal-ΔLL 监督**（J5 手调 cosine 混合、D1 post-L2 蒸馏均 NO-GO，本方向三特征均不同）。

## E1 causal-importance gate（预注册，进行中）
- 计划 `notes/qrbm_e1_plan.md`（FINAL）；gate log `experiments/qrbm_e1_gate.md`；判据 §4 locked。
- 协议：Qwen3-VL-8B 全冻结；4 基准 × dev_64 + 新 held-out_64（均与 gate_200 不相交，审计 PASS）；G=8 块 drop-out occlusion，teacher-forced GT-LL；比较 pre/post-L2/FastV/qsim vs causal utility。
- 实现：build_e1_splits.py ✅ / e1_metrics.py ✅（synthetic 验证过）/ e1_causal_import.py ⏳。
- GPU 预算：≈0.5–1.1 A40·h（G=8，<6 ✓）。**G1 失败即 NO-GO，不进 scorer 训练。**

## GQA claim 待修（user 第 9 项，记录不改，投稿前必修）
- main.tex:557 n=200 "pre-final==post 0.0pp" vs full-split n=12578 Δ=−5.64pp CI[−6.42,−4.86] p=5e-05（reports/acmmm_final_controls_artifacts.md:83）→ 0.0pp 是采样假象，必须修订。

## 下一步
1. e1_causal_import.py 交付 → py_compile + CPU self-test → 4 样本 GPU smoke（测 s/pass、验 ΔLL）。
2. 预算复核 <6 GPU·h → 跑 512 样本 gate → e1_metrics.py → 判 G1–G4。
3. verdict 写入 experiments/qrbm_e1_gate.md；更新 STATE/DECISIONS；commit + push（Code-Yan-ZX，禁 AI 署名）。
4. G1 GO 才设计轻量 scorer（E2，计划 §9）；NO-GO 则回投稿硬化。

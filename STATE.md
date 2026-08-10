# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：Rank-Before-Merge -> ACM MM'27
> 最近更新：2026-08-10 · **新创新任务已锁定：Deferred RBM 零代码 gate；失败才进入 RBM-OT。**

## ★ RankBridge gate = NO-GO（预注册判据终判）
- 方法：全 visual tokens 到 FastV layer-3，pre-merger L2 rank × attention rank 融合（quota rho=0.2 locked），25% keep；实现 `baselines_hf.py --mode rankbridge`（dry-check 全过，rho=0≡FastV）。
- Dev n=64：rho=0.2 lock（TextVQA 0.7344 / OCRBench 0.4310，mean 0.5827 > FastV-k3 0.5671 > RBM 0.4758）。
- Locked n=200：TextVQA 0.7950 > FastV-k3 0.7633（z=+2.11）；DocVQA 0.6023 > 0.5863；GQA 0.5100 ≈ 0.5050；OCRBench 0.4641 ≪ RBM 0.5801（-11.6pp，z=-3.66）→ cond1 失败。
- GO 要求四基准均 ≥max(RBM,FastV-k3)-1pp 且至少一项严格超强 parent，故 **NO-GO**；停止调参，不搜 rho/RRF/K，方法维持 plain RBM。
- RankBridge 全胜 FastV-k3，但纯 OCR 仍明显落后 RBM；非保护 80% 由 merger-distorted attention 排序，符合 text-dense 误排机制。training-free 组合四连负，支持 fixed-point framing。
- 预算：RankBridge≡RBM per-sample ptid 全等，wall≈FastV，GPU≈2.3h；产物 `experiments/rankbridge_gate.md`，论文 §6 是否引用待 user 决定。

## ★ 论文与既有结论
- 投稿权威入口：`drafts/overleaf_submission/main.tex`；正文排版稿：`drafts/overleaf_body_draft/main.tex`（仅实测 Fig.1、无附录、正文+参考文献 10 页）。
- GLM 集成因 think loop blocker 已全撤；stage law 保留 Qwen3-VL/Qwen2.5-VL/InternVL3 三族 full-split 证据。
- paired 统计零 mismatch；三族 text-dense pre≫post 显著；InternVL3 GQA indistinguishable；Qwen2.5VL RBM 的 GQA/DocVQA 小边为 exploratory。
- pre-final control 确认 text-dense stage effect；GQA 无真 stage effect，原反转为 layer-8 ranking artifact。
- cross-arch M1 预测律 DEAD；LLaVA no-merger 对照确认空间 merger 为必要条件；D2 selector MIXED；InternVL3 swap GENERALIZES；效率仅 r=0.50 统计确认 stage-neutral。

## ★ 红线与下一步
- 不宣 RBM beats/SOTA；RBM=鲁棒 OCR/text-dense 默认，FastV=query-conditioned 强 baseline；RankBridge 只作 bounded negative。
- 投稿前/claim 推翻/凭据/单次 >6 GPU·h 升级 user；GPU 单卡 A40 串行。
- 服务器任务书：`experiments/rbm_ot_server_task.md`。先核验 CVPR'26 新颖性，再跑 `rho=1,K=3` Deferred RBM；其失败才实现固定 tau=0.05/20-iter RBM-OT。
- Locked gate：Qwen3-VL keep25% n=200×4；逐项 ≥max(RBM,FastV-k3)-1pp 且 ≥1 项 paired z≥1.5 超强 parent 才 GO；否则停止所有 OT 变体。

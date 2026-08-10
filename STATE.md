# STATE.md - 当前项目状态（主窗口维护，保持 <=30 行）

> 项目：VLM 视觉 token 压缩 · 目标：Rank-Before-Merge -> ACM MM'27
> 最近更新：2026-08-07 · **RankBridge gate = NO-GO；方法冻结为 plain RBM。**

## ★ RankBridge gate = NO-GO（预注册判据终判）
- 方法：全 visual tokens 到 FastV layer-3，pre-merger L2 rank × attention rank 融合（quota rho=0.2 locked），25% keep；实现 `baselines_hf.py --mode rankbridge`。
- Dev n=64：rho=0.2 lock（TextVQA 0.7344 / OCRBench 0.4310，mean 0.5827 > FastV-k3 0.5671 > RBM 0.4758）。
- Locked n=200：TextVQA 0.7950 > 0.7633（z=+2.11）；DocVQA 0.6023 > 0.5863；GQA 0.5100 ~= 0.5050；OCRBench 0.4641 << RBM 0.5801（-11.6pp，z=-3.66）-> cond1 失败。
- 判据要求四基准均 >= max(RBM, FastV-k3)-1pp 且至少一项严格超强 parent，因此 **NO-GO**；停止调参，不搜 rho/RRF/K。
- RankBridge 全胜 FastV-k3，但纯 OCR 仍明显落后 RBM；training-free 组合四连负，支持 fixed-point framing。预算约 2.3 GPU·h；产物 `experiments/rankbridge_gate.md`。

## ★ 论文与既有结论
- 投稿权威入口：`drafts/overleaf_submission/main.tex`；正文排版稿：`drafts/overleaf_body_draft/main.tex`（仅实测 Fig.1、无附录、正文+参考文献 10 页，旧投稿包不动）。
- GLM 集成因 think loop blocker 已全撤；stage law 保留三族 full-split 证据。
- paired 统计零 mismatch；三族 text-dense pre>>post 显著；InternVL3 GQA indistinguishable；Qwen2.5VL RBM 的 GQA/DocVQA 小边为 exploratory。
- pre-final control 确认 text-dense stage effect；GQA 无真 stage effect，原反转为 layer-8 ranking artifact。
- cross-arch M1 预测律 DEAD；LLaVA no-merger 对照确认空间 merger 为必要条件；D2 selector MIXED；InternVL3 swap GENERALIZES；效率仅 r=0.50 统计确认 stage-neutral。

## ★ 红线与下一步
- 不宣 RBM beats/SOTA；RBM=鲁棒 OCR/text-dense 默认，FastV=query-conditioned 强 baseline；RankBridge 只作 bounded negative。
- 投稿前/claim 推翻/凭据/单次 >6 GPU·h 升级 user；GPU 单卡 A40 串行。
- 下一步：决定是否在论文 §6 引用 RankBridge negative；随后做投稿前 go/no-go 与行政检查。

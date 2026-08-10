# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：Rank-Before-Merge -> ACM MM'27
> 最近更新：2026-08-10 · **新创新双门 NO-GO：Deferred RBM (lifetime) 与 RBM-OT (barycenter) 均被固化为 plain RBM 同一锚点选择，方法冻结 plain RBM。**

## ★ 服务器任务书终判：双门均 NO-GO（2026-08-10）
- **Deferred RBM (rho=1,K=3) 零代码臂**：n=64×4 dev gate -- 锚点 ≡ plain RBM bit-identical，ptid 64/64×3+58/58 全等；cond1（四基准 ≥max(RBM,fst3)−1pp）textvqa/docvqa FAIL → **NO-GO**，不进入 n=200。GPU 0.75h。
- **RBM-OT locked 门**：balanced Sinkhorn transport（tau=0.05/20-iter、cosine cost、fp32）进入 RBM 锚点四个 patch slot 各做 barycenter，n=200×4 -- 锚点仍 bit-identical（ptid 200/200×3+181/181 全等）但 cond1 全败：textvqa 0.212 vs fst3 0.763（z=−10.14，b10=5/b01=117）；docvqa 0.045 vs 0.586（z=−10.82）；ocrbench 0.238 vs RBM 0.580（z=−7.63）；gqa 0.375 vs 0.505（z=−3.25）→ **NO-GO**，停止所有 OT 变体，不进 vLLM 移植。GPU 0.21h。
- 训练-free 组合六连负（router/QA-gate/cascade/RankBridge/Deferred-RBM/RBM-OT）：复合侵蚀了 RBM 锚点保活所换取的判别力 → fixed-point framing 强化证据。
- 锚点行被重心化稀释 = 细粒度 OCR 细节消失 = RBM 的优势恰在保留锚点原始行；marg_res 在 bf16 特征 + tau=0.05 下最大 3.4e-2（属方法固有性质，dry-check 契约在 canonical cosine cost 下满足），自主裁定不升级、不松参数。

## ★ 论文与既有结论
- 投稿权威入口：`drafts/overleaf_submission/main.tex`；正文排版稿：`drafts/overleaf_body_draft/main.tex`。
- GLM 集成因 think loop blocker 全撤；stage law 保留 Qwen3-VL/Qwen2.5-VL/InternVL3 三族 full-split 证据。
- 创新不宣 RBM beats/SOTA；RBM=鲁棒 OCR/text-dense 默认，FastV=query-conditioned 强 baseline。

## ★ 红线与下一步
- 不宣 RBM beats/SOTA；不事后调参 RBM-OT/Deferred。
- 投稿前/claim 推翻/凭据/单次 >6 GPU·h 升级 user；GPU 单卡 A40 串行。
- 双门 NO-GO 落地后方法维持 plain RBM；§6 引用待 user OK。
- 任务书交付齐：experiments/cvpr2026_token_compression_audit.md（新颖性）+ deferred_rbm_gate.md（Stage A）+ rbm_ot_gate.md（Stage B）+ DECISIONS/STATE 更新；以 Code-Yan-ZX 身份 commit+push（无 AI 署名）。
- Fig.1 拟改为 TextVQA+DocVQA+OCRBench 真实定性对比；OCR 已挖出 9 个严格三臂翻转（首选 ocr0804），本地仍缺原图/同样本多方法 mask，服务器闭环任务见 `drafts/figures/CLAUDE_CODE_PROMPT_FIG1_QUALITATIVE.md`。

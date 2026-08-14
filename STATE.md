# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：Rank-Before-Merge -> ACM MM'27
> 最近更新：2026-08-14 · **Overleaf 权威稿已完成投稿级重构：8 页正文，后接参考文献与补充材料；仅待作者及官方投稿元数据。**

## ★ 服务器任务书终判：双门均 NO-GO（2026-08-10）
- **Deferred RBM (rho=1,K=3) 零代码臂**：n=64×4 dev gate -- 锚点 ≡ plain RBM bit-identical，ptid 64/64×3+58/58 全等；cond1（四基准 ≥max(RBM,fst3)−1pp）textvqa/docvqa FAIL → **NO-GO**，不进入 n=200。GPU 0.75h。
- **RBM-OT locked 门**：balanced Sinkhorn transport（tau=0.05/20-iter、cosine cost、fp32）进入 RBM 锚点四个 patch slot 各做 barycenter，n=200×4 -- 锚点仍 bit-identical（ptid 200/200×3+181/181 全等）但 cond1 全败：textvqa 0.212 vs fst3 0.763（z=−10.14，b10=5/b01=117）；docvqa 0.045 vs 0.586（z=−10.82）；ocrbench 0.238 vs RBM 0.580（z=−7.63）；gqa 0.375 vs 0.505（z=−3.25）→ **NO-GO**，停止所有 OT 变体，不进 vLLM 移植。GPU 0.21h。
- 训练-free 组合六连负（router/QA-gate/cascade/RankBridge/Deferred-RBM/RBM-OT）：复合侵蚀了 RBM 锚点保活所换取的判别力 → fixed-point framing 强化证据。
- 锚点行被重心化稀释 = 细粒度 OCR 细节消失 = RBM 的优势恰在保留锚点原始行；marg_res 在 bf16 特征 + tau=0.05 下最大 3.4e-2（属方法固有性质，dry-check 契约在 canonical cosine cost 下满足），自主裁定不升级、不松参数。

## ★ 论文与既有结论
- 投稿权威入口：`drafts/overleaf_submission/main.tex`；正文排版稿：`drafts/overleaf_body_draft/main.tex`。
- GLM 集成因 think loop blocker 全撤；stage law 保留 Qwen3-VL/Qwen2.5-VL/InternVL3 三族 full-split 证据。
- 创新不宣 RBM beats/SOTA；RBM=鲁棒 OCR/text-dense 默认，FastV=query-conditioned 强 baseline。

## ★ Direction A/B 升级 scorer（2026-08-13，training-free）
- **freq scorer（B）α=1,β=0.6**：textvqa +1.2pp@25%/+0.5@50%（official）；docvqa@0.5 +0.6pp；但 docvqa/ocrbench/gqa@0.75 −1.6..−2.2pp。text-dense+高压缩受益。
- **adaptive router（A）ACCEPTANCE FAIL**（2 轮已尽）：τ_hf=0.08 全 PRE ≡ RBM（text 0 回归 ✓ 但 GQA +0.00 < +1pp ✗）；τ=0.13 flip GQA→POST 致 TextVQA −7pp 超限。GQA/TextVQA hf 分布重叠，无全局 τ 两全。如实记录。
- **combined ≈ freq-alone**（τ=0.08 router no-op）；ChartQA 加入（combined @25% 0.195 > RBM 0.170/FastV 0.180）。
- 交付：experiments/{freq_aware,adaptive_stage,combined}/results.json+grid.json + experiments/freq_adaptive.md；代码 freq selector + adaptive mode 已入 runner（dry-check 全过）。

## ★ 红线与下一步
- 不宣 RBM beats/SOTA；不事后调参 RBM-OT/Deferred。
- 投稿前/claim 推翻/凭据/单次 >6 GPU·h 升级 user；GPU 单卡 A40 串行。
- 双门 NO-GO 落地后方法维持 plain RBM；§6 引用待 user OK。
- 任务书交付齐：experiments/cvpr2026_token_compression_audit.md（新颖性）+ deferred_rbm_gate.md（Stage A）+ rbm_ot_gate.md（Stage B）+ DECISIONS/STATE 更新；以 Code-Yan-ZX 身份 commit+push（无 AI 署名）。
- 2026-08-14 最终图件：Fig.1=`fig2_rows1and2_combined.pdf`（两例 OCRBench × RBM/Post-L2/FastV-k3），题为 “Three selectors retain different visual evidence”；Fig.2 forest plot 不变；Fig.3=merger mechanism 三联图，题为 “What the native merger changes”；旧 Fig.1 SVG 删除，retention curve 仅保留 Supplementary Table S3 引用。
- 下一步仅人工项：核验 ACM MM'27 官方日期/地点/模板元数据；保持匿名投稿时不填作者，camera-ready 再填作者/单位/DOI/ISBN；实际上传/投稿须 user 确认。

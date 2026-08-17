# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：Rank-Before-Merge -> ACM MM'27
> 最近更新：2026-08-14 · **权威稿 8 页正文 + 2 页参考文献；三图与 49 条引用已终验，仅待官方投稿元数据。**

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
- 2026-08-14 终稿：正文 Fig.1--3 不变；补充 Fig.4 采用 TextVQA+DocVQA token-survival 合并版，Fig.5 合排 M1 分布+retention 曲线；`main.pdf` 22页（8页正文+2页参考文献，补充第11页起），无遮挡/越界/未定义引用。
- 参考文献 49 条且全部正文引用，集中补充 CVPR 2026（正式 CVF proceedings 与 arXiv 预印本严格区分）；无 undefined citation。已安装 `nature-academic-search` 到个人 Codex skills（下轮可用）。
- 下一步仅人工项：核验 ACM MM'27 官方日期/地点/模板元数据；保持匿名投稿时不填作者，camera-ready 再填作者/单位/DOI/ISBN；实际上传/投稿须 user 确认。

## ★ S6 2026-论文缝合（2026-08-17）：Diversity/Adaptive-RBM 达到当前 SOTA 平手
- **方法**：RBM + PRUNESID 式多样性-NMS（`--diversity nms`）+ AgilePruner 式自适应 τ
  （`--div-adaptive`：τ_r=min(τ, r·erank/scale·0.01)）。选择级、iso-token、占位符零改动。
- **dev n=200 r=0.25（official）**：ADAPTIVE 唯一赢 GQA（0.510>post 0.505）且 4/4≥现任；
  DV(τ0.6) textvqa 0.735/docvqa 0.737；freqDV docvqa 0.753。
- **n=500（决定性）**：8/8 格与现任为噪声内平手；textvqa-r0.25 DV 可靠超 FastV 臂
  （+8.3pp，20k bootstrap CI 排除 0）；其余 7 格 CI 含 0，dev 的 4/4 hold 未显著复现。
  → **结论：缝合方法"达到/持平当前 SOTA"（训练-free、零计数变化），未证明可靠全面超越**。
- **结构性发现**：hook 覆盖率 pre 126/200、post 84/200（vLLM 内部编码路径绕过 hook）→
  既有 pre/post "每图 k"契约仅对 ~42-63% 图严格成立；每图自适应预算与占位符契约不兼容
  （特征 pass 部分图不触发）→ E-AdaPrune 式预算判结构性 NO-GO，如实记录不宣。
- 交付：notes/s6_report_draft.md（终稿）+ notes/s6_stitching_design.md；代码与数据已提交
  （Code-Yan-ZX，无 AI 署名）。
- **下一步（待 user 定）**：① 接受"达到 SOTA 平手"为停点；② 继续推可靠超越
  （n=500 边界格加大样本、per-bench 自适应 τ、或 freq×diversity 进 n500）。

# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：Rank-Before-Merge -> ACM MM'27
> 最近更新：2026-08-13 · **Overleaf 权威稿已完成投稿级重构：8 页正文 + 1 页参考文献，后接补充材料；仅待作者及官方投稿元数据。**

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
- Fig.1 改为 frontispiece：单例 OCRBench ocr0422（机场信息牌，GT A105-108，ptid=69），3 区域（Input/Where to rank/Evidence），7.0×3.6 inch；候选 a_balanced/b_visual/c_minimal 渲染于 drafts/figures/frontispiece_fig1/candidates/；选 b_visual 为终版；validator 2026-08-12 扩展为 29/29（fail-closed 检 method-specific mask）；旧版备份为 previous_pipeline_fig1.pdf；overleaf caption 重写、§6 不宣通用 SOTA。
- 2026-08-13：Fig.1 视觉修复：重排为 input/order/evidence 三栏，证据 crop 改上下对齐，配色改低饱和陶土橙+深青灰+砖红；修复长 footer 撑大 PDF 裁切框（最终 520.9×243.6 pt），主稿 18 页编译及 page-3 渲染通过。
- 2026-08-13：Fig.2 forest plot 配色改为低饱和钢蓝/砖陶红/灰橄榄，背景改近白灰+暖象牙；数据与统计不变。主稿重新编译为 18 页，Fig.2 所在 page-6 渲染通过。
- 2026-08-13：按用户给定 TransPrune 风格重做 Fig.1：新增 `drafts/figures/frontispiece_fig1/render_main_framework.py`，采用输入→双排序路径→总体排名/结果→stage accumulation 的横向主框图；嵌入 OCRBench ocr0422 真实图像与 RBM mask，主稿 caption 同步改写。`C:\Python314\python.exe` 渲染通过，主稿 18 页编译、page-3 视觉检查通过。image2 风格探索因当前进程无 `OPENAI_API_KEY` 未执行；最终图保持代码原生、可复现。
- 2026-08-13：按用户进一步要求将 Fig.1 改为 CVPR insight-driven overview：标题突出 “Saliency should be measured before information mixing”，四区为 Problem→Insight→Method→Evidence；核心对照为 FastV `Merge→Rank` vs RBM `Rank→Merge`，方法分支只改变 ranking 位置，证据卡显示同预算 κ=25% 下 `✗ A115-126` vs `✓ A105-108`。新增可编辑 SVG `drafts/overleaf_submission/figs/fig1.svg`，PDF/PNG 及主稿 18 页编译通过。
- 2026-08-12：Post-L2/FastV-k3 源 200-sample run 未存 per-sample kept indices（仅 pre 存了），原 frontispiece 三方法共用 RBM mask（违反 task spec「Do not reuse the RBM mask for other methods and do not derive masks from image pixels」）。scripts/capture_frontispiece_masks.py 单例重捕：--mode post --r-post 0.25 与 --mode fastv --r 0.75 --fastv-k 3，ptid=69 + answer-prefix + OCRBench correctness 全等；mask 差异 RBM∩Post=18 RBM∩FastV=14 Post∩FastV=15（皆 < 54，distinct 真实）；PDF SHA 同；overleaf 同步。GPU 0.4 min，未升级。
- User 2026-08-12 否决三例 contact-sheet 作为首页图；改为单案例旗舰图：沿用原 Overleaf pipeline 流向/配色，以 OCRBench ocr0422（备选 ocr0804）串起 input→rank stage→真实局部 mask/答案；缺 FastV/Post mask 必须单例补捕获，禁复用 RBM mask。任务书见 `drafts/figures/CLAUDE_CODE_PROMPT_FIG1_QUALITATIVE.md`。
- 2026-08-12 投稿稿收口：摘要约 230 词；Fig.2 改为三模型×四基准 paired-95%CI forest plot；Fig.3 改为双模型共享 TextVQA/DocVQA retention 曲线；删除重复正文表、压缩负结果/局限/结论；统计主口径=20k paired bootstrap + paired sign-flip，McNemar 仅补充二值结果。
- Claim 全面按证据范围收紧：不宣 universal/SOTA/architecture-general/necessity/fixed point；GQA 写 no detected pure-stage difference；三模型=两 architecture families；四扩展=prespecified bounded negatives。
- `latexmk` 成功：总 18 页（body 1--8、refs 8--9、supp 10--18）；无 undefined cite/ref、无 overfull、主图及补充图均有 Description；最终本地 PDF=`drafts/overleaf_submission/main.pdf`（不入 git）。
- 下一步仅人工项：核验 ACM MM'27 官方日期/地点/模板元数据；保持匿名投稿时不填作者，camera-ready 再填作者/单位/DOI/ISBN；实际上传/投稿须 user 确认。

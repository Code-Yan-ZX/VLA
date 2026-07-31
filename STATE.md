# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：**Rank-Before-Merge → ACM MM'27**
> 最近更新：2026-07-31 · **真实数据图管线全绿**：`drafts/figures/real_data_pipeline/` 端到端跑通——10 张候选图经 Qwen3-VL-8B-Instruct 真实捕获 merger-input/output L2（repo 原版 `_score_units/_score_tokens`，vLLM enforce_eager，max_pixels=1.5M，25% 预算），NPZ 8 门校验全过（重复捕获比特一致 tol=0.0），逐图产出 4 面板对比（原图/pre top-k/post top-k/4 态 diff）+ 真实 Jaccard 0.22–0.35、Spearman 0.28–0.48（渲染值与校验报告逐图匹配）；FIG:1 扩展为 measured 模式（无 NPZ 即拒渲，`--allow-layout-proxy` 才许代理）；FIG:2/3 从审计 JSON 代码绘制（24 值与 j7/internvl3/j8 digest 逐格核对一致）；run_all.sh 六阶段断点续跑 + CPU smoke test 全过；23 个 PDF 回渲染 QA 全清（1 页/7.09in/无空白）。FIG:1 主图作者已定 **df7282e1** 并晋升 `drafts/figs/fig1.{pdf,png}`（measured merger L2，proxy 已替换）。下一步：嵌图渲染压页 ≤8+2→**二次升级 user 投稿 go/no-go**。

## ★ 实验已完成（全官方指标·full split）
- **Qwen 主表 26/26**：text-dense pre 全胜 vs post（Q3 +38.4/+24.3pp/+363pts，z≥12）；GQA post 显著微胜 +2.6–2.8pp（1/10 量级，无 crossover）。digest j7_main_table.md
- **InternVL3-8B 16/16**：pre−post text-dense **+37.4pp(TextVQA 0.789/0.415) / +34.6pp ANLS(DocVQA 0.728/0.382) / +432pts(OCRBench 753/321) = 与 Qwen3 量级同构，三族确立 stage law**；GQA tie（0.599/0.603）；none 锚 0.834/0.922/852/0.629；r0.875 pre 0.723/0.505 vs post 0.306/0.245；req/s pre≈2.2–2.5× none。注意：DocVQA 4M px cap（≠Qwen 600k，跨模绝对值不可比）；pre vs none DocVQA −19.4pp（非近无损，text-dense 内 TextVQA 才 −4.5pp）。digest internvl3_main_matrix.md
- **FastV k3 最优 K 基线 8/8**（同 scope 官方重打分）：FastV-k3 胜 RBM TextVQA +17.2/GQA +8.9/DocVQA +16.2pp；**RBM 胜 OCR +16.5pp（最优 K 也救不回 OCR）**。digest r2b_fastv_k3.md
- **Cascade gate = NO-GO**：8/8 cell 落后 max(pre,fst) 8–18pp → 方法冻结 plain RBM、入负结果节。digest cascade_gate.md
- **机制**：Jaccard≡1.000 双架构（因果=选择级）。**效率**：25% 保留 +68% 吞吐。**负结果链**：QA-gate/hybrid/router/cascade 四连（pre 不动点）。
- **定性图（10 图实测）**：pre/post L2 top-k Jaccard 0.22–0.35、Spearman 0.28–0.48 → merger 大幅重排单元秩（stage law 的图像级证据）；FastV 面板跳过（无真实 question，manifest 缺）。

## ★ 待办
1. ✅ **FIG:1 主图 = df7282e1**（作者选，已晋升 `drafts/figs/fig1.{pdf,png}`，实测数据版；contact sheet + 9 张备选 fig1_* 留 outputs/ 备换；compare_* 10 张可作 supp 定性材料）
2. **嵌图 + 渲染 + 实测压页 ≤8+2**（预估 9–11pp；4 个 COMPRESS-OPTIONAL 杠杆已埋于 tex；编译环境届时定：装轻量 TeX 链（需 sudo，升级 user）或 user 环境编）→ **二次升级 user 投稿 go/no-go（charter）**
3. **投稿行政**：OpenReview 账号/profile、author list 早锁（abstract 截止后不可改）、MM'27 官方 CFP 待出（现按 MM'26 proxy）

## ★ 红线（判据锁定 DECISIONS.md）
不跨模型宣 SOTA、不写 beats existing methods；GQA 只报微胜/tie·无 crossover；VZ 官方数仅 mismatched 锚；预注册判据不改；claim 标配置边界（像素 cap/预算/n）。图只用实测 merger L2，禁 CLIP/SigLIP/attention 代理（管线已硬编码拒渲）。

## ★ 约束/资产
env qwen3vl_clean；权重齐；runner=v3_premerger_runner.py；HF harness=baselines_hf.py；runs/ gitignore、digest 入 experiments/；commit=Code-Yan-ZX 禁 AI 署名；升级=凭据/>6GPU·h/claim 推翻/投稿前。真实图管线=real_data_pipeline/（run_all.sh 断点续跑；data/*.npz 不入库，审计 JSON 白名单入库）。手册 ORCHESTRATION.md。

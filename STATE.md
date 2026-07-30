# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：**Rank-Before-Merge → ACM MM'27**
> 最近更新：2026-07-30 · 全部实验落定 + **r2c 同 scope 配对数齐**（Table 3  backing：无全域胜者，RBM 胜 OCR 双族+Q2.5 微胜，FastV-k3 胜 Q3 text-dense）→ ④ 终稿修订进行中，完成后升级 user（投稿前 charter）。

## ★ 实验已完成（全官方指标·full split）
- **Qwen 主表 26/26**：text-dense pre 全胜 vs post（Q3 +38.4/+24.3pp/+363pts，z≥12）；GQA post 显著微胜 +2.6–2.8pp（1/10 量级，无 crossover）。digest j7_main_table.md
- **InternVL3-8B 16/16**：pre−post text-dense **+37.4pp(TextVQA 0.789/0.415) / +34.6pp ANLS(DocVQA 0.728/0.382) / +432pts(OCRBench 753/321) = 与 Qwen3 量级同构，三族确立 stage law**；GQA tie（0.599/0.603）；none 锚 0.834/0.922/852/0.629；r0.875 pre 0.723/0.505 vs post 0.306/0.245；req/s pre≈2.2–2.5× none。注意：DocVQA 4M px cap（≠Qwen 600k，跨模绝对值不可比）；pre vs none DocVQA −19.4pp（非近无损，text-dense 内 TextVQA 才 −4.5pp）。digest internvl3_main_matrix.md
- **FastV k3 最优 K 基线 8/8**（同 scope 官方重打分）：FastV-k3 胜 RBM TextVQA +17.2/GQA +8.9/DocVQA +16.2pp；**RBM 胜 OCR +16.5pp（最优 K 也救不回 OCR）**。digest r2b_fastv_k3.md
- **Cascade gate = NO-GO**：8/8 cell 落后 max(pre,fst) 8–18pp → 方法冻结 plain RBM、入负结果节。digest cascade_gate.md
- **机制**：Jaccard≡1.000 双架构（因果=选择级）。**效率**：25% 保留 +68% 吞吐。**负结果链**：QA-gate/hybrid/router/cascade 四连（pre 不动点）。

## ★ 待办
1. **论文重构 ACM MM 版（进行中）**：新稿 drafts/paper_acmmm.md（v4 留作 Qwen 版快照）；约束 notes/acm_mm27_cfp.md（≤8+2pp、acmart sigconf、双盲、截稿 ~2027-03/04、track=Foundation Models area、**framing 锚多模态贡献禁纯效率**）；内容=方法泛化 merger-equipped VLMs + InternVL3 泛化表 + FastV-k3 最强基线 + §6 cascade NO-GO
2. **审稿模拟二轮**（nature-reviewer，重构稿完成后）
3. **投稿前升级 user**（charter 强制）

## ★ 红线（判据锁定 DECISIONS.md）
不跨模型宣 SOTA、不写 beats existing methods；GQA 只报微胜/tie·无 crossover；VZ 官方数仅 mismatched 锚；预注册判据不改；claim 标配置边界（像素 cap/预算/n）。

## ★ 约束/资产
env qwen3vl_clean；权重齐；runner=v3_premerger_runner.py；HF harness=baselines_hf.py；runs/ gitignore、digest 入 experiments/；commit=Code-Yan-ZX 禁 AI 署名；升级=凭据/>6GPU·h/claim 推翻/投稿前。手册 ORCHESTRATION.md。

# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：**Rank-Before-Merge → ACM MM'27**
> 最近更新：2026-08-05 · **第四族 GLM gate + Overleaf 证据审计完成**：GLM-4.1V-9B-Thinking n=200/keep25% text-dense pre≫post 复制（+16.8pp TextVQA / +9.8pp ANLS）；GQA greedy 协议下 inconclusive。Overleaf 新增 S11 全 cell/iso-token/4096 探针，并修正 S9.2/S9.3 共 13 个误标 raw acc 的 official 索引值与 FastV 过宽 claim；机制优先正文保留。GPU 5.5/6 h。**下一步：升级 user 投稿 go/no-go（charter）**。

## ★ 第四族 gate（GLM-4.1V-9B-Thinking · 2026-08-03）
- 候选审计（ModelScope 20+ 模型筛选、代码级 merger 证据）= experiments/latest_vlm_model_audit.md；TOP-1 GLM-4.6V-Flash processor 类需 transformers≥5.0rc 加载失败 → 按预注册切同架构 4.1V，无 config hack
- 官方分 n=200/keep25%/L2（none/pre/post）：textvqa 0.242/0.218/0.050；docvqa ANLS 0.104/0.130/0.031；gqa EM 0.150/0.160/0.115
- Δ(pre−post)：textvqa +16.8pp、docvqa +9.8pp → **text-dense stage law 跨第四独立谱系复制**；iso-token 逐样本精确（pre≡post ptid≈none 25%）
- GQA +4.5pp 方向反转但 **inconclusive**：greedy 解码 floor-collapse 全臂（containment 锚 0.775≈官方 0.77 证能力完好；模型官方协议=sampling）→ 双向不作方向 claim
- Runner glm4v 族分支 +340 行纯增量（四族 dry-check 回归全过）；digest = experiments/glm4v_stage_gate.md（含 v2 4096 探针附录）

## ★ 论文（投稿权威入口 = drafts/overleaf_submission/main.tex；drafts/latex 仅科学正文同步、仍含图占位）
- 入口重写：贡献①=因果机制（M1 秩重排/M2 被贬=文本笔画单元/M3 ranking-swap kept-set identity 定果）→ ②=stage law 为其泛化（三族 full splits）→ ③=封闭设计空间+regime map
- Table 3 → failure-mode/regime map；FastV=query-conditioned strong baseline；RBM=query-blind OCR-preserving robust default（not uniformly optimal）
- §4.2 第四族段落+表（claim 边界：仅 text-dense 臂 n=200 复制、GQA inconclusive、greedy 协议/像素 cap/n 边界）；二轮 must-fix 全部保留
- Overleaf S11 补齐 GLM 高精度分数/ptid/boxed/4096 诊断；S9.2/S9.3 已统一为 r2b/r2c official rescore；审计 = experiments/recent_submission_commit_audit.md

## ★ 待办
1. **升级 user：投稿 go/no-go（charter）**，附 GLM GQA-arm 判决与第四族口径
2. 嵌图 + 渲染 + 实测压页 ≤8+2（FIG:1 实测版已晋升 drafts/figs/；4 个 COMPRESS-OPTIONAL 杠杆已埋）
3. 投稿行政：OpenReview 账号/profile、author list 早锁、MM'27 官方 CFP 待出

## ★ 红线（判据锁定 DECISIONS.md）
不跨模型宣 SOTA、不写 beats existing methods；Qwen/InternVL GQA 只报微胜/tie·无 crossover；**GLM GQA 臂只报 inconclusive**；claim 标配置边界（像素 cap/预算/n/解码协议）；图只用实测 merger L2。

## ★ 约束/资产
env qwen3vl_clean；权重齐（qwen3vl/qwen2vl/internvl3/glm4v，ModelScope 布局）；runner=v3_premerger_runner.py（4 族）；runs/ gitignore、digest 入 experiments/；commit=Code-Yan-ZX 禁 AI 署名；升级=凭据/>6GPU·h/claim 推翻/投稿前。手册 ORCHESTRATION.md。

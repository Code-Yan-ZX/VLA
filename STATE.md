# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：**Rank-Before-Merge → ACM MM'27**
> 最近更新：2026-08-06 · **正文优先排版稿完成**：`drafts/overleaf_body_draft/main.tex` 仅保留实测 Fig.1（第2页），撤下 Fig.2/3 与附录接入；正文+参考文献 10 页、无越栏/未定义引用。原 22 页投稿包保持不动；服务器 GLM sampling gate 执行中。

## ★ 第四族 gate（GLM-4.1V-9B-Thinking · 2026-08-03）
- 候选审计（ModelScope 20+ 模型筛选、代码级 merger 证据）= experiments/latest_vlm_model_audit.md；TOP-1 GLM-4.6V-Flash processor 类需 transformers≥5.0rc 加载失败 → 按预注册切同架构 4.1V，无 config hack
- 官方分 n=200/keep25%/L2（none/pre/post）：textvqa 0.242/0.218/0.050；docvqa ANLS 0.104/0.130/0.031；gqa EM 0.150/0.160/0.115
- Δ(pre−post)：textvqa +16.8pp、docvqa +9.8pp → **text-dense stage law 跨第四独立谱系复制**；iso-token 逐样本精确（pre≡post ptid≈none 25%）
- GQA +4.5pp 方向反转但 **inconclusive**：greedy 解码 floor-collapse 全臂（containment 锚 0.775≈官方 0.77 证能力完好；模型官方协议=sampling）→ 双向不作方向 claim
- Runner glm4v 族分支 +340 行纯增量（四族 dry-check 回归全过）；digest = experiments/glm4v_stage_gate.md（含 v2 4096 探针附录）

## ★ 论文（投稿权威入口 = drafts/overleaf_submission/main.tex；drafts/latex 仅科学正文同步、仍含图占位）
- 入口重写：贡献①=因果机制（M1 秩重排/M2 被贬=文本笔画单元/M3 ranking-swap kept-set identity 定果）→ ②=stage law 为其泛化（三族 full splits）→ ③=封闭设计空间+regime map
- Table 3 → failure-mode/regime map；FastV=query-conditioned strong baseline；RBM=query-blind OCR-preserving robust default（not uniformly optimal）
- Table 1 = 三族 full-split 主矩阵；Table 2 = GLM 第四族 gate；Table 3 = FastV/RBM regime map；Table 4/5/6 = ranking-swap/cascade/efficiency，正文与补充索引一致
- 正文排版工作入口 = `drafts/overleaf_body_draft/main.tex`（仅 Fig.1、无附录、10页）；科学权威投稿包仍为 `drafts/overleaf_submission/`；服务器清单 = experiments/next_server_experiments.md
- Overleaf S11 补齐 GLM 高精度分数/ptid/boxed/4096 诊断；S9.2/S9.3 已统一为 r2b/r2c official rescore；审计 = experiments/recent_submission_commit_audit.md

## ★ 待办
1. 服务器 P0：GLM 官方 sampling（temp0.8/top-p0.6/top-k2）9-cell n=200 seed0；none 锚恢复才继续
2. 条件式 P1/P2：sampling seeds1/2 稳健性；InternVL3 ranking-swap kept-set identity n=200
3. 投稿 go/no-go + 行政：OpenReview/profile、author list、MM'27 CFP

## ★ 红线（判据锁定 DECISIONS.md）
不跨模型宣 SOTA、不写 beats existing methods；Qwen/InternVL GQA 只报微胜/tie·无 crossover；**GLM GQA 臂只报 inconclusive**；claim 标配置边界（像素 cap/预算/n/解码协议）；图只用实测 merger L2。

## ★ 约束/资产
env qwen3vl_clean；权重齐（qwen3vl/qwen2vl/internvl3/glm4v，ModelScope 布局）；runner=v3_premerger_runner.py（4 族）；runs/ gitignore、digest 入 experiments/；commit=Code-Yan-ZX 禁 AI 署名；升级=凭据/>6GPU·h/claim 推翻/投稿前。手册 ORCHESTRATION.md。

# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：Rank-Before-Merge -> ACM MM'27
> 最近更新：2026-08-06 · **P0-1 GLM BLOCKER：sampling 不恢复 anchor -> 停 GLM、撤 headline（已升级 user）**。P0-2 统计完成（0 mismatch，stage law 全显著）。GPU 串行续 P0-3/P1-1/P1-3。

## ★ P0-1 GLM 判决（BLOCKER，已升级）
- 官方 sampling（temp0.8/top_p0.6/top_k2/seed0/maxtok1024）n=200 none×3：textvqa 0.287 / docvqa 0.155 / gqa 0.165（greedy 0.242/0.104/0.150；官方 ~0.8/0.8/0.77）-> **anchor 不恢复**。
- 根因=模型在 vLLM 集成下陷入退化 think loop（95-100% 输出含重复环、98-100% 非boxed 输出是 loop；仅 22-32% 产出 boxed；95-98% 撞 maxtok）。非 max_tokens 问题（旧 4096 greedy 探针亦不收敛）。greedy/sampling 均远低于发布水平。
- 记账：runner acc 用 lenient score_textvqa（0.825 误导）；官方 VQA-acc=0.287（extract_final_answer+official_scorers）。paired_stats.py 已用正确抽取。
- 决策（按 user P0-1 规则）：停 GLM（不跑 seeds1/2、不 full split）；blocker=experiments/glm4v_sampling_blocker.md；**user 选 A 移除 GLM 全部**（commit e772a18 已 push：abstract/intro/contrib/Table1/§5.3 删 GLM、删 GLM 表 tab:glm、四族->三族、删 §S11 GLM-audit、修悬挂 ref；P0-2b 统计措辞同 commit 套用）。stage law 现三族（Qwen3-VL/Qwen2.5-VL/InternVL3，锚点+paired CI 全 valid）。待 Overleaf 编译验证。

## ★ P0-2 完成（commit d5f1a55，已 push）
- paired bootstrap/permutation(20k)/McNemar 重算 Table1/3，0 mismatch。stage law 全 text-dense pre≫post 显著（p≈5e-5）；InternVL3 GQA -0.4pp CI跨0 -> indistinguishable（非 tie）；Table3 Qwen2.5VL RBM GQA/DocVQA 小边 CI跨0 -> exploratory。产物 experiments/paired_metric_statistics.{md,json}+src/v3_premerger/paired_stats.py。
- P0-2b 完成（commit e772a18，已 push）：InternVL3 GQA 'indistinguishable'（非 tie，CI[-0.93,+0.17]）、Table3 Qwen2.5VL RBM GQA/DocVQA 小边 exploratory（CI 跨0）、GLM 全删；paper 双份同步（main.tex≡paper_acmmm.tex）；修了旧表注 factual error。待 Overleaf 编译。

## ★ 进行中（CPU 子agent）
- P0-3 Qwen3 tap-point 审计（pre=deepstack blk8 输出？post=main-merger 输出？+ pre-final control 设计）。
- P1-2 强基线审计（Hi-Lo/QuietPrune/IF-Prun 代码可用性）。

## ★ 队列（GPU 串行，GLM 已移除）
P0-3(Qwen3 pre-final control) -> P1-1(InternVL3 ranking-swap+Jaccard) -> P1-3(效率重复性)。P1-2 audit CPU 并行。

## ★ 红线（DECISIONS.md）
不跨模型宣 SOTA；Qwen/InternVL GQA 只报微胜/tie；GLM 待 user 定（建议撤）；claim 标配置边界；图只用实测 merger L2。

## ★ 约束/资产
env qwen3vl_clean；权重齐；runner=v3_premerger_runner.py（4族+sampling flags，commit a029c89）；runs/ gitignore；commit=Code-Yan-ZX 禁 AI 署名；升级=凭据/>6GPU·h/claim推翻/投稿前。手册 ORCHESTRATION.md。user paper WIP（DECISIONS/overleaf×3/glm4v_stage_gate）在 stash@{0} 待 user 合。

# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：Rank-Before-Merge -> ACM MM'27
> 最近更新：2026-08-06 (晚) · **创新突破轮（user 12h 自主）**：cross-arch M1 预测律已证伪（negative，commit 8de4505）；LLaVA-NeXT no-merger 对照跑中（测 merger 必要性）；D2 edge/var selector 实现中。P0-1..P1-2 完成。GPU 链 LLaVA->P1-1->P1-3->GLM->D2。

## ★ P0-1 GLM 判决（BLOCKER，已升级）
- 官方 sampling（temp0.8/top_p0.6/top_k2/seed0/maxtok1024）n=200 none×3：textvqa 0.287 / docvqa 0.155 / gqa 0.165（greedy 0.242/0.104/0.150；官方 ~0.8/0.8/0.77）-> **anchor 不恢复**。
- 根因=模型在 vLLM 集成下陷入退化 think loop（95-100% 输出含重复环、98-100% 非boxed 输出是 loop；仅 22-32% 产出 boxed；95-98% 撞 maxtok）。非 max_tokens 问题（旧 4096 greedy 探针亦不收敛）。greedy/sampling 均远低于发布水平。
- 记账：runner acc 用 lenient score_textvqa（0.825 误导）；官方 VQA-acc=0.287（extract_final_answer+official_scorers）。paired_stats.py 已用正确抽取。
- 决策（按 user P0-1 规则）：停 GLM（不跑 seeds1/2、不 full split）；blocker=experiments/glm4v_sampling_blocker.md；**user 选 A 移除 GLM 全部**（commit e772a18 已 push：abstract/intro/contrib/Table1/§5.3 删 GLM、删 GLM 表 tab:glm、四族->三族、删 §S11 GLM-audit、修悬挂 ref；P0-2b 统计措辞同 commit 套用）。stage law 现三族（Qwen3-VL/Qwen2.5-VL/InternVL3，锚点+paired CI 全 valid）。待 Overleaf 编译验证。

## ★ P0-2 完成（commit d5f1a55，已 push）
- paired bootstrap/permutation(20k)/McNemar 重算 Table1/3，0 mismatch。stage law 全 text-dense pre≫post 显著（p≈5e-5）；InternVL3 GQA -0.4pp CI跨0 -> indistinguishable（非 tie）；Table3 Qwen2.5VL RBM GQA/DocVQA 小边 CI跨0 -> exploratory。产物 experiments/paired_metric_statistics.{md,json}+src/v3_premerger/paired_stats.py。
- P0-2b 完成（commit e772a18，已 push）：InternVL3 GQA 'indistinguishable'（非 tie，CI[-0.93,+0.17]）、Table3 Qwen2.5VL RBM GQA/DocVQA 小边 exploratory（CI 跨0）、GLM 全删；paper 双份同步（main.tex≡paper_acmmm.tex）；修了旧表注 factual error。待 Overleaf 编译。

## ★ P0-3 完成 + paper 已写入（commit c8f7480，已 push）
- pre-final pure-stage control：TextVQA +30.0pp / DocVQA +12.5pp（pre-final≫post）-> text-dense stage effect 是真 stage effect、非 artifact。GQA pre-final==post（0.0pp）-> 原"反转"是 layer-8 ranking artifact。
- **user 选 A：GQA 重框为"text-dense 专属、GQA 无 stage effect"**（paper c8f7480 已写入：pre-final control 段 + GQA 重框，0 stray GLM）。digest=experiments/p0-3_prefinal_control.md。

## ★ 创新突破轮（user 指令：12h 自主 + 创新）
- **cross-arch M1 预测律 = DEAD**（experiments/cross_arch_predictive_law.md，commit 8de4505）：pre/post ranking divergence 不预测 stage-effect magnitude（per-image r≈0）。诚实 negative，不写入为预测律。
- **LLaVA-NeXT no-merger 对照 = CONFIRMED**（experiments/llava_nomergers_control.md，commit c9eb5cb）：LLaVA-1.5（per-token MLP，无空间 merger）pre/post-MLP L2 @25% n=200 -> TextVQA -2.17pp（McNemar p=0.125 n.s.）、GQA -0.5pp（p=1.0 n.s.）-> **无空间 merger 则无显著 stage effect**；对比 Qwen3 +38.4pp（p≈5e-5）。**空间 merger 是 pre>post stage effect 的必要条件**（非仅 amplifier，方向反转）。理论框为 characterization（stage effect ⟺ lossy spatial merger + text-dense），非预测律。
- **D2 merger-loss-aware selector = MIXED**（runs/qwen3_d2_selector/，commit c68d637）：--selector {edge,var}。**实际 paired delta（脚本 verdict 有 n_paired=0 bug，以 summary JSON 为准）**：textvqa edge -1.17pp/var -5.50pp（L2 胜）、docvqa edge +2.33pp/var +1.53pp（edge 胜）、gqa edge +2.50pp/var +1.00pp（edge 胜）。-> 机制导出 selector benchmark-conditional（docvqa/gqa 胜，textvqa 输），非 uniform 方法，不写入为 method。诚实 negative-ish。
- **P1-1 InternVL3 swap = GENERALIZES**（experiments/p1-1_internvl3_swap.md）：M3 ranking-swap identity 在 InternVL3 成立——swap≡pre（textvqa +0.00pp、docvqa +0.11pp、gqa -0.50pp）、Jaccard(pre,swap)=1.0、ans-id 0.98-1.0。stage law 亦在 InternVL3 成立（textvqa +39.3pp、docvqa +31.9pp、gqa +1.0pp）。机制 claim 现 architecture-general（3 族，2 种 merger 设计）。已写入论文（commit 见下）。
- **GPU chain 自主跑中**（run_gpu_chain.sh，commit 2156b31）：D2 rerun ✅(mixed) -> P1-1 ✅(GENERALIZES) -> P1-3 ✅ -> GLM(跑中). chain.log = runs/gpu_chain/chain.log。
- **P1-3 效率重复性 = stage-neutral within 3%, 部分确认**（runs/qwen3_efficiency/efficiency_summary.json）：pre vs post req/s 全 r 在 3% 内（r=0.25 +0.04%、r=0.50 +2.64%、r=0.75 -1.41%）。r=0.50 above-noise RETAINED；r=0.25/r=0.75 within-noise INCONCLUSIVE。verdict: "DO NOT RETAIN ≤3% claim as-is"（≥1 r inconclusive）-> paper 的 stage-neutral 措辞需软化（"within ~3%, confirmed at r=0.50"），不得宣称全部 r 统计确认。
- brainstorm：RBM 非 SOTA（FastV 胜 TextVQA/GQA）；贡献须 law/mechanism。routing/hybrid 已死；selective bypass 训练-free 不可行。
- **配额**：5h quota 超，reset ~2026-08-07 03:05。子agent 429 fail；主窗口直做 paper 集成。

## ★ 队列（GPU 串行，ollama 并发 OOM 风险）
LLaVA对照(跑中) -> P1-1 InternVL3 swap(已修 swap bug) -> P1-3 效率 -> GLM pre/post(seed0) -> D2 selector test。ollama(qwen3:32b 29GB) 并发会 OOM，各 gate 有 GPU-wait(>=40GB free)+retry。

## ★ 红线（DECISIONS.md）
不跨模型宣 SOTA；GLM 已撤；GQA 无真 stage effect（layer-8 artifact）；M1 预测律已证伪不写；LLaVA 对照若 pre>post 则 law WRONG 需 flag；claim 标配置边界；图只用实测 merger L2。

## ★ 约束/资产
env qwen3vl_clean（+fastv for LLaVA）；权重齐；runner=v3_premerger_runner.py（4族+sampling+pre-final+internvl3 swap+edge/var 待加）；runs/ gitignore；commit=Code-Yan-ZX 禁 AI 署名；升级=凭据/>6GPU·h/claim推翻/投稿前。手册 ORCHESTRATION.md。paper 双份同步；101 notes/scripts WIP 不触碰。

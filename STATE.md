# STATE.md - 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标：Rank-Before-Merge -> ACM MM'27
> 最近更新：2026-08-06 · **投稿前 NO-GO 修复轮启动**（user 6 任务计划 P0-1->P1-3）。科学主干已成（四族 stage law + 机制因果 M1/M2/M3），投稿前修统计/混杂/基线/效率。GPU 单卡串行。

## ★ 进行中（2026-08-06）
- **P0-1 GLM 官方 sampling gate**（GPU 跑中）：runner 加 --temperature/--top-p/--top-k（greedy 默认 bit-identical）+ 逐样本 gen_len/boxed/finish_reason；glm4v_sampling_gate.sh 用官方协议 temp=0.8/top_p=0.6/top_k=2/seed=0 重跑（greedy 违 do_sample 致 GQA floor-collapse none 0.15 vs 官方 0.77）。Stage1=none×3(seed0) 检 anchor 恢复；恢复->seeds1/2 pre+post 出 paired delta mean±std；不恢复->停 GLM 撤 headline。产物 runs/glm4v_gate_sampling/。commit 4c2791f。
- **P0-2 主统计**（CPU 子agent 跑中）：paired bootstrap/permutation(≥10k,固定seed)/McNemar 重算 Table1/3；InternVL3 全补 CI；GQA CI 跨0 只写 indistinguishable（禁无界 statistical tie）；Table3 小差值不稳->exploratory。产物 experiments/paired_metric_statistics.{md,json} + src/v3_premerger/paired_stats.py。
- **P0-3 Qwen3 tap-point 混杂**（代码审计子agent）：确认 pre=deepstack blk8 输出、post=main-merger 输出；将加 pre-final 独立模式（merger 真实输入 L2）vs post(merger 输出 L2) 作 pure-stage control，iso-selector/iso-budget/同样本。GPU 待 P0-1 释放。方向消失=claim 级事件停+报告。
- **P1-2 强基线审计**（web 子agent）：Hi-Lo/QuietPrune/IF-Prun 代码可用性；无代码->不手工复现、出 digest；评估 FastV 扩 full split。

## ★ 队列（GPU 串行）
P0-1(GLM sampling) -> P0-3(Qwen3 pre-final control) -> P1-1(InternVL3 ranking-swap+kept-set Jaccard) -> P1-3(效率重复性 5×)。P1-2 audit CPU 并行；GPU gate 视审计结果。

## ★ 红线（判据锁定 DECISIONS.md）
不跨模型宣 SOTA、不写 beats existing；Qwen/InternVL GQA 只报微胜/tie·无 crossover；GLM GQA 待 sampling 重判（旧 greedy 只 inconclusive）；claim 标配置边界（像素 cap/预算/n/解码协议）；图只用实测 merger L2；pre-final 方向消失即停不包装。

## ★ 约束/资产
env qwen3vl_clean；权重齐（qwen3vl/qwen2vl/internvl3/glm4v ModelScope 布局）；runner=v3_premerger_runner.py（4族+sampling flags）；runs/ gitignore、digest 入 experiments/；commit=Code-Yan-ZX 禁 AI 署名；升级=凭据/>6GPU·h/claim推翻/投稿前。手册 ORCHESTRATION.md。

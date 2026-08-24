# MALT 方法发现报告（malt_method_discovery.md）

> 分支 `exp/deferred-rbm-n200`；2026-08-25。工作名暂用 MALT-Memory/KV（未锁定）。
> 实时日志：`experiments/malt_goal_mode_log.md`；数据：`runs/malt_goal_mode/`。

## 1. 研究问题
为什么让最终会被删除的视觉 token 多参与 1–2 个 decoder block，使 macro 相对
immediate-RBM 提升 ~12.5pp？transient tokens 是作为完整 token 更新，还是只需要
作为一次性视觉 K/V memory 被永久 anchors 或文本 token 读取？

（待填：Phase-1 数据 + 回答）

## 2. 冻结事实与方法基线
- MALT-1 = rankbridge rho=1.0, keep 25%, fastv-k 1（transient 存活 2 个 block）。
- keep-set 与 immediate-RBM 100% 相同；adaptive gate NO-GO；7 项禁试已遵守。

## 3. 因果路径消融（H0–H4）
| 假设 | 定义 | 结果（待填）|
|---|---|---|
| H0 | immediate-RBM（decoder 前删除）| ... |
| H1 | MALT-1（2-block deferred）| ... |
| H2 | block text←transient 读取 | ... |
| H3 | block anchor←transient 读取 | ... |
| H4 | transient K/V-only（无自更新）| ... |
| H5 | no_both（text+anchor 都禁读）| ... |

## 4. 五个问题的回答（待填）
1. 去掉 text←transient 后增益是否消失？
2. 去掉 anchor←transient 后增益是否消失？
3. transient 自身 hidden/MLP 更新是否必要？
4. K/V-only 是否能复现 MALT-1 收益？
5. 主要信息接收者：text / anchor / 共同？

## 5. Phase-2 候选（≤3）
（待填：假设、唯一差别、预计 acc、预计 FLOPs、可证伪条件、Gate B 结果、否决原因）

## 6. 计算效率模型
- ΣN_l、ΣN_l²、真实 FLOPs（H4-real ragged vs MALT-1）。
（待填：来自 analyze_malt_compute.py）

## 7. 正确性验证
- Gate A smoke 全过（keep-set identity、2-block lifetime、L_after、no extra
  merge、H1 复现）；tiny-model 单测（含 kv_only 原始 K/V 投影验证）。

## 8. Novelty collision audit
- 子 agent 结论：read-only transient K/V memory 在 SwiftVLM/Reroute/ET-Prune/
  AnchorPrune/FastV/PyramidDrop/SparseVLM/GMC 中均无覆盖；最近邻为 deferred
  pruning（FastV 风格 full-aliveness）——"K/V-only no-self-update" 组合未见。
（待填：最终判定 + 引用）

## 9. GO/NO-GO
（待填：最终候选通过 Gate C → GO；全失败 → 冻结 MALT-1）

## 10. 复现清单
- 探针 manifest：eval/subsets/{bench}_explore64.jsonl（disjoint 验证过）
- 运行：scripts/run_malt_smoke.sh、scripts/run_malt_explore.sh
- 分析：scripts/analyze_malt_phase1.py、scripts/analyze_malt_compute.py
- 单测：scripts/test_malt_ablations.py
- 提交：见 git log（每 Gate 一次）

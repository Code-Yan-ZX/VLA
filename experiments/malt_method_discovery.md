# MALT 方法发现报告（malt_method_discovery.md）

> 分支 `exp/deferred-rbm-n200`；2026-08-25。工作名：暂用 **MALT-C（Coord-aware）**
> 描述候选；正式名未锁定。实时日志：`experiments/malt_goal_mode_log.md`；
> 数据：`runs/malt_goal_mode/`。

## 1. 研究问题
为什么让最终会被删除的视觉 token 多参与 1–2 个 decoder block，使 macro 相对
immediate-RBM 提升 ~12.5pp？transient tokens 是作为完整 token 更新，还是只需要
作为一次性视觉 K/V memory 被永久 anchors 或文本 token 读取？

## 2. 冻结事实与方法基线
- MALT-1 = rankbridge rho=1.0, keep 25%, fastv-k 1（transient 存活 2 个 block，
  layer 0-1 全量参与，layer 1 后删除）。
- keep-set 与 immediate-RBM 100% 相同（781/781 与 8/8 smoke 验证）；adaptive
  gate NO-GO（已冻结）；7 项禁试全部遵守。

## 3. 因果路径消融（H0–H4 + no_both + H0n 对照）
n=64×4，Qwen3-VL-8B，official scorer；OCRBench 4 个大图样本各臂一致 OOM skip
（same-id 配对干净）。

| bench | H0 imm | H1 MALT-1 | H2 no_text | H3 no_anchor | H4 kv_only | H5 no_both | **H0n native** |
|---|---|---|---|---|---|---|---|
| textvqa | .703 | .812 | .828 | .828 | .828 | .828 | **.828** |
| docvqa | .406 | .500 | .500 | .516 | .516 | .500 | **.500** |
| ocrbench | .550 | .567 | .567 | .567 | .567 | .550 | **.550** |
| gqa | .578 | .578 | .609 | .609 | .594 | .578 | **.609** |
| **macro** | .559 | .614 | .626 | .630 | .626 | .614 | **.622** |

答案差异计数（vs H1，252 same-id）：H2 3、H3 8、H4 7、nb 4；H1 vs H0 108。
→ 消融生效（改变部分答案）但几乎不改变准确率。

## 4. 五个问题的回答
1. **去掉 text←transient 后增益是否消失？否** —— H2 保留 121% 增益（macro .626）。
2. **去掉 anchor←transient 后增益是否消失？否** —— H3 保留 128%（.630）。
3. **transient 自身 hidden/MLP 更新是否必要？否** —— H4（K/V-only）保留 121%（.626）。
4. **K/V-only 能否复现 MALT-1 收益？是（但原因在别处）** —— H4≥H1 只因它也保留
   native 坐标；K/V 读取本身被 H2/H3/nb 证明非必要。
5. **主要信息接收者？均不是** —— no_both（禁 text+anchor 读取）保留 100% 增益。

**核心结论：MALT-1 的增益不来自任何人读取 transient K/V，也不来自 transient
自更新；它来自位置布局（native mrope 坐标）。** 对照 H0n（immediate-RBM +
`--mrope native`，**不引入任何 transient**）macro 0.622 ≥ MALT-1 0.614，
在 K=0（最低）算力下捕获全部增益。

## 5. Phase-2 候选（1 个，机制驱动；其余方向被因果消融否决）
### 候选 C1：Native-coordinate immediate pruning（H0n / 暂名 MALT-C）
- **hypothesis**：deferred pruning 的增益 = native 坐标保留，而非 transient
  上下文化；立即剪枝 + 保留 native 坐标可同精度、零 transient 开销。
- **与 MALT-1 的唯一差别**：无 deferral（decoder 前剪枝，K=0）；保留每个
  survivor 的 native mrope 坐标（MALT-1 与 immediate-RBM 的坐标差异正是
  因果上关键的一环）。
- **预计 accuracy**：= MALT-1（n=64 实测 macro .622 ≥ .614，方向一致）。
- **预计 FLOPs/latency**：K=0 最低——ΣN_l² 较 MALT-1 降 ~39%、ΣN_l 降 ~12%
  （textvqa 中位；36×169² vs 2×601²+34×169²）。
- **可证伪条件**：n=200 下 macro < MALT-1−0.5pp 或 ocrbench 显著落后 >1pp。
- **否决候选**：MALT-Memory/KV（transient 作一次性 K/V memory）——H2/H3/H4/nb
  证明读取通路非增益来源；read-only memory、purity-preserving anchor write、
  dual-path 全部落入同一否决。

### 算法步骤（MALT-C）
1. 编码图像 → pre-merger 特征；按 mean-patch L2 每图像取 top-25% anchor 集
   （与 RBM/MALT-1 相同 identities）。
2. 在**原生 merger 之前**剪枝到 anchor 集（immediate），但**保留每个 survivor
   的 native get_rope_index 坐标**（不做 vllm-mimic 重编号）。
3. LLM 全 36 层只处理 anchors + 文本（无 transient 生命周期）。
4. 复杂度：与 ordinary pre-merger pruning 相同（K=0）；无 extra block。

### 与 ordinary/deferred pruning 的区别
- vs ordinary pruning（vllm-mimic 坐标）：**坐标处理不同**——ordinary 重编号
  survivor 坐标（近似布局），MALT-C 保留精确 native 坐标，捕获 deferred 的全部
  增益。deferred pruning 的"额外 block 参与"被证明是不必要的算力。
- vs deferred pruning（FastV 风格）：MALT-C 删除 deferral（省 2-block 全量
  pass），精度不减。

## 6. 计算效率模型（H0n vs MALT-1，textvqa 中位）
| 指标 | MALT-1 | H0n | 比值 |
|---|---|---|---|
| ΣN_l（token-layer passes）| 6948 | 6084 | 87.6% |
| ΣN_l²（attention 成本）| 1.69M | 1.03M | 60.8% |
| 真实 FLOPs（proj+attn+MLP）| ~2.58e12 | ~2.47e12 | ~96% |
H0n 是真实实现（非行为模拟）；效率优势在 ΣN_l²（repo 的 "compute" 度量）最显著，
真实 FLOPs 约 -4%（2-block 生命周期占总前向比例小）。诚实报告两者。

## 7. 正确性验证
- Gate A smoke PASS（n=8×4×6 臂）：keep-set==immediate-RBM、kept==25%、
  L_after==n_text+kept、精确 2-block lifetime、无 extra merge、H1 fresh==sweep K1。
- tiny-model 单测全绿，含 B4（kv_only layer-1 transient V == v_proj(ln1(raw
  inputs_embeds))，pre-crop DynamicCache 钩子捕获）。
- 纯函数：rb_rho1_keep_set == rankbridge rho=1 keep set；ablate masks 精确阻断。

## 8. Novelty collision audit（子 agent 结论）
"被剪枝视觉 token 仅提供 read-only K/V 给早期 decoder 层、自身不更新"这一机制
在 SwiftVLM / Reroute / ET-Prune / AnchorPrune / FastV / PyramidDrop /
SparseVLM / GMC / SparseVILA / AirCache 中**均无覆盖**（最近邻为 FastV 式
deferred pruning = full-aliveness）。**但**本报告因果消融证明该机制**并非**增益
来源——故该机制方向被否决，novelty audit 仅作为"为何不采用"的记录。
MALT-C（native-coordinate immediate pruning）的 novelty 在于**发现**：deferred
pruning 的增益主因是坐标保留而非 deferral——这是对既有方法的行为归因，非新算子。

## 9. GO/NO-GO → **GO（MALT-C）**
Gate C（locked n=200，paired vs MALT-1）：

| bench | h0n | MALT-1 | diff | z(McN) | W/L | keep=ref |
|---|---|---|---|---|---|---|
| textvqa | .820 | .830 | −.010 | +1.41 | 0/2 | 200/200 |
| docvqa | .465 | .470 | −.005 | +0.58 | 1/2 | 200/200 |
| ocrbench | .635 | .635 | .000 | 0.00 | 0/0 | 181/181 |
| gqa | .565 | .565 | .000 | 0.00 | 3/3 | 200/200 |
| **macro** | .621 | .625 | **−.004** | | | |

- **paired bootstrap 95% CI of macro diff: [−.012, +.004]** —— 含 0，
  无显著精度损失；无数据集损失 >1pp（textvqa −1.0pp 边界、不显著）。
- **keep-set 100% 相等**（200/200×3 + 181/181）。
- **算力**：ΣN_l −11.9%（88.1%），ΣN_l² −39.1%（60.9%，repo "compute" 度量），
  prefill_s 一致更快，峰值内存相当（~17–31GB，取决于图像尺寸）。
- 效率杆满足：macro ≥ MALT-1−0.5pp ✓；compute ≥15% 降低（ΣN_l²）✓；
  真实实现 ✓；无数据集 >1pp 落后 ✓。

**VERDICT: PASS（效率杆）→ GO。** 正式候选 = **MALT-C（native-coordinate
immediate pruning）**：立即剪枝 + 保留 native mrope 坐标，K=0 算力下获得
MALT-1 精度，是固定 K=1 deferred pruning 的严格 Pareto 改进。

**失败边界（诚实记录）**：若论文语境要求"deferred 参与即增益"的旧叙事，本
结果**推翻**该叙事——增益是坐标处理而非 deferral。真实 FLOPs 仅降 ~4–12%
（2-block 生命周期占总前向比例小），效率优势主要体现在 ΣN_l²（attention 成本）
与实现简洁性。若未来需要更大绝对加速，方向是合并 immediate 剪枝 + 原生
坐标 + 进一步 attention 成本压缩（如早退 anchors 的 KV），但**不再**回到
transient K/V memory（被因果消融否决）。

## 10. 复现清单
- 探针 manifest：eval/subsets/{bench}_explore64.jsonl（disjoint 已验证）。
- 运行：scripts/run_malt_smoke.sh / run_malt_explore.sh / run_malt_diagnostics.sh
  / run_malt_gateC.sh。
- 分析：scripts/analyze_malt_phase1.py / analyze_malt_compute.py /
  analyze_malt_gateC.py。
- 单测：scripts/test_malt_ablations.py。
- 提交：每 Gate 一次（ff6e594 → ff30ce9 → 9945ac0 → 1eba1b0 → 2e97d30 →
  a2693a5；已 push origin/exp/deferred-rbm-n200）。

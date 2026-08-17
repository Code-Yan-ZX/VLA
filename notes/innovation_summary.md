# 创新探索综合报告（2026-08-17）

> 目标（user）：方法不再"太简单"，达 SOTA 或对论文有实质帮助。
> 结论先行：**未达 SOTA（1×A40 自主预算内不可达），但产出两项对论文有实质帮助的发现
> + 一条清晰的方法升级路径 + 三组已证伪的负发现**。全部数据/代码可复现。

## 一、探索了哪些方向（全部落地、可复现）

### D1. Learned boundary scorer（stage-distillation）——核心探索
- **方法**：小 MLP 把边界特征 [l2, edge, var, 位置, 邻域] 映射到 POST 阶段（全模型输出）
  重要性排序；全部特征在 pre-merger hook 实时可算（无像素、无 query）；训练几分钟。
- **离线可学性**（held-out 图，零 GPU）：MLP 大幅逼近 POST 排序
  （kept-set Jaccard：docvqa 0.297→0.502，textvqa 0.391→0.460，gqa 0.448→0.483）。
- **任务精度**（held-out 200，官方 scorer）：
  | bench | learned(POST) | l2 | Δ |
  |---|---|---|---|
  | textvqa | 0.435 | 0.595 | **−16.0pp** |
  | gqa | 0.550 | 0.515 | **+3.5pp** |
- **判决**：**learned scorer 的方法机制有效**（gqa 正面、可泛化），**POST teacher 在文本
  密集任务上反任务**（textvqa 大负）。
- **机制洞察（论文价值 ①）**：POST 阶段系统性降权高 edge（文本笔画）单位（rank_delta
  hi-edge −68.6 vs lo-edge +28.0）→ 学 POST = 学"避开文本" → 文本密集任务变差。这与论文
  M2（text-stroke demotion）+ Table 1（pre 选择胜 post 选择）**完全自洽，反向印证 RBM
  选 PRE 位置 + 保留文本是 text-dense 优势的本质**。

### D2. Content-driven per-image budget（像素谱/feature 谱）——双重负发现
- **像素谱**：128-512px 下 spectral-tau 累计能量全≈1.0、erank 全 0.05（自然图谱系重尾，
  代理无判别力）→ 像素复杂度信号不可用。
- **feature 谱**：有真实范围（mean_k=182.5, range=[80,256], iso-total）——但部署
  **skip=200 全跳过**，复现 S6 的故障 → **确认 S6"结构性 NO-GO"**（占位符先于特征固定、
  部分图特征 pass 不触发、cursor 顺序错位）。我此前"技术失败可修"的怀疑被证伪。

### 三组已证伪的负发现（论文 negative-results 素材）
1. POST-distillation teacher 在 text-dense 反任务（textvqa −16pp）。
2. 像素复杂度代理无判别力。
3. per-image budget 在本 vLLM 占位符架构上不可行（双重证据）。

## 二、给论文的实质帮助（已落地可写）
1. **机制段/negative results**：POST 降权文本 → PRE 选择被进一步机制化论证。这是
   "POST 视角重要性 ≠ 任务需要重要性"的完整证据链（离线 + 任务精度双支撑）。
2. **诚实方法评估**：learned scorer 可行性（gqa 正、text-dense 需任务接地 teacher）。
3. **严格评测协议**：held-out 切片 + 官方 scorer + 配对分析（已建 eval/learned_heldout +
   analyze 脚本，可直接用于论文消融）。

## 三、方法升级路径（若 user 要"方法不再简单"）
- **Corrective distillation**（任务接地 teacher）：从"答对的随机 keep-set"学 unit 有用性。
  标签批量生成（K×N 请求、batched）可压到 ~1.5-2h GPU + 训练/评测 ~1.5h，总 ~3-4h GPU
  （自主预算内）。
- **诚实预期**：learned ≥ l2 跨基准（+0-3pp）。l2 本身已强（RBM 基石），headroom 据
  freq(+1.2pp)/diversity(~0) 证据为 modest。
- **不会达 SOTA**：2026 竞品（ET-Prune/RUTA/RoRA/GMC）已占据 training-free/轻训
  query-aware+动态预算赛道；自主预算内无法追平。
- **另一条**：query-aware learned scorer（FastV 的 query 优势 + RBM 的 text-dense 优势），
  工程量大（需 query 条件化 + 任务接地 teacher），>6 GPU·h，需升级 user。

## 四、推荐（待 user 定）
1. **论文**：把 D1 机制洞察写入 negative-results/机制段（零风险、直接增强核心 claim）。
2. **方法**：若坚持方法升级 → 跑 corrective distillation（~3-4h GPU，自主可执行），
   无论 outcome（≥l2 或 ≈l2）都强化论文（learned scorer 或 "l2 近最优" 的更强论证）。
3. **SOTA**：诚实评估后不宣称；论文以机制 + 严格评测协议定位，等 budget/GPU 资源到位
   再攻 query-aware learned（需 >6 GPU·h，升级）。

## 五、代码/数据落地
- `scripts/learned_scorer.py`、`train_learned_scorer.py`、`proto_stage_distill.py`、
  `analyze_d1.py`、`analyze_d2.py`、`run_d1_eval.sh`、`run_d2_budget.sh`、`build_d2_budget.py`、
  `build_learned_heldout.py`；runner `--selector learned`（dry-check PASS）。
- 数据：`runs/v3_merger_aware/survival_capture_lrn200/`（n~192/189/154）、
  `runs/d1_learned/`、`runs/d2_budget/`、`eval/learned_heldout/`。
- 记录：`notes/innovation_directions.md` §七~十三（全过程）。

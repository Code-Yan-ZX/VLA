# S6 缝合实验价值评估（2026-08-17）

> 结论：**对目前论文帮助不大，属中小幅补强，不建议升级为主方法。**
> 依据：`notes/s6_report_draft.md` + `notes/s6_stitching_design.md` + 代码语义核对。

## 1. 实验预算与论文核心设置不一致（最致命）
- `src/v3_premerger/v3_premerger_runner.py:812`：`--r` 是**剪枝率**，`k = round(full×(1−r))`。
- 因此 S6 的 r=0.25/0.5 实际是**保留 75%/50%**；论文 headline 是保留 25%/12.5%（r=0.75/0.875）。
- 当前结果**没有证明缝合方法在论文强调的深压缩区间有效**。

## 2. 没有一个固定方法稳定胜出（n=500，逐格对现任 max(RBM pre, FastV post)）
- **Diversity-RBM：4 胜 1 平 3 负**（胜 textvqa.25/ocrbench.25/ocrbench.5/gqa.5，平 textvqa.5，负 docvqa 两格+gqa.25）。
- **Adaptive：3 胜 0 平 5 负**（只胜 3 个 r=0.5 格，r=0.25 深压缩侧全负）。
- “5/8 更高”是逐格挑选两种方法拼出来的结果，**不能作为单一方法成绩**。

## 3. 没有显著超过 RBM
- TextVQA 的 +8.3pp 是**相对 FastV post**（0.665）；相对该格真正 incumbent RBM（0.745）仅 +0.3pp，bootstrap CI 含 0。
- 其余 7 格 CI 全部含 0，报告自身结论为“8/8 格与现任噪声内平手”。

## 4. “CI 包含零”不能证明持平（方法论红线）
- 要声称非劣 / SOTA parity，需预设**非劣界**（如 Δ>−0.5pp 为可接受）并做**非劣性检验**；
- 普通双侧检验 CI 含 0 只是“无证据”，不是“证明持平”。
- 报告标题“达到/持平 SOTA”在统计上站不住。

## 5. 创新性稀释主线
- Diversity-NMS 与 adaptive τ 明确来自 PRUNESID / AgilePruner（ICLR'26）模块，属拼接他人模块；
- 弱于论文“stage law + 极简 RBM”的原创性与因果叙事。

## 6. 数据不可复算
- `.gitignore:62-63` 排除 `experiments/sota_stitch/`，git 中 0 个文件被跟踪；
- 报告中的 bootstrap CI 目前**无法独立复算**。

## 附：关于“hook 只覆盖 42–63%，其余样本没压缩”
- 该判断**很可能不成立**：项目早期已确认 encoder-cache replay 复用首次生成的已剪枝 embedding；
  每请求占位符仍被缩短，若缓存返回全量 embedding 通常会直接数量不匹配。
- 建议做一次**禁缓存对照审计**核实，但**不应因此推翻现有主表**。

## 若仍要补强论文（正确做法）
1. 固定单一方法（建议 Diversity-RBM，不混 Adaptive）；
2. 跑 **r=0.75/0.875**（论文的 25%/12.5% retention）；
3. 用**与调参前 200 条不重叠**的 slice 或 full split；
4. 用**非劣性检验**（预设非劣界），而非普通差异检验。

S6 现阶段只能作为阶段性探索记录，**不进主文**。

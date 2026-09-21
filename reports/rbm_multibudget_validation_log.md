# RBM Multi-Budget Validation — 总日志

## 会话信息
- 起始时间：2026-09-21
- 基 commit：`460c7d28145884b7ac6568a78e851123a5a439a8`（dcc-precheck exp B complete）
- 分支：`exp/rbm-multibudget-validation`
- 环境：conda `qwen3vl_clean`（python 3.10.20, torch 2.10.0+cu128, vLLM 0.19.0, transformers 4.57.6）
- GPU：1× A40 46GB，串行
- 模式：实验/审计/分析，不改论文、不做 figure。

## 协议要点（子 agent 精读得出，2026-09-21）
- **两个 harness 用相反约定**：vLLM `--r`=丢弃比(keep=1−r)；HF `--r-pre`=保留比。
  | keep | vLLM `--r` | HF pre `--r-pre` | HF FastV `--r` |
  |---|---|---|---|
  | 50% | 0.5 | 0.5 | 0.5 |
  | 25% | 0.75 | 0.25 | 0.75 |
  | 12.5% | 0.875 | 0.125 | 0.875 |
- `--mode pre`=merge 前按 merge-unit L2 排(剪最低)；`--mode post`=merge 后按 token L2 排；`--mode fastv`(仅 HF)=LLM 第 k 层按 last-query attention 剪。
- **RBM(pre)与 FastV 只在 HF harness 同表可比**；FastV 不在 vLLM runner。跨 vLLM/HF 绝对分不可直接比。
- 官方评分器=`src/v3_premerger/official_scorers.py`；生成时内联轻量 scorer，离线官方重打分。
- 任意 retention 无硬校验（clamp k≥1）；12.5% 已验证(`run_p0_2_ocrbench_matched.sh` r=0.875)。
- 多预算模板=`scripts/run_p0_2_ocrbench_matched.sh`。
- 关键文件：`src/v3_premerger/v3_premerger_runner.py`(vLLM)、`src/v3_premerger/baselines_hf.py`(HF)、`official_scorers.py`。

## 每 Gate 提交记录
- [x] Step 0：建分支 + 总日志（commit fe9a401）
- [x] Gate A：audit existing multibudget experiments（commit 见下）
- [ ] Gate B：add qwen3vl multibudget evaluation
- [ ] Gate C：analyze retention-dependent behavior
- [ ] Gate D：audit recent baseline feasibility
- [ ] Gate E：evaluate recent compression baseline
- [ ] Gate F：benchmark end-to-end efficiency
- [ ] Gate G：summarize rbm viability

## 实验登记表
| # | 日期 | 命令 | 数据集 | 模型 | retention | 样本 | metric | runtime | 输出目录 | commit |
|---|---|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |  |  |

# DCC 投稿前实验 B — 同 HF harness 的 Qwen3 TextVQA 全量对照（RBM vs FastV-k3）

日期：2026-09-19 ｜ 分支：`exp/qrbm-e1`（脚本 `scripts/run_dcc_expB_hf_textvqa.sh`，
分析 `scripts/analyze_dcc_expB.py`）｜ 状态：**完成**。

## 1. 协议

参照 `scripts/run_p1_fastv_hf_ocrbench.sh`（P1），benchmark 换 TextVQA，其余逐项相同：
- 模型：Qwen/Qwen3-VL-8B-Instruct（HF transformers eager），双臂**同一 harness**。
- RBM 臂：`--mode pre --r-pre 0.25 --mrope native`（保 25% merger 输入 unit，L2）。
- FastV 臂：`--mode fastv --r 0.75 --fastv-k 3`（LLM 第 3 层后按注意力保 25%）。
- 共同：`--benchmark textvqa --subset eval/full_splits/textvqa_val.jsonl --n 5000
  --max-pixels 4000000 --seed 0 --max-tokens 32`，greedy，同 prompt/处理器/评分器。
- 环境：conda `qwen3vl_clean`，`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  VLLM_NO_USAGE_STATS=1 VLLM_USE_MODELSCOPE=False VLLM_ENABLE_V1_MULTIPROCESSING=0`；
  1×A40；每臂 wall ≈ 2990 s（~50 min），日志 `results/dcc_presubmit/expB/*.log`。
- 冒烟：n=8 双臂 PASS（0 skip）后才进全量。

## 2. 旧 FastV 全量 TextVQA 结果可复用性核验（user 要求先核实）

`runs/r2_same_scope/r2b_qwen3vl_fastv_k3_textvqa_r0.75_full5000.json`：
- 字段直查：`engine = "hf-transformers (eager)"`（并带 `vllm_note = "engine differs
  from runner (vLLM); efficiency numbers use vLLM"`）——**该结果本来就是 HF harness
  跑的**（早前探索 agent 误报为 vLLM runner，已纠正）。
- 本次复跑逐位复现：acc **0.8494 = 0.8494**、skip **5 = 5**（且为**同一批 5 个
  sample ID**：35389/36283/36747/36759/36760）、mean_ptid **213.7 = 213.7**。
- 唯一差异：旧跑 max_pixels=0（native），本次 4M cap——实测不生效（TextVQA 小图
  不触及），逐位一致即证。**结论：旧结果与当前协议/代码版本一致，可复用成立**；
  本次两臂仍为全新独立跑（FastV 臂即复现实证）。

## 3. 结果（n=5000/臂，配对 n=4995）

| 臂 | 完成/失败 | VQA-acc | mean_ptid | n_image_kept |
|---|---|---|---|---|
| RBM pre r0.25 native-mrope | 4995/5000（5 加载失败） | **0.8374** | 213.7 | — |
| FastV r0.75 k3 | 4995/5000（同 5 个 ID） | **0.8494** | 213.7 | — |

- **相同最终视觉 token 数核验**：逐样本 `n_image_kept` 两臂 **0 mismatch**（4995/4995）；
  mean_ptid 213.7 = 213.7。
- **配对差值（RBM − FastV）= −1.20 pp**，95% CI [−2.18, −0.22]，置换 p=0.018；
  两臂 skip 完全同 ID（rbm-only 0 / fastv-only 0），无 skip 不对称。
- 完整统计：`results/dcc_presubmit/expB/analysis_full.json`。

## 4. 解读（无论方向如实报告）

1. 同 harness、同预算下 **FastV 在 TextVQA 上显著略优（+1.20 pp）**。
2. 与论文现状**不冲突**：tab:main 无 FastV 列；FastV 同 harness 对照目前只有
   OCRBench（tab:fastv，RBM +146）；`:195` 已明示 "FastV shows that RBM is not
   uniformly strongest"；`:91` 已限定 "Comparisons stay within backend"。
   本结果是该边界的又一个同 harness 数据点（TextVQA 侧 FastV 略优）。
3. **顺带发现（需投稿前知悉，非本实验 spearhead）**：HF RBM TextVQA 0.8374 vs
   vLLM 矩阵 RBM TextVQA 0.605（tab:main），同预算同方法 backend 差 ~23 pp；
   OCRBench 上同模式（HF 0.608 vs vLLM 0.547，已在 tab:fastv 并存披露）。方向
   一致、TextVQA 幅度更大。论文 `:91/:161` 的 within-backend 限定句已覆盖此风险，
   但建议投稿前明确 vLLM pre 路径（mrope 布局）与 HF native 的系统差来源
   （P1 脚本注：vllm-mimic 对 pre 是 known-degenerate path；qwen3vl vLLM j7 pre
   用的何种布局值得复核），避免审稿人两表对照时质疑 tab:main 的 TV RBM 格子。

## 5. 产物路径

- 原始结果：`results/dcc_presubmit/expB/{smoke_textvqa_r25,smoke_textvqa_k3,
  full_textvqa_rbm_pre_r25,full_textvqa_fastv_k3}.{json,log}` + `analysis_full.json`
- 命令/日志：`experiments/dcc_expB.log`；脚本见文件头。
- 旧结果核验对象：`runs/r2_same_scope/r2b_qwen3vl_fastv_k3_textvqa_r0.75_full5000.json`（未改动）。

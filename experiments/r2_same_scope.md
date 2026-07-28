# R2 — 同 scope baseline 补缺（2026-07-28，进行中）

状态：`scripts/r2_same_scope_baselines.sh` 已启动（后台长跑，预计 12–15 GPU·h）。
进度/日志：`runs/r2_same_scope/_campaign.log` + 每 cell `runs/r2_same_scope/r2_*.log`；断点续跑（json 存在即 skip）。
完成后自动官方 rescore → `runs/r2_same_scope/r2_official_summary.json`（TextVQA VQA-acc / GQA 官方 acc / DocVQA ANLS，OCRBench 元数据按 full-split 顶层字段 question_type/category，复用 j7_main_table.py 模式）。

## Cell 清单（全 qwen3vl，与我方 full split 同 scope）

| cell | 命令要点 | 审稿点 |
|---|---|---|
| r2_qwen3vl_fastv_k{1,2,3,4}_textvqa_r0.75_n500 | baselines_hf.py --mode fastv --fastv-k K, textvqa_val n=500 | R2: FastV K 敏感性 |
| r2_qwen3vl_none_docvqa_full5349 | runner --mode none --max-model-len 49152 --max-num-seqs 1 --chunk-size 200 | 主表 none 锚（full 5349） |
| r2_qwen3vl_fastv_k2_textvqa_r0.75_full5000 | baselines_hf.py fastv-k 2, textvqa_val 全 5000 | R2-1 同 scope |
| r2_qwen3vl_fastv_k2_gqa_r0.75_full12578 | baselines_hf.py fastv-k 2, gqa_testdev 全 12578 | R2-1 同 scope |

docvqa/ocrbench 的 baseline 沿用既有 n=500 cell（j7hf，runs/full_matrix/），不重跑。

## 预判规则

- docvqa none full：脚本会打印 skip 比例；skip>15% → 退报 subset none 0.976（n=200）入脚注并在此注明；否则 full 锚入主表。
- 完成后主窗口将本 digest 的"结果"节补上 r2_official_summary.json 数字并对照 j7_main_table.json 的我方 full-split cell。

## 结果（待 campaign 完成后填写）

_pending_

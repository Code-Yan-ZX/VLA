# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · /goal 锁定：Rank-Before-Merge → **CCF-B 主（CCF-A 扩展）submission-ready 方法论文**
> 最近更新：2026-07-23 · **暂停等 user GO（GPU 被占）**。J1：mrope 修复已写入 runner（`setup_qwen2vl_mrope_fix`，诊断=位置游标按 full grid 推进 vs 占位符已缩到 k→尾部文本继承 2D 网格位，block-mrope 聚集错误崩溃；qwen3vl interleaved 容忍故无事），**未验证**（sues 占卡），验证脚本就绪 `runs/v3_crossarch_cells/j1_smoke_wait.sh`（自动等卡空→6 cell：none 卫检→n8 快验→预注册 n50 pre/post≥0.40→textvqa→qwen3vl 回归），user 命令后执行。J0a 数据/J0b scorer/J4 HF-baseline harness=纯 CPU 后台继续不占卡。runner 改动未 commit（未验证不 push）。

## ★ 完成条件（user）：Qwen3-VL-8B + Qwen2.5-VL-7B 结论一致 / 官方完整 split（TextVQA·DocVQA·OCRBench·GQA）/ 强 baseline 充分（FastV·VisionZip-port·PyramidDrop）/ 机制+效率消融闭环 / 论文 submission-ready。**不跨模型宣 SOTA；仅同模型同 harness 领先才写"超现有方法"**。
## ★ 队列（串行 A40，每 job<6GPU·h）：J1 修通 Qwen2.5-VL → J2 跨代矩阵+官方 rescore+GQA n=200 tie 双模型确证 → J3 机制跨代复制 → J4 FastV/PyramidDrop port 探针 → **J5 QA 单次止损 gate（预注册：均值≥RBM+1pp 且 ≥3/4 基准不回退>0.5pp 且 z≥1.5，否则冻结 plain RBM、不再搜 hybrid/router，≤2GPU·h）** → J6 效率表 → J7 官方完整 split 主表（headline 先行，~50–60h 断点续跑）→ J8 消融闭环 → J9 paper_v4+venue（投稿前升级 user）。
## ★ 测绘要点：① Qwen2.5-VL 崩溃根因=`_patched_pii` 绕过 vllm mrope 重算（block [16,24,24] θ1e6 剪后三轴错位；Qwen3 interleaved 自洽）→ 剪后调 `recompute_mrope_positions` 或 wrap 不替换（qwen2_5_vl.py:1342-1397）；`--model-family qwen2vl` 已全链路、baseline 正常 → 仅压缩路径坏 ② VisionZip-style≡post（11/11），官方码不可跑→port+mismatched 锚 ③ 官方 scorer 补 OCRBench/1000+GQA exact-match ④ 完整 split：GQA 在盘，余三下载到 runs/data/（~ 90% 满勿写）⑤ fairness=同 keep ratio+报绝对 token 数+统一 min/max pixels（patch14≠16 需 family 校准 iso-token）。
## ★ 已确立 claim（推翻才动）：pre 弱占优无 crossover（GQA tie，n=200 待双模型确证）；text-dense 大胜（textvqa +38.3pp / docvqa +26.5pp / ocrbench +41.5pp）；M1–M3 机制因果链（swap≡pre）；selector-invariant；hybrid/router/adaptive 负结果如实报（强化机制 claim）。
## ★ 约束：env qwen3vl_clean（vllm0.19 V1）；权重 /data/models/huggingface/hub（~/.cache 同）；runs/ gitignore，每 run 交 experiments/<exp>.md digest；commit=**Code-Yan-ZX 禁 AI 署名**；升级=凭据/>6GPU·h 训练/claim 推翻/投稿前。
## 资产：runner src/v3_premerger/v3_premerger_runner.py（mode none/post/pre/hybrid, mask-ranking swap, selector l2/attn, visionzip-style, dry-check）；official_scorers.py（+J0b 补 OCR/GQA）；paper drafts/paper_v3.md；细节 ORCHESTRATION.md。

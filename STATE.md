# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md** + **next_plan.md**
> 最近更新：2026-07-03

## 当前阶段
**v2 P0 DONE（V1 引擎迁移 + Table A 复现，commit bff6871）。F1 在 V1 上成立且更强（c12/r75 1.86× vs V0 1.75×）。下一步 P1 = Qwen3-VL-8B 复测。**

## v2 转向（v1 论文待大改，不投）
v1 致命伤：V0 废弃路径 + LLaVA-1.5 过时 + toy 规模 + 0/37 脆弱 + 方法 n=500 null。→ 重做：**V1 引擎 + 现代模型 + 真实 serving 规模**。

## ★ v2 kickoff 结论（notes/v2_ecosystem_assessment.md, commit 597bf72）
- **V1 可行**：vLLM **0.19.0+cu128** 在 driver560/cu12.8/A40 跑通 V1（LLaVA-1.5-7B 3.2s、Qwen3-VL-8B 20.2s init）。env = `qwen3vl_clean`（vllm0.19.0/torch2.10+cu128）；vtc_serve(V0) 留 rollback。注：0.10.2 默认就是 V1，是我们 `serve_bench.py:41` 强制 V0 才走的老路径。
- **V1 controller 信号**：`llm.get_metrics()`→Prometheus（`vllm:num_requests_running`、`vllm:gpu_cache_usage_perc`），替代 V0 失效的 in-process scheduler 读取。
- **现代模型**：**Qwen3-VL-8B-Instruct**（A40 ~16GB bf16，已验证）。原生 2×2 MLP merger + 动态分辨率 → 变长 token（274~2300/图）→ F1/F2/F3 须复测，pruning 价值更大。
- **novelty 仍成立（0/N）**：RTP-LLM 是引擎非压缩机测量；vLLM-Omni 无 token pruning。**补相关工作**：DeepSeek-OCR(2510.18234, 互补)、EarlyTom(2605.30010, TTFT 拆解重叠 F2 须区分)、ADSC(2602.12618)。

## P0 结果（commit bff6871, notes/v2_p0_v1_tableA.md）
- **V1 集成路径**：`VLLM_ENABLE_V1_MULTIPROCESSING=0`（in-process EngineCore，保留 V1 scheduler：chunked prefill/prefix caching）→ V0 的 projector hook + `LlavaProcessingInfo.get_num_image_tokens` patch 直接复用（同一 model 访问链 + 同一 processor 类）。子进程 plugin **不需要**（§4.3 故事：multiproc 是与 scheduler 正交的隔离旋钮，测量时关掉无损科学性）。
- **V1 controller**：`llm.get_metrics()` → `vllm:num_requests_running` gauge 跑通（峰值满并发；需 `disable_log_stats=False`）。
- **★ F1 在 V1 上成立且更强**：r75/r0 c1→c12 bonus V1 **+0.65** vs V0 +0.49；**c12/r75 = 1.86×**（V0 1.75×）。机制（KV/并发放大）对 V1 scheduler 鲁棒，高并发下更强。低并发（c1）V1 略低（chunked prefill 摊销 prefill）。

## 立即下一步 —— P1（Qwen3-VL-8B）
1. Qwen3-VL-8B 的 V1 compressor hook：原生 2×2 MLP merger → pruner 在 merger 下游，r 相对 post-merger token 数（动态分辨率 274~2300/图）。
2. F1/F2/F3 复测 + load-adaptive controller on Qwen3-VL。假设：动态分辨率→vision 占比更高→pruning 价值更大。

## 之后
P2 c≥64 + p50/p99 TTFT + goodput｜P3 VisionZip 类横向对比｜P4 重写（§4.3 V1-migration-as-contribution、§2.3 gap 改写、加 Qwen3-VL 列）。

## 关键约束
- 算力 1× A40 46GB 串行；v2 serving env = `qwen3vl_clean`(V1)；vtc_serve(V0)/fastv 留存。
- 提交以用户本人名义，禁 AI 署名。每步前 web 核实版本+novelty-threat 监控。
- 升级找人：凭据 / >6GPU·h / claim 推翻 / V1 不可行 / 投稿前。

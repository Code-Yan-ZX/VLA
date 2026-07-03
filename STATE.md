# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md** + **next_plan.md**
> 最近更新：2026-07-03

## 当前阶段
**v2 进行中（V1 引擎迁移）。V1 命门 GREEN。P0 = V1 压缩器集成 + 复现 Table A。**

## v2 转向（v1 论文待大改，不投）
v1 致命伤：V0 废弃路径 + LLaVA-1.5 过时 + toy 规模 + 0/37 脆弱 + 方法 n=500 null。→ 重做：**V1 引擎 + 现代模型 + 真实 serving 规模**。

## ★ v2 kickoff 结论（notes/v2_ecosystem_assessment.md, commit 597bf72）
- **V1 可行**：vLLM **0.19.0+cu128** 在 driver560/cu12.8/A40 跑通 V1（LLaVA-1.5-7B 3.2s、Qwen3-VL-8B 20.2s init）。env = `qwen3vl_clean`（vllm0.19.0/torch2.10+cu128）；vtc_serve(V0) 留 rollback。注：0.10.2 默认就是 V1，是我们 `serve_bench.py:41` 强制 V0 才走的老路径。
- **V1 controller 信号**：`llm.get_metrics()`→Prometheus（`vllm:num_requests_running`、`vllm:gpu_cache_usage_perc`），替代 V0 失效的 in-process scheduler 读取。
- **现代模型**：**Qwen3-VL-8B-Instruct**（A40 ~16GB bf16，已验证）。原生 2×2 MLP merger + 动态分辨率 → 变长 token（274~2300/图）→ F1/F2/F3 须复测，pruning 价值更大。
- **novelty 仍成立（0/N）**：RTP-LLM 是引擎非压缩机测量；vLLM-Omni 无 token pruning。**补相关工作**：DeepSeek-OCR(2510.18234, 互补)、EarlyTom(2605.30010, TTFT 拆解重叠 F2 须区分)、ADSC(2602.12618)。

## 立即下一步 —— P0（Dev subagent）
1. **V1 压缩器集成**（最硬）：V1 模型在 EngineCore 子进程 → V0 的 in-process projector hook 失效。找 V1 机制（plugin/collector/mm-processor）在子进程内挂压缩器。这块本身是论文新工程贡献（§4.3 重写）。
2. **V1 controller**：`get_metrics()` 读取 num_running + gpu_cache_usage。
3. **复现 Table A**（V1 + LLaVA-1.5-7B, 并发×prune）→ **F1（e2e>prefill）在 V1 的 prefill/decode 分离调度下是否仍成立**？（V1 改变调度，F1 可能变——这是关键科学问题）

## 之后
P1 Qwen3-VL-8B F1/F2/F3 复测｜P2 c≥64 + p50/p99 TTFT + goodput｜P3 VisionZip 类横向对比｜P4 重写（§4.3 V1-migration-as-contribution、§2.3 gap 改写、加 Qwen3-VL 列）。

## 关键约束
- 算力 1× A40 46GB 串行；v2 serving env = `qwen3vl_clean`(V1)；vtc_serve(V0)/fastv 留存。
- 提交以用户本人名义，禁 AI 署名。每步前 web 核实版本+novelty-threat 监控。
- 升级找人：凭据 / >6GPU·h / claim 推翻 / V1 不可行 / 投稿前。

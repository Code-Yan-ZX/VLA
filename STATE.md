# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md** + **next_plan.md**
> 最近更新：2026-07-03

## 当前阶段
**v2 P2 DONE（真实 serving 规模 c≥64 + p50/p99 TTFT + goodput）。★ 放大不饱和：r75/r0 = 1.19×(c1)→1.96×(c16)→**2.22×(c64)**（P0 c12 1.86× 非天花板）。c64 r0 KV-bound（peak 41GB≈预算）→ c128 infeasible，c64 是单-A40 天花板（正是压缩价值最大处）。r75 在 c64 **strictly dominate** r0：2.22×吞吐 + 2.84×更低 p99-TTFT（无 tradeoff，纯 win）。goodput：紧 SLO(500ms/1s) 全 0（并发 prefill floor~3s），现实 SLO(TTFT≤5s) r75=13.7 vs r0=1.8 req/s(7.4×)。Qwen3-VL c64 r50/r0=1.34×（c12 1.16×）趋势一致。下一步 = P4 重写（加 P2 scale 表 + goodput Pareto + Qwen3-VL 列）。**

## v2 转向（v1 论文待大改，不投）
v1 致命伤：V0 废弃路径 + LLaVA-1.5 过时 + toy 规模 + 0/37 脆弱 + 方法 n=500 null。→ 重做：**V1 引擎 + 现代模型 + 真实 serving 规模**。

## ★ v2 kickoff 结论（notes/v2_ecosystem_assessment.md, commit 597bf72）
- **V1 可行**：vLLM **0.19.0+cu128** 在 driver560/cu12.8/A40 跑通 V1（LLaVA-1.5-7B 3.2s、Qwen3-VL-8B 20.2s init）。env = `qwen3vl_clean`（vllm0.19.0/torch2.10+cu128）；vtc_serve(V0) 留 rollback。注：0.10.2 默认就是 V1，是我们 `serve_bench.py:41` 强制 V0 才走的老路径。
- **V1 controller 信号**：`llm.get_metrics()`→Prometheus（`vllm:num_requests_running`、`vllm:gpu_cache_usage_perc`），替代 V0 失效的 in-process scheduler 读取。
- **现代模型**：**Qwen3-VL-8B-Instruct**（A40 ~16GB bf16，已验证）。原生 2×2 MLP merger + 动态分辨率 → 变长 token（274~2300/图）→ F1/F2/F3 须复测，pruning 价值更大。
- **novelty 仍成立（0/N）**：RTP-LLM 是引擎非压缩机测量；vLLM-Omni 无 token pruning。**补相关工作**：DeepSeek-OCR(2510.18234, 互补)、EarlyTom(2605.30010, TTFT 拆解重叠 F2 须区分)、ADSC(2602.12618)。

## P0 结果（commit bff6871, notes/v2_p0_v1_tableA.md）
V1 in-process（`VLLM_ENABLE_V1_MULTIPROCESSING=0`）→ V0 projector hook + `get_num_image_tokens` patch 直接复用；`llm.get_metrics()` controller 跑通。**★ F1 在 V1 上更强**：r75/r0 c1→c12 bonus V1 **+0.65** vs V0 +0.49；**c12/r75 = 1.86×**（V0 1.75×）。

## P1 结果（Qwen3-VL-8B, notes/v2_p1_qwen3vl.md）
新集成路径（`_get_prompt_updates` placeholder patch + `_process_image_input` post-split wrap + vLLM 内置 M-RoPE recompute）。**★ F1 衰减但方向成立**：c12/r75=**1.29×**（LLaVA 1.86×），bonus +0.21 vs +0.65。F2 vision+merger 仅占 TTFT 10%（不可降）。F3 成立（TextVQA r50 1.16× > GQA 1.06×，但 acc 0.77→0.48）。**★ merger 与 pruner 是 SUBSTITUTES**（都压 post-merger token）→ 边际递减。

## P2 结果（commit fe8f3e9, notes/v2_p2_scale.md）
- **c64 可行但 KV-bound**（r0 peak KV 41.0GB ≈ 0.90×46GB 预算）；c128 r0 infeasible（~80GB KV）。c64 = 单-A40 serving 天花板（压缩价值最大处）。
- **★ 放大不饱和（KEEPS GROWING）**：r75/r0 = 1.19×(c1) → 1.53×(c4) → 1.96×(c16) → **2.22×(c64)**（增量 +0.34/+0.43/+0.26，减速但仍升）。r0 吞吐 c16→c64 仅 +12%（撞顶），r75 +26%（压缩 lift 天花板）。
- **p99 tail：pruning 大幅降尾**。c64 p99-TTFT r0=18.4s→r75=6.5s（2.84×）；p99-e2e 19.2→7.2s。并发下 pruning 是 tail-reducer 不只是 throughput-booster。
- **★ Pareto：r75 strictly dominate r0 @c64**（2.22×吞吐 + 2.84×更低 p99-TTFT，无 tradeoff）。goodput：紧 SLO 全 0（并发 prefill floor~3s）；现实 SLO TTFT≤5s r75=13.7 vs r0=1.8 req/s(7.4×)，e2e≤8s r75=20.4 vs r0=0.9(23×)。
- **Qwen3-VL c64 泛化**：r50/r0 = 1.06×(c1)→1.16×(c12)→**1.34×(c64)**，c12 后仍升（衰减但同向，native merger 压 token 之故）。
- 工程贡献：`serve_bench.py --batch-submit` 改 streaming add_request+step，每请求 TTFT(vLLM metrics.first_token_latency)+e2e(自计 perf_counter)；加 percentile(p50/p99)+goodput(SLO)。c64/c128 KV 测算 + 53k token budget 记录在案。

## 立即下一步
P2 完成 → **P4 重写**：§2 加 Qwen3-VL 列（architecture-conditioned F1）+ P2 scale 表（c64/r75 2.22×，c≤12→c64 扩展）；§3 goodput Pareto（r75 strictly dominate）作为 deployment-relevant 主图；§4.3 V1-migration + Qwen3-VL processor + P2 streaming-metrics 路径作工程贡献。补 query-aware selector(`clip_query`) 修 r75 acc 0.475（v2 method-design）。可选 open-loop goodput（Poisson）补 P2-extension。

## 关键约束
- 算力 1× A40 46GB 串行；v2 serving env = `qwen3vl_clean`(V1)；vtc_serve(V0)/fastv 留存。
- 提交以用户本人名义，禁 AI 署名。每步前 web 核实版本+novelty-threat 监控。
- 升级找人：凭据 / >6GPU·h / claim 推翻 / V1 不可行 / 投稿前。

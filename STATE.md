# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 详见 ORCHESTRATION.md + notes/elasticvis_design.md
> 最近更新：2026-07-13 · **用户 pivot：ElasticVis allocator 终判 negative → 转 EV-VAR（方差外部性信号验证）** · 计划全文 ~/.claude/plans/warm-tinkering-bird.md

## ★ 用户决策（2026-07-13）：停止把 ElasticVis allocator 当正向 claim
EV-1e 已 GPU 终判 negative（连续批处理下 per-request 视觉 token 预算不提升 goodput@SLO，架构性）。但负结果暴露了**批内延迟外部性**机制（高 k prefill 拖慢同批 decode）。用户指令：低成本受控实验验证该外部性能否转化为**超越 {Σk、SLO slack、token-budget batching} 的预测信号**；只有信号成立 **且** compatible co-batching 在真实 GPU 胜过 {FIFO、deadline-only、token-budget、chunked-prefill、最佳固定压缩}，才作为新方法主线。

## ★ 当前方向：EV-VAR（分阶段 go/no-go，plan: warm-tinkering-bird.md）
**H_var**：连续批处理下 per-request 延迟含"批组成"驱动分量，iso-ΣK 下随方差/兼容性变化、不被三混淆量解释、chunked-prefill 后仍有残余。**铁律**：旧 sim（independent-slot）按构造假设掉干扰 → 只作 null（residual=real−sim），不作正向证据；既有 trace 无 batch 组成 → 必须新跑带 step 级 logging 的受控实验。
- **Stage 0**（去风险，进行中）：serve_bench 加 `--log-step-composition`（monkey-patch vllm0.19 `Scheduler.schedule` 存 `SchedulerOutput`，读 `num_scheduled_tokens`/`scheduled_new_reqs` → 成员/相位/Σk/var/wallclock）+ `--chunked-prefill {on,off}` + force `enforce_eager=True`。env `qwen3vl_clean`(vllm0.19 V1-only)，3 图 smoke 验证 primitive。
- **Stage 1**（微基准 gate-before-gate）：batch_submit + `--ev-debug-k` 构造 iso-ΣK 受控 batch；ΣK×{同构/温和/双峰极端}×{prefill-only/prefill+decode}×{chunked ON/OFF}。→GO2 若 composition 效应 >10% & p<0.05（chunked ON）。
- **Stage 2**（预测回归）：openloop 多 policy{FIFO/deadline/token-budget/random} sweep + 新 step 级 logging；M0(~Σk+slack+tokenbudget) vs M1(+var/compatibility) 嵌套回归 + sim null-residual 交叉验证。→GO3 若 M1≫M0 & 效应>噪声。
- **Stage 3 GATE**（仅 1+2 GO，>6GPU·h 找人）：compatible co-batching 须胜全部 5 baseline（mean±std ≥3 repeats）。

## 评测制度（沿用）
open-loop 变载到达为主 + 混合-SLO 为辅。baseline：fixed-{r0,r25,r50,r75}+控制器+oracle。

## v2 资产（substrate，全现成）
framework `src/serve_bench.py`(V1 c64 goodput+openloop+per-req k via ev_state["cur_k"])+`load_controller.py`+`compressors.py`+`src/elasticvis/`；数据 eval/subsets/{gqa,textvqa,mme,mmbench,scienceqa}_{200,500}.jsonl + runs/v2_p*/；模型 LLaVA-1.5-7B(runs/models/)+Qwen3-VL-8B；env `qwen3vl_clean`(V1)。v2 论文 drafts/paper_v2.md 可独立投。

## 已完成（背景）
P0-P4 全完成；ElasticVis EV-0..EV-1e（allocator 终判 negative，已得：门控刻画(5 benchmark)/placeholder-shrink 集成/batch-interference 发现）。详见 DECISIONS.md。

## 关键约束
- 算力 1× A40 46GB 串行；env `qwen3vl_clean`(V1, vllm0.19)。
- 提交以用户本人名义，禁 AI 署名。每步 web 核实版本+novelty 监控。
- 升级找人：凭据 / >6GPU·h / claim 推翻 / 投稿前。

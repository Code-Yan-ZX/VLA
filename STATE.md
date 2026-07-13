# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 详见 ORCHESTRATION.md + notes/elasticvis_design.md
> 最近更新：2026-07-13 · **EV-VAR Stage 1 ❌ NO-GO（variance 信号被 GPU 证伪）→ variance 主线关闭，待 user 定** · 计划 ~/.claude/plans/warm-tinkering-bird.md

## ★ 用户决策（2026-07-13）：停止把 ElasticVis allocator 当正向 claim
EV-1e 已 GPU 终判 negative（连续批处理下 per-request 视觉 token 预算不提升 goodput@SLO，架构性）。但负结果暴露了**批内延迟外部性**机制（高 k prefill 拖慢同批 decode）。用户指令：低成本受控实验验证该外部性能否转化为**超越 {Σk、SLO slack、token-budget batching} 的预测信号**；只有信号成立 **且** compatible co-batching 在真实 GPU 胜过 {FIFO、deadline-only、token-budget、chunked-prefill、最佳固定压缩}，才作为新方法主线。

## ★ 状态：EV-VAR Stage 1 ❌ NO-GO（high conf，2026-07-13）→ variance 主线关闭，待 user 定
H_var（视觉 token 方差有"批内延迟外部性"预测信号，超 Σk/SLO-slack/token-budget）被 GPU step-log 回归证伪：4 cell（homo/bimodal k × chunked on/off，**同 seed 同到达**，R=8，n=300，2048 steps）—— `{var_k,max_k}` 零预测力（**F=0.18 p=0.84**），chunked-ON 残余 **F=0.20 p=0.82**，max_k 系数反方向；EV-1e"外部性"=纯 **sum_k（总 token）** 效应，chunked-prefill 封顶 sum_k/step→wallclock 由 n_prefill/n_decode 驱动，与 k 方差无关，无信号供 scheduler 榨取。**按预注册规则 + 用户指令"信号成立才继续"，停 variance 线，不进 Stage 2/3（省 GPU）。** Stage 0（serve_bench `--log-step-composition`，commit 14c3645）primitive 可用、留作工具。
**待 user 定**：A 折入 v2 negative-finding 节｜B 独立 negative-result 贡献｜C 回 v2 测量论文为主线｜D 纯 deadline-routing 当**独立新假设**先 novelty 扫描（注意：与 SLOs-Serve/AdaServe 重叠，且 EV-1e 的 k-assignment 版已 GPU 输 → likely non-viable）。详见 DECISIONS.md 2026-07-13 Stage 1 条 + plan warm-tinkering-bird.md。

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

# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 详见 ORCHESTRATION.md
> 最近更新：2026-07-14 · **用户重锚：要创新型论文（新方法），非测量论文；方向 = pre-merger pruning（Qwen3-VL 原生 merger 之前剪枝）· 验证中**

## ★ 当前主线：创新型论文 — pre-merger pruning（gated 验证中）
用户 2026-07-13/14 明确：要**创新型（新方法）论文**，非测量论文（v2 降 fallback；ERA 又打破其 0/37 头条）。charter 原目标即"新方法"。**方向 = pre-merger pruning**：在 Qwen3-VL 原生 2×2 merger **之前**剪视觉 token，使 pruner 与 merger 互补（非替代）。动机 F4：post-hoc pruning 在 Qwen3-VL 衰减（1.08×→1.29× vs LLaVA 1.19×→2.22×）因 merger+pruner 替代；pre-merger 可恢复放大。**正交于全部失败尝试**（非 selector、非 scheduling）。
**Gate（EV-VAR 纪律，cheap-first）**：① novelty scan（pre-merger-on-native-merger 是否未被做）② feasibility（Qwen3-VL+vLLM 能否 hook pre-merger 不 fork）③ GPU go/no-go（pre-merger vs post-merger on served 吞吐+精度）。三通过才写论文；失败则 ~1 GPU·h 止损。两 agent 进行中。

## ★ method search 历史（均 GPU 证伪，作 negative 库）
selector 三连败（CLS/LLM-cosine/CLIP，OCR 失败）｜load-adaptive controller（n=500 null）｜ElasticVis allocator（EV-1e 负）｜EV-VAR variance（Stage 1 负，{var_k,max_k} F=0.18 p=0.84）。**教训**：boundary TF selector 打不过 intra-LLM OCR；scheduling-based 被 total-token bound。pre-merger = 新维度（compression-architecture，非 selector/scheduling）。

## ★ v2 测量论文（fallback，drafts/paper_v2.md，ERA 后降级）
v2 = 0/37 served-throughput framework（**ERA arXiv:2606.31982 打破头条**：在 vLLM 内测了 served req/s）。salvage = sharpen 到 continuous-batching/goodput/p99/TTFT。已做：§5.9 negative 节 + 5 图 + RTP-LLM 修（commit f3449f6）。留作 fallback/支撑，非主线。

## 资产 + 约束
framework `src/serve_bench.py`(V1 c64 goodput+openloop+per-req k+`--log-step-composition`)+`compressors.py`+`src/elasticvis/`；数据 eval/subsets/* + runs/v2_p*/；模型 LLaVA-1.5-7B + Qwen3-VL-8B；env `qwen3vl_clean`(vllm0.19 V1)。算力 1× A40 46GB 串行。提交用户名义禁 AI 署名。升级找人：凭据/>6GPU·h/claim推翻/投稿前。详见 DECISIONS.md。

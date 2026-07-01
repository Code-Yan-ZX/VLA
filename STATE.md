# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-01

## 当前阶段
**P2 进行中**（方法设计 + go/no-go 闸门）—— P1 已完成

## P1 已完成（2026-07-01，commit de03843）
- `notes/lit-survey.md`（23 法 survey，arXiv 核验；**0/23 在 serving engine 内测吞吐**）
- `notes/positioning.md`：**Gap A = serving-engine-aware 压缩**（方法+测量框定）；基座 LLaVA-1.5-7B→Qwen2.5-VL-7B
- DECISIONS：gap/基座 + **go/no-go kill-switch**（P2 第一里程碑）+ novelty 复核闸门

## 立即下一步 —— P2 Step1（Dev subagent 已派，后台）
1. **novelty 复核**（cheap 先跑）：last-6mo 复扫 vLLM/SGLang/lmdeploy/TRT-LLM 集成压缩+吞吐论文 → Gap A 仍开放?
2. **建 `vtc_serve` env**（vLLM 自带兼容 transformers/torch；与 vtc 隔离）→ 下 LLaVA-1.5-7B（公开，无凭据）→ vLLM smoke。
3. **写 `notes/method-design.md`**：go/no-go 探针设计（边界级 training-free 压缩器 × vLLM mm_processor hook × {0,25,50,75}% × GQA+TextVQA × tok/s,req/s,TTFT,KV-cache）+ GO/NO-GO 阈值(positioning) + 方法假设骨架 + 基线(FastV@fastv env)。
4. **实现 `src/`** 探针 harness（压缩器模块 + serve_bench.py）；CPU 自测，**不跑 GPU job**。
5. 交回 `notes/p2_probe_jobs.json` → Main 入 configs/queue.json 串行跑 GPU 探针。

## P2 之后（go/no-go 结果定方向）
- **GO**(≥1.5×prefill & ≥1.2×e2e req/s @ ≤2%掉点) → 设计 serving-aware 方法 → P3 全基准。
- **NO-GO**(<1.2×e2e) → 转 negative-result paper 或 Gap D；**触发 charter §6 升级找人**。
细节见 ORCHESTRATION.md §4 P2/P3。

## 关键约束
- 算力 1× A40 46GB 串行；GPU job 一律走 `scripts/queue`。
- env：`vtc`(方法 dev,tf5.x) / `vtc_serve`(vLLM serving) / `fastv`(FastV 基线复现)。
- 升级找人：凭据 / >6GPU·h / claim 被推翻(NO-GO) / 投稿前。

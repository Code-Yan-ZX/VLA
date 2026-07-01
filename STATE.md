# STATE.md — 当前项目状态（主窗口维护，保持 ≤30 行）

> 项目：VLM 视觉 token 压缩 · 目标 Q1/Q2 SCI · 3–6 月 · 详见 **ORCHESTRATION.md**
> 最近更新：2026-07-01

## 当前阶段
**P2 进行中** —— step1 完成，正在建探针子集 → 跑 go/no-go 探针

## P2 step1 已完成（commit 626de21，Dev subagent）
- novelty 复核：**Gap A 仍 OPEN，无 blocker**（AgilePruner/VisionTrim/PRUNESID 挤占 accuracy/FLOPs 组合研究；serving-throughput 仍是空白；EffiVLM-BENCH 仅离线 latency）
- env `vtc_serve`：**vLLM 0.10.2 + torch2.8+cu128 + tf4.55.2**（锁版，driver560 跑通；详见 DECISIONS）。LLaVA-1.5-7B 已下到 `runs/models/`，vLLM 烟测 2.4s/答。
- 探针设计：`ClsAttnSelector`(CLS-attn top-k) × hook `LlavaMultiModalProjector.forward` 输出（post-projector/pre-LLM）。`src/compressors.py`+`serve_bench.py` CPU 自测过。
- `notes/method-design.md` + `notes/p2_probe_jobs.json`（7 job，~2.5 GPU·h）已就绪。

## 立即下一步
1. **建探针子集**（Dev subagent 进行中）：`eval/subsets/gqa_200.jsonl` + `textvqa_200.jsonl`（200 例×2，seed=0，格式 `{"id,image,question,gt,choices?}`）；最小下载（仅取所需图，不入全量 VG）。
2. 子集就绪 → 合并 `p2_probe_jobs.json` 进 `configs/queue.json` → **串行跑 7 job 探针**（背景 driver）。
3. 读 metrics → **GO/NO-GO 判定**：
   - GO = GQA@r50 ≥1.5×prefill & ≥1.2×e2e req/s & ≤2%掉点；TextVQA@r50 ≤5%掉点 → 进 serving-aware 方法设计。
   - NO-GO = r75 仍 <1.2×e2e → 转 negative-result paper 或 Gap D；**触发 §6 升级找人**。

## 排队（probe 运行时并行）
- Lit subagent：把你提供的 ~10 篇论文补进 `lit-survey.md` §2 对比表 + 核 arXiv ID（你提示"少数凭记忆填，投稿前核"）。

## 关键约束
- 算力 1× A40 46GB 串行；GPU job 走 `scripts/queue`。
- env：`vtc`(方法 dev) / `vtc_serve`(vLLM 0.10.2 serving) / `fastv`(FastV 基线，需 clone 上游)。
- 提交以用户本人名义，**禁 AI 署名/Co-Authored-By**。
- 升级找人：凭据 / >6GPU·h / claim 被推翻(NO-GO) / 投稿前。

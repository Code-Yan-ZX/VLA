# P2 Method-v1 Comparison — proxy vs v1(true CLS-attn) vs FastV (n=200, LLaVA-1.5-7B)

> Goal: test whether replacing the probe's proxy selector (hidden-state deviation) with REAL vision-tower CLS attention improves accuracy at iso-throughput, and anchor against FastV (intra-LLM, the canonical TF baseline). Outcome: **v1-trueCLS is NOT the method winner** → method redirects to query-aware selection (see bottom).

## Accuracy (re-scored with fixed GQA/TextVQA scorer)
| benchmark | method | r=0 (576) | r=0.25 (432) | r=0.50 (288) | r=0.75 (144) |
|---|---|---|---|---|---|
| **GQA** | proxy (hidden-state) | 0.585 | 0.555 | **0.565** | 0.470 |
|  | v1 (true CLS-attn) | 0.585 | **0.570** | 0.545 | 0.450 |
|  | FastV (intra-LLM L2) | 0.570 | — | 0.535 | **0.515** |
| **TextVQA** | proxy (hidden-state) | 0.555 | — | **0.530** | 0.440 |
|  | v1 (true CLS-attn) | 0.555 | — | 0.445 | 0.275 |
|  | FastV (intra-LLM L2) | — | — | **0.555** | — |

## E2E served req/s speedup (vLLM V0; FastV is accuracy-only — no serving engine)
| benchmark | method | r=0.25 | r=0.50 | r=0.75 |
|---|---|---|---|---|
| **GQA** | proxy | 1.17× | **1.33×** | **1.43×** |
|  | v1 (true CLS-attn) | 1.14× | 1.04× | 1.31× |
| **TextVQA** | proxy | — | **1.16×** | 1.16× |
|  | v1 (true CLS-attn) | — | 0.99× | 1.10× |

## Findings (3, all method-shaping)
1. **Vision-tower CLS attention is the WRONG selector for OCR/task-accuracy.** v1-trueCLS catastrophically degrades TextVQA (r50 0.445 vs proxy 0.530; r75 0.275 vs 0.440) and is not better on GQA (r50 0.545 ≤ proxy 0.565; r75 0.450 < 0.470). Cause: the vision-tower [CLS] head attends to coarse salient objects/scenes, NOT fine-grained text/task-relevant regions → high prune drops the text patches. **A task-relevant selector is required.**
2. **FastV (intra-LLM attention-rank on the task token) is the most accuracy-robust**, especially at extreme prune / OCR: GQA r75 0.515 (best), TextVQA r50 0.555 (best). Its selection is conditioned on the actual question (the last decode token's attention), so it keeps task-relevant patches. **But FastV is intra-LLM → not integrable into vLLM** (V1 subprocess + CUDA-graph-locked decode), so it can only be an accuracy anchor, not our serving method.
3. **The true-CLS capture adds throughput overhead** (manual softmax per request) — v1 e2e speedup lags proxy at every ratio (GQA r50 1.04× vs 1.33×; TextVQA r50 0.99× vs 1.16×). The proxy probe's speedup numbers remain the clean throughput reference. A v2 selector must be cheap (no per-layer attention materialization).

## Implication: the serving-engine constraint pins us to BOUNDARY pruning
- Intra-LLM pruning (FastV-style, the accuracy winner) **cannot run in vLLM** → any serving-engine method must prune at the boundary (projector output / processor), as the probe did.
- But boundary CLS-attn (vision saliency) fails OCR. **⇒ The open problem: a BOUNDARY selector that is QUERY-AWARE** (uses the question text to select visual patches before the LLM), so it is both vLLM-integrable AND task/OCR-relevant. (The question is available at preprocessing time, so boundary query-aware selection is feasible.)

## ★ Method redirect → v2 (next Dev task)
1. **Query-aware boundary selector**: score visual patches by relevance to the question (text-embedding ↔ patch-embedding similarity, or a tiny cross-attention), select top-k at the projector output. Keep contiguous-compaction + placeholder-shrink (working). Target: recover FastV-level OCR accuracy at the boundary.
2. **Early-prune** (probe finding #2): move selection into/after the vision tower so the encoder also does less work → larger prefill + e2e speedup. (V0 vision tower is always eager → mid-encoder prune is feasible, no CUDA-graph fight.)
3. **KV-cache/batch-aware budget** (probe finding #1, v2 sketch): prune rate responds to engine concurrency to amplify the e2e>prefill serving win.
4. **Served throughput is the evaluation differentiator** (0/37 papers measure it) — keep reporting tok/s, req/s, TTFT.

Provisional GO unchanged: the core claim (compression → served speedup) was established by the proxy probe and is selector-independent. v1 refined the question, it did not overturn it.

## Artifacts (gitignored)
- proxy: `runs/p2_probe/*` · v1: `runs/p2_probe_v1/*` · FastV: `runs/fastv_baseline/*` (raw answers saved for re-scoring)

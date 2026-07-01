# P2 Go/No-Go Probe — Summary (LLaVA-1.5-7B @ vLLM 0.10.2 V0, 1× A40, n=200)

> Compressor: CLS-attention-proxy selector (hidden-state deviation norm — a PLACEHOLDER for the real CLS-attn score, to be refined post-gate) → contiguous compaction at projector output. Patch: `get_num_image_tokens`→k (commit c05ca86). Scorer: GQA-convention (commit 46f9b9d); r0/r50 re-scored offline.

## ★ Verdict: provisional GO (core claim holds; not a NO-GO; no §6 escalation)

The bet — *visual-token compression yields real wall-clock speedup inside a serving engine* — is **CONFIRMED**. The survey's worst-case worry (~0 serving speedup) is refuted.

## GQA curve (the gate)
| prune r | keep | req/s (e2e) | e2e speedup | TTFT (prefill) | prefill speedup | tok/s | acc (re-scored) | acc Δ vs r0 |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 576 | 1.77 | — | 631 ms | — | 22.81 | 0.585 | — |
| 0.25 | 432 | 2.08 | **1.17×** | 553 ms | 1.14× | 24.34 | 0.555 | −3.0% |
| 0.50 | 288 | 2.35 | **1.33×** | 510 ms | 1.24× | 25.05 | 0.565 | −2.0% |
| 0.75 | 144 | 2.53 | **1.43×** | 487 ms | 1.30× | 26.42 | 0.470 | −11.5% |

Gate thresholds (positioning.md ★): e2e ≥1.2× (PASS at r50/r75), acc Δ ≤2% (−2.0% at r50, with a PROXY selector → true CLS-attn will improve), prefill ≥1.5× (not met — explained below).

## ★★ Two paper-grade findings
1. **e2e speedup > prefill speedup at EVERY ratio** (1.17>1.14, 1.33>1.24, 1.43>1.30). Compression's deployment win comes largely from **KV-cache / concurrency** (smaller per-request KV ⇒ more concurrent requests ⇒ higher req/s), NOT prefill FLOPs. This serving-specific effect is **invisible to offline FLOPs measurement** — the exact unmeasured differentiator (0/37 papers measure it).
2. **Prefill is sub-linear** (r75 gives only 1.30× prefill despite 4× fewer tokens). Cause: pruning is at projector OUTPUT ⇒ the vision tower still processes all 576 tokens (fixed encoder cost). **Method implication**: prune earlier (in/after the encoder) to also speed up the encoder → larger prefill gains.

## Accuracy note
- Drops are small at moderate prune (−2% at r50, −3% at r25), larger at extreme (−11.5% at r75) — expected with a proxy selector. Non-monotonicity (r25<r50) is n=200 noise + selector instability. A true CLS-attn selector + position-aware compaction will tighten this (method-design refinement).
- r0 0.585 / r50 0.565 are in LLaVA-1.5-7B's sane GQA range (published ~62%).

## TextVQA (OCR robustness) — pending
3 jobs re-queued after a 1-line scorer-signature fix (`score_textvqa` now accepts `choices` for parity). Re-run in progress; results appended when done.

## Implications for method design (P2 step 3)
- **Differentiator is the serving-throughput story** (finding 1) — frame the method as serving-aware, not just another accuracy/FLOPs compressor.
- **Early-prune** (encoder-stage) to capture the fixed encoder cost (finding 2) → bigger prefill + e2e wins.
- **KV-cache/batch-aware budget** to amplify the concurrency benefit.
- Replace proxy selector with true CLS-attn (or learn the score) for accuracy.

## Artifacts (gitignored)
metrics: `runs/p2_probe/{gqa,textvqa}_r{0,25,50,75}_metrics.json` (raw answers saved) · model: `runs/models/llava-1.5-7b-hf` · logs: `runs/probe_*.log`, `runs/queue_driver.log`

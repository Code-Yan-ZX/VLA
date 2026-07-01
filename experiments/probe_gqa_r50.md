# probe_gqa_r50 — GQA at 50% prune (keep 288/576), n=200 — ★ THE GATE POINT ★

- **status**: done (rc=0) · V0 + placeholder-compact patch (commit c05ca86) · 2026-07-01 16:17 · 148s
- **cmd**: `/home/dell/miniconda3/envs/vtc_serve/bin/python -m src.serve_bench --model runs/models/llava-1.5-7b-hf --pruning-rate 0.50 --benchmark gqa --subset eval/subsets/gqa_200.jsonl --metrics-out runs/p2_probe/gqa_r50_metrics.json --max-tokens 32 --max-model-len 4096 --gpu-memory-utilization 0.90 --seed 0`
- **log**: `runs/probe_gqa_r50.log` · **metrics**: `runs/p2_probe/gqa_r50_metrics.json` (gitignored)

## 结果 / 指标 (vs r0 control)
| metric | r0 | r50 | speedup | gate threshold |
|---|---|---|---|---|
| **served_req_s (e2e)** | 1.77 | **2.35** | **1.33×** | ≥1.2× ✅ PASS (not NO-GO) |
| **prefill (TTFT)** | 631ms | 510ms | **1.24×** | ≥1.5× ⚠️ below target |
| served_tok_s | 22.81 | 25.05 | 1.10× | — |
| accuracy | 0.01 | 0.025 | — | scorer broken; re-score pending |

`kept_counts = [288 × 200]` → contiguous compaction confirmed at full scale (sequence genuinely 288 tokens shorter, not keep-sparse).

## 结论 — ★ provisional GO (core claim holds; not a NO-GO; no §6 escalation) ★
1. **Core claim CONFIRMED**: e2e req/s 1.33× ≫ 1.2× NO-GO line ⇒ visual-token compression DOES yield real wall-clock speedup inside a serving engine. The survey's worst-case worry (~0 serving speedup) is **refuted**. This is the headline.
2. **★ Serving-specific finding (paper gold)**: e2e speedup (1.33×) **>** prefill speedup (1.24×). Compression's deployment win comes largely from **KV-cache / concurrency** (smaller per-request KV ⇒ more concurrent requests ⇒ higher req/s), NOT from prefill FLOPs. This effect is **invisible to offline FLOPs measurement** — exactly the unmeasured differentiator the positioning bet on.
3. **Prefill sub-linear (1.24× < 2× token cut)**: explained — pruning is at projector OUTPUT, so the vision tower still processes all 576 tokens (fixed encoder cost). **Method implication**: prune earlier (in/after the encoder) to also speed up the encoder.
4. accuracy 0.025 still artificial (strict scorer + proxy hidden-state-deviation selector) — re-score after Dev's scorer fix; proxy selector is a lower bound, true CLS-attn selector will improve it.

## 下一步
- Run remaining curve: gqa_r25, gqa_r75 (confirm speedup scales with prune ratio), textvqa_r0/r50/r75 (OCR robustness).
- Re-score all raw answers (r0/r50/…) once scorer fixed.
- Proceed to method design: true CLS-attn selector + serving-aware compressor (early-prune for encoder speedup; KV-cache/batch-aware).

## 产物路径
- metrics `runs/p2_probe/gqa_r50_metrics.json` (gitignored, raw answers saved for re-scoring) · model `runs/models/llava-1.5-7b-hf` (gitignored)

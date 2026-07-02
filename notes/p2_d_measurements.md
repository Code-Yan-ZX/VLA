# P2 Method-D Scoping Measurements (M1 + M2)

> Decides the scope of the serving-aware method D-on-proxy BEFORE implementing it.
> - **M1** (prefill breakdown): is mid-encoder / early-ViT prune worth the surgery?
> - **M2** (concurrency × prune-rate): is the load-adaptive budget justified?
> Infra: `src/serve_bench.py` (vLLM V0, LLaVA-1.5-7B, GQA). Code committed at `028f472`.

---

## M1 — prefill cost breakdown (vision tower vs LLM prefill)

**Setup**: GQA r0 (no pruning), n=100, max-tokens=1 (TTFT ≈ pure prefill, no decode),
serial mode (per-request TTFT), proxy selector. Vision-tower forward timed via
pre/post hooks on `vision_tower.vision_model` (full CLIPVisionTransformer =
embeddings + 24 encoder layers); projector forward timed separately; LLM prefill
estimated as TTFT − vision_tower − projector.

| phase | mean (ms) | fraction of TTFT |
|---|---|---|
| vision tower (CLIP ViT-L/14@336, 576 patches) | **12.7** | **6.6%** |
| projector (linear, boundary) | 0.16 | 0.1% |
| LLM prefill (estimated remainder) | 180.3 ± 8.2 | 93.3% |
| **TTFT (total)** | **193.2** | 100% |

Artifacts: `runs/p2_d/m1_r0_n100.json`, `runs/p2_d/m1.log`.

### Verdict M1: **SKIP early-prune / mid-encoder ViT surgery.**
Vision tower = **6.6% of prefill** — well below the 10% threshold. The vision
tower is a fixed cost but a SMALL one; even dropping it entirely (impossible
without losing all visual signal) caps the extra prefill win at ~7%. The real
prefill cost is the **LLM prefill over the image+text sequence (93%)**, which
boundary pruning already shortens (fewer image-token placeholders → shorter LLM
sequence → proportionally less LLM prefill). **⇒ D should NOT invest in mid-
encoder pruning (high surgery, low ceiling). The win is all LLM-sequence-
shortening, already captured at the boundary by the proxy selector.**

This quantitatively confirms Finding #2 (prefill sub-linear because vision tower
is fixed) AND shows the fixed part is too small to chase: the sub-linearity is
mild (a 50% prune removes ~50% of LLM prefill but 0% of the 6.6% vision cost →
prefill speedup ≈ 0.93 × sequence-shortening factor, only slightly attenuated).

---

## M2 — concurrency × prune-rate tradeoff (load-adaptive budget)

**Setup**: GQA, n=100, max-tokens=16, batch-submit mode (all 100 in one
`llm.chat()` so `max_num_seqs` engages continuous batching). Matrix:
max_num_seqs ∈ {1, 12} × pruning ∈ {r0, r50, r75}, proxy selector.
- **c1** (max_num_seqs=1): no batching → req/s = 1/latency (per-request benefit).
- **c12** (max_num_seqs=12): full continuous batching → req/s = throughput
  (KV-pressure-bound).

**Hypothesis**: the req/s gain from pruning GROWS with concurrency (under high
load KV pressure is the bottleneck, so pruning relieves it disproportionately).

<!-- M2_RESULTS_TABLE -->

Artifacts: `runs/p2_d/m2_{c1,c12}_{r0,r50,r75}.json`, `runs/p2_d/m2_matrix.log`.

### Verdict M2: <!-- M2_VERDICT -->

---

## D-scope recommendation

<!-- D_SCOPE -->

(Based on M1 + M2 verdicts — filled when M2 completes.)

---

## Method-D levers under consideration (context)
1. **Load-adaptive budget** (prune rate responds to engine concurrency) — the
   most novel serving-specific lever (0/37 papers). M2 decides if it's real.
2. **Early / mid-encoder prune** (drop patches after an early ViT layer so the
   encoder also does less work). M1 decides if it's worth the surgery.
3. **KV-cache-aware budget** (prune rate responds to KV-cache occupancy) —
   related to #1, the runtime-side analog.
4. **Served-throughput reporting** (tok/s, req/s, TTFT) — the evaluation
   differentiator, kept regardless.

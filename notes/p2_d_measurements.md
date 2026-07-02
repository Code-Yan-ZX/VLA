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

|  | r0 (576 tok) | r50 (288 tok) | r75 (144 tok) |
|---|---|---|---|
| **max_num_seqs=1** (no batching) | 1.815 req/s (55s) [acc .580] | 2.132 (47s) [.550] | 2.294 (44s) [.450] |
| **max_num_seqs=12** (full batching) | 5.754 req/s (17s) [.580] | 8.181 (12s) [.550] | 10.095 (10s) [.450] |

**Pruning speedup over r0, split by concurrency (the decisive comparison):**
| prune | speedup @ c1 (latency-only) | speedup @ c12 (KV-pressure-bound) | delta (concurrency bonus) |
|---|---|---|---|
| r50 | **1.17×** | **1.42×** | +0.25 |
| r75 | **1.26×** | **1.76×** | +0.49 |

Artifacts: `runs/p2_d/m2_{c1,c12}_{r0,r50,r75}.json`, `runs/p2_d/m2_matrix.log`,
`scripts/analyze_m2.py`.

### Verdict M2: **load-adaptive budget STRONGLY JUSTIFIED.**
The req/s speedup from pruning GROWS markedly with engine concurrency: r50 goes
1.17×→1.42× and r75 goes 1.26×→1.76× as concurrency rises 1→12. Under high load,
KV-cache pressure is the bottleneck and pruning relieves it disproportionately
(smaller per-request KV → more concurrent requests fit → higher throughput). The
effect is monotone AND grows with prune depth (r75's concurrency bonus +0.49 is
~2× r50's +0.25). **This is the most novel serving-specific lever (0/37 papers
measure it) and M2 confirms it is real and large.** ⇒ D's core should be a
prune-rate that responds to engine concurrency / KV-cache occupancy, amplifying
the serving win exactly where the speedup is largest.

---

## D-scope recommendation

Based on M1 (vt_frac=6.6%) + M2 (speedup grows with concurrency):

**BUILD (high value, validated):**
1. **Load-adaptive / KV-cache-aware budget** — D's CORE. Prune rate r(concurrency,
   KV-occupancy) rises under high load. M2 shows r75's speedup grows +0.49 (1.26×
   →1.76×) from c1→c12 — the largest, most novel serving-specific win. Concretely:
   monitor vLLM's running `max_num_seqs` / KV-cache usage, set r ∈ [r_min, r_max]
   (e.g. [0.25, 0.75]) adaptively. This is the contribution 0/37 papers make.

**SKIP (low value, validated):**
2. **Early / mid-encoder ViT prune** — M1 shows the vision tower is only 6.6% of
   prefill. Surgery on the ViT (drop patches after an early layer) caps the extra
   prefill win at ~7% and risks breaking the CLIP features the proxy selector
   relies on. NOT worth it; the win is all LLM-sequence-shortening, already captured.

**KEEP (already working, the evaluation differentiator):**
3. **Served-throughput reporting** (req/s, tok/s, TTFT) at multiple concurrency
   levels — the 0/37-papers gap; report the M2-style concurrency×prune matrix as a
   headline result, not just single-ratio speedups.

**Net D = proxy selector (kept, best boundary accuracy) + load-adaptive budget
(new core) + concurrency-aware served-throughput eval (new headline).** No ViT
surgery, no new selector (the 3 boundary selectors all failed on OCR). D's novelty
is entirely on the SERVING side, which is exactly where the open gap is.

**Accuracy guardrail (from the matrix):** r50 drops GQA acc 0.580→0.550 (−3.0pp),
r75 drops to 0.450 (−13pp). So the load-adaptive budget's r_max should stay ≤0.50
for GQA-class tasks (r75 is too lossy); OCR/TextVQA may need an even tighter cap.
The proxy selector's accuracy (established in v1 comparison) is the floor.

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

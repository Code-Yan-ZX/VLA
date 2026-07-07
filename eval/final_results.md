# SUPERSEDED — see final_results_v3.md

> **This V0-era file (vLLM 0.10.2 V0, LLaVA-1.5-7B) is superseded by
> `/final_results_v3.md` at the repo root**, the single source of truth for all
> numbers in the v3 paper. Retained for historical V0 comparison only.

# Final Results — Consolidated Paper Data (ALL numbers, honest)

> P4 prep. Every number the paper will cite, organized as paper tables. Each row
> is tagged with its source file. **Read this doc, not the underlying logs, for
> paper numbers.** Base: LLaVA-1.5-7B-hf · Engine: vLLM 0.10.2 V0 · 1× A40 46GB.
> Last updated 2026-07-02 (P3-step-3 n=500 GATE added).

---

## The served-throughput gap (the headline novelty statement)

> Sources: `notes/lit-survey.md` §2.1, §5, §7; `notes/new-papers-and-qwen35vl.md` Task 2.

- Survey of **37 methods** (2023–2026, all arXiv-verified 2026-07-01).
- **13/37** report *some* wall-clock-style number (raw CUDA latency / offline prefill / decode speedup).
- **0/37** measure **served throughput (req/s or tok/s) inside a production serving engine** (vLLM / SGLang / lmdeploy / TRT-LLM).
  - SparseVILA (ICCV'25) is the closest — 4.0× prefill / 2.6× e2e — but on its own AWQ pipeline, **not** a serving engine.
  - FastV/SparseVLM/VisionZip/FasterVLM/PRUNESID/Fourier-VLM/PLPHP all measure on research code / authors' own harness.
  - GlimpsePrune uses vLLM only as an *lmms-eval measurement backend*; pruning runs in a transformers fork, not the serving path.
- Independent corroboration: the Westlake survey (arXiv 2507.20198, v5) §6.5.3 documents the FlashAttention-score root cause of non-integrability; §6.5.4 names TTFT/per-token latency as "missing"; Eval-Framework (arXiv 2510.07143) demands this eval. vLLM RFC #45098 (`--image-pruning-rate`) is unfinished infra, not a method.
- **⇒ Gap is OPEN. This paper is the first served-throughput measurement of a VLM visual-token compressor inside a serving engine.**

---

## Table A — Served-throughput scaling (concurrency × prune matrix)

> Sources: `notes/p2_d_measurements.md` M2; `notes/p3s2_pareto.md` §5. Engine: vLLM V0, GQA, batch-submit mode (continuous batching), proxy selector, n=100 per cell.

| max_num_seqs | r0 (576 tok) req/s | r50 (288 tok) req/s | r75 (144 tok) req/s | r50/r0 | r75/r0 |
|---|---|---|---|---|---|
| c1  (no batching) | 1.815 | 2.132 | 2.294 | **1.17×** | 1.26× |
| c4  | 3.436 | 5.022 | — | 1.46× | — |
| c12 (full batching) | 5.754 | 8.181 | **10.095** | **1.42×** | **1.75×** |

### Headline numbers (the measurement contribution)
- **c12/r75 = 1.75× served req/s** over c12/r0 — the strongest single speedup.
- **Concurrency amplification:** the prune speedup *grows* with concurrency: r50 goes 1.17× (c1) → 1.42× (c12); r75 goes 1.26× (c1) → **1.75× (c12)** — a +0.49 concurrency bonus at r75. (M2 verdict.)
- **Constant-vs-bursty contrast:** GQA adaptive controller, bursty req/s 3.38 vs constant req/s 6.95 = **2.06× faster under sustained high load** (controller prunes at r_max-equivalent throughout). `notes/p2_d_results.md` §"Constant-vs-bursty".

### Prefill cost breakdown (why prefill is sub-linear)
> Source: `notes/p2_d_measurements.md` M1 (GQA r0, n=100, max-tokens=1, serial).

| phase | mean ms | % of TTFT |
|---|---|---|
| Vision tower (CLIP ViT-L/14@336, 576 patches) | **12.7** | **6.6%** |
| Projector (linear, boundary) | 0.16 | 0.1% |
| LLM prefill (remainder over image+text) | 180.3 ± 8.2 | 93.3% |
| **TTFT (total)** | **193.2** | 100% |

- **Vision tower = 6.6% of TTFT.** ⇒ Mid-encoder / early-prune surgery caps the extra prefill win at ~7% — **NOT worth the surgery**. The win is LLM-sequence-shortening, already captured at the boundary. This is the quantitative root of finding B2 (below).
- GQA r50 prefill speedup 1.24× (TTFT 631→510 ms); r75 1.30× — sub-linear because the 6.6% vision cost is fixed. `eval/p2_probe_summary.md`.

---

## Table B — The 3 serving-specific findings (the mechanism story)

> Each finding is invisible to offline FLOPs measurement — the differentiation spine.

### Finding B1 — e2e speedup > prefill speedup at every prune ratio
> Source: `eval/p2_probe_summary.md` ★★ (GQA, n=200, c1).

| prune r | e2e speedup | prefill (TTFT) speedup | gap (e2e−prefill) |
|---|---|---|---|
| r25 | 1.17× | 1.14× | +0.03 |
| r50 | **1.33×** | 1.24× | **+0.09** |
| r75 | 1.43× | 1.30× | +0.13 |

**Mechanism:** under continuous batching, smaller per-request KV-cache ⇒ more concurrent requests fit ⇒ higher req/s. The deployment win lives in **KV-cache/concurrency, not prefill FLOPs** — exactly the effect offline FLOPs cannot see. Quantitatively confirmed by M2 (the c1→c12 amplification).

### Finding B2 — prefill is sub-linear (fixed vision-tower cost)
> Source: M1 (above) + `eval/p2_probe_summary.md` ★★ finding 2.

r75 cuts tokens 4× but yields only 1.30× prefill speedup. Cause: pruning is at projector *output* ⇒ the vision tower still processes all 576 tokens (6.6% fixed cost). Implication (acted on): mid-encoder prune is NOT worth the surgery (M1).

### Finding B3 — speedup is workload-dependent (visual-token fraction)
> Source: `eval/p2_probe_summary.md` ★ Finding 3.

| benchmark | r50 e2e speedup | r50 acc drop | notes |
|---|---|---|---|
| GQA (visual-heavy, short Q/A) | **1.33×** | −2.0% | pruning pays off |
| TextVQA (text-heavy, long Q/A) | 1.16× | −2.5% | saturates by r50 (r50≈r75 TTFT) |

**Mechanism:** TextVQA is text-heavier ⇒ visual tokens are a smaller sequence fraction ⇒ pruning them yields less wall-clock; the floor is text length + fixed vision cost. Offline FLOPs reduction is identical regardless of text length, but wall-clock scales with visual fraction — a serving-specific dimension.

---

## Table C — Method Pareto across 5 benchmarks (the supporting method result)

> Frame: **throughput-optimal under a per-benchmark accuracy guardrail** — NOT "Pareto-dominant everywhere" (honest).
> Controller: num_running/max_num_seqs signal, conc_lo=0.25/conc_hi=0.75, r∈[0.25,0.50], mt64, c12, bursty load.
> Sources: `notes/p3s1_pareto.md` (GQA, TextVQA n=200), `notes/p3s2_pareto.md` (MME, MMBench, ScienceQA), **`notes/p3s3_pareto_n500.md` (n=500 tightening — the GATE)**.

> **⚠ P3-step-3 n=500 GATE RESULT (2026-07-02, load-bearing):** re-running the
> Pareto comparison at n=500 (acc stderr ~±0.022 vs n=200's ~±0.031) REVERSED
> the two n=200 Pareto-dominate cases. **MME and ScienceQA no longer
> Pareto-dominate at n=500** (acc margins ±0.014/−0.010 are noise; MME's req/s
> win over r25 also evaporated to a dead heat). MMBench's strict "Pareto" label
> at n=500 is a +0.014 acc-margin artifact (|z|=0.49, noise) — honestly a
> req/s-win + acc-tie. **True clean Pareto-dominate count at n=500 = 0/5.**
> The method is therefore reframed as "free throughput over r25 at iso-accuracy-
> to-r25" (NOT a Pareto win). The n=200 sub-tables below are RETAINED for the
> record but the paper must cite n=500 numbers (Table C-n500). See
> `notes/p3s3_pareto_n500.md` §5–6 for the full honest interpretation + framing.

### C1 — GQA (n=200, mt32, bursty)
| config | req/s | acc | Δreq/s vs r25 | Δacc vs r50 | verdict |
|---|---|---|---|---|---|
| adaptive | 2.346 | 0.550 | +0.024 (WIN) | **−0.015 (LOSS)** | req-only |
| fixed r25 | 2.322 | 0.550 | — | — | — |
| fixed r50 | 2.552 | 0.565 | — | — | **dominates (r50 acc-neutral on GQA)** |

**Honest verdict:** GQA's short yes/no answers recover from aggressive pruning at mt32 ⇒ r50 is acc-NEUTRAL (0.565 ≥ r25 0.550) ⇒ fixed-r50 dominates. **Adaptive does NOT win on GQA.** (The prior mt16 "win" was a truncation artifact — r50 acc was depressed to 0.522 at mt16.)

### C2 — TextVQA  ⚠ n=500 CORRECTION (load-bearing)
> Source: `notes/p3s2_pareto.md` §3.

| config | n | acc | vs adaptive |
|---|---|---|---|
| adaptive | 500 | 0.510 | — |
| fixed r25 | 500 | 0.510 | tie |
| fixed r50 | 500 | **0.526** | adaptive − r50 = **−0.016 (z=−0.51, NOT significant)** |

**⚠ CORRECTION:** P3-step-1 n=200 reported adaptive 0.540 > r50 0.520 (+0.020) and claimed "clean Pareto-dominate." **n=500 REVERSES the sign** (adaptive 0.510 < r50 0.526). The n=200 acc win was NOISE. **TextVQA is downgraded to req/s-only advantage** (adaptive 2.270 > r25 2.233, +0.037). The paper MUST present n=500 TextVQA, not n=200.

| config | req/s (n=500 bursty) | acc (n=500) | Δreq/s vs r25 |
|---|---|---|---|
| adaptive | 2.270 | 0.510 | **+0.037 (WIN)** |
| fixed r25 | 2.233 | 0.510 | — |
| fixed r50 | 2.390 | 0.526 | (beats adaptive on both — r50 wins TextVQA at n=500) |

### C3 — MME (n=200, mt64) — **PARETO-DOMINATES**
| config | req/s | acc | Δreq/s vs r25 | Δacc vs r50 | verdict |
|---|---|---|---|---|---|
| adaptive | 2.980 | **0.700** | +0.045 (WIN) | **+0.015 (WIN)** | **PARETO-DOMINATES** |
| fixed r25 | 2.935 | 0.690 | — | — | — |
| fixed r50 | 3.160 | 0.685 | — | — | — |

### C4 — MMBench (n=200, mt64)
| config | req/s | acc | Δreq/s vs r25 | Δacc vs r50 | verdict |
|---|---|---|---|---|---|
| adaptive | 3.048 | 0.720 | +0.072 (WIN) | −0.010 (LOSS) | req-only |
| fixed r25 | 2.976 | 0.725 | — | — | — |
| fixed r50 | 3.212 | 0.730 | — | — | **dominates (r50 acc-neutral)** |

**Honest verdict:** like GQA, MMBench's r50 is acc-neutral/positive (proxy pruning recovers) ⇒ fixed-r50 dominates. Adaptive loses. NOT a method failure — same "free recovery" regime.

### C5 — ScienceQA (n=200, mt64) — **PARETO-DOMINATES**
| config | req/s | acc | Δreq/s vs r25 | Δacc vs r50 | verdict |
|---|---|---|---|---|---|
| adaptive | 3.005 | **0.685** | +0.070 (WIN) | **+0.015 (WIN)** | **PARETO-DOMINATES** |
| fixed r25 | 2.936 | 0.675 | — | — | — |
| fixed r50 | 3.165 | 0.670 | — | — | — |

### Cross-benchmark summary (the honest pattern) — n=200, SUPERSEDED by n=500 below
> The discriminator is **per-benchmark r50-acc-cost**, NOT answer density.
> ⚠ **This n=200 summary is RETAINED FOR THE RECORD ONLY.** The n=500 re-run
> (Table C-n500) REVERSED MME/ScienceQA — the "2/5 Pareto-dominate" claim does
> NOT survive the noise gate. Use Table C-n500 for all paper numbers.

| benchmark | r50 acc-costly? | adaptive verdict | req/s win over r25 |
|---|---|---|---|
| **MME** | yes (r50 0.685 < r25 0.690) | **PARETO-DOMINATES** | +0.045 |
| **ScienceQA** | yes (r50 0.670 < r25 0.675) | **PARETO-DOMINATES** | +0.070 |
| TextVQA (n=500) | yes (r50 0.526 > r25 0.510, BUT adaptive ≈ r50) | req/s only | +0.037 |
| GQA | no (r50 0.565 ≥ r25 0.550) | req/s only | +0.024 |
| MMBench | no (r50 0.730 ≥ r25 0.725) | req/s only | +0.072 |

**Robust signal (across ALL 5 benchmarks):** adaptive beats r25 on req/s (+2–7%). The accuracy win over r50 is real ONLY where r50 is accuracy-costly (MME, ScienceQA = 2/5). **This is the paper's honest method claim.**

### Table C-n500 — n=500 Pareto (the GATE; supersedes the n=200 sub-tables above)
> Source: `notes/p3s3_pareto_n500.md`. All runs c12, bursty, num_running controller,
> r∈[0.25,0.50], conc-lo 0.25/conc-hi 0.75, seed=0. GQA mt32; MME/MMBench/ScienceQA mt64.
> acc sig: |Δacc|<0.022=noise (|z|<1); 0.022–0.044=suggestive; ≥0.044=meaningful (n=500).

| benchmark | config | req/s | acc | Δreq/s vs r25 | Δacc vs r50 | acc sig | verdict (n=500) |
|---|---|---|---|---|---|---|---|
| **GQA** (mt32) | adaptive | 2.383 | 0.556 | −0.006 (LOSS) | −0.006 (LOSS) | noise (z=−0.19) | dominated (r50 best on both) |
|  | fixed r25 | 2.389 | 0.556 | — | — | — | — |
|  | fixed r50 | 2.607 | 0.562 | — | — | — | dominates |
| **MME** (mt64) | adaptive | 2.588 | 0.766 | −0.002 (tie) | +0.014 (WIN) | **noise (z=+0.52)** | acc-tie + req/s-tie → **REVERSED** (was Pareto at n=200) |
|  | fixed r25 | 2.590 | 0.758 | — | — | — | — |
|  | fixed r50 | 2.749 | 0.752 | — | — | — | highest req/s |
| **MMBench** (mt64) | adaptive | 3.101 | 0.726 | +0.053 (WIN) | +0.014 (WIN) | **noise (z=+0.49)** | req/s-win + acc-tie (strict-Pareto label is a noise artifact) |
|  | fixed r25 | 3.048 | 0.726 | — | — | — | — |
|  | fixed r50 | 3.299 | 0.712 | — | — | — | highest req/s |
| **ScienceQA** (mt64) | adaptive | 3.050 | 0.620 | +0.052 (WIN) | −0.010 (LOSS) | **noise (z=−0.33)** | req/s-win only → **REVERSED** (was Pareto at n=200) |
|  | fixed r25 | 2.999 | 0.618 | — | — | — | — |
|  | fixed r50 | 3.229 | 0.630 | — | — | — | highest req/s + acc |
| **TextVQA** (n=500, mt32) | adaptive | 2.270 | 0.510 | +0.037 (WIN) | −0.016 (LOSS) | noise (z=−0.51) | req/s-win only (unchanged from n=200) |
|  | fixed r25 | 2.233 | 0.510 | — | — | — | — |
|  | fixed r50 | 2.390 | 0.526 | — | — | — | dominates |

**★ n=500 GATE verdict (load-bearing):** clean Pareto-dominate count = **0/5** at n=500.
- **MME, ScienceQA REVERSED** — the n=200 acc Pareto half (±0.015) was noise; MME's n=200 req/s win (+0.045) also collapsed to a dead heat (−0.002).
- **req/s-over-r25 is NOT uniform**: clean WIN on MMBench (+0.053), ScienceQA (+0.052), TextVQA (+0.037); TIE on GQA (−0.006) and MME (−0.002).
- **r50 has the highest req/s on 4/5 benchmarks** (GQA, MME, MMBench, ScienceQA) — when the deployer can tolerate r50's acc, fixed-r50 is the throughput winner; adaptive's niche is recovering throughput at the r25 accuracy floor.

**Honest reframed method claim (use this in the paper):** *"a load-adaptive prune-depth controller that delivers a throughput win over the accuracy-favoring fixed point (r25) on the dense/MC benchmarks (MMBench +5.3%, ScienceQA +5.2%, TextVQA +3.7%) at iso-accuracy-to-r25 — a free throughput gain at the r25 floor. It does not beat fixed-r50 on accuracy or throughput; the deployer choosing r50 gets higher throughput still. The method's value is recovering ~half the r25→r50 throughput gap when r25 is the mandated accuracy floor."* This is a supporting (non-Pareto) method result; the measurement contribution carries the paper.

---

## Table D — Selector ablation history (why we use the proxy)

> Source: `eval/p2_method_v1_comparison.md` + addendum; `eval/p2_method_a2_findings.md`.
> TextVQA r50, matched samples (the OCR stress test). All are TRAINING-FREE boundary selectors.

| selector | signal | TextVQA r50 acc | vs proxy |
|---|---|---|---|
| **proxy (hidden-state deviation)** | post-projector hidden-state norm | **0.530** (n=200) / 0.500 (n=50) | — (best boundary) |
| v1 true CLS-attn | vision-tower last-layer [CLS]→patch attention | 0.445 (n=200) | −0.085 |
| v2 LLM-embed cosine | LLM embed_tokens(question) × post-projector patch | ~0.380 (n=50) | −0.120 |
| A'' CLIP contrastive | CLIP text enc × CLIP ViT patch (visual_proj) | **0.180** (n=50) | −0.320 |
| FastV (intra-LLM, anchor) | LLM layer-2 attention-rank on task token | **0.555** (n=200) | +0.025 (but NOT vLLM-integrable) |

**Pattern: 3 distinct training-free boundary signals (vision-saliency, LLM-cosine, CLIP-contrastive) ALL underperform the proxy on OCR.** CLIP-contrastive is catastrophic (0.180) because CLIP's contrastive loss aligns only the pooled [CLS], not per-patch features (verified on synthetic "STOP 123" image: 0/10 text-region overlap across projection/CLS-attn/rollout/MaskCLIP).

**Conclusion (acted on):** the boundary-training-free OCR ceiling is the proxy. We ACCEPT proxy-level accuracy. FastV (intra-LLM) is more accurate but **cannot run in vLLM** (V1 subprocess + CUDA-graph-locked decode + FlashAttention fuses softmax) — hence accuracy-only anchor, not our serving method. This is structural, not a missed trick — the literature has 5+ OCR-specific methods using trained/learned components precisely because TF boundary signals are too weak.

---

## Table E — FastV accuracy anchor (accuracy-only, NOT in vLLM)

> Source: `notes/p3s2_pareto.md` §4; `eval/p2_method_v1_comparison.md`. FastV = intra-LLM layer-2 attention-rank prune, keep=288 (r50). LLaVA-1.5-7B.

| benchmark | r0 control | FastV r50 | our proxy r50 |
|---|---|---|---|
| MME | 0.715 | 0.720 | 0.685 |
| MMBench | 0.755 | 0.740 | 0.730 |
| ScienceQA | 0.705 | 0.700 | 0.670 |
| GQA | 0.585 | 0.535 (r75: 0.515) | 0.565 |
| TextVQA | 0.555 | 0.555 | 0.530 |

**Positioning:** FastV is accuracy-comparable or higher (prunes inside the LLM where task-relevance is clearer), but **CANNOT run in vLLM** ⇒ no serving throughput. Our method trades a small accuracy delta for vLLM compatibility + the load-adaptive throughput gain. **Complementary to FastV, not replacing it.** Reported honestly as a stronger accuracy baseline that lacks the deployment story.

---

## Table F — Controller behavior (load-tracking proof)

> Source: `notes/p2_d_results.md` (realized-r); `notes/p3s1_pareto.md` (step profile).

### F1 — realized-r distribution (the controller adapts)
> num_running signal, c12, bursty (alternating 2/12 bursts). All 5 benchmarks.

| benchmark | r_min | r_mean | r_max | conc_frac range |
|---|---|---|---|---|
| GQA | 0.250 | 0.367 | 0.500 | 0.00–1.00 |
| TextVQA | 0.250 | 0.367 | 0.500 | 0.00–1.00 |
| MME | 0.250 | 0.367 | 0.500 | 0.00–0.92 |
| MMBench | 0.250 | 0.367 | 0.500 | 0.00–0.83 |
| ScienceQA | 0.250 | 0.367 | 0.500 | 0.00–0.83 |

Realized-r is **bimodal** (0.25 or 0.50) because alternating bursts saturate to conc<conc_lo or conc>conc_hi. The full [r_min, r_max] swing is exercised ⇒ the controller genuinely reacts to engine load.

### F2 — step-profile time-series (the controller figure)
> Source: `notes/p3s1_pareto.md` "Step-profile realized-r". GQA, step profile (low→high→low staircase).

- 141 decisions (30 low one-at-a-time + 1 high batch of 60 + 110 tail).
- **r run-length: `0.25 × 31 → 0.50 × 1 → 0.25 × 109`** — controller sits at r_min through low phase, jumps to r_max for the segment following the high batch (one-segment lag), returns to r_min for the tail.
- conc_frac spans full [0, 1] (vs prior kv_occupancy signal's 0.00–0.04).
- **This is Fig 3 (controller load-tracking):** realized-r clearly rises to r_max in the high phase and falls to r_min in low phases.

### F3 — controller policy (locked)
- Signal: `num_running / max_num_seqs` (concurrency fraction, spans [0,1]).
- Map: piecewise-linear, conc_lo=0.25 / conc_hi=0.75 (at c12: r_min below 3 concurrent, r_max above 9).
- Fallback: KV-occupancy (`get_num_free_gpu_blocks/num_total_gpu_blocks`) for long-sequence regimes; each signal cross-falls back if its reading is None.
- Accuracy guardrail: **r_max = 0.50** per benchmark (r75 drops GQA ~11% — too lossy). Per-benchmark r_max tuning is the guardrail mechanism.

---

## Implementation hurdles (for the paper's "why hard" subsection)

> Source: `notes/p2_d_results.md` §"Implementation notes".

1. **Engine-load read (SOLVED, V0):** `llm.llm_engine.scheduler[0].running` (deque → num running seqs) + `.block_manager.get_num_free_gpu_blocks() / .num_total_gpu_blocks` (→ KV-occupancy). No fallback needed in V0 (in-process model).
2. **Sync `llm.chat()` drains the engine** ⇒ a controller reading load at call boundaries always sees an empty engine. Fix: **engine-level streaming loop** (`add_request` + `step`), one segment at a time, draining fully between segments. Load sampled mid-drain (peak), fed to NEXT segment's decision (one-segment lag — a legitimate reactive control loop).
3. **Batched forward + shared projector-hook k** ⇒ all requests in flight during a forward must share the same k (else masked_scatter placeholder/kept-count mismatch). **Per-segment r (not per-request)** guarantees this. Plus `engine.reset_mm_cache()` between segments (else stale placeholder count reused). Plus `enforce_eager=True` for adaptive (varying seq length vs CUDA graph capture).

---

## Numbers flagged for paper readiness (gaps to close before/during drafting)

> **Honest risk register.** These are flagged so the paper does not over-claim.

1. **TextVQA: MUST report n=500, NOT n=200.** The n=200 adaptive>r50 acc win (+0.020) reverses at n=500 (−0.016, n.s.). ✅ RESOLVED — n=500 data in `notes/p3s2_pareto.md` §3.
2. **GQA acc at n=200 is noisy.** ✅ RESOLVED at n=500 (`notes/p3s3_pareto_n500.md`): adaptive 0.556 / r25 0.556 / r50 0.562 — r50 remains acc-neutral/best; adaptive loses. n=200 directional finding holds.
3. **MME/MMBench/ScienceQA at n=200 — Pareto claims noise-checked.** ✅ RESOLVED at n=500 (`notes/p3s3_pareto_n500.md`): the n=200 Pareto-dominate verdicts (MME, ScienceQA) **REVERSED** at n=500 — acc margins were noise; MME's req/s win also collapsed to a tie. **True clean Pareto-dominate count at n=500 = 0/5.** Method reframed as "free throughput over r25 at iso-acc-to-r25" (NOT a Pareto win). See Table C-n500 above.
4. **Single base (LLaVA-1.5-7B).** No Qwen3-VL-8B generalization row yet (deferred — see limitations).
5. **Single concurrency level for Pareto (c12).** Pareto table is at c12 only; c1/c4 Pareto not run (would strengthen the concurrency-amplification story).
6. **Realized-r is bimodal** (0.25/0.50, no intermediate) because bursts saturate the thresholds. A middle-tier burst would produce intermediate r — cosmetic for the figure, not load-bearing.

**Recommended pre-submission runs (for Main to schedule, NOT this synthesis task):**
- ~~GQA + 3 new benchmarks at n=500 (or full val) to tighten accuracy.~~ ✅ DONE 2026-07-02 (P3-step-3).
- Pareto at c4 (mid-concurrency) to show the amplification curve on the throughput frontier.
- (Stretch) Qwen3-VL-8B generalization row — reframe compression as redundancy-elimination at high visual budgets.

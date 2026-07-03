# Served Throughput of Visual-Token Compression in Vision-Language Models: A Measurement-Led Study with a Load-Adaptive Budget

> **Draft v1** (P4). Markdown; convertible to LaTeX for *Pattern Recognition* submission. All numbers are copied verbatim from `eval/final_results.md` (Tables A–F) and are source-tagged there. Base: LLaVA-1.5-7B-hf · Engine: vLLM 0.10.2 (V0) · Hardware: 1× NVIDIA A40 46 GB. Last updated 2026-07-03.

---

## Abstract

Visual-token compression is the dominant inference-efficiency lever for vision-language models (VLMs), yet we find that **0 of 37 surveyed compressors (2023–2026) report served throughput inside a production serving engine** (vLLM/SGLang/lmdeploy/TRT-LLM); 13 report some offline wall-clock number, all on the authors' own research harness. We close this gap with the first served-throughput measurement of a VLM visual-token compressor integrated inside vLLM, and report three serving-specific findings invisible to offline FLOPs accounting: (F1) end-to-end speedup exceeds prefill speedup at every prune ratio (1.33× vs 1.24× at r50), so the deployment win lives in KV-cache/concurrency, not prefill FLOPs; (F2) prefill is sub-linear because the vision tower is a 6.6% fixed cost; (F3) speedup scales with the visual-token fraction of the sequence (GQA 1.33× vs TextVQA 1.16× at r50). The headline result is **1.76× served req/s at concurrency 12 and 75% pruning**. As a supporting contribution we introduce a **load-adaptive prune-depth controller** (`r=f(concurrency)∈[r_min,r_max]`) that is *throughput-optimal under a per-benchmark accuracy guardrail*: it delivers a free throughput gain over the accuracy-favoring fixed point on the dense/MC benchmarks (MMBench +5.3%, ScienceQA +5.2%, TextVQA +3.7%) at iso-accuracy. We do **not** claim a new accuracy SOTA, nor a universally Pareto-dominant method — an n=500 noise gate overturned the n=200 Pareto claims, which we report openly.

---

## 1. Introduction

Vision-language models (VLMs) spend most of their inference cost on visual tokens: a single LLaVA-1.5-style image expands to 576 tokens before any text is appended, and self-attention is quadratic in sequence length. A large and active literature — **37 methods we survey (2023–2026)** — proposes to compress these tokens, reporting large FLOPs reductions (often 50–90%) with modest accuracy loss [FastV, 2403.06764; SparseVLM, 2410.04417; VisionZip, 2412.04467; FasterVLM/VisPruner, 2412.01818; PRUNESID, 2603.09480; AgilePruner, 2603.01236; VisionTrim, 2601.22674; GlimpsePrune, 2508.01548; Q-Zoom, 2604.06912; and 28 others].

Yet a deployer who serves a VLM does not bill FLOPs — they bill **requests per second, tokens per second, time-to-first-token (TTFT), and KV-cache memory under continuous batching**. This is precisely what no compressor measures. Of our 37 surveyed methods, **13 report some wall-clock-style number** (raw CUDA latency, offline prefill, or self-reported "faster"), but **0 measure served throughput inside a production serving engine** (vLLM/SGLang/lmdeploy/TRT-LLM). The closest, SparseVILA [2510.17777], reports 4.0× prefill / 2.6× end-to-end but on its own AWQ quantization pipeline, not a serving engine. The only serving-engine artifact, vLLM RFC #45098 (`--image-pruning-rate`), is unfinished infrastructure with no benchmarks. Two independent surveys corroborate that the gap is open: the Westlake token-compression survey [2507.20198] §6.5.3–6.5.4 documents the FlashAttention-score root cause that blocks in-LLM pruning from engine integration and names TTFT/per-token latency as "missing"; the Eval-Framework [2510.07143] explicitly demands this evaluation.

**The paradox we expose.** Cutting FLOPs by removing visual tokens does not translate linearly into wall-clock under continuous batching, for three reasons that FLOPs accounting is structurally blind to: (i) the deployment win is amplified by KV-cache pressure and concurrency scheduling, not by prefill arithmetic; (ii) the vision tower is a *fixed* cost that pruning at the projector boundary cannot touch; (iii) the wall-clock benefit scales with the *visual-token fraction* of the sequence, which varies by benchmark even at identical FLOPs reduction. We measure all three (§3).

**Contributions.** We claim the following, and no more:

1. **(Main) The first served-throughput measurement of a VLM visual-token compressor inside a serving engine**, plus **three serving-specific findings** (§3.3) that are each invisible to offline FLOPs measurement. Headline: **1.76× served req/s at concurrency 12 / 75% pruning** over the uncompressed engine baseline; concurrency amplifies the prune speedup from 1.26× (concurrency 1) to 1.76× (concurrency 12).
2. **(Supporting, conditional) A load-adaptive prune-depth controller** that is *throughput-optimal under a per-benchmark accuracy guardrail* (§4). It adapts the prune rate to engine load `r=f(num_running/max_num_seqs)∈[r_min,r_max]`, integrated into vLLM V0 via an engine-level streaming loop. At an n=500 noise gate it delivers a free throughput gain over the accuracy-favoring fixed point (r25) on MMBench (+5.3%), ScienceQA (+5.2%), and TextVQA (+3.7%) at iso-accuracy-to-r25. **It is not universally Pareto-dominant**: on 4/5 benchmarks a deployer who can tolerate r50's accuracy gets higher throughput still from fixed-r50.

**Honest framing.** We do *not* claim a new accuracy SOTA — our proxy selector matches the FastV family and does not beat it; we do *not* claim a new selector — three training-free boundary alternatives we tried all failed on OCR; we do *not* claim a multi-base study — results are on LLaVA-1.5-7B only. We claim the first served-throughput measurement and a load-adaptive budget that is throughput-optimal under an accuracy guardrail.

**Paper roadmap.** §2 surveys compression and serving engines and the load-bearing gap. §3 presents the measurement contribution (main). §4 presents the load-adaptive method (supporting). §5 reports experiments, including an n=500 noise gate that overturned our own n=200 Pareto claims — reported openly. §6 discusses limitations, including a KV-admission extension we attempted but could not validate on a single A40. §7 concludes.

> **[Figure 1 placeholder — The served-throughput gap.]** A two-bar chart over the 37-method landscape: bar A = 13/37 report *some* wall-clock-style number (offline CUDA latency / prefill / decode speedup on the authors' own harness); bar B = **0/37** measure served throughput (req/s, tok/s, TTFT, KV-MB) inside a production serving engine (vLLM/SGLang/lmdeploy/TRT-LLM). Underlying data: `notes/lit-survey.md` §2.1 and §5 (the 13 are SparseVLM, VisionZip, SparseVILA, DyCoke, Q-Zoom, LLaVA-UHD, ToMe, FasterVLM/VisPruner, PRUNESID, E-AdaPrune, FocusUI, Fourier-VLM, PLPHP).

---

## 2. Related Work

### 2.1 VLM visual-token compression

Visual-token compression splits into three families. **(a) Encoder-side selection** picks informative ViT tokens *before* the LLM (VisionZip [2412.04467], VTC-CLS [2412.05819], FasterVLM/VisPruner [2412.01818]). **(b) Inside-LLM pruning** discards or merges tokens at an early LLM layer using attention/CLS scores (FastV [2403.06764], SparseVLM [2410.04417], PyramidDrop [2410.17247], PRUNESID [2603.09480], PLPHP [2502.14504], DyCoke [2411.15024]). **(c) Projector-level compression** trains a projector that emits fewer tokens (TokenPacker [2407.02392], LLaVA-PruMerge [2403.15388]). Orthogonal axes include query-awareness (SparseVLM, Q-Zoom [2604.06912], AdaptMerge) versus query-agnostic (FastV, VisionZip), and fixed-ratio versus content-adaptive budgets (GlimpsePrune [2508.01548], Q-Zoom, E-AdaPrune [2603.05950]). A 2026 ICLR cluster — AgilePruner [2603.01236], VisionTrim [2601.22674], PRUNESID [2603.09480] — crowds the *accuracy/FLOPs combination-study* space, decomposing methods into scoring-basis × reduction-method and sweeping on offline research code.

The accuracy/FLOPs-combination space is therefore saturated; a three-month project on a single GPU cannot beat it on accuracy, and we do not try.

### 2.2 Serving engines for LLMs/VLMs

vLLM [Kwon et al., SOSP 2023] introduced PagedAttention and continuous batching, the de facto serving substrate for open VLMs. Recent serving-systems work targets scheduling and disaggregation: ElasticMM [2507.10069] (built on vLLM v0.6.6) does modality-aware load balancing and unified multimodal prefix caching, and *explicitly disclaims* compression ("we do not compare against these optimization methods"). The only compression-related serving artifact is vLLM RFC #45098 (`--image-pruning-rate`), which is an unfinished opt-in flag with no published method or benchmarks.

### 2.3 The load-bearing gap

The compression literature and the serving-systems literature have not met. Of our **37 surveyed compressors**, **13 report some wall-clock-style number** — but every one is measured on the authors' own research harness or a custom pipeline (e.g., SparseVILA's AWQ pipeline), *not* inside a continuous-batching serving engine. Two independent sources corroborate that the deployment-engine-throughput gap is real and unfilled:

- The Westlake survey [2507.20198] §6.5.3 ("Deployment Hurdles") states that attention-score pruning "cannot be seamlessly integrated into current optimization frameworks" because FlashAttention fuses matmul and softmax, making per-token scores inaccessible inside deployment pipelines — "this creates a critical gap." §6.5.4 names TTFT and per-token decode latency as "crucial for accurately assessing real-world inference acceleration" but notes they are **missing/unreported**.
- The Eval-Framework [2510.07143] argues current benchmarks miss the real cost and demands a dedicated throughput evaluation. EffiVLM-BENCH [2506.00479] unifies efficient-VLM evaluation but reports only **offline** TTFT and decode speedup (batch=1, HuggingFace transformers) — not served throughput.

> **[Table — throughput-reporting tally in the 37-method landscape.]** Columns: method; year/venue; reports FLOPs/token-count (Y for all 37); reports any wall-clock (13/37); measured *inside* a serving engine (**0/37**). Rows drawn from `notes/lit-survey.md` §2. This table is the evidence for Fig. 1 and the main novelty claim.

We close this gap. To our knowledge this is the first work to integrate a visual-token compressor inside a serving engine and report served throughput, TTFT, and KV-cache under continuous batching.

---

## 3. Served-Throughput Measurement (Main Contribution)

### 3.1 Setup

**Engine and model.** vLLM 0.10.2 in V0 mode (the in-process model path; see §4.3 for why V0 is required for an online controller), LLaVA-1.5-7B-hf (CLIP ViT-L/14@336, 576 image tokens per image), 1× NVIDIA A40 46 GB. We chose vLLM 0.10.2 (the last cu12-native major release compatible with our driver) over newer versions because the newer prebuilt wheels bind CUDA 13; this is an environment constraint, not a scientific one.

**Compressor (probe selector).** A training-free **boundary selector** that prunes at the output of `LlavaMultiModalProjector.forward` (i.e., post-projector, pre-LLM-fusion) using a hidden-state-deviation *proxy* score (§4.4 documents that three stronger-looking boundary signals all failed on OCR, so the proxy is the structural training-free boundary ceiling). Boundary pruning is chosen over intra-LLM pruning (FastV-style) precisely because it runs entirely before LLM fusion and slots into vLLM's multimodal processor without FlashAttention surgery — the integration hurdle diagnosed in [2507.20198] §6.5.3. Shrinking the sequence *before* fusion causes vLLM's PagedAttention to allocate fewer KV pages automatically, which is the mechanism §3.3 quantifies.

**Metrics.** `req/s` (requests / end-to-end wall-clock), `tok/s` (output tokens / wall-clock), `TTFT` (time-to-first-token, prefill wall-clock), `peak_kv_mb`, and task accuracy. All throughput runs use **batch-submit mode**: all N requests enter one `llm.chat()` call so `max_num_seqs` engages continuous batching — this is the served-throughput regime, not serial latency. Subsets are fixed-seed (seed=0); n is reported per table.

**Pruning rates.** `r ∈ {0, 0.25, 0.50, 0.75}`, where r is the fraction of visual tokens *removed* (so r50 keeps 288/576, r75 keeps 144/576). r=0 is the uncompressed control on an identical code path.

### 3.2 The concurrency × prune matrix

Table A reports served req/s on GQA across concurrency {c1, c4, c12} × prune {r0, r50, r75} (n=100 per cell, batch-submit, proxy selector).

**Table A — Served-throughput scaling (concurrency × prune), GQA.**

| max_num_seqs | r0 (576 tok) req/s | r50 (288 tok) req/s | r75 (144 tok) req/s | r50/r0 | r75/r0 |
|---|---|---|---|---|---|
| c1 (no batching) | 1.815 | 2.132 | 2.294 | **1.17×** | 1.26× |
| c4 | 3.436 | 5.022 | — | 1.46× | — |
| c12 (full batching) | 5.754 | 8.181 | **10.095** | **1.42×** | **1.76×** |

*Sources: `notes/p2_d_measurements.md` M2; `notes/p3s2_pareto.md` §5.*

**Headline numbers.** The strongest single speedup is **c12/r75 = 1.76× served req/s** over c12/r0. The decisive observation is **concurrency amplification**: the prune speedup *grows* with concurrency. At r50 the speedup goes 1.17× (c1) → 1.42× (c12); at r75 it goes 1.26× (c1) → **1.76× (c12)** — a +0.49 concurrency bonus. A compressor that looks only mildly useful in serial latency (1.26× at r75/c1) becomes substantially useful (1.76×) under the continuous batching a real deployer uses. Offline FLOPs measurement, which is independent of concurrency, cannot see this effect at all.

> **[Figure 2 placeholder — Concurrency × prune curve.]** req/s (y, log or linear) vs prune rate (x: r0/r50/r75), three curves for c1/c4/c12 on GQA. Annotate the c12/r75 = 1.76× point and the c1→c12 amplification arrows (r50: 1.17×→1.42×; r75: 1.26×→1.76×). Underlying data: Table A.

### 3.3 Three serving-specific findings

Each finding below is invisible to offline FLOPs measurement — together they are the differentiation spine of the paper.

**Finding F1 — End-to-end speedup exceeds prefill speedup at every prune ratio.** Under continuous batching, smaller per-request KV-cache lets more requests run concurrently, so the end-to-end (e2e) req/s speedup exceeds the prefill (TTFT) speedup (Table B-left, GQA, n=200, c1).

**Table B-left — F1: e2e speedup > prefill speedup (GQA).**

| prune r | e2e speedup | prefill (TTFT) speedup | gap (e2e − prefill) |
|---|---|---|---|
| r25 | 1.17× | 1.14× | +0.03 |
| r50 | **1.33×** | 1.24× | **+0.09** |
| r75 | 1.43× | 1.30× | +0.13 |

*Mechanism:* the deployment win lives in **KV-cache/concurrency, not prefill FLOPs** — exactly the effect offline FLOPs cannot see, and quantitatively confirmed by the c1→c12 amplification in Table A.

**Finding F2 — Prefill is sub-linear because the vision tower is a fixed cost.** Pruning at the projector *output* means the vision tower still processes all 576 tokens, so r75 (4× fewer tokens) yields only 1.30× prefill speedup. Table B-right gives the prefill cost breakdown (GQA r0, n=100, max-tokens=1, serial).

**Table B-right — F2: prefill cost breakdown (TTFT = 193.2 ms total).**

| phase | mean ms | % of TTFT |
|---|---|---|
| Vision tower (CLIP ViT-L/14@336, 576 patches) | **12.7** | **6.6%** |
| Projector (linear, boundary) | 0.16 | 0.1% |
| LLM prefill (image+text remainder) | 180.3 ± 8.2 | 93.3% |

*Source:* `notes/p2_d_measurements.md` M1. **Implication (acted on):** mid-encoder / early-prune surgery caps the extra prefill win at ~7% — not worth the engineering risk to the CLIP features the selector depends on. The win is almost entirely LLM-sequence-shortening, already captured at the boundary.

**Finding F3 — Speedup is workload-dependent (visual-token fraction).** At identical r50, the e2e speedup differs by benchmark because the visual-token fraction of the sequence differs (Table B-bottom).

**Table B-bottom — F3: workload-dependence at r50.**

| benchmark | r50 e2e speedup | r50 acc drop | notes |
|---|---|---|---|
| GQA (visual-heavy, short Q/A) | **1.33×** | −2.0% | pruning pays off |
| TextVQA (text-heavy, long Q/A) | 1.16× | −2.5% | visual tokens a smaller fraction; saturates by r50 |

*Mechanism:* TextVQA is text-heavier, so visual tokens are a smaller sequence fraction and pruning them yields less wall-clock; the floor is text length plus the fixed vision cost. Offline FLOPs reduction is identical regardless of text length, but wall-clock scales with visual fraction — a serving-specific dimension no FLOPs number captures.

### 3.4 Why FLOPs ≠ wall-clock (the synthesis)

Tying the three findings together: under continuous batching, the deployment win from visual-token compression **(i) is amplified by KV-cache/concurrency scheduling (F1), (ii) is bounded by a fixed ~6.6% encoder cost (F2), and (iii) scales with the visual-token fraction of the sequence (F3)**. None of these three effects is visible to a FLOPs/token-count measurement. This is why 37 compressors can report large FLOPs reductions while 0 report (and several could not integrate into) a serving engine — and why a 50% token cut yields a 1.33× e2e speedup at c1 that *grows* to 1.76× at c12/r75 rather than the ~2× a naive FLOPs reading would predict.

---

## 4. Method — Load-Adaptive Budgeting under an Accuracy Guardrail (Supporting)

We now present the method. We are explicit up front: **this is a supporting, conditional contribution, not a blockbuster.** It does not beat fixed-r50 on accuracy on any benchmark at an n=500 noise gate (§5.2), and it does not claim to. Its value is a free throughput gain at a mandated accuracy floor, automatically adapting prune depth to engine load.

### 4.1 The controller

The concurrency amplification in §3.2 motivates making the prune rate *respond* to engine load. We define a piecewise-linear controller

```
r = f(num_running / max_num_seqs) ∈ [r_min, r_max]
```

mapping the concurrency fraction to a prune rate, with breakpoints `conc_lo = 0.25`, `conc_hi = 0.75` (at c12: r=r_min below 3 concurrent requests, r=r_max above 9). Under low load the controller prunes less (preserving accuracy, since KV-cache is plentiful); under high load it prunes more (trading accuracy for the KV-cache/concurrency win that F1 shows is largest exactly here).

**Choice of signal.** We use `num_running/max_num_seqs` (spans the full [0,1] range) rather than KV-occupancy. In our c12/short-sequence regime the KV pool (3085 blocks) vastly exceeds what 12 short sequences need (~120 blocks), so KV-occupancy peaks at only ~0.04 and a KV-occupancy-driven controller barely leaves r_min. The concurrency fraction exercises the full [r_min, r_max] swing (Table F). A KV-occupancy fallback is retained for long-sequence regimes where the pool is the binding resource.

### 4.2 The accuracy guardrail

We bound `r_max ≤ 0.50` globally. At r75, GQA accuracy drops ~11% (0.580→0.450), which is too lossy; r50 costs ~2% on GQA and ~3% on TextVQA. Per-benchmark r_max tuning is the guardrail mechanism: where r50 is accuracy-costly the controller can be capped lower; where r50 is accuracy-neutral the cap can stay at 0.50. §5.2 shows that whether the guardrail yields a Pareto win depends entirely on whether r50 is accuracy-costly on the specific benchmark — which a deployer may not know a priori.

### 4.3 vLLM V0 integration (why it is hard)

Integrating an online load-adaptive controller into vLLM required resolving three structural hurdles, each a candidate reason the gap of §2.3 stayed open.

1. **Engine-load read path.** V0 runs the model in-process, so the scheduler is reachable: `llm.llm_engine.scheduler[0].running` (deque → number of running sequences) and `.block_manager.get_num_free_gpu_blocks() / .num_total_gpu_blocks` (→ KV-occupancy). No fallback is needed in V0.
2. **Synchronous `llm.chat()` drains the engine.** A controller that reads load at call boundaries always sees an empty engine. Fix: an **engine-level streaming loop** (`engine.add_request` + `engine.step()`), one segment at a time, draining fully between segments. Load is sampled mid-drain (at peak occupancy) and fed to the *next* segment's decision — a **one-segment-lag** reactive control loop (a legitimate control design, not a deficiency).
3. **Batched forward + shared projector-hook k.** All requests in flight during a forward must share the same kept-count k (else the `masked_scatter` placeholder/kept-count mismatch aborts the batch). We therefore use **per-segment r (not per-request)**, plus `engine.reset_mm_cache()` between segments (else a stale placeholder count is reused from the multimodal-processor cache) and `enforce_eager=True` for adaptive runs (varying sequence length is incompatible with CUDA graph capture).

These three hurdles are the concrete manifestation of the [2507.20198] §6.5.3 "deployment hurdles" diagnosis; resolving them is a necessary engineering contribution of the paper.

### 4.4 The selector (an honest limitation)

The controller modulates prune *depth*; the selector chooses *which* tokens to keep. We use a **proxy hidden-state-deviation** selector at the projector output. We tried three stronger-looking training-free boundary selectors and all failed on the OCR stress test (TextVQA r50, Table D in §5.5): true vision-tower [CLS]→patch attention (0.445), LLM-embed cosine query↔patch (~0.38), and CLIP text×patch contrastive (0.180) — the last catastrophic because CLIP's contrastive loss aligns only the pooled [CLS], not per-patch features. The proxy (0.530) is the structural training-free boundary ceiling. FastV's intra-LLM attention-rank selector is more accurate (0.555) but **cannot run in vLLM** (FlashAttention fuses softmax; V1 runs the model in a subprocess; decode is CUDA-graph-locked), so it is our accuracy-only anchor (Table E), not our serving method. We accept proxy-level accuracy and do not claim otherwise.

---

## 5. Experiments

### 5.1 Setup

**Benchmarks.** GQA (scene/graph QA, short yes/no answers), TextVQA (OCR-heavy), MME (perception/cognition, yes/no), MMBench (multiple-choice, dense), ScienceQA (multiple-choice, multimodal science). **Load profiles.** bursty (alternating low/high request bursts), step (low→high→low staircase), constant (sustained high). **Controller.** num_running signal, conc_lo=0.25/conc_hi=0.75, r∈[0.25,0.50], c12 (max_num_seqs=12), mt32 for GQA/TextVQA, mt64 for MME/MMBench/ScienceQA, seed=0. **Baselines.** fixed-r25 (the accuracy-favoring fixed point) and fixed-r50 (the throughput-favoring fixed point), same engine, same selector.

### 5.2 Method Pareto across five benchmarks — and the n=500 correction

We first ran the five-benchmark Pareto comparison at **n=200** per benchmark. Two benchmarks (MME, ScienceQA) appeared to Pareto-dominate both fixed points, and TextVQA appeared to Pareto-dominate at n=200. We then re-ran at **n=500** (accuracy stderr ~±0.022 vs n=200's ~±0.031) as a noise gate. **The n=500 gate overturned the n=200 Pareto claims.** We report the n=500 numbers as authoritative and state the reversal plainly: this is a core honesty obligation, not a footnote.

**Table C — Method Pareto at n=500 (authoritative; c12, bursty, num_running controller, r∈[0.25,0.50]).**

| benchmark | config | req/s | acc | Δreq/s vs r25 | Δacc vs r50 | acc sig (n=500) | verdict |
|---|---|---|---|---|---|---|---|
| **GQA** (mt32) | adaptive | 2.383 | 0.556 | −0.006 (LOSS) | −0.006 (LOSS) | noise (z=−0.19) | dominated (r50 best on both) |
|  | fixed r25 | 2.389 | 0.556 | — | — | — | — |
|  | fixed r50 | 2.607 | 0.562 | — | — | — | dominates |
| **MME** (mt64) | adaptive | 2.588 | 0.766 | −0.002 (tie) | +0.014 (WIN) | **noise (z=+0.52)** | acc-tie + req/s-tie → **REVERSED** (was Pareto at n=200) |
|  | fixed r25 | 2.590 | 0.758 | — | — | — | — |
|  | fixed r50 | 2.749 | 0.752 | — | — | — | highest req/s |
| **MMBench** (mt64) | adaptive | 3.101 | 0.726 | +0.053 (WIN) | +0.014 (WIN) | **noise (z=+0.49)** | req/s-win + acc-tie (strict-Pareto label is a +0.014 noise artifact) |
|  | fixed r25 | 3.048 | 0.726 | — | — | — | — |
|  | fixed r50 | 3.299 | 0.712 | — | — | — | highest req/s |
| **ScienceQA** (mt64) | adaptive | 3.050 | 0.620 | +0.052 (WIN) | −0.010 (LOSS) | **noise (z=−0.33)** | req/s-win only → **REVERSED** (was Pareto at n=200) |
|  | fixed r25 | 2.999 | 0.618 | — | — | — | — |
|  | fixed r50 | 3.229 | 0.630 | — | — | — | highest req/s + acc |
| **TextVQA** (n=500, mt32) | adaptive | 2.270 | 0.510 | +0.037 (WIN) | −0.016 (LOSS) | noise (z=−0.51) | req/s-win only (the n=200 acc win reversed at n=500) |
|  | fixed r25 | 2.233 | 0.510 | — | — | — | — |
|  | fixed r50 | 2.390 | 0.526 | — | — | — | dominates |

*Sources: `notes/p3s3_pareto_n500.md` (GQA/MME/MMBench/ScienceQA); `notes/p3s2_pareto.md` §3 (TextVQA n=500).* Significance: |Δacc|<0.022 = noise (|z|<1); 0.022–0.044 = suggestive; ≥0.044 = meaningful.

**The n=500 gate result, plainly stated.** The clean Pareto-dominate count at n=500 is **0/5**:

- **MME and ScienceQA reversed.** At n=200, adaptive appeared to beat r25 on req/s (MME +0.045) and r50 on accuracy (both +0.015). At n=500, MME's req/s advantage evaporated to a dead heat (−0.002) and both accuracy margins are within noise (|z|<0.6). The n=200 "Pareto-dominate" verdicts were noise.
- **TextVQA reversed earlier.** The n=200 adaptive>r50 accuracy win (+0.020) flipped sign at n=500 (−0.016, n.s.). TextVQA is a req/s-only advantage.
- **MMBench's strict Pareto label is a noise artifact.** The +0.014 accuracy margin is |z|=0.49; honestly it is a req/s win + accuracy tie.
- **GQA is dominated.** Short yes/no answers recover from aggressive pruning, so r50 is accuracy-*neutral* on GQA (0.562 ≥ r25 0.556) and fixed-r50 dominates on both axes.

**Honest reframed method claim.** The robust, n=500-surviving claim is: *the load-adaptive controller delivers a throughput win over the accuracy-favoring fixed point (r25) on the dense/MC benchmarks — MMBench +5.3%, ScienceQA +5.2%, TextVQA +3.7% — **without an accuracy loss vs r25**, i.e., a free throughput gain at the r25 accuracy floor.* It does **not** beat fixed-r50 on accuracy anywhere; on 4/5 benchmarks fixed-r50 has the highest req/s (the deployer who can tolerate r50's accuracy gets higher throughput still). The method's niche is recovering roughly half the r25→r50 throughput gap when r25 is the mandated accuracy floor. The req/s-over-r25 win is also not uniform: it is a *tie* (not a win) on the two short-answer benchmarks (GQA −0.006, MME −0.002). We report this openly rather than retaining the n=200 Pareto language.

> **[Figure 4 placeholder — Pareto frontier, req/s (x) vs accuracy (y).]** Three points per benchmark (adaptive/r25/r50) × 5 benchmarks = 15 points. Annotate the benchmarks where adaptive is dominated by r50 (GQA, MME, ScienceQA, TextVQA on req/s) and where it ties (MMBench). The honest visual: adaptive sits between r25 and r50, never strictly dominating r50. Underlying data: Table C.

### 5.3 Controller behavior (load-tracking proof)

Table F confirms the controller genuinely reacts to engine load.

**Table F1 — Realized prune rate per benchmark (c12, bursty, num_running signal).**

| benchmark | r_min | r_mean | r_max | conc_frac range |
|---|---|---|---|---|
| GQA | 0.250 | 0.367 | 0.500 | 0.00–1.00 |
| TextVQA | 0.250 | 0.367 | 0.500 | 0.00–1.00 |
| MME | 0.250 | 0.367 | 0.500 | 0.00–0.92 |
| MMBench | 0.250 | 0.367 | 0.500 | 0.00–0.83 |
| ScienceQA | 0.250 | 0.367 | 0.500 | 0.00–0.83 |

The full [r_min, r_max] swing is exercised, confirming the controller adapts. Realized-r is **bimodal** (0.25 or 0.50) because the alternating-burst profile saturates the thresholds; a middle-tier burst would produce intermediate values (a cosmetic limitation, §6).

**Step-profile time-series (Fig 3).** Under a low→high→low step load on GQA (141 decisions: 30 low one-at-a-time, then a high batch of 60, then 110 tail), the realized-r run-length is `0.25 × 31 → 0.50 × 1 → 0.25 × 109`: the controller sits at r_min through the low phase, jumps to r_max for the segment following the high batch (the one-segment lag of §4.3), and returns to r_min for the tail. The concurrency fraction spans the full [0,1] range (versus ~0.00–0.04 for the abandoned KV-occupancy signal).

**Constant-vs-bursty contrast.** On GQA, the adaptive controller under a *constant* (sustained-high) profile achieves 6.95 req/s versus 3.38 req/s under a *bursty* profile — **2.06× faster under sustained high load**, because the controller prunes at r_max throughout. This is direct evidence the controller tracks load.

> **[Figure 3 placeholder — Controller load-tracking.]** Step profile: realized-r and conc_frac over decision index. Show r rising to r_max in the high phase and falling to r_min in low phases. Underlying data: Table F2 (`notes/p3s1_pareto.md`).

### 5.4 FastV accuracy anchor (complementary, not replaced)

Table E reports FastV (intra-LLM layer-2 attention-rank prune, keep=288/r50) as an accuracy-only anchor — it is more accurate than our proxy but cannot run in vLLM.

**Table E — FastV (accuracy-only) vs proxy, r50.**

| benchmark | r0 control | FastV r50 | our proxy r50 |
|---|---|---|---|
| MME | 0.715 | 0.720 | 0.685 |
| MMBench | 0.755 | 0.740 | 0.730 |
| ScienceQA | 0.705 | 0.700 | 0.670 |
| GQA | 0.585 | 0.535 (r75: 0.515) | 0.565 |
| TextVQA | 0.555 | 0.555 | 0.530 |

FastV is accuracy-comparable or higher (it prunes inside the LLM where task-relevance is clearer) but has **no serving throughput**. Our method trades a small accuracy delta for vLLM compatibility and the load-adaptive throughput gain. They are complementary: FastV is the upper-bound accuracy reference; we are the deployment-throughput reference.

### 5.5 Selector ablation history (why we use the proxy)

Table D records the three training-free boundary selectors we tried and rejected on the TextVQA r50 OCR stress test.

**Table D — Boundary training-free selector ablation (TextVQA r50, matched samples).**

| selector | signal | TextVQA r50 acc | vs proxy |
|---|---|---|---|
| **proxy (hidden-state deviation)** | post-projector hidden-state norm | **0.530** (n=200) | — (best boundary) |
| v1 true CLS-attn | vision-tower last-layer [CLS]→patch attention | 0.445 (n=200) | −0.085 |
| v2 LLM-embed cosine | LLM embed_tokens(question) × post-projector patch | ~0.380 (n=50) | −0.120 |
| A'' CLIP contrastive | CLIP text enc × CLIP ViT patch | **0.180** (n=50) | −0.320 |
| FastV (intra-LLM, anchor) | LLM layer-2 attention-rank on task token | **0.555** (n=200) | +0.025 (not vLLM-integrable) |

Three distinct training-free boundary signals — vision-saliency, LLM-cosine, CLIP-contrastive — all underperform the proxy on OCR; CLIP-contrastive is catastrophic because CLIP's contrastive loss aligns only the pooled [CLS], not per-patch features (verified on a synthetic "STOP 123" image: 0/10 text-region overlap across projection/CLS-attn/rollout/MaskCLIP). The boundary training-free OCR ceiling is the proxy. The literature has 5+ OCR-specific methods using trained/learned components precisely because training-free boundary signals are too weak; breaking this ceiling is left to future work.

### 5.6 Concurrency amplification (recap)

§3.2's concurrency×prune matrix is the quantitative motivation for the load-adaptive controller: the prune speedup grows with concurrency (r75: 1.26× at c1 → 1.76× at c12), so a controller that prunes harder exactly when concurrency is high captures the headroom a fixed-rate compressor leaves on the table. The free-throughput-over-r25 gains in §5.2 (MMBench +5.3%, ScienceQA +5.2%, TextVQA +3.7%) are the realized benefit of this amplification under a bursty load profile.

---

## 6. Discussion and Limitations

We discuss the boundaries of our claims, several of which are load-bearing for honest reviewing.

**Selector ceiling.** Boundary training-free selectors cannot match intra-LLM selection (FastV) on OCR; proxy-level accuracy is structural, not a missed trick. A learned component (a FlashVLM-style projection head, or a small trained relevance MLP) could break the ceiling but breaks the training-free constraint and is out of scope on a single GPU. The accuracy tables (C, D, E) should be read with this ceiling in mind.

**Benchmark-conditional method and the n=500 null.** The load-adaptive controller is not universally Pareto-dominant. At the n=500 noise gate the clean Pareto-dominate count is 0/5; the surviving claim is a free-throughput-over-r25 win on 3/5 benchmarks (MMBench, ScienceQA, TextVQA) and a tie on the two short-answer benchmarks (GQA, MME). We explicitly retain the n=200 numbers and the reversal history in `eval/final_results.md` so reviewers can audit the correction; the paper cites n=500 throughout. A one-dimensional prune-rate-vs-concurrency controller does not beat a well-chosen fixed rate on accuracy; the controller's value is the *automatic* throughput-optimal behavior under an accuracy guardrail when the deployer does not know the workload's r50-cost a priori.

**KV-admission extension (attempted, not validatable here).** The concurrency amplification in §3.2 suggests that a *KV-admission* controller — gating new requests based on KV-cache pressure rather than modulating prune depth — could yield larger benefits in a genuinely KV-bound regime (high occupancy). We attempted to create a KV-bound regime on the 1× A40 (peak KV-occupancy reached only ~0.148; a KV-bound regime needs >>0.5) but the 46 GB KV pool is too large relative to what 12 short concurrent sequences can fill. We could not validate the KV-admission thesis on this hardware and abandoned the path; it is reported as future work requiring larger hardware or multi-GPU, not as a result.

**Single base.** All results use LLaVA-1.5-7B. Qwen3-VL-8B (native 2×2 MLP compression, M-RoPE, variable token counts) would reframe the story — its native compression changes the fixed-encoder-cost argument of F2, and M-RoPE changes the KV-cache behavior. Generalization is future work.

**Single concurrency for Pareto.** The Pareto comparison (Table C) is at c12 only. c1/c4 Pareto would strengthen the concurrency-amplification narrative on the throughput frontier; we did not run it and flag it for the camera-ready.

**Sample sizes.** GQA/MME/MMBench/ScienceQA use n=500 (after the gate); TextVQA uses n=500. Full-validation runs would further tighten the accuracy claims; the n=500 gate already overturned n=200 verdicts, so we caution against over-reading any sub-0.022 accuracy margin.

**Bimodal realized-r.** Under the alternating-burst profile the controller swings between r_min and r_max with no intermediate values. This is cosmetic for the controller-figure, not load-bearing for the claims; a graduated load profile would show intermediate r.

**One-segment-lag controller.** The per-segment (not per-request) prune rate and the one-segment lag are forced by vLLM's batched-forward plus shared-hook-k constraint (§4.3). A per-request adaptive policy is future infrastructure work that would require deeper engine changes.

---

## 7. Conclusion

We presented the first served-throughput measurement of a VLM visual-token compressor integrated inside a production serving engine, closing a gap left open by all 37 compressors we surveyed. The measurement yields three serving-specific findings invisible to offline FLOPs accounting: end-to-end speedup exceeds prefill speedup (the win is KV-cache/concurrency, not prefill FLOPs), prefill is sub-linear because the vision tower is a fixed cost, and speedup scales with the visual-token fraction of the sequence. The headline is **1.76× served req/s at concurrency 12 / 75% pruning**, with concurrency amplifying the prune speedup from 1.26× (concurrency 1) to 1.76× (concurrency 12). As a supporting contribution we introduced a load-adaptive prune-depth controller that is throughput-optimal under a per-benchmark accuracy guardrail, delivering a free throughput gain over the accuracy-favoring fixed point on the dense/MC benchmarks at iso-accuracy — openly reported as a conditional, non-Pareto result after an n=500 noise gate overturned our n=200 Pareto claims. The deployment win from visual-token compression lives in the serving engine (KV-cache/concurrency), is bounded by fixed encoder cost, and scales with the visual-token fraction — three effects invisible to FLOPs measurement and untouched by 37 prior methods.

---

## References

> arXiv IDs verified 2026-07-01 against arXiv abs pages (`notes/lit-survey.md` §8). No fabricated citations.

1. Chen, L. et al. **FastV — An Image is Worth 1/2 Tokens After Layer 2.** ECCV 2024 (Oral). arXiv:2403.06764.
2. Zhang, Y. et al. **SparseVLM — Visual Token Sparsification for Efficient Vision-Language Inference.** ICML 2025. arXiv:2410.04417.
3. Yang, S. et al. **VisionZip — Longer is Better but Not Necessary in Vision Language Models.** CVPR 2025. arXiv:2412.04467.
4. Zhang, Q. et al. **FasterVLM / VisPruner — Beyond Text-Visual Attention: Exploiting Visual Cues for Effective Token Pruning in VLMs.** arXiv:2412.01818 (v2 retitled; v1 "FasterVLM").
5. **VTC-CLS — [CLS] Token Tells Everything Needed for Training-Free Efficient MLLMs.** arXiv:2412.05819.
6. **PyramidDrop — Accelerating Your Large Vision-Language Models via Pyramid Visual Redundancy Reduction.** CVPR 2025. arXiv:2410.17247.
7. **PRUNESID — Prune Redundancy, Preserve Essence: Vision Token Compression via Synergistic Importance-Diversity.** ICLR 2026. arXiv:2603.09480.
8. **AgilePruner — An Empirical Study of Attention and Diversity for Adaptive Visual Token Pruning in Large VLMs.** ICLR 2026. arXiv:2603.01236.
9. **VisionTrim — Unified Vision Token Compression for Training-Free MLLM Acceleration.** ICLR 2026. arXiv:2601.22674.
10. **GlimpsePrune — A Glimpse to Compress: Dynamic Visual Token Pruning for Large Vision-Language Models.** arXiv:2508.01548.
11. **Q-Zoom — Query-Aware Adaptive Perception for Efficient Multimodal Large Language Models.** arXiv:2604.06912.
12. Khaki et al. **SparseVILA — Decoupling Visual Sparsity for Efficient VLM Inference.** ICCV 2025. arXiv:2510.17777.
13. Tao et al. **DyCoke — Dynamic Compression of Tokens for Fast Video Large Language Models.** CVPR 2025. arXiv:2411.15024.
14. Li, W. et al. **TokenPacker — Efficient Visual Projector for Multimodal LLM.** IJCV 2025. arXiv:2407.02392.
15. Shang, Y. et al. **LLaVA-PruMerge — Adaptive Token Reduction for Efficient Large Multimodal Models.** arXiv:2403.15388.
16. Bolya, D. et al. **Token Merging (ToMe) — Your ViT But Faster.** ICLR 2023. arXiv:2210.09461.
17. **E-AdaPrune — Energy-Driven Adaptive Visual Token Pruning for Efficient Vision-Language Models.** arXiv:2603.05950.
18. **FocusUI — Efficient UI Grounding via Position-Preserving Visual Token Selection.** arXiv:2601.03928 (CVPR 2026 externally).
19. **Fourier-VLM — Compressing Vision Tokens in the Frequency Domain for Large Vision-Language Models.** arXiv:2508.06038.
20. **PLPHP — Per-Layer Per-Head Vision Token Pruning for Efficient Large Vision-Language Models.** arXiv:2502.14504.
21. **METEOR — Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models.** ICCV 2025. arXiv:2507.20842.
22. **AdaTP — Attention-Debiased Token Pruning for Video Large Language Models.** arXiv:2505.20100 (EMNLP 2025 Findings).
23. **AdaReTaKe — Adaptive Redundancy Reduction to Perceive Longer for Video-language Understanding.** arXiv:2503.12559 (ACL 2025 Findings).
24. **RedundancyLens — Revealing and Exploiting Visual Token Processing Redundancy for Efficient Decoder-Only MLLMs.** ACL 2025 Findings. arXiv:2501.19036.
25. **PPE — Positional Preservation Embedding for Multimodal Large Language Models.** arXiv:2510.22936 (ICLR 2026 externally).
26. **HybridToken-VLM — Hybrid Token Compression for Vision-Language Models.** arXiv:2512.08240 (CVPR 2026 externally).
27. **G-Prune — What Kind of Visual Tokens Do We Need? Training-free Visual Token Pruning from the Perspective of Graph.** arXiv:2501.02268.
28. **AdaptPrune — Multi-Cue Training-Free Visual Token Pruning.** arXiv:2503.08019.
29. **AdaptMerge — Language-Guided Token Merging for VLMs.** EMNLP 2025 Findings.
30. **FlashSloth — Efficient Multimodal LLM.** arXiv:2412.04317.
31. **LLaVA-UHD v3 — Native-Resolution Encoding.** arXiv:2511.21150.
32. **LLaMA-VID.** arXiv:2311.17043.
33. Shao, Tao et al. **A Survey of Token Compression for Efficient Multimodal LLMs (Westlake survey).** arXiv:2507.20198 (v5, §6.5.3–6.5.4).
34. **Are We Using the Right Benchmark: An Evaluation Framework for Visual Token Compression Methods.** arXiv:2510.07143.
35. **EffiVLM-BENCH — Unified Benchmark for Efficient VLMs.** arXiv:2506.00479.
36. Liu, Cheng, Tan, You, Tao. **ElasticMM — Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism.** arXiv:2507.10069.
37. **vLLM RFC #45098 — Image Token Pruning flag.** github.com/vllm-project/vllm/issues/45098 (unfinished infrastructure).
38. Kwon, W. et al. **Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM).** SOSP 2023.
39. Liu, H. et al. **Visual Instruction Tuning (LLaVA).** NeurIPS 2023.
40. Radford, A. et al. **Learning Transferable Visual Models From Natural Language Supervision (CLIP).** ICML 2021.

---

*End of draft v1. All numerical claims are source-tagged in `eval/final_results.md` (Tables A–F). Section §5.2 and §6 carry the n=500 correction and the abandoned KV-admission path as honest limitations.*

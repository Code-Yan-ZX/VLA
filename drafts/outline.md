# Paper Outline — Served-Throughput of Visual-Token Compression in VLMs

> P4 prep. Synthesized from `eval/final_results.md`, `notes/positioning.md`,
> `notes/lit-survey.md`, `DECISIONS.md`. **Honesty-first**: the method is
> supporting/conditional, the measurement is the main contribution.
> Last updated 2026-07-02.

---

## Target venue

**Primary: Pattern Recognition (Elsevier, Q1, IF~8).**
- Fits: Pattern Recognition publishes measurement-gap + method papers in visual recognition/efficiency; the served-throughput measurement contribution (0/37 gap) is exactly its "characterize an overlooked evaluation dimension" mold; the method is a recognition-pipeline efficiency technique.
- Alternative backups (in priority order): **Information Sciences** (Elsevier, Q1, broader CS, accepts measurement+method); **Neurocomputing** (Elsevier, Q2, efficiency/deployment focus).
- **Why not CVPR/ECCV/NeurIPS (stretch):** the method is supporting/conditional (benchmark-conditional Pareto, not a blockbuster), and the 1× A40 + single-base + n≤500 evidence is thinner than a top-tier ML venue expects. The measurement contribution is strong but is better framed as an SCI-tier "first measurement + 3 findings + a load-adaptive method" package than a top-tier "new SOTA" paper. Pattern Recognition rewards this shape.
- **Decision (lockable): Pattern Recognition as primary.**

---

## Contribution hierarchy (HONEST — load-bearing)

> Reviewers will catch exaggeration. The hierarchy below is what the evidence supports, no more.

### Contribution 1 (MAIN) — First served-throughput measurement for VLM token compression in a serving engine + 3 serving-specific findings.
- **The gap:** 0/37 surveyed methods measure served throughput (req/s, tok/s, TTFT) inside a production engine (vLLM/SGLang/lmdeploy/TRT-LLM). We close it.
- **The 3 findings (each invisible to offline FLOPs):**
  - F1: e2e speedup > prefill speedup at every ratio (1.33× vs 1.24× at r50 GQA) — the win is KV-cache/concurrency, not FLOPs.
  - F2: prefill is sub-linear (r75 = 1.30× despite 4× fewer tokens) — vision tower is a 6.6% fixed cost.
  - F3: speedup is workload-dependent (GQA 1.33× vs TextVQA 1.16× at r50) — scales with visual-token fraction.
- **Headline number:** c12/r75 = **1.76× served req/s**; concurrency amplifies the prune speedup (r75: 1.26× at c1 → 1.76× at c12).
- This is the paper's spine. Reviewers cannot dismiss it (it is a measurement, validated on a real engine).

### Contribution 2 (SUPPORTING, conditional) — Load-adaptive budgeting = throughput-optimal under per-benchmark accuracy guardrails.
- A prune rate `r = f(num_running/max_num_seqs) ∈ [r_min, r_max]` that adapts to engine load, integrated into vLLM V0 via an engine-level streaming loop.
- **Robust claim:** adaptive beats fixed-r25 on req/s on ALL 5 benchmarks (+2–7%).
- **Conditional claim (honest):** adaptive beats fixed-r50 on accuracy ONLY where r50 is accuracy-costly (MME, ScienceQA = 2/5). Where r50 is acc-neutral (GQA, MMBench) or where the n=500 TextVQA reversal holds, fixed-r50 dominates — adaptive is not universally Pareto-dominant.
- **Frame explicitly as "throughput-optimal under a per-benchmark accuracy guardrail, automatically adapting prune depth to load" — NOT "Pareto-dominant."** The deployer who doesn't know a priori whether r50 costs accuracy on their workload gets a safe throughput win + a guardrailed accuracy floor.

### Non-contributions (stated to pre-empt reviewers)
- NOT a new accuracy SOTA — proxy selector matches FastV-ish, does not beat it.
- NOT a new selector — 3 boundary TF selectors failed on OCR (Table D); we accept proxy-level accuracy as the structural ceiling.
- NOT a multi-base study — single base (LLaVA-1.5-7B); Qwen3-VL-8B is future work.

---

## Section-by-section outline

### 1. Introduction (~1.5 pages)
- Hook: VLM inference is visual-token-bound; 37 methods compress them; all report FLOPs/accuracy; **none measure what a deployer sees — served throughput under continuous batching.**
- The paradox we expose: FLOPs-cut ≠ wall-clock under continuous batching (KV-cache scheduling, prefill saturation, decode-bandwidth binding).
- Contribution hierarchy (above) — measurement [MAIN] + load-adaptive method [SUPPORTING].
- Paper roadmap + the headline figure (Fig 1, the 0/37 gap).
- *Honest framing sentence:* "We do not claim a new accuracy SOTA; we claim the first served-throughput measurement and a load-adaptive budget that is throughput-optimal under an accuracy guardrail."

### 2. Related Work (~1 page)
- **VLM token compression** — 3 families (encoder-side / intra-LLM / projector-level); cite FastV, SparseVLM, VisionZip, FasterVLM, PRUNESID, AgilePruner, VisionTrim, GlimpsePrune, Q-Zoom. Position: accuracy/FLOPs-combination space is crowded (ICLR'26 cluster).
- **Serving engines for LLMs/VLMs** — vLLM PagedAttention, continuous batching, ElasticMM (scheduling, explicitly avoids compression), vLLM RFC #45098 (unfinished infra). Position: serving-systems work and compression work have not met.
- **The gap (load-bearing paragraph):** 13/37 report some wall-clock; **0/37 inside a serving engine.** Cite the Westlake survey (2507.20198 §6.5.3) as independent corroboration of the FlashAttention root cause and the Eval-Framework (2510.07143) as demanding this eval. **Table: the 37-method throughput-reporting tally** (subset of Table A's gap statement).

### 3. Served-Throughput Measurement (MAIN; ~2 pages)
- **3.1 Setup.** vLLM 0.10.2 V0, LLaVA-1.5-7B, 1× A40, proxy boundary selector, GQA/TextVQA subsets, metrics (req/s, tok/s, TTFT, KV-MB), batch-submit mode for continuous batching.
- **3.2 The concurrency × prune matrix (Table A, Fig 2).** req/s at {c1,c4,c12} × {r0,r50,r75}; the c12/r75 = 1.76× headline; concurrency amplification (r75: 1.26×→1.76× c1→c12).
- **3.3 The 3 serving-specific findings (Table B).**
  - F1 e2e>prefill: the KV-cache/concurrency mechanism (1.33× vs 1.24×).
  - F2 prefill sub-linear: M1 prefill breakdown (vision tower 6.6%); why mid-encoder prune is not worth it.
  - F3 workload-dependence: GQA 1.33× vs TextVQA 1.16×; visual-fraction argument.
- **3.4 Why FLOPs≠wall-clock (the synthesis).** Tie the 3 findings: the deployment win lives in (i) KV-cache/concurrency (F1), is bounded by (ii) fixed encoder cost (F2), and scales with (iii) visual-token fraction (F3) — none visible to FLOPs.

### 4. Method — Load-Adaptive Budgeting under Accuracy Guardrail (SUPPORTING; ~1.5 pages)
- **4.1 The controller.** r = f(num_running/max_num_seqs) ∈ [r_min, r_max], piecewise-linear with conc_lo=0.25/conc_hi=0.75. Why num_running (spans [0,1]) beats KV-occupancy (peaks at 0.04 in c12/short-seq regime).
- **4.2 The accuracy guardrail.** r_max per benchmark (≤0.50 — r75 drops GQA ~11%, too lossy). Per-benchmark r_max tuning = the mechanism that bounds accuracy loss.
- **4.3 vLLM V0 integration (the "why hard" subsection).** Engine-load read path; sync `llm.chat()` drains engine → engine-level streaming loop (`add_request`+`step`), one-segment-lag control; batched-forward + shared-hook-k → per-segment r (not per-request); `reset_mm_cache` + `enforce_eager=True`. (Table F3.)
- **4.4 Selector (the honest limitation).** Proxy (hidden-state deviation) — best training-free boundary selector. 3 alternatives failed on OCR (forward-ref Table D). FastV (intra-LLM) is more accurate but not vLLM-integrable. We accept proxy-level accuracy.

### 5. Experiments (~2.5 pages)
- **5.1 Setup.** 5 benchmarks (GQA, TextVQA, MME, MMBench, ScienceQA), bursty/step/constant load profiles, c12, mt64, n=200 (n=500 for TextVQA).
- **5.2 Method Pareto across 5 benchmarks (Table C, Fig 4).** Honest per-benchmark verdict: MME/ScienceQA Pareto-dominate; GQA/MMBench req/s-only (r50 acc-neutral); TextVQA req/s-only at n=500 (⚠ correction noted). Cross-benchmark summary: adaptive beats r25 on req/s everywhere (+2–7%); acc win over r50 only where r50 is costly.
- **5.3 Controller behavior (Table F, Fig 3).** Realized-r distribution (bimodal 0.25/0.50, full [r_min,r_max] swing); step-profile time-series (r rises to r_max in high phase, falls to r_min in low); constant-vs-bursty = 2.06× (controller tracks load).
- **5.4 FastV accuracy anchor (Table E).** Complementary, not replaced — FastV higher acc, no serving throughput.
- **5.5 Selector ablation history (Table D).** The 3 boundary TF failures → proxy is the ceiling → why we use proxy.
- **5.6 Concurrency amplification (recap of §3.2).** The motivation for load-adaptive pruning, quantified.

### 6. Discussion / Limitations (~0.75 page)
- **Selector ceiling:** boundary TF selectors cannot match intra-LLM (FastV) on OCR; proxy-level accuracy is structural, not a missed trick. A learned component (FlashVLM-style) could break it — left to future work.
- **Benchmark-conditional method:** adaptive is not universally Pareto-dominant; the value proposition is "safe throughput win + guardrailed accuracy" for a deployer who doesn't know the workload's r50-cost a priori.
- **Single base:** LLaVA-1.5-7B only. Qwen3-VL-8B (native 2×2 MLP compression + M-RoPE + variable tokens) reframes the story — generalization is future work.
- **Single concurrency level for Pareto (c12):** c1/c4 Pareto not run (would strengthen the amplification curve).
- **Bimodal realized-r:** cosmetic — a middle-tier burst would show intermediate r values.
- **Sample sizes:** GQA/MME/MMBench/ScienceQA at n=200; TextVQA corrected to n=500. Full-val runs would tighten accuracy claims.

### 7. Conclusion (~0.25 page)
- First served-throughput measurement of VLM token compression in a serving engine; 3 serving-specific findings; a load-adaptive budget that is throughput-optimal under an accuracy guardrail. The deployment win lives in the serving engine (KV-cache/concurrency), is bounded by fixed encoder cost, and scales with visual-token fraction — three effects invisible to FLOPs measurement and untouched by 37 prior methods.

---

## Figure list

- **Fig 1 — The served-throughput gap.** Bar chart: of 37 methods, 13 report some wall-clock, **0 inside a serving engine**. (From Table A gap statement; `notes/lit-survey.md` §2.1.)
- **Fig 2 — Concurrency × prune curve.** req/s vs prune rate, 3 curves (c1/c4/c12) on GQA; annotate c12/r75 = 1.76× and the c1→c12 amplification. (Table A.)
- **Fig 3 — Controller load-tracking.** Step profile: realized-r (and conc_frac) over decision index; show r rising to r_max in the high phase, falling to r_min in low phases. (Table F2.)
- **Fig 4 — Pareto frontier.** req/s (x) vs acc (y), 3 points per benchmark (adaptive/r25/r50) × 5 benchmarks; annotate where adaptive Pareto-dominates (MME, ScienceQA) vs where it doesn't. (Table C.)

## Table list (from `eval/final_results.md`)
- **Table A** — concurrency×prune matrix + prefill breakdown.
- **Table B** — the 3 serving-specific findings (with backing numbers).
- **Table C** — method Pareto across 5 benchmarks (GQA, TextVQA[n=500], MME, MMBench, ScienceQA) + cross-benchmark summary.
- **Table D** — selector ablation history (3 boundary failures).
- **Table E** — FastV accuracy anchor.
- **Table F** — controller behavior (realized-r + step profile + policy).
- (Related-work table) — 37-method throughput-reporting tally (subset, in §2).

---

## Candid limitations list (for §6, expanded)

1. **Selector ceiling** — boundary TF selectors (CLS-attn/LLM-cosine/CLIP-contrastive) all underperform proxy on OCR; FastV (intra-LLM) is more accurate but not vLLM-integrable. Proxy-level accuracy is the structural TF-boundary ceiling.
2. **Benchmark-conditional method** — adaptive Pareto-dominates only where r50 is acc-costly (MME, ScienceQA = 2/5); on GQA/MMBench (r50 acc-neutral) and TextVQA (n=500 reversal) fixed-r50 dominates. NOT a universal win.
3. **Single base** — LLaVA-1.5-7B only; Qwen3-VL-8B (native compression, M-RoPE, variable tokens) is future work and would reframe the compression story.
4. **Single concurrency for Pareto** — c12 only; c1/c4 Pareto would strengthen the amplification narrative.
5. **Sample sizes** — n=200 for 4 benchmarks, n=500 for TextVQA (corrected); full-val runs needed to tighten the accuracy claims for the camera-ready.
6. **Bimodal realized-r** — controller swings between r_min/r_max, no intermediate values under the alternating-burst profile; cosmetic, not load-bearing.
7. **One-segment-lag controller** — required by vLLM's batched-forward + shared-hook-k constraint; a per-request adaptive policy is future infra work.

---

## Open question for Main (before drafting)

- **Confirm primary venue = Pattern Recognition** (vs Information Sciences). PR favors the measurement-gap shape; IS is broader. Recommend PR; flag for user.
- **Decision needed:** run n=500 (or full-val) for GQA/MME/MMBench/ScienceQA BEFORE submission (recommended — tightens accuracy claims), or draft with n=200 + the n=500 TextVQA correction and note full-val as camera-ready? Recommend the former for the Pareto-dominate claims to survive review.

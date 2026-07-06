# ElasticVis (working name) / TokenSched — Per-Request Visual-Token Budget Allocation as a Serving Method

> **Pivoted primary direction (2026-07-03).** The v2 measurement framework is the SUBSTRATE; ElasticVis is the new method contribution. **A fresh main conversation (clean context) executes this** — read this doc + `STATE.md` first.

## The one-sentence idea
All 37 visual-token compressors (and our v2 controller) use a **global** prune rate — one r for every request. **ElasticVis makes the per-request visual-token budget `k_i` a scheduling decision the serving engine makes at admission time, optimizing goodput@SLO directly.**

## Why this is the right pivot
- **Sidesteps the saturated selector problem.** The stuck problem was *which* tokens to prune (3 boundary selectors failed on OCR; proxy is the TF ceiling; native merger substitutes for post-hoc pruning). ElasticVis doesn't touch the selector — it asks *how many* tokens per request, given system state. That's a **scheduling/allocation** problem where the serving engine has unique leverage (it sees load, SLO headroom, request mix) that a static compressor ignores.
- **Leverages our v2 assets directly** (nothing wasted):
  - **served-throughput measurement framework** (V1 engine, LLaVA-1.5 + Qwen3-VL, c1–c64, goodput@SLO) = ElasticVis's eval harness + objective.
  - **accuracy(k) curves** (probe data, per benchmark) = the objective's accuracy term.
  - **latency(k, load) data** (concurrency×prune×TTFT/e2e) = the latency predictor (the constraint).
  - **architecture-conditional finding** (native 2×2 merger substitutes) = ElasticVis must be architecture-aware (per-request k on Qwen3-VL respects the merger's token count).

## Formalization — online per-request budget allocation at admission
- **Decision (per arriving request i)**: visual-token budget `k_i ∈ [k_min, k_max]`.
- **Signals**: engine load (`num_running`, `gpu_cache_usage` via `llm.get_metrics()`), SLO headroom (remaining batch latency budget), request features (image resolution/complexity, question length, task type).
- **Objective**: maximize **goodput@SLO** = `Σ_i 1{batch_meets_SLO} · accuracy(k_i, request_i)`.
- **Constraint**: `LatencyPred(Σ k_i, current_load, batch_composition) ≤ SLO`.
- **Problem class**: online allocation (requests arrive over time; allocation irrevocable; state evolves). Candidate solvers: greedy (k_min default, raise toward k_max for SLO-safe / accuracy-sensitive requests), Lagrangian on the constraint, or a small learned controller.

## Novelty (vs the 37 + our v2 controller)
- **0/37** compressors do per-request budget (all global r).
- Our v2 controller is **per-segment** (coarse, threshold-heuristic, n=500 null). ElasticVis is **per-request** + **optimization-based** (objective = goodput@SLO, not a threshold).
- It's a **serving-SYSTEM** method (engine allocates), not a compressor (which tokens) — different problem class where the engine has the leverage.

## Components to build (first main conversation)
1. **Latency predictor** `LatencyPred(Σ k_i, load)` — fit from the v2 concurrency×prune×latency data (`notes/v2_p2_scale.md`). Forecasts batch latency for a given allocation.
2. **Per-request accuracy(k) model** — from probe data; refine by request-feature bucket (image complexity, task type). accuracy(k) is the objective term.
3. **The allocator** (admission-time): given load + SLO headroom + request features, solve for k_i.
4. **Integration**: admission-time per-request k → processor-level placeholder-shrink (V1, already built).
5. **Evaluation**: ElasticVis vs fixed-{r0,r25,r50,r75} vs v2 per-segment controller, on **goodput@SLO @ c64**, LLaVA-1.5 + Qwen3-VL. **Headline claim**: ElasticVis > any fixed rate on goodput@SLO (gives high-k to SLO-safe/accuracy-sensitive requests, low-k to latency-pressured ones) — per-request allocation beats the global knob.

## Relationship to the v2 measurement paper
- **v2 paper** (`drafts/paper_v2.md`, DONE: 9 tables + 5 figures + 47 refs) is **submittable as-is** (measurement-led). Keep as fallback / companion.
- **Decision for later** (based on ElasticVis results + timing): (a) fold ElasticVis into v2 as the method (paper = framework + ElasticVis), or (b) submit v2-measurement first, ElasticVis as follow-up.

## Assets on disk (the fresh conversation's starting point)
- **Framework code**: `src/serve_bench.py` (V1, c64, goodput, per-request TTFT via streaming add_request+step), `src/compressors.py` (proxy/cls/tome/random + hooks), `src/load_controller.py` (**the per-segment controller — ElasticVis's per-request successor**; has `get_metrics()` V1 signal path).
- **Data**: `runs/v2_p{0,1,2,3}/` + `notes/v2_p{0,1,2,3}_*.md` (accuracy(k) curves, latency(k,load), concurrency matrices, goodput, p50/p99).
- **Models**: LLaVA-1.5-7B (`runs/models/llava-1.5-7b-hf`), Qwen3-VL-8B-Instruct (HF cache `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct`).
- **Env**: `qwen3vl_clean` (vLLM 0.19.0 V1, in-process EngineCore via `VLLM_ENABLE_V1_MULTIPROCESSING=0`); `vtc_serve`(V0) + `fastv` retained.
- **v2 paper**: `drafts/paper_v2.md` + `drafts/figures/v2/`.

## First step for the fresh conversation
Read this doc + `STATE.md` + `notes/v2_p2_scale.md` (goodput@SLO, c64, the constraint data) + `notes/v2_p3_crosscompressor.md` (accuracy(k) per compressor). Then: formalize the online-allocation objective → build the latency predictor + per-request accuracy(k) model → implement the admission-time allocator → validate vs fixed rates on goodput@SLO @ c64. **The accuracy-k and latency(k,load) curves are already in the probe data — this is an allocation/optimization problem on measured inputs, not a new measurement.**

## Open design questions (to resolve early)
- Per-request vs per-batch allocation granularity (admission decides k_i per request; the batch forms dynamically under continuous batching).
- The latency predictor's accuracy (it gates the constraint) — calibrate on the c1–c64 data; report its error.
- SLO definition (TTFT-bound? e2e-bound? both?) — pick the deployment-relevant one (goodput@TTFT≤5s is the v2 headline SLO).
- Architecture-awareness: on Qwen3-VL, k_i is relative to the per-image native-merger token count (variable).

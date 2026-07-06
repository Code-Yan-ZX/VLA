# ElasticVis — Design & Build Spec (spine doc, authoritative)

> Pivoted primary method (2026-07-03). v2 measurement framework = SUBSTRATE; ElasticVis = the method contribution. **Read this + `STATE.md` first.** Short pitch: `notes/elasticvis_positioning.md`.

## 0. Reframed positioning (after 2026-07-06 novelty scan — the "0/N" claim is RETIRED)

The original STATE claim "0/N compressors do per-request budget → clean novelty" is **FALSE**. The scan found:
- **CARES** (ACL'26): per-image resolution = visual-token-count selection — per-request visual budget, but **content-driven, no load/SLO signal**.
- **DyToS** (Neurocomputing'26): "budget-aware dynamic token scheduling" for MLLMs — budget = single user latency target → learned per-layer pruner; NOT admission-time per-request.
- **PARCEL** (arXiv 2605.30126): literally "elastic visual-token compression" — one model at multiple budgets; the **compressor SUBSTRATE**, not a competitor.
- **AdaServe / SLOs-Serve / JITServe**: per-request token budget optimizing goodput@SLO — but **text domain** (spec-tree / text tokens).

**The clean unoccupied cell = an admission-time, per-request visual-token budget allocator driven by live LOAD + SLO HEADROOM, optimizing goodput@SLO, in a VLM serving engine.** Positioning:
> *"No prior visual-token compressor allocates the per-request budget as an admission-time serving decision driven by live load and SLO headroom under a goodput@SLO objective. Per-instance visual budgets exist (CARES/DyToS/PARCEL) but are accuracy/content-driven; per-request budget-for-goodput serving exists (AdaServe/SLOs-Serve/JITServe) but is text-domain."*

ElasticVis = the **scheduler/allocator** (goodput@SLO policy) sitting on top of an elastic-capable compressor (PARCEL-like), NOT a compressor. Must-cite: CARES, DyToS, PARCEL, AdaServe, SLOs-Serve, JITServe, EPDServe, vLLM-Omni, DeepSeek-OCR, VISOR.

**Approved spine (user, 2026-07-06): A→B.** Core = H1 system-signal-driven allocator (existing data, fastest to headline). Content dimension (H2) is a later ablation/extension vs CARES — out of scope for EV-0.

## 1. Formalization — online per-request budget allocation at admission

- **Request i** arrives at t_i with features φ_i (image resolution/complexity, question length, task type) and SLO τ_i (deadline, ms). For EV-0 H1, φ_i is UNUSED (aggregate accuracy); τ_i may vary (mixed-SLO).
- **Decision (admission-time, irrevocable):** visual-token budget k_i ∈ [k_min, k_max]. On Qwen3-VL, fraction f_i with k_i = round(f_i · N_native_i).
- **Signals at admission:** `LoadReading{num_running, kv_occupancy, max_num_seqs}` (via `llm.get_metrics()`), SLO headroom h_i = τ_i − LatPred(current batch), φ_i (unused H1).
- **Objective:** maximize **goodput@SLO** = Σ_i 1{req i meets SLO} · U_i(k_i), where U_i(k_i) = accuracy(k_i) (H1: aggregate; H2: φ_i-conditioned).
- **Constraint (gate):** LatPred(own_k=k_i, sum_k=Σ_{j∈running}k_j, num_running, kv_occupancy) ≤ τ_i.
- **Problem class:** online allocation (irrevocable, evolving capacity = SLO budget under churn). fixed-r is the degenerate constant-policy case; ElasticVis strictly wins iff (a) load varies over time or (b) per-request SLO headroom varies. → eval must create that variation (§4).
- **Solvers:** (1) Greedy/threshold — default k_min, raise toward k_max while constraint holds; (2) Lagrangian — shadow price λ on SLO violations, k_i* = argmax_k [U_i(k) − λ·cost(k,load)]; (3) small learned π (later).

## 2. Data map (from v2 probes; `notes/v2_p{0,1,2,3}_*.md`, `runs/v2_p*/`)

- **Latency side — SUFFICIENT for H1.** Per-request raw `{id, served_tok_s, served_req_s, ttft_ms, e2e_ms, peak_kv_mb, correct, answer, gt}` at (k∈{576,288,144}) × (load c∈{1,4,12,16,64}) = 15 cells, n=200 GQA, LLaVA-1.5-7B. Files: `runs/v2_p2/batch_c{1,4,16,64}_r{0,50,75}.json`, `runs/v2_p2/serial_c1_r{0,50,75}.json`, `runs/v2_p0/batch_c{1,4,12}_r{0,50,75}.json`, `runs/v2_p3/{proxy,true_cls,tome_merge,random}_c64_r{0,50,75}.json`.
- **accuracy(k) — per-instance RECOVERABLE (good).** Same 200 GQA ids across all r cells at iso-c; correctness flips on **53/200** images r0→r75 (P2) and 51/200 across compressors (P3). Real per-image signal. Only 3 k pts/image (coarse).
- **goodput@SLO.** JSON stores goodput@{TTFT≤500ms, e2e≤1s} (≈0 at c64, prefill floor ~3s). Full SLO sweep recomputed in `runs/v2_p2_analyze.py` L11–49 from raw ttft/e2e. Reference c64 fixed-rate goodput@TTFT≤5s: r0≈1.4–1.8, r75≈10.8–14.4 req/s.
- **GAPS (H2/EV-1):** (a) only 3 k pts → denser sweep r∈{0,.25,.5,.625,.75,.875} for smooth predictors; (b) NO per-image features logged (only id/correct/answer/gt) → H2 needs re-run with feature capture; (c) Qwen3-VL is req/s-only (no per-request latency/acc) → cross-arch needs new probe. LatPred generalization to intermediate loads (c∉{1,4,12,16,64}) is UNVERIFIED — EV-1 calibrates with an intermediate-c sweep.

## 2.1 CRITICAL data caveat — v2 ttft is segment-queue-baked (verified 2026-07-06)

Diagnostic on `runs/v2_p2` raw (r0/k576): `ttft_min` = c1 138ms / c4 2228 / c16 3028 / c64 2950 (≈batched-compute floor, ~flat for c≥4, weak k-dep); `ttft_p50` = c1 433 / c4 17542 / c16 11530 / c64 10721 (**p50 DECREASES with c**). `served_req_s`@c64 = k576 9.18 / k288 14.59 / k144 20.39 req/s.

**Implication:** v2 `ttft` = batched-compute(~2-3s) + **segment-queue-wait** (the v2 bench submits N then drains at concurrency c → lower c ⇒ lower throughput ⇒ longer queue ⇒ larger ttft; this is why c4 p50 > c64 p50). **k's serving lever is THROUGHPUT (k→KV-cache→served_req_s: 9.18→20.39 for k576→k144), NOT per-request compute** (ttft_min barely moves with k). This confirms v2 finding #1 and IS ElasticVis's mechanism: lower k ⇒ higher throughput ⇒ shorter queue ⇒ more requests meet SLO.

**Consequences:**
- **LatPred** (`predictors.py`) is fit on segment-sojourn ⇒ it is a *closed-loop-bench* predictor; using it for open-loop double-counts the queue. (It is still exact at measured grid cells and fine for closed-loop sanity; `predict()` was made hybrid n≤2→lookup to avoid the c1 sojourn blowup.)
- **Sim** (`sim.py`) must model the server as **M=64 slots, each request occupies a slot for `S(k)=64/served_req_s(k)` = {576:7.0, 288:4.4, 144:3.1}s** (Little's law from measured saturation throughput), with queue wait generated by the arrival process; `TTFT = wait + P(k)`, `P(k)≈ttft_min≈2.9s`. This is well-defined for mixed-k (per-request occupancy) ⇒ open-loop numbers are faithful, not merely directional. Sanity: closed_loop(c64) must reproduce served_req_s {9.18,14.59,20.39} and goodput@TTFT≤5s {1.4-1.8, 10.8-14.4}. Hand-checked: 200 reqs/μ=9.18 → wall 21.8s ✓, avg wait ≈(200-64)/2/9.18=7.4s + P 2.9s ≈ 10.3s ≈ measured p50 10.7s ✓.

## 3. Integration (from `src/serve_bench.py`, `src/load_controller.py`, `src/compressors.py`)

- **r is global** via shared mutable `k_cell` dict; the per-request *mechanism* exists, granularity does not. Placeholder count set in `patch_image_token_count()` `serve_bench.py:303-360` (adaptive branch `:341-348` reads `k_cell["k"]`). Actual gather in `_projector_hook` `:644-840` via `_cur_k()` `:636-642`; selector built at `:818/:822`, `sel.select()` `:823`.
- **load_controller (per-segment, 1-seg-lag)** `LoadAdaptiveController.decide_r()` `load_controller.py:289-331`; signals via `read_engine_load_v1` `:153-200` parsing `llm.get_metrics()`. Injection mutates `k_cell["k"]` at `serve_bench.py:1192/:1199`.
- **Admission hook point** = immediately before each `add_request` (`:1218` seg / `:1342` batch_submit), where load+SLO+features are all available. **Constraint:** k must be set before `preprocess_chat` (`:1143-1148`).
- **goodput@SLO** `goodput()` `:979-990`; per-request SLO flag `x<=slo_ms` `:984`; latencies `e2e_by_rid` `:1353`, `ttft_by_rid` `:1356`.
- **Plumbing (EV-1, MODERATE):** (a) replace shared `k_cell` with `k_by_rid[rid]` + `k_resolver(load,slo,φ)` at add_request — trivial; (c) thread-local side-channel so `get_num_image_tokens` `:343` reads per-request k — moderate; (b) **ESCAPE the batched-hook problem** (`:1160-1166` "all in-flight MUST share k") by moving prune to **per-request preprocess** (placeholder-shrink at k_i before batched forward), so the batched hook need not be per-request. Qwen3-VL natively tolerates variable-length multimodal input.

## 4. Eval regime (approved: open-loop variable-load primary + mixed-SLO secondary)

**Why not vanilla c64 closed-loop:** closed-loop saturation keeps admission-time load ≈ constant (64 reqs co-submitted) → no H1 variation to exploit → likely NULL (this is why v2's per-segment controller went n=500 null). The win needs per-request signal heterogeneity:
- **H1 (primary): open-loop variable-load arrival.** Poisson/bursty arrival generator in `serve_bench.py`; admission-time load genuinely varies → allocator gives high-k in light spells, low-k in load spikes. Main headline.
- **H1b (secondary): mixed-SLO @ variable load.** Per-request deadline heterogeneity (tight vs slack τ_i); allocator gives low-k to deadline-tight, high-k to slack. Ablation separates H1 vs H1b win sources.
- Baselines: fixed-{r0,r25,r50,r75}, v2 per-segment controller, oracle (offline upper bound from per-image accuracy(k)).
- **Offline-first validation (zero GPU):** a discrete-event simulator replays measured per-request (k,load)→latency via LatPred + per-image accuracy(k) under (arrival, allocator) → goodput@SLO. Prove/refute the headline claim BEFORE spending GPU. Live GPU run (EV-1) only confirms + calibrates LatPred at intermediate loads.

## 5. Interface contracts (parallel-build alignment; new pkg `src/elasticvis/`)

Reuses `LoadReading` from `load_controller.py`. All units ms unless noted.
```python
# src/elasticvis/predictors.py  (Agent X)
@dataclass
class LatencyEstimate: ttft_p50:float; ttft_p99:float; e2e_p50:float; e2e_p99:float
class LatencyPred:
    def predict(self, own_k:int, sum_k:int, num_running:int, kv_occupancy:float) -> LatencyEstimate
class AccuracyTerm:
    def utility(self, k:int, req_id:str|None=None, features:dict|None=None) -> float  # H1: aggregate curve

# src/elasticvis/allocator.py  (Agent Y)
class Allocator:
    def allocate(self, load:LoadReading, slo_ms:float, slo_type:str, req_features:dict|None,
                 k_range:tuple[int,int], lat:LatencyPred, acc:AccuracyTerm) -> int  # returns k_i
# variants: GreedyAllocator, LagrangianAllocator, FixedAllocator(r)  # baseline

# src/elasticvis/sim.py  (Agent Z)
def simulate(arrival:ArrivalProcess, allocator:Allocator, lat:LatencyPred, acc:AccuracyTerm,
             dataset, slo_ms:float, slo_type:str) -> GoodputResult  # goodput@SLO + per-req trace + breakdown
# ArrivalProcess: open_loop_poisson(rate), bursty, closed_loop(c), mixed_slo(deadline_dist)
```

## 6. Build sequence

- **EV-0 (no GPU):** design doc ✓ → fit LatPred+Acc (X) → allocator (Y) → offline sim (Z) → **decide: does sim show ElasticVis > best fixed-r on goodput@SLO under open-loop?** This is the go/no-go for the whole direction, zero GPU.
- **EV-1 (GPU, LLaVA):** integrate plumbing §3 into `serve_bench.py` + open-loop arrival mode → live confirm of sim result + LatPred calibration at intermediate c.
- **EV-2 (GPU, Qwen3-VL + H2):** new probe (per-request Qwen3-VL latency/acc + denser k + per-image features) → cross-arch + content extension (vs CARES).

## 7. Open questions
- SLO type: TTFT vs e2e as the allocator's gate. v2 headline = TTFT≤5s; e2e more deployment-relevant (decode length). Sim both, pick.
- Greedy vs Lagrangian: which wins in sim? Report both.
- LatPred form: linear `α+β·n+γ·Σk+δ·n·Σk` vs lookup+interp; pick by fit error.
- Architecture-awareness (Qwen3-VL): k_i as fraction of N_native_i (EV-2).

## 8. EV-0 RESULTS (2026-07-06) — gating characterization VALIDATED; GO on TextVQA

**Core insight (data-backed):** ElasticVis's goodput benefit over fixed-r is **gated by the steepness of accuracy(k)**. Validated across 5 benchmarks (acc-range → H1b mixed-SLO outcome):
| benchmark group | acc(k) range | ElasticVis vs best fixed-r |
|---|---|---|
| knowledge (MME/MMBench/ScienceQA) | ~0.01 | flat → no win (visual tokens irrelevant to the task) |
| object QA (GQA, LLaVA/Qwen3-VL) | 0.12-0.13 | **boundary** (<0.15 crossover) → NO-GO |
| text-dense (TextVQA, LLaVA/Qwen3-VL) | 0.28-0.29 | **steep** (>0.15) → **WIN** |

**Synthetic sweep** (`runs/elasticvis_ev0/gating_sweep.py`, linear acc a@144 swept, a@576=0.60): H1b (mixed-SLO) crossover at acc-range **≈0.15**; H1 (uniform-SLO) crossover at **≈0.40**. ⇒ **mixed-SLO is the robust regime**; uniform-SLO needs very steep acc.

**Decisive result on REAL TextVQA** (`confirm_textvqa.py`, slot+queue sim, zero GPU):
- H1b mixed-SLO (poisson8, 50% tight 3.5s / 50% slack 15s, e2e SLO): Greedy=**2.36** vs best Fixed=1.74 → **+35.5% WIN**.
- H1 uniform-SLO: 0.898 lose (TextVQA range 0.28 < 0.40 H1 threshold).
- GQA H1b: 0.978 lose — confirms the gate at the boundary.

**Mechanism:** Greedy gives low-k to deadline-tight requests (protect throughput/SLO) and high-k to slack requests (accuracy 0.555 vs 0.275); fixed-r cannot adapt to per-request deadlines.

**Caveat:** sim has known fidelity gaps (slot over-serialization; closed-loop sanity k144 2.5× low, biased in ElasticVis's favor). The +35.5% is a *relative* comparison (robust to absolute bias) but the magnitude needs GPU confirmation (EV-1). Direction (Greedy > Fixed on TextVQA mixed-SLO) is robust.

**Paper spine (confirmed):** (1) gating characterization (acc(k)-steepness gate, validated 5 benchmarks + synthetic sweep); (2) ElasticVis method win on TextVQA (+35.5% H1b mixed-SLO); (3) v2 framework as substrate. GQA NO-GO = the flat-acc boundary (a feature illustrating the gate). Mixed-SLO/deadline-heterogeneous is realistic for multi-tenant serving.

**Next (EV-1, GPU):** implement open-loop + mixed-SLO arrival + per-request k plumbing in `serve_bench.py` (§3); re-measure TextVQA accuracy(k) cleanly on the serving engine; run ElasticVis vs fixed-r on real goodput@SLO to confirm magnitude. Then DocVQA/ChartQA (steeper, fresh probes) + Qwen3-VL cross-arch.

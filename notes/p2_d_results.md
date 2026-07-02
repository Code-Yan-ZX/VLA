# P2 Method-D Validation: adaptive vs fixed (load-adaptive prune-rate budget)

> The method's headline claim: a prune rate `r` that ADAPTS to the engine's
> runtime load (KV-occupancy / num-running) Pareto-dominates any FIXED prune
> rate on the served-req/s ↔ accuracy frontier -- it prunes MORE under high
> load (where M2 showed the req/s speedup is largest: r75 1.26×→1.76× as
> concurrency rises 1→12, KV-pressure relief amplifying with prune depth) and
> LESS under light load (where accuracy matters and throughput isn't the
> bottleneck). This is the serving-aware contribution (0/37 prior methods
> condition compression on engine state).
>
> D-scope LOCKED by `notes/p2_d_measurements.md` (M1: vision tower = 6.6% of
> prefill → SKIP early-prune; M2: speedup grows with concurrency → BUILD
> load-adaptive budget). Selector = proxy (hidden-state deviation; selector
> chase closed). Accuracy guardrail: r_max ≤ 0.50 (GQA r75 drops acc ~11%;
> r50 ~-2%).

## Engine-load read (the integration crux) — SOLVED in vLLM V0

V0 runs the model in-process, so the scheduler is reachable. The load-read path
(`src/load_controller.py:read_engine_load`):

```
llm.llm_engine.scheduler            # list[Scheduler], one per TP rank (len 1 for us)
llm.llm_engine.scheduler[0].running # deque[SequenceGroup]  -> len() = num running seqs
llm.llm_engine.scheduler[0].waiting # deque                  -> num waiting (backpressure)
llm.llm_engine.scheduler[0].swapped # deque                  -> num swapped (KV-pressure)
llm.llm_engine.scheduler[0].block_manager.get_num_free_gpu_blocks()  # free KV blocks
llm.llm_engine.scheduler[0].block_manager.num_total_gpu_blocks       # total KV blocks
kv_occupancy = 1 - free/total
```

Both KV-occupancy AND num-running are reachable cleanly. **No fallback needed.**
This is the primary signal (KV-occupancy is the M2 bottleneck — req/s is
KV-pressure-bound at high concurrency).

## Controller policy

Piecewise-linear map load → r ∈ [r_min, r_max] (`src/load_controller.py`):
- `occ < occ_lo (0.40)` → r_min (0.25)            [light load: keep accuracy]
- `occ > occ_hi (0.70)` → r_max (0.50)            [heavy load: max KV relief]
- `occ_lo ≤ occ ≤ occ_hi` → linear interp r_min → r_max
- Fallback signal `num_running` (run_lo=4, run_hi=8) if KV-occupancy unreadable.
- All thresholds/bounds are CLI args (`--r-min/--r-max/--occ-lo/--occ-hi`).

**Per-request plumbing**: the patched `get_num_image_tokens` reads a MUTABLE
`k_cell["k"]` each call; `run()` updates `k_cell["k"]` from
`controller.decide_r(read_engine_load())` before every submission (segment or
request). The projector hook ALSO reads `k_cell` so the kept-count matches the
per-request placeholder count exactly. The `realized[]` log accumulates
(r, reading) pairs → `realized_summary` gives the r distribution (adaptation proof).

## Varying-load profiles (`--load-profile {constant,bursty,step}`)

The adaptive benefit only appears under VARYING load (constant max → just use
r_max; constant low → r_min). Three generators in `src/load_controller.py`:
- `constant` : one big batch (M2's constant-high case; controller → ~r_max).
- `bursty`   : small bursts (default 4) with short gaps (0.3s) so prior bursts'
  decode overlaps the next arrival → residual KV-occupancy rises across bursts.
- `step`     : low-rate → high-rate → low-rate staircase (cleanest visualization).

## Validation matrix (GQA, n=100, c12, bursty; quick check)

<TO FILL from scripts/analyze_d.py output once runs/p2_d/dval_*.json complete>

| config | req/s | wall_s | acc | kept{min,max} | realized r (mean/min/max) |
|---|---|---|---|---|---|
| adaptive (bursty) | TBD | | | | |
| fixed r25 (bursty) | TBD | | | | |
| fixed r50 (bursty) | TBD | | | | |
| adaptive (constant) | TBD | | | | |

### Pareto-dominance verdict
<TBD: does adaptive beat fixed-r25 on req/s AND fixed-r50 on acc?>

### Realized-r distribution (adaptation proof)
<TBD: did the controller actually adapt (r_min ≠ r_max in realized)? what
occupancy did it observe?>

## Artifacts
- Code: `src/load_controller.py` (controller + profiles), `src/serve_bench.py`
  (adaptive mode + k_cell plumbing). Commit: <TBD>.
- Quick-check outputs: `runs/p2_d/dval_*.json` + `.log`.
- Full n=200 + TextVQA + step: queued in `notes/d_method_jobs.json` for Main.
- Analyzer: `scripts/analyze_d.py`.

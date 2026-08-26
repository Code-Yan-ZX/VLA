# V2 P0 — V1 Table A + F1-on-V1 Verdict

> Subagent output. V1 migration of the served-throughput measurement. LLaVA-1.5-7B,
> GQA, n=100, 1× A40, env `qwen3vl_clean` (vllm 0.19.0, torch 2.10.0+cu128, py3.10).
> Compare against V0 Table A + B1 in `eval/final_results.md`. Commit bff6871.

## TL;DR (3 lines)
- **F1 HOLDS on V1, and the concurrency amplification is STRONGER.** r75/r0 grows
  1.21× (c1) → **1.86× (c12)** on V1, a +0.65 concurrency bonus vs V0's +0.49.
- **Headline c12/r75 = 1.86× served req/s** on V1 (13.27/7.13), vs V0's 1.75×.
- V1's chunked-prefill scheduler LOWERS the prune speedup at low concurrency
  (c1 r75 1.21× vs V0 1.26× — prefill is amortized, less to save) but RAISES it
  at high concurrency (c12 r75 1.86× vs 1.75× — more efficient packing when KV
  is relieved). The KV-cache/concurrency mechanism is robust to V1's scheduler.

## 1. V1 pruning integration (the §4.3 contribution)

**Path taken: in-process EngineCore + processor-level fixed-k (NOT a subprocess plugin).**

vLLM 0.19 defaults `VLLM_ENABLE_V1_MULTIPROCESSING=1` (EngineCore in a spawned
subprocess), which makes the V0 in-process projector forward-hook and
`scheduler[0].running` reads unreachable. BUT the measurement only needs the V1
**scheduler** (chunked prefill, prefix caching — the features that could change
F1), not the subprocess isolation per se. Setting `VLLM_ENABLE_V1_MULTIPROCESSING=0`
runs EngineCore **in-process** (`vllm/v1/engine/llm_engine.py:131`:
`self.model_executor = self.engine_core.engine_core.model_executor`), so:

- the model is reachable via the SAME attribute chain as V0
  (`llm.llm_engine.model_executor.driver_worker.model_runner.model` →
  `LlavaForConditionalGeneration`); the projector forward-hook + vision-tower
  hook work UNCHANGED (instance-level, in-process).
- `LlavaProcessingInfo.get_num_image_tokens` patch (the V0 `patch_image_token_count`,
  same class in 0.19) shrinks the placeholder count to k=(1-r)×576 in the
  multimodal **processor** (main process); the projector hook prunes the output
  to k to match. Verified `runs/v1_probe.py`: at k=288 the vision tower emits 576
  (`in_shape=(1,576,4096)`), the hook prunes to 288 (`out_shape=(1,288,4096)`),
  and `llm.chat()` succeeds with no shape mismatch ⇒ the LLM input sequence is
  GENUINELY 288-shorter (contiguous compaction, not pad-repeat).
- `llm.get_metrics()` returns a populated Prometheus snapshot for the controller
  (`vllm:num_requests_running` gauge peaks at the full `max_num_seqs` under load
  — verified peak=12.0 at c12; full [0, max_num_seqs] range). `gpu_cache_usage_perc`
  needs `kv_cache_metrics=True` (off by default); `num_running` is the primary
  signal (matches V0's P3-step-1 default) and works.

**Scheduler-equivalence justification (load-bearing for the F1 claim):** V1
scheduler code (`vllm/v1/core/sched/scheduler.py`) and `enable_chunked_prefill`
(a `SchedulerConfig` field, default True) are IDENTICAL in both multiproc modes —
only the IPC wrapper (`InprocClient` vs `SyncMPClient`) differs. So the F1
measurement under multiproc=0 reflects V1's scheduler, not V0's. For server-mode
deployment (`vllm serve`), a subprocess plugin (`collective_rpc`-installed
forward-hook) is the production path; outlined as future work.

**Fallback NOT needed:** the task anticipated a possible subprocess-plugin
fallback; the in-process path made it unnecessary. Cleaner §4.3 story than a
subprocess plugin (insight: multiprocessing is an isolation knob orthogonal to
the scheduler, and disabling it for measurement loses nothing scientifically).

## 2. V1 Table A — served-throughput matrix (batch-submit, n=100)

| max_num_seqs | r0 req/s | r50 req/s | r75 req/s | r50/r0 | r75/r0 | V0 r50/r0 | V0 r75/r0 |
|---|---|---|---|---|---|---|---|
| c1  | 2.036 | 2.213 | 2.469 | 1.09× | 1.21× | 1.17× | 1.26× |
| c4  | 4.969 | 6.208 | 7.491 | 1.25× | 1.51× | 1.46× | — |
| c12 | 7.128 | 10.113 | **13.268** | 1.42× | **1.86×** | 1.42× | 1.75× |

**Concurrency amplification (the F1 mechanism, r/r0 c1→c12):**
- V1 r50: 1.09× → 1.42× (bonus +0.33)
- V1 r75: 1.21× → **1.86× (bonus +0.65)**  ← vs V0 r75: 1.26× → 1.75× (bonus +0.49)

**The V1-vs-V0 crossover:** at c1 V1's speedup is LOWER than V0 (1.21 vs 1.26),
at c12 V1's is HIGHER (1.86 vs 1.75). V1's chunked prefill amortizes prefill at
low load (less to save) but its more efficient packing amplifies the KV-relief
payoff at high load. The concurrency-bonus curve is STEEPER on V1.

## 3. ★ F1 verdict on V1

**(a) Concurrency amplification (the robust F1 mechanism, batch Table A):**
**F1 HOLDS and is STRONGER on V1.** The prune speedup grows MORE with concurrency
on V1 (r75 bonus +0.65 vs V0 +0.49). The KV-cache/concurrency-amplification
mechanism is robust to — even amplified by — V1's prefill/decode-disaggregated +
chunked-prefill scheduler.

**(b) Serial c1 e2e vs prefill-TTFT (the V0 B1 framing):**

| prune r | V1 e2e× | V1 prefill× | V1 gap | V0 e2e× | V0 prefill× | V0 gap |
|---|---|---|---|---|---|---|
| r50 | 1.15× | 1.08× | **+0.08** | 1.33× | 1.24× | +0.09 |
| r75 | 1.28× | 1.21× | **+0.07** | 1.43× | 1.30× | +0.13 |

e2e > prefill at both prune rates on V1 (gap positive). The gap is smaller than
V0 at r75 (+0.07 vs +0.13) because V1's chunked prefill already makes prefill
efficient (less e2e-prefill headroom at c1), but the SIGN holds. Note: the serial
absolute speedups are lower on V1 (r75 e2e 1.28× vs V0 1.43×) for the same reason
— V1's prefill is already amortized so pruning saves less per-request.

## 4. V1-vs-V0 headline deltas

- **c12/r75 served speedup: V1 1.86× vs V0 1.75× (+0.11).**
- Concurrency amplification r75 (c1→c12 bonus): V1 +0.65 vs V0 +0.49.
- c12/r50: V1 1.42× = V0 1.42× (identical — the r50 concurrency endpoint is stable).
- Raw throughput: V1 r0 c12 = 7.13 req/s vs V0 5.75 (V1 ~1.24× faster baseline —
  V1 is a more optimized engine, as expected).

## 5. Accuracy sanity

V1 GQA acc (n=100): r0=0.60, r50=0.52, r75=0.47 — consistent across c1/c4/c12
(throughput is concurrency-dependent, accuracy is not). V0 (Table C-n500):
r0≈0.585, r50≈0.565, r75 lower. V1's r50 acc (0.52) is a touch below V0 (0.565)
but within n=100 noise; the pruning IS working (kept=288/144 at r50/r75, verified
by hook logs). Accuracy is not the focus of P0 (the throughput measurement is).

## 6. Environment + reproduction

- env: `qwen3vl_clean` (vllm 0.19.0, torch 2.10.0+cu128, py3.10). GPU: 1× A40 46GB.
- reproduce: `bash runs/v2_p0_run_matrix.sh` (9 batch cells + 3 serial cells);
  analyze: `python runs/v2_p0_analyze.py` (or see §2-4 above).
- commit: bff6871 (V1 port of src/serve_bench.py + src/load_controller.py).
- probe (integration proof): `runs/v1_probe.py` → `runs/v1_probe.log`.
- total GPU: ~13 min for all 12 cells (each cell = fresh process).

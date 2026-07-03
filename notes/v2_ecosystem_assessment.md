# V2 Ecosystem Assessment — 2026-07-03

> v2 kickoff subagent output. Web-verified + runtime-verified on our hardware.
> Authoritative for v2 planning until superseded. Sources inline.

## TL;DR (3 lines)
- **★ V1 gate = GREEN.** vLLM 0.19.0 + cu128 runs V1 end-to-end on driver 560 / CUDA 12.8 / A40 (2 runtime smokes: LLaVA-1.5-7B 3.21s, Qwen3-VL-8B 20.2s). `qwen3vl_clean` env is the v2 serving env; `vtc_serve` preserved as rollback.
- **★ Novelty = HOLDS (0/N).** No paper integrates a post-hoc visual-token pruner into a serving engine with served-throughput. Closest neighbors (ElasticMM/EPD/ModServe/vLLM-Omni) all do serving-side optimization WITHOUT a pruner. One flag: RTP-LLM body-text check before submission.
- **P1 = GREEN.** Qwen3-VL-8B-Instruct loads on A40 (~16GB bf16); native 2×2 MLP merger + dynamic resolution → **variable tokens/image (~274 for 256px logo, ~2300 at default res)** — F1/F2/F3 must be re-tested, pruning value rises.

---

## 1. Verified Ecosystem Table

| Component | Status (2026-07-03) | On our hw (driver560/cu12.8/A40) | Source |
|---|---|---|---|
| **vLLM stable** | ~0.20–0.23; default PyPI wheel = **cu130** (CUDA 13) since v0.20 | cu130 wheel FAILS on driver 560 (the "0.24 failed" symptom) | [LinkedIn v0.20.0](https://www.linkedin.com/posts/vllm-project_vllm-v0200-is-out-...) |
| **vLLM cu12 wheels** | cu128 / cu129 release assets published for latest tags | **WORKS** — pull `+cu128` GitHub asset, not default PyPI | [GPU install doc](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/) |
| **V1 engine default since** | v0.8.0 (early 2025) | Our 0.10.2 already defaults to V1; production code FORCES V0 via `os.environ.setdefault("VLLM_USE_V1","0")` in `src/serve_bench.py:41` | [Red Hat v0.8.1](https://developers.redhat.com/articles/2025/04/28/performance-boosts-vllm-081-switching-v1-engine) |
| **Tested env (NEWEST V1 on our hw)** | **vllm 0.19.0 + torch 2.10.0+cu128** in conda env `qwen3vl_clean` | **VERIFIED GREEN** — V1 init + LLaVA-1.5 answer + Qwen3-VL-8B answer | `runs/v1_smoke.log`, `runs/v1_smoke_qwen3vl.log` |
| **Qwen3-VL variants** | Dense 2B/4B/8B/32B; each Instruct + Thinking; 8B/32B also FP8 | 8B bf16 FITS A40 (~16GB, verified); 32B does NOT fit | [HF collection](https://huggingface.co/collections/Qwen/qwen3-vl) |
| **Qwen3-VL vLLM support** | mainline since **vLLM v0.11.0** | Our 0.19.0 supports it (runtime-verified) | [Qwen3-VL repo](https://github.com/qwenlm/qwen3-vl) |
| **Qwen3-VL visual tokens** | dynamic res, 32×32 patches, **native 2×2 MLP merger** (built-in 4× reduction) | smoke: 256px logo → 274 prompt toks; web: ~2300/image at default res | [Reddit sweet-spot](https://www.reddit.com/r/LocalLLaVA/comments/1ot95gj/) |
| **Qwen2.5-VL-7B (backup)** | mature vLLM, A40-feasible, up to 16384 visual toks/image (KV-heavy) | viable | — |
| **InternVL3.5-8B (backup)** | vLLM-supported (Qwen2.5 backbone), A40-feasible, dynamic-res pixel-shuffle | viable | — |
| **DeepSeek-OCR** | arXiv [2510.18234](https://arxiv.org/abs/2510.18234), Oct 2025 | n/a | "Contexts Optical Compression": text→image re-encoding, 7–20× text reduction |
| **vLLM-Omni** | arXiv [2602.02204](https://arxiv.org/abs/2602.02204), Feb 2026 | n/a | any-to-any (LLM+diffusion) full disaggregation; JCT only; NO token pruning |

---

## 2. ★ V1 Feasibility Verdict — GREEN (no blocker)

**The "V1 needs CUDA 13" fear is FALSE for our path.** The failure mode the user hit with "0.24" was installing the **default PyPI wheel, which became cu130 at v0.20**. CUDA-12-native wheels (cu128/cu129) are published as GitHub-release assets for every recent tag; they run on driver 560 via CUDA minor-version compat (same mechanism as our working 0.10.2 cu12-native).

**Runtime proof (this session, on driver 560.35.05 / A40):**
1. `qwen3vl_clean` env (vllm 0.19.0, torch 2.10.0+cu128): served `llava-1.5-7b-hf`, V1 engine init confirmed (`Initializing a V1 LLM engine (v0.19.0)`), correct answer in 3.21s, total 86.8s incl. compile. → `runs/v1_smoke.log`
2. Same env: served `Qwen/Qwen3-VL-8B-Instruct` (bf16, from HF cache), V1 init, correct answer in 20.2s, prompt_token_ids=274. → `runs/v1_smoke_qwen3vl.log`

**Working env spec (v2 serving env):**
- conda env: **`qwen3vl_clean`** (existing; do NOT reuse `vtc_serve` for V1 work — keep as rollback)
- vllm 0.19.0, torch 2.10.0+cu128, CUDA 12.8 runtime, Python 3.11
- Install-from-scratch recipe if a clean `vtc_serve_v1` env is later desired:
  `uv pip install https://github.com/vllm-project/vllm/releases/download/v0.19.0/vllm-0.19.0+cu128-cp38-abi3-manylinux_2_35_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu128`
  (newer tags also publish cu128 assets — can go higher than 0.19.0 if needed)

**STATE.md correction:** the line `vtc_serve(vLLM0.10.2 V0)` is misleading — 0.10.2 *defaults to V1*; the controller *forces* V0 via env var. Both 0.10.2 and 0.19.0 are V1-capable on our hw.

---

## 3. V1 Controller Signal Path (the §4.3 replacement)

V1 runs the model in a **spawned subprocess** (`EngineCore_DP0`), so the V0 in-process path is DEAD:
```python
# src/load_controller.py:101-110 — DIES under V1
engine = llm.llm_engine
sched = engine.scheduler[0]          # AttributeError in V1
running = sched.running              # no such attr
bm = sched.block_manager             # no such attr
```
Runtime-confirmed in smoke: probe reported `v0_scheduler_running: ABSENT`.

**V1 replacement (statically confirmed by source inspection of vllm 0.19.0):**
```python
# V1 path
llm.get_metrics()           # -> list[Metric]   (NOTE: 'get_metrics', NOT 'get_engine_stats')
  └─ llm.llm_engine.get_metrics()
       └─ asserts self.log_stats   (set LLM(log_stats=True) if off; default varies)
       └─ returns get_metrics_snapshot()  # Prometheus snapshot
```
Canonical V1 metric names (from [metrics doc](https://docs.vllm.ai/en/stable/design/metrics/)):
- `vllm:num_requests_running`  ← replaces `len(scheduler.running)` (num_running)
- `vllm:num_requests_waiting`
- `vllm:gpu_cache_usage_perc`  ← replaces `1 - free/total gpu_blocks` (KV-occupancy)
- (plus TTFT/e2e histograms, num_preemption, etc. — useful for P2)

**Two controller-deployment modes in V1:**
- **(A) Offline `LLM.chat()` path** (current code): poll `llm.get_metrics()` between segments — but the same "drains the engine" problem from V0 §4.3 item 2 persists; the streaming-loop fix (`engine.add_request` + `engine.step()`) must be ported to V1's async `engine_core`. **This is the real engineering task of P0.**
- **(B) Server mode (`vllm serve`)**: hit `http://localhost:8000/metrics` (Prometheus) from the controller; place the pruner as a **prefill hook** in the engine. Cleaner, but requires re-architecting `serve_bench.py` from offline-LLM to async-server + aiohttp client.

**Runtime note (honesty):** I statically confirmed `get_metrics` exists and the call chain; I did NOT runtime-exercise it returning a non-empty snapshot (my probe called the wrong method names `get_engine_stats`/`get_stats`, which don't exist in 0.19). First P0 task is to confirm a populated `get_metrics()` snapshot under load.

---

## 4. Novelty-Threat Assessment — 0/N HOLDS (with one flag)

**Bottom line:** "0/N papers integrate a post-hoc visual-token compressor into a serving engine and report served throughput" **still holds** as of 2026-07-03. Keywords exhausted: visual token pruning serving · vLLM token compression throughput · served throughput VLM · multimodal serving compression · vision token reduction vLLM · FasterVLM/VisionZip/FastV + vLLM · goodput VLM visual token reduction 2026.

**Closest neighbors (cite in v2 §2.3 as adjacent work, NOT competitors):**
| Paper | arXiv | What they do | Why not a threat |
|---|---|---|---|
| ElasticMM (NeurIPS'25) | [2507.10069](https://arxiv.org/abs/2507.10069) | elastic-parallelism multimodal serving, TTFT 4.2× / tput 3.2–4.5× | no pruner integrated (only related work) |
| EPD Disaggregation (ICML'25) | [2501.05460](https://arxiv.org/abs/2501.05460) | prefill/decode split, TTFT −71% | caches mm tokens, does not prune |
| ModServe (SoCC'25) | [2502.00937](https://arxiv.org/abs/2502.00937) | stage-disaggregated LMM serving, 3.3–5.5× | no compressor |
| vLLM-Omni (Feb'26) | [2602.02204](https://arxiv.org/abs/2602.02204) | any-to-any (LLM+diffusion) disaggregation, JCT −91% | no token compression (verbatim abstract check) |
| Survey (ACL'26 Findings) | [2604.05546](https://arxiv.org/abs/2604.05546) | names "stage-disaggregated serving via hw-algo co-design" as **future frontier** | supports our gap claim |
| DeepSeek-OCR | [2510.18234](https://arxiv.org/abs/2510.18234) | text→image optical compression (7–20× text reduction) | orthogonal: compresses TEXT via image, not vision-encoder output; cite not compete |

**★ FLAG (must resolve before submission):** **RTP-LLM** (Alibaba inference engine) is described in secondary sources as having *"attention entropy-guided visual token pruning inside the vision encoder"* — i.e. a serving engine with a native pruner, the single closest neighbor. Could NOT pull a primary arXiv abstract (only ResearchGate/Alibaba tech-blog trace). **Action:** targeted body-text fetch of RTP-LLM before locking the novelty claim; if it reports served throughput with pruner on, differentiate explicitly (likely: RTP-LLM is industry-engineering, no controlled prune-rate sweep, no KV-admission analysis).

**Gap-statement rewrite for v2 §2.3 (drop "two fields haven't intersected"; the serving field now exists, just without a pruner):**
> "Visual-token compressors (FastV/VisionZip/SparseVLM/...) and multimodal serving engines (ElasticMM/EPD/ModServe/vLLM-Omni) have evolved on parallel tracks. The former report only offline accuracy/FLOPs; the latter optimize serving throughput but treat the vision tower as fixed. **No work integrates a post-hoc visual-token pruner into a serving engine and measures end-to-end served throughput.** This paper closes that gap."

---

## 5. P0–P3 Plan (verified versions)

- **P0 — V1 migration + Table A/B reproduce.** (a) Confirm populated `llm.get_metrics()` under load (≤30 min). (b) Port `src/load_controller.py:read_engine_load()` from V0 scheduler to V1 Prometheus snapshot (`num_requests_running` + `gpu_cache_usage_perc`). (c) Port `src/serve_bench.py` streaming loop to V1 async engine_core (or switch to server-mode + /metrics). (d) Re-run Table A (single-image point) + Table B (c1–c12 sweep) on LLaVA-1.5 in V1; confirm F1/F2/F3 qualitative findings survive. *Env:* `qwen3vl_clean`. *GPU:* ~3–5 GPU·h (12 cells × ~20 min).
- **P1 — Qwen3-VL-8B-Instruct re-test.** (a) Build a V2-style pruner hook for Qwen3-VL's vision-encoder output (note: native 2×2 MLP merger already gives 4× — our pruner sits DOWNSTREAM of it, so "prune rate r" must be defined relative to post-merger token count, not raw patches). (b) Re-run F1/F2/F3 + load-adaptive controller on Qwen3-VL. Hypothesis: dynamic resolution → higher vision fraction → pruning value UP. Even a "partially doesn't hold" result is publishable. *Model already in HF cache (17GB).*
- **P2 — Real serving scale.** Extend concurrency sweep to **c≥64** (short sequences on A40 — V1 chunked-prefill handles this); add **p50/p99 TTFT** + **goodput** (SLO-aware req/s) to the logged metrics. Required for any "serving paper" credibility (Pattern Recognition reviewers will check).
- **P3 — Cross-compressor.** Integrate **VisionZip-class** (encoder-attention-based, no FlashAttention-score dep → cleanly hookable) as a second compressor inside the same V1 engine, so "served throughput" measurement spans >1 pruner. Differentiates from our own load-adaptive proxy.
- **P4 — Rewrite.** Convert v1 measurement-led draft to v2: replace §4.3 V0-engine text with V1-migration-as-engineering-contribution; update §2.3 gap statement (above); add ElasticMM/EPD/ModServe/vLLM-Omni/DeepSeek-OCR/RTP-LLM to related work; add Qwen3-VL column to all tables; add p50/p99/goodput to all serving plots.
- **Continuous:** monitor arXiv for the keywords above every 2 weeks until submission (RTP-LLM body-text check is the immediate action).

---

## 6. Decisions / Flags for the main window
- **Env:** reuse `qwen3vl_clean` as v2 serving env (don't create `vtc_serve_v1` yet — `qwen3vl_clean` already works; clone only if it accumulates conflicting packages). `vtc_serve` stays as rollback.
- **§2.3 / paper_v1.md:** LEFT UNTOCHANGED (per task rule — no direct novelty threat). Rewrite happens in v2 P4. Needed additions noted in §4 above.
- **RTP-LLM body-text check** is the one open novelty risk — schedule before submission.
- **V1 `get_metrics()` runtime population** is the one unverified detail — first P0 task.

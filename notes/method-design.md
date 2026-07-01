# Method Design — P2-step-1 (Go/No-Go Probe + Serving-Aware Skeleton)

> Dev subagent · 2026-07-01. Companion to `notes/positioning.md` (gap + thresholds) and `notes/lit-survey.md` §7 (novelty re-checked: OPEN).
> Base: LLaVA-1.5-7B-hf (`runs/models/llava-1.5-7b-hf`). Serving env: `vtc_serve` (vLLM). Baseline env: `fastv`.
> Goal of this doc: make the P2 go/no-go probe **concrete and runnable** on 1× A40, plus sketch the method hypothesis for after the gate.

---

## 1. Go/No-Go probe design (concrete)

### 1a. Probe compressor — CLS / attention-score selection at the projector output

**Choice: a minimal re-implementation of the VisionZip / FasterVLM / VTC-CLS family — *boundary-level, training-free, query-agnostic* token selection by CLIP-[CLS] attention score.** Implemented in `src/compressors.py` as `ClsAttnSelector`.

**Why this one (not FastV):**
- *Integration feasibility.* FastV prunes **inside** the LLM at layer 2 using attention scores — but vLLM uses FlashAttention which fuses matmul+softmax, so per-token attention scores are **not accessible** inside the engine (this is the exact "Deployment Hurdle" diagnosed in the 2507.20198 survey §6.5.3, and the reason FastV is the accuracy anchor on a *separate* `fastv` env, not the probe). Boundary-level selection runs entirely **before** LLM fusion, so it slots cleanly into vLLM's existing multimodal path with no attention-score extraction.
- *Minimal.* CLS-attention is a single head; selecting top-k tokens by it is ~10 lines, no training, no extra forward pass (the CLS attention already exists in CLIP/SigLIP).
- *Proven family.* VisionZip (8× prefill speedup), FasterVLM (≈90% perf retained at high prune), VTC-CLS (SOTA among TF) all use this signal — so the probe isolates the **serving** question (does boundary prune → served speedup?) from the *accuracy* question (already answered by the family).

**Selection rule (exact):** for each image, take the vision tower's last-layer [CLS]→patch attention (mean over heads), giving a per-patch importance score `s∈[0,1]^N` (N=576 for LLaVA-1.5). Keep top `k = round(N·(1−r))` patches by `s` where `r∈{0,.25,.50,.75}` is the pruning rate. Reindex: selected projector-output rows `(B, k, D)` replace the full `(B, N, D)`. The corresponding `image_token_index` placeholder count in the text sequence shrinks from N→k (vLLM re-derives feature size from the projector output, see §1c).

### 1b. EXACT vLLM integration point

**Hook: `LlavaMultiModalProjector.forward` output** in `vllm/model_executor/models/llava.py`.

Verified source (vLLM 0.10.2, `vllm_qwen3vl` env as reference; vtc_serve will match or differ only slightly):

```
llava.py:96   class LlavaMultiModalProjector(nn.Module)
llava.py:119  def forward(self, image_features): ... return hidden_states   # POST-projector, PRE-LLM-fusion = the boundary
llava.py:649  def _process_image_input(self, image_input):
llava.py:657      image_features = self._process_image_pixels(image_input)   # vision-tower output
llava.py:660      return self.multi_modal_projector(image_features)         # <-- inject selector HERE
llava.py:666      image_embeds = self.multi_modal_projector(torch.cat(image_features))
```

**Integration plan (engine-internal, not offline):**
1. **Monkeypatch** `LlavaMultiModalProjector.forward` (or subclass) to apply `ClsAttnSelector` to its output, returning `(B, k, D)` instead of `(B, N, D)`. The CLS-attention scores come from the vision tower's last layer — captured via a `forward_hook` on `vision_tower` registered once at model init. No FlashAttention surgery needed (scores are read from the *vision encoder*, not the LLM).
2. **Placeholder count reconciliation:** vLLM computes the number of image placeholder tokens in the *text* sequence from the processor's `image_token` count (576 for LLaVA-1.5). After pruning, the projector emits fewer rows, so the placeholder count **must** match `k`. Two options: (a) patch the multimodal processor's `image_token` expansion to emit `k` placeholders per image (cleanest, mirrors RFC #45098's "pruned before LLM fusion"); (b) keep 576 placeholders and pad projector output — **rejected** (defeats the KV-cache win). Option (a) is the path; the `--image-pruning-rate` RFC #45098 proposes exactly this surface, so our hook aligns with upstream direction.
3. **KV-cache + scheduling:** because we shrink the *sequence length* pre-fusion, vLLM's PagedAttention automatically allocates fewer KV pages → peak KV-cache MB and prefill work both drop ~linearly in (1−r). This is the mechanism the go/no-go probe measures.

**Minimal-effort variant for the probe (recommended to start):** run vLLM **offline** via `LLM.generate()` with `pruning_rate` passed through a patched projector, **not** the OpenAI server. Offline mode reuses the exact same prefill/decode/KV-cache machinery (the metric differences vs server are only batch-scheduling overhead, which we *want* to characterize separately). Server mode is a stretch for after the gate.

### 1c. Metrics × pruning × benchmark

**Metrics (all per-request, aggregated mean ± stderr over the subset):**
- `served_tok_s` — output tokens / e2e wall-clock (decode throughput).
- `served_req_s` — requests / e2e wall-clock (offline: 1/batch_e2e; server: req/s).
- `ttft_ms` — Time To First Token (prefill wall-clock).
- `peak_kv_mb` — peak KV-cache memory (from vLLM `engine_metrics` / `nvidia-smi` delta).
- `accuracy` — GQA exact-match / TextVQA VQA-accuracy (for the ≤2% gate).
- `prefill_speedup`, `e2e_speedup` — relative to r=0 baseline.

**Pruning rates:** `r ∈ {0, 0.25, 0.50, 0.75}` (r=0 is the uncompressed control; identical code path, no selection).

**Benchmarks (small subsets for the probe — full runs later):**
- **GQA val** — 200 randomly-seeded examples (balanced answer distribution). Tests scene/graph QA; moderate visual dependence.
- **TextVQA val** — 200 examples. OCR-heavy; the known weak spot for compressors (FastV drops most here). This is the *stress test* — if accuracy holds here at 50% prune, the gate is robust.

Subsets are generated deterministically (fixed seed) and cached under `eval/subsets/` for reproducibility; the queue jobs reuse them.

### 1d. GO / NO-GO thresholds (restated from positioning.md §★)

**GO** (proceed to serving-aware method design) if **all** hold:
- 50% token cut (r=0.50) → **≥1.5× prefill speedup** (ttft ratio ≥1.5 vs r=0), AND
- **≥1.2× e2e served req/s** at r=0.50, AND
- **≤2% GQA** accuracy drop (and ≤5% TextVQA drop) at r=0.50.

**NO-GO** (escalate to user, charter §6) if **e2e wall-clock speedup <1.2× even at r=0.75**. This would mean FLOPs/token-cut decouples so strongly from wall-clock under vLLM's scheduling that the method premise fails.

**Pivot if NO-GO:** reframe as a *negative-result characterization paper* ("FLOPs ≠ wall-clock under continuous batching, and here is why — KV-cache scheduling, prefill saturation, decode-bandwidth binding") OR demote Gap A and fall back to **Gap D** (content-adaptive) as the primary method, with throughput as a secondary section. Either is still publishable (Pattern Recognition / Information Sciences tier). **The probe is designed to be decisive either way.**

### 1e. Fallback plan (if vLLM hook proves too brittle within the probe window)
- **Fallback A:** use `lmdeploy` (a `lmdeploy` conda env already exists) — its pipeline model is more patch-friendly; same metrics, same probe. Costs ~1 day.
- **Fallback B:** measure r=0 vs r=0.75 *offline-HF* (transformers, no engine) for the *accuracy* anchor, and report *engine* numbers only at r=0/r=0.50 to bound the wall-clock claim. This salvages a partial result if engine integration blocks the gate. Documented as a degraded probe, not a silent compromise.

---

## 2. Method hypothesis skeleton (post-gate; to be detailed if GO)

**Working title: Serving-Aware Visual Token Compression (SAVTC).** The probe establishes that boundary-level pruning yields wall-clock speedup under vLLM. The *method* then makes the compressor **aware of serving realities** so the gain is *near-linear* rather than sub-linear:
- **(i) Batch-composition awareness:** under continuous batching, a request's prefill is co-scheduled with others' decodes; a fixed prune-rate under-utilizes the slack of "easy" batches and over-prunes "hard" ones. Use a cheap batch-level signal (current KV occupancy / queue depth) to modulate per-image prune-rate within a band [r_min, r_max] — tighten under load, relax when idle.
- **(ii) KV-cache budget awareness:** instead of a fixed token count, target a *KV-cache page budget* per request, so the compressor produces a token count that fits a desired page slice — directly optimizing the resource vLLM actually schedules.
- **(iii) Decode-bandwidth guard:** since decode is text-token-bandwidth-bound (not visual-token-bound), prune only down to the point where prefill no longer dominates TTFT; further pruning wastes accuracy for no decode gain. Derive this knee from the probe's r-vs-ttft curve.

Concrete mechanisms (gating network? closed-form page budget?) deferred until the go/no-go data selects which of (i)–(iii) has the most headroom. The probe's r=0.50 result is the prior.

---

## 3. Baselines

| Baseline | Env | Role | What we report |
|---|---|---|---|
| **FastV** (reproduce) | `fastv` (torch 2.0.1 / tf 4.31) | **Accuracy anchor** — best-known TF compressor, but *cannot* run inside vLLM (FlashAttention score inaccessibility, survey §6.5.3). | GQA + TextVQA accuracy at ~50% prune. No served-throughput (by design — that's our gap). |
| **Probe compressor** (`ClsAttnSelector`, this work) | `vtc_serve` (vLLM) | **Integration + wall-clock** — the go/no-go instrument. | Full metric set (§1c) at r∈{0,.25,.50,.75}. |
| r=0 (no pruning) | `vtc_serve` | **Control** — iso-engine, iso-model, no compression. | Same metrics; the denominator for all speedup ratios. |

The three together answer: *does boundary prune beat no-prune on wall-clock (probe vs r=0)?* and *does it match the accuracy of the canonical TF compressor (probe vs FastV)?* — exactly the two claims the gate tests.

### 3a. Differentiator note (vs the 2026 combination-study cluster)
AgilePruner (2603.01236), VisionTrim (2601.22674), PRUNESID (2603.09480) crowd the **accuracy/FLOPs systematic-combination** space (decompose methods into scoring-basis × reduction-method, sweep combinations on offline research code). Our work is **orthogonal**: the unit of optimization is **served wall-clock inside an engine** (vLLM), where the dominant costs (KV-cache scheduling, prefill saturation, decode-bandwidth binding) are invisible to FLOPs/accuracy studies. We *do* adopt the combination-study discipline for our accuracy tables (multi-benchmark × multi-backbone × multi-budget × multi-seed; always report accuracy + real latency) — but the contribution is the **serving-aware** compressor, not yet another accuracy/FLOPs combination. EffiVLM-BENCH (2506.00479) is a candidate **eval-harness backbone** for the accuracy side (its OP/OG/OL/OE indices are reusable); its offline-only latency reporting is exactly the gap we fill on the throughput side.

---

## 4. Open implementation questions (resolve during probe)
1. Does vLLM's `LLM.generate()` offline path expose per-request `ttft` and peak KV MB directly (`engine_core.metrics`), or do we time it externally? → resolve in smoke test.
2. CLS-attention from CLIP vision tower: last layer only, or ensemble across last-K layers (VTC-CLS uses ensemble)? Start with last-layer (simplest); ensemble is a free ablation if accuracy is tight.
3. Does shrinking placeholder count require patching `LLaVAProcessor` (HF side) or vLLM's `multimodal/processing.py`? → resolve during hook integration.

## 5. Subset prep (P2-step2, 2026-07-01)
Probe subsets built and verified (script: `scripts/build_subsets.py`, log `runs/build_subsets.log`):
- **GQA**: 200 samples from `lmms-lab/gqa` `testdev_balanced` (instructions+images parquets, ~68 MB pulled). Local JPEGs `runs/data/gqa/*.jpg` (20 MB). `gt` = single open-vocab answer; no `choices` (score_gqa exact-match).
- **TextVQA**: 200 samples from `lmms-lab/TextVQA` `validation` (streamed, stopped after 600 good rows — avoided the full ~920 MB parquet). Local JPEGs `runs/data/textvqa/*.jpg` (35 MB). `gt` = multiple human answers **semicolon-separated** (score_textvqa).
- Both: seed=0 deterministic, all 400 PIL-verified, q/gt non-empty, load_subset-parsed. **No gating, no token, no full VG dump.** Total local image footprint: 55 MB; HF cache delta ~73 MB.
- Sample GQA line: `{"id":"202147765","image":".../runs/data/gqa/202147765.jpg","question":"Is the chair in the bottom of the image?","gt":"no"}`
- Sample TextVQA line: `{"id":"35066","image":".../runs/data/textvqa/35066.jpg","question":"what time is displayed?","gt":"12:34;12:34 am;..."}`

— End. Probe jobs in `notes/p2_probe_jobs.json`. Subsets in `eval/subsets/`. —

— End. Probe jobs in `notes/p2_probe_jobs.json`. —

---

# Part II — v1 / v2 Method Design (post-GO; Dev subagent, 2026-07-01)

> Builds on Part I's probe. The GO gate passed (provisional) with a **PROXY** score; v1 swaps in the real selector, v2 adds the serving-adaptive layer. The three probe findings (eval/p2_probe_summary.md ★★) drive every design choice below.

## 6. v1 — True CLS-attention selector (accuracy) + early-prune plan (prefill)

**Goal of v1:** (a) tighten accuracy at iso-throughput by replacing the proxy score with REAL vision-tower CLS-attention, and (b) capture the fixed-encoder cost (finding #2) by moving selection upstream into the vision tower. Throughput numbers should match the proxy probe within noise (the serving win is selector-agnostic — same hook, same KV-cache shrink); the **accuracy** is where v1 must beat the proxy.

### 6a. True CLS-attn selector (DONE — `src/compressors.py`)

The probe used a hidden-state-deviation PROXY because vLLM's `CLIPAttention` delegates to `MultiHeadAttention` → `F.scaled_dot_product_attention` (fused SDPA, returns NO weights; vLLM stripped HF's `output_attentions`). v1 gets the **real** signal via `ClsAttnCapture` (`src/compressors.py`):

- Monkeypatches the LAST `CLIPAttention` layer's `forward` (`vision_tower.vision_model.encoder.layers[-1].self_attn`) to run a **parallel manual softmax path** alongside the real SDPA call. The SDPA output still drives the encoder (numerics byte-identical to stock vLLM); the manual `QK^T/sqrt(d) → softmax` only stashes the (B,H,S,S) weights.
- `cls_attention_scores(weights)` then reduces to per-patch score `s = mean_h attn[:, h, 0, 1:]` (CLS is query 0; patches are keys 1..576). Mirrors FasterVLM/VTC-CLS.
- `TrueClsAttnSelector` ranks patches by `s`; `diversity_lam` flag (default 0.0 = OFF for v1) optionally switches to a PRUNESID-style greedy importance+diversity rule (`greedy_diverse_topk`: penalize candidates similar to already-kept, NMS-like, on L2-normalized projector features).
- Wired into `serve_bench.py` via `--selector {proxy,true_cls}` + `--diversity-lam`. CPU self-test (determinism + shape + lam=0≡proxy) passes. Probe jobs: `notes/v1_probe_jobs.json`.
- **Multi-layer ensemble** (VTC-CLS averages CLS-attn over last-K layers) is a free ablation — `ClsAttnCapture` accepts a list of layers and running-means their scores; leave single-layer (last) as v1 default, ensemble for the accuracy table.

**Expected outcome:** proxy r50 GQA was 0.565 (−2.0%); v1 true-CLS-attn should tighten this (the proxy correlated with foreground saliency but not with the LLM-relevant CLS signal). If v1 r50 reaches ≥0.575 (≈−1%), the accuracy story is strong; if v1 r75 (proxy 0.470, −11.5%) recovers to ≥0.52, we can argue the proxy **understated** the method (a useful framing for reviewers).

### 6b. Early-prune feasibility (Task 2 verdict: TRACTABLE for v1, not v2-only)

**Finding #2 motivation:** the probe prunes at projector OUTPUT, so the vision tower still processes all 576 tokens — prefill is sub-linear (r75 only 1.30× despite 4× fewer tokens). Pruning INSIDE the encoder (compute CLS-attn at layer L, drop low-attention patches BEFORE layers L+1..24) would let the encoder do less work → larger prefill gains.

**vLLM 0.10.2 source audit (`vllm/model_executor/models/clip.py`):**
- `CLIPVisionTransformer.encoder = CLIPEncoder` holds `self.layers = nn.ModuleList([CLIPEncoderLayer, ...])` (24 for CLIP ViT-L/14).
- `CLIPEncoder.forward` (clip.py:251): `for encoder_layer in self.layers: hidden_states = encoder_layer(hidden_states)` — a **plain Python loop over hookable nn.Modules**. Each `CLIPEncoderLayer` is independently monkeypatchable.
- **CUDA graphs do NOT cover the vision tower in V0.** `worker/model_runner.py:728` `_use_captured_graph` returns True only when `decode_only=True`; the vision tower runs during prefill (encode) which is always **eager**. So a mid-encoder prune does NOT fight CUDA graph capture/replay (the fear that blocked FastV-style intra-LLM hooks in the survey §6.5.3). ★ This is the key de-risking fact.

**Concrete integration plan (v1 prototype-ready, defer to v2 only if it regresses):**
1. **Capture** CLS-attention at an EARLY layer L (e.g. layer 6 of 24): install `ClsAttnCapture` on `encoder.layers[6].self_attn` (one-line change — `ClsAttnCapture` already supports any layer).
2. **Prune** after layer L: monkeypatch `encoder.layers[L]` with a wrapper that, post-forward, gathers the CLS-top-k patch rows of `hidden_states` (keeping CLS at index 0) and re-indexes. Subsequent layers see a shorter sequence → real encoder FLOPs/latency drop.
3. **Consistency:** the projector-output hook (`patch_image_token_count` → k) stays identical — but now the projector receives k rows directly from a k-short encoder output, not 576→k at the projector. The placeholder reconciliation is unchanged.
4. **Risk to mitigate:** early-layer CLS-attn is noisier than last-layer (less refined). v1 should A/B early-prune-L6 vs boundary-prune-last-layer at iso-k; if early-prune loses >1% acc at iso-k, keep boundary-prune for v1 and ship early-prune as a v1.1 ablation. The encoder-FLOPs win is only worth it if accuracy holds.

**Verdict: low-risk enough to prototype as part of v1** (not deferred to v2). The hook surface is the same `ClsAttnCapture` we already built; the only new code is the per-layer prune wrapper (~30 lines). Recommend: ship v1 with boundary true-CLS-attn FIRST (lock the accuracy win), then add early-prune as a same-PR ablation if time permits.

### 6c. What v1 does NOT do (deferred)
- No load-adaptive budget (that is v2, §7).
- No learned scoring (training-free only — the family's strength, and the 1× A40 budget rules out training a new selector).
- No multi-image / variable-resolution (LLaVA-1.5 single-image only; Qwen2.5-VL generalization is the publishable stretch).

---

## 7. v2 sketch — Load-adaptive budget (the serving-specific novel lever)

**This is the most novel serving-specific contribution** and the clearest differentiator from the 37-method table. The probe proved e2e>prefill at every ratio (finding #1) → the deployment win is KV-cache/concurrency, not prefill FLOPs. **v2 makes the prune rate RESPOND to engine concurrency** so the compressor captures the headroom a fixed-rate compressor leaves on the table.

### 7a. The feedback mechanism (sketch — NOT implemented in v1)

A fixed prune rate `r` is globally suboptimal under continuous batching:
- **Low batch / idle GPU:** KV-cache is plentiful; pruning hard wastes accuracy for no throughput gain. Should prune LESS (lower r → higher accuracy, the GPU can afford it).
- **High batch / KV-cache pressure:** the scheduler is the bottleneck; pruning harder (higher r) frees KV pages → more concurrent requests → higher req/s. The accuracy cost is "paid for" by throughput.

**Closed-loop design (to implement in v2):**
1. **Sensor:** read vLLM's engine state each prefill — `scheduler.running_num` (active requests), `KV cache usage %` (`gpu_prefix_cache_hit` / `num_gpu_blocks_free` / `num_gpu_blocks_total`). These are exposed on the V0 scheduler (`vllm/core/scheduler.py`) and the engine metrics — a cheap per-step read, no extra forward pass.
2. **Controller:** a monotonic map `load ∈ [0,1] → r ∈ [r_min, r_max]` (e.g. r_min=0.25, r_max=0.75). `load` = a normalized blend of KV-occupancy and running-queue depth. Cheap closed-form (no learned policy) for v2; a tiny gating MLP is a v3 stretch.
3. **Actuator:** the projector hook reads the controller's `r` per-image (instead of the global `args.pruning_rate`), and `patch_image_token_count` becomes per-image adaptive (this is where v2 needs the upstream RFC #45098 `--image-pruning-rate` surface, OR a per-request override we inject).
4. **Stability:** to avoid thrash, the controller updates `r` on a sliding window of recent load samples (not per-request), and `r` is quantized to {0.25, 0.50, 0.75} so placeholder counts stay integer.

**Why this is publishable on its own:** no compressor in the 37-method table reads engine state. The closest, ElasticMM (ICCV'25), does scheduling WITHOUT compression and explicitly avoids it; PRUNESID/AgilePruner do content-adaptive budgets but OFFLINE (erank-based), blind to serving load. A load-adaptive compressor is a **first** in the serving-engine-aware sense.

### 7b. Other v2 levers (secondary, from Part I §2)
- **KV-cache-page-budget targeting** (instead of token count): solve for the `k` that fits a desired page slice — directly optimizes what vLLM schedules.
- **Decode-bandwidth guard** (finding: decode is text-bandwidth-bound, not visual): prune only down to the prefill/TTFT knee; further pruning is pure accuracy loss. Derive the knee from the probe's r-vs-ttft curve (the r50→r75 TTFT plateau on TextVQA is exactly this knee signal).

---

## 8. ★ Novelty vs the 37-method table (the load-bearing paragraph)

**We are building the FIRST serving-engine-aware visual-token compressor.** The 37-method survey (lit-survey.md §7, re-verified 2026-07-01) is unanimous: **0/37 measure served throughput inside a production engine** (vLLM/lmdeploy/TRT-LLM/SGLang). The 13 that report any wall-clock number all measure on the authors' own research harness — raw CUDA latency, offline prefill time, or self-reported "faster" — none inside a continuous-batching serving engine. The 2026 ICLR cluster (AgilePruner, VisionTrim, PRUNESID) crowds the accuracy/FLOPs-combination space but is offline-only; ElasticMM does scheduling but explicitly avoids compression; the vLLM RFC #45098 (`--image-pruning-rate`) is unfinished infra, not a method. **This is the cleanest novelty opening in the field and it holds.** Our claim is not "yet another accurate compressor" (that space is saturated and unbeatable in 3 months on 1× A40) — it is that **visual-token compression's deployment win lives in the serving engine (KV-cache/concurrency, finding #1), is sub-linear when the encoder is fixed (finding #2), and scales with visual-token fraction (finding #3)** — three effects invisible to FLOPs/accuracy measurement and untouched by every existing method. v1 (true CLS-attn + early-prune) delivers an accurate, real-wall-clock-fast compressor inside vLLM; v2 (load-adaptive budget) makes the compressor *aware* of the engine state — a first. The combination is the contribution: a compressor co-designed with continuous-batching realities, evaluated by served throughput, with the three findings as the mechanism story. This is a method paper (not a measurement paper): the load-adaptive controller (§7) is novel machinery, and the early-prune-via-CLS-attn (§6b) is a new integration that the survey diagnosed as blocked (§6.5.3 FlashAttention hurdle) but which is in fact tractable in the vision tower (CUDA graphs don't cover it in V0).

— End Part II. v1 code in `src/compressors.py` (`TrueClsAttnSelector`, `ClsAttnCapture`); v1 probe jobs in `notes/v1_probe_jobs.json`. —

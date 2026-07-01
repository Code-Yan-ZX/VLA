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

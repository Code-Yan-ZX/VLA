# V2 P1 — Qwen3-VL-8B generalization (F1/F2/F3 re-test)

> Subagent output. Do the three findings (F1 concurrency amplification, F2
> vision-fixed-cost, F3 workload-dependence) generalize from LLaVA-1.5 to a
> modern dynamic-resolution model with a NATIVE 2x2 MLP merger? Model:
> Qwen3-VL-8B-Instruct (bf16, ~16GB). Env `qwen3vl_clean` (vllm 0.19.0 V1,
> in-process EngineCore). 1× A40 46GB. Compare against P0 LLaVA-1.5-7B V1
> (notes/v2_p0_v1_tableA.md).

## TL;DR (4 lines)
- **Integration WORKS**: hook `model._process_image_input` (prune each post-split
  per-image embed) + patch `Qwen3VLMultiModalProcessor._get_prompt_updates`
  (variable per-image k). M-RoPE handled by vLLM's built-in `recompute_mrope_positions`.
- **F1 ATTENUATED, not absent**: c12/r75 = **1.29×** (vs LLaVA-1.5 1.86×); r75
  concurrency bonus c1→c12 = **+0.21** (vs +0.65). Mechanism holds directionally
  but is ~1/3 the strength.
- **F2**: vision tower + native merger = **only 10% of TTFT** (the merger is
  efficient, ~18ms/req). Pruning is post-merger, so this 10% is IRREDUCIBLE by us.
- **F3 HOLDS**: TextVQA (748 visual tok) r50 = **1.16×** > GQA (279 tok) r50 =
  **1.06×** — more visual tokens → bigger prune speedup. But TextVQA acc drops
  steeply (0.77→0.48). **★ The native 2x2 merger and our post-merger pruner are
  SUBSTITUTES, not complements** → diminishing returns on GQA.

## 1. Integration: post-split hook + variable-k processor (the new architecture)

Qwen3-VL ≠ LLaVA-1.5 in three load-bearing ways. Resolution of each:

**(a) Vision path = ViT → native 2×2 MLP merger → (deepstack multiscale cat) → LLM.**
`Qwen3_VisionTransformer.forward` (qwen3_vl.py:616) runs the ViT blocks, then
`self.merger` (line 653, a `Qwen3_VisionPatchMerger` that reshapes 4 spatial
patches → 1 token via an MLP), then concatenates 3 deepstack merger outputs
along the FEATURE dim (line 654-656: `[main(4096), ds1, ds2, ds3]` → 16384-wide
rows). So the seq dim == num_post_merger_tokens; each row holds the main +
deepstack features for ONE spatial position. **Implication for pruning: dropping
row j removes position j consistently across main + deepstack** — a single
per-row score is well-defined.

**(b) The placeholder path is NOT `ProcessingInfo.get_num_image_tokens`.**
Unlike LLaVA (where patching `LlavaProcessingInfo.get_num_image_tokens` suffices),
Qwen3-VL builds image placeholders via `Qwen3VLMultiModalProcessor._get_prompt_updates`
→ `get_image_replacement_qwen3vl` (qwen3_vl.py:1193-1199): `[image_token_id] *
(grid_thw.prod() // merge_length)`. `get_num_image_tokens` is only for dummy/max-
token estimation. **Fix**: wrap `_get_prompt_updates` (dataclass-replace the image
`PromptReplacement.replacement` callable) to scale the returned token list to
`k = int(per_image_full * (1-r))` per image. (Confirmed in probe: proc_full=[260],
proc_k=[130] at r50; placeholder shrinks 279→149 mean_ptid.)

**(c) Dynamic resolution → per-image k, and `_process_image_input`'s split.**
The post-merger count varies per image (GQA ~260, TextVQA ~730). `model._process_
image_input` (line 1803-1823) does `image_embeds.split(sizes)` with `sizes =
grid_thw.prod(-1)//4` (the UNPRUNED per-image count). **Hooking `model.visual`
fails**: pruning visual's output first starves that split (`RuntimeError:
split_with_sizes expects split_sizes to sum exactly to 130 ... but got [260]`).
**Fix**: hook ONE level up — wrap `model._process_image_input` so the split runs
on the full visual output, then prune each per-image embed to its own k_i (top-k
by L2-norm of the full 16384-wide row), return the tuple. The placeholder (set in
(a)) matches sum(k_i).

**M-RoPE**: vLLM 0.19 ships `recompute_mrope_positions` (qwen3_vl.py:2299,
docstring: "once we prune media tokens we should reflect this in the
mrope_positions") — it recomputes M-RoPE from the ACTUAL (pruned) multimodal
embedding lengths. So variable-k pruning "just works" for M-RoPE. No V0-style
CUBLAS shape crash at any r (verified r0/r50/r75, probe + 13 cells).

**Selector**: L2-norm proxy (Qwen3-VL has no CLS token; CLS-attention path of
LLaVA-1.5 is inapplicable). Crude but a fair probe-grade baseline; accuracy
section shows it degrades gracefully on GQA, steeply on TextVQA.

Probe: `runs/v2_p1_probe.py` → integration proven (r50 260→130 kept == placeholder
130; forward OK; ptid 279→149→84 across r0/r50/r75).

## 2. F1 — served-throughput matrix (Qwen3-VL-8B, GQA, n=100, mt=32, V1)

| c | r0 req/s | r50 req/s | r75 req/s | r50/r0 | r75/r0 | mean_ptid r0 |
|---|---|---|---|---|---|---|
| c1  | 0.971 | 1.030 | 1.045 | 1.06× | 1.08× | 279 |
| c4  | 3.211 | 3.417 | 3.541 | 1.06× | 1.10× | 279 |
| c12 | 6.212 | 7.204 | **8.011** | 1.16× | **1.29×** | 279 |

**Concurrency amplification (r/r0 bonus c1→c12):**
- Qwen3-VL r50: 1.06× → 1.16× (bonus **+0.10**); r75: 1.08× → 1.29× (bonus **+0.21**)
- LLaVA-1.5 V1 (P0): r50 +0.33 ; r75 **+0.65** (c12/r75 = 1.86×)

**★ F1 verdict — ATTENUATED, not absent.** The prune speedup DOES still grow with
concurrency on Qwen3-VL (r75 bonus +0.21 > 0 — the KV-cache/concurrency mechanism
is robust to the architecture), but it is ~1/3 the strength of LLaVA-1.5
(+0.21 vs +0.65; headline c12/r75 1.29× vs 1.86×). The mechanism generalizes
directionally; the magnitude does not. (LLaVA-1.5 c1 numbers were higher in
absolute req/s — 2.036 vs 0.971 — because Qwen3-VL-8B is a heavier decoder; the
SPEEDUP RATIOS are the architecture-controlled comparison.)

## 3. F2 — prefill breakdown (Qwen3-VL native merger does NOT create a big fixed cost)

GQA c1, n=20, mt=4 (decode-negligible → wall ≈ TTFT), r=0, vision timed via a
`cuda.synchronize`-wrapped hook on `_process_image_input`:

| component | time | % of wall |
|---|---|---|
| vision tower + native 2×2 merger + deepstack (all 4 mergers) | 0.36s | **10%** |
| LLM prefill + 4-token decode | 3.42s | 90% |
| **total wall (20 reqs)** | 3.78s | |

- **Vision is only 10% of TTFT** (~18ms/req) — the native merger is EFFICIENT, not
  the heavy fixed cost the "merger hurts pruning" story would predict. F2's
  "vision-tower fixed cost" framing (from LLaVA-1.5) is SMALLER on Qwen3-VL.
- Pruning is **post-merger** → the 10% vision cost is IRREDUCIBLE by our method
  (the merger+ViT run in full regardless of r; confirmed: vision_ms r0=365 ≈
  r50=358). The addressable headroom is the 90% LLM-prefill.
- r50 wall speedup at c1/mt4 = 3.78/3.31 = **1.14×** (prefill-dominated); at the
  production mt=32 (decode-heavy) e2e drops to 1.06× — decode dilutes the prefill
  saving. vs EarlyTom 2605.30010 (decomposes TTFT similarly) — ours is the
  serving-engine, post-merger-pruning version (cite + differentiate).

## 4. F3 — workload-dependence (HOLDS, but accuracy-sensitive)

c4, n=100, mt=32:

| workload | visual tok r0 | r50/r0 speedup | acc r0 → r50 |
|---|---|---|---|
| GQA     | 279 (149 at r50) | **1.06×** | 0.49 → 0.47 (robust) |
| TextVQA | 748 (383 at r50) | **1.16×** | 0.77 → 0.48 (steep drop) |

- **F3 HOLDS on Qwen3-VL**: TextVQA (2.7× more visual tokens via dynamic
  resolution) gets a LARGER r50 speedup than GQA (1.16× vs 1.06×). The
  visual-fraction effect is robust to the architecture. Dynamic resolution makes
  the effect MORE pronounced — it auto-allocates more tokens to text-dense images,
  where pruning then has more to remove.
- **Workload-dependent accuracy cost**: GQA (object QA) is nearly accuracy-flat
  under pruning (0.49→0.47 even at r75 0.43); TextVQA (fine-grained text reading)
  collapses (0.77→0.48 at r50). Pruning text-region tokens is the known FastV
  failure mode — confirms a selector-accuracy tradeoff that the L2 proxy exposes
  worst on OCR-style tasks.

## 5. ★ Hypothesis verdict — DIMINISHING RETURNS (merger & pruner are SUBSTITUTES)

The P1 hypothesis posed two outcomes:
- (a) dynamic resolution → MORE visual tokens → pruning value LARGER; OR
- (b) native 2×2 merger already compressed → diminishing returns.

**Answer: (b), with a twist.** The native 2×2 merger compresses 4 patches → 1
token BEFORE our pruner; our pruner then operates on the post-merger tokens. On
GQA, smart_resize + the merger leaves only ~260 post-merger tokens (vs LLaVA-1.5's
FIXED 576 pre-projector tokens). So there is LESS for our pruner to remove
(pruning 75% saves ~195 tokens vs LLaVA's 432) → smaller absolute KV/prefill
relief → smaller F1 speedup and weaker concurrency amplification. **The merger
and our pruner both consume the same resource (post-merger tokens), so they are
SUBSTITUTES, not complements — the merger's compression reduces the marginal
value of our pruning.** This is the architectural reason F1 attenuates.

The twist: dynamic resolution RESTORES pruning value on text-dense workloads
(TextVQA → 748 tokens, F3 speedup 1.16× > GQA 1.06×) — but at a steep accuracy
cost. So the value of post-merger pruning on a merger-equipped model is
**workload-conditional**: low on already-compressed object QA, higher (but
accuracy-fragile) on text-dense images. This is the key generalization insight
for the paper: F1's concurrency amplification is NOT a universal constant — it
scales with the visual-token budget that survives the model's own compression.

## 6. Accuracy sanity (n=100, Qwen3-VL-8B, L2 proxy)

| cell | acc | | cell | acc |
|---|---|---|---|---|
| GQA r0 c1 | 0.49 | | GQA r75 c1 | 0.43 |
| GQA r50 c1 | 0.47 | | TextVQA r0 c4 | 0.77 |
| GQA r50 c12 | 0.45 | | TextVQA r50 c4 | 0.48 |

GQA accuracy is essentially flat across concurrency (throughput is c-dependent,
accuracy is not — consistent across r). The L2 proxy degrades GQA gently
(0.49→0.43 at r75) but TextVQA sharply (0.77→0.48 at r50) — a query-aware
selector (the v2 method-design path) should close the TextVQA gap, deferred.

## 7. Environment + reproduction

- env: `qwen3vl_clean` (vllm 0.19.0, torch 2.10.0+cu128, py3.10). GPU: 1× A40 46GB.
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` (in-process EngineCore, same as P0).
- reproduce: `bash runs/v2_p1_run_matrix.sh` (13 cells = 9 F1 + 2 F2 + 2 F3;
  ~24 min, each cell = fresh process); analyze: `python runs/v2_p1_analyze.py`.
- probe (integration proof): `runs/v2_p1_probe.py` → `runs/v2_p1_probe.log`.
- runner: `runs/v2_p1_runner.py` (processor patch + _process_image_input wrap).
- total GPU: ~24 min for all 13 cells + ~3 min probe. Qwen3-VL-8B bf16 ~16.8GB,
  KV budget 20.88GB (max concurrency 18.5× at 8192 len).

# GLM-4V Fourth-Family Stage-Law Gate — Digest

Date: 2026-08-03 · Host: 1× A40 46GB · Env: qwen3vl_clean (transformers 4.57.6, torch 2.10, vLLM 0.19.0)
Purpose: cross-family replication of the workload-conditional pre-vs-post stage law
(text-dense pre≫post; GQA post≥pre) on a fully Qwen/InternVL-independent architecture,
per the pre-registered criteria in `experiments/latest_vlm_model_audit.md`.

## Model used

- **Audit TOP-1 `ZhipuAI/GLM-4.6V-Flash` FAILED at processor load** (Phase 1):
  config.json parses fine on transformers 4.57.6 (the feared 5.0.0rc `rope_parameters`
  key is accepted), but `preprocessor_config.json` demands
  `Glm46VImageProcessor`/`Glm46VProcessor` — classes that only exist in
  transformers ≥5.0rc. AutoProcessor silently falls back to the bare tokenizer
  (`PreTrainedTokenizerFast has no attribute image_processor`), which makes the
  model unusable under vLLM's multimodal pipeline. Per the pre-registered plan
  (and the explicit no-config-hack rule) → **fallback adopted**.
- **Model actually used: `ZhipuAI/GLM-4.1V-9B-Thinking`** (same `glm4v`
  architecture, config targets transformers 4.57.1), ModelScope snapshot at
  `/data/models/modelscope/hub/models/ZhipuAI/GLM-4___1V-9B-Thinking`
  (modelscope cache convention: dots escaped to `___`). Loaded as a local path;
  no trust_remote_code, no remote modeling code.
- Config facts: text 40L/4096d GQA 32/2, mrope **block sections [8,12,12]**
  (`rope_scaling.mrope_section`), image_token_id 151343; vision 24L/1536d ViT,
  patch 14, `spatial_merge_size=2`, out_hidden 4096. Merge stage = strided 2×2
  `downsample` conv (1536→4096) + `Glm4vVisionPatchMerger` MLP — deterministic,
  positional 4× compression (identical layout in vLLM `glm4_1v.py` and
  transformers `modeling_glm4v.py`).
- **Thinking-model note**: GLM-4.1V-9B-Thinking emits spontaneous `<think>`
  blocks; the runner's `--max-tokens` was raised 32 → **1024** for all glm4v
  cells (32 would truncate reasoning before any answer; the n=8 smoke showed
  degraded arms loop inside `<think>` when uncertain and need room to still
  reach the boxed answer). Identical across arms; only the within-model
  PATTERN is read. Containment/official scorers see the full answer text for
  every arm.

## Weight acquisition note (Phase 0)

ModelScope pull of the fallback started at ~5 MB/s and degraded to ~1 MB/s;
a parallel 4-shard fetch of the identical `THUDM/GLM-4.1V-9B-Thinking`
snapshot from the HF mirror (hf-mirror.com) won the race (~25 MB/s aggregate,
all 4 shards size-verified against Content-Length, safetensors headers +
704-tensor index validated with `safe_open`). The staging copy was moved to
the canonical ModelScope-layout path
`/data/models/modelscope/hub/models/ZhipuAI/GLM-4___1V-9B-Thinking`
(byte-identical config vs the ModelScope partial; same repo snapshot).

## Load smoke (Phase 1 outcome on the fallback model) — PASS

`scripts/glm4v_load_smoke.py` (transformers 4.57.6, bf16, single A40):
- (a) config loads natively: Glm4vConfig, image_token_id 151343,
  mrope_section [8,12,12], spatial_merge_size 2 (no accelerate in env →
  CPU load + .to('cuda'); weights 20.6 GB on GPU).
- (b) merger exists: `model.visual.merger = Glm4vVisionPatchMerger`,
  preceded by `downsample = Conv2d` (the strided 2×2 stage).
- (d) tokens: grid_thw [1,74,42] → ViT tokens 3108 → merged 777;
  ratio 4.00; placeholder run in input_ids = 777 (exact match).
  (The 4× happens at the downsample conv: merger in/out = (777,4096)→(777,4096).)
- (c) generation on an OCR-ish TextVQA image ("what time is displayed?"):
  `<think>…</think><answer><|begin_of_box|>12:34 am<|end_of_box|></answer>`
  — correct (gt "12:34"), confirms the spontaneous thinking format.

## Runner integration (Phase 2)

All changes family-scoped in `src/v3_premerger/v3_premerger_runner.py`
(+~380 lines; 15 lines retouched = family gates now including glm4v):

- `MODELS["glm4v"]` (local ModelScope path); `--model-family` choices;
  `detect_family` (`glm-4`/`glm4` → glm4v); `_proc_class` →
  vLLM `glm4_1v.Glm4vMultiModalProcessor`.
- **Placeholder scaling**: the generic list-scaler branch works unchanged
  (glm4v's image replacement is a plain `[image_token_id]*num_units` list;
  per-image k = max(1, round(n·(1−r)))).
- **PRE arm** `setup_pre_merger_glm4v`: forward_pre_hook on `visual` captures
  `grid_thw` → `PreMergerPruner.begin_pass`; `visual.forward` replaced
  (line-by-line copy of vLLM 0.19 `Glm4vVisionTransformer.forward`) with ONE
  change — after `post_layernorm`, units are top-k selected (L2 over the 4×1536
  pre-merge ViT stream, per-image k_i) **before** the native downsample-conv +
  merger MLP run on survivors only. The 2×2 strided conv has no cross-unit
  receptive field ⇒ kept units' merged tokens are bitwise native.
  `_process_image_input` replaced to split by the pruned counts.
  (Semantics = qwen2vl pre hook: rank on the merger input stream, pre-merge.)
- **POST arm**: the family-agnostic `setup_post_merger`
  (`_process_image_input` split prune) — verified to fit glm4v's per-image
  merged-token layout (dry-check (c)).
- **mrope fix**: `setup_qwen2vl_mrope_fix` now also gated to glm4v
  (block mrope [8,12,12] ⇒ the same placeholder-run position repair as
  qwen2vl; contract `get_mrope_input_positions(input_tokens, mm_features)` +
  `mm_feature.data["image_grid_thw"]` verified identical).
- **Capability guards**: swap/visionzip/hybrid/QA-pre are Qwen-only machinery
  and raise SystemExit for glm4v (same policy as internvl3).
- **trust_remote_code**: unchanged (`family == "internvl3"` only) — glm4v is
  transformers-native.
- **--max-pixels enforcement**: glm4v excluded from the `mm_processor_kwargs`
  path (Glm4vImageProcessor has no max_pixels parameter; its budget is the
  `size={shortest_edge,longest_edge}` AREA dict) → PIL pre-resize enforcement
  only, same as internvl3. The gate itself ran at the model's DEFAULT
  processor resolution (`--max-pixels 0`).
- **Isolation proof**: `git diff` shows only additive glm4v code + the gate
  retouches above; runner `--dry-check` PASSES for all four families
  (glm4v new; qwen3vl / qwen2vl / internvl3 unchanged behavior).

## Verify (Phase 3) — n=8 textvqa smoke

Runner `--dry-check` family=glm4v: ALL PASS (processor patch on
Glm4vMultiModalProcessor; pre visual.forward slice units [8,4]→k[2,1]; post
family-agnostic split prune; mrope-fix install). Dry-check regression:
qwen3vl / qwen2vl / internvl3 all still ALL PASS (unchanged behavior).

n=8 smoke on textvqa_200 (runs/glm4v_smoke8/), all arms 0 skips,
non-degenerate per-image answers:

| arm  | runner acc | mean prompt tokens | visual tokens (≈ptid−36 text) |
|---|---|---|---|
| none | 0.875 | 1035.5 | ≈ 999 merged (full) |
| pre  | 0.875 | 279.4  | ≈ 243 merged = 25% (k e.g. 777→194, 1147→287) |
| post | 0.125 | 279.4  | ≈ 243 merged = 25% (same k) |

- **iso-token EXACT**: pre and post mean_ptid identical (279.4) with
  identical per-sample prompt_token_ids; both ≈ 25% of the none-arm visual
  tokens. Placeholder (n,k) pairs agree with pruner k at every site.
- pre arm at 25% keeps baseline-level text accuracy; post arm collapses on
  text-dense samples — the answers stay coherent and image-specific (e.g.
  reads the clock as ~10 o'clock, the sign as "Pizza Shop") but lose the
  fine text detail, and the thinking model then loops inside `<think>` and
  often fails to emit the boxed answer within the cap. Same protocol for all
  arms; this is the stage effect's expected DIRECTION on text-dense input
  (amplified by the thinking format), not a harness fault (positions/splits
  are identical to pre, which is intact).
- Multi-image encoder batches verified correct: one visual call can carry
  several images; the per-image full_units/k_units lists keep the split exact
  (diag nk logs the first image per call only).

## Protocol

9 cells = {none, pre, post} × {textvqa_200, docvqa_200, gqa_200}, `--r 0.75`
(keep 25%), selector l2, fresh process per cell, `--max-tokens 512`,
outputs `runs/glm4v_gate/`. Official offline rescore (same functions as
`scripts/rescore_official.py`, applied inline as in
`scripts/internvl3_main_matrix.sh`): TextVQA VQA-acc, DocVQA ANLS, GQA
exact-match.

**Effective pixel budget (model default)**: `size = {shortest_edge 12544,
longest_edge 9633792}` AREAS, patch 14, merge 2, factor 28; for single images
the t=2 temporal padding halves the spatial cap ⇒ effective max ≈ 4.82M px
(≤ 6144 merged tokens/image), min ≈ 6272 px. Estimated mean merged tokens:
textvqa ≈ 970 (max 1369), docvqa ≈ 4839 (max 6072), gqa ≈ 347 (max 529).
Cross-family absolute values are NOT comparable — only the PATTERN
(direction × workload condition) is.

## Results

All 9 cells: n=200, 0 skipped, r=0.75 (keep 25%), selector l2.
Runner containment acc (in-harness metric, for reference): textvqa
0.785/0.745/0.18 · docvqa 0.68/0.70/0.145 · gqa 0.29/0.31/0.245.

### Official scores (none / pre / post @ keep 25%)

Raw-text official scoring gives 0.0000 on EVERY sample because the thinking
wrapper never string-matches short golds; the canonical thinking-model
protocol = parse the final answer (last box, else post-think, else full),
applied IDENTICALLY to all arms (`scripts/glm4v_rescore_official.py`):

| benchmark | metric | none | pre | post | Δ(pre−post) pp |
|---|---|---|---|---|---|
| textvqa_200 | VQA-acc | 0.2417 | 0.2183 | 0.0500 | **+16.83** |
| docvqa_200  | ANLS    | 0.1039 | 0.1297 | 0.0313 | **+9.84** |
| gqa_200     | exact-match | 0.1500 | 0.1600 | 0.1150 | **+4.50** |

Convergence note: fraction of samples reaching the boxed final answer —
textvqa none/pre/post = 56/52/15, docvqa = 33/30/11, gqa = 46/47/39 (of 200).
The post arm's degraded visual input makes the model loop in-think and miss
the 1024-token cap more often; the extraction fallback then scores the
unfinished text. Identical protocol across arms.

### Mean prompt token counts per arm (iso-token check)

| benchmark | none | pre | post | pre/post visual ≈ 25% of none |
|---|---|---|---|---|
| textvqa_200 | 996.8 | 269.1 | 269.1 | ≈960 → ≈233 (24%) |
| docvqa_200 | 4868.1 | 1238.8 | 1238.8 | ≈4830 → ≈1200 (25%) |
| gqa_200 | 373.7 | 113.8 | 113.8 | ≈347 → ≈87 (25%) |

EXACT iso-token: pre and post mean_ptid identical per benchmark (per-sample
prompt_token_ids identical too); both ≈ 25% of the none-arm visual tokens
(rounding per image pushes gqa to 25.4%).

## VERDICT (against pre-registered criteria)

Pre-registered success = BOTH text-dense pre≫post AND GQA post≥pre;
failure = text-dense Δ<5pp OR reversed GQA crossover.

- **Text-dense half: TRANSFERS.** textvqa Δ=+16.83pp, docvqa Δ=+9.84pp —
  both far above the 5pp bar, same direction as Qwen3-VL / InternVL3.
- **GQA half: DOES NOT REPLICATE at full n.** post 0.115 < pre 0.160
  (Δ=+4.5pp in the WRONG direction vs the expected post≥pre). At n=200 the
  reversal is within binomial noise (z≈1.3, not significant), but the
  pre-registered criterion is directional: not met.
- **Diagnostic (post-hoc, not the headline)**: on the n=25 GQA samples where
  ALL THREE arms converged to a boxed answer, none/pre/post = 0.640/0.680/
  0.680 → Δ(pre−post)=0.0pp, i.e. post≥pre (tie) among converged samples.
  The full-n reversal is carried by the post arm's lower answer convergence
  (39 vs 47 boxed), an inherent thinking-model behavior under degraded
  vision, scored under the identical protocol.

**Net: stage law transfers PARTIALLY to the independent GLM family — the
workload-conditional text-dense pre≫post effect replicates strongly; the
GQA post≥pre half fails the pre-registered directional test at n=200
(noise-level, convergence-driven).** Per the audit's failure branch this
triggers the honest-downgrade wording + escalation to user/coordinator for
any claim that the FULL workload-conditional pattern is architecture-universal.

## Resource accounting

- 9 gate cells (wall+load): 9512 s = 2.64 GPU·h. Plus: load smoke ~4 min,
  n=8 smoke 3 cells ~8 min, 1 debug re-run ~2 min, one OOM'd attempt ~5 min,
  killed-run reload overhead ~5 min. **Total ≈ 3.1 GPU·h < 6 GPU·h budget.**
- Ops note: the harness killed three long-running background gate-script
  parents mid-campaign; the final two cells were run under `setsid` (detached)
  and completed cleanly. No GPU ever ran two processes concurrently.

## Harness changes / fixes

- Runner: glm4v family branch (additive; see integration section). No existing
  family behavior changed (dry-check regression all-pass for all 4 families).
- Gate script docvqa flags: `--max-num-seqs 8 → 4`, `--gpu-memory-utilization
  0.9 → 0.85` for the pruned arms — pre_docvqa OOM'd at step 0 with mns 8:
  short pruned placeholders let the scheduler pack 8 concurrent long-document
  prefills and the 20k+-patch ViT activations overflowed the 0.9 headroom
  (none arm survived at mns 8; safe flags applied uniformly, recorded here).
- New files: `scripts/glm4v_load_smoke.py`, `scripts/glm4v_rescore_official.py`
  (thinking-answer extraction + official metrics), `src/v3_premerger/glm4v_gate.sh`.

## Addendum: max_tokens=4096 probe (v2, coordinator-ordered)

4 cells re-run at `--max-tokens 4096` (everything else identical), outputs in
`runs/glm4v_gate_v2/` (v1 untouched). Mean generated tokens ≈ 3930–4060 in
ALL cells (≈ at the cap); complete boxed answers 0–2/200; mid-loop/no-box
fraction 0.74–0.81 — at 4× budget the greedy thinking still essentially never
converges (repetition loops dominate the tails).

| cell | official | runner | boxed | cut_frac | mean_gen | wall_s |
|---|---|---|---|---|---|---|
| none_gqa@4096 | 0.1650 | 0.290 | 1/200 | 0.740 | 4011 | 2160 |
| pre_gqa@4096 | 0.1650 | 0.310 | 0/200 | 0.765 | 3997 | 2137 |
| post_gqa@4096 | 0.1150 | 0.245 | 0/200 | 0.810 | 4057 | 2137 |
| none_textvqa@4096 | 0.2383 | 0.775 | 2/200 | 0.745 | 3927 | 2090 |

Verdicts: (1) GQA post≥pre? **NO** — and no tie (Δ(pre−post)=+5.0pp at 4096,
same as v1): the reversal does NOT ride the truncation channel. (2) TextVQA
none anchor ≥0.65? **NO** — 0.2383 at 4096 vs 0.2417 at 1024; the anchor gap
vs published ≈0.77 is NOT truncation. The runner containment metric (0.775)
sits right at the published ≈0.77: the gap is a **decoding/protocol effect** —
greedy thinking rarely converges to a concise boxed answer (the repo
generation_config specifies temp 0.8/top_p 0.6 sampling, likely used by the
published evals; our gate is greedy for deterministic arm comparison,
identical across arms). v1 conclusions stand: pre≫post text-dense replicates;
GQA post≥pre fails directionally at both caps (n=200 noise-level, loop-driven).

Budget: v2 = 2.40 GPU·h (≤2.6); cumulative v1+v2 ≈ 5.5 GPU·h (<6). Abort rule
checked: first two cells = 1.21 GPU·h < 1.6 → all four cells run.

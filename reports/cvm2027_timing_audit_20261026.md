# P3 timing audit — 2026-10-26

The original `plan/timing_harness.json` and `plan/timing_hilo.json` are retained as historical raw artifacts. They do not support a fair inference-cost ranking. Use newly measured `*_corrected.json` artifacts only after their parity, actual-token-budget and request-completion gates pass.

## Confirmed defects in the original script

1. `scripts/cvm24h_timing.py` runs its manual generation replica without `no_grad` or `inference_mode`. The authoritative `baselines_hf.generate_pruned` is decorated with `torch.no_grad`. The replica therefore builds inference-irrelevant autograd graphs through the decoder. Original harness peak allocated memory was 41.03–42.70 GiB, with 7–18 OOM requests depending on configuration; the same 200 attempted IDs consequently yield different sets of survivors.
2. Each original pre-mode request installs a new `MergerTap` and never calls `remove`. Hooks retain their first captured visual feature tensor and accumulate across requests. The corrected script installs one tap and resets it per request.
3. The original FastV timing passes `deepstack=None`, despite capturing all three deepstack tensors. Quality evaluation passes the three tensors. The replica times a different model computation. Its parity check covers `none` only.
4. Original harness `total_s` starts after processor/transfer; original Hi-Lo `total_s` starts before processor/transfer. Their request boundaries differ. Harness uses eager attention throughout, whereas Hi-Lo uses SDPA throughout.
5. Original Hi-Lo constructs one model at keep=.25, then modifies the external configuration for .50/.75. Installed Transformers 4.57.6 deep-copies the configuration in `modeling_utils.py:4968`, so those later writes do not update the model's live configuration. All 200 request output lengths are identical across its three nominal budgets, corroborating the code defect. The original timing files contain no actual-K trace that could validate the claimed .50/.75 budgets.
6. Original harness does not save a direct `ttft_s` field, excludes processor from its implied TTFT, and uses different successful-request subsets for latency quantiles. Original Hi-Lo has no per-configuration warmup and uses callback order as a proxy for recognizing the prompt. Raw repetition records must be inspected instead of relying on the old generated table.

## Corrected protocol

`scripts/cvm24h_timing_v2.py` preserves the original fixed 200 IDs (TextVQA 100 + OCRBench 100) and repeats each request three times after warmup. A single A40 runs configurations serially. The common model is Qwen3-VL-8B, bf16, greedy, maximum 32 new tokens, 4,000,000 pixel cap, native positions, vision SDPA and text eager. The live vision/text attention implementations are asserted. This shared attention protocol differs from the previous full quality runs, so quality-cost pairs must use answers from these new fixed timing requests.

The pre-fixed TextVQA timing selector used a legacy token-count proxy and excluded the five previously skipped records. The legacy pre-mode `n_image_full` field actually contains the retained 25% count (for example ID 34607 is recorded as 168 although its full input contains 672 image tokens). This count is approximately monotone in resolution and remains usable as a stratification proxy; it is not an authoritative full-token count. The IDs remain frozen. The corrected run measures actual full and retained counts directly. This is a fixed evaluation workload rather than a random representative estimate of full-split latency.

Each measured request begins with an already decoded RGB PIL image. Direct timestamps measure CPU processor work, host-to-device transfer, full visual encoding and merger, LLM prefill/first-token time, decoding and complete request time. CUDA events also record transfer, visual, selection and fusion intervals. Selection/fusion within LLM prefill overlap that segment and must not be added twice. End-to-end latency is measured directly rather than reconstructed by summing medians. Workload wall time includes the full repeat loop, local decode once per ID, bookkeeping, synchronization and any failure recovery; model loading and warmup are excluded.

The corrected harness projects only the last prefill hidden state through the vocabulary head, matching native HF generation's `logits_to_keep=1`. The original helper projected every prefill position although greedy generation consumed only the last one. This avoids charging unused vocabulary projections to None/RBM/FastV in the comparison. Complete generated-token parity against the original helper remains mandatory. These are timings of the corrected, equivalently optimized generation implementation, not unchanged historical-harness latency; paired quality comes from the newly saved 200-request answers.

Every native-harness configuration checks exact generated-token parity against the existing `generate_pruned` implementation on three warmup requests. The None configuration additionally checks full-token parity against native `model.generate`. A separate native-versus-vendored no-op check uses the first 32 fixed timing requests, recording complete greedy token sequences (maximum 32 new tokens), first-token logit differences, input/position/feature tensor byte hashes and fixed numerical feature probes, including all three deepstack streams. Hi-Lo selection is explicitly bypassed (`image_tensor=None`) and PA-fusion disabled on the live attention modules. Input and position identity and complete greedy-token parity are gated. BF16 feature and logit differences are reported numerically; byte identity is claimed only if the recorded tensor hashes actually match.

Every repeat retains its answer, generated token IDs, output length, original and retained visual-token counts, layer-by-layer active visual-token counts, direct timing fields and peak allocated/reserved memory. Hi-Lo changes the live model configuration and asserts actual K for every request and budget. FastV preserves all three deepstack streams. Per-request median values are computed over the three repetitions; p50/p95 then use those 200 request medians. Repetitions are not treated as independent samples. Failed requests remain in the artifact and completion counts accompany latency summaries.

## Reproduction

Run only when the A40 evaluation queue is idle. The historical GPU deadline is 2026-10-26 15:07:06 China time (epoch 1792998426). The deadline flag stops before a new request and preserves any partial cell.

```bash
P=/home/dell/miniconda3/envs/qwen3vl_clean/bin/python
$P scripts/cvm24h_timing_v2.py --engine harness \
  --out results/cvm2027_supplement_20261026/plan/timing_harness_corrected.json \
  --deadline-epoch 1792998426
$P scripts/cvm24h_timing_v2.py --engine hilo \
  --out results/cvm2027_supplement_20261026/plan/timing_hilo_corrected.json \
  --deadline-epoch 1792998426
```

The script refuses to resume an artifact from a different script hash. Original timing raw files are neither rewritten nor reused as valid cells. Final counts and any no-op mismatch or failed requests must be checked before exporting a paper table.

An independent original-quality-protocol gate is prepared in `scripts/cvm24h_noop_verify.py`. It uses eager vision/text attention and the previously fixed 32-request TextVQA smoke set to compare actual native generation with pre keep=1, post keep=1 and FastV drop=0. All three deepstack streams are passed to every manual call. A vocabulary-head hook captures first-step logits from the actual generation calls; input/position/mask/deepstack identity, kept index sets and complete greedy token sequences are retained. Numerical errors are reported explicitly. A structural/token mismatch writes evidence and fails instead of relaxing a tolerance. This additional gate is to be run only after mandatory cost measurements, with enough margin before the same GPU deadline; preparation is not a claim that the gate has passed.

```bash
$P scripts/cvm24h_noop_verify.py \
  --out results/cvm2027_supplement_20261026/plan/p0_noop32_verified.json \
  --deadline-epoch 1792998426
```

Completed corrected timing files can be checked without the GPU:

```bash
$P scripts/cvm24h_validate_timing_v2.py \
  results/cvm2027_supplement_20261026/plan/timing_harness_corrected.json \
  results/cvm2027_supplement_20261026/plan/timing_hilo_corrected.json \
  --out results/cvm2027_supplement_20261026/plan/timing_corrected_validation.json
```

## Matched full main-table quality extension

After the author explicitly authorized extending the historical GPU deadline, a fresh common-backend main-table runner was prepared in `scripts/cvm24h_matched_quality.py`. It leaves the frozen corrected timing runner unchanged. The new quality run uses the same vision SDPA/text eager, bf16, native-position, greedy-32, 4M-pixel protocol, with weights pinned to the exact cached snapshot `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` used by P3. All four methods freshly infer TextVQA 5000 and OCRBench 1000 at the frozen main comparison point (None 100%; RBM/FastV/Hi-Lo 25%). Full-prediction reuse is disabled.

The new runner directly imports the frozen corrected manual generation function for None/RBM/FastV and uses official Hi-Lo generation for the modern baseline. Three warmups and every common fixed P3 request compare complete generated token sequences, answers, full N and actual K against the first timing repeat without reading scores. Every planned ID is present from initialization, including explicit pending/failed states; checkpoints occur every 100 attempts, and deadline exits preserve partial results. TextVQA stores the original ten official answers. Weight/processor/implementation fingerprints and each request's full input-ID hash and actual K support cross-method CPU validation. Any protocol or blind-agreement mismatch saves evidence and aborts the cell.

The intended destination is `results/cvm2027_supplement_20261026/matched_main/{mode}_{benchmark}.json`. Before full inference, the root supervisor is to run all eight mode/task combinations with n=3 in a separate `smoke/` directory; these are runtime checks and never full-main results. The new deadline is supplied explicitly by the author-controlled serial supervisor. At script preparation, CPU metadata checks over all 5000 TextVQA records passed; GPU smoke and full-main completion remain to be verified from the resulting artifacts.

## Completed validation at 14:53, October 26

All ten corrected cost cells completed at 14:48: 200 common IDs and three raw repeats per configuration, 6,000 requests in total. The independent CPU validator passed actual counts, 36-layer traces, all raw medians, identical repeated answers and zero failures. Complete workload wall times sum to 4,009.6 seconds; model loading and warmup are excluded. Official quality is paired only with these same 100 requests per task in `reports/cvm2027_corrected_quality_cost.{md,json}`. For OCRBench-100, displayed scores are normalized by 1,000, not a claim to have scored the full split in this cost sample.

The shared-attention native/vendored no-op gate passed all 32 fixed TextVQA requests with selection and live fusion disabled: full input/position/main/three-deepstack hashes and complete greedy sequences agree, and first-step logit error is zero. The separate original eager keep-all gate also passed pre/post/FastV full greedy generation on all 32 requests, with input/position/mask/stream/index identity; its first-step BF16 logit maximum absolute error is 0.125 for each manual path, so those logits are **not** byte-identical. The two checks must not be conflated.

All eight n=3 matched-main runtime smokes passed protocol/source/grid/budget/blind-token checks. Fresh complete-split inference started at 14:51:25 in the pre-frozen order; this startup verification does not itself establish completion of the eight full cells. The original invalid timings remain retained only for audit.

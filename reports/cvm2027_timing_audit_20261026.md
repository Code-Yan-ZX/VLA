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

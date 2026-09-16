# Evidence audit for the 2026-09-16 DCC rewrite

Audit scope: existing local reports, committed artifacts/manifests, and targeted implementation inspection only. No local skills, new GPU experiments, new significance calculations, or external literature searches. This is an internal evidence ledger, not a claim that all historical experiment narratives are correct.

## 1. Implementation facts and limits of attribution

The executable selector paths take precedence over shorthand in experiment reports. In particular, the phrase “pure-stage” in `reports/acmmm_final_controls.md` overstates the Qwen3 final-input control: the post score uses the concatenated main and deepstack outputs. This was already acknowledged in `experiments/p0-3_prefinal_control.md:41`.

| Model/path | Pre score and tap | Post score | Native computation, allocation and ordering |
|---|---|---|---|
| Qwen3 headline RBM | Mean of the four patch L2 norms, on the **raw input of the first-called deepstack merger**, ViT layer-8/deepstack[0] input. Float conversion for norm, no added score normalization. The wrapper acts before the merger's internal normalization/projection. | L2 of each **entire returned visual vector**, comprising main output and three deepstack outputs concatenated across channels. It is not main-output-only scoring. | One cached mask slices the inputs of all four mergers; native merger functions compute survivors at their respective depths. Units remain index ordered. Mask creation at an early merger does not mean the ViT blocks stop processing discarded patches. |
| Qwen3 pre-final control | Mean patch L2 at the **main merger's own final-ViT input**, before the native merger forward. | Same full-concatenation post scorer above. | Main and deepstack mergers all execute on full inputs, then the cached final-input mask selects corresponding rows of their concatenated visual output. This holds the output computation pathway fixed, but does not equate the scoring streams or isolate the main merger as a single causal variable. |
| Qwen2.5 RBM | Mean four-patch L2 at the raw final-ViT merger input, before merger-internal normalization. | L2 of the merged visual output. | One merger, no deepstack. The vLLM pre path keeps selected units in window order and explicitly skips the native reverse_indices restoration; post operates after native output restoration. Thus there is an ordering difference in addition to scoring stage. |
| InternVL3 RBM | L2 of the **4c concatenated vector after parameter-free pixel shuffle and before mlp1** (including its initial LN). This is sqrt(sum of squared patch norms), not the mean patch norm used in Qwen. | L2 of mlp1 output. | RBM chooses k per tile; Post-L2 pools all image tiles and chooses a total K across the image. Totals agree, but spatial allocation differs. Both preserve sorted survivor order, use native mlp1 and 1-D RoPE, and have no deepstack. |

Code anchors, all in `src/v3_premerger/v3_premerger_runner.py`:
- Token/unit scoring: `_score_tokens` around 347; `_score_units` around 415.
- Qwen post returned-vector scoring: `setup_post_merger`, 1360–1390.
- InternVL3 pre/pixel shuffle and tile budgets: 1436–1501; image-pooled post: 1541–1572.
- Qwen score tap/mask: `PreMergerPruner.slice_input`, 2459 onward; `_wrap_merger_forward`, 2826 onward.
- Qwen2 window-order change: 2864–2988.
- Shared Qwen3 mask and targets: 3022 onward.
- Final-input control: 3135–3210.

**Position semantics:** do not claim preservation of original retained spatial coordinates. The vLLM Qwen2 adapter counts compressed placeholders and assigns the **first k grid positions**, correcting trailing-text offsets; it does not gather the original coordinates of selected units (1224–1348). The Qwen3 vLLM path retains its stock positional handling. HF experiments using `--mrope native` follow a different position route; do not label all HF/vLLM paths equivalent. Native merger parameters/interface compatibility are defensible; exact unmodified native position/order semantics across every reported implementation are not.

**Known versus unverified:** the score taps and aggregation rules above are directly visible in code. Exact runtime internal normalization variants and model revision internals were not freshly loaded in this audit; say “before the native merger/projector forward (and its internal normalization)” rather than inventing an exact per-model LN topology. Headline multi-model gains are implementation comparisons, not isolated cross-architecture stage effects.

## 2. Qwen3 final-input control at 25% retention

Existing official rescoring and paired 95% bootstrap CIs, reproduced from `reports/acmmm_final_controls.md:36–46` and committed `results/acmmm_final_controls/cell_summary.csv`.

| Benchmark | n paired | None anchor | Pre-final | Post-L2 | Difference in percentage points | Paired 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| TextVQA | 5000 | .8443 | .4985 | .2217 | +27.68 | [26.21, 29.19] |
| DocVQA | 5349 | .9562 | .2836 | .2377 | +4.59 | [3.43, 5.78] |
| OCRBench | 1000 | 760/1000 | 419/1000 | 184/1000 | +23.50 (+235 points) | [19.90, 27.10] |
| GQA | 12578 | .6165 | .4207 | .4771 | −5.64 | [−6.42, −4.86] |

Protocol: Qwen3-VL-8B-Instruct, vLLM 0.19.0, eager, native resolution, greedy, maximum 32 output tokens, L2, retention .25. TextVQA/OCRBench/GQA max_num_seqs=8, model length=8192; DocVQA max_num_seqs=4, model length=32768 and max_num_batched_tokens=32768. Same IDs and per-sample token counts verified; zero skips in either compressed arm. None anchors are reused, not freshly rerun; TextVQA two and OCRBench 18 anchor failures are scored zero, others zero failures.

Analysis uses 20,000 paired bootstrap/sign-flip resamples with seed 0 campaign settings (`scripts/analyze_acmmm_final_controls.py:53–54,123–138`), with benchmark-specific seeds. Existing Holm p-values are .00020/.00015/.00010/.00005, respectively. The rewrite can simply provide the CIs without foregrounding significance.

Traceability: `results/acmmm_final_controls/analysis.json`, `cell_summary.csv`, `MANIFEST.sha256`; gzip source predictions under `artifacts/acmmm_final_controls/`; mapping documentation `reports/acmmm_final_controls_artifacts.md`.

Safe interpretation: choosing units using final-ViT input features beats this particular concatenated-output L2 selector on three text benchmarks and loses on GQA. It removes the early-tap difference, but residual scoring-stream/normalization differences prevent a “pure main-merger stage effect” claim.

## 3. Strong baseline evidence

### Same-HF full OCRBench campaign: recommended table

Source: `reports/acmmm_final_controls.md:83–111`, corresponding P1 rows in `cell_summary.csv`.

| Model | Planned n | Common answered n | Shared skips | RBM total /1000 | FastV-k3 total /1000 | Total gain | Paired gain on common answered set | CI of paired gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-VL-8B | 1000 | 921 | 79 | 559 | 413 | +146 points | +15.9 pp | [12.8, 18.9] pp |
| Qwen2.5-VL-7B | 1000 | 912 | 88 | 382 | 295 | +87 points | +9.5 pp | [5.9, 13.2] pp |

Both use `baselines_hf.py`, transformers eager, native-mRoPE RBM, FastV layer 3, retention .25, same IDs/prompts, seed 0, greedy/max32, same 4,000,000-pixel cap. Failures are common long-OCR overrun/OOM images; score them zero in the /1000 totals. **Do not attach the common-answered-set CI to the /1000 difference without explaining its denominator.** No matching uncompressed same-HF/cap full anchor was found in this campaign. A blank/not-measured anchor is more accurate than importing the vLLM none score.

This is a valid same-harness comparison at the same final token budget. FastV processes all image tokens through layer 3, so equal final budget is not equal total compute.

### Existing full FastV TextVQA/GQA: useful but backend-limited

Source: `experiments/r2b_fastv_k3.md`; manifest `experiments/artifact_anonymous_tcsvt_20260904/manifests/S9_file_mapping.tsv:47–49`.
- Qwen3 TextVQA FastV-k3 .7771, full attempted n=5000, five skipped; HF eager. RBM headline .6056 comes from vLLM, so this is not a same-backend direct causal comparison.
- Qwen3 GQA FastV-k3 .5376, n=12578, zero skips; HF eager. RBM headline .4488 comes from vLLM.
- vLLM full none anchors: .8443 TextVQA and .6165 GQA; they are backend-specific references, not proven full HF equivalents.
- Existing HF-vLLM uncompressed equivalence evidence is only 16/16 GQA probe answers (`experiments/j4_baselines_hf_design.md`, `j4_probe_qwen3vl.md`), insufficient to prove full benchmark equivalence.
- FastV manifest mean prompt tokens: TextVQA 213.7; GQA 96.8. These are prompt lengths, not pure visual token counts.
- Filename “n500” for DocVQA/OCRBench FastV is misleading: manifest verifies actual n=200. These are not full-split results.
- Historical OCR probe .415 uses attempted denominator 200, whereas .4586 uses 83/181 common completed denominator; do not mix these with RBM .5801 without consistent denominator.

## 4. Efficiency: what the existing numbers actually measured

Primary numeric source `experiments/j6_efficiency.md`; executable protocol `scripts/j6_efficiency.sh`.
- One A40, Qwen3-VL-8B, vLLM offline generation of TextVQA subset `eval/subsets/textvqa_200.jsonl`, n=200.
- max_num_seqs=8, max_model_len=8192, gpu_memory_utilization=.9; no explicit pixel cap (runner default native), greedy default, maximum 32 generated tokens (variable actual answer lengths).
- Runner defaults/measurement: one first-sample eager warmup excluded; timer surrounds `llm.chat` generation after image loading/message preparation and engine/model loading. Throughput = completed/scored requests divided by timed generation wall time. See runner 4338–4430.
- Batch passed 200 requests; max_num_seqs=8 is engine concurrency cap, not batch size 8. It is not online serving under a specified arrival process.
- Throughput none 4.11 req/s, RBM at 75/50/25% retention 4.61/5.39/6.91, Post-L2 4.52/5.54/6.89. Timed batch walls 48.7, RBM 43.4/37.1/28.9, Post 44.2/36.1/29.0 seconds.
- Mean **prompt** lengths: none 766; compressed 582/397/213, including text. Do not call these visual-only counts.
- Separate latency proxy: first TextVQA example only, n=1, max_num_seqs=1, five independent invocations; mean none .36 s and RBM25% .23 s. This is not latency of the throughput batch, TTFT, tail latency, or an average over representative input lengths.
- No actual output-length distribution or pixel-size distribution was found in the committed J6 digest; do not assert matched generated output totals.
- Memory numbers are allocation-level observations with fixed utilization, not evidence of a measured peak memory reduction.

**Repeatability caveat:** `src/v3_premerger/qwen3_efficiency_repeatability.sh` specifies a separate experiment: max_num_seqs=16, chunk_size=250, max_num_batched_tokens=32768, one discarded process invocation and five repeats, max32. No committed summary/results for this repeat experiment were found. Therefore do not combine its protocol with J6 numbers or state “five throughput repeats within 3%” on the basis of the script alone. J6's five repeats refer to the n=1 latency proxy.

Safe short paper text: “On a TextVQA n=200 offline batch with at most eight concurrent sequences, greedy decoding capped at 32 tokens, and one untimed warmup, RBM at 25% retention processes 6.91 requests/s versus 4.11 without compression and 6.89 for Post-L2. Timing excludes model loading and initial image loading. These measurements concern the specified workload rather than an intrinsic selector speedup.”

## 5. RankBridge: exploratory evidence and unresolved split provenance

Source `experiments/rankbridge_gate.md`.
- Development: planned n=64 each TextVQA/OCRBench, rho in {.1,.2,.3}; maximize their mean score, tie to smaller rho; selected rho=.2. Budget matches reported for 122 answered pairs (consistent with 64 TextVQA +58 OCRBench).
- Evaluation: planned 200 each; effective TextVQA/DocVQA/GQA=200, OCRBench=181 with 19 common loading failures. Common-ID pairing verified in the gate digest. Mean prompt tokens approximately 213/176/115/95.
- DocVQA uses HF pixel cap 600,000; describe the actual cap rather than claiming it is equivalent to a vLLM batch-token limit.
- Native-mRoPE HF eager, FastV-k3, final retention .25, protected quota .2.
- Development/evaluation ID files and the actual gate scripts were not found in the local tracked artifacts; the locked example points to `eval/subsets/textvqa_200.jsonl`. **Disjointness and independent held-out evaluation are unverified.** Do not call this an independent test set or imply proven non-overlap.
- No existing official soft-score paired CI or permutation result for RankBridge was found. Drop the “TextVQA significant” claim; binary discordance z is not an adequate official soft-score test.

| Benchmark | Effective n | None | RBM | FastV-k3 | RankBridge | RankBridge minus FastV |
|---|---:|---:|---:|---:|---:|---:|
| TextVQA | 200 | .8667 | .5967 | .7633 | .7950 | +3.17 pp |
| DocVQA | 200 | .9487 | .4239 | .5863 | .6023 | +1.60 pp |
| OCRBench | 181 | .7569 | .5801 | .4586 | .4641 | +0.55 pp |
| GQA | 200 | .6050 | .4150 | .5050 | .5100 | +0.50 pp |

The original **pre-registered gate failed (NO-GO)** because RankBridge had to stay within one point of the stronger parent on every task; OCRBench was 11.6 points below RBM. It observed small gains over FastV but did not establish a generally superior combined selector. Report as an exploratory, bounded fusion experiment; it is not a validated second primary method contribution.

## 6. Required editorial corrections

1. Replace “pure-stage/matched-input causal isolation” by “final-input control”, with the concatenated-output residual difference stated.
2. State model-specific score aggregation and allocation; no universal mean-patch equation for all architectures.
3. Native weights and merger operations preserved does not imply original retained-position coordinates/order preserved in every backend.
4. Separate main RBM implementation comparisons, final-input control, same-HF strong baseline, and exploratory fusion.
5. Keep numerator/denominator conventions explicit for OCRBench and failures.
6. Report the negative GQA result and failed fusion gate; avoid generic “never loses/robust default”.
7. Separate throughput and n=1 latency protocols; remove unverified repeatability assertion.
8. None of the unresolved evidence should be filled by assumptions or new numbers.


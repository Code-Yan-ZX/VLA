# GQA pinned lmms-eval CPU rescore (2026-10-25)

No new model inference. The exact pinned upstream function is executed for every non-failed example.

Reference commit: `8f40097532a7bf4db6123713f1a905e02c8f5971`. [Metric source](https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/8f40097532a7bf4db6123713f1a905e02c8f5971/lmms_eval/api/metrics.py); [GQA task configuration](https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/8f40097532a7bf4db6123713f1a905e02c8f5971/lmms_eval/tasks/gqa/gqa.yaml).

The task enables ignore_case and ignore_punctuation only. It does not strip whitespace, drop articles, map number words, expand contractions, or split semicolon answer variants.

All attempted splits contain 12,578 IDs; failed samples receive zero in means and paired intervals.

| Cell | Attempted / answered | Official attempted | Legacy attempted | Changed examples |
|---|---:|---:|---:|---:|
| qwen3vl_none | 12578 / 12578 | 0.61647321 | 0.61647321 | 0 |
| qwen3vl_pre | 12578 / 12578 | 0.44864048 | 0.44879949 | 2 |
| qwen3vl_post | 12578 / 12578 | 0.47694387 | 0.47710288 | 2 |
| qwen2vl_none | 12578 / 12578 | 0.60446812 | 0.60446812 | 0 |
| qwen2vl_pre | 12578 / 12578 | 0.55708380 | 0.55938941 | 29 |
| qwen2vl_post | 12578 / 12578 | 0.58522818 | 0.58522818 | 0 |
| internvl3_none | 12578 / 12578 | 0.62871681 | 0.62927333 | 7 |
| internvl3_pre | 12578 / 12578 | 0.59850533 | 0.59930037 | 10 |
| internvl3_post | 12578 / 12578 | 0.60279854 | 0.60311655 | 4 |
| qwen3vl_pre_final | 12578 / 12578 | 0.41978057 | 0.42073462 | 12 |
| qwen3vl_post_main | 12578 / 12578 | 0.48163460 | 0.48179361 | 2 |

| A minus B | Paired n | Delta pp | Paired 95% CI pp |
|---|---:|---:|---:|
| qwen3vl_pre_minus_qwen3vl_post | 12578 | -2.83034 | [-3.52202, -2.15456] |
| qwen2vl_pre_minus_qwen2vl_post | 12578 | -2.81444 | [-3.52997, -2.11480] |
| internvl3_pre_minus_internvl3_post | 12578 | -0.42932 | [-0.98585, +0.11926] |
| qwen3vl_pre_final_minus_qwen3vl_post | 12578 | -5.71633 | [-6.49547, -4.92129] |
| qwen3vl_post_main_minus_qwen3vl_pre_final | 12578 | +6.18540 | [+5.43012, +6.95659] |

Each interval uses 20,000 paired empirical bootstrap draws. Multinomial counts over the empirical score-difference support implement exactly the same bootstrap distribution as resampling paired IDs, without allocating a large index array.

Seven targeted normalization checks passed. JSON records source digests, changed IDs, all seeds, failure handling, and pairing checks. Raw answers are not copied into the report.

The Qwen3 control Post-concat artifact and operational-matrix baseline have identical IDs, answers, GT, and skip flags on all 12,578 examples. All five reported comparison directions remain unchanged after official rescoring.

Missing sources: 0.

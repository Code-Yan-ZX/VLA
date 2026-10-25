# OCRBench pinned lmms-eval CPU audit (2026-10-25)

Source: https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/8f40097532a7bf4db6123713f1a905e02c8f5971/lmms_eval/tasks/ocrbench/utils.py

All 1000 attempted IDs; shared skips score zero. Includes fullwidth ASCII folding and HME-specific rules.

| Cell | Score /1000 | Legacy /1000 | Changed items |
|---|---:|---:|---:|
| qwen3vl_none | 760 | 760 | 0 |
| qwen3vl_pre | 547 | 547 | 0 |
| qwen3vl_post | 184 | 184 | 0 |
| qwen3vl_pre_final | 419 | 419 | 0 |
| qwen3vl_post_main | 356 | 356 | 0 |
| qwen2vl_none | 817 | 817 | 0 |
| qwen2vl_pre | 480 | 480 | 0 |
| qwen2vl_post | 182 | 182 | 0 |
| internvl3_none | 852 | 852 | 0 |
| internvl3_pre | 753 | 753 | 0 |
| internvl3_post | 321 | 321 | 0 |
| qwen3vl_hf_pre | 559 | 559 | 0 |
| qwen3vl_hf_fastv | 413 | 413 | 0 |
| qwen2vl_hf_pre | 382 | 382 | 0 |
| qwen2vl_hf_fastv | 295 | 295 | 0 |

| A minus B | Delta /1000 | 95% CI /1000 |
|---|---:|---:|
| qwen3vl_pre_minus_qwen3vl_post | +363.0 | [+326.0, +399.0] |
| qwen2vl_pre_minus_qwen2vl_post | +298.0 | [+263.0, +333.0] |
| internvl3_pre_minus_internvl3_post | +432.0 | [+397.0, +466.0] |
| qwen3vl_pre_final_minus_qwen3vl_post | +235.0 | [+200.0, +271.0] |
| qwen3vl_pre_final_minus_qwen3vl_post_main | +63.0 | [+27.0, +100.0] |
| qwen3vl_hf_pre_minus_qwen3vl_hf_fastv | +146.0 | [+118.0, +174.0] |
| qwen2vl_hf_pre_minus_qwen2vl_hf_fastv | +87.0 | [+54.0, +120.0] |

# TextVQA official metric CPU rescore (2026-10-25)

Reference: https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/8f40097532a7bf4db6123713f1a905e02c8f5971/lmms_eval/tasks/_task_utils/vqa_eval_metric.py

No new model inference. Skips receive zero in attempted scores.

| Cell | Attempted/answered | Official attempted | Official answered | Legacy scorer answered |
|---|---:|---:|---:|---:|
| qwen3vl_none | 5000/4998 | 0.834300 | 0.834634 | 0.844338 |
| qwen3vl_pre | 5000/5000 | 0.597840 | 0.597840 | 0.605333 |
| qwen3vl_post | 5000/5000 | 0.218100 | 0.218100 | 0.221667 |
| qwen2vl_none | 5000/4998 | 0.852860 | 0.853201 | 0.862345 |
| qwen2vl_pre | 5000/5000 | 0.694100 | 0.694100 | 0.702267 |
| qwen2vl_post | 5000/5000 | 0.436040 | 0.436040 | 0.441733 |
| internvl3_none | 5000/5000 | 0.825120 | 0.825120 | 0.833800 |
| internvl3_pre | 5000/5000 | 0.780420 | 0.780420 | 0.789000 |
| internvl3_post | 5000/5000 | 0.409260 | 0.409260 | 0.414800 |
| qwen3vl_pre_final | 5000/5000 | 0.491700 | 0.491700 | 0.498467 |
| qwen3vl_post_main | 5000/5000 | 0.312300 | 0.312300 | 0.317000 |
| qwen3vl_hf_rbm | 5000/4995 | 0.755040 | 0.755796 | 0.764831 |
| qwen3vl_hf_fastv | 5000/4995 | 0.768260 | 0.769029 | 0.777845 |
| qwen3vl_hf_old_fastv | 5000/4995 | 0.768260 | 0.769029 | 0.777845 |

Paired deltas use common answered IDs, except the explicitly labelled all-attempted row (shared skips score zero).

| A minus B | Paired n | Delta pp | 95% CI pp |
|---|---:|---:|---:|
| qwen3vl_pre_minus_qwen3vl_post | 5000 | +37.9740 | [+36.5080, +39.4480] |
| qwen2vl_pre_minus_qwen2vl_post | 5000 | +25.8060 | [+24.3240, +27.3040] |
| internvl3_pre_minus_internvl3_post | 5000 | +37.1160 | [+35.7079, +38.5401] |
| qwen3vl_pre_final_minus_qwen3vl_post | 5000 | +27.3600 | [+25.8819, +28.8060] |
| qwen3vl_pre_final_minus_qwen3vl_post_main | 5000 | +17.9400 | [+16.4500, +19.4520] |
| qwen3vl_hf_rbm_minus_qwen3vl_hf_fastv | 4995 | -1.3233 | [-2.3303, -0.3263] |
| qwen3vl_hf_rbm_minus_qwen3vl_hf_fastv_all_attempted | 5000 | -1.3220 | [-2.3021, -0.3200] |

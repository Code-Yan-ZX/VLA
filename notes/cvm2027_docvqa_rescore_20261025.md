# DocVQA pinned lmms-eval ANLS CPU rescore (2026-10-25)

Source: https://raw.githubusercontent.com/EvolvingLMMs-Lab/lmms-eval/8f40097532a7bf4db6123713f1a905e02c8f5971/lmms_eval/api/metrics.py

Saved GT variants; no new inference; all 5349 attempted IDs included.

| Cell | Score | Legacy score | Changed items |
|---|---:|---:|---:|
| qwen3vl_none | 0.95633616 | 0.95622654 | 14 |
| qwen3vl_pre | 0.48089675 | 0.48063906 | 12 |
| qwen3vl_post | 0.23780170 | 0.23765761 | 7 |
| qwen3vl_pre_final | 0.28369566 | 0.28357423 | 5 |
| qwen3vl_post_main | 0.37368419 | 0.37353868 | 8 |
| qwen2vl_none | 0.94915529 | 0.94905768 | 12 |
| internvl3_none | 0.92234990 | 0.92214272 | 15 |
| qwen2vl_pre | 0.63612581 | 0.63609349 | 4 |
| internvl3_pre | 0.72849848 | 0.72838161 | 13 |
| qwen2vl_post | 0.52582762 | 0.52564694 | 14 |
| internvl3_post | 0.38203448 | 0.38196704 | 6 |

| A minus B | Delta pp | 95% CI pp |
|---|---:|---:|
| qwen3vl_pre_minus_qwen3vl_post | +24.309505 | [+23.009919, +25.590573] |
| qwen2vl_pre_minus_qwen2vl_post | +11.029819 | [+9.550128, +12.494736] |
| internvl3_pre_minus_internvl3_post | +34.646400 | [+33.203741, +36.105979] |
| qwen3vl_pre_final_minus_qwen3vl_post | +4.589396 | [+3.411259, +5.767857] |
| qwen3vl_pre_final_minus_qwen3vl_post_main | -8.998853 | [-10.165729, -7.822192] |

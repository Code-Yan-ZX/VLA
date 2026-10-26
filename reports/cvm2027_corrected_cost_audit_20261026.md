# Independent corrected P3 audit (2026-10-26)

All ten configurations pass: 200 fixed requests each, three raw repeats, 2,000 request medians and 6,000 raw measurements over the same 200 input IDs. No request fails, and all repeat token sequences and answers agree. Actual K and every one of the 36 decoder-layer token counts match the frozen rules.

Independent execution of the pinned upstream task functions reproduces all 20 task/config scores. Recomputed per-request medians, task p50/p95 and raw-repeat memory maxima reproduce the published cost summary. Hi-Lo no-op checks match all 32 native input, position, embedding and deepstack hashes, full greedy outputs and first logits.

| Task (n=100) | 95% target | RBM minimum measured p50 | FastV minimum measured p50 | Hi-Lo minimum measured p50 |
|---|---:|---|---|---|
| textvqa | 77.615 | pre_0.5: 0.295s | fastv_0.25: 0.269s | hilo_0.75: 0.494s |
| ocrbench | 741.000 | pre_0.5: 0.642s | fastv_0.75: 0.729s | hilo_0.75: 0.978s |

The rule chooses actual p50 among observed qualifying points. On this TextVQA sample FastV25 qualifies; on OCRBench RBM50 qualifies. These do not replace the older mixed-protocol curve thresholds. Some qualifying Hi-Lo points and the OCRBench FastV75 point are slower than uncompressed None; meeting the quality threshold does not imply speedup.

Preserve the limitations recorded in the companion JSON: fixed n=100 per task, output-length differences, and no inference of quality equivalence from a CI crossing zero. Memory is GiB and uses maxima over raw repeats, rather than maxima of per-request median peaks.

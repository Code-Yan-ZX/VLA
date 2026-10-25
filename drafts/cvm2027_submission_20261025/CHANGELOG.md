# CVM 2027 revision, 2026-10-25

- Copied only the manuscript, bibliography, and four used figure assets; the
  original September 16 directory is unchanged.
- Adopted the official anonymous `cvmpaper_for_review.tex` structure: 10-point
  Times, two columns, US Letter paper, the unchanged `cvm.sty`, and `cvm.bst`.
- Removed names, affiliations, email addresses, corresponding-author details,
  the identifiable code link, and PDF author metadata.
- Retained the official paper-ID placeholder; no submission number was invented.
- Changed the title to `Native-Unit Selection for Visual Token Compression`.
- Preserved the original Figure 1 image content through TeX viewport crops, but
  omitted the independent `Only the selection domain changes` slogan and
  `text evidence demoted` label. Added neutral TeX labels.
- Preserved the six original Figure 4 spatial panels through TeX viewport crops.
  Omitted gray causal-interpretation boxes and historical aggregate-score headings;
  kept the historical diagnostic provenance in the caption.

- Replaced repeated defensive sentences with explicit adapter definitions and
  one discussion of the measured scope. Defined separate Qwen mean-patch-norm
  and InternVL3 concatenated-vector-norm scores and image/tile budget groups.
- Added the Post-main control alongside Pre-final and Post-concat, including the
  DocVQA reversal and task-dependent input/output ranking results.
- Recalculated every reported TextVQA cell and associated confidence interval
  from original predictions using the official normalization and consensus
  metric. Corrected the prior simplified-consensus and binary-containment scores.
- Added the matched-harness TextVQA result favoring FastV; point estimates and
  bootstrap intervals both use all 5,000 attempted examples, with five common
  failures assigned zero.
- Recalculated DocVQA scores and control confidence intervals with the pinned
  lmms-eval ANLS implementation. The main-only DocVQA advantage remains 9.00
  percentage points; displayed control scores and interval endpoints are updated.
- Recalculated all GQA scores with the pinned official case/punctuation-insensitive
  exact-match rule. The main-only control advantage is 6.19 percentage points.
  Audited all 15 OCRBench cells with the pinned upstream scorer including
  fullwidth normalization: all reproduce the saved scores without changes.
- Updated control and FastV confidence intervals using 20,000 resamples and the
  complete attempted-split denominator, including shared zero-score failures.
- Replaced unsupported byte-identical/all-prediction-equivalence language with
  the verified probe counts and answer-agreement counts.
- Reported the 64-image-per-benchmark diagnostic sampling scope. Removed the
  single-example latency proxy and retained the specified 200-example throughput
  measurement.
- Removed the old three-arm control chart and its stale TextVQA scores; the
  updated control table directly presents the added main-only variant.
- Preserved the original 51 substantively cited works, including the closest
  query-guided native-unit method. Restored grouped comparisons of pruning
  stage, selection signal, and evidence preservation, without using `\nocite`.
- Used a standard float barrier so all figures precede the reference section,
  with final-page column balancing. The official style, fonts, and margins are
  unchanged.

This is a scientific revision review draft. The narrowed contribution and
remaining competitiveness questions require author review before submission.

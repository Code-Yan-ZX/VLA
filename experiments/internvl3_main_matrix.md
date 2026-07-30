# InternVL3-8B main matrix — digest (third model family, official metrics, full splits)

## Config
Model OpenGVLab/InternVL3-8B (pixel-shuffle merger family; runner family branch).
Modes none / pre (RBM L2 top-κ pre-merger) / post (FastV-style layer-2 attn prune),
@ r0.75 (25% keep; waves 1–3) + pre/post @ r0.875 (12.5% keep; wave 4, text-dense only).
Benchmarks full splits: TextVQA val 5000 / GQA testdev 12578 / OCRBench 1000 / DocVQA val 5349.
DocVQA pixel cap = 4M px (script STD; PIL pre-resize path — NOT Qwen's 600k: InternVL3
dynamic tiling tolerates more, and InternVLImageProcessor rejects max_pixels kwarg, see
DECISIONS 2026-07-29 fix f2ff06d). vLLM 0.19 V1, mns4/chunk500 (docvqa: mns4/chunk300,
max-model-len 32768). Official rescore: VQA-acc / ANLS / exact-match / OCRBench /1000.
**16/16 cells, zero missing** (wave-1 none_docvqa backfilled 2026-07-30 after the
max_pixels fix; all cells skip=0 except noted).

## Official results
| benchmark (metric) | none | pre@25% | post@25% | pre−post | pre vs none | ptid (pre/post) |
|---|---|---|---|---|---|---|
| TextVQA (VQA-acc) | 0.8338 | **0.7890** | 0.4148 | **+37.4pp** | −4.5pp | 690 |
| DocVQA (ANLS) | 0.9221 | **0.7284** | 0.3820 | **+34.6pp** | −19.4pp | 860 |
| OCRBench (/1000) | 852 | **753** | 321 | **+432pts** | −99 | 555 |
| GQA (acc) | 0.6293 | 0.5993 | 0.6031 | −0.4pp (tie) | −3.0pp | 690 |

Deep budget r0.875 (12.5% keep): TextVQA pre 0.7230 vs post 0.3064 (+41.7pp);
DocVQA pre 0.5054 vs post 0.2451 (+26.0pp). ptid 378/464.

## Efficiency (within-run relative, mns4 config — NOT comparable to Qwen runs)
req/s none→pre: TextVQA 1.716→3.770 (2.2×) · GQA 1.888→4.630 (2.5×) ·
DocVQA 1.518→3.683 (2.4×) · OCRBench 1.565→2.830 (1.8×). post ≈ pre req/s
(identical ptid): e.g. TextVQA post 3.652. Compression speedup reproduces.

## Cross-family isomorphism (vs Qwen3-VL main table, official)
pre−post text-dense gaps: InternVL3 +37.4/+34.6pp/+432pts ≈ Qwen3-VL
+38.4/+24.3/+363pts — same order, same direction, all three text-dense benches.
GQA: InternVL3 tie (−0.4pp) vs Qwen3 post-narrow-win (+2.8pp) — both within the
"no text-dense crossover, object-centric GQA at most a small post edge" red line.
Stage law (pre ≫ post on text-dense; query-blind selection at the merger boundary
protects raw-patch OCR information that layer-2 attention pruning destroys) is now
established on a THIRD model family with a different merger (pixel-shuffle vs
Qwen PatchMerger) — supports the "merger-equipped VLMs" generalization framing.

## Caveats (disclose in paper)
- DocVQA absolute values use a 4M-px cap on InternVL3 vs 600k on Qwen — within-model
  comparisons valid; cross-model DocVQA absolutes NOT comparable (state per-model config).
- pre@25% DocVQA costs −19.4pp vs none (TextVQA only −4.5pp): near-lossless claim is
  TextVQA/GQA-scoped; DocVQA/OCR keep a large pre>post margin but pre≠none there.
- GQA "tie" at n=12578: −0.4pp, report as no meaningful difference (contrast Qwen3
  full-split +2.8pp post edge — family-dependent, both small).

## Assets
runs/internvl3/internvl3_official_summary.json (16 entries) · per-cell
runs/internvl3/internvl3_*.json · logs runs/internvl3/*.log ·
backfill log runs/internvl3_backfill.out · fix f2ff06d.

# Rank-Before-Merge (RBM) — VLM Visual Token Compression

Training-free, stage-level visual token pruning for large vision-language
models. We isolate *where* pruning happens relative to the native 2×2
patch-merger, show that **ranking before merge** materially outperforms the
common **merge-then-rank** recipe on text-dense / OCR workloads, and bound the
effect with mechanism + negative-result controls.

> **Standing result (do not overclaim):** RBM is a *robust text-dense/OCR
> default*, **not** a universal SOTA winner. On scene-text/object-centric cells
> a query-conditioned baseline (FastV-class) remains stronger. Read the paper's
> claims in `drafts/overleaf_submission/main.tex` before citing numbers.

---

## TL;DR — what we did and what we found

| Question | Answer |
|---|---|
| Does *pre-merge* vs *post-merge* ranking matter? | Yes — a pure **stage** change (same L2 scorer, same token budget) is worth up to **+38.4pp** (TextVQA) / **+24.3pp** (DocVQA) over post-merger pruning at 25% retention on Qwen3-VL-8B (+37.4/+34.6pp on InternVL3-8B). |
| Why? | The merger is *systematically anti-text*: it demotes high-edge (text-stroke) units (M2), and swapping the post ranking into the pre path reproduces the pre result at the *kept-set* level (M3, Jaccard = 1.000 with pre top-κ). |
| Across models? | Qwen3-VL-8B (byte-level causal), Qwen2.5-VL-7B (kept-set level), InternVL3-8B (accuracy level). Two architecture families, native 2×2 merger + deepstack design. |
| When does the gap appear? | **Compression-activated**: near zero at 75% retention, widens monotonically through 25%. |
| Why not a universal SOTA? | GQA shows no detected pure-stage difference; FastV-style query-conditioned pruning still wins several scene-text / object cells. Extensions we prespecified (router, hybrid masking, QA-blend, cascade, frequency-aware scorer, adaptive stage router) all fail their acceptance gates. |

**12-sentence abstract + full method/experiments:** `drafts/overleaf_submission/main.tex` (submission authority).

---

## Method in one line

Score each 2×2 merge unit on its **merger-*input*** features (query-blind L2),
keep the top-κ units, and let the unmodified native merger pass only the
survivors. Because the merger is a per-unit op, a kept unit's merged token is
bit-identical regardless of which stage selected it — RBM changes only *which*
units reach the merger, at iso-token budget with the post-merger (FastV-style)
baseline.

```
merge-then-rank (post/FastV-style)   :  [patch grid] → 2×2 merge → rank merged → keep κ
rank-before-merge (RBM, ours)         :  [patch grid] → rank units  → keep κ → 2×2 merge
```

---

## Headline experiments

All official metrics (TextVQA VQA-acc, DocVQA ANLS, OCR-Bench /1000 final score,
GQA exact-match); full splits; `κ` = kept fraction.

### Stage law (Table 1) — full-split, three models

Exact values from the submission (`main.tex` Table~\ref{tab:tab1}); OCR-Bench is
reported as the official /1000 final score.

| model | bench | none | RBM @25% | post @25% | Δpre−post |
|---|---|---|---|---|---|
| Qwen3-VL-8B | TextVQA (VQA-acc) | 0.844 | **0.605** | 0.222 | **+38.4pp** |
| Qwen3-VL-8B | DocVQA (ANLS) | 0.956 | **0.481** | 0.238 | **+24.3pp** |
| Qwen3-VL-8B | OCR-Bench (/1000) | 760 | **547** | 184 | **+363 pts** |
| Qwen3-VL-8B | GQA (exact) | 0.616 | 0.449 | 0.477 | −2.8pp (n.s.) |
| Qwen2.5-VL-7B | TextVQA (VQA-acc) | 0.862 | **0.702** | 0.442 | **+26.1pp** |
| Qwen2.5-VL-7B | DocVQA (ANLS) | 0.949 | **0.636** | 0.526 | **+11.0pp** |
| Qwen2.5-VL-7B | OCR-Bench (/1000) | 817 | **476** | 183 | **+293 pts** |
| Qwen2.5-VL-7B | GQA (exact) | 0.604 | 0.559 | 0.585 | −2.6pp (n.s.) |
| InternVL3-8B | TextVQA (VQA-acc) | 0.834 | **0.789** | 0.415 | **+37.4pp** |
| InternVL3-8B | DocVQA (ANLS) | 0.922 | **0.728** | 0.382 | **+34.6pp** |
| InternVL3-8B | OCR-Bench (/1000) | 852 | **753** | 321 | **+432 pts** |
| InternVL3-8B | GQA (exact) | 0.629 | 0.599 | 0.603 | −0.4pp (n.s.) |

\*RBM remains lossy vs *no compression* (it prunes 75% of visual tokens at
κ=25%); the contrast claimed is pre-vs-post **at fixed budget**, not
RBM-vs-full. GQA rows carry a dagger in the paper: no detected pure-stage gain,
and RBM trails FastV there (Table 2).

### Regime map vs FastV (Table 2) — query-blind RBM default vs query-conditioned baseline

FastV (ECCV 2024 Oral, FastV-k3 setting) leads on TextVQA/GQA/DocVQA@600k-cap
(Qwen3-VL and mostly Qwen2.5-VL); RBM retains dense OCR better. Headline rows
(`main.tex` Table~\ref{tab:tab2}): Qwen3-VL OCR-Bench **RBM 0.575 vs FastV
0.415 (+16.0 R)**; Qwen2.5-VL OCR-Bench **RBM 0.370 vs 0.285 (+8.5 R)**,
DocVQA **0.506 vs 0.485 (+2.1 R)**, GQA **0.520 vs 0.475 (+4.5 R)**. FastV is
stronger on Qwen3-VL TextVQA (0.777 vs 0.605, +17.2 F) and GQA (0.538 vs 0.449,
+8.9 F). Full table: `drafts/overleaf_submission/main.tex` +
`runs/full_matrix/j7_official_summary.json` (n=5000/12578 full; n=200 dev cells
marked).

### Mechanism (M1–M3)

- **M1** rank reshuffle: pre-vs-post unit-ranking order changes under the merger.
- **M2** the demoted units are text strokes: post↔pre rank shift correlates with
  Sobel edge energy (ρ = +0.439 DocVQA / +0.155 TextVQA / +0.036 GQA);
  dropped units are the highest-edge of all (DocVQA mean Sobel **0.641 vs 0.124**).
- **M3** causal kept-set identity: post+swap (pre ranking, post path) selects
  **exactly** the pre top-κ set (Jaccard 1.000, both Qwen archs) and restores
  pre accuracy; the collapse then reappears with the post ranking.

### Prespecified negative results (we do not claim these compose)

Six query-/regime-aware extensions all fail on Qwen3-VL-8B: image-level router,
merger-aware hybrid mask, query-embedding QA-blend, RBM→FastV cascade, a
variance-weighted (frequency-aware) scorer, and an adaptive PRE/POST stage
router. Details + official tables in §Negative Results and Supplementary S4b/S4c.

---

## Where the current SOTA stands (and where we cite it)

These are the baselines we compare against, with their publication venues
(entries already in `drafts/overleaf_submission/references.bib`):

| Baseline | Method | Venue | Verified id |
|---|---|---|---|
| **FastV** | prune image tokens by text→image attention after LLM layer 2 | **ECCV 2024 (Oral)** | arXiv 2403.06764 |
| **VisionZip** | dominant tokens + redundant-token compression between vision encoder and LLM | **CVPR 2025** | arXiv 2412.04467 |
| **PyramidDrop** | layer-wise pyramidal visual-token dropping in the LLM | **CVPR 2025** | arXiv 2410.17247 |

**Positioning on the SOTA map:** the mature vision-token-compression line
(FastV, VisionZip, PyramidDrop) prunes **after** the native 2×2 merger (or
inside the LLM) using local attention/dominance signals. RBM is orthogonal to
that axis: it changes only the *ranking hook point* relative to the merger, is
fully training-free and query-blind, and its mechanism is verified down to the
kept-set level. We do **not** claim RBM beats these baselines as a universal
compressor; we claim a *stage law* (pre-post gap, compression-activated,
workload-conditional) and position RBM as the robust text-dense default. Note:
our FastV/PyramidDrop numbers are official-fastv-parameter ports per the J4
protocol (`experiments/j4_baselines_hf_design.md`), not re-trained models.

---

## Repo layout

```
src/v3_premerger/        runner + selectors (pre/post/hybrid/pre-final/adaptive),
                         official_scorers, baseline HF ports (FastV/PyramidDrop)
scripts/                 grids, drivers, analysis, figure/mask capture
runs/                    per-cell JSONs + logs (gitignored where large)
experiments/             analysis notes + gates (.md) — incl. this work's
                         freq_aware/ adaptive_stage/ combined/ results
notes/                   design notes, survey, mechanism analyses
drafts/overleaf_submission/   authoritative paper (main.tex, supp.tex, references.bib)
eval/                    full-split jsonl + subsets
```

## Reproduce

Qwen3-VL-8B on a single A40 46GB; every cell resumable and self-healing.

```bash
# stage law cell (plain RBM, TextVQA full split @ 25% keep)
python src/v3_premerger/v3_premerger_runner.py \
  --model-family qwen3vl --mode pre --benchmark textvqa \
  --subset eval/full_splits/textvqa_val.jsonl --n 5000 --r 0.75 \
  --selector l2 --max-num-seqs 8 --max-model-len 8192 \
  --gpu-memory-utilization 0.9 --out runs/rbm_textvqa_r0.75.json

# Frequency-aware scorer probe (α=1, β=0.6; see experiments/freq_aware/results.json)
python src/v3_premerger/v3_premerger_runner.py \
  --model-family qwen3vl --mode pre --benchmark textvqa \
  --subset eval/full_splits/textvqa_val.jsonl --n 500 --r 0.75 \
  --selector freq --alpha 1.0 --beta 0.6 \
  --max-num-seqs 8 --max-model-len 8192 \
  --gpu-memory-utilization 0.9 --out runs/freq_probe.json

# Adaptive PRE/POST stage router (see experiments/adaptive_stage/results.json)
python src/v3_premerger/v3_premerger_runner.py \
  --model-family qwen3vl --mode adaptive --benchmark gqa \
  --subset eval/full_splits/gqa_testdev.jsonl --n 500 --r 0.75 \
  --tau-hf 0.08 --tau-ent 2.0 --hf-var-mode mean1sd \
  --max-num-seqs 8 --max-model-len 8192 \
  --gpu-memory-utilization 0.9 --out runs/adapt_probe.json
```

`--dry-check` runs a no-GPU sanity pass (hook wiring, selector blend, adaptive
routing) before any weighted run.

## Offline / fidelity notes

- Official metrics are the peer-review metric (`src/v3_premerger/official_scorers.py`);
  the runner's inline `acc` is a fast loose approximation and must not be cited.
- Use `eval/full_splits/*.jsonl` (they carry the prompt's "Answer … single word"
  suffix); the n-small `eval/subsets/*.jsonl` drop that suffix and yield verbose
  answers that official scoring marks 0.
- Weights/datasets/logs are gitignored; per-cell reproducibility index in
  Supplementary S9 (run JSONs released with code).

## Status

Research main-line for the ACM MM'27 submission. All experiments on 1× A40
(single-GPU, serial). See `STATE.md` for the current vertex.
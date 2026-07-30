# Supplementary Material — Rank Before You Merge: A Stage Law for Training-Free Visual Token Compression in Merger-Equipped Vision-Language Models

*ACM MM'27 submission — supplementary file (≤ 50 MB; the CFP allows no appendix after the body, so
all overflow tables live here). Double-blind. Every number is labeled with its source digest
(`experiments/*.md`) or run-JSON path; `[TODO: runs JSON]` marks cells to be pasted from the cited
runs at packaging time (no cross-run subtraction; no unsourced numbers). Tables S1–S3/S7 support §5;
S4–S6 support §4 (S4b cascade counts support §6); S8 supports §3; S9 is the per-cell reproducibility
index, with the full per-cell listing (cell / n / official / ptid / skip / JSON path) in S9.1–S9.4.*

---

## S1. M1 — merger ranking reshuffle, full table (Qwen3-VL-8B, seed-0, n = 64/benchmark)

Per image, Spearman correlation of the pre ranking (merger-input unit L2) with the post ranking
(merged-token L2) over all units, plus top-25% kept-set overlap. Source: mechanism_verification_report
via `drafts/paper_v4.md` §4 (measured under `experiments/j3_mechanism_crossarch.md` campaign).

| Benchmark (type) | Spearman ρ | Jaccard@25% | Kendall τ [TODO: runs JSON] |
|---|---|---|---|
| DocVQA (text-dense document) | **0.137** | **0.180** | [TODO: runs JSON] |
| TextVQA (text-dense scene) | 0.332 | 0.243 | [TODO: runs JSON] |
| GQA (object-centric) | **0.360** | **0.278** | [TODO: runs JSON] |

Reading: the reshuffle is largest on the most text-dense benchmark and smallest (but still
substantial) on object-centric GQA — ρ = 0.137 < 0.332 < 0.360; Jaccard = 0.180 < 0.243 < 0.278.
Chance-level Jaccard at κ = 25% is 0.25 (overlap is below random for DocVQA, i.e. actively
anti-correlated in the decision band).

## S2. M2 — demotion directionality vs text-stroke energy, full table (same sample as S1)

rank_shift = post_rank − pre_rank (+ ⇒ demoted); per-unit Sobel edge energy as the text-stroke proxy.
Source: mechanism_verification_report via `drafts/paper_v4.md` §4.

| Benchmark | ρ(rank_shift, Sobel edge) | mean Sobel, post-drops-pre-keeps | mean Sobel, reverse group | % demoted above per-image median |
|---|---|---|---|---|
| DocVQA | **+0.439** | **0.641** | 0.124 | **92% (vs 35%)** |
| TextVQA | +0.155 | [TODO: runs JSON] | [TODO: runs JSON] | [TODO: runs JSON] |
| GQA | +0.036 | [TODO: runs JSON] | [TODO: runs JSON] | [TODO: runs JSON] |

Reading: the merger preferentially demotes high-edge (text-stroke) units ~12× more strongly on
documents than on object scenes; on DocVQA the units post drops that pre keeps are the highest-edge
of all (92% above the per-image median vs 35% for the reverse group) — post-merger selection is
systematically anti-text on documents. Directionality hypothesis (merger trained on natural content
attenuates high-frequency stroke energy): testable hypothesis, not a measured result. Second
text-stroke proxy (Laplacian variance): future work (single-proxy disclosure).

## S3. Retention curve — full number strings (n = 200, both Qwen models)

pre−post Δ at 75% / 50% / 25% retention (the body §5.4 prints endpoints only). Source:
`experiments/j8_ablations.md` via `drafts/paper_v4.md` §5.6.

| Model | Benchmark | Δ @75% | Δ @50% | Δ @25% |
|---|---|---|---|---|
| Qwen3-VL-8B | TextVQA | +8.7 pp | +29.0 pp | **+38.4 pp** |
| Qwen3-VL-8B | DocVQA | −1.3 pp | +5.9 pp | **+24.3 pp** |
| Qwen2.5-VL-7B | TextVQA | +7.0 pp | +15.3 pp | **+26.1 pp** |
| Qwen2.5-VL-7B | DocVQA | −2.1 pp | −1.4 pp | **+11.0 pp** |

Reading: the two stages are indistinguishable under shallow compression (DocVQA post even leads
1–2 pp within noise at 75%); the pre-merger lead emerges and widens monotonically with depth. Same
shallow regime as the GQA sign and the Qwen2.5-VL 600k-cap reversal (body §5.4 boundary; §8 item 3).
Data source for FIG:3.

## S4. Primary paired McNemar statistics — per cell (full splits, both Qwen models)

Primary test = paired McNemar z on per-sample correctness, z = (b − c)/√(b + c), with b/c the
pre-only / post-only correct counts; positive z = pre-merger lead, negative = post-merger lead (GQA
only). Cross-check = independent-binomial z (SE √(se²_pre + se²_post)); "agree %" = per-question
agreement of the recorded correctness indicators. Source: `drafts/paper_v4.md` §5.2 paired-test table
(per-sample rescore of `runs/full_matrix/j7_*_full.json`, paired by question id, restricted to ids
answered in both arms; 200/200 ground-truth self-test) + `experiments/j7_main_table.md`.

| Model | Benchmark | κ | Δ (official) | McNemar z (primary) | b (pre-only) | c (post-only) | agree % | indep-binom z |
|---|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | TextVQA | .25 | +38.4 pp | **+43.0** | 2349 | 183 | 49.4 | 42.3 |
| Qwen3-VL-8B | DocVQA | .25 | +24.3 pp | **+34.7** | 1601 | 150 | 67.3 | 27.1 |
| Qwen3-VL-8B | OCR-Bench | .25 | +363 pts | **+16.7** | 422 | 56 | 52.2 | 18.2 |
| Qwen3-VL-8B | GQA | .25 | −2.8 pp † | **−8.0** | 834 | 1196 | 83.9 | 4.5 |
| Qwen3-VL-8B | TextVQA | .125 | +34.0 pp | **+40.8** | 2101 | 161 | 54.8 | 39.9 |
| Qwen3-VL-8B | DocVQA | .125 | +24.9 pp | **+33.1** | 1437 | 127 | 70.8 | 32.1 |
| Qwen3-VL-8B | OCR-Bench | .125 | +297 pts | **+16.0** | 323 | 25 | 65.2 | 17.8 |
| Qwen2.5-VL-7B | TextVQA | .25 | +26.1 pp | **+30.8** | 1682 | 309 | 60.2 | 27.3 |
| Qwen2.5-VL-7B | DocVQA | .25 | +11.0 pp | **+15.6** | 1406 | 693 | 60.8 | 11.6 |
| Qwen2.5-VL-7B | OCR-Bench | .25 | +293 pts | **+14.6** | 347 | 54 | 59.9 | 14.7 |
| Qwen2.5-VL-7B | GQA | .25 | −2.6 pp † | **−5.7** | 892 | 1150 | 83.8 | 4.1 |
| Qwen2.5-VL-7B | TextVQA | .125 | +27.8 pp | **+33.1** | 1836 | 305 | 57.2 | 29.1 |
| Qwen2.5-VL-7B | DocVQA | .125 | +21.0 pp | **+25.6** | 1325 | 294 | 69.7 | 23.3 |
| Qwen2.5-VL-7B | OCR-Bench | .125 | +268 pts | **+15.0** | 294 | 26 | 68.0 | 15.9 |

† the only significant post-stage lead. GQA official word-normalized exact-match rescore gives a
concordant McNemar z = 8.1 / 7.1 (Qwen3-VL / Qwen2.5-VL). Smallest text-dense Δ₂₅ |z| = 14.6
(Qwen2.5-VL OCR-Bench); smallest text-dense Δ₁₂.₅ |z| = 15.0 (Qwen2.5-VL OCR-Bench); smallest |z|
anywhere = 5.7 (Qwen2.5-VL GQA). On OCR-Bench @25% the independent-binomial cross-check (18.2 / 14.7)
marginally exceeds McNemar (16.7 / 14.6) because the binary correctness indicator coarsens the /1000
item score (body footnote e); McNemar remains the primary test and is ≥-consistent on every other cell.

### S4b. Cascade two-stage gate — paired McNemar counts (Qwen3-VL-8B, n = 200, official rescore)

Cascade gate (body Table 4 / §6): single budget, Stage 1 = pre-merger RBM keeping fraction X of
merger-input units, Stage 2 = FastV at LLM layer 2 dropping fraction Y of the survivors; total keep =
X·(1−Y). Three arms (pre-alone / fastv-alone / cascade) are compared pairwise on identical images
(n_common matched; mean post-merger token count identical across all three arms in every cell, so the
comparison is iso-budget). cas_only / pre_only / fst_only = per-sample counts correct in only the
named arm of the pair. Verdict = NO-GO (cascade fails the Pareto rule on all 8 budget×benchmark
cells). Source: `runs/cascade/gate_result.json` (`mcnemar_cas_vs_pre` / `mcnemar_cas_vs_fst`);
`experiments/cascade_gate.md`.

| total keep κ | bench | n | pre | fst | cas | z (cas vs pre) | cas_only | pre_only | z (cas vs fst) | cas_only | fst_only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 25% | TextVQA | 200 | 0.5967 | 0.6800 | 0.5950 | −0.152 | 21 | 22 | −2.335 | 18 | 35 |
| 25% | DocVQA | 200 | 0.4239 | 0.5183 | 0.4779 | +2.066 | 38 | 22 | −1.500 | 26 | 38 |
| 25% | OCR-Bench | 181 | 0.5801 | 0.1713 | 0.3978 | −5.154 | 4 | 37 | +6.112 | 43 | 2 |
| 25% | GQA | 200 | 0.4150 | 0.4900 | 0.4050 | −0.333 | 17 | 19 | −2.429 | 16 | 33 |
| 12.5% | TextVQA | 200 | 0.5000 | 0.4450 | 0.4050 | −2.540 | 21 | 41 | −0.896 | 27 | 34 |
| 12.5% | DocVQA | 200 | 0.2980 | 0.2779 | 0.2070 | −2.012 | 31 | 49 | −2.064 | 16 | 30 |
| 12.5% | OCR-Bench | 181 | 0.3591 | 0.0884 | 0.1713 | −5.013 | 6 | 40 | +3.441 | 17 | 2 |
| 12.5% | GQA | 200 | 0.3950 | 0.4550 | 0.3500 | −1.286 | 20 | 29 | −3.130 | 12 | 33 |

Reading: cascade sits strictly between the two single methods and is dominated by the better arm on
every cell — pre-alone on OCR-Bench (z = −5.15 / −5.01 vs pre at 25% / 12.5%; the FastV stage destroys
exactly the raw-patch information Stage 1 was built to protect) and FastV-alone on TextVQA / GQA
(z = −2.34 / −2.43 vs fst at 25%). The single positive cas-vs-pre z (DocVQA 25%, +2.07) is paired
with a loss to FastV (−1.50), so the "strictly beats both arms" condition holds nowhere. Two
training-free selectors composed in series do not complement (body §6).

## S5. Binomial stderrs + skip accounting (Table 1 cells)

σ = √(p(1 − p)/n), greedy decoding (no temperature variance; error bars are binomial only).
Source: `experiments/j7_main_table.md` (stderr column) + Table 1 footnotes.

- Qwen3-VL none DocVQA: re-run at max-model-len 49,152, native resolution, **0/5349 skips** (ANLS 0.956).
- OCR-Bench none cells skip **18/1000 (Qwen3-VL) / 24/1000 (Qwen2.5-VL)** long-OCR images on context
  overrun, scored 0 (conservative); compressed cells skip ≤ 5. The uncompressed anchor is therefore
  understated and the relative gaps are conservative.
- Qwen2.5-VL OCR-Bench @25% RBM: 4 M-pixel cap (mean tokens 229.6 vs post 282.0) → its +293-pt gap is
  conservative (the one iso-token exception in Table 1, note d).
- Per-cell σ values (full splits, computed from the printed official metric p and full-split n;
  greedy decoding ⇒ binomial only):

| Model | Benchmark | n | none | pre@25% | post@25% | pre@12.5% | post@12.5% |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | TextVQA | 5000 | 0.0051 | 0.0069 | 0.0059 | 0.0071 | 0.0048 |
| Qwen3-VL-8B | DocVQA | 5349 | — ᵃ | 0.0068 | 0.0058 | 0.0065 | 0.0042 |
| Qwen3-VL-8B | OCR-Bench | 1000 | 0.0135 | 0.0157 | 0.0123 | 0.0151 | 0.0071 |
| Qwen3-VL-8B | GQA | 12578 | 0.0043 | 0.0044 | 0.0045 | — ᶜ | — ᶜ |
| Qwen2.5-VL-7B | TextVQA | 5000 | 0.0049 | 0.0065 | 0.0070 | 0.0069 | 0.0066 |
| Qwen2.5-VL-7B | DocVQA | 5349 | 0.0030 | 0.0066 | 0.0068 | 0.0068 | 0.0059 |
| Qwen2.5-VL-7B | OCR-Bench | 1000 | 0.0122 | 0.0158 | 0.0122 | 0.0149 | 0.0079 |
| Qwen2.5-VL-7B | GQA | 12578 | 0.0044 | 0.0044 | 0.0044 | — ᶜ | — ᶜ |

  σ = √(p(1−p)/n); each cell's every text-dense Δ₂₅ is ≥ 15σ of its own cells' pooled stderr, and the
  full-split GQA post lead (+2.8 / +2.6 pp) is ≈ 5σ of the pooled GQA stderr (≈0.0063) — significant
  but an order of magnitude smaller than any text-dense Δ. ᵃ Qwen3-VL none-DocVQA native-resolution
  anchor: full-split run partly skipped on context overrun (re-run at max-model-len 49,152 gives
  ANLS 0.956, 0/5349 skips); σ not printed for the partial native cell. ᶜ GQA evaluated at κ = 0.25
  only on the full split (no full-split κ = 0.125 cells; body footnote c).
- Qwen3-VL OCR-Bench none = 760/1000 over 982 attempted (18 skips scored 0); compressed @25%: 547 (pre)
  / 184 (post), skip ≤ 5 — ratio 3.0×, widening to 6.6× at 12.5% (350 vs 53).

## S6. n = 200 cross-split consistency (dev-subset vs full-split direction)

The n = 200 dev scope used by Tables 3–4 and §5.3 is direction-consistent with the full splits.
Source: `experiments/j2_crossgen_matrix.md` (n = 200 official) + `drafts/paper_v4.md` Table A2
(j0a/j2/j4 campaigns). Δ₂₅ = pre − post at κ = 0.25.

| Model | Benchmark | full-split Δ₂₅ (Table 1) | n = 200 Δ₂₅ | same sign? |
|---|---|---|---|---|
| Qwen3-VL-8B | TextVQA | +38.4 pp | +38.3 pp | ✓ (pre) |
| Qwen3-VL-8B | DocVQA | +24.3 pp | +26.5 pp | ✓ (pre) |
| Qwen2.5-VL-7B | TextVQA | +26.1 pp | +32.0 pp | ✓ (pre) |
| Qwen2.5-VL-7B | DocVQA | +11.0 pp | +18.8 pp | ✓ (pre) |

Reading: all four text-dense cells agree in sign (pre-merger lead) across the two scopes; magnitudes
are within run noise (the n = 200 subset SE is ≈8× larger, so the subset is retained only as a
direction check, never as the headline). The full Qwen3-VL n = 200 cross-method view (RBM vs FastV vs
Pyramid) and the compact Qwen2.5-VL n = 200 cross-generation matrix are printed in paper_v4 Table A2
and §5.4 Table 2 respectively.

## S7. Selector invariance (second scorer family, Qwen3-VL)

The stage law under a second, non-L2 scorer (global-centroid attention). Source: method_gate_report
via `drafts/paper_v4.md` §5 (j3/j5 campaigns).

| Selector | Benchmark | pre | post | Δ |
|---|---|---|---|---|
| L2 (paper selector) | TextVQA | 0.598–0.605 | 0.215–0.222 | ≈ +38 pp |
| Global-centroid attention | TextVQA | **0.553** | 0.200 | **+35.3 pp** |
| Global-centroid attention | TextVQA/DocVQA (doubled n) | — | — | **+35.5 / +24.4 pp** |

Reading: invariant in sign and magnitude on Qwen3-VL — the effect is not an L2 artifact. On
Qwen2.5-VL the L2 sign is invariant but the attention proxy degrades (proxy-family specificity, not
a counter-example); L2 remains the paper's selector, frozen as plain RBM (§3.2).

## S8. Merger tap points and M-RoPE configuration details (supports §3)

- **Qwen3-VL-8B**: first-invoked merger = `deepstack_merger_list[0]` (deepstack_visual_indexes =
  [8, 16, 24]), so unit scores are computed on **ViT block-8 outputs** (intermediate depth); the
  single resulting mask is shared by all mergers, including the main merger that consumes the
  final-block output. Runner diagnostic `mask_computed_at = 'deepstack_0'` on every run. Source:
  `experiments/r1_1_swap_jaccard.md` Part B (three-evidence verification: call order in
  `Qwen3VLVisionModel.forward`, model config, run diagnostics).
- **Qwen2.5-VL-7B**: no deepstack; scores are computed on the **main merger input (final ViT block
  output)**. `mask_computed_at = 'main'`.
- **InternVL3-8B**: the pixel-shuffle downsampling stage is the merger; L2 taps its input.
- **M-RoPE fix**: Qwen2.5-VL uses a block M-RoPE layout and Qwen3-VL an interleaved one; under
  pruning the position cursor must advance by the actual surviving count. A family-scoped fix
  (bit-degrading to the original at full retention) makes the Qwen2.5-VL compression path
  well-formed; the block-vs-interleaved contrast is itself a diagnostic of why naive pruning
  collapses one generation and is tolerated by the other.
- **Table 3 / r2c harness**: HF transformers eager attention, `--mrope native` (family-agnostic
  correct positions) on all same-scope cells; the Qwen2.5-VL × pre vllm-mimic positional degeneration
  (repeated "addCriterion…" outputs → pseudo-zero scores) was detected and replaced before any
  reported number; on Qwen3-VL native vs mimic give 0.575 vs 0.525 (same ordering). Source:
  `experiments/r2c_rbm_scope.md`, DECISIONS 2026-07-30.

## S9. Per-cell reproducibility index (run JSONs; gitignored, released with code)

| Body location | Cells | Runs path | Summary digest |
|---|---|---|---|
| Table 1 (Qwen full splits) | 26/26 | `runs/full_matrix/j7_*.json` | `experiments/j7_main_table.md` |
| Table 2 (InternVL3) | 16/16 | `runs/internvl3/internvl3_*.json` + `internvl3_official_summary.json` | `experiments/internvl3_main_matrix.md` |
| Table 3 FastV-k3 | 8 | `runs/r2_same_scope/r2b_*.json` | `experiments/r2b_fastv_k3.md` |
| Table 3 RBM same-scope | 5 | `runs/r2_same_scope/r2c_*.json` | `experiments/r2c_rbm_scope.md` |
| Table 4 cascade gate | 24/24 | `runs/cascade/gate_*.json` + `gate_result.json` | `experiments/cascade_gate.md` |
| Table 5 efficiency | 7 configs | `runs/full_matrix/efficiency/*.eff.json` | `experiments/j6_efficiency.md` |
| §5 kept-set Jaccard / swap | 8 | `runs/r1_1_swap_jaccard/*.json` + `jaccard_summary.json` | `experiments/r1_1_swap_jaccard.md` |
| §5.1–5.2 M1/M2 | n = 64 × 3 | mechanism campaign (released with code) | mechanism_verification_report via `drafts/paper_v4.md` §4 |
| §5.4 retention curve | n = 200 × 4 | j8 ablation runs | `experiments/j8_ablations.md` |
| §6 (a)(b)(c) router/hybrid/QA-gate | — | gate/j5 runs | `experiments/j5_qa_gate_result.md` + paper_v4 §6 |
| §8 600k reversal | 3 | j8b runs | `experiments/j8b_q2vl_docvqa_600k.md` |
| Engine fairness | r=0 8/8; none 16/16 | j4 runs | `experiments/j4_step2_fix.md` |
| PyramidDrop iso-25% collapse | n = 500 | j7hf runs | `experiments/j7hf_baselines_n500.md` |

### S9.1 InternVL3-8B full-split cells (Table 2; 16/16)

Model `OpenGVLab/InternVL3-8B`, vLLM 0.19, bf16, eager attention, greedy; r = TOTAL drop (0.750 →
κ = 25%, 0.875 → κ = 12.5%); skip = context-overrun cells scored 0. Source:
`runs/internvl3/internvl3_official_summary.json` (+ per-cell `*_full.json`, each carries
`n_skipped`; all 0 here). Digest: `experiments/internvl3_main_matrix.md`.

| cell | mode | r | bench | n | official | metric | ptid | skip | JSON |
|---|---|---|---|---|---|---|---|---|---|
| none_docvqa | none | 0.000 | docvqa | 5349 | 0.9221 | ANLS | 3230.8 | 0 | internvl3_none_docvqa_r0.000_full.json |
| none_gqa | none | 0.000 | gqa | 12578 | 0.6293 | acc-official | 2557.2 | 0 | internvl3_none_gqa_r0.000_full.json |
| none_ocrbench | none | 0.000 | ocrbench | 1000 | 0.852 | Final/1000=852 | 2040.1 | 0 | internvl3_none_ocrbench_r0.000_full.json |
| none_textvqa | none | 0.000 | textvqa | 5000 | 0.8338 | VQA-acc | 2562.9 | 0 | internvl3_none_textvqa_r0.000_full.json |
| pre_docvqa_r0.750 | pre | 0.750 | docvqa | 5349 | 0.7284 | ANLS | 859.5 | 0 | internvl3_pre_docvqa_r0.750_full.json |
| pre_docvqa_r0.875 | pre | 0.875 | docvqa | 5349 | 0.5054 | ANLS | 464.3 | 0 | internvl3_pre_docvqa_r0.875_full.json |
| pre_gqa_r0.750 | pre | 0.750 | gqa | 12578 | 0.5993 | acc-official | 690.1 | 0 | internvl3_pre_gqa_r0.750_full.json |
| pre_ocrbench_r0.750 | pre | 0.750 | ocrbench | 1000 | 0.753 | Final/1000=753 | 555.4 | 0 | internvl3_pre_ocrbench_r0.750_full.json |
| pre_textvqa_r0.750 | pre | 0.750 | textvqa | 5000 | 0.789 | VQA-acc | 690.5 | 0 | internvl3_pre_textvqa_r0.750_full.json |
| pre_textvqa_r0.875 | pre | 0.875 | textvqa | 5000 | 0.723 | VQA-acc | 378.4 | 0 | internvl3_pre_textvqa_r0.875_full.json |
| post_docvqa_r0.750 | post | 0.750 | docvqa | 5349 | 0.382 | ANLS | 859.5 | 0 | internvl3_post_docvqa_r0.750_full.json |
| post_docvqa_r0.875 | post | 0.875 | docvqa | 5349 | 0.2451 | ANLS | 464.3 | 0 | internvl3_post_docvqa_r0.875_full.json |
| post_gqa_r0.750 | post | 0.750 | gqa | 12578 | 0.6031 | acc-official | 690.1 | 0 | internvl3_post_gqa_r0.750_full.json |
| post_ocrbench_r0.750 | post | 0.750 | ocrbench | 1000 | 0.321 | Final/1000=321 | 555.4 | 0 | internvl3_post_ocrbench_r0.750_full.json |
| post_textvqa_r0.750 | post | 0.750 | textvqa | 5000 | 0.4148 | VQA-acc | 690.5 | 0 | internvl3_post_textvqa_r0.750_full.json |
| post_textvqa_r0.875 | post | 0.875 | textvqa | 5000 | 0.3064 | VQA-acc | 378.4 | 0 | internvl3_post_textvqa_r0.875_full.json |

(All paths relative to `runs/internvl3/`. Pre-merger leads on every text-dense benchmark at both
budgets — e.g. DocVQA ANLS 0.7284 vs 0.382 at κ = 25%; GQA is the near-null object-centric case,
0.5993 vs 0.6031 — replicating the stage law on the pixel-shuffle merger design.)

### S9.2 Table 3 FastV-k3 same-scope cells (8/8)

FastV one-shot attention pruning at LLM layer K = 3, iso-budget r = 0.75 (κ = 25%); HF-transformers
4.57.6 eager harness, official metrics. Source: per-cell `runs/r2_same_scope/r2b_*.json` (fields
`n` / `acc` / `n_skipped` / `mean_ptid_len` / `engine`). Digest: `experiments/r2b_fastv_k3.md`.

| cell | model | bench | n | official | ptid | skip | engine | JSON |
|---|---|---|---|---|---|---|---|---|
| r2b_qwen3vl_fastv_k3_textvqa_r0.75_full5000 | Qwen3-VL | textvqa | 5000 | 0.8494 | 213.7 | 5 | hf eager | r2b_qwen3vl_fastv_k3_textvqa_r0.75_full5000.json |
| r2b_qwen3vl_fastv_k3_gqa_r0.75_full12578 | Qwen3-VL | gqa | 12578 | 0.5561 | 96.8 | 0 | hf eager | r2b_qwen3vl_fastv_k3_gqa_r0.75_full12578.json |
| r2b_qwen3vl_fastv_k3_docvqa_r0.75_n500 | Qwen3-VL | docvqa | 200 | 0.465 | 176.0 | 0 | hf eager | r2b_qwen3vl_fastv_k3_docvqa_r0.75_n500.json |
| r2b_qwen3vl_fastv_k3_ocrbench_r0.75_n500 | Qwen3-VL | ocrbench | 200 | 0.4586 | 114.8 | 19 | hf eager | r2b_qwen3vl_fastv_k3_ocrbench_r0.75_n500.json |
| r2b_qwen2vl_fastv_k3_textvqa_r0.75_n200 | Qwen2.5-VL | textvqa | 200 | 0.815 | 282.1 | 0 | hf eager | r2b_qwen2vl_fastv_k3_textvqa_r0.75_n200.json |
| r2b_qwen2vl_fastv_k3_gqa_r0.75_n200 | Qwen2.5-VL | gqa | 200 | 0.51 | 126.8 | 0 | hf eager | r2b_qwen2vl_fastv_k3_gqa_r0.75_n200.json |
| r2b_qwen2vl_fastv_k3_docvqa_r0.75_n200 | Qwen2.5-VL | docvqa | 200 | 0.39 | 231.4 | 0 | hf eager | r2b_qwen2vl_fastv_k3_docvqa_r0.75_n200.json |
| r2b_qwen2vl_fastv_k3_ocrbench_r0.75_n200 | Qwen2.5-VL | ocrbench | 200 | 0.3167 | 143.6 | 20 | hf eager | r2b_qwen2vl_fastv_k3_ocrbench_r0.75_n200.json |

(Filenames marked `_n500` for the Qwen3-VL DocVQA / OCR-Bench cells ran at the n = 200 same-scope
budget recorded in the JSON `n` field; OCR-Bench skips are long-OCR context overruns, scored 0.)

### S9.3 Table 3 RBM same-scope cells (5/5)

RBM (pre-merger L2) at iso-budget r = 0.75, HF-transformers eager, `--mrope native` (family-agnostic
correct positions), matched to the FastV-k3 cells above. Source: per-cell
`runs/r2_same_scope/r2c_*.json`. Digest: `experiments/r2c_rbm_scope.md` (+ DECISIONS 2026-07-30).

| cell | model | bench | n | official | ptid | skip | engine | JSON |
|---|---|---|---|---|---|---|---|---|
| r2c_qwen2vl_pre_r0.75_textvqa_n200 | Qwen2.5-VL | textvqa | 200 | 0.725 | 282.1 | 0 | hf eager | r2c_qwen2vl_pre_r0.75_textvqa_n200.json |
| r2c_qwen2vl_pre_r0.75_gqa_n200 | Qwen2.5-VL | gqa | 200 | 0.545 | 126.8 | 0 | hf eager | r2c_qwen2vl_pre_r0.75_gqa_n200.json |
| r2c_qwen2vl_pre_r0.75_docvqa_n200 | Qwen2.5-VL | docvqa | 200 | 0.405 | 231.4 | 0 | hf eager | r2c_qwen2vl_pre_r0.75_docvqa_n200.json |
| r2c_qwen2vl_pre_r0.75_ocrbench_n200 | Qwen2.5-VL | ocrbench | 200 | 0.4111 | 143.6 | 20 | hf eager | r2c_qwen2vl_pre_r0.75_ocrbench_n200.json |
| r2c_qwen3vl_pre_r0.75_ocrbench_n200 | Qwen3-VL | ocrbench | 200 | 0.6354 | 114.8 | 19 | hf eager | r2c_qwen3vl_pre_r0.75_ocrbench_n200.json |

### S9.4 Cascade gate cells (Table 4; 24/24, Qwen3-VL-8B)

Three arms × two budgets × four benchmarks; n = n_common (matched by question id); official =
`acc_official` from the paired rescore; ptid = mean post-merger token count (identical across the
three arms in every cell). Source: `runs/cascade/gate_result.json` (authoritative paired summary)
with per-arm run cells `runs/cascade/gate_{pre,fst,cas}{25,12}_<bench>.json`. Digest:
`experiments/cascade_gate.md`. (McNemar counts for these cells: S4b.)

| arm | κ | bench | n | official | ptid | JSON |
|---|---|---|---|---|---|---|
| pre-alone (RBM) | 25% | textvqa | 200 | 0.5967 | 212.8 | gate_pre25_textvqa.json |
| fastv-alone | 25% | textvqa | 200 | 0.6800 | 212.8 | gate_fst25_textvqa.json |
| cascade | 25% | textvqa | 200 | 0.5950 | 212.8 | gate_cas25_textvqa.json |
| pre-alone (RBM) | 25% | docvqa | 200 | 0.4239 | 176.0 | gate_pre25_docvqa.json |
| fastv-alone | 25% | docvqa | 200 | 0.5183 | 176.0 | gate_fst25_docvqa.json |
| cascade | 25% | docvqa | 200 | 0.4779 | 176.0 | gate_cas25_docvqa.json |
| pre-alone (RBM) | 25% | ocrbench | 181 | 0.5801 | 114.8 | gate_pre25_ocrbench.json |
| fastv-alone | 25% | ocrbench | 181 | 0.1713 | 114.8 | gate_fst25_ocrbench.json |
| cascade | 25% | ocrbench | 181 | 0.3978 | 114.8 | gate_cas25_ocrbench.json |
| pre-alone (RBM) | 25% | gqa | 200 | 0.4150 | 95.4 | gate_pre25_gqa.json |
| fastv-alone | 25% | gqa | 200 | 0.4900 | 95.4 | gate_fst25_gqa.json |
| cascade | 25% | gqa | 200 | 0.4050 | 95.4 | gate_cas25_gqa.json |
| pre-alone (RBM) | 12.5% | textvqa | 200 | 0.5000 | 120.7 | gate_pre12_textvqa.json |
| fastv-alone | 12.5% | textvqa | 200 | 0.4450 | 120.7 | gate_fst12_textvqa.json |
| cascade | 12.5% | textvqa | 200 | 0.4050 | 120.7 | gate_cas12_textvqa.json |
| pre-alone (RBM) | 12.5% | docvqa | 200 | 0.2980 | 103.7 | gate_pre12_docvqa.json |
| fastv-alone | 12.5% | docvqa | 200 | 0.2779 | 103.7 | gate_fst12_docvqa.json |
| cascade | 12.5% | docvqa | 200 | 0.2070 | 103.7 | gate_cas12_docvqa.json |
| pre-alone (RBM) | 12.5% | ocrbench | 181 | 0.3591 | 68.7 | gate_pre12_ocrbench.json |
| fastv-alone | 12.5% | ocrbench | 181 | 0.0884 | 68.7 | gate_fst12_ocrbench.json |
| cascade | 12.5% | ocrbench | 181 | 0.1713 | 68.7 | gate_cas12_ocrbench.json |
| pre-alone (RBM) | 12.5% | gqa | 200 | 0.3950 | 62.3 | gate_pre12_gqa.json |
| fastv-alone | 12.5% | gqa | 200 | 0.4550 | 62.3 | gate_fst12_gqa.json |
| cascade | 12.5% | gqa | 200 | 0.3500 | 62.3 | gate_cas12_gqa.json |

(All paths relative to `runs/cascade/`. OCR-Bench n_common = 181 after 19 long-OCR skips; all other
cells n = 200. `missing_cells: []` — 24/24 present.)

---

*Packaging note: this file converts to the ACM MM supplementary PDF/archive; run JSONs ship in the
code release (anonymous archive at submission, repository at de-anonymization). Nothing in this
supplement identifies authors, institutions, or repositories (double-blind).*

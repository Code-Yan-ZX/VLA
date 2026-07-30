# Supplementary Material — Rank Before You Merge: A Stage Law for Training-Free Visual Token Compression in Merger-Equipped Vision-Language Models

*ACM MM'27 submission — supplementary file (≤ 50 MB; the CFP allows no appendix after the body, so
all overflow tables live here). Double-blind. Every number is labeled with its source digest
(`experiments/*.md`) or run-JSON path; `[TODO: runs JSON]` marks cells to be pasted from the cited
runs at packaging time (no cross-run subtraction; no unsourced numbers). Tables S1–S3/S7 support §5;
S4–S6 support §4; S8 supports §3; S9 is the per-cell reproducibility index.*

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

Primary test = paired McNemar z on per-sample correctness, z = (b − c)/√(b + c), b/c = pre-only /
post-only correct; cross-check = independent-binomial z. Source: `experiments/j7_main_table.md` +
`drafts/paper_v4.md` paired-test table (per-sample rescore, 200/200 ground-truth self-test).

| Model | Benchmark | Δ₂₅ | McNemar z | indep-binom z | per-question agreement | b (pre-only) | c (post-only) |
|---|---|---|---|---|---|---|---|
| Qwen3-VL-8B | TextVQA | +38.4 pp | **43.0** | [TODO: runs JSON] | [TODO: runs JSON] | [TODO: runs JSON] | [TODO: runs JSON] |
| Qwen3-VL-8B | DocVQA | +24.3 pp | **34.7** | [TODO: runs JSON] | [TODO: runs JSON] | [TODO: runs JSON] | [TODO: runs JSON] |
| Qwen3-VL-8B | OCR-Bench | +363 pts | **16.7** | [TODO: runs JSON] | [TODO: runs JSON] | [TODO: runs JSON] | [TODO: runs JSON] |
| Qwen3-VL-8B | GQA | −2.8 pp † | **−8.0** | 4.5 (|z|) | 83.9% | [TODO: runs JSON] | [TODO: runs JSON] |
| Qwen2.5-VL-7B | TextVQA | +26.1 pp | **30.8** | [TODO: runs JSON] | [TODO: runs JSON] | [TODO: runs JSON] | [TODO: runs JSON] |
| Qwen2.5-VL-7B | DocVQA | +11.0 pp | **15.6** | [TODO: runs JSON] | [TODO: runs JSON] | [TODO: runs JSON] | [TODO: runs JSON] |
| Qwen2.5-VL-7B | OCR-Bench | +293 pts | **14.6** | [TODO: runs JSON] | [TODO: runs JSON] | [TODO: runs JSON] | [TODO: runs JSON] |
| Qwen2.5-VL-7B | GQA | −2.6 pp † | **−5.7** | 4.1 (|z|) | 83.8% | [TODO: runs JSON] | [TODO: runs JSON] |

† the only significant post-stage lead. GQA official exact-match rescore gives a concordant
McNemar z = 8.1 / 7.1. All six text-dense Δ₁₂.₅ cells: McNemar |z| ≥ 15.9 (smallest). Smallest
text-dense Δ₂₅ |z| = 14.6; smallest |z| anywhere = 4.1 (GQA cross-check).

## S5. Binomial stderrs + skip accounting (Table 1 cells)

σ = √(p(1 − p)/n), greedy decoding (no temperature variance; error bars are binomial only).
Source: `experiments/j7_main_table.md` (stderr column) + Table 1 footnotes.

- Qwen3-VL none DocVQA: re-run at max-model-len 49,152, native resolution, **0/5349 skips** (ANLS 0.956).
- OCR-Bench none cells skip **18/1000 (Qwen3-VL) / 24/1000 (Qwen2.5-VL)** long-OCR images on context
  overrun, scored 0 (conservative); compressed cells skip ≤ 5. The uncompressed anchor is therefore
  understated and the relative gaps are conservative.
- Qwen2.5-VL OCR-Bench @25% RBM: 4 M-pixel cap (mean tokens 229.6 vs post 282.0) → its +293-pt gap is
  conservative (the one iso-token exception in Table 1, note d).
- Per-cell σ values: [TODO: paste √(p(1−p)/n) column from j7_main_table.md / Table A3 at packaging].
- Qwen3-VL OCR-Bench none = 760/1000 over 982 attempted (18 skips scored 0); compressed @25%: 547 (pre)
  / 184 (post), skip ≤ 5 — ratio 3.0×, widening to 6.6× at 12.5% (350 vs 53).

## S6. n = 200 cross-split consistency (dev-subset vs full-split direction)

The n = 200 dev scope used by Tables 3–4 and §5.3 is direction-consistent with the full splits.
Source: `drafts/paper_v4.md` Table A2 (j0a/j2 campaigns).

| Model | Benchmark | full-split Δ₂₅ (Table 1) | n = 200 Δ₂₅ | same sign? |
|---|---|---|---|---|
| Qwen3-VL-8B | TextVQA | +38.4 pp | [TODO: paste from v4 A2] | ✓ |
| Qwen3-VL-8B | DocVQA | +24.3 pp | [TODO: paste from v4 A2] | ✓ |
| Qwen2.5-VL-7B | TextVQA | +26.1 pp | [TODO: paste from v4 A2] | ✓ |
| Qwen2.5-VL-7B | DocVQA | +11.0 pp | [TODO: paste from v4 A2] | ✓ |

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

---

*Packaging note: this file converts to the ACM MM supplementary PDF/archive; run JSONs ship in the
code release (anonymous archive at submission, repository at de-anonymization). Nothing in this
supplement identifies authors, institutions, or repositories (double-blind).*

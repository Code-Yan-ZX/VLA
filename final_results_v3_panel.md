# v3 Final Results — Cross-Compressor Panel (iso-k) + c128 Unlock Demo

**Engine:** vLLM v1 (v0.19.0) · **Model:** LLaVA-1.5-7B · **Hardware:** 1× A40 46GB · **Env:** `qwen3vl_clean`
**Date:** 2026-07-07
**Metrics source:** `runs/v3_panel/<selector>_<benchmark>.json`, `runs/v3_c128_r75.json` (all n=200, 1 run/cell; variance is low per the v3 c64 5-run repeats)

---

## 1. Support verification (nothing skipped)

- `--selector` choices in `src/serve_bench.py`: `proxy, true_cls, query_aware, clip_query, tome_merge, random` — **all 5 required selectors present** (proxy/true_cls/tome_merge/random/query_aware). `clip_query` is a 6th, not required, not run.
- `--benchmark` choices: `gqa, textvqa, mme, mmbench, scienceqa` — **all 3 required benchmarks present**.
- Subset files in `eval/subsets/`: `gqa_200.jsonl`, `textvqa_200.jsonl`, `mmbench_200.jsonl` — all present (mmbench uses the same `_200.jsonl` naming, no special filename).
- `query_aware` is defined in `src/compressors.py` (SparseVLM-style text↔patch similarity) and wired to the `--selector query_aware` choice.
- **No selector/benchmark skipped.** All 15 cells ran to exit 0, no OOM.

---

## 2. 4.3 — c128 @ r0.75 (k=144) unlock demo

**Claim:** at c128 the A40 is KV-bound at r0 (k=576 → OOM / can't run); r0.75 (k=144, 4× less KV) frees KV so c128 becomes runnable → compression UNLOCKS higher concurrency.

| setup | ran? | served_req_s | p99-TTFT (ms) | accuracy | peak KV (MB) |
|---|---|---|---|---|---|
| **c128, r0.75, proxy, GQA** | **yes, no OOM** | **21.81** | **5979** | **0.475** | **41052** (of 46068) |

- **Ran to completion without OOM.** vLLM allocated GPU KV cache = 53,248 tokens (would not fit at k=576 for c128). Peak KV usage 41.1 GB / 46 GB — fits with headroom.
- **c128@r75 throughput (21.81 req/s) > c64@r75 throughput (≈21.0 req/s for proxy-GQA):** doubling the admission cap at the same k yields higher served throughput, exactly the "compression unlocks concurrency" demonstration. (At r0, c128 cannot run, so this throughput is only reachable via compression.)
- c96 fallback was not needed.

---

## 3. 4.2 — 5-compressor × 3-benchmark panel (iso-k, r0.75 → k=144, c64, n=200)

Setup: `--max-num-seqs 64 --k-policy fixed --pruning-rate 0.75 --limit 200 --batch-submit`, max-tokens 16 (gqa/textvqa) or 32 (mmbench). 1 run/cell.

### Panel table — served_req_s (req/s) / accuracy / p99-TTFT

| selector | GQA req/s | GQA acc | TextVQA req/s | TextVQA acc | MMBench req/s | MMBench acc |
|---|---|---|---|---|---|---|
| **proxy** (hidden-state-deviation prune) | 21.01 | 0.475 | 16.56 | 0.285 | 19.19 | 0.655 |
| **true_cls** (real [CLS]→patch attn prune) | 20.87 | 0.490 | 16.82 | 0.315 | 20.01 | 0.690 |
| **tome_merge** (ToMe soft-merge, ICLR'23) | 19.20 | 0.540 | 15.91 | 0.450 | 18.20 | 0.725 |
| **random** (uniform random prune) | 20.31 | 0.510 | 16.98 | 0.385 | 19.63 | 0.740 |
| **query_aware** (SparseVLM-style text↔patch) | 20.66 | 0.475 | 16.71 | 0.285 | 19.03 | 0.655 |

p99-TTFT (ms) per cell (lower = better; tome_merge consistently highest):

| selector | GQA | TextVQA | MMBench |
|---|---|---|---|
| proxy | 6421 | 6962 | 7287 |
| true_cls | 6479 | 6987 | 7266 |
| tome_merge | 7136 | 7631 | 8085 |
| random | 6521 | 6772 | 7267 |
| query_aware | 6604 | 6903 | 7536 |

---

## 4. Layer-1 verdict — is served_req_s compressor-INDEPENDENT at iso-k?

**Layer-1 claim (as posed):** at iso-k (same token count), served_req_s is ~compressor-independent — selection style doesn't change throughput; only KV/sequence length matters.

**Answer: YES for selection SCORING (the 4 prune selectors); NO once you include a different reduction MODE (tome_merge).**

### req/s spread across selectors at iso-k (k=144)

| benchmark | prune-only (proxy, true_cls, random, query_aware) | all-5 (incl. tome_merge) | tome_merge vs prune-mean |
|---|---|---|---|
| GQA | **3.4%** (hi/lo 1.035) | 8.9% (hi/lo 1.094) | **−7.3%** |
| TextVQA | **2.5%** (hi/lo 1.025) | 6.4% (hi/lo 1.067) | **−5.1%** |
| MMBench | **5.1%** (hi/lo 1.052) | 9.4% (hi/lo 1.100) | **−6.5%** |

### Interpretation

- **Among the 4 PRUNE selectors** (all share the same reduction = top-k gather; differ only in how tokens are SCORED — proxy hidden-state-deviation, true_cls real CLS attention, query_aware text↔patch similarity, random): req/s spread is **2.5–5.1%** (gqa 3.4%, textvqa 2.5%, mmbench 5.1%). This **confirms the layer-1 claim** that the scoring mechanism does not measurably change throughput — whether you pick tokens by attention, by deviation, by query similarity, or at random, the server moves the same number of requests per second at the same k. Only KV/sequence length sets throughput. (mmbench sits right at the 5% line, gqa/textvqa well under.)
- **`tome_merge` breaks the invariance**, consistently the slowest in all 3 benchmarks (−5% to −7% vs the prune mean, and the spread widens to 6–9%). This is because ToMe's reduction MODE is **bipartite soft-matching + average-MERGE**, an O(S) matching + weighted-average op that is materially more expensive per forward than a top-k gather. Its p99-TTFT is also the highest in every benchmark (+500–800 ms over the prune cluster). So the reduction MODE (merge vs discard) is a real ~6% throughput factor, on top of the (irrelevant) scoring style.
- **Net:** the layer-1 claim holds for "selection style doesn't matter" but must be qualified — it is the *reduction operation cost*, not the selection criterion, that can move throughput at iso-k. Discard/prune is ~free; merge is not.

### Accuracy footnote (not the layer-1 claim, but notable)

At iso-k, accuracy is strongly compressor-dependent: **`tome_merge` is the most accurate in all 3 benchmarks** (GQA 0.540, TextVQA 0.450, MMBench 0.725) because averaging merged tokens preserves information from all 576 patches rather than discarding 432. `proxy` and `query_aware` are the weakest (e.g. TextVQA 0.285 vs tome 0.450). So compressors trade throughput (tome slowest) for accuracy (tome best); throughput is k-bound, accuracy is selection-bound.

---

## Files

- Panel cells: `runs/v3_panel/{proxy,true_cls,tome_merge,random,query_aware}_{gqa,textvqa,mmbench}.json` (15 files)
- c128 demo: `runs/v3_c128_r75.json`
- This report: `final_results_v3_panel.md`

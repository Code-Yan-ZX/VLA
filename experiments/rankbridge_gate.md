# RankBridge gate — digest (pre-registered, user goal 2026-08-07)

## Method
Cross-stage rank FUSION at FastV layer K over the FULL visual-token candidate
set (vs cascade's serial pruning, NO-GO 2026-07-29): ALL visual tokens stay
through LLM layer K=3; each native merger unit's pre-merger L2 RANK (1 = best
mean-patch L2 within image; first-called merger input, runner-parity mask
source) is cached; at layer K the FastV query-conditioned attention rank
(mean-over-heads, last-query row, image columns) is fused with the pre-rank
into the final keep set at the RBM per-image budget k_i=max(1,round(f_i·0.25))
(total keep 25%, `--r 0.75`). Fusion = protected quota: q_i=round(rho·k_i)
seats reserved for the best pre-ranks, remainder filled by attention rank
(rho=0 ≡ FastV, rho=1 ≡ pre-rank top-k). RRF fusion implemented as fallback
(score=1/(c+r_pre)+λ/(c+r_query); NOT searched). Survivors keep native mrope
coords (in-LLM prune, same as FastV). Hypothesis: pre-rank quota shields
dense/OCR units that merger-distorted query attention misses, while FastV
ranks the full (unstarved) candidate set for query relevance.

## Implementation & reproduction
- Code: src/v3_premerger/baselines_hf.py `--mode rankbridge` (commit
  5fa3933); runner untouched. New flags: `--rb-fuse {quota,rrf}`,
  `--rb-rho`, `--rb-lambda`, `--rb-rrf-c`; reuses `--r` (total drop) and
  `--fastv-k`.
- Dry-check ALL PASS incl. degenerate identities (quota rho=0 == FastV keep
  set AND hidden; rho=1 == pre top-k); real-weights smoke n=4: rho=0 answers
  and ptid IDENTICAL to FastV (4/4), protected counts = round(rho·k_i) exact.
- Example cell:
  `python src/v3_premerger/baselines_hf.py --mode rankbridge --r 0.75 \
     --fastv-k 3 --rb-fuse quota --rb-rho 0.2 --model \
     Qwen/Qwen3-VL-8B-Instruct --benchmark textvqa \
     --subset eval/subsets/textvqa_200.jsonl --n 200 --seed 0 \
     --max-pixels 0 --out runs/rankbridge/locked_rb_textvqa_n200.json`
  (docvqa: `--max-pixels 600000` — the HF-harness stable config equivalent of
  the vLLM runner's --max-num-batched-tokens 32768 / max_num_seqs=4.)
- Scripts: runs/rankbridge/{smoke,dev_gate,locked_gate}.sh +
  gate_analyze.py (official rescore; runs/ is gitignored).

## Dev gate (n=64, textvqa+ocrbench, K=3, keep 25%) — DONE
Lock rule (pre-registered): rho maximizing mean(official textvqa VQA-acc,
ocrbench 0/1 acc); exact tie -> smaller rho.

| arm | textvqa | ocrbench | mean |
|---|---|---|---|
| rho=0.1 | 0.7031 | 0.4310 | 0.5671 |
| **rho=0.2** | **0.7188** | **0.4310** | **0.5749** |
| rho=0.3 | 0.7031 | 0.4310 | 0.5671 |
| FastV-k3 | 0.6875 | 0.4310 | 0.5593 |
| RBM (pre25 slice) | 0.4531 | 0.4828 | 0.4679 |

Budget parity: rb vs RBM per-sample ptid 122/122 equal (both benches).
Masks genuinely differ (answers vs fst3: 20/29/37 of 64 differ at rho
0.1/0.2/0.3); identical ocrbench 0/1 sums and the rho0.1==rho0.3 textvqa tie
are score-level coincidences at n=58/64 (SE ~0.06).
**LOCKED: rho=0.2** (quota). Dev signal: rb >= fst3 on textvqa, ocrbench
quota holds fst3-level but does NOT yet recover RBM-level OCR (0.43 vs 0.48;
underpowered at n=58 -> locked gate decides).

## Locked gate (n=200 × 4, official metrics)
GO = every bench rb ≥ max(RBM, FastV-k3) − 1pp AND ≥1 bench strictly above
the stronger parent; binomial SE (ANLS: sample SE) + paired McNemar z.

| benchmark | none | RBM | FastV-k2 | FastV-k3 | RankBridge |
|---|---|---|---|---|---|
| textvqa | TBD | TBD | TBD | TBD | TBD |
| docvqa | TBD | TBD | TBD | TBD | TBD |
| ocrbench | TBD | TBD | TBD | TBD | TBD |
| gqa | TBD | TBD | TBD | TBD | TBD |

Budget: rb mean_ptid vs RBM arm TBD (per-sample ptid equality check).

## Verdict — TBD

## Failure modes / observations
TBD

## Next steps
TBD

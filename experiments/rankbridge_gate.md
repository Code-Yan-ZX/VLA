# RankBridge gate — digest (pre-registered, user goal 2026-08-07)

## Method
Cross-stage rank FUSION at FastV layer K=3 over the FULL visual-token
candidate set (vs cascade's serial pruning, NO-GO 2026-07-29): ALL visual
tokens stay through layer K; per-unit pre-merger L2 RANK (1=best mean-patch
L2 within image; first-called merger input = runner-parity mask source) is
cached; at layer K FastV's query-conditioned attention rank (mean-over-heads,
last-query row, image columns) fuses with the pre-rank into the final keep
set at the RBM per-image budget k_i=max(1,round(f_i·0.25)) (keep 25%, `--r
0.75`). Fusion = protected quota: q_i=round(rho·k_i) seats for best
pre-ranks, rest by attention rank (rho=0 ≡ FastV, rho=1 ≡ pre top-k). RRF
implemented as fallback, NOT searched. Native mrope (in-LLM prune == FastV).
Hypothesis: quota shields dense/OCR units; FastV ranks the unstarved full set.

## Implementation & reproduction
- Code: src/v3_premerger/baselines_hf.py `--mode rankbridge` (commit 5fa3933);
  flags `--rb-fuse {quota,rrf} --rb-rho --rb-lambda --rb-rrf-c`; runner
  untouched. Dry-check ALL PASS (rho=0 == FastV keep set AND hidden; rho=1 ==
  pre top-k); smoke n=4: rho=0 answers/ptid IDENTICAL to FastV (4/4).
- Cell: `python src/v3_premerger/baselines_hf.py --mode rankbridge --r 0.75
  --fastv-k 3 --rb-fuse quota --rb-rho 0.2 --model Qwen/Qwen3-VL-8B-Instruct
  --benchmark textvqa --subset eval/subsets/textvqa_200.jsonl --n 200 --seed 0
  --out runs/rankbridge/locked_rb_textvqa_n200.json` (docvqa adds
  `--max-pixels 600000` = HF stable config equivalent of the vLLM runner's
  mnbt 32768 / mns 4). Scripts: runs/rankbridge/{smoke,dev_gate,locked_gate}
  .sh + gate_analyze.py (official rescore, cascade-gate convention; runs/
  gitignored).

## Dev gate (n=64, K=3, keep 25%) — DONE, LOCKED rho=0.2
Lock rule (pre-registered): rho maximizing mean(textvqa, ocrbench), tie ->
smaller rho; budget parity rb==RBM ptid 122/122; masks genuinely differ.

| arm | textvqa | ocrbench | mean |
|---|---|---|---|
| rho=0.1 | 0.7135 | 0.4310 | 0.5723 |
| **rho=0.2** | **0.7344** | **0.4310** | **0.5827** |
| rho=0.3 | 0.7188 | 0.4310 | 0.5749 |
| FastV-k3 | 0.7031 | 0.4310 | 0.5671 |
| RBM (pre25) | 0.4688 | 0.4828 | 0.4758 |

## Locked gate (n=200 × 4, official metrics, common-id pairing)
| benchmark | none | RBM | FastV-k2 | FastV-k3 | RankBridge |
|---|---|---|---|---|---|
| textvqa | 0.8667±.023 | 0.5967±.034 | 0.6800±.032 | 0.7633±.029 | **0.7950±.027** |
| docvqa | 0.9487±.014 | 0.4239±.033 | 0.5183±.031 | 0.5863±.032 | **0.6023±.031** |
| ocrbench | 0.7569±.032 | **0.5801±.037** | 0.1713±.028 | 0.4586±.037 | 0.4641±.037 |
| gqa | 0.6050±.035 | 0.4150±.035 | 0.4900±.035 | 0.5050±.035 | **0.5100±.035** |

SE = sample SE (soft VQA-acc/ANLS) or binomial. Paired z rb vs stronger
parent: textvqa **+2.11** (b10=9,b01=2) / docvqa +0.58 / ocrbench **−3.66**
(b10=6,b01=27) / gqa +0.33. Budget: rb vs RBM per-sample ptid equal 200/200
every bench (ocrbench 181/181; 19 shared load skips); mean_ptid identical
across pre/fst2/fst3/rb (213/176/115/95); wall rb ≈ fst3 (HF eager,
disclosed); GPU ≈2.3 h (<6, no training).

## Verdict — NO-GO
Pre-registered: GO = every bench rb ≥ max(RBM, FastV-k3) − 1pp AND ≥1 bench
strictly above the stronger parent. Cond1 FAILS on ocrbench (0.4641 vs
0.5801 = −11.6pp ≫ −1pp, z=−3.66); cond2 holds on 3 benches. Per the lock:
no further tuning (no rho/RRF/K search); bounded negative; frozen = RBM.

## Failure modes / observations (bounded)
- rb beats FastV-k3 on ALL 4 benches (+3.2/+1.6/+0.6/+0.5pp; only textvqa
  z=2.11 ~significant, borderline under multiplicity): the quota reliably
  adds pre-rank value on top of query ranking, recovering part of the OCR
  FastV destroys (0.171 -> 0.464).
- OCRBench stays −11.6pp below RBM: protected seats keep RBM-identical
  merged tokens (M3 unit equivalence), but the non-protected 80% are chosen
  by attention over MERGER-DISTORTED features — the text-dense misranking
  the paper's mechanism describes. Full pre-selection remains the OCR fixed
  point; one global rho can't be workload-adaptive (rho->1 ≡ RBM, giving
  back textvqa/docvqa). Fourth consecutive training-free composition
  negative (router/QA-gate/cascade/RankBridge) — every composition lands
  between its parents; evidence FOR the fixed-point framing.

## Next steps
None for RankBridge (frozen); paper spine unchanged (stage law + mechanism +
RBM robust default). §6 may cite: "fusion on the full candidate set still
OCR-dominated; quota partially shields" (needs user OK).

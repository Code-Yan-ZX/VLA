# Deferred RBM lifetime gate — digest (pre-registered, task book 2026-08-10)

## Hypothesis (fixed, no parameter search)
Token *lifetime* alone closes the parent trade-off: kept identities are
BIT-IDENTICAL to plain RBM (pre-merger L2 top-k, keep 25%), but ALL visual
tokens participate in LLM layers 0..2 and are deleted only at layer K=3
(RankBridge quota `rho=1.0` realizes exactly this arm — zero new method code).
Locked: rho=1.0, K=3, keep 25% (--r 0.75), Qwen3-VL-8B-Instruct, seed 0.
No other rho or K may be tested (task book Stage A).

## Identity checks (required step 1) — PASS
- Dry-check (B9, tiny model, CPU): deferred rho=1 kept indices == plain-RBM
  kept indices bit-for-bit (survivor mrope positions `torch.equal`); survivor
  hidden states at the deletion boundary NON-identical (maxdiff=1.93e-02) —
  anchors are contextualized with tokens later deleted. Full dry-check ALL PASS.
- Real model, dev n=64: per-sample prompt_token_ids deferred vs RBM(pre25)
  equal on textvqa 64/64, docvqa 64/64, ocrbench 58/58, gqa 64/64 → per-image
  kept-index equality confirmed on real weights.

## Exact command (all 4 dev cells)
```
python src/v3_premerger/baselines_hf.py --mode rankbridge --r 0.75 \
  --rb-fuse quota --rb-rho 1.0 --fastv-k 3 \
  --model Qwen/Qwen3-VL-8B-Instruct --benchmark $B \
  --subset eval/subsets/${B}_200.jsonl --n 64 --seed 0 \
  --out runs/deferred_rbm/dev_deferred_${B}_n64.json   # docvqa: + --max-pixels 600000
```
Script `runs/deferred_rbm/dev_gate.sh`; analysis `gate_analyze.py --phase dev`
(official rescore: VQA-acc / ANLS / exact-match / OCRBench batch). Parents
reused after exact config/ID parity check (2026-08-10): RBM(pre25) =
runs/cascade/gate_pre25_*[:64], FastV-k3 = runs/rankbridge/locked_fst3_*_n200[:64]
(same model/r/seed/maxtok=32/maxpix per bench; first-64 ID prefixes equal).

## Dev gate results (n=64, official metrics, common-id)
| bench | deferred | RBM(pre) | FastV-k3 | cond1 (≥max−1pp) | strict>both |
|---|---|---|---|---|---|
| textvqa | 0.6562±.056 | 0.4688 | **0.7031** | FAIL (−4.7pp) | no |
| docvqa | 0.5168±.060 | 0.3932 | **0.5607** | FAIL (−4.4pp) | no |
| ocrbench | **0.5862±.065** | 0.4828 | 0.4310 | PASS (+10.3pp) | YES |
| gqa | **0.5781±.062** | 0.4531 | 0.5156 | PASS (+6.3pp) | YES |

Paired z vs stronger parent (McNemar, common-id): textvqa −0.53 (b10=6,b01=8),
docvqa −0.93 (12/17), ocrbench +1.90 (8/2), gqa +1.15 (8/4).

## Verdict — NO-GO (terminal for Stage A)
Pre-registered ADVANCE rule: ALL 4 benches ≥ max(RBM,FastV-k3)−1pp AND ≥1
strictly above BOTH parents. cond1 FAILS on textvqa and docvqa → **no locked
n=200 follow-up** (2.3 GPU·h not spent). Per task book: bounded negative,
proceed to Stage B (RBM-OT) without changing rho/K.

## Bounded observations (not claims)
- Lifetime extension moves deferred strictly above BOTH parents on OCR-heavy
  benches (ocrbench +10.3pp vs RBM, z=+1.90; gqa +6.3pp) — letting dropped
  tokens write context into anchors before deletion helps dense recognition.
- On text-centric benches it lands BETWEEN the parents (0.656/0.517 vs RBM
  0.469/0.393 and FastV-k3 0.703/0.561): lifetime recovers part of the
  query-conditioning gap but not enough — consistent with the fixed-point
  pattern of every training-free composition (router/QA/cascade/RankBridge)
  landing between its parents.
- GPU ≈0.75 A40·h (4 serial cells), no training; runs/deferred_rbm/ gitignored.

# RBM-OT locked gate — digest (pre-registered, task book 2026-08-10, NO-GO)

## Method (locked; no parameter search performed)
Keep 25% of pre-merger spatial units selected by plain-RBM pre-merger L2
ranking as FIXED ANCHORS.  BEFORE the native nonlinear spatial merger runs,
compute a balanced Sinkhorn optimal-transport plan (tau=0.05, exactly 20
log-domain alternating u/v updates, cosine cost in fp32, row mass 1, equal
anchor capacity n_drop/n_anchor) from every dropped pre-merger unit
descriptor `normalize(mean(H[u], dim=patch_slot))` to the anchor descriptors.
For each of the four patch slots of an anchor j, replace it with the
transport barycenter
  H'_j,p = (H_j,p + sum_i P_ij H_i,p) / (1 + sum_i P_ij)
INDEPENDENTLY across slots.  Then run the model's native merger on the
enriched full-length tensor.  Mask_source and main/deepstack alignment:
compute the plan ONCE from the first-called merger input (Qwen3-VL:
deepstack_0) and reuse on every merger's own features via a forward-pre-hook
(`MergerEnrichTap`).  vllm-mimic mrope (matches locked RBM pre25 parent).
**No learned parameters, no query attention, no router, no cascade, no
evolution-score fusion.**  Constants live in `RBOT_TAU=0.05`,
`RBOT_ITERS=20` (no CLI knobs by design).

## Implementation & reproduction
- Code: `src/v3_premerger/baselines_hf.py` `--mode rbmot` (commits
  5c72ecb implementation, 5da6c4e marg_res ruling).  New pure functions
  `rbot_descriptors`, `sinkhorn_balanced_plan`, `rbot_plan`, `rbot_apply`;
  hook class `MergerEnrichTap`.  rbmot main branch: tap.reset(grid) → native
  vision+merger forward (hook enriches) → apply_premerger(kept=plan.kept) →
  generate_pruned("pre", empty plan) over the survivors.
- Dry-checks ALL PASS (tiny Qwen2.5-VL, CPU): (C1) Sinkhorn marginals on
  canonical cosine cost: row res 5.96e-7, col res 2.38e-7 < 1e-3 (iters=20,
  tau=0.05); (C2) anchors == plain-RBM kept BIT-EXACT, budget sum(k_i)=4
  merged tokens / 16 pre-merger rows; (C3) each enriched patch slot == the
  specified weighted barycenter (all 4 slots, atol 1e-5) and dropped rows
  untouched; (C4) keep_frac=1.0 -> bitwise identity; (C5) NO cross-image
  transport, deterministic repeated output, finite; (B10) keep=100% through
  the REAL merger pre-hooks == native capture bitwise; (B11) plan reuse:
  deepstack_0-first, ONE plan, both mergers row-aligned (anchors enriched,
  drops bitwise untouched).
- n=8 TextVQA real-weight smoke (plain RBM vs RBM-OT): ptid 8/8 equal, 0 skip,
  8/8 non-empty answers, hook pattern (deepstack_0-first x4) 8/8, wall
  ratio 0.98.  marg_res distribution (observation, NOT gate): min 6.5e-5,
  max 3.66e-3 (per-sample diag field `rbmot.marg_res`).  Recurrent dev study:
  this is a property of tau=0.05 on bf16 merger inputs (near-duplicate flat-
  region descriptors => block-binary kernel => Sinkhorn contraction rate ->1
  in the degenerate regime; fp32 vs fp64 identical at 20 iters; no canonical
  20-iter scheme clears 1e-3 on these inputs).  Dry-check contract is met
  on canonical cosine cost; smoke contract per the task book is met.
  Recorded autonomously in DECISIONS.md (no escalation, no relaxation).

## Exact command (locked cells)
```
python src/v3_premerger/baselines_hf.py --mode rbmot --r 0.75 \
  --model Qwen/Qwen3-VL-8B-Instruct --benchmark $B \
  --subset eval/subsets/${B}_200.jsonl --n 200 --seed 0 \
  --out runs/rbm_ot/locked_rbmot_${B}_n200.json   # docvqa: + --max-pixels 600000
```
Script `runs/rbm_ot/locked_gate.sh`; analysis `gate_analyze.py --phase locked`
(official rescore, common-id pairing, paired McNemar z vs stronger parent).
Parents reused after exact config/ID parity check (2026-08-10): none/fst3 =
runs/rankbridge/locked_*_n200, pre25 = runs/cascade/gate_pre25_*,
fst25 (reference) = runs/cascade/gate_fst25_*.

## Locked-gate results (n=200, official metrics, common-id)
| benchmark | none | RBM(pre) | FastV-k2 | FastV-k3 | RBM-OT |
|---|---|---|---|---|---|
| textvqa | 0.8667±.023 | 0.5967±.034 | 0.6800±.032 | **0.7633**±.029 | 0.2117±.028 |
| docvqa | 0.9487±.014 | 0.4239±.033 | 0.5183±.031 | **0.5863**±.032 | 0.0452±.013 |
| ocrbench | 0.7569±.032 | **0.5801**±.037 | 0.1713±.028 | 0.4586±.037 | 0.2376±.032 |
| gqa | 0.6050±.035 | 0.4150±.035 | 0.4900±.035 | **0.5050**±.035 | 0.3750±.034 |

Marg_res (per-sample diag): textvqa 1.0e-6..1.0e-2 (82/200 >1e-3),
docvqa 1.0e-6..3.4e-2 (6/200), ocrbench 0.0..1.9e-2 (23/181), gqa 4.0e-6..2.6e-2
(96/200).  Mean_ptid RBM-OT == RBM (213/176/115/95 vs 213/176/115/95;
docvqa 176); wall_s RBM-OT (98/146/215/57 s) similar to RBM (117/144/220/47 s).
Budget / identity: per-sample prompt_token_ids RBM-OT == RBM on
200/200, 200/200, 181/181, 200/200 across the 4 benches.

## Paired evidence (McNemar common-id, vs the stronger parent)
| benchmark | stronger | RBM-OT vs stronger (z) | b10 | b01 | n_common |
|---|---|---|---|---|---|
| textvqa | fst3 (0.7633) | -10.14 | 5 | 117 | 200 |
| docvqa | fst3 (0.5863) | -10.82 | 1 | 120 | 200 |
| ocrbench | pre (0.5801) | -7.63 | 2 | 64 | 181 |
| gqa | fst3 (0.5050) | -3.25 | 19 | 45 | 200 |

## Verdict — NO-GO
Pre-registered GO rule: EVERY bench rbmot >= max(RBM,FastV-k3)-1pp AND
>=1 bench strictly above the stronger parent with paired z >= 1.5.
cond1 FAILS on ALL 4 benches (textvqa -55.2pp / docvqa -54.1pp / ocrbench
-34.3pp / gqa -13.0pp vs stronger parent; all paired z strongly negative).
cond2 trivially fails (no strict-over-parent bench).  Per the lock: stop all
RBM-OT variants, NO search of tau, iterations, layer, cost mixture, spatial
penalty, anchor ratio, unbalanced OT, or evolution-score fusion after
seeing results.  Method NOT promoted to a SOTA claim; passing this gate
would have been method-candidate ONLY, and it has not passed.

## Failure mode (bounded, mechanistic)
Anchors are bit-identical to plain RBM (200/200 ptid match, dry-check
identity holds).  Yet accuracy collapses well below RBM on all 4 benches:
RBM-OT answers remain coherent (non-empty, on-topic) but with degraded
text reading (e.g. textvqa "1:00" vs gt "12:34;12:34 am;...").  The four
patch slots of every anchor are each replaced by a transport barycenter
(anchors + 3 dropped on average per anchor, n_drop/n_anchor~3) of its
assigned cluster -- each anchor becomes a local average of ~4 units'
worth of information, diluting OCR-critical fine detail that plain RBM
preserved by keeping the anchor's original rows untouched.  RBM's
strength is the purity of the kept rows; the barycenter mixes them away.
This is the sixth consecutive bounded negative on the
training-free-composition ladder (router / QA-gate / cascade / RankBridge /
Deferred-RBM / RBM-OT), reinforcing the fixed-point framing: composition
cannibalizes the discriminator that RBM's anchor selection was buying.

## GPU accounting
Compute sum Σ(wall_s+load_s) across all cells this session ≈ 0.21 GPU·h:
- deferred dev 4 cells: 35+22, 38+5, 65+5, 12+5 s
- rbmot smoke 2 cells: 5+5, 5+5 s
- rbmot locked 4 cells: 98+5, 146+5, 215+5, 57+5 s
Session wall-clock longer (model loads, process startup, inter-cell gaps).
Total well under the task 6 GPU·h per-run budget.  No SOTA claim, no
port to the vLLM runner (per task book "Port to the canonical vLLM runner
only after the accuracy GO gate" — gate did not pass).

## Artifacts
- Code: src/v3_premerger/baselines_hf.py (commits 5c72ecb, 5da6c4e).
- Run JSONs: runs/rbm_ot/{smoke,locked}_*.json (gitignored, on disk).
- Scripts: runs/rbm_ot/{smoke,locked_gate}.sh, gate_analyze.py
  (gitignored, on disk).
- Digests: experiments/cvpr2026_token_compression_audit.md (Step 0),
  experiments/deferred_rbm_gate.md (Stage A), this file (Stage B).
- DECISIONS entries: novelty audit (NO-COLLISION) 2026-08-10;
  marg_res ruling (method-inherent) 2026-08-10;
  terminal gate (this NO-GO) appended 2026-08-10.

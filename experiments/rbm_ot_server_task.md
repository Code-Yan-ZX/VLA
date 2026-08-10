# Server Claude Code task: Deferred RBM -> Rank-Transport-Before-Merge

Date: 2026-08-10. Hardware: 1x A40 46 GB. Run GPU jobs serially.

## Role and reporting contract

Execute this task end to end on the server repository at
`/media/disk2/YZX/research/vla`. Use subagents for paper/code reading and keep
their replies to at most 20 lines plus artifact path and next recommendation.
Do not paste full papers, logs, or source files into the main context.

Commit and push each meaningful checkpoint to `origin/main` as `Code-Yan-ZX`.
Do not add AI attribution or `Co-Authored-By`. Do not commit runs, weights,
datasets, caches, or logs. One GPU job at a time. Stop and ask the user before
any single run estimated above 6 GPU hours, on credentials, on a novelty
collision that removes the gap, or if an existing core paper claim is
overturned.

## Objective

Run a pre-registered two-stage innovation ladder. First test whether token
*lifetime* alone closes the parent trade-off with an existing zero-code arm.
Only if that fails, test one genuinely different compression primitive rather
than another selector mixture:

> Keep the exact plain-RBM anchors, but replace hard deletion of all other
> pre-merger units with balanced semantic transport into those anchors before
> the model's native nonlinear spatial merger.

Stage A is **Deferred RBM**: keep identities are exactly RBM, but all visual
tokens participate in the first three LLM layers before deletion. Stage B is
**Rank-Transport-Before-Merge (RBM-OT)**. Together they ask whether the
fixed-point failures of router, QA selector, cascade, and RankBridge arise from
insufficient token lifetime or from irreversible hard discard rather than from
the RBM ranking itself.

## Step 0 - novelty and claim audit (CPU/network, must precede implementation)

Verify the official CVPR 2026 papers and released code for TransPrune, V2Drop,
Recursive Token Reduction, AOT, and MacTok. Record exact title, task, model,
compression stage, training requirement, metric, baseline, code URL, and what
the advertised percentage actually measures in
`experiments/cvpr2026_token_compression_audit.md`.

Use `experiments/literature_frontier_20260810.md` as a candidate/source map,
but independently recheck its official URLs and repository status before using
any statement as evidence.

Specifically answer:

1. Does AOT transport discarded tokens at a VLM vision-merger input, keep a
   separately selected anchor set fixed, aggregate the four patch offsets
   independently, and then execute the native nonlinear merger?
2. Do TransPrune/V2Drop score evolution inside the vision encoder or the LLM,
   and are their results video-only, image-VLM, or tokenizer/generation tasks?
3. Is any released method already identical to the locked RBM-OT definition
   below?

MacTok is an image tokenizer/generation paper unless the official paper proves
otherwise; Recursive Token Reduction targets learned multi-turn compression.
Neither is a direct baseline for the current training-free single-turn VQA
gate. Do not quote their numbers as comparable SOTA evidence.

**Collision rule:** if an existing paper already implements the same stage,
anchor policy, transport operation, and native-merger placement, stop before
GPU work. Write the collision precisely to `DECISIONS.md` and report it. A
shared use of optimal transport alone is not a collision.

## Stage A - Deferred RBM lifetime gate (zero code, run first)

The existing HF harness already realizes the exact arm:

```bash
python src/v3_premerger/baselines_hf.py \
  --mode rankbridge --rb-fuse quota --rb-rho 1.0 \
  --fastv-k 3 --r 0.75 ...
```

At `rho=1`, selected identities equal RBM top-k, while all tokens remain through
LLM layer 3 and can write context into the eventual anchors. This is a fixed
token-lifetime hypothesis, not a reopened rho/K search. Do not test any other
rho or K.

1. Verify exact per-image kept-index equality with plain RBM and non-identity
   of survivor hidden states at the deletion boundary.
2. Run the four locked `n=64` prefixes for TextVQA, DocVQA, OCRBench, and GQA,
   reusing parent prefixes only after exact config/ID checks.
3. Advance to locked `n=200` only if all four dev metrics are at least
   `max(RBM, FastV-k3)-1pp` and at least one strictly exceeds both parents.
4. At `n=200`, apply the same terminal GO rule and paired `z>=1.5` requirement
   defined below for RBM-OT.

If Deferred RBM passes the locked gate, stop: it is the winning method
candidate and RBM-OT must not be implemented. Report that its first-three-layer
compute differs from plain RBM and measure efficiency later before any SOTA
claim. If it fails either gate, record a bounded NO-GO in
`experiments/deferred_rbm_gate.md` and proceed to Stage B without changing
rho/K. Expected dev cost is about 0.7 A40 GPU-hours; locked follow-up about 2.3
GPU-hours only if dev passes.

## Stage B - locked RBM-OT method (conditional, no parameter search)

Initial scope: Qwen3-VL-8B-Instruct, plain RBM L2 ranking, keep 25%, seed 0.
Do not add query attention, a router, cascade, learned parameters, or an
evolution-score fusion in this gate.

For each image and each pre-merger representation:

1. Reshape merger input to units `H in [U, 4, D]` using the existing exact
   block-major spatial-unit convention.
2. Select anchor indices with the existing plain-RBM L2 implementation. The
   anchor indices must be bit-identical to the current RBM arm.
3. Define each unit descriptor as `normalize(mean(H[u], dim=patch_slot))`.
4. For dropped descriptors `D_i` and anchor descriptors `A_j`, use cosine cost
   `C_ij = 1 - D_i dot A_j` in fp32.
5. Compute a balanced Sinkhorn plan with fixed `tau=0.05`, exactly 20
   iterations, row mass 1, and equal anchor capacity `n_drop / n_anchor`.
   Implement with PyTorch only and stable log-domain or max-shifted updates.
6. For each of the four patch offsets independently, replace an anchor by the
   transport barycenter

   `H'_j,p = (H_j,p + sum_i P_ij H_i,p) / (1 + sum_i P_ij)`.

7. Pass enriched anchors through every original native merger. For Qwen3-VL,
   compute/cache the anchor mask and transport plan from the same first-called
   merger input used by plain RBM, then reuse that plan on each merger's own
   features so main/deepstack alignment remains exact.
8. Preserve the current placeholder, mRoPE, deepstack, per-image budget, and
   output-token-count behavior. At keep=100%, the method is identity.

The preferred accuracy-gate integration is
`src/v3_premerger/baselines_hf.py`, because the locked RankBridge/RBM/FastV-k3
parents were evaluated in that harness. A merger pre-hook may enrich only the
anchor rows while leaving the full tensor length unchanged during native HF
preparation; the existing `apply_premerger` then selects the same anchors.
Factor Sinkhorn/barycenter math into pure functions. Port to the canonical
vLLM runner only after the accuracy GO gate.

Do not substitute nearest-neighbor merging, unbalanced softmax, sequential
averaging, VisionZip grouping, or ToMe bipartite pairing and still call it the
locked method.

## CPU and real-model correctness gates

Extend `--dry-check` with all of the following:

- keep=100% is bit-identical to plain native execution;
- selected indices equal plain RBM exactly;
- exact output budget is `sum(k_i) * 4` pre-merger rows and `sum(k_i)` merged
  tokens;
- Sinkhorn row/column marginal maximum residual is below `1e-3`;
- finite outputs, deterministic repeated output, and no cross-image transport;
- each enriched patch slot is the specified weighted barycenter;
- deepstack and main merger reuse the same plan without row misalignment.

Then run a real-weight `n=8` TextVQA smoke for plain RBM and RBM-OT. Require
identical per-sample ptid, zero skips, nonempty answers, expected hook counts,
and no material wall-time anomaly. This smoke is functional only and must not
be used to tune anything.

## Pre-registered RBM-OT GPU gate

Use exactly the RankBridge locked-gate model, subsets, image limits, generation
protocol, official scorers, and common-ID paired analysis documented in
`experiments/rankbridge_gate.md`. Reuse parent JSON only after checking config
and sample IDs exactly; otherwise rerun the parent in the same harness.

Run one fixed RBM-OT candidate cell at `n=200` for each:

- TextVQA
- DocVQA
- OCRBench
- GQA

Report none, plain RBM, FastV-k3, and RBM-OT with official metric, sample SE,
mean ptid, skips, wall time, and paired discordant counts/z against the stronger
parent `max(RBM, FastV-k3)`.

**GO:** every benchmark is at least `max(RBM, FastV-k3) - 1.0 percentage point`,
and at least one benchmark is strictly above its stronger parent with paired
`z >= 1.5`.

**NO-GO:** any benchmark misses the per-benchmark floor, or none exceeds a
stronger parent at the locked evidence threshold. Stop all RBM-OT variants. Do
not search `tau`, iterations, layer, cost mixture, spatial penalty, anchor
ratio, unbalanced OT, or an evolution-score fusion after seeing results.

Passing this gate makes RBM-OT a method candidate, not a SOTA claim. A SOTA
claim additionally requires full splits, same-model/same-harness published
baselines, efficiency measurement, and statistical support.

## Evolution trajectory direction (deferred, conditional only)

TransPrune/V2Drop motivate a separate score based on layer-to-layer magnitude
and angular change. It is deliberately excluded from the first gate because it
changes the selector while RBM-OT is meant to isolate the reduction primitive.
Only if RBM-OT passes may you propose a new, separately pre-registered ablation
for evolution-ranked anchors. Do not run it automatically.

## Required deliverables

1. `experiments/cvpr2026_token_compression_audit.md` - concise verified audit.
2. `experiments/deferred_rbm_gate.md` - exact commands, identity checks,
   paired result, and terminal/advance verdict.
3. Conditional RBM-OT implementation plus CPU dry-checks in the minimal
   existing harness files.
4. `experiments/rbm_ot_gate.md` if Stage B runs - method, commands, hashes,
   results, paired statistics, GO/NO-GO, failures, and GPU hours.
5. `STATE.md` updated to at most 30 lines.
6. `DECISIONS.md` entry recording novelty audit and terminal gate decision.
7. Commit and push code/digests only; leave run directories gitignored.

At completion, reply with at most 20 lines: verdict, four benchmark deltas,
paired evidence, GPU hours, commit, artifact paths, and the single next action.

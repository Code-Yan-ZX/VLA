# Next server experiments after the four-family submission audit

Date: 2026-08-05. Hardware: one A40 46 GB; run GPU jobs serially.

## P0 - GLM official-sampling gate (must run first)

The existing GLM-4.1V-9B-Thinking gate is scientifically useful only for the
large text-dense pre-vs-post direction. Greedy decoding makes the uncompressed
GQA anchor collapse (`0.150` versus the released approximately `0.77`) and
therefore cannot test the workload crossover. Re-run the same 9 cells with the
model's released sampling settings, keeping all other controls unchanged:

```bash
cd /media/disk2/YZX/research/vla
OUT=runs/glm4v_sampling_gate_seed0 \
TEMPERATURE=0.8 TOP_P=0.6 TOP_K=2 SEED=0 \
bash src/v3_premerger/glm4v_gate.sh
```

Locked design: `n=200`, none/pre/post, TextVQA/DocVQA/GQA, keep 25%, L2,
identical per-example seed across arms, max tokens 1024, default image budget.
Expected cost is about 3 GPU hours from the prior 9-cell gate. Record official
score, containment score, boxed-answer convergence, ptid, skips, and wall time.

Decision gate:

- If the none anchors recover near the released range, use the sampling result
  as the GLM headline protocol and retain greedy only as a diagnostic.
- If the anchors still collapse, stop GLM expansion; do not spend on full
  splits. Keep the current `n=200` text-dense replication with its protocol
  limitation.
- Regardless of direction, never compare GLM absolute scores across families.

## P1 - sampling robustness (conditional)

Only if P0 restores the none anchors, repeat seeds 1 and 2 in separate output
directories. The text-dense gaps are large, but the GQA difference is small, so
the primary purpose is to report mean/std and determine whether GQA is a tie or
a stable crossover. Do not pool stochastic samples as if they were independent
images; report per-seed paired deltas.

## P2 - causal generalization on InternVL3 (high value, lower priority)

Run the ranking-swap/kept-set identity control on InternVL3 at `n=200`, keep
25%, for TextVQA, DocVQA, and GQA. This closes the current mechanism gap noted
in the conclusion: the accuracy stage law already transfers to InternVL3, but
the kept-set causal identity is currently demonstrated only on the two Qwen
architectures.

## Do not run yet

- No fifth model family: it adds breadth but does not repair a current claim.
- No GLM full split before P0: scaling a protocol-invalid anchor is wasted GPU.
- No additional 4096-token greedy cells: the completed probe already ruled out
  a short output cap.
- No new cascade/router sweep: the negative-design-space evidence is already
  saturated relative to the paper's main claim.

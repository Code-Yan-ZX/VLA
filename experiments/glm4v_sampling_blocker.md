# GLM-4.1V-9B-Thinking Official-Sampling Gate — BLOCKER (P0-1)

**Date:** 2026-08-06 · **Verdict:** ANCHOR DOES NOT RECOVER → stop GLM expansion.
**Rule applied:** user P0-1 — "若 anchor 不恢复：停止 GLM 扩展，报告 blocker，正文应撤下 GLM headline".

## What was run

Re-ran the fourth-family stage-law gate with the model's PUBLISHED sampling protocol
(`generation_config.json`: `do_sample=True, temperature=0.8, top_p=0.6, top_k=2`), seed=0,
n=200, `none` arm only (stage 1), TextVQA/DocVQA/GQA, keep-25% controls unchanged,
max_tokens=1024 (locked in `experiments/next_server_experiments.md`). Greedy (temperature=0)
was the prior protocol and floor-collapsed GQA (`none` EM 0.15 vs released ~0.77).

Outputs: `runs/glm4v_gate_sampling/glm4v_none_{textvqa,docvqa,gqa}_r0.750_s0_n200.json`.
Runner: `src/v3_premerger/v3_premerger_runner.py` (+ `--temperature/--top-p/--top-k`, per-sample
`gen_len`/`finish_reason`/`boxed`). Scoring: OFFICIAL per-sample metrics via
`extract_final_answer` (`<|begin_of_box|>...<|end_of_box|>` → `</think>`-post → full) +
`official_scorers.py` (VQA-acc / ANLS / exact-match). NOTE: the gate's inline rescore
printed 0.0 because it scored the RAW think-block text without extraction — that was a
rescorer bug, NOT the model; the numbers below use correct extraction.

## Result — anchor does NOT recover

| benchmark | greedy none | sampling none | released (~) | box_rate (samp) | len_cutoff (samp) |
|-----------|------------|---------------|--------------|-----------------|-------------------|
| TextVQA (VQA-acc) | 0.242 | **0.287** | ~0.8 | 32.5% | 95.5% |
| DocVQA (ANLS)     | 0.104 | **0.155** | ~0.8 | 22.5% | 97.0% |
| GQA (exact-match) | 0.150 | **0.165** | ~0.77 | 25.5% | 98.5% |

Sampling improves over greedy by only ~+4.5pp at most — nowhere near the released range.
GQA (the decisive arm: 0.77 released) stays at 0.165. **Anchor not recovered.**

## Root cause — degenerate thinking loops (NOT max_tokens)

The model emits a boxed answer (`<|begin_of_box|>…<|end_of_box|>`) only 22–32% of the time.
Repetition analysis (8-word span repeated ≥3× in the last 600 chars of each output):

| benchmark | any-loop | no-box & loop (of no-box) |
|-----------|---------|---------------------------|
| TextVQA | 95% | 98% |
| DocVQA  | 96% | 98% |
| GQA     | 100% | 100% |

98–100% of non-boxed outputs are **stuck in repetitive think loops** (e.g. GQA:
*"Wait, the question is… Wait, the question is…"*) — not slow progress. 95–98% hit the
1024-token length cutoff (model does not stop). This is a **model-integration failure in
the vLLM setup**, not a token-budget issue: more max_tokens would only lengthen the loops
(the prior greedy 4096 probe also failed to converge). The model runs far below its
released level under BOTH greedy and sampling.

Likely integration factors (not yet diagnosed): chat-template / thinking-mode prompt,
stop-token ids (`<|end_of_box|>` / eos not honored), or a repetition-penalty/sampling-config
mismatch in vLLM 0.19 V1 for this thinking model.

## Scoring-methodology note (important for the paper)

The runner's `acc` field uses `score_textvqa` (defined in the runner, a LENIENT
containment match; `score_docvqa = score_textvqa`) — NOT the official metric. It reports
TextVQA none = 0.825, which is misleading. The OFFICIAL VQA-acc (strict, with extraction)
= 0.287. Any paper number must use `official_scorers.score_*_vqaacc/anls` + extraction
(the `paired_stats.py` pipeline already does this correctly).

## Decision (per user P0-1 rule)

1. **STOP GLM expansion.** No seeds 1/2. No full split. (Stage 1 only; GPU ~5.5h total incl. load.)
2. **Report blocker** (this file).
3. **Pull the GLM headline from the paper.** The GLM fourth-family result (greedy AND
   sampling) is from a model running at 0.155–0.287 absolute vs its released ~0.77–0.8 —
   the absolute scores are invalid. The pre≫post text-dense delta (P0-2: +16.8pp TextVQA,
   +9.8pp DocVQA, p≈5e-5) holds numerically BUT is measured on a degenerate integration,
   so it does not support an architecture-generalization claim.

## Options for the user (claim-level, pre-submission → escalated)

- **A. Remove GLM entirely** from the paper (drop the fourth-family section + Table 1 GLM
  rows + GLM mechanism paragraph). Cleanest; stage law then rests on Qwen3-VL / Qwen2.5-VL /
  InternVL3 (3 families, all with valid anchors + paired CIs).
- **B. Retain GLM text-dense delta with a strong "integration-failure / protocol-limitation"
  caveat.** Not recommended: a delta from a model stuck in repetition loops is not
  interpretable, and reviewers will flag the 0.16–0.29 absolutes.
- **C. Attempt to fix the GLM-vLLM integration** (chat template, stop tokens, repetition
  penalty) and re-run. Uncertain outcome, needs research + GPU; the prior greedy 4096 probe
  already failed. Would delay submission.
- **D. Replace the fourth family** with a different non-thinking model (e.g. GLM-4.1V-9B
  non-Thinking, or another independent lineage). Out of scope of the current plan (user:
  no fifth model).

**Recommendation:** A (remove GLM) or C (fix integration) — decision is the user's.

## Artifacts
- Per-sample JSONs: `runs/glm4v_gate_sampling/glm4v_none_{textvqa,docvqa,gqa}_r0.750_s0_n200.json`
- Rescore/analysis: inline in this file; reproducible via `official_scorers.py` + `extract_final_answer` (see `src/v3_premerger/paired_stats.py`).
- Gate script: `src/v3_premerger/glm4v_sampling_gate.sh`; runner sampling flags: commit `a029c89`.

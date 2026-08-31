# Q-RBM E1 — Pre-registered Plan: Causal-Importance Gate

> Status: **FINAL — pre-registered 2026-09-01 before any E1 run.** Branch `exp/qrbm-e1` (base origin/main 370fb7b).
> Model: Qwen3-VL-8B-Instruct, ALL VLM params frozen, forward-only. Env: `qwen3vl_clean` (torch 2.10.0, transformers 4.57.6).
> Acceptance criteria are locked in §4. No gate-sample tuning. This plan is written BEFORE the experiment runs.

## 0. Direction (user decision; overrides the "method-innovation freeze" of STATE.md 08-25)
Reopen P2 method innovation: **Query-Conditioned Pre-Merger Utility Ranking (Q-RBM)**.
Current RBM submission is kept as fallback; E1 changes no paper claims (§7).
Do NOT revisit: frequency, edge, diversity, image-only router, cascade, OT, RankBridge,
deferred-RBM, merger-representation, or other training-free patchworks (boundary audit §3).

## 1. Motivation (verified against main.tex, DECISIONS, three audit sub-agents)
- The query-dependent headroom is real but currently unreachable pre-merger: FastV (query-conditioned,
  POST-merge, layer-K attention) leads RBM on TextVQA (+8.2pp n=200 / +17.2pp full) and GQA, while
  RBM holds OCRBench/DocVQA. Routing analysis: only ~27% of the per-sample oracle gap is
  workload-level → ~73% is sample- and query-dependent (main.tex:893-894).
- **Boundary audit conclusion (sub-agent A):** the ONLY untested cell is a **trained**
  query-conditioned pre-merger ranker with **causal-utility (occlusion-ΔLL) supervision**.
  - J5 (2026-07-24): hand-fixed query-embedding cosine blend `s=(1−λ)l2+λ·qsim` → every λ>0 ≤ L2
    (−1.7/−5.2/−3.3pp) → rules out hand-tuned training-free query mixing (DECISIONS.md:120).
  - D1 (2026-08-18): learned MLP distilling a **POST-L2 teacher** → TextVQA −16.0pp → rules out
    post-merger L2 as supervision (main.tex:918-924).
  - Cascade / RankBridge / deferred-RBM / RBM-OT / freq / adaptive router / merger-representation:
    all NO-GO; no failed run combines learned-predictor + causal supervision + query-conditioning.
- **E1's causal claim (what this gate falsifies):** does adding the query representation to
  pre-merger unit features significantly improve prediction of the *causal* unit-utility ranking
  (units whose removal most lowers the GT-answer log-likelihood) on held-out data, relative to
  image-only signals (pre-L2/post-L2)? If NO → NO-GO, no scorer training (E2 not entered).
  If YES → design a lightweight query-conditioned pre-merger ranker (E2; §8).

## 2. Data & splits (strictly fixed; no tuning on gate samples)
- Dev/gate set: existing `runs/merger_repr/dev_{bench}_64.jsonl` (n=64×4; verified **disjoint**
  from the official `eval/subsets/*_200.jsonl` gate sets; rebuilt deterministically by
  `scripts/build_e1_splits.py` if missing).
- Held-out set (NEW): n=64×4 built by `scripts/build_e1_splits.py` from
  `eval/full_splits/{textvqa_val,docvqa_val,gqa_testdev,ocrbench}.jsonl`, deterministic
  `random.Random("E1-{bench}")`, **disjoint from both dev_64 and gate_200** (j5-pattern).
- Total N = 4 benches × 128 = 512 samples. Predictors are fit on dev_64 only, evaluated on
  held-out_64 only.

## 3. Causal utility via occlusion (forward-only, frozen VLM)
- Unit = pre-merger 2×2 merge unit (4 consecutive patch tokens), the RBM selection unit.
- Occlusion = **drop-out** (slice the unit's 4 rows at the merger input — identical to RBM pruning),
  NOT zero-out (zeroing changes attention normalization: a different intervention).
- Cheap scheme (pre-registered): partition each image's units into **G=8 raster-order contiguous
  spatial blocks**; each block's units are dropped in one forward. Passes/sample = G+1.
- GT-answer log-likelihood: teacher-forced over the accepted-answer token sequence.
  Canonical answer a* = accepted GT answer with max LL, **chosen once on the FULL pass**;
  all G occlusion passes score only a*. (GT is semicolon-joined multi-answer for
  TextVQA/DocVQA; GQA single answer.)
- Causal utility per block b: u_b = LL_full(a*) − LL_without-b(a*). Block-granularity (all units
  in a block share u_b). Per-image z-scored for the prediction task.
- Signals (all from the full pass; no extra forwards): pre-L2 (`premerger_keep_units`/
  `_score_units`), post-L2 (`postmerger_keep_tokens`), FastV (layer-K query→image attention,
  `rank_keep_indices`, mean over heads, last-query row), qsim (`max_t cos(merger(unit),
  LLM_embed(q_t))` — the existing J5 query feature in the runner).

## 4. Metrics & pre-registered GO/NO-GO (locked; paired, 95% bootstrap CI clustered by image + Wilcoxon, via `paired_stats.py`)

| # | Metric (all on held-out_64) | Definition | GO | NO-GO |
|---|---|---|---|---|
| G1 (KEY) | ΔnDCG@25% of query-conditioned predictor | nDCG@25% of predictor B [pre-L2, post-L2, qsim] minus A [pre-L2, post-L2], both fit on dev_64 (simple linear/logistic ranker on standardized features), eval on held-out_64; k=ceil(0.25·G) | mean Δ ≥ +0.05 AND CI excludes 0 AND ≥3/4 benches positive | mean Δ ≤ +0.02 OR CI includes 0 |
| G2 | Answer-bearing-unit Recall@25% (TextVQA/DocVQA/OCRBench) | per-sample \|causal top-25% ∩ predictor-B top-25%\| / \|causal top-25%\| vs pre-L2 (RBM) | B ≥ RBM on ≥2/3 benches AND ≥ RBM−0.05 on the 3rd | B < RBM on ≥2/3 benches |
| G3 | GQA causal ranking | per-sample nDCG@25% of predictor B vs RBM (same def) | B − RBM ≥ +0.05 | ≤ +0.02 |
| G4 | Per-sample win predictor | logistic on x_s = Spearman(qsim, pre-L2) per sample (query-image alignment); label = FastV-vs-RBM winner by higher causal Recall@25%; fit dev, eval held-out | accuracy > 0.60 with lower CI > 0.50 | ≤ 0.55 or CI ∋ 0.50 |

- **Global rule (per user): if G1 fails → NO-GO, stop, do NOT train any scorer (E2 not entered).**
  G2–G4 are additional GO evidence, not alternatives to G1.
- **J5-differentiation (pre-registered):** E1 differs from the ruled-out J5 in exactly three ways —
  (1) learned predictor fit on dev (J5: hand-fixed λ blend), (2) supervision = causal downstream
  utility ΔLL (J5: similarity heuristic, no supervision), (3) evaluation on a held-out set disjoint
  from all prior gate sets (J5: dev-slice λ selection). A G1 failure is therefore attributable to
  "query signal does not help causal-utility prediction pre-merger", not to "J5 repeated".
- The fixed-λ qsim blend (λ=0.5, minmax-normalized) is reported **as a diagnostic only**
  (J5-replication sanity check), never as the Q-RBM arm.

## 5. GPU budget (est. before running; target ~1 A40·h, must be <6)
- Pass count: per bench 2·64·(G+1) = 1,152 (dev+held-out); total **4,608** passes (G=8).
- Anchors: Qwen3-VL-8B prefill ≈888 tok/s (~0.16 s/req, vLLM) and HF-eager batched occlusion
  ≈0.4–0.9 s/pass (prefill 111–300 tok + ~8–16 answer tok).
- **Estimate: 4,608 × 0.4–0.9 s ≈ 0.5–1.1 A40·h (G=8).** Safely <6 GPU·h — no reduction needed;
  escalation to G=16 only if a 4-sample smoke shows <0.4 s/pass (then 1.0–2.2 A·h, still <6).
- A 4-sample × G=8 smoke is run first: measure s/pass, validate ΔLL (drop a unit containing a
  known answer-bearing region → LL must drop), check a* selection.

## 6. Implementation plan (code, config, smoke)
- `scripts/build_e1_splits.py` — held-out_64 build (+ deterministic dev_64 rebuild if missing) + audit JSON.
- `src/v3_premerger/e1_causal_import.py` — driver: full pass (capture pre/post/attention/qsim + GT-LL),
  G drop-masks via `apply_premerger`-style block keep-mask, ΔLL per block → per-sample JSON
  `runs/e1/{dev,heldout}_{bench}.json` (gitignored).
- `scripts/e1_metrics.py` — fit predictors on dev_64, eval held-out_64, compute G1–G4, paired CIs,
  write `runs/e1/metrics.json` + verdict.
- Smoke test: n=4/bench × G=8, verify pass count, s/pass, ΔLL sanity, a* selection, and that the
  pipeline runs end-to-end before the 512-sample gate.
- GPU etiquette: reuse `wait_gpu` (free ≥ 30 GiB) before launch; cells resumable/idempotent.

## 7. Deliverables & guardrails
- This plan, `experiments/qrbm_e1_gate.md` (live gate log), code + config + smoke, STATE/DECISIONS update.
- **No paper-claim changes in this phase.** Recorded GQA issue (below) is deferred to submission hardening.
- Commit + push `origin main` under user identity `Code-Yan-ZX`, no AI attribution; never commit
  weights/data/logs (runs/ gitignored).

## 8. Recorded issue (item 9 of the direction; NOT part of E1)
**GQA claim conflict — pre-submission must-fix:**
- Draft main.tex:557-558 (n=200): "GQA, pre-final equals post (0.450=0.450, 0.0pp, paired mean
  0.000±0.032)".
- Full-split n=12578 (reports/acmmm_final_controls_artifacts.md:83): pre-final 0.4207 vs post
  0.4771, **Δ = −5.64pp, 95% CI [−6.42, −4.86], p = 5e-05, McNemar 941 vs 1650, iso-token verified**
  — a significant *negative* stage effect on object-QA. The n=200 "0.0pp" statement is a sampling
  artifact and must be revised before submission. Recorded here and in DECISIONS.md; not to be mixed into E1.

## 9. E2 sketch (ONLY if G1 GO; not entered now)
Lightweight query-conditioned pre-merger ranker: input = query representation (qsim or richer
cross-attention probe), pre-merger unit feature (hs.reshape(num_units,4,ctx=3584)), position/grid;
supervision = causal utility (ΔLL) from E1 (NOT post-L2 distillation); RBM-preservation anchor to
avoid breaking OCR; VLM frozen, only the light scorer trained. Final method gate (user spec):
all 4 tasks ≥ max(RBM, FastV)−1pp, ≥1 task paired-significantly above a strong constituent,
OCRBench no substantial regression.

# Merger Representation Goal — Live Log

> Branch: `exp/merger-representation-goal`
> Base commit: `cad68f902fa4606d4c41f9d7b64e0dcdd86ccb82`
> Goal: discover a representation-level method strictly stronger than plain RBM
> (training-free visual-token compression). Change the *compression action / token
> representation*, NOT score/ranking/routing/position/pruning layer.
> Frozen baseline: Qwen3-VL-8B, native mRoPE, official scorer, 25% final LLM visual-token
> budget, pre-native-merger mean-patch L2, native RBM, FastV-k3, Full model, paired locked manifests.

## Timeline

- 2026-08-26: Branch created from cad68f9. Novelty audit + pipeline mapping launched (parallel subagents).
- 2026-08-26: Research question defined — Dual-Granularity Merger Representation:
  base token b_i = M(X_i); residual detail token r_i = M(ΔX_i) - M(0), ΔX_i = X_i - mean(X_i);
  M reused from model's native merger (no new params).
- 2026-08-26: **Novelty audit COMPLETE — GAP OPEN.** No existing method implements
  "native merged semantic token + independent within-group residual-detail token".
  Closest cousins all fail on ≥1 defining element:
  - HTC-VLM/HybridToken-VLM: trained, single <voco> bottleneck (not 2 independent tokens)
  - GMC: dropped-token message aggregated INTO base token (one stream)
  - VisionZip: dominant+context is cross-group, not within a fixed 2×2 native group
  - RBM-OT (own repo): one-stream OT enrichment of anchors, no separate residual stream
  Candidate proceeds. Matrix in final report.

- 2026-08-26: **Pipeline map complete.** Key facts for implementation:
  - Injection point: `_patched_pii` in `setup_pre_merger` (runner:3087) returns
    per-image splits of pruned merged tokens → residual tokens appended here.
  - Residual computation: `pruner.merger_origs["main"]` = original main-merger
    forward; `slice_input` (runner:2459) holds `feats=hs.reshape(U,4,ctx)`.
    M = LayerNorm→concat→MLP (NOT pure linear; M(0)=GELU-bias response ≠ 0, so
    subtracting M(0) is required).
  - Positions (vLLM 0.19, qwen3vl): `get_mrope_input_positions` appends FULL
    grid positions; `_calc_mrope_positions` truncates to prompt length →
    visual tokens get **first-K cells by rank** (compacted). RBM itself uses
    these compacted positions. Trailing-text overshoot tolerated by qwen3vl.
  - Key consequence: if total image tokens stay exactly K (= plain RBM budget),
    the placeholder count AND all positions AND trailing-text positions are
    IDENTICAL to plain RBM → P1 (append-tail) needs zero position-code change
    and bit-degrades at ratio=0.
  - DeepStack: visual output rows = [main | ds1 | ds2 | ds3] concatenated dim=-1.
    `_compute_deepstack_embeds` splits main vs multiscale. Residual rows must be
    [r | 0 | 0 | 0] (zero deepstack channels) → deepstack adds 0 at residual
    positions, no shape mismatch.
  - Official scorer: `src/v3_premerger/official_scorers.py`; runner's inline
    `correct` = ad-hoc containment (must NOT be used as official). Offline
    `scripts/rescore_official.py` for official metrics.
  - Manifests: locked n=200 × 4 in eval/subsets/{gqa,textvqa,docvqa,ocrbench}_200.jsonl.
    No dedicated disjoint n=64 manifest exists → must build disjoint dev sets.
  - Job launching: `python scripts/queue` (configs/queue.json), or direct
    `src/v3_premerger/v3_premerger_runner.py` invocation. Env: qwen3vl_clean
    (vLLM 0.19), model at /data/models/huggingface/hub (HF_HUB_CACHE).

- 2026-08-26: **Mechanism pre-check COMPLETE (n=15501 units, 12 mixed samples).**
  - Task-spec'd r_a = M(ΔX) − M(0): **REFUTED.** LayerNorm inside native merger
    normalizes de-meaned patches to ~unit variance → r_a norms 2–7× base
    (p50 ratio 2.22, p90 7.35), and UNcorrelated with detail (spearman vs
    within-var −0.26, vs edge +0.03, vs demotion −0.28). M(0) is a large
    constant (norm 34.5). No NaN/explosion, but severe distribution shift.
  - Mechanism-derived r_b = M(X) − M(X̄) (X̄ = group-mean repeated): **VIABLE.**
    Same magnitude regime as base (ratio p10 0.32, p50 0.81, p90 0.95),
    positively correlated with within-var (+0.31) and edge (+0.24), max-abs
    mean 0.75. Caveat: cos(r_b, base) ≈ 0.92 → residual largely parallel to
    base (39% orthogonal). Base token b_i = M(X_i) ALREADY encodes within-group
    structure (the merger is norm+MLP, not an average).
  - RBM at 25% already keeps high-detail units (kept within-var 1.36–1.64× image
    mean) → residuals add detail on top of already-detail-rich kept units.
  - DECISION: implement C1/C2 with **r_b** (r_a mechanism-refuted; r_b is the
    mechanism-derived representation — falls under C3 allowance for mechanism-
    derived alternatives). Empirical dev test is the arbiter of the redundancy
    concern (cos 0.92).

- 2026-08-26: **HARNESS PIVOT (critical).** The frozen baseline is
  `baselines_hf.py --mode pre --r-pre 0.25 --mrope NATIVE` (native mRoPE —
  kept units keep their true native coordinates; the paper's formal config).
  The vLLM runner's stock positions are first-K COMPACTED (vllm-mimic) —
  verified: native vs vllm-mimic give different answers on sample 16837
  ($200 vs $500). 严禁 vllm-mimic ⇒ the candidate MUST be evaluated in the
  native harness. vLLM-runner dual mode (P1 append) remains mechanically
  validated (Gate A vLLM: ratio=0 bit-degrades 10/10, token counts exact) but
  is NOT the evaluation harness.
  - baselines_hf dual mode implemented: dualrepr_keep_units (C1 K_b by L2,
    C2 dual+single by distortion), dualrepr_residuals (r_b from MAIN merger
    input), inject_residual (residuals after image block, native positions:
    duplicate = base cell, adjacent = cell right of base), CLI
    --repr/--repr-candidate/--repr-ratio/--repr-pos. Dry-check ALL PASS.
  - Position schemes for dev-lock: duplicate (P2, task scheme 1) vs adjacent
    (avoids duplicate-coordinate attention anomalies). Task scheme 2
    (max-deviation patch coordinate) == group coordinate in the merged grid.

## Gates

- 2026-08-26: **Gate B pre-registration (committed before any Gate B results):**
  C1 ρ ∈ {0.1, 0.2, 0.3} (task-fixed); C2 ρ ∈ {0.05, 0.1, 0.15} (D = dual
  groups / K, my pre-registered choice). Positions to compare: duplicate vs
  adjacent (C1-0.2 probe); ratio search: at most these 3 per candidate, no
  post-hoc points. One global config, no per-dataset tuning.
- [x] Gate A: CPU/10-sample correctness — **PASS (baselines_hf native harness)**
  - Dry-check ALL PASS (qwen3vl): ratio=0 → bit-identical plain RBM; C1/C2
    split sizes; main-call residuals; base+residual==K accounting.
  - GPU 10-sample: A1 plain vs A2 c1-ratio=0 → **10/10 identical answers**
    (bit-degrade); all arms skipped=0; per-image n_base+n_res == K exactly
    (e.g. 390+98=488, 14+4=18); ptid_mean identical across arms (270.9);
    residual norms sane (C1 mean 15.7, C2 mean 20-22); official rescore
    wrapper verified.
- [ ] Gate B: disjoint exploration n=64×4 (baselines + position lock + C1/C2 ratios)
- [ ] Gate C: locked n=200×4 confirmation
- [ ] Final: report + GO/NO-GO

## Decisions

- (pending audit) 如果 "native merged semantic token + independent within-group
  residual-detail token" 已被现有工作完全覆盖 → 立即停止该候选。

## 2026-08-26 Gate B Phase 1 COMPLETE (disjoint dev n=64 × 4, official metrics)

| arm | textvqa | docvqa | ocrbench | gqa | macro |
|-----|---------|--------|----------|-----|-------|
| full | 0.8281 | 0.9573 | 0.1969 | 0.6406 | 0.6557 |
| rbm_native | 0.7188 | 0.5497 | 0.1906 | 0.6406 | 0.5249 |
| fastv_k3 | 0.6667 | 0.5917 | 0.1844 | 0.5938 | 0.5091 |
| c1r2_dup | 0.7083 | 0.4678 | 0.1906 | 0.6406 | 0.5018 |
| c1r2_adj | 0.7083 | 0.4671 | 0.1906 | 0.6406 | 0.5017 |

- **Position lock: duplicate ≡ adjacent** (all datasets within 0.0007) — locked
  duplicate (task scheme 1, principled). The mRoPE position of the residual
  token has no measurable dev effect (novel token type).
- **C1-0.2 is WORSE than plain RBM**: macro 0.5018 vs 0.5249 (−2.3pp); DocVQA
  −8.2pp; textvqa −1.1pp; ocrbench/gqa 0. Mechanism prediction confirmed:
  r_b is redundant with the base (cos 0.92), and the base token already encodes
  within-group structure; trading K_r spatial groups for redundant residual
  tokens loses coverage.
- Phase 2 (C1 {0.1,0.3}, C2 {0.05,0.1,0.15}, duplicate) launched.

## 2026-08-26 Gate B COMPLETE — NO candidate passes the bar

| arm | textvqa | docvqa | ocrbench | gqa | macro | Δmacro vs RBM |
|-----|---------|--------|----------|-----|-------|---------------|
| rbm_native | 0.7188 | 0.5497 | 0.1906 | 0.6406 | 0.5249 | — |
| c1r1_dup | 0.7188 | 0.5539 | 0.1875 | 0.6562 | 0.5291 | **+0.42** |
| c1r2_dup | 0.7083 | 0.4678 | 0.1906 | 0.6406 | 0.5018 | −2.31 |
| c1r3_dup | 0.6875 | 0.4847 | 0.1875 | 0.6094 | 0.4923 | −3.26 |
| c2r05_dup | 0.7083 | 0.5382 | 0.1875 | 0.6406 | 0.5187 | −0.62 |
| c2r1_dup | 0.7083 | 0.4842 | 0.1906 | 0.6719 | 0.5138 | −1.11 |
| c2r15_dup | 0.6823 | 0.4457 | 0.1906 | 0.6250 | 0.4859 | −3.90 |

- **Monotonic degradation with residual ratio** (all datasets): candidate ≈ RBM at
  ρ→0, worse as ρ grows (DocVQA −6.5 to −10.4pp at higher ratios). No config
  beats RBM on any dataset by a meaningful margin.
- **Best = c1r1 (ρ=0.1)**: macro 0.5291 vs RBM 0.5249 (+0.42pp, within noise).
  FAILS criterion 1 (needs +1.5pp macro); FAILS criterion 4 (only gqa +1.56pp
  ≥1pp, needs ≥2 datasets); FAILS criterion 3 (textvqa 0.7188 vs full 0.8281
  = −11pp below stronger parent).
- Mechanism fully explains: r_b redundant with base (cos 0.92); base token
  already encodes within-group structure; trading K_r spatial groups for
  redundant residual tokens loses coverage.
- Per protocol: best candidate c1r1_dup enters Gate C (locked n=200) — launched.

## 2026-08-26 Gate C COMPLETE — NO-GO confirmed (locked n=200, official)

| bench | full | RBM | FastV | candidate (C1-ρ=0.1) | cand−RBM |
|-------|------|-----|-------|---------------------|----------|
| textvqa | 0.8667 | 0.7433 | 0.7633 | 0.7333 | −1.00pp |
| docvqa | 0.9487 | 0.5924 | 0.5863 | 0.5501 | −4.23pp (p=0.028) |
| ocrbench /1000 | 690.5 | 579.1 | 418.3 | 533.8 | −45.3pts (p=0.012) |
| gqa | 0.6050 | 0.5400 | 0.5050 | 0.5200 | −2.00pp |
| macro | 0.7943 | 0.6278 | 0.5783 | **0.5973** | **−3.05pp** |

- **Verdict: NO-GO.** Candidate below native RBM on all 4 locked datasets
  (macro −3.05pp; DocVQA/OCRBench significantly negative). Fails all 8 formal
  criteria. Mechanism: r_b redundant with base (cos 0.92) + spatial-coverage
  loss → monotonic degradation with ρ on both dev and locked data.
- Harness validated: my native-RBM DocVQA 0.5924 == paper frozen cell exactly.
- **No representation-level extension passed the preregistered gate; retain
  RBM as the finding-driven minimal method; proceed to submission hardening.
  Stop this round and all subsequent method-variant search (per task 止损).**

## 2026-08-26 FINAL — branch pushed
- `exp/merger-representation-goal` pushed to origin (13 commits, base cad68f9).
- Commits: d23ba82 (audit+map+mech), ec09971 (vLLM dual impl), b54dd04 (native
  harness), b880acf (Gate A), 586c23d/0a8c2cd (runners+pre-reg), f01f354 (Gate C
  analysis), 5288bba (Gate B P1), cea397c (Gate B), 8001970/4e73e2d/9571e03
  (report/state), 329c001 (final report).
- **GO/NO-GO: NO-GO** (locked n=200, macro -3.05pp below RBM, 2 datasets
  significantly negative). Retain RBM as the finding-driven minimal method.
- **是否替换 RBM: 否** — no representation-level extension passed the gate.

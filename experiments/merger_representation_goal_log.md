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

## Gates

- [ ] Gate A: CPU/10-sample correctness
- [ ] Gate B: disjoint exploration n=64×4
- [ ] Gate C: locked n=200×4 confirmation
- [ ] Final: report + GO/NO-GO

## Decisions

- (pending audit) 如果 "native merged semantic token + independent within-group
  residual-detail token" 已被现有工作完全覆盖 → 立即停止该候选。

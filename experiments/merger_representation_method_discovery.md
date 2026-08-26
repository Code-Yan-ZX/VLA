# Merger Representation Goal — Method Discovery Final Report

> Branch: `exp/merger-representation-goal`
> Base: `cad68f902fa4606d4c41f9d7b64e0dcdd86ccb82`
> Goal: a training-free visual-token compression method **strictly stronger than
> plain RBM** by changing the compression action / token representation (NOT
> score / ranking / routing / position / pruning layer). Frozen baseline:
> Qwen3-VL-8B, native mRoPE, official scorer, 25% LLM visual-token budget,
> pre-native-merger mean-patch L2, native RBM, FastV-k3, Full model.

_(DRAFT — results being filled as Gates complete.)_

## 1. Research question and candidate

For a native 2×2 merger group X_i = [x_i1..x_i4] with merged token b_i = M(X_i):

- **Candidate residual token** r_i = M(ΔX_i) − M(0), ΔX_i = X_i − mean(X_i),
  M = the model's native merger (no new parameters).
- Hypothesis: b_i carries low-frequency semantics; r_i carries within-group
  high-frequency detail (text strokes, edges); both enter the LLM independently
  under a strictly fixed total token budget K.

## 2. Novelty / gap audit — GAP OPEN

Matrix (selection location / representation action / aggregation / independent
detail tokens / trained / overlap):

| 方法 | 选择位置 | 表示动作 | 聚合? | 独立细节token? | 训练? | 与候选重叠 |
| -- | ---- | ---- | ---- | -------------- | ---- | ----- |
| VisionZip | encoder out | dominant+context | yes (跨组) | no | TF | low |
| TokenPacker | projector | coarse-fine resample | yes | no | FT | low |
| HybridToken-VLM | arch | semantic anchor + detail → single <voco> | yes | no (合成1token) | FT | mid |
| GMC | post-merger | dropped-token message into kept token | yes (并入base) | no | TF | mid |
| RBM-OT (repo) | pre-merger | OT barycenter into anchor → 1 token | yes | no | TF | mid-high |
| Fourier Compressor | freq domain | DCT low-pass | no | no | TF | low |
| PruneSID/AnchorPrune/ET-Prune | encoder/decoder | pure pruning | no | no | TF | none |
| **This candidate** | **pre-merger** | **base + within-group residual, both to LLM** | **no (独立)** | **yes** | **TF** | **—** |

**Verdict: NOT covered.** No existing method implements "native merged semantic
token + independent within-group residual-detail token built by reusing the
native merger weights, both entering the LLM under a fixed budget". Closest
cousins fail on ≥1 defining element (HTC-VLM trained + single-token bottleneck;
GMC aggregates into the base; VisionZip cross-group; RBM-OT one-stream).

## 3. Mechanism pre-checks (n = 15,501 units, 12 mixed samples)

- **r_a = M(ΔX) − M(0) — REFUTED.** The merger's internal LayerNorm normalizes
  each de-meaned patch to ~unit variance, so r_a norms are 2–7× the base
  (p50 ratio 2.22, p90 7.35) and uncorrelated with within-group detail
  (spearman vs within-var −0.26, vs edge +0.03, vs demotion −0.28). M(0) is a
  large constant (norm 34.5). No NaN, but a severe distribution shift.
- **r_b = M(X) − M(X̄) (X̄ = group-mean repeated) — VIABLE.** Same magnitude
  regime as the base (ratio p10 0.32, p50 0.81, p90 0.95), positively
  correlated with within-var (+0.31) and edge (+0.24), max-abs mean 0.75.
  Caveat: cos(r_b, b) ≈ 0.92 → the residual is largely parallel to the base
  (≈39% orthogonal).
- The native merger is NOT a simple average (it is norm+MLP), so the base token
  **already encodes much of the within-group structure**; and RBM at 25% already
  keeps the high-detail units (kept within-var 1.36–1.64× the image mean).
- ⇒ Implemented **C1/C2 with r_b** (the mechanism-derived construction; r_a is
  the documented refutation). The redundancy concern is arbitrated empirically
  on the disjoint dev sets.

## 4. Candidates (final definitions)

- **C1 — Base + Residual** (K_b + K_r = K):
  base = top-K_b merger groups by frozen RBM L2 score; residual tokens from
  those base groups, selected by residual energy ||r_b||. K_r = ρK,
  ρ ∈ {0.1, 0.2, 0.3} (pre-registered; locked, no per-dataset tuning).
- **C2 — Distortion-Constrained Granularity** (2D + (K−2D) = K):
  d_i = ||M(X_i) − M(X̄_i)||₂ / (||M(X_i)||₂ + ε) over ALL units; the top-D
  groups by distortion are protected with base+residual; the remaining budget
  (K−2D single tokens) is filled by RBM L2 among the non-dual units.
  D = ρK, ρ ∈ {0.05, 0.1, 0.15} (pre-registered).
- Hard budget / hard constraint in both; no scalar score blend, no training,
  no OT/barycenter, no query embedding, no router, no dynamic lifetime, no
  second FastV stage, no diversity/NMS, no native-coordinate modification.
- **Position schemes** (dev-locked, at most two): P2-duplicate (residual
  inherits the base group's native cell) vs P2-adjacent (the cell immediately
  right of the base). (Task scheme 2 — max-deviation patch coordinate — equals
  the group coordinate in the merged grid.)

## 5. Results

### 5.1 Gate A — correctness (PASS)

Dry-check ALL PASS (qwen3vl); GPU 10-sample: `--repr dual --repr-ratio 0.0`
is **bit-identical to plain native RBM (10/10 answers)**, per-image
`n_base + n_res == K` exactly (e.g. 390+98=488, 14+4=18), `skipped=0`,
`ptid_mean` identical across arms, residual norms sane, official rescore
verified.

### 5.2 Gate B Phase 1 — baselines + position lock (disjoint dev n=64 × 4, official)

| arm | textvqa | docvqa | ocrbench | gqa | macro |
|-----|---------|--------|----------|-----|-------|
| full | 0.8281 | 0.9573 | 0.1969 | 0.6406 | 0.6557 |
| rbm_native | 0.7188 | 0.5497 | 0.1906 | 0.6406 | 0.5249 |
| fastv_k3 | 0.6667 | 0.5917 | 0.1844 | 0.5938 | 0.5091 |
| **C1-0.2 duplicate** | 0.7083 | 0.4678 | 0.1906 | 0.6406 | **0.5018** |
| C1-0.2 adjacent | 0.7083 | 0.4671 | 0.1906 | 0.6406 | 0.5017 |

- **Position lock: duplicate ≡ adjacent** (all datasets within 0.0007) →
  locked **duplicate** (task scheme 1: residual inherits the base group's
  native cell). No duplicate-coordinate attention anomaly beyond this (the
  residual is a novel token type; its exact mRoPE position has no measurable
  dev effect).
- **C1-0.2 is below plain RBM** (macro −2.3pp; DocVQA −8.2pp, TextVQA −1.1pp,
  OCRBench/GQA 0). Confirms the mechanism prediction (redundancy + loss of
  spatial coverage).

### 5.3 Gate B — full ratio sweep (C1 {0.1,0.2,0.3}, C2 {0.05,0.1,0.15}, duplicate)

| arm | textvqa | docvqa | ocrbench | gqa | macro | Δmacro vs RBM |
|-----|---------|--------|----------|-----|-------|---------------|
| rbm_native | 0.7188 | 0.5497 | 0.1906 | 0.6406 | 0.5249 | — |
| **c1r1_dup** | 0.7188 | 0.5539 | 0.1875 | 0.6562 | **0.5291** | **+0.42** |
| c1r2_dup | 0.7083 | 0.4678 | 0.1906 | 0.6406 | 0.5018 | −2.31 |
| c1r3_dup | 0.6875 | 0.4847 | 0.1875 | 0.6094 | 0.4923 | −3.26 |
| c2r05_dup | 0.7083 | 0.5382 | 0.1875 | 0.6406 | 0.5187 | −0.62 |
| c2r1_dup | 0.7083 | 0.4842 | 0.1906 | 0.6719 | 0.5138 | −1.11 |
| c2r15_dup | 0.6823 | 0.4457 | 0.1906 | 0.6250 | 0.4859 | −3.90 |

- **Monotonic degradation with residual ratio**: candidate ≈ RBM at ρ→0,
  strictly worse as ρ grows (DocVQA −6.5pp at C1-0.3, −10.4pp at C2-0.15).
- Best config C1-ρ=0.1: macro +0.42pp vs RBM (within noise). Fails criterion 1
  (needs +1.5pp), criterion 4 (only GQA strictly exceeds its parent by ≥1pp),
  and criterion 3 (TextVQA −11pp below the Full parent).
- ⇒ Per protocol the best candidate (C1-ρ=0.1, duplicate) enters Gate C.

### 5.4 Gate C — locked n=200 × 4 confirmation (C1-ρ=0.1, duplicate)

_(running — to fill)_

## 6. Methodological acceptance (for the record — the candidate is fully
specified even though it fails the empirical gate)

- **Problem definition**: the fixed one-token-per-group representation of a
  pre-merger compression scheme forces each kept 2×2 group into a single merged
  token; we ask whether the *representation action* (not the selection) can be
  changed — under a strictly fixed LLM token budget — to carry both semantic
  and within-group detail.
- **Design principle**: semantic/detail decoupling — a base token for the group's
  semantic content and an independent residual-detail token for the within-group
  structure the base might lose.
- **Algorithm** (C1, the best config): (1) score groups by frozen pre-merger
  mean-patch L2; (2) keep K_b = K − round(ρK) groups; (3) compute
  r_b = M(X) − M(X̄) via the native main merger; (4) pick K_r = ρK of the kept
  groups by ||r_b||; (5) inject the residual rows after the image block with the
  base group's native coordinate (duplicate); (6) zero deepstack channels.
- **Fixed budget**: K_b + K_r = K per image, K = round(f·(1−r)) — byte-exact
  (Gate A). No per-dataset tuning.
- **Complexity**: per image, C1 adds O(1) extra merger forwards (a norm + 2
  linear + GELU on the K_b kept units ≈ O(K_b·d²) ≈ small fraction of the ViT);
  C2 adds O(1) merger forwards on the full unit set for the distortion. Both are
  counted in the selection/representation overhead (criterion 7).
- **Causal ablation**: ratio sweep {0.1,0.2,0.3} / {0.05,0.1,0.15} is the causal
  knob — the method is neutral at ρ→0 and strictly worse as ρ grows, isolating
  the residual tokens as the (negative) causal factor.
- **Cross-architecture path**: the construction reuses any native merger with
  the block-major 2×2 contract (Qwen2.5-VL's single merger, InternVL3's
  pixel-shuffle+mlp1, GLM-4.1V's conv+merger); only the merger handle and the
  deepstack-zero-padding differ.
- **Distinctions**: vs RBM — changes the representation action, not the
  selection; vs VisionZip — residual is within-group and independent, not
  cross-group context pooling; vs Nüwa/DUET — representation not lifetime/stage;
  vs HybridToken-VLM — training-free, reuses native merger, keeps two
  independent tokens instead of a learned single-token bottleneck.

## 7. Verdict

_(GO / NO-GO — to fill after Gate C)_

## 7. Verdict

_(GO / NO-GO — to fill)_

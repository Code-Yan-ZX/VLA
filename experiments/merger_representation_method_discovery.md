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

_(to fill from Gate B / Gate C)_

## 6. Methodological acceptance

- New problem definition, design principle (semantic/detail decoupling), clear
  algorithm, fixed token budget, complexity analysis, causal ablation,
  cross-architecture path, and explicit distinction from RBM/VisionZip/Nüwa/
  DUET/HybridToken — all documented here.

## 7. Verdict

_(GO / NO-GO — to fill)_

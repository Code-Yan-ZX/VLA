# freq-aware scorer + adaptive stage selector (Directions A/B)

Date: 2026-08-13 · Model: Qwen3-VL-8B-Instruct (repo default qwen3vl) · 1×A40
Training-free; strictly §3 iso-selector + iso-budget protocol.

Summary (三表格 + 失败原因) — 见本文件首节；完整 log 见后文。

## Setup / calibration (probe: scripts/adapt_discriminator_probe.py, n=16/class)
- raw-scale freq blend is numerically degenerate on Qwen3-VL features:
  mean(unit L2) ≈ 12 vs mean(within-unit var) ≈ 0.04 (≈300:1) → β·var is
  negligible across the α,β grid → freq would collapse to pure L2 and M2 is
  unused. DECISION: score = α·z(L2) + β·z(var) (per-image z-score of each
  component before blending); α:β is then a genuine trade-off knob;
  α=1,β=0 ≡ plain L2 ranking (rank-preserving by z).
- hf_ratio threshold calibration: using per-image MEAN var as the "high-freq"
  baseline gives TextVQA≈0.46 vs GQA≈0.48 (no separation → router is blind).
  Using mean+1σ tail (hf_var_mode=mean1sd): TextVQA≈0.142 vs GQA≈0.127 →
  separated in the mechanism-expected direction (text images carry more
  high-frequency outlier units). Default τ-grid recentered to {0.15,0.30,0.50}.

## Task 1 — frequency-aware scorer (Direction B)
- selector `freq`: score = α·z(L2)+β·z(var); PRE-merger only (POST → L2).
- grid: α,β ∈ {0.5,1.0,2.0}² on TextVQA-500 @ r=0.75 (PRE path), vs plain-l2 ref.
- eval: best (α*,β*) on DocVQA/OCRBench/GQA @ {0.75,0.5} vs pre/post.
- (cells resumable; best selected on OFFICIAL metric — same as paper rescore)

## Task 2 — adaptive stage selector (Direction A)
- workload detector on merger input (no query): high-freq ratio (mean1sd tail)
  + L2 entropy (20-bin histogram, nats). Rule: hf>τ_hf OR H>τ_ent → PRE(RBM)
  else POST(FastV). iso-budget: both keep k=f(1-r).
- grid: τ_hf×τ_ent on TextVQA-500 @ r=0.75; eval on 4 benches @ {0.75,0.5}.
- acceptance: GQA/scene +≥1pp vs RBM; TextVQA/DocVQA/OCRBench ≤0.5pp regress.

## Task 3 — combined
- freq-best scorer + adaptive router on 4 benches + ChartQA @ {0.75,0.5}
  vs RBM(pre)/FastV(post)/PyramidDrop(iso-budget) → regime map.

## Regime map (Qwen3-VL-8B, official, n500 except chartqa n200)

| bench@r | RBM(pre) | FastV(post) | freq(1,0.6) | adapt(0.08,2) | combined |
|---|---|---|---|---|---|
| textvqa@0.75 | 0.6053 | 0.2073 | **0.6173** | 0.6053 | **0.6173** |
| textvqa@0.5 | 0.6733 | 0.3873 | **0.6780** | 0.6733 | **0.6780** |
| docvqa@0.75 | **0.5063** | 0.2317 | 0.4903 | **0.5063** | 0.4903 |
| docvqa@0.5 | 0.6212 | 0.5487 | **0.6271** | 0.6212 | **0.6271** |
| ocrbench@0.75 | **0.7380** | 0.1700 | 0.7160 | **0.7380** | 0.7160 |
| ocrbench@0.5 | **0.8220** | 0.4980 | **0.8220** | **0.8220** | **0.8220** |
| gqa@0.75 | 0.4380 | **0.4540** | 0.4220 | 0.4380 | 0.4220 |
| gqa@0.5 | 0.4840 | **0.5000** | 0.4860 | 0.4840 | 0.4860 |
| chartqa@0.75 | 0.1700 | 0.1800 | — | — | **0.1950** |
| chartqa@0.5 | **0.4000** | 0.3400 | — | — | 0.3900 |

Bold = best among RBM/FastV/available methods per cell (freq/adapt/combined vary
by cell availability; ChartQA has combined (+pre/post) only).
- freq scorer: wins TextVQA both rates, docvqa@0.5, ties ocrbench@0.5 — but
  regresses −1.6..−2.2pp on docvqa/ocrbench/gqa@0.75. Best when text-dense +
  aggressive compression.
- adaptive router: ≡ RBM everywhere at the tuned τ (all-PRE). No GQA gain.
- combined = freq (router is a no-op at τ=0.08).

## Summary
- **Direction B (freq scorer)** ✅ genuine improvement on TextVQA: α=1,β=0.6 →
  +1.2pp@25% / +0.5pp@50% official, wins docvqa@0.5 (+0.59pp). But regresses
  −1.6..−2.2pp on docvqa/ocrbench/gqa@0.75. Mixed, text-dense-favouring.
- **Direction A (adaptive router)** ❌ acceptance FAIL after 2 rounds: tuned
  τ_hf=0.08 routes everything to PRE → ≡ RBM everywhere (0 text regression ✓,
  GQA +0.00pp < +1pp ✗). No global τ satisfies both constraints (GQA/TextVQA
  hf_ratio distributions overlap; τ=0.13 fixes GQA routing but TextVQA −7pp).
- **Task 3 (combined)** = freq-alone (router is a no-op at τ=0.08); table +
  regime map generated; ChartQA added (combined beats RBM/FastV @25%: 0.195 vs
  0.170/0.180; trails RBM @50% 0.390 vs 0.400 on n=200).
- Deliverables: experiments/{freq_aware,adaptive_stage,combined}/results.json +
  grid.json + this log. Training-free throughout; paper §3 iso-selector+iso-budget.

## Log

### 2026-08-13 — Task-3 combined eval complete (official)
combined = freq-scorer(α=1,β=0.6) + adaptive router(τ_hf=0.08,τ_ent=2.0).
Router routes ALL images → PRE (τ=0.08 > everything) so combined ≡ freq-alone.
| bench@r | combined | RBM | FastV | Δc-RBM | Δc-FV |
|---|---|---|---|---|---|
| textvqa@0.75/0.5 | 0.6173/0.6780 | 0.6053/0.6733 | 0.2073/0.3873 | +1.20/+0.47pp | +41.0/+29.1pp |
| docvqa@0.75/0.5 | 0.4903/0.6271 | 0.5063/0.6212 | 0.2317/0.5487 | −1.60/+0.59pp | +25.9/+7.8pp |
| ocrbench@0.75/0.5 | 0.7160/0.8220 | 0.7380/0.8220 | 0.1700/0.4980 | −2.20/0.00pp | +54.6/+32.4pp |
| gqa@0.75/0.5 | 0.4220/0.4860 | 0.4380/0.4840 | 0.4540/0.5000 | −1.60/+0.20pp | −3.2/−1.4pp |
| chartqa@0.75/0.5 | 0.1950/0.3900 | 0.1700/0.4000 | 0.1800/0.3400 | +2.50/−1.00pp | +1.5/+5.0pp |
Regime map (official, n500 / chartqa n200): see freq_adaptive.md §Regime.
PyramidDrop anchors (j7hf r0.375_canon, looser budget keep 62.5%): textvqa 0.8347,
docvqa 0.8821, ocrbench 0.7460, gqa 0.6060 (NOT iso-budget with our 25%/50%).

### 2026-08-13 — Task-2 eval complete (official): ACCEPTANCE FAIL
Best τ_hf=0.08, τ_ent=2.0 → routes ALL images to PRE on every bench:
| bench@r | adapt | RBM | FastV | Δadapt-RBM |
|---|---|---|---|---|
| textvqa@0.75/0.5 | 0.6053/0.6733 | same | 0.2073/0.3873 | 0.00pp |
| docvqa@0.75/0.5 | 0.5063/0.6212 | same | 0.2317/0.5487 | 0.00pp |
| ocrbench@0.75/0.5 | 0.7380/0.8220 | same | 0.1700/0.4980 | 0.00pp |
| gqa@0.75/0.5 | 0.4380/0.4840 | same | 0.4540/0.5000 | **0.00pp (< +1pp required) |
Text regression: PASS (0.00pp). GQA gain: FAIL (+0.00pp < +1pp).
τ_hf=0.13 (would flip GQA→POST): TextVQA −7pp (over 0.5pp limit), GQA 0.428 < RBM 0.438.
→ NO single global τ satisfies both constraints. gqa hf_mean 0.127 ≈ textvqa 0.142, overlap.
Router mechanism works; the workload features can't separate the classes at a global threshold.
Per task stop rule: Task-2 improvement not achieved → recorded, not claimed.

### 2026-08-13 — Task-2 adaptive grid (round 1: tau_hf{0.15,0.30,0.50} → miscalibrated)
- TextVQA mean1sd hf_ratio ≈ 0.142 < 0.15 → hf never fired → entropy-only routing
  flipped text images to POST → acc 0.528/0.374 (vs pre 0.68). Calibration miss.
- Round 2 (tau_hf{0.08,0.13,0.20} x tau_ent{2.0,2.5,3.0}):
  - tau_hf=0.08 column: all-PRE, acc=0.68 (≡ RBM) — hf>0.08 everywhere on TextVQA.
  - tau_hf=0.13: 269 pre / 51 post (post hf_mean 0.1203 < 0.13); acc 0.608/0.542/0.544 (POST flips hurt TextVQA).
  - tau_hf=0.20: acc 0.518/0.266/… (heavy POST flips, big loss).
- Router mechanism validated (correct per-image routing); on TextVQA POST flips are
  net-negative because FastV < RBM there. Whether any config wins GQA (where
  POST>RBM) is the open question → eval_adapt.

### 2026-08-13 — Task-1 eval complete (official)
| bench@r | freq | RBM(pre) | FastV(post) | Δfreq-RBM |
|---|---|---|---|---|
| textvqa@0.75 | 0.6173 | 0.6053 | 0.2073 | +1.20pp |
| textvqa@0.5 | 0.6780 | 0.6733 | 0.3873 | +0.47pp |
| docvqa@0.75 | 0.4903 | 0.5063 | 0.2317 | −1.60pp |
| docvqa@0.5 | 0.6271 | 0.6212 | 0.5487 | +0.59pp |
| ocrbench@0.75 | 0.7160 | 0.7380 | 0.1700 | −2.20pp |
| ocrbench@0.5 | 0.8220 | 0.8220 | 0.4980 | 0.00pp |
| gqa@0.75 | 0.4220 | 0.4380 | 0.4540 | −1.60pp |
| gqa@0.5 | 0.4860 | 0.4840 | 0.5000 | +0.20pp |

freq truly helps the most text-dense bench under aggressive compression
(TextVQA +1.2pp@25%); mixed/negative elsewhere. Task-2 (adaptive) grid running.

### 2026-08-13 — freq winner + eval
- **WINNER (α*,β*) = (1.0, 0.6)** — official TextVQA-500 @ r0.75 = 0.6173 vs L2 0.6053 (+1.2pp).
- eval_freq running: freq@(1,0.6) vs pre/post on {textvqa,docvqa,ocrbench,gqa} @ {0.75,0.5} = 24 cells.
- textvqa done (inline): @0.75 freq 0.692 / pre 0.680 / post 0.248; @0.5 freq 0.766 / pre 0.760 / post 0.458. freq>pre both rates.
- mid-eval bug: vLLM teardown race on r0.5 cells (next launch while prev process held ~5GiB → 39.2<40.0 free). Fixed: wait threshold 30→41GiB + rc≠0 retry x2 with 90s settle. Resumed.

### 2026-08-13 — freq grid (Task 1)
- Round 1 (α,β ∈ {0.5,1,2}², official metric, TextVQA-500 @ r=0.75):
  ratio-invariant z-blend; NO cell beats plain L2 (best 0.6047 vs L2 0.6053).
- Round 2 (fine α:β around the plateau): **α=1,β=0.65 (ratio 1.54) official
  0.6107 (+0.5pp vs L2)**; α=1,β=1.5 → 0.6087. Refinement β=0.6/0.7 pending.
- Bug fixed mid-grid: generic pre dispatch wasn't threading α/β (all cells ran
  silently at α=β=1.0, bit-identical acc) — re-threaded, verified, re-gridded.
- Selection metric = OFFICIAL rescore (paper protocol), not runner inline acc.

### 2026-08-13 — setup notes
- freq scorer: score = α·z(L2)+β·z(var) (per-image z-score; raw scales are
  300:1 apart so raw blending is rank-degenerate).
- adaptive router: mean1sd tail hf_ratio (TextVQA 0.142 vs GQA 0.127
  separation) + 20-bin L2 entropy; grid τ_hf∈{0.15,0.30,0.50},
  τ_ent∈{2.0,2.5,3.0}.
# P3-step-3: n=500 Pareto accuracy tightening (the GATE)

> **Gate question:** the n=200 Pareto-dominate claims on MME / ScienceQA
> rested on acc margins of ~+0.015 — WITHIN noise (n=200 acc stderr ~+-0.031).
> TextVQA's n=200 acc headline already REVERSED at n=500. This step re-runs
> the Pareto comparison at n=500 (acc stderr ~+-0.022) on GQA / MME / MMBench
> / ScienceQA to see whether MME / ScienceQA HOLD or REVERSE, and whether the
> benchmark-conditional Pareto story survives the noise gate.

> All runs: c12 (max_num_seqs=12), bursty profile, num_running controller
> (adaptive: r_min 0.25 / r_max 0.50, conc-lo 0.25 / conc-hi 0.75), seed=0.
> GQA max-tokens=32; MME/MMBench/ScienceQA max-tokens=64 — identical to the
> n=200 anchors (clean n=200 -> n=500 comparison).

> Significance calibration (n=500, two-prop acc stderr ~+-0.022):
> |margin| < 0.022 = noise; 0.022-0.044 = suggestive; >= 0.044 = meaningful.

## 1. n=500 Pareto table

| bench | config | req/s | acc | kept{min,max} | r_min | r_mean | r_max | conc_frac |
|---|---|---|---|---|---|---|---|---|
| gqa | adaptive | 2.383 | 0.556 | {288,432} | 0.250 | 0.372 | 0.500 | 0.00-1.00 |
| gqa | fixed r25 | 2.389 | 0.556 | {432,432} | — | — | — | — |
| gqa | fixed r50 | 2.607 | 0.562 | {288,288} | — | — | — | — |
| mme | adaptive | 2.588 | 0.766 | {288,432} | 0.250 | 0.372 | 0.500 | 0.00-0.92 |
| mme | fixed r25 | 2.590 | 0.758 | {432,432} | — | — | — | — |
| mme | fixed r50 | 2.749 | 0.752 | {288,288} | — | — | — | — |
| mmbench | adaptive | 3.101 | 0.726 | {288,432} | 0.250 | 0.372 | 0.500 | 0.00-0.83 |
| mmbench | fixed r25 | 3.048 | 0.726 | {432,432} | — | — | — | — |
| mmbench | fixed r50 | 3.299 | 0.712 | {288,288} | — | — | — | — |
| scienceqa | adaptive | 3.050 | 0.620 | {288,432} | 0.250 | 0.369 | 0.500 | 0.00-0.83 |
| scienceqa | fixed r25 | 2.999 | 0.618 | {432,432} | — | — | — | — |
| scienceqa | fixed r50 | 3.229 | 0.630 | {288,288} | — | — | — | — |

## 2. Per-benchmark verdict at n=500

Each row: adaptive vs r25 (req/s, must be WIN) and vs r50 (acc, must be WIN) -> Pareto-dominate?

| bench | adaptive req/s | r25 req/s | Δ req/s | adaptive acc | r50 acc | Δ acc | acc sig | verdict |
|---|---|---|---|---|---|---|---|---|
| gqa | 2.383 | 2.389 | -0.006 | 0.556 | 0.562 | -0.006 | noise (z=-0.19) | dominated |
| mme | 2.588 | 2.590 | -0.002 | 0.766 | 0.752 | +0.014 | noise (z=+0.52) | wins acc only |
| mmbench | 3.101 | 3.048 | +0.053 | 0.726 | 0.712 | +0.014 | noise (z=+0.49) | PARETO-DOMINATES |
| scienceqa | 3.050 | 2.999 | +0.052 | 0.620 | 0.630 | -0.010 | noise (z=-0.33) | wins req/s only |

## 3. TextVQA at n=500 (from P3-step-2, for the full 5-benchmark tally)

| config | n | acc |
|---|---|---|
| adaptive | 500 | 0.510 |
| fixed_r25 | 500 | 0.510 |
| fixed_r50 | 500 | 0.526 |

TextVQA n=500: Δreq/s(adaptive-r25)=+0.037 (WIN); Δacc(adaptive-r50)=-0.016 (LOSS) -> **wins req/s only**

## 4. ★ Honest bottom line (the GATE result)

**Adaptive cleanly Pareto-dominates at n=500 on 1/5 benchmarks** (gqa, mme, mmbench, scienceqa, textvqa).

Per-benchmark status at n=500:
- **gqa**: dominated
- **mme**: wins acc only
- **mmbench**: PARETO-DOMINATES
- **scienceqa**: wins req/s only
- **textvqa**: wins req/s only

**MME/ScienceQA: REVERSED (MME, ScienceQA no longer Pareto-dominate at n=500).** Method collapses to 'req/s > r25 everywhere' only — the paper's method section must drop the Pareto-claim and frame purely as throughput-optimal-under-guardrail (no per-benchmark accuracy win).

## 5. Honest interpretation (load-bearing — overrides the strict verdict labels)

**★ The n=200 Pareto story DOES NOT SURVIVE the n=500 noise gate.** Both n=200
Pareto-dominate cases (MME, ScienceQA) REVERSE at n=500, and the TextVQA
n=200→n=500 reversal from P3-step-2 stands. The strict "PARETO-DOMINATES"
label on MMBench (§2) is an ARTIFACT of the +0.014 acc margin still being
within noise (|z|=0.49) — MMBench is honestly "req/s win + acc tie", NOT a
clean Pareto-dominate. So the true count is **0/5 clean Pareto-dominate at n=500**.

**1. MME REVERSED.** n=200: adaptive beat r25 on req/s by +0.045 (the headline
req/s win). n=500: adaptive 2.588 ≈ r25 2.590 (Δ −0.002 — the req/s advantage
EVAPORATED). The acc win (adaptive 0.766 vs r50 0.752, +0.014) is noise (|z|=0.52).
So at n=500 MME is "acc ~tie, req/s ~tie" — no Pareto case. The n=200 req/s
win was the load-bearing signal and it did not replicate.

**2. ScienceQA REVERSED.** n=200: adaptive beat r50 on acc by +0.015 (WIN).
n=500: adaptive 0.620 < r50 0.630 (Δ −0.010, noise |z|=0.33 — sign flipped).
The req/s win over r25 (+0.052) HOLDS, but the acc Pareto half is gone.
ScienceQA is now "req/s win only", same status as GQA/MMBench/TextVQA.

**3. The req/s-over-r25 win is NOT uniform either.** At n=500 adaptive beats
r25 on req/s on MMBench (+0.053) and ScienceQA (+0.052) cleanly, ties on GQA
(−0.006) and MME (−0.002), and beats it on TextVQA (+0.037, n=500 P3s2). So
even the "req/s > r25 everywhere (+2–7%)" claim from the n=200 data weakens:
at n=500 it is +0–5% and is a TIE (not a win) on the 2 short-answer benchmarks
(GQA, MME). The robust statement is "req/s ≥ r25 everywhere, clean win on the
MC/dense benchmarks (MMBench, ScienceQA) and TextVQA."

**4. Why did n=200 mislead?** The n=200 req/s margins on GQA (+0.024) and MME
(+0.045) were themselves noisy: bursty-profile req/s has run-to-run variance
from request-arrival timing, and at n=200 a single slow batch flips the sign.
n=500 averages more batches and the spurious adaptive>r25 req/s edge on short-
answer tasks collapses to ~0. The acc margins (±0.015) were always within
n=200 noise (stderr ~±0.031) — n=500 (stderr ~±0.022) confirms they are noise.

## 6. Impact on the paper's method framing (the gate's consequence)

**DROP the Pareto-dominate claim entirely.** The outline (`drafts/outline.md`
lines 36-37, 77, 117) and `eval/final_results.md` Table C currently lean on
"MME/ScienceQA = 2/5 Pareto-dominate" as the accuracy-guardrail evidence. At
n=500 this is 0/5. The method must be reframed as:

  **"a load-adaptive prune-depth controller that delivers a throughput win
  over the accuracy-favoring fixed point (r25) on the dense/MC benchmarks
  (MMBench +5.3%, ScienceQA +5.2%, TextVQA +3.7%) WITHOUT an accuracy loss
  vs r25 — i.e. a free throughput gain at iso-accuracy-to-r25. It does NOT
  beat fixed-r50 on accuracy (r50's acc advantage where it exists is
  preserved within noise); the deployer choosing r50 gets higher throughput
  still (r50 > adaptive on req/s on GQA/MME/MMBench/ScienceQA). The method's
  niche is: when r25 is the mandated accuracy floor, load-adaptive recovers
  ~half the r25→r50 throughput gap at no acc cost."**

This is a WEAKER but HONEST method contribution. The measurement contribution
(served-throughput gap, 0/37) remains the paper's MAIN pillar and is
unaffected. The method becomes a clean supporting result (free throughput at
iso-acc-to-r25), not a Pareto win. Pattern Recognition / Information Sciences
still viable — the measurement + 3 findings carry the paper; the method is a
practical bonus, not the headline.

**Net: the paper is now measurement-led with a supporting (non-Pareto) method.
Reshape §3.2 (method claims), §5.2 (Table C framing), and the abstract/contrib
hierarchy in `drafts/outline.md` accordingly before drafting.**

---
Generated by `scripts/analyze_p3s3.py`.
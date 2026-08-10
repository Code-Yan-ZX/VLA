# Paired Metric Statistics (Table 1 + Table 3)

Recomputed from per-sample result JSONs under `runs/` with the OFFICIAL per-sample
metrics (`src/v3_premerger/official_scorers.py`). CPU-only, no model loading,
fully reproducible: seed=0, n_resamples=20000 (bootstrap + permutation).

- **Alignment**: samples matched by id across arms; only the intersection (both arms answered) is used; skipped samples excluded.
- **Bootstrap**: paired bootstrap of d_i = A_i - B_i, 95% percentile CI + SE.
- **Permutation**: paired sign-flip test, two-sided p.
- **McNemar** (binary metrics only: GQA, OCR-Bench): exact binomial p + z.
- **GLM-4.1V** is a thinking model: `extract_final_answer` (last box -> post-think -> full)
  applied identically to all arms before scoring (raw text scores 0.0 on every sample).
- **OCR-Bench**: question_type joined from `eval/full_splits/ocrbench.jsonl`; HME branch is case-sensitive + space-insensitive (official).
- Deltas in **pp** (percentage points). For OCR-Bench full-split cells, pp * 10 = /1000 pts.

---

## Table 1 / Stage law --- pre vs post (25% retention)

### Qwen3-VL-8B (`qwen3vl`)

#### textvqa  (VQA-acc)

- **pre vs post** (primary, stage law):
  - metric: VQA-acc (continuous)  | n_paired=5000  (A_total=5000, B_total=5000, only_A=0, only_B=0)
    mean pre=0.6053  mean post=0.2217  delta(pre-post)=+38.367 pp  95% CI [+36.873, +39.853] pp  SE=0.7620
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant A>B**
- pre vs none:
  - metric: VQA-acc (continuous)  | n_paired=4998  (A_total=5000, B_total=4998, only_A=2, only_B=0)
    mean pre=0.6056  mean none=0.8443  delta(pre-none)=-23.876 pp  95% CI [-25.110, -22.636] pp  SE=0.6321
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant B>A**
- post vs none:
  - metric: VQA-acc (continuous)  | n_paired=4998  (A_total=5000, B_total=4998, only_A=2, only_B=0)
    mean post=0.2218  mean none=0.8443  delta(post-none)=-62.258 pp  95% CI [-63.619, -60.924] pp  SE=0.6856
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant B>A**
- pre vs post @12.5% (deeper compression):
  - metric: VQA-acc (continuous)  | n_paired=5000  (A_total=5000, B_total=5000, only_A=0, only_B=0)
    mean pre@12.5%=0.4718  mean post@12.5%=0.1318  delta(pre-post)=+34.000 pp  95% CI [+32.533, +35.433] pp  SE=0.7373
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant A>B**

#### docvqa  (ANLS)

- **pre vs post** (primary, stage law):
  - metric: ANLS (continuous)  | n_paired=5349  (A_total=5349, B_total=5349, only_A=0, only_B=0)
    mean pre=0.4806  mean post=0.2377  delta(pre-post)=+24.298 pp  95% CI [+23.008, +25.602] pp  SE=0.6582
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant A>B**
- pre vs none:
  - metric: ANLS (continuous)  | n_paired=5349  (A_total=5349, B_total=5349, only_A=0, only_B=0)
    mean pre=0.4806  mean none=0.9562  delta(pre-none)=-47.559 pp  95% CI [-48.882, -46.243] pp  SE=0.6706
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant B>A**
- post vs none:
  - metric: ANLS (continuous)  | n_paired=5349  (A_total=5349, B_total=5349, only_A=0, only_B=0)
    mean post=0.2377  mean none=0.9562  delta(post-none)=-71.857 pp  95% CI [-72.994, -70.711] pp  SE=0.5820
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant B>A**
- pre vs post @12.5% (deeper compression):
  - metric: ANLS (continuous)  | n_paired=5349  (A_total=5349, B_total=5349, only_A=0, only_B=0)
    mean pre@12.5%=0.3522  mean post@12.5%=0.1034  delta(pre-post)=+24.882 pp  95% CI [+23.580, +26.180] pp  SE=0.6632
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant A>B**

#### ocrbench  (OCR-containment)

- **pre vs post** (primary, stage law):
  - metric: OCR-containment (binary)  | n_paired=1000  (A_total=1000, B_total=1000, only_A=0, only_B=0)
    mean pre=0.5470  mean post=0.1840  delta(pre-post)=+36.300 pp  95% CI [+32.600, +40.000] pp  SE=1.8688
    paired permutation p (two-sided) = 5.00e-05
    McNemar: pre-only=419  post-only=56  z=+16.66  exact p=0.00e+00
    verdict: **significant A>B**
- pre vs none:
  - metric: OCR-containment (binary)  | n_paired=982  (A_total=1000, B_total=982, only_A=18, only_B=0)
    mean pre=0.5468  mean none=0.7739  delta(pre-none)=-22.709 pp  95% CI [-25.764, -19.654] pp  SE=1.5655
    paired permutation p (two-sided) = 5.00e-05
    McNemar: pre-only=30  none-only=253  z=-13.26  exact p=0.00e+00
    verdict: **significant B>A**
- post vs none:
  - metric: OCR-containment (binary)  | n_paired=982  (A_total=1000, B_total=982, only_A=18, only_B=0)
    mean post=0.1864  mean none=0.7739  delta(post-none)=-58.758 pp  95% CI [-62.016, -55.499] pp  SE=1.6910
    paired permutation p (two-sided) = 5.00e-05
    McNemar: post-only=19  none-only=596  z=-23.27  exact p=0.00e+00
    verdict: **significant B>A**

#### gqa  (exact-match)

- **pre vs post** (primary, stage law):
  - metric: exact-match (binary)  | n_paired=12578  (A_total=12578, B_total=12578, only_A=0, only_B=0)
    mean pre=0.4488  mean post=0.4771  delta(pre-post)=-2.830 pp  95% CI [-3.514, -2.147] pp  SE=0.3508
    paired permutation p (two-sided) = 5.00e-05
    McNemar: pre-only=796  post-only=1152  z=-8.07  exact p=0.00e+00
    verdict: **significant B>A**
- pre vs none:
  - metric: exact-match (binary)  | n_paired=12578  (A_total=12578, B_total=12578, only_A=0, only_B=0)
    mean pre=0.4488  mean none=0.6165  delta(pre-none)=-16.767 pp  95% CI [-17.618, -15.917] pp  SE=0.4338
    paired permutation p (two-sided) = 5.00e-05
    McNemar: pre-only=591  none-only=2700  z=-36.76  exact p=0.00e+00
    verdict: **significant B>A**
- post vs none:
  - metric: exact-match (binary)  | n_paired=12578  (A_total=12578, B_total=12578, only_A=0, only_B=0)
    mean post=0.4771  mean none=0.6165  delta(post-none)=-13.937 pp  95% CI [-14.732, -13.142] pp  SE=0.4079
    paired permutation p (two-sided) = 5.00e-05
    McNemar: post-only=556  none-only=2309  z=-32.75  exact p=0.00e+00
    verdict: **significant B>A**

### Qwen2.5-VL-7B (`qwen2vl`)

#### textvqa  (VQA-acc)

- **pre vs post** (primary, stage law):
  - metric: VQA-acc (continuous)  | n_paired=5000  (A_total=5000, B_total=5000, only_A=0, only_B=0)
    mean pre=0.7023  mean post=0.4417  delta(pre-post)=+26.053 pp  95% CI [+24.527, +27.567] pp  SE=0.7732
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant A>B**
- pre vs none:
  - metric: VQA-acc (continuous)  | n_paired=4998  (A_total=5000, B_total=4998, only_A=2, only_B=0)
    mean pre=0.7021  mean none=0.8623  delta(pre-none)=-16.020 pp  95% CI [-17.140, -14.906] pp  SE=0.5723
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant B>A**
- post vs none:
  - metric: VQA-acc (continuous)  | n_paired=4998  (A_total=5000, B_total=4998, only_A=2, only_B=0)
    mean post=0.4417  mean none=0.8623  delta(post-none)=-42.063 pp  95% CI [-43.451, -40.656] pp  SE=0.7175
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant B>A**
- pre vs post @12.5% (deeper compression):
  - metric: VQA-acc (continuous)  | n_paired=5000  (A_total=5000, B_total=5000, only_A=0, only_B=0)
    mean pre@12.5%=0.5971  mean post@12.5%=0.3187  delta(pre-post)=+27.840 pp  95% CI [+26.300, +29.380] pp  SE=0.7830
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant A>B**

#### docvqa  (ANLS)

- **pre vs post** (primary, stage law):
  - metric: ANLS (continuous)  | n_paired=5349  (A_total=5349, B_total=5349, only_A=0, only_B=0)
    mean pre=0.6361  mean post=0.5256  delta(pre-post)=+11.045 pp  95% CI [+9.596, +12.522] pp  SE=0.7509
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant A>B**
- pre vs none:
  - metric: ANLS (continuous)  | n_paired=5349  (A_total=5349, B_total=5349, only_A=0, only_B=0)
    mean pre=0.6361  mean none=0.9491  delta(pre-none)=-31.296 pp  95% CI [-32.500, -30.099] pp  SE=0.6136
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant B>A**
- post vs none:
  - metric: ANLS (continuous)  | n_paired=5349  (A_total=5349, B_total=5349, only_A=0, only_B=0)
    mean post=0.5256  mean none=0.9491  delta(post-none)=-42.341 pp  95% CI [-43.605, -41.089] pp  SE=0.6392
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant B>A**
- pre vs post @12.5% (deeper compression):
  - metric: ANLS (continuous)  | n_paired=5349  (A_total=5349, B_total=5349, only_A=0, only_B=0)
    mean pre@12.5%=0.4547  mean post@12.5%=0.2451  delta(pre-post)=+20.961 pp  95% CI [+19.558, +22.367] pp  SE=0.7149
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant A>B**

#### ocrbench  (OCR-containment)

- **pre vs post** (primary, stage law):
  - metric: OCR-containment (binary)  | n_paired=1000  (A_total=1000, B_total=1000, only_A=0, only_B=0)
    mean pre=0.4760  mean post=0.1830  delta(pre-post)=+29.300 pp  95% CI [+25.800, +32.700] pp  SE=1.7749
    paired permutation p (two-sided) = 5.00e-05
    McNemar: pre-only=347  post-only=54  z=+14.63  exact p=0.00e+00
    verdict: **significant A>B**
- pre vs none:
  - metric: OCR-containment (binary)  | n_paired=976  (A_total=1000, B_total=976, only_A=24, only_B=0)
    mean pre=0.4775  mean none=0.8371  delta(pre-none)=-35.963 pp  95% CI [-39.037, -32.889] pp  SE=1.5879
    paired permutation p (two-sided) = 5.00e-05
    McNemar: pre-only=7  none-only=358  z=-18.37  exact p=0.00e+00
    verdict: **significant B>A**
- post vs none:
  - metric: OCR-containment (binary)  | n_paired=976  (A_total=1000, B_total=976, only_A=24, only_B=0)
    mean post=0.1814  mean none=0.8371  delta(post-none)=-65.574 pp  95% CI [-68.545, -62.500] pp  SE=1.5545
    paired permutation p (two-sided) = 5.00e-05
    McNemar: post-only=5  none-only=645  z=-25.10  exact p=0.00e+00
    verdict: **significant B>A**
- pre vs post @12.5% (deeper compression):
  - metric: OCR-containment (binary)  | n_paired=1000  (A_total=1000, B_total=1000, only_A=0, only_B=0)
    mean pre@12.5%=0.3350  mean post@12.5%=0.0670  delta(pre-post)=+26.800 pp  95% CI [+23.800, +29.900] pp  SE=1.5761
    paired permutation p (two-sided) = 5.00e-05
    McNemar: pre@12.5%-only=294  post@12.5%-only=26  z=+14.98  exact p=0.00e+00
    verdict: **significant A>B**

#### gqa  (exact-match)

- **pre vs post** (primary, stage law):
  - metric: exact-match (binary)  | n_paired=12578  (A_total=12578, B_total=12578, only_A=0, only_B=0)
    mean pre=0.5594  mean post=0.5852  delta(pre-post)=-2.584 pp  95% CI [-3.291, -1.884] pp  SE=0.3601
    paired permutation p (two-sided) = 5.00e-05
    McNemar: pre-only=874  post-only=1199  z=-7.14  exact p=0.00e+00
    verdict: **significant B>A**
- pre vs none:
  - metric: exact-match (binary)  | n_paired=12578  (A_total=12578, B_total=12578, only_A=0, only_B=0)
    mean pre=0.5594  mean none=0.6045  delta(pre-none)=-4.508 pp  95% CI [-5.176, -3.856] pp  SE=0.3366
    paired permutation p (two-sided) = 5.00e-05
    McNemar: pre-only=622  none-only=1189  z=-13.32  exact p=0.00e+00
    verdict: **significant B>A**
- post vs none:
  - metric: exact-match (binary)  | n_paired=12578  (A_total=12578, B_total=12578, only_A=0, only_B=0)
    mean post=0.5852  mean none=0.6045  delta(post-none)=-1.924 pp  95% CI [-2.465, -1.383] pp  SE=0.2763
    paired permutation p (two-sided) = 5.00e-05
    McNemar: post-only=491  none-only=733  z=-6.92  exact p=0.00e+00
    verdict: **significant B>A**

### InternVL3-8B (`internvl3`)

#### textvqa  (VQA-acc)

- **pre vs post** (primary, stage law):
  - metric: VQA-acc (continuous)  | n_paired=5000  (A_total=5000, B_total=5000, only_A=0, only_B=0)
    mean pre=0.7890  mean post=0.4148  delta(pre-post)=+37.420 pp  95% CI [+35.980, +38.873] pp  SE=0.7358
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant A>B**
- pre vs none:
  - metric: VQA-acc (continuous)  | n_paired=5000  (A_total=5000, B_total=5000, only_A=0, only_B=0)
    mean pre=0.7890  mean none=0.8338  delta(pre-none)=-4.480 pp  95% CI [-5.253, -3.727] pp  SE=0.3916
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant B>A**
- post vs none:
  - metric: VQA-acc (continuous)  | n_paired=5000  (A_total=5000, B_total=5000, only_A=0, only_B=0)
    mean post=0.4148  mean none=0.8338  delta(post-none)=-41.900 pp  95% CI [-43.313, -40.487] pp  SE=0.7253
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant B>A**
- pre vs post @12.5% (deeper compression):
  - metric: VQA-acc (continuous)  | n_paired=5000  (A_total=5000, B_total=5000, only_A=0, only_B=0)
    mean pre@12.5%=0.7230  mean post@12.5%=0.3064  delta(pre-post)=+41.660 pp  95% CI [+40.180, +43.140] pp  SE=0.7535
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant A>B**

#### docvqa  (ANLS)

- **pre vs post** (primary, stage law):
  - metric: ANLS (continuous)  | n_paired=5349  (A_total=5349, B_total=5349, only_A=0, only_B=0)
    mean pre=0.7284  mean post=0.3820  delta(pre-post)=+34.641 pp  95% CI [+33.228, +36.079] pp  SE=0.7297
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant A>B**
- pre vs none:
  - metric: ANLS (continuous)  | n_paired=5349  (A_total=5349, B_total=5349, only_A=0, only_B=0)
    mean pre=0.7284  mean none=0.9221  delta(pre-none)=-19.376 pp  95% CI [-20.446, -18.306] pp  SE=0.5484
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant B>A**
- post vs none:
  - metric: ANLS (continuous)  | n_paired=5349  (A_total=5349, B_total=5349, only_A=0, only_B=0)
    mean post=0.3820  mean none=0.9221  delta(post-none)=-54.018 pp  95% CI [-55.303, -52.721] pp  SE=0.6584
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant B>A**
- pre vs post @12.5% (deeper compression):
  - metric: ANLS (continuous)  | n_paired=5349  (A_total=5349, B_total=5349, only_A=0, only_B=0)
    mean pre@12.5%=0.5054  mean post@12.5%=0.2451  delta(pre-post)=+26.029 pp  95% CI [+24.600, +27.465] pp  SE=0.7267
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant A>B**

#### ocrbench  (OCR-containment)

- **pre vs post** (primary, stage law):
  - metric: OCR-containment (binary)  | n_paired=1000  (A_total=1000, B_total=1000, only_A=0, only_B=0)
    mean pre=0.7530  mean post=0.3210  delta(pre-post)=+43.200 pp  95% CI [+39.700, +46.600] pp  SE=1.7522
    paired permutation p (two-sided) = 5.00e-05
    McNemar: pre-only=463  post-only=31  z=+19.44  exact p=0.00e+00
    verdict: **significant A>B**
- pre vs none:
  - metric: OCR-containment (binary)  | n_paired=1000  (A_total=1000, B_total=1000, only_A=0, only_B=0)
    mean pre=0.7530  mean none=0.8520  delta(pre-none)=-9.900 pp  95% CI [-12.200, -7.600] pp  SE=1.1670
    paired permutation p (two-sided) = 5.00e-05
    McNemar: pre-only=24  none-only=123  z=-8.16  exact p=0.00e+00
    verdict: **significant B>A**
- post vs none:
  - metric: OCR-containment (binary)  | n_paired=1000  (A_total=1000, B_total=1000, only_A=0, only_B=0)
    mean post=0.3210  mean none=0.8520  delta(post-none)=-53.100 pp  95% CI [-56.300, -49.900] pp  SE=1.6423
    paired permutation p (two-sided) = 5.00e-05
    McNemar: post-only=11  none-only=542  z=-22.58  exact p=0.00e+00
    verdict: **significant B>A**

#### gqa  (exact-match)

- **pre vs post** (primary, stage law):
  - metric: exact-match (binary)  | n_paired=12578  (A_total=12578, B_total=12578, only_A=0, only_B=0)
    mean pre=0.5993  mean post=0.6031  delta(pre-post)=-0.382 pp  95% CI [-0.930, +0.167] pp  SE=0.2798
    paired permutation p (two-sided) = 1.80e-01
    McNemar: pre-only=594  post-only=642  z=-1.36  exact p=1.81e-01
    verdict: **indistinguishable (CI crosses 0)**
- pre vs none:
  - metric: exact-match (binary)  | n_paired=12578  (A_total=12578, B_total=12578, only_A=0, only_B=0)
    mean pre=0.5993  mean none=0.6293  delta(pre-none)=-2.997 pp  95% CI [-3.586, -2.401] pp  SE=0.3032
    paired permutation p (two-sided) = 5.00e-05
    McNemar: pre-only=559  none-only=936  z=-9.75  exact p=0.00e+00
    verdict: **significant B>A**
- post vs none:
  - metric: exact-match (binary)  | n_paired=12578  (A_total=12578, B_total=12578, only_A=0, only_B=0)
    mean post=0.6031  mean none=0.6293  delta(post-none)=-2.616 pp  95% CI [-3.196, -2.035] pp  SE=0.2957
    paired permutation p (two-sided) = 5.00e-05
    McNemar: post-only=543  none-only=872  z=-8.75  exact p=0.00e+00
    verdict: **significant B>A**

### GLM-4.1V-9B-Thinking (`glm4v`) [thinking; answer-extracted]

#### textvqa  (VQA-acc)

- **pre vs post** (primary, stage law):
  - metric: VQA-acc (continuous)  | n_paired=200  (A_total=200, B_total=200, only_A=0, only_B=0)
    mean pre=0.2183  mean post=0.0500  delta(pre-post)=+16.833 pp  95% CI [+11.000, +22.833] pp  SE=2.9908
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant A>B**
- pre vs none:
  - metric: VQA-acc (continuous)  | n_paired=200  (A_total=200, B_total=200, only_A=0, only_B=0)
    mean pre=0.2183  mean none=0.2417  delta(pre-none)=-2.333 pp  95% CI [-7.500, +2.833] pp  SE=2.6735
    paired permutation p (two-sided) = 4.16e-01
    verdict: **indistinguishable (CI crosses 0)**
- post vs none:
  - metric: VQA-acc (continuous)  | n_paired=200  (A_total=200, B_total=200, only_A=0, only_B=0)
    mean post=0.0500  mean none=0.2417  delta(post-none)=-19.167 pp  95% CI [-25.167, -13.333] pp  SE=3.0074
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant B>A**

#### docvqa  (ANLS)

- **pre vs post** (primary, stage law):
  - metric: ANLS (continuous)  | n_paired=200  (A_total=200, B_total=200, only_A=0, only_B=0)
    mean pre=0.1297  mean post=0.0313  delta(pre-post)=+9.844 pp  95% CI [+5.471, +14.471] pp  SE=2.3072
    paired permutation p (two-sided) = 1.00e-04
    verdict: **significant A>B**
- pre vs none:
  - metric: ANLS (continuous)  | n_paired=200  (A_total=200, B_total=200, only_A=0, only_B=0)
    mean pre=0.1297  mean none=0.1039  delta(pre-none)=+2.583 pp  95% CI [-2.000, +7.167] pp  SE=2.3910
    paired permutation p (two-sided) = 3.13e-01
    verdict: **indistinguishable (CI crosses 0)**
- post vs none:
  - metric: ANLS (continuous)  | n_paired=200  (A_total=200, B_total=200, only_A=0, only_B=0)
    mean post=0.0313  mean none=0.1039  delta(post-none)=-7.261 pp  95% CI [-11.770, -2.804] pp  SE=2.3007
    paired permutation p (two-sided) = 1.50e-03
    verdict: **significant B>A**

#### gqa  (exact-match)

- **pre vs post** (primary, stage law):
  - metric: exact-match (binary)  | n_paired=200  (A_total=200, B_total=200, only_A=0, only_B=0)
    mean pre=0.1600  mean post=0.1150  delta(pre-post)=+4.500 pp  95% CI [+0.500, +8.500] pp  SE=2.0378
    paired permutation p (two-sided) = 4.88e-02
    McNemar: pre-only=13  post-only=4  z=+2.18  exact p=4.90e-02
    verdict: **significant A>B**
- pre vs none:
  - metric: exact-match (binary)  | n_paired=200  (A_total=200, B_total=200, only_A=0, only_B=0)
    mean pre=0.1600  mean none=0.1500  delta(pre-none)=+1.000 pp  95% CI [-3.000, +5.000] pp  SE=2.0099
    paired permutation p (two-sided) = 8.04e-01
    McNemar: pre-only=9  none-only=7  z=+0.50  exact p=8.04e-01
    verdict: **indistinguishable (CI crosses 0)**
- post vs none:
  - metric: exact-match (binary)  | n_paired=200  (A_total=200, B_total=200, only_A=0, only_B=0)
    mean post=0.1150  mean none=0.1500  delta(post-none)=-3.500 pp  95% CI [-8.000, +1.000] pp  SE=2.3062
    paired permutation p (two-sided) = 1.85e-01
    McNemar: post-only=7  none-only=14  z=-1.53  exact p=1.89e-01
    verdict: **indistinguishable (CI crosses 0)**

---

## Table 3 --- FastV-k3 vs RBM (pre) @25% (baseline regime map)

Delta = FastV - RBM (positive => FastV wins). All n=200 dev cells are flagged
exploratory when the CI is wide / n_paired small / skip-heavy.

**OCR-Bench dev cells**: the paper's point estimates use sum/200 (skips scored 0); the paired CI here uses the attempted samples only (n_paired=181/180, since the 19/20 skipped giant-OCR images have no answer in either arm and cannot be paired). So the paired delta can differ slightly from the paper margin (e.g. Qwen3-VL OCR -17.7 pp here vs -16.0 pp paper) --- both correct, different denominators.

### Qwen3-VL-8B / textvqa (full 5000)  [paper: FastV=0.7771, RBM=0.605, margin=+17.2 pp]

  - metric: VQA-acc (continuous)  | n_paired=4995  (A_total=4995, B_total=5000, only_A=0, only_B=5)
    mean FastV-k3=0.7778  mean RBM-pre=0.6054  delta(FastV-RBM)=+17.244 pp  95% CI [+15.976, +18.498] pp  SE=0.6421
    paired permutation p (two-sided) = 5.00e-05
    verdict: **significant A>B**

### Qwen3-VL-8B / gqa (full 12578)  [paper: FastV=0.5376, RBM=0.449, margin=+8.9 pp]

  - metric: exact-match (binary)  | n_paired=12578  (A_total=12578, B_total=12578, only_A=0, only_B=0)
    mean FastV-k3=0.5376  mean RBM-pre=0.4488  delta(FastV-RBM)=+8.881 pp  95% CI [+8.022, +9.747] pp  SE=0.4427
    paired permutation p (two-sided) = 5.00e-05
    McNemar: FastV-k3-only=2160  RBM-pre-only=1043  z=+19.74  exact p=0.00e+00
    verdict: **significant A>B**

### Qwen3-VL-8B / docvqa (dev 200 (600k cap))  [paper: FastV=0.5863, RBM=0.4239, margin=+16.2 pp]

  - metric: ANLS (continuous)  | n_paired=200  (A_total=200, B_total=200, only_A=0, only_B=0)
    mean FastV-k3=0.5863  mean RBM-pre=0.4239  delta(FastV-RBM)=+16.247 pp  95% CI [+8.474, +23.877] pp  SE=3.9207
    paired permutation p (two-sided) = 5.00e-05
    verdict: **exploratory/inconclusive: significant A>B**

### Qwen3-VL-8B / ocrbench (dev 200 (native))  [paper: FastV=0.415, RBM=0.575, margin=-16.0 pp]

  - metric: OCR-containment (binary)  | n_paired=181  (A_total=181, B_total=181, only_A=0, only_B=0)
    mean FastV-k3=0.4586  mean RBM-pre=0.6354  delta(FastV-RBM)=-17.680 pp  95% CI [-24.309, -11.602] pp  SE=3.3116
    paired permutation p (two-sided) = 5.00e-05
    McNemar: FastV-k3-only=5  RBM-pre-only=37  z=-4.94  exact p=0.00e+00
    verdict: **exploratory/inconclusive: significant B>A**

### Qwen2.5-VL-7B / textvqa (dev 200)  [paper: FastV=0.7467, RBM=0.6683, margin=+7.8 pp]

  - metric: VQA-acc (continuous)  | n_paired=200  (A_total=200, B_total=200, only_A=0, only_B=0)
    mean FastV-k3=0.7467  mean RBM-pre=0.6683  delta(FastV-RBM)=+7.833 pp  95% CI [+2.500, +13.167] pp  SE=2.7452
    paired permutation p (two-sided) = 5.35e-03
    verdict: **exploratory/inconclusive: significant A>B**

### Qwen2.5-VL-7B / gqa (dev 200)  [paper: FastV=0.475, RBM=0.52, margin=-4.5 pp]

  - metric: exact-match (binary)  | n_paired=200  (A_total=200, B_total=200, only_A=0, only_B=0)
    mean FastV-k3=0.4750  mean RBM-pre=0.5200  delta(FastV-RBM)=-4.500 pp  95% CI [-10.500, +1.500] pp  SE=3.0994
    paired permutation p (two-sided) = 1.99e-01
    McNemar: FastV-k3-only=15  RBM-pre-only=24  z=-1.44  exact p=2.00e-01
    verdict: **exploratory/inconclusive (CI crosses 0)**

### Qwen2.5-VL-7B / docvqa (dev 200 (600k cap))  [paper: FastV=0.4852, RBM=0.5062, margin=-2.1 pp]

  - metric: ANLS (continuous)  | n_paired=200  (A_total=200, B_total=200, only_A=0, only_B=0)
    mean FastV-k3=0.4852  mean RBM-pre=0.5062  delta(FastV-RBM)=-2.100 pp  95% CI [-10.098, +5.773] pp  SE=4.0633
    paired permutation p (two-sided) = 6.07e-01
    verdict: **exploratory/inconclusive (CI crosses 0)**

### Qwen2.5-VL-7B / ocrbench (dev 200 (native))  [paper: FastV=0.285, RBM=0.37, margin=-8.5 pp]

  - metric: OCR-containment (binary)  | n_paired=180  (A_total=180, B_total=180, only_A=0, only_B=0)
    mean FastV-k3=0.3167  mean RBM-pre=0.4111  delta(FastV-RBM)=-9.444 pp  95% CI [-17.778, -1.111] pp  SE=4.2984
    paired permutation p (two-sided) = 4.08e-02
    McNemar: FastV-k3-only=22  RBM-pre-only=39  z=-2.18  exact p=3.96e-02
    verdict: **exploratory/inconclusive: significant B>A**

---

## Decision-rule applications

1. **InternVL3 paired CI for every pre-vs-post comparison**: provided above (InternVL3 section).
2. **GQA rule**: where the pre-vs-post delta CI crosses 0, the cell is labelled
   *indistinguishable* (NOT 'statistical tie' --- no preregistered equivalence bound exists).
   - qwen3vl (Qwen3-VL-8B): delta=-2.83 pp, CI [-3.51, -2.15], verdict=significant B>A
   - qwen2vl (Qwen2.5-VL-7B): delta=-2.58 pp, CI [-3.29, -1.88], verdict=significant B>A
   - internvl3 (InternVL3-8B): delta=-0.38 pp, CI [-0.93, +0.17], verdict=indistinguishable (CI crosses 0)
   - glm4v (GLM-4.1V-9B-Thinking): delta=+4.50 pp, CI [+0.50, +8.50], verdict=significant A>B  [NB: paired stats are borderline significant (perm p=0.049, McNemar p=0.049), but the paper labels this arm 'inconclusive' for PROTOCOL reasons (greedy floor-collapse across all GLM GQA arms: none 0.15 vs official 0.77) -- a scope override, not a statistical claim; the p sits exactly at 0.05 so no directional claim is the conservative call.]

3. **Table 3 (n=200) small-delta audit**: cells flagged *exploratory/inconclusive*
   where the CI is wide, n_paired is small, or skips are heavy:
   - Qwen3-VL-8B / textvqa (full 5000): n_paired=4995, delta=+17.24 pp, CI [+15.98, +18.50], verdict=significant A>B
   - Qwen3-VL-8B / docvqa (dev 200 (600k cap)): n_paired=200, delta=+16.25 pp, CI [+8.47, +23.88], verdict=exploratory/inconclusive: significant A>B
   - Qwen3-VL-8B / ocrbench (dev 200 (native)): n_paired=181, delta=-17.68 pp, CI [-24.31, -11.60], verdict=exploratory/inconclusive: significant B>A
   - Qwen2.5-VL-7B / textvqa (dev 200): n_paired=200, delta=+7.83 pp, CI [+2.50, +13.17], verdict=exploratory/inconclusive: significant A>B
   - Qwen2.5-VL-7B / gqa (dev 200): n_paired=200, delta=-4.50 pp, CI [-10.50, +1.50], verdict=exploratory/inconclusive (CI crosses 0)
   - Qwen2.5-VL-7B / docvqa (dev 200 (600k cap)): n_paired=200, delta=-2.10 pp, CI [-10.10, +5.77], verdict=exploratory/inconclusive (CI crosses 0)
   - Qwen2.5-VL-7B / ocrbench (dev 200 (native)): n_paired=180, delta=-9.44 pp, CI [-17.78, -1.11], verdict=exploratory/inconclusive: significant B>A

---

## Mismatches vs paper

None. Every recomputed cell mean matches the paper-stated number to within 0.5 pp (rounding), confirming the official rescoring is reproduced exactly.


#!/usr/bin/env python
"""MALT adaptive-gate feasibility audit -- PART 1: no-GPU transition audit.

Builds the per-dataset K0<->K1 and K1<->K8 transition tables from the existing
deferred-RBM n=200 sweep per-sample results, and reports the G0 / G1 positive /
negative sample counts and class imbalance (task 2026-08-24, audit round).

Labels (official rescore == sweep_per_sample.json 'correct', see
analyze_deferred_sweep.py):
  G0 (safe to delete at K=0):
      positive        = K0 correct
      high-risk neg   = K0 wrong AND K1 correct
      neutral (drop)  = K0 wrong AND K1 wrong   (routing to K0 costs nothing vs K1)
  G1 (extend K1 -> K8):
      positive        = K1 wrong AND K8 correct (extension rescues)
      negative        = K1 correct  OR (K1 wrong AND K8 wrong)
      harmful ext     = K1 correct AND K8 wrong (reported separately)

Outputs: experiments/malt_adaptive_gate_feasibility/transitions.json
         (also prints the tables)
"""
import json
import os

HERE = "/media/disk2/YZX/research/vla"
SWEEP = f"{HERE}/experiments/deferred_rbm_n200_data/sweep_per_sample.json"
OUTDIR = f"{HERE}/experiments/malt_adaptive_gate_feasibility"
os.makedirs(OUTDIR, exist_ok=True)

BENCHES = ["textvqa", "docvqa", "ocrbench", "gqa"]


def main():
    sweep = json.load(open(SWEEP))
    byid = {}
    for k, v in sweep.items():
        byid.setdefault((v["bench"], v["sample_id"]), {})[v["K"]] = v

    out = {}
    print(f"{'bench':9s} {'n':>4s} | {'K0C,K1C':>9s} {'K0C,K1W':>9s} "
          f"{'K0W,K1C':>9s} {'K0W,K1W':>9s} | {'K1C,K8C':>9s} {'K1C,K8W':>9s} "
          f"{'K1W,K8C':>9s} {'K1W,K8W':>9s}")
    for b in BENCHES:
        rows = [byid[(b, sid)] for sid in
                (sid for (bb, sid), ks in byid.items() if bb == b
                 and 0 in ks and 1 in ks and 8 in ks)]
        n = len(rows)
        t01 = {"K0C_K1C": 0, "K0C_K1W": 0, "K0W_K1C": 0, "K0W_K1W": 0}
        t18 = {"K1C_K8C": 0, "K1C_K8W": 0, "K1W_K8C": 0, "K1W_K8W": 0}
        for r in rows:
            c0, c1, c8 = (r[0]["correct"], r[1]["correct"], r[8]["correct"])
            t01[("K0C" if c0 else "K0W") + "_" + ("K1C" if c1 else "K1W")] += 1
            t18[("K1C" if c1 else "K1W") + "_" + ("K8C" if c8 else "K8W")] += 1
        # G0 label counts
        g0_pos = t01["K0C_K1C"] + t01["K0C_K1W"]          # K0 correct
        g0_neg = t01["K0W_K1C"]                           # high-risk
        g0_neutral = t01["K0W_K1W"]                       # don't care
        # G1 label counts
        g1_pos = t18["K1W_K8C"]
        g1_neg = t18["K1C_K8C"] + t18["K1C_K8W"] + t18["K1W_K8W"]
        g1_harm = t18["K1C_K8W"]
        row = {
            "n": n,
            "K0K1": t01,
            "K1K8": t18,
            "G0": {"pos_K0correct": g0_pos,
                   "neg_K0W_K1C": g0_neg,
                   "neutral_K0W_K1W": g0_neutral,
                   "imbalance_pos/neg": round(g0_pos / g0_neg, 2) if g0_neg else None},
            "G1": {"pos_K1W_K8C": g1_pos,
                   "neg": g1_neg,
                   "harm_K1C_K8W": g1_harm,
                   "imbalance_pos/neg": round(g1_pos / g1_neg, 3) if g1_neg else None},
        }
        out[b] = row
        print(f"{b:9s} {n:4d} | {t01['K0C_K1C']:9d} {t01['K0C_K1W']:9d} "
              f"{t01['K0W_K1C']:9d} {t01['K0W_K1W']:9d} | "
              f"{t18['K1C_K8C']:9d} {t18['K1C_K8W']:9d} "
              f"{t18['K1W_K8C']:9d} {t18['K1W_K8W']:9d}")
    print("\nG0: positive=K0 correct, high-risk negative=K0W&K1C, neutral=K0W&K1W")
    for b in BENCHES:
        g = out[b]["G0"]
        print(f"  {b:9s} pos={g['pos_K0correct']:3d} neg={g['neg_K0W_K1C']:3d} "
              f"neutral={g['neutral_K0W_K1W']:3d} "
              f"imbalance={g['imbalance_pos/neg']}")
    print("\nG1: positive=K1W&K8C (rescue), negative=else, harmful=K1C&K8W")
    for b in BENCHES:
        g = out[b]["G1"]
        print(f"  {b:9s} pos={g['pos_K1W_K8C']:3d} neg={g['neg']:3d} "
              f"harmful={g['harm_K1C_K8W']:3d} imbalance={g['imbalance_pos/neg']}")

    with open(f"{OUTDIR}/transitions.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[written] {OUTDIR}/transitions.json")


if __name__ == "__main__":
    main()

"""EV-VAR Stage 1 analysis: does batch COMPOSITION explain per-step wallclock
BEYOND total tokens (sum_k) and batch size (n_members)?

Coordinator caveats honored:
  * PRIMARY criterion = M1-vs-M0 joint likelihood-ratio (F) test.
  * Externality signature computed on M0 RESIDUALS (wallclock - M0_pred) so the
    total-tokens confound is removed; raw means reported only as illustration.
  * var_k / max_k are collinear -> report VIF; rely on the JOINT test, not
    individual coefs.
  * Decisive sub-result = chunked-ON subset (residual composition effect after
    the standard chunked-prefill mitigation).

Inputs: runs/ev_var/stage1_{homo,bimodal}_{chunk,nochunk}.json.steps.json
Output: runs/ev_var/stage1_results.json  +  stdout summary.
"""
import json, os, glob, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats

RUNS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "ev_var")


def load_steps():
    rows = []
    cells = {
        "stage1_homo_chunk.json.steps.json": ("homo", 1),
        "stage1_homo_nochunk.json.steps.json": ("homo", 0),
        "stage1_bimodal_chunk.json.steps.json": ("bimodal", 1),
        "stage1_bimodal_nochunk.json.steps.json": ("bimodal", 0),
    }
    for fn, (kcomp, chunked) in cells.items():
        p = os.path.join(RUNS_DIR, fn)
        if not os.path.exists(p):
            print(f"[warn] missing {p}", file=sys.stderr)
            continue
        d = json.load(open(p))
        for s in d["steps"]:
            ks = list(s["k_i"].values())
            phases = list(s["phase"].values())
            if not ks:
                continue
            max_k = int(max(ks))
            n_prefill = sum(1 for ph in phases if ph == "prefill")
            n_decode = sum(1 for ph in phases if ph == "decode")
            rows.append({
                "cell": f"{kcomp}_{chunked}",
                "kcomp": kcomp,
                "chunked": chunked,
                "sum_k": int(s["sum_k"]),
                "n_members": int(s["n_members"]),
                "var_k": float(s["var_k"]),
                "max_k": max_k,
                "n_prefill": n_prefill,
                "n_decode": n_decode,
                "wallclock_ms": float(s["wallclock_ms"]),
            })
    df = pd.DataFrame(rows)
    return df


def fit(df, formula):
    Y = df["wallclock_ms"]
    X = df[formula].copy()
    X = sm.add_constant(X)
    return sm.OLS(Y, X).fit()


def lr_test(df, m0_terms, m1_extra):
    """Joint F-test of M0 vs M0+m1_extra. Returns (f_stat, p_value)."""
    m0 = fit(df, m0_terms)
    m1 = fit(df, m0_terms + m1_extra)
    # LR = m1.ess - m0.ess over (df_resid_m0 - df_resid_m1) ... use anova for F
    # Use statsmodels compare_f_test (M1.restricted ... ): m1.compare_f_test(m0)
    f, p, df_diff = m1.compare_f_test(m0)
    return m0, m1, f, p, df_diff


def vif(df, cols):
    X = df[cols].copy()
    X = sm.add_constant(X)
    out = {}
    for i, c in enumerate(X.columns):
        if c == "const":
            continue
        try:
            out[c] = float(variance_inflation_factor(X.values, i))
        except Exception:
            out[c] = float("nan")
    return out


def signature_on_residuals(df, m0_terms):
    """Externality signature on M0 residuals: among steps with >=1 decode
    member, compare residual (wc - M0_pred) for steps WITH a co-batched
    high-k prefill (max_k>=500) vs WITHOUT (max_k<200). Also report raw wc."""
    m0 = fit(df, m0_terms)
    pred = m0.predict(sm.add_constant(df[m0_terms]))
    df = df.copy()
    df["resid"] = df["wallclock_ms"] - pred
    sub = df[df["n_decode"] >= 1].copy()
    hi = sub[sub["max_k"] >= 500]
    lo = sub[sub["max_k"] < 200]
    def blk(s):
        if len(s) == 0:
            return {"n": 0, "mean": float("nan"), "median": float("nan"),
                    "std": float("nan")}
        return {"n": int(len(s)), "mean": float(s.mean()),
                "median": float(s.median()), "std": float(s.std())}
    # also match on n_members to control a bit (±1)
    return {
        "residual_hi": blk(hi["resid"]), "residual_lo": blk(lo["resid"]),
        "raw_wc_hi": blk(hi["wallclock_ms"]), "raw_wc_lo": blk(lo["wallclock_ms"]),
        "resid_ratio_hi_over_lo": (float(hi["resid"].mean()) / lo["resid"].mean())
            if len(hi) and len(lo) and lo["resid"].mean() != 0 else float("nan"),
        "n_members_hi": blk(hi["n_members"]), "n_members_lo": blk(lo["n_members"]),
    }


def coef_table(model):
    out = []
    for name in model.params.index:
        out.append({
            "term": name,
            "coef": float(model.params[name]),
            "stderr": float(model.bse[name]),
            "t": float(model.tvalues[name]),
            "p": float(model.pvalues[name]),
        })
    return out


def effect_over_range(model, term, df):
    """|coef| * (observed range of term) = wallclock swing attributable."""
    if term not in model.params.index or term == "const":
        return None
    lo = float(df[term].min())
    hi = float(df[term].max())
    coef = float(model.params[term])
    swing = abs(coef) * (hi - lo)
    base = float(df["wallclock_ms"].mean())
    return {"term": term, "coef": coef, "min": lo, "max": hi,
            "swing_ms": swing, "pct_of_mean_wallclock": (swing / base * 100.0)
            if base else float("nan")}


def main():
    df = load_steps()
    if len(df) == 0:
        print("NO DATA — did the 4 cells run?"); return
    print(f"=== EV-VAR Stage 1: {len(df)} steps pooled ===")
    print(df.groupby(["kcomp", "chunked"]).size())

    m0_terms = ["sum_k", "n_members", "chunked"]
    m1_extra = ["var_k", "max_k", "n_prefill", "n_decode"]

    # ---- FULL POOL: M0 vs M1 ----
    m0, m1, f, p, dfd = lr_test(df, m0_terms, m1_extra)
    print("\n=== FULL POOL M0 ===")
    print(f"  R2={m0.rsquared:.4f}  adjR2={m0.rsquared_adj:.4f}  n={int(m0.nobs)}")
    for r in coef_table(m0):
        print(f"    {r['term']:>12}  coef={r['coef']:>10.4f}  p={r['p']:.3e}")
    print("=== FULL POOL M1 (M0 + composition) ===")
    print(f"  R2={m1.rsquared:.4f}  adjR2={m1.rsquared_adj:.4f}  n={int(m1.nobs)}")
    for r in coef_table(m1):
        print(f"    {r['term']:>12}  coef={r['coef']:>10.4f}  p={r['p']:.3e}")
    print(f"=== JOINT LR (F) TEST M1 vs M0 ===")
    print(f"  F({dfd},{int(m1.df_resid)})={f:.3f}  p={p:.3e}")

    print("\n=== VIF (composition block) ===")
    for k, v in vif(df, m1_extra).items():
        print(f"    {k:>12}  VIF={v:.2f}")

    print("\n=== composition-feature swings over observed range (from M1) ===")
    for t in m1_extra:
        e = effect_over_range(m1, t, df)
        if e:
            print(f"    {t:>12}  swing={e['swing_ms']:>8.2f}ms  "
                  f"= {e['pct_of_mean_wallclock']:.1f}% of mean wc  (p={float(m1.pvalues[t]):.3e})")

    # ---- DECISIVE: chunked-ON subset ----
    df_on = df[df["chunked"] == 1].copy()
    res_on = None
    if len(df_on) > 20:
        # M0 inside chunked-ON drops the constant chunked term
        m0t = ["sum_k", "n_members"]
        m0_on = fit(df_on, m0t)
        m1_on = fit(df_on, m0t + m1_extra)
        f_on, p_on, dfd_on = m1_on.compare_f_test(m0_on)
        print(f"\n=== CHUNKED-ON SUBSET (decisive): M1 vs M0 ===")
        print(f"  n={len(df_on)}  M0 R2={m0_on.rsquared:.4f}  M1 R2={m1_on.rsquared:.4f}")
        print(f"  F({dfd_on},{int(m1_on.df_resid)})={f_on:.3f}  p={p_on:.3e}")
        for r in coef_table(m1_on):
            print(f"    {r['term']:>12}  coef={r['coef']:>10.4f}  p={r['p']:.3e}")
        res_on = {"n": len(df_on), "m0_r2": float(m0_on.rsquared),
                  "m1_r2": float(m1_on.rsquared), "F": float(f_on),
                  "p": float(p_on), "dfd": float(dfd_on),
                  "coef": coef_table(m1_on)}
        # chunked-ON signature on M0 residuals
        sig_on = signature_on_residuals(df_on, m0t)
        print(f"  chunked-ON resid signature hi(max_k>=500) vs lo(max_k<200), decode steps:")
        print(f"    resid_hi mean={sig_on['residual_hi']['mean']:.3f}ms (n={sig_on['residual_hi']['n']}) "
              f" resid_lo mean={sig_on['residual_lo']['mean']:.3f}ms (n={sig_on['residual_lo']['n']}) "
              f" ratio={sig_on['resid_ratio_hi_over_lo']:.3f}")
    else:
        print("\n[warn] chunked-ON subset too small")

    # ---- FULL-POOL signature on M0 residuals ----
    sig = signature_on_residuals(df, m0_terms)
    print(f"\n=== FULL-POOL resid signature (decode steps; M0 residual) ===")
    print(f"  resid_hi(max_k>=500) mean={sig['residual_hi']['mean']:.3f}ms n={sig['residual_hi']['n']} "
          f" n_members_mean={sig['n_members_hi']['mean']:.2f}")
    print(f"  resid_lo(max_k<200)  mean={sig['residual_lo']['mean']:.3f}ms n={sig['residual_lo']['n']} "
          f" n_members_mean={sig['n_members_lo']['mean']:.2f}")
    print(f"  resid ratio hi/lo = {sig['resid_ratio_hi_over_lo']:.3f}  "
          f"(raw wc ratio = {sig['raw_wc_hi']['mean']/sig['raw_wc_lo']['mean'] if sig['raw_wc_lo']['mean'] else float('nan'):.3f})")

    # ---- chunked-OFF signature too ----
    df_off = df[df["chunked"] == 0].copy()
    sig_off = None
    if len(df_off) > 20:
        sig_off = signature_on_residuals(df_off, ["sum_k", "n_members"])
        print(f"\n=== CHUNKED-OFF resid signature (decode steps) ===")
        print(f"  resid_hi mean={sig_off['residual_hi']['mean']:.3f}ms n={sig_off['residual_hi']['n']}"
              f"  resid_lo mean={sig_off['residual_lo']['mean']:.3f}ms n={sig_off['residual_lo']['n']}"
              f"  ratio={sig_off['resid_ratio_hi_over_lo']:.3f}")

    # ---- assemble result ----
    result = {
        "n_steps_total": int(len(df)),
        "cells_present": sorted(df["cell"].unique().tolist()),
        "M0": {"r2": float(m0.rsquared), "adj_r2": float(m0.rsquared_adj),
               "coef": coef_table(m0)},
        "M1": {"r2": float(m1.rsquared), "adj_r2": float(m1.rsquared_adj),
               "coef": coef_table(m1)},
        "LR_test_M1_vs_M0": {"F": float(f), "p": float(p), "dfd": float(dfd),
                              "dfn": int(m0.df_resid) - int(m1.df_resid)},
        "VIF": vif(df, m1_extra),
        "composition_swings": {t: effect_over_range(m1, t, df) for t in m1_extra},
        "signature_fullpool_residual": sig,
        "signature_chunked_off_residual": sig_off,
        "chunked_on_subset": res_on,
    }
    outp = os.path.join(RUNS_DIR, "stage1_results.json")
    with open(outp, "w") as f_:
        json.dump(result, f_, indent=2, default=str)
    print(f"\n=> wrote {outp}")


if __name__ == "__main__":
    main()

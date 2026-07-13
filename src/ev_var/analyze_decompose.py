"""Decompose the Stage-1 LR signal: is it k-COMPOSITION (var_k, max_k -- the
externality) or merely PHASE-MIX (n_prefill, n_decode -- trivial work volume)?

Also: clean iso-composition signature -- BIMODAL vs HOMO decode-step M0-residual
at iso n_members (the controlled contrast the design was built for)."""
import json, os
import numpy as np, pandas as pd
import statsmodels.api as sm

RUNS_DIR = "runs/ev_var"
cells = {
 "stage1_homo_chunk.json.steps.json": ("homo", 1),
 "stage1_homo_nochunk.json.steps.json": ("homo", 0),
 "stage1_bimodal_chunk.json.steps.json": ("bimodal", 1),
 "stage1_bimodal_nochunk.json.steps.json": ("bimodal", 0),
}
rows = []
for fn,(kc,ch) in cells.items():
    d = json.load(open(os.path.join(RUNS_DIR, fn)))
    for s in d["steps"]:
        ks = list(s["k_i"].values()); ph = list(s["phase"].values())
        if not ks: continue
        rows.append(dict(kcomp=kc, chunked=ch, sum_k=s["sum_k"],
            n_members=s["n_members"], var_k=s["var_k"], max_k=max(ks),
            n_prefill=sum(1 for p in ph if p=="prefill"),
            n_decode=sum(1 for p in ph if p=="decode"),
            wallclock_ms=s["wallclock_ms"]))
df = pd.DataFrame(rows)

def fit(cols):
    X = sm.add_constant(df[cols]); return sm.OLS(df["wallclock_ms"], X).fit()

m0 = fit(["sum_k","n_members","chunked"])
# k-composition only
mk = fit(["sum_k","n_members","chunked","var_k","max_k"])
# phase only
mp = fit(["sum_k","n_members","chunked","n_prefill","n_decode"])

print("=== LR: +{var_k,max_k} only (the k-composition externality) ===")
f,p,dd = mk.compare_f_test(m0)
print(f"  F({dd},{int(mk.df_resid)})={f:.3f}  p={p:.3e}   "
      f"(var_k p={mk.pvalues['var_k']:.3f}, max_k p={mk.pvalues['max_k']:.3f}, "
      f"max_k coef={mk.params['max_k']:.4f})")
print("=== LR: +{n_prefill,n_decode} only (phase-mix / work-volume) ===")
f2,p2,dd2 = mp.compare_f_test(m0)
print(f"  F({dd2},{int(mp.df_resid)})={f2:.3f}  p={p2:.3e}   "
      f"(n_prefill p={mp.pvalues['n_prefill']:.3e}, n_decode p={mp.pvalues['n_decode']:.3e})")

print("\n=== DECISIVE: chunked-ON, +{var_k,max_k} only ===")
df_on = df[df.chunked==1].copy()
def fit_on(cols):
    X = sm.add_constant(df_on[cols]); return sm.OLS(df_on["wallclock_ms"], X).fit()
m0on = fit_on(["sum_k","n_members"])
mkon = fit_on(["sum_k","n_members","var_k","max_k"])
f3,p3,dd3 = mkon.compare_f_test(m0on)
print(f"  n={len(df_on)} F({dd3},{int(mkon.df_resid)})={f3:.3f}  p={p3:.3e}  "
      f"(var_k p={mkon.pvalues['var_k']:.3f}, max_k p={mkon.pvalues['max_k']:.3f}, "
      f"max_k coef={mkon.params['max_k']:.4f})")

# Clean iso-composition signature: BIMODAL vs HOMO decode-step residual at iso n_members
print("\n=== iso-composition: BIMODAL vs HOMO decode-step M0-residual (matched n_members) ===")
for ch in (1,0):
    sub = df[(df.chunked==ch)&(df.n_decode>=1)].copy()
    m0s = sm.OLS(sub["wallclock_ms"], sm.add_constant(sub[["sum_k","n_members"]])).fit()
    sub["resid"] = sub["wallclock_ms"] - m0s.predict(sm.add_constant(sub[["sum_k","n_members"]]))
    # match on n_members within +-1
    ho = sub[sub.kcomp=="homo"]; bi = sub[sub.kcomp=="bimodal"]
    # bin by n_members
    import statistics
    print(f"  chunked={ch}: HOMO decode-step resid mean={ho['resid'].mean():.3f}ms (n={len(ho)}, n_mem mean={ho['n_members'].mean():.2f}) "
          f"sum_k mean={ho['sum_k'].mean():.0f}")
    print(f"  chunked={ch}: BIMO decode-step resid mean={bi['resid'].mean():.3f}ms (n={len(bi)}, n_mem mean={bi['n_members'].mean():.2f}) "
          f"sum_k mean={bi['sum_k'].mean():.0f}")
    # matched comparison: for each n_members bucket present in both
    nb = sorted(set(ho.n_members) & set(bi.n_members))
    dres = []
    for nm in nb:
        rh = ho[ho.n_members==nm]["resid"].mean()
        rb = bi[bi.n_members==nm]["resid"].mean()
        dres.append((nm, rh, rb, rb-rh, len(ho[ho.n_members==nm]), len(bi[bi.n_members==nm])))
    if dres:
        # weighted mean of (bimo-homo) residual diff
        wdh = sum((r[3]*r[4]) for r in dres)/sum(r[4] for r in dres)
        print(f"  chunked={ch}: matched n_members buckets={len(nb)}; "
              f"weighted (BIMO-HOMO) resid diff={wdh:+.3f}ms "
              f"(>0 = bimodal decode steps slower at iso sum_k+n_members)")

import json, os
import numpy as np, pandas as pd
import statsmodels.api as sm
cells={'stage1_homo_chunk.json.steps.json':('homo',1),'stage1_homo_nochunk.json.steps.json':('homo',0),'stage1_bimodal_chunk.json.steps.json':('bimodal',1),'stage1_bimodal_nochunk.json.steps.json':('bimodal',0)}
rows=[]
for fn,(kc,ch) in cells.items():
  for s in json.load(open(os.path.join('runs/ev_var',fn)))['steps']:
    ks=list(s['k_i'].values()); ph=list(s['phase'].values())
    if not ks: continue
    rows.append(dict(kcomp=kc,chunked=ch,sum_k=s['sum_k'],n_members=s['n_members'],
      var_k=s['var_k'],max_k=max(ks),
      n_prefill=sum(1 for x in ph if x=='prefill'),
      n_decode=sum(1 for x in ph if x=='decode'),
      wallclock_ms=s['wallclock_ms']))
df=pd.DataFrame(rows)
print('HIGH-CONCURRENCY subset (n_members>=12):')
sub=df[df.n_members>=12].copy()
print(f'  n={len(sub)} (bimodal={len(sub[sub.kcomp=="bimodal"])})')
m0=sm.OLS(sub['wallclock_ms'],sm.add_constant(sub[['sum_k','n_members','chunked']])).fit()
mk=sm.OLS(sub['wallclock_ms'],sm.add_constant(sub[['sum_k','n_members','chunked','var_k','max_k']])).fit()
f,p,dd=mk.compare_f_test(m0)
print(f'  LR +var_k,max_k: F({dd:.0f},{int(mk.df_resid)})={f:.3f} p={p:.3e}  var_k p={mk.pvalues["var_k"]:.3f} max_k p={mk.pvalues["max_k"]:.3f} max_k coef={mk.params["max_k"]:.4f}')
sub2=sub[sub.n_decode>=1].copy()
print(f'  decode-steps n={len(sub2)}: mean wc HOMO={sub2[sub2.kcomp=="homo"]["wallclock_ms"].mean():.1f}ms BIMO={sub2[sub2.kcomp=="bimodal"]["wallclock_ms"].mean():.1f}ms')

"""03 — Dinamica dei livelli: medie per sottoperiodo, break (scala), OU/ADF, stagionalità."""
import numpy as np, pandas as pd
from config import *
from utils import nw_mean, one_break, ou_fit, adf_p, deseasonalise, save_txt

lm = pd.read_csv(PROC/"lambdas_monthly.csv", index_col=0, parse_dates=True); lm.columns=lm.columns.astype(int)
L=[]; P=L.append
P("=== 03 DYNAMICS ===")
periods = {"FLL 04/07-09/11":(FLL_A[:7],FLL_B[:7]), "2010-2012":("2010-01","2012-12"),
           "2013-2019":("2013-01","2019-12"), "2020-2026":("2020-01","2026-12"),
           "post-2013":(POST13,"2026-12")}
P("\nsubperiod means [NW t]:")
for nm,(a,b) in periods.items():
    row=f"  {nm:<16}"
    for m in MATS:
        x=lm[m].loc[a:b]
        row += (f"{nw_mean(x)[0]:7.1f}[{nw_mean(x)[1]:5.1f}]" if x.notna().sum()>12 else f"{'--':>14}")
    P(row)
P("\nbreaks (1st + 2nd sul segmento lungo):")
for m in MATS:
    x=lm[m].dropna()
    if len(x)<60: continue
    d1,a1,b1,F1,k = one_break(x)
    seg = x[:k] if k>=len(x)-k else x[k:]
    if len(seg)>=60:
        d2,a2,b2,F2,_ = one_break(seg)
        P(f"  {m:>2}Y: {d1.date()} ({a1:5.1f}->{b1:5.1f}, F={F1:6.1f}); 2nd {d2.date()} ({a2:5.1f}->{b2:5.1f}, F={F2:6.1f})")
P("\nOU/ADF post-2013 + stagionalità (dummies post-2010):")
for m in [2,5,10,30]:
    x=lm[m].loc[POST13:]
    k,th,s,hl = ou_fit(x)
    line=f"  {m:>2}Y: ADF p={adf_p(x):5.3f}  HL={hl:4.1f}m  theta={th:5.1f}"
    if m in (2,5,10):
        des,amp,fp = deseasonalise(lm[m].loc['2010-01':])
        line += f"  | seas amp={amp:4.1f} (F p={fp:.3f})  ADF deseas(post13)={adf_p(des.loc[POST13:]):5.3f}"
    P(line)
des2,_,_ = deseasonalise(lm[2].loc['2010-01':])
post = des2.loc['2024-11':]
P(f"\n(U7) 2Y deseasonalised mean post-2024-10: {post.mean():.1f} bp (n={len(post)}) -> PROVISIONAL")
save_txt("03_dynamics.txt", L); print("\n".join(L))

"""02 — Validazione tripla della costruzione: breakeven, CUSIP, benchmark FLL."""
import numpy as np, pandas as pd
from config import *
from utils import nw_mean, save_txt

lam  = pd.read_csv(PROC/"lambdas_daily.csv", index_col=0, parse_dates=True)
lamB = pd.read_csv(PROC/"lambdas_be_daily.csv", index_col=0, parse_dates=True)
st   = pd.read_csv(PROC/"cusip_stats_daily.csv", index_col=0, parse_dates=True)
lam.columns = lam.columns.astype(int); lamB.columns = lamB.columns.astype(int)

L=[]; P=L.append
P("=== 02 VALIDATION ===")
P("\n(D2) direct vs breakeven (daily):")
for m in MATS:
    d=(lam[m]-lamB[m]).dropna()
    if len(d)<100: P(f"  {m:>2}Y: overlap insufficiente"); continue
    P(f"  {m:>2}Y: mean {d.mean():6.2f}  sd {d.std():5.2f}  corr {lam[m].corr(lamB[m]):.4f}  n={len(d)}")
P("\n(B2) curve vs bond-level median:")
z=pd.concat([st['median'],lam[10]],axis=1).dropna()
P(f"  corr level {z.iloc[:,0].corr(z.iloc[:,1]):.3f}  corr d-monthly "
  f"{z.iloc[:,0].resample('ME').last().diff().corr(z.iloc[:,1].resample('ME').last().diff()):.3f}")
mu,t,_ = nw_mean(lam[10].resample('ME').last().loc[FLL_A:FLL_B])
P(f"\n(D3) FLL window: lambda10 = {mu:.1f} bp [t={t:.1f}]  vs paper {FLL_MEAN_BP}")
save_txt("02_validation.txt", L); print("\n".join(L))

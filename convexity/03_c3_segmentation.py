"""03 - C3: segmentazione curva-vs-swaption sui 5 mercati (3M e 1Y x 10Y, NORM) + episodi."""
import pandas as pd, numpy as np
from config import *; from utils import *
print("== 03 c3 ==")
BE=pd.read_csv(PROC/"sigbe_monthly.csv",index_col=0,parse_dates=True)
W=pd.read_csv(PROC/"vols_monthly.csv",index_col=0,parse_dates=True)
def iv(ccy,e): 
    c=f"{ccy}_{e}_10Y_NORM"; return W[c] if c in W.columns else None
L=[];P=L.append; P("=== 03 C3 SEGMENTATION ===")
P(f"{'mercato':9}{'exp':4}{'N':>5}{'liv':>7}{'DELTA':>8}{'08-14':>8}{'15-26':>8}")
ORDER=["USDswap","USTgovt","EUR","DEgovt","GBP","UKgovt","JPY","JPgovt"]
for mkt in [m for m in ORDER if m in BE.columns]:
    for e in ["3M","1Y"]:
        s=iv(IVMAP[mkt],e)
        if s is None: P(f"{mkt:9}{e:4}  serie assente"); continue
        j=pd.concat([BE[mkt],s],axis=1).dropna(); d=j.diff().dropna()
        a=d.loc["2008":"2014"]; b=d.loc["2015":]
        P(f"{mkt:9}{e:4}{len(j):5d}{j.iloc[:,0].corr(j.iloc[:,1]):+7.2f}{d.iloc[:,0].corr(d.iloc[:,1]):+8.2f}"
          f"{(a.iloc[:,0].corr(a.iloc[:,1]) if len(a)>12 else np.nan):+8.2f}{(b.iloc[:,0].corr(b.iloc[:,1]) if len(b)>12 else np.nan):+8.2f}")
P("\nepisodi: BE firmata / IV(3Mx10Y):")
for d in EPISODES:
    P("  "+d[:7]+" | "+" | ".join(f"{m} {BE[m].asof(pd.Timestamp(d)):+5.0f}/{(iv(IVMAP[m],'3M').asof(pd.Timestamp(d))):3.0f}" for m in [x for x in ORDER if x in BE.columns]))
save_txt("03_c3.txt",L); print("\n".join(L))

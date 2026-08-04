"""07 - test della storia 'one object, two clienteles': S2 (ordinamento di integrazione
tra venue predetto dalla sovrapposizione delle clientele) e S5 (half-life del gap
ordinate dalla profondita' della segmentazione)."""
import pandas as pd, numpy as np
from config import *; from utils import *
print("== 07 story tests ==")
BE=pd.read_csv(PROC/"sigbe_monthly.csv",index_col=0,parse_dates=True)
W=pd.read_csv(PROC/"vols_monthly.csv",index_col=0,parse_dates=True)
S2v=pd.read_csv(PROC/"s2be_monthly.csv",index_col=0,parse_dates=True)
mid=pd.read_csv(PROC/"mids_daily.csv",index_col=0,parse_dates=True)
L=[];P=L.append
P("=== S2: integrazione tra venue (corr livelli / Delta) -- predizione: JPY max, GBP min(<0) ===")
for mkt in ["JPY","EUR","USDswap","USTgovt","GBP"]:
    c=f"{IVMAP[mkt]}_3M_10Y_NORM"
    j=pd.concat([BE[mkt],W[c]],axis=1).dropna(); d=j.diff().dropna()
    P(f"  {mkt:8}: liv {j.iloc[:,0].corr(j.iloc[:,1]):+.2f} | Delta {d.iloc[:,0].corr(d.iloc[:,1]):+.2f}")
P("\n=== S5: half-life AR(1) del gap G = s2_TRAIL - s2_BE -- predizione: piu' lenta dove piu' segmentato ===")
for mkt in ["JPY","EUR","USDswap","USTgovt","GBP"]:
    if mkt in MK:
        dy=(mid[MK[mkt][0][1]]/100.0).diff()
        trail=(dy.rolling(63).std()*np.sqrt(252)).resample("ME").last()
        G=(trail**2 - S2v[mkt]).dropna()
    else:
        G=(-S2v[mkt]).dropna()
    x=G-G.mean(); rho=x.autocorr(1)
    hl=np.log(0.5)/np.log(rho) if 0<rho<1 else np.nan
    P(f"  {mkt:8}: rho1 {rho:+.2f} | half-life {hl:5.1f} mesi")
save_txt("07_story.txt",L); print("\n".join(L))

"""08 - audit di robustezza: igiene mid/half-spread (spike-and-revert), episodio-dipendenza
(LOO sulle 4 finestre di crisi, ex-ALL, Spearman, rolling36) su C3/S2/S5, e outlier/winsor
sulle strategie nette e sul tercile HIGH. Esiti run 01-ago-2026: C3 e S2 robusti (ex-ALL
entro +-0.05; GBP livelli -0.56 ex-crisi); EUR netA robusto (worst-LOO +7.9[2.5], drop3 +11.7);
GBP netA concentrato nelle crisi (61% P&L, ex +3.6[0.7]); GBP HIGH fragile (drop3 +11.4[1.0]);
S5 stabile solo agli estremi (JPY/UST); mid: 21 tick/75k, 0 su fine-mese, sigma_BE invariata."""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *; from utils import nw_t, fly_weights, sigbe_and_returns, save_txt
BE=pd.read_csv(PROC/"sigbe_monthly.csv",index_col=0,parse_dates=True)
W=pd.read_csv(PROC/"vols_monthly.csv",index_col=0,parse_dates=True)
NET=pd.read_csv(PROC/"strat_monthly.csv",index_col=0,parse_dates=True)
mid=pd.read_csv(PROC/"mids_daily.csv",index_col=0,parse_dates=True)
hs=pd.read_csv(PROC/"halfspreads_daily.csv",index_col=0,parse_dates=True)
S2v=pd.read_csv(PROC/"s2be_monthly.csv",index_col=0,parse_dates=True)
WIN={"GFC":("2008-09-01","2009-06-30"),"EURO":("2011-07-01","2012-08-31"),
     "COVID":("2020-02-01","2020-06-30"),"QTLDI":("2022-09-01","2023-03-31")}
def mask_ex(idx,wins):
    m=pd.Series(True,index=idx)
    for a,b in wins: m[(idx>=a)&(idx<=b)]=False
    return m
def spike_revert(s,floor=0.10,k=6.0):
    dx=s.diff(); mad=dx.abs().rolling(250,min_periods=60).median()
    big=dx.abs()>np.maximum(k*1.4826*mad,floor)
    rev=(s.shift(-1)-s.shift(1)).abs()<0.35*(s-s.shift(1)).abs()
    return (big&rev).fillna(False)
L=[];P=L.append
P("=== 08 ROBUSTNESS AUDIT ===")
P("\n[igiene mid] tick spike-and-revert per gamba (0 attesi su fine-mese):")
tot=0
for c in mid.columns:
    f=spike_revert(mid[c].dropna()); tot+=int(f.sum())
    if f.sum(): P(f"  {c}: {int(f.sum())} ({', '.join(str(d.date()) for d in f[f].index[:3])})")
P(f"  totale: {tot} su {int(mid.notna().sum().sum()):,} punti")
P("\n[A] C3 Delta-corr: pieno | exALL | Spearman:")
for mkt in ["USDswap","USTgovt","EUR","GBP","JPY"]:
    c=f"{IVMAP[mkt]}_3M_10Y_NORM"
    d=pd.concat([BE[mkt],W[c]],axis=1).dropna().diff().dropna()
    ex=d[mask_ex(d.index,list(WIN.values()))]
    P(f"  {mkt:8}: {d.iloc[:,0].corr(d.iloc[:,1]):+.2f} | {ex.iloc[:,0].corr(ex.iloc[:,1]):+.2f} | {d.iloc[:,0].corr(d.iloc[:,1],method='spearman'):+.2f}")
P("\n[B] S2 livelli pieno|ex-crisi:")
for mkt in ["JPY","EUR","USDswap","USTgovt","GBP"]:
    c=f"{IVMAP[mkt]}_3M_10Y_NORM"
    j=pd.concat([BE[mkt],W[c]],axis=1).dropna(); ex=j[mask_ex(j.index,list(WIN.values()))]
    P(f"  {mkt:8}: {j.iloc[:,0].corr(j.iloc[:,1]):+.2f} | {ex.iloc[:,0].corr(ex.iloc[:,1]):+.2f}")
P("\n[D] netA: full[t] | exALL[t] | drop3 | %P&L crisi:")
for mkt in NET.columns:
    x=NET[mkt].dropna(); ex=x[mask_ex(x.index,list(WIN.values()))]
    d3=x.drop(x.abs().nlargest(3).index)
    pcr=100*x[~mask_ex(x.index,list(WIN.values()))].sum()/x.sum() if x.sum()!=0 else np.nan
    P(f"  {mkt:8}: {x.mean():6.2f}[{nw_t(x):4.1f}] | {ex.mean():6.2f}[{nw_t(ex):4.1f}] | {d3.mean():6.2f} | {pcr:4.0f}%")
# [E] staleness: run piatti nelle IV e C3 sui soli mesi freschi
import re as _re, glob as _glob
_F={"Normalised Vol ATM":"NORM","RFR Normalised Vol ATM":"NORM","ATM_IVOL_NOM":"NORM"}
IVdd={}
for p in VOLS:
    sr=pd.read_excel(p,sheet_name="Series",header=None)
    nr=next((i for i in range(4) if any(("Vol ATM" in str(x)) or ("IVOL" in str(x)) for x in sr.iloc[i].tolist())),None)
    if nr is None: continue
    names=[str(x).strip() for x in sr.iloc[nr].tolist()]; dd=pd.to_datetime(sr.iloc[nr+1:,0],errors="coerce")
    for j,n in enumerate(names):
        m=_re.match(r"([A-Z]{3})SW(3M|1Y)10YF (.+)",n)
        if not m or m.group(3) not in _F: continue
        k=(m.group(1),m.group(2))
        if k in IVdd: continue
        v=pd.to_numeric(sr.iloc[nr+1:,j],errors="coerce"); v.index=dd
        v=v.dropna().sort_index(); IVdd[k]=v[v>0]
P("\n[E] staleness IV usate: maxrun (giorni) e C3 pieno vs mesi-freschi (quota mossa <=10gg):")
for mkt in ["USDswap","USTgovt","EUR","GBP","JPY"]:
    ccy=IVMAP[mkt]; sIV=IVdd[(ccy,"3M")]
    grp=(sIV.diff()!=0).cumsum(); mx=int(sIV.groupby(grp).size().max())
    fresh=(sIV.diff()!=0).rolling(10,min_periods=1).max().astype(bool).resample("ME").last()
    ivm=sIV.resample("ME").last()
    d=pd.concat([BE[mkt],ivm],axis=1).dropna().diff().dropna()
    dfr=d[fresh.reindex(d.index).fillna(False)]
    P(f"  {mkt:8}: maxrun {mx:2d}g | C3 {d.iloc[:,0].corr(d.iloc[:,1]):+.2f} -> freschi {dfr.iloc[:,0].corr(dfr.iloc[:,1]):+.2f} (N {len(d)}->{len(dfr)})")
save_txt("08_robustness.txt",L); print("\n".join(L))

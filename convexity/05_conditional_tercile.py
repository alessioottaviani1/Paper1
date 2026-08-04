"""05 - raffinamento condizionale: strategia netta per terzile di dealer-CDS regionale (livelli)."""
import pandas as pd, numpy as np
from config import *; from utils import nw_t, save_txt, load_dealer_cds, load_market_states
print("== 05 tercile ==")
NET=pd.read_csv(PROC/"strat_monthly.csv",index_col=0,parse_dates=True)
CDS,_,_=load_dealer_cds()   # composite regionale, letta direttamente dal foglio cds di bbg_paper2
L=[];P=L.append; _lab="NET" if APPLY_COSTS else "GROSS"
P(f"=== 05 {_lab} timed per terzile dealer-CDS (LOW/MID/HIGH) ===")
P(f"{'mercato':8}{'reg':>4}{'LOW':>8}{'MID':>8}{'HIGH':>9}{'[t HIGH]':>9}{'H-L':>8}")
for mkt in NET.columns:
    reg=CDSREGION[mkt]; s=CDS[reg].reindex(NET.index)
    q1,q2=s.quantile(1/3),s.quantile(2/3)
    lo=NET[mkt][s<=q1].dropna(); mi=NET[mkt][(s>q1)&(s<=q2)].dropna(); hi=NET[mkt][s>q2].dropna()
    P(f"{mkt:8}{reg:>4}{lo.mean():8.2f}{mi.mean():8.2f}{hi.mean():9.2f}{nw_t(hi):9.1f}{hi.mean()-lo.mean():8.2f}")
# --- pannello robustezza: sort per SnrFin-Main (stress bancario purgato del credito broad; globale) ---
try:
    ms=load_market_states()
    S=ms["FIN_MINUS_MAIN"].resample("ME").last()
    q=S.dropna().quantile([1/3,2/3]).values
    reg=pd.Series(np.where(S<q[0],"LOW",np.where(S<q[1],"MID","HIGH")),index=S.index)
    P("")
    P("=== robustezza: NET per terzile di SnrFin-Main (variabile GLOBALE, dal 2004) ===")
    P(f"{'mercato':8}{'LOW':>8}{'MID':>8}{'HIGH':>9} [t HIGH]{'H-L':>8}")
    for mkt in NET.columns:
        x=NET[mkt].dropna(); r=reg.reindex(x.index)
        lo,mi,hi=(x[r==k] for k in ("LOW","MID","HIGH"))
        P(f"{mkt:8}{lo.mean():8.2f}{mi.mean():8.2f}{hi.mean():9.2f}   [{nw_t(hi):4.1f}]{hi.mean()-lo.mean():8.2f}")
except Exception as e:
    P(f"[05] pannello SnrFin-Main saltato: {e}")
save_txt("05_tercile.txt",L); print("\n".join(L))

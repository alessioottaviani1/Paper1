"""04 - strategia timed. Con APPLY_COSTS=False: risultati GROSS (skill del segnale) + shift-test.
Con APPLY_COSTS=True: aggiunge netA/netB (due bound, modello MOVE-tiered) e li riporta."""
import pandas as pd, numpy as np
from config import *; from utils import *
print("== 04 strategy ==")
mid=pd.read_csv(PROC/"mids_daily.csv",index_col=0,parse_dates=True)
hs =pd.read_csv(PROC/"halfspreads_daily.csv",index_col=0,parse_dates=True)
S2 =pd.read_csv(PROC/"s2be_monthly.csv",index_col=0,parse_dates=True)
RR =pd.read_csv(PROC/"pack_returns_monthly.csv",index_col=0,parse_dates=True)
L=[];P=L.append
_mode="NET, costi MOVE-tiered" if APPLY_COSTS else "GROSS - costi OFF, in attesa quote dealer"
P("=== 04 STRATEGY ("+_mode+") ===")
hdr=f"{'mercato':8}{'N':>5}{'gross':>8}{'[t]':>6}{'p-shift':>8}"
if APPLY_COSTS: hdr+=f"{'netA':>8}{'[t]':>6}{'pA':>7}{'netB':>8}{'[t]':>6}"
hdr+=f"{'flip':>6}"; P(hdr)
STRAT={}
for mkt,(legs,taus) in MK.items():
    if mkt in GOVT_MKTS: continue   # costi swap non applicabili alle curve governative
    t1,t2,t3=taus; w1,w3,_=fly_weights(t1,t2,t3)
    idx=S2[mkt].dropna().index
    dy=(mid[legs[1]]/100.0).diff()
    trail=(dy.rolling(63).std()*np.sqrt(252)).resample("ME").last().reindex(idx)
    pos=np.sign(trail**2-S2[mkt].reindex(idx)).fillna(0.0)
    R=RR[mkt].reindex(idx)
    gross=(pos.shift(1)*R).dropna()
    HSm=hs[list(legs)].resample("ME").mean().reindex(idx).ffill()
    pack=w1*t1*HSm[legs[0]]+t2*HSm[legs[1]]+w3*t3*HSm[legs[2]]
    dpos=pos.diff().abs().reindex(gross.index).fillna(0)
    netA=gross-dpos*pack.reindex(gross.index)
    netB=netA-pos.shift(1).abs().reindex(gross.index)*pack.reindex(gross.index)
    primary=netA if APPLY_COSTS else gross
    pv=pos.shift(1).reindex(gross.index).values; Rv=R.reindex(gross.index).values; pc=pack.reindex(gross.index).values
    K=len(gross); obs=primary.mean()
    def perm(k):
        pr=np.roll(pv,k); g=pr*Rv
        if APPLY_COSTS: g=g-np.abs(np.diff(np.concatenate([[0],pr])))*pc
        return g.mean()
    psh=sum(perm(k)>=obs for k in range(1,K))/(K-1)
    row=f"{mkt:8}{K:5d}{gross.mean():8.2f}{nw_t(gross):6.1f}{psh:8.3f}"
    if APPLY_COSTS:
        cntA=sum(((np.roll(pv,k)*Rv-np.abs(np.diff(np.concatenate([[0],np.roll(pv,k)])))*pc).mean()>=netA.mean()) for k in range(1,K))
        row+=f"{netA.mean():8.2f}{nw_t(netA):6.1f}{cntA/(K-1):7.3f}{netB.mean():8.2f}{nw_t(netB):6.1f}"
    row+=f"{int((pos.diff().abs()==2).sum()):6d}"; P(row)
    STRAT[mkt]=primary
pd.DataFrame(STRAT).to_csv(PROC/"strat_monthly.csv")
POOL=pd.DataFrame(STRAT).dropna().mean(axis=1)
P(f"{'POOL4':8}{len(POOL):5d}{POOL.mean():8.2f}{nw_t(POOL):6.1f}")
save_txt("04_strategy.txt",L); print("\n".join(L))

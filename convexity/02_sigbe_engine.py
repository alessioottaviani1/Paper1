"""02 - motore sigma_BE (3 punti) sui 5 mercati + ritorni pacchetto; validazione topologica vs Tabella 4."""
import pandas as pd, numpy as np
from config import *; from utils import *
print("== 02 sigbe ==")
mid=pd.read_csv(PROC/"mids_daily.csv",index_col=0,parse_dates=True)
BE={}; RR={}; S2={}
for mkt,(legs,taus) in MK.items():
    Zm=mid[list(legs)].dropna().resample("ME").last()/100.0; Zm.columns=["z1","z2","z3"]
    BE[mkt],RR[mkt],S2[mkt]=sigbe_and_returns(Zm,taus)
Z=pd.read_csv(PROC/"gsw_nodes_monthly.csv",index_col=0,parse_dates=True); Z.columns=["z1","z2","z3"]
BE["USTgovt"],RR["USTgovt"],S2["USTgovt"]=sigbe_and_returns(Z,(2,10,30))
pd.DataFrame(BE).to_csv(PROC/"sigbe_monthly.csv"); pd.DataFrame(RR).to_csv(PROC/"pack_returns_monthly.csv")
pd.DataFrame(S2).to_csv(PROC/"s2be_monthly.csv")
L=[];P=L.append; P("=== 02 SIGMA_BE (3pt) ===")
for k,v in BE.items(): P(f"  {k:8}: {v.dropna().index.min().date()} -> {v.dropna().index.max().date()} | media {v.mean():6.0f} bp/yr")
P("\nvalidazione topologica (segni attesi da Report Tab.4):")
ok=tot=0
for d,exp in EXPECTED_SIGN.items():
    for mkt,sg in exp.items():
        val=BE[mkt].asof(pd.Timestamp(d)); tot+=1
        hit = (sg==0 and abs(val)<60) or (sg!=0 and np.sign(val)==sg)
        ok+=hit; P(f"  {d[:7]} {mkt:8}: BE {val:+6.0f}  atteso {'rich' if sg>0 else 'cheap' if sg<0 else '~0'}  {'OK' if hit else 'X'}")
P(f"  -> {ok}/{tot} celle coerenti")
save_txt("02_sigbe.txt",L); print("\n".join(L))

"""01 - import e QC: gambe MID da bbg_paper2 (foglio swap), half-spread (bid/ask storici se
presenti, altrimenti livelli BASE), vol swaption (batch Barclays), GSW -> processed."""
import pandas as pd, numpy as np
from config import *; from utils import *
print("== 01 import ==")
mid = load_legs_mid()
hs, hs_src = load_halfspreads(mid.index)
print(f"[01] half-spread: {hs_src}")
pd.DataFrame(mid).to_csv(PROC/"mids_daily.csv"); pd.DataFrame(hs).to_csv(PROC/"halfspreads_daily.csv")
IV = load_vols()
W = pd.DataFrame({f"{c}_{e}_{t}_{f}":s for (c,e,t,f),s in IV.items()})
W.to_csv(PROC/"vols_monthly.csv")
Z = load_gsw_nodes(); Z.to_csv(PROC/"gsw_nodes_monthly.csv")
L=[]; P=L.append
P("=== 01 IMPORT/QC ===")
P(f"gambe (mid, bbg_paper2): {len(mid)} gg | hs [{hs_src}] medio bp: "+", ".join(f"{k} {hs[k].mean():.2f}" for k in list(hs)[:6])+" ...")
per={}
for (c,e,t,f) in IV: per[c]=per.get(c,0)+1
P(f"vol swaption: {len(IV)} serie | per valuta: {per}")
P(f"GSW nodi: {Z.index.min().date()} -> {Z.index.max().date()}")
save_txt("01_import.txt", L); print("\n".join(L))

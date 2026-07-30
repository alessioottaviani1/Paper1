"""06 — Superficie forced-unwind: stop-out per vintage × buffer (monitoraggio DAILY)."""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

med_d = pd.read_csv(PROC/"cusip_stats_daily.csv", index_col=0, parse_dates=True)["median"].dropna()
L=[]; P=L.append
P("=== 06 FORCED-UNWIND SURFACE (daily monitoring, D=8, 24m) ===")
entries = med_d.resample("MS").first().dropna()
rows={}
for buf,lab in [(100,"1%"),(200,"2%")]:
    stop={}
    for t0,l0 in entries.items():
        win = med_d.loc[t0: t0+pd.DateOffset(months=H_RISK_M)]
        if len(win)<200: continue
        stop[t0.year] = stop.get(t0.year,[]) + [ (D_DUR*(win-l0)).max() > buf ]
    rows[lab]={yy:np.mean(v) for yy,v in stop.items()}
yrs=sorted(set(rows["1%"])|set(rows["2%"]))
P("  vintage : " + " ".join(f"{yy%100:>4}" for yy in yrs))
for lab in ["1%","2%"]:
    P(f"  stop {lab}: " + " ".join(f"{rows[lab].get(yy,np.nan)*100:4.0f}" for yy in yrs))
pd.DataFrame(rows).to_csv(RES/"06_unwind_surface.csv")
save_txt("06_unwind_replay.txt", L); print("\n".join(L))

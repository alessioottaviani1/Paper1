"""Diagnostica del fit NSS sui prezzi nelle date a RMSE alto: mostra, per le N date
peggiori, i bond con residuo di prezzo maggiore (in bp di yield equivalente). Dice se
l'RMSE alto viene da pochi bond stale (-> filtro outlier) o da curva irregolare diffusa
(-> limite NSS, da accettare). Sola lettura. Lancia dalla root:
    python .\src\inflation_linked\_diag_fit.py
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import bbg, curves
from config import MARKETS, CACHE
from basis import settlement

MKT = "IT"; N_DATE = 6; N_BOND = 6
nss = pd.read_parquet(CACHE / f"nss_{MKT}.parquet")
worst = nss.sort_values("rmse_bp", ascending=False).head(N_DATE)
px = pd.read_parquet(CACHE / f"px_nom_{MKT}.parquet"); px.index = pd.to_datetime(px.index)
ytm = pd.read_parquet(CACHE / f"ytm_{MKT}.parquet"); ytm.index = pd.to_datetime(ytm.index)
ref = bbg.load("ref_nominal"); ref = ref[ref["mkt"] == MKT]
mats = pd.to_datetime(ref["MATURITY"]); cpns = pd.to_numeric(ref["CPN"], errors="coerce")
freqs = pd.to_numeric(ref.get("CPN_FREQ"), errors="coerce").fillna(2)
m = MARKETS[MKT]; hol = m.holidays; nset = getattr(m, "settle_days", 2)

for dt, prow in worst.iterrows():
    p = px.loc[dt].dropna() if dt in px.index else pd.Series(dtype=float)
    settle = pd.Timestamp(settlement(dt.date(), nset, hol))
    par = [prow[c] for c in ("b0","b1","b2","b3","t1","t2")]
    res = []
    for isin, pc in p.items():
        if isin not in mats.index or pd.isna(cpns.get(isin)): continue
        mat = mats[isin]; tau_m = (mat - settle).days/365.25
        if not (0.25 < tau_m < 50): continue
        freq = int(freqs.get(isin,2)) or 2; step = max(1, round(12/freq))
        dts=[mat]
        while dts[-1] > settle: dts.append(dts[-1]-pd.DateOffset(months=step))
        nxt = sorted(d for d in dts if d > settle)
        if not nxt: continue
        last = nxt[0]-pd.DateOffset(months=step); ced=float(cpns[isin])/freq
        den=(nxt[0]-last).days; acc=ced*(settle-last).days/den if den>0 else 0
        taus=np.array([(d-settle).days/365.25 for d in nxt]); amts=np.full(len(nxt),ced); amts[-1]+=100
        z=curves.nss_yield(taus,*par); pm=float(np.sum(amts*np.exp(-z/100*taus)))
        res.append((isin, tau_m, float(pc)+acc, pm, pm-(float(pc)+acc)))
    d = pd.DataFrame(res, columns=["isin","tau","dirty_obs","dirty_fit","err_price"])
    d["err_bp"] = d["err_price"]/d["dirty_obs"].abs()*1e4/d["tau"].clip(lower=0.5)  # ~bp yield
    d = d.reindex(d["err_price"].abs().sort_values(ascending=False).index)
    print(f"\n=== {dt.date()}  rmse={prow['rmse_bp']:.1f}bp  n={int(prow['n_bonds'])}  {prow['model']} ===")
    print(d.head(N_BOND)[["isin","tau","dirty_obs","dirty_fit","err_price"]].to_string(index=False))
    big = (d["err_price"].abs() > 0.5).sum()
    print(f"  bond con |err prezzo|>0.5: {big}/{len(d)}  "
          f"({'POCHI outlier' if big<=2 else 'DIFFUSO -> curva irregolare'})")

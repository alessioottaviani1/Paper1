"""r1 - BEI US e UK e premio residuo (richiesta RR n.1), specificazione market-based
di Maffei-Rebonato: projected inflation = zero-coupon inflation swap rate (ISR).
  BEI - projected = BEI - ISR = -lambda   (mispricing / liquidita' dei linker,
                                          il lambda di Fleckenstein et al. e di Maffei)
  BEI - spot      = BEI - YoY corrente     (premio grezzo)
Confronto US vs UK per sottoperiodi + correlazione. Output: an_bei_premia.csv.
Curve: BoE (UK) e GSW (US) + ISR dal cache della pipeline (BPSWIT / USSWIT).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import pandas as pd
import rp
from config import CACHE

MATS = (5.0, 10.0)
out = {}
for mkt in ("US", "UK"):
    b, s, lam = rp.lam_gamma(mkt, MATS)
    pi = rp.yoy(rp.cpi(mkt))
    for m in MATS:
        k = int(m)
        out[f"BEI_{mkt}_{k}y"] = b[m]
        out[f"ISR_{mkt}_{k}y"] = s[m]
        out[f"BEI-ISR_{mkt}_{k}y"] = -lam[m]           # BEI - projected
        out[f"BEI-spot_{mkt}_{k}y"] = b[m] - pi.reindex(b.index).ffill()
out = pd.DataFrame(out).sort_index()
out.to_csv(CACHE / "an_bei_premia.csv")

idx = out.dropna(subset=["BEI-ISR_US_10y", "BEI-ISR_UK_10y"]).index
per = {"pieno": idx, "pre-2009": idx[idx < "2009"], "2009-2019": idx[(idx >= "2009") & (idx < "2020")],
       "2020-oggi": idx[idx >= "2020"]}
print(f"{'periodo':10s} {'BEI-ISR US10':>13s} {'BEI-ISR UK10':>13s} {'BEI-spot US10':>14s} "
      f"{'BEI-spot UK10':>14s} {'corr(BEI-ISR)':>14s}")
for k, ii in per.items():
    if len(ii) < 12:
        continue
    d = out.loc[ii]
    print(f"{k:10s} {d['BEI-ISR_US_10y'].mean()*100:12.0f}bp {d['BEI-ISR_UK_10y'].mean()*100:12.0f}bp "
          f"{d['BEI-spot_US_10y'].mean():13.2f}% {d['BEI-spot_UK_10y'].mean():13.2f}% "
          f"{d['BEI-ISR_US_10y'].corr(d['BEI-ISR_UK_10y']):14.2f}")
print(f"\nsalvato: an_bei_premia.csv ({len(out)} mesi, campione {idx.min().date()} -> {idx.max().date()})")
print("BEI-ISR = -lambda: negativo = i linker rendono PIU' del sintetico (premio di")
print("liquidita'/mispricing a favore del nominale, come in FLL). Confronto US/UK: livelli,")
print("correlazione e episodi (2008, 2020, 2022) leggibili in an_bei_premia.csv.")

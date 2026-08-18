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
for mkt in ("US", "UK", "IT", "FR", "DE"):
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

MKTS = ("US", "UK", "IT", "FR", "DE")
PER = {"pre-2009": lambda x: x < "2009", "2009-2019": lambda x: (x >= "2009") & (x < "2020"),
       "2020-oggi": lambda x: x >= "2020"}

# livello medio di lambda per mercato, ognuno sul PROPRIO campione (niente intersezione a 5,
# che la Germania giovane taglierebbe al 2012): ogni linker sfrutta tutta la sua storia.
print("=== livello medio  -lambda = BEI-ISR (10y), ogni mercato sul proprio campione ===")
print("periodo    " + "".join(f"{m:>7s}" for m in MKTS))
for k, mask in PER.items():
    row = f"{k:10s}"
    for m in MKTS:
        col = f"BEI-ISR_{m}_10y"
        d = out[col].dropna(); d = d[mask(d.index)]
        row += f"{d.mean()*100:5.0f}bp" if len(d) >= 12 else f"{'--':>7s}"
    print(row)
sp = {m: (out[f"BEI-ISR_{m}_10y"].dropna().index.min(), out[f"BEI-ISR_{m}_10y"].dropna().index.max()) for m in MKTS}
print("campioni:", "  ".join(f"{m} {a.year}-{b.year}" for m,(a,b) in sp.items()))

# correlazione BILATERALE con US: ogni coppia sul suo campione comune
print("\n=== correlazione di -lambda(10y) con US, BILATERALE (campione comune di coppia) ===")
print("periodo   " + "".join(f"{'c(US,'+m+')':>10s}" for m in MKTS[1:]))
for k, mask in PER.items():
    row = f"{k:10s}"
    for m in MKTS[1:]:
        d = out[[f"BEI-ISR_US_10y", f"BEI-ISR_{m}_10y"]].dropna()
        d = d[mask(d.index)]
        row += f"{d.iloc[:,0].corr(d.iloc[:,1]):10.2f}" if len(d) >= 12 else f"{'--':>10s}"
    print(row)
print(f"\nsalvato: an_bei_premia.csv. -lambda<0 = linker cheap al sintetico (FLL). Confronto")
print("core (DE~-12, FR~-3: piu' cari, risk-free area) vs periferia (IT~-28: piu' a sconto);")
print("c(US,IT) crolla nel 2009-19 (crisi sovrana), c(US,DE/FR) resta positiva (core globale).")

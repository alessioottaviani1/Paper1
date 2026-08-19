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

print(">>> r1 US/UK-only v2 <<<  (se non vedi questa riga, gira un file diverso)")
MATS = (5.0, 10.0)
out = {}
for mkt in ("US", "UK"):        # SOLO i due mercati che RR chiede (curve solide)
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

MKTS = ("US", "UK")             # RR n.1 riguarda US e UK
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
# ============================================================================ RR domanda 1
# "Calculate US BEI and UK BEI inflation. Subtract projected and spot inflation from both.
#  Compare the residual risk premium for US and UK. Similar? If different, why?"
# Le due sottrazioni sono entrambe richieste: projected = ISR (swap zero-coupon),
# spot = YoY corrente. Qui il confronto DIRETTO US vs UK sui due residui, per scadenza
# e per sottoperiodo, con la differenza e la sua dispersione.
print("\n" + "=" * 78)
print("RR n.1 -- CONFRONTO DIRETTO US vs UK del premio residuo")
print("=" * 78)
print("  due residui, come chiesto: BEI-ISR (projected) e BEI-spot (inflazione corrente)")

for tag, lbl in (("BEI-ISR", "BEI - projected (ISR)"), ("BEI-spot", "BEI - spot (YoY)")):
    print(f"\n  {lbl}   [bp]")
    print(f"    {'periodo':10s}" + "".join(f"{h:>12s}" for h in
                                           ("US 5y", "UK 5y", "diff 5y", "US 10y", "UK 10y", "diff 10y")))
    for k, mask in PER.items():
        row = f"    {k:10s}"
        for ten in (5, 10):
            us = out[f"{tag}_US_{ten}y"].dropna(); us = us[mask(us.index)]
            uk = out[f"{tag}_UK_{ten}y"].dropna(); uk = uk[mask(uk.index)]
            both = out[[f"{tag}_US_{ten}y", f"{tag}_UK_{ten}y"]].dropna()
            both = both[mask(both.index)]
            for v, n in ((us, len(us)), (uk, len(uk))):
                row += f"{v.mean()*100:11.0f}" + " " if n >= 12 else f"{'--':>12s}"
            row += (f"{(both.iloc[:,0]-both.iloc[:,1]).mean()*100:11.0f}" + " "
                    if len(both) >= 12 else f"{'--':>12s}")
        print(row)

# la differenza US-UK: livello medio, dispersione e stabilita' del segno (10y, BEI-ISR)
d = out[["BEI-ISR_US_10y", "BEI-ISR_UK_10y"]].dropna()
if len(d) >= 24:
    diff = (d.iloc[:, 0] - d.iloc[:, 1]) * 100
    same_sign = (diff > 0).mean()
    print(f"\n  differenza US-UK (BEI-ISR, 10y): media {diff.mean():+.0f}bp | "
          f"sd {diff.std():.0f}bp | US>UK nel {100*same_sign:.0f}% dei mesi | "
          f"correlazione dei livelli {d.iloc[:,0].corr(d.iloc[:,1]):+.2f}")
    print(f"  campione comune: {d.index.min().date()} -> {d.index.max().date()} ({len(d)} mesi)")
    verdetto = ("SIMILI" if abs(diff.mean()) < 10 and diff.std() < 20 else
                "DIVERSI" if abs(diff.mean()) >= 10 else "simili in media, volatili")
    print(f"  -> i due premi residui sono {verdetto}")
    print("  [il PERCHE' si legge in r2b: se il lambda dei due mercati carica in modo")
    print("   diverso su liquidita' (PCA) e sorprese (InfS), quella e' la ragione --")
    print("   e' la stessa scomposizione della tesi, non un test aggiuntivo]")

print(f"\nsalvato: an_bei_premia.csv. -lambda<0 = linker cheap al sintetico (FLL).")

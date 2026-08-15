"""Ultimo check autonomo: la BEI US (=GSW nom - GSW TIPS) reagisce alle sorprese quasi
il doppio della BEI UK (+27 vs +15.6 bp). E' plausibile o e' un artefatto della nostra
costruzione? Confronto le reattivita' delle DUE gambe (nominale e reale) separatamente,
US vs UK, per vedere DOVE nasce l'asimmetria. Sola lettura.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np, pandas as pd
import rp

liq = rp.liquidity()
MATS = (5.0, 10.0)

def reg_leg(series_by_mat, surp, liq, mats, lags=6):
    out = {}
    for n in mats:
        parts = [(series_by_mat[float(n)]*100).rename("y"), surp.rename("s")] + \
                [liq[c].rename(c) for c in liq.columns]
        d = pd.concat(parts, axis=1).dropna()
        if len(d) < 30: continue
        b,e,fit,r2,X = rp.ols(d["y"].values, d[["s"]+list(liq.columns)].values)
        out[float(n)] = b[1]
    return out

print("Reattivita' delle gambe alle sorprese (bp per unita', netto liquidita'):\n")
print(f"{'mkt':>4} {'mat':>4} {'d(nominale)':>12} {'d(reale)':>10} {'d(BEI)=nom-reale':>17} {'d(ISR)':>8}")
for mkt in ("US", "UK"):
    surp = rp.surprise_maffei(mkt)
    if mkt == "US":
        nom = rp.gsw("nominal", MATS); real = rp.gsw("tips", MATS)
    else:
        nom = rp.interp_cols(rp.boe("nominal"), MATS); real = rp.interp_cols(rp.boe("real"), MATS)
    bei = nom.sub(real)  # nostra BEI
    _, isr, _ = rp.lam_gamma(mkt, MATS)
    dn = reg_leg(nom, surp, liq, MATS); dr = reg_leg(real, surp, liq, MATS)
    db = reg_leg(bei, surp, liq, MATS); di = reg_leg(isr, surp, liq, MATS)
    for n in MATS:
        print(f"{mkt:>4} {int(n):>3}y {dn.get(n,np.nan):>+11.1f} {dr.get(n,np.nan):>+9.1f} "
              f"{db.get(n,np.nan):>+16.1f} {di.get(n,np.nan):>+7.1f}")
print("\nLettura: la BEI reagisce = nominale REAGISCE - reale REAGISCE. Se la BEI US e'")
print("ipersensibile, e' perche' il nominale US reagisce molto o il reale poco. Confronto")
print("con UK dice se e' un fatto di mercato o della nostra costruzione GSW.")
print("Nota: per RQ1/RQ2 la BEI e' nom-tips; qui il nominale UK e' BoE nominal (coerente).")

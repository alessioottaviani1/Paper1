"""Diagnostica del segno di lambda US (opposto a Maffei): decompone lambda = ISR - BEI
nelle due gambe e regredisce OGNUNA sulla sorpresa (netto liquidita'), per capire quale
guida il segno. Sola lettura. Lancia:  python .\\src\\linker_premia\\_diag_lambda_us.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np, pandas as pd
import rp

MATS = {"US": (3.0, 5.0, 10.0, 15.0), "UK": (3.0, 5.0, 10.0, 20.0)}
liq = rp.liquidity()

def reg_on_surp(series_by_mat, surp, liq, mats, lags=6):
    """regredisce ogni maturita' (in bp) su [sorpresa, liquidita']; ritorna beta sorpresa."""
    out = {}
    for n in mats:
        parts = [(series_by_mat[float(n)] * 100).rename("y"), surp.rename("s")] + \
                [liq[c].rename(c) for c in liq.columns]
        d = pd.concat(parts, axis=1).dropna()
        if len(d) < 30: continue
        b, e, fit, r2, X = rp.ols(d["y"].values, d[["s"] + list(liq.columns)].values)
        t = rp.nw_t(e, X, b, lags)
        out[float(n)] = (b[1], t[1])
    return out

for mkt in ("US", "UK"):
    surp = rp.surprise_maffei(mkt)
    b, s, lam = rp.lam_gamma(mkt, MATS[mkt])   # b=BEI, s=ISR, lam=ISR-BEI
    print(f"\n=== {mkt} | decomposizione del segno di lambda ===")
    bei_reg = reg_on_surp(b, surp, liq, MATS[mkt])
    isr_reg = reg_on_surp(s, surp, liq, MATS[mkt])
    lam_reg = reg_on_surp(lam, surp, liq, MATS[mkt])
    print(f"  {'mat':>4} {'dBEI/dsurp':>12} {'dISR/dsurp':>12} {'dlam=dISR-dBEI':>15}")
    for n in MATS[mkt]:
        bb = bei_reg.get(float(n), (np.nan,0))[0]
        ss = isr_reg.get(float(n), (np.nan,0))[0]
        ll = lam_reg.get(float(n), (np.nan,0))[0]
        print(f"  {int(n):>3}y {bb:>+11.1f} {ss:>+11.1f} {ll:>+14.1f}")
    print("  lettura: se dBEI>dISR la lambda SCENDE con la sorpresa (segno neg). Maffei: "
          "sui TIPS US la domanda spinge i prezzi -> BEI su -> se ISR fermo, lambda giu.")

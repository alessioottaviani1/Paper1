"""Replica sul CAMPIONE ESATTO di Maffei (ago-2007 -> set-2024) per capire se il segno
di lambda US dipende dal campione. Confronta full-sample (2004-2026) vs Maffei-sample.
Sola lettura. Lancia:  python .\\src\\linker_premia\\_diag_maffei_sample.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np, pandas as pd
import rp

MATS = {"US": (3.0, 5.0, 10.0, 15.0), "UK": (3.0, 5.0, 10.0, 20.0)}
MAFFEI = (pd.Timestamp("2007-08-31"), pd.Timestamp("2024-09-30"))
liq = rp.liquidity()

def beta_surp(mkt, win=None):
    surp = rp.surprise_maffei(mkt)
    b, s, lam = rp.lam_gamma(mkt, MATS[mkt])
    if win:
        lam = lam.loc[(lam.index >= win[0]) & (lam.index <= win[1])]
    t = rp.reg_lambda(lam, surp, liq)
    return t.set_index("mat")

for mkt in ("US", "UK"):
    full = beta_surp(mkt)
    maf = beta_surp(mkt, MAFFEI)
    print(f"\n=== {mkt}: beta sorpresa su lambda (netto liquidita') ===")
    print(f"  {'mat':>4} {'FULL 2004-26':>16} {'MAFFEI 2007-24':>16}")
    for n in MATS[mkt]:
        f = full.loc[float(n)]; m = maf.loc[float(n)]
        print(f"  {int(n):>3}y  {f.beta_surp:+7.1f} (t{f.t_surp:+.1f})   {m.beta_surp:+7.1f} (t{m.t_surp:+.1f})")
    sf = np.sign(full["beta_surp"].iloc[0]); sm = np.sign(maf["beta_surp"].iloc[0])
    print(f"  segno corto: full {'+' if sf>0 else '-'}, Maffei-sample {'+' if sm>0 else '-'}"
          f"  {'-> GIRA col campione!' if sf!=sm else '-> stesso segno'}")
print(f"\nMaffei trova beta POSITIVO sui TIPS US. Se il nostro US gira a + sul suo campione,")
print(f"la differenza era il periodo (2004-07 e 2024-26 esclusi da lui).")

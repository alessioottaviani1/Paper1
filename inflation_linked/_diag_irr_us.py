"""Diagnostica 'IRR: bracket non trovato' (US nearest): replica build_market su poche
date fallite, intercetta _solve_irr e al fallimento stampa QUALE GAMBA fallisce
(linker / sintetico / nominale, dallo stack), i cashflow e f(y) su una griglia.
Sola lettura (save=False). Lancia:  python .\\src\\linker_premia\\_diag_irr_us.py
"""
import sys, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np, pandas as pd
import basis, pipeline

CACHE = pipeline.CACHE

# 1) coppie (data, isin) fallite: prezzo c'e', total_ytm no
ty = pd.read_parquet(CACHE / "totalytm_US.parquet")
px = pd.read_parquet(CACHE / "px_mid_US.parquet")
px.index, ty.index = pd.to_datetime(px.index), pd.to_datetime(ty.index)
miss = (px.notna() & ty.reindex_like(px).isna())
by_date = miss.sum(axis=1)
# date candidate distribuite sui regimi: una per anno tra i piu' colpiti
cand = []
for yr in (2015, 2020, 2022, 2024):
    d = by_date[by_date.index.year == yr]
    if len(d) and d.max() > 0:
        cand.append(d.idxmax())
print("date diagnostiche:", [str(d.date()) for d in cand])

# 2) patch di _solve_irr: al fallimento, stampa gamba + cashflow + f(y)
_orig = basis._solve_irr
_dumped = [0]
def _patched(c, t):
    try:
        return _orig(c, t)
    except ValueError:
        if _dumped[0] < 6:
            _dumped[0] += 1
            stack = [fr.name for fr in traceback.extract_stack()]
            gamba = next((n for n in reversed(stack) if n in
                          ("irr", "synthetic_irr", "bond_basis_row", "match_nominals")), "?")
            f = lambda y: float(np.sum(np.asarray(c) / (1 + y) ** np.asarray(t)))
            grid = [-0.95, -0.60, -0.30, -0.10, 0.0, 0.10, 0.30, 1.0, 4.0, 64.0]
            print(f"\n--- FALLIMENTO #{_dumped[0]} | gamba (stack): {gamba} | stack: {stack[-4:]}")
            print(f"    n flussi: {len(c)} | c[0] (prezzo): {c[0]:+.3f} | "
                  f"flussi>0: {np.sum(np.asarray(c)>0)} somma {np.sum(np.asarray(c)[np.asarray(c)>0]):+.2f} | "
                  f"flussi<0 oltre c[0]: {np.sum(np.asarray(c)[1:]<0)}")
            print(f"    t: [{t[0]:.3f} .. {t[-1]:.3f}] anni")
            print(f"    primi 4 flussi: {np.round(np.asarray(c)[:4], 3).tolist()}")
            print(f"    ultimi 2 flussi: {np.round(np.asarray(c)[-2:], 3).tolist()}")
            print("    f(y): " + "  ".join(f"{y:+.2f}:{f(y):+.2f}" for y in grid))
        raise
basis._solve_irr = _patched
# anche il riferimento eventualmente importato dentro pipeline
if hasattr(pipeline, "_solve_irr"):
    pipeline._solve_irr = _patched

# 3) replica del run sulle sole date candidate (config come il 04, senza salvare)
res = pipeline.build_market("US", methods=("nearest",),
                            exclude_tail=False, ytm_convention="annual",
                            dates=pd.DatetimeIndex(cand), save=False)
print("\nfatto: vedi i blocchi FALLIMENTO sopra (gamba, cashflow, f(y)).")

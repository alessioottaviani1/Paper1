"""Le BEI 'nominale - reale' coincidono a seconda della fonte? Confronto, per UK e US,
le costruzioni disponibili della stessa BEI. Se differiscono, il segno di lambda dipende
da QUALE usiamo -> non e' universale come sembra. Sola lettura.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np, pandas as pd
import rp

MATS = (5.0, 10.0)

print("=== UK: tre modi di ottenere la BEI ===")
bei_infl = rp.interp_cols(rp.boe("inflation"), MATS)        # curva inflation BoE (RQ1/RQ2)
bei_nr   = rp.interp_cols(rp.boe("nominal"), MATS).sub(rp.interp_cols(rp.boe("real"), MATS))  # nom-real BoE
idx = bei_infl.index.intersection(bei_nr.index)
for n in MATS:
    a, b = bei_infl.loc[idx, n], bei_nr.loc[idx, n]
    diff = (a - b) * 100  # bp
    print(f"  {int(n):>3}y: inflation-curve media {a.mean():.2f}% | (nom-real) media {b.mean():.2f}% "
          f"| differenza media {diff.mean():+.1f}bp, sd {diff.std():.1f}bp, max|.| {diff.abs().max():.1f}bp")

print("\n=== US: GSW nom - TIPS (la nostra) vs colonne diagnostiche ===")
bei_us = rp.gsw("nominal", MATS).sub(rp.gsw("tips", MATS))
for n in MATS:
    s = bei_us[n].dropna()
    print(f"  {int(n):>3}y: media {s.mean():.2f}%, sd {s.std():.2f}%  (unica fonte: GSW nom-TIPS)")

print("\nSe UK inflation-curve e (nom-real) differiscono di parecchi bp, e RQ1/RQ2 usano la")
print("PRIMA mentre il diagnostico gambe usa la SECONDA, il confronto di reattivita' non e'")
print("pulito: la BEI 'vera' di RQ2 potrebbe reagire diversamente da nom-real.")

"""
07c - Coerenza fra misura (1) e misura (3) PER SINGOLO CCT.

L'ASPETTATIVA CORRETTA (e' quella giusta). Per un dato CCT, confrontarsi con il BTP piu'
vicino oppure con la curva dei nominali letta alla sua scadenza deve dare quasi lo stesso
numero, perche' quel BTP sta quasi sulla curva. Se per qualche coppia le due misure
divergono molto, la causa e' identificabile: o quel BTP e' davvero fuori curva (caro perche'
on-the-run, o economico perche' illiquido), oppure la curva e' mal stimata a quella scadenza.

Il report precedente confrontava le MEDIANE DELLE SERIE, che con distribuzioni asimmetriche
non si compongono: la mediana di una differenza non e' la differenza delle mediane. Qui si
guarda la differenza OSSERVAZIONE PER OSSERVAZIONE e poi si aggrega per coppia.

  basis1 - basis3 - cuneo  ==  y_BTP(curva) - y_BTP(mercato)   [algebricamente esatto]
  positivo -> il BTP rende MENO di quanto la curva implichi -> BTP CARO
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

B = pd.read_csv(PROC/"basis_daily.csv", parse_dates=["date"])
B["off"] = B.basis1_y - B.basis3_y - B.wedge_y
B["absdiff"] = (B.basis1_y - B.basis3_y).abs()

L=[]; P=L.append
P("=== 07c COERENZA (1) vs (3), PER COPPIA ===")
P(f"osservazioni con entrambe le misure: {B.dropna(subset=['basis1_y','basis3_y']).shape[0]:,}")
P("\n--- distribuzione della differenza OSSERVAZIONE PER OSSERVAZIONE ---")
d = (B.basis1_y - B.basis3_y).dropna()
for q in [0.05, 0.25, 0.50, 0.75, 0.95]:
    P(f"  p{int(q*100):02d}: {d.quantile(q):8.1f} bp")
P(f"  |differenza| < 10 bp in {(d.abs()<10).mean():.1%} delle osservazioni")
P(f"  |differenza| < 25 bp in {(d.abs()<25).mean():.1%}")
P(f"  |differenza| > 50 bp in {(d.abs()>50).mean():.1%}   <-- coppie da guardare")

P("\n--- per COPPIA: le 12 peggiori (BTP piu' lontano dalla curva) ---")
g = (B.dropna(subset=["basis1_y","basis3_y"])
       .groupby(["CCT_ISIN","BTP_ISIN","regime"])
       .agg(n=("basis1_y","size"), b1=("basis1_y","median"), b3=("basis3_y","median"),
            off=("off","median"), absd=("absdiff","median"),
            mism=("mismatch_d","first"), tau=("tau_cct","median"))
       .reset_index().sort_values("absd", ascending=False))
P(f"  {'CCT':>14}{'BTP':>14}{'n':>7}{'b1':>8}{'b3':>8}{'off':>8}{'|d|':>7}{'mism':>6}{'tau':>6}")
for _, r in g.head(12).iterrows():
    P(f"  {r.CCT_ISIN:>14}{r.BTP_ISIN:>14}{r.n:>7,}{r.b1:>8.1f}{r.b3:>8.1f}"
      f"{r.off:>8.1f}{r.absd:>7.1f}{r.mism:>6.0f}{r.tau:>6.1f}")
P(f"\n  coppie con |differenza| mediana < 10 bp: {(g.absd<10).sum()}/{len(g)}")
P(f"  coppie con |differenza| mediana > 50 bp: {(g.absd>50).sum()}/{len(g)}")

P("\n--- il BTP fuori curva dipende dalla scadenza? ---")
B["bucket"] = pd.cut(B.tau_cct, [0,1.5,3,4.5,8], labels=["0-1.5y","1.5-3y","3-4.5y","4.5-8y"])
for b, gg in B.groupby("bucket", observed=True):
    v = gg["off"].dropna()
    if len(v) > 200:
        P(f"  {str(b):>8}: mediana {v.median():7.1f} bp | IQR [{v.quantile(.25):7.1f},{v.quantile(.75):7.1f}] | n {len(v):,}")
P("  [lettura] se il divario esplode solo sul tratto cortissimo, e' il rendimento a breve")
P("            a essere ipersensibile al prezzo, non un difetto della costruzione.")

P("\n--- e dal periodo? ---")
B["yr"] = B.date.dt.year
for a, b in [(1999,2007),(2008,2010),(2011,2012),(2013,2016),(2017,2021),(2022,2026)]:
    v = B[(B.yr>=a)&(B.yr<=b)]["off"].dropna()
    if len(v) > 200:
        P(f"  {a}-{str(b)[-2:]}: mediana {v.median():7.1f} bp | IQR [{v.quantile(.25):7.1f},{v.quantile(.75):7.1f}]")
save_txt("07c_pair_diagnostic.txt", L); print("\n".join(L))

"""
01 - Anagrafica: import, pulizia, classificazione dei regimi, diagnostica del campione.
Non richiede prezzi: gira subito e dice se il disegno del campione regge.
Output: PROC/static_{cct,btp,bot}.csv + results/01_static.txt
"""
import numpy as np, pandas as pd
from config import *
from utils import load_static, save_txt

print("== 01 static ==")
S = load_static(); C, O, B = S["CCTS"], S["BOTS"], S["BTPS"]
L = []; P = L.append
P("=== 01 ANAGRAFICA E DISEGNO DEL CAMPIONE ===")
P(f"universo grezzo: {len(C)} CCT | {len(B)} BTP | {len(O)} BOT")
P(f"  CCT  scadenze {C.maturity.min().date()} -> {C.maturity.max().date()}")
P(f"  BTP  scadenze {B.maturity.min().date()} -> {B.maturity.max().date()}")
P(f"  BOT  scadenze {O.maturity.min().date()} -> {O.maturity.max().date()}")
P(f"  cedole BTP estratte: {B.coupon.notna().sum()}/{len(B)}")

P("\nregime dei CCT (indicizzazione):")
for r, g in C.groupby("regime"):
    P(f"  {r:9s}: {len(g):3d} titoli | emissioni {g.issue.min().date() if g.issue.notna().any() else 'n/d'}"
      f" -> {g.issue.max().date() if g.issue.notna().any() else 'n/d'}")

for lab, s in [("PRIMARIO (euro)", START_PRIMARY), ("ESTESO (pre-euro)", START_EXTENDED)]:
    S0 = pd.Timestamp(s)
    cc = C[C.maturity > S0]
    P(f"\n{lab}, da {s}: {len(cc)} CCT vivi in campione "
      f"({(cc.regime=='CCTeu').sum()} CCTeu, {(cc.regime=='CCT-BOT').sum()} CCT-BOT)")

P("\ntitoli VIVI per data (emessi e non scaduti) - densita' per la curva:")
P(f"  {'data':>12}{'CCT':>6}{'BTP':>6}{'BOT':>6}")
for y in [1995, 1999, 2003, 2008, 2012, 2016, 2020, 2024, 2026]:
    d = pd.Timestamp(f"{y}-06-30")
    P(f"  {str(d.date()):>12}{((C.issue<=d)&(C.maturity>d)).sum():>6}"
      f"{((B.issue<=d)&(B.maturity>d)).sum():>6}{((O.issue<=d)&(O.maturity>d)).sum():>6}")

P("\ndisponibilita' di un BTP vicino per scadenza (condizione necessaria per ISIN-vs-ISIN):")
cc = C[C.maturity > pd.Timestamp(START_EXTENDED)]
for tol in (31, 62, 92, 183):
    n = sum(1 for m in cc.maturity if ((B.maturity - m).abs() <= pd.Timedelta(days=tol)).any())
    P(f"  entro {tol:>3} gg: {n}/{len(cc)} CCT")
P(f"\n[nota] soglia scelta MAX_MISMATCH_D={MAX_MISMATCH_D} gg: i CCT scadono il 15 di")
P( "       gen/apr/lug/ott, i BTP il 1 o il 15 di mesi vari, quindi una soglia stretta")
P( "       alla TIPS (31gg) scarterebbe titoli appaiabili senza ragione economica.")

for n, d in [("static_cct.csv", C), ("static_btp.csv", B), ("static_bot.csv", O)]:
    d.to_csv(PROC / n, index=False); P(f"[saved] {PROC/n}")
save_txt("01_static.txt", L); print("\n".join(L))

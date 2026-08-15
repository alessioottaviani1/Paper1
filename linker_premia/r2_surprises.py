"""r2 - Sorprese d'inflazione e linker (richiesta RR n.2), specificazione Maffei:
  sorpresa = YoY corrente - media mobile 10y dello YoY (dai CPI in cache, no download)
  lambda(T) = ISR(T) - BEI(T)  regredito sulla sorpresa, per scadenza ('by T').
Se lambda reagisce alle sorprese pur essendo i linker indicizzati, la spiegazione RR-
Maffei e' lo spostamento della domanda di protezione; il pattern dei beta lungo T =
segmentazione. Riportiamo anche Delta y_reale(T) sulla sorpresa (lettura diretta).
Output: an_surprise_regs.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import pandas as pd
import rp
from config import CACHE

# tenor allineati a Maffei (2-20yr). UK: la curva reale BoE parte da 2.5y -> minimo 2.5.
MATS = {"UK": (2.5, 5.0, 10.0, 15.0, 20.0), "US": (2.0, 5.0, 10.0, 15.0, 20.0)}
REAL = {"UK": lambda: rp.boe("real"), "US": lambda: rp.gsw("tips", MATS["US"])}

tabs = []
for mkt in ("US", "UK"):
    surp = rp.surprise_maffei(mkt)
    b, s, lam = rp.lam_gamma(mkt, MATS[mkt])
    try:
        liq = rp.liquidity()
    except FileNotFoundError:
        liq = None
        print("  (an_liq assente: regressione solo-sorpresa; lanciare r0 per VIX/MOVE)")
    tl = rp.reg_lambda(lam, surp, liq); tl.insert(0, "var", "lambda"); tl.insert(0, "mkt", mkt)
    dy = rp.reg_surprise(REAL[mkt](), pd.DataFrame({"maffei": surp}), MATS[mkt])
    dy = dy.rename(columns={"sorpresa": "var"}); dy["var"] = "dy_real"; dy.insert(0, "mkt", mkt)
    tabs += [tl, dy]
    print(f"\n=== {mkt} | lambda(T)=ISR-BEI su LIQUIDITA' + SORPRESA (Maffei), "
          f"{lam.index.min().date()}->{lam.index.max().date()} ===")
    print("  beta SORPRESA (al netto liquidita') [bp/unit]:  " +
          "  ".join(f"{int(r.mat)}y: {r.beta_surp:+.1f} (t={r.t_surp:+.1f})" for r in tl.itertuples()))
    if tl["beta_liq"].notna().any():
        print("  beta LIQUIDITA' [bp/sd]:                        " +
              "  ".join(f"{int(r.mat)}y: {r.beta_liq:+.1f} (t={r.t_liq:+.1f})" for r in tl.itertuples()))
        print("  R2 full vs solo-liquidita' (contributo sorprese): " +
              "  ".join(f"{int(r.mat)}y: {r.R2_full:.2f}/{r.R2_liq_only:.2f}" for r in tl.itertuples()))
    print("  Delta y_reale(T) [bp] su sorpresa:              " +
          "  ".join(f"{int(r.mat)}y: {r.beta_bp:+.1f} (t={r.t_NW:+.1f})" for r in dy.itertuples()))
    g = tl.sort_values("mat")
    print(f"  segmentazione (beta sorpresa): |corto| {abs(g.beta_surp.iloc[0]):.1f} vs "
          f"|lungo| {abs(g.beta_surp.iloc[-1]):.1f}"
          f" -> {'concentrata sul CORTO' if abs(g.beta_surp.iloc[0])>abs(g.beta_surp.iloc[-1]) else 'concentrata sul LUNGO'}")
pd.concat(tabs).to_csv(CACHE / "an_surprise_regs.csv", index=False)
print("\nsalvato: an_surprise_regs.csv")

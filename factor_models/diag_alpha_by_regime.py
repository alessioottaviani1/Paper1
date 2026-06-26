# -*- coding: utf-8 -*-
"""
diag_alpha_by_regime.py — DIAGNOSTICO (non per il paper).

Alpha dei 3 benchmark (Duarte / ActiveFI / Fung-Hsieh) decomposto per regime di
stress (livello iTraxx Europe Main 5Y, bps), confrontando:
  - full sample (1 alpha)
  - 2 bande  : NORMAL (z < HIGH_CUT) / HIGH (z >= HIGH_CUT)        <- com'e' ora (soglia 100)
  - 3 bande  : LOW (z < LOW_CUT) / MID / HIGH                       <- come su AEN (60/100)

Metodo: una sola OLS con dummy d'intercetta per regime e BETA COMUNI (no costante)
    r_t = sum_g alpha_g * D_g,t + beta' F_t + e_t
cosi' ogni alpha_g e' l'intercetta del regime g, beta stimate sull'intero campione
(robusto: i mesi HIGH sono pochi). Inferenza Newey-West HAC (auto-lag, come 03/07/oos).

Allineato ai 75 fattori nuovi: liste fattori CORRETTE (FH = SNPMRF/SCMLC/R10/BAAMTSY,
ActiveFI con EU_Term/US_Term -> 'Term'), iTraxx dal foglio 'cds' di bbg.xlsx.

Da mettere in factor_models/ e lanciare: python diag_alpha_by_regime.py
Author: Alessio Ottaviani, EDHEC.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path

from inference import auto_hac_lags
import factor_sources as src

# ----------------------------- parametri -----------------------------
REGION   = "eur"          # "eur" primario; "us" per controllo
LOW_CUT  = 60             # soglia LOW/MID (bps)
HIGH_CUT = 100            # soglia MID/HIGH (bps)  -> per provare 80, metti 80 qui (o in LOW_CUT)
FREQ     = "monthly"
ANN      = 12

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROC         = PROJECT_ROOT / "data" / "processed"

# liste fattori = nomi REALI dei CSV (post-fix)
FRAMEWORKS = {
    "Duarte":    (["Mkt-RF", "SMB", "HML", "UMD", "RS", "RI", "RB", "R2", "R5", "R10"],
                  "regression_data_{s}_{r}_{f}.csv"),
    "ActiveFI":  (["Term", "Global_Term", "Global_Aggregate", "Inflation_Linkers",
                   "Corporate_Credit", "Emerging_Debt", "Emerging_Currency", "UST_Volatility"],
                  "regression_data_active_fi_{s}_{r}_{f}.csv"),
    "FungHsieh": (["SNPMRF", "SCMLC", "PTFSBD", "PTFSFX", "PTFSCOM", "R10", "BAAMTSY"],
                  "regression_data_fung_hsieh_{s}_{r}_{f}.csv"),
}
STRATS = ["BTP_Italia", "iTraxx_Combined", "CDS_Bond_Basis"]


def stress_monthly():
    """iTraxx Europe Main 5Y (bps), mensile, dal foglio 'cds' di bbg.xlsx."""
    s = src.load_bloomberg("cds")["ITRX EUR CDSI GEN 5Y Corp"].dropna()
    return s.resample("ME").last().dropna().rename("ITRX")


def load(fw, strat):
    fp = PROC / FRAMEWORKS[fw][1].format(s=strat.lower(), r=REGION, f=FREQ)
    if not fp.exists():
        return None
    d = pd.read_csv(fp, index_col=0, parse_dates=True)
    return d.rename(columns={"EU_Term": "Term", "US_Term": "Term"})   # ActiveFI


def regime_alphas(y, F, dummies):
    """OLS r su [dummies | F] senza costante; HAC. Ritorna {regime:(alpha_ann,t)}, {regime:n}."""
    A = pd.concat([y.rename("y"), dummies, F], axis=1).dropna()
    cols = [c for c in dummies.columns if A[c].sum() > 0]    # salta regimi vuoti
    n = {c: int(A[c].sum()) for c in dummies.columns}
    if not cols or len(A) <= len(cols) + F.shape[1] + 2:
        return {c: (np.nan, np.nan) for c in dummies.columns}, n
    X = A[cols + list(F.columns)]
    m = sm.OLS(A["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": auto_hac_lags(len(A))})
    out = {c: (np.nan, np.nan) for c in dummies.columns}
    for c in cols:
        out[c] = (m.params[c] * ANN, m.tvalues[c])
    return out, n


def star(t):
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return ""
    a = abs(t)
    return "***" if a > 2.58 else "**" if a > 1.96 else "*" if a > 1.64 else ""


def cell(alpha, t, nobs=None):
    if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)):
        return "   n/a   "
    s = f"{alpha:+6.2f}{star(t):<3}(t={t:+.2f}"
    s += f",n={nobs})" if nobs is not None else ")"
    return s


def main():
    z = stress_monthly()
    print(f"iTraxx Main 5Y: {z.index.min().date()}..{z.index.max().date()}  "
          f"min={z.min():.0f}/med={z.median():.0f}/max={z.max():.0f} bps")
    print(f"Regioni={REGION.upper()}  cutoff LOW<{LOW_CUT}<=MID<{HIGH_CUT}<=HIGH  (alpha %% p.a., HAC NW)\n")

    for fw, (facs, _) in FRAMEWORKS.items():
        print("=" * 100)
        print(f"{fw}  ({REGION.upper()})")
        print("=" * 100)
        for strat in STRATS:
            d = load(fw, strat)
            if d is None or "Strategy_Return" not in d.columns:
                print(f"  {strat:16s}  (no CSV)")
                continue
            y = d["Strategy_Return"]
            F = d[[f for f in facs if f in d.columns]]
            if F.shape[1] < 2:
                print(f"  {strat:16s}  (solo {F.shape[1]} fattori nel CSV — controlla i nomi)")
                continue
            zt = z.reindex(y.index, method="nearest")

            full = sm.OLS(y, sm.add_constant(F)).fit(
                cov_type="HAC", cov_kwds={"maxlags": auto_hac_lags(len(y))})
            aF, tF = full.params["const"] * ANN, full.tvalues["const"]

            D2 = pd.DataFrame({"NORM": (zt < HIGH_CUT).astype(float),
                               "HIGH": (zt >= HIGH_CUT).astype(float)}, index=y.index)
            r2, n2 = regime_alphas(y, F, D2)

            D3 = pd.DataFrame({"LOW":  (zt < LOW_CUT).astype(float),
                               "MID":  ((zt >= LOW_CUT) & (zt < HIGH_CUT)).astype(float),
                               "HIGH": (zt >= HIGH_CUT).astype(float)}, index=y.index)
            r3, n3 = regime_alphas(y, F, D3)

            print(f"  {strat:16s}  K={F.shape[1]}  N={int(full.nobs)}")
            print(f"      full    {cell(aF, tF)}")
            print(f"      2-band  NORM {cell(*r2['NORM'], n2['NORM'])}   "
                  f"HIGH {cell(*r2['HIGH'], n2['HIGH'])}")
            print(f"      3-band  LOW  {cell(*r3['LOW'], n3['LOW'])}   "
                  f"MID  {cell(*r3['MID'], n3['MID'])}   "
                  f"HIGH {cell(*r3['HIGH'], n3['HIGH'])}")
        print()
    print("Nota: beta comuni tra regimi (dummy d'intercetta); regime classificato sul livello")
    print("iTraxx contemporaneo. Significativita' HAC: * 10%, ** 5%, *** 1%.")


if __name__ == "__main__":
    main()

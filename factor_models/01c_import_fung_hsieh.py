"""
Script 1c: Import Fung & Hsieh (2004) 7-Factor Model - US & EUR - dal NUOVO DATABASE
=====================================================================================
Costruisce i dataset di regressione FH per TUTTE LE STRATEGIE pescando dal nuovo
database (build_all_factors.py): EUR-native dal parquet, US dal RAW (Bloomberg
tr_indices + Hsieh TF + Ken French RF), eurizzato via GHS. Stessa logica di 01a.

Reference: Fung & Hsieh (2004) "Hedge Fund Benchmarks: A Risk-Based Approach", FAJ
Trend-following: Fung & Hsieh (2001), RFS — David Hsieh data library (TF-Fac.xls).

I 7 FATTORI (nomi del paper FH, TRANNE il bond factor — vedi nota):
  SNPMRF   Equity market   = S&P 500 total-return in eccesso (R(SPXT) - rf)
  SCMLC    Size spread     = Russell 2000 TR - S&P 500 TR (small - large, zero-cost)
  PTFSBD   Bond trend      \\
  PTFSFX   FX trend         > Hsieh PTFS lookback straddles (globali, local-ccy, NO FX)
  PTFSCOM  Commodity trend /
  R10      Bond market     = 10Y govt total-return (TRADABILE) -- vedi nota
  BAAMTSY  Credit spread   = R(BBB corp) - R(Treasury) = CRED_SPR (corp - govt, tradabile)

NOTA SUL BOND FACTOR (perche' R10 e non "Δ10Y"/"BD10RET"):
  In FH il "bond market factor" e' la VARIAZIONE del rendimento del Treasury 10Y,
  cioe' una variazione di yield: NON e' il rendimento di una posizione -> fallisce
  "intercetta = alpha di Jensen" (Tessaromatis). Lo sostituiamo con il tradabile
  R10 (rendimento del portafoglio governativo 10Y); idem per il credito, dove
  usiamo CRED_SPR (corporate - Treasury) invece della variazione di spread Δ(Baa-10Y).
  Tutti i 7 regressori sono cosi' excess return realizzati -> intercetta = alpha.

VERSIONI EUR vs US:
  - EUR: SNPMRF=MKT_EU, SCMLC=SMB_EU (FF Dev-Europe; non esiste un Russell europeo),
         R10=R10_EU, BAAMTSY=CRED_SPR_EU, PTFS* globali (dal parquet).
  - US : SNPMRF=R(SPXT)-rf, SCMLC=R(RUTTR)-R(SPXT), R10=R(MLTAU10E),
         BAAMTSY=CRED_SPR_US=R(LCB1TRUU)-R(SPBDU1BT), PTFS* globali (dal RAW Hsieh).
         I _US sono eurizzati via GHS e NON entrano nel parquet/AEN (come in 01a).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETRI
# ============================================================================

FACTOR_FREQ = "monthly"  # "daily", "weekly", "monthly"

# ============================================================================
# DEFINIZIONE STRATEGIE
# ============================================================================

STRATEGIES = {
    'BTP_Italia': 'btp_italia/index_daily.csv',
    'iTraxx_Combined': 'itraxx_combined/index_daily.csv',
    'CDS_Bond_Basis': 'cds_bond_basis/index_daily.csv'
}

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FATTORI DAL PARQUET (build_all_factors.py): EUR-native + US (dal RAW)
# ============================================================================
import statsmodels.api as sm
import sys
sys.path.insert(0, str(Path(__file__).parent))   # per importare i moduli sibling
from build_all_factors import (to_monthly_last, monthly_return,
                                ghs_longshort, ghs_market, tr_index_spread)
import factor_sources as src

FACTORS_PARQUET = PROCESSED_DATA_DIR / "all_factors_monthly.parquet"
assert FACTORS_PARQUET.exists(), f"manca {FACTORS_PARQUET} (gira prima build_all_factors.py)"
_fac = pd.read_parquet(FACTORS_PARQUET)

# Fung & Hsieh 7-factor -> colonne del nuovo database (EUR-native dal parquet).
# Nomi = mnemonici del paper FH; bond factor = R10 (tradabile) al posto del Δ10Y.
FH_EUR = {"SNPMRF": "MKT_EU",  "SCMLC": "SMB_EU",
          "PTFSBD": "PTFSBD",  "PTFSFX": "PTFSFX",  "PTFSCOM": "PTFSCOM",
          "R10": "R10_EU",     "BAAMTSY": "CRED_SPR_EU"}
FH_ORDER = ["SNPMRF", "SCMLC", "PTFSBD", "PTFSFX", "PTFSCOM", "R10", "BAAMTSY"]

def _block(mapping):
    cols = {k: v for k, v in mapping.items() if v in _fac.columns}
    return None if len(cols) < len(mapping) else _fac[list(cols.values())].rename(
        columns={v: k for k, v in cols.items()})

factors_eur_final = _block(FH_EUR)

def _build_us_fh():
    """7 fattori FH lato US dal RAW (Bloomberg/bbg.xlsx tr_indices + Hsieh + Ken French RF),
    eurizzati via GHS. Riusa le helper di build_all_factors SENZA far entrare i _US nel parquet/AEN.
    SNPMRF = R(SPXT)-rf (S&P500 TR in eccesso); SCMLC = R(RUTTR)-R(SPXT) (Russell-S&P, zero-cost);
    R10 = R(MLTAU10E) (UST 10Y excess-return index); BAAMTSY = CRED_SPR_US = R(LCB1TRUU)-R(SPBDU1BT);
    PTFS* = Hsieh trend-following (globali: local currency, NESSUN aggiustamento FX, per D. Hsieh)."""
    try:
        # rates & FX: stesse formule di build_all_factors.build()
        bbg_rates = src.load_bloomberg("rates_fx")
        eurusd = to_monthly_last(bbg_rates["EURUSD Curncy"])
        getb1  = to_monthly_last(bbg_rates["GETB1 Index"])
        r_fx   = eurusd.pct_change()
        _days  = pd.Series(getb1.index.day, index=getb1.index)
        rf_eur = getb1.shift(1) * _days / 360.0
        bbg_tr = src.load_bloomberg("tr_indices")
        ff5_us = src.load_french("F-F_Research_Data_5_Factors_2x3.csv")   # solo per RF (US T-bill, %)
        rf_usd = ff5_us["RF"]
        hsieh  = src.load_hsieh()
        U = {}
        # equity market (excess) e size (Russell - S&P, zero-cost) dai TOTAL-RETURN Bloomberg
        spx = monthly_return(bbg_tr["SPXT Index"]) * 100      # S&P 500 total return, %
        rut = monthly_return(bbg_tr["RUTTR Index"]) * 100     # Russell 2000 total return, %
        U["SNPMRF"]  = ghs_market(spx - rf_usd, rf_usd, r_fx, rf_eur)      # R(SPXT)-rf, poi eurizz.
        U["SCMLC"]   = ghs_longshort(rut - spx, r_fx)                      # gia' excess/zero-cost
        # PTFS (globali): decimale -> %, nessun FX (per Hsieh)
        U["PTFSBD"]  = to_monthly_last(hsieh["PTFSBD"])  * 100
        U["PTFSFX"]  = to_monthly_last(hsieh["PTFSFX"])  * 100
        U["PTFSCOM"] = to_monthly_last(hsieh["PTFSCOM"]) * 100
        # bond market: 10Y govt TR tradabile (UST 10Y excess-return index -> niente -rf), eurizz.
        U["R10"]     = ghs_longshort(monthly_return(bbg_tr["MLTAU10E Index"]) * 100, r_fx)
        # credit spread: CRED_SPR_US = R(BBB) - R(Treasury), eurizz. (come build_all_factors)
        U["BAAMTSY"] = ghs_longshort(
            tr_index_spread(bbg_tr["LCB1TRUU Index"], bbg_tr["SPBDU1BT Index"]) * 100, r_fx)
        return pd.DataFrame(U)[FH_ORDER].dropna(how="all")
    except (KeyError, FileNotFoundError) as e:
        print(f"   ⚠️  US FH non costruiti ({type(e).__name__}: {e}); proseguo solo EUR.")
        return None

factors_us_final = _build_us_fh()   # dal RAW excel, NON dal parquet (fuori dall'AEN)
assert factors_eur_final is not None, "mancano colonne FH _EU nel parquet (MKT_EU/SMB_EU/PTFS*/R10_EU/CRED_SPR_EU)"
print(f"✅ EUR: {list(factors_eur_final.columns)}")
print(f"   US : {'pronti' if factors_us_final is not None else 'non ancora (mancano i RAW US)'}")

for region, F in [("eur", factors_eur_final), ("us", factors_us_final)]:
    if F is not None:
        F.to_csv(PROCESSED_DATA_DIR / f"all_fung_hsieh_factors_{region}_{FACTOR_FREQ}.csv")

FACTOR_COLS = FH_ORDER
for name, rel in STRATEGIES.items():
    p = RESULTS_DIR / rel
    if not p.exists():
        print(f"skip {name}: {p} assente"); continue
    s = (pd.read_csv(p, index_col=0, parse_dates=True)[["index_return"]]
           .resample("M").apply(lambda x: ((1 + x/100).prod() - 1) * 100)
           .rename(columns={"index_return": "Strategy_Return"}))
    # scrivi i dataset di regressione (gli stessi nomi che 02c si aspetta)
    for region, F in [("eur", factors_eur_final), ("us", factors_us_final)]:
        if F is None: continue
        reg = s.join(F, how="inner").dropna()
        reg.to_csv(PROCESSED_DATA_DIR / f"regression_data_fung_hsieh_{name.lower()}_{region}_{FACTOR_FREQ}.csv")
    # alpha rapido (HAC) per EUR e US, cosi' la regressione "gira" subito
    for region, F in [("EUR", factors_eur_final), ("US ", factors_us_final)]:
        if F is None: continue
        d = s.join(F, how="inner").dropna()
        if len(d) <= 12: continue
        rr = sm.OLS(d["Strategy_Return"], sm.add_constant(d[FACTOR_COLS])).fit(
            cov_type="HAC", cov_kwds={"maxlags": int(len(d)**0.25)})
        pv = rr.pvalues["const"]
        sig = "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .10 else ""
        lbl = name if region == "EUR" else ""
        print(f"  {lbl:16s} {region} α={rr.params['const']*12:+.2f}%/yr "
              f"(t={rr.tvalues['const']:+.2f}){sig}  R²adj={rr.rsquared_adj:.2f}  N={int(rr.nobs)}")
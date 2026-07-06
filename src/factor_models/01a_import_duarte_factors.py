"""
Script 1: Import All Risk Factors - US & EUROPEAN VERSION - MULTI STRATEGY
===========================================================================
Importa TUTTI i fattori di rischio per le regressioni, sia in versione US che EUR.
Crea dataset di regressione per TUTTE LE STRATEGIE:
1. BTP-Italia
2. iTraxx Main
3. iTraxx SnrFin
4. iTraxx SubFin
5. iTraxx Xover
6. iTraxx Combined

FATTORI IMPORTATI (US):
1-4. Fama-French: Mkt-RF, SMB, HML, UMD (da Kenneth French)
5. RS: S&P Bank Stock Index (da Bloomberg)
6. RI: Industrial Bonds A/BBB (da FRED - composite 50-50)
7. RB: Corporate Bonds A/BBB (composite 50-50)
8-10. R2, R5, R10: Treasury portfolios 2Y/5Y/10Y (da Bloomberg)

FATTORI IMPORTATI (EUR):
1-4. Fama-French EUR: Mkt-RF, SMB, HML, UMD (European markets, convertiti in EUR)
5-10. RS, RI, RB, R2, R5, R10: fattori EUR specifici o fallback US

CONVERSIONI EUR:
- SMB, HML, UMD: LS_t^EUR = LS_t^USD / (1 + r_t^{USD/EUR})
- Mkt-RF: formula specifica con conversione FX

FIX APPLICATO:
- RI_RB.xlsx: skiprows=2 per saltare header rows
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

EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

FACTORS_FILE = EXTERNAL_DATA_DIR / "Duarte_factors.xlsx"

# ============================================================================
# FATTORI DAL PARQUET (build_all_factors.py): EUR-native + US (se già generati)
# ============================================================================
import sys
import importlib
import statsmodels.api as sm                        # serve al loop alpha HAC in fondo (sm.OLS)
sys.path.insert(0, str(Path(__file__).parent))   # cartella factor_models: per i moduli sibling
import factor_sources as src                       # serve a _build_us_duarte (src.load_bloomberg, src.CDS_IDX, ...)
_baf = importlib.import_module("00_build_all_factors")
to_monthly_last, monthly_return = _baf.to_monthly_last, _baf.monthly_return
ghs_longshort, ghs_market       = _baf.ghs_longshort, _baf.ghs_market

FACTORS_PARQUET = PROCESSED_DATA_DIR / "all_factors_monthly.parquet"
assert FACTORS_PARQUET.exists(), f"manca {FACTORS_PARQUET} (gira prima build_all_factors.py)"
_fac = pd.read_parquet(FACTORS_PARQUET)

DUARTE_EUR = {"Mkt-RF":"MKT_EU","SMB":"SMB_EU","HML":"HML_EU","UMD":"UMD_EU",
              "RS":"RS_EU","RI":"RI_EU","RB":"RB_EU","R2":"R2_EU","R5":"R5_EU","R10":"R10_EU"}
DUARTE_US  = {"Mkt-RF":"MKT_US","SMB":"SMB_US","HML":"HML_US","UMD":"UMD_US",
              "RS":"RS_US","RI":"RI_US","RB":"RB_US","R2":"R2_US","R5":"R5_US","R10":"R10_US"}

def _block(mapping):
    cols = {k:v for k,v in mapping.items() if v in _fac.columns}
    return None if len(cols) < len(mapping) else _fac[list(cols.values())].rename(
        columns={v:k for k,v in cols.items()})

factors_eur_final = _block(DUARTE_EUR)

def _build_us_duarte():
    """10 fattori Duarte US dal RAW (Bloomberg/bbg.xlsx tr_indices + Ken French US),
    eurizzati via GHS. Riusa le helper di build_all_factors SENZA far entrare i _US nel parquet/AEN."""
    try:
        # rates & FX: stesse formule di build_all_factors.build() (righe 442-453)
        bbg_rates = src.load_bloomberg("rates_fx")
        eurusd = to_monthly_last(bbg_rates["EURUSD Curncy"])
        getb1  = to_monthly_last(bbg_rates["GETB1 Index"])
        r_fx   = eurusd.pct_change()
        _days  = pd.Series(getb1.index.day, index=getb1.index)
        rf_eur = getb1.shift(1) * _days / 360.0
        rf_usd = src.load_french("Europe_5_Factors.csv")["RF"]            # US T-bill, %
        bbg_tr = src.load_bloomberg("tr_indices")
        ff5_us = src.load_french("F-F_Research_Data_5_Factors_2x3.csv")   # US FF (in ken_french/)
        mom_us = src.load_french("F-F_Momentum_Factor.csv")              # US momentum
        U = {}
        U["Mkt-RF"] = ghs_market(ff5_us["Mkt-RF"], ff5_us["RF"], r_fx, rf_eur)
        U["SMB"]    = ghs_longshort(ff5_us["SMB"], r_fx)
        U["HML"]    = ghs_longshort(ff5_us["HML"], r_fx)
        _wml = mom_us["WML"] if "WML" in mom_us.columns else mom_us["Mom"]  # US: 'Mom'; Europe: 'WML'
        U["UMD"]    = ghs_longshort(_wml, r_fx)
        # Treasury US = UBS excess-return (come i MLTAG* EUR): niente -rf, solo eurizzazione
        U["R2"]  = ghs_longshort(monthly_return(bbg_tr["MLTAUS2E Index"]) * 100, r_fx)
        U["R5"]  = ghs_longshort(monthly_return(bbg_tr["MLTAUS5E Index"]) * 100, r_fx)
        U["R10"] = ghs_longshort(monthly_return(bbg_tr["MLTAU10E Index"]) * 100, r_fx)
        # single-leg total-return: -rf_usd poi eurizzazione
        U["RS"] = ghs_longshort(monthly_return(bbg_tr["SPTR5BNK Index"]) * 100 - rf_usd, r_fx)
        U["RI"] = ghs_longshort((0.5 * monthly_return(bbg_tr["I04945US Index"]) * 100
                               + 0.5 * monthly_return(bbg_tr["I04948US Index"]) * 100) - rf_usd, r_fx)
        U["RB"] = ghs_longshort((0.5 * monthly_return(bbg_tr["I04941US Index"]) * 100
                               + 0.5 * monthly_return(bbg_tr["I04944US Index"]) * 100) - rf_usd, r_fx)  # Finance A + Baa (50-50)
        return pd.DataFrame(U)[list(DUARTE_US.keys())].dropna(how="all")
    except (KeyError, FileNotFoundError) as e:
        print(f"   ⚠️  US Duarte non costruiti ({type(e).__name__}: {e}); proseguo solo EUR.")
        return None

factors_us_final = _build_us_duarte()   # dal RAW excel, NON dal parquet (fuori dall'AEN)
assert factors_eur_final is not None, "mancano colonne _EU nel parquet"
print(f"✅ EUR: {list(factors_eur_final.columns)}")
print(f"   US : {'pronti' if factors_us_final is not None else 'non ancora (mancano _US)'}")

for region, F in [("eur", factors_eur_final), ("us", factors_us_final)]:
    if F is not None:
        F.to_csv(PROCESSED_DATA_DIR / f"all_risk_factors_{region}_{FACTOR_FREQ}.csv")

FACTOR_COLS = list(DUARTE_EUR)
for name, rel in STRATEGIES.items():
    p = RESULTS_DIR / rel
    if not p.exists():
        print(f"skip {name}: {p} assente"); continue
    s = (pd.read_csv(p, index_col=0, parse_dates=True)[["index_return"]]
           .resample("M").apply(lambda x: ((1 + x/100).prod() - 1) * 100)
           .rename(columns={"index_return": "Strategy_Return"}))
    # scrivi i dataset di regressione (gli stessi nomi che 02a si aspetta)
    for region, F in [("eur", factors_eur_final), ("us", factors_us_final)]:
        if F is None: continue
        reg = s.join(F, how="inner").dropna()
        reg.to_csv(PROCESSED_DATA_DIR / f"regression_data_{name.lower()}_{region}_{FACTOR_FREQ}.csv")
    # alpha rapido (HAC) per EUR e US, così la regressione "gira" subito
    for region, F in [("EUR", factors_eur_final), ("US ", factors_us_final)]:
        if F is None: continue
        d = s.join(F, how="inner").dropna()
        if len(d) <= 12: continue
        rr = sm.OLS(d["Strategy_Return"], sm.add_constant(d[FACTOR_COLS])).fit(
            cov_type="HAC", cov_kwds={"maxlags": int(len(d)**0.25)})
        pv = rr.pvalues["const"]
        sig = "***" if pv<.01 else "**" if pv<.05 else "*" if pv<.10 else ""
        lbl = name if region == "EUR" else ""
        print(f"  {lbl:16s} {region} α={rr.params['const']*12:+.2f}%/yr "
              f"(t={rr.tvalues['const']:+.2f}){sig}  R²adj={rr.rsquared_adj:.2f}  N={int(rr.nobs)}")
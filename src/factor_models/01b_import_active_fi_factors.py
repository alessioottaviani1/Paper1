"""
Script 1b: Import Active Fixed Income factors - US & EUR - MULTI STRATEGY
==========================================================================
Benchmark "Active Fixed Income Illusions" (Brooks, Gould, Richardson, JFI 2020).
Stessa architettura di 01a/01c: lato EUR dal parquet (build_all_factors.py) via
_block(); lato US dal RAW (Bloomberg/bbg.xlsx tr_indices) eurizzato via GHS, SENZA
far entrare i _US nel parquet/AEN.

GLI 8 FATTORI (Exhibit 3 del paper):
1. Term            US/EU Treasury excess return (duration premium)
2. Global Term     Bloomberg Global Treasury (hedged) excess return
3. Global Aggregate Bloomberg Global Aggregate (hedged) excess return
4. Inflation-Linkers Bloomberg Global Inflation-Linked (hedged) excess return
5. Corporate Credit High-Yield in eccesso (gamba leveraged-loan SCARTATA)
6. Emerging Debt   EM bond index excess return
7. Emerging Currency EM FX basket excess return
8. UST Volatility  "selling volatility on rates" (varswap 10Y Treasury USA)

NUOVI TICKER US (tr_indices) -> simmetrici agli EUR-native del parquet:
  US_Term            LUATTRUU   (US Treasury TR)              ~ TERM_EU (I01656EU)
  Global_Term        BTSYTRUH   (Global Treasury USD-hedged)  ~ GLOBAL_TERM (H00023EU)
  Global_Aggregate   LEGATRUH   (Global Agg USD-hedged)       ~ GLOBAL_AGG (LEGATREH)
  Inflation_Linkers  LF94TRUH   (Global ILB USD-hedged)       ~ INFL_LINK (LF94TREH)
  Corporate_Credit   LF98TRUU   (US High Yield)               ~ HY_CORP (H02500EU)
  Emerging_Debt      EMUSTRUU   (EM USD Aggregate)            ~ EMERG_DEBT (H04386EU)
  Emerging_Currency  MXEF0CX0   (EM FX) -- condiviso US/EUR
  UST_Volatility     IV_TSY     (varswap TY1, dal parquet) -- condiviso US/EUR

CORPORATE CREDIT — gamba leveraged loan SCARTATA:
  Nel paper Corporate Credit = 50% (HY in eccesso su Treasury duration-matched)
  + 50% (S&P/LSTA Leveraged Loan in eccesso sul 3M Libor). Il leg loan NON e'
  tradabile in senso Tessaromatis (indice non investibile, settlement in
  settimane, prezzi da quote -> return smoothing/autocorrelazione, illiquidita'
  non raccoglibile). Lo scartiamo: Corporate Credit = sola gamba HY.
  - US : LF98TRUU (Bloomberg US Corporate High Yield) in eccesso sul cash USD.
  - EUR: HY_CORP = H02500EU (Pan-European HY) in eccesso sul cash EUR (parquet;
         loan SPBDEL gia' fuori per costruzione).
  CAVEAT: e' HY in eccesso sul CASH, non su Treasury duration-matched come nel
  paper; la differenza e' il rendimento di un Treasury di pari duration (~4y).
  Per la versione duration-matched basta sottrarre il 5Y Treasury excess
  (MLTAUS5E lato US, MLTAGB5E lato EUR), gia' nel bbg.

CONVERSIONI EUR (US -> EUR), come 01a:
  - excess self-financing: LS_t^EUR = LS_t^USD / (1 + r_t^{EUR/USD})  [ghs_longshort]
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

# ============================================================================
# FATTORI DAL PARQUET (build_all_factors.py): EUR-native; US dal RAW (eurizzato)
# ============================================================================
import statsmodels.api as sm
import sys
import importlib
sys.path.insert(0, str(Path(__file__).parent))   # per importare i moduli sibling
import factor_sources as src
_baf = importlib.import_module("00_build_all_factors")   # nome con cifra iniziale: import diretto impossibile
to_monthly_last, monthly_return, ghs_longshort = _baf.to_monthly_last, _baf.monthly_return, _baf.ghs_longshort

FACTORS_PARQUET = PROCESSED_DATA_DIR / "all_factors_monthly.parquet"
assert FACTORS_PARQUET.exists(), f"manca {FACTORS_PARQUET} (gira prima build_all_factors.py)"
_fac = pd.read_parquet(FACTORS_PARQUET)

# Active FI (Brooks 2020) -> colonne del nuovo database (lato EUR).
# Solo il Term e' regione-specifico (US_Term / EU_Term); gli altri nomi-colonna
# sono identici in US/EUR: 02b normalizza US_Term/EU_Term -> "Term".
AFI_EUR = {"EU_Term": "TERM_EU",  "Global_Term": "GLOBAL_TERM",
           "Global_Aggregate": "GLOBAL_AGG", "Inflation_Linkers": "INFL_LINK",
           "Corporate_Credit": "HY_CORP",    # sola gamba HY (loan SPBDEL gia' fuori)
           "Emerging_Debt": "EMERG_DEBT",    "Emerging_Currency": "EMERG_FX",
           "UST_Volatility": "IV_BUND"}    # euro-area counterpart of the Brooks rates-vol leg
EUR_ORDER = ["EU_Term", "Global_Term", "Global_Aggregate", "Inflation_Linkers",
             "Corporate_Credit", "Emerging_Debt", "Emerging_Currency", "UST_Volatility"]
US_ORDER  = ["US_Term", "Global_Term", "Global_Aggregate", "Inflation_Linkers",
             "Corporate_Credit", "Emerging_Debt", "Emerging_Currency", "UST_Volatility"]

def _block(mapping, order):
    cols = {k: v for k, v in mapping.items() if v in _fac.columns}
    return None if len(cols) < len(mapping) else _fac[[mapping[k] for k in order]].rename(
        columns={v: k for k, v in mapping.items()})[order]

factors_eur_final = _block(AFI_EUR, EUR_ORDER)

def _build_us_afi():
    """8 fattori Active FI US dal RAW (bbg tr_indices, ticker USD / USD-hedged),
    eurizzati via GHS. La vol (IV_TSY) e' gia' nel parquet (varswap dai futures).
    Riusa le helper di build_all_factors SENZA far entrare i _US nel parquet/AEN."""
    try:
        # FX: stessa logica di build_all_factors.build()
        bbg_rates = src.load_bloomberg("rates_fx")
        r_fx   = to_monthly_last(bbg_rates["EURUSD Curncy"]).pct_change()
        rf_usd = src.load_french("Europe_5_Factors.csv")["RF"]   # US T-bill, %
        bbg_tr = src.load_bloomberg("tr_indices")
        U = {}
        # single-leg total-return: -rf_usd (excess sul cash USD) poi eurizzazione GHS
        U["US_Term"]           = ghs_longshort(monthly_return(bbg_tr["LUATTRUU Index"]) * 100 - rf_usd, r_fx)
        U["Global_Term"]       = ghs_longshort(monthly_return(bbg_tr["BTSYTRUH Index"]) * 100 - rf_usd, r_fx)
        U["Global_Aggregate"]  = ghs_longshort(monthly_return(bbg_tr["LEGATRUH Index"]) * 100 - rf_usd, r_fx)
        U["Inflation_Linkers"] = ghs_longshort(monthly_return(bbg_tr["LF94TRUH Index"]) * 100 - rf_usd, r_fx)
        U["Corporate_Credit"]  = ghs_longshort(monthly_return(bbg_tr["LF98TRUU Index"]) * 100 - rf_usd, r_fx)  # US HY, loan dropped
        U["Emerging_Debt"]     = ghs_longshort(monthly_return(bbg_tr["EMUSTRUU Index"]) * 100 - rf_usd, r_fx)
        U["Emerging_Currency"] = ghs_longshort(monthly_return(bbg_tr["MXEF0CX0 Index"]) * 100 - rf_usd, r_fx)
        # vol sui tassi USA: identica per US/EUR, gia' nel parquet (varswap TY1)
        U["UST_Volatility"]    = _fac["IV_TSY"]
        return pd.DataFrame(U)[US_ORDER].dropna(how="all")
    except (KeyError, FileNotFoundError) as e:
        print(f"   ⚠️  US Active FI non costruiti ({type(e).__name__}: {e}); proseguo solo EUR.")
        return None

factors_us_final = _build_us_afi()   # dal RAW bbg, NON dal parquet (fuori dall'AEN)
assert factors_eur_final is not None, "mancano colonne Active FI _EU nel parquet"
for _F in (factors_eur_final, factors_us_final):
    if _F is not None:
        _F["UST_Volatility"] = _F["UST_Volatility"].astype(float) * 100.0

print(f"✅ EUR: {list(factors_eur_final.columns)}")
print(f"   US : {'pronti' if factors_us_final is not None else 'non ancora (mancano ticker US nel bbg)'}")

for region, F in [("eur", factors_eur_final), ("us", factors_us_final)]:
    if F is not None:
        F.to_csv(PROCESSED_DATA_DIR / f"active_fi_factors_{region}_{FACTOR_FREQ}.csv")

for name, rel in STRATEGIES.items():
    p = RESULTS_DIR / rel
    if not p.exists():
        print(f"skip {name}: {p} assente"); continue
    s = (pd.read_csv(p, index_col=0, parse_dates=True)[["index_return"]]
           .resample("M").apply(lambda x: ((1 + x/100).prod() - 1) * 100)
           .rename(columns={"index_return": "Strategy_Return"}))
    # scrivi i dataset di regressione (gli stessi nomi che 02b si aspetta)
    for region, F in [("eur", factors_eur_final), ("us", factors_us_final)]:
        if F is None: continue
        reg = s.join(F, how="inner").dropna()
        reg.to_csv(PROCESSED_DATA_DIR / f"regression_data_active_fi_{name.lower()}_{region}_{FACTOR_FREQ}.csv")
    # alpha rapido (HAC) per EUR e US, cosi' la regressione "gira" subito
    for region, F, order in [("EUR", factors_eur_final, EUR_ORDER), ("US ", factors_us_final, US_ORDER)]:
        if F is None: continue
        d = s.join(F, how="inner").dropna()
        if len(d) <= 12: continue
        rr = sm.OLS(d["Strategy_Return"], sm.add_constant(d[order])).fit(
            cov_type="HAC", cov_kwds={"maxlags": int(len(d)**0.25)})
        pv = rr.pvalues["const"]
        sig = "***" if pv < .01 else "**" if pv < .05 else "*" if pv < .10 else ""
        lbl = name if region == "EUR" else ""
        print(f"  {lbl:16s} {region} α={rr.params['const']*12:+.2f}%/yr "
              f"(t={rr.tvalues['const']:+.2f}){sig}  R²adj={rr.rsquared_adj:.2f}  N={int(rr.nobs)}")

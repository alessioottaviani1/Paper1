"""
================================================================================
Stationarity Check (ENHANCED for Financial Series) - MINIMAL OUTPUT
================================================================================
Analisi basata su ADF + KPSS (level 'c' e trend 'ct').

Classificazione:
- STATIONARY:        ADF rejects unit root  AND  KPSS(c) does NOT reject
  -> non stampiamo nulla (pulizia)

- NON-STATIONARY:    ADF does NOT reject    AND  KPSS(c) rejects
  -> entrambi concordano sulla non-stazionarietà (Strong)

- TREND-STATIONARY:  ADF rejects unit root  AND  KPSS(c) rejects  AND  KPSS(ct) does NOT reject
  -> stazionaria solo attorno a un trend deterministico

- CONFLICT/PERSISTENT: tutti gli altri casi "non risolti" (serie molto persistenti / bassa potenza test)

Output:
- Stampa SOLO i fattori NON chiaramente STATIONARY, con p-value test.
================================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, kpss
import warnings

warnings.filterwarnings("ignore")

# =========================
# CONFIG
# =========================
ALPHA = 0.05

# Path default + fallback (stesso stile che usi nel progetto)
FACTORS_PATH = Path(__file__).resolve().parent / "data" / "processed" / "all_factors_monthly.parquet"
_candidates = [
    FACTORS_PATH,
    Path(__file__).resolve().parents[2] / "data" / "processed" / "all_factors_monthly.parquet",
    Path("/mnt/user-data/uploads/all_factors_monthly.parquet"),
]
for _p in _candidates:
    if _p.exists():
        FACTORS_PATH = _p
        break


def run_tests(series):
    s = series.dropna().astype(float)

    # --- ADF (H0: unit root / non-stationary) ---
    adf_res = adfuller(s, autolag="AIC")
    adf_p = float(adf_res[1])
    adf_reject = adf_p < ALPHA  # True => stationary (reject unit root)

    # --- KPSS level (H0: level-stationary) ---
    kpss_c = kpss(s, regression="c", nlags="auto")
    kpss_c_p = float(kpss_c[1])
    kpss_c_reject = kpss_c_p < ALPHA  # True => rejects stationarity

    # --- KPSS trend (H0: trend-stationary) ---
    # (Serve per distinguere trend-stationary da non-stationary / conflict)
    kpss_ct = kpss(s, regression="ct", nlags="auto")
    kpss_ct_p = float(kpss_ct[1])
    kpss_ct_reject = kpss_ct_p < ALPHA

    return adf_p, adf_reject, kpss_c_p, kpss_c_reject, kpss_ct_p, kpss_ct_reject


def classify(adf_reject, kpss_c_reject, kpss_ct_reject):
    # Clearly stationary
    if adf_reject and (not kpss_c_reject):
        return "STATIONARY"

    # Strong non-stationary agreement
    if (not adf_reject) and kpss_c_reject:
        return "!!! NON-STATIONARY (Strong) !!!"

    # Trend-stationary: not stationary around a level, but stationary around a trend
    if adf_reject and kpss_c_reject and (not kpss_ct_reject):
        return "TREND-STATIONARY (detrend ok)"

    # Everything else: typical for persistent financial series / low power tests
    return "? CONFLICT / PERSISTENT ?"


def main():
    if not FACTORS_PATH.exists():
        raise FileNotFoundError(f"File non trovato: {FACTORS_PATH}")

    factors = pd.read_parquet(FACTORS_PATH)

    print(f"{'FACTOR':<25} | {'CLASSIFICATION':<30} | {'ADF p':>8} | {'KPSS(c) p':>10} | {'KPSS(ct) p':>11}")
    print("-" * 95)

    for col in sorted(factors.columns):
        s = factors[col]

        # Skip too-short / constant (silenzioso)
        if len(s.dropna()) < 36 or np.isclose(s.dropna().var(), 0.0):
            continue

        try:
            adf_p, adf_reject, kpss_c_p, kpss_c_reject, kpss_ct_p, kpss_ct_reject = run_tests(s)
            status = classify(adf_reject, kpss_c_reject, kpss_ct_reject)
        except Exception:
            continue

        # Print ONLY non-STATIONARY cases (as requested)
        if status != "STATIONARY":
            print(
                f"{col:<25} | {status:<30} | "
                f"{adf_p:8.4f} | {kpss_c_p:10.4f} | {kpss_ct_p:11.4f}"
            )


if __name__ == "__main__":
    main()

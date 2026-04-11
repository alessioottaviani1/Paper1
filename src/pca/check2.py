"""
================================================================================
Stationarity Check (ENHANCED for Financial Series) - MINIMAL TERMINAL OUTPUT
================================================================================
ADF + KPSS with both level ('c') and trend ('ct') specifications.

Goal:
- Print ONLY non-clearly-stationary series (minimal terminal output).
- Provide a robust classification to guide transformations before PCA / alpha tests.

Classification (using ALPHA):
- STATIONARY (level):        ADF(c) rejects unit root AND KPSS(c) does NOT reject
  -> do NOT print

- NON-STATIONARY (Strong):   ADF(c) does NOT reject AND KPSS(c) rejects
  -> print

- TREND-STATIONARY:          ADF(ct) rejects unit root AND KPSS(c) rejects AND KPSS(ct) does NOT reject
  -> print

- CONFLICT / PERSISTENT:     everything else
  -> print

Notes:
- Adds ADF with trend ("ct") to properly identify trend-stationarity.
- KPSS uses 'auto' lags, with a safe fallback if it fails (common with 'ct').

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
MIN_OBS = 60  # recommended for monthly data; lower if you must (e.g., 36)

# Path default + fallback
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


def _kpss_safe(s: pd.Series, regression: str):
    """
    KPSS can fail or be unstable for financial series, especially with regression='ct' and nlags='auto'.
    We try auto first, then fallback to a conservative fixed lag.
    Returns (p_value, rejected_stationarity_bool).
    """
    n = len(s)
    try:
        res = kpss(s, regression=regression, nlags="auto")
        p = float(res[1])
        return p, (p < ALPHA)
    except Exception:
        # Fallback lag (monthly data): up to 12, but not too large relative to sample size
        # Conservative: min(12, n//10) with lower bound 1
        nlags = max(1, min(12, n // 10))
        res = kpss(s, regression=regression, nlags=nlags)
        p = float(res[1])
        return p, (p < ALPHA)


def run_tests(series: pd.Series):
    s = series.dropna().astype(float)
    n = len(s)

    # --- ADF (H0: unit root / non-stationary) ---
    # ADF with constant
    adf_c = adfuller(s, regression="c", autolag="AIC")
    adf_c_p = float(adf_c[1])
    adf_c_reject = adf_c_p < ALPHA

    # ADF with constant + trend
    adf_ct = adfuller(s, regression="ct", autolag="AIC")
    adf_ct_p = float(adf_ct[1])
    adf_ct_reject = adf_ct_p < ALPHA

    # --- KPSS (H0: stationary) ---
    kpss_c_p, kpss_c_reject = _kpss_safe(s, regression="c")
    kpss_ct_p, kpss_ct_reject = _kpss_safe(s, regression="ct")

    return {
        "n": n,
        "adf_c_p": adf_c_p,
        "adf_c_reject": adf_c_reject,
        "adf_ct_p": adf_ct_p,
        "adf_ct_reject": adf_ct_reject,
        "kpss_c_p": kpss_c_p,
        "kpss_c_reject": kpss_c_reject,
        "kpss_ct_p": kpss_ct_p,
        "kpss_ct_reject": kpss_ct_reject,
    }


def classify(t):
    """
    Minimal but robust classification.

    STATIONARY (level):
      ADF(c) rejects unit root AND KPSS(c) does NOT reject

    NON-STATIONARY (Strong):
      ADF(c) does NOT reject AND KPSS(c) rejects

    TREND-STATIONARY:
      ADF(ct) rejects unit root AND KPSS(c) rejects AND KPSS(ct) does NOT reject

    Else:
      CONFLICT / PERSISTENT
    """
    if t["adf_c_reject"] and (not t["kpss_c_reject"]):
        return "STATIONARY"

    if (not t["adf_c_reject"]) and t["kpss_c_reject"]:
        return "!!! NON-STATIONARY (Strong) !!!"

    if t["adf_ct_reject"] and t["kpss_c_reject"] and (not t["kpss_ct_reject"]):
        return "TREND-STATIONARY (detrend ok)"

    return "? CONFLICT / PERSISTENT ?"


def main():
    if not FACTORS_PATH.exists():
        raise FileNotFoundError(f"File non trovato: {FACTORS_PATH}")

    factors = pd.read_parquet(FACTORS_PATH)

    print(
        f"{'FACTOR':<28} | {'CLASSIFICATION':<30} | "
        f"{'ADF(c) p':>8} | {'ADF(ct) p':>9} | {'KPSS(c) p':>10} | {'KPSS(ct) p':>11} | {'N':>5}"
    )
    print("-" * 120)

    for col in sorted(factors.columns):
        s = factors[col]

        # Skip too-short / constant (silent)
        s_nonan = s.dropna()
        if len(s_nonan) < MIN_OBS:
            continue
        if np.isclose(float(s_nonan.var()), 0.0):
            continue

        try:
            t = run_tests(s)
            status = classify(t)
        except Exception:
            continue

        # Print ONLY non-STATIONARY cases (as requested)
        if status != "STATIONARY":
            print(
                f"{col:<28} | {status:<30} | "
                f"{t['adf_c_p']:8.4f} | {t['adf_ct_p']:9.4f} | {t['kpss_c_p']:10.4f} | {t['kpss_ct_p']:11.4f} | {t['n']:5d}"
            )


if __name__ == "__main__":
    main()

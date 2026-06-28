"""
================================================================================
verify_marginal_vs_partial.py — Sign-Flip Diagnostic for AEN Coefficients
================================================================================
For each AEN-selected factor in Paper 1 Table 33, this script computes:
  1. Marginal correlation Corr(r_strategy, f_j)
  2. Univariate OLS beta from r = a + b * f_j + e
  3. Partial OLS beta from the multivariate AEN-selected regression
  4. Sign-flip flag (when marginal and partial have opposite signs)
  5. Variance Inflation Factor (VIF) for multicollinearity diagnostic

Purpose: diagnose whether counterintuitive coefficient signs in Table 33 are
driven by genuine data patterns (Cho 2020 endogenous risk channel,
slow-moving capital rebound) or by multicollinearity-induced sign flips in
multivariate regression.

Reference for VIF interpretation:
  - VIF > 10:  serious multicollinearity (collapse caution)
  - VIF 5-10:  moderate, acceptable for inference
  - VIF < 5:   no multicollinearity concern

Author:      Alessio Ottaviani
Institution: EDHEC Business School – PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path

# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACTORS_PATH = PROJECT_ROOT / "data" / "processed" / "all_factors_monthly.parquet"
RESULTS_DIR = PROJECT_ROOT / "results"

STRATEGIES = {
    "BTP_Italia":      RESULTS_DIR / "btp_italia"      / "index_daily.csv",
    "CDS_Bond_Basis":  RESULTS_DIR / "cds_bond_basis"  / "index_daily.csv",
    "iTraxx_Combined": RESULTS_DIR / "itraxx_combined" / "index_daily.csv",
}

# AEN-selected factors from Table 33, in same order as paper
AEN_SELECTED = {
    "BTP_Italia": [
        "SMB_EU", "ILLIQ", "SS10Y", "UMD_EU", "CDX_IG", "LIBOR_OIS"
    ],
    "CDS_Bond_Basis": [
        "\u0394UF", "RI_EU", "EMERG_FX", "PB_EU_CDS_1Y", "SS5Y",
        "CRED_SPR_US", "EP_SVIX_1M", "PTFSFX"
    ],
    "iTraxx_Combined": [
        "EP_SVIX_1M", "LIBOR_OIS", "\u0394V2X", "\u0394UR",
        "\u0394FAILS_PCT_TSY"
    ],
}

HAC_LAGS = 6  # Newey-West lag, same as paper

# ============================================================================
# HELPERS
# ============================================================================

def load_monthly_returns(csv_path: Path) -> pd.Series:
    """Load daily returns and compound to monthly (%)."""
    daily = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if 'index_return' not in daily.columns:
        raise ValueError(f"'index_return' column not found in {csv_path}")
    monthly = daily['index_return'].resample('ME').apply(
        lambda x: ((1 + x / 100).prod() - 1) * 100 if len(x) > 0 else np.nan
    ).dropna()
    return monthly


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Compute VIF for each column of X (without constant)."""
    Xc = sm.add_constant(X)
    vif = pd.Series(
        [variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])],
        index=X.columns,
        name='VIF'
    )
    return vif


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def diagnose_strategy(strategy_name: str, factors: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print(f"  STRATEGY: {strategy_name}")
    print("=" * 90)

    # Load returns
    csv_path = STRATEGIES[strategy_name]
    if not csv_path.exists():
        print(f"  ❌ File not found: {csv_path}")
        return

    r = load_monthly_returns(csv_path)
    selected = AEN_SELECTED[strategy_name]

    # Build aligned (y, X)
    missing = [f for f in selected if f not in factors.columns]
    if missing:
        print(f"  ❌ Missing factors in parquet: {missing}")
        return

    X = factors[selected].dropna()
    common = r.index.intersection(X.index)
    r = r.loc[common]
    X = X.loc[common]

    print(f"  Sample: T={len(r)} monthly observations, k={len(selected)} factors")
    print(f"  Range:  {r.index.min().strftime('%Y-%m')} → {r.index.max().strftime('%Y-%m')}")

    # === Multivariate regression (partial betas) ===
    Xc = sm.add_constant(X)
    multi = sm.OLS(r, Xc).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

    # === VIF for multicollinearity ===
    vif = compute_vif(X)

    # === Per-factor diagnostics ===
    print(f"\n  {'Factor':<22} {'Marg corr':>10} {'Uni β':>10} "
          f"{'Uni p':>8} {'Multi β':>10} {'Multi p':>8} {'VIF':>7} {'Flag':>12}")
    print("  " + "-" * 88)

    for f in selected:
        # Marginal correlation
        marg_corr = r.corr(X[f])

        # Univariate regression
        Xu = sm.add_constant(X[[f]])
        uni = sm.OLS(r, Xu).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
        uni_beta = uni.params.iloc[1]
        uni_pval = uni.pvalues.iloc[1]

        # Multivariate (partial) coefficient
        multi_beta = multi.params[f]
        multi_pval = multi.pvalues[f]

        # VIF
        vif_val = vif[f]

        # Sign flip detection
        flip_flag = ""
        if uni_beta * multi_beta < 0:
            flip_flag = "⚠️ SIGN FLIP"
        elif vif_val > 10:
            flip_flag = "⚠️ HIGH VIF"
        elif abs(np.sign(marg_corr) - np.sign(multi_beta)) > 1:
            flip_flag = "⚠️ CORR-β MISMATCH"

        print(f"  {f:<22} {marg_corr:>+10.4f} {uni_beta:>+10.4f} "
              f"{uni_pval:>8.4f} {multi_beta:>+10.4f} {multi_pval:>8.4f} "
              f"{vif_val:>7.2f} {flip_flag:>12}")

    # === Multivariate alpha ===
    alpha = multi.params['const']
    alpha_pval = multi.pvalues['const']
    print(f"\n  Multivariate α (monthly %): {alpha:+.4f} (p = {alpha_pval:.4f})")
    print(f"  Multivariate α (annualized %): {alpha * 12:+.4f}")
    print(f"  Adj R²: {multi.rsquared_adj:.4f}")


def main():
    print("=" * 90)
    print("SIGN-FLIP DIAGNOSTIC FOR AEN-SELECTED COEFFICIENTS")
    print("Marginal vs Partial Correlation + VIF Multicollinearity Test")
    print("=" * 90)

    # Load factors
    if not FACTORS_PATH.exists():
        print(f"\n❌ Factors parquet not found: {FACTORS_PATH}")
        print("   Run 00_import_all_factors.py first.")
        return

    factors = pd.read_parquet(FACTORS_PATH)
    print(f"\n✅ Loaded {len(factors.columns)} factors, "
          f"range {factors.index.min().strftime('%Y-%m')} → "
          f"{factors.index.max().strftime('%Y-%m')}")

    for strategy in STRATEGIES.keys():
        diagnose_strategy(strategy, factors)

    print("\n" + "=" * 90)
    print("DIAGNOSTIC LEGEND:")
    print("  Marg corr      = Pearson correlation Corr(r, f_j) [marginal]")
    print("  Uni β / Uni p  = Univariate OLS coefficient and p-value")
    print("  Multi β        = Partial coefficient from multivariate AEN regression")
    print("  Multi p        = HAC p-value from multivariate AEN regression")
    print("  VIF            = Variance Inflation Factor")
    print("                   > 10: serious multicollinearity")
    print("                   5-10: moderate")
    print("                   < 5: clean")
    print("  Flag           = ⚠️ SIGN FLIP        : Uni β and Multi β opposite signs")
    print("                   ⚠️ HIGH VIF         : VIF > 10")
    print("                   ⚠️ CORR-β MISMATCH  : sign(corr) ≠ sign(Multi β)")
    print("=" * 90)


if __name__ == "__main__":
    main()
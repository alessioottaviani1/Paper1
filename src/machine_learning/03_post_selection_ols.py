"""
================================================================================
03_aen_post_selection_ols.py - Post-Selection OLS Inference
================================================================================
After AEN selects factors in Step 02, this script estimates an unrestricted
OLS regression on the selected factors to:

1. Obtain UNBIASED coefficient estimates (AEN coefficients are shrunk).
2. Estimate α (intercept) — the strategy's risk-adjusted return.
3. Compute HAC (Newey-West) standard errors for valid t-tests under
   serial correlation and heteroskedasticity.
4. Test H₀: α = 0  (is there genuine alpha after controlling for factors?)

The regression is:
    yₜ = α + β₁·X₁ₜ + β₂·X₂ₜ + ... + βₖ·Xₖₜ + εₜ

where y is strategy returns (UN-centered, original scale) and X are the
selected factors (also original scale, NOT L2-normalized).

IMPORTANT: We use RAW (pre-standardization) data here, not the centered/
L2-normed data from preprocessing.  The AEN standardization was for the
penalized estimator; OLS inference needs original units for interpretable
coefficients and a meaningful intercept (α).

Inputs:
    - aen_results.json from Step 02 (selected factor names)
    - Raw strategy returns and factor data

Outputs (per strategy):
    - ols_results.json  (full regression output)
    - ols_table.csv     (publication-ready table)

Author:      Alessio Ottaviani
Institution: EDHEC Business School – PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm

import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"

spec = importlib.util.spec_from_file_location("aen_config", config_path)
aen_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aen_config)

FACTORS_PATH         = aen_config.FACTORS_PATH
STRATEGIES           = aen_config.STRATEGIES
FACTORS_END_DATE     = aen_config.FACTORS_END_DATE
HAC_LAGS             = aen_config.HAC_LAGS
AEN_TUNING_CRITERION = aen_config.AEN_TUNING_CRITERION
get_strategy_aen_dir = aen_config.get_strategy_aen_dir
get_aen_output_dir   = aen_config.get_aen_output_dir


def print_header(title, char="="):
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")


def load_strategy_returns(strategy_path):
    """Load daily returns → monthly compounding (same as 01)."""
    daily_df = pd.read_csv(strategy_path, index_col=0, parse_dates=True)
    daily_returns = daily_df['index_return'].dropna()
    monthly = daily_returns.resample('ME').apply(
        lambda x: ((1 + x / 100).prod() - 1) * 100 if len(x) > 0 else np.nan
    )
    return monthly.dropna()


def significance_stars(pval):
    if pval < 0.01:
        return "***"
    elif pval < 0.05:
        return "**"
    elif pval < 0.10:
        return "*"
    return ""


def run_ols_for_strategy(strategy_name, strategy_path):
    """Run post-selection OLS for a single strategy."""

    strategy_dir = get_strategy_aen_dir(strategy_name)

    # ── Load AEN results ───────────────────────────────────────────────
    with open(strategy_dir / "aen_results.json", 'r') as f:
        aen_results = json.load(f)

    selected_factors = aen_results['selected_factors']
    n_selected = aen_results['n_selected']

    if n_selected == 0:
        print(f"\n   ⚠️  No factors selected by AEN. Skipping OLS.")
        print(f"      (The AEN selected the null model for this strategy.)")
        return None

    print(f"\n   AEN selected {n_selected} factors: {selected_factors}")

    # ── Load RAW data (not standardized) ───────────────────────────────
    # We need original-scale data for interpretable OLS
    all_factors = pd.read_parquet(FACTORS_PATH)
    factors_end = pd.Timestamp(FACTORS_END_DATE)
    all_factors = all_factors[all_factors.index <= factors_end]

    returns = load_strategy_returns(strategy_path)
    returns = returns[returns.index <= factors_end]

    # Align
    common_dates = returns.index.intersection(all_factors.index)
    y = returns.loc[common_dates]
    X = all_factors.loc[common_dates][selected_factors].copy()

    # Drop any NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    y, X = y[mask], X[mask]
    T = len(y)

    print(f"   T = {T}, k = {n_selected}")
    print(f"   y mean = {y.mean():.4f}% monthly")

    # ── OLS with intercept ─────────────────────────────────────────────
    X_const = sm.add_constant(X, prepend=True)

    # OLS fit
    model = sm.OLS(y, X_const)
    results_ols = model.fit()

    # HAC (Newey-West) standard errors
    results_hac = model.fit(
        cov_type='HAC',
        cov_kwds={'maxlags': HAC_LAGS}
    )

    # ── Extract results ────────────────────────────────────────────────
    alpha_coeff = results_hac.params['const']
    alpha_se = results_hac.bse['const']
    alpha_tstat = results_hac.tvalues['const']
    alpha_pval = results_hac.pvalues['const']

    print(f"\n   ── OLS Results (HAC SE, Newey-West {HAC_LAGS} lags) ──")
    print(f"\n   {'Variable':<25} {'Coeff':>10} {'HAC SE':>10}"
          f" {'t-stat':>10} {'p-value':>10}")
    print(f"   {'─' * 67}")

    rows = []
    for var in X_const.columns:
        coeff = results_hac.params[var]
        se = results_hac.bse[var]
        tstat = results_hac.tvalues[var]
        pval = results_hac.pvalues[var]
        stars = significance_stars(pval)
        label = "α (intercept)" if var == 'const' else var
        print(f"   {label:<25} {coeff:>+10.4f} {se:>10.4f}"
              f" {tstat:>10.3f} {pval:>10.4f} {stars}")
        rows.append({
            'variable': label,
            'coefficient': round(coeff, 6),
            'hac_se': round(se, 6),
            't_statistic': round(tstat, 4),
            'p_value': round(pval, 4),
            'significance': stars
        })

    print(f"   {'─' * 67}")
    print(f"   R²         = {results_ols.rsquared:.4f}")
    print(f"   R² adj     = {results_ols.rsquared_adj:.4f}")
    print(f"   Durbin-W   = {sm.stats.durbin_watson(results_ols.resid):.4f}")
    print(f"   F-stat     = {results_ols.fvalue:.4f}"
          f" (p = {results_ols.f_pvalue:.4f})")

    # Alpha interpretation
    print(f"\n   ── Alpha Inference ──")
    print(f"   α = {alpha_coeff:+.4f}% monthly"
          f" ({alpha_coeff * 12:+.2f}% annualized)")
    print(f"   t-stat = {alpha_tstat:.3f}, p-value = {alpha_pval:.4f}"
          f" {significance_stars(alpha_pval)}")
    if alpha_pval < 0.05:
        print(f"   → Significant alpha at 5% level ✅")
    elif alpha_pval < 0.10:
        print(f"   → Significant alpha at 10% level (marginal)")
    else:
        print(f"   → Alpha not statistically significant")

    # ── Save ───────────────────────────────────────────────────────────
    table_df = pd.DataFrame(rows)
    table_df.to_csv(strategy_dir / "ols_table.csv", index=False)

    ols_output = {
        'strategy': strategy_name,
        'tuning_criterion': AEN_TUNING_CRITERION,
        'n_factors': n_selected,
        'selected_factors': selected_factors,
        'T': T,
        'hac_lags': HAC_LAGS,
        'alpha': {
            'coefficient': round(alpha_coeff, 6),
            'hac_se': round(alpha_se, 6),
            't_statistic': round(alpha_tstat, 4),
            'p_value': round(alpha_pval, 4),
            'annualized_pct': round(alpha_coeff * 12, 4),
            'significant_5pct': bool(alpha_pval < 0.05),
            'significant_10pct': bool(alpha_pval < 0.10),
        },
        'factors': {
            var: {
                'coefficient': round(float(results_hac.params[var]), 6),
                'hac_se': round(float(results_hac.bse[var]), 6),
                't_statistic': round(float(results_hac.tvalues[var]), 4),
                'p_value': round(float(results_hac.pvalues[var]), 4),
            }
            for var in selected_factors
        },
        'r_squared': round(results_ols.rsquared, 6),
        'r_squared_adj': round(results_ols.rsquared_adj, 6),
        'durbin_watson': round(float(sm.stats.durbin_watson(results_ols.resid)), 4),
        'f_statistic': round(float(results_ols.fvalue), 4),
        'f_pvalue': round(float(results_ols.f_pvalue), 6),
    }

    with open(strategy_dir / "ols_results.json", 'w') as f:
        json.dump(ols_output, f, indent=2)

    print(f"\n   💾 ols_results.json")
    print(f"   💾 ols_table.csv")

    return ols_output


# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("POST-SELECTION OLS INFERENCE")
    print(f"\n   Criterion: {AEN_TUNING_CRITERION}")
    print(f"   HAC lags (Newey-West): {HAC_LAGS}")
    print(f"   H₀: α = 0 (no risk-adjusted return)")

    all_results = {}

    for strategy_name, strategy_path in STRATEGIES.items():
        print_header(f"STRATEGY: {strategy_name}")

        strategy_dir = get_strategy_aen_dir(strategy_name)
        if not (strategy_dir / "aen_results.json").exists():
            print(f"\n   ❌ AEN results not found. Run 02 first.")
            continue

        try:
            result = run_ols_for_strategy(strategy_name, strategy_path)
            if result is not None:
                all_results[strategy_name] = result
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
            import traceback; traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────────
    if all_results:
        print_header("ALPHA SUMMARY")

        print(f"\n   {'Strategy':<20} {'α (mo)':>8} {'α (yr)':>8}"
              f" {'t-stat':>8} {'p-val':>8} {'R²adj':>6} {'k':>3}")
        print(f"   {'─' * 63}")

        for name, res in all_results.items():
            a = res['alpha']
            print(f"   {name:<20}"
                  f" {a['coefficient']:>+8.4f}"
                  f" {a['annualized_pct']:>+8.2f}"
                  f" {a['t_statistic']:>8.3f}"
                  f" {a['p_value']:>8.4f}"
                  f" {res['r_squared_adj']:>6.3f}"
                  f" {res['n_factors']:>3}"
                  f" {significance_stars(a['p_value'])}")

        # Save global summary
        aen_output_dir = get_aen_output_dir()
        summary = {
            'tuning_criterion': AEN_TUNING_CRITERION,
            'hac_lags': HAC_LAGS,
            'strategies': {name: {
                'alpha_monthly': res['alpha']['coefficient'],
                'alpha_annualized': res['alpha']['annualized_pct'],
                'alpha_tstat': res['alpha']['t_statistic'],
                'alpha_pval': res['alpha']['p_value'],
                'r_squared_adj': res['r_squared_adj'],
                'n_factors': res['n_factors'],
                'selected_factors': res['selected_factors']
            } for name, res in all_results.items()}
        }
        with open(aen_output_dir / "ols_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n   💾 {aen_output_dir / 'ols_summary.json'}")

    print(f"\n{'=' * 80}")
    print(f"✅ POST-SELECTION OLS COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n   🎯 Next: python src/aen/04_bootstrap_stability.py")


if __name__ == "__main__":
    main()
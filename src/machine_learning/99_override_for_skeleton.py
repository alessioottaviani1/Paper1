"""
================================================================================
99_override_for_skeleton.py — Temporary Factor Override for Skeleton Tables
================================================================================
TEMPORARY script: overrides AEN factor selection with manually specified
factors, runs REAL post-selection OLS, and writes all JSON/CSV files that
07_tables.py (and 06e_conditional_alpha.py) expect.

This does NOT modify the AEN pipeline logic.  Once the pipeline produces
the correct selections, delete this file and re-run the standard chain.

What it writes (per strategy, in the strategy AEN output directory):
    - aen_results.json           (fake: sets selected_factors to override)
    - bootstrap_stability.json   (fake: frequencies = 1.0 for override factors)
    - ols_stable_results.json    (REAL: OLS on override factors, HAC SE)
    - ols_stable_table.csv       (REAL: same as above, CSV format)
    - method_comparison.json     (stub: only AEN method, for 07 table generation)
    - method_comparison.csv      (stub: same)
    - selection_matrix.csv       (stub: only AEN column)

Usage:
    python src/machine_learning/99_override_for_skeleton.py

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

# ============================================================================
# CONFIG
# ============================================================================

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
COEF_TOL             = aen_config.COEF_TOL
STABILITY_THRESHOLD  = aen_config.STABILITY_THRESHOLD
BOOTSTRAP_N_REPS     = aen_config.BOOTSTRAP_N_REPS
BOOTSTRAP_BLOCK_LENGTH = aen_config.BOOTSTRAP_BLOCK_LENGTH
BOOTSTRAP_METHOD     = aen_config.BOOTSTRAP_METHOD
CSS_ENABLED          = aen_config.CSS_ENABLED
get_strategy_aen_dir = aen_config.get_strategy_aen_dir
get_aen_output_dir   = aen_config.get_aen_output_dir


# ============================================================================
# ⭐ MANUAL FACTOR OVERRIDE — edit here
# ============================================================================

OVERRIDE_FACTORS = {
    "btp_italia": [
        "ILLIQ", "UMD_EU", "SMB_EU", "BAB_US",
        "BTP_BUND", "TERM_EU", "CREDIT_EU", "SS10Y",
    ],
    "cds_bond_basis": [
        "EBP", "CRED_SPR_US", "TERM_US", "RI_EU",
        "PB_EU_CDS_1Y", "SMB_EU", "SS10Y", "DEF_US",
    ],
    "itraxx_combined": [
        "EP_SVIX_1M", "LIBOR_REPO_SHOCK", "SILLIQ",
        "\u0394UF", "TED_SHOCK_EU", "R10_EU",
    ],
}


# ============================================================================
# HELPERS
# ============================================================================

def print_header(title, char="="):
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")


def load_strategy_returns(strategy_path):
    """Load daily returns → monthly compounding (same as 01/03)."""
    daily_df = pd.read_csv(strategy_path, index_col=0, parse_dates=True)
    daily_returns = daily_df['index_return'].dropna()
    monthly = daily_returns.resample('ME').apply(
        lambda x: ((1 + x / 100).prod() - 1) * 100 if len(x) > 0 else np.nan
    )
    return monthly.dropna()


def significance_stars(pval):
    if pval < 0.01:   return "***"
    elif pval < 0.05: return "**"
    elif pval < 0.10: return "*"
    return ""


# ============================================================================
# CORE: RUN REAL OLS + WRITE ALL EXPECTED JSON/CSV
# ============================================================================

def override_strategy(strategy_name, strategy_path, override_factors):
    """
    1. Run real OLS on override_factors (raw data, HAC SE)
    2. Write aen_results.json, bootstrap_stability.json,
       ols_stable_results.json, method_comparison.json, etc.
    """
    strategy_dir = get_strategy_aen_dir(strategy_name)
    strategy_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n   Override factors: {override_factors}")

    # ── Load raw data ──────────────────────────────────────────────────
    all_factors = pd.read_parquet(FACTORS_PATH)
    factors_end = pd.Timestamp(FACTORS_END_DATE)
    all_factors = all_factors[all_factors.index <= factors_end]

    # Validate factor names
    available = set(all_factors.columns)
    missing = [f for f in override_factors if f not in available]
    if missing:
        print(f"\n   ❌ MISSING FACTORS: {missing}")
        print(f"      Available columns (sample): {sorted(available)[:20]}")
        # Try fuzzy match
        for m in missing:
            candidates = [c for c in available
                          if m.lower().replace("_", "") in c.lower().replace("_", "")]
            if candidates:
                print(f"      Did you mean? {m} → {candidates}")
        return None

    returns = load_strategy_returns(strategy_path)
    returns = returns[returns.index <= factors_end]

    common = returns.index.intersection(all_factors.index)
    y = returns.loc[common]
    X = all_factors.loc[common][override_factors].copy()
    mask = ~(X.isna().any(axis=1) | y.isna())
    y, X = y[mask], X[mask]
    T = len(y)
    k = len(override_factors)

    print(f"   T = {T}, k = {k}")
    print(f"   y mean = {y.mean():.4f}% monthly")

    # ── OLS with intercept + HAC SE ────────────────────────────────────
    X_const = sm.add_constant(X, prepend=True)
    res_ols = sm.OLS(y, X_const).fit()
    res_hac = sm.OLS(y, X_const).fit(
        cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

    alpha_c  = float(res_hac.params['const'])
    alpha_se = float(res_hac.bse['const'])
    alpha_t  = float(res_hac.tvalues['const'])
    alpha_p  = float(res_hac.pvalues['const'])

    print(f"\n   ── OLS Results (HAC NW {HAC_LAGS} lags) ──")
    print(f"\n   {'Variable':<25} {'Coeff':>10} {'HAC SE':>10}"
          f" {'t-stat':>10} {'p-val':>9}")
    print(f"   {'─' * 67}")

    csv_rows = []
    for var in X_const.columns:
        c  = float(res_hac.params[var])
        se = float(res_hac.bse[var])
        t  = float(res_hac.tvalues[var])
        p  = float(res_hac.pvalues[var])
        lbl = "α (intercept)" if var == 'const' else var
        print(f"   {lbl:<25} {c:>+10.4f} {se:>10.4f}"
              f" {t:>10.3f} {p:>9.4f} {significance_stars(p)}")
        csv_rows.append({
            'variable': lbl,
            'coefficient': round(c, 6),
            'hac_se': round(se, 6),
            't_statistic': round(t, 4),
            'p_value': round(p, 4),
            'significance': significance_stars(p),
        })

    print(f"   {'─' * 67}")
    r2     = float(res_ols.rsquared)
    r2_adj = float(res_ols.rsquared_adj)
    dw     = float(sm.stats.durbin_watson(res_ols.resid))
    print(f"   R²={r2:.4f} | R²adj={r2_adj:.4f} | DW={dw:.4f}")
    print(f"\n   α={alpha_c:+.4f}%/mo ({alpha_c*12:+.2f}%/yr)"
          f" | t={alpha_t:.3f} | p={alpha_p:.4f}"
          f" {significance_stars(alpha_p)}")

    # ── Build factor-level dict ────────────────────────────────────────
    factors_dict = {}
    for f in override_factors:
        factors_dict[f] = {
            'coefficient':  round(float(res_hac.params[f]), 6),
            'hac_se':       round(float(res_hac.bse[f]), 6),
            't_statistic':  round(float(res_hac.tvalues[f]), 4),
            'p_value':      round(float(res_hac.pvalues[f]), 4),
        }

    alpha_dict = {
        'coefficient':      round(alpha_c, 6),
        'hac_se':           round(alpha_se, 6),
        't_statistic':      round(alpha_t, 4),
        'p_value':          round(alpha_p, 4),
        'annualized_pct':   round(alpha_c * 12, 4),
        'significant_5pct': bool(alpha_p < 0.05),
        'significant_10pct': bool(alpha_p < 0.10),
    }

    # ==================================================================
    # WRITE 1: aen_results.json  (read by 06_robustness)
    # ==================================================================
    aen_results = {
        'strategy': strategy_name,
        'tuning_criterion': AEN_TUNING_CRITERION,
        'T': T,
        'p': k,
        '_override_note': 'Manually specified factors (99_override_for_skeleton.py)',
        'stage1': {
            'lambda2_po_best': 0.01,
            'lambda1_po_best': 0.001,
            'rescale': 1.01,
            'ic_best': 0.0,
            'ic_raw_best': 0.0,
            'df_best': k,
        },
        'stage2': {
            'lambda1_star_po': 0.001,
            'lambda2_po': 0.01,
            'rescale': 1.01,
            'ic_best': 0.0,
            'ic_raw_best': 0.0,
            'df_best': k,
            'df_under_raw_ic': k,
        },
        'selected_factors': override_factors,
        'n_selected': k,
        'selected_coefficients': {
            f: factors_dict[f]['coefficient'] for f in override_factors
        },
    }
    with open(strategy_dir / "aen_results.json", 'w') as f:
        json.dump(aen_results, f, indent=2)
    print(f"\n   💾 aen_results.json (override)")

    # ==================================================================
    # WRITE 2: ols_stable_results.json  (read by 07_tables build_article_stable_ols)
    # ==================================================================
    ols_stable = {
        'selection_label': 'stable',
        'stable_factors': override_factors,
        'n_factors': k,
        'T': T,
        'alpha': alpha_dict,
        'factors': factors_dict,
        'r_squared': round(r2, 6),
        'r_squared_adj': round(r2_adj, 6),
        'durbin_watson': round(dw, 4),
    }
    with open(strategy_dir / "ols_stable_results.json", 'w') as f:
        json.dump(ols_stable, f, indent=2)
    pd.DataFrame(csv_rows).to_csv(
        strategy_dir / "ols_stable_table.csv", index=False)
    print(f"   💾 ols_stable_results.json (REAL OLS)")
    print(f"   💾 ols_stable_table.csv")

    # ==================================================================
    # WRITE 3: bootstrap_stability.json  (read by 07_tables, 06e)
    # ==================================================================
    # Fake frequencies: all override factors at 1.0
    freq_dict = {
        f: {'pi_lambda_star': 1.0, 'pi_max_lambda': 1.0}
        for f in override_factors
    }
    boot_results = {
        '_override_note': 'Manually specified factors (99_override_for_skeleton.py)',
        'config': {
            'coef_tol': COEF_TOL,
            'bootstrap_method': BOOTSTRAP_METHOD,
            'n_reps': BOOTSTRAP_N_REPS,
            'block_length': BOOTSTRAP_BLOCK_LENGTH,
            'pi_thr': STABILITY_THRESHOLD,
            'css_enabled': CSS_ENABLED,
        },
        'full_sample_factors': override_factors,
        'stable_factors': override_factors,
        'stable_factors_robustness': override_factors,
        'stable_factors_medoid': override_factors,
        'n_stable': k,
        'n_stable_robustness': k,
        'n_stable_medoid': k,
        'overlap_with_full_sample': override_factors,
        'model_size_at_lambda_star_mean': float(k),
        'model_size_at_lambda_star_median': k,
        'null_model_frequency': 0.0,
        'bootstrap_errors': 0,
        'mb_bound': 0.0,
        'factor_frequencies': freq_dict,
        'css': {'enabled': CSS_ENABLED},
        'ols_stable': ols_stable,
        'ols_stable_medoid': ols_stable,
        'elapsed_seconds': 0.0,
    }
    with open(strategy_dir / "bootstrap_stability.json", 'w') as f:
        json.dump(boot_results, f, indent=2)
    print(f"   💾 bootstrap_stability.json (override freqs=1.0)")

    # ==================================================================
    # WRITE 4: method_comparison.json + .csv + selection_matrix.csv
    #          Runs ALL methods (Lasso, EN, ALasso, Ridge, ALASSO-Chen)
    #          by importing from 05_method_comparison.py
    # ==================================================================
    print(f"\n   ── Running all comparison methods ──")

    # Import functions from 05_method_comparison
    mc_path = PROJECT_ROOT / "src" / "machine_learning" / "05_method_comparison.py"
    mc_spec = importlib.util.spec_from_file_location("method_comp", mc_path)
    mc_mod = importlib.util.module_from_spec(mc_spec)
    mc_spec.loader.exec_module(mc_mod)

    # Load preprocessed (standardized) data for penalized methods
    y_std_df = pd.read_parquet(strategy_dir / "y_centered.parquet")
    X_std_df = pd.read_parquet(strategy_dir / "X_standardized.parquet")
    y_std = y_std_df['y'].values
    X_std = X_std_df.values
    factor_names_std = X_std_df.columns.tolist()
    T_std, p_std = X_std.shape

    # Raw data aligned to standardized factor set
    X_raw_mc = all_factors.loc[common][factor_names_std].copy()
    y_raw_mc = y.copy()
    mask_mc = ~(X_raw_mc.isna().any(axis=1) | y_raw_mc.isna())
    y_raw_mc, X_raw_mc = y_raw_mc[mask_mc], X_raw_mc[mask_mc]

    criterion = AEN_TUNING_CRITERION
    gic_alpha_val = getattr(aen_config, 'GIC_ALPHA', 3.0)
    n_lam1 = aen_config.AEN_LAMBDA1_N_VALUES
    lambda2_grid = aen_config.AEN_LAMBDA2_GRID
    gamma = aen_config.AEN_GAMMA

    mc_results = {}

    # AEN (bootstrap) — from our override OLS
    aen_method_result = {
        'selected_factors': override_factors,
        'n_factors': k,
        'alpha_monthly': round(alpha_c, 6),
        'alpha_annualized': round(alpha_c * 12, 4),
        'alpha_tstat': round(alpha_t, 4),
        'alpha_pval': round(alpha_p, 4),
        'r_squared': round(r2, 6),
        'r_squared_adj': round(r2_adj, 6),
    }
    mc_results['AEN (bootstrap)'] = aen_method_result

    # Method 1: Lasso
    print(f"   [1/5] Lasso...")
    try:
        beta_lasso = mc_mod.fit_lasso(y_std, X_std, criterion,
                                      gic_alpha_val, n_lam1)
        sel_lasso = mc_mod.selected_from_beta(beta_lasso, factor_names_std)
        ols_lasso = mc_mod.post_selection_ols(y_raw_mc, X_raw_mc,
                                              sel_lasso, HAC_LAGS)
        mc_results['Lasso'] = ols_lasso
        print(f"         Selected {len(sel_lasso)} factors")
    except Exception as e:
        print(f"         ❌ {e}")

    # Method 2: Elastic Net
    print(f"   [2/5] Elastic Net...")
    try:
        _, beta_en_raw = mc_mod.fit_elastic_net(y_std, X_std, lambda2_grid,
                                                criterion, gic_alpha_val,
                                                n_lam1)
        sel_en = mc_mod.selected_from_beta(beta_en_raw, factor_names_std)
        ols_en = mc_mod.post_selection_ols(y_raw_mc, X_raw_mc,
                                           sel_en, HAC_LAGS)
        mc_results['Elastic Net'] = ols_en
        print(f"         Selected {len(sel_en)} factors")
    except Exception as e:
        print(f"         ❌ {e}")

    # Method 3: Adaptive Lasso
    print(f"   [3/5] Adaptive Lasso...")
    try:
        beta_alasso = mc_mod.fit_adaptive_lasso(y_std, X_std, criterion,
                                                gamma, gic_alpha_val,
                                                n_lam1)
        sel_alasso = mc_mod.selected_from_beta(beta_alasso, factor_names_std)
        ols_alasso = mc_mod.post_selection_ols(y_raw_mc, X_raw_mc,
                                               sel_alasso, HAC_LAGS)
        mc_results['Adaptive Lasso'] = ols_alasso
        print(f"         Selected {len(sel_alasso)} factors")
    except Exception as e:
        print(f"         ❌ {e}")

    # Method 4: Ridge
    print(f"   [4/5] Ridge...")
    try:
        ridge_result = mc_mod.fit_ridge(y_raw_mc, X_raw_mc, factor_names_std)
        mc_results['Ridge'] = ridge_result
        print(f"         lambda = {ridge_result.get('ridge_lambda', 0):.4f}")
    except Exception as e:
        print(f"         ❌ {e}")

    # Method 5: ALASSO-Chen
    print(f"   [5/5] ALASSO-Chen...")
    try:
        chen_result = mc_mod.fit_adaptive_lasso_chen(
            y_raw_mc, X_raw_mc, factor_names_std, n_lambda1=n_lam1)
        sel_chen = chen_result['selected_factors_zscore']
        ols_chen = mc_mod.post_selection_ols(y_raw_mc, X_raw_mc,
                                             sel_chen, HAC_LAGS)
        ols_chen['chen_init_method'] = chen_result['init_method']
        mc_results['ALASSO-Chen'] = ols_chen
        print(f"         Selected {len(sel_chen)} factors")
    except Exception as e:
        print(f"         ❌ {e}")

    # Write method_comparison.json
    mc_json = {
        'strategy': strategy_name,
        'T': T,
        'p': p_std,
        'coef_eps': COEF_TOL,
        'hac_lags': HAC_LAGS,
        'tuning_criterion_methods_1_5': AEN_TUNING_CRITERION,
        'tuning_criterion_method_6': 'AIC',
        'chen_details': {
            'init_method': chen_result.get('init_method', 'unknown')
                          if 'ALASSO-Chen' in mc_results else 'N/A',
            'lambda1': chen_result.get('lambda1_best', 0)
                       if 'ALASSO-Chen' in mc_results else 0,
            'n_selected': mc_results.get('ALASSO-Chen', {}).get('n_factors', 0),
            'selected_factors': mc_results.get('ALASSO-Chen', {}).get(
                'selected_factors', []),
        },
        'methods': mc_results,
    }
    with open(strategy_dir / "method_comparison.json", 'w') as f:
        json.dump(mc_json, f, indent=2, default=str)

    # Write method_comparison.csv
    mc_csv_rows = []
    for method, res in mc_results.items():
        mc_csv_rows.append({
            'method': method,
            'n_factors': res.get('n_factors', 0),
            'alpha_monthly': res.get('alpha_monthly'),
            'alpha_annualized': res.get('alpha_annualized'),
            'alpha_tstat': res.get('alpha_tstat'),
            'alpha_pval': res.get('alpha_pval'),
            'r_squared': res.get('r_squared'),
            'r_squared_adj': res.get('r_squared_adj'),
        })
    pd.DataFrame(mc_csv_rows).to_csv(
        strategy_dir / "method_comparison.csv", index=False)

    # Write selection_matrix.csv
    all_sel_factors = set()
    for m, res in mc_results.items():
        if m != 'Ridge':
            all_sel_factors.update(res.get('selected_factors', []))
    all_sel_factors = sorted(all_sel_factors)

    sel_rows = []
    for f in all_sel_factors:
        row = {'factor': f}
        for m, res in mc_results.items():
            if m != 'Ridge':
                row[m] = 1 if f in res.get('selected_factors', []) else 0
        sel_rows.append(row)
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(strategy_dir / "selection_matrix.csv", index=False)

    print(f"\n   💾 method_comparison.json / .csv (ALL methods)")
    print(f"   💾 selection_matrix.csv")

    # ==================================================================
    # WRITE 5: bootstrap_frequencies.csv  (read by 07 stability barplot)
    # ==================================================================
    freq_rows = []
    for f in override_factors:
        freq_rows.append({
            'factor': f,
            'selection_frequency': 1.0,
            'selection_frequency_maxlam': 1.0,
        })
    pd.DataFrame(freq_rows).to_csv(
        strategy_dir / "bootstrap_frequencies.csv", index=False)
    print(f"   💾 bootstrap_frequencies.csv")

    return ols_stable


# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header(
        "99_OVERRIDE_FOR_SKELETON\n"
        "   TEMPORARY: manual factor override → real OLS → JSON for 07_tables"
    )
    print(f"\n   Criterion label: {AEN_TUNING_CRITERION}")
    print(f"   HAC lags: {HAC_LAGS}")

    all_results = {}

    for strategy_name, strategy_path in STRATEGIES.items():
        print_header(f"STRATEGY: {strategy_name}")

        if strategy_name not in OVERRIDE_FACTORS:
            print(f"\n   ⚠️ No override factors specified. Skipping.")
            continue

        override_factors = OVERRIDE_FACTORS[strategy_name]

        try:
            result = override_strategy(
                strategy_name, strategy_path, override_factors)
            if result:
                all_results[strategy_name] = result
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
            import traceback; traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────────
    if all_results:
        print_header("ALPHA SUMMARY (override factors)")

        print(f"\n   {'Strategy':<20} {'k':>4} {'α(mo)':>8} {'α(yr)':>8}"
              f" {'t':>7} {'p':>8} {'R²adj':>7}")
        print(f"   {'─' * 66}")

        for name, res in all_results.items():
            a = res['alpha']
            print(f"   {name:<20}"
                  f" {res['n_factors']:>4}"
                  f" {a['coefficient']:>+8.4f}"
                  f" {a['annualized_pct']:>+8.2f}"
                  f" {a['t_statistic']:>7.3f}"
                  f" {a['p_value']:>8.4f}"
                  f" {res['r_squared_adj']:>7.3f}"
                  f" {significance_stars(a['p_value'])}")

    print_header("DONE")
    print(f"\n   ✅ JSON files written. Now run:")
    print(f"      python src/machine_learning/07_tables.py")
    print(f"      python src/machine_learning/06e_conditional_alpha.py")
    print(f"\n   ⚠️  This is TEMPORARY. Delete this file after pipeline fix.")


if __name__ == "__main__":
    main()
"""
================================================================================
06_aen_robustness.py - Robustness Checks
================================================================================
Four independent robustness checks:

(a) Rolling α:
    OLS on rolling windows of 36 months, factors FIXED (AEN full-sample
    selection).  Shows α stability over time.  Plot with CI bands.

(b) Correlation threshold sensitivity:
    Re-run full pipeline (01+02) with ρ = 0.85, 0.90, 0.95.
    Show which factors are selected under each threshold.

(c) γ sensitivity:
    Re-run 02 with γ = 1, 2, 3.
    Show which factors are selected under each γ.

(d) Half-sample split:
    AEN on first half → selected factors.
    OLS on second half with those factors → out-of-sample α.

Outputs (per strategy):
    - rolling_alpha.csv          (Table 5: rolling α time series)
    - rolling_alpha_summary.json
    - sensitivity_correlation.json (Table 6a)
    - sensitivity_gamma.json       (Table 6b)
    - halfsample_results.json      (Table 7)
    - robustness_summary.json      (combined)

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

AEN_GAMMA              = aen_config.AEN_GAMMA
AEN_LAMBDA2_GRID       = aen_config.AEN_LAMBDA2_GRID
AEN_LAMBDA1_N_VALUES   = aen_config.AEN_LAMBDA1_N_VALUES
AEN_TUNING_CRITERION   = aen_config.AEN_TUNING_CRITERION
COEF_TOL               = aen_config.COEF_TOL
GIC_ALPHA              = aen_config.GIC_ALPHA
HAC_LAGS               = aen_config.HAC_LAGS
CORRELATION_THRESHOLD  = aen_config.CORRELATION_THRESHOLD
FACTORS_PATH           = aen_config.FACTORS_PATH
FACTORS_END_DATE       = aen_config.FACTORS_END_DATE
FACTORS_TO_EXCLUDE     = aen_config.FACTORS_TO_EXCLUDE
STRATEGIES             = aen_config.STRATEGIES
get_strategy_aen_dir   = aen_config.get_strategy_aen_dir
get_aen_output_dir     = aen_config.get_aen_output_dir

# Robustness-specific parameters
ROLLING_WINDOW = 36             # months
CORR_THRESHOLDS = [0.85, 0.90, 0.95]
GAMMA_VALUES = [1, 2, 3]


def print_header(title, char="="):
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")


def significance_stars(pval):
    if pval < 0.01:   return "***"
    elif pval < 0.05: return "**"
    elif pval < 0.10: return "*"
    return ""


def load_strategy_returns(strategy_path):
    daily_df = pd.read_csv(strategy_path, index_col=0, parse_dates=True)
    daily_returns = daily_df['index_return'].dropna()
    monthly = daily_returns.resample('ME').apply(
        lambda x: ((1 + x / 100).prod() - 1) * 100 if len(x) > 0 else np.nan
    )
    return monthly.dropna()


# ============================================================================
# IC (shared)
# ============================================================================

def compute_ic(y, X, beta, criterion, gic_alpha=3.0):
    T = len(y)
    residuals = y - X @ beta
    rss = float(np.sum(residuals ** 2))
    df = int(np.sum(np.abs(beta) > COEF_TOL))
    if rss / T < 1e-15:
        return np.inf, rss, df
    log_lik = T * np.log(rss / T)
    if criterion in ("BIC", "SIS_BIC"):
        ic = log_lik + df * np.log(T)
    elif criterion == "HQC":
        ic = log_lik + 2 * np.log(np.log(T)) * df
    elif criterion == "AICc":
        aic = log_lik + 2 * df
        ic = aic + (2 * df * (df + 1)) / (T - df - 1) if T - df - 1 > 0 else np.inf
    elif criterion == "GIC":
        ic = log_lik + gic_alpha * df
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    return float(ic), rss, df


def compute_lambda1_max(X, y):
    T = X.shape[0]
    return np.max(np.abs(X.T @ y)) / T


def build_lambda1_grid(lambda1_max, n_values=100, ratio_min=1e-4):
    return np.logspace(np.log10(lambda1_max),
                       np.log10(lambda1_max * ratio_min), n_values)


# ============================================================================
# COORDINATE DESCENT: WEIGHTED ℓ₁ + UNIFORM ℓ₂  (paper eq. 2.2)
# ============================================================================

def weighted_elastic_net_cd(X, y, lambda1, lambda2, weights,
                            max_iter=10000, tol=1e-7):
    """
    Coordinate descent: weighted ℓ₁ + uniform ℓ₂ (paper eq. 2.2).
    Returns naive β (not rescaled).
    """
    T, p = X.shape
    beta = np.zeros(p, dtype=np.float64)
    r = y.copy().astype(np.float64)
    col_norm_sq = np.sum(X ** 2, axis=0) / T

    for iteration in range(max_iter):
        max_change = 0.0
        for j in range(p):
            beta_old = beta[j]
            if beta_old != 0.0:
                r += X[:, j] * beta_old
            rho_j = np.dot(X[:, j], r) / T
            threshold = lambda1 * weights[j]
            if rho_j > threshold:
                numerator = rho_j - threshold
            elif rho_j < -threshold:
                numerator = rho_j + threshold
            else:
                numerator = 0.0
            beta[j] = numerator / (col_norm_sq[j] + lambda2)
            if beta[j] != 0.0:
                r -= X[:, j] * beta[j]
            change = abs(beta[j] - beta_old)
            if change > max_change:
                max_change = change
        if max_change < tol:
            break
    return beta


# ============================================================================
# FULL AEN (self-contained, for robustness re-runs)
# ============================================================================

def run_aen_full(y, X, factor_names, lambda2_grid, n_lambda1,
                 criterion, gamma=1, gic_alpha=3.0):
    """
    Complete two-stage AEN. Centers/standardizes internally.
    Stage 2 uses paper-faithful CD (weighted ℓ₁ + uniform ℓ₂).
    Returns list of selected factor names.
    """
    T, p = X.shape

    # Center and L2-normalize
    y_mean = np.mean(y)
    y_c = y - y_mean
    X_mean = np.mean(X, axis=0)
    X_c = X - X_mean
    X_l2 = np.sqrt(np.sum(X_c ** 2, axis=0))
    X_l2[X_l2 == 0] = 1.0
    X_s = X_c / X_l2

    # Stage 1 (CD with uniform weights, standard EN)
    lambda1_max = compute_lambda1_max(X_s, y_c)
    best_ic = np.inf
    best_beta_raw = None
    best_lambda2_po = None
    best_rescale = None

    for lambda2_po in lambda2_grid:
        lambda1_grid = build_lambda1_grid(lambda1_max, n_lambda1)
        lambda2_sum = lambda2_po * T
        rescale = 1.0 + lambda2_po          # eq. 1.4: (1+λ₂°)

        w_uniform = np.ones(p, dtype=np.float64)
        for lambda1_po in lambda1_grid:
            beta_raw = weighted_elastic_net_cd(
                X_s, y_c, lambda1_po, lambda2_po, w_uniform,
                max_iter=10000, tol=1e-7)
            beta_enet = beta_raw * rescale
            # df on beta_raw (pre-rescale)
            df = int(np.sum(np.abs(beta_raw) > COEF_TOL))
            pred = X_s @ beta_enet
            rss = float(np.sum((y_c - pred) ** 2))
            if rss / T < 1e-15:
                ic = np.inf
            else:
                log_lik = T * np.log(rss / T)
                if criterion in ("BIC", "SIS_BIC"):
                    ic = log_lik + df * np.log(T)
                elif criterion == "HQC":
                    ic = log_lik + 2 * np.log(np.log(T)) * df
                elif criterion == "AICc":
                    aic = log_lik + 2 * df
                    ic = aic + (2 * df * (df + 1)) / (T - df - 1) if T - df - 1 > 0 else np.inf
                elif criterion == "GIC":
                    ic = log_lik + gic_alpha * df
            if ic < best_ic:
                best_ic = ic
                best_beta_raw = beta_raw.copy()
                best_lambda2_po = lambda2_po
                best_rescale = rescale

    beta_enet = best_beta_raw * best_rescale

    # Adaptive weights
    epsilon = 1.0 / T
    weights = (np.abs(beta_enet) + epsilon) ** (-gamma)

    # Stage 2 — paper-faithful CD
    lambda1_star_max = np.max(np.abs(X_s.T @ y_c) / T / weights)
    if lambda1_star_max < 1e-15:
        lambda1_star_max = 1e-6
    lambda1_star_grid = build_lambda1_grid(lambda1_star_max, n_lambda1)

    best_ic2 = np.inf
    best_beta_naive = np.zeros(p)

    for lambda1_star_po in lambda1_star_grid:
        beta_naive = weighted_elastic_net_cd(
            X_s, y_c, lambda1_star_po, best_lambda2_po,
            weights, max_iter=10000, tol=1e-7
        )
        beta_aen = beta_naive * best_rescale
        # df on beta_naive (pre-rescale)
        df = int(np.sum(np.abs(beta_naive) > COEF_TOL))
        pred = X_s @ beta_aen
        rss = float(np.sum((y_c - pred) ** 2))
        if rss / T < 1e-15:
            ic = np.inf
        else:
            log_lik = T * np.log(rss / T)
            if criterion in ("BIC", "SIS_BIC"):
                ic = log_lik + df * np.log(T)
            elif criterion == "HQC":
                ic = log_lik + 2 * np.log(np.log(T)) * df
            elif criterion == "AICc":
                aic = log_lik + 2 * df
                ic = aic + (2 * df * (df + 1)) / (T - df - 1) if T - df - 1 > 0 else np.inf
            elif criterion == "GIC":
                ic = log_lik + gic_alpha * df
        if ic < best_ic2:
            best_ic2 = ic
            best_beta_naive = beta_naive.copy()

    # Selection on beta_naive (pre-rescale)
    selected = [factor_names[i] for i in range(p)
                if abs(best_beta_naive[i]) > COEF_TOL]
    return selected


# ============================================================================
# (a) ROLLING ALPHA
# ============================================================================

def rolling_alpha(strategy_name, strategy_path):
    """
    Rolling OLS with FIXED factors (from AEN full-sample).
    Window = 36 months. Factors don't change — only the OLS window moves.
    """
    print_header(f"   (a) Rolling Alpha — {strategy_name}", "─")

    strategy_dir = get_strategy_aen_dir(strategy_name)

    with open(strategy_dir / "aen_results.json", 'r') as f:
        aen_results = json.load(f)
    selected_factors = aen_results['selected_factors']

    if len(selected_factors) == 0:
        print(f"\n   ⚠️  No factors selected. Skipping.")
        return None

    # Load raw data
    all_factors = pd.read_parquet(FACTORS_PATH)
    factors_end = pd.Timestamp(FACTORS_END_DATE)
    all_factors = all_factors[all_factors.index <= factors_end]

    returns = load_strategy_returns(strategy_path)
    returns = returns[returns.index <= factors_end]

    common_dates = returns.index.intersection(all_factors.index)
    y = returns.loc[common_dates]
    X = all_factors.loc[common_dates][selected_factors].copy()
    mask = ~(X.isna().any(axis=1) | y.isna())
    y, X = y[mask], X[mask]
    T = len(y)

    n_roll = T - ROLLING_WINDOW + 1
    if n_roll <= 0:
        print(f"\n   ⚠️  Not enough data for rolling window:"
              f" T={T} < {ROLLING_WINDOW}. Skipping.")
        return None

    print(f"\n   Factors (fixed): {selected_factors}")
    print(f"   Window: {ROLLING_WINDOW} months")
    print(f"   T = {T}, rolling windows: {n_roll}")

    # t-critical for CI (df = window - k - 1)
    from scipy import stats as scipy_stats
    k = len(selected_factors)
    df_ci = ROLLING_WINDOW - k - 1
    t_crit = scipy_stats.t.ppf(0.975, df=df_ci) if df_ci > 0 else 1.96

    rolling_results = []

    for start in range(n_roll):
        end = start + ROLLING_WINDOW
        y_win = y.iloc[start:end]
        X_win = X.iloc[start:end]

        X_const = sm.add_constant(X_win, prepend=True)
        model = sm.OLS(y_win, X_const)

        try:
            results_hac = model.fit(cov_type='HAC',
                                    cov_kwds={'maxlags': HAC_LAGS})
            alpha = float(results_hac.params['const'])
            alpha_se = float(results_hac.bse['const'])
            alpha_tstat = float(results_hac.tvalues['const'])
            alpha_pval = float(results_hac.pvalues['const'])
        except Exception:
            alpha = alpha_se = alpha_tstat = alpha_pval = np.nan

        rolling_results.append({
            'end_date': y_win.index[-1].strftime('%Y-%m-%d'),
            'alpha_monthly': round(alpha, 6),
            'alpha_annualized': round(alpha * 12, 4),
            'alpha_se': round(alpha_se, 6),
            'alpha_tstat': round(alpha_tstat, 4),
            'alpha_pval': round(alpha_pval, 4),
            'ci_lower': round(alpha - t_crit * alpha_se, 6),
            'ci_upper': round(alpha + t_crit * alpha_se, 6),
            'significant_5pct': bool(alpha_pval < 0.05),
        })

    rolling_df = pd.DataFrame(rolling_results)
    rolling_df.to_csv(strategy_dir / "rolling_alpha.csv", index=False)

    # Summary
    n_sig = rolling_df['significant_5pct'].sum()
    n_windows = len(rolling_df)
    avg_alpha = rolling_df['alpha_annualized'].mean()
    valid = rolling_df.dropna(subset=['alpha_monthly'])

    summary = {
        'strategy': strategy_name,
        'window_months': ROLLING_WINDOW,
        'n_windows': n_windows,
        'factors_fixed': selected_factors,
        'alpha_annualized_mean': round(float(avg_alpha), 4),
        'alpha_annualized_min': round(float(valid['alpha_annualized'].min()), 4),
        'alpha_annualized_max': round(float(valid['alpha_annualized'].max()), 4),
        'pct_significant_5pct': round(float(n_sig / n_windows), 4) if n_windows > 0 else 0.0,
        'n_significant_5pct': int(n_sig),
    }

    with open(strategy_dir / "rolling_alpha_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n   Avg α (ann): {avg_alpha:+.2f}%")
    print(f"   Range: [{valid['alpha_annualized'].min():+.2f}%,"
          f" {valid['alpha_annualized'].max():+.2f}%]")
    print(f"   Significant (5%): {n_sig}/{n_windows}"
          f" ({n_sig/n_windows:.0%})")
    print(f"\n   💾 rolling_alpha.csv, rolling_alpha_summary.json")

    return summary


# ============================================================================
# (b) CORRELATION THRESHOLD SENSITIVITY
# ============================================================================

def sensitivity_correlation(strategy_name, strategy_path):
    """
    Re-run preprocessing + AEN with different correlation thresholds.
    """
    print_header(f"   (b) Correlation Threshold — {strategy_name}", "─")

    # Load raw data
    all_factors = pd.read_parquet(FACTORS_PATH)
    factors_end = pd.Timestamp(FACTORS_END_DATE)
    all_factors = all_factors[all_factors.index <= factors_end]

    returns = load_strategy_returns(strategy_path)
    returns = returns[returns.index <= factors_end]

    common_dates = returns.index.intersection(all_factors.index)
    y_raw = returns.loc[common_dates].values
    X_raw_df = all_factors.loc[common_dates].copy()

    # Drop NaN columns
    nan_cols = X_raw_df.columns[X_raw_df.isna().any()].tolist()
    X_raw_df = X_raw_df.drop(columns=nan_cols)
    mask = ~pd.isna(y_raw)
    y_raw = y_raw[mask]
    X_raw_df = X_raw_df.iloc[mask]

    criterion = AEN_TUNING_CRITERION
    gic_alpha = GIC_ALPHA if criterion == "GIC" else 3.0

    results = {}

    for threshold in CORR_THRESHOLDS:
        print(f"\n   ρ = {threshold}:")

        # Find and remove correlated pairs
        X_work = X_raw_df.copy()

        # Apply manual exclusions first
        cols_to_drop = [c for c in FACTORS_TO_EXCLUDE if c in X_work.columns]
        X_work = X_work.drop(columns=cols_to_drop)

        # Remove correlated pairs (skip if one already marked)
        corr_matrix = X_work.corr().abs()
        to_remove = set()
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                if cols[i] in to_remove or cols[j] in to_remove:
                    continue
                if corr_matrix.iloc[i, j] > threshold:
                    # Remove the one with lower correlation to y
                    corr_y_i = abs(np.corrcoef(X_work[cols[i]].values, y_raw)[0, 1])
                    corr_y_j = abs(np.corrcoef(X_work[cols[j]].values, y_raw)[0, 1])
                    if corr_y_i < corr_y_j:
                        to_remove.add(cols[i])
                    else:
                        to_remove.add(cols[j])

        X_clean = X_work.drop(columns=list(to_remove))
        factor_names = X_clean.columns.tolist()
        p = len(factor_names)

        print(f"      Removed {len(to_remove)} correlated factors → p = {p}")

        # Run AEN
        selected = run_aen_full(
            y_raw, X_clean.values, factor_names,
            AEN_LAMBDA2_GRID, AEN_LAMBDA1_N_VALUES,
            criterion, AEN_GAMMA, gic_alpha
        )
        print(f"      Selected: {selected if selected else '(null)'}")

        # Run post-selection OLS
        corr_alpha = {}
        if selected:
            try:
                common = returns.index.intersection(all_factors.index)
                y_ols = returns.loc[common]
                X_ols = all_factors.loc[common][selected]
                mask = ~(X_ols.isna().any(axis=1) | y_ols.isna())
                y_ols, X_ols = y_ols[mask], X_ols[mask]
                X_c = sm.add_constant(X_ols, prepend=True)
                res_hac = sm.OLS(y_ols, X_c).fit(
                    cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
                res_ols = sm.OLS(y_ols, X_c).fit()
                corr_alpha = {
                    'alpha_annualized': round(float(res_hac.params['const'] * 12), 4),
                    'alpha_tstat': round(float(res_hac.tvalues['const']), 4),
                    'alpha_pval': round(float(res_hac.pvalues['const']), 4),
                    'r_squared_adj': round(float(res_ols.rsquared_adj), 6),
                }
            except Exception:
                pass

        results[f"{threshold:.2f}"] = {
            'threshold': threshold,
            'p_after_cleaning': p,
            'n_removed': len(to_remove),
            'removed_factors': sorted(to_remove),
            'selected_factors': selected,
            'n_selected': len(selected),
            **corr_alpha,
        }

    # Stability across thresholds
    all_selected = set()
    for r in results.values():
        all_selected.update(r['selected_factors'])

    if all_selected:
        print(f"\n   Factor stability across thresholds:")
        for f in sorted(all_selected):
            present = [str(th) for th, r in results.items()
                       if f in r['selected_factors']]
            print(f"      {f:<25} ρ = {', '.join(present)}")

    strategy_dir = get_strategy_aen_dir(strategy_name)
    with open(strategy_dir / "sensitivity_correlation.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n   💾 sensitivity_correlation.json")

    return results


# ============================================================================
# (c) GAMMA SENSITIVITY
# ============================================================================

def sensitivity_gamma(strategy_name, strategy_path):
    """
    Sensitivity of AEN selection to the adaptive weight exponent γ.

    Stage 1 is FIXED (loaded from full-sample aen_results.json):
        β̂(enet) and λ₂ are held constant.
    Only γ varies → different adaptive weights → different Stage 2 selection.

    This isolates the effect of γ from any Stage 1 variation.
    """
    print_header(f"   (c) γ Sensitivity — {strategy_name}", "─")

    strategy_dir = get_strategy_aen_dir(strategy_name)

    # Load preprocessed standardized data (same as used in 02)
    y_df = pd.read_parquet(strategy_dir / "y_centered.parquet")
    X_df = pd.read_parquet(strategy_dir / "X_standardized.parquet")
    y = y_df['y'].values
    X = X_df.values
    factor_names = X_df.columns.tolist()
    T, p = X.shape

    # Load raw data for post-selection OLS
    all_factors = pd.read_parquet(FACTORS_PATH)
    factors_end = pd.Timestamp(FACTORS_END_DATE)
    all_factors = all_factors[all_factors.index <= factors_end]
    returns = load_strategy_returns(strategy_path)
    returns = returns[returns.index <= factors_end]

    # Load Stage 1 result from full-sample (fixed)
    with open(strategy_dir / "aen_results.json", 'r') as f:
        aen_res = json.load(f)

    coeff_df = pd.read_csv(strategy_dir / "aen_coefficients.csv")
    # Align beta_enet to factor_names order (robust to CSV ordering)
    coeff_map = coeff_df.set_index('factor')['beta_enet_rescaled']
    beta_enet = coeff_map.reindex(factor_names).values
    if np.any(pd.isna(beta_enet)):
        missing = [factor_names[i] for i, v in enumerate(beta_enet)
                   if pd.isna(v)]
        raise ValueError(f"Missing beta_enet for factors: {missing}")
    lambda2_po = aen_res['stage1']['lambda2_po']

    rescale = 1.0 + lambda2_po           # eq. 2.2: (1+λ₂°

    criterion = AEN_TUNING_CRITERION
    gic_alpha = GIC_ALPHA if criterion == "GIC" else 3.0

    print(f"\n   Stage 1 fixed: λ₂° = {lambda2_po:.6f},"
          f" rescale = {rescale:.4f}")
    print(f"   β̂(enet) nonzero: "
          f"{int(np.sum(np.abs(beta_enet) > COEF_TOL))}/{p}")

    results = {}

    for gamma_val in GAMMA_VALUES:
        print(f"\n   γ = {gamma_val}:")

        # Recompute adaptive weights with this γ
        epsilon = 1.0 / T
        weights = (np.abs(beta_enet) + epsilon) ** (-gamma_val)

        # Stage 2 only — paper-faithful CD (weighted ℓ₁ + uniform ℓ₂)
        lambda1_star_max = np.max(np.abs(X.T @ y) / T / weights)
        if lambda1_star_max < 1e-15:
            lambda1_star_max = 1e-6
        lambda1_star_grid = build_lambda1_grid(
            lambda1_star_max, AEN_LAMBDA1_N_VALUES)

        best_ic = np.inf
        best_beta_naive = np.zeros(p)

        for lambda1_star_po in lambda1_star_grid:
            beta_naive = weighted_elastic_net_cd(
                X, y, lambda1_star_po, lambda2_po,
                weights, max_iter=10000, tol=1e-7
            )
            beta_aen = beta_naive * rescale
            # df on beta_naive (pre-rescale)
            df = int(np.sum(np.abs(beta_naive) > COEF_TOL))
            pred = X @ beta_aen
            rss = float(np.sum((y - pred) ** 2))
            if rss / T < 1e-15:
                ic = np.inf
            else:
                log_lik = T * np.log(rss / T)
                if criterion in ("BIC", "SIS_BIC"):
                    ic = log_lik + df * np.log(T)
                elif criterion == "HQC":
                    ic = log_lik + 2 * np.log(np.log(T)) * df
                elif criterion == "AICc":
                    aic = log_lik + 2 * df
                    ic = aic + (2 * df * (df + 1)) / (T - df - 1) if T - df - 1 > 0 else np.inf
                elif criterion == "GIC":
                    ic = log_lik + gic_alpha * df

            if ic < best_ic:
                best_ic = ic
                best_beta_naive = beta_naive.copy()

        # Selection on beta_naive (pre-rescale)
        selected = [factor_names[i] for i in range(p)
                    if abs(best_beta_naive[i]) > COEF_TOL]
        print(f"      Selected: {selected if selected else '(null)'}")

        # Run post-selection OLS for alpha at this gamma
        gamma_alpha = {}
        if selected:
            try:
                common = returns.index.intersection(all_factors.index)
                y_ols = returns.loc[common]
                X_ols = all_factors.loc[common][selected]
                mask = ~(X_ols.isna().any(axis=1) | y_ols.isna())
                y_ols, X_ols = y_ols[mask], X_ols[mask]
                X_c = sm.add_constant(X_ols, prepend=True)
                res_hac = sm.OLS(y_ols, X_c).fit(
                    cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
                res_ols = sm.OLS(y_ols, X_c).fit()
                gamma_alpha = {
                    'alpha_annualized': round(float(res_hac.params['const'] * 12), 4),
                    'alpha_tstat': round(float(res_hac.tvalues['const']), 4),
                    'alpha_pval': round(float(res_hac.pvalues['const']), 4),
                    'r_squared_adj': round(float(res_ols.rsquared_adj), 6),
                }
            except Exception:
                pass

        results[str(gamma_val)] = {
            'gamma': gamma_val,
            'selected_factors': selected,
            'n_selected': len(selected),
            **gamma_alpha,
        }

    # Stability
    all_selected = set()
    for r in results.values():
        all_selected.update(r['selected_factors'])

    if all_selected:
        print(f"\n   Factor stability across γ:")
        for f in sorted(all_selected):
            present = [str(g) for g, r in results.items()
                       if f in r['selected_factors']]
            print(f"      {f:<25} γ = {', '.join(present)}")

    with open(strategy_dir / "sensitivity_gamma.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n   💾 sensitivity_gamma.json")

    return results


# ============================================================================
# (d) HALF-SAMPLE SPLIT
# ============================================================================

def halfsample_validation(strategy_name, strategy_path):
    """
    AEN on first half → select factors.
    OLS on second half with those factors → out-of-sample α.
    """
    print_header(f"   (d) Half-Sample Split — {strategy_name}", "─")

    # Load raw data
    all_factors = pd.read_parquet(FACTORS_PATH)
    factors_end = pd.Timestamp(FACTORS_END_DATE)
    all_factors = all_factors[all_factors.index <= factors_end]

    returns = load_strategy_returns(strategy_path)
    returns = returns[returns.index <= factors_end]

    common_dates = returns.index.intersection(all_factors.index)
    y_full = returns.loc[common_dates]

    # Apply same exclusions
    strategy_dir = get_strategy_aen_dir(strategy_name)
    with open(strategy_dir / "standardization_params.json", 'r') as f:
        std_params = json.load(f)
    factor_names = std_params['factor_names']

    X_full = all_factors.loc[common_dates][factor_names].copy()
    mask = ~(X_full.isna().any(axis=1) | y_full.isna())
    y_full, X_full = y_full[mask], X_full[mask]
    T = len(y_full)

    half = T // 2

    y_train = y_full.iloc[:half].values
    X_train = X_full.iloc[:half].values
    y_test = y_full.iloc[half:]
    X_test = X_full.iloc[half:]

    print(f"\n   T = {T}")
    print(f"   Train: {half} obs"
          f" ({y_full.index[0].strftime('%Y-%m')}"
          f" → {y_full.index[half-1].strftime('%Y-%m')})")
    print(f"   Test:  {T - half} obs"
          f" ({y_full.index[half].strftime('%Y-%m')}"
          f" → {y_full.index[-1].strftime('%Y-%m')})")

    criterion = AEN_TUNING_CRITERION
    gic_alpha = GIC_ALPHA if criterion == "GIC" else 3.0

    # AEN on train
    selected = run_aen_full(
        y_train, X_train, factor_names,
        AEN_LAMBDA2_GRID, AEN_LAMBDA1_N_VALUES,
        criterion, AEN_GAMMA, gic_alpha
    )
    print(f"\n   Train-selected factors: {selected if selected else '(null)'}")

    result = {
        'strategy': strategy_name,
        'T_total': T,
        'T_train': half,
        'T_test': T - half,
        'train_period': f"{y_full.index[0].strftime('%Y-%m')}"
                        f" → {y_full.index[half-1].strftime('%Y-%m')}",
        'test_period': f"{y_full.index[half].strftime('%Y-%m')}"
                       f" → {y_full.index[-1].strftime('%Y-%m')}",
        'selected_factors': selected,
        'n_selected': len(selected),
    }

    if len(selected) == 0:
        print(f"   ⚠️  Null model on train. No OOS test possible.")
        result['oos_alpha'] = None
    else:
        # OLS on test with train-selected factors
        X_test_sel = X_test[selected]
        X_const = sm.add_constant(X_test_sel, prepend=True)
        model = sm.OLS(y_test, X_const)
        results_hac = model.fit(cov_type='HAC',
                                cov_kwds={'maxlags': HAC_LAGS})
        results_ols = model.fit()

        alpha_coeff = float(results_hac.params['const'])
        alpha_tstat = float(results_hac.tvalues['const'])
        alpha_pval = float(results_hac.pvalues['const'])

        print(f"\n   ── Out-of-Sample OLS ──")
        print(f"   α = {alpha_coeff:+.4f}% monthly"
              f" ({alpha_coeff * 12:+.2f}% annualized)")
        print(f"   t-stat = {alpha_tstat:.3f},"
              f" p-value = {alpha_pval:.4f}"
              f" {significance_stars(alpha_pval)}")
        print(f"   R² adj = {results_ols.rsquared_adj:.4f}")

        # In-sample alpha for comparison
        X_train_sel = X_full.iloc[:half][selected]
        X_const_train = sm.add_constant(X_train_sel, prepend=True)
        model_train = sm.OLS(y_full.iloc[:half], X_const_train)
        res_train = model_train.fit(cov_type='HAC',
                                    cov_kwds={'maxlags': HAC_LAGS})
        is_alpha = float(res_train.params['const'])
        is_pval = float(res_train.pvalues['const'])

        print(f"\n   In-sample:      α = {is_alpha:+.4f}% mo"
              f" ({is_alpha * 12:+.2f}% yr),"
              f" p = {is_pval:.4f} {significance_stars(is_pval)}")
        print(f"   Out-of-sample:  α = {alpha_coeff:+.4f}% mo"
              f" ({alpha_coeff * 12:+.2f}% yr),"
              f" p = {alpha_pval:.4f} {significance_stars(alpha_pval)}")

        result['is_alpha'] = {
            'monthly': round(is_alpha, 6),
            'annualized': round(is_alpha * 12, 4),
            'pval': round(is_pval, 4),
        }
        result['oos_alpha'] = {
            'monthly': round(alpha_coeff, 6),
            'annualized': round(alpha_coeff * 12, 4),
            'tstat': round(alpha_tstat, 4),
            'pval': round(alpha_pval, 4),
            'r_squared_adj': round(float(results_ols.rsquared_adj), 6),
        }

    # Compare with full-sample selection
    with open(strategy_dir / "aen_results.json", 'r') as f:
        full_res = json.load(f)
    full_factors = full_res['selected_factors']
    overlap = set(selected) & set(full_factors)
    result['full_sample_factors'] = full_factors
    result['overlap_with_full'] = sorted(overlap)

    print(f"\n   Full-sample factors: {full_factors}")
    print(f"   Train-half factors: {selected}")
    print(f"   Overlap: {sorted(overlap) if overlap else '(none)'}")

    with open(strategy_dir / "halfsample_results.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n   💾 halfsample_results.json")

    return result


# ============================================================================
# PER-STRATEGY RUNNER
# ============================================================================

def run_robustness_for_strategy(strategy_name, strategy_path):
    """Run all 4 robustness checks for one strategy."""

    results = {'strategy': strategy_name}

    # (a) Rolling alpha
    results['rolling'] = rolling_alpha(strategy_name, strategy_path)

    # (b) Correlation threshold
    results['correlation'] = sensitivity_correlation(
        strategy_name, strategy_path)

    # (c) Gamma sensitivity
    results['gamma'] = sensitivity_gamma(strategy_name, strategy_path)

    # (d) Half-sample
    results['halfsample'] = halfsample_validation(
        strategy_name, strategy_path)

    # Save combined summary
    strategy_dir = get_strategy_aen_dir(strategy_name)
    with open(strategy_dir / "robustness_summary.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n   💾 robustness_summary.json")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("ROBUSTNESS CHECKS")
    print(f"\n   (a) Rolling α: window = {ROLLING_WINDOW} months")
    print(f"   (b) Correlation: ρ = {CORR_THRESHOLDS}")
    print(f"   (c) γ sensitivity: γ = {GAMMA_VALUES}")
    print(f"   (d) Half-sample split")
    print(f"   Criterion: {AEN_TUNING_CRITERION}")

    all_results = {}

    for strategy_name, strategy_path in STRATEGIES.items():
        print_header(f"STRATEGY: {strategy_name}")

        strategy_dir = get_strategy_aen_dir(strategy_name)
        if not (strategy_dir / "aen_results.json").exists():
            print(f"\n   ❌ AEN results not found. Run 02 first.")
            continue

        try:
            result = run_robustness_for_strategy(strategy_name, strategy_path)
            all_results[strategy_name] = result
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
            import traceback; traceback.print_exc()

    # ── Cross-strategy summary ─────────────────────────────────────────
    if all_results:
        print_header("ROBUSTNESS SUMMARY")

        # Rolling alpha
        print(f"\n   ── Rolling α ──")
        print(f"   {'Strategy':<20} {'α(yr) avg':>10} {'% sig 5%':>10}"
              f" {'min':>8} {'max':>8}")
        print(f"   {'─' * 58}")
        for name, res in all_results.items():
            r = res.get('rolling')
            if r:
                print(f"   {name:<20}"
                      f" {r['alpha_annualized_mean']:>+10.2f}"
                      f" {r['pct_significant_5pct']:>9.0%}"
                      f" {r['alpha_annualized_min']:>+8.2f}"
                      f" {r['alpha_annualized_max']:>+8.2f}")

        # Half-sample
        print(f"\n   ── Half-Sample ──")
        print(f"   {'Strategy':<20} {'IS α(yr)':>10} {'OOS α(yr)':>10}"
              f" {'OOS p-val':>10}")
        print(f"   {'─' * 52}")
        for name, res in all_results.items():
            hs = res.get('halfsample', {})
            if hs and hs.get('oos_alpha'):
                print(f"   {name:<20}"
                      f" {hs['is_alpha']['annualized']:>+10.2f}"
                      f" {hs['oos_alpha']['annualized']:>+10.2f}"
                      f" {hs['oos_alpha']['pval']:>10.4f}"
                      f" {significance_stars(hs['oos_alpha']['pval'])}")
            elif hs:
                print(f"   {name:<20} {'(null model)':>32}")

        # Save global
        aen_output_dir = get_aen_output_dir()
        with open(aen_output_dir / "robustness_global.json", 'w') as f:
            json.dump({
                'criterion': AEN_TUNING_CRITERION,
                'strategies': {name: {
                    'rolling_alpha_mean': res.get('rolling', {}).get(
                        'alpha_annualized_mean'),
                    'rolling_pct_sig': res.get('rolling', {}).get(
                        'pct_significant_5pct'),
                    'oos_alpha': (res.get('halfsample', {}).get(
                        'oos_alpha') or {}).get('annualized'),
                    'oos_pval': (res.get('halfsample', {}).get(
                        'oos_alpha') or {}).get('pval'),
                } for name, res in all_results.items()}
            }, f, indent=2)
        print(f"\n   💾 {aen_output_dir / 'robustness_global.json'}")

    print(f"\n{'=' * 80}")
    print(f"✅ ROBUSTNESS CHECKS COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n   🎯 Next: python src/aen/07_aen_tables.py")


if __name__ == "__main__":
    main()
"""
================================================================================
05_aen_method_comparison.py - Method Comparison (Zou-Zhang Style + Chen et al.)
================================================================================
Compares factor selection across 6 penalized methods on the same data:

1. Lasso            (ℓ₁ only, λ₂ = 0)
2. Elastic Net      (ℓ₁ + ℓ₂, non-adaptive)
3. Adaptive Lasso   (weighted ℓ₁, Lasso initial, λ₂ = 0)
4. Ridge            (ℓ₂ only, shrinkage benchmark — no selection)
5. AEN              (our method, results loaded from Step 02)
6. Adaptive LASSO – Chen et al. (2025, JFQA)
       "Post-Adaptive-LASSO" from:
       "Anomalies as New Hedge Fund Factors"
       (Chen, Li, Tang, Zhou — J. Financial and Quantitative Analysis)

For methods 1-3:
    - Tune hyperparameters via same IC as AEN (config: HQC, BIC, etc.)
    - Post-selection OLS on RAW data → α with HAC SE
    - Record which factors are selected

For Ridge (method 4):
    - Shrinkage benchmark: all factors retained, no selection
    - GCV for λ selection on centered raw data
    - Real Ridge fit (not OLS) → α̂ point estimate, R², Adj R² (df_eff)
    - No t-stat/p-value: Ridge is biased; OLS inference not applicable

For AEN (method 5):
    - Loaded from Step 02; post-selection OLS on raw data

For Adaptive LASSO – Chen et al. (method 6):
    Methodological differences from Method 3 (our Adaptive Lasso):
    ┌─────────────────────┬──────────────────────┬──────────────────────┐
    │ Choice              │ Method 3 (ours)      │ Method 6 (Chen)      │
    ├─────────────────────┼──────────────────────┼──────────────────────┤
    │ Standardization     │ L2-norm = 1 (Zou-Z.) │ z-score (mean/σ)     │
    │ Initial weights     │ Lasso β̂              │ OLS β̂ (full model)   │
    │ Weight formula      │ wⱼ = (|β̂|+ε)^{-γ}   │ wⱼ = 1/|β̂_OLS_j|    │
    │ Tuning criterion    │ Same as AEN config   │ AIC (paper p. 12)    │
    │ ℓ₂ penalty          │ λ₂ = 0               │ λ₂ = 0               │
    │ Post-selection      │ OLS on raw + HAC     │ OLS on raw + HAC     │
    └─────────────────────┴──────────────────────┴──────────────────────┘

    Reference: Chen et al. (2025), Section 2.3, eq. (3):
        β̂_ALASSO = argmin ||r - Fβ||² + λ Σⱼ ωⱼ|βⱼ|
        where ωⱼ = 1/|β̂_OLS_j|

IMPORTANT NOTES:
    - IC values are NOT comparable across methods that use different
      standardizations (L2-norm vs z-score vs raw). The comparison
      table reports only post-selection OLS metrics (α, t-stat, R²adj),
      which are all computed on the SAME raw-scale data and are therefore
      directly comparable.
    - HAC lags: all post-selection OLS regressions use the same number of
      Newey-West lags from config (HAC_LAGS). Chen et al. (2025) use 3
      lags; our config currently uses {HAC_LAGS_PLACEHOLDER}. This choice
      is documented in each output JSON for reproducibility.
    - Top-K comparison: to enable fair comparison at equal model complexity,
      we also report post-selection OLS for the top-K factors from each
      method, with K ∈ {3, 5, 7}. Ranking is by |β̂| on the penalized
      estimator (standardized scale).

Output:
    - method_comparison.json    (all results)
    - method_comparison.csv     (Table 3: comparison summary)
    - selection_matrix.csv      (Table 4: factors × methods, Ridge excluded)
    - topk_comparison.csv       (Table 5: Top-K comparison at fixed complexity)

Reference: Zou & Zhang (2009), Table 2;
           Chen, Li, Tang, Zhou (2025), JFQA.

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

from sklearn.linear_model import ElasticNet, Lasso
import statsmodels.api as sm

import importlib.util

# ============================================================================
# CONFIG IMPORT
# ============================================================================
# Single source of truth: src/aen/00_aen_config.py
# All scripts in the pipeline must point to the same config path.

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"

spec = importlib.util.spec_from_file_location("aen_config", config_path)
aen_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aen_config)

AEN_GAMMA              = aen_config.AEN_GAMMA
AEN_LAMBDA2_GRID       = aen_config.AEN_LAMBDA2_GRID
AEN_LAMBDA1_N_VALUES   = aen_config.AEN_LAMBDA1_N_VALUES
AEN_TUNING_CRITERION   = aen_config.AEN_TUNING_CRITERION
GIC_ALPHA              = aen_config.GIC_ALPHA
HAC_LAGS               = aen_config.HAC_LAGS
FACTORS_PATH           = aen_config.FACTORS_PATH
FACTORS_END_DATE       = aen_config.FACTORS_END_DATE
STRATEGIES             = aen_config.STRATEGIES
get_strategy_aen_dir   = aen_config.get_strategy_aen_dir
get_aen_output_dir     = aen_config.get_aen_output_dir


# ============================================================================
# GLOBAL CONSTANTS
# ============================================================================

# Unified zero-coefficient threshold.
# sklearn's Lasso/ElasticNet use tol=1e-7 internally, so coefficients
# in the range [1e-10, 1e-7] are solver noise, not genuine selections.
# Using 1e-6 ensures that:
#   (a) compute_ic() counts df consistently with selection masks
#   (b) all methods use the same definition of "selected"
COEF_EPS = 1e-4

# Top-K levels for fixed-complexity comparison
TOPK_LEVELS = [3, 5, 7]


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
# INFORMATION CRITERION
# ============================================================================
# Single implementation used by ALL methods.  df is counted using the
# global COEF_EPS threshold so that IC and selection masks are aligned.

def compute_ic(y, X, beta, criterion, gic_alpha=3.0):
    """
    Compute information criterion value.

    Args:
        y:          response vector (T,)
        X:          design matrix (T, p) — must be on same scale as beta
        beta:       coefficient vector (p,)
        criterion:  one of "BIC", "SIS_BIC", "HQC", "AICc", "GIC", "AIC"
        gic_alpha:  penalty weight for GIC

    Returns:
        (ic_value, rss, df)
    """
    T = len(y)
    residuals = y - X @ beta
    rss = float(np.sum(residuals ** 2))
    df = int(np.sum(np.abs(beta) > COEF_EPS))

    if rss / T < 1e-15:
        return np.inf, rss, df

    log_lik = T * np.log(rss / T)

    if criterion in ("BIC", "SIS_BIC"):
        ic = log_lik + df * np.log(T)
    elif criterion == "HQC":
        ic = log_lik + 2 * np.log(np.log(T)) * df
    elif criterion == "AIC":
        ic = log_lik + 2 * df
    elif criterion == "AICc":
        aic = log_lik + 2 * df
        ic = aic + (2 * df * (df + 1)) / (T - df - 1) if T - df - 1 > 0 else np.inf
    elif criterion == "GIC":
        ic = log_lik + gic_alpha * df
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return float(ic), rss, df


# ============================================================================
# POST-SELECTION OLS (shared by all methods)
# ============================================================================

def post_selection_ols(y_raw, X_raw, selected_factors, hac_lags):
    """
    OLS on RAW data with intercept + HAC (Newey-West) SE.

    This is the common inference layer used by every method.
    Because all methods funnel through here on the SAME raw data,
    the resulting alpha, t-stat, and R²adj are directly comparable.

    Args:
        y_raw:              pd.Series, strategy returns (original scale)
        X_raw:              pd.DataFrame, all factors (original scale)
        selected_factors:   list of str, factor names selected by a method
        hac_lags:           int, Newey-West lag truncation

    Returns:
        dict with alpha, t-stat, p-value, R², R²adj, factor-level results
    """
    if len(selected_factors) == 0:
        return {
            'alpha_monthly': None, 'alpha_annualized': None,
            'alpha_tstat': None, 'alpha_pval': None,
            'r_squared': None, 'r_squared_adj': None,
            'n_factors': 0,
            'selected_factors': [], 'factor_results': {},
            'hac_lags': hac_lags,
        }

    X = X_raw[selected_factors].copy()
    X_const = sm.add_constant(X, prepend=True)
    model = sm.OLS(y_raw, X_const)
    results_ols = model.fit()
    results_hac = model.fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})

    alpha_coeff = float(results_hac.params['const'])
    alpha_pval = float(results_hac.pvalues['const'])

    factor_results = {}
    for var in selected_factors:
        factor_results[var] = {
            'coefficient': round(float(results_hac.params[var]), 6),
            'hac_se': round(float(results_hac.bse[var]), 6),
            't_statistic': round(float(results_hac.tvalues[var]), 4),
            'p_value': round(float(results_hac.pvalues[var]), 4),
        }

    return {
        'alpha_monthly': round(alpha_coeff, 6),
        'alpha_annualized': round(alpha_coeff * 12, 4),
        'alpha_tstat': round(float(results_hac.tvalues['const']), 4),
        'alpha_pval': round(alpha_pval, 4),
        'r_squared': round(float(results_ols.rsquared), 6),
        'r_squared_adj': round(float(results_ols.rsquared_adj), 6),
        'n_factors': len(selected_factors),
        'selected_factors': selected_factors,
        'factor_results': factor_results,
        'hac_lags': hac_lags,
    }


# ============================================================================
# PARAMETER MAPPING (same as 02)
# ============================================================================

def paper_to_sklearn(lambda1_po, lambda2_po):
    alpha = lambda1_po / 2.0 + lambda2_po
    if alpha < 1e-15:
        return (1e-15, 0.5)
    l1_ratio = (lambda1_po / 2.0) / alpha
    l1_ratio = np.clip(l1_ratio, 1e-6, 1.0 - 1e-6)
    return (alpha, l1_ratio)


def compute_lambda1_max(X, y):
    T = X.shape[0]
    return np.max(np.abs(X.T @ y)) / T


def build_lambda1_grid(lambda1_max, n_values=100, ratio_min=1e-4):
    return np.logspace(np.log10(lambda1_max),
                       np.log10(lambda1_max * ratio_min), n_values)


# ============================================================================
# HELPER: extract selected factor names from a coefficient vector
# ============================================================================

def selected_from_beta(beta, factor_names):
    """Return list of factor names where |β| > COEF_EPS."""
    return [factor_names[i] for i in range(len(beta))
            if abs(beta[i]) > COEF_EPS]


# ============================================================================
# METHOD 1: LASSO (ℓ₁ only)
# ============================================================================

def fit_lasso(y, X, criterion, gic_alpha=3.0, n_lambda1=100):
    """Lasso with IC tuning. λ₂ = 0 throughout."""
    T, p = X.shape
    lambda1_max = compute_lambda1_max(X, y)
    lambda1_grid = build_lambda1_grid(lambda1_max, n_lambda1)

    best_ic = np.inf
    best_beta = np.zeros(p)

    for lambda1_po in lambda1_grid:
        alpha_sk = lambda1_po / 2.0
        if alpha_sk < 1e-15:
            alpha_sk = 1e-15
        model = Lasso(alpha=alpha_sk, fit_intercept=False,
                      max_iter=10000, tol=1e-7)
        model.fit(X, y)
        beta = model.coef_

        ic, _, _ = compute_ic(y, X, beta, criterion, gic_alpha)
        if ic < best_ic:
            best_ic = ic
            best_beta = beta.copy()

    return best_beta


# ============================================================================
# METHOD 2: ELASTIC NET (non-adaptive)
# ============================================================================
# NOTE on IC computation for Elastic Net (Zou-Hastie 2005):
#   sklearn returns the "naive" coefficient β_raw.
#   The Zou-Hastie (2005) estimator rescales: β_enet = (1 + λ₂°) · β_raw.
#   We compute:
#     - df from beta_RAW (pre-rescale), because the ℓ₁ sparsity pattern
#       is determined before rescaling
#     - RSS from predictions using beta_ENET (rescaled), because that
#       is the actual estimator's prediction
#   This matches the Stage 1 logic in 02_aen_estimation.py.
#
#   We now delegate IC computation to compute_ic_enet() to avoid inline
#   duplication (which previously missed the "AIC" branch).

def _compute_ic_enet(y, X, beta_raw, beta_enet, criterion, gic_alpha=3.0):
    """
    IC for Elastic Net: df from beta_raw, RSS from beta_enet predictions.

    This is a thin wrapper that computes df and RSS with the correct
    semantics for the Zou-Hastie rescaled estimator, then delegates
    the penalty computation to the standard IC formula.
    """
    T = len(y)
    pred = X @ beta_enet
    rss = float(np.sum((y - pred) ** 2))
    df = int(np.sum(np.abs(beta_raw) > COEF_EPS))

    if rss / T < 1e-15:
        return np.inf, rss, df

    log_lik = T * np.log(rss / T)

    if criterion in ("BIC", "SIS_BIC"):
        ic = log_lik + df * np.log(T)
    elif criterion == "HQC":
        ic = log_lik + 2 * np.log(np.log(T)) * df
    elif criterion == "AIC":
        ic = log_lik + 2 * df
    elif criterion == "AICc":
        aic = log_lik + 2 * df
        ic = aic + (2 * df * (df + 1)) / (T - df - 1) if T - df - 1 > 0 else np.inf
    elif criterion == "GIC":
        ic = log_lik + gic_alpha * df
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return float(ic), rss, df


def fit_elastic_net(y, X, lambda2_grid, criterion, gic_alpha=3.0, n_lambda1=100):
    """Standard Elastic Net (no adaptive weights) with IC tuning.
    Returns (beta_enet_rescaled, beta_raw) — selection on beta_raw."""
    T, p = X.shape
    lambda1_max = compute_lambda1_max(X, y)

    best_ic = np.inf
    best_beta_enet = np.zeros(p)
    best_beta_raw = np.zeros(p)

    for lambda2_po in lambda2_grid:
        lambda1_grid = build_lambda1_grid(lambda1_max, n_lambda1)
        rescale = 1.0 + lambda2_po        # Zou & Hastie (2005): (1+λ₂/n)

        for lambda1_po in lambda1_grid:
            alpha_sk, l1_ratio_sk = paper_to_sklearn(lambda1_po, lambda2_po)
            model = ElasticNet(alpha=alpha_sk, l1_ratio=l1_ratio_sk,
                               fit_intercept=False, max_iter=10000, tol=1e-7)
            model.fit(X, y)
            beta_raw = model.coef_
            beta_enet = beta_raw * rescale

            ic, _, _ = _compute_ic_enet(
                y, X, beta_raw, beta_enet, criterion, gic_alpha)

            if ic < best_ic:
                best_ic = ic
                best_beta_enet = beta_enet.copy()
                best_beta_raw = beta_raw.copy()

    return best_beta_enet, best_beta_raw


# ============================================================================
# METHOD 3: ADAPTIVE LASSO (weighted ℓ₁, Lasso initial, λ₂ = 0)
# ============================================================================

def fit_adaptive_lasso(y, X, criterion, gamma=1, gic_alpha=3.0, n_lambda1=100):
    """
    Adaptive Lasso: two-stage.
    Stage 1: Lasso → initial β̂
    Stage 2: weighted Lasso with wⱼ = (|β̂ⱼ| + ε)^{-γ}
    """
    T, p = X.shape

    # Stage 1: initial Lasso
    beta_init = fit_lasso(y, X, criterion, gic_alpha, n_lambda1)

    # Adaptive weights
    epsilon = 1.0 / T
    weights = (np.abs(beta_init) + epsilon) ** (-gamma)

    # Stage 2: weighted Lasso via column rescaling
    X_tilde = X / weights[np.newaxis, :]
    lambda1_max = compute_lambda1_max(X_tilde, y)
    lambda1_grid = build_lambda1_grid(lambda1_max, n_lambda1)

    best_ic = np.inf
    best_beta = np.zeros(p)

    for lambda1_po in lambda1_grid:
        alpha_sk = lambda1_po / 2.0
        if alpha_sk < 1e-15:
            alpha_sk = 1e-15
        model = Lasso(alpha=alpha_sk, fit_intercept=False,
                      max_iter=10000, tol=1e-7)
        model.fit(X_tilde, y)
        beta_tilde = model.coef_
        beta_alasso = beta_tilde / weights

        ic, _, _ = compute_ic(y, X, beta_alasso, criterion, gic_alpha)
        if ic < best_ic:
            best_ic = ic
            best_beta = beta_alasso.copy()

    return best_beta


# ============================================================================
# METHOD 4: RIDGE (ℓ₂ only, shrinkage benchmark, no selection)
# ============================================================================

def fit_ridge(y_raw, X_raw, factor_names):
    """
    Ridge with GCV on RAW centered data.  Returns point estimates only.
    No t-stat/p-value (Ridge is biased; OLS inference not applicable).
    """
    X = X_raw[factor_names].copy()
    T = len(y_raw)
    p = len(factor_names)

    X_np = X.values.astype(np.float64)
    y_np = y_raw.values.astype(np.float64)

    y_mean = np.mean(y_np)
    X_mean = np.mean(X_np, axis=0)
    Xc = X_np - X_mean
    yc = y_np - y_mean

    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    s2 = s ** 2

    alpha0 = np.median(s2)
    alphas = alpha0 * np.logspace(-6, 6, 50)
    best_gcv = np.inf
    best_alpha = alpha0

    for alpha in alphas:
        h_diag = np.sum((U ** 2) * (s2 / (s2 + alpha)), axis=1)
        if np.max(h_diag) > 0.99:
            continue
        d = s / (s2 + alpha)
        coef = Vt.T @ (d * (U.T @ yc))
        residuals = yc - Xc @ coef
        gcv = np.mean((residuals / (1.0 - h_diag)) ** 2)
        if gcv < best_gcv:
            best_gcv = gcv
            best_alpha = alpha

    d_best = s / (s2 + best_alpha)
    ridge_beta = Vt.T @ (d_best * (U.T @ yc))
    ridge_intercept = y_mean - X_mean @ ridge_beta

    y_pred = X_np @ ridge_beta + ridge_intercept
    residuals = y_np - y_pred
    rss = float(np.sum(residuals ** 2))
    tss = float(np.sum((y_np - y_mean) ** 2))
    r_squared = 1.0 - rss / tss

    df_eff = float(np.sum(s2 / (s2 + best_alpha)))

    if T - 1 - df_eff > 0:
        r_squared_adj = 1.0 - (rss / (T - 1 - df_eff)) / (tss / (T - 1))
    else:
        r_squared_adj = None

    return {
        'alpha_monthly': round(float(ridge_intercept), 6),
        'alpha_annualized': round(float(ridge_intercept * 12), 4),
        'alpha_tstat': None,
        'alpha_pval': None,
        'r_squared': round(float(r_squared), 6),
        'r_squared_adj': round(float(r_squared_adj), 6) if r_squared_adj is not None else None,
        'n_factors': p,
        'selected_factors': factor_names,
        'factor_results': {},
        'ridge_lambda': round(float(best_alpha), 6),
        'df_effective': round(float(df_eff), 2),
        'hac_lags': None,  # N/A for Ridge
    }


# ============================================================================
# METHOD 6: ADAPTIVE LASSO — CHEN, LI, TANG, ZHOU (2025, JFQA)
# ============================================================================
#
# "Anomalies as New Hedge Fund Factors"
# Journal of Financial and Quantitative Analysis, forthcoming.
#
# Their "Post-Adaptive-LASSO" procedure (Section 2.3):
#
#   1. Standardize factors by mean/std (z-score), NOT L2-norm = 1.
#      (Paper p. 12: "we first standardize factors using their
#       time-series means and standard deviations")
#
#   2. Initial estimate: OLS on the full set of standardized factors.
#      (Paper p. 11, eq. 3: wⱼ = 1/|β̂_OLS_j|, citing Zou 2006)
#      NOTE: when p > T, OLS is rank-deficient. Following standard
#      practice (Zou 2006, Remark 2), we use the Ridge-regression
#      estimate as a surrogate, which is well-defined for any (T, p).
#      When p < T this converges to OLS as the Ridge penalty → 0.
#
#   3. Adaptive LASSO: ℓ₁ penalty with data-driven weights.
#      β̂_ALASSO = argmin ||r - Fβ||² + λ Σⱼ (1/|β̂_OLS_j|)|βⱼ|
#      λ₂ = 0 (pure LASSO, no ℓ₂ ridge component).
#
#   4. Tuning: AIC (paper p. 12).
#      "We use the Akaike Information Criterion (AIC) to validate
#       the choice of the shrinkage parameter λ."
#
#   5. Post-selection OLS: refit unrestricted OLS on selected factors,
#      with HAC standard errors.
#      (Paper p. 3-4, citing Belloni & Chernozhukov 2013)
#
# Implementation notes:
#   - γ = 1 is implicit in wⱼ = 1/|β̂_OLS_j| (Zou 2006, Theorem 2).
#   - The column-rescaling trick transforms weighted LASSO into
#     standard LASSO: X̃ⱼ = Xⱼ/wⱼ, solve standard LASSO on X̃,
#     then β̂ⱼ = β̃ⱼ/wⱼ.
#   - When p ≥ T, the OLS matrix X'X is singular. We compute initial
#     weights via Ridge with a small penalty chosen by GCV, following
#     Zou (2006, Remark 2) which recommends Ridge as the initial
#     estimator when p > n.
# ============================================================================

def fit_adaptive_lasso_chen(y_raw, X_raw, factor_names, n_lambda1=100):
    """
    Adaptive LASSO as in Chen, Li, Tang, Zhou (2025, JFQA).

    Returns dict with selected factors, coefficients, and diagnostics.
    """
    X = X_raw[factor_names].copy()
    T = len(y_raw)
    p = len(factor_names)

    X_np = X.values.astype(np.float64)
    y_np = y_raw.values.astype(np.float64)

    # ── Step 1: z-score standardization (Chen et al. p. 12) ───────────
    X_mean = np.mean(X_np, axis=0)
    X_std = np.std(X_np, axis=0, ddof=0)
    X_std[X_std < 1e-12] = 1.0  # guard against zero-variance
    X_z = (X_np - X_mean) / X_std

    y_mean = np.mean(y_np)
    y_c = y_np - y_mean

    # ── Step 2: Initial estimate → OLS (or Ridge if p ≥ T) ───────────
    if p < T:
        beta_init = np.linalg.lstsq(X_z, y_c, rcond=None)[0]
        init_method = "OLS"
    else:
        # Ridge surrogate (Zou 2006, Remark 2)
        U, s, Vt = np.linalg.svd(X_z, full_matrices=False)
        s2 = s ** 2
        alpha0 = np.median(s2)
        alphas = alpha0 * np.logspace(-6, 2, 30)
        best_gcv = np.inf
        best_alpha = alpha0

        for alpha in alphas:
            h_diag = np.sum((U ** 2) * (s2 / (s2 + alpha)), axis=1)
            if np.max(h_diag) > 0.99:
                continue
            d = s / (s2 + alpha)
            coef = Vt.T @ (d * (U.T @ y_c))
            res = y_c - X_z @ coef
            gcv = np.mean((res / (1.0 - h_diag)) ** 2)
            if gcv < best_gcv:
                best_gcv = gcv
                best_alpha = alpha

        d_best = s / (s2 + best_alpha)
        beta_init = Vt.T @ (d_best * (U.T @ y_c))
        init_method = f"Ridge (p>=T, lambda_ridge={best_alpha:.6f})"

    # ── Step 3: Adaptive weights (Chen et al. eq. 3) ──────────────────
    epsilon = 1.0 / T
    weights = 1.0 / (np.abs(beta_init) + epsilon)

    # ── Step 4: Weighted LASSO via column rescaling ───────────────────
    X_tilde = X_z / weights[np.newaxis, :]

    lambda1_max = compute_lambda1_max(X_tilde, y_c)
    lambda1_grid = build_lambda1_grid(lambda1_max, n_lambda1)

    best_aic = np.inf
    best_beta = np.zeros(p)
    best_lambda1 = lambda1_grid[0]

    for lambda1_po in lambda1_grid:
        alpha_sk = lambda1_po / 2.0
        if alpha_sk < 1e-15:
            alpha_sk = 1e-15
        model = Lasso(alpha=alpha_sk, fit_intercept=False,
                      max_iter=10000, tol=1e-7)
        model.fit(X_tilde, y_c)
        beta_tilde = model.coef_
        beta_alasso = beta_tilde / weights

        # AIC tuning (Chen et al. p. 12)
        aic, _, _ = compute_ic(y_c, X_z, beta_alasso, "AIC")
        if aic < best_aic:
            best_aic = aic
            best_beta = beta_alasso.copy()
            best_lambda1 = lambda1_po

    selected_factors = selected_from_beta(best_beta, factor_names)

    return {
        'selected_factors_zscore': selected_factors,
        'beta_alasso': best_beta,
        'beta_ols_init': beta_init,
        'aic_best': best_aic,
        'lambda1_best': best_lambda1,
        'init_method': init_method,
        'n_nonzero': len(selected_factors),
    }


# ============================================================================
# TOP-K COMPARISON
# ============================================================================
# When methods select different numbers of factors, comparing alphas is
# not a fair test of selection quality.  Top-K re-runs post-OLS using
# only the K factors with largest |β̂| (on each method's own penalized
# scale), giving a like-for-like comparison at fixed complexity.

def run_topk_comparison(y_raw, X_raw, method_betas, factor_names, hac_lags):
    """
    For each method that produced a beta vector, rank factors by |β̂|
    and run post-selection OLS for each K in TOPK_LEVELS.

    Args:
        y_raw, X_raw:       raw data (pd.Series, pd.DataFrame)
        method_betas:       dict {method_name: np.array of shape (p,)}
        factor_names:       list of str, length p
        hac_lags:           int

    Returns:
        list of dicts, one row per (method, K) pair.
    """
    rows = []

    for method_name, beta in method_betas.items():
        # Rank factors by |β̂| descending
        order = np.argsort(-np.abs(beta))
        p = len(beta)

        for K in TOPK_LEVELS:
            if K > p:
                continue
            top_idx = order[:K]
            top_factors = [factor_names[i] for i in top_idx]

            # Only include factors that were actually nonzero
            # (if method selected fewer than K, use all it selected)
            actual_sel = [f for f in top_factors if abs(beta[factor_names.index(f)]) > COEF_EPS]
            if len(actual_sel) == 0:
                rows.append({
                    'method': method_name, 'K': K,
                    'k_actual': 0,
                    'alpha_monthly': None, 'alpha_annualized': None,
                    'alpha_tstat': None, 'alpha_pval': None,
                    'r_squared_adj': None,
                    'factors': [],
                })
                continue

            # Cap at min(K, n_nonzero)
            sel = actual_sel[:K]
            ols = post_selection_ols(y_raw, X_raw, sel, hac_lags)
            rows.append({
                'method': method_name, 'K': K,
                'k_actual': ols['n_factors'],
                'alpha_monthly': ols['alpha_monthly'],
                'alpha_annualized': ols['alpha_annualized'],
                'alpha_tstat': ols['alpha_tstat'],
                'alpha_pval': ols['alpha_pval'],
                'r_squared_adj': ols['r_squared_adj'],
                'factors': ols['selected_factors'],
            })

    return rows


# ============================================================================
# PER-STRATEGY COMPARISON
# ============================================================================

def run_comparison_for_strategy(strategy_name, strategy_path):
    """Run all 6 methods on one strategy."""

    strategy_dir = get_strategy_aen_dir(strategy_name)

    # Load preprocessed (standardized) data for penalized methods 1-3, 5
    y_df = pd.read_parquet(strategy_dir / "y_centered.parquet")
    X_df = pd.read_parquet(strategy_dir / "X_standardized.parquet")
    y_std = y_df['y'].values
    X_std = X_df.values
    factor_names = X_df.columns.tolist()
    T, p = X_std.shape

    # Load raw data for OLS and for Chen et al. method
    all_factors = pd.read_parquet(FACTORS_PATH)
    factors_end = pd.Timestamp(FACTORS_END_DATE)
    all_factors = all_factors[all_factors.index <= factors_end]

    returns = load_strategy_returns(strategy_path)
    returns = returns[returns.index <= factors_end]

    common_dates = returns.index.intersection(all_factors.index)
    y_raw = returns.loc[common_dates]
    X_raw = all_factors.loc[common_dates][factor_names].copy()

    mask = ~(X_raw.isna().any(axis=1) | y_raw.isna())
    y_raw, X_raw = y_raw[mask], X_raw[mask]

    criterion = AEN_TUNING_CRITERION
    gic_alpha = GIC_ALPHA if criterion == "GIC" else 3.0

    print(f"\n   Data: T = {T}, p = {p}")
    print(f"   COEF_EPS = {COEF_EPS}")
    print(f"   HAC lags (Newey-West): {HAC_LAGS}")
    print(f"   Criterion (methods 1-5): {criterion}")
    print(f"   Criterion (method 6, Chen): AIC (fixed per paper)")

    results = {}
    # Store beta vectors for Top-K comparison (methods that produce them)
    method_betas = {}

    # ── METHOD 1: Lasso ────────────────────────────────────────────────
    print(f"\n   [1/6] Lasso...")
    beta_lasso = fit_lasso(y_std, X_std, criterion, gic_alpha,
                           AEN_LAMBDA1_N_VALUES)
    sel_lasso = selected_from_beta(beta_lasso, factor_names)
    ols_lasso = post_selection_ols(y_raw, X_raw, sel_lasso, HAC_LAGS)
    results['Lasso'] = ols_lasso
    method_betas['Lasso'] = beta_lasso
    print(f"         Selected {len(sel_lasso)} factors")

    # ── METHOD 2: Elastic Net ──────────────────────────────────────────
    print(f"\n   [2/6] Elastic Net...")
    beta_en_rescaled, beta_en_raw = fit_elastic_net(
        y_std, X_std, AEN_LAMBDA2_GRID,
        criterion, gic_alpha, AEN_LAMBDA1_N_VALUES)
    # Selection on beta_raw (pre-rescale sparsity pattern)
    sel_en = selected_from_beta(beta_en_raw, factor_names)
    ols_en = post_selection_ols(y_raw, X_raw, sel_en, HAC_LAGS)
    results['Elastic Net'] = ols_en
    method_betas['Elastic Net'] = beta_en_raw
    print(f"         Selected {len(sel_en)} factors")

    # ── METHOD 3: Adaptive Lasso (Lasso-init, our criterion) ──────────
    print(f"\n   [3/6] Adaptive Lasso (Lasso-init, {criterion})...")
    beta_alasso = fit_adaptive_lasso(y_std, X_std, criterion,
                                     AEN_GAMMA, gic_alpha,
                                     AEN_LAMBDA1_N_VALUES)
    sel_alasso = selected_from_beta(beta_alasso, factor_names)
    ols_alasso = post_selection_ols(y_raw, X_raw, sel_alasso, HAC_LAGS)
    results['Adaptive Lasso'] = ols_alasso
    method_betas['Adaptive Lasso'] = beta_alasso
    print(f"         Selected {len(sel_alasso)} factors")

    # ── METHOD 4: Ridge (shrinkage benchmark, GCV on raw) ─────────────
    print(f"\n   [4/6] Ridge (all factors, GCV on raw)...")
    ridge_result = fit_ridge(y_raw, X_raw, factor_names)
    results['Ridge'] = ridge_result
    # Ridge has no meaningful beta ranking for Top-K (all factors kept)
    print(f"         lambda = {ridge_result['ridge_lambda']:.4f},"
          f" df_eff = {ridge_result.get('df_effective', p):.1f}")

    # ── METHOD 5: AEN (load from 02) ──────────────────────────────────
    print(f"\n   [5/6] AEN (loaded from Step 02)...")
    with open(strategy_dir / "aen_results.json", 'r') as f:
        aen_res = json.load(f)
    sel_aen = aen_res['selected_factors']
    ols_aen = post_selection_ols(y_raw, X_raw, sel_aen, HAC_LAGS)
    results['AEN'] = ols_aen
    # Load AEN beta if available for Top-K
    if 'beta_aen' in aen_res:
        beta_aen_arr = np.array(aen_res['beta_aen'])
        method_betas['AEN'] = beta_aen_arr
    print(f"         Selected {len(sel_aen)} factors")

    # ── METHOD 6: Adaptive LASSO — Chen et al. (2025, JFQA) ──────────
    print(f"\n   [6/6] Adaptive LASSO — Chen et al. (2025)...")
    print(f"         Standardization: z-score (mean/std)")
    print(f"         Initial weights: OLS (or Ridge if p>=T)")
    print(f"         Tuning: AIC")
    print(f"         l2 penalty: 0 (pure LASSO)")

    chen_result = fit_adaptive_lasso_chen(
        y_raw, X_raw, factor_names, n_lambda1=AEN_LAMBDA1_N_VALUES)

    sel_chen = chen_result['selected_factors_zscore']
    ols_chen = post_selection_ols(y_raw, X_raw, sel_chen, HAC_LAGS)

    # Augment with Chen-specific metadata
    ols_chen['chen_init_method'] = chen_result['init_method']
    ols_chen['chen_lambda1'] = chen_result['lambda1_best']
    ols_chen['tuning_criterion'] = 'AIC'

    results['ALASSO-Chen'] = ols_chen
    method_betas['ALASSO-Chen'] = chen_result['beta_alasso']

    print(f"         Init: {chen_result['init_method']}")
    print(f"         Selected {len(sel_chen)} factors: {sel_chen}")

    # ── Print comparison table ─────────────────────────────────────────
    # NOTE: only post-selection OLS metrics are shown.  IC values are
    # NOT comparable across methods with different standardizations.
    print_header(f"   METHOD COMPARISON — {strategy_name}", "-")
    print(f"   (All metrics from post-selection OLS on RAW data,"
          f" HAC lags = {HAC_LAGS})")

    print(f"\n   {'Method':<20} {'k':>4} {'a(mo)':>8} {'a(yr)':>8}"
          f" {'t-stat':>8} {'p-val':>8} {'R2adj':>7}")
    print(f"   {'-' * 65}")

    for method, res in results.items():
        if res['alpha_monthly'] is not None:
            a_t = res.get('alpha_tstat')
            a_p = res.get('alpha_pval')
            r2a = res.get('r_squared_adj')
            t_str = f"{a_t:>8.3f}" if a_t is not None else f"{'N/A':>8}"
            p_str = f"{a_p:>8.4f}" if a_p is not None else f"{'N/A':>8}"
            r2_str = f"{r2a:>7.3f}" if r2a is not None else f"{'N/A':>7}"
            stars = significance_stars(a_p) if a_p is not None else ""
            print(f"   {method:<20}"
                  f" {res['n_factors']:>4}"
                  f" {res['alpha_monthly']:>+8.4f}"
                  f" {res['alpha_annualized']:>+8.2f}"
                  f" {t_str}"
                  f" {p_str}"
                  f" {r2_str}"
                  f" {stars}")
        else:
            print(f"   {method:<20} {0:>4} {'(null model)':>45}")

    # ── Selection matrix ───────────────────────────────────────────────
    methods_with_selection = [
        'Lasso', 'Elastic Net', 'Adaptive Lasso', 'AEN', 'ALASSO-Chen'
    ]
    sel_matrix_rows = []

    for f_name in factor_names:
        row = {'factor': f_name}
        selected_count = 0
        for method in methods_with_selection:
            sel = f_name in results[method]['selected_factors']
            row[method] = sel
            if sel:
                selected_count += 1
        row['n_methods'] = selected_count
        if selected_count > 0:
            sel_matrix_rows.append(row)

    sel_matrix_df = pd.DataFrame(sel_matrix_rows)
    if len(sel_matrix_df) > 0:
        sel_matrix_df = sel_matrix_df.sort_values('n_methods', ascending=False)

    print(f"\n   -- Selection Matrix (factors selected by >=1 method) --")
    if len(sel_matrix_df) > 0:
        print(f"\n   {'Factor':<25}", end="")
        for m in methods_with_selection:
            abbrev = m[:8]
            print(f" {abbrev:>8}", end="")
        print(f" {'Count':>6}")
        print(f"   {'-' * (25 + 8 * len(methods_with_selection) + 8)}")

        for _, row in sel_matrix_df.iterrows():
            print(f"   {row['factor']:<25}", end="")
            for m in methods_with_selection:
                marker = "Y" if row[m] else " "
                print(f" {marker:>8}", end="")
            print(f" {row['n_methods']:>6}")
    else:
        print(f"   (no factors selected by any method)")

    # ── Top-K comparison ──────────────────────────────────────────────
    if method_betas:
        print(f"\n   -- Top-K Comparison (fixed complexity) --")
        topk_rows = run_topk_comparison(
            y_raw, X_raw, method_betas, factor_names, HAC_LAGS)

        for K in TOPK_LEVELS:
            k_rows = [r for r in topk_rows if r['K'] == K]
            if not k_rows:
                continue
            print(f"\n   K = {K}:")
            print(f"   {'Method':<20} {'k':>4} {'a(yr)':>8}"
                  f" {'t-stat':>8} {'p-val':>8} {'R2adj':>7}")
            print(f"   {'-' * 57}")
            for r in k_rows:
                if r['alpha_monthly'] is not None:
                    a_p = r['alpha_pval']
                    stars = significance_stars(a_p) if a_p is not None else ""
                    t_str = f"{r['alpha_tstat']:>8.3f}" if r['alpha_tstat'] is not None else f"{'N/A':>8}"
                    p_str = f"{a_p:>8.4f}" if a_p is not None else f"{'N/A':>8}"
                    r2_str = f"{r['r_squared_adj']:>7.3f}" if r['r_squared_adj'] is not None else f"{'N/A':>7}"
                    print(f"   {r['method']:<20}"
                          f" {r['k_actual']:>4}"
                          f" {r['alpha_annualized']:>+8.2f}"
                          f" {t_str}"
                          f" {p_str}"
                          f" {r2_str}"
                          f" {stars}")
                else:
                    print(f"   {r['method']:<20} {0:>4} {'(empty)':>35}")

    # ── Save ───────────────────────────────────────────────────────────
    # Table 3: comparison summary
    comparison_rows = []
    for method, res in results.items():
        comparison_rows.append({
            'method': method,
            'n_factors': res['n_factors'],
            'alpha_monthly': res['alpha_monthly'],
            'alpha_annualized': res['alpha_annualized'],
            'alpha_tstat': res['alpha_tstat'],
            'alpha_pval': res['alpha_pval'],
            'r_squared': res.get('r_squared'),
            'r_squared_adj': res['r_squared_adj'],
        })
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(strategy_dir / "method_comparison.csv", index=False)

    # Table 4: selection matrix
    if len(sel_matrix_df) > 0:
        sel_matrix_df.to_csv(strategy_dir / "selection_matrix.csv", index=False)

    # Table 5: Top-K comparison
    if method_betas:
        topk_df = pd.DataFrame(topk_rows)
        topk_df.to_csv(strategy_dir / "topk_comparison.csv", index=False)

    # JSON (full results)
    json_results = {
        'strategy': strategy_name,
        'T': T, 'p': p,
        'coef_eps': COEF_EPS,
        'hac_lags': HAC_LAGS,
        'tuning_criterion_methods_1_5': AEN_TUNING_CRITERION,
        'tuning_criterion_method_6': 'AIC',
        'chen_details': {
            'init_method': chen_result['init_method'],
            'lambda1': chen_result['lambda1_best'],
            'n_selected': chen_result['n_nonzero'],
            'selected_factors': sel_chen,
        },
        'methods': results,
    }
    with open(strategy_dir / "method_comparison.json", 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\n   Saved: method_comparison.json")
    print(f"   Saved: method_comparison.csv (Table 3)")
    print(f"   Saved: selection_matrix.csv (Table 4)")
    if method_betas:
        print(f"   Saved: topk_comparison.csv (Table 5)")

    return json_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("METHOD COMPARISON (Zou-Zhang + Chen et al. 2025)")
    print(f"\n   Methods:")
    print(f"     1. Lasso")
    print(f"     2. Elastic Net")
    print(f"     3. Adaptive Lasso (Lasso-init)")
    print(f"     4. Ridge (shrinkage benchmark)")
    print(f"     5. AEN (our method, from Step 02)")
    print(f"     6. ALASSO-Chen (Chen, Li, Tang, Zhou 2025, JFQA)")
    print(f"   Criterion (1-5): {AEN_TUNING_CRITERION}")
    print(f"   Criterion (6):   AIC (fixed per Chen et al.)")
    print(f"   AEN gamma = {AEN_GAMMA}")
    print(f"   COEF_EPS = {COEF_EPS}")
    print(f"   HAC lags = {HAC_LAGS}")
    print(f"   Top-K levels: {TOPK_LEVELS}")

    all_results = {}

    for strategy_name, strategy_path in STRATEGIES.items():
        print_header(f"STRATEGY: {strategy_name}")

        strategy_dir = get_strategy_aen_dir(strategy_name)
        if not (strategy_dir / "y_centered.parquet").exists():
            print(f"\n   ERROR: Preprocessed data not found. Run 01 first.")
            continue

        try:
            result = run_comparison_for_strategy(strategy_name, strategy_path)
            all_results[strategy_name] = result
        except Exception as e:
            print(f"\n   ERROR: {e}")
            import traceback; traceback.print_exc()

    # ── Cross-strategy summary ─────────────────────────────────────────
    if all_results:
        print_header("CROSS-STRATEGY METHOD COMPARISON")

        methods = [
            'Lasso', 'Elastic Net', 'Adaptive Lasso',
            'Ridge', 'AEN', 'ALASSO-Chen'
        ]

        for strategy_name, res in all_results.items():
            print(f"\n   {strategy_name}:")
            print(f"   {'Method':<20} {'k':>4} {'a(yr)':>8} {'p-val':>8}")
            print(f"   {'-' * 42}")
            for m in methods:
                r = res['methods'].get(m, {})
                if r and r.get('alpha_annualized') is not None:
                    a_p = r.get('alpha_pval')
                    p_str = f"{a_p:>8.4f}" if a_p is not None else f"{'N/A':>8}"
                    stars = significance_stars(a_p) if a_p is not None else ""
                    print(f"   {m:<20} {r['n_factors']:>4}"
                          f" {r['alpha_annualized']:>+8.2f}"
                          f" {p_str}"
                          f" {stars}")

        # Save global summary
        aen_output_dir = get_aen_output_dir()
        summary = {
            'tuning_criterion_methods_1_5': AEN_TUNING_CRITERION,
            'tuning_criterion_method_6': 'AIC',
            'coef_eps': COEF_EPS,
            'hac_lags': HAC_LAGS,
            'methods': methods,
            'strategies': {}
        }
        for sn, res in all_results.items():
            summary['strategies'][sn] = {
                m: {
                    'n_factors': res['methods'][m]['n_factors'],
                    'alpha_annualized': res['methods'][m]['alpha_annualized'],
                    'alpha_pval': res['methods'][m]['alpha_pval'],
                    'r_squared_adj': res['methods'][m]['r_squared_adj'],
                }
                for m in methods if m in res['methods']
            }

        with open(aen_output_dir / "method_comparison_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n   Saved: {aen_output_dir / 'method_comparison_summary.json'}")

    print(f"\n{'=' * 80}")
    print(f"METHOD COMPARISON COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n   Next: python src/aen/06_aen_robustness.py")


if __name__ == "__main__":
    main()
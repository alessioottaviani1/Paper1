"""
================================================================================
02_aen_estimation.py - Adaptive Elastic-Net Estimation
================================================================================
Two-stage AEN from Zou & Zhang (2009), with support for:
    - BIC      (paper-faithful baseline)
    - HQC      (Hannan-Quinn 1979, penalty = 2·log(log(T))·df)
    - AIC      (Akaike 1973, penalty = 2·df)
    - AICc     (Hurvich-Tsai 1989, small-sample corrected AIC)
    - GIC      (Generalized IC, custom penalty α·df)
    - SIS_BIC  (Sure Independence Screening + BIC)
    - CV       (Rolling-origin cross-validation for time series)

Set AEN_TUNING_CRITERION in 00_aen_config.py to choose.

Pipeline (per strategy):
    Stage 1 — Elastic Net (eq. 1.4):
        β̂(enet) = (1+λ₂°)·argmin{ (1/2T)||y-Xβ||² + (λ₂/2)||β||² + λ₁||β||₁ }
        Pick (λ₁, λ₂) by information criterion or CV.
        Solver: coordinate descent with uniform weights (same solver as Stage 2).

    Adaptive weights (eq. 2.1):
        wⱼ = (|β̂ⱼ(enet)| + 1/T)^{-γ}
        Computed on RESCALED β̂(enet).

    Stage 2 — Adaptive Elastic Net (eq. 2.2):
        β̂(AEN) = (1+λ₂°)·argmin{ (1/2T)||y-Xβ||² + (λ₂/2)||β||² + λ₁* Σⱼ wⱼ|βⱼ| }
        Same λ₂ as Stage 1 (p. 1737). Pick λ₁* by same criterion.
        Solver: custom coordinate descent with weighted ℓ₁ + UNIFORM ℓ₂.
        (The column rescaling trick distorts ℓ₂ when λ₂ > 0;
         this solver is faithful to eq. 2.2.)

    CV mode:
        Rolling-origin CV (expanding window) preserves temporal structure.
        Initial training window = 60% of T. Step-ahead h = 1.
        Select (λ₁, λ₂) jointly in Stage 1, then λ₁* in Stage 2,
        minimizing mean squared prediction error (MSPE).

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

import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"

spec = importlib.util.spec_from_file_location("aen_config", config_path)
aen_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aen_config)

AEN_GAMMA             = aen_config.AEN_GAMMA
AEN_LAMBDA2_GRID      = aen_config.AEN_LAMBDA2_GRID
AEN_LAMBDA1_N_VALUES  = aen_config.AEN_LAMBDA1_N_VALUES
AEN_TUNING_CRITERION  = aen_config.AEN_TUNING_CRITERION
COEF_TOL              = aen_config.COEF_TOL
GIC_ALPHA             = aen_config.GIC_ALPHA
CV_INITIAL_FRAC       = getattr(aen_config, 'CV_INITIAL_FRAC', 0.60)
CV_STEP_AHEAD         = getattr(aen_config, 'CV_STEP_AHEAD', 1)
STRATEGIES            = aen_config.STRATEGIES
get_strategy_aen_dir  = aen_config.get_strategy_aen_dir
get_aen_output_dir    = aen_config.get_aen_output_dir


def print_header(title, char="="):
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")



# ============================================================================
# INFORMATION CRITERIA
# ============================================================================
#
# All criteria share the form: IC = T·log(RSS/T) + penalty(df)
#
# BIC:   penalty = df · log(T)                     (Schwarz 1978)
# HQC:   penalty = df · 2·log(log(T))              (Hannan-Quinn 1979)
# AIC:   penalty = 2·df                            (Akaike 1973)
# AICc:  penalty = 2·df + 2·df·(df+1)/(T-df-1)    (Hurvich-Tsai 1989)
# GIC:   penalty = df · α                          (custom penalty)
#
# Penalità per fattore (T = 158):
#   BIC  = 5.06,  HQC = 3.24,  AIC = 2.00,  AICc ≈ 2.2,  GIC(α=3) = 3.00

def compute_ic(y, X, beta, criterion, gic_alpha=3.0):
    """
    Compute information criterion.

    Args:
        y:          response (T,)
        X:          predictors (T × p)
        beta:       coefficients (p,)
        criterion:  "BIC", "HQC", "AIC", "AICc", "GIC", or "SIS_BIC"
        gic_alpha:  penalty per df for GIC (default 3.0)

    Returns:
        (ic_value, rss, df)
    """
    T = len(y)
    residuals = y - X @ beta
    rss = float(np.sum(residuals ** 2))
    df = int(np.sum(np.abs(beta) > COEF_TOL))

    if rss / T < 1e-15:
        return (np.inf, rss, df)

    log_likelihood = T * np.log(rss / T)

    if criterion in ("BIC", "SIS_BIC"):
        ic = log_likelihood + df * np.log(T)
    elif criterion == "HQC":
        ic = log_likelihood + 2 * np.log(np.log(T)) * df
    elif criterion == "AIC":
        ic = log_likelihood + 2 * df
    elif criterion == "AICc":
        aic = log_likelihood + 2 * df
        if T - df - 1 > 0:
            ic = aic + (2 * df * (df + 1)) / (T - df - 1)
        else:
            ic = np.inf  # model too complex for sample size
    elif criterion == "GIC":
        ic = log_likelihood + gic_alpha * df
    elif criterion == "CV":
        # CV doesn't use IC for selection; return BIC as reference
        ic = log_likelihood + df * np.log(T)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    return (float(ic), rss, df)


# ============================================================================
# λ₁ GRID
# ============================================================================

def compute_lambda1_max(X, y):
    """λ₁_max: smallest λ₁ for which β̂ = 0 (KKT of CD objective)."""
    T = X.shape[0]
    return np.max(np.abs(X.T @ y)) / T


def build_lambda1_grid(lambda1_max, n_values=100, ratio_min=1e-4):
    return np.logspace(np.log10(lambda1_max),
                       np.log10(lambda1_max * ratio_min), n_values)


# ============================================================================
# ROLLING-ORIGIN CROSS-VALIDATION (time-series faithful)
# ============================================================================
#
# Expanding window: train on t = 1, ..., τ and predict t = τ + h.
# Initial training window: τ₀ = ⌊CV_INITIAL_FRAC × T⌋.
# Step-ahead: h = 1 (one-step forecast).
#
# LEAKAGE PREVENTION:
#   Every quantity that depends on y is computed INSIDE each fold using
#   only the training window [1, τ]:
#     - λ₁_max (depends on Xᵀy)
#     - λ₁*_max (depends on Xᵀy / weights)
#     - adaptive weights wⱼ (depend on β̂(enet) from Stage 1)
#   The λ₁ grid is rebuilt per-fold from λ₁_max(τ).
#   The λ₂ grid is fixed (depends only on the user, not on data).
#
# For Stage 1: select (λ₁, λ₂) minimizing MSPE across all folds.
# For Stage 2: for each candidate (λ₂_fixed, λ₁*), re-run the full
#   Stage 1 → weights → Stage 2 pipeline inside each fold.
#   This is computationally expensive but avoids leakage.
#
# Reference: Hyndman & Athanasopoulos (2021), "Forecasting: Principles
#            and Practice", Ch. 5.8 (time series cross-validation).

def run_stage1_cv(y, X, lambda2_grid, n_lambda1, initial_frac=0.60,
                  step_ahead=1, max_iter=10000, tol=1e-7):
    """
    Stage 1: select (λ₁, λ₂) by rolling-origin CV.

    λ₁_max is recomputed inside each fold to avoid leakage.
    The λ₁ grid is defined as a ratio grid [1, ratio_min] × λ₁_max(τ),
    so the RELATIVE positions are the same across folds even though
    the absolute values shift with λ₁_max(τ).

    Returns same dict as run_stage1 (IC fields contain MSPE instead).
    """
    T, p = X.shape
    y_np = np.asarray(y, dtype=np.float64)
    X_np = np.asarray(X, dtype=np.float64)

    tau_0 = max(int(initial_frac * T), p + 2)  # need at least p+1 obs
    n_folds = T - tau_0 - step_ahead + 1
    if n_folds < 5:
        print(f"\n   ⚠️  CV: only {n_folds} folds (T={T}, τ₀={tau_0})."
              f" Consider using an IC criterion instead.")

    # ── Accumulate MSPE per (λ₁_ratio_idx, λ₂) ───────────────────────
    # Since λ₁_max varies per fold, we index λ₁ by its POSITION in the
    # ratio grid (0 = λ₁_max, ..., n_lambda1-1 = λ₁_max × ratio_min).
    ratio_min = 1e-4
    mspe_accum = np.zeros((len(lambda2_grid), n_lambda1))
    fold_count = 0

    for tau in range(tau_0, T - step_ahead + 1):
        y_train = y_np[:tau]
        X_train = X_np[:tau, :]
        y_test = y_np[tau + step_ahead - 1]
        X_test = X_np[tau + step_ahead - 1, :]

        T_train = len(y_train)

        # Per-fold λ₁_max (no leakage: uses only training data)
        l1_max_fold = compute_lambda1_max(X_train, y_train)
        l1_max_fold = max(l1_max_fold, 1e-6)
        l1_grid_fold = build_lambda1_grid(l1_max_fold, n_lambda1,
                                           ratio_min)

        for i_l2, lambda2_po in enumerate(lambda2_grid):
            rescale_train = 1.0 + lambda2_po    # eq. 1.4: (1+λ₂°)

            w_uniform = np.ones(p, dtype=np.float64)
            for i_l1, lambda1_po in enumerate(l1_grid_fold):
                beta_raw = weighted_elastic_net_cd(
                    X_train, y_train, lambda1_po, lambda2_po, w_uniform,
                    max_iter=max_iter, tol=tol)
                beta_enet = beta_raw * rescale_train

                pred = X_test @ beta_enet
                mspe_accum[i_l2, i_l1] += (y_test - pred) ** 2

        fold_count += 1

    mspe_accum /= fold_count

    # ── Find best (λ₂, λ₁_ratio_idx) ─────────────────────────────────
    best_idx = np.unravel_index(np.argmin(mspe_accum), mspe_accum.shape)
    best_i_l2, best_i_l1 = best_idx
    best_mspe = float(mspe_accum[best_i_l2, best_i_l1])
    best_l2 = lambda2_grid[best_i_l2]

    # Reconstruct λ₁ on full-sample scale for refit
    l1_max_full = compute_lambda1_max(X_np, y_np)
    l1_max_full = max(l1_max_full, 1e-6)
    l1_grid_full = build_lambda1_grid(l1_max_full, n_lambda1, ratio_min)
    best_l1 = l1_grid_full[best_i_l1]

    # Build results grid for diagnostics
    results = []
    for i_l2, lambda2_po in enumerate(lambda2_grid):
        for i_l1 in range(n_lambda1):
            results.append({
                'lambda1_po': l1_grid_full[i_l1],
                'lambda2_po': lambda2_po,
                'ic_rescaled': float(mspe_accum[i_l2, i_l1]),
                'rss_rescaled': np.nan,
                'ic_raw': np.nan, 'rss_raw': np.nan, 'df': 0
            })
    grid_df = pd.DataFrame(results)

    # ── Refit on full sample with best (λ₁, λ₂) ─────────────────────
    w_uniform = np.ones(p, dtype=np.float64)
    beta_raw = weighted_elastic_net_cd(
        X_np, y_np, best_l1, best_l2, w_uniform,
        max_iter=max_iter, tol=tol)

    lambda2_sum = best_l2 * T
    rescale = 1.0 + best_l2             # eq. 1.4: (1+λ₂°)
    beta_enet = beta_raw * rescale
    df = int(np.sum(np.abs(beta_raw) > COEF_TOL))

    # IC on full sample (for comparison)
    pred_resc = X_np @ beta_enet
    rss_resc = float(np.sum((y_np - pred_resc) ** 2))
    ic_raw_val, rss_raw_val, _ = compute_ic(
        y_np, X_np, beta_raw, "BIC", 3.0)

    # Grid boundary check
    l2_min, l2_max = min(lambda2_grid), max(lambda2_grid)
    if best_l2 == l2_min and best_l2 > 0:
        print(f"\n   ⚠️  CV: Best λ₂° = {l2_min} at LOWER boundary.")
    if best_l2 == l2_max:
        print(f"\n   ⚠️  CV: Best λ₂° = {l2_max} at UPPER boundary.")

    print(f"\n   Stage 1 results (Rolling-origin CV, {n_folds} folds):")
    print(f"      Best λ₂° = {best_l2:.6f}"
          f"  (sum scale: {lambda2_sum:.4f},"
          f" rescale: {rescale:.4f})")
    print(f"      Best λ₁° = {best_l1:.6f}"
          f"  (grid position {best_i_l1}/{n_lambda1})")
    print(f"      MSPE = {best_mspe:.6f}")
    print(f"      df (full sample) = {df}")
    print(f"      Grid: {len(lambda2_grid)} × {n_lambda1}"
          f" = {len(lambda2_grid) * n_lambda1} pairs"
          f" × {n_folds} folds")

    return {
        'beta_raw': beta_raw,
        'beta_enet': beta_enet,
        'rescale': rescale,
        'lambda1_po_best': best_l1,
        'lambda2_po_best': best_l2,
        'lambda2_sum_best': lambda2_sum,
        'ic_best': best_mspe,       # MSPE stored in ic_best field
        'ic_raw_best': ic_raw_val,   # BIC on full sample (reference)
        'rss_best': rss_resc,
        'df_best': df,
        'grid_results': grid_df,
        'cv_n_folds': n_folds,
    }


def run_stage2_cv(y, X, weights_full, lambda2_po_fixed, T,
                  n_lambda1, gamma=1, initial_frac=0.60, step_ahead=1,
                  max_iter=10000, tol=1e-7):
    """
    Stage 2: select λ₁* by rolling-origin CV (same λ₂ as Stage 1).

    LEAKAGE PREVENTION: In each fold, the full Stage 1 → weights →
    Stage 2 pipeline is re-run on the training window [1, τ]:
      1. Fit Stage 1 elastic net on [1, τ] with (best_λ₁, λ₂_fixed)
      2. Compute adaptive weights from β̂(enet) on [1, τ]
      3. Fit Stage 2 weighted CD with (λ₁*_candidate, λ₂_fixed, w(τ))
      4. Predict y_{τ+h}

    `weights_full` is only used for the full-sample refit at the end.

    Returns same dict as run_stage2 (IC fields contain MSPE instead).
    """
    T_obs, p = X.shape
    y_np = np.asarray(y, dtype=np.float64)
    X_np = np.asarray(X, dtype=np.float64)

    lambda2_sum = lambda2_po_fixed * T
    rescale_full = 1.0 + lambda2_po_fixed   # eq. 2.2

    tau_0 = max(int(initial_frac * T), p + 2)
    n_folds = T - tau_0 - step_ahead + 1

    # λ₁* grid: use ratio positions (same logic as Stage 1 CV)
    ratio_min = 1e-4
    mspe_accum = np.zeros(n_lambda1)
    fold_count = 0

    for tau in range(tau_0, T - step_ahead + 1):
        y_train = y_np[:tau]
        X_train = X_np[:tau, :]
        y_test = y_np[tau + step_ahead - 1]
        X_test = X_np[tau + step_ahead - 1, :]

        T_train = len(y_train)
        rescale_train = 1.0 + lambda2_po_fixed   # eq. 2.2: (1+λ₂°)

        # ── Stage 1 on training window (for fold-specific weights) ────
        # Use the SAME λ₂_fixed. For λ₁, use fold-specific λ₁_max
        # and pick λ₁ by IC (BIC as neutral choice) on training data.
        l1_max_fold = compute_lambda1_max(X_train, y_train)
        l1_max_fold = max(l1_max_fold, 1e-6)
        # Coarse grid for Stage 1 inside CV (speed: n_lambda1 // 3)
        n_l1_inner = max(n_lambda1 // 3, 20)
        l1_grid_inner = build_lambda1_grid(l1_max_fold, n_l1_inner,
                                            ratio_min)

        best_bic_fold = np.inf
        best_beta_enet_fold = None

        w_uniform = np.ones(p, dtype=np.float64)
        for lambda1_po in l1_grid_inner:
            beta_raw_fold = weighted_elastic_net_cd(
                X_train, y_train, lambda1_po, lambda2_po_fixed,
                w_uniform, max_iter=max_iter, tol=tol)
            beta_enet_fold = beta_raw_fold * rescale_train

            pred_fold = X_train @ beta_enet_fold
            rss_fold = float(np.sum((y_train - pred_fold) ** 2))
            df_fold = int(np.sum(np.abs(beta_raw_fold) > COEF_TOL))
            if rss_fold / T_train < 1e-15:
                bic_fold = np.inf
            else:
                bic_fold = (T_train * np.log(rss_fold / T_train)
                            + df_fold * np.log(T_train))
            if bic_fold < best_bic_fold:
                best_bic_fold = bic_fold
                best_beta_enet_fold = beta_enet_fold.copy()

        # ── Fold-specific adaptive weights ────────────────────────────
        weights_fold = compute_adaptive_weights(
            best_beta_enet_fold, T_train, gamma)

        # ── λ₁* grid for Stage 2 (fold-specific) ─────────────────────
        l1s_max_fold = np.max(
            np.abs(X_train.T @ y_train) / T_train / weights_fold)
        if l1s_max_fold < 1e-15:
            l1s_max_fold = 1e-6
        l1s_grid_fold = build_lambda1_grid(l1s_max_fold, n_lambda1,
                                            ratio_min)

        # ── Evaluate each λ₁* candidate ──────────────────────────────
        for i_l1, lambda1_star_po in enumerate(l1s_grid_fold):
            beta_naive = weighted_elastic_net_cd(
                X_train, y_train, lambda1_star_po, lambda2_po_fixed,
                weights_fold, max_iter=max_iter, tol=tol
            )
            beta_aen = beta_naive * rescale_train
            pred = X_test @ beta_aen
            mspe_accum[i_l1] += (y_test - pred) ** 2

        fold_count += 1

    mspe_accum /= fold_count

    # ── Find best λ₁* (by ratio position) ────────────────────────────
    best_i_l1 = int(np.argmin(mspe_accum))
    best_mspe = float(mspe_accum[best_i_l1])

    # Reconstruct λ₁* on full-sample scale for refit
    l1s_max_full = np.max(
        np.abs(X_np.T @ y_np) / T / weights_full)
    if l1s_max_full < 1e-15:
        l1s_max_full = 1e-6
    l1s_grid_full = build_lambda1_grid(l1s_max_full, n_lambda1,
                                        ratio_min)
    best_l1_star = l1s_grid_full[best_i_l1]

    # ── Refit on full sample ──────────────────────────────────────────
    beta_naive = weighted_elastic_net_cd(
        X_np, y_np, best_l1_star, lambda2_po_fixed,
        weights_full, max_iter=max_iter, tol=tol)
    beta_aen = beta_naive * rescale_full
    df = int(np.sum(np.abs(beta_naive) > COEF_TOL))

    pred_resc = X_np @ beta_aen
    rss_resc = float(np.sum((y_np - pred_resc) ** 2))

    ic_raw_val, _, _ = compute_ic(y_np, X_np, beta_naive, "BIC", 3.0)

    selected_mask = np.abs(beta_naive) > COEF_TOL
    selected_idx = np.where(selected_mask)[0]

    # Results grid for diagnostics
    results = []
    for i_l1 in range(n_lambda1):
        results.append({
            'lambda1_star_po': l1s_grid_full[i_l1],
            'ic_rescaled': float(mspe_accum[i_l1]),
            'rss_rescaled': np.nan,
            'ic_raw': np.nan, 'rss_raw': np.nan, 'df': 0,
        })
    grid_df = pd.DataFrame(results)

    print(f"\n   Stage 2 results (Rolling-origin CV, {n_folds} folds,"
          f" weights re-estimated per fold):")
    print(f"      λ₂° (fixed) = {lambda2_po_fixed:.6f},"
          f" rescale = {rescale_full:.4f}")
    print(f"      Best λ₁*° = {best_l1_star:.6f}"
          f"  (grid position {best_i_l1}/{n_lambda1})")
    print(f"      MSPE = {best_mspe:.6f}")
    print(f"      df (full sample) = {df}")

    return {
        'beta_aen': beta_aen,
        'beta_raw': beta_naive,
        'rescale': rescale_full,
        'lambda1_star_po': best_l1_star,
        'ic_best': best_mspe,
        'ic_raw_best': ic_raw_val,
        'rss_best': rss_resc,
        'df_best': df,
        'selected_idx': selected_idx.tolist(),
        'grid_results': grid_df,
        'raw_ic_df': df,
        'cv_n_folds': n_folds,
    }


# ============================================================================
# STAGE 1: ELASTIC NET
# ============================================================================

def run_stage1(y, X, lambda2_grid, n_lambda1, criterion, gic_alpha=3.0,
               max_iter=10000, tol=1e-7):
    T, p = X.shape
    y_np = np.asarray(y, dtype=np.float64)
    X_np = np.asarray(X, dtype=np.float64)
    lambda1_max = compute_lambda1_max(X_np, y_np)

    results = []
    best_ic = np.inf
    best_result = None

    for lambda2_po in lambda2_grid:
        lambda1_grid = build_lambda1_grid(lambda1_max, n_lambda1)
        lambda2_sum = lambda2_po * T       # sum-scale λ₂ (for logging)
        rescale = 1.0 + lambda2_po         # Zou & Zhang (2009) eq. 1.4: (1+λ₂/n)

        w_uniform = np.ones(p, dtype=np.float64)
        for lambda1_po in lambda1_grid:
            beta_raw = weighted_elastic_net_cd(
                X_np, y_np, lambda1_po, lambda2_po, w_uniform,
                max_iter=max_iter, tol=tol)

            # Zou & Zhang (2009) eq. 1.4: β̂(enet) = (1+λ₂°)·β̂_naive
            beta_enet = beta_raw * rescale

            # df on beta_raw (pre-rescale) to avoid inflating df
            df = int(np.sum(np.abs(beta_raw) > COEF_TOL))

            # IC on rescaled predictions (paper-consistent)
            pred_resc = X_np @ beta_enet
            rss_resc = float(np.sum((y_np - pred_resc) ** 2))
            if rss_resc / T < 1e-15:
                ic_resc = np.inf
            else:
                log_lik_resc = T * np.log(rss_resc / T)
                if criterion in ("BIC", "SIS_BIC"):
                    ic_resc = log_lik_resc + df * np.log(T)
                elif criterion == "HQC":
                    ic_resc = log_lik_resc + 2 * np.log(np.log(T)) * df
                elif criterion == "AIC":
                    ic_resc = log_lik_resc + 2 * df
                elif criterion == "AICc":
                    aic = log_lik_resc + 2 * df
                    ic_resc = aic + (2 * df * (df + 1)) / (T - df - 1) if T - df - 1 > 0 else np.inf
                elif criterion == "GIC":
                    ic_resc = log_lik_resc + gic_alpha * df

            # IC on raw (robustness)
            ic_raw, rss_raw, _ = compute_ic(
                y_np, X_np, beta_raw, criterion, gic_alpha)

            results.append({
                'lambda1_po': lambda1_po, 'lambda2_po': lambda2_po,
                'ic_rescaled': ic_resc, 'rss_rescaled': rss_resc,
                'ic_raw': ic_raw, 'rss_raw': rss_raw, 'df': df
            })

            if ic_resc < best_ic:
                best_ic = ic_resc
                best_result = {
                    'beta_raw': beta_raw.copy(),
                    'beta_enet': beta_enet.copy(),
                    'lambda1_po': lambda1_po, 'lambda2_po': lambda2_po,
                    'lambda2_sum': lambda2_sum, 'rescale': rescale,
                    'ic_resc': ic_resc, 'rss_resc': rss_resc,
                    'ic_raw': ic_raw, 'rss_raw': rss_raw, 'df': df
                }

    grid_df = pd.DataFrame(results)

    # Robustness: raw-IC best
    raw_best_idx = grid_df['ic_raw'].idxmin()
    raw_best = grid_df.loc[raw_best_idx]

    # Grid boundary check
    l2_min, l2_max = min(lambda2_grid), max(lambda2_grid)
    if best_result['lambda2_po'] == l2_min:
        print(f"\n   ⚠️  Best λ₂° = {l2_min} at LOWER boundary. Consider extending.")
    if best_result['lambda2_po'] == l2_max:
        print(f"\n   ⚠️  Best λ₂° = {l2_max} at UPPER boundary. Consider extending.")

    crit_label = criterion if criterion != "SIS_BIC" else "BIC (post-SIS)"
    print(f"\n   Stage 1 results ({crit_label} on rescaled β̂(enet)):")
    print(f"      Best λ₂° = {best_result['lambda2_po']:.6f}"
          f"  (sum scale: {best_result['lambda2_sum']:.4f},"
          f" rescale: {best_result['rescale']:.4f})")
    print(f"      Best λ₁° = {best_result['lambda1_po']:.6f}")
    print(f"      {crit_label} (rescaled) = {best_result['ic_resc']:.4f}"
          f"  |  {crit_label} (raw) = {best_result['ic_raw']:.4f}")
    print(f"      df = {best_result['df']}")
    print(f"      Grid: {len(results)} pairs")

    # Robustness
    resc_best_idx = grid_df['ic_rescaled'].idxmin()
    if raw_best_idx != resc_best_idx:
        print(f"\n   📊 ROBUSTNESS: Raw-{crit_label} selects different model:"
              f" df={int(raw_best['df'])}")
    else:
        print(f"\n   ✅ ROBUSTNESS: Raw and rescaled {crit_label} agree.")

    return {
        'beta_raw': best_result['beta_raw'],
        'beta_enet': best_result['beta_enet'],
        'rescale': best_result['rescale'],
        'lambda1_po_best': best_result['lambda1_po'],
        'lambda2_po_best': best_result['lambda2_po'],
        'lambda2_sum_best': best_result['lambda2_sum'],
        'ic_best': best_result['ic_resc'],
        'ic_raw_best': best_result['ic_raw'],
        'rss_best': best_result['rss_resc'],
        'df_best': best_result['df'],
        'grid_results': grid_df
    }


# ============================================================================
# ADAPTIVE WEIGHTS (eq. 2.1)
# ============================================================================

def compute_adaptive_weights(beta_enet, T, gamma=1):
    """
    wⱼ = (|β̂ⱼ(enet)| + 1/T)^{-γ}
    On RESCALED β̂(enet) = (1+λ₂)·β_naive (paper definition).
    """
    epsilon = 1.0 / T
    return (np.abs(beta_enet) + epsilon) ** (-gamma)


#============================================================================
#COORDINATE DESCENT: WEIGHTED ℓ₁ + UNIFORM ℓ₂  (paper eq. 2.2)
# ============================================================================
#
# Solves:  min_β  (1/2T)||y - Xβ||² + (λ₂/2)||β||² + λ₁* Σⱼ wⱼ|βⱼ|
#
# The column rescaling trick (X̃ⱼ = Xⱼ/wⱼ) is exact only when λ₂ = 0.
# When λ₂ > 0, it distorts the ridge penalty into λ₂ Σ wⱼ²βⱼ²
# (feature-specific), which is NOT what the paper prescribes.
#
# This custom solver keeps ℓ₁ weighted and ℓ₂ uniform, faithful to eq. 2.2.
#
# Update rule for coordinate j:
#   βⱼ ← S( (1/T)Xⱼᵀr₋ⱼ , λ₁*wⱼ ) / ( (1/T)||Xⱼ||² + λ₂ )
#
# where S(z, γ) = sign(z)(|z| - γ)₊ is the soft-thresholding operator.

def weighted_elastic_net_cd(X, y, lambda1, lambda2, weights,
                            max_iter=10000, tol=1e-7):
    """
    Coordinate descent for weighted ℓ₁ + uniform ℓ₂ elastic net.

    Minimizes: (1/2T)||y - Xβ||² + (λ₂/2)||β||² + λ₁ Σⱼ wⱼ|βⱼ|

    Args:
        X:        (T, p) predictor matrix
        y:        (T,) response vector
        lambda1:  ℓ₁ penalty (per-observation scale, λ₁*° in paper)
        lambda2:  ℓ₂ penalty (per-observation scale, λ₂° in paper)
        weights:  (p,) adaptive weights wⱼ
        max_iter: max iterations
        tol:      convergence tolerance on max |Δβⱼ|

    Returns:
        beta: (p,) coefficient vector (naive, NOT rescaled by (1+λ₂))
    """
    T, p = X.shape
    beta = np.zeros(p, dtype=np.float64)
    r = y.copy().astype(np.float64)  # residual = y - Xβ

    # Precompute (1/T)||Xⱼ||²
    col_norm_sq = np.sum(X ** 2, axis=0) / T  # (p,)

    for iteration in range(max_iter):
        max_change = 0.0

        for j in range(p):
            beta_old = beta[j]

            # Add back contribution of j to residual
            if beta_old != 0.0:
                r += X[:, j] * beta_old

            # Compute (1/T) Xⱼᵀr
            rho_j = np.dot(X[:, j], r) / T

            # Soft-thresholding with weighted ℓ₁
            threshold = lambda1 * weights[j]
            if rho_j > threshold:
                numerator = rho_j - threshold
            elif rho_j < -threshold:
                numerator = rho_j + threshold
            else:
                numerator = 0.0

            # Denominator: (1/T)||Xⱼ||² + λ₂ (uniform!)
            denominator = col_norm_sq[j] + lambda2

            beta[j] = numerator / denominator

            # Update residual
            if beta[j] != 0.0:
                r -= X[:, j] * beta[j]

            change = abs(beta[j] - beta_old)
            if change > max_change:
                max_change = change

        if max_change < tol:
            break

    return beta


# ============================================================================
# STAGE 2: ADAPTIVE ELASTIC NET (eq. 2.2) — paper-faithful solver
# ============================================================================

def run_stage2(y, X, weights, lambda2_po_fixed, T,
               n_lambda1, criterion, gic_alpha=3.0,
               max_iter=10000, tol=1e-7):
    T_obs, p = X.shape
    y_np = np.asarray(y, dtype=np.float64)
    X_np = np.asarray(X, dtype=np.float64)

    lambda2_sum = lambda2_po_fixed * T     # sum-scale (for logging)
    rescale = 1.0 + lambda2_po_fixed       # eq. 2.2: (1+λ₂/n)

    # λ₁* max: largest λ₁* that keeps all β = 0
    # For weighted ℓ₁: λ₁*_max = max_j { (1/T)|Xⱼᵀy| / wⱼ }
    lambda1_star_max = np.max(np.abs(X_np.T @ y_np) / T / weights)
    if lambda1_star_max < 1e-15:
        lambda1_star_max = 1e-6
    lambda1_star_grid = build_lambda1_grid(lambda1_star_max, n_lambda1)

    results = []

    for lambda1_star_po in lambda1_star_grid:
        # Paper-faithful: weighted ℓ₁ + uniform ℓ₂
        beta_naive = weighted_elastic_net_cd(
            X_np, y_np, lambda1_star_po, lambda2_po_fixed,
            weights, max_iter=max_iter, tol=tol
        )

        # df on beta_naive (pre-rescale)
        df = int(np.sum(np.abs(beta_naive) > COEF_TOL))

        # Paper rescaling: β̂(AEN) = (1 + λ₂) · β_naive
        beta_aen = beta_naive * rescale

        # IC on rescaled predictions, but df from beta_naive
        pred_resc = X_np @ beta_aen
        rss_resc = float(np.sum((y_np - pred_resc) ** 2))
        if rss_resc / T < 1e-15:
            ic_resc = np.inf
        else:
            log_lik_resc = T * np.log(rss_resc / T)
            if criterion in ("BIC", "SIS_BIC"):
                ic_resc = log_lik_resc + df * np.log(T)
            elif criterion == "HQC":
                ic_resc = log_lik_resc + 2 * np.log(np.log(T)) * df
            elif criterion == "AIC":
                ic_resc = log_lik_resc + 2 * df
            elif criterion == "AICc":
                aic = log_lik_resc + 2 * df
                ic_resc = aic + (2 * df * (df + 1)) / (T - df - 1) if T - df - 1 > 0 else np.inf
            elif criterion == "GIC":
                ic_resc = log_lik_resc + gic_alpha * df

        ic_raw, rss_raw, _ = compute_ic(
            y_np, X_np, beta_naive, criterion, gic_alpha)

        results.append({
            'lambda1_star_po': lambda1_star_po,
            'ic_rescaled': ic_resc, 'rss_rescaled': rss_resc,
            'ic_raw': ic_raw, 'rss_raw': rss_raw, 'df': df,
            'beta_original': beta_naive.copy(),
            'beta_aen': beta_aen.copy()
        })

    grid_df = pd.DataFrame([{k: v for k, v in r.items()
                             if k not in ('beta_original', 'beta_aen')}
                            for r in results])

    best_idx = grid_df['ic_rescaled'].idxmin()
    beta_aen = results[best_idx]['beta_aen']
    beta_raw = results[best_idx]['beta_original']

    raw_best_idx = grid_df['ic_raw'].idxmin()

    # Selection on beta_naive (pre-rescale)
    selected_mask = np.abs(beta_raw) > COEF_TOL
    selected_idx = np.where(selected_mask)[0]
    chosen = grid_df.loc[best_idx]

    crit_label = criterion if criterion != "SIS_BIC" else "BIC (post-SIS)"
    print(f"\n   Stage 2 results ({crit_label} on rescaled β̂(AEN)):")
    print(f"      λ₂° (fixed) = {lambda2_po_fixed:.6f},"
          f" rescale = {rescale:.4f}")
    print(f"      Best λ₁*° = {chosen['lambda1_star_po']:.6f}")
    print(f"      {crit_label} (rescaled) = {chosen['ic_rescaled']:.4f}"
          f"  |  {crit_label} (raw) = {chosen['ic_raw']:.4f}")
    print(f"      df = {int(chosen['df'])}")

    if raw_best_idx != best_idx:
        raw_row = grid_df.loc[raw_best_idx]
        print(f"\n   📊 ROBUSTNESS: Raw-{crit_label} selects df={int(raw_row['df'])}")
    else:
        print(f"\n   ✅ ROBUSTNESS: Raw and rescaled agree.")

    return {
        'beta_aen': beta_aen,
        'beta_raw': beta_raw,
        'rescale': rescale,
        'lambda1_star_po': float(chosen['lambda1_star_po']),
        'ic_best': float(chosen['ic_rescaled']),
        'ic_raw_best': float(grid_df.loc[raw_best_idx, 'ic_raw']),
        'rss_best': float(chosen['rss_rescaled']),
        'df_best': int(chosen['df']),
        'selected_idx': selected_idx.tolist(),
        'grid_results': grid_df,
        'raw_ic_df': int(grid_df.loc[raw_best_idx, 'df'])
    }


# ============================================================================
# PER-STRATEGY RUNNER
# ============================================================================

def run_aen_for_strategy(strategy_name):
    strategy_dir = get_strategy_aen_dir(strategy_name)

    y_df = pd.read_parquet(strategy_dir / "y_centered.parquet")
    X_df = pd.read_parquet(strategy_dir / "X_standardized.parquet")
    with open(strategy_dir / "standardization_params.json", 'r') as f:
        std_params = json.load(f)

    y = y_df['y'].values
    X = X_df.values
    factor_names = X_df.columns.tolist()
    T, p = X.shape

    # Determine criterion
    criterion = AEN_TUNING_CRITERION
    gic_alpha = GIC_ALPHA if criterion == "GIC" else 3.0

    crit_label = criterion if criterion != "SIS_BIC" else "SIS + BIC"
    if criterion == "CV":
        crit_label = "Rolling-origin CV"
    print(f"\n   Data: T = {T}, p = {p}")
    print(f"   Criterion: {crit_label}")
    if criterion == "HQC":
        print(f"   HQC penalty per factor: 2·log(log({T}))"
              f" = {2*np.log(np.log(T)):.4f}")
    elif criterion == "AIC":
        print(f"   AIC penalty per factor: 2")
    elif criterion == "GIC":
        print(f"   GIC α = {gic_alpha}")
    elif criterion == "CV":
        print(f"   CV initial window: {CV_INITIAL_FRAC:.0%} of T"
              f" = {int(CV_INITIAL_FRAC * T)} obs")
        print(f"   CV step-ahead: h = {CV_STEP_AHEAD}")
    elif criterion == "BIC" or criterion == "SIS_BIC":
        print(f"   BIC penalty per factor: log({T}) = {np.log(T):.4f}")
    print(f"   AEN γ = {AEN_GAMMA}")

    # ── Stage 1 ────────────────────────────────────────────────────────
    print_header(f"   STAGE 1: Elastic Net — {strategy_name}", "─")

    if criterion == "CV":
        stage1 = run_stage1_cv(
            y=y, X=X, lambda2_grid=AEN_LAMBDA2_GRID,
            n_lambda1=AEN_LAMBDA1_N_VALUES,
            initial_frac=CV_INITIAL_FRAC,
            step_ahead=CV_STEP_AHEAD)
    else:
        stage1 = run_stage1(
            y=y, X=X, lambda2_grid=AEN_LAMBDA2_GRID,
            n_lambda1=AEN_LAMBDA1_N_VALUES,
            criterion=criterion, gic_alpha=gic_alpha)

    beta_enet = stage1['beta_enet']
    nonzero_mask = np.abs(beta_enet) > COEF_TOL
    nonzero_names = [f for f, nz in zip(factor_names, nonzero_mask) if nz]
    print(f"\n   Stage 1 nonzero factors ({len(nonzero_names)}):")
    for name in nonzero_names:
        i = factor_names.index(name)
        print(f"      {name:30s}  β̂(enet) = {beta_enet[i]:+.6f}")

    # ── Adaptive weights ───────────────────────────────────────────────
    print_header(f"   ADAPTIVE WEIGHTS — {strategy_name}", "─")

    weights = compute_adaptive_weights(beta_enet, T, AEN_GAMMA)
    print(f"\n   Weights on rescaled β̂(enet) (rescale = {stage1['rescale']:.4f})")
    print(f"   min: {weights.min():.4f}, median: {np.median(weights):.4f},"
          f" max: {weights.max():.4f}")

    # ── Stage 2 ────────────────────────────────────────────────────────
    print_header(f"   STAGE 2: Adaptive Elastic Net — {strategy_name}", "─")

    if criterion == "CV":
        stage2 = run_stage2_cv(
            y=y, X=X, weights_full=weights,
            lambda2_po_fixed=stage1['lambda2_po_best'], T=T,
            n_lambda1=AEN_LAMBDA1_N_VALUES,
            gamma=AEN_GAMMA,
            initial_frac=CV_INITIAL_FRAC,
            step_ahead=CV_STEP_AHEAD)
    else:
        stage2 = run_stage2(
            y=y, X=X, weights=weights,
            lambda2_po_fixed=stage1['lambda2_po_best'], T=T,
            n_lambda1=AEN_LAMBDA1_N_VALUES,
            criterion=criterion, gic_alpha=gic_alpha)

    # ── Results ────────────────────────────────────────────────────────
    print_header(f"   SELECTED FACTORS — {strategy_name}", "─")

    beta_aen = stage2['beta_aen']
    selected_idx = stage2['selected_idx']
    selected_names = [factor_names[i] for i in selected_idx]

    print(f"\n   Selected {len(selected_names)} factors:")
    print(f"   {'Factor':<30} {'β̂(AEN)':>12} {'β̂(enet)':>12} {'weight':>12}")
    print(f"   {'─' * 68}")
    for i in selected_idx:
        print(f"   {factor_names[i]:<30} {beta_aen[i]:>+12.6f}"
              f" {beta_enet[i]:>+12.6f} {weights[i]:>12.4f}")

    # ── Save ───────────────────────────────────────────────────────────
    coeff_df = pd.DataFrame({
        'factor': factor_names,
        'beta_enet_raw': stage1['beta_raw'],
        'beta_enet_rescaled': stage1['beta_enet'],
        'adaptive_weight': weights,
        'beta_aen_raw': stage2['beta_raw'],
        'beta_aen_rescaled': stage2['beta_aen'],
        'selected': np.abs(stage2['beta_raw']) > COEF_TOL
        
    })

    aen_results = {
        'strategy': strategy_name,
        'T': T, 'p': p,
        'tuning_criterion': AEN_TUNING_CRITERION,
        'gic_alpha': gic_alpha if criterion == "GIC" else None,
        'aen_gamma': AEN_GAMMA,
        'conventions': {
            'standardization': 'L2-norm = 1 (paper p. 1735)',
            'lambda_scale': 'per-observation',
            'rescaling': '(1 + lambda2_po * T)',
            'bic_df_proxy': 'n_nonzero',
            'weights_on': 'rescaled beta_enet',
        },
        'stage1': {
            'lambda1_po': stage1['lambda1_po_best'],
            'lambda2_po': stage1['lambda2_po_best'],
            'lambda2_sum': stage1['lambda2_sum_best'],
            'rescale': stage1['rescale'],
            'ic_rescaled': stage1['ic_best'],
            'ic_raw': stage1['ic_raw_best'],
            'df': stage1['df_best'],
        },
        'stage2': {
            'lambda1_star_po': stage2['lambda1_star_po'],
            'lambda2_po': stage1['lambda2_po_best'],
            'rescale': stage2['rescale'],
            'ic_rescaled': stage2['ic_best'],
            'ic_raw': stage2['ic_raw_best'],
            'df': stage2['df_best'],
            'df_under_raw_ic': stage2['raw_ic_df'],
        },
        'selected_factors': selected_names,
        'n_selected': len(selected_names),
        'selected_coefficients': {
            factor_names[i]: float(beta_aen[i]) for i in selected_idx
        }
    }

    with open(strategy_dir / "aen_results.json", 'w') as f:
        json.dump(aen_results, f, indent=2)
    coeff_df.to_csv(strategy_dir / "aen_coefficients.csv", index=False)
    stage1['grid_results'].to_csv(strategy_dir / "aen_stage1_grid.csv", index=False)
    stage2['grid_results'].to_csv(strategy_dir / "aen_stage2_grid.csv", index=False)

    print(f"\n   💾 aen_results.json, aen_coefficients.csv")
    print(f"   💾 aen_stage1_grid.csv, aen_stage2_grid.csv")

    return aen_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    crit_label = AEN_TUNING_CRITERION
    if crit_label == "SIS_BIC":
        crit_label = "SIS + BIC"
    elif crit_label == "CV":
        crit_label = "Rolling-origin CV"

    print_header("ADAPTIVE ELASTIC-NET ESTIMATION")
    print(f"\n   Zou & Zhang (2009), Annals of Statistics, 37(4)")
    print(f"   Criterion: {crit_label}")
    if AEN_TUNING_CRITERION == "GIC":
        print(f"   GIC α = {GIC_ALPHA}")
    elif AEN_TUNING_CRITERION == "CV":
        print(f"   CV initial window: {CV_INITIAL_FRAC:.0%}")
        print(f"   CV step-ahead: h = {CV_STEP_AHEAD}")
    print(f"   AEN γ = {AEN_GAMMA}")
    print(f"   λ₂ same in Stage 1 and Stage 2 (paper p. 1737)")

    all_results = {}

    for strategy_name in STRATEGIES:
        print_header(f"STRATEGY: {strategy_name}")
        strategy_dir = get_strategy_aen_dir(strategy_name)
        if not (strategy_dir / "y_centered.parquet").exists():
            print(f"\n   ❌ Preprocessed data not found. Run 01 first.")
            continue
        try:
            results = run_aen_for_strategy(strategy_name)
            all_results[strategy_name] = results
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
            import traceback; traceback.print_exc()

    # ── Cross-strategy comparison ──────────────────────────────────────
    if all_results:
        print_header("CROSS-STRATEGY COMPARISON")

        print(f"\n   Criterion: {crit_label}")
        print(f"\n   {'Strategy':<20} {'Sel':>4} {'IC':>10}"
              f" {'λ₂°':>10} {'λ₁*°':>10} {'df(raw)':>8}")
        print(f"   {'─' * 64}")

        for name, res in all_results.items():
            print(f"   {name:<20} {res['n_selected']:>4}"
                  f" {res['stage2']['ic_rescaled']:>10.2f}"
                  f" {res['stage2']['lambda2_po']:>10.6f}"
                  f" {res['stage2']['lambda1_star_po']:>10.6f}"
                  f" {res['stage2']['df_under_raw_ic']:>8}")

        print(f"\n   Selected factors per strategy:")
        for name, res in all_results.items():
            print(f"\n   {name}:")
            if res['selected_coefficients']:
                for factor, coeff in res['selected_coefficients'].items():
                    print(f"      {factor:<30} β̂(AEN) = {coeff:+.6f}")
            else:
                print(f"      (none)")

    # Save global summary
    aen_output_dir = get_aen_output_dir()
    summary = {
        'tuning_criterion': AEN_TUNING_CRITERION,
        'gic_alpha': GIC_ALPHA if AEN_TUNING_CRITERION == "GIC" else None,
        'config': {
            'aen_gamma': AEN_GAMMA,
            'lambda2_same_both_stages': True,
            'lambda2_grid_po': AEN_LAMBDA2_GRID.tolist(),
            'standardization': 'L2-norm = 1',
            'weights_on': 'rescaled_beta_enet',
        },
        'results': {name: {
            'selected_factors': res['selected_factors'],
            'n_selected': res['n_selected'],
            'stage1_ic': res['stage1']['ic_rescaled'],
            'stage2_ic': res['stage2']['ic_rescaled'],
            'selected_coefficients': res['selected_coefficients']
        } for name, res in all_results.items()}
    }
    with open(aen_output_dir / "aen_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n   💾 {aen_output_dir / 'aen_summary.json'}")
    print(f"\n{'=' * 80}")
    print(f"✅ AEN ESTIMATION COMPLETE (criterion: {crit_label})")
    print(f"{'=' * 80}")
    print(f"\n   🎯 Next: python src/aen/03_aen_post_selection_ols.py")


if __name__ == "__main__":
    main()
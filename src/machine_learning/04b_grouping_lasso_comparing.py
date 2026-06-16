"""
================================================================================
04b_grouping_lasso_comparison.py - Grouping Effect & LASSO vs AEN Stability
================================================================================
Two diagnostics motivated by Zou & Hastie (2005, JRSS-B):

Part A — Grouping Effect Verification (Theorem 1, p. 305-306):
    For each strategy, compute the Zou-Hastie coefficient distance
        D(i,j) = |β̂_i − β̂_j| / ||y||₁
    for all selected-factor pairs with |ρ(i,j)| > 0.30.
    Verify that D(i,j) ≤ (1/λ₂)·√(2(1−ρ)) — the paper's upper bound.

    This shows that the elastic net penalty actively induces the
    grouping effect on YOUR data, not just in theory.

Part B — LASSO vs AEN Bootstrap Stability Comparison:
    Run LASSO (λ₂ = 0) bootstrap replicates with the same infrastructure
    as the AEN bootstrap (Step 04). Compare:
      - Jaccard similarity of selected sets across replicates
      - Frequency distribution (concentrated vs dispersed)
      - Which factors are "vote-split" by LASSO but not by AEN

    This provides the empirical evidence that the elastic net's
    grouping effect + stability selection outperforms pure LASSO
    in the presence of correlated factors (Sections 4-5 of the paper).

Dependencies:
    - Step 01 outputs: y_centered.parquet, X_standardized.parquet
    - Step 02 outputs: aen_results.json (for stage 1 λ₂)
    - Step 04 outputs: bootstrap_stability.json, bootstrap_frequencies.csv

Outputs (per strategy, in strategy AEN dir):
    - grouping_effect.json          (Part A: Theorem 1 verification)
    - lasso_vs_aen_stability.json   (Part B: comparison metrics)
    - lasso_bootstrap_frequencies.csv (Part B: per-factor LASSO freqs)

Author:      Alessio Ottaviani
Institution: EDHEC Business School – PhD Thesis
================================================================================
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Lasso
from joblib import Parallel, delayed

import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"

spec = importlib.util.spec_from_file_location("aen_config", config_path)
aen_config = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(aen_config)

AEN_GAMMA             = aen_config.AEN_GAMMA
AEN_LAMBDA2_GRID      = aen_config.AEN_LAMBDA2_GRID
AEN_LAMBDA1_N_VALUES  = aen_config.AEN_LAMBDA1_N_VALUES
AEN_TUNING_CRITERION  = aen_config.AEN_TUNING_CRITERION
GIC_ALPHA             = aen_config.GIC_ALPHA
STRATEGIES            = aen_config.STRATEGIES
BOOTSTRAP_METHOD      = aen_config.BOOTSTRAP_METHOD
COEF_TOL               = aen_config.COEF_TOL
BOOTSTRAP_N_REPS      = aen_config.BOOTSTRAP_N_REPS
BOOTSTRAP_BLOCK_LENGTH = aen_config.BOOTSTRAP_BLOCK_LENGTH
STABILITY_THRESHOLD   = aen_config.STABILITY_THRESHOLD
get_strategy_aen_dir  = aen_config.get_strategy_aen_dir
get_aen_output_dir    = aen_config.get_aen_output_dir


def print_header(title, char="="):
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")


# ============================================================================
# BOOTSTRAP INDEX GENERATION (same as 04_bootstrap.py — duplicated to
# keep this file standalone; uses same seed=42 for reproducibility)
# ============================================================================

def circular_block_bootstrap(T, block_length, rng):
    indices = np.empty(T, dtype=np.int64)
    for t in range(T):
        if t % block_length == 0:
            indices[t] = rng.integers(0, T)
        else:
            indices[t] = (indices[t - 1] + 1) % T
    return indices


def stationary_bootstrap(T, block_length, rng):
    p_new = 1.0 / block_length
    indices = np.empty(T, dtype=np.int64)
    indices[0] = rng.integers(0, T)
    for t in range(1, T):
        if rng.random() < p_new:
            indices[t] = rng.integers(0, T)
        else:
            indices[t] = (indices[t - 1] + 1) % T
    return indices


def generate_bootstrap_indices(T, block_length, method, rng):
    if method == "circular":
        return circular_block_bootstrap(T, block_length, rng)
    elif method == "stationary":
        return stationary_bootstrap(T, block_length, rng)
    else:
        raise ValueError(f"Unknown BOOTSTRAP_METHOD: '{method}'.")


# ============================================================================
# IC computation (lightweight, same as 04)
# ============================================================================

def compute_ic(y, X_beta, T, df, criterion, gic_alpha=3.0):
    rss = np.sum((y - X_beta) ** 2)
    if rss / T < 1e-15:
        return np.inf, rss
    log_lik = T * np.log(rss / T)
    if criterion in ("BIC", "SIS_BIC"):
        ic = log_lik + df * np.log(T)
    elif criterion == "HQC":
        ic = log_lik + 2 * np.log(np.log(T)) * df
    elif criterion == "AIC":
        ic = log_lik + 2 * df
    elif criterion == "AICc":
        aic = log_lik + 2 * df
        ic = aic + (2 * df * (df + 1)) / (T - df - 1) \
            if T - df - 1 > 0 else np.inf
    elif criterion == "GIC":
        ic = log_lik + gic_alpha * df
    elif criterion == "CV":
        ic = log_lik + df * np.log(T)
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    return ic, rss


def compute_lambda1_max(X, y):
    T = X.shape[0]
    return np.max(np.abs(X.T @ y)) / T


def build_lambda1_grid(lambda1_max, n_values=100, ratio_min=1e-4):
    lambda1_max = max(float(lambda1_max), 1e-12)
    return np.logspace(np.log10(lambda1_max),
                       np.log10(lambda1_max * ratio_min), n_values)


# ============================================================================
# PART A: GROUPING EFFECT VERIFICATION (Zou & Hastie 2005, Theorem 1)
# ============================================================================
#
# Theorem 1 (p. 306): For the naïve elastic net with parameters (λ₁, λ₂),
# if β̂_i · β̂_j > 0 (same sign, both nonzero), then:
#
#     D_{λ₁,λ₂}(i,j) = |β̂_i − β̂_j| / ||y||₁  ≤  (1/λ₂)·√(2(1−ρ))
#
# where ρ = x_i^T x_j (sample correlation on standardized predictors).
#
# We verify this on the STAGE 1 elastic net coefficients (β̂_enet),
# which are the (1+λ₂)-rescaled naïve estimates.  The bound applies
# to the naïve estimates, so we undo the rescaling before computing D.
#
# NOTE: The theorem requires both β̂_i and β̂_j to be nonzero and
# same-sign. We check all pairs with |ρ| > 0.30 among selected factors.

def verify_grouping_effect(strategy_name):
    """Verify Zou-Hastie Theorem 1 on Stage 1 elastic net."""

    strategy_dir = get_strategy_aen_dir(strategy_name)

    # Load preprocessed data
    y_df = pd.read_parquet(strategy_dir / "y_centered.parquet")
    X_df = pd.read_parquet(strategy_dir / "X_standardized.parquet")
    y = y_df['y'].values
    X = X_df.values
    factor_names = X_df.columns.tolist()
    T, p = X.shape

    # Load AEN results (need Stage 1 lambda2 and coefficients)
    with open(strategy_dir / "aen_results.json", 'r') as f:
        aen_results = json.load(f)

    # Load coefficients file for Stage 1 elastic net
    coeff_df = pd.read_csv(strategy_dir / "aen_coefficients.csv")

    # β̂_enet (rescaled) and β̂_naive (raw) from Stage 1
    beta_enet = coeff_df['beta_enet_rescaled'].values
    beta_raw = coeff_df['beta_enet_raw'].values  # naïve (pre-rescale)

    lambda2_po = aen_results['stage1']['lambda2_po']
    lambda2_sum = aen_results['stage1']['lambda2_sum']
    rescale = aen_results['stage1']['rescale']

    # The theorem operates on the naïve elastic net coefficients.
    # β̂_naive = β̂_enet / (1 + λ₂)  (eq. 12 of the paper)
    beta_naive = beta_enet / rescale

    # ||y||₁ (L1-norm of centered y)
    y_l1 = np.sum(np.abs(y))

    # Correlation matrix on standardized X
    corr_matrix = np.corrcoef(X.T)

    # Find all pairs of nonzero, same-sign coefficients with |ρ| > 0.30
    nonzero_idx = np.where(np.abs(beta_naive) > COEF_TOL)[0]
    MIN_RHO = 0.30

    pair_results = []
    for ii, i in enumerate(nonzero_idx):
        for j in nonzero_idx[ii + 1:]:
            rho = corr_matrix[i, j]
            # Same sign check
            if beta_naive[i] * beta_naive[j] <= 0:
                continue
            if abs(rho) < MIN_RHO:
                continue

            # D(i,j) = |β̂_i(naive) − β̂_j(naive)| / ||y||₁
            D_ij = abs(beta_naive[i] - beta_naive[j]) / y_l1

            # Theorem bound: (1/λ₂_sum) · √(2(1−ρ))
            # NOTE: λ₂ in the theorem is the total penalty λ₂ (not per-obs).
            # In our notation, λ₂_sum = λ₂_po × T is the total λ₂.
            if lambda2_sum > 0:
                bound = (1.0 / lambda2_sum) * np.sqrt(2.0 * (1.0 - rho))
                satisfied = D_ij <= bound + 1e-10  # small tolerance
            else:
                bound = np.inf
                satisfied = True  # trivially true for λ₂ = 0

            pair_results.append({
                'factor_i': factor_names[i],
                'factor_j': factor_names[j],
                'rho': round(float(rho), 4),
                'beta_naive_i': round(float(beta_naive[i]), 6),
                'beta_naive_j': round(float(beta_naive[j]), 6),
                'D_ij': round(float(D_ij), 8),
                'bound': round(float(bound), 8) if np.isfinite(bound) else None,
                'bound_satisfied': bool(satisfied),
                'ratio_D_over_bound': round(float(D_ij / bound), 4)
                    if np.isfinite(bound) and bound > 0 else None,
            })

    # Also compute D for ALL nonzero pairs (regardless of ρ) for completeness
    all_nonzero_pairs = 0
    same_sign_pairs = 0
    for ii, i in enumerate(nonzero_idx):
        for j in nonzero_idx[ii + 1:]:
            all_nonzero_pairs += 1
            if beta_naive[i] * beta_naive[j] > 0:
                same_sign_pairs += 1

    result = {
        'strategy': strategy_name,
        'theorem': 'Zou & Hastie (2005, JRSS-B), Theorem 1',
        'description': (
            'D(i,j) = |β_naive_i - β_naive_j| / ||y||_1  ≤  '
            '(1/λ₂) · √(2(1-ρ))'
        ),
        'lambda2_po': lambda2_po,
        'lambda2_sum': lambda2_sum,
        'rescale': rescale,
        'y_l1_norm': round(float(y_l1), 4),
        'n_nonzero_stage1': len(nonzero_idx),
        'n_all_nonzero_pairs': all_nonzero_pairs,
        'n_same_sign_pairs': same_sign_pairs,
        'min_rho_for_table': MIN_RHO,
        'n_qualifying_pairs': len(pair_results),
        'all_bounds_satisfied': all(p['bound_satisfied']
                                    for p in pair_results),
        'pairs': pair_results,
    }

    # Print
    print(f"\n   ── Grouping Effect (Theorem 1) ──")
    print(f"   λ₂ (per-obs) = {lambda2_po:.6f},"
          f" λ₂ (total) = {lambda2_sum:.4f},"
          f" rescale = {rescale:.4f}")
    print(f"   Stage 1 nonzero: {len(nonzero_idx)} factors,"
          f" {all_nonzero_pairs} pairs,"
          f" {same_sign_pairs} same-sign")
    print(f"   Qualifying pairs (|ρ| > {MIN_RHO}, same sign):"
          f" {len(pair_results)}")

    if pair_results:
        print(f"\n   {'Factor i':<20} {'Factor j':<20} {'ρ':>6}"
              f" {'D(i,j)':>12} {'Bound':>12} {'D/Bound':>8} {'OK':>3}")
        print(f"   {'─' * 85}")
        for p in sorted(pair_results, key=lambda x: -abs(x['rho'])):
            bound_str = f"{p['bound']:.8f}" if p['bound'] is not None else "∞"
            ratio_str = f"{p['ratio_D_over_bound']:.4f}" \
                if p['ratio_D_over_bound'] is not None else "—"
            ok = "✅" if p['bound_satisfied'] else "❌"
            print(f"   {p['factor_i']:<20} {p['factor_j']:<20}"
                  f" {p['rho']:>6.3f}"
                  f" {p['D_ij']:>12.8f}"
                  f" {bound_str:>12}"
                  f" {ratio_str:>8} {ok:>3}")

        if result['all_bounds_satisfied']:
            print(f"\n   ✅ All {len(pair_results)} bounds satisfied."
                  f" Grouping effect verified.")
        else:
            n_violated = sum(1 for p in pair_results
                             if not p['bound_satisfied'])
            print(f"\n   ⚠️  {n_violated}/{len(pair_results)} bounds violated."
                  f" Check numerical tolerances.")
    else:
        print(f"\n   ℹ️  No qualifying pairs found"
              f" (no same-sign, |ρ| > {MIN_RHO} pairs among"
              f" Stage 1 nonzero factors).")

    # Save
    with open(strategy_dir / "grouping_effect.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n   💾 grouping_effect.json")

    return result


# ============================================================================
# PART B: LASSO vs AEN BOOTSTRAP STABILITY COMPARISON
# ============================================================================
#
# Run a pure LASSO (λ₂ = 0) bootstrap with the SAME infrastructure:
#   - Same block bootstrap indices (same seed=42)
#   - Same IC criterion (HQC/BIC/etc.)
#   - Same λ₁ grid (100 log-spaced values)
#   - Same number of replicates (B)
#   - No adaptive weights (Stage 1 LASSO only, no Stage 2)
#
# Compare AEN (from Step 04) vs LASSO on:
#   1. Jaccard similarity across bootstrap replicates:
#          J(b₁, b₂) = |S(b₁) ∩ S(b₂)| / |S(b₁) ∪ S(b₂)|
#      Averaged over all pairs. Higher = more stable selection.
#
#   2. Coefficient of variation of selection frequencies:
#      CV = std(freq) / mean(freq) across factors with freq > 5%.
#      Lower = more concentrated (fewer factors "split").
#
#   3. Effective number of factors "above noise":
#      Factors with selection frequency > STABILITY_THRESHOLD.
#
#   4. Vote-splitting detection:
#      Factors with AEN freq > 50% but LASSO freq < 30%,
#      or vice versa.

def run_lasso_single(y, X, n_lambda1, criterion, gic_alpha=3.0,
                     max_iter=10000, tol=1e-7):
    """
    Pure LASSO (λ₂ = 0) with IC-based tuning on a single (y, X) sample.
    Returns: (selected_idx, beta_lasso)
    """
    T, p = X.shape

    # Re-standardize (same as AEN bootstrap)
    y_mean = np.mean(y)
    y_c = y - y_mean
    X_mean = np.mean(X, axis=0)
    X_c = X - X_mean
    X_l2 = np.sqrt(np.sum(X_c ** 2, axis=0))
    X_l2[X_l2 == 0] = 1.0
    X_s = X_c / X_l2

    lambda1_max = compute_lambda1_max(X_s, y_c)
    lambda1_max = max(lambda1_max, 1e-6)
    lambda1_grid = build_lambda1_grid(lambda1_max, n_lambda1)

    best_ic = np.inf
    best_beta = None

    for lambda1_po in lambda1_grid:
        # sklearn Lasso: (1/2T)||y - Xβ||² + alpha·||β||₁
        # Our form: (1/T)||y - Xβ||² + λ₁||β||₁
        # Match: alpha_sk = λ₁ / 2
        alpha_sk = max(lambda1_po / 2.0, 1e-15)

        model = Lasso(alpha=alpha_sk, fit_intercept=False,
                      max_iter=max_iter, tol=tol, warm_start=False)
        model.fit(X_s, y_c)
        beta = model.coef_.copy()

        pred = X_s @ beta
        df = int(np.sum(np.abs(beta) > 1e-6))
        ic, _ = compute_ic(y_c, pred, T, df, criterion, gic_alpha)

        if ic < best_ic:
            best_ic = ic
            best_beta = beta.copy()

    selected = np.where(np.abs(best_beta) > 1e-6)[0].tolist()
    return selected, best_beta


def mean_pairwise_jaccard(selection_matrix):
    """
    Average Jaccard similarity across all pairs of bootstrap replicates.

    J(b₁, b₂) = |S₁ ∩ S₂| / |S₁ ∪ S₂|

    Returns mean Jaccard (float). Higher = more stable.
    """
    B = selection_matrix.shape[0]
    if B < 2:
        return np.nan

    # Vectorized: for all pairs (i, j), compute intersection and union
    # Use float for efficiency
    S = selection_matrix.astype(np.float32)

    # Sample random pairs for speed if B is large
    n_pairs = min(B * (B - 1) // 2, 5000)
    rng = np.random.default_rng(seed=123)

    jaccards = []
    if n_pairs < B * (B - 1) // 2:
        # Random sampling of pairs
        for _ in range(n_pairs):
            i, j = rng.choice(B, size=2, replace=False)
            inter = np.sum(S[i] * S[j])
            union = np.sum(np.clip(S[i] + S[j], 0, 1))
            jaccards.append(inter / union if union > 0 else 0.0)
    else:
        # All pairs
        for i in range(B):
            for j in range(i + 1, B):
                inter = np.sum(S[i] * S[j])
                union = np.sum(np.clip(S[i] + S[j], 0, 1))
                jaccards.append(inter / union if union > 0 else 0.0)

    return float(np.mean(jaccards))


def run_lasso_bootstrap_comparison(strategy_name):
    """Run LASSO bootstrap and compare with AEN bootstrap results."""

    strategy_dir = get_strategy_aen_dir(strategy_name)

    # Load preprocessed data
    y_df = pd.read_parquet(strategy_dir / "y_centered.parquet")
    X_df = pd.read_parquet(strategy_dir / "X_standardized.parquet")
    y = y_df['y'].values
    X = X_df.values
    factor_names = X_df.columns.tolist()
    T, n_factors = X.shape

    criterion = AEN_TUNING_CRITERION
    gic_alpha = GIC_ALPHA if criterion == "GIC" else 3.0

    # Reduced λ₁ grid (same as AEN bootstrap)
    BOOT_N_LAMBDA1 = max(AEN_LAMBDA1_N_VALUES // 2, 30)

    # ── Load AEN bootstrap results ─────────────────────────────────────
    with open(strategy_dir / "bootstrap_stability.json", 'r') as f:
        aen_boot = json.load(f)

# factor_frequencies has structure {factor: {'pi_lambda_star': x, 'pi_max_lambda': y}}
    # Use pi_lambda_star (primary rule, Π̂(j, λ*)) for comparison with LASSO,
    # since that is the frequency cutoff enforced by STABILITY_THRESHOLD in 04.
    def _get_freq(fname):
        entry = aen_boot['factor_frequencies'].get(fname, 0.0)
        if isinstance(entry, dict):
            return float(entry.get('pi_lambda_star', 0.0))
        return float(entry)  # backward compat with old flat format
    
    aen_freqs = np.array([_get_freq(f) for f in factor_names])
    aen_stable = aen_boot['stable_factors']

    # Load AEN selection matrix from bootstrap_coefficients.csv
    # (nonzero coefficient → selected)
    aen_coeff_df = pd.read_csv(strategy_dir / "bootstrap_coefficients.csv")
    aen_selection_matrix = (aen_coeff_df.abs().values > 1e-6).astype(bool)

    # ── Generate bootstrap indices (SAME seed as Step 04) ──────────────
    rng = np.random.default_rng(seed=42)
    all_boot_idx = [
        generate_bootstrap_indices(T, BOOTSTRAP_BLOCK_LENGTH,
                                   BOOTSTRAP_METHOD, rng)
        for _ in range(BOOTSTRAP_N_REPS)
    ]

    # ── Run LASSO bootstrap ────────────────────────────────────────────
    print(f"\n   Running LASSO (λ₂=0) bootstrap:"
          f" B={BOOTSTRAP_N_REPS}, IC={criterion}...")

    def _single_lasso_rep(boot_idx):
        y_boot = y[boot_idx]
        X_boot = X[boot_idx, :]
        try:
            selected_idx, beta_lasso = run_lasso_single(
                y_boot, X_boot,
                n_lambda1=BOOT_N_LAMBDA1,
                criterion=criterion,
                gic_alpha=gic_alpha)
            sel_mask = np.zeros(n_factors, dtype=bool)
            sel_mask[selected_idx] = True
            return sel_mask, beta_lasso, len(selected_idx), None
        except Exception as e:
            return (np.zeros(n_factors, dtype=bool),
                    np.zeros(n_factors), 0, str(e))

    t_start = time.time()
    results_list = Parallel(n_jobs=-1, verbose=5)(
        delayed(_single_lasso_rep)(idx) for idx in all_boot_idx
    )
    elapsed = time.time() - t_start
    print(f"\n   ✅ LASSO bootstrap: {elapsed:.1f}s"
          f" ({elapsed/BOOTSTRAP_N_REPS:.2f}s/rep)")

    lasso_sel_matrix = np.array([r[0] for r in results_list])
    lasso_coeff_matrix = np.array([r[1] for r in results_list])
    lasso_n_selected = np.array([r[2] for r in results_list])
    lasso_errors = [r[3] for r in results_list if r[3] is not None]

    if lasso_errors:
        print(f"   ⚠️  LASSO errors: {len(lasso_errors)}/{BOOTSTRAP_N_REPS}")

    # ── LASSO frequencies ──────────────────────────────────────────────
    lasso_freqs = lasso_sel_matrix.mean(axis=0)

    # ── Metrics ────────────────────────────────────────────────────────

    # 1. Jaccard similarity
    print(f"\n   Computing Jaccard similarity...")
    jaccard_aen = mean_pairwise_jaccard(aen_selection_matrix)
    jaccard_lasso = mean_pairwise_jaccard(lasso_sel_matrix)

    # 2. Model size stats
    aen_n_sel_arr = aen_selection_matrix.sum(axis=1)

    # 3. Number of factors above threshold
    n_stable_aen = int(np.sum(aen_freqs >= STABILITY_THRESHOLD))
    n_stable_lasso = int(np.sum(lasso_freqs >= STABILITY_THRESHOLD))

    # 4. Frequency CV (among factors with freq > 5%)
    mask_aen = aen_freqs > 0.05
    mask_lasso = lasso_freqs > 0.05
    cv_aen = (float(np.std(aen_freqs[mask_aen]) /
                     np.mean(aen_freqs[mask_aen]))
              if mask_aen.sum() > 1 else np.nan)
    cv_lasso = (float(np.std(lasso_freqs[mask_lasso]) /
                       np.mean(lasso_freqs[mask_lasso]))
                if mask_lasso.sum() > 1 else np.nan)

    # 5. Vote-splitting: factors with high AEN freq but low LASSO freq
    vote_split_detected = []
    for j, f in enumerate(factor_names):
        aen_f = aen_freqs[j]
        lasso_f = lasso_freqs[j]
        if aen_f >= 0.50 and lasso_f < 0.30:
            vote_split_detected.append({
                'factor': f,
                'aen_freq': round(float(aen_f), 4),
                'lasso_freq': round(float(lasso_f), 4),
                'interpretation': (
                    'AEN grouping effect preserves this factor; '
                    'LASSO splits votes with correlated alternatives')
            })
        elif lasso_f >= 0.50 and aen_f < 0.30:
            vote_split_detected.append({
                'factor': f,
                'lasso_freq': round(float(lasso_f), 4),
                'aen_freq': round(float(aen_f), 4),
                'interpretation': (
                    'LASSO favors this factor; AEN may prefer a '
                    'correlated alternative via grouping')
            })

    # 6. Per-factor comparison table
    comparison_rows = []
    for j, f in enumerate(factor_names):
        if aen_freqs[j] >= 0.05 or lasso_freqs[j] >= 0.05:
            comparison_rows.append({
                'factor': f,
                'aen_freq': round(float(aen_freqs[j]), 4),
                'lasso_freq': round(float(lasso_freqs[j]), 4),
                'diff': round(float(aen_freqs[j] - lasso_freqs[j]), 4),
                'aen_stable': f in aen_stable,
            })
    comparison_df = pd.DataFrame(comparison_rows).sort_values(
        'aen_freq', ascending=False)

    # ── Print ──────────────────────────────────────────────────────────
    print(f"\n   ══════════════════════════════════════════════════════")
    print(f"   LASSO vs AEN BOOTSTRAP STABILITY COMPARISON")
    print(f"   ══════════════════════════════════════════════════════")

    print(f"\n   {'Metric':<40} {'AEN':>10} {'LASSO':>10}")
    print(f"   {'─' * 62}")
    print(f"   {'Jaccard similarity (mean pairwise)':<40}"
          f" {jaccard_aen:>10.4f} {jaccard_lasso:>10.4f}")
    print(f"   {'Model size (mean)':<40}"
          f" {aen_n_sel_arr.mean():>10.1f}"
          f" {lasso_n_selected.mean():>10.1f}")
    print(f"   {'Model size (median)':<40}"
          f" {np.median(aen_n_sel_arr):>10.0f}"
          f" {np.median(lasso_n_selected):>10.0f}")
    print(f"   {'Factors ≥ π_thr ({STABILITY_THRESHOLD:.0%})':<40}"
          f" {n_stable_aen:>10} {n_stable_lasso:>10}")
    print(f"   {'Freq CV (factors > 5%)':<40}"
          f" {cv_aen:>10.4f} {cv_lasso:>10.4f}")
    print(f"   {'Null model freq':<40}"
          f" {(aen_n_sel_arr == 0).mean():>10.1%}"
          f" {(lasso_n_selected == 0).mean():>10.1%}")

    if jaccard_aen > jaccard_lasso:
        pct_improvement = (jaccard_aen / jaccard_lasso - 1) * 100 \
            if jaccard_lasso > 0 else np.inf
        print(f"\n   → AEN is {pct_improvement:.0f}% more stable than"
              f" LASSO (Jaccard)")
    else:
        print(f"\n   → LASSO is as stable or more stable than AEN (Jaccard)")

    if vote_split_detected:
        print(f"\n   Vote-splitting detected ({len(vote_split_detected)} factors):")
        for vs in vote_split_detected:
            print(f"     {vs['factor']:<25}"
                  f" AEN={vs['aen_freq']:.0%}"
                  f" LASSO={vs['lasso_freq']:.0%}"
                  f" — {vs['interpretation']}")

    print(f"\n   Top factors (freq > 5%):")
    print(f"   {'Factor':<25} {'AEN':>8} {'LASSO':>8} {'Diff':>8}")
    print(f"   {'─' * 52}")
    for _, row in comparison_df.head(15).iterrows():
        marker = "✅" if row['aen_stable'] else "  "
        print(f"   {row['factor']:<25}"
              f" {row['aen_freq']:>7.1%}"
              f" {row['lasso_freq']:>7.1%}"
              f" {row['diff']:>+7.1%} {marker}")

    # ── Save ───────────────────────────────────────────────────────────
    result = {
        'strategy': strategy_name,
        'reference': 'Zou & Hastie (2005, JRSS-B), Sections 4-5',
        'description': (
            'Bootstrap stability comparison: AEN (grouping effect via '
            'ℓ₂ penalty) vs pure LASSO (λ₂=0). Same bootstrap indices, '
            'IC criterion, and λ₁ grid.'
        ),
        'config': {
            'bootstrap_method': BOOTSTRAP_METHOD,
            'n_reps': BOOTSTRAP_N_REPS,
            'block_length': BOOTSTRAP_BLOCK_LENGTH,
            'criterion': AEN_TUNING_CRITERION,
            'stability_threshold': STABILITY_THRESHOLD,
            'lasso_lambda1_grid_size': BOOT_N_LAMBDA1,
        },
        'metrics': {
            'jaccard_aen': round(jaccard_aen, 4),
            'jaccard_lasso': round(jaccard_lasso, 4),
            'jaccard_improvement_pct': round(
                (jaccard_aen / jaccard_lasso - 1) * 100, 1)
                if jaccard_lasso > 0 else None,
            'model_size_mean_aen': round(float(aen_n_sel_arr.mean()), 2),
            'model_size_mean_lasso': round(float(lasso_n_selected.mean()), 2),
            'n_stable_aen': n_stable_aen,
            'n_stable_lasso': n_stable_lasso,
            'freq_cv_aen': round(cv_aen, 4) if np.isfinite(cv_aen) else None,
            'freq_cv_lasso': round(cv_lasso, 4)
                if np.isfinite(cv_lasso) else None,
        },
        'vote_splitting': vote_split_detected,
        'elapsed_seconds': round(elapsed, 1),
    }

    with open(strategy_dir / "lasso_vs_aen_stability.json", 'w') as f:
        json.dump(result, f, indent=2)

    # Save LASSO frequencies
    lasso_freq_df = pd.DataFrame({
        'factor': factor_names,
        'lasso_freq': lasso_freqs,
        'aen_freq': aen_freqs,
        'diff': aen_freqs - lasso_freqs,
    }).sort_values('lasso_freq', ascending=False)
    lasso_freq_df.to_csv(
        strategy_dir / "lasso_bootstrap_frequencies.csv", index=False)

    print(f"\n   💾 lasso_vs_aen_stability.json")
    print(f"   💾 lasso_bootstrap_frequencies.csv")

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("GROUPING EFFECT & LASSO vs AEN STABILITY")
    print(f"\n   Reference: Zou & Hastie (2005, JRSS-B)")
    print(f"   Part A: Theorem 1 — grouping effect verification")
    print(f"   Part B: Bootstrap stability comparison (LASSO vs AEN)")
    print(f"   Criterion: {AEN_TUNING_CRITERION}")
    print(f"   Bootstrap: B={BOOTSTRAP_N_REPS},"
          f" L={BOOTSTRAP_BLOCK_LENGTH},"
          f" method={BOOTSTRAP_METHOD}")

    all_grouping = {}
    all_comparison = {}

    for strategy_name in STRATEGIES:
        print_header(f"STRATEGY: {strategy_name}")

        strategy_dir = get_strategy_aen_dir(strategy_name)

        # Check prerequisites
        if not (strategy_dir / "y_centered.parquet").exists():
            print(f"\n   ❌ Preprocessed data not found. Run 01 first.")
            continue
        if not (strategy_dir / "aen_results.json").exists():
            print(f"\n   ❌ AEN results not found. Run 02 first.")
            continue
        if not (strategy_dir / "bootstrap_stability.json").exists():
            print(f"\n   ❌ Bootstrap results not found. Run 04 first.")
            continue

        # Part A: Grouping effect
        print_header(f"   PART A: Grouping Effect — {strategy_name}", "─")
        try:
            grouping = verify_grouping_effect(strategy_name)
            all_grouping[strategy_name] = grouping
        except Exception as e:
            print(f"\n   ❌ Error (Part A): {e}")
            import traceback; traceback.print_exc()

        # Part B: LASSO comparison
        print_header(f"   PART B: LASSO vs AEN Stability — {strategy_name}", "─")
        try:
            comparison = run_lasso_bootstrap_comparison(strategy_name)
            all_comparison[strategy_name] = comparison
        except Exception as e:
            print(f"\n   ❌ Error (Part B): {e}")
            import traceback; traceback.print_exc()

    # ── Cross-strategy summary ─────────────────────────────────────────
    if all_comparison:
        print_header("CROSS-STRATEGY SUMMARY")

        print(f"\n   {'Strategy':<20}"
              f" {'J(AEN)':>8} {'J(LASSO)':>10}"
              f" {'Impr%':>7}"
              f" {'Stable_AEN':>11} {'Stable_L':>9}")
        print(f"   {'─' * 68}")
        for name, res in all_comparison.items():
            m = res['metrics']
            impr = m.get('jaccard_improvement_pct', 0) or 0
            print(f"   {name:<20}"
                  f" {m['jaccard_aen']:>8.4f}"
                  f" {m['jaccard_lasso']:>10.4f}"
                  f" {impr:>+6.0f}%"
                  f" {m['n_stable_aen']:>11}"
                  f" {m['n_stable_lasso']:>9}")

    # Save global summary
    aen_output_dir = get_aen_output_dir()
    summary = {
        'grouping_effect': {
            name: {
                'n_qualifying_pairs': res['n_qualifying_pairs'],
                'all_bounds_satisfied': res['all_bounds_satisfied'],
                'lambda2_sum': res['lambda2_sum'],
            }
            for name, res in all_grouping.items()
        },
        'stability_comparison': {
            name: res['metrics']
            for name, res in all_comparison.items()
        },
    }
    with open(aen_output_dir / "grouping_lasso_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n   💾 {aen_output_dir / 'grouping_lasso_summary.json'}")

    print(f"\n{'=' * 80}")
    print(f"✅ GROUPING EFFECT & LASSO COMPARISON COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
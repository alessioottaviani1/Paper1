"""
================================================================================
04_bootstrap_stability.py  —  Bootstrap Stability Selection + Final OLS
================================================================================

METHODOLOGY
-----------
Fixed-grid adaptation of Meinshausen & Bühlmann (2010) stability selection
for short, autocorrelated financial time series (T ≈ 150–200, p ≈ 60).

Three documented departures from M&B (2010):

  1. BLOCK BOOTSTRAP instead of iid subsampling n/2
     Stationary bootstrap (Politis & Romano 1994) preserves temporal
     dependence.  Reference: Bühlmann (2002, Statistical Science).

  2. FIXED λ1 GRID instead of IC-inside-bootstrap
     The λ1 grid Λ is built ONCE on the full sample and held fixed.
     For each (replicate b, λ ∈ Λ), Stage-2 weighted AEN is run and
     selections are recorded.  Selection frequency matrix:

         Π̂(j, λ)  =  (1/B) Σ_b  1[j selected in rep b at λ]

     Motivation: IC-inside-bootstrap produced model sizes from 3 to 54
     per replicate, destroying the comparability of frequencies across
     replicates.  A replica-dependent λ grid means two replicates may
     search entirely different regions of the penalty landscape.

  3. FIXED λ2 AND ADAPTIVE WEIGHTS from full-sample Stage-1
     Only source of variation across replicates is the bootstrap sample.
     Reduces computational cost ~50× vs re-running Stage 1 per cell.

STABILITY RULE
--------------
  Primary:    Π̂(j) = Π̂(j, λ*)              where λ* = full-sample optimal λ1.
  Robustness: Π̂(j) = max_{λ∈Λ_q} Π̂(j, λ)  M&B (2010) original rule.

  Both are reported. The primary rule at λ* is more conservative and avoids
  the saturation artefact of max_λ when Λ_q contains near-zero λ values.

  Λ_q = {λ : q̄(λ) ≤ q_max}  when STABILITY_MAX_Q is set (M&B Thm 1 req.).
  Otherwise Λ_q = Λ (full grid, no restriction).

  CSS (Faletto & Bien 2022) cluster frequencies use sel_star — the same
  (B, p) slice at λ* used for individual_freq — to ensure logical
  consistency between individual and cluster-level stability rules.

SCALING DESIGN (separation of concerns)
-----------------------------------------
Preprocessing (01) produces X_standardized.parquet and y_centered.parquet:
    X: mean-centred, L2-norm = 1 per column  (paper p. 1735)
    y: mean-centred only  (paper p. 1733)

This script loads those files WITHOUT re-standardising on the full sample.
Re-standardisation happens ONLY inside _aen_boot_cell, where bootstrap
resampling genuinely destroys the original normalisation.

Two separate functions make this explicit:
    _compute_weights_and_grid(X_prepared, y_c)
        → X already mean-centred + L2-normed, y already centred (from 01)
        → Stage-1: CD with uniform weights  (no sklearn, no rescaling)
        → Stage-2: CD with adaptive weights (AEN)
        → called once on the full sample

    _aen_boot_cell(y_boot, X_boot, ...)
        → re-centres and re-L2-norms internally (bootstrap destroys norms)
        → uses fixed weights and fixed λ2 from full sample
        → no q_max truncation: model-size control via eligible Λ_q in Step 5

COEF_TOL
---------
COEF_TOL = 1e-4 is used uniformly for:
    - Stage-1 df count  (full-sample AEN)
    - Stage-2 selection  (full-sample AEN and each bootstrap cell)
    - Stable factor identification

COEF_TOL >> solver tol (1e-7): coefficients below 1e-4 are numerical
noise on L2-normalised predictors with T ≈ 150.  A sensitivity analysis
across {1e-5, 1e-4, 1e-3} is provided in 04b_coef_tol_sensitivity.py.

OUTPUTS (per strategy)
-----------------------
  bootstrap_stability.json     frequencies, stable factors, OLS results
  bootstrap_frequencies.csv    per-factor table:
                                  selection_frequency        = Π̂(j, λ*)  [primary]
                                  selection_frequency_maxlam = max_λ Π̂   [robustness]
  bootstrap_freq_matrix.csv    full Π̂(j, λ) matrix  (factors × λ-grid)
  ols_stable_results.json      OLS on primary stable factors (HAC SE)
  ols_stable_table.csv
  ols_medoid_results.json      OLS on medoid stable factors (HAC SE)
  ols_medoid_table.csv

References
----------
  Meinshausen & Bühlmann (2010), JRSS-B 72(4), 417-473.
  Shah & Samworth (2013), JRSS-B 75(1), 55-80.
  Faletto & Bien (2022), arXiv:2201.00494.
  Politis & Romano (1994), JASA 89(428), 1303-1313.
  Bühlmann (2002), Statistical Science 17(1), 52-72.
  Zou & Zhang (2009), Annals of Statistics 37(4), 1733-1751.

Author:      Alessio Ottaviani
Institution: EDHEC Business School — PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# NOTE: sklearn is NOT imported. Stage-1 uses the same CD solver as Stage-2
# (uniform weights), eliminating any sklearn parameterisation ambiguity.
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import statsmodels.api as sm

import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"

spec = importlib.util.spec_from_file_location("aen_config", config_path)
aen_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aen_config)

AEN_GAMMA                      = aen_config.AEN_GAMMA
AEN_LAMBDA2_GRID               = aen_config.AEN_LAMBDA2_GRID
AEN_LAMBDA1_N_VALUES           = aen_config.AEN_LAMBDA1_N_VALUES
AEN_TUNING_CRITERION           = aen_config.AEN_TUNING_CRITERION
GIC_ALPHA                      = aen_config.GIC_ALPHA
BOOTSTRAP_METHOD               = aen_config.BOOTSTRAP_METHOD
BOOTSTRAP_N_REPS               = aen_config.BOOTSTRAP_N_REPS
BOOTSTRAP_BLOCK_LENGTH         = aen_config.BOOTSTRAP_BLOCK_LENGTH
STABILITY_MAX_Q                = aen_config.STABILITY_MAX_Q
STABILITY_THRESHOLD            = aen_config.STABILITY_THRESHOLD
STABILITY_THRESHOLD_ROBUSTNESS = aen_config.STABILITY_THRESHOLD_ROBUSTNESS
CSS_ENABLED                    = aen_config.CSS_ENABLED
CSS_CORRELATION_THRESHOLD      = aen_config.CSS_CORRELATION_THRESHOLD
CSS_MIN_GAP                    = getattr(aen_config, 'CSS_MIN_GAP', 0.15)
CSS_REPRESENTATIVE_METHOD      = aen_config.CSS_REPRESENTATIVE_METHOD
HAC_LAGS                       = aen_config.HAC_LAGS
FACTORS_PATH                   = aen_config.FACTORS_PATH
FACTORS_END_DATE               = aen_config.FACTORS_END_DATE
STRATEGIES                     = aen_config.STRATEGIES
get_strategy_aen_dir           = aen_config.get_strategy_aen_dir
get_aen_output_dir             = aen_config.get_aen_output_dir

# ── Coefficient threshold ──────────────────────────────────────────────────
# Single constant used everywhere: Stage-1 df count, Stage-2 selection,
# stable-factor identification.  Must be >> solver tol (1e-7).
# Sensitivity analysis: see 04b_coef_tol_sensitivity.py.
COEF_TOL = 1e-4


# ============================================================================
# UTILITIES
# ============================================================================

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
    """Daily returns → monthly compounding."""
    daily_df = pd.read_csv(strategy_path, index_col=0, parse_dates=True)
    daily_returns = daily_df['index_return'].dropna()
    monthly = daily_returns.resample('ME').apply(
        lambda x: ((1 + x / 100).prod() - 1) * 100 if len(x) > 0 else np.nan
    )
    return monthly.dropna()


# ============================================================================
# BOOTSTRAP RESAMPLING
# ============================================================================

def circular_block_bootstrap(T, block_length, rng):
    """Circular block bootstrap (Künsch 1989; Politis & Romano 1992c)."""
    block_length = int(block_length)
    if block_length <= 0:
        raise ValueError("block_length must be > 0.")
    n_blocks = int(np.ceil(T / block_length))
    starts = rng.integers(0, T, size=n_blocks)
    indices = []
    for s in starts:
        indices.extend([(s + j) % T for j in range(block_length)])
    return np.array(indices[:T])


def stationary_bootstrap(T, block_length, rng):
    """Stationary bootstrap (Politis & Romano 1994). E[L] = block_length."""
    if block_length < 1:
        raise ValueError(f"block_length must be ≥ 1 (got {block_length}).")
    p_new = 1.0 / block_length
    indices = np.empty(T, dtype=np.intp)
    indices[0] = rng.integers(0, T)
    for t in range(1, T):
        indices[t] = (rng.integers(0, T) if rng.random() < p_new
                      else (indices[t - 1] + 1) % T)
    return indices


def generate_bootstrap_indices(T, block_length, method, rng):
    if method == "circular":
        return circular_block_bootstrap(T, block_length, rng)
    elif method == "stationary":
        return stationary_bootstrap(T, block_length, rng)
    else:
        raise ValueError(f"Unknown BOOTSTRAP_METHOD: '{method}'.")


# ============================================================================
# CLUSTER-LEVEL AGGREGATION  (Faletto & Bien 2022)
# ============================================================================

def build_factor_clusters(X_df, threshold=0.70):
    """
    Cluster factors by |ρ| (hierarchical, complete linkage).
    Returns: clusters (dict, multi-member only), cluster_labels, singletons.
    """
    factor_names = X_df.columns.tolist()
    p = len(factor_names)

    corr = X_df.corr().abs().values.copy()
    np.fill_diagonal(corr, 1.0)
    corr = (corr + corr.T) / 2.0
    dist = np.clip(1.0 - corr, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)

    Z = linkage(squareform(dist, checks=False), method='complete')
    labels = fcluster(Z, t=1.0 - threshold, criterion='distance')

    clusters, singletons = {}, []
    for cid in np.unique(labels):
        members = [factor_names[j] for j in range(p) if labels[j] == cid]
        if len(members) >= 2:
            clusters[int(cid)] = members
        else:
            singletons.append(members[0])

    return clusters, labels, singletons


def compute_cluster_medoids(clusters, X_df):
    """
    Medoid: cluster member with minimum mean distance (1−|ρ|) to others.
    Purely X-based (unsupervised), no y involved.
    """
    abs_corr = X_df.corr().abs()
    medoids = {}
    for cid, members in clusters.items():
        if len(members) == 1:
            medoids[cid] = members[0]
            continue
        sub = abs_corr.loc[members, members].values
        medoids[cid] = members[int(np.argmin((1.0 - sub).mean(axis=1)))]
    return medoids


def compute_cluster_frequencies(sel_2d, factor_names, cluster_labels):
    """
    Cluster-level selection frequencies from (B, p) boolean array.
    Cluster is selected in rep b iff ANY member is selected.

    sel_2d must be the SAME (B, p) slice used for individual_freq
    (i.e. sel_star at λ*), ensuring logical consistency between
    individual and cluster-level stability rules.
    """
    cluster_freq, cluster_members_dict = {}, {}
    for cid in np.unique(cluster_labels):
        idx = np.where(cluster_labels == cid)[0]
        members = [factor_names[j] for j in idx]
        cluster_members_dict[int(cid)] = members
        cluster_freq[int(cid)] = float(sel_2d[:, idx].any(axis=1).mean())
    return cluster_freq, cluster_members_dict


def select_css_stable_factors(cluster_freq, cluster_members_dict,
                               individual_freq, factor_names, threshold,
                               min_gap=0.15, representative_method="max_freq",
                               medoids=None):
    """
    Stable factors: individual rule + vote-splitting correction.

    Step 1: factors with individual_freq ≥ threshold.
    Step 2: for multi-member clusters where
            cluster_freq ≥ threshold AND gap ≥ min_gap,
            add representative if not already selected.
    """
    n2i = {n: i for i, n in enumerate(factor_names)}
    stable_set = {n for j, n in enumerate(factor_names)
                  if individual_freq[j] >= threshold}
    stable_clusters = {}

    for cid, cfreq in cluster_freq.items():
        members = cluster_members_dict[cid]
        midx = [n2i[m] for m in members]
        mfreqs = individual_freq[midx]
        max_ind = float(np.max(mfreqs))
        gap = cfreq - max_ind

        if len(members) == 1:
            if cfreq >= threshold:
                stable_clusters[cid] = {
                    'members': members,
                    'cluster_freq': round(cfreq, 4),
                    'max_individual_freq': round(max_ind, 4),
                    'gap': round(gap, 4),
                    'representative': members[0],
                    'representative_freq': round(max_ind, 4),
                    'selection_reason': 'individual',
                    'is_singleton': True,
                }
            continue

        if representative_method == "max_freq":
            rep = members[int(np.argmax(mfreqs))]
        elif representative_method == "medoid":
            rep = (medoids[cid]
                   if medoids and cid in medoids
                   else members[int(np.argmax(mfreqs))])
        else:
            raise ValueError(f"Unknown representative_method: "
                             f"'{representative_method}'")

        rep_freq = float(individual_freq[n2i[rep]])

        if rep_freq >= threshold:
            reason = 'individual'
        elif cfreq >= threshold and gap >= min_gap:
            stable_set.add(rep)
            reason = 'vote_splitting_correction'
        else:
            reason = None

        if cfreq >= threshold:
            stable_clusters[cid] = {
                'members': members,
                'cluster_freq': round(cfreq, 4),
                'max_individual_freq': round(max_ind, 4),
                'gap': round(gap, 4),
                'representative': rep,
                'representative_freq': round(rep_freq, 4),
                'member_freqs': {m: round(float(individual_freq[n2i[m]]), 4)
                                 for m in members},
                'selection_reason': reason,
                'is_singleton': False,
            }

    return (sorted(stable_set, key=lambda f: -individual_freq[n2i[f]]),
            stable_clusters)


# ============================================================================
# AEN UTILITIES
# ============================================================================

def compute_ic(y_c, y_pred, T, df, criterion, gic_alpha=3.0):
    """IC on centred residuals. df computed externally with COEF_TOL."""
    rss = np.sum((y_c - y_pred) ** 2)
    if rss / T < 1e-15:
        return np.inf, rss
    log_lik = T * np.log(rss / T)
    if criterion in ("BIC", "SIS_BIC"):
        return log_lik + df * np.log(T), rss
    elif criterion == "HQC":
        return log_lik + 2 * np.log(np.log(T)) * df, rss
    elif criterion == "AIC":
        return log_lik + 2 * df, rss
    elif criterion == "AICc":
        aic = log_lik + 2 * df
        return aic + (2 * df * (df + 1)) / max(T - df - 1, 1), rss
    elif criterion == "GIC":
        return log_lik + gic_alpha * df, rss
    elif criterion == "CV":
        return log_lik + df * np.log(T), rss   # BIC fallback
    else:
        raise ValueError(f"Unknown criterion: {criterion}")


def build_lambda1_grid(lambda1_max, n_values, ratio_min=1e-4):
    lambda1_max = max(float(lambda1_max), 1e-12)
    return np.logspace(np.log10(lambda1_max),
                       np.log10(lambda1_max * ratio_min),
                       n_values)


def weighted_elastic_net_cd(X, y_c, lambda1, lambda2, weights,
                             max_iter=10000, tol=1e-7):
    """
    Coordinate descent for:
        (1/2T)||y - Xβ||² + λ1·Σ_j w_j|β_j| + (λ2/2)||β||²

    With uniform weights (w_j = 1 for all j): standard Elastic Net (Stage 1).
    With adaptive weights: Adaptive Elastic Net (Stage 2).

    col_sq[j] = Σ_t x_{tj}² / T = 1/T when ||X_j||_2 = 1 (L2-norm convention).
    The λ1_max formula (2/T)·max|X'y| is consistent with this scaling.
    """
    T, p = X.shape
    beta = np.zeros(p, dtype=np.float64)
    r = y_c.copy().astype(np.float64)
    col_sq = np.sum(X ** 2, axis=0) / T   # = 1/T when ||X_j||_2 = 1

    for _ in range(max_iter):
        max_chg = 0.0
        for j in range(p):
            b_old = beta[j]
            if b_old != 0.0:
                r += X[:, j] * b_old
            rho = np.dot(X[:, j], r) / T
            thr = lambda1 * weights[j]
            num = (rho - thr if rho > thr
                   else rho + thr if rho < -thr
                   else 0.0)
            beta[j] = num / (col_sq[j] + lambda2)
            if beta[j] != 0.0:
                r -= X[:, j] * beta[j]
            chg = abs(beta[j] - b_old)
            if chg > max_chg:
                max_chg = chg
        if max_chg < tol:
            break
    return beta


# ============================================================================
# SCALING-SEPARATED AEN FUNCTIONS
# ============================================================================

def _compute_weights_and_grid(X_prep, y_c, lambda2_grid, n_lambda1,
                               criterion, gamma, gic_alpha,
                               max_iter=10000, tol=1e-7):
    """
    Compute adaptive weights and Stage-2 λ1 grid from already-prepared data.

    ASSUMES:
        X_prep : mean-centred, L2-norm = 1 per column  (output of 01)
        y_c    : mean-centred only  (output of 01)

    Stage-1 uses the SAME CD solver as Stage-2 with uniform weights (w_j = 1),
    eliminating any sklearn parameterisation or rescaling ambiguity.
    No re-standardisation is performed here.

    Returns
    -------
    weights       : (p,) adaptive penalty weights  w_j = (|β̂_j(enet)| + 1/T)^{-γ}
                    computed on rescaled β̂(enet) = (1+λ₂°)·β̂_naive
    lambda2_opt   : optimal λ2 (selected in Stage-1)
    lambda1_opt   : optimal Stage-2 λ1  (= λ* used for primary frequencies)
    lambda1_grid  : Stage-2 λ1 grid  (held fixed across bootstrap replicates)
    lambda1_max   : λ1_max used to build Stage-2 grid
    beta_aen      : Stage-2 AEN coefficients on full sample (for diagnostics)
    selected      : list of selected column indices on full sample
    """
    T, p = X_prep.shape

    # ── Stage 1: CD with uniform weights (standard EN) ─────────────────
    # w_j = 1 for all j: identical to Elastic Net objective.
    # Using the same CD solver as Stage-2 removes all sklearn ambiguity.
    lambda1_max_s1 = max(np.max(np.abs(X_prep.T @ y_c)) / T, 1e-6)
    best_ic1 = np.inf
    best_beta_s1 = None
    best_lambda2_po = None
    w1 = np.ones(p, dtype=float)   # uniform weights for Stage 1

    for lambda2_po in lambda2_grid:
        rescale_s1 = 1.0 + lambda2_po       # eq. 1.4: (1+λ₂°)
        for lam1 in build_lambda1_grid(lambda1_max_s1, n_lambda1):
            beta_naive_s1 = weighted_elastic_net_cd(
                X_prep, y_c, lam1, lambda2_po, w1,
                max_iter=max_iter, tol=tol)
            beta_enet_s1 = beta_naive_s1 * rescale_s1
            df = int(np.sum(np.abs(beta_naive_s1) > COEF_TOL))
            ic, _ = compute_ic(y_c, X_prep @ beta_enet_s1, T, df,
                                criterion, gic_alpha)
            if ic < best_ic1:
                best_ic1 = ic
                best_beta_naive_s1 = beta_naive_s1.copy()
                best_lambda2_po = lambda2_po

    # ── Adaptive weights  (Zou & Zhang 2009, eq. 2.1) ──────────────────
    # Weights on β̂(enet) = (1+λ₂°)·β̂_naive  (paper definition).
    # ε = 1/T guards against division by zero when β̂_j = 0.
    beta_enet_fs = best_beta_naive_s1 * (1.0 + best_lambda2_po)
    weights = (np.abs(beta_enet_fs) + 1.0 / T) ** (-gamma)

    # ── Stage 2: weighted CD (AEN) ─────────────────────────────────────
    lam1_max_s2 = max(
        np.max(np.abs(X_prep.T @ y_c) / T / weights), 1e-6)
    lambda1_grid = build_lambda1_grid(lam1_max_s2, n_lambda1)

    best_ic2 = np.inf
    best_lam1_opt = None
    best_beta_naive_s2 = None
    rescale_s2 = 1.0 + best_lambda2_po     # eq. 2.2: (1+λ₂°)

    for lam1 in lambda1_grid:
        beta_naive = weighted_elastic_net_cd(
            X_prep, y_c, lam1, best_lambda2_po, weights,
            max_iter=max_iter, tol=tol)
        beta_aen = beta_naive * rescale_s2
        df = int(np.sum(np.abs(beta_naive) > COEF_TOL))
        ic, _ = compute_ic(y_c, X_prep @ beta_aen, T, df,
                            criterion, gic_alpha)
        if ic < best_ic2:
            best_ic2 = ic
            best_lam1_opt = lam1
            best_beta_naive_s2 = beta_naive.copy()

    best_beta_aen = best_beta_naive_s2 * rescale_s2
    selected = np.where(np.abs(best_beta_naive_s2) > COEF_TOL)[0].tolist()

    return (weights, best_lambda2_po, best_lam1_opt,
            lambda1_grid, lam1_max_s2, best_beta_aen, selected)


def _aen_boot_cell(y_boot, X_boot, lambda1_po, lambda2_po, weights_fs,
                   max_iter=10000, tol=1e-7):
    """
    Stage-2 weighted AEN on one bootstrap sample at one fixed λ1.

    Re-centres and re-L2-norms INTERNALLY — bootstrap resampling destroys
    the original normalisation from 01.  Adaptive weights (weights_fs) and
    λ2 are fixed from the full sample; only the data changes.

    q_max is NOT applied here.  Model-size control is handled in Step 5
    via restriction of the eligible set Λ_q = {λ : q̄(λ) ≤ q_max},
    which is the formally correct M&B (2010) approach.

    Returns boolean selection mask of length p.
    """
    T, p = X_boot.shape

    # Re-centre y (mean-centred only, consistent with 01)
    y_c = y_boot - np.mean(y_boot)

    # Re-L2-normalise X (bootstrap destroys original L2-norm = 1)
    X_c = X_boot - np.mean(X_boot, axis=0)
    X_l2 = np.sqrt(np.sum(X_c ** 2, axis=0))
    X_l2[X_l2 < 1e-12] = 1.0
    X_s = X_c / X_l2

    beta = weighted_elastic_net_cd(
        X_s, y_c, lambda1_po, lambda2_po, weights_fs,
        max_iter=max_iter, tol=tol)

    mask = np.zeros(p, dtype=bool)
    mask[np.abs(beta) > COEF_TOL] = True
    return mask


# ============================================================================
# MAIN BOOTSTRAP LOOP
# ============================================================================

def run_bootstrap_for_strategy(strategy_name):
    """
    Bootstrap stability selection for one strategy.

    Procedure:
      1. Load X_standardized.parquet and y_centered.parquet from 01.
         Do NOT re-standardise on full sample.
      2. _compute_weights_and_grid: Stage-1 CD (uniform w) → adaptive weights,
         Stage-2 CD → λ1 grid Λ and optimal λ* on full sample.
      3. For each (rep b, λ ∈ Λ): _aen_boot_cell — fixed weights/λ2.
      4. Primary:    Π̂(j) = Π̂(j, λ*)           ≥ π_thr  →  stable
         Robustness: Π̂(j) = max_{λ∈Λ_q} Π̂(j,λ) ≥ π_thr  →  stable
    """
    strategy_dir = get_strategy_aen_dir(strategy_name)

    # ── Load preprocessed data from 01 ────────────────────────────────
    y_df = pd.read_parquet(strategy_dir / "y_centered.parquet")
    X_df = pd.read_parquet(strategy_dir / "X_standardized.parquet")
    y = y_df['y'].values          # already mean-centred
    X = X_df.values               # already mean-centred + L2-normed
    factor_names = X_df.columns.tolist()
    T, n_factors = X.shape

    # Load standardisation params for auditing
    std_params_path = strategy_dir / "standardization_params.json"
    if std_params_path.exists():
        with open(std_params_path) as f:
            std_params = json.load(f)
        std_desc = std_params.get('standardization', 'unknown')
    else:
        std_desc = 'unknown (standardization_params.json not found)'

    with open(strategy_dir / "aen_results.json") as f:
        aen_results = json.load(f)
    full_sample_factors = aen_results['selected_factors']

    criterion = AEN_TUNING_CRITERION
    gic_alpha = GIC_ALPHA if criterion == "GIC" else 3.0

    print(f"\n   Data: T={T}, p={n_factors}")
    print(f"   Standardisation (from 01): {std_desc}")
    print(f"   X passed WITHOUT re-standardisation on full sample.")
    print(f"   Full-sample factors (from 02): {full_sample_factors}")
    print(f"   Criterion: {criterion}")
    print(f"   Bootstrap: {BOOTSTRAP_METHOD} | B={BOOTSTRAP_N_REPS}"
          f" | L={BOOTSTRAP_BLOCK_LENGTH}")
    print(f"   π_thr={STABILITY_THRESHOLD:.0%}"
          f" | rob={STABILITY_THRESHOLD_ROBUSTNESS:.0%}"
          f" | q_max={STABILITY_MAX_Q}")
    print(f"   COEF_TOL={COEF_TOL}  (>> solver tol 1e-7)")

    # ── Step 1: Weights and grid from full-sample prepared data ───────
    # Full grid density (AEN_LAMBDA1_N_VALUES) used in bootstrap: a reduced
    # grid causes IC to jump to wrong penalty regions and distorts λ*.
    print(f"\n   Computing weights and fixed λ1 grid (CD, same as 02)...")

    (fs_weights, fs_lambda2_opt, fs_lambda1_opt,
     lambda1_grid_fs, lambda1_max_fs,
     fs_beta_aen, fs_selected) = _compute_weights_and_grid(
        X, y,
        lambda2_grid=AEN_LAMBDA2_GRID,
        n_lambda1=AEN_LAMBDA1_N_VALUES,
        criterion=criterion,
        gamma=AEN_GAMMA,
        gic_alpha=gic_alpha)

    # Bootstrap grid spans the full Stage-2 range, same density as full-sample
    lambda1_grid_boot = build_lambda1_grid(lambda1_max_fs, AEN_LAMBDA1_N_VALUES)
    n_lambda = len(lambda1_grid_boot)

    # Cross-check: re-derived factors vs stored in aen_results.json
    fs_factor_names = [factor_names[i] for i in fs_selected]
    match = set(fs_factor_names) == set(full_sample_factors)
    status = "✅ match" if match else "⚠️  MISMATCH"
    print(f"   Re-derived factors: {fs_factor_names}  [{status}]")
    if not match:
        print(f"   Stored (02): {full_sample_factors}")
        print(f"   Using re-derived weights for bootstrap consistency.")
        print(f"   Check COEF_TOL consistency between 02 and 04.")

    print(f"   λ2 fixed={fs_lambda2_opt:.3e}"
          f" | λ1* (full-sample)={fs_lambda1_opt:.3e}")
    print(f"   λ1 grid: {n_lambda} values"
          f" [{lambda1_grid_boot[-1]:.2e}, {lambda1_grid_boot[0]:.2e}]")

    # ── Step 2: Bootstrap indices ──────────────────────────────────────
    rng = np.random.default_rng(seed=42)
    all_boot_idx = [
        generate_bootstrap_indices(T, BOOTSTRAP_BLOCK_LENGTH,
                                   BOOTSTRAP_METHOD, rng)
        for _ in range(BOOTSTRAP_N_REPS)
    ]

    # ── Step 3: Parallel (rep, λ) cells ───────────────────────────────
    tasks = [(b, li, all_boot_idx[b], lambda1_grid_boot[li])
             for b in range(BOOTSTRAP_N_REPS)
             for li in range(n_lambda)]

    def _run_cell(b, li, boot_idx, lam1):
        try:
            mask = _aen_boot_cell(
                y[boot_idx], X[boot_idx, :],
                lambda1_po=lam1,
                lambda2_po=fs_lambda2_opt,
                weights_fs=fs_weights)
            return b, li, mask, None
        except Exception as e:
            return b, li, np.zeros(n_factors, dtype=bool), str(e)

    t_start = time.time()
    print(f"\n   Running {BOOTSTRAP_N_REPS}×{n_lambda}"
          f"={len(tasks)} cells (parallel)...")

    cell_results = Parallel(n_jobs=-1, verbose=3)(
        delayed(_run_cell)(b, li, idx, lam)
        for b, li, idx, lam in tasks)

    elapsed = time.time() - t_start
    print(f"\n   ✅ Done: {elapsed:.1f}s"
          f" ({elapsed / len(tasks) * 1000:.1f}ms per cell)")

    # ── Step 4: Reconstruct (B, p, n_lambda) tensor ───────────────────
    sel_3d = np.zeros((BOOTSTRAP_N_REPS, n_factors, n_lambda), dtype=bool)
    boot_errors = []
    for b, li, mask, err in cell_results:
        sel_3d[b, :, li] = mask
        if err is not None:
            boot_errors.append(err)

    n_errors = len(boot_errors)
    if n_errors:
        rate = n_errors / len(tasks)
        print(f"\n   ⚠️  Errors: {n_errors}/{len(tasks)} ({rate:.1%})")
        for e in list(set(boot_errors))[:5]:
            print(f"      {e[:100]}")
        if rate > 0.05:
            print(f"   🔴 Error rate > 5%: results may be unreliable.")

    # ── Step 5: Frequency matrix Π̂(j, λ) ─────────────────────────────
    freq_matrix = sel_3d.mean(axis=0)               # (p, n_lambda)

    # Mean model size per λ, averaged over bootstrap replicates
    model_size_per_lambda = sel_3d.sum(axis=1).mean(axis=0)  # (n_lambda,)

    # Eligible Λ_q: restrict to λ with q̄(λ) ≤ q_max  [M&B (2010) Thm 1]
    if STABILITY_MAX_Q is not None:
        eligible = model_size_per_lambda <= STABILITY_MAX_Q
        if not np.any(eligible):
            # Fallback: most regularised λ (smallest mean model size)
            j_best = int(np.argmin(model_size_per_lambda))
            eligible = np.zeros(n_lambda, dtype=bool)
            eligible[j_best] = True
            print(f"   ⚠ No λ satisfies q̄≤{STABILITY_MAX_Q}. "
                  f"Falling back to most regularised: "
                  f"λ1={lambda1_grid_boot[j_best]:.3e}, "
                  f"q̄={model_size_per_lambda[j_best]:.2f}")
        else:
            print(f"   Eligible Λ_q (q̄≤{STABILITY_MAX_Q}): "
                  f"{int(eligible.sum())}/{n_lambda} λ values "
                  f"(q̄ range [{model_size_per_lambda[eligible].min():.1f}, "
                  f"{model_size_per_lambda[eligible].max():.1f}])")
    else:
        eligible = np.ones(n_lambda, dtype=bool)

    elig_idx = np.where(eligible)[0]   # integer indices of eligible λ values

    # ── Primary: Π̂(j, λ*) ─────────────────────────────────────────────
    # Find the bootstrap-grid λ closest to fs_lambda1_opt within Λ_q.
    j_star_local = int(np.argmin(
        np.abs(lambda1_grid_boot[elig_idx] - fs_lambda1_opt)))
    j_star = elig_idx[j_star_local]
    lam_star_used = lambda1_grid_boot[j_star]
    print(f"   λ* for primary frequencies: {lam_star_used:.3e}"
          f"  (full-sample optimal: {fs_lambda1_opt:.3e})")

    sel_star = sel_3d[:, :, j_star].astype(bool)    # (B, p) at λ*
    individual_freq = sel_star.mean(axis=0)          # (p,)  PRIMARY

    # ── Robustness: max_{λ∈Λ_q} Π̂(j, λ)  (M&B 2010 original rule) ───
    freq_matrix_elig = freq_matrix[:, elig_idx]      # (p, n_elig)
    individual_freq_maxlam = freq_matrix_elig.max(axis=1)   # (p,)
    lambda_at_max = lambda1_grid_boot[
        elig_idx[np.argmax(freq_matrix_elig, axis=1)]]

    # Model-size diagnostics at λ*
    n_sel_per_rep = sel_star.sum(axis=1)             # (B,)
    n_sel_mean = float(n_sel_per_rep.mean())

    # q̂ = max q̄(λ) over Λ_q — used in formal M&B E(V) bound
    q_hat = float(model_size_per_lambda[eligible].max())

    # ── Step 6: CSS clustering ─────────────────────────────────────────
    # IMPORTANT: cluster_freq uses sel_star (same base as individual_freq) to
    # ensure the cluster-level rule is logically consistent with the individual
    # rule. Using a different (B, p) slice would break the consistency.
    css_info = {}
    clusters, cluster_labels = {}, np.arange(n_factors, dtype=int)
    singletons = list(factor_names)
    cluster_medoids = {}
    stable_factors_css_medoid = []
    stable_clusters = {}
    cluster_freq = {}
    cluster_members_dict = {}

    if CSS_ENABLED:
        clusters, cluster_labels, singletons = build_factor_clusters(
            X_df, threshold=CSS_CORRELATION_THRESHOLD)

        for cid, mems in clusters.items():
            if len(mems) > 5:
                print(f"\n   ⚠️  Large cluster [{cid}]: {len(mems)} members"
                      f" at |ρ|≥{CSS_CORRELATION_THRESHOLD}")

        cluster_medoids = compute_cluster_medoids(clusters, X_df)

        # Use sel_star at λ* — consistent with individual_freq
        cluster_freq, cluster_members_dict = compute_cluster_frequencies(
            sel_star, factor_names, cluster_labels)

        # Primary selection (max-freq representative)
        stable_factors_css, stable_clusters = select_css_stable_factors(
            cluster_freq, cluster_members_dict, individual_freq,
            factor_names, threshold=STABILITY_THRESHOLD,
            min_gap=CSS_MIN_GAP,
            representative_method=CSS_REPRESENTATIVE_METHOD)

        # Medoid selection (robustness)
        stable_factors_css_medoid, _ = select_css_stable_factors(
            cluster_freq, cluster_members_dict, individual_freq,
            factor_names, threshold=STABILITY_THRESHOLD,
            min_gap=CSS_MIN_GAP,
            representative_method="medoid",
            medoids=cluster_medoids)

        # Higher threshold robustness
        stable_factors_css_rob, _ = select_css_stable_factors(
            cluster_freq, cluster_members_dict, individual_freq,
            factor_names, threshold=STABILITY_THRESHOLD_ROBUSTNESS,
            min_gap=CSS_MIN_GAP,
            representative_method=CSS_REPRESENTATIVE_METHOD)

        # Annotate medoid in stable_clusters info
        for cid, info in stable_clusters.items():
            if not info['is_singleton'] and cid in cluster_medoids:
                info['medoid'] = cluster_medoids[cid]
                info['medoid_matches_representative'] = (
                    cluster_medoids[cid] == info['representative'])

        # Warn on low-freq vote-split representative
        for cid, info in stable_clusters.items():
            if (not info['is_singleton']
                    and info.get('selection_reason') == 'vote_splitting_correction'
                    and info['representative_freq'] < 0.40):
                print(f"\n   ⚠️  Cluster [{cid}]: vote-split adds"
                      f" {info['representative']}"
                      f" (ind={info['representative_freq']:.0%},"
                      f" cluster={info['cluster_freq']:.0%})."
                      f" Verify economic plausibility.")

        stable_factors     = stable_factors_css
        stable_factors_rob = stable_factors_css_rob

        css_info = {
            'enabled': True,
            'correlation_threshold': CSS_CORRELATION_THRESHOLD,
            'min_gap': CSS_MIN_GAP,
            'representative_method': CSS_REPRESENTATIVE_METHOD,
            'n_clusters': int(len(np.unique(cluster_labels))),
            'n_multi_member': len(clusters),
            'n_singletons': len(singletons),
            'multi_member_clusters': {
                str(k): v for k, v in clusters.items()},
            'cluster_medoids': {
                str(k): v for k, v in cluster_medoids.items()},
            'cluster_frequencies': {
                str(k): round(v, 4)
                for k, v in sorted(cluster_freq.items(),
                                   key=lambda x: -x[1])},
            'stable_clusters': {
                str(k): v for k, v in stable_clusters.items()},
        }
    else:
        stable_factors = [factor_names[j] for j in range(n_factors)
                          if individual_freq[j] >= STABILITY_THRESHOLD]
        stable_factors_rob = [factor_names[j] for j in range(n_factors)
                               if individual_freq[j]
                               >= STABILITY_THRESHOLD_ROBUSTNESS]

    # ── Frequency DataFrame ────────────────────────────────────────────
    freq_df = pd.DataFrame({
        'factor': factor_names,
        'selection_frequency':        individual_freq,          # at λ* [primary]
        'selection_frequency_maxlam': individual_freq_maxlam,   # max_λ [robustness]
        'lambda_at_max': lambda_at_max,
        'stable_individual': individual_freq >= STABILITY_THRESHOLD,
        'in_full_sample': [f in full_sample_factors for f in factor_names],
    }).sort_values('selection_frequency', ascending=False)

    if CSS_ENABLED:
        n2c = {factor_names[j]: int(cluster_labels[j]) for j in range(n_factors)}
        cfreq_map = {factor_names[j]: cluster_freq.get(
            int(cluster_labels[j]), 0.0) for j in range(n_factors)}
        freq_df['cluster_id']     = freq_df['factor'].map(n2c)
        freq_df['cluster_freq']   = freq_df['factor'].map(cfreq_map)
        freq_df['stable_css']     = freq_df['factor'].isin(stable_factors)
        freq_df['stable_css_rob'] = freq_df['factor'].isin(stable_factors_rob)
        freq_df['stable_medoid']  = freq_df['factor'].isin(
            stable_factors_css_medoid)

    # ── PART A: Diagnostics ────────────────────────────────────────────
    print_header(f"   PART A: Selection Frequencies — {strategy_name}", "─")
    print(f"\n   Framework: Fixed-grid M&B (2010), Stage-1/2 via CD (no sklearn)")
    print(f"   IC NOT re-optimised in bootstrap | λ1 grid: {n_lambda} values"
          f" fixed from full sample")
    print(f"   λ2 fixed={fs_lambda2_opt:.3e} | weights from full-sample Stage-1 CD")
    print(f"   PRIMARY:    Π̂(j) = Π̂(j, λ*={lam_star_used:.3e}) ≥"
          f" π_thr={STABILITY_THRESHOLD:.0%}")
    print(f"   ROBUSTNESS: Π̂(j) = max_{{λ∈Λ_q}} Π̂(j,λ) ≥"
          f" π_thr={STABILITY_THRESHOLD:.0%}")
    print(f"   COEF_TOL={COEF_TOL}")

    print(f"\n   Model size at λ*:"
          f" mean={n_sel_mean:.1f},"
          f" median={np.median(n_sel_per_rep):.0f},"
          f" range=[{n_sel_per_rep.min()},{n_sel_per_rep.max()}]")
    print(f"   Null model (at λ*): {(n_sel_per_rep==0).sum()}/{BOOTSTRAP_N_REPS}"
          f" ({(n_sel_per_rep==0).mean():.1%})")

    if STABILITY_MAX_Q is not None:
        ev_m = q_hat**2 / (n_factors * (2*STABILITY_THRESHOLD - 1))
        ev_r = q_hat**2 / (n_factors * (2*STABILITY_THRESHOLD_ROBUSTNESS - 1))
        print(f"\n   M&B E(V) bound (formal — Λ_q restricted by q_max={STABILITY_MAX_Q}):")
        print(f"     q̂={q_hat:.2f}: E(V)≤{ev_m:.2f} (π={STABILITY_THRESHOLD:.2f}),"
              f" E(V)≤{ev_r:.2f} (π={STABILITY_THRESHOLD_ROBUSTNESS:.2f})")
    else:
        ev_ind = n_sel_mean**2 / (n_factors * (2*STABILITY_THRESHOLD - 1))
        print(f"\n   M&B E(V) — indicative only (q_max not set):")
        print(f"     q̄(λ*)={n_sel_mean:.2f}: E(V)≤{ev_ind:.2f}"
              f" (π={STABILITY_THRESHOLD:.2f})")
        print(f"     ⚠  Set STABILITY_MAX_Q for a formal bound.")

    if CSS_ENABLED:
        print(f"\n   ── Clusters (|ρ|≥{CSS_CORRELATION_THRESHOLD}) ──")
        n_uniq = len(np.unique(cluster_labels))
        print(f"   {n_uniq} total: {len(clusters)} multi-member,"
              f" {len(singletons)} singletons")
        n2i = {n: i for i, n in enumerate(factor_names)}
        for cid, mems in sorted(clusters.items()):
            cf = cluster_freq.get(cid, 0.0)
            mi = float(np.max(individual_freq[[n2i[m] for m in mems]]))
            gap = cf - mi
            tag = ("✅ vote-split"
                   if cf >= STABILITY_THRESHOLD and gap >= CSS_MIN_GAP
                   else "── stable" if cf >= STABILITY_THRESHOLD
                   else "   ")
            print(f"     [{cid}] freq={cf:.0%} max_ind={mi:.0%}"
                  f" gap={gap:+.0%}  {tag}")
            print(f"           {', '.join(mems)}")

    print(f"\n   Top factors  [Π̂(λ*) = primary | Π̂(max) = robustness]:")
    print(f"   {'Factor':<25} {'Π̂(λ*)':>7} {'Π̂(max)':>7}"
          + (f" {'CluFreq':>7} {'CSS':>4}" if CSS_ENABLED else f" {'Stable':>6}")
          + f" {'Full':>5}")
    print(f"   {'─' * 63}")
    for _, row in freq_df.head(25).iterrows():
        if row['selection_frequency'] < 0.01 and not row['in_full_sample']:
            continue
        s = (f"   {row['factor']:<25}"
             f" {row['selection_frequency']:>6.1%}"
             f" {row['selection_frequency_maxlam']:>6.1%}")
        if CSS_ENABLED:
            s += (f" {row.get('cluster_freq', 0):>6.1%}"
                  f" {'✅' if row.get('stable_css') else '  ':>3}")
        else:
            s += f" {'✅' if row['stable_individual'] else '  ':>5}"
        s += f" {'●' if row['in_full_sample'] else ' ':>4}"
        print(s)

    # ── PART B: Stable factors ─────────────────────────────────────────
    print_header(f"   PART B: Stable Factors — {strategy_name}", "─")

    overlap   = set(stable_factors) & set(full_sample_factors)
    only_full = set(full_sample_factors) - set(stable_factors)
    only_boot = set(stable_factors) - set(full_sample_factors)

    print(f"\n   Primary (π={STABILITY_THRESHOLD:.0%},"
          f" {CSS_REPRESENTATIVE_METHOD}, at λ*): {stable_factors or '(none)'}")
    print(f"   Robustness (π={STABILITY_THRESHOLD_ROBUSTNESS:.0%}):"
          f" {stable_factors_rob or '(none)'}")
    if CSS_ENABLED:
        print(f"   Medoid (π={STABILITY_THRESHOLD:.0%}, X-only):"
              f" {stable_factors_css_medoid or '(none)'}")
    print(f"   Full-sample (02): {full_sample_factors}")
    print(f"   Overlap: {sorted(overlap) or '(none)'}")
    if only_full:
        print(f"   ⚠  In full-sample but NOT stable: {sorted(only_full)}")
    if only_boot:
        print(f"   ⚠  Stable but NOT in full-sample: {sorted(only_boot)}")

    for cid, info in stable_clusters.items():
        if info['is_singleton']:
            print(f"     [{cid}] (singleton) {info['representative']}"
                  f" Π̂={info['cluster_freq']:.1%}")
        else:
            tag = (" ← VOTE-SPLIT"
                   if info.get('selection_reason') == 'vote_splitting_correction'
                   else "")
            print(f"     [{cid}] {info['members']}")
            print(f"       cluster={info['cluster_freq']:.1%}"
                  f"  max_ind={info['max_individual_freq']:.1%}"
                  f"  gap={info['gap']:+.1%}")
            print(f"       repr={info['representative']}"
                  f" (Π̂={info['representative_freq']:.1%}){tag}")
            med = info.get('medoid')
            if med:
                print(f"       medoid={med}"
                      f" ({'=repr' if med == info['representative'] else '≠repr'})")

    # ── PART C: OLS ────────────────────────────────────────────────────
    print_header(f"   PART C: Final OLS — {strategy_name}", "─")

    ols_stable_result = None
    ols_medoid_result = None

    if not stable_factors:
        print(f"\n   ⚠️  No stable factors. Skipping OLS (primary).")
    else:
        ols_stable_result = run_ols_on_stable_factors(
            strategy_name, stable_factors, label="stable")

    if CSS_ENABLED:
        if not stable_factors_css_medoid:
            print(f"\n   ⚠️  No medoid stable factors. Skipping OLS (medoid).")
        else:
            if stable_factors_css_medoid == stable_factors:
                print(f"\n   ℹ️  Medoid set = primary set.")
            print(f"\n   ── OLS: medoid set ──")
            ols_medoid_result = run_ols_on_stable_factors(
                strategy_name, stable_factors_css_medoid, label="medoid")

    # ── Save ───────────────────────────────────────────────────────────
    freq_df.to_csv(strategy_dir / "bootstrap_frequencies.csv", index=False)

    # Full Π̂(j, λ) matrix: rows=factors, cols=lambda values
    pd.DataFrame(
        freq_matrix,
        index=factor_names,
        columns=[f"lam_{i}" for i in range(n_lambda)]
    ).to_csv(strategy_dir / "bootstrap_freq_matrix.csv")

    if STABILITY_MAX_Q is not None:
        mb_bound = {
            'valid': True,
            'q_max': STABILITY_MAX_Q,
            'q_hat': round(q_hat, 4),   # max q̄(λ) over Λ_q (empirical)
            'n_factors': n_factors,
            f'EV_pi_{STABILITY_THRESHOLD:.2f}': round(
                q_hat**2 / (n_factors*(2*STABILITY_THRESHOLD-1)), 4),
            f'EV_pi_{STABILITY_THRESHOLD_ROBUSTNESS:.2f}': round(
                q_hat**2 / (n_factors*(2*STABILITY_THRESHOLD_ROBUSTNESS-1)), 4),
            'note': ('E(V) ≤ q̂²/[p·(2π-1)], M&B(2010) Thm1. '
                     'q̂ = max_{λ∈Λ_q} q̄(λ). '
                     'Conditions satisfied: fixed grid + Λ_q restriction.'),
        }
    else:
        mb_bound = {
            'valid': False,
            'q_max': None,
            'q_bar_at_lambda_star': round(n_sel_mean, 4),
            f'EV_indicative_pi_{STABILITY_THRESHOLD:.2f}': round(
                n_sel_mean**2 / (n_factors*(2*STABILITY_THRESHOLD-1)), 4),
            'note': 'Indicative only. Set STABILITY_MAX_Q for formal bound.',
        }

    boot_results = {
        'strategy': strategy_name,
        'tuning_criterion': AEN_TUNING_CRITERION,
        'methodology': {
            'framework': ('Fixed-grid M&B (2010) adapted for TS; '
                          'no IC in bootstrap loop'),
            'stage1': 'CD with uniform weights (no sklearn)',
            'stage2': 'CD with adaptive weights (AEN)',
            'scaling': {
                'source': 'X_standardized.parquet from 01_preprocessing',
                'description': std_desc,
                'restandarised_in_bootstrap': True,
                'restandarised_on_fullsample': False,
            },
            'coef_tol': COEF_TOL,
            'y_treatment': 'mean-centred only (no variance scaling)',
            'bootstrap_method': BOOTSTRAP_METHOD,
            'n_reps': BOOTSTRAP_N_REPS,
            'block_length': BOOTSTRAP_BLOCK_LENGTH,
            'n_lambda': n_lambda,
            'lambda1_range': [round(float(lambda1_grid_boot[-1]), 8),
                               round(float(lambda1_grid_boot[0]), 8)],
            'lambda2_fixed': round(float(fs_lambda2_opt), 8),
            'lambda1_star': round(float(lam_star_used), 8),
            'q_max': STABILITY_MAX_Q,
            'pi_thr': STABILITY_THRESHOLD,
            'pi_thr_robustness': STABILITY_THRESHOLD_ROBUSTNESS,
            'primary_rule':    'Π̂(j, λ*) ≥ π_thr',
            'robustness_rule': 'max_{λ∈Λ_q} Π̂(j,λ) ≥ π_thr',
            'css_enabled': CSS_ENABLED,
            'css_base': 'sel_star at λ* (consistent with individual_freq)',
            'references': [
                'Meinshausen & Bühlmann (2010, JRSS-B)',
                'Politis & Romano (1994, JASA)',
                'Bühlmann (2002, Statistical Science)',
                'Faletto & Bien (2022, arXiv:2201.00494)',
            ],
        },
        'full_sample_factors': full_sample_factors,
        'stable_factors': stable_factors,
        'stable_factors_robustness': stable_factors_rob,
        'stable_factors_medoid': stable_factors_css_medoid,
        'n_stable': len(stable_factors),
        'n_stable_robustness': len(stable_factors_rob),
        'n_stable_medoid': len(stable_factors_css_medoid),
        'overlap_with_full_sample': sorted(overlap),
        'model_size_at_lambda_star_mean':   round(n_sel_mean, 2),
        'model_size_at_lambda_star_median': int(np.median(n_sel_per_rep)),
        'null_model_frequency': round(float((n_sel_per_rep==0).mean()), 4),
        'bootstrap_errors': n_errors,
        'mb_bound': mb_bound,
        'factor_frequencies': {
            row['factor']: {
                'pi_lambda_star': round(float(row['selection_frequency']), 4),
                'pi_max_lambda':  round(float(row['selection_frequency_maxlam']), 4),
            }
            for _, row in freq_df.iterrows()
            if row['selection_frequency'] >= 0.01
               or row['selection_frequency_maxlam'] >= 0.01},
        'css': css_info if CSS_ENABLED else {'enabled': False},
        'ols_stable': ols_stable_result,
        'ols_stable_medoid': ols_medoid_result,
        'elapsed_seconds': round(time.time() - t_start, 1),
    }

    with open(strategy_dir / "bootstrap_stability.json", 'w') as f:
        json.dump(boot_results, f, indent=2)

    print(f"\n   💾 bootstrap_stability.json")
    print(f"   💾 bootstrap_frequencies.csv"
          f"  [selection_frequency=λ* (primary), "
          f"selection_frequency_maxlam=max_λ (robustness)]")
    print(f"   💾 bootstrap_freq_matrix.csv  [Π̂(j,λ) — used by 04b]")

    return boot_results


# ============================================================================
# OLS ON STABLE FACTORS
# ============================================================================

def run_ols_on_stable_factors(strategy_name, stable_factors, label="stable"):
    """
    Post-selection OLS on RAW (un-standardised) data.
    HAC (Newey-West) SE.  label controls output filenames.
    """
    strategy_dir  = get_strategy_aen_dir(strategy_name)
    strategy_path = STRATEGIES[strategy_name]

    all_factors = pd.read_parquet(FACTORS_PATH)
    factors_end = pd.Timestamp(FACTORS_END_DATE)
    all_factors = all_factors[all_factors.index <= factors_end]

    returns = load_strategy_returns(strategy_path)
    returns = returns[returns.index <= factors_end]

    common = returns.index.intersection(all_factors.index)
    y = returns.loc[common]
    X = all_factors.loc[common][stable_factors].copy()
    mask = ~(X.isna().any(axis=1) | y.isna())
    y, X = y[mask], X[mask]
    T = len(y)

    print(f"\n   [{label}] factors: {stable_factors}")
    print(f"   T={T}, k={len(stable_factors)}")

    X_c = sm.add_constant(X, prepend=True)
    res_ols = sm.OLS(y, X_c).fit()
    res_hac = sm.OLS(y, X_c).fit(
        cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

    alpha_c = res_hac.params['const']
    alpha_se = res_hac.bse['const']
    alpha_t  = res_hac.tvalues['const']
    alpha_p  = res_hac.pvalues['const']

    print(f"\n   ── OLS [{label}] (HAC NW {HAC_LAGS} lags) ──")
    print(f"\n   {'Variable':<25} {'Coeff':>10} {'HAC SE':>10}"
          f" {'t-stat':>10} {'p-val':>9}")
    print(f"   {'─' * 67}")
    rows = []
    for var in X_c.columns:
        c  = res_hac.params[var]
        se = res_hac.bse[var]
        t  = res_hac.tvalues[var]
        p  = res_hac.pvalues[var]
        lbl = "α (intercept)" if var == 'const' else var
        print(f"   {lbl:<25} {c:>+10.4f} {se:>10.4f}"
              f" {t:>10.3f} {p:>9.4f} {significance_stars(p)}")
        rows.append({'variable': lbl,
                     'coefficient':  round(float(c), 6),
                     'hac_se':       round(float(se), 6),
                     't_statistic':  round(float(t), 4),
                     'p_value':      round(float(p), 4),
                     'significance': significance_stars(p)})
    print(f"   {'─' * 67}")
    print(f"   R²={res_ols.rsquared:.4f} | R²adj={res_ols.rsquared_adj:.4f}"
          f" | DW={sm.stats.durbin_watson(res_ols.resid):.4f}")
    print(f"\n   α={alpha_c:+.4f}%/mo ({alpha_c*12:+.2f}%/yr)"
          f" | t={alpha_t:.3f} | p={alpha_p:.4f}"
          f" {significance_stars(alpha_p)}")
    if alpha_p < 0.05:   print(f"   → Significant at 5% ✅")
    elif alpha_p < 0.10: print(f"   → Significant at 10% (marginal)")
    else:                print(f"   → Not statistically significant")

    pd.DataFrame(rows).to_csv(
        strategy_dir / f"ols_{label}_table.csv", index=False)

    result = {
        'selection_label': label,
        'stable_factors':  stable_factors,
        'n_factors': len(stable_factors),
        'T': T,
        'alpha': {
            'coefficient':       round(float(alpha_c), 6),
            'hac_se':            round(float(alpha_se), 6),
            't_statistic':       round(float(alpha_t), 4),
            'p_value':           round(float(alpha_p), 4),
            'annualized_pct':    round(float(alpha_c * 12), 4),
            'significant_5pct':  bool(alpha_p < 0.05),
            'significant_10pct': bool(alpha_p < 0.10),
        },
        'factors': {
            v: {'coefficient': round(float(res_hac.params[v]), 6),
                'hac_se':      round(float(res_hac.bse[v]), 6),
                't_statistic': round(float(res_hac.tvalues[v]), 4),
                'p_value':     round(float(res_hac.pvalues[v]), 4)}
            for v in stable_factors
        },
        'r_squared':     round(float(res_ols.rsquared), 6),
        'r_squared_adj': round(float(res_ols.rsquared_adj), 6),
        'durbin_watson': round(
            float(sm.stats.durbin_watson(res_ols.resid)), 4),
    }

    with open(strategy_dir / f"ols_{label}_results.json", 'w') as f:
        json.dump(result, f, indent=2)

    print(f"   💾 ols_{label}_results.json | ols_{label}_table.csv")
    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header(
        "BOOTSTRAP STABILITY SELECTION\n"
        "   Fixed-grid M&B (2010) · Stationary bootstrap (PR 1994)\n"
        "   CSS vote-splitting correction (Faletto & Bien 2022)\n"
        "   Stage-1/2: CD solver (no sklearn, no rescaling)\n"
        "   Primary: Π̂(j, λ*)  |  Robustness: max_{λ∈Λ_q} Π̂(j, λ)"
    )
    print(f"\n   Criterion (full sample): {AEN_TUNING_CRITERION}")
    print(f"   Bootstrap: {BOOTSTRAP_METHOD}"
          f" | B={BOOTSTRAP_N_REPS} | L={BOOTSTRAP_BLOCK_LENGTH}")
    print(f"   π_thr={STABILITY_THRESHOLD:.0%}"
          f" | rob={STABILITY_THRESHOLD_ROBUSTNESS:.0%}"
          f" | q_max={STABILITY_MAX_Q}")
    print(f"   COEF_TOL={COEF_TOL}"
          f"  (sensitivity: run 04b_coef_tol_sensitivity.py)")
    print(f"   CSS: {'ENABLED' if CSS_ENABLED else 'disabled'}"
          f"  (|ρ|≥{CSS_CORRELATION_THRESHOLD}, gap≥{CSS_MIN_GAP:.0%})")
    print(f"   HAC lags: {HAC_LAGS}")

    all_results = {}

    for strategy_name in STRATEGIES:
        print_header(f"STRATEGY: {strategy_name}")
        strategy_dir = get_strategy_aen_dir(strategy_name)

        if not (strategy_dir / "y_centered.parquet").exists():
            print(f"\n   ❌ Preprocessed data not found. Run 01 first.")
            continue
        if not (strategy_dir / "aen_results.json").exists():
            print(f"\n   ❌ AEN results not found. Run 02 first.")
            continue

        try:
            result = run_bootstrap_for_strategy(strategy_name)
            all_results[strategy_name] = result
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        return

    # ── Cross-strategy summary ─────────────────────────────────────────
    print_header("FINAL SUMMARY")

    def _print_summary(results, ols_key, label, n_key):
        print(f"\n   ── {label} ──")
        print(f"\n   {'Strategy':<20} {'k':>4} {'α(mo)':>8} {'α(yr)':>8}"
              f" {'t':>7} {'p':>8} {'R²adj':>7}")
        print(f"   {'─' * 66}")
        for name, res in results.items():
            ols = res.get(ols_key)
            k   = res.get(n_key, 0)
            if ols:
                a = ols['alpha']
                print(f"   {name:<20} {k:>4}"
                      f" {a['coefficient']:>+8.4f}"
                      f" {a['annualized_pct']:>+8.2f}"
                      f" {a['t_statistic']:>7.3f}"
                      f" {a['p_value']:>8.4f}"
                      f" {ols['r_squared_adj']:>7.3f}"
                      f" {significance_stars(a['p_value'])}")
            else:
                print(f"   {name:<20} {k:>4} {'(no stable factors)':>48}")

    _print_summary(all_results, 'ols_stable',
                   'primary selection (λ*)', 'n_stable')
    if CSS_ENABLED:
        _print_summary(all_results, 'ols_stable_medoid',
                       'medoid selection (robustness)', 'n_stable_medoid')

    # Save global summary
    aen_output_dir = get_aen_output_dir()
    summary = {
        'tuning_criterion': AEN_TUNING_CRITERION,
        'bootstrap_config': {
            'framework': ('Fixed-grid M&B (2010), Stage-1/2 via CD (no sklearn), '
                          'primary Π̂(j,λ*), robustness max_{λ∈Λ_q} Π̂(j,λ)'),
            'method': BOOTSTRAP_METHOD,
            'n_reps': BOOTSTRAP_N_REPS,
            'block_length': BOOTSTRAP_BLOCK_LENGTH,
            'coef_tol': COEF_TOL,
            'threshold': STABILITY_THRESHOLD,
            'threshold_robustness': STABILITY_THRESHOLD_ROBUSTNESS,
            'q_max': STABILITY_MAX_Q,
            'css_enabled': CSS_ENABLED,
            'references': [
                'Meinshausen & Bühlmann (2010, JRSS-B)',
                'Politis & Romano (1994, JASA)',
                'Bühlmann (2002, Statistical Science)',
                'Faletto & Bien (2022, arXiv:2201.00494)',
            ],
        },
        'strategies': {name: {
            'full_sample':       res['full_sample_factors'],
            'stable':            res['stable_factors'],
            'stable_robustness': res['stable_factors_robustness'],
            'stable_medoid':     res['stable_factors_medoid'],
            'overlap':           res['overlap_with_full_sample'],
            'mb_bound':          res['mb_bound'],
            'alpha_monthly':     (res['ols_stable']['alpha']['coefficient']
                                  if res.get('ols_stable') else None),
            'alpha_annualized':  (res['ols_stable']['alpha']['annualized_pct']
                                  if res.get('ols_stable') else None),
            'alpha_pval_hac':    (res['ols_stable']['alpha']['p_value']
                                  if res.get('ols_stable') else None),
            'alpha_monthly_medoid':    (
                res['ols_stable_medoid']['alpha']['coefficient']
                if res.get('ols_stable_medoid') else None),
            'alpha_annualized_medoid': (
                res['ols_stable_medoid']['alpha']['annualized_pct']
                if res.get('ols_stable_medoid') else None),
            'alpha_pval_hac_medoid': (
                res['ols_stable_medoid']['alpha']['p_value']
                if res.get('ols_stable_medoid') else None),
        } for name, res in all_results.items()}
    }
    with open(aen_output_dir / "bootstrap_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n   💾 {aen_output_dir / 'bootstrap_summary.json'}")

    print(f"\n{'=' * 80}")
    print(f"✅ BOOTSTRAP STABILITY SELECTION COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n   🎯 Sensitivity analysis: python src/aen/04b_coef_tol_sensitivity.py")
    print(f"   🎯 Method comparison:    python src/aen/05_aen_method_comparison.py")


if __name__ == "__main__":
    main()
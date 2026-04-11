"""
================================================================================
00_aen_config.py - Configuration for Adaptive Elastic-Net Pipeline
================================================================================
Centralized parameters for the AEN factor-selection pipeline.

References:
    Zou, H. and Zhang, H.H. (2009),
        "On the Adaptive Elastic-Net with a Diverging Number of Parameters",
        Annals of Statistics, 37(4), 1733-1751.
    Chen, J. and Chen, Z. (2008),
        "Extended Bayesian Information Criteria for Model Selection with
        Large Model Spaces", Biometrika, 95(3), 759-771.

Three configurations supported (set AEN_TUNING_CRITERION):
    "BIC"      — classical BIC, paper-faithful baseline
    "HQC"      — Hannan-Quinn (penalty = 2·log(log(T))·df)
    "AICc"     — Small-sample corrected AIC (Hurvich-Tsai 1989)
    "GIC"      — Generalized IC (custom penalty α·df)
    "SIS_BIC"  — Sure Independence Screening (paper Section 5) + BIC

Author:      Alessio Ottaviani
Institution: EDHEC Business School – PhD Thesis
================================================================================
"""

from pathlib import Path
import numpy as np

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================================
# INPUT FILES
# ============================================================================

FACTORS_PATH = DATA_DIR / "processed" / "all_factors_monthly.parquet"

STRATEGIES = {
    "btp_italia":      RESULTS_DIR / "btp_italia"      / "index_daily.csv",
    "cds_bond_basis":  RESULTS_DIR / "cds_bond_basis"  / "index_daily.csv",
    "itraxx_combined": RESULTS_DIR / "itraxx_combined" / "index_daily.csv",
}

FACTORS_END_DATE = "2025-05-31"

# ============================================================================
# ⭐ TUNING CRITERION — main switch
# ============================================================================
# Choose ONE of: "BIC", "HQC", "AIC", "AICc", "GIC", "SIS_BIC", "CV"
#
# Penalità per fattore (T = 158):
#   BIC:       log(T)              = 5.06
#   HQC:       2·log(log(T))       = 3.24   (Hannan-Quinn)
#   AIC:       2                   = 2.00   (Akaike)
#   AICc:      2 + correction      ≈ 2.2    (small-sample AIC)
#   GIC(α=3):  α                   = 3.00   (custom penalty)
#   SIS_BIC:   log(T) on reduced p = 5.06   (after pre-screening)
#   CV:        Rolling-origin CV   = data-driven (no penalty term)
#
# Note: AIC (penalty=2) selects denser models than HQC and BIC.
#       AICc dominates AIC for small T; AIC included for completeness.
#       CV uses rolling-origin (expanding window) to preserve temporal
#       dependence; optimizes MSPE, not parsimony.

AEN_TUNING_CRITERION = "HQC"

# ============================================================================
# ⭐ CORRELATION PRE-CLEANING
# ============================================================================

CORRELATION_THRESHOLD = 0.95

FACTORS_TO_EXCLUDE: list[str] = [
'R2_EU', 'GLOBAL_TERM', 'Δ10Y_YIELD_EU', 
]

# ============================================================================
# ⭐ SIS — Sure Independence Screening (paper Section 5)
# ============================================================================
# Only active when AEN_TUNING_CRITERION = "SIS_BIC".
# Reduces p to d_n before running AEN.
#
# Paper formula: d_n = ⌊T / log(T)⌋
# Keeps the d_n factors with highest |corr(Xⱼ, y)| (marginal screening).
#
# Set SIS_D_N_RULE to "paper" to use ⌊T/log(T)⌋, or to an integer for a
# manual override.

SIS_D_N_RULE = "paper"                # "paper" or an integer (e.g. 25)

# ============================================================================
# ⭐ ADAPTIVE ELASTIC-NET PARAMETERS (Zou & Zhang 2009)
# ============================================================================

AEN_GAMMA = 1                          # adaptive weight exponent
AEN_WEIGHT_EPSILON_RULE = "1/n"        # zero-avoidance (paper p. 1736)
AEN_SAME_LAMBDA2 = True                # paper p. 1737
COEF_TOL = 1e-4

# ============================================================================
# ⭐ TUNING GRIDS
# ============================================================================
#
# λ₂ grid:
#   Includes λ₂ = 0 explicitly (adaptive lasso, Zou 2006).
#   Fine resolution in [0, 0.01] where the optimum typically falls
#   for p < T (ridge unnecessary). Upper range [0.1, 1.0] retained
#   for completeness but rarely selected.
#
# λ₁ grid:
#   100 log-spaced points from λ₁_max to λ₁_max × 1e-4.
#   λ₁_max = (1/T)·max_j|Xⱼᵀy| is the smallest λ₁ that zeros all β.
#   For CV (computationally expensive), consider reducing to 50-70.

AEN_LAMBDA2_GRID = np.array([
0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15               # moderate-to-heavy ridge
])

AEN_LAMBDA1_N_VALUES = 100

# ============================================================================
# ⭐ IC SETTINGS
# ============================================================================

# df proxy (same for all criteria)
AEN_BIC_DF_PROXY = "n_nonzero"

# GIC custom penalty: IC = T·log(RSS/T) + α·df
# Only used when AEN_TUNING_CRITERION = "GIC"
GIC_ALPHA = 3.0

# ============================================================================
# ⭐ CROSS-VALIDATION SETTINGS (rolling-origin)
# ============================================================================
# Only used when AEN_TUNING_CRITERION = "CV".
#
# Rolling-origin (expanding window) preserves temporal dependence:
#   Train on t = 1,...,τ  →  predict t = τ + h.
#   τ starts at ⌊CV_INITIAL_FRAC × T⌋ and expands to T − h.
#
# CV_INITIAL_FRAC:  fraction of T for initial training window.
#                   0.60 → ~95 obs for T=158. Must leave enough folds.
# CV_STEP_AHEAD:    forecast horizon h (1 = one-step-ahead).
#
# Reference: Hyndman & Athanasopoulos (2021), FPP Ch. 5.8.

CV_INITIAL_FRAC = 0.70
CV_STEP_AHEAD = 1

# ============================================================================
# ⭐ STANDARDIZATION
# ============================================================================
# Paper p. 1735: ||Xⱼ||₂ = 1 (L2-norm one, NOT unit variance).
AEN_STANDARDIZATION = "L2-norm = 1"

# ============================================================================
# ⭐ INFERENCE (post-selection OLS)
# ============================================================================

HAC_LAGS = 6

# ============================================================================
# ⭐ BOOTSTRAP / STABILITY SELECTION
# ============================================================================
# Adapted from Meinshausen & Bühlmann (2010, JRSS-B) for time-series
# dependence using block bootstrap (Künsch 1989; Politis & Romano 1994).
#
# BOOTSTRAP_METHOD:
#   "circular"    — Circular block bootstrap (Politis & Romano 1992c).
#                   Fixed block length L. Künsch-style, wraps around.
#   "stationary"  — Stationary bootstrap (Politis & Romano 1994, JASA).
#                   Random block lengths ~ Geometric(p = 1/L).
#                   Expected block length = L. Less sensitive to L choice.
#
# Key parameters:
#   STABILITY_MAX_Q:    maximum model size per bootstrap replicate (q-cap).
#                       Controls E(V) via the M&B bound:
#                           E(V) ≤ q² / [p · (2·π_thr − 1)]
#                       If AEN+IC selects more than q factors in a replicate,
#                       only the top-q by |β̂_AEN| (stage 2) are kept.
#   STABILITY_THRESHOLD: selection frequency cutoff π_thr.
#                       M&B recommend π_thr ∈ (0.6, 0.9).
#   STABILITY_THRESHOLD_ROBUSTNESS: lower threshold for sensitivity check.

BOOTSTRAP_METHOD = "stationary"          # "circular" or "stationary"
BOOTSTRAP_N_REPS = 100
BOOTSTRAP_BLOCK_LENGTH = 9
STABILITY_MAX_Q = None                 # M&B model-size cap: None = disabled
STABILITY_THRESHOLD = 0.80             # main threshold (π_thr)
STABILITY_THRESHOLD_ROBUSTNESS = 0.70  # higher robustness-check threshold

# ============================================================================
# ⭐ CLUSTER-LEVEL FREQUENCY AGGREGATION
# ============================================================================
# When factors are correlated, standard stability selection suffers from
# "vote splitting": the lasso picks one proxy per replicate, splitting
# frequency across cluster members so that none passes the threshold.
#
# Motivated by Faletto & Bien (2022, arXiv:2201.00494), we extend the
# M&B framework by computing selection frequencies at the cluster level:
# a cluster is "selected" in a replicate if ANY member is selected.
# After identifying stable clusters, the member with the highest
# individual frequency serves as the cluster representative.
#
# NOTE: This is NOT the full CSS procedure of Faletto & Bien (which uses
# complementary pairs subsampling and weighted representatives). We retain
# M&B block bootstrap for time-series dependence and adopt cluster-level
# aggregation as a post-hoc correction.
#
# CSS_ENABLED:
#   True  — Cluster-level frequency aggregation (recommended with
#           correlated factors). Individual + cluster frequencies both
#           reported.
#   False — Classic M&B stability selection (individual frequencies only).
#
# CSS_CORRELATION_THRESHOLD:
#   |ρ| above which factors are grouped into the same cluster.
#   0.70 is a reasonable default for financial factors.
#   Clusters built via hierarchical clustering (complete linkage)
#   on the factor correlation matrix.
#
# CSS_MIN_GAP:
#   Minimum gap (cluster_freq − max_individual_freq) to trigger
#   vote-splitting correction.  Only clusters with gap ≥ this value
#   are candidates for representative inclusion.
#   0.15 (15 percentage points) filters out clusters where one member
#   already dominates and no correction is needed.
#
# CSS_REPRESENTATIVE_METHOD:
#   "max_freq"       — pick the cluster member with highest individual
#                      bootstrap selection frequency.

CSS_ENABLED = True
CSS_CORRELATION_THRESHOLD = 0.70
CSS_MIN_GAP = 0
CSS_REPRESENTATIVE_METHOD = "max_freq"

# ============================================================================
# OUTPUT DIRECTORIES — separate per criterion
# ============================================================================

def get_aen_output_dir() -> Path:
    suffix = AEN_TUNING_CRITERION.lower()
    output_dir = RESULTS_DIR / "aen" / suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def get_strategy_aen_dir(strategy_name: str) -> Path:
    suffix = AEN_TUNING_CRITERION.lower()
    output_dir = RESULTS_DIR / strategy_name / "aen" / suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# ============================================================================
# HELPER
# ============================================================================

def print_config_summary():
    print("=" * 72)
    print("AEN CONFIGURATION SUMMARY")
    print("=" * 72)
    print(f"  AEN_TUNING_CRITERION:    {AEN_TUNING_CRITERION}")
    print(f"  CORRELATION_THRESHOLD:   {CORRELATION_THRESHOLD}")
    print(f"  FACTORS_TO_EXCLUDE:      {FACTORS_TO_EXCLUDE if FACTORS_TO_EXCLUDE else '(none yet)'}")
    if AEN_TUNING_CRITERION == "SIS_BIC":
        print(f"  SIS_D_N_RULE:            {SIS_D_N_RULE}")
    if AEN_TUNING_CRITERION == "GIC":
        print(f"  GIC_ALPHA:               {GIC_ALPHA}")
    print(f"  AEN_GAMMA:               {AEN_GAMMA}")
    print(f"  AEN_LAMBDA2_GRID:        {AEN_LAMBDA2_GRID}")
    print(f"  AEN_LAMBDA1_N_VALUES:    {AEN_LAMBDA1_N_VALUES}")
    print(f"  AEN_BIC_DF_PROXY:        {AEN_BIC_DF_PROXY}")
    print(f"  AEN_STANDARDIZATION:     {AEN_STANDARDIZATION}")
    print("=" * 72)

if __name__ == "__main__":
    print_config_summary()
    print(f"\nPROJECT_ROOT:  {PROJECT_ROOT}")
    print(f"FACTORS_PATH:  {FACTORS_PATH}  (exists: {FACTORS_PATH.exists()})")
    print(f"\nStrategies:")
    for name, path in STRATEGIES.items():
        print(f"  {name}: {path}  (exists: {path.exists()})")
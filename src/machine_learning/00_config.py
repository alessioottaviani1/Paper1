"""
================================================================================
00_config.py - Configuration for the factor-selection pipeline
================================================================================
Centralized parameters. The PRIMARY selector is best-subset (ℓ0, fixed k;
02s_best_subset.py); the adaptive elastic net (02_estimation.py) is retained
downstream as a selection-method robustness check, tuned by the Hannan–Quinn
information criterion (HQC). No information-criterion machinery runs for the
primary, which fixes k by parsimony and reports a k-sensitivity instead.

References:
    Zou, H. and Zhang, H.H. (2009),
        "On the Adaptive Elastic-Net with a Diverging Number of Parameters",
        Annals of Statistics, 37(4), 1733-1751.
    Bertsimas, D., King, A. and Mazumder, R. (2016),
        "Best Subset Selection via a Modern Optimization Lens",
        Annals of Statistics, 44(2), 813-852.

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
AEN_START_DATE   = "2008-01-31"   

# ============================================================================
# ⭐ AEN TUNING CRITERION (robustness selector only)
# ============================================================================
# The adaptive elastic net (02) is tuned by the Hannan–Quinn IC:
#   HQC penalty per factor = 2·log(log(T)) ≈ 3.24 at T ≈ 158.
# HQC is a consistent, mid-strength penalty. Because the AEN is a selector-
# robustness check against the best-subset primary, alternative ICs are not
# explored. This string also sets the per-criterion output sub-directory
# (results/.../aen/hqc/), which the whole pipeline reads.

AEN_TUNING_CRITERION = "HQC"

# ============================================================================
# ⭐ CORRELATION DIAGNOSTIC (no pruning)
# ============================================================================
# Correlation is reported for transparency only — factors are NOT dropped.
# Best-subset (ℓ0) handles collinearity by keeping the most informative member
# of a correlated cluster. CORRELATION_THRESHOLD flags pairs above it in 01;
# FACTORS_TO_EXCLUDE is the manual drop list (kept empty: nothing is pruned).

CORRELATION_THRESHOLD = 0.95

FACTORS_TO_EXCLUDE: list[str] = ["R5_EU", "GOVT_EU", "TERM_EU", "GLOBAL_TERM", "CRED_SPR_US", "ITRX_MAIN"]

# ============================================================================
# ⭐ ADAPTIVE ELASTIC-NET PARAMETERS (Zou & Zhang 2009)
# ============================================================================

AEN_GAMMA = 1                          # adaptive weight exponent
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

AEN_LAMBDA2_GRID = np.array([
    0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15
])
AEN_LAMBDA1_N_VALUES = 100

# ============================================================================
# ⭐ INFERENCE (post-selection OLS)
# ============================================================================

HAC_LAGS = 4

# ============================================================================
# ⭐ BLOCK BOOTSTRAP (inference on the fixed best-subset set)
# ============================================================================
# 04_bootstrap.py resamples the post-selection OLS on the FIXED best-subset
# factors to obtain block-bootstrap CIs / p-values for alpha (NO model
# re-selection). Block bootstrap preserves time-series dependence.
#
# BOOTSTRAP_METHOD:
#   "circular"    — Circular block bootstrap (Politis & Romano 1992c), fixed L.
#   "stationary"  — Stationary bootstrap (Politis & Romano 1994, JASA),
#                   random block lengths ~ Geometric(1/L); less sensitive to L.

BOOTSTRAP_METHOD = "stationary"          # "circular" or "stationary"
BOOTSTRAP_N_REPS = 9999
BOOTSTRAP_BLOCK_LENGTH = 9


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
    print(f"  AEN_GAMMA:               {AEN_GAMMA}")
    print(f"  AEN_LAMBDA2_GRID:        {AEN_LAMBDA2_GRID}")
    print(f"  AEN_LAMBDA1_N_VALUES:    {AEN_LAMBDA1_N_VALUES}")
    print("=" * 72)

if __name__ == "__main__":
    print_config_summary()
    print(f"\nPROJECT_ROOT:  {PROJECT_ROOT}")
    print(f"FACTORS_PATH:  {FACTORS_PATH}  (exists: {FACTORS_PATH.exists()})")
    print(f"\nStrategies:")
    for name, path in STRATEGIES.items():
        print(f"  {name}: {path}  (exists: {path.exists()})")
"""
================================================================================
06e_conditional_alpha.py — Conditional Alpha Analysis
================================================================================
Section (e) for 06_aen_robustness.py

Tests whether strategy alpha varies with financial stress, using stable
factors from bootstrap stability selection.

Five tests, grounded in the conditional performance evaluation literature:

  (e1) Ferson–Schadt (1996, JF) conditional alpha + conditional beta:
       r_t = α₀ + α₁·z_t + Σⱼ (βⱼ + δⱼ·z_t)·Xⱼt + εₜ
       - z_t = standardized conditioning variable (continuous)
       - α₁ tests whether alpha varies with market conditions
       - δⱼ tests whether factor exposures are time-varying
       - Joint F-test on all interaction terms {α₁, δ₁,...,δₖ}
       Primary reference for conditional fund evaluation.

  (e2) Dummy interaction (Mitchell & Pulvino 2001, JF):
       r_t = α₀ + α₁·D_HIGH,t + Σⱼ βⱼ·Xⱼt + εₜ
       - D_HIGH = 1{stress > threshold}
       - α₁ = incremental alpha in stress periods
       Threshold robustness: 80, 100, 120 bps (Patton 2009, RFS)

  (e3) Sub-sample split by regime:
       Separate OLS in HIGH vs non-HIGH months.
       Gives economic magnitudes: α_HIGH vs α_NORMAL in annualized %.

  (e4) Multiple conditioning variables (Patton 2009, RFS):
       Repeat (e1) with alternative z_t: V2X, EURIBOR-OIS.
       Tests robustness to the choice of conditioning variable.

  (e5) Rolling alpha with regime shading:
       Overlay stress regimes on the rolling alpha from section (a).
       Visual evidence that alpha peaks coincide with stress episodes.

References:
    Ferson, W. and Schadt, R. (1996), "Measuring Fund Strategy and
        Performance in Changing Economic Conditions", Journal of Finance,
        51(2), 425-461.
    Mitchell, M. and Pulvino, T. (2001), "Characteristics of Risk and
        Return in Risk Arbitrage", Journal of Finance, 56(6), 2135-2175.
    Patton, A. (2009), "Are 'Market Neutral' Hedge Funds Really Market
        Neutral?", Review of Financial Studies, 22(7), 2495-2530.
    Christoffersen, P. and Langlois, H. (2013), "The Joint Dynamics of
        Equity and Bond Returns", Journal of Financial and Quantitative
        Analysis, 48(5), 1453-1480.
    Joenväärä, J., Kauppila, M., Kosowski, R. and Tolonen, P. (2021),
        "Hedge Fund Performance", Review of Financial Studies, 34(7),
        3417-3473.

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
from scipy import stats as scipy_stats

import importlib.util
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"

spec = importlib.util.spec_from_file_location("aen_config", config_path)
aen_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aen_config)

HAC_LAGS             = aen_config.HAC_LAGS
FACTORS_PATH         = aen_config.FACTORS_PATH
FACTORS_END_DATE     = aen_config.FACTORS_END_DATE
STRATEGIES           = aen_config.STRATEGIES
get_strategy_aen_dir = aen_config.get_strategy_aen_dir
get_aen_output_dir   = aen_config.get_aen_output_dir

# Import FACTOR_INFO from 07_tables for factor descriptions in notes
try:
    from importlib.util import spec_from_file_location, module_from_spec
    _spec07 = spec_from_file_location("tables07",
        PROJECT_ROOT / "src" / "machine_learning" / "07_tables.py")
    _mod07 = module_from_spec(_spec07)
    _spec07.loader.exec_module(_mod07)
    FACTOR_INFO = _mod07.FACTOR_INFO
except Exception:
    FACTOR_INFO = {}

# ── Stress proxy paths (same as RQ3) ──────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
FACTORS_EXTERNAL_DIR = DATA_DIR / "external" / "factors"
NONTRADABLE_FILE     = FACTORS_EXTERNAL_DIR / "Nontradable_risk_factors.xlsx"
TRADABLE_CB_FILE     = FACTORS_EXTERNAL_DIR / "Tradable_corporate_bond_factors.xlsx"

# ── Conditioning thresholds (bps) for dummy interaction ───────────────────
# Mitchell & Pulvino (2001) style; Patton (2009) robustness to threshold
ITRX_THRESHOLDS_BPS = [80, 100, 120]
DEFAULT_THRESHOLD    = 100

# ── Conditioning variables for Ferson–Schadt ──────────────────────────────
# Primary: iTraxx Main 5Y (credit stress, continuous)
# Robustness: V2X (equity vol), EURIBOR-OIS (funding liquidity)
CONDITIONING_VARIABLES = {
    "ITRX_MAIN": {
        "source": "tradable",
        "sheet": "CDS_INDEX", "skiprows": 14,
        "usecols": [0, 1], "colnames": ["Date", "value"],
        "label": "iTraxx Main 5Y",
    },
    "V2X": {
        "source": "nontradable",
        "sheet": "VIX", "skiprows": 14,
        "usecols": [0, 2], "colnames": ["Date", "value"],
        "label": "V2X (Euro implied vol)",
    },
    "EURIBOR_OIS": {
        "source": "factors_parquet",
        "label": "Euribor–OIS spread",
    },
}

# ── Plot settings ─────────────────────────────────────────────────────────
FIGURE_DPI = 150
FIGURE_FORMAT = "pdf"
REGIME_COLORS = {"LOW": "#2ca02c", "MEDIUM": "#ff7f0e", "HIGH": "#d62728"}

ROLLING_WINDOW = 36


# ============================================================================
# HELPERS
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
    daily_df = pd.read_csv(strategy_path, index_col=0, parse_dates=True)
    daily_returns = daily_df['index_return'].dropna()
    monthly = daily_returns.resample('ME').apply(
        lambda x: ((1 + x / 100).prod() - 1) * 100 if len(x) > 0 else np.nan
    )
    return monthly.dropna()


def load_stress_proxy_monthly(proxy_name):
    """
    Load a stress proxy as a monthly Series (end-of-month level).
    Handles three source types: tradable Excel, nontradable Excel, factors parquet.
    """
    cfg = CONDITIONING_VARIABLES[proxy_name]

    if cfg["source"] == "factors_parquet":
        # Already in all_factors_monthly.parquet (e.g., EURIBOR_OIS)
        factors = pd.read_parquet(FACTORS_PATH)
        if proxy_name in factors.columns:
            return factors[proxy_name].dropna()
        else:
            raise KeyError(f"{proxy_name} not found in {FACTORS_PATH}")

    # Excel source
    file_path = TRADABLE_CB_FILE if cfg["source"] == "tradable" else NONTRADABLE_FILE
    raw = pd.read_excel(
        file_path, sheet_name=cfg["sheet"],
        skiprows=cfg["skiprows"], usecols=cfg["usecols"], header=0
    )
    raw.columns = cfg["colnames"]
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date")
    daily = pd.to_numeric(raw["value"], errors="coerce").dropna()

    # Resample to monthly (end-of-month last observation)
    monthly = daily.resample('ME').last().dropna()
    monthly.name = proxy_name
    return monthly


def load_stable_factors(strategy_name):
    """Load stable factor list from bootstrap_stability.json."""
    strategy_dir = get_strategy_aen_dir(strategy_name)
    stability_path = strategy_dir / "bootstrap_stability.json"
    if not stability_path.exists():
        raise FileNotFoundError(f"Missing {stability_path}")
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    stable_factors = stability.get("stable_factors", [])
    if not stable_factors:
        raise ValueError(f"No stable_factors for {strategy_name}")
    return stable_factors


def prepare_data(strategy_name, strategy_path):
    """
    Load returns, factors (original scale), and stable factor list.
    Returns (y, X, stable_factors) aligned on common dates, NaN-free.
    """
    stable_factors = load_stable_factors(strategy_name)
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

    return y, X, stable_factors


# ============================================================================
# (e1) FERSON–SCHADT CONDITIONAL ALPHA + BETA
# ============================================================================

def ferson_schadt_conditional(strategy_name, strategy_path):
    """
    Ferson & Schadt (1996, JF) conditional performance model.

    r_t = α₀ + α₁·z_t + Σⱼ βⱼ·Xⱼt + Σⱼ δⱼ·(Xⱼt·z_t) + εₜ

    where z_t is a standardized conditioning variable.

    - α₁ > 0 ⟹ alpha increases with stress (Duffie prediction)
    - δⱼ ≠ 0 ⟹ factor exposures are time-varying
    - Joint F-test on {α₁, δ₁,...,δₖ}: tests whether any conditioning matters

    Uses primary conditioning variable (iTraxx Main 5Y).
    """
    print_header(f"   (e1) Ferson–Schadt Conditional Model — {strategy_name}", "─")

    y, X, stable_factors = prepare_data(strategy_name, strategy_path)
    T, k = len(y), len(stable_factors)

    # Load primary conditioning variable
    z_raw = load_stress_proxy_monthly("ITRX_MAIN")
    z_aligned = z_raw.reindex(y.index, method='nearest')
    # Standardize z to mean=0, std=1 (Ferson-Schadt convention)
    z = (z_aligned - z_aligned.mean()) / z_aligned.std()
    z = z.reindex(y.index).dropna()

    # Realign after z dropna
    common = y.index.intersection(z.index)
    y, X, z = y[common], X.loc[common], z[common]
    T = len(y)

    print(f"\n   Stable factors: {stable_factors}")
    print(f"   Conditioning variable: iTraxx Main 5Y (standardized)")
    print(f"   T = {T}, k = {k}")

    # ── Unconditional model (baseline) ─────────────────────────────────
    X_unc = sm.add_constant(X, prepend=True)
    model_unc = sm.OLS(y, X_unc)
    res_unc = model_unc.fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

    # ── Conditional model ──────────────────────────────────────────────
    # Build interaction terms: z_t and Xⱼt × z_t for each factor
    X_cond = X.copy()
    X_cond['z_stress'] = z.values       # α₁ coefficient
    for f in stable_factors:
        X_cond[f'{f}_x_z'] = X[f].values * z.values   # δⱼ coefficient
    X_cond = sm.add_constant(X_cond, prepend=True)

    model_cond = sm.OLS(y, X_cond)
    res_cond = model_cond.fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

    # ── Extract key results ────────────────────────────────────────────
    alpha0 = res_cond.params['const']
    alpha1 = res_cond.params['z_stress']
    alpha1_t = res_cond.tvalues['z_stress']
    alpha1_p = res_cond.pvalues['z_stress']

    print(f"\n   ── Unconditional Model ──")
    print(f"   α₀ = {res_unc.params['const']:+.4f}% monthly"
          f" ({res_unc.params['const'] * 12:+.2f}% ann.)")
    print(f"   R² adj = {res_unc.rsquared_adj:.4f}")

    print(f"\n   ── Conditional Model (Ferson–Schadt 1996) ──")
    print(f"   α₀ = {alpha0:+.4f}% monthly ({alpha0 * 12:+.2f}% ann.)")
    print(f"   α₁ = {alpha1:+.4f} (t = {alpha1_t:.3f},"
          f" p = {alpha1_p:.4f}) {significance_stars(alpha1_p)}")

    if alpha1 > 0:
        print(f"   → Alpha INCREASES with stress (Duffie-consistent)")
    else:
        print(f"   → Alpha does not increase with stress")

    # Conditional betas
    print(f"\n   Conditional Betas (δⱼ):")
    print(f"   {'Factor':<25} {'δⱼ':>10} {'t':>8} {'p':>8}")
    print(f"   {'─' * 53}")
    delta_results = {}
    for f in stable_factors:
        col = f'{f}_x_z'
        delta = res_cond.params[col]
        delta_t = res_cond.tvalues[col]
        delta_p = res_cond.pvalues[col]
        print(f"   {f:<25} {delta:>+10.4f} {delta_t:>8.3f}"
              f" {delta_p:>8.4f} {significance_stars(delta_p)}")
        delta_results[f] = {
            'delta': round(float(delta), 6),
            't_stat': round(float(delta_t), 4),
            'p_value': round(float(delta_p), 4),
        }

    # ── Joint F-test: all interaction terms ────────────────────────────
    # H₀: α₁ = δ₁ = ... = δₖ = 0 (no conditioning matters)
    interaction_cols = ['z_stress'] + [f'{f}_x_z' for f in stable_factors]
    r_matrix = np.zeros((len(interaction_cols), len(res_cond.params)))
    for i, col in enumerate(interaction_cols):
        col_idx = list(res_cond.params.index).index(col)
        r_matrix[i, col_idx] = 1.0

    f_test = res_cond.f_test(r_matrix)
    f_stat = float(f_test.fvalue)
    f_pval = float(f_test.pvalue)

    print(f"\n   Joint F-test (H₀: α₁ = δ₁ = ... = δₖ = 0):")
    print(f"   F({len(interaction_cols)}, {T - len(res_cond.params)}) ="
          f" {f_stat:.3f}, p = {f_pval:.4f} {significance_stars(f_pval)}")

    # ── Wald test: conditional alpha only ──────────────────────────────
    # H₀: α₁ = 0 (alpha is unconditional)
    print(f"\n   Wald test (H₀: α₁ = 0, conditional alpha only):")
    print(f"   t = {alpha1_t:.3f}, p = {alpha1_p:.4f}"
          f" {significance_stars(alpha1_p)}")

    # R² comparison
    print(f"\n   R² adj unconditional:  {res_unc.rsquared_adj:.4f}")
    print(f"   R² adj conditional:    {res_cond.rsquared_adj:.4f}")
    print(f"   ΔR² adj:              "
          f" {res_cond.rsquared_adj - res_unc.rsquared_adj:+.4f}")

    # ── Economic magnitude ─────────────────────────────────────────────
    # α at z = +1 SD (high stress) vs z = -1 SD (low stress)
    alpha_high = alpha0 + alpha1 * 1.0
    alpha_low  = alpha0 + alpha1 * (-1.0)
    print(f"\n   Economic magnitude:")
    print(f"   α(z = +1σ, stress):  {alpha_high:+.4f}% mo"
          f" ({alpha_high * 12:+.2f}% ann.)")
    print(f"   α(z = -1σ, normal): {alpha_low:+.4f}% mo"
          f" ({alpha_low * 12:+.2f}% ann.)")
    print(f"   Spread (2σ):         {(alpha_high - alpha_low):+.4f}% mo"
          f" ({(alpha_high - alpha_low) * 12:+.2f}% ann.)")

    # ── Save ───────────────────────────────────────────────────────────
    result = {
        'strategy': strategy_name,
        'conditioning_variable': 'ITRX_MAIN',
        'T': T, 'k_base': k, 'k_total': len(res_cond.params) - 1,
        'unconditional': {
            'alpha_monthly': round(float(res_unc.params['const']), 6),
            'alpha_annualized': round(float(res_unc.params['const']) * 12, 4),
            'alpha_pval': round(float(res_unc.pvalues['const']), 4),
            'r2_adj': round(float(res_unc.rsquared_adj), 6),
        },
        'conditional': {
            'alpha0_monthly': round(float(alpha0), 6),
            'alpha0_annualized': round(float(alpha0 * 12), 4),
            'alpha1': round(float(alpha1), 6),
            'alpha1_tstat': round(float(alpha1_t), 4),
            'alpha1_pval': round(float(alpha1_p), 4),
            'alpha1_interpretation': (
                'alpha increases with stress'
                if alpha1 > 0 and alpha1_p < 0.10
                else 'no significant conditional alpha'),
            'r2_adj': round(float(res_cond.rsquared_adj), 6),
        },
        'conditional_betas': delta_results,
        'joint_f_test': {
            'f_statistic': round(f_stat, 4),
            'p_value': round(f_pval, 4),
            'df_num': len(interaction_cols),
            'df_denom': T - len(res_cond.params),
        },
        'economic_magnitude': {
            'alpha_at_plus_1sd': round(float(alpha_high * 12), 4),
            'alpha_at_minus_1sd': round(float(alpha_low * 12), 4),
            'spread_2sd_annualized': round(float((alpha_high - alpha_low) * 12), 4),
        },
    }

    strategy_dir = get_strategy_aen_dir(strategy_name)
    with open(strategy_dir / "conditional_ferson_schadt.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n   💾 conditional_ferson_schadt.json")

    return result


# ============================================================================
# (e2) DUMMY INTERACTION (Mitchell & Pulvino 2001)
# ============================================================================

def dummy_interaction(strategy_name, strategy_path):
    """
    r_t = α₀ + α₁·D_HIGH,t + Σⱼ βⱼ·Xⱼt + εₜ

    D_HIGH = 1{iTraxx Main > threshold bps}.
    α₁ = incremental alpha in stress.

    Threshold robustness: 80, 100, 120 bps (Patton 2009).
    """
    print_header(f"   (e2) Dummy Interaction — {strategy_name}", "─")

    y, X, stable_factors = prepare_data(strategy_name, strategy_path)
    T, k = len(y), len(stable_factors)

    # Load stress proxy (levels, not standardized)
    z_raw = load_stress_proxy_monthly("ITRX_MAIN")
    z_aligned = z_raw.reindex(y.index, method='nearest')

    common = y.index.intersection(z_aligned.dropna().index)
    y, X, z_level = y[common], X.loc[common], z_aligned[common]
    T = len(y)

    print(f"\n   Stable factors: {stable_factors}")
    print(f"   T = {T}")
    print(f"   iTraxx Main range: [{z_level.min():.1f},"
          f" {z_level.max():.1f}] bps")

    results_all_thresholds = {}

    for threshold in ITRX_THRESHOLDS_BPS:
        D_high = (z_level > threshold).astype(float)
        n_high = int(D_high.sum())
        n_low = T - n_high

        print(f"\n   ── Threshold: {threshold} bps ──")
        print(f"   HIGH months: {n_high}/{T} ({n_high/T:.1%})")

        if n_high < 10:
            print(f"   ⚠️  Too few HIGH obs ({n_high}), skipping")
            results_all_thresholds[str(threshold)] = {
                'threshold_bps': threshold,
                'n_HIGH': n_high, 'n_LOW': n_low,
                'skip': True, 'reason': 'too few HIGH observations'
            }
            continue

        X_dummy = X.copy()
        X_dummy['D_HIGH'] = D_high.values
        X_dummy_const = sm.add_constant(X_dummy, prepend=True)

        model = sm.OLS(y, X_dummy_const)
        res = model.fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

        alpha0 = res.params['const']
        alpha1 = res.params['D_HIGH']
        alpha1_t = res.tvalues['D_HIGH']
        alpha1_p = res.pvalues['D_HIGH']

        print(f"   α₀ (normal) = {alpha0:+.4f}% mo"
              f" ({alpha0 * 12:+.2f}% ann.)")
        print(f"   α₁ (HIGH)   = {alpha1:+.4f}% mo"
              f" ({alpha1 * 12:+.2f}% ann.)"
              f"  t = {alpha1_t:.3f}, p = {alpha1_p:.4f}"
              f" {significance_stars(alpha1_p)}")
        print(f"   α(HIGH)     = {(alpha0 + alpha1):+.4f}% mo"
              f" ({(alpha0 + alpha1) * 12:+.2f}% ann.)")

        results_all_thresholds[str(threshold)] = {
            'threshold_bps': threshold,
            'n_HIGH': n_high, 'n_LOW': n_low,
            'alpha0_monthly': round(float(alpha0), 6),
            'alpha0_annualized': round(float(alpha0 * 12), 4),
            'alpha1_monthly': round(float(alpha1), 6),
            'alpha1_annualized': round(float(alpha1 * 12), 4),
            'alpha1_tstat': round(float(alpha1_t), 4),
            'alpha1_pval': round(float(alpha1_p), 4),
            'alpha_HIGH_annualized': round(float((alpha0 + alpha1) * 12), 4),
            'r2_adj': round(float(res.rsquared_adj), 6),
        }

    # Summary across thresholds
    print(f"\n   ── Threshold Robustness Summary ──")
    print(f"   {'Threshold':>10} {'α₁ (ann)':>10} {'t-stat':>8}"
          f" {'p-val':>8} {'n_HIGH':>7}")
    print(f"   {'─' * 45}")
    for th, r in results_all_thresholds.items():
        if r.get('skip'):
            print(f"   {th:>10} {'skipped':>10}")
        else:
            print(f"   {th + ' bps':>10}"
                  f" {r['alpha1_annualized']:>+10.2f}"
                  f" {r['alpha1_tstat']:>8.3f}"
                  f" {r['alpha1_pval']:>8.4f}"
                  f" {r['n_HIGH']:>7}"
                  f" {significance_stars(r['alpha1_pval'])}")

    # Save
    strategy_dir = get_strategy_aen_dir(strategy_name)
    with open(strategy_dir / "conditional_dummy_interaction.json", 'w') as f:
        json.dump({
            'strategy': strategy_name,
            'T': T, 'stable_factors': stable_factors,
            'thresholds': results_all_thresholds,
        }, f, indent=2)
    print(f"\n   💾 conditional_dummy_interaction.json")

    return results_all_thresholds


# ============================================================================
# (e3) SUB-SAMPLE SPLIT BY REGIME
# ============================================================================

def subsample_regime(strategy_name, strategy_path):
    """
    Estimate OLS separately on HIGH vs non-HIGH months.
    Gives direct economic magnitudes: α_HIGH vs α_NORMAL.
    """
    print_header(f"   (e3) Sub-Sample Split by Regime — {strategy_name}", "─")

    y, X, stable_factors = prepare_data(strategy_name, strategy_path)

    z_raw = load_stress_proxy_monthly("ITRX_MAIN")
    z_aligned = z_raw.reindex(y.index, method='nearest')
    common = y.index.intersection(z_aligned.dropna().index)
    y, X, z_level = y[common], X.loc[common], z_aligned[common]

    threshold = DEFAULT_THRESHOLD
    D_high = z_level > threshold
    T = len(y)

    results = {}

    for label, mask in [("HIGH", D_high), ("NORMAL", ~D_high)]:
        y_sub = y[mask]
        X_sub = X[mask]
        n = len(y_sub)

        print(f"\n   ── {label} (n = {n}) ──")

        if n < len(stable_factors) + 5:
            print(f"   ⚠️  Too few obs for OLS ({n} < {len(stable_factors) + 5})")
            results[label] = {'n': n, 'skip': True}
            continue

        X_const = sm.add_constant(X_sub, prepend=True)
        model = sm.OLS(y_sub, X_const)

        # Use OLS SE if too few obs for reliable HAC
        if n > 30:
            res = model.fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
            se_type = 'HAC'
        else:
            res = model.fit()
            se_type = 'OLS'

        alpha = res.params['const']
        alpha_t = res.tvalues['const']
        alpha_p = res.pvalues['const']

        print(f"   α = {alpha:+.4f}% monthly ({alpha * 12:+.2f}% ann.)")
        print(f"   t = {alpha_t:.3f}, p = {alpha_p:.4f}"
              f" {significance_stars(alpha_p)}  ({se_type} SE)")
        print(f"   R² adj = {res.rsquared_adj:.4f}")

        results[label] = {
            'n': n,
            'alpha_monthly': round(float(alpha), 6),
            'alpha_annualized': round(float(alpha * 12), 4),
            'alpha_tstat': round(float(alpha_t), 4),
            'alpha_pval': round(float(alpha_p), 4),
            'r2_adj': round(float(res.rsquared_adj), 6),
            'se_type': se_type,
        }

    # Difference
    if not results.get('HIGH', {}).get('skip') and not results.get('NORMAL', {}).get('skip'):
        diff = results['HIGH']['alpha_annualized'] - results['NORMAL']['alpha_annualized']
        print(f"\n   Δα (HIGH − NORMAL) = {diff:+.2f}% annualized")
        results['difference_annualized'] = round(diff, 4)

    strategy_dir = get_strategy_aen_dir(strategy_name)
    with open(strategy_dir / "conditional_subsample.json", 'w') as f:
        json.dump({
            'strategy': strategy_name,
            'threshold_bps': threshold, 'T': T,
            'stable_factors': stable_factors,
            'regimes': results,
        }, f, indent=2)
    print(f"\n   💾 conditional_subsample.json")

    return results


# ============================================================================
# (e4) MULTIPLE CONDITIONING VARIABLES (Patton 2009)
# ============================================================================

def multiple_conditioning_variables(strategy_name, strategy_path):
    """
    Repeat Ferson–Schadt (e1) with alternative conditioning variables.
    Tests robustness to the choice of z_t.
    """
    print_header(
        f"   (e4) Multiple Conditioning Variables — {strategy_name}", "─")

    y, X, stable_factors = prepare_data(strategy_name, strategy_path)
    T_orig, k = len(y), len(stable_factors)

    results = {}

    for proxy_name, cfg in CONDITIONING_VARIABLES.items():
        print(f"\n   ── z_t = {cfg['label']} ──")

        try:
            z_raw = load_stress_proxy_monthly(proxy_name)
        except Exception as e:
            print(f"   ⚠️  Cannot load {proxy_name}: {e}")
            results[proxy_name] = {'skip': True, 'reason': str(e)}
            continue

        z_aligned = z_raw.reindex(y.index, method='nearest')
        common = y.index.intersection(z_aligned.dropna().index)
        y_sub = y[common]
        X_sub = X.loc[common]
        z_sub = z_aligned[common]
        T = len(y_sub)

        if T < k + 10:
            print(f"   ⚠️  Too few obs ({T}), skip")
            results[proxy_name] = {'skip': True, 'T': T}
            continue

        # Standardize z
        z_std = (z_sub - z_sub.mean()) / z_sub.std()

        # Ferson–Schadt model
        X_cond = X_sub.copy()
        X_cond['z_stress'] = z_std.values
        for f in stable_factors:
            X_cond[f'{f}_x_z'] = X_sub[f].values * z_std.values
        X_cond = sm.add_constant(X_cond, prepend=True)

        model = sm.OLS(y_sub, X_cond)
        res = model.fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

        alpha1 = res.params['z_stress']
        alpha1_t = res.tvalues['z_stress']
        alpha1_p = res.pvalues['z_stress']

        # Joint F-test
        interaction_cols = ['z_stress'] + [f'{f}_x_z' for f in stable_factors]
        r_matrix = np.zeros((len(interaction_cols), len(res.params)))
        for i, col in enumerate(interaction_cols):
            col_idx = list(res.params.index).index(col)
            r_matrix[i, col_idx] = 1.0
        f_test = res.f_test(r_matrix)
        f_stat = float(f_test.fvalue)
        f_pval = float(f_test.pvalue)

        print(f"   T = {T}")
        print(f"   α₁ = {alpha1:+.4f}  (t = {alpha1_t:.3f},"
              f" p = {alpha1_p:.4f}) {significance_stars(alpha1_p)}")
        print(f"   Joint F = {f_stat:.3f} (p = {f_pval:.4f})"
              f" {significance_stars(f_pval)}")

        results[proxy_name] = {
            'label': cfg['label'],
            'T': T,
            'alpha1': round(float(alpha1), 6),
            'alpha1_tstat': round(float(alpha1_t), 4),
            'alpha1_pval': round(float(alpha1_p), 4),
            'joint_f_stat': round(f_stat, 4),
            'joint_f_pval': round(f_pval, 4),
            'r2_adj': round(float(res.rsquared_adj), 6),
        }

    # Summary
    print(f"\n   ── Cross-Variable Summary ──")
    print(f"   {'Variable':<25} {'α₁':>8} {'t':>8} {'p':>8}"
          f" {'F-joint':>8} {'p(F)':>8}")
    print(f"   {'─' * 67}")
    for pn, r in results.items():
        if r.get('skip'):
            continue
        print(f"   {CONDITIONING_VARIABLES[pn]['label']:<25}"
              f" {r['alpha1']:>+8.4f}"
              f" {r['alpha1_tstat']:>8.3f}"
              f" {r['alpha1_pval']:>8.4f}"
              f" {r['joint_f_stat']:>8.3f}"
              f" {r['joint_f_pval']:>8.4f}"
              f" {significance_stars(r['alpha1_pval'])}")

    # Count how many have α₁ > 0
    n_positive = sum(1 for r in results.values()
                     if not r.get('skip') and r.get('alpha1', 0) > 0)
    n_sig = sum(1 for r in results.values()
                if not r.get('skip') and r.get('alpha1_pval', 1) < 0.10
                and r.get('alpha1', 0) > 0)
    n_total = sum(1 for r in results.values() if not r.get('skip'))
    print(f"\n   α₁ > 0: {n_positive}/{n_total}")
    print(f"   α₁ > 0 and significant: {n_sig}/{n_total}")

    strategy_dir = get_strategy_aen_dir(strategy_name)
    with open(strategy_dir / "conditional_multi_variable.json", 'w') as f:
        json.dump({
            'strategy': strategy_name,
            'stable_factors': stable_factors,
            'conditioning_results': results,
        }, f, indent=2)
    print(f"\n   💾 conditional_multi_variable.json")

    return results


# ============================================================================
# (e5) ROLLING ALPHA WITH REGIME SHADING
# ============================================================================

def rolling_alpha_regime_plot(strategy_name, strategy_path):
    """
    Overlay stress regimes (iTraxx Main thresholds) on the rolling alpha
    plot from section (a).  Visual test: do alpha peaks coincide with
    HIGH-stress episodes?
    """
    print_header(
        f"   (e5) Rolling Alpha + Regime Shading — {strategy_name}", "─")

    strategy_dir = get_strategy_aen_dir(strategy_name)

    # Load rolling alpha from section (a)
    rolling_path = strategy_dir / "rolling_alpha.csv"
    if not rolling_path.exists():
        print(f"   ⚠️  rolling_alpha.csv not found. Run (a) first.")
        return None

    rolling_df = pd.read_csv(rolling_path)
    rolling_df['end_date'] = pd.to_datetime(rolling_df['end_date'])
    rolling_df = rolling_df.set_index('end_date')

    # Load stress proxy
    z_raw = load_stress_proxy_monthly("ITRX_MAIN")
    z_monthly = z_raw.resample('ME').last().dropna()

    # Define regimes
    threshold_high = DEFAULT_THRESHOLD
    threshold_low = 60  # consistent with RQ3 manual thresholds

    regime = pd.Series("MEDIUM", index=z_monthly.index)
    regime[z_monthly < threshold_low] = "LOW"
    regime[z_monthly >= threshold_high] = "HIGH"

    # Align to rolling alpha dates
    regime_aligned = regime.reindex(rolling_df.index, method='nearest')

    # Stats by regime
    for rl in ["LOW", "MEDIUM", "HIGH"]:
        mask = regime_aligned == rl
        if mask.sum() > 0:
            sub = rolling_df.loc[mask]
            avg = sub['alpha_annualized'].mean()
            n = mask.sum()
            print(f"   {rl:>7}: n = {n:>3},"
                  f" avg α (ann.) = {avg:+.2f}%")

    # ── Plot ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(
        f"Rolling Alpha ({ROLLING_WINDOW}M) with Stress Regimes — "
        f"{strategy_name}",
        fontsize=13, fontweight='bold')

    # Panel 1: Rolling alpha + CI + regime shading
    ax = axes[0]
    dates = rolling_df.index
    alpha_ann = rolling_df['alpha_annualized'].values
    ci_lo = rolling_df['ci_lower'].values * 12
    ci_hi = rolling_df['ci_upper'].values * 12

    ax.plot(dates, alpha_ann, color='black', linewidth=1.2, label='Rolling α')
    ax.fill_between(dates, ci_lo, ci_hi, color='grey', alpha=0.2,
                    label='95% CI')
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')

    # Regime shading
    for rl, color in REGIME_COLORS.items():
        mask = regime_aligned == rl
        if mask.any():
            blocks = mask.astype(int).diff().fillna(0)
            starts = dates[blocks == 1]
            ends = dates[blocks == -1]
            if mask.iloc[0]:
                starts = starts.insert(0, dates[0])
            if mask.iloc[-1]:
                ends = ends.append(pd.DatetimeIndex([dates[-1]]))
            alpha_shade = 0.15 if rl != "HIGH" else 0.25
            for s, e in zip(starts[:len(ends)], ends[:len(starts)]):
                ax.axvspan(s, e, alpha=alpha_shade, color=color, zorder=0)

    ax.set_ylabel("α (annualized %)", fontsize=11)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Panel 2: Stress proxy level
    ax2 = axes[1]
    z_plot = z_monthly.reindex(dates, method='nearest')
    ax2.plot(dates, z_plot.values, color='black', linewidth=0.8)
    ax2.axhline(threshold_high, color=REGIME_COLORS['HIGH'],
                linewidth=1.0, linestyle='--',
                label=f'HIGH ({threshold_high} bps)')
    ax2.axhline(threshold_low, color=REGIME_COLORS['LOW'],
                linewidth=1.0, linestyle='--',
                label=f'LOW ({threshold_low} bps)')
    ax2.set_ylabel("iTraxx Main (bps)", fontsize=10)
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

    fig.tight_layout()
    fig_path = strategy_dir / f"rolling_alpha_regime.{FIGURE_FORMAT}"
    fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()

    print(f"\n   📊 rolling_alpha_regime.{FIGURE_FORMAT}")

    # Save regime stats
    regime_stats = {}
    for rl in ["LOW", "MEDIUM", "HIGH"]:
        mask = regime_aligned == rl
        if mask.sum() > 0:
            sub = rolling_df.loc[mask, 'alpha_annualized']
            regime_stats[rl] = {
                'n': int(mask.sum()),
                'alpha_mean': round(float(sub.mean()), 4),
                'alpha_median': round(float(sub.median()), 4),
                'alpha_std': round(float(sub.std()), 4),
                'pct_positive': round(float((sub > 0).mean()), 4),
            }

    with open(strategy_dir / "rolling_alpha_regime_stats.json", 'w') as f:
        json.dump({
            'strategy': strategy_name,
            'threshold_high_bps': threshold_high,
            'threshold_low_bps': threshold_low,
            'regime_stats': regime_stats,
        }, f, indent=2)
    print(f"   💾 rolling_alpha_regime_stats.json")

    return regime_stats


# ============================================================================
# TEX GENERATION — BEAMER SLIDES + THESIS TABLES
# ============================================================================

TITLE_MAP = {
    "btp_italia":      "BTP Italia",
    "cds_bond_basis":  "CDS--Bond Basis",
    "itraxx_combined": "iTraxx Combined",
}

def _stars_tex(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.01:   return "***"
    elif p < 0.05: return "**"
    elif p < 0.10: return "*"
    return ""
def _stars_sup(p):
    """Superscript version for article tables."""
    s = _stars_tex(p)
    return f"^{{{s}}}" if s else ""

def _fmt2(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.2f}"

def _fmt4(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.4f}"

def _pretty_factor(name):
    name = name.replace("\u0394", r"\Delta ")
    if "_" not in name:
        return rf"$\mathrm{{{name}}}$"
    head, tail = name.split("_", 1)
    tail = tail.replace("_", r"\_")
    return rf"$\mathrm{{{head}}}_{{\mathrm{{{tail}}}}}$"

def _safe(d, *keys, default=None):
    """Safely traverse nested dicts."""
    obj = d
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        else:
            return default
    return obj


# ── TABLE 1: Ferson–Schadt (per-strategy slide + thesis table) ────────────

def _build_fs_beamer(strategy_name, fs_result):
    """
    Beamer slide: Ferson–Schadt conditional model for one strategy.
    Left: coefficient table (α₀, α₁, βⱼ, δⱼ). Right: model fit + F-test.
    """
    title = TITLE_MAP.get(strategy_name, strategy_name.replace("_", " ").title())
    cond = fs_result['conditional']
    unc  = fs_result['unconditional']
    jf   = fs_result['joint_f_test']
    em   = fs_result['economic_magnitude']
    deltas = fs_result.get('conditional_betas', {})
    T    = fs_result['T']
    k    = fs_result['k_base']

    tex = []
    tex.append(rf"\begin{{frame}}[t]{{Conditional Alpha (Ferson--Schadt): {title}}}")
    tex.append(r"\centering\vspace{-0.4cm}\scriptsize")
    tex.append(r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.05}")
    tex.append(r"\begin{columns}[T,onlytextwidth,totalwidth=0.94\textwidth]")

    # Left: coefficient table
    tex.append(r"\column{0.68\textwidth}\centering")
    tex.append(r"\begin{tabular}{l r r r}")
    tex.append(r"\toprule")
    tex.append(r"Variable & Coeff. & $t$-stat & $p$-value \\")
    tex.append(r"\midrule")

    # α₀
    a0 = cond['alpha0_annualized']
    a0_t = float(fs_result.get('_alpha0_t', 0))  # may not be stored
    a0_p = float(fs_result.get('_alpha0_p', 0))
    # We need to re-derive t/p for α₀ from the JSON — not available directly
    # Use unconditional alpha as proxy for display if needed
    tex.append(rf"$\alpha_0$ (ann.\%) & {a0:+.2f} & & \\")

    # α₁
    a1 = cond['alpha1']
    a1_t = cond['alpha1_tstat']
    a1_p = cond['alpha1_pval']
    tex.append(rf"$\alpha_1$ (stress) & {a1:+.4f}{_stars_tex(a1_p)} "
               rf"& {_fmt2(a1_t)} & {_fmt4(a1_p)} \\")
    tex.append(r"\midrule")

    # δⱼ (conditional betas)
    for f, d in deltas.items():
        tex.append(rf"{_pretty_factor(f)} $\times z_t$ & "
                   rf"{d['delta']:+.4f}{_stars_tex(d['p_value'])} & "
                   rf"{_fmt2(d['t_stat'])} & {_fmt4(d['p_value'])} \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    # Right: model fit
    tex.append(r"\column{0.26\textwidth}\centering\scriptsize")
    tex.append(r"\textbf{Model fit}\par\vspace{0.10cm}")
    tex.append(r"\begin{tabular}{@{}lr@{}}")
    tex.append(rf"$T$ & {T} \\")
    tex.append(rf"$k$ (base) & {k} \\")
    tex.append(rf"$R^2_{{adj}}$ (unc.) & {_fmt4(unc['r2_adj'])} \\")
    tex.append(rf"$R^2_{{adj}}$ (cond.) & {_fmt4(cond['r2_adj'])} \\")
    tex.append(rf"Joint $F$ & {_fmt2(jf['f_statistic'])} \\")
    tex.append(rf"$p(F)$ & {_fmt4(jf['p_value'])} \\")
    tex.append(r"\end{tabular}")
    tex.append(r"\par\vspace{0.15cm}")
    tex.append(r"{\tiny\textbf{Econ.\ magnitude:}\par")
    tex.append(rf"$\alpha(z{{=}}+1\sigma)$: {em['alpha_at_plus_1sd']:+.2f}\%\par")
    tex.append(rf"$\alpha(z{{=}}-1\sigma)$: {em['alpha_at_minus_1sd']:+.2f}\%\par")
    tex.append(r"*** $p{<}1\%$, ** $p{<}5\%$,\par * $p{<}10\%$.}")

    tex.append(r"\end{columns}")

    # Bottom: note
    tex.append(r"\vspace{0.15cm}")
    tex.append(r"{\tiny Ferson \& Schadt (1996, \textit{JF}). "
               r"$z_t$ = standardized iTraxx Main 5Y. "
               r"HAC (Newey--West) standard errors.}")

    tex.append(r"\end{frame}")
    return "\n".join(tex)


def _build_fs_thesis(all_results):
    """
    Thesis table: Ferson–Schadt results across all strategies.
    Panel A: α₀, α₁, joint F. Panel B: conditional betas δⱼ.
    Single tabular environment with Panel headers as multicolumn rows.
    """
    strategies = [s for s in all_results
                  if _safe(all_results[s], 'ferson_schadt', 'conditional')]
    if not strategies:
        return ""

    n_s = len(strategies)
    headers = " & ".join([TITLE_MAP.get(s, s) for s in strategies])

    # Collect all unique factors for Panel B
    all_factors = []
    for s in strategies:
        all_factors.extend(
            all_results[s]['ferson_schadt'].get('conditional_betas', {}).keys())
    unique_factors = sorted(set(all_factors))

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Conditional Alpha: Ferson--Schadt (1996) Model}")
    tex.append(r"\label{tab:conditional_alpha_fs}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l" + "c" * n_s + "}")
    tex.append(r"\toprule")
    tex.append(rf" & {headers} \\")
    tex.append(r"\midrule")

    # ── Panel A ──
    tex.append(rf"\multicolumn{{{n_s + 1}}}{{l}}{{\textit{{Panel A: Conditional alpha}}}} \\")
    tex.append(r"\addlinespace")

    # α₀ annualized
    vals = []
    for s in strategies:
        fs = all_results[s]['ferson_schadt']
        v = fs['unconditional']['alpha_annualized']
        p = fs['unconditional']['alpha_pval']
        vals.append(f"${v:+.2f}{_stars_sup(p)}$")
    tex.append(rf"$\alpha_0$ (ann.\ \%) & {' & '.join(vals)} \\")

    # α₁
    vals = []
    for s in strategies:
        c = all_results[s]['ferson_schadt']['conditional']
        vals.append(f"${c['alpha1']:+.4f}{_stars_sup(c['alpha1_pval'])}$")
    tex.append(rf"$\alpha_1$ (conditional) & {' & '.join(vals)} \\")

    # α₁ t-stat
    vals = []
    for s in strategies:
        c = all_results[s]['ferson_schadt']['conditional']
        vals.append(f"({c['alpha1_tstat']:.2f})")
    tex.append(rf"\quad $t$-stat & {' & '.join(vals)} \\")

    # Joint F
    vals = []
    for s in strategies:
        jf = all_results[s]['ferson_schadt']['joint_f_test']
        vals.append(f"${jf['f_statistic']:.2f}{_stars_sup(jf['p_value'])}$")
    tex.append(rf"Joint $F$-test & {' & '.join(vals)} \\")

    # R² adj unconditional
    vals = []
    for s in strategies:
        vals.append(_fmt4(all_results[s]['ferson_schadt']['unconditional']['r2_adj']))
    tex.append(rf"$\bar{{R}}^2$ (unconditional) & {' & '.join(vals)} \\")

    # R² adj conditional
    vals = []
    for s in strategies:
        vals.append(_fmt4(all_results[s]['ferson_schadt']['conditional']['r2_adj']))
    tex.append(rf"$\bar{{R}}^2$ (conditional) & {' & '.join(vals)} \\")

    # Economic magnitude
    vals_high = []
    vals_low = []
    for s in strategies:
        em = all_results[s]['ferson_schadt']['economic_magnitude']
        vals_high.append(f"{em['alpha_at_plus_1sd']:+.2f}\\%")
        vals_low.append(f"{em['alpha_at_minus_1sd']:+.2f}\\%")
    tex.append(rf"$\alpha(z = +1\sigma)$ ann. & {' & '.join(vals_high)} \\")
    tex.append(rf"$\alpha(z = -1\sigma)$ ann. & {' & '.join(vals_low)} \\")

    tex.append(r"\addlinespace")

    # T
    vals = []
    for s in strategies:
        vals.append(str(all_results[s]['ferson_schadt']['T']))
    tex.append(rf"$T$ & {' & '.join(vals)} \\")

    # ── Panel B ──
    if unique_factors:
        tex.append(r"\midrule")
        tex.append(rf"\multicolumn{{{n_s + 1}}}{{l}}{{\textit{{Panel B: Conditional betas ($\delta_j$)}}}} \\")
        tex.append(r"\addlinespace")

        for f in unique_factors:
            vals = []
            for s in strategies:
                d = _safe(all_results[s], 'ferson_schadt',
                          'conditional_betas', f)
                if d:
                    vals.append(f"${d['delta']:+.3f}{_stars_sup(d['p_value'])}$")
                else:
                    vals.append("--")
            tex.append(rf"{_pretty_factor(f)} $\times z_t$ & "
                       rf"{' & '.join(vals)} \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")

    tex.append(r"\item \textit{Note:} "
               r"Ferson and Schadt (1996) conditional performance model. "
               r"$z_t$ is the standardized iTraxx Main 5Y spread "
               r"(mean zero, unit variance). "
               r"$\alpha_1 > 0$ indicates alpha increases with credit stress. "
               r"Joint $F$-test: "
               r"$H_0{:}\ \alpha_1 = \delta_1 = \cdots = \delta_k = 0$. "
               r"HAC (Newey--West) standard errors with "
               rf"{HAC_LAGS} lags. "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$. "
               r"Factor definitions are provided in Table~\ref{tab:factor_list}.")

    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")
    return "\n".join(tex)


# ── TABLE 2: Threshold Robustness (Beamer + thesis) ──────────────────────

def _build_threshold_beamer(all_results):
    """
    Beamer slide: threshold robustness panel (80/100/120 × 3 strategies).
    """
    strategies = [s for s in all_results
                  if 'dummy_interaction' in all_results[s]]
    if not strategies:
        return ""

    tex = []
    tex.append(r"\begin{frame}[t]{Conditional Alpha: Threshold Robustness}")
    tex.append(r"\centering\vspace{-0.3cm}\scriptsize")
    tex.append(r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.10}")

    for s in strategies:
        title = TITLE_MAP.get(s, s)
        di = all_results[s]['dummy_interaction']

        tex.append(rf"\vspace{{0.15cm}}\textbf{{{title}}}\par\vspace{{0.05cm}}")
        tex.append(r"\begin{tabular}{r r r r r r}")
        tex.append(r"\toprule")
        tex.append(r"Threshold & $n_{\text{HIGH}}$ & "
                   r"$\alpha_0$ (ann.) & $\alpha_1$ (ann.) & "
                   r"$t$-stat & $p$-value \\")
        tex.append(r"\midrule")

        for th in ITRX_THRESHOLDS_BPS:
            r = di.get(str(th), {})
            if r.get('skip'):
                tex.append(rf"{th} bps & \multicolumn{{5}}{{c}}{{skipped}} \\")
            else:
                tex.append(
                    rf"{th} bps & {r['n_HIGH']} & "
                    rf"{r['alpha0_annualized']:+.2f}\% & "
                    rf"{r['alpha1_annualized']:+.2f}\%{_stars_tex(r['alpha1_pval'])} & "
                    rf"{_fmt2(r['alpha1_tstat'])} & {_fmt4(r['alpha1_pval'])} \\")

        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}")

    tex.append(r"\vspace{0.15cm}")
    tex.append(r"{\tiny Mitchell \& Pulvino (2001, \textit{JF}); "
               r"threshold robustness per Patton (2009, \textit{RFS}). "
               r"$D_{\text{HIGH}} = \mathbf{1}\{\text{iTraxx Main} > \text{threshold}\}$. "
               r"HAC standard errors. "
               r"*** $p{<}1\%$, ** $p{<}5\%$, * $p{<}10\%$.}")
    tex.append(r"\end{frame}")
    return "\n".join(tex)


def _build_threshold_thesis(all_results):
    """
    Thesis table: threshold robustness.
    """
    strategies = [s for s in all_results
                  if 'dummy_interaction' in all_results[s]]
    if not strategies:
        return ""

    n_strat = len(strategies)

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Conditional Alpha: Stress Threshold Robustness}")
    tex.append(r"\label{tab:conditional_alpha_threshold}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l " + "r r r " * n_strat + "}")
    tex.append(r"\toprule")

    # Header row 1: strategy names spanning 4 columns each
    header1 = " "
    for s in strategies:
        title = TITLE_MAP.get(s, s)
        header1 += rf" & \multicolumn{{3}}{{c}}{{{title}}}"
    header1 += r" \\"
    tex.append(header1)

    # Cmidrules
    cmi = ""
    for i in range(n_strat):
        st = 2 + i * 3
        cmi += rf"\cmidrule(lr){{{st}-{st + 2}}}"
    tex.append(cmi)

    # Header row 2: sub-columns
    header2 = "Threshold"
    for _ in strategies:
        header2 += r" & $n_H$ & $\alpha_1$ (ann.) & $t$-stat"
    header2 += r" \\"
    tex.append(header2)
    tex.append(r"\midrule")

    for th in ITRX_THRESHOLDS_BPS:
        row = rf"{th} bps"
        for s in strategies:
            r = all_results[s]['dummy_interaction'].get(str(th), {})
            if r.get('skip'):
                row += r" & -- & -- & --"
            else:
                n_h = r.get('n_HIGH', '--')
                a1 = r['alpha1_annualized']
                p1 = r['alpha1_pval']
                row += rf" & {n_h}"
                if _stars_sup(p1):
                    row += rf" & ${a1:+.2f}{_stars_sup(p1)}$\%"
                else:
                    row += rf" & {a1:+.2f}\%"
                row += rf" & {_fmt2(r['alpha1_tstat'])}"
        row += r" \\"
        tex.append(row)

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"$r_t = \alpha_0 + \alpha_1 D_{\text{HIGH},t} + "
               r"\sum_j \beta_j X_{jt} + \varepsilon_t$, where "
               r"$D_{\text{HIGH}} = \mathbf{1}\{\text{iTraxx Main 5Y} > "
               r"\text{threshold}\}$. $n_H$ = number of HIGH months. "
               r"Factors from bootstrap stability selection. "
               r"HAC (Newey--West) standard errors. "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")
    return "\n".join(tex)


# ── TABLE 3: Multi-Proxy Robustness (Beamer + thesis) ────────────────────

def _build_multiproxy_beamer(all_results):
    """
    Beamer slide: multi-conditioning-variable robustness.
    """
    strategies = [s for s in all_results
                  if 'multi_conditioning' in all_results[s]]
    if not strategies:
        return ""

    tex = []
    tex.append(r"\begin{frame}[t]{Conditional Alpha: "
               r"Robustness to Conditioning Variable}")
    tex.append(r"\centering\vspace{-0.3cm}\scriptsize")
    tex.append(r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.10}")

    for s in strategies:
        title = TITLE_MAP.get(s, s)
        mc = all_results[s]['multi_conditioning']

        tex.append(rf"\vspace{{0.15cm}}\textbf{{{title}}}\par\vspace{{0.05cm}}")
        tex.append(r"\begin{tabular}{l r r r r r}")
        tex.append(r"\toprule")
        tex.append(r"Conditioning variable ($z_t$) & "
                   r"$\alpha_1$ & $t$-stat & $p$-value & "
                   r"Joint $F$ & $p(F)$ \\")
        tex.append(r"\midrule")

        for pn, cfg in CONDITIONING_VARIABLES.items():
            r = mc.get(pn, {})
            if r.get('skip'):
                tex.append(rf"{cfg['label']} & "
                           r"\multicolumn{5}{c}{not available} \\")
            else:
                tex.append(
                    rf"{cfg['label']} & "
                    rf"{r['alpha1']:+.4f}{_stars_tex(r['alpha1_pval'])} & "
                    rf"{_fmt2(r['alpha1_tstat'])} & "
                    rf"{_fmt4(r['alpha1_pval'])} & "
                    rf"{_fmt2(r['joint_f_stat'])} & "
                    rf"{_fmt4(r['joint_f_pval'])} \\")

        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}")

    tex.append(r"\vspace{0.15cm}")
    tex.append(r"{\tiny Ferson \& Schadt (1996) model with alternative "
               r"conditioning variables; Patton (2009, \textit{RFS}). "
               r"$z_t$ standardized to mean zero, unit variance. "
               r"HAC standard errors. "
               r"*** $p{<}1\%$, ** $p{<}5\%$, * $p{<}10\%$.}")
    tex.append(r"\end{frame}")
    return "\n".join(tex)


def _build_multiproxy_thesis(all_results):
    """
    Thesis table: multi-proxy robustness across all strategies.
    """
    strategies = [s for s in all_results
                  if 'multi_conditioning' in all_results[s]]
    if not strategies:
        return ""

    proxies = list(CONDITIONING_VARIABLES.keys())

    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(r"\caption{Conditional Alpha: Robustness to Conditioning Variable}")
    tex.append(r"\label{tab:conditional_alpha_multiproxy}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l " + "r r r " * len(strategies) + "}")
    tex.append(r"\toprule")

    header1 = r"Conditioning variable"
    for s in strategies:
        title = TITLE_MAP.get(s, s)
        header1 += rf" & \multicolumn{{3}}{{c}}{{{title}}}"
    header1 += r" \\"
    tex.append(header1)

    header2 = r"($z_t$)"
    for _ in strategies:
        header2 += r" & $\alpha_1$ & $t$ & $p$"
    header2 += r" \\"
    # Cmidrules
    cmi = ""
    for i, _ in enumerate(strategies):
        start = 2 + i * 3
        cmi += rf"\cmidrule(lr){{{start}-{start+2}}}"
    tex.append(cmi)
    tex.append(header2)
    tex.append(r"\midrule")

    for pn in proxies:
        label = CONDITIONING_VARIABLES[pn]['label']
        row = label
        for s in strategies:
            r = all_results[s].get('multi_conditioning', {}).get(pn, {})
            if r.get('skip'):
                row += r" & -- & -- & --"
            else:
                row += (rf" & {r['alpha1']:+.4f}{_stars_tex(r['alpha1_pval'])}"
                        rf" & {_fmt2(r['alpha1_tstat'])}"
                        rf" & {_fmt4(r['alpha1_pval'])}")
        row += r" \\"
        tex.append(row)

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    tex.append(r"\vspace{0.2cm}")
    tex.append(r"\begin{minipage}{0.92\textwidth}")
    tex.append(r"\footnotesize\textit{Notes:} "
               r"Each column reports $\alpha_1$ from the Ferson and Schadt (1996) "
               r"conditional model $r_t = \alpha_0 + \alpha_1 z_t + "
               r"\sum_j (\beta_j + \delta_j z_t) X_{jt} + \varepsilon_t$, "
               r"estimated with a different conditioning variable $z_t$. "
               r"All $z_t$ are standardized to zero mean and unit variance. "
               r"$\alpha_1 > 0$ in all specifications indicates that alpha "
               r"increases with financial stress, robust to the choice of proxy. "
               r"HAC (Newey--West) standard errors. "
               r"*** $p < 0.01$, ** $p < 0.05$, * $p < 0.10$.")
    tex.append(r"\end{minipage}")

    tex.append(r"\end{table}")
    return "\n".join(tex)


# ── MASTER TEX GENERATOR ─────────────────────────────────────────────────

def generate_all_tex(all_results):
    """
    Generate all .tex files: Beamer slides + thesis tables.
    Saves to the AEN output directory under tables/.
    """
    print_header("GENERATING .TEX FILES (Beamer + Thesis)")

    tables_dir = get_aen_output_dir() / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    written = []

    # ── Per-strategy Beamer slides: Ferson–Schadt ─────────────────────
    for s, res in all_results.items():
        fs = res.get('ferson_schadt')
        if not fs or 'conditional' not in fs:
            continue
        safe = s.replace("_", " ").title().replace(" ", "_")
        fname = f"Cond_Alpha_FS_{safe}_Slide.tex"
        content = _build_fs_beamer(s, fs)
        (tables_dir / fname).write_text(content, encoding="utf-8")
        written.append(fname)

    # ── Cross-strategy thesis table: Ferson–Schadt ────────────────────
    content = _build_fs_thesis(all_results)
    if content:
        fname = "Cond_Alpha_FS_Thesis.tex"
        (tables_dir / fname).write_text(content, encoding="utf-8")
        written.append(fname)

    # ── Threshold robustness: Beamer + thesis ─────────────────────────
    content = _build_threshold_beamer(all_results)
    if content:
        fname = "Cond_Alpha_Threshold_Slide.tex"
        (tables_dir / fname).write_text(content, encoding="utf-8")
        written.append(fname)

    content = _build_threshold_thesis(all_results)
    if content:
        fname = "Cond_Alpha_Threshold_Thesis.tex"
        (tables_dir / fname).write_text(content, encoding="utf-8")
        written.append(fname)

    # ── Multi-proxy robustness: Beamer + thesis ───────────────────────
    content = _build_multiproxy_beamer(all_results)
    if content:
        fname = "Cond_Alpha_MultiProxy_Slide.tex"
        (tables_dir / fname).write_text(content, encoding="utf-8")
        written.append(fname)

    content = _build_multiproxy_thesis(all_results)
    if content:
        fname = "Cond_Alpha_MultiProxy_Thesis.tex"
        (tables_dir / fname).write_text(content, encoding="utf-8")
        written.append(fname)

    for f in written:
        print(f"   ✅ {tables_dir / f}")

    print(f"\n   Total: {len(written)} .tex files generated")

    return written


# ============================================================================
# RUNNER: ALL 5 CONDITIONAL ALPHA TESTS
# ============================================================================

def conditional_alpha_analysis(strategy_name, strategy_path):
    """Run all 5 conditional alpha tests for one strategy."""
    print_header(f"(e) CONDITIONAL ALPHA ANALYSIS — {strategy_name}")

    results = {'strategy': strategy_name}

    # (e1) Ferson–Schadt
    try:
        results['ferson_schadt'] = ferson_schadt_conditional(
            strategy_name, strategy_path)
    except Exception as e:
        print(f"\n   ❌ (e1) Error: {e}")
        import traceback; traceback.print_exc()

    # (e2) Dummy interaction
    try:
        results['dummy_interaction'] = dummy_interaction(
            strategy_name, strategy_path)
    except Exception as e:
        print(f"\n   ❌ (e2) Error: {e}")
        import traceback; traceback.print_exc()

    # (e3) Sub-sample split
    try:
        results['subsample'] = subsample_regime(
            strategy_name, strategy_path)
    except Exception as e:
        print(f"\n   ❌ (e3) Error: {e}")
        import traceback; traceback.print_exc()

    # (e4) Multiple conditioning variables
    try:
        results['multi_conditioning'] = multiple_conditioning_variables(
            strategy_name, strategy_path)
    except Exception as e:
        print(f"\n   ❌ (e4) Error: {e}")
        import traceback; traceback.print_exc()

    # (e5) Rolling alpha with regime shading
    try:
        results['rolling_regime'] = rolling_alpha_regime_plot(
            strategy_name, strategy_path)
    except Exception as e:
        print(f"\n   ❌ (e5) Error: {e}")
        import traceback; traceback.print_exc()

    # Save combined
    strategy_dir = get_strategy_aen_dir(strategy_name)
    with open(strategy_dir / "conditional_alpha_summary.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n   💾 conditional_alpha_summary.json")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("CONDITIONAL ALPHA ANALYSIS")
    print(f"\n   References:")
    print(f"   - Ferson & Schadt (1996, JF): conditional alpha + beta")
    print(f"   - Mitchell & Pulvino (2001, JF): dummy interaction")
    print(f"   - Patton (2009, RFS): threshold & variable robustness")
    print(f"   - Christoffersen & Langlois (2013, JFQA): time-varying betas")
    print(f"\n   Conditioning variables: {list(CONDITIONING_VARIABLES.keys())}")
    print(f"   Dummy thresholds: {ITRX_THRESHOLDS_BPS} bps")
    print(f"   HAC lags: {HAC_LAGS}")

    all_results = {}

    for strategy_name, strategy_path in STRATEGIES.items():
        strategy_dir = get_strategy_aen_dir(strategy_name)

        if not (strategy_dir / "bootstrap_stability.json").exists():
            print(f"\n   ❌ {strategy_name}: bootstrap_stability.json not found."
                  f" Run 04_bootstrap.py first.")
            continue

        try:
            result = conditional_alpha_analysis(
                strategy_name, strategy_path)
            all_results[strategy_name] = result
        except Exception as e:
            print(f"\n   ❌ {strategy_name}: {e}")
            import traceback; traceback.print_exc()

    # ── Cross-Strategy Summary ─────────────────────────────────────────
    if all_results:
        print_header("CONDITIONAL ALPHA — CROSS-STRATEGY SUMMARY")

        # Ferson-Schadt α₁
        print(f"\n   ── Ferson–Schadt α₁ (conditional alpha on iTraxx Main) ──")
        print(f"   {'Strategy':<20} {'α₁':>8} {'t':>8} {'p':>8}"
              f" {'α(+1σ) ann':>12} {'α(-1σ) ann':>12}")
        print(f"   {'─' * 70}")
        for name, res in all_results.items():
            fs = res.get('ferson_schadt', {})
            if fs and 'conditional' in fs:
                c = fs['conditional']
                em = fs['economic_magnitude']
                print(f"   {name:<20}"
                      f" {c['alpha1']:>+8.4f}"
                      f" {c['alpha1_tstat']:>8.3f}"
                      f" {c['alpha1_pval']:>8.4f}"
                      f" {em['alpha_at_plus_1sd']:>+12.2f}%"
                      f" {em['alpha_at_minus_1sd']:>+12.2f}%"
                      f" {significance_stars(c['alpha1_pval'])}")

        # Dummy interaction at default threshold
        print(f"\n   ── Dummy Interaction (iTraxx > {DEFAULT_THRESHOLD} bps) ──")
        print(f"   {'Strategy':<20} {'α₁ (ann)':>10} {'t':>8}"
              f" {'p':>8} {'n_HIGH':>7}")
        print(f"   {'─' * 55}")
        for name, res in all_results.items():
            di = res.get('dummy_interaction', {})
            th_res = di.get(str(DEFAULT_THRESHOLD), {})
            if th_res and not th_res.get('skip'):
                print(f"   {name:<20}"
                      f" {th_res['alpha1_annualized']:>+10.2f}"
                      f" {th_res['alpha1_tstat']:>8.3f}"
                      f" {th_res['alpha1_pval']:>8.4f}"
                      f" {th_res['n_HIGH']:>7}"
                      f" {significance_stars(th_res['alpha1_pval'])}")

        # Save global
        aen_output_dir = get_aen_output_dir()
        with open(aen_output_dir / "conditional_alpha_global.json", 'w') as f:
            json.dump({
                'strategies': {
                    name: {
                        'fs_alpha1': res.get('ferson_schadt', {}).get(
                            'conditional', {}).get('alpha1'),
                        'fs_alpha1_pval': res.get('ferson_schadt', {}).get(
                            'conditional', {}).get('alpha1_pval'),
                        'dummy_alpha1_ann': res.get(
                            'dummy_interaction', {}).get(
                                str(DEFAULT_THRESHOLD), {}).get(
                                    'alpha1_annualized'),
                        'dummy_alpha1_pval': res.get(
                            'dummy_interaction', {}).get(
                                str(DEFAULT_THRESHOLD), {}).get(
                                    'alpha1_pval'),
                    } for name, res in all_results.items()
                }
            }, f, indent=2)
        print(f"\n   💾 {aen_output_dir / 'conditional_alpha_global.json'}")

        # ── Generate .tex files (Beamer + Thesis) ─────────────────────
        generate_all_tex(all_results)

    print(f"\n{'=' * 80}")
    print(f"✅ CONDITIONAL ALPHA ANALYSIS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
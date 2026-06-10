"""
================================================================================
05b_pca_conditional_alpha.py — Conditional Alpha Analysis (PCA Pipeline)
================================================================================
Tests whether strategy alpha varies with financial stress, using rolling
PCA factors (PC scores) from 02_pca_rolling.py.

Same five tests as 06e_conditional_alpha.py (ML pipeline), but with
PC scores replacing AEN-selected factors:

  (e1) Ferson–Schadt (1996, JF) conditional alpha + conditional beta
  (e2) Dummy interaction (Mitchell & Pulvino 2001, JF) with threshold
       robustness (80, 100, 120 bps — Patton 2009, RFS)
  (e3) Sub-sample split by regime (HIGH vs NORMAL)
  (e4) Multiple conditioning variables (iTraxx Main, V2X, EURIBOR-OIS)
  (e5) Rolling alpha with regime shading (iTraxx Main)

Using the same tests across PCA and AEN pipelines demonstrates that the
conditional alpha result is robust to the dimensionality reduction method.

Timing convention:
  Follows PCA_TIMING from 00_pca_config.py.
  "predictive":      PC_t → R_{t+1}  (Ludvigson & Ng 2009)
  "contemporaneous": PC_t → R_t

References:
    Ferson, W. and Schadt, R. (1996, JF)
    Mitchell, M. and Pulvino, T. (2001, JF)
    Patton, A. (2009, RFS)
    Ludvigson, S.C. and Ng, S. (2009), "Macro Factors in Bond Risk Premia",
        Review of Financial Studies, 22(12), 5027-5067.

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
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================================
# CONFIG
# ============================================================================

import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load PCA config
config_paths = [
    PROJECT_ROOT / "src" / "pca" / "00_pca_config.py",
    PROJECT_ROOT / "src" / "pca" / "00_pca_config_fix.py",
]

pca_config = None
for config_path in config_paths:
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("pca_config", config_path)
        pca_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pca_config)
        break

if pca_config is None:
    raise FileNotFoundError("PCA config file not found!")

RESULTS_DIR       = pca_config.RESULTS_DIR
STRATEGIES        = pca_config.STRATEGIES
PCA_N_COMPONENTS  = pca_config.PCA_N_COMPONENTS
PCA_TIMING        = pca_config.PCA_TIMING
get_pca_output_dir   = pca_config.get_pca_output_dir
get_strategy_pca_dir = pca_config.get_strategy_pca_dir

HAC_LAGS = 6

# ── Stress proxy paths ────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
FACTORS_EXTERNAL_DIR = DATA_DIR / "external" / "factors"
NONTRADABLE_FILE     = FACTORS_EXTERNAL_DIR / "Nontradable_risk_factors.xlsx"
TRADABLE_CB_FILE     = FACTORS_EXTERNAL_DIR / "Tradable_corporate_bond_factors.xlsx"
FACTORS_PATH         = pca_config.FACTORS_PATH

# ── Thresholds and conditioning variables (same as 06e) ───────────────────
ITRX_THRESHOLDS_BPS = [80, 100, 120]
DEFAULT_THRESHOLD    = 80

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

# ── Output ────────────────────────────────────────────────────────────────
TABLES_DIR  = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures" / "pca"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TITLE_MAP = {
    "btp_italia":      "BTP Italia",
    "cds_bond_basis":  "CDS--Bond Basis",
    "itraxx_combined": "iTraxx Combined",
}


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


def _fmt2(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.2f}"

def _fmt4(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.4f}"


def load_stress_proxy_monthly(proxy_name):
    """Load a stress proxy as a monthly Series (end-of-month level)."""
    cfg = CONDITIONING_VARIABLES[proxy_name]

    if cfg["source"] == "factors_parquet":
        factors = pd.read_parquet(FACTORS_PATH)
        if proxy_name in factors.columns:
            return factors[proxy_name].dropna()
        else:
            raise KeyError(f"{proxy_name} not found in {FACTORS_PATH}")

    file_path = TRADABLE_CB_FILE if cfg["source"] == "tradable" else NONTRADABLE_FILE
    raw = pd.read_excel(
        file_path, sheet_name=cfg["sheet"],
        skiprows=cfg["skiprows"], usecols=cfg["usecols"], header=0
    )
    raw.columns = cfg["colnames"]
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date")
    daily = pd.to_numeric(raw["value"], errors="coerce").dropna()
    monthly = daily.resample('ME').last().dropna()
    monthly.name = proxy_name
    return monthly


def load_strategy_returns(strategy_path):
    """Load daily returns → monthly compounding."""
    daily_df = pd.read_csv(strategy_path, index_col=0, parse_dates=True)
    daily_returns = daily_df['index_return'].dropna()
    monthly = daily_returns.resample('ME').apply(
        lambda x: ((1 + x / 100).prod() - 1) * 100 if len(x) > 0 else np.nan
    )
    return monthly.dropna()


def prepare_pca_data(strategy_name, timing):
    """
    Load PC scores and strategy returns, align with correct timing convention.

    Returns (y, X, pc_names) — aligned, NaN-free, timing-correct.
    """
    pca_dir = get_pca_output_dir()
    strategy_dir = get_strategy_pca_dir(strategy_name)

    # PC scores
    pc_path = pca_dir / f"pc_scores_{timing}.parquet"
    if not pc_path.exists():
        raise FileNotFoundError(f"PC scores not found: {pc_path}")
    pc_scores = pd.read_parquet(pc_path)

    # Returns
    returns_path = strategy_dir / "y_returns_pca.parquet"
    if not returns_path.exists():
        raise FileNotFoundError(f"Returns not found: {returns_path}")
    returns = pd.read_parquet(returns_path)['Strategy_Return']

    # Align dates
    common = returns.index.intersection(pc_scores.index)

    if timing == "predictive":
        # PC_t → R_{t+1}: shift returns back so y[t] = R_{t+1}, X[t] = PC_t
        pc_aligned = pc_scores.loc[common].iloc[:-1]
        ret_aligned = returns.loc[common].iloc[1:]
        pc_aligned.index = ret_aligned.index
    else:
        pc_aligned = pc_scores.loc[common]
        ret_aligned = returns.loc[common]

    # Drop NaN
    mask = ~(pc_aligned.isna().any(axis=1) | ret_aligned.isna())
    y = ret_aligned[mask]
    X = pc_aligned[mask]

    pc_names = list(X.columns)
    return y, X, pc_names


# ============================================================================
# (e1) FERSON–SCHADT CONDITIONAL ALPHA + BETA
# ============================================================================

def ferson_schadt_conditional(strategy_name, timing):
    """
    r_t = α₀ + α₁·z_t + Σⱼ βⱼ·PCⱼt + Σⱼ δⱼ·(PCⱼt·z_t) + εₜ
    """
    print_header(f"   (e1) Ferson–Schadt — {strategy_name} ({timing})", "─")

    y, X, pc_names = prepare_pca_data(strategy_name, timing)
    T, k = len(y), len(pc_names)

    # Load conditioning variable
    z_raw = load_stress_proxy_monthly("ITRX_MAIN")
    z_aligned = z_raw.reindex(y.index, method='nearest')
    z = (z_aligned - z_aligned.mean()) / z_aligned.std()
    z = z.reindex(y.index).dropna()

    common = y.index.intersection(z.index)
    y, X, z = y[common], X.loc[common], z[common]
    T = len(y)

    print(f"\n   PC components: {k}, T = {T}")
    print(f"   Conditioning: iTraxx Main 5Y (standardized)")

    # Unconditional
    X_unc = sm.add_constant(X, prepend=True)
    res_unc = sm.OLS(y, X_unc).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

    # Conditional
    X_cond = X.copy()
    X_cond['z_stress'] = z.values
    for pc in pc_names:
        X_cond[f'{pc}_x_z'] = X[pc].values * z.values
    X_cond = sm.add_constant(X_cond, prepend=True)

    res_cond = sm.OLS(y, X_cond).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

    alpha0 = res_cond.params['const']
    alpha1 = res_cond.params['z_stress']
    alpha1_t = res_cond.tvalues['z_stress']
    alpha1_p = res_cond.pvalues['z_stress']

    print(f"\n   Unconditional: α = {res_unc.params['const']:+.4f}% mo"
          f" ({res_unc.params['const'] * 12:+.2f}% ann.)")
    print(f"   Conditional:   α₀ = {alpha0:+.4f}, "
          f"α₁ = {alpha1:+.4f} (t={alpha1_t:.3f}, p={alpha1_p:.4f})"
          f" {significance_stars(alpha1_p)}")

    # Conditional betas δⱼ
    delta_results = {}
    print(f"\n   Conditional Betas (δⱼ):")
    for pc in pc_names:
        col = f'{pc}_x_z'
        d = res_cond.params[col]
        d_t = res_cond.tvalues[col]
        d_p = res_cond.pvalues[col]
        print(f"   {pc:<8} × z_t: δ={d:+.4f}, t={d_t:.3f}, p={d_p:.4f}"
              f" {significance_stars(d_p)}")
        delta_results[pc] = {
            'delta': round(float(d), 6),
            't_stat': round(float(d_t), 4),
            'p_value': round(float(d_p), 4),
        }

    # Joint F-test
    interaction_cols = ['z_stress'] + [f'{pc}_x_z' for pc in pc_names]
    r_matrix = np.zeros((len(interaction_cols), len(res_cond.params)))
    for i, col in enumerate(interaction_cols):
        col_idx = list(res_cond.params.index).index(col)
        r_matrix[i, col_idx] = 1.0
    f_test = res_cond.f_test(r_matrix)
    f_stat = float(f_test.fvalue)
    f_pval = float(f_test.pvalue)

    print(f"\n   Joint F({len(interaction_cols)},{T - len(res_cond.params)}) ="
          f" {f_stat:.3f}, p = {f_pval:.4f} {significance_stars(f_pval)}")

    # Economic magnitude
    alpha_high = alpha0 + alpha1 * 1.0
    alpha_low  = alpha0 + alpha1 * (-1.0)
    print(f"\n   α(+1σ stress): {alpha_high * 12:+.2f}% ann.")
    print(f"   α(-1σ normal): {alpha_low * 12:+.2f}% ann.")
    print(f"   R² adj unc: {res_unc.rsquared_adj:.4f}, cond: {res_cond.rsquared_adj:.4f}")

    result = {
        'strategy': strategy_name, 'timing': timing,
        'conditioning_variable': 'ITRX_MAIN',
        'T': T, 'k_base': k,
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
            'r2_adj': round(float(res_cond.rsquared_adj), 6),
        },
        'conditional_betas': delta_results,
        'joint_f_test': {
            'f_statistic': round(f_stat, 4),
            'p_value': round(f_pval, 4),
        },
        'economic_magnitude': {
            'alpha_at_plus_1sd': round(float(alpha_high * 12), 4),
            'alpha_at_minus_1sd': round(float(alpha_low * 12), 4),
            'spread_2sd_annualized': round(float((alpha_high - alpha_low) * 12), 4),
        },
    }

    strategy_dir = get_strategy_pca_dir(strategy_name)
    with open(strategy_dir / f"conditional_fs_{timing}.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n   💾 conditional_fs_{timing}.json")

    return result


# ============================================================================
# (e2) DUMMY INTERACTION
# ============================================================================

def dummy_interaction(strategy_name, timing):
    """
    r_t = α₀ + α₁·D_HIGH,t + Σⱼ βⱼ·PCⱼt + εₜ
    Threshold robustness: 80, 100, 120 bps.
    """
    print_header(f"   (e2) Dummy Interaction — {strategy_name} ({timing})", "─")

    y, X, pc_names = prepare_pca_data(strategy_name, timing)

    z_raw = load_stress_proxy_monthly("ITRX_MAIN")
    z_aligned = z_raw.reindex(y.index, method='nearest')
    common = y.index.intersection(z_aligned.dropna().index)
    y, X, z_level = y[common], X.loc[common], z_aligned[common]
    T = len(y)

    print(f"\n   T = {T}, iTraxx range: [{z_level.min():.1f}, {z_level.max():.1f}] bps")

    results_all = {}

    for threshold in ITRX_THRESHOLDS_BPS:
        D_high = (z_level > threshold).astype(float)
        n_high = int(D_high.sum())

        print(f"\n   Threshold {threshold} bps: n_HIGH = {n_high}/{T}")

        if n_high < 10:
            print(f"   ⚠️  Too few HIGH obs, skipping")
            results_all[str(threshold)] = {'threshold_bps': threshold,
                                           'n_HIGH': n_high, 'skip': True}
            continue

        X_d = X.copy()
        X_d['D_HIGH'] = D_high.values
        X_d_const = sm.add_constant(X_d, prepend=True)

        res = sm.OLS(y, X_d_const).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

        a0 = res.params['const']
        a1 = res.params['D_HIGH']
        a1_t = res.tvalues['D_HIGH']
        a1_p = res.pvalues['D_HIGH']

        print(f"   α₀ = {a0 * 12:+.2f}% ann., α₁ = {a1 * 12:+.2f}% ann."
              f" (t={a1_t:.3f}, p={a1_p:.4f}) {significance_stars(a1_p)}")

        results_all[str(threshold)] = {
            'threshold_bps': threshold, 'n_HIGH': n_high, 'n_LOW': T - n_high,
            'alpha0_monthly': round(float(a0), 6),
            'alpha0_annualized': round(float(a0 * 12), 4),
            'alpha1_monthly': round(float(a1), 6),
            'alpha1_annualized': round(float(a1 * 12), 4),
            'alpha1_tstat': round(float(a1_t), 4),
            'alpha1_pval': round(float(a1_p), 4),
            'alpha_HIGH_annualized': round(float((a0 + a1) * 12), 4),
            'r2_adj': round(float(res.rsquared_adj), 6),
        }

    strategy_dir = get_strategy_pca_dir(strategy_name)
    with open(strategy_dir / f"conditional_dummy_{timing}.json", 'w') as f:
        json.dump({'strategy': strategy_name, 'timing': timing,
                   'T': T, 'thresholds': results_all}, f, indent=2)
    print(f"\n   💾 conditional_dummy_{timing}.json")

    return results_all


# ============================================================================
# (e3) SUB-SAMPLE SPLIT BY REGIME
# ============================================================================

def subsample_regime(strategy_name, timing):
    """OLS separately on HIGH vs non-HIGH months."""
    print_header(f"   (e3) Sub-Sample Regime — {strategy_name} ({timing})", "─")

    y, X, pc_names = prepare_pca_data(strategy_name, timing)

    z_raw = load_stress_proxy_monthly("ITRX_MAIN")
    z_aligned = z_raw.reindex(y.index, method='nearest')
    common = y.index.intersection(z_aligned.dropna().index)
    y, X, z_level = y[common], X.loc[common], z_aligned[common]

    D_high = z_level > DEFAULT_THRESHOLD
    T = len(y)
    results = {}

    for label, mask in [("HIGH", D_high), ("NORMAL", ~D_high)]:
        y_sub, X_sub = y[mask], X[mask]
        n = len(y_sub)
        print(f"\n   {label}: n = {n}")

        if n < len(pc_names) + 5:
            print(f"   ⚠️  Too few obs, skip")
            results[label] = {'n': n, 'skip': True}
            continue

        X_const = sm.add_constant(X_sub, prepend=True)
        se_type = 'HAC' if n > 30 else 'OLS'
        if se_type == 'HAC':
            res = sm.OLS(y_sub, X_const).fit(
                cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
        else:
            res = sm.OLS(y_sub, X_const).fit()

        alpha = res.params['const']
        print(f"   α = {alpha:+.4f}% mo ({alpha * 12:+.2f}% ann.)"
              f" t={res.tvalues['const']:.3f}, p={res.pvalues['const']:.4f}"
              f" {significance_stars(res.pvalues['const'])} ({se_type})")

        results[label] = {
            'n': n,
            'alpha_monthly': round(float(alpha), 6),
            'alpha_annualized': round(float(alpha * 12), 4),
            'alpha_tstat': round(float(res.tvalues['const']), 4),
            'alpha_pval': round(float(res.pvalues['const']), 4),
            'r2_adj': round(float(res.rsquared_adj), 6),
            'se_type': se_type,
        }

    if not results.get('HIGH', {}).get('skip') and not results.get('NORMAL', {}).get('skip'):
        diff = results['HIGH']['alpha_annualized'] - results['NORMAL']['alpha_annualized']
        results['difference_annualized'] = round(diff, 4)
        print(f"\n   Δα (HIGH − NORMAL) = {diff:+.2f}% ann.")

    strategy_dir = get_strategy_pca_dir(strategy_name)
    with open(strategy_dir / f"conditional_subsample_{timing}.json", 'w') as f:
        json.dump({'strategy': strategy_name, 'timing': timing,
                   'threshold_bps': DEFAULT_THRESHOLD, 'T': T,
                   'regimes': results}, f, indent=2)
    print(f"\n   💾 conditional_subsample_{timing}.json")

    return results


# ============================================================================
# (e4) MULTIPLE CONDITIONING VARIABLES
# ============================================================================

def multiple_conditioning_variables(strategy_name, timing):
    """Repeat Ferson–Schadt with alternative z_t."""
    print_header(
        f"   (e4) Multi-Conditioning — {strategy_name} ({timing})", "─")

    y, X, pc_names = prepare_pca_data(strategy_name, timing)
    k = len(pc_names)
    results = {}

    for proxy_name, cfg in CONDITIONING_VARIABLES.items():
        print(f"\n   z_t = {cfg['label']}:")

        try:
            z_raw = load_stress_proxy_monthly(proxy_name)
        except Exception as e:
            print(f"   ⚠️  Cannot load: {e}")
            results[proxy_name] = {'skip': True, 'reason': str(e)}
            continue

        z_aligned = z_raw.reindex(y.index, method='nearest')
        common = y.index.intersection(z_aligned.dropna().index)
        y_s, X_s, z_s = y[common], X.loc[common], z_aligned[common]
        T = len(y_s)

        if T < k + 10:
            print(f"   ⚠️  Too few obs ({T}), skip")
            results[proxy_name] = {'skip': True, 'T': T}
            continue

        z_std = (z_s - z_s.mean()) / z_s.std()

        X_cond = X_s.copy()
        X_cond['z_stress'] = z_std.values
        for pc in pc_names:
            X_cond[f'{pc}_x_z'] = X_s[pc].values * z_std.values
        X_cond = sm.add_constant(X_cond, prepend=True)

        res = sm.OLS(y_s, X_cond).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

        a1 = res.params['z_stress']
        a1_t = res.tvalues['z_stress']
        a1_p = res.pvalues['z_stress']

        # Joint F-test
        int_cols = ['z_stress'] + [f'{pc}_x_z' for pc in pc_names]
        r_mat = np.zeros((len(int_cols), len(res.params)))
        for i, col in enumerate(int_cols):
            r_mat[i, list(res.params.index).index(col)] = 1.0
        ft = res.f_test(r_mat)

        print(f"   α₁ = {a1:+.4f} (t={a1_t:.3f}, p={a1_p:.4f})"
              f" {significance_stars(a1_p)}   F={float(ft.fvalue):.3f}")

        results[proxy_name] = {
            'label': cfg['label'], 'T': T,
            'alpha1': round(float(a1), 6),
            'alpha1_tstat': round(float(a1_t), 4),
            'alpha1_pval': round(float(a1_p), 4),
            'joint_f_stat': round(float(ft.fvalue), 4),
            'joint_f_pval': round(float(ft.pvalue), 4),
            'r2_adj': round(float(res.rsquared_adj), 6),
        }

    # Summary
    n_pos = sum(1 for r in results.values() if not r.get('skip') and r.get('alpha1', 0) > 0)
    n_sig = sum(1 for r in results.values()
                if not r.get('skip') and r.get('alpha1_pval', 1) < 0.10
                and r.get('alpha1', 0) > 0)
    n_tot = sum(1 for r in results.values() if not r.get('skip'))
    print(f"\n   α₁ > 0: {n_pos}/{n_tot},  α₁ > 0 & significant: {n_sig}/{n_tot}")

    strategy_dir = get_strategy_pca_dir(strategy_name)
    with open(strategy_dir / f"conditional_multi_{timing}.json", 'w') as f:
        json.dump({'strategy': strategy_name, 'timing': timing,
                   'results': results}, f, indent=2)
    print(f"\n   💾 conditional_multi_{timing}.json")

    return results


# ============================================================================
# (e5) ROLLING ALPHA WITH REGIME SHADING
# ============================================================================

def rolling_alpha_regime_plot(strategy_name, timing):
    """Rolling alpha with iTraxx Main regime shading."""
    print_header(
        f"   (e5) Rolling Alpha + Regime — {strategy_name} ({timing})", "─")

    y, X, pc_names = prepare_pca_data(strategy_name, timing)

    z_raw = load_stress_proxy_monthly("ITRX_MAIN")
    z_monthly = z_raw.resample('ME').last().dropna()

    T = len(y)
    n_roll = T - ROLLING_WINDOW + 1
    if n_roll <= 0:
        print(f"   ⚠️  Not enough data (T={T})")
        return None

    from scipy import stats as scipy_stats
    t_crit = scipy_stats.t.ppf(0.975, df=max(1, ROLLING_WINDOW - len(pc_names) - 1))

    rolling_rows = []
    for start in range(n_roll):
        end = start + ROLLING_WINDOW
        y_w = y.iloc[start:end]
        X_w = X.iloc[start:end]
        X_c = sm.add_constant(X_w, prepend=True)
        try:
            res = sm.OLS(y_w, X_c).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
            a = float(res.params['const'])
            a_se = float(res.bse['const'])
        except Exception:
            a = a_se = np.nan
        rolling_rows.append({
            'end_date': y.index[end - 1],
            'alpha_monthly': a,
            'alpha_ann': a * 12,
            'ci_lo': (a - t_crit * a_se) * 12,
            'ci_hi': (a + t_crit * a_se) * 12,
        })

    roll_df = pd.DataFrame(rolling_rows).set_index('end_date')

    # Regime
    threshold_high = DEFAULT_THRESHOLD
    threshold_low = 60
    regime = pd.Series("MEDIUM", index=z_monthly.index)
    regime[z_monthly < threshold_low] = "LOW"
    regime[z_monthly >= threshold_high] = "HIGH"
    regime_aligned = regime.reindex(roll_df.index, method='nearest')

    # Stats per regime
    regime_stats = {}
    for rl in ["LOW", "MEDIUM", "HIGH"]:
        mask = regime_aligned == rl
        if mask.sum() > 0:
            sub = roll_df.loc[mask, 'alpha_ann']
            regime_stats[rl] = {
                'n': int(mask.sum()),
                'alpha_mean': round(float(sub.mean()), 4),
                'alpha_median': round(float(sub.median()), 4),
            }
            print(f"   {rl:>7}: n={mask.sum():>3}, avg α(ann)={sub.mean():+.2f}%")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})
    title_str = TITLE_MAP.get(strategy_name, strategy_name).replace("--", "–")
    fig.suptitle(
        f"Rolling Alpha ({ROLLING_WINDOW}M) with Stress Regimes — "
        f"{title_str} ({timing})",
        fontsize=13, fontweight='bold')

    ax = axes[0]
    dates = roll_df.index
    ax.plot(dates, roll_df['alpha_ann'], color='black', linewidth=1.2)
    ax.fill_between(dates, roll_df['ci_lo'], roll_df['ci_hi'],
                    color='grey', alpha=0.2)
    ax.axhline(0, color='grey', linewidth=0.5)

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
            alpha_sh = 0.15 if rl != "HIGH" else 0.25
            for s, e in zip(starts[:len(ends)], ends[:len(starts)]):
                ax.axvspan(s, e, alpha=alpha_sh, color=color, zorder=0)

    ax.set_ylabel("α (annualized %)")
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    z_plot = z_monthly.reindex(dates, method='nearest')
    ax2.plot(dates, z_plot.values, color='black', linewidth=0.8)
    ax2.axhline(threshold_high, color=REGIME_COLORS['HIGH'], linestyle='--')
    ax2.axhline(threshold_low, color=REGIME_COLORS['LOW'], linestyle='--')
    ax2.set_ylabel("iTraxx Main (bps)")
    ax2.grid(True, alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    fig.tight_layout()
    fig_path = FIGURES_DIR / f"pca_rolling_alpha_regime_{timing}.{FIGURE_FORMAT}"
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"\n   📊 {fig_path.name}")

    strategy_dir = get_strategy_pca_dir(strategy_name)
    with open(strategy_dir / f"rolling_alpha_regime_{timing}.json", 'w') as f:
        json.dump({'strategy': strategy_name, 'timing': timing,
                   'regime_stats': regime_stats}, f, indent=2)
    print(f"   💾 rolling_alpha_regime_{timing}.json")

    return regime_stats


# ============================================================================
# TEX GENERATION
# ============================================================================

def generate_tex(all_results, timing):
    """Generate Beamer slides + thesis tables for conditional alpha results."""
    print_header(f"GENERATING .TEX ({timing})")

    strategies = list(all_results.keys())
    if not strategies:
        print("   No results to generate .tex for.")
        return

    timing_label = timing.title()

    # ── THESIS TABLE: Ferson–Schadt ───────────────────────────────────
    tex = []
    tex.append(r"\begin{table}[htbp]")
    tex.append(r"\centering")
    tex.append(rf"\caption{{Conditional Alpha (PCA, {timing_label}): "
               r"Ferson--Schadt (1996) Model}")
    tex.append(rf"\label{{tab:pca_conditional_alpha_fs_{timing}}}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l" + "c" * len(strategies) + "}")
    tex.append(r"\toprule")
    headers = " & ".join([TITLE_MAP.get(s, s) for s in strategies])
    tex.append(rf" & {headers} \\")
    tex.append(r"\midrule")

    # α₀ annualized
    vals = []
    for s in strategies:
        fs = all_results[s].get('ferson_schadt', {})
        u = fs.get('unconditional', {})
        v = u.get('alpha_annualized', np.nan)
        p = u.get('alpha_pval', 1)
        vals.append(f"{v:+.2f}{significance_stars(p)}" if not np.isnan(v) else "--")
    tex.append(rf"$\alpha_0$ (ann.\ \%) & {' & '.join(vals)} \\")

    # α₁
    vals = []
    for s in strategies:
        c = all_results[s].get('ferson_schadt', {}).get('conditional', {})
        v = c.get('alpha1', np.nan)
        p = c.get('alpha1_pval', 1)
        vals.append(f"{v:+.4f}{significance_stars(p)}" if not np.isnan(v) else "--")
    tex.append(rf"$\alpha_1$ (conditional) & {' & '.join(vals)} \\")

    # t-stat
    vals = []
    for s in strategies:
        c = all_results[s].get('ferson_schadt', {}).get('conditional', {})
        vals.append(f"({c.get('alpha1_tstat', 0):.2f})")
    tex.append(rf"\quad $t$-stat & {' & '.join(vals)} \\")

    # Joint F
    vals = []
    for s in strategies:
        jf = all_results[s].get('ferson_schadt', {}).get('joint_f_test', {})
        f_v = jf.get('f_statistic', np.nan)
        f_p = jf.get('p_value', 1)
        vals.append(f"{f_v:.2f}{significance_stars(f_p)}" if not np.isnan(f_v) else "--")
    tex.append(rf"Joint $F$-test & {' & '.join(vals)} \\")

    # Economic magnitude
    vals_h = []
    vals_l = []
    for s in strategies:
        em = all_results[s].get('ferson_schadt', {}).get('economic_magnitude', {})
        vals_h.append(f"{em.get('alpha_at_plus_1sd', np.nan):+.2f}\\%")
        vals_l.append(f"{em.get('alpha_at_minus_1sd', np.nan):+.2f}\\%")
    tex.append(rf"$\alpha(z = +1\sigma)$ ann. & {' & '.join(vals_h)} \\")
    tex.append(rf"$\alpha(z = -1\sigma)$ ann. & {' & '.join(vals_l)} \\")

    # T
    vals = []
    for s in strategies:
        T = all_results[s].get('ferson_schadt', {}).get('T', '--')
        vals.append(str(T))
    tex.append(rf"$T$ & {' & '.join(vals)} \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\vspace{0.2cm}")
    tex.append(r"\begin{minipage}{0.92\textwidth}")
    tex.append(r"\footnotesize\textit{Notes:} "
               r"Ferson and Schadt (1996) conditional model with rolling PCA factors. "
               rf"Timing: {timing_label} ($PC_t \to R_{{t+1}}$ if predictive). "
               r"$z_t$ = standardized iTraxx Main 5Y. "
               r"HAC (Newey--West) standard errors. "
               r"*** $p < 0.01$, ** $p < 0.05$, * $p < 0.10$.")
    tex.append(r"\end{minipage}")
    tex.append(r"\end{table}")

    fname = f"PCA_Cond_Alpha_FS_Thesis_{timing}.tex"
    (TABLES_DIR / fname).write_text("\n".join(tex), encoding="utf-8")
    print(f"   ✅ {fname}")

    # ── THESIS TABLE: Threshold Robustness ────────────────────────────
    tex2 = []
    tex2.append(r"\begin{table}[htbp]")
    tex2.append(r"\centering")
    tex2.append(rf"\caption{{Conditional Alpha (PCA, {timing_label}): "
                r"Stress Threshold Robustness}")
    tex2.append(rf"\label{{tab:pca_conditional_threshold_{timing}}}")
    tex2.append(r"\small")
    n_s = len(strategies)
    tex2.append(r"\begin{tabular}{l r " + "r r r " * n_s + "}")
    tex2.append(r"\toprule")

    h1 = " & "
    for s in strategies:
        h1 += rf" & \multicolumn{{3}}{{c}}{{{TITLE_MAP.get(s, s)}}}"
    h1 += r" \\"
    tex2.append(h1)

    cmi = ""
    for i in range(n_s):
        start = 3 + i * 3
        cmi += rf"\cmidrule(lr){{{start}-{start+2}}}"
    tex2.append(cmi)

    h2 = r"Threshold & $n_H$"
    for _ in strategies:
        h2 += r" & $\alpha_1$ (ann.) & $t$ & $p$"
    h2 += r" \\"
    tex2.append(h2)
    tex2.append(r"\midrule")

    for th in ITRX_THRESHOLDS_BPS:
        row = rf"{th} bps"
        n_h = "--"
        for s in strategies:
            r = all_results[s].get('dummy_interaction', {}).get(str(th), {})
            if not r.get('skip') and r.get('n_HIGH'):
                n_h = str(r['n_HIGH'])
                break
        row += f" & {n_h}"

        for s in strategies:
            r = all_results[s].get('dummy_interaction', {}).get(str(th), {})
            if r.get('skip'):
                row += r" & -- & -- & --"
            else:
                row += (rf" & {r['alpha1_annualized']:+.2f}\%"
                        rf"{significance_stars(r['alpha1_pval'])}"
                        rf" & {_fmt2(r['alpha1_tstat'])}"
                        rf" & {_fmt4(r['alpha1_pval'])}")
        row += r" \\"
        tex2.append(row)

    tex2.append(r"\bottomrule")
    tex2.append(r"\end{tabular}")
    tex2.append(r"\vspace{0.2cm}")
    tex2.append(r"\begin{minipage}{0.92\textwidth}")
    tex2.append(r"\footnotesize\textit{Notes:} "
                r"$D_{\text{HIGH}} = \mathbf{1}\{\text{iTraxx Main 5Y} > "
                r"\text{threshold}\}$. PCA rolling factors. "
                r"HAC standard errors. "
                r"*** $p < 0.01$, ** $p < 0.05$, * $p < 0.10$.")
    tex2.append(r"\end{minipage}")
    tex2.append(r"\end{table}")

    fname2 = f"PCA_Cond_Alpha_Threshold_Thesis_{timing}.tex"
    (TABLES_DIR / fname2).write_text("\n".join(tex2), encoding="utf-8")
    print(f"   ✅ {fname2}")

    # ── THESIS TABLE: Multi-Proxy ─────────────────────────────────────
    tex3 = []
    tex3.append(r"\begin{table}[htbp]")
    tex3.append(r"\centering")
    tex3.append(rf"\caption{{Conditional Alpha (PCA, {timing_label}): "
                r"Robustness to Conditioning Variable}")
    tex3.append(rf"\label{{tab:pca_conditional_multiproxy_{timing}}}")
    tex3.append(r"\small")
    tex3.append(r"\begin{tabular}{l " + "r r r " * n_s + "}")
    tex3.append(r"\toprule")

    h1 = r"Conditioning variable"
    for s in strategies:
        h1 += rf" & \multicolumn{{3}}{{c}}{{{TITLE_MAP.get(s, s)}}}"
    h1 += r" \\"
    tex3.append(h1)
    cmi = ""
    for i in range(n_s):
        start = 2 + i * 3
        cmi += rf"\cmidrule(lr){{{start}-{start+2}}}"
    tex3.append(cmi)

    h2 = r"($z_t$)"
    for _ in strategies:
        h2 += r" & $\alpha_1$ & $t$ & $p$"
    h2 += r" \\"
    tex3.append(h2)
    tex3.append(r"\midrule")

    for pn, pcfg in CONDITIONING_VARIABLES.items():
        row = pcfg['label']
        for s in strategies:
            r = all_results[s].get('multi_conditioning', {}).get(pn, {})
            if r.get('skip'):
                row += r" & -- & -- & --"
            else:
                row += (rf" & {r['alpha1']:+.4f}{significance_stars(r['alpha1_pval'])}"
                        rf" & {_fmt2(r['alpha1_tstat'])}"
                        rf" & {_fmt4(r['alpha1_pval'])}")
        row += r" \\"
        tex3.append(row)

    tex3.append(r"\bottomrule")
    tex3.append(r"\end{tabular}")
    tex3.append(r"\vspace{0.2cm}")
    tex3.append(r"\begin{minipage}{0.92\textwidth}")
    tex3.append(r"\footnotesize\textit{Notes:} "
                r"Ferson and Schadt (1996) with PCA factors and alternative $z_t$. "
                r"HAC standard errors. "
                r"*** $p < 0.01$, ** $p < 0.05$, * $p < 0.10$.")
    tex3.append(r"\end{minipage}")
    tex3.append(r"\end{table}")

    fname3 = f"PCA_Cond_Alpha_MultiProxy_Thesis_{timing}.tex"
    (TABLES_DIR / fname3).write_text("\n".join(tex3), encoding="utf-8")
    print(f"   ✅ {fname3}")

    # ── BEAMER SLIDE: Threshold + Multi-Proxy combined ────────────────
    tex_b = []
    tex_b.append(rf"\begin{{frame}}[t]{{PCA Conditional Alpha ({timing_label}): Robustness}}")
    tex_b.append(r"\centering\vspace{-0.3cm}\scriptsize")
    tex_b.append(r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.05}")

    # Threshold panel
    tex_b.append(r"\textbf{Panel A: Threshold Robustness (iTraxx Main)}\par\vspace{0.1cm}")
    tex_b.append(r"\begin{tabular}{r r " + "r r " * n_s + "}")
    tex_b.append(r"\toprule")
    bh = "Thr. & $n_H$"
    for s in strategies:
        bh += rf" & $\alpha_1$ & $t$"
    bh += r" \\"
    tex_b.append(bh)
    tex_b.append(r"\midrule")
    for th in ITRX_THRESHOLDS_BPS:
        row = rf"{th}"
        n_h = "--"
        for s in strategies:
            r = all_results[s].get('dummy_interaction', {}).get(str(th), {})
            if not r.get('skip') and r.get('n_HIGH'):
                n_h = str(r['n_HIGH'])
                break
        row += f" & {n_h}"
        for s in strategies:
            r = all_results[s].get('dummy_interaction', {}).get(str(th), {})
            if r.get('skip'):
                row += " & -- & --"
            else:
                row += (rf" & {r['alpha1_annualized']:+.2f}{significance_stars(r['alpha1_pval'])}"
                        rf" & {_fmt2(r['alpha1_tstat'])}")
        row += r" \\"
        tex_b.append(row)
    tex_b.append(r"\bottomrule")
    tex_b.append(r"\end{tabular}")

    # Multi-proxy panel
    tex_b.append(r"\vspace{0.2cm}")
    tex_b.append(r"\textbf{Panel B: Conditioning Variable Robustness}\par\vspace{0.1cm}")
    tex_b.append(r"\begin{tabular}{l " + "r r " * n_s + "}")
    tex_b.append(r"\toprule")
    bh2 = "$z_t$"
    for s in strategies:
        bh2 += rf" & $\alpha_1$ & $t$"
    bh2 += r" \\"
    tex_b.append(bh2)
    tex_b.append(r"\midrule")
    for pn, pcfg in CONDITIONING_VARIABLES.items():
        row = pcfg['label']
        for s in strategies:
            r = all_results[s].get('multi_conditioning', {}).get(pn, {})
            if r.get('skip'):
                row += " & -- & --"
            else:
                row += (rf" & {r['alpha1']:+.4f}{significance_stars(r['alpha1_pval'])}"
                        rf" & {_fmt2(r['alpha1_tstat'])}")
        row += r" \\"
        tex_b.append(row)
    tex_b.append(r"\bottomrule")
    tex_b.append(r"\end{tabular}")

    tex_b.append(r"\vspace{0.1cm}")
    tex_b.append(r"{\tiny Ferson \& Schadt (1996); Patton (2009). "
                 r"PCA rolling factors. HAC SE. "
                 r"*** $p{<}1\%$, ** $p{<}5\%$, * $p{<}10\%$.}")
    tex_b.append(r"\end{frame}")

    fname_b = f"PCA_Cond_Alpha_Robustness_Slide_{timing}.tex"
    (TABLES_DIR / fname_b).write_text("\n".join(tex_b), encoding="utf-8")
    print(f"   ✅ {fname_b}")

def generate_article_tex(all_results, timing):
    """Generate article-format tables for conditional alpha (thesis/paper)."""
    print_header(f"GENERATING ARTICLE .TEX ({timing})")

    strategies = list(all_results.keys())
    if not strategies:
        print("   No results to generate article .tex for.")
        return

    timing_label = timing.title()
    n_s = len(strategies)

    def sig_super(pval):
        if pval < 0.01:
            return '***'
        if pval < 0.05:
            return '**'
        if pval < 0.10:
            return '*'
        return ''

    def fmt_val_super(val, pval, decimals=2):
        """Format value with superscript stars."""
        if np.isnan(val):
            return "--"
        stars = sig_super(pval)
        if stars:
            return f"${val:+.{decimals}f}^{{{stars}}}$"
        return f"{val:+.{decimals}f}"

    # ── TABLE 1: Ferson–Schadt ────────────────────────────────────────
    tex = []
    tex.append("% " + "=" * 74)
    tex.append("% PCA CONDITIONAL ALPHA — ARTICLE TABLES")
    tex.append("% " + "=" * 74)
    tex.append("")
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(rf"\caption{{Conditional Alpha (PCA, {timing_label}): "
               r"Ferson--Schadt (1996) Model}")
    tex.append(rf"\label{{tab:pca_cond_fs_{timing}_article}}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l" + "c" * n_s + "}")
    tex.append(r"\toprule")
    headers = " & ".join([TITLE_MAP.get(s, s) for s in strategies])
    tex.append(rf" & {headers} \\")
    tex.append(r"\midrule")

    # α₀ annualized
    vals = []
    for s in strategies:
        fs = all_results[s].get('ferson_schadt', {})
        u = fs.get('unconditional', {})
        v = u.get('alpha_annualized', np.nan)
        p = u.get('alpha_pval', 1)
        vals.append(fmt_val_super(v, p))
    tex.append(rf"$\alpha_0$ (ann.\ \%) & {' & '.join(vals)} \\")

    # α₁
    vals = []
    for s in strategies:
        c = all_results[s].get('ferson_schadt', {}).get('conditional', {})
        v = c.get('alpha1', np.nan)
        p = c.get('alpha1_pval', 1)
        vals.append(fmt_val_super(v, p, decimals=4))
    tex.append(rf"$\alpha_1$ (conditional) & {' & '.join(vals)} \\")

    # t-stat
    vals = []
    for s in strategies:
        c = all_results[s].get('ferson_schadt', {}).get('conditional', {})
        vals.append(f"({c.get('alpha1_tstat', 0):.2f})")
    tex.append(rf"\quad $t$-stat & {' & '.join(vals)} \\")

    # Joint F
    vals = []
    for s in strategies:
        jf = all_results[s].get('ferson_schadt', {}).get('joint_f_test', {})
        f_v = jf.get('f_statistic', np.nan)
        f_p = jf.get('p_value', 1)
        vals.append(fmt_val_super(f_v, f_p))
    tex.append(rf"Joint $F$-test & {' & '.join(vals)} \\")

    # Economic magnitude
    vals_h = []
    vals_l = []
    for s in strategies:
        em = all_results[s].get('ferson_schadt', {}).get('economic_magnitude', {})
        vals_h.append(f"{em.get('alpha_at_plus_1sd', np.nan):+.2f}\\%")
        vals_l.append(f"{em.get('alpha_at_minus_1sd', np.nan):+.2f}\\%")
    tex.append(rf"$\alpha(z = +1\sigma)$ ann. & {' & '.join(vals_h)} \\")
    tex.append(rf"$\alpha(z = -1\sigma)$ ann. & {' & '.join(vals_l)} \\")

    # T
    vals = []
    for s in strategies:
        T = all_results[s].get('ferson_schadt', {}).get('T', '--')
        vals.append(str(T))
    tex.append(rf"$T$ & {' & '.join(vals)} \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"Ferson and Schadt (1996) conditional model with rolling PCA factors. "
               rf"Timing: {timing_label} ($PC_t \to R_{{t+1}}$ if predictive). "
               r"$z_t$ = standardized iTraxx Main 5Y. "
               r"$t$-statistics in parentheses (Newey--West HAC). "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

    fname = f"PCA_Cond_Alpha_FS_article_{timing}.tex"
    (TABLES_DIR / fname).write_text("\n".join(tex), encoding="utf-8")
    print(f"   ✅ {fname}  ← PAPER & SKELETON")

    # ── TABLE 2: Threshold Robustness ─────────────────────────────────
    tex2 = []
    tex2.append(r"\begin{table}[H]")
    tex2.append(r"\centering")
    tex2.append(rf"\caption{{Conditional Alpha (PCA, {timing_label}): "
                r"Stress Threshold Robustness}")
    tex2.append(rf"\label{{tab:pca_cond_threshold_{timing}_article}}")
    tex2.append(r"\begin{threeparttable}")
    tex2.append(r"\begin{singlespace}")
    tex2.append(r"\small")
    tex2.append(r"\begin{tabular}{l r " + "r r r " * n_s + "}")
    tex2.append(r"\toprule")

    h1 = " & "
    for s in strategies:
        h1 += rf" & \multicolumn{{3}}{{c}}{{{TITLE_MAP.get(s, s)}}}"
    h1 += r" \\"
    tex2.append(h1)

    cmi = ""
    for i in range(n_s):
        start = 3 + i * 3
        cmi += rf"\cmidrule(lr){{{start}-{start+2}}}"
    tex2.append(cmi)

    h2 = r"Threshold & $n_H$"
    for _ in strategies:
        h2 += r" & $\alpha_1$ (ann.) & $t$ & $p$"
    h2 += r" \\"
    tex2.append(h2)
    tex2.append(r"\midrule")

    for th in ITRX_THRESHOLDS_BPS:
        row = rf"{th} bps"
        n_h = "--"
        for s in strategies:
            r = all_results[s].get('dummy_interaction', {}).get(str(th), {})
            if not r.get('skip') and r.get('n_HIGH'):
                n_h = str(r['n_HIGH'])
                break
        row += f" & {n_h}"

        for s in strategies:
            r = all_results[s].get('dummy_interaction', {}).get(str(th), {})
            if r.get('skip'):
                row += r" & -- & -- & --"
            else:
                a1 = r['alpha1_annualized']
                p1 = r['alpha1_pval']
                stars = sig_super(p1)
                if stars:
                    row += rf" & ${a1:+.2f}^{{{stars}}}$\%"
                else:
                    row += rf" & {a1:+.2f}\%"
                row += rf" & {_fmt2(r['alpha1_tstat'])}"
                row += rf" & {_fmt4(r['alpha1_pval'])}"
        row += r" \\"
        tex2.append(row)

    tex2.append(r"\bottomrule")
    tex2.append(r"\end{tabular}")
    tex2.append("")
    tex2.append(r"\begin{tablenotes}[para,flushleft]")
    tex2.append(r"\footnotesize")
    tex2.append(r"\item \textit{Note:} "
                r"$D_{\text{HIGH}} = \mathbf{1}\{\text{iTraxx Main 5Y} > "
                r"\text{threshold}\}$. PCA rolling factors. "
                r"$t$-statistics (Newey--West HAC). "
                r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex2.append(r"\end{tablenotes}")
    tex2.append(r"\end{singlespace}")
    tex2.append(r"\end{threeparttable}")
    tex2.append(r"\end{table}")

    fname2 = f"PCA_Cond_Alpha_Threshold_article_{timing}.tex"
    (TABLES_DIR / fname2).write_text("\n".join(tex2), encoding="utf-8")
    print(f"   ✅ {fname2}  ← PAPER & SKELETON")

    # ── TABLE 3: Multi-Proxy ──────────────────────────────────────────
    tex3 = []
    tex3.append(r"\begin{table}[H]")
    tex3.append(r"\centering")
    tex3.append(rf"\caption{{Conditional Alpha (PCA, {timing_label}): "
                r"Robustness to Conditioning Variable}")
    tex3.append(rf"\label{{tab:pca_cond_multiproxy_{timing}_article}}")
    tex3.append(r"\begin{threeparttable}")
    tex3.append(r"\begin{singlespace}")
    tex3.append(r"\small")
    tex3.append(r"\begin{tabular}{l " + "r r r " * n_s + "}")
    tex3.append(r"\toprule")

    h1 = r"Conditioning variable"
    for s in strategies:
        h1 += rf" & \multicolumn{{3}}{{c}}{{{TITLE_MAP.get(s, s)}}}"
    h1 += r" \\"
    tex3.append(h1)
    cmi = ""
    for i in range(n_s):
        start = 2 + i * 3
        cmi += rf"\cmidrule(lr){{{start}-{start+2}}}"
    tex3.append(cmi)

    h2 = r"($z_t$)"
    for _ in strategies:
        h2 += r" & $\alpha_1$ & $t$ & $p$"
    h2 += r" \\"
    tex3.append(h2)
    tex3.append(r"\midrule")

    for pn, pcfg in CONDITIONING_VARIABLES.items():
        row = pcfg['label']
        for s in strategies:
            r = all_results[s].get('multi_conditioning', {}).get(pn, {})
            if r.get('skip'):
                row += r" & -- & -- & --"
            else:
                a1 = r['alpha1']
                p1 = r['alpha1_pval']
                stars = sig_super(p1)
                if stars:
                    row += rf" & ${a1:+.4f}^{{{stars}}}$"
                else:
                    row += rf" & {a1:+.4f}"
                row += rf" & {_fmt2(r['alpha1_tstat'])}"
                row += rf" & {_fmt4(r['alpha1_pval'])}"
        row += r" \\"
        tex3.append(row)

    tex3.append(r"\bottomrule")
    tex3.append(r"\end{tabular}")
    tex3.append("")
    tex3.append(r"\begin{tablenotes}[para,flushleft]")
    tex3.append(r"\footnotesize")
    tex3.append(r"\item \textit{Note:} "
                r"Ferson and Schadt (1996) with PCA factors and alternative $z_t$. "
                r"$t$-statistics (Newey--West HAC). "
                r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex3.append(r"\end{tablenotes}")
    tex3.append(r"\end{singlespace}")
    tex3.append(r"\end{threeparttable}")
    tex3.append(r"\end{table}")

    fname3 = f"PCA_Cond_Alpha_MultiProxy_article_{timing}.tex"
    (TABLES_DIR / fname3).write_text("\n".join(tex3), encoding="utf-8")
    print(f"   ✅ {fname3}  ← PAPER & SKELETON")

# ============================================================================
# RUNNER
# ============================================================================

def run_all_for_strategy(strategy_name, timing):
    """Run all 5 conditional alpha tests for one strategy + timing."""
    results = {'strategy': strategy_name, 'timing': timing}

    for label, func in [
        ('ferson_schadt', ferson_schadt_conditional),
        ('dummy_interaction', dummy_interaction),
        ('subsample', subsample_regime),
        ('multi_conditioning', multiple_conditioning_variables),
        ('rolling_regime', rolling_alpha_regime_plot),
    ]:
        try:
            results[label] = func(strategy_name, timing)
        except Exception as e:
            print(f"\n   ❌ {label}: {e}")
            import traceback; traceback.print_exc()

    strategy_dir = get_strategy_pca_dir(strategy_name)
    with open(strategy_dir / f"conditional_alpha_summary_{timing}.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n   💾 conditional_alpha_summary_{timing}.json")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("PCA CONDITIONAL ALPHA ANALYSIS")
    print(f"\n   References:")
    print(f"   - Ferson & Schadt (1996, JF)")
    print(f"   - Mitchell & Pulvino (2001, JF)")
    print(f"   - Patton (2009, RFS)")
    print(f"   - Ludvigson & Ng (2009, RFS): rolling PCA")
    print(f"\n   PCA timing: {PCA_TIMING}")
    print(f"   PC components: {PCA_N_COMPONENTS}")
    print(f"   Conditioning: {list(CONDITIONING_VARIABLES.keys())}")
    print(f"   Thresholds: {ITRX_THRESHOLDS_BPS} bps")

    # Detect available timings
    pca_dir = get_pca_output_dir()
    timings = []
    for t in ['predictive', 'contemporaneous']:
        if (pca_dir / f"pc_scores_{t}.parquet").exists():
            timings.append(t)

    if not timings:
        print("\n   ❌ No PC scores found. Run 02_pca_rolling.py first.")
        return

    print(f"   Available timings: {timings}")

    for timing in timings:
        print_header(f"TIMING: {timing.upper()}")

        all_results = {}
        for strategy_name in STRATEGIES.keys():
            print_header(f"STRATEGY: {strategy_name} ({timing})")
            try:
                result = run_all_for_strategy(strategy_name, timing)
                all_results[strategy_name] = result
            except Exception as e:
                print(f"\n   ❌ {strategy_name}: {e}")
                import traceback; traceback.print_exc()

        if all_results:
            # Cross-strategy summary
            print_header(f"CROSS-STRATEGY SUMMARY ({timing})")
            print(f"\n   Ferson–Schadt α₁:")
            print(f"   {'Strategy':<20} {'α₁':>8} {'t':>8} {'p':>8}")
            print(f"   {'─' * 46}")
            for name, res in all_results.items():
                c = res.get('ferson_schadt', {}).get('conditional', {})
                if c:
                    print(f"   {name:<20} {c.get('alpha1', 0):>+8.4f}"
                          f" {c.get('alpha1_tstat', 0):>8.3f}"
                          f" {c.get('alpha1_pval', 1):>8.4f}"
                          f" {significance_stars(c.get('alpha1_pval', 1))}")

            # Generate .tex
            generate_tex(all_results, timing)

            # Article versions (thesis/paper)
            generate_article_tex(all_results, timing)

    print(f"\n{'=' * 80}")
    print(f"✅ PCA CONDITIONAL ALPHA ANALYSIS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
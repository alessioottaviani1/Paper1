"""
================================================================================
04_pca_conditional_alpha.py — Conditional Alpha Analysis (PCA Pipeline)
================================================================================
Tests whether strategy alpha varies with financial stress, using full-sample
PCA factors (PC scores) from 02_pca_estimation.py.

Same three tests as 06e_conditional_alpha.py (ML pipeline) and
factor_models/03 (benchmark), but with PC scores as factors:

  (e1) Ferson–Schadt (1996, JF) conditional alpha, z_{t-1} headline
       (bootstrap); HAC and contemporaneous z_t as robustness
  (e2) Regime dummies, 3-state, common betas (LOW<60, MEDIUM∈[60,100),
       HIGH≥100 bps on iTraxx Main 5Y)
  (e5) Rolling alpha with LOW/MEDIUM/HIGH regime shading (iTraxx Main)

Using the same tests across PCA and AEN pipelines demonstrates that the
conditional alpha result is robust to the dimensionality reduction method.

Timing convention:
  Follows PCA_TIMING from 00_pca_config.py.
  "contemporaneous": PC_t → R_t   (baseline; Connor & Korajczyk 1986, 1988)

References:
    Ferson, W. and Schadt, R. (1996, JF) — conditional alpha & beta
    Connor, G. and Korajczyk, R. (1986, 1988) — PCs as factors

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

import importlib.util as _ilu, os as _os
_inf_spec = _ilu.spec_from_file_location(
    "inference",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "factor_models", "inference.py"))
_inf = _ilu.module_from_spec(_inf_spec); _inf_spec.loader.exec_module(_inf)
cfg_alpha1_bootstrap_p = _inf.cfg_alpha1_bootstrap_p
HAC_LAGS = 4

# ── Stress proxy paths ────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"
FACTORS_PATH         = pca_config.FACTORS_PATH

# ── Stress-regime cutoffs su iTraxx Main 5Y (= benchmark / 03) ───────────
LOW_CUT = 60
HIGH_CUT = 100
DEFAULT_THRESHOLD = HIGH_CUT           # cutoff HIGH binario usato da (e3)/(e5)

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
    "itraxx_combined": "iTraxx Skew",
}


# ============================================================================
# HELPERS
# ============================================================================

def print_header(title, char="="):
    print(f"\n{char * 2} {title.strip()}")


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


def load_stress_proxy_monthly(proxy_name="ITRX_MAIN"):
    """iTraxx Main 5Y level (bps), monthly last — dal foglio 'cds' di bbg.xlsx
    (come 03 e benchmark factor_models/03). Caricato via importlib per evitare
    il warning statico di Pylance; il sys.path serve agli import interni."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "factor_models"))
    _spec = _ilu.spec_from_file_location(
        "factor_sources", str(PROJECT_ROOT / "src" / "factor_models" / "factor_sources.py"))
    src = _ilu.module_from_spec(_spec); _spec.loader.exec_module(src)
    s = src.load_bloomberg("cds")["ITRX EUR CDSI GEN 5Y Corp"].dropna()
    monthly = s.resample("ME").last().dropna()
    monthly.name = "ITRX_MAIN"
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

def cfg_conditional(strategy_name, timing):
    """
    r_t = α₀ + α₁·z_t + Σⱼ βⱼ·PCⱼt + Σⱼ δⱼ·(PCⱼt·z_t) + εₜ
    """
    print_header(f"   (e1) Ferson–Schadt — {strategy_name} ({timing})", "─")

    y, X, pc_names = prepare_pca_data(strategy_name, timing)
    T, k = len(y), len(pc_names)

    # Load conditioning variable (iTraxx Main 5Y), levels
    z_raw = load_stress_proxy_monthly("ITRX_MAIN")
    z_level = z_raw.reindex(y.index, method='nearest').dropna()
    common = y.index.intersection(z_level.index)
    y, X, z_level = y[common], X.loc[common], z_level[common]

    # CFG (1998): condition on the PREDETERMINED (one-period-lagged) stress state z_{t-1}.
    z_lag_level = z_level.shift(1)
    keep = z_lag_level.dropna().index
    y, X = y.loc[keep], X.loc[keep]
    z_level, z_lag_level = z_level.loc[keep], z_lag_level.loc[keep]
    z = (z_lag_level - z_lag_level.mean()) / z_lag_level.std()      # z_{t-1}  (primary CFG)
    z_contemp = (z_level - z_level.mean()) / z_level.std()          # z_t      (robustness)
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
    # Headline inference on alpha1 = moving-block bootstrap; HAC kept as robustness.
    alpha1_p_hac = float(res_cond.pvalues['z_stress'])
    alpha1_p = cfg_alpha1_bootstrap_p(y, X, z)                 # z_{t-1}  (primary CFG)
    alpha1_p_contemp = cfg_alpha1_bootstrap_p(y, X, z_contemp)  # z_t  (robustness)


    print(f"\n   Unconditional: α = {res_unc.params['const'] * 12:+.2f}% ann.")
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
            'alpha1_pval': round(float(alpha1_p), 4),                  # bootstrap p, z_{t-1} (HEADLINE)
            'alpha1_pval_hac': round(float(alpha1_p_hac), 4),          # HAC p, z_{t-1} (robustness)
            'alpha1_pval_contemp': round(float(alpha1_p_contemp), 4),  # bootstrap p, z_t (robustness)
            'r2_adj': round(float(res_cond.rsquared_adj), 6),
        },
        'conditional_betas': delta_results,
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
    Regime dummies a 3 stati con beta COMUNI (come factor_models/03):
        r_t = a_LOW·D_LOW + a_MED·D_MED + a_HIGH·D_HIGH + Σⱼ βⱼ·PCⱼt + εₜ
    Niente costante: le 3 dummy partizionano il campione; le beta sono comuni a
    tutti i regimi (robusto quando i mesi HIGH sono pochi). Cutoff su iTraxx Main
    5Y: LOW < LOW_CUT <= MEDIUM < HIGH_CUT <= HIGH. Inferenza Newey-West HAC.
    """
    print_header(f"   (e2) Regime dummies 3-state — {strategy_name} ({timing})", "─")

    y, X, pc_names = prepare_pca_data(strategy_name, timing)

    z_raw = load_stress_proxy_monthly("ITRX_MAIN")
    z_aligned = z_raw.reindex(y.index, method='nearest')
    common = y.index.intersection(z_aligned.dropna().index)
    y, X, z_level = y[common], X.loc[common], z_aligned[common]
    T = len(y)

    D = pd.DataFrame({
        'LOW':    (z_level < LOW_CUT).astype(float),
        'MEDIUM': ((z_level >= LOW_CUT) & (z_level < HIGH_CUT)).astype(float),
        'HIGH':   (z_level >= HIGH_CUT).astype(float),
    }, index=y.index)
    regimes = [r for r in ('LOW', 'MEDIUM', 'HIGH') if D[r].sum() >= 1]

    print(f"\n   T = {T}  |  LOW={int(D['LOW'].sum())}, "
          f"MEDIUM={int(D['MEDIUM'].sum())}, HIGH={int(D['HIGH'].sum())}")

    X_d = pd.concat([D[regimes], X], axis=1)   # no costante: le dummy partizionano
    res = sm.OLS(y, X_d).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})

    results = {}
    for r in ('LOW', 'MEDIUM', 'HIGH'):
        n = int(D[r].sum())
        if r in regimes and n >= 5:
            a = float(res.params[r])
            results[r] = {
                'alpha_monthly': round(a, 6),
                'alpha_annualized': round(a * 12, 4),
                'alpha_tstat': round(float(res.tvalues[r]), 4),
                'alpha_pval': round(float(res.pvalues[r]), 4),
                'n': n,
            }
            print(f"   {r:<7} α = {a*12:+6.2f}% ann. "
                  f"(t={res.tvalues[r]:5.2f}, p={res.pvalues[r]:.4f}) "
                  f"{significance_stars(res.pvalues[r])}  N={n}")
        else:
            results[r] = {'skip': True, 'n': n}
            print(f"   {r:<7} skip (N={n})")

    strategy_dir = get_strategy_pca_dir(strategy_name)
    with open(strategy_dir / f"conditional_dummy_{timing}.json", 'w') as f:
        json.dump({'strategy': strategy_name, 'timing': timing, 'T': T,
                   'cutoffs_bps': [LOW_CUT, HIGH_CUT], 'regimes': results}, f, indent=2)
    print(f"\n   💾 conditional_dummy_{timing}.json")

    return results


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
        print(f"   α = {alpha * 12:+.2f}% ann."
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
    tex.append(r"\singlespacing")
    tex.append(rf"\caption{{Conditional Alpha (PCA, {timing_label}): "
               r"Christopherson, Ferson, and Glassman (1998) Model}")
    tex.append(rf"\label{{tab:pca_cond_fs_{timing}_article}}")
    tex.append(r"\begin{minipage}{\textwidth}")
    tex.append(r"{\footnotesize\noindent "
               r"Christopherson, Ferson, and Glassman (1998) full conditional model, "
               r"$r_t = \alpha_0 + \alpha_1 z_{t-1} + (\beta + \delta z_{t-1})' \mathbf{PC}_t + \varepsilon_t$, "
               r"with the $K=8$ full-sample PCA factors: both the alpha and the factor loadings "
               r"are conditioned on $z_{t-1}$, the one-month-lagged standardized iTraxx Main 5Y "
               r"spread (a predetermined conditioning variable). "
               rf"Timing: {timing_label}. "
               r"Significance of $\alpha_1$ from a moving-block bootstrap "
               r"($B=9{,}999$, $H_0{:}\,\alpha_1=0$ imposed, block length~9); Newey--West HAC "
               r"$t$-statistics in parentheses; the HAC $p$-value is a robustness check. "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.}")
    tex.append(r"\end{minipage}")
    tex.append(r"\par\vspace{6pt}")
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

    # α₁ (annualized ×12, to match Table VI Panel B)
    vals = []
    for s in strategies:
        c = all_results[s].get('ferson_schadt', {}).get('conditional', {})
        v = c.get('alpha1', np.nan)
        p = c.get('alpha1_pval', 1)
        vals.append(fmt_val_super(v * 12 if v == v else v, p, decimals=2))
    tex.append(rf"$\alpha_1$ (\% p.a.\ per $1\sigma$) & {' & '.join(vals)} \\")

    # t-stat
    vals = []
    for s in strategies:
        c = all_results[s].get('ferson_schadt', {}).get('conditional', {})
        vals.append(f"({c.get('alpha1_tstat', 0):.2f})")
    tex.append(rf"\quad $t$-stat & {' & '.join(vals)} \\")

    # alpha1 bootstrap p (headline) + HAC p (robustness)
    vals = []
    for s in strategies:
        c = all_results[s].get('ferson_schadt', {}).get('conditional', {})
        vals.append(_fmt4(c.get('alpha1_pval', np.nan)))
    tex.append(rf"\quad $p$ (bootstrap) & {' & '.join(vals)} \\")
    vals = []
    for s in strategies:
        c = all_results[s].get('ferson_schadt', {}).get('conditional', {})
        vals.append(_fmt4(c.get('alpha1_pval_hac', np.nan)))
    tex.append(rf"\quad $p$ (HAC, robustness) & {' & '.join(vals)} \\")

    # Economic magnitude
    vals_h = []
    vals_l = []
    for s in strategies:
        em = all_results[s].get('ferson_schadt', {}).get('economic_magnitude', {})
        vals_h.append(f"{em.get('alpha_at_plus_1sd', np.nan):+.2f}")
        vals_l.append(f"{em.get('alpha_at_minus_1sd', np.nan):+.2f}")
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
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{table}")

    fname = f"PCA_Cond_Alpha_FS_article_{timing}.tex"
    (TABLES_DIR / fname).write_text("\n".join(tex), encoding="utf-8")
    print(f"   ✅ {fname}  ← PAPER & SKELETON")

    # POST
    # ── TABLE 2: Regime alpha (dummy 3-stati, beta comuni) ────────────
    tex2 = []
    tex2.append(r"\begin{table}[H]")
    tex2.append(r"\centering")
    tex2.append(rf"\caption{{Conditional Alpha (PCA, {timing_label}): "
                r"Stress-Regime Dummies (common $\beta$)}")
    tex2.append(rf"\label{{tab:pca_cond_threshold_{timing}_article}}")
    tex2.append(r"\begin{threeparttable}")
    tex2.append(r"\begin{singlespace}")
    tex2.append(r"\small")
    tex2.append(r"\begin{tabular}{l " + "r r " * n_s + "}")
    tex2.append(r"\toprule")

    h1 = " "
    for s in strategies:
        h1 += rf" & \multicolumn{{2}}{{c}}{{{TITLE_MAP.get(s, s)}}}"
    h1 += r" \\"
    tex2.append(h1)

    cmi = ""
    for i in range(n_s):
        start = 2 + i * 2
        cmi += rf"\cmidrule(lr){{{start}-{start+1}}}"
    tex2.append(cmi)

    h2 = r"Regime"
    for _ in strategies:
        h2 += r" & $\alpha$ (ann.) & $t$"
    h2 += r" \\"
    tex2.append(h2)
    tex2.append(r"\midrule")

    for reg in ('LOW', 'MEDIUM', 'HIGH'):
        row = reg
        for s in strategies:
            r = all_results[s].get('dummy_interaction', {}).get(reg, {})
            if r.get('skip') or 'alpha_annualized' not in r:
                row += r" & -- & --"
            else:
                a = r['alpha_annualized']
                stars = sig_super(r['alpha_pval'])
                if stars:
                    row += rf" & ${a:+.2f}^{{{stars}}}$\%"
                else:
                    row += rf" & {a:+.2f}\%"
                row += rf" & {_fmt2(r['alpha_tstat'])}"
        row += r" \\"
        tex2.append(row)

    tex2.append(r"\bottomrule")
    tex2.append(r"\end{tabular}")
    tex2.append("")
    tex2.append(r"\begin{tablenotes}[para,flushleft]")
    tex2.append(r"\footnotesize")
    tex2.append(r"\item \textit{Note:} "
                r"Common-$\beta$ regime dummies (no intercept): "
                r"$r_t = \sum_g a_g D_{g,t} + \sum_j \beta_j \text{PC}_{jt} + \varepsilon_t$, "
                r"regimes on iTraxx Main 5Y (LOW $<$ 60, MEDIUM $\in$ [60,100), HIGH $\geq$ 100 bps). "
                r"$a_g$ is the regime-$g$ alpha (\% p.a.). "
                r"$t$-statistics (Newey--West HAC, 4 lags). "
                r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex2.append(r"\end{tablenotes}")
    tex2.append(r"\end{singlespace}")
    tex2.append(r"\end{threeparttable}")
    tex2.append(r"\end{table}")

    fname2 = f"PCA_Cond_Alpha_Threshold_article_{timing}.tex"
    (TABLES_DIR / fname2).write_text("\n".join(tex2), encoding="utf-8")
    print(f"   ✅ {fname2}  ← PAPER")


# ============================================================================
# RUNNER
# ============================================================================

def run_all_for_strategy(strategy_name, timing):
    """Run the 3 conditional alpha tests (CFG, 3-state regime, rolling)."""
    results = {'strategy': strategy_name, 'timing': timing}

    for label, func in [
        ('ferson_schadt', cfg_conditional),
        ('dummy_interaction', dummy_interaction),
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
    print(f"\n   PCA timing: {PCA_TIMING}")
    print(f"   PC components: {PCA_N_COMPONENTS}")
    print(f"   Conditioning: iTraxx Main 5Y  |  regime cutoffs {LOW_CUT}/{HIGH_CUT} bps")

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

            # Tabelle usate dal paper (FS + Threshold)
            generate_article_tex(all_results, timing)

    print(f"\n{'=' * 80}")
    print(f"✅ PCA CONDITIONAL ALPHA ANALYSIS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
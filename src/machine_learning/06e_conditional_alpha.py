"""
================================================================================
06e_conditional_alpha.py — Conditional Alpha Analysis
================================================================================
Conditional-alpha layer on the frozen primary (best-subset) factor set.

Tests whether strategy alpha varies with financial stress, using the
best-subset factor set (02s), frozen — loadings are conditional but the
factor selection itself is not re-run here.

Three tests, IDENTICAL to factor_models/03 and pca/04 (conditional alpha):

  (e1) Ferson–Schadt (1996, JF) conditional alpha + conditional beta:
       r_t = α₀ + α₁·z_(t-1) + Σⱼ (βⱼ + δⱼ·z_(t-1))·Xⱼt + εₜ
       - z_(t-1) = standardized PREDETERMINED stress (iTraxx Main 5Y, lagged)
       - α₁ tests whether alpha is higher when stress is already elevated
       - δⱼ tests whether factor exposures are time-varying
       Headline inference = moving-block bootstrap; HAC and contemporaneous
       z_t reported as robustness.

  (e2) Regime dummies, 3-state, common betas (cfr. factor_models/03):
       r_t = a_LOW·D_LOW + a_MED·D_MED + a_HIGH·D_HIGH + Σⱼ βⱼ·Xⱼt + εₜ
       Cutoffs on iTraxx Main 5Y: LOW<60, MEDIUM∈[60,100), HIGH≥100 bps.

  (e5) Rolling alpha with regime shading:
       Overlay LOW/MEDIUM/HIGH stress regimes on the rolling alpha.

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

import importlib.util as _ilu
_inf_spec = _ilu.spec_from_file_location(
    "inference", str(PROJECT_ROOT / "src" / "factor_models" / "inference.py"))
_inf = _ilu.module_from_spec(_inf_spec); _inf_spec.loader.exec_module(_inf)
cfg_alpha1_bootstrap_p = _inf.cfg_alpha1_bootstrap_p

# Stress proxy dalla STESSA fonte del benchmark/PCA (factor_models), così la
# variabile di condizionamento è identica in tutto il paper. Caricato via
# importlib (come inference.py) per evitare il warning statico di Pylance;
# il sys.path serve solo agli import interni di factor_sources.
import sys as _sys
_sys.path.insert(0, str(PROJECT_ROOT / "src" / "factor_models"))
_src_spec = _ilu.spec_from_file_location(
    "factor_sources", str(PROJECT_ROOT / "src" / "factor_models" / "factor_sources.py"))
_src = _ilu.module_from_spec(_src_spec); _src_spec.loader.exec_module(_src)

HAC_LAGS             = aen_config.HAC_LAGS   # = 4 once set in 00_config.py (above)
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


# ── iTraxx Main 5Y stress-regime cutoffs (bps) — IDENTICI a factor_models/03 e
# pca/04: LOW < LOW_CUT <= MEDIUM < HIGH_CUT <= HIGH (dummy a 3 stati, beta comuni).
LOW_CUT = 60
HIGH_CUT = 100
DEFAULT_THRESHOLD = HIGH_CUT   # cutoff HIGH binario usato dallo shading rolling (e5)



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


def load_stress_proxy_monthly(proxy_name="ITRX_MAIN"):
    """iTraxx Europe Main 5Y level (bps), monthly last — STESSA fonte di
    factor_models/03 e pca/04 (factor_sources.load_bloomberg), così la
    variabile di stress è identica in tutto il paper."""
    s = _src.load_bloomberg("cds")["ITRX EUR CDSI GEN 5Y Corp"].dropna()
    monthly = s.resample("ME").last().dropna()
    monthly.name = "ITRX_MAIN"
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

    # Load primary conditioning variable (iTraxx Main 5Y level)
    z_raw = load_stress_proxy_monthly("ITRX_MAIN")
    stress_aligned = z_raw.reindex(y.index, method='nearest')
    common = y.index.intersection(stress_aligned.dropna().index)
    y, X = y[common], X.loc[common]
    stress_level = stress_aligned[common]

    # CFG (1998): condition on the PREDETERMINED one-period-lagged stress state z_{t-1}
    # — the canonical CFG timing, IDENTICAL to the benchmark/PCA pipelines
    # (factor_models/03 analysis_ferson_schadt). Standardize over the estimation sample.
    z_lag_level = stress_level.shift(1)
    keep = z_lag_level.dropna().index
    y, X = y.loc[keep], X.loc[keep]
    zl = z_lag_level.loc[keep]
    z = (zl - zl.mean()) / zl.std()       # z_{t-1}  (primary CFG, headline)
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
    # Headline inference on alpha1 = moving-block bootstrap; HAC kept as robustness.
    alpha1_p_hac = float(res_cond.pvalues['z_stress'])
    alpha1_p = cfg_alpha1_bootstrap_p(y, X, z)                          # z_{t-1} (HEADLINE)
    # z_t (contemporaneo) bootstrap p — solo robustezza, come il benchmark
    # (factor_models/03 'alpha1_pval_contemp'). Ri-deriva il livello
    # contemporaneo sullo STESSO campione di stima (y.index).
    _sl = load_stress_proxy_monthly("ITRX_MAIN").reindex(y.index, method="nearest")
    z_contemp = (_sl - _sl.mean()) / _sl.std()
    alpha1_p_contemp = cfg_alpha1_bootstrap_p(y, X, z_contemp)          # z_t (robustezza)

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
            'alpha1_pval': round(float(alpha1_p), 4),              # block-bootstrap p, z_{t-1} (HEADLINE)
            'alpha1_pval_hac': round(float(alpha1_p_hac), 4),      # HAC p, z_{t-1} (robustness)
            'alpha1_pval_contemp': round(float(alpha1_p_contemp), 4),  # block-bootstrap p, z_t (robustness)
            'alpha1_interpretation': (
                'alpha increases with stress'
                if alpha1 > 0 and alpha1_p < 0.10
                else 'no significant conditional alpha'),
            'r2_adj': round(float(res_cond.rsquared_adj), 6),
        },
        'conditional_betas': delta_results,
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
    Regime dummies a 3 stati con beta COMUNI (come factor_models/03 e pca/04):
        r_t = a_LOW·D_LOW + a_MED·D_MED + a_HIGH·D_HIGH + Σⱼ βⱼ·Xⱼt + εₜ
    Niente costante: le 3 dummy partizionano il campione; le beta sono comuni a
    tutti i regimi (robusto quando i mesi HIGH sono pochi). Cutoff su iTraxx Main
    5Y: LOW < LOW_CUT <= MEDIUM < HIGH_CUT <= HIGH. Inferenza Newey-West HAC.
    """
    print_header(f"   (e2) Regime dummies 3-state — {strategy_name}", "─")

    y, X, stable_factors = prepare_data(strategy_name, strategy_path)

    z_raw = load_stress_proxy_monthly("ITRX_MAIN")
    z_aligned = z_raw.reindex(y.index, method='nearest')
    common = y.index.intersection(z_aligned.dropna().index)
    y, X, z_level = y[common], X.loc[common], z_aligned[common]
    T = len(y)

    print(f"\n   Stable factors: {stable_factors}")
    print(f"   T = {T}  |  LOW={int((z_level < LOW_CUT).sum())}, "
          f"MEDIUM={int(((z_level >= LOW_CUT) & (z_level < HIGH_CUT)).sum())}, "
          f"HIGH={int((z_level >= HIGH_CUT).sum())}")

    D = pd.DataFrame({
        'LOW':    (z_level < LOW_CUT).astype(float),
        'MEDIUM': ((z_level >= LOW_CUT) & (z_level < HIGH_CUT)).astype(float),
        'HIGH':   (z_level >= HIGH_CUT).astype(float),
    }, index=y.index)
    regimes = [r for r in ('LOW', 'MEDIUM', 'HIGH') if D[r].sum() >= 1]

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

    strategy_dir = get_strategy_aen_dir(strategy_name)
    with open(strategy_dir / "conditional_dummy_interaction.json", 'w') as f:
        json.dump({'strategy': strategy_name, 'T': T,
                   'cutoffs_bps': [LOW_CUT, HIGH_CUT], 'regimes': results}, f, indent=2)
    print(f"\n   💾 conditional_dummy_interaction.json")

    return results


# ============================================================================
# (e3) SUB-SAMPLE SPLIT BY REGIME
# ============================================================================


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
    em   = fs_result['economic_magnitude']
    deltas = fs_result.get('conditional_betas', {})
    T    = fs_result['T']
    k    = fs_result['k_base']

    tex = []
    tex.append(rf"\begin{{frame}}[t]{{Conditional Alpha (Christopherson--Ferson--Glassman): {title}}}")
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
    tex.append(rf"$\alpha_1$ $p$ (boot) & {_fmt4(cond['alpha1_pval'])} \\")
    tex.append(rf"$\alpha_1$ $p$ (HAC) & {_fmt4(cond.get('alpha1_pval_hac', float('nan')))} \\")
    tex.append(r"\end{tabular}")
    tex.append(r"\par\vspace{0.15cm}")
    tex.append(r"{\tiny\textbf{Econ.\ magnitude:}\par")
    tex.append(rf"$\alpha(z{{=}}+1\sigma)$: {em['alpha_at_plus_1sd']:+.2f}\%\par")
    tex.append(rf"$\alpha(z{{=}}-1\sigma)$: {em['alpha_at_minus_1sd']:+.2f}\%\par")
    tex.append(r"*** $p{<}1\%$, ** $p{<}5\%$,\par * $p{<}10\%$.}")

    tex.append(r"\end{columns}")

    # Bottom: note
    tex.append(r"\vspace{0.15cm}")
    tex.append(r"{\tiny Christopherson, Ferson \& Glassman (1998, \textit{RFS}). "
               r"$z_t$ = standardized iTraxx Main 5Y. "
               r"$\alpha_1$ significance: moving-block bootstrap; HAC $t$ in parentheses; HAC $p$ as robustness.}")

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
    tex.append(r"\caption{Conditional Alpha: Christopherson, Ferson, and Glassman (1998) Model}")
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

    # alpha1 bootstrap p (headline) + HAC p (robustness)
    vals = []
    for s in strategies:
        c = all_results[s]['ferson_schadt']['conditional']
        vals.append(_fmt4(c['alpha1_pval']))
    tex.append(rf"\quad $p$ (bootstrap) & {' & '.join(vals)} \\")
    vals = []
    for s in strategies:
        c = all_results[s]['ferson_schadt']['conditional']
        vals.append(_fmt4(c.get('alpha1_pval_hac', float('nan'))))
    tex.append(rf"\quad $p$ (HAC, robustness) & {' & '.join(vals)} \\")

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
               r"Christopherson, Ferson, and Glassman (1998) full conditional performance model: "
               r"both the alpha and the factor loadings are conditioned on $z_t$, the standardized "
               r"iTraxx Main 5Y spread (mean zero, unit variance). "
               r"$\alpha_1 > 0$ indicates alpha increases with credit stress. "
               r"Significance of $\alpha_1$ is from a moving-block bootstrap "
               r"($B=9{,}999$, $H_0{:}\,\alpha_1=0$ imposed, block length~9); Newey--West HAC "
               rf"$t$-statistics in parentheses are Newey--West HAC (lag {HAC_LAGS}). "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$. "
               r"Factor definitions are provided in Table~\ref{tab:factor_list}.")

    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")
    return "\n".join(tex)


# ── TABLE 2: Threshold Robustness (Beamer + thesis) ──────────────────────

def _build_threshold_beamer(all_results):
    """Beamer slide: 3-state regime alpha (common-beta dummies, cfr. factor_models/03, pca/04)."""
    strategies = [s for s in all_results if 'dummy_interaction' in all_results[s]]
    if not strategies:
        return ""

    tex = []
    tex.append(r"\begin{frame}[t]{Conditional Alpha: Stress Regimes}")
    tex.append(r"\centering\vspace{-0.2cm}\scriptsize")
    tex.append(r"\setlength{\tabcolsep}{6pt}\renewcommand{\arraystretch}{1.15}")
    tex.append(r"\begin{tabular}{l r r r}")
    tex.append(r"\toprule")
    tex.append(r"Strategy & $\alpha_{\text{LOW}}$ & $\alpha_{\text{MED}}$ & $\alpha_{\text{HIGH}}$ \\")
    tex.append(r"\midrule")
    for s in strategies:
        title = TITLE_MAP.get(s, s)
        di = all_results[s]['dummy_interaction']
        cells = []
        for reg in ('LOW', 'MEDIUM', 'HIGH'):
            r = di.get(reg, {})
            if r and not r.get('skip'):
                cells.append(rf"{r['alpha_annualized']:+.2f}\%{_stars_tex(r['alpha_pval'])}")
            else:
                cells.append("--")
        tex.append(rf"{title} & {cells[0]} & {cells[1]} & {cells[2]} \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\vspace{0.15cm}")
    tex.append(r"{\tiny 3-state common-beta dummies on iTraxx Main 5Y "
               r"(LOW $<60$, MEDIUM $\in[60,100)$, HIGH $\geq100$ bps). "
               r"Annualized $\alpha$, HAC SE. *** $p{<}1\%$, ** $p{<}5\%$, * $p{<}10\%$.}")
    tex.append(r"\end{frame}")
    return "\n".join(tex)


def _build_threshold_thesis(all_results):
    """Thesis table: 3-state regime alpha (common-beta dummies)."""
    strategies = [s for s in all_results if 'dummy_interaction' in all_results[s]]
    if not strategies:
        return ""

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Conditional Alpha by Stress Regime}")
    tex.append(r"\label{tab:conditional_alpha_regime}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l r r r}")
    tex.append(r"\toprule")
    tex.append(r"Strategy & LOW & MEDIUM & HIGH \\")
    tex.append(r" & ($<60$ bps) & ($[60,100)$ bps) & ($\geq100$ bps) \\")
    tex.append(r"\midrule")
    for s in strategies:
        title = TITLE_MAP.get(s, s)
        di = all_results[s]['dummy_interaction']
        row = rf"{title}"
        trow = " "
        for reg in ('LOW', 'MEDIUM', 'HIGH'):
            r = di.get(reg, {})
            if r and not r.get('skip'):
                a = r['alpha_annualized']
                sup = _stars_sup(r['alpha_pval'])
                row += rf" & ${a:+.2f}{sup}$\%" if sup else rf" & {a:+.2f}\%"
                trow += rf" & ({_fmt2(r['alpha_tstat'])})"
            else:
                row += r" & --"
                trow += r" & "
        tex.append(row + r" \\")
        tex.append(trow + r" \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"$r_t = \alpha_{\text{LOW}} D_{\text{LOW},t} + \alpha_{\text{MED}} D_{\text{MED},t} + "
               r"\alpha_{\text{HIGH}} D_{\text{HIGH},t} + \sum_j \beta_j X_{jt} + \varepsilon_t$ "
               r"(no intercept; the three regime dummies partition the sample, betas common across "
               r"regimes). Regimes on iTraxx Main 5Y: LOW $<60$, MEDIUM $\in[60,100)$, HIGH $\geq100$ bps. "
               r"Annualized $\alpha$; $t$-statistics in parentheses. Factors from best-subset selection. "
               r"HAC (Newey--West) SE. $^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
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
        canonical = PROJECT_ROOT / "results" / "tables" / fname
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text(content, encoding="utf-8")
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

    for f in written:
        print(f"   ✅ {tables_dir / f}")

    print(f"\n   Total: {len(written)} .tex files generated")

    return written


# ============================================================================
# RUNNER: ALL 5 CONDITIONAL ALPHA TESTS
# ============================================================================

def conditional_alpha_analysis(strategy_name, strategy_path):
    """Run the 3 conditional alpha tests (CFG, 3-state regime, rolling)."""
    print_header(f"(e) CONDITIONAL ALPHA ANALYSIS — {strategy_name}")

    results = {'strategy': strategy_name}

    # (e1) Ferson–Schadt
    try:
        results['ferson_schadt'] = ferson_schadt_conditional(
            strategy_name, strategy_path)
    except Exception as e:
        print(f"\n   ❌ (e1) Error: {e}")
        import traceback; traceback.print_exc()

    # (e2) Regime dummies (3-state, common betas)
    try:
        results['dummy_interaction'] = dummy_interaction(
            strategy_name, strategy_path)
    except Exception as e:
        print(f"\n   ❌ (e2) Error: {e}")
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
    print("\n   Conditioning variable: iTraxx Main 5Y (z[t-1], lagged headline)")
    print(f"   Regime cutoffs (iTraxx Main 5Y): LOW<{LOW_CUT}, MEDIUM∈[{LOW_CUT},{HIGH_CUT}), HIGH≥{HIGH_CUT} bps")
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

        # Regime alpha (3-state common-beta dummies)
        print(f"\n   ── Regime Alpha (iTraxx Main 5Y, cutoffs {LOW_CUT}/{HIGH_CUT} bps) ──")
        print(f"   {'Strategy':<20} {'LOW':>12} {'MEDIUM':>12} {'HIGH':>12}")
        print(f"   {'─' * 58}")
        for name, res in all_results.items():
            di = res.get('dummy_interaction', {})
            def _ra(reg):
                r = di.get(reg, {})
                if r and not r.get('skip'):
                    return f"{r['alpha_annualized']:+.2f}{significance_stars(r['alpha_pval'])}"
                return "--"
            print(f"   {name:<20} {_ra('LOW'):>12} {_ra('MEDIUM'):>12} {_ra('HIGH'):>12}")

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
                        'regime_alpha_ann': {
                            reg: res.get('dummy_interaction', {}).get(reg, {}).get('alpha_annualized')
                            for reg in ('LOW', 'MEDIUM', 'HIGH')},
                        'regime_alpha_pval': {
                            reg: res.get('dummy_interaction', {}).get(reg, {}).get('alpha_pval')
                            for reg in ('LOW', 'MEDIUM', 'HIGH')},
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
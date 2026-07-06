"""
================================================================================
03_subperiod_rolling_analysis.py — Sub-Period & Regime Analysis
================================================================================
Robustness analysis for all strategies across all factor model frameworks.

Analyses:
  (A) Full sample + Half-sample split (temporal stability)
  (B) Regime analysis: LOW/MEDIUM/HIGH on iTraxx Main 5Y (common-beta dummy alphas)
      - Thresholds: 60 / 100 bps (same cutoffs as the fee tiers and RQ3)
      - Consistent with RQ3 and ML/PCA conditional alpha pipelines
  (C) Rolling alpha with iTraxx Main regime shading + confidence bands

Frameworks:
  - Duarte et al. (2007): Mkt-RF, SMB, HML, UMD, RS, RI, RB, R2, R5, R10
  - Active FI (Brooks et al.\\ (2020)): Term, Global_Term, ... , UST_Volatility
  - Fung & Hsieh (2004): SNP, SIZE, PTFSBD, PTFSFX, PTFSCOM, TERM, CREDIT

Strategies:
  - BTP Italia, iTraxx Combined, CDS-Bond Basis

Outputs:
  - Per strategy: subperiod .tex + rolling alpha .pdf
  - Aggregated: cross-strategy regime comparison .tex (thesis + Beamer)
  - JSON summaries

Author:      Alessio Ottaviani
Institution: EDHEC Business School – PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json
import warnings
from inference import cfg_alpha1_bootstrap_p
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

REGRESSION_FREQ = "monthly"
HAC_LAGS = 4  # Newey-West (1994) rule floor(4*(T/100)^(2/9)) = 4 for all our samples; was ad hoc 6
ROLLING_WINDOW = 36  # months

# Strategies (same as 02a)
STRATEGIES = ['BTP_Italia', 'iTraxx_Combined', 'CDS_Bond_Basis']

STRATEGY_LABELS = {
    'BTP_Italia':      'BTP Italia',
    'iTraxx_Combined': 'iTraxx Combined',
    'CDS_Bond_Basis':  'CDS--Bond Basis',
}

# Frameworks and their factor lists
FRAMEWORKS = {
    'Duarte': {
        'factors': ['Mkt-RF', 'SMB', 'HML', 'UMD', 'RS', 'RI', 'RB', 'R2', 'R5', 'R10'],
        'data_pattern': 'regression_data_{strategy}_{region}_{freq}.csv',
        'label': 'Duarte et al.\\ (2007)',
        'short': 'Duarte',
    },
    'ActiveFI': {
        'factors': ['Term', 'Global_Term', 'Global_Aggregate', 'Inflation_Linkers',
                     'Corporate_Credit', 'Emerging_Debt', 'Emerging_Currency', 'UST_Volatility'],
        'data_pattern': 'regression_data_active_fi_{strategy}_{region}_{freq}.csv',
        'label': 'Active FI (Brooks et al.\\ (2020))',
        'short': 'ActiveFI',
    },
    'FungHsieh': {
        'factors': ['SNPMRF', 'SCMLC', 'PTFSBD', 'PTFSFX', 'PTFSCOM', 'R10', 'BAAMTSY'],
        'data_pattern': 'regression_data_fung_hsieh_{strategy}_{region}_{freq}.csv',
        'label': 'Fung \\& Hsieh (2004)',
        'short': 'FH',
    },
}

# Use EUR factors as primary (European strategies)
PRIMARY_REGION = "eur"

# iTraxx Main 5Y stress-regime cutoffs (bps): LOW < LOW_CUT <= MEDIUM < HIGH_CUT <= HIGH
LOW_CUT = 60
HIGH_CUT = 100
DEFAULT_THRESHOLD = HIGH_CUT          # used by the rolling-alpha regime shading

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Stress proxy: iTraxx Main 5Y dal foglio 'cds' di bbg.xlsx (vedi load_stress_proxy_monthly)

# Plot settings
FIGURE_DPI = 150
REGIME_COLORS = {"LOW": "#2ca02c", "MEDIUM": "#ff7f0e", "HIGH": "#d62728"}
plt.style.use('seaborn-v0_8-whitegrid')


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


def load_stress_proxy_monthly():
    """iTraxx Europe Main 5Y level (bps), monthly last -- dal foglio 'cds' di
    bbg.xlsx (PX_LAST), in sostituzione del ritirato
    Tradable_corporate_bond_factors.xlsx."""
    import factor_sources as src
    s = src.load_bloomberg("cds")["ITRX EUR CDSI GEN 5Y Corp"].dropna()
    monthly = s.resample('ME').last().dropna()
    monthly.name = "ITRX_MAIN"
    return monthly


def load_regression_data(strategy, framework, region, freq):
    """Load preprocessed regression data for a strategy/framework combo."""
    strategy_lower = strategy.lower()
    pattern = FRAMEWORKS[framework]['data_pattern']
    filename = pattern.format(strategy=strategy_lower, region=region, freq=freq)
    filepath = PROCESSED_DATA_DIR / filename
    if not filepath.exists():
        return None
    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    # ActiveFI: il CSV ha EU_Term/US_Term (region-specific) -> canonicalizza a 'Term'
    return data.rename(columns={"EU_Term": "Term", "US_Term": "Term"})


def run_ols_hac(y, X, hac_lags=HAC_LAGS):
    """Run OLS with HAC standard errors. Returns results dict or None."""
    if len(y) < 20:
        return None
    X_const = sm.add_constant(X, prepend=True)
    model = sm.OLS(y, X_const)
    try:
        res = model.fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
    except Exception:
        return None

    alpha = float(res.params['const'])
    if REGRESSION_FREQ == "monthly":
        ann = 12
    elif REGRESSION_FREQ == "weekly":
        ann = 52
    else:
        ann = 252

    return {
        'alpha': alpha,
        'alpha_ann': alpha * ann,
        'alpha_tstat': float(res.tvalues['const']),
        'alpha_pval': float(res.pvalues['const']),
        'r2_adj': float(res.rsquared_adj),
        'nobs': int(res.nobs),
    }


# ============================================================================
# (A) FULL SAMPLE + HALF-SAMPLE SPLIT
# ============================================================================

def analysis_subperiod(strategy, framework, region, freq, stress_monthly):
    """
    Full sample, first half, second half (time sub-samples only).
    Returns dict of period → results.
    """
    data = load_regression_data(strategy, framework, region, freq)
    if data is None:
        return None

    y_col = 'Strategy_Return'
    if y_col not in data.columns:
        return None

    factor_list = FRAMEWORKS[framework]['factors']
    available = [f for f in factor_list if f in data.columns]
    if len(available) < 2:
        return None

    data = data.dropna(subset=[y_col] + available)
    y = data[y_col]
    X = data[available]
    T = len(y)

    if T < 30:
        return None

    # Sotto-periodi TEMPORALI (i regimi di stress LOW/MEDIUM/HIGH sono nel Panel B,
    # via analysis_threshold_robustness, metodo dummy a beta comuni)
    mid = T // 2
    periods = {
        'Full Sample': y.index,
        'First Half': y.index[:mid],
        'Second Half': y.index[mid:],
    }

    results = {}
    for period_name, idx in periods.items():
        y_p = y.loc[idx].dropna()
        X_p = X.loc[idx].dropna()
        common = y_p.index.intersection(X_p.index)
        res = run_ols_hac(y_p[common], X_p.loc[common])
        if res:
            res['period'] = period_name
            results[period_name] = res

    return results


# ============================================================================
# (B) THRESHOLD ROBUSTNESS
# ============================================================================

def analysis_threshold_robustness(strategy, framework, region, freq, stress_monthly):
    """
    Alpha per regime di stress LOW / MEDIUM / HIGH (livello iTraxx Main 5Y a
    LOW_CUT / HIGH_CUT), via dummy d'intercetta a beta COMUNI (niente costante):
        r_t = a_LOW*D_LOW + a_MED*D_MED + a_HIGH*D_HIGH + sum_j b_j*X_jt + e_t
    Ogni a_g e' l'alpha (intercetta) del regime g; le beta sono comuni a tutti i
    regimi (robusto quando i mesi HIGH sono pochi). Inferenza Newey-West HAC.
    Ritorna {regime: {alpha_ann, alpha_tstat, alpha_pval, n}} oppure {skip, n}.
    """
    data = load_regression_data(strategy, framework, region, freq)
    if data is None:
        return None

    y_col = 'Strategy_Return'
    factor_list = FRAMEWORKS[framework]['factors']
    available = [f for f in factor_list if f in data.columns]
    if y_col not in data.columns or len(available) < 2:
        return None
    data = data.dropna(subset=[y_col] + available)
    y = data[y_col]
    X = data[available]

    z = stress_monthly.reindex(y.index, method='nearest')
    common = y.index.intersection(z.dropna().index)
    y, X, z = y.loc[common], X.loc[common], z.loc[common]
    if len(y) < 30:
        return None

    D = pd.DataFrame({
        'LOW':    (z < LOW_CUT).astype(float),
        'MEDIUM': ((z >= LOW_CUT) & (z < HIGH_CUT)).astype(float),
        'HIGH':   (z >= HIGH_CUT).astype(float),
    }, index=y.index)
    regimes = [r for r in ('LOW', 'MEDIUM', 'HIGH') if D[r].sum() >= 1]

    ann = 12 if REGRESSION_FREQ == "monthly" else (52 if REGRESSION_FREQ == "weekly" else 252)
    X_d = pd.concat([D[regimes], X], axis=1)   # no costante: le 3 dummy partizionano
    try:
        res = sm.OLS(y, X_d).fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
    except Exception:
        return None

    results = {}
    for r in ('LOW', 'MEDIUM', 'HIGH'):
        n = int(D[r].sum())
        if r in regimes and n >= 5:
            results[r] = {
                'alpha_ann': float(res.params[r]) * ann,
                'alpha_tstat': float(res.tvalues[r]),
                'alpha_pval': float(res.pvalues[r]),
                'n': n,
            }
        else:
            results[r] = {'skip': True, 'n': n}

    return results

# ============================================================================
# (B2) Christopherson, Ferson & Glassman CONDITIONAL ALPHA
# ============================================================================

def analysis_cfg_conditional_alpha(strategy, framework, region, freq, stress_monthly):
    """
    Christopherson, Ferson & Glassman (1998) full conditional model:
    r_t = α₀ + α₁·z_t + Σⱼ (βⱼ + δⱼ·z_t)·Xⱼt + ε
    where z_t = standardized iTraxx Main 5Y. Both the alpha (α₁) and the loadings
    (δⱼ) are conditioned on stress. Inference on α₁ is by moving-block bootstrap.
    """
    data = load_regression_data(strategy, framework, region, freq)
    if data is None:
        return None

    y_col = 'Strategy_Return'
    factor_list = FRAMEWORKS[framework]['factors']
    available = [f for f in factor_list if f in data.columns]
    data = data.dropna(subset=[y_col] + available)
    y = data[y_col]
    X = data[available]

    stress_aligned = stress_monthly.reindex(y.index, method='nearest')
    common = y.index.intersection(stress_aligned.dropna().index)
    y, X = y[common], X.loc[common]
    stress_level = stress_aligned[common]

    if len(y) < 30:
        return None

    # CFG (1998): condition on the PREDETERMINED (one-period-lagged) stress state z_{t-1}.
    # Lag the level, drop the first (undefined) row, realign y/X, then standardize the
    # lagged (primary) and contemporaneous (robustness) series over the estimation sample.
    z_lag_level = stress_level.shift(1)
    keep = z_lag_level.dropna().index
    y, X = y.loc[keep], X.loc[keep]
    stress_level, z_lag_level = stress_level.loc[keep], z_lag_level.loc[keep]
    z = (z_lag_level - z_lag_level.mean()) / z_lag_level.std()             # z_{t-1}  (primary CFG)
    z_contemp = (stress_level - stress_level.mean()) / stress_level.std()  # z_t      (robustness)

    # Full CFG (Christopherson, Ferson & Glassman 1998): condition BOTH the alpha
    # and the loadings on z -> z_stress carries alpha1, X_j*z carry the conditional betas.
    X_cfg = X.copy()
    X_cfg['z_stress'] = z.values
    for f in available:
        X_cfg[f'{f}_x_z'] = X[f].values * z.values
    X_const = sm.add_constant(X_cfg, prepend=True)

    try:
        res = sm.OLS(y, X_const).fit(
            cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
    except Exception:
        return None

    ann = 12 if REGRESSION_FREQ == "monthly" else (52 if REGRESSION_FREQ == "weekly" else 252)

    alpha0 = float(res.params['const']) * ann
    alpha0_t = float(res.tvalues['const'])
    alpha0_p = float(res.pvalues['const'])

    alpha1 = float(res.params['z_stress']) * ann
    alpha1_t = float(res.tvalues['z_stress'])
    # Headline inference on alpha1 = moving-block bootstrap (HAC over-rejects at any
    # lag in finite samples; Kiefer-Vogelsang 2005, Lazarus et al. 2018). HAC kept
    # as robustness in 'alpha1_pval_hac'.
    alpha1_p_hac = float(res.pvalues['z_stress'])
    alpha1_p = cfg_alpha1_bootstrap_p(y, X, z)                 # z_{t-1}  (primary CFG)
    alpha1_p_contemp = cfg_alpha1_bootstrap_p(y, X, z_contemp)  # z_t  (robustness)

    # Conditional alpha at +1σ and -1σ
    alpha_high = alpha0 + alpha1   # z = +1
    alpha_low = alpha0 - alpha1    # z = -1

    return {
        'alpha0_ann': alpha0,
        'alpha0_tstat': alpha0_t,
        'alpha0_pval': alpha0_p,
        'alpha1_ann': alpha1,
        'alpha1_tstat': alpha1_t,
        'alpha1_pval': alpha1_p,                  # bootstrap p, z_{t-1} (HEADLINE)
        'alpha1_pval_hac': alpha1_p_hac,          # HAC p, z_{t-1} (robustness)
        'alpha1_pval_contemp': alpha1_p_contemp,  # bootstrap p, z_t (robustness)
        'alpha_high_1sigma': alpha_high,
        'alpha_low_1sigma': alpha_low,
        'r2_adj': float(res.rsquared_adj),
        'nobs': int(res.nobs),
    }

# ============================================================================
# (C) ROLLING ALPHA WITH REGIME SHADING
# ============================================================================

def analysis_rolling_alpha(strategy, framework, region, freq, stress_monthly):
    """Rolling window alpha with iTraxx Main regime shading."""
    data = load_regression_data(strategy, framework, region, freq)
    if data is None:
        return None

    y_col = 'Strategy_Return'
    factor_list = FRAMEWORKS[framework]['factors']
    available = [f for f in factor_list if f in data.columns]
    data = data.dropna(subset=[y_col] + available)
    y = data[y_col]
    X = data[available]
    T = len(y)
    n_roll = T - ROLLING_WINDOW + 1

    if n_roll <= 5:
        return None

    ann = 12 if REGRESSION_FREQ == "monthly" else (52 if REGRESSION_FREQ == "weekly" else 252)

    from scipy import stats as sp_stats
    t_crit = sp_stats.t.ppf(0.975, df=max(1, ROLLING_WINDOW - len(available) - 1))

    rows = []
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
        rows.append({
            'end_date': y.index[end - 1],
            'alpha_ann': a * ann,
            'ci_lo': (a - t_crit * a_se) * ann,
            'ci_hi': (a + t_crit * a_se) * ann,
        })

    roll_df = pd.DataFrame(rows).set_index('end_date')

    # Regime
    stress_aligned = stress_monthly.reindex(roll_df.index, method='nearest')
    regime = pd.Series("MEDIUM", index=roll_df.index)
    regime[stress_aligned < 60] = "LOW"
    regime[stress_aligned >= DEFAULT_THRESHOLD] = "HIGH"

    return roll_df, regime


# ============================================================================
# PLOT: ROLLING ALPHA PER STRATEGY (best framework)
# ============================================================================

def plot_rolling_alpha(strategy, roll_df, regime, framework_label, framework_key=""):
    """Plot rolling alpha with regime shading for one strategy."""
    fig, ax = plt.subplots(figsize=(14, 4))
    title_str = STRATEGY_LABELS.get(strategy, strategy).replace("--", "–")
    fig.suptitle(
        f"Rolling Alpha ({ROLLING_WINDOW}M) — {title_str}\n({framework_label})",
        fontsize=13, fontweight='bold')

    dates = roll_df.index
    ax.plot(dates, roll_df['alpha_ann'], color='black', linewidth=1.2)
    ax.fill_between(dates, roll_df['ci_lo'], roll_df['ci_hi'],
                    color='grey', alpha=0.2)
    ax.axhline(0, color='grey', linewidth=0.5)

    for rl, color in REGIME_COLORS.items():
        mask = regime == rl
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
    ax.set_xlabel("")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    safe_name = strategy.lower()
    fw_suffix = f"_{framework_key.lower()}" if framework_key else ""
    fig_path = FIGURES_DIR / f"rolling_alpha_regime_{safe_name}{fw_suffix}_{REGRESSION_FREQ}.pdf"
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"   📊 {fig_path.name}")
    return fig_path


# ============================================================================
# TEX GENERATION
# ============================================================================

def generate_tex_thesis(all_subperiod, all_threshold, framework_key):
    """
    Thesis table: Panel A = sub-period alpha, Panel B = threshold robustness.
    Cross-strategy (one column per strategy).
    """
    fw = FRAMEWORKS[framework_key]
    strats = [s for s in STRATEGIES if s in all_subperiod]
    if not strats:
        return ""
    n_s = len(strats)

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\singlespacing")
    tex.append(rf"\caption{{Sub-Period and Regime Analysis: {fw['label']}}}")
    tex.append(rf"\label{{tab:subperiod_regime_{fw['short'].lower()}}}")
    tex.append(r"\begin{minipage}{\textwidth}")
    tex.append(rf"{{\footnotesize\noindent Annualized alpha (\% p.a.) of the three strategies under the "
               rf"{fw['label']} specification, EUR factors, monthly frequency. "
               r"Panel A: OLS with Newey--West HAC standard errors on time sub-samples. "
               r"Panel B: regime alphas from common-beta dummy intercepts "
               r"$r_t = \sum_g \alpha_g D_{g,t} + \sum_j \beta_j X_{jt} + \varepsilon_t$, "
               r"with $g \in \{$LOW, MEDIUM, HIGH$\}$ defined on the iTraxx Main 5Y level "
               rf"(LOW $<{LOW_CUT}$ bps, MEDIUM $[{LOW_CUT},{HIGH_CUT})$, HIGH $\geq {HIGH_CUT}$). "
               r"$N$ = months per sub-sample or regime. "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.}")
    tex.append(r"\end{minipage}")
    tex.append(r"\par\vspace{6pt}")
    tex.append(r"\small")

    # Panel A: Sub-period
    tex.append(r"\vspace{0.2cm}\textit{Panel A: Sub-period alpha (\% p.a.)}")
    tex.append(r"\vspace{0.1cm}")
    tex.append(r"\begin{tabular}{l" + "r r r " * n_s + "}")
    tex.append(r"\toprule")
    h1 = "Period"
    for s in strats:
        h1 += rf" & \multicolumn{{3}}{{c}}{{{STRATEGY_LABELS.get(s, s)}}}"
    h1 += r" \\"
    tex.append(h1)
    cmi = ""
    for i in range(n_s):
        st = 2 + i * 3
        cmi += rf"\cmidrule(lr){{{st}-{st+2}}}"
    tex.append(cmi)
    h2 = " "
    for _ in strats:
        h2 += r" & $\alpha$ & $t$ & $N$"
    h2 += r" \\"
    tex.append(h2)
    tex.append(r"\midrule")

    for period in ['Full Sample', 'First Half', 'Second Half']:
        row = period
        for s in strats:
            r = all_subperiod.get(s, {}).get(period)
            if r is None:
                row += r" & -- & -- & --"
            else:
                row += (rf" & {r['alpha_ann']:+.2f}{significance_stars(r['alpha_pval'])}"
                        rf" & {r['alpha_tstat']:.2f}"
                        rf" & {r['nobs']}")
        row += r" \\"
        tex.append(row)

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    # Panel B: stress regime LOW / MEDIUM / HIGH (alpha via common-beta dummy intercepts)
    tex.append(r"\par\vspace{0.3cm}")
    tex.append(r"\textit{Panel B: Alpha by stress regime (\% p.a.)}")
    tex.append(r"\vspace{0.1cm}")
    tex.append(r"\begin{tabular}{l" + "r r r " * n_s + "}")
    tex.append(r"\toprule")
    h1b = "Regime"
    for s in strats:
        h1b += rf" & \multicolumn{{3}}{{c}}{{{STRATEGY_LABELS.get(s, s)}}}"
    h1b += r" \\"
    tex.append(h1b)
    cmi2 = ""
    for i in range(n_s):
        st = 2 + i * 3
        cmi2 += rf"\cmidrule(lr){{{st}-{st+2}}}"
    tex.append(cmi2)
    h2b = " "
    for _ in strats:
        h2b += r" & $\alpha$ & $t$ & $N$"
    h2b += r" \\"
    tex.append(h2b)
    tex.append(r"\midrule")

    for regime in ['LOW', 'MEDIUM', 'HIGH']:
        row = regime
        for s in strats:
            r = all_threshold.get(s, {}).get(regime)
            if r is None or r.get('skip'):
                n_cell = "--" if r is None else str(r.get('n', '--'))
                row += rf" & -- & -- & {n_cell}"
            else:
                row += (rf" & {r['alpha_ann']:+.2f}{significance_stars(r['alpha_pval'])}"
                        rf" & {r['alpha_tstat']:.2f}"
                        rf" & {r['n']}")
        row += r" \\"
        tex.append(row)

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    tex.append(r"\end{table}")
    return "\n".join(tex)


def generate_tex_beamer(all_subperiod, all_threshold, framework_key):
    """Beamer slide: compact sub-period + threshold for one framework."""
    fw = FRAMEWORKS[framework_key]
    strats = [s for s in STRATEGIES if s in all_subperiod]
    if not strats:
        return ""
    n_s = len(strats)

    tex = []
    tex.append(rf"\begin{{frame}}[t]{{Regime Analysis: {fw['label']}}}")
    tex.append(r"\centering\vspace{-0.3cm}\scriptsize")
    tex.append(r"\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.05}")

    # Panel A
    tex.append(r"\textbf{Panel A: Alpha by sub-period (\% p.a.)}\par\vspace{0.08cm}")
    tex.append(r"\begin{tabular}{l " + "r r " * n_s + "}")
    tex.append(r"\toprule")
    bh = "Period"
    for s in strats:
        bh += rf" & \multicolumn{{2}}{{c}}{{{STRATEGY_LABELS.get(s, s)}}}"
    bh += r" \\"
    tex.append(bh)
    bh2 = " "
    for _ in strats:
        bh2 += r" & $\alpha$ & $t$"
    bh2 += r" \\"
    tex.append(bh2)
    tex.append(r"\midrule")

    for period in ['Full Sample', 'First Half', 'Second Half']:
        row = period
        for s in strats:
            r = all_subperiod.get(s, {}).get(period)
            if r is None:
                row += " & -- & --"
            else:
                row += (rf" & {r['alpha_ann']:+.2f}{significance_stars(r['alpha_pval'])}"
                        rf" & {r['alpha_tstat']:.2f}")
        row += r" \\"
        tex.append(row)

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    # Panel B
    tex.append(r"\vspace{0.15cm}")
    tex.append(r"\textbf{Panel B: Alpha by stress regime (\% p.a.)}"
               r"\par\vspace{0.08cm}")
    tex.append(r"\begin{tabular}{l " + "r r " * n_s + "}")
    tex.append(r"\toprule")
    bh3 = r"Regime"
    for s in strats:
        bh3 += rf" & $\alpha$ & $t$"
    bh3 += r" \\"
    tex.append(bh3)
    tex.append(r"\midrule")

    for regime in ['LOW', 'MEDIUM', 'HIGH']:
        row = regime
        for s in strats:
            r = all_threshold.get(s, {}).get(regime)
            if r is None or r.get('skip'):
                row += " & -- & --"
            else:
                row += (rf" & {r['alpha_ann']:+.2f}{significance_stars(r['alpha_pval'])}"
                        rf" & {r['alpha_tstat']:.2f}")
        row += r" \\"
        tex.append(row)

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    tex.append(r"\vspace{0.1cm}")
    tex.append(rf"{{\tiny {fw['label']}. EUR factors. HAC SE. "
               rf"LOW $<{LOW_CUT}$, MED $[{LOW_CUT},{HIGH_CUT})$, HIGH $\geq {HIGH_CUT}$ bps (iTraxx Main 5Y). "
               r"*** $p{<}1\%$, ** $p{<}5\%$, * $p{<}10\%$.}}")
    tex.append(r"\end{frame}")
    return "\n".join(tex)


# ============================================================================
# COMBINED VIF TABLE (Panels A-C across the three benchmarks)
# ============================================================================

def write_combined_vif_table():
    """Single VIF table across the three benchmark specifications (EUR factors),
    assembled from the JSON dumps of 02a/02b/02c. Note above (JoF style)."""
    import json
    panels = [("Duarte",    "Panel A: Duarte et al.\\ (2007)"),
              ("FungHsieh", "Panel B: Fung \\& Hsieh (2004)"),
              ("ActiveFI",  "Panel C: Brooks et al.\\ (2020)")]
    strat_order = ['BTP_Italia', 'CDS_Bond_Basis', 'iTraxx_Combined']
    data = {}
    for key, _ in panels:
        fp = TABLES_DIR / f"{key}_VIF_{REGRESSION_FREQ}.json"
        if not fp.exists():
            print(f"   ⚠ {fp.name} not found — run 02a/02b/02c first; combined VIF table skipped")
            return
        data[key] = json.loads(fp.read_text(encoding='utf-8'))
    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\singlespacing")
    tex.append(r"\caption{Variance Inflation Factors --- Benchmark Factor Models, EUR Factors}")
    tex.append(r"\label{tab:vif_benchmarks}")
    tex.append(r"\begin{minipage}{\textwidth}")
    tex.append(r"{\footnotesize\noindent Variance inflation factors of the EUR factors of the three "
               r"benchmark specifications, computed on each factor matrix over each strategy's "
               r"estimation sample. VIF $> 10$ indicates serious multicollinearity; VIF $> 5$ "
               r"moderate concerns. Elevated VIF for $R_2$, $R_5$, $R_{10}$ (Panel A) reflects the "
               r"inherent correlation of maturity-sorted government bond returns; elevated VIF for "
               r"Global\_Term and Global\_Aggregate (Panel C) reflects the strong correlation "
               r"between global bond benchmarks. Neither affects inference on alpha.}")
    tex.append(r"\end{minipage}")
    tex.append(r"\par\vspace{6pt}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l r r r}")
    tex.append(r"\toprule")
    tex.append(r"Factor & BTP Italia & CDS--Bond Basis & iTraxx Combined \\")
    tex.append(r"\midrule")
    for key, panel_label in panels:
        tex.append(rf"\multicolumn{{4}}{{l}}{{\textbf{{{panel_label}}}}} \\")
        tex.append(r"\addlinespace")
        for factor in FRAMEWORKS[key]['factors']:
            row = factor.replace('_', r'\_')
            for s in strat_order:
                v = data[key].get(s, {}).get(factor)
                row += f" & {v:.2f}" if v is not None else " & --"
            tex.append(row + r" \\")
        if key != panels[-1][0]:
            tex.append(r"\addlinespace")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{table}")
    out = TABLES_DIR / f"VIF_combined_article_{REGRESSION_FREQ}.tex"
    out.write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"   ✅ {out.name}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("SUB-PERIOD & REGIME ANALYSIS — ALL FRAMEWORKS")
    print(f"\n   Strategies: {STRATEGIES}")
    print(f"   Frameworks: {list(FRAMEWORKS.keys())}")
    print(f"   Frequency:  {REGRESSION_FREQ}")
    print(f"   Region:     {PRIMARY_REGION}")
    print(f"   Stress regimes: LOW <{LOW_CUT} | MEDIUM [{LOW_CUT},{HIGH_CUT}) | HIGH >={HIGH_CUT} bps")

    # Load stress proxy
    print_header("Loading Stress Proxy", "-")
    stress_monthly = load_stress_proxy_monthly()
    print(f"   iTraxx Main: {len(stress_monthly)} months,"
          f" range [{stress_monthly.min():.0f}, {stress_monthly.max():.0f}] bps")

    all_results = {}  # framework → strategy → analysis results

    for fw_key, fw_cfg in FRAMEWORKS.items():
        print_header(f"FRAMEWORK: {fw_cfg['label']}")

        fw_subperiod = {}
        fw_threshold = {}
        fw_rolling = {}

        for strategy in STRATEGIES:
            print(f"\n   ── {strategy} ──")

            # (A) Sub-period
            sp_results = analysis_subperiod(
                strategy, fw_key, PRIMARY_REGION, REGRESSION_FREQ, stress_monthly)
            if sp_results:
                fw_subperiod[strategy] = sp_results
                for period, r in sp_results.items():
                    print(f"   {period:15s}: α={r['alpha_ann']:+6.2f}%"
                          f" (t={r['alpha_tstat']:5.2f})"
                          f" {significance_stars(r['alpha_pval']):3s}"
                          f" N={r['nobs']}")
            else:
                print(f"   ⚠️  No data for {strategy}/{fw_key}/{PRIMARY_REGION}")

            # (B) Threshold robustness
            th_results = analysis_threshold_robustness(
                strategy, fw_key, PRIMARY_REGION, REGRESSION_FREQ, stress_monthly)
            if th_results:
                fw_threshold[strategy] = th_results
                for reg in ('LOW', 'MEDIUM', 'HIGH'):
                    r = th_results.get(reg, {})
                    if not r or r.get('skip'):
                        print(f"   {reg:<7}: skipped (n={r.get('n', 0)})")
                    else:
                        print(f"   {reg:<7}: α={r['alpha_ann']:+.2f}%"
                              f" (t={r['alpha_tstat']:.2f},"
                              f" p={r['alpha_pval']:.4f}, n={r['n']})"
                              f" {significance_stars(r['alpha_pval'])}")
             # (B2) CFG (Christopherson-Ferson-Glassman) conditional alpha
            fs_result = analysis_cfg_conditional_alpha(
                strategy, fw_key, PRIMARY_REGION, REGRESSION_FREQ, stress_monthly)
            if fs_result:
                fw_subperiod[strategy]['CFG'] = fs_result
                print(f"   CFG: α₀={fs_result['alpha0_ann']:+.2f}%"
                      f" α₁={fs_result['alpha1_ann']:+.2f}%"
                      f" (t={fs_result['alpha1_tstat']:.2f},"
                      f" p={fs_result['alpha1_pval']:.4f})"
                      f" {significance_stars(fs_result['alpha1_pval'])}")
                
            # (C) Rolling alpha for ALL frameworks
            roll_result = analysis_rolling_alpha(
                strategy, fw_key, PRIMARY_REGION, REGRESSION_FREQ, stress_monthly)
            if roll_result:
                roll_df, regime = roll_result
                fw_rolling[strategy] = (roll_df, regime)
                plot_rolling_alpha(strategy, roll_df, regime, fw_cfg['label'], fw_key)

        all_results[fw_key] = {
            'subperiod': fw_subperiod,
            'threshold': fw_threshold,
            'rolling': fw_rolling,
        }

        # Generate .tex for this framework
        if fw_subperiod:
            # Thesis table
            thesis_tex = generate_tex_thesis(fw_subperiod, fw_threshold, fw_key)
            if thesis_tex:
                fname = f"subperiod_regime_{fw_cfg['short'].lower()}_{REGRESSION_FREQ}.tex"
                (TABLES_DIR / fname).write_text(thesis_tex, encoding="utf-8")
                print(f"\n   ✅ {fname}")

            # Beamer slide
            beamer_tex = generate_tex_beamer(fw_subperiod, fw_threshold, fw_key)
            if beamer_tex:
                fname = f"subperiod_regime_{fw_cfg['short'].lower()}_{REGRESSION_FREQ}_slide.tex"
                (TABLES_DIR / fname).write_text(beamer_tex, encoding="utf-8")
                print(f"   ✅ {fname}")

    # ── Cross-Framework Summary ───────────────────────────────────────
    print_header("CROSS-FRAMEWORK SUMMARY")

    print(f"\n   Full-Sample Alpha (% ann.):")
    print(f"   {'Strategy':<20}", end="")
    for fw_key in FRAMEWORKS:
        print(f" {FRAMEWORKS[fw_key]['short']:>10}", end="")
    print()
    print(f"   {'─' * (20 + 11 * len(FRAMEWORKS))}")

    for strategy in STRATEGIES:
        row = f"   {strategy:<20}"
        for fw_key in FRAMEWORKS:
            r = all_results.get(fw_key, {}).get('subperiod', {}).get(
                strategy, {}).get('Full Sample')
            if r:
                row += f" {r['alpha_ann']:>+9.2f}{significance_stars(r['alpha_pval'])}"
            else:
                row += f" {'--':>10}"
        print(row)

    print(f"\n   HIGH − LOW Δα (% ann.):")
    print(f"   {'Strategy':<20}", end="")
    for fw_key in FRAMEWORKS:
        print(f" {FRAMEWORKS[fw_key]['short']:>10}", end="")
    print()
    print(f"   {'─' * (20 + 11 * len(FRAMEWORKS))}")

    for strategy in STRATEGIES:
        row = f"   {strategy:<20}"
        for fw_key in FRAMEWORKS:
            rg = all_results.get(fw_key, {}).get('threshold', {}).get(strategy, {})
            rh = rg.get('HIGH')
            rl = rg.get('LOW')
            if rh and not rh.get('skip') and rl and not rl.get('skip'):
                diff = rh['alpha_ann'] - rl['alpha_ann']
                row += f" {diff:>+10.2f}"
            else:
                row += f" {'--':>10}"
        print(row)

    # Save JSON summary
    json_summary = {}
    for fw_key in FRAMEWORKS:
        fw_data = all_results.get(fw_key, {})
        json_summary[fw_key] = {}
        for strategy in STRATEGIES:
            sp = fw_data.get('subperiod', {}).get(strategy, {})
            th = fw_data.get('threshold', {}).get(strategy, {})
            json_summary[fw_key][strategy] = {
                'subperiod': {k: {kk: vv for kk, vv in v.items()
                                  if kk != 'period'} for k, v in sp.items()},
                'threshold': {str(k): v for k, v in th.items()} if th else {},
            }

    with open(TABLES_DIR / f"subperiod_regime_summary_{REGRESSION_FREQ}.json", 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f"\n   💾 subperiod_regime_summary_{REGRESSION_FREQ}.json")
    
    # ── Composite Rolling Alpha Figure (for paper body) ──
    plot_composite_rolling_alpha(all_results, stress_monthly)

    # ── Combined VIF table (Panels A-C across benchmarks) ──
    write_combined_vif_table()

    # ── Alpha Synthesis Table (article) ───────────────────────────────
    synthesis_path = TABLES_DIR / f"alpha_synthesis_across_models_{REGRESSION_FREQ}.tex"

    fw_labels = {
        'Duarte': 'Duarte et al.\\ (2007)',
        'ActiveFI': 'Brooks et al.\\ (2020)',
        'FungHsieh': 'Fung \\& Hsieh (2004)',
    }

    with open(synthesis_path, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\singlespacing\n")
        f.write("\\caption{Alpha Estimates Across Benchmark Factor Models}\n")
        f.write("\\label{tab:alpha_synthesis}\n")
        f.write("\\begin{minipage}{\\textwidth}\n")
        f.write("{\\footnotesize\\noindent \\textit{Note:} ")
        f.write("Panel A reports annualized alpha (\\% p.a.) from monthly OLS regressions ")
        f.write("with Newey--West HAC standard errors using EUR factors. ")
        f.write("Panel B reports the intercept $\\alpha_0$ and the stress-loading ")
        f.write("coefficient $\\alpha_1$ from the \\citet{christopherson1998conditioning} ")
        f.write("full conditional model ")
        f.write("$r_t = \\alpha_0 + \\alpha_1 z_{t-1} + (\\beta + \\delta z_{t-1})' X_t + \\varepsilon_t$, ")
        f.write("in which both the alpha and the factor loadings are conditioned on $z_{t-1}$, ")
        f.write("the one-month-lagged standardized iTraxx Main 5Y spread (a predetermined ")
        f.write("conditioning variable); $\\bar{R}^2$ is the adjusted $R^2$. ")
        f.write("Significance of $\\alpha_1$ is assessed by a moving-block bootstrap ")
        f.write("($B=9{,}999$, $H_0{:}\\,\\alpha_1=0$ imposed, block length~9). ")
        f.write("Panel C reports out-of-sample alpha in the recursive design of ")
        f.write("\\citet{welch2008comprehensive}: loadings estimated by OLS on the expanding ")
        f.write("past-only window $[1,t)$ (60-month burn-in), and the realized abnormal return ")
        f.write("$e_{i,t} = r_{i,t} - \\hat{\\boldsymbol{\\beta}}_{i,t-1}'\\,\\mathbf{f}_t$ tested with ")
        f.write("Newey--West HAC; IR is the annualized information ratio of the hedged return, ")
        f.write("and $N$ the number of out-of-sample months, common to the three benchmarks ")
        f.write("within each strategy ")
        f.write("(rolling-window counterpart: Appendix~\\ref{app:oos_rolling}). ")
        f.write("$t$-statistics in parentheses are Newey--West HAC throughout. ")
        f.write("$^{***}\\,p<0.01$, $^{**}\\,p<0.05$, $^{*}\\,p<0.10$.}\n")
        f.write("\\end{minipage}\n")
        f.write("\\vspace{2pt}\n\n")
        f.write("\\footnotesize\n\n")

        n_fw = len(FRAMEWORKS)

        # ── Panel A: Full-Sample Alpha ──
        f.write("\\centerline{\\textit{Panel A: Full-sample alpha (\\% p.a.)}}\n")
        f.write("\\vspace{2pt}\n\n")
        f.write("\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}l" + " r r r" * n_fw + "}\n")
        f.write("\\toprule\n")

        # Header row 1: framework names
        f.write("Strategy")
        for fw_key in FRAMEWORKS:
            lab = fw_labels.get(fw_key, FRAMEWORKS[fw_key]['label'])
            f.write(f" & \\multicolumn{{3}}{{c}}{{{lab}}}")
        f.write(" \\\\\n")

        # Cmidrules
        for i, fw_key in enumerate(FRAMEWORKS):
            st = 2 + i * 3
            f.write(f"\\cmidrule(lr){{{st}-{st + 2}}}")
        f.write("\n")

        # Header row 2
        f.write(" ")
        for _ in FRAMEWORKS:
            f.write(" & $\\alpha$ & $t$ & $N$")
        f.write(" \\\\\n")
        f.write("\\midrule\n")

        # Data rows
        for strategy in STRATEGIES:
            display = STRATEGY_LABELS.get(strategy, strategy)
            f.write(f"\\textit{{{display}}}")
            for fw_key in FRAMEWORKS:
                r = all_results.get(fw_key, {}).get('subperiod', {}).get(
                    strategy, {}).get('Full Sample')
                if r:
                    stars = significance_stars(r['alpha_pval'])
                    if stars:
                        f.write(f" & ${r['alpha_ann']:+.2f}^{{{stars}}}$")
                    else:
                        f.write(f" & {r['alpha_ann']:+.2f}")
                    f.write(f" & {r['alpha_tstat']:.2f}")
                    f.write(f" & {r['nobs']}")
                else:
                    f.write(" & -- & -- & --")
            f.write(" \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular*}\n")
        f.write("\\par\\vspace{10pt}\n\n")


        # ── Panel B: CFG (Christopherson-Ferson-Glassman) Conditional Alpha ──
        f.write("\\centerline{\\textit{Panel B: Christopherson, Ferson, and Glassman (1998) conditional alpha (\\% p.a.)}}\n")
        f.write("\\vspace{2pt}\n\n")
        f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + " rrr" * n_fw + "}\n")
        f.write("\\toprule\n")

        # Header row 1: framework names (span 3 columns each)
        f.write(" ")
        for fw_key in FRAMEWORKS:
            lab = fw_labels.get(fw_key, FRAMEWORKS[fw_key]['label'])
            f.write(f" & \\multicolumn{{3}}{{c}}{{{lab}}}")
        f.write(" \\\\\n")

        for i, fw_key in enumerate(FRAMEWORKS):
            st = 2 + i * 3
            f.write(f"\\cmidrule(lr){{{st}-{st + 2}}}")
        f.write("\n")

        # Header row 2
        f.write(" ")
        for _ in FRAMEWORKS:
            f.write(r" & $\alpha_0$ & $\alpha_1$ & $\bar{R}^2$")
        f.write(" \\\\\n")
        f.write("\\midrule\n")

        # Data rows: each strategy spans two lines (coefficients, then t-stats)
        for strategy in STRATEGIES:
            display = STRATEGY_LABELS.get(strategy, strategy)

            # Line 1: alpha0, alpha1 (with stars), R^2
            f.write(f"\\textit{{{display}}}")
            for fw_key in FRAMEWORKS:
                fs = all_results.get(fw_key, {}).get('subperiod', {}).get(
                    strategy, {}).get('CFG')
                if fs:
                    s0 = significance_stars(fs['alpha0_pval'])
                    s1 = significance_stars(fs['alpha1_pval'])
                    a0 = (f"${fs['alpha0_ann']:+.2f}^{{{s0}}}$" if s0
                          else f"{fs['alpha0_ann']:+.2f}")
                    a1 = (f"${fs['alpha1_ann']:+.2f}^{{{s1}}}$" if s1
                          else f"{fs['alpha1_ann']:+.2f}")
                    f.write(f" & {a0} & {a1} & {fs['r2_adj']:.2f}")
                else:
                    f.write(" & -- & -- & --")
            f.write(" \\\\\n")

            # Line 2: t-statistics in parentheses under alpha0 and alpha1
            f.write(" ")
            for fw_key in FRAMEWORKS:
                fs = all_results.get(fw_key, {}).get('subperiod', {}).get(
                    strategy, {}).get('CFG')
                if fs:
                    f.write(f" & ({fs['alpha0_tstat']:.2f})"
                            f" & ({fs['alpha1_tstat']:.2f}) & ")
                else:
                    f.write(" & & & ")
            f.write(" \\\\\n")
            f.write("\\addlinespace\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular*}\n")
        f.write("\\par\\vspace{10pt}\n\n")

        # ── Panel C: Out-of-sample alpha (inlined from 07_oos_alpha.py output) ──
        # NB: run 07_oos_alpha.py BEFORE this script so panel_d_oos_alpha.tex exists.
        panel_d_tex = TABLES_DIR / "panel_d_oos_alpha.tex"
        if panel_d_tex.exists():
            f.write(panel_d_tex.read_text(encoding="utf-8"))
            f.write("\n")
        else:
            print("   ⚠ panel_d_oos_alpha.tex not found — run 07_oos_alpha.py first")    
        f.write("\\end{table}\n")

    print(f"\n   ✅ {synthesis_path.name}  ← PAPER & SKELETON")

    # ── Alpha Synthesis Table (PRESENTATION SLIDE) ────────────────────
    presentation_path = TABLES_DIR / f"alpha_synthesis_Presentation_Slide_{REGRESSION_FREQ}.tex"

    # Slide-only ordering: BTP → CDS-Bond → iTraxx (matches benchmark slides)
    slide_strategies = ['BTP_Italia', 'CDS_Bond_Basis', 'iTraxx_Combined']

    with open(presentation_path, 'w', encoding='utf-8') as f:
        f.write("%" + "-" * 60 + "\n")
        f.write("% ALPHA SYNTHESIS — PRESENTATION SLIDE\n")
        f.write("% Layout: Full-sample (TL) + Conditional Alpha (TR) + Regime (bottom centered)\n")
        f.write("% Wrap with \\begin{frame}{...} ... \\end{frame} in main file\n")
        f.write("%" + "-" * 60 + "\n\n")

        # ============================================================
        # TOP ROW: 2 columns side-by-side (Full-sample alpha | Conditional alpha)
        # ============================================================
        f.write("\\begin{columns}[T]\n")

        # ---- LEFT: Full-sample alpha ----
        f.write("\\begin{column}{0.5\\textwidth}\n")
        f.write("\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n\n")

        f.write("\\textbf{Full-sample alpha (\\% p.a.)}\\\\[4pt]\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write(" & \\textit{BTP Italia} & \\textit{CDS--Bond} & \\textit{iTraxx} \\\\\n")
        f.write("\\midrule\n")
        for fw_key in FRAMEWORKS:
            lab = fw_labels.get(fw_key, FRAMEWORKS[fw_key]['label'])
            f.write(f"{lab}")
            for strategy in slide_strategies:
                r = all_results.get(fw_key, {}).get('subperiod', {}).get(
                    strategy, {}).get('Full Sample')
                if r:
                    stars = significance_stars(r['alpha_pval'])
                    if stars:
                        f.write(f" & ${r['alpha_ann']:+.2f}^{{{stars}}}$")
                    else:
                        f.write(f" & {r['alpha_ann']:+.2f}")
                else:
                    f.write(" & --")
            f.write(" \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{column}\n")

        # ---- RIGHT: Conditional Alpha (CFG) ----
        f.write("\\begin{column}{0.5\\textwidth}\n")
        f.write("\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n\n")

        f.write("\\textbf{Conditional Alpha}")
        f.write(" {\\scriptsize\\textit{(CFG(1998), $\\alpha_1$ per 1$\\sigma$ stress)}}")
        f.write("\\\\[4pt]\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write(" & \\textit{BTP Italia} & \\textit{CDS--Bond} & \\textit{iTraxx} \\\\\n")
        f.write("\\midrule\n")
        for fw_key in FRAMEWORKS:
            lab = fw_labels.get(fw_key, FRAMEWORKS[fw_key]['label'])
            f.write(f"{lab}")
            for strategy in slide_strategies:
                fs = all_results.get(fw_key, {}).get('subperiod', {}).get(
                    strategy, {}).get('CFG')
                if fs:
                    stars = significance_stars(fs['alpha1_pval'])
                    is_sig = stars in ['***', '**']
                    cell = f"${fs['alpha1_ann']:+.2f}"
                    if stars:
                        cell += f"^{{{stars}}}"
                    cell += "$"
                    if is_sig:
                        f.write(f" & \\hlt{{{cell}}}")
                    else:
                        f.write(f" & {cell}")
                else:
                    f.write(" & --")
            f.write(" \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{column}\n")

        f.write("\\end{columns}\n\n")

        # ============================================================
        # BOTTOM ROW: Regime decomposition (centered, full width)
        # ============================================================
        f.write("\\vspace{0.4cm}\n\n")
        f.write("\\begin{center}\n")
        f.write("\\scriptsize\n\\setlength{\\tabcolsep}{4pt}\n\n")

        f.write("\\textbf{Regime decomposition}")
        f.write(f" {{\\scriptsize\\textit{{(LOW $<{LOW_CUT}$ | MED $[{LOW_CUT},{HIGH_CUT})$ | HIGH $\\geq {HIGH_CUT}$ bps, iTraxx Main 5Y)}}}}")
        f.write("\\\\[4pt]\n")
        n_fw = len(FRAMEWORKS)
        f.write("\\begin{tabular}{l" + "@{\\hspace{8pt}}ccc" * n_fw + "}\n")
        f.write("\\toprule\n")
        # Header row 1: framework names with multicolumn
        f.write(" ")
        for fw_key in FRAMEWORKS:
            lab = fw_labels.get(fw_key, FRAMEWORKS[fw_key]['label'])
            f.write(f" & \\multicolumn{{3}}{{c}}{{{lab}}}")
        f.write(" \\\\\n")
        # cmidrules
        for i, fw_key in enumerate(FRAMEWORKS):
            st = 2 + i * 3
            f.write(f"\\cmidrule(lr){{{st}-{st + 2}}}")
        f.write("\n")
        # Header row 2: LOW / MED / HIGH
        f.write(" ")
        for _ in FRAMEWORKS:
            f.write(" & LOW & MED & HIGH")
        f.write(" \\\\\n")
        f.write("\\midrule\n")
        # Data rows: one per strategy (slide ordering)
        for strategy in slide_strategies:
            display = STRATEGY_LABELS.get(strategy, strategy)
            f.write(f"\\textit{{{display}}}")
            for fw_key in FRAMEWORKS:
                rg = all_results.get(fw_key, {}).get('threshold', {}).get(strategy, {})
                for reg in ('LOW', 'MEDIUM', 'HIGH'):
                    r = rg.get(reg)
                    if r is None or r.get('skip'):
                        f.write(" & --")
                        continue
                    stars = significance_stars(r['alpha_pval'])
                    cell = (f"${r['alpha_ann']:+.2f}^{{{stars}}}$" if stars
                            else f"{r['alpha_ann']:+.2f}")
                    if reg == 'HIGH':
                        f.write(f" & \\hlt{{{cell}}}")
                    else:
                        f.write(f" & {cell}")
            f.write(" \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{center}\n")

    print(f"   ✅ {presentation_path.name}  ← PRESENTATION SLIDE")

    print(f"\n{'=' * 80}")
    print(f"✅ SUB-PERIOD & REGIME ANALYSIS COMPLETE")
    print(f"{'=' * 80}")

def plot_composite_rolling_alpha(all_results, stress_monthly):
    """
    Composite figure: 3 panels (one per strategy), each with 3 lines
    (one per framework) + regime shading. For the paper body.
    """
    fw_colors = {
        'Duarte': '#1f77b4',
        'ActiveFI': '#ff7f0e',
        'FungHsieh': '#2ca02c',
    }
    fw_labels_short = {
        'Duarte': 'Duarte et al. (2007)',
        'ActiveFI': 'Brooks et al. (2020)',
        'FungHsieh': 'Fung & Hsieh (2004)',
    }

    fig, axes = plt.subplots(len(STRATEGIES), 1, figsize=(14, 4 * len(STRATEGIES)),
                              sharex=True)
    if len(STRATEGIES) == 1:
        axes = [axes]

    for idx, strategy in enumerate(STRATEGIES):
        ax = axes[idx]
        display = STRATEGY_LABELS.get(strategy, strategy)
        ax.set_title(display, fontsize=13, fontweight='bold')

        # Regime shading (from first available framework)
        for fw_key in FRAMEWORKS:
            rolling_data = all_results.get(fw_key, {}).get('rolling', {}).get(strategy)
            if rolling_data:
                _, regime = rolling_data
                dates = regime.index
                for rl, color in [('HIGH', '#d62728')]:
                    mask = (regime == rl)
                    if not mask.any():
                        continue
                    changes = mask.astype(int).diff().fillna(0)
                    starts = dates[changes == 1]
                    ends = dates[changes == -1]
                    if mask.iloc[0]:
                        starts = starts.insert(0, dates[0])
                    if mask.iloc[-1]:
                        ends = ends.append(pd.DatetimeIndex([dates[-1]]))
                    alpha_sh = 0.12 if rl != "HIGH" else 0.20
                    for s, e in zip(starts[:len(ends)], ends[:len(starts)]):
                        ax.axvspan(s, e, alpha=alpha_sh, color=color, zorder=0)
                break  # shading from first framework only

        # Plot rolling alpha for each framework
        for fw_key in FRAMEWORKS:
            rolling_data = all_results.get(fw_key, {}).get('rolling', {}).get(strategy)
            if rolling_data is None:
                continue
            roll_df, _ = rolling_data
            ax.plot(roll_df.index, roll_df['alpha_ann'],
                    color=fw_colors.get(fw_key, 'gray'),
                    label=fw_labels_short.get(fw_key, fw_key),
                    linewidth=1.3, zorder=3)

        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_ylabel("α (% p.a.)", fontsize=11)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=9, loc='upper right')

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].set_xlabel("")

    fig.tight_layout()
    fig_path = FIGURES_DIR / f"rolling_alpha_composite_{REGRESSION_FREQ}.pdf"
    fig.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"\n   📊 COMPOSITE: {fig_path.name}")
    return fig_path

if __name__ == "__main__":
    main()
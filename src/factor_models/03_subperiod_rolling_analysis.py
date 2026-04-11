"""
================================================================================
03_subperiod_rolling_analysis.py — Sub-Period & Regime Analysis
================================================================================
Robustness analysis for all strategies across all factor model frameworks.

Analyses:
  (A) Full sample + Half-sample split (temporal stability)
  (B) Regime analysis: HIGH vs NORMAL using iTraxx Main 5Y thresholds
      - Thresholds: 80, 100, 120 bps (Patton 2009, RFS)
      - Consistent with RQ3 and ML/PCA conditional alpha pipelines
  (C) Rolling alpha with iTraxx Main regime shading + confidence bands

Frameworks:
  - Duarte et al. (2007): Mkt-RF, SMB, HML, UMD, RS, RI, RB, R2, R5, R10
  - Active FI (Brooks et al.\\ (2020)): Term, Global_Term, ... , UST_Volatility
  - Fung & Hsieh (2001): SNP, SIZE, PTFSBD, PTFSFX, PTFSCOM, TERM, CREDIT

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
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

REGRESSION_FREQ = "monthly"
HAC_LAGS = 6
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
        'factors': ['SNP', 'SIZE', 'PTFSBD', 'PTFSFX', 'PTFSCOM', 'TERM', 'CREDIT'],
        'data_pattern': 'regression_data_fung_hsieh_{strategy}_{region}_{freq}.csv',
        'label': 'Fung \\& Hsieh (2001)',
        'short': 'FH',
    },
}

# Use EUR factors as primary (European strategies)
PRIMARY_REGION = "eur"

# iTraxx Main thresholds for regime analysis (bps)
ITRX_THRESHOLDS = [80, 100, 120]
DEFAULT_THRESHOLD = 100

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

# Stress proxy
FACTORS_EXTERNAL_DIR = DATA_DIR / "external" / "factors"
TRADABLE_CB_FILE = FACTORS_EXTERNAL_DIR / "Tradable_corporate_bond_factors.xlsx"

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
    """Load iTraxx Main 5Y as monthly Series."""
    raw = pd.read_excel(
        TRADABLE_CB_FILE, sheet_name="CDS_INDEX",
        skiprows=14, usecols=[0, 1], header=0
    )
    raw.columns = ["Date", "value"]
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date")
    daily = pd.to_numeric(raw["value"], errors="coerce").dropna()
    monthly = daily.resample('ME').last().dropna()
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
    return data


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
    Full sample, first half, second half, HIGH regime, NORMAL regime.
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

    # Align stress proxy
    stress_aligned = stress_monthly.reindex(y.index, method='nearest')

    # Define periods
    mid = T // 2
    periods = {
        'Full Sample': y.index,
        'First Half': y.index[:mid],
        'Second Half': y.index[mid:],
    }

    # Regime split at default threshold
    D_high = stress_aligned > DEFAULT_THRESHOLD
    if D_high.sum() >= 10 and (~D_high).sum() >= 10:
        periods['HIGH'] = y.index[D_high.reindex(y.index, fill_value=False).astype(bool)]
        periods['NORMAL'] = y.index[(~D_high).reindex(y.index, fill_value=False).astype(bool)]

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
    Test α₁ from dummy interaction at 80, 100, 120 bps.
    r_t = α₀ + α₁·D_HIGH + Σ βⱼ·Xⱼt + ε
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

    results = {}
    for th in ITRX_THRESHOLDS:
        D_high = (stress_level > th).astype(float)
        n_high = int(D_high.sum())
        if n_high < 10 or (len(y) - n_high) < 10:
            results[th] = {'threshold': th, 'n_HIGH': n_high, 'skip': True}
            continue

        X_d = X.copy()
        X_d['D_HIGH'] = D_high.values
        X_const = sm.add_constant(X_d, prepend=True)
        try:
            res = sm.OLS(y, X_const).fit(
                cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS})
        except Exception:
            results[th] = {'threshold': th, 'skip': True}
            continue

        ann = 12 if REGRESSION_FREQ == "monthly" else (52 if REGRESSION_FREQ == "weekly" else 252)

        results[th] = {
            'threshold': th,
            'n_HIGH': n_high,
            'n_LOW': len(y) - n_high,
            'alpha0_ann': float(res.params['const']) * ann,
            'alpha1_ann': float(res.params['D_HIGH']) * ann,
            'alpha1_tstat': float(res.tvalues['D_HIGH']),
            'alpha1_pval': float(res.pvalues['D_HIGH']),
            'alpha_HIGH_ann': float(res.params['const'] + res.params['D_HIGH']) * ann,
            'r2_adj': float(res.rsquared_adj),
        }

    return results

# ============================================================================
# (B2) FERSON-SCHADT CONDITIONAL ALPHA
# ============================================================================

def analysis_ferson_schadt(strategy, framework, region, freq, stress_monthly):
    """
    Ferson-Schadt (1996) conditional model:
    r_t = α₀ + α₁·z_t + Σ βⱼ·Xⱼt + ε
    where z_t = standardized iTraxx Main 5Y.
    α₁ > 0: alpha increases in stress.
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

    # Standardize stress proxy
    z = (stress_level - stress_level.mean()) / stress_level.std()

    X_fs = X.copy()
    X_fs['z_stress'] = z.values
    X_const = sm.add_constant(X_fs, prepend=True)

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
    alpha1_p = float(res.pvalues['z_stress'])

    # Conditional alpha at +1σ and -1σ
    alpha_high = alpha0 + alpha1   # z = +1
    alpha_low = alpha0 - alpha1    # z = -1

# Joint F-test: α₀ = α₁ = 0
    try:
        r_matrix = np.zeros((2, len(res.params)))
        r_matrix[0, 0] = 1  # const
        r_matrix[1, list(X_const.columns).index('z_stress')] = 1
        wald = res.f_test(r_matrix)
        f_stat = float(wald.statistic)
        f_pval = float(wald.pvalue)
    except Exception:
        f_stat = np.nan
        f_pval = np.nan

    return {
        'alpha0_ann': alpha0,
        'alpha0_tstat': alpha0_t,
        'alpha0_pval': alpha0_p,
        'alpha1_ann': alpha1,
        'alpha1_tstat': alpha1_t,
        'alpha1_pval': alpha1_p,
        'alpha_high_1sigma': alpha_high,
        'alpha_low_1sigma': alpha_low,
        'f_stat': f_stat,
        'f_pval': f_pval,
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
    tex.append(rf"\caption{{Sub-Period and Regime Analysis: {fw['label']}}}")
    tex.append(rf"\label{{tab:subperiod_regime_{fw['short'].lower()}}}")
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

    for period in ['Full Sample', 'First Half', 'Second Half', 'HIGH', 'NORMAL']:
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

    # Panel B: Threshold robustness — alpha levels by regime
    tex.append(r"\par\vspace{0.3cm}")
    tex.append(r"\textit{Panel B: Alpha by stress threshold (\% p.a.)}")
    tex.append(r"\vspace{0.1cm}")
    tex.append(r"\begin{tabular}{l r " + "r r " * n_s + "}")
    tex.append(r"\toprule")
    h1b = r"Threshold & $n_H$"
    for s in strats:
        h1b += rf" & \multicolumn{{2}}{{c}}{{{STRATEGY_LABELS.get(s, s)}}}"
    h1b += r" \\"
    tex.append(h1b)
    cmi2 = ""
    for i in range(n_s):
        st = 3 + i * 2
        cmi2 += rf"\cmidrule(lr){{{st}-{st+1}}}"
    tex.append(cmi2)
    h2b = r" & "
    for _ in strats:
        h2b += r" & HIGH & NORMAL"
    h2b += r" \\"
    tex.append(h2b)
    tex.append(r"\midrule")

    for th in ITRX_THRESHOLDS:
        row = rf"{th} bps"
        # n_HIGH from first strategy
        n_h = "--"
        for s in strats:
            r = all_threshold.get(s, {}).get(th)
            if r and not r.get('skip') and r.get('n_HIGH'):
                n_h = str(r['n_HIGH'])
                break
        row += f" & {n_h}"
        for s in strats:
            r = all_threshold.get(s, {}).get(th)
            if r is None or r.get('skip'):
                row += r" & -- & --"
            else:
                row += (rf" & {r['alpha_HIGH_ann']:+.2f}"
                        rf"{significance_stars(r['alpha1_pval'])}"
                        rf" & {r['alpha0_ann']:+.2f}")
        row += r" \\"
        tex.append(row)

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    # Notes
    tex.append(r"\vspace{0.2cm}")
    tex.append(r"\begin{minipage}{0.92\textwidth}")
    tex.append(rf"\footnotesize\textit{{Notes:}} "
               rf"Factor model: {fw['label']}. "
               r"EUR factors. Monthly frequency. "
               r"Panel A: OLS with HAC (Newey--West) standard errors. "
               r"HIGH = iTraxx Main 5Y above threshold. "
               r"Panel B: $\alpha_{\text{HIGH}} = \alpha_0 + \alpha_1$ and "
               r"$\alpha_{\text{NORMAL}} = \alpha_0$ from "
               r"$r_t = \alpha_0 + \alpha_1 D_{\text{HIGH}} + "
               r"\sum_j \beta_j X_{jt} + \varepsilon_t$. "
               r"Significance stars on HIGH from $t$-test on $\alpha_1$. "
               r"*** $p < 0.01$, ** $p < 0.05$, * $p < 0.10$.")
    tex.append(r"\end{minipage}")

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

    for period in ['Full Sample', 'First Half', 'Second Half', 'HIGH', 'NORMAL']:
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
    tex.append(r"\textbf{Panel B: Threshold robustness ($\alpha_1$, \% p.a.)}"
               r"\par\vspace{0.08cm}")
    tex.append(r"\begin{tabular}{r r " + "r r " * n_s + "}")
    tex.append(r"\toprule")
    bh3 = r"Thr. & $n_H$"
    for s in strats:
        bh3 += rf" & $\alpha_1$ & $t$"
    bh3 += r" \\"
    tex.append(bh3)
    tex.append(r"\midrule")

    for th in ITRX_THRESHOLDS:
        row = rf"{th}"
        n_h = "--"
        for s in strats:
            r = all_threshold.get(s, {}).get(th)
            if r and not r.get('skip') and r.get('n_HIGH'):
                n_h = str(r['n_HIGH'])
                break
        row += f" & {n_h}"
        for s in strats:
            r = all_threshold.get(s, {}).get(th)
            if r is None or r.get('skip'):
                row += " & -- & --"
            else:
                row += (rf" & {r['alpha1_ann']:+.2f}{significance_stars(r['alpha1_pval'])}"
                        rf" & {r['alpha1_tstat']:.2f}")
        row += r" \\"
        tex.append(row)

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    tex.append(r"\vspace{0.1cm}")
    tex.append(rf"{{\tiny {fw['label']}. EUR factors. "
               r"HAC SE. iTraxx Main thresholds. "
               r"*** $p{<}1\%$, ** $p{<}5\%$, * $p{<}10\%$.}}")
    tex.append(r"\end{frame}")
    return "\n".join(tex)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("SUB-PERIOD & REGIME ANALYSIS — ALL FRAMEWORKS")
    print(f"\n   Strategies: {STRATEGIES}")
    print(f"   Frameworks: {list(FRAMEWORKS.keys())}")
    print(f"   Frequency:  {REGRESSION_FREQ}")
    print(f"   Region:     {PRIMARY_REGION}")
    print(f"   Thresholds: {ITRX_THRESHOLDS} bps")

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
                for th, r in th_results.items():
                    if r.get('skip'):
                        print(f"   Threshold {th}: skipped")
                    else:
                        print(f"   Threshold {th}: α₁={r['alpha1_ann']:+.2f}%"
                              f" (t={r['alpha1_tstat']:.2f},"
                              f" p={r['alpha1_pval']:.4f})"
                              f" {significance_stars(r['alpha1_pval'])}")
            # (B2) Ferson-Schadt conditional alpha
            fs_result = analysis_ferson_schadt(
                strategy, fw_key, PRIMARY_REGION, REGRESSION_FREQ, stress_monthly)
            if fs_result:
                fw_subperiod[strategy]['Ferson-Schadt'] = fs_result
                print(f"   Ferson-Schadt: α₀={fs_result['alpha0_ann']:+.2f}%"
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

    print(f"\n   HIGH − NORMAL Δα (% ann.):")
    print(f"   {'Strategy':<20}", end="")
    for fw_key in FRAMEWORKS:
        print(f" {FRAMEWORKS[fw_key]['short']:>10}", end="")
    print()
    print(f"   {'─' * (20 + 11 * len(FRAMEWORKS))}")

    for strategy in STRATEGIES:
        row = f"   {strategy:<20}"
        for fw_key in FRAMEWORKS:
            sp = all_results.get(fw_key, {}).get('subperiod', {}).get(strategy, {})
            rh = sp.get('HIGH')
            rn = sp.get('NORMAL')
            if rh and rn:
                diff = rh['alpha_ann'] - rn['alpha_ann']
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

    # ── Alpha Synthesis Table (article) ───────────────────────────────
    synthesis_path = TABLES_DIR / f"alpha_synthesis_across_models_{REGRESSION_FREQ}.tex"

    fw_labels = {
        'Duarte': 'Duarte et al.\\ (2007)',
        'ActiveFI': 'Brooks et al.\\ (2020)',
        'FungHsieh': 'Fung \\& Hsieh (2004)',
    }

    with open(synthesis_path, 'w', encoding='utf-8') as f:
        f.write("% " + "=" * 74 + "\n")
        f.write("% ALPHA COMPARISON ACROSS BENCHMARK MODELS — ARTICLE TABLE\n")
        f.write("% " + "=" * 74 + "\n\n")

        f.write("\\begin{table}[H]\n")
        f.write("\\caption{Alpha Estimates Across Benchmark Factor Models}\n")
        f.write("\\label{tab:alpha_synthesis}\n")
        f.write("\\begin{center}\n")
        f.write("\\begin{threeparttable}\n")
        f.write("\\begin{singlespace}\n")
        f.write("\\small\n\n")

        n_fw = len(FRAMEWORKS)

        # ── Panel A: Full-Sample Alpha ──
        f.write("\\begin{center}\\textit{Panel A: Full-sample alpha (\\% p.a.)}\\end{center}\n")
        f.write("\\vspace{0.1cm}\n\n")
        f.write("\\begin{tabular}{l" + " r r r" * n_fw + "}\n")
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
        f.write("\\end{tabular}\n\n")

        # ── Panel B: Alpha by Regime ──
        f.write("\\vspace{0.4cm}\n")
        f.write("\\begin{center}\\textit{Panel B: Alpha by stress regime (\\% p.a.)}\\end{center}\n")
        f.write("\\vspace{0.1cm}\n\n")
        f.write("\\begin{tabular}{l" + " r r" * n_fw + "}\n")
        f.write("\\toprule\n")

        # Header row 1: framework names
        f.write(" ")
        for fw_key in FRAMEWORKS:
            lab = fw_labels.get(fw_key, FRAMEWORKS[fw_key]['label'])
            f.write(f" & \\multicolumn{{2}}{{c}}{{{lab}}}")
        f.write(" \\\\\n")

        # Cmidrules
        for i, fw_key in enumerate(FRAMEWORKS):
            st = 2 + i * 2
            f.write(f"\\cmidrule(lr){{{st}-{st + 1}}}")
        f.write("\n")

        # Header row 2
        f.write(" ")
        for _ in FRAMEWORKS:
            f.write(" & HIGH & NORMAL")
        f.write(" \\\\\n")
        f.write("\\midrule\n")

        # Data rows: alpha + t-stat for each regime
        for strategy in STRATEGIES:
            display = STRATEGY_LABELS.get(strategy, strategy)
            # Alpha row
            f.write(f"\\textit{{{display}}}")
            for fw_key in FRAMEWORKS:
                sp = all_results.get(fw_key, {}).get('subperiod', {}).get(strategy, {})
                rh = sp.get('HIGH')
                rn = sp.get('NORMAL')
                if rh and rn:
                    stars_h = significance_stars(rh['alpha_pval'])
                    stars_n = significance_stars(rn['alpha_pval'])
                    if stars_h:
                        f.write(f" & ${rh['alpha_ann']:+.2f}^{{{stars_h}}}$")
                    else:
                        f.write(f" & {rh['alpha_ann']:+.2f}")
                    if stars_n:
                        f.write(f" & ${rn['alpha_ann']:+.2f}^{{{stars_n}}}$")
                    else:
                        f.write(f" & {rn['alpha_ann']:+.2f}")
                else:
                    f.write(" & -- & --")
            f.write(" \\\\\n")

            # t-stat row
            f.write(" ")
            for fw_key in FRAMEWORKS:
                sp = all_results.get(fw_key, {}).get('subperiod', {}).get(strategy, {})
                rh = sp.get('HIGH')
                rn = sp.get('NORMAL')
                if rh and rn:
                    f.write(f" & ({rh['alpha_tstat']:.2f})")
                    f.write(f" & ({rn['alpha_tstat']:.2f})")
                else:
                    f.write(" & & ")
            f.write(" \\\\\n")
            f.write("\\addlinespace\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n\n")

        # ── Panel C: Ferson-Schadt Conditional Alpha ──
        f.write("\\vspace{0.4cm}\n")
        f.write("\\begin{center}\\textit{Panel C: Ferson--Schadt (1996) conditional alpha (\\% p.a.)}\\end{center}\n")
        f.write("\\vspace{0.1cm}\n\n")
        f.write("\\begin{tabular}{l" + " r r" * n_fw + "}\n")
        f.write("\\toprule\n")

        # Header row 1: framework names
        f.write(" ")
        for fw_key in FRAMEWORKS:
            lab = fw_labels.get(fw_key, FRAMEWORKS[fw_key]['label'])
            f.write(f" & \\multicolumn{{2}}{{c}}{{{lab}}}")
        f.write(" \\\\\n")

        for i, fw_key in enumerate(FRAMEWORKS):
            st = 2 + i * 2
            f.write(f"\\cmidrule(lr){{{st}-{st + 1}}}")
        f.write("\n")

        # Header row 2
        f.write(" ")
        for _ in FRAMEWORKS:
            f.write(r" & $\alpha_1$ & $t$-stat")
        f.write(" \\\\\n")
        f.write("\\midrule\n")

        # Data rows
        for strategy in STRATEGIES:
            display = STRATEGY_LABELS.get(strategy, strategy)
            f.write(f"\\textit{{{display}}}")
            for fw_key in FRAMEWORKS:
                fs = all_results.get(fw_key, {}).get('subperiod', {}).get(
                    strategy, {}).get('Ferson-Schadt')
                if fs:
                    stars = significance_stars(fs['alpha1_pval'])
                    if stars:
                        f.write(f" & ${fs['alpha1_ann']:+.2f}^{{{stars}}}$")
                    else:
                        f.write(f" & {fs['alpha1_ann']:+.2f}")
                    f.write(f" & ({fs['alpha1_tstat']:.2f})")
                else:
                    f.write(" & -- & --")
            f.write(" \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n\n")


        # ── Notes ──
        f.write("\\begin{tablenotes}[para,flushleft]\n")
        f.write("\\footnotesize\n")
        f.write("\\item \\textit{Note:} ")
        f.write("Panel A reports annualized alpha (\\% p.a.) from monthly OLS regressions ")
        f.write("with Newey--West HAC standard errors using EUR factors. ")
        f.write("Panel B reports alpha separately for HIGH and NORMAL regimes, ")
        f.write("where HIGH denotes months in which the 5-year iTraxx Europe Main ")
        f.write(f"spread exceeds {DEFAULT_THRESHOLD} bps. ")
        f.write("Panel C reports the stress-loading coefficient $\\alpha_1$ from the ")
        f.write("\\citet{ferson1996measuring} conditional model ")
        f.write("$r_t = \\alpha_0 + \\alpha_1 z_t + \\beta' X_t + \\varepsilon_t$, ")
        f.write("where $z_t$ is the standardized iTraxx Main 5Y spread. ")
        f.write("$\\alpha_1 > 0$: alpha increases during funding stress. ")
        f.write("$t$-statistics in parentheses. ")
        f.write("$^{***}\\,p<0.01$, $^{**}\\,p<0.05$, $^{*}\\,p<0.10$.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{singlespace}\n")
        f.write("\\end{threeparttable}\n")
        f.write("\\end{center}\n")
        f.write("\\end{table}\n")

    print(f"\n   ✅ {synthesis_path.name}  ← PAPER & SKELETON")

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
                for rl, color in [('HIGH', '#d62728'), ('LOW', '#1f77b4')]:
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
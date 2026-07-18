# -*- coding: utf-8 -*-
"""
EW vs SW comparison: Basic Metrics + MPPM + Moreira-Muir
Generates 4 LaTeX tables (Panel A: EW / Panel B: SW):
  1. SW_EW_basic_metrics_3strategies.tex
  2. MPPM_analysis_monthly.tex
  3. moreira_muir_performance_monthly.tex
  4. moreira_muir_robustness_monthly.tex

Author: Alessio Ottaviani
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

def _sig_stars(pvalue):
    """Significance stars: *** p<0.01, ** p<0.05, * p<0.10."""
    if pvalue is None or (isinstance(pvalue, float) and np.isnan(pvalue)):
        return ""
    if pvalue < 0.01:
        return "$^{***}$"
    if pvalue < 0.05:
        return "$^{**}$"
    if pvalue < 0.10:
        return "$^{*}$"
    return ""

# =============================================================================
# CONFIGURATION
# =============================================================================

# MPPM
FREQUENCY = "monthly"
RHO_VALUES = [2, 3, 4]
# Risk-free: 1-month Euribor from the Bloomberg workbook (bbg.xlsx, sheet
# 'rates_fx'), read via factor_sources.load_bloomberg for consistency with the
# rest of the pipeline. Quoted in percent, annualized.
EURIBOR_SHEET = "rates_fx"
EURIBOR_TICKER = "EUR001M Index"

# Moreira-Muir
PERIODS_PER_YEAR = 12
USE_DEMEANING = False
INCLUDE_ROBUSTNESS_TABLE = True

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# --- Bloomberg loader (for the 1-month Euribor risk-free), located in either
#     src/factor_models/ or factor_models/ ---
import sys
_FM_DIR = next(
    (p for p in (PROJECT_ROOT / "src" / "factor_models",
                 PROJECT_ROOT / "factor_models")
     if (p / "factor_sources.py").exists()),
    None,
)
if _FM_DIR is None:
    raise FileNotFoundError("factor_sources.py not found in src/factor_models/ or factor_models/")
sys.path.insert(0, str(_FM_DIR))
from factor_sources import load_bloomberg  # type: ignore

# Strategies — Rebonato versions save to same folders, with both _sw and _ew columns
STRATEGIES = {
    "BTP_Italia": "btp_italia/index_daily.csv",
    "CDS_Bond_Basis": "cds_bond_basis/index_daily.csv",
    "iTraxx_Combined": "itraxx_combined/index_daily.csv",
}

STRATEGY_NAMES_LATEX = {
    "BTP_Italia": "BTP Italia",
    "CDS_Bond_Basis": "CDS--Bond Basis",
    "iTraxx_Combined": "iTraxx Skew",
}

# Column mapping: files contain index_return_sw (signal-weighted) and index_return_ew (equal-weight)
VARIANTS = {
    "EW": "index_return_ew",
    "SW": "index_return_sw",
}

# =============================================================================
# MPPM calculation (Goetzmann et al. 2007)
# =============================================================================

def calculate_mppm(returns_dec, rf_dec, rho, delta_t):
    total_return = returns_dec + rf_dec
    ratio = (1 + total_return) / (1 + rf_dec)
    power_term = ratio ** (1 - rho)
    mean_power = power_term.mean()
    theta_hat = (1 / ((1 - rho) * delta_t)) * np.log(mean_power)
    return theta_hat * 100

# =============================================================================
# Basic performance metrics
# =============================================================================

def _max_drawdown_from_returns(r):
    r = r.dropna()
    if len(r) == 0:
        return np.nan
    wealth = (1.0 + r).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return dd.min()

def compute_basic_metrics(returns, rf, ann_factor=252, rho_theta=3):
    r = returns.dropna()
    if len(r) == 0:
        return {}

    # Performance metrics use the FULL return sample: returns are excess returns
    # by construction, so they do not require alignment to the risk-free series.
    # This keeps them consistent with 04_summary_statistics.py (Table 24), which
    # also uses the full daily sample without rf alignment.
    r_al = r
    avg_ret = r_al.mean() * ann_factor
    vol = r_al.std() * np.sqrt(ann_factor)
    sharpe = avg_ret / vol if vol > 0 else np.nan
    mdd = _max_drawdown_from_returns(r_al)
    skew = r_al.skew()
    kurt = r_al.kurtosis()

    # The risk-free is needed ONLY for the MPPM (Theta): align rf to the return
    # dates and restrict to the overlapping sub-sample for that single metric.
    rf_theta = rf.reindex(r_al.index).dropna()
    r_theta = r_al.reindex(rf_theta.index).dropna()
    rf_theta = rf_theta.reindex(r_theta.index).dropna()
    delta_t = 1.0 / ann_factor
    theta = calculate_mppm(r_theta, rf_theta, rho=rho_theta, delta_t=delta_t)

    n_months = r_al.index.to_period("M").nunique()

    return {
        "Return": avg_ret * 100,
        "Vol": vol * 100,
        "Sharpe": sharpe,
        "MaxDD": mdd * 100,
        "Skew": skew,
        "Kurt": kurt,
        "Theta": theta,
        "N": int(n_months),
    }


def compute_monthly_panelB(daily_returns_dec):
    """Panel-B summary statistics on MONTHLY returns, matching Table 24 Panel B
    in 04_summary_statistics.py (Duarte-style). Input: daily returns in DECIMAL."""
    import statsmodels.api as sm
    d = daily_returns_dec.dropna()
    if len(d) == 0:
        return {}
    monthly = (((1 + d).resample("M").prod() - 1) * 100.0).dropna()   # percent
    n = len(monthly)
    if n == 0:
        return {}
    mean = monthly.mean()
    std = monthly.std()
    if n > 1 and std > 0:
        nw_lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
        _ols = sm.OLS(monthly.values, np.ones(n)).fit(
            cov_type="HAC", cov_kwds={"maxlags": max(nw_lags, 1)}
        )
        t_stat = float(_ols.tvalues[0])
    else:
        t_stat = np.nan
    return {
        "n": n,
        "mean": mean * 12,
        "t_stat": t_stat,
        "std": std * np.sqrt(12),
        "minimum": monthly.min(),
        "maximum": monthly.max(),
        "skewness": monthly.skew(),
        "kurtosis": monthly.kurtosis(),
        "pct_neg": (monthly < 0).sum() / n * 100,
        "ac1": monthly.autocorr(lag=1),
        "sharpe": (mean * 12) / (std * np.sqrt(12)) if std > 0 else np.nan,
    }


# =============================================================================
# Moreira-Muir calculation
# =============================================================================

def moreira_muir_one_variant(daily_returns_dec):
    daily_returns = daily_returns_dec.dropna()

    if USE_DEMEANING:
        rv_list = []
        for period, group in daily_returns.groupby(daily_returns.index.to_period("M")):
            mean_return = group.mean()
            rv = ((group - mean_return) ** 2).sum()
            rv_list.append({"date": period.to_timestamp("M"), "rv": rv})
        rv_df = pd.DataFrame(rv_list).set_index("date")
        realized_variance = rv_df["rv"]
    else:
        realized_variance = (daily_returns ** 2).resample("M").sum()

    monthly_returns = (1 + daily_returns).resample("M").prod() - 1
    monthly_returns = monthly_returns.dropna()

    rv_lagged = realized_variance.shift(1)
    common_idx = monthly_returns.index.intersection(rv_lagged.index)
    returns_clean = monthly_returns.loc[common_idx]
    rv_clean = rv_lagged.loc[common_idx]

    valid_idx = (~returns_clean.isna()) & (~rv_clean.isna()) & (rv_clean > 0)
    returns_clean = returns_clean[valid_idx]
    rv_clean = rv_clean[valid_idx]

    if len(returns_clean) == 0:
        return {"status": "no_valid_data"}

    scaling_raw = 1.0 / rv_clean
    vm_returns_raw = scaling_raw * returns_clean

    if vm_returns_raw.std() <= 0 or np.isnan(vm_returns_raw.std()):
        return {"status": "no_valid_data"}

    c = returns_clean.std() / vm_returns_raw.std()
    vm_returns = c * vm_returns_raw
    effective_scaling = c * scaling_raw

    # Regression: VM on BH (Moreira-Muir Table III)
    # statsmodels OLS with Newey-West HAC to obtain alpha t-stat / p-value,
    # consistent with the inference used elsewhere in the paper and with the
    # original Moreira and Muir (2017) reporting of alpha significance.
    import statsmodels.api as sm
    X = sm.add_constant(returns_clean.values)
    y = vm_returns.values
    nw_lags = int(np.floor(4 * (len(y) / 100.0) ** (2.0 / 9.0)))
    ols = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": max(nw_lags, 1)})

    alpha_monthly = ols.params[0]
    alpha_annual = alpha_monthly * PERIODS_PER_YEAR * 100
    alpha_tstat = ols.tvalues[0]
    alpha_pvalue = ols.pvalues[0]

    bh_sharpe = (returns_clean.mean() / returns_clean.std()) * np.sqrt(PERIODS_PER_YEAR)
    vm_sharpe = (vm_returns.mean() / vm_returns.std()) * np.sqrt(PERIODS_PER_YEAR)

    return {
        "status": "ok",
        "bh_return_pct": (returns_clean.mean() * PERIODS_PER_YEAR) * 100,
        "bh_vol_pct": (returns_clean.std() * np.sqrt(PERIODS_PER_YEAR)) * 100,
        "bh_sharpe": bh_sharpe,
        "vm_return_pct": (vm_returns.mean() * PERIODS_PER_YEAR) * 100,
        "vm_vol_pct": (vm_returns.std() * np.sqrt(PERIODS_PER_YEAR)) * 100,
        "vm_sharpe": vm_sharpe,
        "alpha_annual_pct": alpha_annual,
        "alpha_tstat": alpha_tstat,
        "alpha_pvalue": alpha_pvalue,
        "n_obs": len(returns_clean),
        "effective_scaling": effective_scaling,
        "returns_clean": returns_clean,
    }

# =============================================================================
# LaTeX WRITERS — All use Panel A (EW) / Panel B (SW)
# =============================================================================

def write_basic_metrics_latex(metrics_by_strategy, out_path):
    """
    Table 1: Basic Metrics (beamer version, no \\begin{table}).
    Panel A: EW, Panel B: SW.
    Columns: Strategy | Return | Vol | Sharpe | MaxDD | Skew | Kurt | Theta | N
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\\begin{threeparttable}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lrrrrrrrc}\n")
        f.write("\\toprule\n")
        f.write("Strategy & Return & Vol & Sharpe & MaxDD & Skew & Kurt & $\\Theta_{\\rho=3}$ & N \\\\\n")
        f.write(" & (\\% p.a.) & (\\% p.a.) & & (\\%) & & & (\\% p.a.) & \\\\\n")
        f.write("\\midrule\n")

        for panel_label, variant in [("A", "EW"), ("B", "SW")]:
            f.write(f"\\multicolumn{{9}}{{l}}{{\\textbf{{Panel {panel_label}: {variant}}}}} \\\\\n")
            f.write("\\addlinespace\n")

            for strat_name in STRATEGIES:
                m = metrics_by_strategy.get((strat_name, variant), {})
                if not m:
                    continue
                latex_name = STRATEGY_NAMES_LATEX.get(strat_name, strat_name)
                f.write(
                    f"{latex_name} & "
                    f"{m.get('Return', np.nan):.2f} & "
                    f"{m.get('Vol', np.nan):.2f} & "
                    f"{m.get('Sharpe', np.nan):.2f} & "
                    f"{m.get('MaxDD', np.nan):.2f} & "
                    f"{m.get('Skew', np.nan):.2f} & "
                    f"{m.get('Kurt', np.nan):.2f} & "
                    f"{m.get('Theta', np.nan):.2f} & "
                    f"{int(m.get('N', 0)):d} \\\\\n"
                )

            if panel_label == "A":
                f.write("\\addlinespace\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        

def write_basic_metrics_latex_article(metrics_by_strategy, out_path):
    """
    Equal-weighted counterpart of Table 24, Panel B (monthly returns).
    Same columns as the signal-weighted summary; EW only (SW is in the main text).
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\singlespacing\n")
        f.write("\\caption{Equal-Weighted Strategy Returns}\n")
        f.write("\\label{tab:sw_ew_basic}\n")

        # --- Descriptive legend ABOVE the table (matches Table 24) ---
        f.write("\\begin{minipage}{\\textwidth}\n")
        f.write("{\\footnotesize\\noindent This table reports monthly percentage excess "
                "returns, net of transaction costs, for the equal-weighted indices, in "
                "which each active trade receives weight $1/K_t$; it is the "
                "equal-weighted counterpart of Panel~B of "
                "Table~\\ref{tab:summary_stats_monthly}, whose signal-weighted indices "
                "are the baseline of the paper. The mean and standard deviation "
                "are annualized and the Sharpe ratio is their ratio; kurtosis is excess "
                "kurtosis; the $t$-statistic for the mean is corrected for serial "
                "correlation (Newey--West); and \\%Neg is the proportion of negative "
                "monthly excess returns.}\n")
        f.write("\\end{minipage}\n")
        f.write("\\par\\vspace{6pt}\n")

        f.write("\\begin{singlespace}\n")
        f.write("\\footnotesize\n")
        f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lrrrrrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Strategy & $n$ & Mean & $t$-Stat & Std Dev & Min & Max "
                "& Skew & Kurt & \\%Neg & AC(1) & Sharpe \\\\\n")
        f.write(" & & (\\% p.a.) & & (\\% p.a.) & (\\%) & (\\%) & & & & & \\\\\n")
        f.write("\\midrule\n")

        for strat_name in STRATEGIES:
            m = metrics_by_strategy.get((strat_name, "EW"), {})
            if not m or "mean" not in m:
                continue
            latex_name = STRATEGY_NAMES_LATEX.get(strat_name, strat_name)
            ac1_str = f"{m.get('ac1', np.nan):.2f}"
            if ac1_str == "-0.00":
                ac1_str = "0.00"
            f.write(
                f"{latex_name} & "
                f"{int(m.get('n', 0)):d} & "
                f"{m.get('mean', np.nan):.2f} & "
                f"{m.get('t_stat', np.nan):.2f} & "
                f"{m.get('std', np.nan):.2f} & "
                f"{m.get('minimum', np.nan):.2f} & "
                f"{m.get('maximum', np.nan):.2f} & "
                f"{m.get('skewness', np.nan):.2f} & "
                f"{m.get('kurtosis', np.nan):.2f} & "
                f"{m.get('pct_neg', np.nan):.1f} & "
                f"{ac1_str} & "
                f"{m.get('sharpe', np.nan):.2f} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular*}\n")
        f.write("\\end{singlespace}\n")
        f.write("\\end{table}\n")


def write_mppm_latex(mppm_data, out_path):
    """
    Table 2: MPPM by rho (beamer version, no \\begin{table}).
    Panel A: EW, Panel B: SW.
    Columns: Strategy | rho=2 | rho=3 | rho=4 | Avg Ret | Vol | Sharpe | N
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\\begin{threeparttable}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lrrrrrrc}\n")
        f.write("\\toprule\n")
        f.write("Strategy & $\\rho=2$ & $\\rho=3$ & $\\rho=4$ & Avg Ret & Vol & Sharpe & N \\\\\n")
        f.write(" & (\\% p.a.) & (\\% p.a.) & (\\% p.a.) & (\\% p.a.) & (\\% p.a.) & & \\\\\n")
        f.write("\\midrule\n")

        for panel_label, variant in [("A", "EW"), ("B", "SW")]:
            f.write(f"\\multicolumn{{8}}{{l}}{{\\textbf{{Panel {panel_label}: {variant}}}}} \\\\\n")
            f.write("\\addlinespace\n")

            for strat_name in STRATEGIES:
                d = mppm_data.get((strat_name, variant), {})
                if not d:
                    continue
                latex_name = STRATEGY_NAMES_LATEX.get(strat_name, strat_name)
                f.write(
                    f"{latex_name} & "
                    f"{d.get('rho2', np.nan):.2f} & "
                    f"{d.get('rho3', np.nan):.2f} & "
                    f"{d.get('rho4', np.nan):.2f} & "
                    f"{d.get('avg_ret', np.nan):.2f} & "
                    f"{d.get('vol', np.nan):.2f} & "
                    f"{d.get('sharpe', np.nan):.2f} & "
                    f"{int(d.get('n_obs', 0)):d} \\\\\n"
                )

            if panel_label == "A":
                f.write("\\addlinespace\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}% end resizebox\n")
        f.write("\\begin{tablenotes}[para]\n\\footnotesize\n")
        f.write("\\item \\parbox{\\textwidth}{MPPM is the Manipulation-Proof Performance Measure ")
        f.write("(Goetzmann et al. 2007). Computed from monthly returns compounded from daily. ")
        f.write("Risk-free is Euribor 1M (ACT/360), compounded to monthly.}\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{threeparttable}\n")
def write_mppm_latex_article(mppm_data, out_path, variant="SW",
                             caption="Manipulation-Proof Performance Measure by Risk Aversion",
                             label="tab:mppm"):
    """MPPM by rho (article version), single weighting scheme.
    JF layout: bold title + descriptive legend above the table."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\begin{singlespace}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{minipage}{\\textwidth}\n")
        f.write("{\\footnotesize\\noindent MPPM is the manipulation-proof performance "
                "measure of Goetzmann et al.\\ (2007), computed from monthly returns; "
                "$\\rho$ is the coefficient of relative risk aversion and the risk-free "
                "rate is 1-month Euribor. Returns are net of transaction costs.}\n")
        f.write("\\end{minipage}\n")
        f.write("\\par\\vspace{6pt}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")
        f.write("Strategy & $\\rho=2$ & $\\rho=3$ & $\\rho=4$ \\\\\n")
        f.write(" & (\\% p.a.) & (\\% p.a.) & (\\% p.a.) \\\\\n")
        f.write("\\midrule\n")

        for strat_name in STRATEGIES:
            d = mppm_data.get((strat_name, variant), {})
            if not d:
                continue
            latex_name = STRATEGY_NAMES_LATEX.get(strat_name, strat_name)
            f.write(
                f"{latex_name} & "
                f"{d.get('rho2', np.nan):.2f} & "
                f"{d.get('rho3', np.nan):.2f} & "
                f"{d.get('rho4', np.nan):.2f} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{singlespace}\n")
        f.write("\\end{table}\n")

def write_moreira_muir_performance_latex(mm_data, out_path):
    """
    Table 3: Moreira-Muir Performance (beamer version, no \\begin{table}).
    Panel A: EW, Panel B: SW.
    Columns: Strategy | BH Ret | BH Vol | BH Sharpe | VM Ret | VM Vol | VM Sharpe | Alpha | N
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\\begin{threeparttable}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write("\\begin{tabular}{lrrrrrrrrc}\n")
        f.write("\\toprule\n")
        f.write("Strategy & \\multicolumn{3}{c}{Buy-and-Hold} & \\multicolumn{3}{c}{Vol-Managed} & Alpha & $t(\\alpha)$ & N \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n")
        f.write(" & Ret & Vol & Sharpe & Ret & Vol & Sharpe & & & \\\\\n")
        f.write(" & (\\% p.a.) & (\\% p.a.) & & (\\% p.a.) & (\\% p.a.) & & (\\% p.a.) & & \\\\\n")
        f.write("\\midrule\n")

        for panel_label, variant in [("A", "EW"), ("B", "SW")]:
            f.write(f"\\multicolumn{{10}}{{l}}{{\\textbf{{Panel {panel_label}: {variant}}}}} \\\\\n")
            f.write("\\addlinespace\n")

            for strat_name in STRATEGIES:
                d = mm_data.get((strat_name, variant), {})
                if not d:
                    continue
                latex_name = STRATEGY_NAMES_LATEX.get(strat_name, strat_name)
                f.write(
                    f"{latex_name} & "
                    f"{d.get('bh_return_pct', np.nan):.2f} & "
                    f"{d.get('bh_vol_pct', np.nan):.2f} & "
                    f"{d.get('bh_sharpe', np.nan):.2f} & "
                    f"{d.get('vm_return_pct', np.nan):.2f} & "
                    f"{d.get('vm_vol_pct', np.nan):.2f} & "
                    f"{d.get('vm_sharpe', np.nan):.2f} & "
                    f"{d.get('alpha_annual_pct', np.nan):+.2f}{_sig_stars(d.get('alpha_pvalue'))} & "
                    f"{d.get('alpha_tstat', np.nan):.2f} & "
                    f"{int(d.get('n_obs', 0)):d} \\\\\n"
                )

            if panel_label == "A":
                f.write("\\addlinespace\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}% end resizebox\n")
        f.write("\\begin{tablenotes}[para]\n\\footnotesize\n")
        f.write("\\item \\parbox{\\textwidth}{Following Moreira and Muir (2017). ")
        f.write("The volatility-managed strategy scales positions inversely to realized variance: ")
        f.write("$f^\\sigma_{t+1} = (c/\\sigma^2_t) \\times f_{t+1}$, where $c$ matches buy-and-hold volatility. ")
        f.write("Alpha is from regressing VM returns on BH returns, with Newey--West HAC "
                "standard errors; $t(\\alpha)$ is the corresponding $t$-statistic. "
                "$^{*}$, $^{**}$, $^{***}$ denote significance at the 10\\%, 5\\%, and 1\\% levels.}\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{threeparttable}\n")

def write_moreira_muir_performance_latex_article(mm_data, out_path, variant="SW",
        caption="Moreira--Muir Volatility-Managed Portfolio Performance",
        label="tab:moreira_muir"):
    """Moreira-Muir performance (article version), single weighting scheme.
    JF layout: bold title + descriptive legend above the table."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\begin{singlespace}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{minipage}{\\textwidth}\n")
        f.write("{\\footnotesize\\noindent Following Moreira and Muir (2017). The "
                "volatility-managed strategy scales the position inversely to the "
                "previous month's realized variance, "
                "$f^{\\sigma}_{t+1}=(c/\\sigma^2_t)\\,f_{t+1}$, with $c$ chosen to match "
                "buy-and-hold volatility. Alpha is the intercept from regressing "
                "volatility-managed on buy-and-hold monthly returns, with Newey--West "
                "standard errors (4 lags). Returns are net of transaction costs. "
                "$^{***}\\,p<0.01$, $^{**}\\,p<0.05$, $^{*}\\,p<0.10$.}\n")
        f.write("\\end{minipage}\n")
        f.write("\\par\\vspace{6pt}\n")
        f.write("\\begin{tabular}{lrrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Strategy & \\multicolumn{3}{c}{Buy-and-Hold} & "
                "\\multicolumn{3}{c}{Vol-Managed} & Alpha & $t(\\alpha)$ \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n")
        f.write(" & Ret & Vol & Sharpe & Ret & Vol & Sharpe & & \\\\\n")
        f.write(" & (\\% p.a.) & (\\% p.a.) & & (\\% p.a.) & (\\% p.a.) & & (\\% p.a.) & \\\\\n")
        f.write("\\midrule\n")

        for strat_name in STRATEGIES:
            d = mm_data.get((strat_name, variant), {})
            if not d:
                continue
            latex_name = STRATEGY_NAMES_LATEX.get(strat_name, strat_name)
            f.write(
                f"{latex_name} & "
                f"{d.get('bh_return_pct', np.nan):.2f} & "
                f"{d.get('bh_vol_pct', np.nan):.2f} & "
                f"{d.get('bh_sharpe', np.nan):.2f} & "
                f"{d.get('vm_return_pct', np.nan):.2f} & "
                f"{d.get('vm_vol_pct', np.nan):.2f} & "
                f"{d.get('vm_sharpe', np.nan):.2f} & "
                f"{d.get('alpha_annual_pct', np.nan):.2f}{_sig_stars(d.get('alpha_pvalue'))} & "
                f"{d.get('alpha_tstat', np.nan):.2f} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{singlespace}\n")
        f.write("\\end{table}\n")


def write_moreira_muir_performance_2panel_article(mm_data, out_path,
        caption="Moreira--Muir Volatility-Managed Portfolio Performance",
        label="tab:moreira_muir"):
    """Moreira-Muir performance, single table with Panel A (SW) and Panel B (EW)."""
    def _rows(variant):
        out = []
        for strat_name in STRATEGIES:
            d = mm_data.get((strat_name, variant), {})
            if not d:
                continue
            latex_name = STRATEGY_NAMES_LATEX.get(strat_name, strat_name)
            out.append(
                f"{latex_name} & "
                f"{d.get('bh_return_pct', np.nan):.2f} & "
                f"{d.get('bh_vol_pct', np.nan):.2f} & "
                f"{d.get('bh_sharpe', np.nan):.2f} & "
                f"{d.get('vm_return_pct', np.nan):.2f} & "
                f"{d.get('vm_vol_pct', np.nan):.2f} & "
                f"{d.get('vm_sharpe', np.nan):.2f} & "
                f"{d.get('alpha_annual_pct', np.nan):.2f}{_sig_stars(d.get('alpha_pvalue'))} & "
                f"{d.get('alpha_tstat', np.nan):.2f} \\\\\n"
            )
        return out

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\singlespacing\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{minipage}{\\textwidth}\n")
        f.write("{\\footnotesize\\noindent Following Moreira and Muir (2017). The "
                "volatility-managed strategy scales the position inversely to the "
                "previous month's realized variance, "
                "$f^{\\sigma}_{t+1}=(c/\\sigma^2_t)\\,f_{t+1}$, with $c$ chosen to match "
                "buy-and-hold volatility. Alpha is the intercept from regressing "
                "volatility-managed on buy-and-hold monthly returns, with Newey--West "
                "standard errors (4 lags). Panel~A uses signal-weighted indices; "
                "Panel~B, equal-weighted. Returns are net of transaction costs. "
                "$^{***}\\,p<0.01$, $^{**}\\,p<0.05$, $^{*}\\,p<0.10$.}\n")
        f.write("\\end{minipage}\n")
        f.write("\\par\\vspace{6pt}\n")
        f.write("\\begin{singlespace}\n")
        f.write("\\begin{tabular}{lrrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Strategy & \\multicolumn{3}{c}{Buy-and-Hold} & "
                "\\multicolumn{3}{c}{Vol-Managed} & Alpha & $t(\\alpha)$ \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n")
        f.write(" & Ret & Vol & Sharpe & Ret & Vol & Sharpe & & \\\\\n")
        f.write(" & (\\% p.a.) & (\\% p.a.) & & (\\% p.a.) & (\\% p.a.) & & (\\% p.a.) & \\\\\n")
        f.write("\\midrule\n")
        f.write("\\multicolumn{9}{l}{\\textit{Panel A: Signal-Weighted}} \\\\\n")
        for r in _rows("SW"):
            f.write(r)
        f.write("\\addlinespace\n")
        f.write("\\multicolumn{9}{l}{\\textit{Panel B: Equal-Weighted}} \\\\\n")
        for r in _rows("EW"):
            f.write(r)
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{singlespace}\n")
        f.write("\\end{table}\n")


def write_moreira_muir_robustness_latex(mm_data, robustness_rows, out_path):
    """
    Table 4: Moreira-Muir Robustness (beamer version, no \\begin{table}).
    Panel A: EW, Panel B: SW.
    Columns: Strategy | Metric | No Leverage | 50% Leverage | Table V
    """
    rob_df = pd.DataFrame(robustness_rows) if robustness_rows else pd.DataFrame()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\\begin{threeparttable}\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{llrrr}\n")
        f.write("\\toprule\n")
        f.write("Strategy & Metric & No Leverage & 50\\% Leverage & Table V \\\\\n")
        f.write(" & & $(\\leq 1)$ & $(\\leq 1.5)$ & $[0.1,\\,2.0]$ \\\\\n")
        f.write("\\midrule\n")

        for panel_label, variant in [("A", "EW"), ("B", "SW")]:
            f.write(f"\\multicolumn{{5}}{{l}}{{\\textbf{{Panel {panel_label}: {variant}}}}} \\\\\n")
            f.write("\\addlinespace\n")

            for strat_name in STRATEGIES:
                latex_name = STRATEGY_NAMES_LATEX.get(strat_name, strat_name)

                if rob_df.empty:
                    continue
                sub = rob_df[(rob_df["strategy"] == strat_name) & (rob_df["variant"] == variant)]

                def get_val(constraint, field):
                    ss = sub[sub["constraint"] == constraint]
                    return ss.iloc[0][field] if not ss.empty else np.nan

                # Alpha row
                f.write(f"{latex_name} & Alpha (\\% p.a.) ")
                for c in ["No Leverage", "50% Leverage", "Table V"]:
                    val = get_val(c, "alpha_pct")
                    f.write(f"& {val:+.2f} " if pd.notna(val) else "& -- ")
                f.write("\\\\\n")

                # Sharpe row
                f.write(f" & Sharpe ")
                for c in ["No Leverage", "50% Leverage", "Table V"]:
                    val = get_val(c, "sharpe")
                    f.write(f"& {val:.2f} " if pd.notna(val) else "& -- ")
                f.write("\\\\\n")
                f.write("\\addlinespace\n")

            if panel_label == "A":
                f.write("\\addlinespace\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("\\begin{tablenotes}[para]\n\\scriptsize\n")
        f.write("\\item \\parbox{\\linewidth}{Robustness tests following Table V in Moreira and Muir (2017). ")
        f.write("\\textit{No Leverage}: scaling $\\in [0,\\,1]$. ")
        f.write("\\textit{50\\% Leverage}: scaling $\\in [0,\\,1.5]$. ")
        f.write("\\textit{Table V}: scaling $\\in [0.1,\\,2.0]$ as in the paper.}\n")
        f.write("\\end{tablenotes}\n")

        f.write("\\end{threeparttable}\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    # --- Load the risk-free rate: 1-month Euribor from bbg.xlsx (rates_fx) ---
    # Quoted in percent (annualized). Convert to a daily decimal rate on an
    # ACT/360 basis: r_daily = (Euribor1M / 100) / 360.
    rates = load_bloomberg(EURIBOR_SHEET)
    if EURIBOR_TICKER not in rates.columns:
        raise KeyError(f"'{EURIBOR_TICKER}' not found in sheet '{EURIBOR_SHEET}' of bbg.xlsx")
    rf_daily_dec = (rates[EURIBOR_TICKER].dropna() / 100.0) / 360.0

    # --- Storage ---
    basic_metrics = {}     # (strategy, variant) -> dict
    mppm_data = {}         # (strategy, variant) -> dict
    mm_data = {}           # (strategy, variant) -> dict
    robustness_rows = []   # list of dicts

    Delta_t = 1 / 12

    for strategy_name, strategy_path in STRATEGIES.items():
        full_path = RESULTS_DIR / strategy_path
        if not full_path.exists():
            print(f"⚠️  File non trovato: {full_path}")
            continue

        df = pd.read_csv(full_path, index_col=0, parse_dates=True)

        # Check columns exist
        missing = [col for vname, col in VARIANTS.items() if col not in df.columns]
        if missing:
            print(f"⚠️  {strategy_name}: colonne mancanti {missing}")
            continue

        for variant_name, col_name in VARIANTS.items():
            # Daily returns in decimals
            daily_dec = (df[col_name].copy() / 100.0).dropna()

            # ---- BASIC METRICS (daily, annualized) — for the beamer/plain table ----
            m = compute_basic_metrics(daily_dec, rf_daily_dec, ann_factor=252, rho_theta=3)
            # ---- MONTHLY Panel-B stats (Duarte-style) — for the article (EW) table ----
            m.update(compute_monthly_panelB(daily_dec))
            basic_metrics[(strategy_name, variant_name)] = m

            # ---- MOREIRA-MUIR (monthly) ----
            # Computed first so that MPPM below can be aligned to the identical
            # set of valid months (identical sample, hence identical N).
            mm_valid_index = None
            mm_out = moreira_muir_one_variant(daily_dec)
            if mm_out.get("status") == "ok":
                mm_data[(strategy_name, variant_name)] = mm_out
                mm_valid_index = mm_out["returns_clean"].index

                # Robustness
                if INCLUDE_ROBUSTNESS_TABLE:
                    constraint_tests = [
                        {"name": "No Leverage", "min": 0.0, "max": 1.0},
                        {"name": "50% Leverage", "min": 0.0, "max": 1.5},
                        {"name": "Table V", "min": 0.1, "max": 2.0},
                    ]
                    for test in constraint_tests:
                        scaling_c = mm_out["effective_scaling"].clip(lower=test["min"], upper=test["max"])
                        vm_c = scaling_c * mm_out["returns_clean"]

                        X_c = mm_out["returns_clean"].values.reshape(-1, 1)
                        y_c = vm_c.values
                        reg_c = LinearRegression()
                        reg_c.fit(X_c, y_c)
                        alpha_c = reg_c.intercept_ * PERIODS_PER_YEAR * 100
                        sharpe_c = (vm_c.mean() / vm_c.std()) * np.sqrt(PERIODS_PER_YEAR)

                        robustness_rows.append({
                            "strategy": strategy_name,
                            "variant": variant_name,
                            "constraint": test["name"],
                            "alpha_pct": alpha_c,
                            "sharpe": sharpe_c,
                        })

            # ---- MPPM (monthly, aligned to the Moreira-Muir valid-month sample) ----
            data_daily = pd.DataFrame({
                "ret_dec": daily_dec,
                "rf_dec": rf_daily_dec,
            }).dropna()

            if len(data_daily) == 0:
                continue

            data_monthly = pd.DataFrame({
                "ret_dec": (1 + data_daily["ret_dec"]).resample("M").prod() - 1,
                "rf_dec": (1 + data_daily["rf_dec"]).resample("M").prod() - 1,
            }).dropna()

            # Restrict MPPM to the same valid months used by Moreira-Muir, so the
            # two performance tables are computed on an identical sample.
            if mm_valid_index is not None:
                data_monthly = data_monthly.loc[
                    data_monthly.index.intersection(mm_valid_index)
                ]

            if len(data_monthly) == 0:
                continue

            avg_ret_annual = data_monthly["ret_dec"].mean() * 12 * 100
            vol_annual = data_monthly["ret_dec"].std() * np.sqrt(12) * 100
            sharpe_annual = avg_ret_annual / vol_annual if vol_annual > 0 else 0

            mppm_row = {
                "avg_ret": avg_ret_annual,
                "vol": vol_annual,
                "sharpe": sharpe_annual,
                "n_obs": len(data_monthly),
            }
            for rho in RHO_VALUES:
                theta = calculate_mppm(data_monthly["ret_dec"], data_monthly["rf_dec"], rho, Delta_t)
                mppm_row[f"rho{rho}"] = theta

            mppm_data[(strategy_name, variant_name)] = mppm_row

    # =================================================================
    # WRITE 4 LaTeX files
    # =================================================================

    # 1. Basic Metrics
    out1 = TABLES_DIR / "SW_EW_basic_metrics_3strategies.tex"
    write_basic_metrics_latex(basic_metrics, out1)
    out1_article = TABLES_DIR / "SW_EW_basic_metrics_3strategies_article.tex"
    write_basic_metrics_latex_article(basic_metrics, out1_article)
    print(f"✅ Saved: {out1.name}")

    # 2. MPPM — SW (main text) + EW (appendix)
    out2 = TABLES_DIR / "MPPM_analysis_monthly.tex"
    write_mppm_latex(mppm_data, out2)  # beamer version (both panels), unchanged
    print(f"✅ Saved: {out2.name}")
    out2_article = TABLES_DIR / "MPPM_analysis_monthly_article.tex"
    write_mppm_latex_article(
        mppm_data, out2_article, variant="SW",
        caption="Manipulation-Proof Performance Measure by Risk Aversion",
        label="tab:mppm",
    )
    print(f"✅ Saved: {out2_article.name}")
    out2_ew_article = TABLES_DIR / "MPPM_analysis_monthly_ew_article.tex"
    write_mppm_latex_article(
        mppm_data, out2_ew_article, variant="EW",
        caption="Manipulation-Proof Performance Measure by Risk Aversion (Equal-Weighted)",
        label="tab:mppm_ew",
    )
    print(f"✅ Saved: {out2_ew_article.name}")

    # 3. Moreira-Muir Performance — SW (main text) + EW (appendix)
    out3 = TABLES_DIR / "moreira_muir_performance_monthly.tex"
    write_moreira_muir_performance_latex(mm_data, out3)  # beamer version, unchanged
    print(f"✅ Saved: {out3.name}")
    out3_2panel = TABLES_DIR / "moreira_muir_performance_monthly_2panel_article.tex"
    write_moreira_muir_performance_2panel_article(
        mm_data, out3_2panel,
        caption="Moreira--Muir Volatility-Managed Portfolio Performance",
        label="tab:moreira_muir",
    )
    print(f"✅ Saved: {out3_2panel.name}")

    # 4. Moreira-Muir Robustness
    out4 = TABLES_DIR / "moreira_muir_robustness_monthly.tex"
    write_moreira_muir_robustness_latex(mm_data, robustness_rows, out4)
    print(f"✅ Saved: {out4.name}")

    # --- Console summary ---
    print("\n" + "=" * 80)
    print("SW vs EW — Basic Metrics (Daily, Annualized)")
    print("=" * 80)
    for variant in ["EW", "SW"]:
        print(f"\n  --- {variant} ---")
        for strat in STRATEGIES:
            m = basic_metrics.get((strat, variant), {})
            if m:
                print(f"  {strat:<20s}  Ret={m['Return']:.2f}%  Vol={m['Vol']:.2f}%  "
                      f"Sharpe={m['Sharpe']:.2f}  Theta={m['Theta']:.2f}%")

    print("\n" + "=" * 80)
    print("SW vs EW — MPPM (Monthly)")
    print("=" * 80)
    for variant in ["EW", "SW"]:
        print(f"\n  --- {variant} ---")
        for strat in STRATEGIES:
            d = mppm_data.get((strat, variant), {})
            if d:
                print(f"  {strat:<20s}  rho2={d['rho2']:.2f}  rho3={d['rho3']:.2f}  "
                      f"rho4={d['rho4']:.2f}  Sharpe={d['sharpe']:.2f}")

    print("\n" + "=" * 80)
    print("SW vs EW — Moreira-Muir (Monthly)")
    print("=" * 80)
    for variant in ["EW", "SW"]:
        print(f"\n  --- {variant} ---")
        for strat in STRATEGIES:
            d = mm_data.get((strat, variant), {})
            if d:
                print(f"  {strat:<20s}  BH_Sharpe={d['bh_sharpe']:.2f}  VM_Sharpe={d['vm_sharpe']:.2f}  "
                      f"Alpha={d['alpha_annual_pct']:+.2f}%  "
                      f"t={d.get('alpha_tstat', float('nan')):+.2f}  "
                      f"p={d.get('alpha_pvalue', float('nan')):.3f}")


    print("\n✅ Tutti i file generati in:", TABLES_DIR)


if __name__ == "__main__":
    main()
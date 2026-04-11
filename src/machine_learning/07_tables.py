"""
================================================================================
07_tables.py — Generate LaTeX Tables + Figures for Paper & Slides
================================================================================
Reads JSON/CSV outputs from Steps 02–06 and produces publication-ready
LaTeX tables in TWO formats:
    (a) Beamer slides  — for presentations (frame environment)
    (b) Article/Thesis — for paper submission (table/threeparttable)

Also generates PDF figures for the paper:
    - Rolling alpha with regime shading (from 06 rolling_alpha.csv)

Tables generated:
  1. Post-Selection OLS on Stable Factors (from 04 bootstrap)
     → Main result table for Section 5.2.4
  2. Bootstrap Stability Selection Frequencies (from 04)
     → Section 5.2.3
  3. Method Comparison: AEN vs Adaptive LASSO vs LASSO vs Ridge (from 05)
     → Section 5.4
  4. Selection Matrix: factors × methods (from 05)
     → Section 5.4 or Appendix
  5. Sensitivity: correlation threshold + gamma (from 06)
     → Appendix
  6. Half-Sample Validation (from 06)
     → Appendix

All tables follow the same formatting conventions as 06e_conditional_alpha.py
(thesis tables) and 05b_pca_conditional_alpha.py (PCA article tables):
    - threeparttable + singlespace + tablenotes[para,flushleft]
    - Superscript significance stars (^{***})
    - Newey–West HAC note

References:
    Zou & Zhang (2009, Annals of Statistics)
    Meinshausen & Bühlmann (2010, JRSS-B)
    Chen & Chen (2008, Biometrika)

Author:      Alessio Ottaviani
Institution: EDHEC Business School – PhD Thesis
================================================================================
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================================
# CONFIG
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"

spec = importlib.util.spec_from_file_location("aen_config", CONFIG_PATH)
aen_config = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(aen_config)

HAC_LAGS             = aen_config.HAC_LAGS
FACTORS_PATH         = aen_config.FACTORS_PATH
FACTORS_END_DATE     = aen_config.FACTORS_END_DATE
STRATEGIES           = aen_config.STRATEGIES
AEN_TUNING_CRITERION = aen_config.AEN_TUNING_CRITERION
get_strategy_aen_dir = aen_config.get_strategy_aen_dir
get_aen_output_dir   = aen_config.get_aen_output_dir

RESULTS_DIR = aen_config.RESULTS_DIR
TABLES_DIR  = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures" / "aen"
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TITLE_MAP = {
    "btp_italia":      "BTP Italia",
    "cds_bond_basis":  "CDS--Bond Basis",
    "itraxx_combined": "iTraxx Combined",
}

# Stress proxy for rolling alpha shading
DATA_DIR = PROJECT_ROOT / "data"
FACTORS_EXTERNAL_DIR = DATA_DIR / "external" / "factors"
TRADABLE_CB_FILE     = FACTORS_EXTERNAL_DIR / "Tradable_corporate_bond_factors.xlsx"
ITRX_THRESHOLD_BPS   = 100

FIGURE_DPI    = 150
FIGURE_FORMAT = "pdf"

plt.style.use('seaborn-v0_8-whitegrid')


# ============================================================================
# FACTOR DESCRIPTIONS (for interpretation bullets on slides)
# ============================================================================

FACTOR_INFO: dict[str, tuple[str, str]] = {
    "ILLIQ":            ("Change in the Amihud illiquidity shock measure.",
                         "Amihud and Mendelson (2015)"),
    "SILLIQ":           ("Stock-market illiquidity innovation (AR(3) residual).",
                         "Acharya et al.\\ (2013)"),
    "LIBOR_REPO_SHOCK": ("Dollar funding liquidity shock (LIBOR $-$ GC repo, AR(2) res.).",
                         "Asness et al.\\ (2013)"),
    "TED_SHOCK_EU":     ("Euro interbank funding shock (Euribor $-$ German Bill, AR(2) res.).",
                         "Asness et al.\\ (2013)"),
    "EURIBOR_OIS":      ("Euro interbank credit/liquidity risk (3M Euribor $-$ OIS).",
                         "Nyborg and Ostberg (2014)"),
    "UMD_EU":           ("European equity momentum factor (winners $-$ losers).",
                         "Carhart (1997)"),
    "SMB_EU":           ("European equity size factor (small $-$ big caps).",
                         "Fama and French (2017)"),
    "BAB_US":           ("US betting-against-beta factor (low $-$ high beta).",
                         "Frazzini and Pedersen (2014)"),
    "BAB_EU":           ("European betting-against-beta factor (low $-$ high beta).",
                         "Frazzini and Pedersen (2014)"),
    "BTP_BUND":         ("Italian sovereign risk (10Y BTP $-$ Bund yield spread).",
                         "Fausch and Sigonius (2018)"),
    "TERM_US":          ("US term premium (long-term gov.\\ bond excess return over T-bill).",
                         "Fama and French (1993)"),
    "TERM_EU":          ("Euro term premium (long-term Bund excess return over German bill).",
                         "Fama and French (1993)"),
    "SS10Y":            ("10Y EUR swap spread (par swap rate $-$ 10Y Bund yield).",
                         "Collin-Dufresne et al.\\ (2001)"),
    "SS5Y":             ("5Y EUR swap spread (par swap rate $-$ 5Y Bobl yield).",
                         "Collin-Dufresne et al.\\ (2001)"),
    "SS2Y":             ("2Y EUR swap spread (par swap rate $-$ 2Y Schatz yield).",
                         "Collin-Dufresne et al.\\ (2001)"),
    "R10_EU":           ("Euro 10Y government bond return factor.",
                         "Cochrane and Piazzesi (2005)"),
    "EBP":              ("Excess bond premium: corporate spread net of expected default.",
                         "Gilchrist and Zakrajsek (2012)"),
    "CREDIT_EU":        ("Euro credit spread factor (BBB $-$ AAA bond yield).",
                         "Fama and French (1989)"),
    "CRED_SPR_US":      ("US credit spread factor (Moody's BAA $-$ AAA yield).",
                         "Fama and French (1989)"),
    "DEF_US":           ("US default factor (corporate $-$ gov.\\ long-term bond return).",
                         "Fama and French (1993)"),
    "RI_EU":            ("Euro investment-grade industrial bond return factor.",
                         "Collin-Dufresne et al.\\ (2001)"),
    "PB_EU_CDS_1Y":     ("European prime broker 1Y CDS spread (counterparty risk).",
                         "Klaus and Rzepkowski (2009)"),
    "PB_EU_CDS_5Y":     ("European prime broker 5Y CDS spread (long-term funding risk).",
                         "Klaus and Rzepkowski (2009)"),
    "EP_SVIX_1M":       ("Option-implied equity premium via SVIX, 1-month horizon.",
                         "Martin (2017)"),
    "EP_SVIX_3M":       ("Option-implied equity premium via SVIX, 3-month horizon.",
                         "Martin (2017)"),
    "\u0394UF":         ("Change in financial uncertainty index.",
                         "Ludvigson et al.\\ (2021)"),
    "\u0394UM":         ("Change in macroeconomic uncertainty index.",
                         "Ludvigson et al.\\ (2021)"),
}


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def _stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""

def _stars_sup(p):
    """Superscript version for article tables."""
    s = _stars(p)
    return f"^{{{s}}}" if s else ""

def _fmt2(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.2f}"

def _fmt4(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.4f}"

def _pretty(name: str) -> str:
    """LaTeX-safe factor name for tables."""
    return name.replace("_", r"\_").replace("Δ", r"$\Delta$")

def _pretty_math(name: str) -> str:
    """Factor name in math mode for slides."""
    name = name.replace("\u0394", r"\Delta ")
    if "_" not in name:
        return rf"$\mathrm{{{name}}}$"
    head, tail = name.split("_", 1)
    tail = tail.replace("_", r"\_")
    return rf"$\mathrm{{{head}}}_{{\\mathrm{{{tail}}}}}$"


def print_header(title, char="="):
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")


# ============================================================================
# DATA LOADERS
# ============================================================================

def _load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def _load_itrx_monthly():
    """Load iTraxx Main 5Y monthly for regime shading."""
    raw = pd.read_excel(
        TRADABLE_CB_FILE, sheet_name="CDS_INDEX",
        skiprows=14, usecols=[0, 1], header=0)
    raw.columns = ["Date", "value"]
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date")
    daily = pd.to_numeric(raw["value"], errors="coerce").dropna()
    return daily.resample('ME').last().dropna()


# ############################################################################
#                        ARTICLE / THESIS TABLES
# ############################################################################

# ── TABLE 1: Post-Selection OLS on Stable Factors (cross-strategy) ─────────

def build_article_stable_ols(strategies_data):
    """
    Main results table: α + betas + model fit for each strategy.
    One Panel per strategy. Single tabular, no sparse '--' grid.
    Reads ols_stable_results.json (from 04_bootstrap).
    """
    strats = list(strategies_data.keys())
    if not strats:
        return ""

    panel_labels = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    # Collect all unique factors for the notes
    all_unique_factors = []
    for s in strats:
        all_unique_factors.extend(strategies_data[s].get('stable_factors', []))
    all_unique_factors = sorted(set(all_unique_factors))

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Adaptive Elastic Net: Post-Selection OLS on Stable Factors}")
    tex.append(r"\label{tab:aen_stable_ols}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l r r}")
    tex.append(r"\toprule")

    for idx, s in enumerate(strats):
        d = strategies_data[s]
        title = TITLE_MAP.get(s, s)
        panel = panel_labels.get(idx, chr(65 + idx))
        factors = d.get('stable_factors', [])

        if idx > 0:
            tex.append(r"\addlinespace")
            tex.append(r"\midrule")

        tex.append(rf"\multicolumn{{3}}{{l}}{{\textit{{Panel {panel}: {title}}}}} \\")
        tex.append(r"\addlinespace")

        # Column headers (only on first panel)
        if idx == 0:
            tex.append(r" & Coefficient & $t$-stat \\")
            tex.append(r"\midrule")

        # Alpha row
        a = d['alpha']['annualized_pct']
        a_t = d['alpha']['t_statistic']
        a_p = d['alpha']['p_value']
        tex.append(rf"$\alpha$ (ann.\ \%) & "
                   rf"${a:+.2f}{_stars_sup(a_p)}$ & "
                   rf"{a_t:.2f} \\")
        tex.append(r"\addlinespace")

        # Factor rows
        for f in factors:
            fdata = d.get('factors', {}).get(f, {})
            c = fdata.get('coefficient', 0)
            t = fdata.get('t_statistic', 0)
            p = fdata.get('p_value', 1)
            tex.append(rf"{_pretty(f)} & "
                       rf"${c:+.3f}{_stars_sup(p)}$ & "
                       rf"{t:.2f} \\")

        tex.append(r"\addlinespace")

        # Model fit (compact, one line)
        r2a = _fmt4(d['r_squared_adj'])
        dw = _fmt4(d.get('durbin_watson'))
        tex.append(rf"$T = {d['T']}$, $k = {d['n_factors']}$, "
                   rf"$\bar{{R}}^2 = {r2a}$, DW $= {dw}$ & & \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"Post-selection OLS on factors identified by bootstrap "
               r"stability selection ($\hat{\Pi}_j \geq 0.60$). "
               r"Factors are in original (un-standardized) units. "
               r"$\alpha$ is annualized (\% p.a.). "
               r"Coefficients rounded to 3 decimals. "
               r"$t$-statistics based on Newey--West HAC "
               rf"standard errors ({HAC_LAGS} lags). "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$. "
               r"Factor definitions are provided in Table~\ref{tab:factor_list}.")

    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


# ── TABLE 2: Bootstrap Stability Selection Frequencies ─────────────────────

def build_article_stability_freq(all_boot_data):
    """
    Selection frequencies from bootstrap stability selection.
    One row per factor, one column per strategy.
    Reads bootstrap_stability.json.
    """
    strats = list(all_boot_data.keys())
    n_s = len(strats)
    if n_s == 0:
        return ""

    # Collect all factors with freq >= 10% in at least one strategy
    all_factors = {}
    for s in strats:
        freqs = all_boot_data[s].get('factor_frequencies', {})
        for f, v in freqs.items():
            pls = v.get('pi_lambda_star', 0)
            if pls >= 0.10:
                if f not in all_factors:
                    all_factors[f] = {}
                all_factors[f][s] = pls

    if not all_factors:
        return ""

    # Sort by max frequency across strategies
    sorted_factors = sorted(all_factors.keys(),
                            key=lambda f: max(all_factors[f].values()),
                            reverse=True)

    # Get thresholds
    pi_thr = all_boot_data[strats[0]].get('config', {}).get('pi_thr', 0.60)

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Bootstrap Stability Selection Frequencies}")
    tex.append(r"\label{tab:aen_stability_freq}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l " + "c " * n_s + "}")
    tex.append(r"\toprule")

    hdrs = " & ".join([TITLE_MAP.get(s, s) for s in strats])
    tex.append(rf"Factor & {hdrs} \\")
    tex.append(r"\midrule")

    for f in sorted_factors:
        vals = []
        for s in strats:
            freq = all_factors.get(f, {}).get(s, 0)
            if freq >= pi_thr:
                vals.append(rf"\textbf{{{freq*100:.0f}\%}}")
            elif freq >= 0.10:
                vals.append(f"{freq*100:.0f}\\%")
            else:
                vals.append("--")
        tex.append(rf"{_pretty(f)} & {' & '.join(vals)} \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"Selection frequency $\hat{\Pi}(j, \lambda^*)$ from "
               r"block bootstrap stability selection "
               r"(Meinshausen and B\"uhlmann, 2010). "
               rf"Bold indicates $\hat{{\Pi}} \geq {pi_thr*100:.0f}\%$ (stable). "
               r"Only factors with frequency $\geq 10\%$ in at least "
               r"one strategy are shown.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


# ── TABLE 3: Method Comparison ─────────────────────────────────────────────

def build_article_method_comparison(all_mc_data):
    """
    Cross-strategy comparison: AEN vs Adaptive LASSO (Chen) vs LASSO vs Elastic Net.
    One row per method, columns for each strategy (k, alpha with t-stat, R2_adj).
    Ridge excluded (retains all factors, not a selection method).
    """
    strats = list(all_mc_data.keys())
    if not strats:
        return ""

    # Get method names from first strategy, exclude Ridge
    first = all_mc_data[strats[0]]
    methods_all = list(first.get('methods', {}).keys())
    method_order = [m for m in methods_all
                    if any(kw in m.lower() for kw in
                           ['aen', 'adaptive', 'lasso', 'elastic'])
                    and 'ridge' not in m.lower()]
    if not method_order:
        method_order = [m for m in methods_all if 'ridge' not in m.lower()]

    n_s = len(strats)

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Factor Selection Method Comparison}")
    tex.append(r"\label{tab:aen_method_comparison}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")

    # Column format: Method | (k, alpha, R2) per strategy
    tex.append(r"\begin{tabular}{l" + " c c c" * n_s + "}")
    tex.append(r"\toprule")

    # Header row 1: strategy names
    header1 = "Method"
    for s in strats:
        title = TITLE_MAP.get(s, s)
        header1 += rf" & \multicolumn{{3}}{{c}}{{{title}}}"
    header1 += r" \\"
    tex.append(header1)

    # Cmidrules
    cmi = ""
    for i in range(n_s):
        st = 2 + i * 3
        cmi += rf"\cmidrule(lr){{{st}-{st + 2}}}"
    tex.append(cmi)

    # Header row 2: k, alpha, R2
    header2 = " "
    for _ in strats:
        header2 += r" & $k$ & $\alpha$ & $\bar{R}^2$"
    header2 += r" \\"
    tex.append(header2)
    tex.append(r"\midrule")

    # Data rows: alpha row + t-stat row per method
    for m in method_order:
        # Clean method name
        m_clean = m.replace("_", " ")

        # Alpha row
        row_alpha = m_clean
        for s in strats:
            mc = all_mc_data[s].get('methods', {})
            r = mc.get(m, {})
            if not r:
                row_alpha += r" & -- & -- & --"
                continue
            k = r.get('n_factors', '--')
            a = r.get('alpha_annualized')
            p = r.get('alpha_pval')
            r2 = r.get('r_squared_adj')

            a_str = f"${a:+.2f}{_stars_sup(p)}$" if a is not None else "--"
            r2_str = _fmt4(r2)
            row_alpha += rf" & {k} & {a_str} & {r2_str}"
        row_alpha += r" \\"
        tex.append(row_alpha)

        # t-stat row (in parentheses)
        row_t = " "
        for s in strats:
            mc = all_mc_data[s].get('methods', {})
            r = mc.get(m, {})
            t = r.get('alpha_tstat')
            if t is not None:
                row_t += rf" & & ({t:.2f}) & "
            else:
                row_t += r" & & & "
        row_t += r" \\"
        tex.append(row_t)
        tex.append(r"\addlinespace")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"$\alpha$ is annualized (\% p.a.) from post-selection OLS. "
               r"$k$ denotes the number of selected factors. "
               r"$t$-statistics (Newey--West HAC) in parentheses. "
               r"AEN, LASSO, and Elastic Net use the same tuning "
               rf"criterion ({AEN_TUNING_CRITERION}). "
               r"Adaptive LASSO (Chen) uses AIC per Chen and Chen (2008). "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

    return "\n".join(tex)





# ── TABLE 4c: Selection Matrix NAMES-IN-CELLS (JoF style) ─────────────────

def build_article_selection_matrix_names(all_mc_data):
    """
    Transposed JoF-style table: rows = methods, columns = strategies,
    cells contain factor names.
    """
    strats = list(all_mc_data.keys())
    if not strats:
        return ""

    # Detect methods from first available CSV
    methods_order = None
    for s in strats:
        strategy_dir = get_strategy_aen_dir(s)
        sel_path = strategy_dir / "selection_matrix.csv"
        if sel_path.exists():
            df_tmp = pd.read_csv(sel_path)
            if not df_tmp.empty:
                methods_order = [c for c in df_tmp.columns
                                 if c not in ('factor', 'n_methods')]
                break
    if methods_order is None:
        return ""

    method_labels = {}
    for m in methods_order:
        if m == 'Lasso':
            method_labels[m] = 'LASSO'
        elif m == 'Adaptive Lasso':
            method_labels[m] = 'Ada.~LASSO'
        else:
            method_labels[m] = m.replace('_', ' ')

    # Load selection data per strategy
    strat_data = {}
    for s in strats:
        strategy_dir = get_strategy_aen_dir(s)
        sel_path = strategy_dir / "selection_matrix.csv"
        if sel_path.exists():
            df = pd.read_csv(sel_path)
            if not df.empty:
                strat_data[s] = df

    if not strat_data:
        return ""

    n_s = len(strats)
    strat_hdrs = " & ".join([rf"\textbf{{{TITLE_MAP.get(s, s)}}}"
                             for s in strats])

    tex = []
    tex.append(r"\begin{singlespace}")
    tex.append(r"\scriptsize")
    col_w = f"{0.82 / n_s:.3f}\\textwidth"
    tex.append(r"\begin{longtable}{"
               r"@{}>{\raggedright\arraybackslash}p{0.18\textwidth}"
               + f">{{\\raggedright\\arraybackslash}}p{{{col_w}}}" * n_s
               + r"@{}}")
    tex.append(r"\caption{Selected Factors by Method and Strategy}")
    tex.append(r"\label{tab:aen_selection_names} \\")
    tex.append(r"\toprule")
    tex.append(rf"Method & {strat_hdrs} \\")
    tex.append(r"\midrule")
    tex.append(r"\endfirsthead")
    tex.append(rf"\multicolumn{{{n_s + 1}}}{{l}}{{\small\textit{{(continued)}}}} \\")
    tex.append(r"\toprule")
    tex.append(rf"Method & {strat_hdrs} \\")
    tex.append(r"\midrule")
    tex.append(r"\endhead")
    tex.append(r"\bottomrule")
    tex.append(r"\endlastfoot")

    # One row per method
    for m in methods_order:
        cells = []
        for s in strats:
            if s in strat_data and m in strat_data[s].columns:
                selected = strat_data[s][strat_data[s][m] == True]['factor'].tolist()
                selected_clean = [_pretty(f).replace(r"\_", r"\_\allowbreak{}")
                                  for f in selected]
                cells.append(", ".join(selected_clean) if selected_clean else "--")
            else:
                cells.append("--")
        tex.append(rf"{method_labels[m]} & {' & '.join(cells)} \\")
        tex.append(r"\addlinespace")

    tex.append(r"\end{longtable}")
    tex.append(r"\begin{minipage}{\textwidth}")
    tex.append(r"\footnotesize")
    tex.append(r"\textit{Note:} "
               r"Each cell lists the factors selected (non-zero coefficient) "
               r"by the corresponding method. Ridge excluded (selects all). "
               r"AEN denotes the Adaptive Elastic Net with bootstrap "
               r"stability selection ($\hat{\pi}_j \geq 60\%$).")
    tex.append(r"\end{minipage}")
    tex.append(r"\end{singlespace}")

    return "\n".join(tex)


# ── TABLE 5: Sensitivity Analysis (correlation + gamma) ────────────────────

def build_article_sensitivity(all_rob_data):
    """
    Sensitivity to correlation pre-cleaning threshold and gamma.
    Reads robustness_summary.json.
    """
    strats = list(all_rob_data.keys())
    if not strats:
        return ""

    n_s = len(strats)

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Sensitivity of Factor Selection to Hyperparameters}")
    tex.append(r"\label{tab:aen_sensitivity}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")

    # Panel A: Correlation threshold
    tex.append(r"\textit{Panel A: Correlation pre-cleaning threshold}")
    tex.append(r"\vspace{0.1cm}")
    tex.append(r"\begin{tabular}{l " + "c c " * n_s + "}")
    tex.append(r"\toprule")

    hdr1 = r"$\rho$ threshold"
    for s in strats:
        hdr1 += rf" & \multicolumn{{2}}{{c}}{{{TITLE_MAP.get(s, s)}}}"
    hdr1 += r" \\"
    tex.append(hdr1)

    cmi = ""
    for i in range(n_s):
        start = 2 + i * 2
        cmi += rf"\cmidrule(lr){{{start}-{start+1}}}"
    tex.append(cmi)

    hdr2 = " "
    for _ in strats:
        hdr2 += r" & $k$ & $\alpha$ (ann.)"
    hdr2 += r" \\"
    tex.append(hdr2)
    tex.append(r"\midrule")

    # Get thresholds from first strategy
    corr_data = all_rob_data[strats[0]].get('correlation', {})
    thresholds = sorted(corr_data.keys())

    for th in thresholds:
        row = f"{th}"
        for s in strats:
            r = all_rob_data[s].get('correlation', {}).get(th, {})
            k = r.get('n_selected', '--')
            a = r.get('alpha_annualized')
            p = r.get('alpha_pval')
            if a is not None:
                row += rf" & {k} & ${a:+.2f}{_stars_sup(p)}$\%"
            else:
                row += rf" & {k} & --"
        row += r" \\"
        tex.append(row)

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\vspace{0.4cm}")

    # Panel B: Gamma
    tex.append(r"\textit{Panel B: Adaptive weight exponent $\gamma$}")
    tex.append(r"\vspace{0.1cm}")
    tex.append(r"\begin{tabular}{l " + "c c " * n_s + "}")
    tex.append(r"\toprule")

    hdr1 = r"$\gamma$"
    for s in strats:
        hdr1 += rf" & \multicolumn{{2}}{{c}}{{{TITLE_MAP.get(s, s)}}}"
    hdr1 += r" \\"
    tex.append(hdr1)
    tex.append(cmi)

    hdr2 = " "
    for _ in strats:
        hdr2 += r" & $k$ & $\alpha$ (ann.)"
    hdr2 += r" \\"
    tex.append(hdr2)
    tex.append(r"\midrule")

    gamma_data = all_rob_data[strats[0]].get('gamma', {})
    gammas = sorted(gamma_data.keys())

    for g in gammas:
        row = f"{g}"
        for s in strats:
            r = all_rob_data[s].get('gamma', {}).get(g, {})
            k = r.get('n_selected', '--')
            a = r.get('alpha_annualized')
            p = r.get('alpha_pval')
            if a is not None:
                row += rf" & {k} & ${a:+.2f}{_stars_sup(p)}$\%"
            else:
                row += r" & -- & --"
        row += r" \\"
        tex.append(row)

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"Panel A varies the pairwise correlation threshold for "
               r"pre-cleaning (baseline: $\rho = 0.95$). "
               r"Panel B varies the adaptive weight exponent "
               r"$\gamma$ (baseline: $\gamma = 1$, Zou and Zhang 2009). "
               r"$k$ = number of selected factors. "
               r"$\alpha$ from post-selection OLS (Newey--West HAC). "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


# ── TABLE 6: Half-Sample Validation ────────────────────────────────────────

def build_article_halfsample(all_rob_data):
    """
    Half-sample stability: train on first half, test on second.
    Reads robustness_summary.json → halfsample.
    """
    strats = list(all_rob_data.keys())
    if not strats:
        return ""

    n_s = len(strats)

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Half-Sample Validation}")
    tex.append(r"\label{tab:aen_halfsample}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l " + "c " * n_s + "}")
    tex.append(r"\toprule")

    hdrs = " & ".join([TITLE_MAP.get(s, s) for s in strats])
    tex.append(rf" & {hdrs} \\")
    tex.append(r"\midrule")

    # In-sample alpha
    vals = []
    for s in strats:
        hs = all_rob_data[s].get('halfsample', {})
        is_a = hs.get('is_alpha', {})
        a = is_a.get('annualized')
        p = is_a.get('pval')
        if a is not None:
            vals.append(rf"${a:+.2f}{_stars_sup(p)}$\%")
        else:
            vals.append("--")
    tex.append(rf"$\alpha_{{IS}}$ (ann.\ \%) & {' & '.join(vals)} \\")

    # OOS alpha
    vals = []
    for s in strats:
        hs = all_rob_data[s].get('halfsample', {})
        oos = hs.get('oos_alpha') or {}
        a = oos.get('annualized')
        p = oos.get('pval')
        if a is not None:
            vals.append(rf"${a:+.2f}{_stars_sup(p)}$\%")
        else:
            vals.append("--")
    tex.append(rf"$\alpha_{{OOS}}$ (ann.\ \%) & {' & '.join(vals)} \\")

    # OOS t-stat
    vals = []
    for s in strats:
        hs = all_rob_data[s].get('halfsample', {})
        t = (hs.get('oos_alpha') or {}).get('tstat')
        vals.append(f"({_fmt2(t)})" if t else "--")
    tex.append(rf"\quad $t$-stat & {' & '.join(vals)} \\")

    # k selected (train half)
    vals = []
    for s in strats:
        hs = all_rob_data[s].get('halfsample', {})
        k = hs.get('n_selected', '--')
        vals.append(str(k))
    tex.append(rf"$k$ (train half) & {' & '.join(vals)} \\")

    # Overlap with full sample
    vals = []
    for s in strats:
        hs = all_rob_data[s].get('halfsample', {})
        ov = hs.get('overlap_with_full', [])
        vals.append(str(len(ov)))
    tex.append(rf"Overlap with full sample & {' & '.join(vals)} \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"AEN estimated on the first half of each strategy's sample. "
               r"$\alpha_{IS}$: in-sample alpha (training period). "
               r"$\alpha_{OOS}$: out-of-sample alpha using training-selected "
               r"factors on the second half. "
               r"Newey--West HAC standard errors. "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


# ############################################################################
#                        BEAMER SLIDES (presentation)
# ############################################################################

def build_slide_stable_ols(strategy_name, ols_data):
    """
    Beamer slide: AEN stable-factor OLS results.
    Same layout as the old 07_tables.py.
    """
    stable_factors = ols_data.get('stable_factors', [])
    if not stable_factors:
        return ""

    title = TITLE_MAP.get(strategy_name,
                          strategy_name.replace("_", " ").title())

    # Alpha
    alpha_ann = ols_data['alpha']['annualized_pct']
    alpha_t = ols_data['alpha']['t_statistic']
    alpha_p = ols_data['alpha']['p_value']

    rows = []
    rows.append((r"$\alpha$",
                 f"{alpha_ann:+.2f}{_stars(alpha_p)}",
                 _fmt2(alpha_t), _fmt2(alpha_p)))
    for f in stable_factors:
        fd = ols_data['factors'].get(f, {})
        c = fd.get('coefficient', np.nan)
        t = fd.get('t_statistic', np.nan)
        p = fd.get('p_value', np.nan)
        rows.append((_pretty_math(f),
                     f"{c:+.2f}{_stars(p)}", _fmt2(t), _fmt2(p)))

    # Model fit
    nobs = ols_data['T']
    k = ols_data['n_factors']
    r2 = ols_data['r_squared']
    r2a = ols_data['r_squared_adj']
    dw = ols_data.get('durbin_watson')

    # Bullets
    bullets = []
    for f in stable_factors:
        info = FACTOR_INFO.get(f)
        if info:
            desc, ref = info
            ref_tex = rf"\hfill\textit{{\scriptsize {ref}}}" if ref else ""
            bullets.append(rf"\item \textbf{{{_pretty_math(f)}}}: {desc}{ref_tex}")

    tex = []
    tex.append(rf"\begin{{frame}}[t]{{AEN Stable-Factor Model: {title}}}")
    tex.append(r"\centering\vspace{-0.58cm}\scriptsize")
    tex.append(r"\setlength{\tabcolsep}{3pt}")
    tex.append(r"\renewcommand{\arraystretch}{1.05}")
    tex.append(r"\begin{center}")
    tex.append(r"\begin{columns}[T,onlytextwidth,totalwidth=0.94\textwidth]")

    # Left: coefficient table
    tex.append(r"\column{0.70\textwidth}\centering")
    tex.append(r"\begin{tabular}{l r r r}")
    tex.append(r"\toprule")
    tex.append(r"Variable & Coefficient & $t$-stat & $p$-value \\")
    tex.append(r"\midrule")
    for var, coef, t, p in rows:
        tex.append(rf"{var} & {coef} & {t} & {p} \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    # Right: model fit
    tex.append(r"\column{0.24\textwidth}\centering\scriptsize")
    tex.append(r"\textbf{Model fit}\par\vspace{0.10cm}")
    tex.append(r"\begin{tabular}{@{}lr@{}}")
    tex.append(rf"$T$ (obs) & {nobs} \\")
    tex.append(rf"$k$ (factors) & {k} \\")
    tex.append(rf"$R^2$ & {_fmt4(r2)} \\")
    tex.append(rf"$R^2_{{adj}}$ & {_fmt4(r2a)} \\")
    if dw is not None:
        tex.append(rf"Durbin--Watson & {_fmt4(dw)} \\")
    tex.append(r"\end{tabular}")
    tex.append(r"\par\vspace{0.20cm}")
    tex.append(r"{\tiny $(\alpha)$ annualized \%.\par")
    tex.append(r"*** $p{<}1\%$,\ ** $p{<}5\%$,\par * $p{<}10\%$.}")

    tex.append(r"\end{columns}")
    tex.append(r"\end{center}")

    # Bullets
    if bullets:
        tex.append(r"\vspace{0.10cm}\footnotesize")
        tex.append(r"\begin{columns}[T,onlytextwidth]")
        tex.append(r"\column{\textwidth}")
        tex.append(r"\begin{itemize}")
        tex.append(r"\setlength\itemsep{-0.06em}")
        tex.append(r"\setlength\parskip{0pt}\setlength\topsep{0pt}")
        for b in bullets:
            tex.append(b)
        tex.append(r"\end{itemize}")
        tex.append(r"\end{columns}")

    tex.append(r"\end{frame}")
    return "\n".join(tex)


# ############################################################################
#                              FIGURES
# ############################################################################

def generate_rolling_alpha_figure(all_rob_data):
    """
    Rolling alpha with iTraxx Main regime shading.
    One subplot per strategy. PDF output.
    """
    strats = [s for s in STRATEGIES.keys()
              if s in all_rob_data and 'rolling' in all_rob_data[s]]
    if not strats:
        print("   ⚠️ No rolling alpha data found")
        return

    # Load iTraxx Main for shading
    try:
        itrx = _load_itrx_monthly()
    except Exception:
        itrx = None

    n = len(strats)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.5 * n), squeeze=False)

    for i, s in enumerate(strats):
        ax = axes[i, 0]
        strategy_dir = get_strategy_aen_dir(s)
        csv_path = strategy_dir / "rolling_alpha.csv"
        if not csv_path.exists():
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            continue

        df = pd.read_csv(csv_path, parse_dates=['end_date'])
        df = df.set_index('end_date')

        alpha_ann = df['alpha_monthly'] * 12
        se_ann = df['alpha_se'] * 12

        ax.plot(df.index, alpha_ann, color='black', linewidth=1.2,
                label=r'Rolling $\alpha$ (ann.)')
        ax.fill_between(df.index,
                        alpha_ann - 1.96 * se_ann,
                        alpha_ann + 1.96 * se_ann,
                        color='grey', alpha=0.2, label='95\\% CI')
        ax.axhline(0, color='grey', linewidth=0.5)

        # Shade stress
        if itrx is not None:
            stress = itrx > ITRX_THRESHOLD_BPS
            stress_aligned = stress.reindex(df.index, method='nearest')
            in_stress = False
            start = None
            for date, is_s in stress_aligned.items():
                if is_s and not in_stress:
                    start = date
                    in_stress = True
                elif not is_s and in_stress:
                    ax.axvspan(start, date, alpha=0.12, color='#C73E1D')
                    in_stress = False
            if in_stress:
                ax.axvspan(start, df.index[-1], alpha=0.12, color='#C73E1D')

        title = TITLE_MAP.get(s, s)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(r'$\alpha$ (% p.a.)', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Date', fontsize=10)
    fig.suptitle(f'Rolling Alpha (AEN Stable Factors)\n'
                 f'Shaded: iTraxx Main > {ITRX_THRESHOLD_BPS} bps',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    fig_path = FIGURES_DIR / f"aen_rolling_alpha.{FIGURE_FORMAT}"
    fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"   💾 {fig_path}")

def generate_stability_barplot(all_boot_data):
    """
    Horizontal bar chart of bootstrap selection frequencies.
    One subplot per strategy. Dashed line at pi_thr.
    """
    strats = [s for s in STRATEGIES.keys() if s in all_boot_data]
    if not strats:
        print("   ⚠️ No bootstrap data for barplot")
        return

    n = len(strats)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), squeeze=False)

    for i, s in enumerate(strats):
        ax = axes[0, i]
        freqs = all_boot_data[s].get('factor_frequencies', {})
        pi_thr = all_boot_data[s].get('config', {}).get('pi_thr', 0.60)

        # Filter factors with freq >= 5%
        items = [(f, v['pi_lambda_star'])
                 for f, v in freqs.items()
                 if v.get('pi_lambda_star', 0) >= 0.05]
        items.sort(key=lambda x: x[1])

        if not items:
            ax.text(0.5, 0.5, 'No factors', ha='center', va='center')
            continue

        names, vals = zip(*items)
        colors = ['#2E86AB' if v >= pi_thr else '#AAAAAA' for v in vals]

        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(pi_thr, color='#C73E1D', linestyle='--', linewidth=1.5,
                   label=f'π_thr = {pi_thr:.0%}')
        ax.set_xlabel(r'Selection frequency $\hat{\Pi}$', fontsize=10)
        ax.set_title(TITLE_MAP.get(s, s), fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    fig_path = FIGURES_DIR / f"aen_stability_frequencies.{FIGURE_FORMAT}"
    fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"   💾 {fig_path}")


# ############################################################################
#                              MAIN
# ############################################################################

def main():
    print_header("GENERATING AEN LaTeX TABLES + FIGURES")
    print(f"   Criterion: {AEN_TUNING_CRITERION}")
    print(f"   Output: {TABLES_DIR}")

    # ── Load all data ──────────────────────────────────────────────────
    all_ols_stable = {}
    all_boot = {}
    all_mc = {}
    all_rob = {}

    for s in STRATEGIES.keys():
        sd = get_strategy_aen_dir(s)

        # OLS on stable factors (from 04_bootstrap)
        d = _load_json(sd / "ols_stable_results.json")
        if d:
            all_ols_stable[s] = d

        # Bootstrap stability
        d = _load_json(sd / "bootstrap_stability.json")
        if d:
            all_boot[s] = d

        # Method comparison
        d = _load_json(sd / "method_comparison.json")
        if d:
            all_mc[s] = d

        # Robustness summary
        d = _load_json(sd / "robustness_summary.json")
        if d:
            all_rob[s] = d

    # ── Generate article tables ────────────────────────────────────────
    print_header("ARTICLE / THESIS TABLES", "-")

    article_tables = {
        "AEN_Stable_OLS_article.tex":
            build_article_stable_ols(all_ols_stable),
        "AEN_Stability_Freq_article.tex":
            build_article_stability_freq(all_boot),
        "AEN_Method_Comparison_article.tex":
            build_article_method_comparison(all_mc),
        "AEN_Selection_Names_article.tex":
            build_article_selection_matrix_names(all_mc),
        "AEN_Sensitivity_article.tex":
            build_article_sensitivity(all_rob),
        "AEN_Halfsample_article.tex":
            build_article_halfsample(all_rob),
    }

    for fname, content in article_tables.items():
        if content:
            (TABLES_DIR / fname).write_text(content, encoding="utf-8")
            print(f"   ✅ {fname}")
        else:
            print(f"   ⚠️ {fname} — no data, skipped")

    # ── Generate Beamer slides ─────────────────────────────────────────
    print_header("BEAMER SLIDES", "-")

    aen_tables_dir = get_aen_output_dir() / "tables"
    aen_tables_dir.mkdir(parents=True, exist_ok=True)

    for s in STRATEGIES.keys():
        if s in all_ols_stable:
            safe = "_".join([w.capitalize()
                             for w in re.split(r"[_\-\s]+", s) if w])
            fname = f"AEN_Stable_OLS_{safe}_Presentation_Slide.tex"
            content = build_slide_stable_ols(s, all_ols_stable[s])
            if content:
                (aen_tables_dir / fname).write_text(content, encoding="utf-8")
                print(f"   ✅ {fname}")

    # ── Generate figures ───────────────────────────────────────────────
    print_header("FIGURES", "-")

    if all_rob:
        generate_rolling_alpha_figure(all_rob)
    else:
        print("   ⚠️ No robustness data — rolling alpha figure skipped")

    if all_boot:
        generate_stability_barplot(all_boot)
    else:
        print("   ⚠️ No bootstrap data — stability barplot skipped")

    # ── Summary ────────────────────────────────────────────────────────
    print_header("SUMMARY")
    n_article = sum(1 for c in article_tables.values() if c)
    print(f"   Article tables: {n_article}")
    print(f"   Beamer slides:  {len(all_ols_stable)}")
    print(f"   Figures:        1 (rolling alpha)")

    print(f"\n{'=' * 80}")
    print(f"✅ AEN TABLE GENERATION COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
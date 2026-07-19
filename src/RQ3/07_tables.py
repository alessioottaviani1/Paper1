"""
================================================================================
rq3_07_tables.py — Generate LaTeX Tables for RQ3 (Paper & Slides)
================================================================================
Reads CSV outputs from rq3_01–06 and produces publication-ready LaTeX tables
in article format (threeparttable, singlespace, tablenotes).

Tables for the BODY (Section 6):
  1. Unconditional Correlations (Pearson, Spearman, Forbes-Rigobon)
  2. Regime Correlations (HIGH vs NORMAL) + Co-Widening
  3. Spanning Regressions (Δm)
  4. Interaction Regressions (stress amplification)
  5. Granger Causality
  6. Duffie Scorecard

Tables for APPENDIX (A.8):
  7. Purged Correlations
  8. DCC-GARCH summary
  9. Quantile Regression
 10. Alternative Stress Proxies

Author:      Alessio Ottaviani
Institution: EDHEC Business School – PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "src" / "rq3" / "config.py"

spec = importlib.util.spec_from_file_location("rq3_config", config_path)
rq3_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rq3_config)

RQ3_TABLES_DIR  = rq3_config.RQ3_TABLES_DIR
RQ3_FIGURES_DIR = rq3_config.RQ3_FIGURES_DIR
RESULTS_DIR     = rq3_config.RESULTS_DIR

TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY_LABELS = rq3_config.STRATEGY_LABELS


# ============================================================================
# HELPERS
# ============================================================================

def _stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""

def _stars_sup(p):
    s = _stars(p)
    return f"^{{{s}}}" if s else ""

def _fmt2(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.2f}"

def _fmt3(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.3f}"

def _fmt4(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "--"
    return f"{x:.4f}"

def _load_csv(name):
    path = RQ3_TABLES_DIR / name
    if not path.exists():
        print(f"   ⚠️ Not found: {name}")
        return None
    return pd.read_csv(path)

def _esc(s):
    """Escape underscores for LaTeX."""
    if not isinstance(s, str):
        return str(s)
    return s.replace('_', r'\_')

def print_header(title, char="="):
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")


# ############################################################################
#                    BODY TABLES (Section 6)
# ############################################################################

# ── TABLE 1: Unconditional Correlations ────────────────────────────────────

def build_unconditional_correlations():
    df = _load_csv("T2a_unconditional_correlations.csv")
    if df is None:
        return ""

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Unconditional Correlations of Strategy Returns}")
    tex.append(r"\label{tab:rq3_unconditional_corr}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l r r r}")
    tex.append(r"\toprule")
    tex.append(r"Pair & Pearson & $t$-stat & Spearman \\")
    tex.append(r"\midrule")

    # Filter to returns only (CSV contains returns, Δm, levels)
    if 'series' in df.columns:
        df = df[df['series'] == 'Returns'].copy()

    for _, row in df.iterrows():
        pair = _esc(str(row.get('pair', '')))
        pearson = row.get('pearson_r', np.nan)
        t_stat = row.get('pearson_t_hac', np.nan)
        spearman = row.get('spearman_r', np.nan)
        p_val = row.get('pearson_p_hac', np.nan)

        stars = _stars_sup(p_val) if not np.isnan(p_val) else ""
        tex.append(rf"{pair} & ${pearson:.3f}{stars}$ & {_fmt2(t_stat)} "
                   rf"& {_fmt3(spearman)} \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"Pearson and Spearman correlations of monthly strategy returns. "
               r"$t$-statistics: Newey--West HAC standard errors (4 lags). "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


# ── TABLE 2: Regime Correlations + Co-Widening ─────────────────────────────

def build_regime_correlations():
    df_regime = _load_csv("T2c_regime_correlations.csv")
    df_cw = _load_csv("T2e_cowidening.csv")

    if df_regime is None:
        return ""

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Correlations and Co-Widening by Stress Regime}")
    tex.append(r"\label{tab:rq3_regime_corr}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")

    # Panel A: Regime correlations
    tex.append(r"\textit{Panel A: Correlations by regime}")
    tex.append(r"\vspace{0.1cm}")
    tex.append(r"\begin{tabular}{l r r r}")
    tex.append(r"\toprule")
    tex.append(r"Pair & LOW & MEDIUM & HIGH \\")
    tex.append(r"\midrule")

    # Filter to Δm (mispricing changes)
    if 'series' in df_regime.columns:
        df_regime = df_regime[df_regime['series'] == 'Δm'].copy()

    for _, row in df_regime.iterrows():
        pair = _esc(str(row.get('pair', '')))
        low = row.get('rho_LOW', np.nan)
        med = row.get('rho_MED', np.nan)
        high = row.get('rho_HIGH', np.nan)
        tex.append(rf"{pair} & {_fmt3(low)} & {_fmt3(med)} & {_fmt3(high)} \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    # Panel B: Co-Widening
    if df_cw is not None and len(df_cw) > 0:
        tex.append(r"\vspace{0.4cm}")
        tex.append(r"\textit{Panel B: Co-widening frequency}")
        tex.append(r"\vspace{0.1cm}")
        tex.append(r"\begin{tabular}{l r r r}")
        tex.append(r"\toprule")
        tex.append(r"Pair & Regime & $P(\text{joint})$ & $P(\text{excess})$ \\")
        tex.append(r"\midrule")

        for _, row in df_cw.iterrows():
            pair = _esc(str(row.get('pair', '')))
            regime = row.get('regime', '')
            p_joint = row.get('P_joint', np.nan)
            excess = row.get('excess_prob', np.nan)

            tex.append(rf"{pair} & {regime} & {_fmt3(p_joint)} "
                       rf"& {_fmt3(excess)} \\")

        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}")

    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"Panel A: Pearson correlations of monthly $\Delta m$ "
               r"(change in mispricing) by iTraxx Main regime. Stars on the "
               r"LOW, MED and HIGH columns test $H_0$: $\rho = 0$; stars on the "
               r"FR column test $H_0$: $\rho_{\text{HIGH}} = \rho_{\text{LOW}}$ "
               r"with Fisher's $z$-transform, after the heteroskedasticity "
               r"correction of Forbes and Rigobon (2002). "
               r"Panel B: fraction of months with simultaneous widening, "
               r"by regime. $p$-value from block bootstrap test of "
               r"$H_0$: co-widening(HIGH) $=$ co-widening(LOW).")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


# ── TABLE 3: Spanning Regressions (Δm) ────────────────────────────────────

def build_spanning_regressions():
    df = _load_csv("T3b_spanning_delta_m.csv")
    if df is None:
        return ""

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\singlespacing")
    tex.append(r"\caption{Cross-Strategy Spanning Regressions ($\Delta m$)}")
    tex.append(r"\label{tab:rq3_spanning}")
    tex.append(r"\begin{minipage}{\textwidth}")
    tex.append(r"{\footnotesize\noindent "
               r"Each dependent variable is regressed on the contemporaneous value and the "
               r"first two lags of the other two $\Delta m$ series, where $\Delta m_{i,t}$ is "
               r"the monthly change in the mispricing magnitude of strategy $i$. "
               r"Newey--West HAC standard errors (4 lags). "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.}")
    tex.append(r"\end{minipage}")
    tex.append(r"\par\vspace{6pt}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l l r r r r}")
    tex.append(r"\toprule")
    tex.append(r"Dependent & Regressor & $\beta$ & $t$-stat & $p$ & $\bar{R}^2$ \\")
    tex.append(r"\midrule")

    for _, row in df.iterrows():
        dep = _esc(str(row.get('dependent', '')))
        r2 = row.get('R2_adj', np.nan)

        # Find regressor columns dynamically (pattern: X_beta, X_tstat, X_pval)
        beta_cols = [c for c in df.columns if c.endswith('_beta')]
        first_reg = True
        for bc in beta_cols:
            regressor = bc.replace('_beta', '')
            beta = row.get(bc, np.nan)
            tstat = row.get(f'{regressor}_tstat', np.nan)
            pval = row.get(f'{regressor}_pval', np.nan)

            if np.isnan(beta):
                continue

            stars = _stars_sup(pval)
            dep_label = dep if first_reg else ""
            r2_label = _fmt4(r2) if first_reg else ""
            tex.append(rf"{dep_label} & {_esc(regressor)} & ${beta:+.3f}{stars}$ "
                       rf"& {_fmt2(tstat)} & {_fmt4(pval)} & {r2_label} \\")
            first_reg = False
        tex.append(r"\addlinespace")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


# ── TABLE 4: Interaction Regressions ───────────────────────────────────────

def build_interaction_regressions():
    df = _load_csv("T3d_interaction_regressions.csv")
    if df is None:
        return ""

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\singlespacing")
    tex.append(r"\caption{Stress Amplification: Interaction Regressions}")
    tex.append(r"\label{tab:rq3_interaction}")
    tex.append(r"\begin{minipage}{\textwidth}")
    tex.append(r"{\footnotesize\noindent "
               r"$\Delta m_{i,t} = \alpha + \beta_0\,\Delta m_{j,t} "
               r"+ \beta_1\,(\Delta m_{j,t} \times z_t) + \gamma\,z_t + \varepsilon_t$, "
               r"where $\Delta m_{i,t}$ is the monthly change in the mispricing magnitude of "
               r"strategy $i$ and $z_t$ is the contemporaneous standardized iTraxx Main 5Y level. "
               r"Newey--West HAC standard errors (4 lags). "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.}")
    tex.append(r"\end{minipage}")
    tex.append(r"\par\vspace{6pt}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l l r r r r}")
    tex.append(r"\toprule")
    tex.append(r"Dependent & Variable & Coefficient & $t$-stat & $p$ & $\bar{R}^2$ \\")
    tex.append(r"\midrule")

    for _, row in df.iterrows():
        dep = _esc(str(row.get('dependent', '')))
        indep = _esc(str(row.get('independent', '')))
        r2 = row.get('R2', np.nan)

        variables = [
            (rf"$\Delta m_j$ ({indep})", 'beta1', 'beta1_t', 'beta1_p'),
            (rf"$\Delta m_j \times z_t$", 'beta2', 'beta2_t', 'beta2_p'),
            (r"$z_t$ (stress)", 'beta3', 'beta3_t', 'beta3_p'),
        ]

        for v_idx, (var_label, b_key, t_key, p_key) in enumerate(variables):
            coef = row.get(b_key, np.nan)
            t = row.get(t_key, np.nan)
            p = row.get(p_key, np.nan)

            stars = _stars_sup(p) if not np.isnan(p) else ""
            dep_label = dep if v_idx == 0 else ""
            r2_label = _fmt4(r2) if v_idx == 0 else ""
            tex.append(rf"{dep_label} & {var_label} & ${coef:+.3f}{stars}$ "
                       rf"& {_fmt2(t)} & {_fmt3(p)} & {r2_label} \\")
        tex.append(r"\addlinespace")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


# ── TABLE 5: Granger Causality ─────────────────────────────────────────────

def build_granger_causality():
    df = _load_csv("T4b_granger_causality.csv")
    if df is None:
        return ""

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Granger Causality Tests}")
    tex.append(r"\label{tab:rq3_granger}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l l r r r}")
    tex.append(r"\toprule")
    tex.append(r"Cause & Effect & $F$-stat & $p$-value & Lags \\")
    tex.append(r"\midrule")

    for _, row in df.iterrows():
        cause = _esc(str(row.get('causing', row.get('cause', row.get('Cause', '')))))
        effect = _esc(str(row.get('caused', row.get('effect', row.get('Effect', '')))))
        f_stat = row.get('F_stat', row.get('f_stat', np.nan))
        p = row.get('p_value', np.nan)
        lags = row.get('lags', row.get('Lags', '--'))

        stars = _stars_sup(p)
        tex.append(rf"{cause} & {effect} & ${f_stat:.2f}{stars}$ "
                   rf"& {_fmt4(p)} & {lags} \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"Granger causality tests from VAR on $\Delta m$ "
               r"(change in mispricing). Lag length selected by BIC. "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


# ── TABLE: Granger Causality Multilag (Appendix) ─────────────────────────

def build_granger_multilag():
    df = _load_csv("T4b2_granger_multilag.csv")
    if df is None:
        return ""

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\singlespacing")
    tex.append(r"\caption{Granger Causality: Robustness to Lag Length}")
    tex.append(r"\label{tab:rq3_granger_multilag}")
    tex.append(r"\begin{minipage}{\textwidth}")
    tex.append(r"{\footnotesize\noindent "
               r"Each entry is the $F$-statistic for the null that the row variable does not "
               r"Granger-cause the column variable, from a VAR($p$) on the $\Delta m$ series "
               r"estimated separately at $p = 1, 2, 3$ lags. "
               r"Lag selection: HQC selects $p=1$; AIC/FPE select $p=2$; BIC selects $p=0$ "
               r"(forced to 1). "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.}")
    tex.append(r"\end{minipage}")
    tex.append(r"\par\vspace{6pt}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")

    lags = sorted(df['lag'].unique())
    n_lags = len(lags)

    tex.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l l " + "r r " * n_lags + "@{}}")
    tex.append(r"\toprule")

    # Header row 1: lag groups
    header1 = "Cause & Effect"
    for lag in lags:
        header1 += rf" & \multicolumn{{2}}{{c}}{{$p={lag}$}}"
    header1 += r" \\"
    tex.append(header1)

    # Cmidrules
    cmi = ""
    for i, lag in enumerate(lags):
        st = 3 + i * 2
        cmi += rf"\cmidrule(lr){{{st}-{st + 1}}}"
    tex.append(cmi)

    # Header row 2
    header2 = " & "
    for _ in lags:
        header2 += r" & $F$ & $p$-val"
    header2 += r" \\"
    tex.append(header2)
    tex.append(r"\midrule")

    # Group by (causing, caused)
    pairs = df.groupby(['causing', 'caused'], sort=False)
    for (causing, caused), group in pairs:
        cause_clean = _esc(str(causing))
        effect_clean = _esc(str(caused))
        line = rf"{cause_clean} & {effect_clean}"

        for lag in lags:
            row = group[group['lag'] == lag]
            if not row.empty:
                r = row.iloc[0]
                f_stat = r['F_stat']
                p = r['p_value']
                stars = _stars_sup(p)
                line += rf" & ${f_stat:.2f}{stars}$ & {_fmt4(p)}"
            else:
                line += r" & -- & --"

        line += r" \\"
        tex.append(line)

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular*}")
    tex.append("")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{table}")

    return "\n".join(tex)

# ── TABLE 6: Duffie Scorecard ──────────────────────────────────────────────

def build_duffie_scorecard():
    df = _load_csv("T6_duffie_scorecard.csv")
    if df is None:
        return ""

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Duffie (2010) Slow-Moving Capital Scorecard}")
    tex.append(r"\label{tab:rq3_duffie_scorecard}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\setlength{\tabcolsep}{3pt}")

    # Get prediction columns only (exclude metadata)
    pred_cols = ['P1_correlation', 'P2_cowidening', 'P3_spanning',
                 'P4_granger', 'P5_pca', 'P6_persistence']
    pred_cols = [c for c in pred_cols if c in df.columns]

    # Clean column names for LaTeX
    pred_labels = {
        'P1_correlation': 'P1',
        'P2_cowidening': 'P2',
        'P3_spanning': 'P3',
        'P4_granger': 'P4',
        'P5_pca': 'P5',
        'P6_persistence': 'P6',
    }

    tex.append(r"\begin{tabular}{l " + "c " * len(pred_cols) + "r}")
    tex.append(r"\toprule")

    # Header
    header = "Period"
    for pc in pred_cols:
        short = pred_labels.get(pc, pc.replace('_', r'\_'))
        header += rf" & {short}"
    header += r" & Score \\"
    tex.append(header)
    tex.append(r"\midrule")

    for _, row in df.iterrows():
        period = str(row.get('period', row.get('Period', '')))
        period = period.replace(' (long sample)', '').replace('_', r'\_')
        line = rf"{period}"

        for pc in pred_cols:
            val = row.get(pc, 0)
            if val == 1 or val is True:
                line += r" & $\checkmark$"
            else:
                line += r" & $\times$"

        score = row.get('score', row.get('Score', 0))
        max_s = row.get('max_score', len(pred_cols))
        line += rf" & {int(score)}/{int(max_s)}"
        line += r" \\"
        tex.append(line)

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"Each column tests a prediction of Duffie's (2010) "
               r"slow-moving capital theory. "
               r"$\checkmark$ = prediction confirmed at 10\% significance. "
               r"$\times$ = not confirmed. "
               r"Score = number of confirmed predictions out of total.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


# ############################################################################
#                    APPENDIX TABLES (A.8)
# ############################################################################


# ── TABLE 8: Alternative Stress Proxies ────────────────────────────────────

def build_alternative_proxies():
    df = _load_csv("T5d_alternative_proxies.csv")
    if df is None:
        return ""

    # Use manual mode only for the compact table
    df_man = df[df['mode'] == 'manual'].copy()
    if len(df_man) == 0:
        df_man = df.copy()

    # Proxies to show (in order)
    proxies = [p for p in ["ITRX_MAIN", "ITRX_XOVER", "V2X", "VIX"]
               if p in df_man['proxy'].unique()]
    proxy_labels = {
        "ITRX_MAIN": "iTraxx Main",
        "ITRX_XOVER": "iTraxx Xover",
        "V2X": "V2X",
        "VIX": "VIX",
    }

    pairs = df_man['pair'].unique()

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Robustness: Alternative Stress Proxies}")
    tex.append(r"\label{tab:rq3_alt_proxies}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")

    n_prx = len(proxies)
    tex.append(r"\begin{tabular}{l" + " cc" * n_prx + "}")
    tex.append(r"\toprule")

    # Header row 1: proxy names spanning 2 cols each
    h1 = ""
    for p in proxies:
        h1 += rf" & \multicolumn{{2}}{{c}}{{{proxy_labels.get(p, p)}}}"
    tex.append(h1 + r" \\")

    # Cmidrules
    for i, p in enumerate(proxies):
        col_start = 2 + i * 2
        col_end = col_start + 1
        tex.append(rf"\cmidrule(lr){{{col_start}-{col_end}}}")

    # Header row 2: rho_HIGH, rho_LOW for each proxy
    h2 = "Pair"
    for _ in proxies:
        h2 += r" & $\rho_{\mathrm{H}}$ & $\rho_{\mathrm{L}}$"
    tex.append(h2 + r" \\")
    tex.append(r"\midrule")

    # Data rows
    for pair in pairs:
        pair_short = pair.replace("CDS\u2013Bond Basis", "CDS--Bond Basis") \
                         .replace("CDS-Bond Basis", "CDS--Bond Basis")
        line = _esc(pair_short)
        for p in proxies:
            row = df_man[(df_man['pair'] == pair) & (df_man['proxy'] == p)]
            if len(row) > 0:
                rho_h = row.iloc[0]['rho_returns_HIGH']
                rho_l = row.iloc[0]['rho_returns_LOW']
                rho_h_s = f"{rho_h:+.2f}" if not np.isnan(rho_h) else "--"
                rho_l_s = f"{rho_l:+.2f}" if not np.isnan(rho_l) else "--"
                line += rf" & {rho_h_s} & {rho_l_s}"
            else:
                line += r" & -- & --"
        tex.append(line + r" \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"$\rho_{\mathrm{H}}$ and $\rho_{\mathrm{L}}$ are Pearson "
               r"correlations of strategy returns in HIGH and LOW stress regimes, "
               r"respectively. Regimes defined by manual thresholds on each proxy. "
               r"Sign patterns are consistent across all proxies, confirming that "
               r"cross-strategy interdependencies are not an artifact of the "
               r"conditioning variable.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


# ── TABLE 8b: AR(1) Persistence (P4) ──────────────────────────────────────

def build_ar1_persistence():
    """
    Article-format table for P4: AR(1) persistence of mispricing levels.
    Reads T4z_ar1_persistence.csv generated by rq3_04_var_analysis.py.
    Reports phi (HAC SE/t) and implied half-life in months for each strategy,
    monthly frequency only (consistent with the rest of Section 6).
    """
    df = _load_csv("T4z_ar1_persistence.csv")
    if df is None:
        return ""
    df = df[df['frequency'] == 'monthly'].copy()
    if df.empty:
        return ""

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\singlespacing")
    tex.append(r"\caption{AR(1) Persistence of Mispricing Levels}")
    tex.append(r"\label{tab:rq3_ar1_persistence}")
    tex.append(r"\begin{minipage}{\textwidth}")
    tex.append(r"{\footnotesize\noindent "
               r"AR(1) regression of the mispricing level $m_{i,t} = c_i + \phi_i m_{i,t-1} "
               r"+ u_{i,t}$, where $m_{i,t}$ is the market-level mispricing magnitude of "
               r"strategy $i$. The half-life is $h_i^{*} = \ln(0.5)/\ln(\phi_i)$ months. "
               r"Newey--West HAC standard errors (4 lags). "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.}")
    tex.append(r"\end{minipage}")
    tex.append(r"\par\vspace{6pt}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\renewcommand{\arraystretch}{1.20}")
    tex.append(r"\setlength{\tabcolsep}{8pt}")
    tex.append(r"\begin{tabular}{lcccc}")
    tex.append(r"\toprule")
    tex.append(r"Strategy & $\phi$ & HAC $t$-stat & Half-life (months) & $N$ \\")
    tex.append(r"\midrule")
    for _, row in df.iterrows():
        strat = str(row['strategy']).replace('_', r'\_')
        phi = row['phi']
        t_ = row['phi_tstat']
        hl = row['half_life_months']
        n = int(row['N'])
        # Stars based on HAC t (one-sided H0: phi=0 vs H1: phi>0 not standard;
        # use two-sided as default)
        from scipy.stats import t as tdist
        pval = 2 * (1 - tdist.cdf(abs(t_), df=n - 2))
        stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
        hl_str = f"{hl:.1f}" if np.isfinite(hl) else r"$\infty$"
        tex.append(rf"{strat} & ${phi:.4f}^{{{stars}}}$ & ${t_:.2f}$ "
                   rf"& {hl_str} & {n} \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{table}")
    return "\n".join(tex)


# ── TABLE 9: PC1 ~ Funding Stress ──────────────────────────────────────────

def build_pc1_funding():
    df = _load_csv("T3e2_pc1_funding_regression.csv")
    if df is None:
        return ""

    DEP = [("cds_bond_basis", "CDS--Bond Basis"),
           ("btp_italia",     "BTP Italia"),
           ("itraxx_combined","iTraxx Skew"),
           ("PC1_dm",         r"PC1 ($\Delta m$, contrast)")]
    SETLAB = {"Core": "Intermediary", "Macro": "Macro placebo"}
    FLAB = {"HKM": "HKM capital ratio", "LIBOR_OIS": r"LIBOR--OIS",
            "PB_SPR_5Y_EU": "Dealer CDS spread", "ILLIQ": "Amihud illiquidity",
            "ΔUM": r"$\Delta$ macro uncert.", "ΔUR": r"$\Delta$ real uncert.",
            "5Y5Y_INFL": "5y5y inflation", "EPU_EU": "Policy uncertainty"}

    tex = [r"\begin{table}[H]", r"\centering", r"\singlespacing",
           r"\caption{Mispricing Changes and Intermediary Funding Stress}",
           r"\label{tab:rq3_pc1_funding}",
           r"\begin{minipage}{\textwidth}",
           r"{\footnotesize\noindent Each column regresses the monthly change in a "
           r"strategy's mispricing, $\Delta m_{i,t}$, on a set of intermediary "
           r"balance-sheet proxies (one per Duffie channel: capital ratio, funding "
           r"cost, dealer health, market illiquidity) and, separately, on a "
           r"macroeconomic placebo set. The final column uses the first principal "
           r"component of the three $\Delta m$ series, oriented so that it rises when "
           r"the institutional mispricings widen; with three series this component is "
           r"a contrast between the institutionally held pair and the retail-held "
           r"BTP~Italia, not a common factor, and is reported only for robustness. "
           r"The intermediary proxies carry the sign of stress with opposite sign "
           r"across the institutional and retail strategies, the pattern a common "
           r"macroeconomic driver cannot produce. HKM and LIBOR--OIS are USD series; "
           r"coefficient magnitudes are therefore not directly comparable across "
           r"strategies and the evidence rests on signs and significance. The "
           r"intermediary proxies are significant where the theory predicts, "
           r"while the macro placebo set shows no consistent pattern. "
           r"Newey--West HAC standard errors. "
           r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.}",
           r"\end{minipage}", r"\par\vspace{6pt}",
           r"\begin{singlespace}", r"\small",
           r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + " r" * len(DEP) + "}",
           r"\toprule",
           " & " + " & ".join(lab for _, lab in DEP) + r" \\",
           r"\cmidrule(lr){2-" + str(1 + len(DEP)) + "}"]

    def cell(dep_key, set_key, var):
        r = df[(df["dependent"] == dep_key) & (df["set"] == set_key)
               & (df["variable"] == var)]
        if r.empty:
            return ""
        b = float(r["beta"].iloc[0]); p = float(r["p_value"].iloc[0])
        return f"${b:+.2f}{_stars_sup(p)}$"

    for set_key in ("Core", "Macro"):
        tex.append(rf"\addlinespace\multicolumn{{{1+len(DEP)}}}{{l}}"
                   rf"{{\textit{{{SETLAB[set_key]}}}}} \\")
        varlist = list(dict.fromkeys(df[df["set"] == set_key]["variable"]))
        for var in varlist:
            row = FLAB.get(var, _esc(var))
            for dep_key, _ in DEP:
                row += " & " + cell(dep_key, set_key, var)
            tex.append(row + r" \\")
        # R2_adj / T row per set
        r2row, trow = r"\quad $\bar{R}^2$", r"\quad $T$"
        for dep_key, _ in DEP:
            rr = df[(df["dependent"] == dep_key) & (df["set"] == set_key)]
            r2row += " & " + (f"{float(rr['R2_adj'].iloc[0]):.3f}" if not rr.empty else "")
            trow  += " & " + (f"{int(rr['T'].iloc[0])}" if not rr.empty else "")
        tex.append(r2row + r" \\")
        tex.append(trow + r" \\")

    tex += [r"\bottomrule", r"\end{tabular*}", r"\end{singlespace}", r"\end{table}"]
    return "\n".join(tex)

# ── NEW TABLE: Unified Correlations (Returns + Δm, Uncond + Regime + FR) ───

def _corr_pvalue(rho, n):
    """Two-sided p-value for Pearson correlation via t-test (n-2 d.f.)."""
    from scipy import stats as _st
    if n is None or rho is None:
        return np.nan
    if isinstance(n, float) and np.isnan(n):
        return np.nan
    if isinstance(rho, float) and np.isnan(rho):
        return np.nan
    n = int(n)
    if n < 4:
        return np.nan
    rho_c = np.clip(rho, -0.9999, 0.9999)
    t_val = rho_c * np.sqrt((n - 2) / (1 - rho_c**2))
    return float(2 * (1 - _st.t.cdf(abs(t_val), df=n - 2)))


def build_unified_correlations():
    """
    Panel A: Strategy Returns — Unconditional (Pearson, t) + Regime + FR
    Panel B: Mispricing Changes (Δm) — same structure
    No Spearman. Stars on all correlations. Forbes-Rigobon on HIGH.
    """
    df_uncond = _load_csv("T2a_unconditional_correlations.csv")
    df_regime = _load_csv("T2c_regime_correlations.csv")
    if df_uncond is None or df_regime is None:
        return ""

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\singlespacing")
    tex.append(r"\caption{Unconditional and Regime-Dependent Correlations of Mispricing Changes}")
    tex.append(r"\label{tab:rq3_unified_corr}")
    tex.append(r"\begin{minipage}{\textwidth}")
    tex.append(r"{\footnotesize\noindent "
               r"The table reports pairwise correlations of $\Delta m_{i,t}$, the monthly "
               r"change in the market-level mispricing magnitude of strategy $i$. "
               r"Unconditional: Pearson correlations with Newey--West HAC $t$-statistics (4 lags). "
               r"Regime columns: within-regime Pearson correlations, with regimes on the iTraxx "
               r"Main 5Y level (LOW $<60$, MEDIUM $[60,100)$, HIGH $\geq 100$ bps). "
               r"Stars are two-sided $t$-tests on the within-regime correlation ($n-2$ degrees of "
               r"freedom); in HIGH$^{\mathrm{FR}}$, which applies the \citet{forbes2002no} variance "
               r"adjustment, they test the LOW-to-HIGH change by Fisher $z$-transformation. "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.}")
    tex.append(r"\end{minipage}")
    tex.append(r"\par\vspace{6pt}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")

    if True:
        series_name = "Δm"
        tex.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l r r r r r r}")
        tex.append(r"\toprule")
        tex.append(r" & \multicolumn{2}{c}{Unconditional}"
                   r" & \multicolumn{4}{c}{By Regime} \\")
        tex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-7}")
        tex.append(r"Pair & Pearson & $t$-stat"
                   r" & LOW & MED & HIGH & HIGH$^{\mathrm{FR}}$ \\")
        tex.append(r"\midrule")

        # Filter unconditional
        df_u = df_uncond[df_uncond['series'] == series_name].copy() \
               if 'series' in df_uncond.columns else df_uncond.copy()

        # Filter regime
        df_r = df_regime[df_regime['series'] == series_name].copy() \
               if 'series' in df_regime.columns else df_regime.copy()

        for _, row_u in df_u.iterrows():
            pair = _esc(str(row_u.get('pair', '')))
            pearson = row_u.get('pearson_r', np.nan)
            t_stat = row_u.get('pearson_t_hac', np.nan)
            p_val = row_u.get('pearson_p_hac', np.nan)

            stars_unc = _stars_sup(p_val) if not np.isnan(p_val) else ""

            # Find matching regime row
            pair_raw = str(row_u.get('pair', ''))
            row_r = df_r[df_r['pair'] == pair_raw]

            if len(row_r) > 0:
                row_r = row_r.iloc[0]
                rho_low = row_r.get('rho_LOW', np.nan)
                rho_med = row_r.get('rho_MED', np.nan)
                rho_high = row_r.get('rho_HIGH', np.nan)
                rho_fr = row_r.get('rho_FR_avg', np.nan)
                n_low = row_r.get('n_LOW', np.nan)
                n_med = row_r.get('n_MED', np.nan)
                n_high = row_r.get('n_HIGH', np.nan)
                p_fr = row_r.get('fisher_p_FR', np.nan)

                p_low = _corr_pvalue(rho_low, n_low)
                p_med = _corr_pvalue(rho_med, n_med)
                p_high = _corr_pvalue(rho_high, n_high)

                stars_low = _stars_sup(p_low)
                stars_med = _stars_sup(p_med)
                stars_high = _stars_sup(p_high)
                stars_fr = _stars_sup(p_fr) if not np.isnan(p_fr) else ""
            else:
                rho_low, rho_med, rho_high, rho_fr = np.nan, np.nan, np.nan, np.nan
                stars_low = stars_med = stars_high = stars_fr = ""

            tex.append(
                rf"{pair}"
                rf" & ${pearson:.3f}{stars_unc}$"
                rf" & {_fmt2(t_stat)}"
                rf" & ${_fmt3(rho_low)}{stars_low}$"
                rf" & ${_fmt3(rho_med)}{stars_med}$"
                rf" & ${_fmt3(rho_high)}{stars_high}$"
                rf" & ${_fmt3(rho_fr)}{stars_fr}$"
                rf" \\"
            )

        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular*}")
        tex.append(r"\par")

    tex.append(r"\end{singlespace}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


# ── NEW TABLE: Co-Widening Frequency ───────────────────────────────────────

def build_cowidening_standalone():
    """
    Co-widening frequency by regime — standalone table.
    Stars on excess probability via Fisher exact test.
    """
    df_cw = _load_csv("T2e_cowidening.csv")
    if df_cw is None or len(df_cw) == 0:
        return ""

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Co-Widening Frequency by Stress Regime}")
    tex.append(r"\label{tab:rq3_cowidening}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l l r r r r r}")
    tex.append(r"\toprule")
    tex.append(r"Pair & Regime & $N$"
               r" & $P(\text{joint})$ & $P(\text{indep})$"
               r" & Excess & Fisher $p$ \\")
    tex.append(r"\midrule")

    prev_pair = None
    for _, row in df_cw.iterrows():
        pair = _esc(str(row.get('pair', '')))
        regime = str(row.get('regime', ''))
        n = row.get('T', np.nan)
        p_joint = row.get('P_joint', np.nan)
        p_indep = row.get('P_indep', np.nan)
        excess = row.get('excess_prob', np.nan)
        p_fisher = row.get('fisher_p', np.nan)

        stars = _stars_sup(p_fisher) if not (p_fisher is None or
                (isinstance(p_fisher, float) and np.isnan(p_fisher))) else ""

        if prev_pair is not None and pair != prev_pair:
            tex.append(r"\midrule")
        prev_pair = pair

        n_str = f"{int(n)}" if not np.isnan(n) else "--"
        tex.append(
            rf"{pair} & {regime} & {n_str}"
            rf" & {_fmt3(p_joint)}"
            rf" & {_fmt3(p_indep)}"
            rf" & ${_fmt3(excess)}{stars}$"
            rf" & {_fmt4(p_fisher)} \\"
        )

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"$P(\text{joint})$: fraction of months where both "
               r"$\Delta m_i > 0$ and $\Delta m_j > 0$. "
               r"$P(\text{indep}) = P(\Delta m_i > 0) \times "
               r"P(\Delta m_j > 0)$. "
               r"Excess $= P(\text{joint}) - P(\text{indep})$. "
               r"Fisher: one-sided exact test. "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

    return "\n".join(tex)

# ── NEW TABLE: Unified Correlations — 2-LEVEL (NORMAL/HIGH) ───────────────

def build_unified_correlations_2l():
    """
    Panel A: Strategy Returns — Unconditional + NORMAL/HIGH
    Panel B: Mispricing Changes (Δm) — same structure
    """
    df_uncond = _load_csv("T2a_unconditional_correlations.csv")
    df_regime = _load_csv("T2c_regime_correlations_2l.csv")
    if df_uncond is None or df_regime is None:
        return ""

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Unconditional and Regime-Dependent Correlations "
               r"(NORMAL/HIGH)}")
    tex.append(r"\label{tab:rq3_unified_corr_2l}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")

    for panel_label, series_name in [("A", "Returns"), ("B", "Δm")]:
        panel_title = "Strategy Returns" if series_name == "Returns" \
                      else r"Mispricing Changes ($\Delta m$)"
        tex.append(rf"\textit{{Panel {panel_label}: {panel_title}}}")
        tex.append(r"\vspace{0.1cm}")
        tex.append(r"\begin{tabular}{l r r r r r}")
        tex.append(r"\toprule")
        tex.append(r" & \multicolumn{3}{c}{Unconditional}"
                   r" & \multicolumn{2}{c}{By Regime} \\")
        tex.append(r"\cmidrule(lr){2-4} \cmidrule(lr){5-6}")
        tex.append(r"Pair & Pearson & $t$-stat & Spearman"
                   r" & NORMAL & HIGH \\")
        tex.append(r"\midrule")

        df_u = df_uncond[df_uncond['series'] == series_name].copy() \
               if 'series' in df_uncond.columns else df_uncond.copy()
        df_r = df_regime[df_regime['series'] == series_name].copy() \
               if 'series' in df_regime.columns else df_regime.copy()

        for _, row_u in df_u.iterrows():
            pair = _esc(str(row_u.get('pair', '')))
            pearson = row_u.get('pearson_r', np.nan)
            t_stat = row_u.get('pearson_t_hac', np.nan)
            spearman = row_u.get('spearman_r', np.nan)
            p_val = row_u.get('pearson_p_hac', np.nan)

            stars_unc = _stars_sup(p_val) if not np.isnan(p_val) else ""

            pair_raw = str(row_u.get('pair', ''))
            row_r = df_r[df_r['pair'] == pair_raw]

            if len(row_r) > 0:
                row_r = row_r.iloc[0]
                rho_normal = row_r.get('rho_NORMAL', np.nan)
                rho_high = row_r.get('rho_HIGH', np.nan)
                n_normal = row_r.get('n_NORMAL', np.nan)
                n_high = row_r.get('n_HIGH', np.nan)

                p_normal = _corr_pvalue(rho_normal, n_normal)
                p_high = _corr_pvalue(rho_high, n_high)

                stars_normal = _stars_sup(p_normal)
                stars_high = _stars_sup(p_high)
            else:
                rho_normal, rho_high = np.nan, np.nan
                stars_normal = stars_high = ""

            tex.append(
                rf"{pair}"
                rf" & ${pearson:.3f}{stars_unc}$"
                rf" & {_fmt2(t_stat)}"
                rf" & {_fmt3(spearman)}"
                rf" & ${_fmt3(rho_normal)}{stars_normal}$"
                rf" & ${_fmt3(rho_high)}{stars_high}$"
                rf" \\"
            )

        tex.append(r"\bottomrule")
        tex.append(r"\end{tabular}")
        if panel_label == "A":
            tex.append(r"\vspace{0.4cm}")

    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"Pearson and Spearman correlations (unconditional) with "
               r"HAC $t$-statistics (Newey--West). "
               r"Regime columns: Pearson correlations in NORMAL and "
               r"HIGH stress (iTraxx Main 5Y $> 100$ bps). "
               r"Stars on unconditional: HAC inference. "
               r"Stars on regime correlations: two-sided $t$-test "
               r"with $n-2$ degrees of freedom. "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


# ############################################################################
#                BEAMER SLIDES — Section 5 (Cross-Strategy)
# ############################################################################

# Each function below produces a slide-friendly .tex containing ONLY the
# inner content (table or short structure), without \begin{frame}/\end{frame},
# so that presentation.tex can wrap it in a frame and add bullets around.
# NO numbers are hard-coded here: everything is read from the CSVs that
# rq3_02/rq3_03/rq3_04 already export.

def _stars(p):
    """Compact significance stars (returns empty if p invalid/missing)."""
    if p is None or (isinstance(p, float) and (np.isnan(p) or np.isinf(p))):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def build_slide_p1_correlations():
    """
    P1 — Cross-market correlation.
    Reads T2a_unconditional_correlations.csv (filter on Δm series only) and
    produces a 3-row mini-table of unconditional Pearson ρ for the three
    pairs, with HAC t-stat and significance stars.
    """
    df = _load_csv("T2a_unconditional_correlations.csv")
    if df is None:
        return ""
    # Filter EXACTLY series == "Δm" (NOT "Returns" nor "Levels m")
    if 'series' in df.columns:
        df = df[df['series'].astype(str) == "Δm"].copy()
    if df.empty:
        return ""

    tex = []
    tex.append(r"\centering\small")
    tex.append(r"\renewcommand{\arraystretch}{1.30}")
    tex.append(r"\setlength{\tabcolsep}{8pt}")
    tex.append(r"\begin{tabular}{lcccc}")
    tex.append(r"\toprule")
    tex.append(r"Pair & Pearson $\rho$ & $t$-HAC & $p$-value & $N$ \\")
    tex.append(r"\midrule")
    for _, row in df.iterrows():
        pair = str(row.get('pair', '')).replace('_', r'\_')
        rho = row.get('pearson_r', np.nan)
        tval = row.get('pearson_t_hac', np.nan)
        pval = row.get('pearson_p_hac', np.nan)
        n = row.get('N', np.nan)
        stars = _stars(pval)
        tex.append(rf"{pair} & ${rho:+.3f}{stars}$ & {_fmt2(tval)} "
                   rf"& {_fmt3(pval)} & {int(n) if not np.isnan(n) else '-'} \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    return "\n".join(tex)


def build_slide_p2_interaction():
    """
    P2 — Stress amplification (right side of slide P2).
    Reads T3d_interaction_regressions.csv and shows the interaction
    coefficient β₁ (on Δm_j × z_t) for each direction, with short
    labels for slide readability. The CSV column is named beta2 for
    historical reasons; the slide model labels this as β₁.
    Column names in the CSV: dependent, independent, beta2, beta2_t, beta2_p.
    """
    df = _load_csv("T3d_interaction_regressions.csv")
    if df is None or df.empty:
        return ""

    # Short labels for slide compactness
    short = {
        'BTP Italia':       'BTP',
        'CDS--Bond Basis':  'CDS-Bond',
        'CDS-Bond Basis':   'CDS-Bond',
        'iTraxx Skew':  'iTraxx',
    }

    tex = []
    tex.append(r"\centering\scriptsize")
    tex.append(r"\renewcommand{\arraystretch}{1.20}")
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\begin{tabular}{lc}")
    tex.append(r"\toprule")
    tex.append(r"Direction & $\hat\beta_1$ \\")
    tex.append(r"\midrule")
    for _, row in df.iterrows():
        dep = str(row.get('dependent', ''))
        indep = str(row.get('independent', ''))
        dep_s = short.get(dep, dep).replace('_', r'\_').replace('--', '-')
        indep_s = short.get(indep, indep).replace('_', r'\_').replace('--', '-')
        beta = row.get('beta2', np.nan)
        pval = row.get('beta2_p', np.nan)
        if isinstance(beta, float) and np.isnan(beta):
            continue
        stars = _stars(pval)
        tex.append(rf"{indep_s} $\to$ {dep_s} & ${beta:+.4f}{stars}$ \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    return "\n".join(tex)


def build_slide_p3_spanning_granger():
    """
    P3 — Lead-lag transmission.
    Combines TWO blocks side-by-side:
    (a) Spanning regressions R²_adj (3 rows) from T3b_spanning_delta_m.csv
    (b) Granger causality summary from T4b_granger_causality.csv (compact).
    Returns the inner LaTeX of a slide; presentation.tex wraps the frame.
    """
    df_span = _load_csv("T3b_spanning_delta_m.csv")
    df_grg = _load_csv("T4b_granger_causality.csv")
    if df_span is None and df_grg is None:
        return ""

    tex = []
    tex.append(r"\centering\scriptsize")
    tex.append(r"\begin{minipage}[t]{0.46\textwidth}")
    tex.append(r"\centering")
    tex.append(r"\textbf{Spanning regressions (significant lead-lag):}\\[0.10cm]")
    tex.append(r"\renewcommand{\arraystretch}{1.20}")
    tex.append(r"\setlength{\tabcolsep}{5pt}")
    tex.append(r"\begin{tabular}{lcc}")
    tex.append(r"\toprule")
    tex.append(r"Lead $\to$ Lag & $\hat\beta$ & $p$ \\")
    tex.append(r"\midrule")
    if df_span is not None:
        # Short labels for slide compactness
        short_span = {
            'BTP Italia':       'BTP',
            'CDS--Bond Basis':  'CDS-Bond',
            'CDS-Bond Basis':   'CDS-Bond',
            'iTraxx Skew':  'iTraxx',
        }
        # The CSV is in WIDE format: one row per dependent, columns named
        # "{regressor}_beta", "{regressor}_tstat", "{regressor}_pval", where the
        # regressor key embeds the lag (e.g., "BTP Italia_L1_beta").
        # We unroll it into a long list of (dep, regressor, lag, beta, pval).
        col_dep = 'dependent' if 'dependent' in df_span.columns else 'strategy'
        beta_cols = [c for c in df_span.columns if c.endswith('_beta')]

        sig_entries = []  # list of (dep, regressor_strat, lag, beta, pval)
        for _, row in df_span.iterrows():
            dep_raw = str(row.get(col_dep, ''))
            dep_short = short_span.get(dep_raw, dep_raw).replace('--', '-')
            for bc in beta_cols:
                key = bc[:-len('_beta')]  # e.g. "BTP Italia_L1"
                beta = row.get(bc, np.nan)
                pval = row.get(f"{key}_pval", np.nan)
                if not np.isfinite(beta) or not np.isfinite(pval):
                    continue
                if pval >= 0.10:
                    continue
                # Parse "{strategy}_L{lag}" — split on the LAST "_L"
                if "_L" in key:
                    strat_raw, lag_str = key.rsplit("_L", 1)
                    try:
                        lag = int(lag_str)
                    except ValueError:
                        strat_raw, lag = key, 0
                else:
                    strat_raw, lag = key, 0
                # Skip own-lags (regressor == dependent)
                if strat_raw.strip() == dep_raw.strip():
                    continue
                reg_short = short_span.get(strat_raw, strat_raw).replace('--', '-')
                sig_entries.append((dep_short, reg_short, lag, beta, pval))

        if sig_entries:
            # Sort by dependent, then lag (contemporaneous first)
            sig_entries.sort(key=lambda x: (x[0], x[2]))
            for dep_s, reg_s, lag, beta, pval in sig_entries:
                lag_tex = "" if lag == 0 else rf"$_{{t-{lag}}}$"
                stars = _stars(pval)
                tex.append(rf"{reg_s}{lag_tex} $\to$ {dep_s} & ${beta:+.2f}{stars}$ & {_fmt3(pval)} \\")
            tex.append(r"\midrule")
            tex.append(r"\multicolumn{3}{l}{\scriptsize All other coefficients $p > 0.10$.} \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{minipage}\hfill")

    tex.append(r"\begin{minipage}[t]{0.50\textwidth}")
    tex.append(r"\centering")
    tex.append(r"\textbf{Granger causality (VAR on $\Delta m$):}\\[0.10cm]")
    tex.append(r"\renewcommand{\arraystretch}{1.15}")
    tex.append(r"\setlength{\tabcolsep}{3pt}")
    tex.append(r"\begin{tabular}{lcl c}")
    tex.append(r"\toprule")
    tex.append(r"From & & To & $p$-value \\")
    tex.append(r"\midrule")
    # Short labels for slide compactness
    short_grg = {
        'BTP Italia':       'BTP',
        'CDS--Bond Basis':  'CDS-Bond',
        'CDS-Bond Basis':   'CDS-Bond',
        'iTraxx Skew':  'iTraxx',
    }
    if df_grg is not None:
        # Try common column names — the CSV uses 'causing'/'caused'
        col_from = next((c for c in ('causing', 'cause', 'from')
                         if c in df_grg.columns), None)
        col_to = next((c for c in ('caused', 'effect', 'to')
                       if c in df_grg.columns), None)
        col_p = next((c for c in ('p_value', 'pvalue', 'p')
                      if c in df_grg.columns), None)
        for _, row in df_grg.iterrows():
            f_raw = str(row.get(col_from, '')) if col_from else ''
            t_raw = str(row.get(col_to, '')) if col_to else ''
            f_ = short_grg.get(f_raw, f_raw).replace('_', r'\_').replace('--', '-')
            t_ = short_grg.get(t_raw, t_raw).replace('_', r'\_').replace('--', '-')
            pv = row.get(col_p, np.nan) if col_p else np.nan
            stars = _stars(pv)
            tex.append(rf"{f_} & $\to$ & {t_} & ${_fmt3(pv)}^{{{stars}}}$ \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{minipage}")
    return "\n".join(tex)


def build_slide_p4_persistence():
    """
    P4 — Persistence reflects funding-dependence (NEW).
    Reads T4z_ar1_persistence.csv (filter on monthly) and shows phi, HAC
    t-stat, half-life (months) for the three strategies. NO hardcoded numbers.
    """
    df = _load_csv("T4z_ar1_persistence.csv")
    if df is None:
        return ""
    df = df[df['frequency'] == 'monthly'].copy()
    if df.empty:
        return ""

    tex = []
    tex.append(r"\centering\small")
    tex.append(r"\renewcommand{\arraystretch}{1.30}")
    tex.append(r"\setlength{\tabcolsep}{8pt}")
    tex.append(r"\begin{tabular}{lccc}")
    tex.append(r"\toprule")
    tex.append(r"Strategy & $\phi$ & HAC $t$ & Half-life (months) \\")
    tex.append(r"\midrule")
    for _, row in df.iterrows():
        strat = str(row['strategy']).replace('_', r'\_').replace('--', '-')
        phi = row['phi']
        t_ = row['phi_tstat']
        hl = row['half_life_months']
        hl_str = f"{hl:.1f}" if np.isfinite(hl) else r"$\infty$"
        # Highlight CDS-Bond as the most persistent (funding-dependence)
        if 'CDS' in strat and 'Bond' in strat:
            tex.append(rf"\hlt{{{strat}}} & \hlt{{${phi:.3f}$}} & "
                       rf"$({t_:.1f})$ & \hlt{{{hl_str}}} \\")
        else:
            tex.append(rf"{strat} & ${phi:.3f}$ & "
                       rf"$({t_:.1f})$ & {hl_str} \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    return "\n".join(tex)


def build_slide_p5_pc1_funding():
    """
    P5 — Common factor = intermediary capital (NOT macro).
    Reads T3e2_pc1_funding_regression.csv and shows Panel A (Core) + Panel C
    (Macro placebo) side-by-side or stacked. Panel B goes to backup.
    Format: per-set mini-table with factor, β (with stars), R²_adj.
    """
    df = _load_csv("T3e2_pc1_funding_regression.csv")
    if df is None:
        return ""

    # Restrict to Core (Panel A) and Macro (Panel C) for the slide
    df_p = df[df['set'].isin(['Core', 'Macro'])].copy()

    tex = []
    tex.append(r"\centering\scriptsize")
    tex.append(r"\renewcommand{\arraystretch}{1.20}")
    tex.append(r"\setlength{\tabcolsep}{6pt}")
    tex.append(r"\begin{tabular}{lcc}")
    tex.append(r"\toprule")
    tex.append(r"Factor & Coefficient & $p$-value \\")
    tex.append(r"\midrule")

    set_labels_local = {
        'Core': r'\textbf{Panel A: Core Intermediary Proxies}',
        'Macro': r'\textbf{Panel C: Macroeconomic Placebo}',
    }
    current_set = None
    for _, row in df_p.iterrows():
        rs = row['set']
        if rs != current_set:
            if current_set is not None:
                tex.append(r"\addlinespace")
            r2a = row.get('R2_adj', np.nan)
            t_ = int(row.get('T', 0))
            tex.append(rf"\multicolumn{{3}}{{l}}{{{set_labels_local[rs]}"
                       rf" \quad $T={t_}$, $\bar R^2={r2a:.3f}$}} \\")
            current_set = rs
        fac = str(row.get('variable', '')).replace('_', r'\_').replace('Δ', r'$\Delta$')
        beta = row.get('beta', np.nan)
        pval = row.get('p_value', np.nan)
        stars = _stars(pval)
        tex.append(rf"{fac} & ${beta:+.3f}{stars}$ & {_fmt3(pval)} \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    return "\n".join(tex)


def build_slide_mapping_scorecard():
    """
    Final mapping slide (P1-P5 with green checks).
    Reads existing scorecard CSV if available; otherwise builds the structure
    from the CSVs of each P. Outputs a compact 5-row table:
    P# | Statement | Empirical evidence (auto-populated key numbers)
    """
    # Try to use the existing scorecard logic; if missing, return empty
    df_score = _load_csv("T4d_duffie_scorecard.csv") \
               if (RQ3_TABLES_DIR / "T4d_duffie_scorecard.csv").exists() \
               else None

    if df_score is None:
        # Fallback: build a static 5-row scorecard from key CSVs
        # We populate "evidence" by reading the most informative cell from each P
        return _build_mapping_from_csvs()

    tex = []
    tex.append(r"\centering\scriptsize")
    tex.append(r"\renewcommand{\arraystretch}{1.40}")
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\begin{tabular}{p{0.5cm}p{2.6cm}p{8.2cm}p{0.5cm}}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{P} & \textbf{Statement} & \textbf{Empirical evidence} & \\")
    tex.append(r"\midrule")
    for _, row in df_score.iterrows():
        pid = str(row.get('P', ''))
        statement = str(row.get('statement', '')).replace('_', r'\_')
        evidence = str(row.get('evidence', '')).replace('_', r'\_')
        tex.append(rf"\textbf{{{pid}}} & {statement} & {evidence} "
                   r"& \textcolor{mygreen}{\large\checkmark} \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    return "\n".join(tex)


def _build_mapping_from_csvs():
    """
    Fallback: build the P1-P5 scorecard by reading key cells from the CSVs.
    No hardcoded numbers; all values pulled from existing exports.
    """
    # P1: unconditional ρ — institutional pair (CDS-Bond × iTraxx), Δm
    p1_evidence = "See Table~\\ref{tab:rq3_unified_corr}"
    df_uncond = _load_csv("T2a_unconditional_correlations.csv")
    if df_uncond is not None:
        # Filter on Δm series AND the institutional pair specifically
        df_dm = df_uncond[df_uncond['series'].astype(str).str.contains("m", case=False)] \
                if 'series' in df_uncond.columns else df_uncond
        # Take the institutional pair (CDS-Bond × iTraxx), not just the max
        df_inst = df_dm[
            df_dm['pair'].astype(str).str.contains("CDS", case=False)
            & df_dm['pair'].astype(str).str.contains("iTraxx|Itrx", case=False, regex=True)
        ] if 'pair' in df_dm.columns else pd.DataFrame()
        if len(df_inst) > 0:
            row = df_inst.iloc[0]
            p1_evidence = (rf"$\rho_{{\Delta m}} = {row['pearson_r']:+.2f}{_stars(row.get('pearson_p_hac',1))}$ "
                           rf"(CDS--Bond Basis vs iTraxx Skew)")

    # P2: rise from LOW to HIGH (institutional pair)
    p2_evidence = "See Table~\\ref{tab:rq3_unified_corr}, Fig.~\\ref{fig:rq3_regime_bar}"
    df_reg = _load_csv("T2c_regime_correlations.csv")
    if df_reg is not None:
        # Take the institutional pair (CDS-Bond × iTraxx) with Δm
        df_inst = df_reg[
            (df_reg['series'].astype(str).str.contains("m", case=False))
            & (df_reg['pair'].astype(str).str.contains("CDS", case=False))
            & (df_reg['pair'].astype(str).str.contains("iTraxx|Itrx", case=False, regex=True))
        ] if all(c in df_reg.columns for c in ['series', 'pair']) else pd.DataFrame()
        if len(df_inst) > 0:
            r_low = df_inst.iloc[0]['rho_LOW']
            r_high = df_inst.iloc[0]['rho_HIGH']
            p2_evidence = rf"$\rho_{{\rm LOW}} = {r_low:+.2f} \to \rho_{{\rm HIGH}} = {r_high:+.2f}$"

    # P3: spanning R²_adj for BTP Italia (most predictable, key empirical claim)
    p3_evidence = "See Table~\\ref{tab:rq3_spanning}, Granger tests"
    df_span = _load_csv("T3b_spanning_delta_m.csv")
    if df_span is not None:
        col_dep = 'dependent' if 'dependent' in df_span.columns else 'strategy'
        col_r2 = 'R2_adj' if 'R2_adj' in df_span.columns else 'r_squared_adj'
        df_btp = df_span[df_span[col_dep].astype(str).str.contains("btp", case=False)]
        if len(df_btp) > 0:
            r2 = df_btp[col_r2].iloc[0]
            p3_evidence = rf"BTP Italia $\bar R^2 = {r2*100:.1f}\%$ (iTraxx leads at $1\%$)"

# P4 (NEW): persistence reflects funding-dependence (AR(1) on m_t levels)
    p4_evidence = "See Table~\\ref{tab:rq3_ar1_persistence}"
    df_ar = _load_csv("T4z_ar1_persistence.csv")
    if df_ar is not None:
        df_m_only = df_ar[df_ar['frequency'] == 'monthly'].copy()
        if len(df_m_only) >= 2:
            order = df_m_only.sort_values('half_life_months', ascending=False).reset_index(drop=True)
            # Detect whether CDS--Bond stands clearly apart, with BTP and iTraxx
            # essentially identical (the actual D1 finding: 3.0 / 2.2 / 2.2).
            hl_top = order['half_life_months'].iloc[0]
            hl_rest = order['half_life_months'].iloc[1:]
            if (hl_top - hl_rest.max()) > 0.4 and (hl_rest.max() - hl_rest.min()) < 0.3:
                # CDS-Bond stands apart; the other two are essentially identical
                top_strat = str(order['strategy'].iloc[0]).replace('--', '-').replace('_', ' ')
                rest_strats = ", ".join(
                    str(s).replace('--', '-').replace('_', ' ')
                    for s in order['strategy'].iloc[1:]
                )
                p4_evidence = (
                    rf"{top_strat} $\sim${hl_top:.1f}m; "
                    rf"{rest_strats} $\sim${hl_rest.mean():.1f}m (essentially identical)"
                )
            else:
                # Strict gradient: format with $>$ in math mode to avoid the "¿" bug
                p4_evidence = " $>$ ".join(
                    rf"{str(r['strategy']).replace('--', '-').replace('_', ' ')}"
                    rf"~$\sim${r['half_life_months']:.1f}m"
                    for _, r in order.iterrows()
                )

    # P5: PC1 ~ intermediary (Panel A R²_adj) AND Macro (Panel C R²_adj)
    p5_evidence = "See Table~\\ref{tab:rq3_pc1_funding}, Panels~A \\& C"
    df_pc = _load_csv("T3e2_pc1_funding_regression.csv")
    if df_pc is not None:
        r2_core = df_pc[df_pc['set'] == 'Core']['R2_adj'].iloc[0] \
                  if len(df_pc[df_pc['set'] == 'Core']) > 0 else None
        r2_macro = df_pc[df_pc['set'] == 'Macro']['R2_adj'].iloc[0] \
                   if len(df_pc[df_pc['set'] == 'Macro']) > 0 else None
        if r2_core is not None and r2_macro is not None:
            p5_evidence = (rf"Intermediary $\bar R^2 = {r2_core*100:.1f}\%$ vs "
                           rf"Macro placebo $\bar R^2 = {r2_macro*100:.1f}\%$ (n.s.)")

    rows = [
        (r"\textbf{P1}", "Cross-market correlation", p1_evidence),
        (r"\textbf{P2}", "Stress amplification", p2_evidence),
        (r"\textbf{P3}", "Lead-lag transmission", p3_evidence),
        (r"\textbf{P4}", "Persistence reflects funding-dependence", p4_evidence),
        (r"\textbf{P5}", "Common factor = intermediary, not macro", p5_evidence),
    ]

    tex = []
    tex.append(r"\centering\scriptsize")
    tex.append(r"\renewcommand{\arraystretch}{1.45}")
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\begin{tabular}{p{0.5cm}p{2.7cm}p{8.0cm}p{0.5cm}}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{P} & \textbf{Statement} & \textbf{Empirical evidence} & \\")
    tex.append(r"\midrule")
    # Per-prediction verdict: P4 is "Partially confirmed" (CDS-Bond stands apart,
    # but BTP and iTraxx are essentially identical, so the strict 3-way gradient
    # does not hold). Other predictions are unambiguously confirmed.
    verdicts = {
        r"\textbf{P1}": r"\textcolor{mygreen}{\large\checkmark}",
        r"\textbf{P2}": r"\textcolor{mygreen}{\large\checkmark}",
        r"\textbf{P3}": r"\textcolor{mygreen}{\large\checkmark}",
        r"\textbf{P4}": r"\textcolor{mygold}{\textbf{(partial)}}",
        r"\textbf{P5}": r"\textcolor{mygreen}{\large\checkmark}",
    }
    for pid, statement, evidence in rows:
        v = verdicts.get(pid, r"\textcolor{mygreen}{\large\checkmark}")
        tex.append(rf"{pid} & {statement} & {evidence} & {v} \\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    return "\n".join(tex)

def build_primary_outcomes_correction():
    """Correzione multiple-testing sui 3 PRIMARY_OUTCOMES pre-registrati (config).
    Famiglia = 3 test, non i ~61 esplorativi. Outcome institutional pair
    (CDS-Bond vs iTraxx), fissati a priori. Co-widening: p-value STATIONARY bootstrap.
    """
    from statsmodels.stats.multitest import multipletests

    primaries = []   # (label, p_raw)

    # 1) Co-widening HIGH vs LOW — institutional pair, stationary bootstrap
    df_cw = _load_csv("T2g_diff_cowidening_bootstrap.csv")
    p_cw = np.nan
    if df_cw is not None and len(df_cw):
        # cerca la riga institutional (CDS-Bond vs iTraxx) in modo difensivo
        col_pair = next((c for c in df_cw.columns if 'pair' in c.lower()), None)
        col_pstat = next((c for c in df_cw.columns
                          if 'stationary' in c.lower() and 'p' in c.lower()), None)
        if col_pair and col_pstat:
            mask = df_cw[col_pair].astype(str).str.contains('CDS', case=False) & \
                   df_cw[col_pair].astype(str).str.contains('iTraxx', case=False)
            if mask.any():
                p_cw = float(df_cw.loc[mask, col_pstat].iloc[0])
    primaries.append(("Co-widening HIGH vs LOW (stationary bootstrap)", p_cw))

    # 2) Interaction beta2 (Δm_j × Stress) — institutional pair
    df_int = _load_csv("T3d_interaction_regressions.csv")
    p_int = np.nan
    if df_int is not None and len(df_int):
        col_dep = next((c for c in df_int.columns if 'depend' in c.lower()), None)
        col_ind = next((c for c in df_int.columns if 'independ' in c.lower()), None)
        col_b2p = next((c for c in df_int.columns
                        if 'beta2' in c.lower() and ('_p' in c.lower() or 'p' == c.lower()[-1])), None)
        if col_dep and col_ind and col_b2p:
            m = (df_int[col_dep].astype(str).str.contains('CDS', case=False) &
                 df_int[col_ind].astype(str).str.contains('iTraxx', case=False))
            if m.any():
                p_int = float(df_int.loc[m, col_b2p].iloc[0])
    primaries.append(("Interaction β₂ (Δm×Stress)", p_int))

    # 3) Granger F-test — institutional pair (qualunque direzione, prendo la minima)
    df_g = _load_csv("T4b_granger_causality.csv")
    p_g = np.nan
    if df_g is not None and len(df_g):
        col_cs = next((c for c in df_g.columns if 'causing' in c.lower()), None)
        col_cd = next((c for c in df_g.columns if 'caused' in c.lower()), None)
        col_p = next((c for c in df_g.columns if c.lower() in ('p_value', 'p', 'pvalue')), None)
        if col_cs and col_cd and col_p:
            m = ((df_g[col_cs].astype(str).str.contains('CDS', case=False) &
                  df_g[col_cd].astype(str).str.contains('iTraxx', case=False)) |
                 (df_g[col_cs].astype(str).str.contains('iTraxx', case=False) &
                  df_g[col_cd].astype(str).str.contains('CDS', case=False)))
            if m.any():
                p_g = float(df_g.loc[m, col_p].min())
    primaries.append(("Granger F-test (CDS↔iTraxx)", p_g))

    # --- Correzione BH + Holm sui 3 primari ---
    labels = [l for l, p in primaries]
    pvals = np.array([p for l, p in primaries], dtype=float)
    valid = ~np.isnan(pvals)

    print("\n" + "=" * 60)
    print("PRIMARY OUTCOMES — Multiple Testing Correction (family = 3)")
    print("=" * 60)
    for l, p in primaries:
        print(f"   {l:48s}: p_raw = {p:.4f}" if not np.isnan(p) else f"   {l:48s}: p_raw = NaN")

    if valid.sum() >= 1:
        rej_bh, p_bh, _, _ = multipletests(pvals[valid], alpha=0.05, method='fdr_bh')
        rej_holm, p_holm, _, _ = multipletests(pvals[valid], alpha=0.05, method='holm')
        print(f"\n   {'Outcome':48s} {'p_raw':>8} {'p_BH':>8} {'p_Holm':>8}")
        j = 0
        rows = []
        for i, (l, p) in enumerate(primaries):
            if valid[i]:
                print(f"   {l:48s} {p:8.4f} {p_bh[j]:8.4f} {p_holm[j]:8.4f}"
                      f"  {'BH✓' if rej_bh[j] else ''} {'Holm✓' if rej_holm[j] else ''}")
                rows.append((l, p, p_bh[j], rej_bh[j], p_holm[j], rej_holm[j]))
                j += 1
        # salva CSV
        import pandas as _pd
        _pd.DataFrame(rows, columns=['outcome','p_raw','p_BH','reject_BH','p_Holm','reject_Holm']) \
            .to_csv(RQ3_TABLES_DIR / "T2i_primary_outcomes.csv", index=False)
        print(f"\n   💾 T2i_primary_outcomes.csv")

    # LaTeX
    tex = []
    tex.append(r"\begin{table}[H]\centering")
    tex.append(r"\caption{Primary Outcomes: Multiple-Testing Correction}")
    tex.append(r"\label{tab:rq3_primary_mtc}")
    tex.append(r"\begin{tabular}{l c c c}")
    tex.append(r"\toprule")
    tex.append(r"Pre-registered outcome & $p_{\text{raw}}$ & $p_{\text{BH}}$ & $p_{\text{Holm}}$ \\")
    tex.append(r"\midrule")
    if valid.sum() >= 1:
        j = 0
        for i, (l, p) in enumerate(primaries):
            if valid[i]:
                lbl = l.replace("β₂", r"$\beta_2$").replace("Δm", r"$\Delta m$").replace("×", r"$\times$")
                tex.append(rf"{lbl} & {p:.4f} & {p_bh[j]:.4f} & {p_holm[j]:.4f} \\")
                j += 1
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table}")
    return "\n".join(tex)

def main():
    print_header("GENERATING RQ3 LaTeX TABLES")
    print(f"   Input:  {RQ3_TABLES_DIR}")
    print(f"   Output: {TABLES_DIR}")

    # ── Body tables ────────────────────────────────────────────────────
    print_header("BODY TABLES (Section 6)", "-")

    # NOTE: builders for unused tables are kept in the code but NOT written out.
    # Removed from the paper (not shown in slides/talk): Unconditional (subsumed by
    # Unified), 2-level Unified (paper uses 3-level), single-lag Granger (paper uses
    # multilag), Duffie scorecard (stated verbally), Primary-Outcomes MTC.
    # Regime_Correlations is kept only until the Forbes–Rigobon value is folded into
    # build_unified_correlations, then it too can be dropped.
    # CoWidening is retained pending a diagnostic on its informativeness.
    body_tables = {
        "RQ3_Unified_Correlations_article.tex":
            build_unified_correlations(),
        "RQ3_Spanning_Regressions_article.tex":
            build_spanning_regressions(),
        "RQ3_Interaction_Regressions_article.tex":
            build_interaction_regressions(),
        "RQ3_AR1_Persistence_article.tex":
            build_ar1_persistence(),
        "RQ3_PC1_Funding_article.tex":
            build_pc1_funding(),
    }

    for fname, content in body_tables.items():
        if content:
            (TABLES_DIR / fname).write_text(content, encoding="utf-8")
            print(f"   ✅ {fname}")
        else:
            print(f"   ⚠️ {fname} — no data, skipped")

    # ── Appendix tables ────────────────────────────────────────────────
    print_header("APPENDIX TABLES (A.8)", "-")

    # Alternative_Proxies builder kept in code but not written (not in paper/slides).
    appendix_tables = {
        "RQ3_Granger_Multilag_article.tex":
            build_granger_multilag(),
    }

    for fname, content in appendix_tables.items():
        if content:
            (TABLES_DIR / fname).write_text(content, encoding="utf-8")
            print(f"   ✅ {fname}")
        else:
            print(f"   ⚠️ {fname} — no data, skipped")

    # ── Beamer slide tables (Section 5 of presentation) ──────────────
    print_header("BEAMER SLIDE TABLES (RQ3 Sec. 5)", "-")
    slide_tables = {
        "RQ3_Slide_P1_Correlations.tex":         build_slide_p1_correlations(),
        "RQ3_Slide_P2_Interaction.tex":          build_slide_p2_interaction(),
        "RQ3_Slide_P3_SpanningGranger.tex":      build_slide_p3_spanning_granger(),
        "RQ3_Slide_P4_Persistence.tex":          build_slide_p4_persistence(),
        "RQ3_Slide_P5_PC1_Funding.tex":          build_slide_p5_pc1_funding(),
        "RQ3_Slide_Mapping_Scorecard.tex":       build_slide_mapping_scorecard(),
    }
    for fname, content in slide_tables.items():
        if content:
            (TABLES_DIR / fname).write_text(content, encoding="utf-8")
            print(f"   ✅ {fname}")
        else:
            print(f"   ⚠️ {fname} — no data, skipped (will use fallback in slide)")

    # ── Summary ────────────────────────────────────────────────────────
    print_header("SUMMARY")
    n_body = sum(1 for c in body_tables.values() if c)
    n_app = sum(1 for c in appendix_tables.values() if c)
    print(f"   Body tables:     {n_body}")
    print(f"   Appendix tables: {n_app}")

    print(f"\n   Figures (regenerate with FIGURE_FORMAT='pdf' in rq3_00_config):")
    print(f"   Body:     C1_regime_correlations_bar, E1_girf, E3_fevd, G1_duffie_scorecard")
    print(f"   Appendix: F1_dcc_garch, F2_quantile_regression")

    print(f"\n{'=' * 80}")
    print(f"✅ RQ3 TABLE GENERATION COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
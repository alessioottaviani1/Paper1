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
config_path = PROJECT_ROOT / "src" / "rq3" / "rq3_00_config.py"

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
               r"$t$-statistics: HAC (Newey--West). "
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
               r"(change in mispricing) by iTraxx Main regime. "
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
    tex.append(r"\caption{Cross-Strategy Spanning Regressions ($\Delta m$)}")
    tex.append(r"\label{tab:rq3_spanning}")
    tex.append(r"\begin{threeparttable}")
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
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"$\Delta m_{i,t} = \alpha + \beta \Delta m_{j,t} + \varepsilon_t$. "
               r"Newey--West HAC standard errors. "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
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
    tex.append(r"\caption{Stress Amplification: Interaction Regressions}")
    tex.append(r"\label{tab:rq3_interaction}")
    tex.append(r"\begin{threeparttable}")
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
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"$\Delta m_{i,t} = \beta_0 \Delta m_{j,t} + "
               r"\beta_1 (\Delta m_{j,t} \times z_t) + \gamma z_t + \varepsilon_t$, "
               r"where $z_t$ is the standardized iTraxx Main 5Y spread. "
               r"$\beta_1 > 0$: interdependence increases under stress. "
               r"Newey--West HAC standard errors. "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
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
    tex.append(r"\caption{Granger Causality: Robustness to Lag Length}")
    tex.append(r"\label{tab:rq3_granger_multilag}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")

    lags = sorted(df['lag'].unique())
    n_lags = len(lags)

    tex.append(r"\begin{tabular}{l l " + "r r " * n_lags + "}")
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
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"Granger causality $F$-tests from VAR($p$) on $\Delta m$, "
               r"estimated separately at $p = 1, 2, 3$ lags. "
               r"Lag selection: HQC selects $p=1$; AIC/FPE select $p=2$; "
               r"BIC selects $p=0$ (forced to 1). "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
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

# ── TABLE 7: Purged Correlations ───────────────────────────────────────────

def build_purged_correlations():
    df = _load_csv("T2h_purged_correlations.csv")
    if df is None:
        return ""

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Correlations After Purging Common Factors}")
    tex.append(r"\label{tab:rq3_purged_corr}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l r r r}")
    tex.append(r"\toprule")
    tex.append(r"Pair & Raw & Purged & Reduction \\")
    tex.append(r"\midrule")

    # Load raw correlations from T2a to compute reduction
    df_raw = _load_csv("T2a_unconditional_correlations.csv")
    raw_lookup = {}
    if df_raw is not None and 'series' in df_raw.columns:
        df_raw_dm = df_raw[df_raw['series'] == 'Δm']
        for _, rr in df_raw_dm.iterrows():
            raw_lookup[rr.get('pair', '')] = rr.get('pearson_r', np.nan)

    for _, row in df.iterrows():
        pair_str = str(row.get('pair', ''))
        pair = _esc(pair_str)
        purged = row.get('pearson_r', np.nan)
        raw = raw_lookup.get(pair_str, np.nan)

        if not np.isnan(raw) and not np.isnan(purged) and abs(raw) > 1e-10:
            reduction = (1 - abs(purged) / abs(raw)) * 100
        else:
            reduction = np.nan

        tex.append(rf"{pair} & {_fmt3(raw)} & {_fmt3(purged)} "
                   rf"& {_fmt2(reduction)}\% \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"Correlations of residuals after regressing each strategy's "
               r"$\Delta m$ on common factors (EURIBOR-OIS, HKM intermediary "
               r"capital, HPW noise). Reduction shows the percentage decline "
               r"in absolute correlation.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

    return "\n".join(tex)


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


# ── TABLE 9: PC1 ~ Funding Stress ──────────────────────────────────────────

def build_pc1_funding():
    df = _load_csv("T3e2_pc1_funding_regression.csv")
    if df is None:
        return ""

    tex = []
    tex.append(r"\begin{table}[H]")
    tex.append(r"\centering")
    tex.append(r"\caption{Common Factor (PC1) and Funding Stress}")
    tex.append(r"\label{tab:rq3_pc1_funding}")
    tex.append(r"\begin{threeparttable}")
    tex.append(r"\begin{singlespace}")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{l r r r}")
    tex.append(r"\toprule")
    tex.append(r"Factor & Coefficient & $t$-stat & $p$-value \\")
    tex.append(r"\midrule")

    current_set = None
    set_labels = {
        'Core': 'Panel A: Core Intermediary Proxies',
        'Extended': 'Panel B: Extended Proxy Set',
    }

    for _, row in df.iterrows():
        row_set = str(row.get('set', ''))
        if row_set != current_set:
            if current_set is not None:
                tex.append(r"\midrule")
            label = set_labels.get(row_set, row_set)
            r2_adj = row.get('R2_adj', np.nan)
            t_obs = int(row.get('T', 0))
            tex.append(rf"\multicolumn{{4}}{{l}}{{\textbf{{{label}}}}} \\")
            if not np.isnan(r2_adj) and t_obs > 0:
                tex.append(rf"\multicolumn{{4}}{{l}}{{$T = {t_obs}$, "
                           rf"$\bar{{R}}^2 = {r2_adj:.3f}$}} \\")
            tex.append(r"\addlinespace")
            current_set = row_set

        factor = str(row.get('factor', row.get('Factor', row.get('variable', ''))))
        factor_clean = factor.replace('_', r'\_').replace('Δ', r'$\Delta$')
        coef = row.get('beta', row.get('coefficient', row.get('coef', np.nan)))
        t = row.get('t_stat', row.get('t_statistic', np.nan))
        p = row.get('p_value', np.nan)

        stars = _stars_sup(p) if not np.isnan(p) else ""
        tex.append(rf"{factor_clean} & ${coef:+.3f}{stars}$ "
                   rf"& {_fmt2(t)} & {_fmt3(p)} \\")
        
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append("")
    tex.append(r"\begin{tablenotes}[para,flushleft]")
    tex.append(r"\footnotesize")
    tex.append(r"\item \textit{Note:} "
               r"OLS regression of the first principal component of the three "
               r"$\Delta m$ series on intermediary stress proxies. "
               r"Panel~A: one proxy per Duffie channel (intermediary capital, "
               r"funding cost, dealer health, market illiquidity). "
               r"Panel~B: all available funding and liquidity proxies. "
               r"Significant coefficients indicate that the common factor "
               r"driving correlated mispricings is linked to intermediary "
               r"balance-sheet conditions \citep{duffie2010presidential, "
               r"he2017intermediary}. "
               r"Newey--West HAC standard errors. "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
    tex.append(r"\end{table}")

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
    tex.append(r"\caption{Unconditional and Regime-Dependent Correlations}")
    tex.append(r"\label{tab:rq3_unified_corr}")
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
        tex.append(r" & \multicolumn{2}{c}{Unconditional}"
                   r" & \multicolumn{3}{c}{By Regime} \\")
        tex.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-6}")
        tex.append(r"Pair & Pearson & $t$-stat"
                   r" & LOW & MED & HIGH \\")
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
                n_low = row_r.get('n_LOW', np.nan)
                n_med = row_r.get('n_MED', np.nan)
                n_high = row_r.get('n_HIGH', np.nan)

                p_low = _corr_pvalue(rho_low, n_low)
                p_med = _corr_pvalue(rho_med, n_med)
                p_high = _corr_pvalue(rho_high, n_high)

                stars_low = _stars_sup(p_low)
                stars_med = _stars_sup(p_med)
                stars_high = _stars_sup(p_high)
            else:
                rho_low, rho_med, rho_high = np.nan, np.nan, np.nan
                stars_low = stars_med = stars_high = ""

            tex.append(
                rf"{pair}"
                rf" & ${pearson:.3f}{stars_unc}$"
                rf" & {_fmt2(t_stat)}"
                rf" & ${_fmt3(rho_low)}{stars_low}$"
                rf" & ${_fmt3(rho_med)}{stars_med}$"
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
               r"Pearson correlations (unconditional) with "
               r"HAC $t$-statistics (Newey--West). "
               r"Regime columns: Pearson correlations in LOW, MEDIUM, and "
               r"HIGH stress (iTraxx Main 5Y thresholds). "
               r"Stars on unconditional: HAC inference. "
               r"Stars on regime correlations: two-sided $t$-test "
               r"with $n-2$ degrees of freedom. "
               r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.")
    tex.append(r"\end{tablenotes}")
    tex.append(r"\end{singlespace}")
    tex.append(r"\end{threeparttable}")
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
#                              MAIN
# ############################################################################

def main():
    print_header("GENERATING RQ3 LaTeX TABLES")
    print(f"   Input:  {RQ3_TABLES_DIR}")
    print(f"   Output: {TABLES_DIR}")

    # ── Body tables ────────────────────────────────────────────────────
    print_header("BODY TABLES (Section 6)", "-")

    body_tables = {
        "RQ3_Unconditional_Correlations_article.tex":
            build_unconditional_correlations(),
        "RQ3_Regime_Correlations_article.tex":
            build_regime_correlations(),
        "RQ3_Unified_Correlations_article.tex":
            build_unified_correlations(),
        "RQ3_Unified_Correlations_2L_article.tex":
            build_unified_correlations_2l(),
        "RQ3_CoWidening_article.tex":
            build_cowidening_standalone(),
        "RQ3_Spanning_Regressions_article.tex":
            build_spanning_regressions(),
        "RQ3_Interaction_Regressions_article.tex":
            build_interaction_regressions(),
        "RQ3_Granger_Causality_article.tex":
            build_granger_causality(),
        "RQ3_Duffie_Scorecard_article.tex":
            build_duffie_scorecard(),
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

    appendix_tables = {
        "RQ3_Purged_Correlations_article.tex":
            build_purged_correlations(),
        "RQ3_Alternative_Proxies_article.tex":
            build_alternative_proxies(),
        "RQ3_Granger_Multilag_article.tex":
            build_granger_multilag(),
    }

    for fname, content in appendix_tables.items():
        if content:
            (TABLES_DIR / fname).write_text(content, encoding="utf-8")
            print(f"   ✅ {fname}")
        else:
            print(f"   ⚠️ {fname} — no data, skipped")

    # ── Summary ────────────────────────────────────────────────────────
    print_header("SUMMARY")
    n_body = sum(1 for c in body_tables.values() if c)
    n_app = sum(1 for c in appendix_tables.values() if c)
    print(f"   Body tables:     {n_body}")
    print(f"   Appendix tables: {n_app}")

    print(f"\n   Figures (regenerate with FIGURE_FORMAT='pdf' in rq3_00_config):")
    print(f"   Body:     C1_regime_correlations_bar, E1_girf, E3_fevd, G1_duffie_scorecard")
    print(f"   Appendix: F1_dcc_garch, F2_quantile_regression, D1_pca_rolling_stability")

    print(f"\n{'=' * 80}")
    print(f"✅ RQ3 TABLE GENERATION COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
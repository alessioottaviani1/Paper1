# -*- coding: utf-8 -*-
"""
07_oos_alpha.py — Out-of-sample alpha for the three fixed-factor benchmarks.

Produces:
  - Panel D of Table 29 (alpha_synthesis_across_models): expanding-window
    out-of-sample alpha (60-month burn-in), EUR factors. This is the main-text
    OOS result, laid out like Panel A (alpha / t / N) for a row-by-row comparison.
  - the rolling 60-month OOS alpha, computed for appendix A.7 (kept in the JSON
    and printed for context; NOT in Panel D).
Expanding is the conventional primary OOS scheme (Welch-Goyal); the rolling
window is the robustness that relaxes the constant-loadings assumption (appendix).

Method (engine in oos.py): loadings estimated on a PAST-ONLY window, the slopes
carried forward to form the hedge (the in-window intercept is the alpha we
measure), and the realized abnormal return e_t = y_t - beta_hat'_{t-1} F_t tested
by Newey-West HAC. This is the implementability / out-of-sample counterpart to
the in-sample benchmark alphas (Tessaromatis; Moreira-Muir / Cederburg et al. 2020).

Burn-in note (this is the footnote argument, not a table): in the expanding scheme
the window at each t is always [0,t), so the estimate at every date is invariant to
the burn-in -- it only sets where the OOS average starts. 60 gives >=5 obs per
parameter on the first fit (K+1=11 for the largest specification).

Reuses oos.py; mirrors 03 (FRAMEWORKS, CSV naming, EUR-primary). Runs where the
regression_data_* CSVs live (produced locally by 01a/01b/01c).

Author: Alessio Ottaviani, EDHEC Business School.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from oos import oos_residuals, summarize, oos_r2, insample_alpha

# ----------------------------- configuration -----------------------------
REGRESSION_FREQ  = "monthly"
PERIODS_PER_YEAR = 12
OOS_MIN_TRAIN    = 60            # expanding burn-in AND rolling window length
ROLL_WINDOW      = 60            # the single rolling window reported (Panel D / A.7)
COND_MAX         = 1e3           # skip only genuinely degenerate windows (tunable)
REGIONS          = ["eur", "us"] # EUR primary (Table 29); US for the appendix
EXP_KEY  = f"exp{OOS_MIN_TRAIN}"
ROLL_KEY = f"roll{ROLL_WINDOW}"

STRATEGIES = ["BTP_Italia", "iTraxx_Combined", "CDS_Bond_Basis"]   # as in 03

FRAMEWORKS = {
    "Duarte": {
        "factors": ["Mkt-RF", "SMB", "HML", "UMD", "RS", "RI", "RB", "R2", "R5", "R10"],
        "data_pattern": "regression_data_{strategy}_{region}_{freq}.csv",
        "label": "Duarte et al. (2007)",
    },
    "ActiveFI": {
        "factors": ["Term", "Global_Term", "Global_Aggregate", "Inflation_Linkers",
                    "Corporate_Credit", "Emerging_Debt", "Emerging_Currency", "UST_Volatility"],
        "data_pattern": "regression_data_active_fi_{strategy}_{region}_{freq}.csv",
        "label": "Brooks et al. (2020)",
    },
    "FungHsieh": {
        "factors": ["SNP", "SIZE", "PTFSBD", "PTFSFX", "PTFSCOM", "TERM", "CREDIT"],
        "data_pattern": "regression_data_fung_hsieh_{strategy}_{region}_{freq}.csv",
        "label": "Fung & Hsieh (2004)",
    },
}

PROJECT_ROOT       = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR         = PROJECT_ROOT / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Only the two OOS schemes that are reported (Panel D + appendix A.7).
_VARIANTS = [(EXP_KEY,  "expanding", OOS_MIN_TRAIN, 0),
             (ROLL_KEY, "rolling",   OOS_MIN_TRAIN, ROLL_WINDOW)]


def load_regression_data(strategy, framework, region):
    fname = FRAMEWORKS[framework]["data_pattern"].format(
        strategy=strategy.lower(), region=region, freq=REGRESSION_FREQ)
    fp = PROCESSED_DATA_DIR / fname
    if not fp.exists():
        return None
    return pd.read_csv(fp, index_col=0, parse_dates=True).dropna()


def _safe_oos(y, X, scheme, min_window, roll_window):
    """Run one OOS variant; return NaNs cleanly if the sample is too short."""
    try:
        em, en, info = oos_residuals(y, X, scheme, min_window=min_window,
                                     roll_window=roll_window, cond_max=COND_MAX)
        s = summarize(em, PERIODS_PER_YEAR)
        return {"alpha": s["alpha_ann"], "t": s["t"], "p": s["p"],
                "n_oos": s["n_oos"], "skip": info["n_skipped"], "r2": oos_r2(em, en)}
    except ValueError:
        return {"alpha": np.nan, "t": np.nan, "p": np.nan, "n_oos": 0, "skip": 0, "r2": np.nan}


def run_cell(strategy, framework, region):
    df = load_regression_data(strategy, framework, region)
    if df is None or "Strategy_Return" not in df.columns:
        return None
    y = df["Strategy_Return"]
    factors = [f for f in FRAMEWORKS[framework]["factors"] if f in df.columns]
    X = df[factors]
    aligned = pd.concat([y, X], axis=1).dropna()
    n_total = len(aligned)
    if OOS_MIN_TRAIN >= n_total:
        return {"strategy": strategy, "framework": framework, "region": region,
                "note": f"too short (T={n_total})"}
    oos_index = aligned.index[OOS_MIN_TRAIN:]                   # common OOS months [60:T]

    row = {"strategy": strategy, "framework": framework, "region": region,
           "K": len(factors), "n_total": n_total, "oos_start": OOS_MIN_TRAIN}

    # in-sample references (HAC lags on each sample's own length; not reported, context only)
    isf = insample_alpha(y, X, periods_per_year=PERIODS_PER_YEAR)
    iss = insample_alpha(y, X, restrict_index=oos_index, periods_per_year=PERIODS_PER_YEAR)
    row.update({"alpha_is_full": isf["alpha_ann"], "t_is_full": isf["t"], "n_is_full": isf["n"],
                "alpha_is_sub": iss["alpha_ann"], "t_is_sub": iss["t"], "n_is_sub": iss["n"]})

    # the two reported OOS schemes
    for key, scheme, mw, rw in _VARIANTS:
        r = _safe_oos(y, X, scheme, mw, rw)
        row.update({f"alpha_oos_{key}": r["alpha"], f"t_oos_{key}": r["t"], f"p_oos_{key}": r["p"],
                    f"n_oos_{key}": r["n_oos"], f"skip_{key}": r["skip"], f"r2_oos_{key}": r["r2"]})
    return row


# ----------------------------- console table -----------------------------
def _star(t):
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return ""
    a = abs(t)
    return "***" if a > 2.58 else "**" if a > 1.96 else "*" if a > 1.64 else ""


def _a(alpha, t):
    if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)):
        return "n/a"
    return f"{alpha:+.2f}{_star(t)}"


def _print_table(rows):
    rows = [r for r in rows if "note" not in r]
    print("\n" + "=" * 92)
    print(f"OOS alpha (annualised %): in-sample vs out-of-sample, common OOS start = {OOS_MIN_TRAIN}")
    print("=" * 92)
    hdr = (f"{'Framework':9s} {'Strategy':15s} {'Reg':3s}  {'IS_full':>9s} {'IS_sub':>9s} "
           f"{'OOSexp':>9s} {'OOSroll'+str(ROLL_WINDOW):>9s}  {'n_oos':>5s} {'skip':>4s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        n_oos = r[f"n_oos_{EXP_KEY}"]; skip = r.get(f"skip_{ROLL_KEY}", 0)
        print(f"{r['framework']:9s} {r['strategy']:15s} {r['region']:3s}  "
              f"{_a(r['alpha_is_full'], r['t_is_full']):>9s} {_a(r['alpha_is_sub'], r['t_is_sub']):>9s} "
              f"{_a(r[f'alpha_oos_{EXP_KEY}'], r[f't_oos_{EXP_KEY}']):>9s} "
              f"{_a(r[f'alpha_oos_{ROLL_KEY}'], r[f't_oos_{ROLL_KEY}']):>9s}  "
              f"{n_oos:>5d} {skip:>4d}")
    print(f"(IS_full: full sample. IS_sub, OOSexp, OOSroll{ROLL_WINDOW}: same OOS months [{OOS_MIN_TRAIN}:T].)")


# ---------- LaTeX: Panel D for Table 29 (alpha_synthesis_across_models) ----------
# Same style as Panel A in 03_subperiod_rolling_analysis.py: {l + " r r r"*n_fw},
# \multicolumn{3}{c}{framework}, alpha/t/N per framework, alpha with $..^{***}$.
# Expanding-window OOS only (rolling-60 is the appendix A.7 robustness).
TEX_FRAMEWORKS = [("Duarte",    r"Duarte et al.\ (2007)"),
                  ("ActiveFI",  r"Brooks et al.\ (2020)"),
                  ("FungHsieh", r"Fung \& Hsieh (2004)")]
TEX_STRAT = {"BTP_Italia": "BTP Italia", "iTraxx_Combined": "iTraxx Combined",
             "CDS_Bond_Basis": "CDS--Bond Basis"}


def _sig_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def _tex_a(alpha, p):
    if alpha is None or (isinstance(alpha, float) and np.isnan(alpha)):
        return "--"
    s = _sig_p(p)
    return f"${alpha:+.2f}^{{{s}}}$" if s else f"{alpha:+.2f}"


def write_panel_d_tex(rows, region="eur"):
    """Emit the Panel D block (expanding-window OOS alpha, Panel A style) for Table 29."""
    idx = {(r["framework"], r["strategy"]): r for r in rows
           if r.get("region") == region and "note" not in r}
    n_fw = len(TEX_FRAMEWORKS)
    L = [r"\vspace{0.4cm}",
         r"\begin{center}\textit{Panel D: Out-of-sample alpha (\% p.a.)}\end{center}",
         r"\vspace{0.1cm}", "",
         r"\noindent\makebox[\linewidth][c]{%",
         r"\begin{tabular}{l" + " r r r" * n_fw + "}",
         r"\toprule"]
    h1 = " "
    for _, lab in TEX_FRAMEWORKS:
        h1 += rf" & \multicolumn{{3}}{{c}}{{{lab}}}"
    L.append(h1 + r" \\")
    L.append("".join(rf"\cmidrule(lr){{{2 + i*3}-{4 + i*3}}}" for i in range(n_fw)))
    L.append(" " + r" & $\alpha$ & $t$ & $N$" * n_fw + r" \\")
    L.append(r"\midrule")
    for strat in STRATEGIES:
        row = rf"\textit{{{TEX_STRAT.get(strat, strat)}}}"
        for fwk, _ in TEX_FRAMEWORKS:
            r = idx.get((fwk, strat))
            a = r.get(f"alpha_oos_{EXP_KEY}") if r else None
            if a is not None and not (isinstance(a, float) and np.isnan(a)):
                row += " & " + _tex_a(a, r.get(f"p_oos_{EXP_KEY}"))
                row += f" & {r[f't_oos_{EXP_KEY}']:.2f} & {r[f'n_oos_{EXP_KEY}']}"
            else:
                row += " & -- & -- & --"
        L.append(row + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", ""]
    out = TABLES_DIR / "panel_d_oos_alpha.tex"
    out.write_text("\n".join(L) + "\n")
    return out


def main():
    rows = []
    for fw in FRAMEWORKS:
        for strat in STRATEGIES:
            for reg in REGIONS:
                r = run_cell(strat, fw, reg)
                if r is None:
                    print(f"  (no CSV) {fw} / {strat} / {reg}")
                    continue
                rows.append(r)
                if "note" in r:
                    print(f"  {fw}/{strat}/{reg}: {r['note']}")

    _print_table(rows)

    out = TABLES_DIR / "oos_alpha_results.json"
    out.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nSaved: {out}")
    pd_path = write_panel_d_tex(rows, region="eur")
    print(f"Saved Panel D (EUR) LaTeX: {pd_path}  -> inlined into Table 29 by 03_subperiod_rolling_analysis.py")
    print("Notes: alpha in Strategy_Return units, annualised x12. Significance (HAC): * 10%, ** 5%, *** 1%.")


if __name__ == "__main__":
    main()

"""
================================================================================
07_tables.py — Article LaTeX tables (machine-learning / best-subset layer)
================================================================================

Builds the paper tables for the best-subset spanning-regression layer from the
canonical outputs of the cleaned pipeline. One builder per table; each table
makes a single point and is written as a self-contained `threeparttable`.

Tables (written to <results>/tables/):
  main_spanning.tex   Spanning regressions: alpha (HAC t + block-bootstrap CI)
                      and factor loadings per strategy.        [03 + 04]
  k_sensitivity.tex   Best-subset support at k in {5, 8, 10}.  [02s]
  oos_alpha.tex       In-sample vs out-of-sample alpha
                      (expanding + rolling).                   [06_aen_oos]
  cross_layer.tex     Alpha and R2adj across the three empirical layers
                      (factor benchmark / PCA / best-subset).  [benchmark + PCA + 03]
  selector_overlap.tex   Selected set under best-subset / AEN / double-selection
                      (columns shown only where the input exists).  [02s + 02 + PDS]
  factor_list.tex     Definitions of the factors that enter any selected set.

Design follows Cochrane's table rules: a self-contained caption carrying the
regression equation and the left-hand variable; two-to-three significant
digits; sensible units (alpha in % p.a.); no number that the paper does not
discuss. Beamer slides and figures are intentionally NOT produced here.

Institution: EDHEC Business School — PhD Thesis
================================================================================
"""

import json
import importlib.util
from pathlib import Path

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"

_spec = importlib.util.spec_from_file_location("aen_config", CONFIG_PATH)
aen_config = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(aen_config)

HAC_LAGS             = aen_config.HAC_LAGS
STRATEGIES           = aen_config.STRATEGIES
get_strategy_aen_dir = aen_config.get_strategy_aen_dir
get_aen_output_dir   = aen_config.get_aen_output_dir

RESULTS_DIR = aen_config.RESULTS_DIR
TABLES_DIR  = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

TITLE_MAP = {
    "btp_italia":      "BTP Italia",
    "cds_bond_basis":  "CDS--Bond Basis",
    "itraxx_combined": "iTraxx Skew",
}

# Filenames the post-double-selection module may write (PDS column is optional).
PDS_CANDIDATES = (
    "postselection_results.json",
    "aen_postselection.json",
    "double_selection.json",
    "pds_results.json",
)

# ── Factor definitions (symbol -> (description, source)) ────────────────────────
FACTOR_INFO: dict[str, tuple[str, str]] = {
    "ILLIQ":            ("Change in the Amihud illiquidity measure.", "Amihud and Mendelson (2015)"),
    "SILLIQ":           ("Stock-market illiquidity innovation (AR(3) residual).", "Acharya et al.\\ (2013)"),
    "LIBOR_REPO_SHOCK": ("Dollar funding-liquidity shock (LIBOR $-$ GC repo, AR(2) res.).", "Asness et al.\\ (2013)"),
    "TED_SHOCK_EU":     ("Euro interbank funding shock (Euribor $-$ German bill, AR(2) res.).", "Asness et al.\\ (2013)"),
    "EURIBOR_OIS":      ("Euro interbank credit/liquidity risk (3M Euribor $-$ OIS).", "Nyborg and Ostberg (2014)"),
    "UMD_EU":           ("European equity momentum (winners $-$ losers).", "Carhart (1997)"),
    "SMB_EU":           ("European equity size (small $-$ big caps).", "Fama and French (2017)"),
    "BAB_US":           ("US betting-against-beta (low $-$ high beta).", "Frazzini and Pedersen (2014)"),
    "BAB_EU":           ("European betting-against-beta (low $-$ high beta).", "Frazzini and Pedersen (2014)"),
    "BTP_BUND":         ("Italian sovereign risk (10Y BTP $-$ Bund spread).", "Fausch and Sigonius (2018)"),
    "TERM_US":          ("US term premium (long gov.\\ bond excess return).", "Fama and French (1993)"),
    "TERM_EU":          ("Euro term premium (long Bund excess return).", "Fama and French (1993)"),
    "SS10Y":            ("10Y EUR swap spread (swap rate $-$ 10Y Bund).", "Collin-Dufresne et al.\\ (2001)"),
    "SS5Y":             ("5Y EUR swap spread (swap rate $-$ 5Y Bobl).", "Collin-Dufresne et al.\\ (2001)"),
    "SS2Y":             ("2Y EUR swap spread (swap rate $-$ 2Y Schatz).", "Collin-Dufresne et al.\\ (2001)"),
    "R10_EU":           ("Euro 10Y government bond return.", "Cochrane and Piazzesi (2005)"),
    "EBP":              ("Excess bond premium: corporate spread net of expected default.", "Gilchrist and Zakrajsek (2012)"),
    "CREDIT_EU":        ("Euro credit spread (BBB $-$ AAA yield).", "Fama and French (1989)"),
    "CRED_SPR_US":      ("US credit spread (Moody's BAA $-$ AAA yield).", "Fama and French (1989)"),
    "DEF_US":           ("US default factor (corporate $-$ gov.\\ long bond return).", "Fama and French (1993)"),
    "RI_EU":            ("Euro investment-grade industrial bond return.", "Collin-Dufresne et al.\\ (2001)"),
    "PB_EU_CDS_1Y":     ("European prime-broker 1Y CDS (counterparty risk).", "Klaus and Rzepkowski (2009)"),
    "PB_EU_CDS_5Y":     ("European prime-broker 5Y CDS (funding risk).", "Klaus and Rzepkowski (2009)"),
    "EP_SVIX_1M":       ("Option-implied equity premium via SVIX, 1-month.", "Martin (2017)"),
    "EP_SVIX_3M":       ("Option-implied equity premium via SVIX, 3-month.", "Martin (2017)"),
    "\u0394UF":         ("Change in financial uncertainty.", "Ludvigson et al.\\ (2021)"),
    "\u0394UM":         ("Change in macroeconomic uncertainty.", "Ludvigson et al.\\ (2021)"),
    "\u0394UR":         ("Change in real-activity uncertainty.", "Ludvigson et al.\\ (2021)"),
    "\u0394V2X":        ("Change in V2X (30-day implied vol, Euro STOXX 50).", "Chung et al.\\ (2019)"),
    "\u0394FAILS_PCT_TSY": ("Change in US Treasury fails / debt outstanding.", "Fleckenstein et al.\\ (2014)"),
    "LIBOR_OIS":        ("Libor--OIS spread (3M LIBOR $-$ 3M OIS).", "Nyborg and Ostberg (2014)"),
    "CDX_IG":           ("First difference of the CDX IG index CDS spread.", "Klaus and Rzepkowski (2009)"),
    "EMERG_FX":         ("Equal-weighted EM currency basket vs USD.", "Brooks et al.\\ (2020)"),
    "PTFSFX":           ("Currency trend-following factor (FX lookback straddles).", "Fung and Hsieh (2004)"),
}


# ── Formatting helpers ──────────────────────────────────────────────────────────
def _is_bad(x):
    return x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))


def _stars(p):
    if _is_bad(p):
        return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _sup(p):
    """Significance as a math superscript, e.g. ^{**}."""
    s = _stars(p)
    return f"^{{{s}}}" if s else ""


def _f1(x):   # one decimal (alpha %, CI bounds)
    return "--" if _is_bad(x) else f"{x:.1f}"


def _f2(x):   # two decimals (t-stats, R2adj)
    return "--" if _is_bad(x) else f"{x:.2f}"


def _f3(x):   # three decimals / sig figs (loadings)
    return "--" if _is_bad(x) else f"{x:.3f}"


def _pretty(name: str) -> str:
    """LaTeX-safe factor symbol for tables."""
    return str(name).replace("_", r"\_").replace("\u0394", r"$\Delta$")


def _load_json(path: Path):
    if not path or not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _print_header(title, char="="):
    print(f"\n{char * 80}\n{title}\n{char * 80}")


def _selected(d) -> list:
    """Pull a selected-factor list from a results dict, tolerant of key names."""
    if not d:
        return []
    for k in ("selected_factors", "stable_factors", "factors_selected"):
        v = d.get(k)
        if isinstance(v, list):
            return v
    return []


# ── TABLE 1 — Spanning regressions (the main result) ────────────────────────────
def build_main_spanning(all_ols: dict, all_boot: dict) -> str:
    """Alpha (HAC t + bootstrap CI) and factor loadings, one panel per strategy."""
    strats = [s for s in STRATEGIES if s in all_ols]
    if not strats:
        return ""

    n_reps = None
    boot_method = None
    for s in strats:
        b = all_boot.get(s)
        if b:
            n_reps = b.get("n_reps", n_reps)
            boot_method = b.get("bootstrap_method", boot_method)

    boot_note = ""
    if n_reps:
        boot_note = (rf"The 95\% confidence interval for $\alpha$ is the percentile interval "
                     rf"from a {boot_method or 'block'} bootstrap ({n_reps} replications). ")
    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\singlespacing",
        r"\caption{Post-Selection OLS on the Best-Subset Factors}",
        r"\label{tab:aen_stable_ols}",
        r"\begin{minipage}{\textwidth}",
        r"{\footnotesize\noindent The table reports OLS estimates of "
        r"$r_{i,t} = \alpha_i + \sum_j \beta_{i,j}\,F_{j,t} + \varepsilon_{i,t}$, "
        r"where $r_{i,t}$ is the monthly excess return of arbitrage strategy $i$ and "
        r"$F_{j,t}$ are the factors selected for that strategy by best-subset selection "
        r"(Bertsimas, King and Mazumder, 2016). Factors enter in original (un-standardized) "
        r"units; $\alpha$ is annualized (\% p.a.). " + boot_note +
        rf"$t$-statistics use Newey--West HAC standard errors ({HAC_LAGS} lags). "
        r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$. "
        r"Factor definitions are in Table~\ref{tab:factor_list}.}",
        r"\end{minipage}",
        r"\par\vspace{6pt}",
        r"\begin{singlespace}",
        r"\small",
        r"\begin{tabular}{l r r}",
        r"\toprule",
        r" & Coefficient & $t$-stat \\",
        r"\midrule",
    ]

    for idx, s in enumerate(strats):
        d = all_ols[s]
        b = all_boot.get(s, {})
        title = TITLE_MAP.get(s, s)
        factors = _selected(d)

        if idx > 0:
            tex.append(r"\addlinespace")
            tex.append(r"\midrule")
        panel = chr(65 + idx)
        tex.append(rf"\multicolumn{{3}}{{l}}{{\textit{{Panel {panel}: {title}}}}} \\")
        tex.append(r"\addlinespace")

        a = d.get("alpha", {})
        a_ann, a_t, a_p = a.get("annualized_pct"), a.get("t_statistic"), a.get("p_value")
        tex.append(rf"$\alpha$ (\% p.a.) & ${_f1(a_ann)}{_sup(a_p)}$ & {_f2(a_t)} \\")

        # block-bootstrap 95% CI for alpha (annualized), if present
        ci = (b.get("alpha", {}) or {}).get("ci95_annualized_pct")
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            tex.append(rf"\quad 95\% CI (bootstrap) & "
                       rf"\multicolumn{{2}}{{r}}{{$[{_f1(ci[0])},\ {_f1(ci[1])}]$}} \\")
        tex.append(r"\addlinespace")

        fac = d.get("factors", {})
        for f in factors:
            fd = fac.get(f, {})
            tex.append(rf"{_pretty(f)} & ${_f3(fd.get('coefficient'))}{_sup(fd.get('p_value'))}$ "
                       rf"& {_f2(fd.get('t_statistic'))} \\")

        tex.append(r"\addlinespace")
        tex.append(rf"$T = {d.get('T','--')}$, $k = {d.get('n_factors', len(factors))}$, "
                   rf"$\bar R^2 = {_f2(d.get('r_squared_adj'))}$ & & \\")

    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{singlespace}",
        r"\end{table}",
    ]
    return "\n".join(tex)


# ── TABLE 2 — k-sensitivity ──────────────────────────────────────────────────────
def build_k_sensitivity(all_sens: dict) -> str:
    """Best-subset support and fit at each k in K_GRID, one panel per strategy."""
    strats = [s for s in STRATEGIES if s in all_sens]
    if not strats:
        return ""

    # discover the k grid from the data (sorted ascending)
    ks: list[int] = []
    for s in strats:
        for k in (all_sens[s].get("k_grid", {}) or {}).keys():
            if int(k) not in ks:
                ks.append(int(k))
    ks = sorted(ks)

    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\singlespacing",
        r"\caption{Best-Subset Support as the Cardinality $k$ Varies}",
        r"\label{tab:ml_ksens}",
        r"\begin{minipage}{\textwidth}",
        r"{\footnotesize\noindent Each row reports the best-subset solution of exactly $k$ factors "
        r"(Bertsimas, King and Mazumder, 2016) on the full factor panel; the primary "
        r"specification fixes $k=8$. $\alpha$ is annualized (\% p.a.), with Newey--West HAC "
        r"$t$-statistics in parentheses; $\bar R^2$ is the in-sample adjusted $R^2$ on the "
        r"standardized design. "
        r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.}",
        r"\end{minipage}",
        r"\par\vspace{6pt}",
        r"\begin{singlespace}",
        r"\small",
        r"\begin{tabular}{l c c p{0.62\linewidth}}",
        r"\toprule",
        r"$k$ & \# & $\alpha$ (\% p.a.) & $\bar R^2$ & Selected factors \\",
        r"\midrule",
    ]
    for idx, s in enumerate(strats):
        title = TITLE_MAP.get(s, s)
        grid = all_sens[s].get("k_grid", {})
        if idx > 0:
            tex.append(r"\addlinespace")
            tex.append(r"\midrule")
        tex.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{title}}}}} \\")
        for k in ks:
            cell = grid.get(str(k))
            if not cell:
                continue
            facs = cell.get("factors", [])
            names = ", ".join(_pretty(f) for f in facs)
            a, t, p = cell.get("alpha"), cell.get("alpha_t"), cell.get("alpha_p")
            acell = "--" if a is None else rf"${a:+.2f}{_stars(p)}$ ({t:.2f})"
            tex.append(rf"{k} & {len(facs)} & {acell} & {_f2(cell.get('r2_adj_std'))} & {names} \\")

    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\end{singlespace}",
        r"\end{table}",
    ]
    return "\n".join(tex)


# ── TABLE 3 — Out-of-sample alpha ────────────────────────────────────────────────
def build_oos(oos_exp: dict, oos_roll: dict) -> str:
    """In-sample vs OOS alpha (expanding and rolling), one row per strategy."""
    strats = [s for s in STRATEGIES if (s in oos_exp or s in oos_roll)]
    if not strats:
        return ""

    def cell(d):
        if not d:
            return r"-- & "
        return rf"${_f1(d.get('alpha_oos_ann'))}{_sup(d.get('p'))}$ & {_f2(d.get('t'))}"

    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{In-sample and out-of-sample alpha}",
        r"\label{tab:ml_oos}",
        r"\begin{threeparttable}",
        r"\begin{singlespace}",
        r"\small",
        r"\begin{tabular}{l c cc cc}",
        r"\toprule",
        r" & In-sample & \multicolumn{2}{c}{OOS, expanding} & \multicolumn{2}{c}{OOS, rolling} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Strategy & $\alpha$ & $\alpha$ & $t$ & $\alpha$ & $t$ \\",
        r"\midrule",
    ]
    for s in strats:
        title = TITLE_MAP.get(s, s)
        de, dr = oos_exp.get(s), oos_roll.get(s)
        ins = (de or dr or {}).get("alpha_insample_ann")
        tex.append(rf"{title} & ${_f1(ins)}$ & {cell(de)} & {cell(dr)} \\")

    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\begin{tablenotes}[para,flushleft]",
        r"\footnotesize",
        r"\item \textit{Note:} The in-sample column is the full-sample OLS intercept; the "
        r"out-of-sample columns freeze the best-subset factor set and re-estimate only the hedge "
        r"loadings on past data, then record the realized abnormal return on the next month "
        r"$e_t = r_t - \hat\beta'_{t-1} F_t$. ``Expanding'' grows the estimation window; "
        r"``rolling'' uses a 60-month window. All figures are annualized (\% p.a.); $t$-statistics "
        r"use Newey--West HAC standard errors. $^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.",
        r"\end{tablenotes}",
        r"\end{singlespace}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    return "\n".join(tex)


# ── TABLE 5 — Selector overlap (best-subset / AEN / double-selection) ────────────
def build_selector_overlap(all_subset: dict, all_aen: dict, all_pds: dict) -> str:
    """Selected set under every selector of Table~\ref{tab:selector_robustness},
    one panel per strategy. Sets are read from the benchmark JSON written by
    08_selector_benchmark.py, so the two tables can never fall out of sync."""
    strats = [s for s in STRATEGIES if s in all_subset]
    if not strats:
        return ""

    # Sets from the selector benchmark (same six methods as Table XIII).
    import json as _json
    _bench_path = get_aen_output_dir() / "benchmark" / "selector_benchmark.json"
    _bench = {}
    if _bench_path.exists():
        try:
            _bench = _json.loads(_bench_path.read_text(encoding="utf-8"))
        except Exception as _e:
            print(f"   WARNING: cannot read {_bench_path} ({_e}); falling back to three columns.")

    def _from_bench(method):
        out = {}
        for s in strats:
            for r in _bench.get(s, []):
                if r.get("method") == method:
                    f = r.get("factors")
                    out[s] = {"selected_factors": f} if isinstance(f, list) else {"selected_factors": []}
        return out

    if _bench:
        cols = [("Best subset", all_subset),
                ("One per risk category", _from_bench("group_rep")),
                ("Adaptive elastic net", all_aen),
                ("Relaxed lasso", _from_bench("relaxed")),
                ("Random forest", _from_bench("rf")),
                ("Model-X knockoffs", _from_bench("knockoffs"))]
        if any(all_pds.get(s) for s in strats):
            cols.append(("Double-selection", all_pds))
    else:
        cols = [("Best subset", all_subset)]
        if any(all_aen.get(s) for s in strats):
            cols.append(("Adaptive elastic net", all_aen))
        if any(all_pds.get(s) for s in strats):
            cols.append(("Double-selection", all_pds))

    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\singlespacing",
        r"\caption{Selected Factors Across Selection Methods}",
        r"\label{tab:aen_selection_names}",
        r"\begin{minipage}{\textwidth}",
        r"{\footnotesize\noindent The factor set selected for each strategy by every method of "
        r"Table~\ref{tab:selector_robustness}; the methods themselves are described in "
        r"Section~\ref{sec:aen}. $\emptyset$ marks a method that retains no factor: the "
        r"relaxed lasso, post-double-selection and the knockoffs let the data choose the "
        r"cardinality, and for the iTraxx skew none of the $69$ candidates clears their "
        r"threshold. "
        r"The identity of the selected factors varies across methods, as expected when "
        r"correlated candidates substitute for one another, while the alpha of "
        r"Table~\ref{tab:selector_robustness} does not.}",
        r"\end{minipage}",
        r"\par\vspace{6pt}",
        r"\begin{singlespace}",
        r"\small",
        r"\begin{tabular}{l p{0.72\linewidth}}",
        r"\toprule",
        r"Method & Selected factors \\",
        r"\midrule",
    ]
    for idx, s in enumerate(strats):
        title = TITLE_MAP.get(s, s)
        if idx > 0:
            tex.append(r"\addlinespace")
            tex.append(r"\midrule")
        tex.append(rf"\multicolumn{{2}}{{l}}{{\textit{{{title}}}}} \\")
        for label, store in cols:
            facs = _selected(store.get(s))
            names = ", ".join(_pretty(f) for f in facs) if facs else r"$\emptyset$"
            tex.append(rf"{label} & {names} \\")

    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{singlespace}",
        r"\end{table}",
    ]
    return "\n".join(tex)


# ── TABLE 6 — Factor definitions ─────────────────────────────────────────────────
def build_factor_list(used_factors: set) -> str:
    """Definitions of the factors that enter any selected set."""
    facs = sorted(used_factors)
    if not facs:
        return ""

    tex = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Factor definitions}",
        r"\label{tab:ml_factor_list}",
        r"\begin{threeparttable}",
        r"\begin{singlespace}",
        r"\small",
        r"\begin{tabular}{l p{0.56\linewidth} l}",
        r"\toprule",
        r"Factor & Definition & Source \\",
        r"\midrule",
    ]
    for f in facs:
        desc, src = FACTOR_INFO.get(f, ("", ""))
        tex.append(rf"{_pretty(f)} & {desc} & {src} \\")

    tex += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\begin{tablenotes}[para,flushleft]",
        r"\footnotesize",
        r"\item \textit{Note:} Factors that enter at least one strategy's selected set. "
        r"All factors are constructed as monthly innovations / returns as described in the source.",
        r"\end{tablenotes}",
        r"\end{singlespace}",
        r"\end{threeparttable}",
        r"\end{table}",
    ]
    return "\n".join(tex)


# ── Driver ────────────────────────────────────────────────────────────────────
def main():
    _print_header("ARTICLE TABLES — best-subset spanning layer")
    print(f"   Output: {TABLES_DIR}")

    all_ols, all_boot, all_sens = {}, {}, {}
    all_subset, all_aen, all_pds = {}, {}, {}
    for s in STRATEGIES:
        sd = get_strategy_aen_dir(s)
        for store, fname in ((all_ols, "ols_results.json"),
                             (all_boot, "bootstrap_stability.json"),
                             (all_sens, "subset_sensitivity.json"),
                             (all_subset, "subset_results.json"),
                             (all_aen, "aen_results.json")):
            d = _load_json(sd / fname)
            if d:
                store[s] = d
        for cand in PDS_CANDIDATES:                       # optional PDS output
            d = _load_json(sd / cand)
            if d:
                all_pds[s] = d
                break

    out = get_aen_output_dir()
    oos_exp = _load_json(out / "aen_oos_alpha_expanding.json") or {}
    oos_roll = _load_json(out / "aen_oos_alpha_rolling.json") or {}

    # factors that appear in any selected set (for the factor-list table)
    used = set()
    for s in STRATEGIES:
        used.update(_selected(all_ols.get(s)) or _selected(all_subset.get(s)))

    tables = {
        "AEN_Stable_OLS_article.tex":        build_main_spanning(all_ols, all_boot),
        "k_sensitivity.tex":    build_k_sensitivity(all_sens),
        "AEN_Selection_Names_article.tex": build_selector_overlap(all_subset, all_aen, all_pds),
        "aen_selected_factor_defs.tex": build_factor_list(used),
    }

    _print_header("WRITING", "-")
    written = 0
    for fname, content in tables.items():
        if content:
            (TABLES_DIR / fname).write_text(content, encoding="utf-8")
            print(f"   \u2705 {fname}")
            written += 1
        else:
            print(f"   \u26a0\ufe0f  {fname} — no input data, skipped")

    _print_header("DONE")
    print(f"   {written}/{len(tables)} tables written to {TABLES_DIR}")


if __name__ == "__main__":
    main()

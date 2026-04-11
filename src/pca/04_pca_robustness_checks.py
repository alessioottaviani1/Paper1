"""
================================================================================
04_pca_robustness_checks.py - PCA Robustness (W x K) for Alpha and R^2
================================================================================

Runs PCA rolling spanning regressions across:
- Window length W in {24, 36, 48} months
- Number of PCs K in {8, 11, 14}
- Timing in {predictive, contemporaneous}

Pipeline per (W,K):
1) Run 01_pca_preprocessing.py  -> creates factors_for_pca.parquet + y_returns_pca.parquet
2) Run 02_pca_rolling.py twice  -> spanning_regression_results_{timing}.json

Outputs (aggregated):
- results/robustness/tables/PCA_robustness_alpha_R2.tex
- results/robustness/tables/PCA_robustness_alpha_R2.csv

Author: Alessio Ottaviani
================================================================================
"""

import json
import importlib.util
from pathlib import Path
import pandas as pd

# =============================================================================
# LOAD CONFIG (same logic as your other scripts)
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

config_paths = [
    PROJECT_ROOT / "src" / "pca" / "00_pca_config.py",
    PROJECT_ROOT / "src" / "pca" / "00_pca_config_fix.py",
]

pca_config = None
loaded_cfg_path = None
for config_path in config_paths:
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("pca_config", config_path)
        pca_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pca_config)
        loaded_cfg_path = config_path
        break

if pca_config is None:
    raise FileNotFoundError("PCA config file not found!")

print(f"✅ Loaded config from: {loaded_cfg_path}")

BASE_RESULTS_DIR = pca_config.RESULTS_DIR  # keep original
ROBUST_ROOT = BASE_RESULTS_DIR / "robustness"
ROBUST_ROOT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# LOAD 01/02 AS MODULES (call main() programmatically)
# =============================================================================

def load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

PCA_DIR = PROJECT_ROOT / "src" / "pca"
mod01 = load_module("pca_preprocessing_01", PCA_DIR / "01_pca_preprocessing.py")
mod02 = load_module("pca_rolling_02", PCA_DIR / "02_pca_rolling.py")

def sync_globals(mod, cfg):
    """Force a module to use the *current* config values (mutated at runtime)."""
    mod.pca_config = cfg
    mod.RESULTS_DIR = cfg.RESULTS_DIR
    mod.STRATEGIES = cfg.STRATEGIES
    mod.PCA_N_COMPONENTS = cfg.PCA_N_COMPONENTS
    mod.PCA_WINDOW_LENGTH = cfg.PCA_WINDOW_LENGTH
    if hasattr(cfg, "PCA_TIMING"):
        mod.PCA_TIMING = cfg.PCA_TIMING
    if hasattr(cfg, "PCA_START_DATE"):
        mod.PCA_START_DATE = cfg.PCA_START_DATE
    mod.get_pca_output_dir = cfg.get_pca_output_dir
    mod.get_strategy_pca_dir = cfg.get_strategy_pca_dir

# =============================================================================
# HELPERS: read results + formatting
# =============================================================================

def sig_stars(pval: float) -> str:
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""

def strategy_label(name: str) -> str:
    return name.replace("_", " ").title()

def add_months(date_str: str, n_months: int) -> str:
    d = pd.Timestamp(date_str)
    d2 = d + pd.DateOffset(months=int(n_months))
    # preserva fine mese (Timestamp lo gestisce bene per month-end già coerenti)
    return d2.strftime("%Y-%m-%d")

def read_spanning_results(cfg, timing: str):
    """
    Return dict[strategy -> json] for current cfg.RESULTS_DIR and given timing.
    """
    out = {}
    for strategy_name in cfg.STRATEGIES.keys():
        sdir = cfg.get_strategy_pca_dir(strategy_name)
        jpath = sdir / f"spanning_regression_results_{timing}.json"
        if jpath.exists():
            with open(jpath, "r", encoding="utf-8") as f:
                out[strategy_name] = json.load(f)
    return out

def read_explained_variance_pct(cfg, timing: str):
    """
    Read avg cumulative explained variance (in %) for current run (W,K)
    from: get_pca_output_dir() / pca_summary_{timing}.json

    NOTE: in 02_pca_rolling.py it is stored under:
      summary['pca_diagnostics']['avg_variance_explained']
    """
    try:
        pca_dir = cfg.get_pca_output_dir()
    except Exception:
        pca_dir = cfg.RESULTS_DIR / "pca"

    summary_path = pca_dir / f"pca_summary_{timing}.json"
    if not summary_path.exists():
        return None

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)

        diag = s.get("pca_diagnostics", {})
        avg = diag.get("avg_variance_explained", None)
        if avg is None:
            return None

        return 100.0 * float(avg)
    except Exception:
        return None

# =============================================================================
# BUILD ONE SINGLE ROBUSTNESS TABLE (predictive + contemporaneous)
# =============================================================================

def build_one_table(records, strategies_order):
    """
    records: list of dict with keys:
      W,K,timing,strategy,alpha_ann,alpha_tstat,alpha_pval,r2adj,nobs,expl_var_pct

    Return a wide dataframe:
      rows: (W,K)
      columns per strategy/timing:
        alpha_est, alpha_t (separate) + r2
      plus:
        ExplVar (single per (W,K))
    """
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame()

    df.sort_values(["W", "K", "timing", "strategy"], inplace=True)

    rows = []
    for (W, K), g in df.groupby(["W", "K"], sort=True):
        row = {"W": int(W), "K": int(K)}

        expl_vals = []
        if "expl_var_pct" in g.columns:
            expl_vals = g["expl_var_pct"].dropna().unique().tolist()
        row["ExplVar"] = f"{float(expl_vals[0]):.1f}" if len(expl_vals) > 0 else ""

        for s in strategies_order:
            for timing in ["predictive", "contemporaneous"]:
                gg = g[(g["strategy"] == s) & (g["timing"] == timing)]

                key_a = f"{s}_{timing}_alpha"
                key_t = f"{s}_{timing}_tstat"
                key_r = f"{s}_{timing}_r2"

                if gg.empty:
                    row[key_a] = ""
                    row[key_t] = ""
                    row[key_r] = ""
                else:
                    r = gg.iloc[0]
                    st = sig_stars(float(r["alpha_pval"]))
                    row[key_a] = f"{float(r['alpha_ann']):.2f}{st}"
                    row[key_t] = f"({float(r['alpha_tstat']):.2f})"
                    row[key_r] = f"{float(r['r2adj']):.3f}"

        rows.append(row)

    return pd.DataFrame(rows)

def write_latex_one_table(wide: pd.DataFrame, strategies_order, out_path: Path):
    """
    One compact table (Duarte-style t-stat row):
      W K | for each strategy: [Pred alpha, Pred R2, Cont alpha, Cont R2] | Expl. Var. (%)
      Next row: t-stats under alpha columns only, blanks elsewhere.

    IMPORTANT (Beamer): use tabular* + extracolsep fill to FORCE width = \\textwidth
    (no resizebox -> keep font readable).
    """
    if wide.empty:
        raise RuntimeError("No robustness data collected. (Likely spanning regressions did not run.)")

    n_strat = len(strategies_order)
    # columns: W,K + 4 per strategy + 1 explained variance
    colspec = "cc" + ("cccc" * n_strat) + "c"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("% ============================================================================\n")
        f.write("% PCA ROBUSTNESS — Alpha and R^2 adj across (W,K) — predictive vs contemporaneous\n")
        f.write("% ============================================================================\n\n")

        f.write("\\begin{threeparttable}\n")
        f.write("\\setlength{\\tabcolsep}{1.1pt}\n")
        f.write("\\renewcommand{\\arraystretch}{1.02}\n")
        f.write("\\tiny\n")
        f.write("\\centering\n")

        # ---- FULL-WIDTH TABLE (NO RESIZEBOX): tabular* stretches to \textwidth ----
        line = f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}@{{\\extracolsep{{\\fill}}}}}}"
        f.write(line + "\n")

        #f.write(f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}}}\n")
        f.write("\\toprule\n")

        total_cols = 2 + 4 * n_strat + 1
        f.write(f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{Robustness (W,K): Alpha (t) and Adjusted $R^2$ — Predictive vs Contemporaneous}}}} \\\\\n")
        f.write("\\addlinespace\n")

        # Header row 1: strategy blocks
        f.write("W & K")
        for s in strategies_order:
            f.write(f" & \\multicolumn{{4}}{{c}}{{\\textit{{{strategy_label(s)}}}}}")
        f.write(" & Expl. Var. (\\%)")
        f.write(" \\\\\n")

        # Header row 2: sub-blocks
        f.write(" & ")
        for _ in strategies_order:
            f.write(" & \\multicolumn{2}{c}{Pred} & \\multicolumn{2}{c}{Cont}")
        f.write(" & ")
        f.write(" \\\\\n")

        # Header row 3: alpha/r2 labels
        f.write(" & ")
        for _ in strategies_order:
            f.write(" & $\\alpha$ & $R^2_{adj}$ & $\\alpha$ & $R^2_{adj}$")
        f.write(" & ")
        f.write(" \\\\\n")

        f.write("\\midrule\n")

        # body: Duarte-style (estimates row + t-stat row)
        for _, r in wide.iterrows():
            # Row A: estimates
            f.write(f"{int(r['W'])} & {int(r['K'])}")
            for s in strategies_order:
                f.write(f" & {r.get(f'{s}_predictive_alpha','')}")
                f.write(f" & {r.get(f'{s}_predictive_r2','')}")
                f.write(f" & {r.get(f'{s}_contemporaneous_alpha','')}")
                f.write(f" & {r.get(f'{s}_contemporaneous_r2','')}")
            f.write(f" & {r.get('ExplVar','')}")
            f.write(" \\\\\n")

            # Row B: t-stats under alpha only
            f.write(" & ")
            for s in strategies_order:
                f.write(f" & {r.get(f'{s}_predictive_tstat','')}")
                f.write(" & ")
                f.write(f" & {r.get(f'{s}_contemporaneous_tstat','')}")
                f.write(" & ")
            f.write(" & ")
            f.write(" \\\\\n")

            f.write("\\addlinespace\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular*}\n")

        # ---- FULL-WIDTH NOTES: wrap in minipage{\textwidth} ----
        f.write("\\begin{minipage}{\\textwidth}\n")
        f.write("\\begin{tablenotes}\n")
        f.write("\\tiny\n")

        f.write(
            "\\item $W$ is the rolling window length (months) used to estimate PCA factors; "
            "$K$ is the number of retained principal components. "
            "Expl. Var. (\\%) is the average cumulative variance explained by the first $K$ PCs "
            "(rolling, time-series average). "
            "Alpha is annualized (\\% p.a.). t-statistics in parentheses (Newey-West HAC). "
            "Pred = $PC_t \\to R_{t+1}$, Cont = $PC_t \\to R_t$. "
            "*** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n"
        )

        f.write("\\end{tablenotes}\n")
        f.write("\\end{minipage}\n")

        f.write("\\end{threeparttable}\n")

def write_latex_article_table(records, strategies_order, out_path: Path):
    """
    Article-format robustness table (for thesis appendix).
    Same content as beamer version but with \begin{table}[H] wrapper,
    \singlespace, superscript stars, \footnotesize notes.
    """
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No robustness data collected.")

    strategy_map = {
        'btp_italia': 'BTP Italia',
        'cds_bond_basis': 'CDS--Bond',
        'itraxx_combined': 'Index Skew',
    }

    def sig_super(pval):
        if pval < 0.01:
            return '***'
        if pval < 0.05:
            return '**'
        if pval < 0.10:
            return '*'
        return ''

    n_strat = len(strategies_order)
    colspec = "cc" + ("cccc" * n_strat) + "c"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("% " + "=" * 74 + "\n")
        f.write("% PCA ROBUSTNESS (W,K) — ARTICLE TABLE\n")
        f.write("% " + "=" * 74 + "\n\n")

        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{PCA Robustness: Alpha and $\\bar{R}^2$ Across Window Lengths and Number of Components}\n")
        f.write("\\label{tab:pca_robustness}\n")
        f.write("\\begin{threeparttable}\n")
        f.write("\\begin{singlespace}\n")
        f.write("\\small\n")
        f.write("\\setlength{\\tabcolsep}{1.5pt}\n\n")

        f.write(f"\\begin{{tabular}}{{{colspec}}}\n")
        f.write("\\toprule\n")

        # Header row 1: strategy blocks
        f.write("$W$ & $K$")
        for s in strategies_order:
            label = strategy_map.get(s, strategy_label(s))
            f.write(f" & \\multicolumn{{4}}{{c}}{{{label}}}")
        f.write(" & Expl.")
        f.write(" \\\\\n")

        # Cmidrules per strategy
        for i, s in enumerate(strategies_order):
            st = 3 + i * 4
            f.write(f"\\cmidrule(lr){{{st}-{st + 3}}}")
        f.write("\n")

        # Header row 2: Pred / Cont sub-blocks
        f.write(" & ")
        for _ in strategies_order:
            f.write(" & \\multicolumn{2}{c}{Pred} & \\multicolumn{2}{c}{Cont}")
        f.write(" & Var.(\\%)")
        f.write(" \\\\\n")

        # Header row 3: alpha / R2
        f.write(" & ")
        for _ in strategies_order:
            f.write(" & $\\alpha$ & $\\bar{R}^2$ & $\\alpha$ & $\\bar{R}^2$")
        f.write(" & ")
        f.write(" \\\\\n")

        f.write("\\midrule\n")

        # Body: group by (W,K)
        df.sort_values(["W", "K", "timing", "strategy"], inplace=True)

        for (W, K), g in df.groupby(["W", "K"], sort=True):
            # Explained variance
            expl_vals = g["expl_var_pct"].dropna().unique().tolist()
            expl_str = f"{float(expl_vals[0]):.1f}" if expl_vals else ""

            # Row A: coefficients
            f.write(f"{int(W)} & {int(K)}")
            for s in strategies_order:
                for timing in ["predictive", "contemporaneous"]:
                    gg = g[(g["strategy"] == s) & (g["timing"] == timing)]
                    if gg.empty:
                        f.write(" & -- & --")
                    else:
                        r = gg.iloc[0]
                        stars = sig_super(float(r["alpha_pval"]))
                        alpha = float(r["alpha_ann"])
                        r2 = float(r["r2adj"])
                        if stars:
                            f.write(f" & ${alpha:.2f}^{{{stars}}}$")
                        else:
                            f.write(f" & {alpha:.2f}")
                        f.write(f" & {r2:.3f}")
            f.write(f" & {expl_str}")
            f.write(" \\\\\n")

            # Row B: t-stats under alpha only
            f.write(" & ")
            for s in strategies_order:
                for timing in ["predictive", "contemporaneous"]:
                    gg = g[(g["strategy"] == s) & (g["timing"] == timing)]
                    if gg.empty:
                        f.write(" & & ")
                    else:
                        r = gg.iloc[0]
                        f.write(f" & ({float(r['alpha_tstat']):.2f})")
                        f.write(" & ")
            f.write(" & ")
            f.write(" \\\\\n")
            f.write("\\addlinespace\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n\n")

        # Notes
        f.write("\\begin{tablenotes}[para,flushleft]\n")
        f.write("\\footnotesize\n")
        f.write("\\item \\textit{Note:} ")
        f.write("$W$ is the rolling window length (months); "
                "$K$ is the number of retained principal components. "
                "Expl.\\ Var.\\ (\\%) is the average cumulative variance explained by the first $K$ PCs "
                "(time-series average across rolling windows). "
                "$\\alpha$ is annualized (\\% p.a.). "
                "$t$-statistics in parentheses (Newey--West HAC). "
                "Pred $= PC_t \\to R_{t+1}$; Cont $= PC_t \\to R_t$. "
                "$^{***}\\,p<0.01$, $^{**}\\,p<0.05$, $^{*}\\,p<0.10$.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{singlespace}\n")
        f.write("\\end{threeparttable}\n")
        f.write("\\end{table}\n")

# =============================================================================
# MAIN
# =============================================================================

def main():
    W_list = [24, 36, 48]
    K_list = [8, 11, 14]
    timings = ["predictive", "contemporaneous"]

    strategies_order = list(pca_config.STRATEGIES.keys())

    # base start date from config (string)
    base_start = getattr(pca_config, "PCA_START_DATE", "2008-01-31")

    records = []

    for W in W_list:
        # shift start date: W=24 -> +0, W=36 -> +12, W=48 -> +24
        start_shift = int(W - 24)
        shifted_start = add_months(base_start, start_shift)

        for K in K_list:
            run_dir = ROBUST_ROOT / f"W{W}_K{K}"
            run_dir.mkdir(parents=True, exist_ok=True)

            # mutate config for this run
            pca_config.RESULTS_DIR = run_dir
            pca_config.PCA_WINDOW_LENGTH = W
            pca_config.PCA_N_COMPONENTS = K
            pca_config.PCA_START_DATE = shifted_start  # IMPORTANT

            print("\n" + "=" * 80)
            print(f"RUN: W={W}, K={K}, START={shifted_start}  ->  {run_dir}")
            print("=" * 80)

            # 1) preprocessing ONCE (creates factors + y_returns for each strategy)
            sync_globals(mod01, pca_config)
            print("\n--- Running 01_pca_preprocessing.py ---")
            mod01.main()

            expl_var_pct = None

            # 2) rolling + spanning for both timings
            for timing in timings:
                pca_config.PCA_TIMING = timing
                sync_globals(mod02, pca_config)

                print(f"\n--- Running 02_pca_rolling.py  (timing={timing}) ---")
                mod02.main()

                # read explained variance once (same across timings for given W,K)
                if expl_var_pct is None:
                    expl_var_pct = read_explained_variance_pct(pca_config, timing)

                # collect results
                res = read_spanning_results(pca_config, timing)
                if not res:
                    print(f"⚠️ No spanning results found for W={W},K={K},timing={timing} (missing JSONs).")
                    continue

                for s, r in res.items():
                    records.append({
                        "W": W,
                        "K": K,
                        "timing": timing,
                        "strategy": s,
                        "alpha_ann": float(r["alpha"]) * 12.0,
                        "alpha_tstat": float(r["alpha_tstat"]),
                        "alpha_pval": float(r["alpha_pvalue"]),
                        "r2adj": float(r["r_squared_adj"]),
                        "nobs": int(r["n_obs"]),
                        "expl_var_pct": expl_var_pct,
                    })

    # restore base results dir
    pca_config.RESULTS_DIR = BASE_RESULTS_DIR
    pca_config.PCA_START_DATE = base_start

    # aggregated outputs
    robust_tables_dir = ROBUST_ROOT / "tables"
    robust_tables_dir.mkdir(parents=True, exist_ok=True)

    wide = build_one_table(records, strategies_order)



    tex_path = robust_tables_dir / "PCA_robustness_alpha_R2.tex"
    write_latex_one_table(wide, strategies_order, tex_path)

    # Article version (thesis appendix)
    tex_article_path = robust_tables_dir / "PCA_robustness_alpha_R2_article.tex"
    write_latex_article_table(records, strategies_order, tex_article_path)

    print("\n[OK] Robustness outputs:")
    print(f" - {tex_path}")
    print(f" - {tex_article_path}  ← PAPER & SKELETON")
    print("\nDONE ✅")

if __name__ == "__main__":
    main()

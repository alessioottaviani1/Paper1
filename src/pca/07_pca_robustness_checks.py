"""
================================================================================
04_pca_robustness_checks.py - PCA spanning robustness across K
================================================================================
Full-sample PCA, contemporaneous spanning. For each K in K_LIST, re-runs
01 (preprocessing) and 02 (estimation + spanning) at the fixed PCA_START_DATE,
then writes the appendix robustness table (alpha and adjusted R^2 across K).

With full-sample PCA there is NO rolling window, so the only robustness
dimension is the number of components K (baseline = config value). 01 is
re-run per K so each run dir is self-contained; the panel itself is K-invariant.

Outputs:
- results/robustness/tables/PCA_robustness_alpha_R2_article.tex   (paper appendix)
- results/robustness/tables/PCA_robustness_alpha_R2.csv

Author: Alessio Ottaviani
================================================================================
"""

import json
import importlib.util
from pathlib import Path
import pandas as pd

# =============================================================================
# CONFIG + MODULES
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

config_paths = [
    PROJECT_ROOT / "src" / "pca" / "00_pca_config.py",
]
pca_config = None
for config_path in config_paths:
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("pca_config", config_path)
        pca_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pca_config)
        break
if pca_config is None:
    raise FileNotFoundError("PCA config file not found!")

BASE_RESULTS_DIR = pca_config.RESULTS_DIR
ROBUST_ROOT = BASE_RESULTS_DIR / "robustness"
ROBUST_ROOT.mkdir(parents=True, exist_ok=True)

PCA_DIR = PROJECT_ROOT / "src" / "pca"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod01 = load_module("pca_preprocessing_01", PCA_DIR / "01_pca_preprocessing.py")
mod02 = load_module("pca_estimation_02", PCA_DIR / "02_pca_estimation.py")


def sync_globals(mod, cfg):
    """Force a module to use the current (runtime-mutated) config values."""
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
# HELPERS
# =============================================================================
def sig_stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def strategy_label(name):
    return name.replace("_", " ").title()


def read_spanning_results(cfg, timing):
    out = {}
    for s in cfg.STRATEGIES.keys():
        jpath = cfg.get_strategy_pca_dir(s) / f"spanning_regression_results_{timing}.json"
        if jpath.exists():
            with open(jpath, "r", encoding="utf-8") as f:
                out[s] = json.load(f)
    return out


def read_explained_variance_pct(cfg, timing):
    try:
        pca_dir = cfg.get_pca_output_dir()
    except Exception:
        pca_dir = cfg.RESULTS_DIR / "pca"
    p = pca_dir / f"pca_summary_{timing}.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            s = json.load(f)
        avg = s.get("pca_diagnostics", {}).get("avg_variance_explained", None)
        return None if avg is None else 100.0 * float(avg)
    except Exception:
        return None


# =============================================================================
# LATEX TABLE (article float, K-only, contemporaneous)
# =============================================================================
def write_article_table(records, strategies, out_path):
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No robustness data collected (spanning regressions did not run).")

    Ks = sorted(df["K"].unique())
    colspec = "c" + ("cc" * len(strategies)) + "c"  # K | (alpha, R2) per strategy | ExplVar

    L = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{PCA Spanning Robustness across the Number of Components}",
        r"\label{tab:pca_robustness}",
        r"\begin{minipage}{\textwidth}",
        r"{\footnotesize\noindent Full-sample contemporaneous spanning "
        r"$r_{i,t}=\alpha_i+\boldsymbol{\beta}_i'\mathbf{PC}_t+\varepsilon_{i,t}$ "
        r"for $K$ principal components (baseline $K=8$). Newey--West HAC $t$ in "
        r"parentheses. Expl.\ Var.\ is the cumulative variance captured by the $K$ "
        r"components. $^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.}",
        r"\end{minipage}",
        r"\par\vspace{6pt}",
        r"\begin{singlespace}",
        r"\begin{tabular}{" + colspec + "}",
        r"\toprule",
    ]

    head = r"$K$ " + "".join(
        r"& \multicolumn{2}{c}{" + strategy_label(s) + "} " for s in strategies
    ) + r"& Expl.\ Var. \\"
    sub = " " + r"& $\alpha$ (\% p.a.) & $\bar{R}^2$ " * len(strategies) + r"& (\%) \\"
    L += [head, sub, r"\midrule"]

    for K in Ks:
        g = df[df["K"] == K]
        expl = g["expl"].dropna()
        expl_s = f"{float(expl.iloc[0]):.1f}" if len(expl) else ""

        row = f"{int(K)} "
        for s in strategies:
            r = g[g["strategy"] == s]
            if r.empty:
                row += r"& -- & -- "
            else:
                r = r.iloc[0]
                row += f"& {r['alpha_ann']:.2f}{sig_stars(r['p'])} & {r['r2adj']:.3f} "
        row += f"& {expl_s} " + r"\\"
        L.append(row)

        trow = " "
        for s in strategies:
            r = g[g["strategy"] == s]
            trow += (f"& ({r.iloc[0]['t']:.2f}) & " if not r.empty else r"& & ")
        trow += r"& \\"
        L += [trow, r"\addlinespace"]

    L += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{singlespace}",
        r"\end{table}",
    ]
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================
K_LIST = [5, 8, 11, 13]      # baseline 8 = Bai-Ng ICp2; 13 = ICp1 (upper bound)
TIMING = "contemporaneous"


def main():
    strategies = list(pca_config.STRATEGIES.keys())
    base_start = getattr(pca_config, "PCA_START_DATE", "2008-01-31")
    records = []

    for K in K_LIST:
        run_dir = ROBUST_ROOT / f"K{K}"
        run_dir.mkdir(parents=True, exist_ok=True)
        pca_config.RESULTS_DIR = run_dir
        pca_config.PCA_N_COMPONENTS = K
        pca_config.PCA_START_DATE = base_start
        pca_config.PCA_TIMING = TIMING

        print(f"\n== robustness K={K} -> {run_dir}")
        sync_globals(mod01, pca_config); mod01.main()
        sync_globals(mod02, pca_config); mod02.main()

        expl = read_explained_variance_pct(pca_config, TIMING)
        res = read_spanning_results(pca_config, TIMING)
        if not res:
            print(f"   (no spanning results for K={K})")
            continue
        for s, r in res.items():
            records.append({
                "K": K, "strategy": s,
                "alpha_ann": float(r["alpha"]) * 12.0,
                "t": float(r["alpha_tstat"]),
                "p": float(r["alpha_pvalue"]),
                "r2adj": float(r["r_squared_adj"]),
                "nobs": int(r["n_obs"]),
                "expl": expl,
            })

    pca_config.RESULTS_DIR = BASE_RESULTS_DIR
    pca_config.PCA_START_DATE = base_start

    tdir = ROBUST_ROOT / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(tdir / "PCA_robustness_alpha_R2.csv", index=False)
    write_article_table(records, strategies,
                        PROJECT_ROOT / "results" / "tables" / "PCA_robustness_alpha_R2_article.tex")
    print(f"\n== Done. Table -> {tdir / 'PCA_robustness_alpha_R2_article.tex'}")


if __name__ == "__main__":
    main()
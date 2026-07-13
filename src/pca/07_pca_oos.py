# -*- coding: utf-8 -*-
"""
06_pca_oos.py
=============
Out-of-sample (past-only-beta) alpha for the PCA spanning model -- the PCA analogue
of the benchmark Panel D (07_oos_alpha.py).

WHY THIS EXISTS
---------------
The in-sample PCA spanning test (02_pca_estimation.py) fits BOTH the principal
components (loadings) AND the spanning beta on the FULL sample, so the reported
in-sample alpha is not out-of-sample on the hedge -- the implementability critique
that Panel D answers for the benchmark. This module re-estimates BOTH the PCs and
the spanning coefficients recursively, past-only, so the hedge at each date uses no
future information.

ROTATION-INVARIANCE (why we re-fit per window instead of reusing in-sample PCs)
-------------------------------------------------------------------------------
Principal components are identified only up to sign / label-order / rotation within
the top-K subspace, and each past window yields its own basis. We use a SELF-CONTAINED
window: at each t we extract the K PCs AND fit beta in the SAME past window, so beta
and the score it multiplies live in the same basis. Any orthogonal rotation R of that
basis sends PC -> R'PC and beta -> R'beta, and the hedge (R'beta)'(R'PC) = beta'PC is
invariant; the realized residual e_t = r_t - beta'PC_t is therefore well defined
regardless of the eigenvector sign/label convention (Giglio & Xiu 2021). PCs are
treated as tradable factors in the asymptotic-PCA sense (Connor & Korajczyk 1986, 1988).

WINDOW MODES (both panels are produced in a single run):
    expanding  ->  base = [start, t-1]   Panel D primary (comparable to the
                                          benchmark expanding OOS)
    rolling    ->  base = [t-L, t-1]      Appendix A.7 robustness (L = OOS_ROLL_WINDOW)

Both the PC extraction and the beta fit use the SAME window (self-contained), so the
rotation-invariance above holds in either mode.

Timing: the OOS hedge is CONTEMPORANEOUS (r_t hedged by PC_t with a past-only beta),
matching PCA_TIMING="contemporaneous" and the benchmark Panel D. (The "predictive"
PC timing is a return-predictability exercise, a different object, not a hedge-OOS.)

Run:  python src/pca/06_pca_oos.py
"""
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

OOS_MIN_TRAIN   = 60          # burn-in / minimum window before the first evaluation
OOS_ROLL_WINDOW = 60          # rolling-window length L (Appendix A.7 robustness)

# ── load PCA config (00_pca_config.py) ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_cfg_path = PROJECT_ROOT / "src" / "pca" / "00_pca_config.py"
_spec = importlib.util.spec_from_file_location("pca_config", _cfg_path)
pca_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pca_config)

FACTORS_PATH         = pca_config.FACTORS_PATH
FACTORS_END_DATE     = pca_config.FACTORS_END_DATE
STRATEGIES           = pca_config.STRATEGIES
N_PC                 = pca_config.PCA_N_COMPONENTS          # 8
get_strategy_pca_dir = pca_config.get_strategy_pca_dir
get_pca_output_dir   = pca_config.get_pca_output_dir

# ── HAC lag rule from the benchmark OOS engine, so inference is IDENTICAL ────
try:
    _oos_path = PROJECT_ROOT / "src" / "factor_models" / "oos.py"
    _ospec = importlib.util.spec_from_file_location("oos", _oos_path)
    _oos = importlib.util.module_from_spec(_ospec); _ospec.loader.exec_module(_oos)
    _nw_lags = _oos._nw_lags
except Exception:
    def _nw_lags(n):
        return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))   # Newey-West (1994)

PERIODS_PER_YEAR = 12


def _stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# ── data loaders ────────────────────────────────────────────────────────────
def load_factor_panel():
    F = pd.read_parquet(FACTORS_PATH)
    F = F[F.index <= pd.Timestamp(FACTORS_END_DATE)]
    return F.sort_index()


def load_returns(strategy_name):
    """Strategy monthly returns (%). Prefer the PCA pipeline's series for
    consistency with the in-sample spanning; fall back to the daily index."""
    p = get_strategy_pca_dir(strategy_name) / "y_returns_pca.parquet"
    if p.exists():
        return pd.read_parquet(p)["Strategy_Return"].dropna().sort_index()
    daily = pd.read_csv(STRATEGIES[strategy_name], index_col=0,
                        parse_dates=True)["index_return"].dropna()
    monthly = daily.resample("ME").apply(
        lambda x: ((1 + x / 100).prod() - 1) * 100 if len(x) > 0 else np.nan)
    return monthly.dropna().sort_index()


# ── core: past-only-beta OOS residuals, self-contained windows ──────────────
def pca_oos_residuals(returns, F, window_mode, n_pc=N_PC,
                      min_train=OOS_MIN_TRAIN, roll_window=OOS_ROLL_WINDOW):
    common = returns.index.intersection(F.index)
    r = returns.loc[common].sort_index()
    X = F.loc[common].sort_index()
    dates = r.index
    T = len(dates)

    e = {}
    n_fac = []
    for i in range(T):
        t = dates[i]
        lo = (max(0, i - roll_window) if window_mode == "rolling" else 0)
        if i - lo < min_train:                       # not enough history yet
            continue
        win = dates[lo:i]                            # training dates [lo, i-1] (excludes t)
        Xw, rw, xt = X.loc[win], r.loc[win], X.loc[[t]]

        # drop factors not fully available in this window or at t
        ok = Xw.columns[~(Xw.isna().any() | xt.isna().any().values)]
        Xw, xt = Xw[ok], xt[ok]
        if Xw.shape[1] < n_pc:
            continue

        # in-window standardisation, reuse window stats for t
        mu, sd = Xw.mean(), Xw.std().replace(0, np.nan)
        Zw, Zt = (Xw - mu) / sd, (xt - mu) / sd
        ok2 = Zw.columns[~(Zw.isna().any() | Zt.isna().any().values)]
        Zw, Zt = Zw[ok2], Zt[ok2]
        if Zw.shape[1] < n_pc:
            continue

        # PCA fit in THIS window's basis; project window + current month in it
        pca = PCA(n_components=n_pc).fit(Zw.values)
        PCw = pca.transform(Zw.values)               # (len(win), n_pc)
        PCt = pca.transform(Zt.values)[0]            # (n_pc,)

        # spanning beta on (r_window, PCw); slopes only -> in-window intercept dropped
        beta = sm.OLS(rw.values, sm.add_constant(PCw)).fit().params
        e[t] = float(r.loc[t] - PCt @ beta[1:])
        n_fac.append(Zw.shape[1])

    return pd.Series(e).sort_index(), (np.mean(n_fac) if n_fac else float("nan"))


def hac_alpha(e):
    lags = _nw_lags(len(e))
    res = sm.OLS(e.values, np.ones(len(e))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    ir = e.mean() / e.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)   # info ratio (annualised)
    return (float(res.params[0]) * PERIODS_PER_YEAR,   # annualised alpha (%)
            float(res.tvalues[0]), float(res.pvalues[0]), lags, float(ir))


def main():
    F = load_factor_panel()
    outdir = get_pca_output_dir()
    all_rows = {}                       # mode -> [(sn, a_ann, t, p, n, avg_fac), ...]

    for mode in ("expanding", "rolling"):
        L = OOS_ROLL_WINDOW if mode == "rolling" else None
        print("=" * 92)
        print(f"PCA OUT-OF-SAMPLE ALPHA  (past-only beta, self-contained windows)   "
              f"window = '{mode.upper()}'{'' if L is None else f'  (L = {L})'}")
        print("=" * 92)
        print(f"  K = {N_PC} PCs | min_train = {OOS_MIN_TRAIN}"
              f"{' | L = ' + str(L) if L else ''}"
              f" | contemporaneous hedge | HAC = Newey-West(1994)\n")
        out, rows = {}, []
        for sn in STRATEGIES:
            r = load_returns(sn)
            e, avg_fac = pca_oos_residuals(r, F, mode)
            if len(e) < OOS_MIN_TRAIN // 2:
                print(f"  {sn:<18} insufficient OOS observations ({len(e)})")
                continue
            a_ann, t, p, lags, ir = hac_alpha(e)
            out[sn] = {"window": mode, "n_oos": int(len(e)),
                       "avg_factors_in_window": round(float(avg_fac), 2),
                       "alpha_oos_annualized": round(a_ann, 4),
                       "t_stat": round(t, 4), "p_value": round(p, 4),
                       "information_ratio": round(float(ir), 4),
                       "hac_lags": int(lags), "n_pc": int(N_PC),
                       "min_train": OOS_MIN_TRAIN, "roll_window": L}
            rows.append((sn, a_ann, t, p, ir, len(e), avg_fac))
            print(f"  {sn:<18} alpha_OOS = {a_ann:+6.2f}% ann   t = {t:5.2f}   "
                  f"p = {p:.4f} {_stars(p):<3}   IR = {ir:+.2f}   [n={len(e)}]")

        tag = "expanding (headline)" if mode == "expanding" else "rolling-60 (robustness)"
        print(f"\n{'-'*60}\n  PCA OOS — {tag} — alpha annualizzato, t tra ()\n{'-'*60}")
        for sn, a, t, p, ir, n, _ in rows:
            print(f"   {sn:<18} {a:+6.2f}{_stars(p):<3}  ({t:.2f})   IR={ir:+.2f}")

        (outdir / f"pca_oos_alpha_{mode}.json").write_text(json.dumps(out, indent=2, default=float))
        print(f"\nSaved: {outdir / f'pca_oos_alpha_{mode}.json'}\n")
        all_rows[mode] = rows

    # ---- Joint LaTeX table: OOS alpha, PCA (Panel A) + Best-Subset (Panel B) ----
    TEX_STRAT = {"btp_italia": "BTP Italia", "cds_bond_basis": "CDS--Bond Basis",
                 "itraxx_combined": "iTraxx Combined"}
    aen_dir = PROJECT_ROOT / "results" / "machine_learning"
    aen = {}
    for mode in ("expanding", "rolling"):
        fp = aen_dir / f"aen_oos_alpha_{mode}.json"
        aen[mode] = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else {}
        if not aen[mode]:
            print(f"  ⚠ {fp} not found — Best-Subset panel will show '--' (run 06_aen_oos.py first)")

    def _pca_cell(sn, mode):
        d = {r[0]: r for r in all_rows.get(mode, [])}.get(sn)
        if d is None:
            return " & -- & -- & --", None
        _, a, t, p, ir, n, _avg = d
        s = _stars(p)
        a_tex = f"${a:+.2f}^{{{s}}}$" if s else f"{a:+.2f}"
        return f" & {a_tex} & {t:.2f} & {ir:+.2f}", n

    def _aen_cell(sn, mode):
        d = aen.get(mode, {}).get(sn.lower())
        if not d:
            return " & -- & -- & --", None
        s = _stars(d["p"])
        a_tex = f"${d['alpha_oos_ann']:+.2f}^{{{s}}}$" if s else f"{d['alpha_oos_ann']:+.2f}"
        return f" & {a_tex} & {d['t']:.2f} & {d['ir']:+.2f}", d.get("n_oos")

    L = [r"\begin{table}[H]",
         r"\centering",
         r"\singlespacing",
         r"\caption{Out-of-Sample Alpha: PCA and Best-Subset}",
         r"\label{tab:pca_oos}",
         r"\begin{minipage}{\textwidth}",
         r"{\footnotesize\noindent Out-of-sample alpha of the PCA (Panel A) and best-subset (Panel B) "
         r"hedges, in the recursive design of \citet{welch2008comprehensive}. At each month $t$ "
         r"the hedge is estimated on a past-only window---expanding with a 60-month burn-in "
         r"(headline) or rolling 60-month, which relaxes the constant-loadings assumption---and "
         r"the realized abnormal return is "
         r"$e_{i,t} = r_{i,t} - \hat{\boldsymbol{\beta}}_{i,t-1}'\,\mathbf{f}_t$; the in-window "
         r"intercept (the alpha under measurement) is excluded from the hedge. In Panel A the "
         r"principal components are re-estimated within each window, so the design is invariant "
         r"to the sign and rotation indeterminacy of PCA; in Panel B the best-subset "
         r"factor set is frozen and only the loadings are re-estimated. $\alpha$ is the "
         r"annualized mean of $e_{i,t}$ (\% p.a.), tested with Newey--West HAC standard errors; "
         r"IR is the annualized information ratio $\bar{e}_i/\sigma(e_i)\times\sqrt{12}$; $N$ is "
         r"the number of out-of-sample months. "
         r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.}",
         r"\end{minipage}",
         r"\par\vspace{6pt}",
         r"\begin{singlespace}",
         r"\small",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l r rrr rrr}",
         r"\toprule",
         r" & & \multicolumn{3}{c}{Expanding (60-month burn-in)} & \multicolumn{3}{c}{Rolling 60-month} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Strategy & $N$ & $\alpha$ & $t$ & IR & $\alpha$ & $t$ & IR \\",
         r"\midrule"]
    for panel_name, cell_fn in (("Panel A: PCA", _pca_cell), ("Panel B: Best-Subset", _aen_cell)):
        L.append(rf"\multicolumn{{8}}{{l}}{{\textbf{{{panel_name}}}}} \\")
        L.append(r"\addlinespace")
        for sn in STRATEGIES:
            ce, n_e = cell_fn(sn, "expanding")
            cr, n_r = cell_fn(sn, "rolling")
            ns = {v for v in (n_e, n_r) if v}
            n_show = next(iter(ns)) if len(ns) == 1 else ("/".join(str(v) for v in (n_e, n_r) if v) or "--")
            L.append(rf"\textit{{{TEX_STRAT.get(sn, sn.replace('_', ' '))}}} & {n_show}{ce}{cr} \\")
        L.append(r"\addlinespace")
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{singlespace}", r"\end{table}"]
    tables_dir = PROJECT_ROOT / "results" / "tables"   # = \tablepath
    tables_dir.mkdir(parents=True, exist_ok=True)
    (tables_dir / "PCA_AEN_oos_alpha.tex").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Saved: {tables_dir / 'PCA_AEN_oos_alpha.tex'}")
    print("\nNotes:")
    print("  - Both panels are produced in one run (expanding = Panel D, rolling = A.7).")
    print("  - Each evaluation date is self-contained: PCs and beta share the same")
    print("    window basis, so the residual is invariant to PC sign/label/rotation.")
    print("  - Beta is slopes-only (in-window intercept dropped), as in oos.py / Panel D.")


if __name__ == "__main__":
    main()

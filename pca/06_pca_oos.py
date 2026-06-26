# -*- coding: utf-8 -*-
"""
06_pca_oos.py
=============
Out-of-sample (past-only-beta) alpha for the PCA spanning model -- the PCA analogue
of the benchmark Panel D (07_oos_alpha.py).

WHY THIS EXISTS
---------------
The rolling PCA already estimates the LOADINGS (eigenvectors, L1) past-only:
the score at t projects x_t onto eigenvectors fit on [t-W, t-1] (excludes t).
But the SPANNING beta (L2) in 02_pca_rolling.run_spanning_regression is fit on the
FULL sample, so the reported PCA alpha is NOT out-of-sample on the hedge -- exactly
the implementability critique that Panel D answers for the benchmark, and exactly
the out-of-sample exercise Ludvigson & Ng (2009) themselves run (factors AND
coefficients re-estimated recursively). This module makes the beta past-only too.

THE PCA-SPECIFIC SUBTLETY (and why we cannot just reuse the saved scores)
-------------------------------------------------------------------------
Rolling-PCA eigenvectors are identified only up to sign / label-order / rotation
within the top-K subspace. The saved score series is stitched from many windows,
each in its own (possibly flipped) basis, so applying a single past beta to that
stitched series is incoherent. We dissolve the problem with a SELF-CONTAINED window:
at each t we extract the K PCs AND fit beta in the SAME past window, so beta and the
score it multiplies live in the same basis. Any orthogonal rotation R of that basis
sends PC -> R'PC and beta -> R'beta, and the hedge (R'beta)'(R'PC) = beta'PC is
invariant. The realized residual e_t = r_t - beta'PC_t is therefore well defined
regardless of the eigenvector sign/label convention.

WINDOW MODE -- switch with ONE string below:
    OOS_WINDOW = "expanding"   ->  base = [start, t-1]      (recommended; comparable
                                                             to the benchmark expanding OOS)
    OOS_WINDOW = "rolling"     ->  base = [t-L, t-1]         (L = OOS_ROLL_WINDOW)

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

# ────────────────────────────────────────────────────────────────────────────
#  ⭐  SINGLE-STRING SWITCH  ⭐
# ────────────────────────────────────────────────────────────────────────────
OOS_WINDOW = "expanding"      # "expanding"  or  "rolling"
# ────────────────────────────────────────────────────────────────────────────

OOS_MIN_TRAIN   = 60          # burn-in / minimum window before the first evaluation
OOS_ROLL_WINDOW = 60          # window length L when OOS_WINDOW == "rolling"

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
COND_TOL = 0.0


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
    n_fac, n_skip = [], 0
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
            n_skip += 1
            continue

        # in-window standardisation (Ludvigson & Ng), reuse window stats for t
        mu, sd = Xw.mean(), Xw.std().replace(0, np.nan)
        Zw, Zt = (Xw - mu) / sd, (xt - mu) / sd
        ok2 = Zw.columns[~(Zw.isna().any() | Zt.isna().any().values)]
        Zw, Zt = Zw[ok2], Zt[ok2]
        if Zw.shape[1] < n_pc:
            n_skip += 1
            continue

        # PCA fit in THIS window's basis; project window + current month in it
        pca = PCA(n_components=n_pc).fit(Zw.values)
        PCw = pca.transform(Zw.values)               # (len(win), n_pc)
        PCt = pca.transform(Zt.values)[0]            # (n_pc,)

        # spanning beta on (r_window, PCw); slopes only -> in-window intercept dropped
        beta = sm.OLS(rw.values, sm.add_constant(PCw)).fit().params
        e[t] = float(r.loc[t] - PCt @ beta[1:])
        n_fac.append(Zw.shape[1])

    return pd.Series(e).sort_index(), (np.mean(n_fac) if n_fac else float("nan")), n_skip


def hac_alpha(e):
    lags = _nw_lags(len(e))
    res = sm.OLS(e.values, np.ones(len(e))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return (float(res.params[0]) * PERIODS_PER_YEAR,   # annualised alpha (%)
            float(res.tvalues[0]), float(res.pvalues[0]), lags)


def main():
    if OOS_WINDOW not in ("expanding", "rolling"):
        raise ValueError("OOS_WINDOW must be 'expanding' or 'rolling'")

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
            e, avg_fac, n_skip = pca_oos_residuals(r, F, mode)
            if len(e) < OOS_MIN_TRAIN // 2:
                print(f"  {sn:<18} insufficient OOS observations ({len(e)})")
                continue
            a_ann, t, p, lags = hac_alpha(e)
            out[sn] = {"window": mode, "n_oos": int(len(e)),
                       "avg_factors_in_window": round(float(avg_fac), 2),
                       "alpha_oos_annualized": round(a_ann, 4),
                       "t_stat": round(t, 4), "p_value": round(p, 4),
                       "hac_lags": int(lags), "n_pc": int(N_PC),
                       "min_train": OOS_MIN_TRAIN, "roll_window": L}
            rows.append((sn, a_ann, t, p, len(e), avg_fac))
            print(f"  {sn:<18} alpha_OOS = {a_ann:+6.2f}% ann   t = {t:5.2f}   "
                  f"p = {p:.4f} {_stars(p):<3}   [n={len(e)}, ~{avg_fac:.0f} factors/window]")

        tag = "Panel D (primario)" if mode == "expanding" else "Appendice A.7 (robustezza)"
        print(f"\n{'-'*60}\n  {tag} — PCA {mode} — alpha annualizzato, t tra ()\n{'-'*60}")
        for sn, a, t, p, n, _ in rows:
            print(f"   {sn:<18} {a:+6.2f}{_stars(p):<3}  ({t:.2f})")

        (outdir / f"pca_oos_alpha_{mode}.json").write_text(json.dumps(out, indent=2, default=float))
        print(f"\nSaved: {outdir / f'pca_oos_alpha_{mode}.json'}\n")
        all_rows[mode] = rows

    # ---- Tabella LaTeX: alpha OOS, expanding (Panel D) + rolling-60 (A.7) ----
    def _fmt(a, t, p): return f"{a:+.2f}{_stars(p)} ({t:.2f})"
    exp = {r[0]: r for r in all_rows.get("expanding", [])}
    rol = {r[0]: r for r in all_rows.get("rolling", [])}
    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             r"Strategy & Expanding (Panel D) & Rolling-60 (App.~A.7) \\", r"\midrule"]
    for sn in STRATEGIES:
        if sn not in exp and sn not in rol:
            continue
        ce = _fmt(exp[sn][1], exp[sn][2], exp[sn][3]) if sn in exp else "--"
        cr = _fmt(rol[sn][1], rol[sn][2], rol[sn][3]) if sn in rol else "--"
        lines.append(f"{sn.replace('_', ' ')} & {ce} & {cr} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tables_dir = PROJECT_ROOT / "results" / "tables"   # = \tablepath
    tables_dir.mkdir(parents=True, exist_ok=True)
    float_tex = [r"\begin{table}[htbp]", r"\centering",
                 r"\caption{Out-of-Sample PCA Spanning Alpha}",
                 r"\label{tab:pca_oos}", *lines, r"\end{table}"]
    (tables_dir / "PCA_oos_alpha.tex").write_text(
        "% PCA out-of-sample alpha (annualised %, HAC Newey-West t in parentheses)\n"
        + "\n".join(float_tex) + "\n")
    print(f"Saved: {tables_dir / 'PCA_oos_alpha.tex'}")
    print("\nNotes:")
    print(f"  - Switch the window by editing ONE line: OOS_WINDOW = "
          f"'{'rolling' if OOS_WINDOW == 'expanding' else 'expanding'}'.")
    print("  - Each evaluation date is self-contained: PCs and beta share the same")
    print("    window basis, so the residual is invariant to PC sign/label/rotation.")
    print("  - Beta is slopes-only (in-window intercept dropped), as in oos.py / Panel D.")


if __name__ == "__main__":
    main()

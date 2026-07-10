# -*- coding: utf-8 -*-
"""
================================================================================
08_selector_benchmark.py — Head-to-head test of factor-selection alternatives
================================================================================

PURPOSE
-------
Runs, on the SAME preprocessed inputs as 02/02s (per strategy:
    <results>/<strategy>/aen/<suffix>/y_centered.parquet
    <results>/<strategy>/aen/<suffix>/X_standardized.parquet ),
all the selection methods worth considering for the paper, and scores them on
criteria a Journal of Finance referee actually cares about:

    n_sel        parsimony (number of selected factors)
    R2adj_IS     in-sample adjusted R^2 of post-selection OLS (standardized X)
    alpha_IS     full-sample post-selection alpha (% p.a.) + HAC t   [optimistic]
    alpha_SPLIT  HONEST alpha: select on 1st half, estimate alpha on 2nd half
                 (valid post-selection inference by sample splitting)
    Jaccard      stability: overlap of supports selected on 1st vs 2nd half
    alpha_OOS    frozen-set expanding-window OOS alpha (% p.a.), HAC t, IR
                 (same estimand as 05_aen_oos / Table IX)
    engine/time  which algorithm actually ran, runtime

METHODS
-------
  ell0        Best subset of exact size k via `abess` (Zhu et al. 2020), the
              polynomial-time solver for the l0 problem of Bertsimas, King &
              Mazumder (2016, Annals of Statistics). HARD-FAILS to forward
              selection only if abess is missing — and says so loudly.
  forward     Greedy forward selection forced to k (Hastie, Tibshirani &
              Tibshirani 2020). Diagnostic: how far is the fallback from l0?
  aen         The paper's Adaptive Elastic Net selection, LOADED from
              aen_results.json (not refit) so it is exactly the paper's set.
  relaxed     Relaxed lasso (Meinshausen 2007; recommended by HTT 2020 as the
              best all-SNR default). Two stages: lasso path -> per-support OLS,
              blend gamma in {0,.25,.5,.75,1}; (lambda, gamma) by 5-fold
              CONTIGUOUS-BLOCK CV (respects serial dependence).
  cpss        Complementary-pairs stability selection (Shah & Samworth 2013):
              B pairs of disjoint half-samples, first-q-to-enter LARS order per
              half, threshold pi from the E[V] <= 1 bound
              ( pi = 1/2 + q^2 / (2 p E[V]) , capped at 0.9 ).
  bch_pds     Post-lasso with the Belloni-Chernozhukov-Hansen (2014) plug-in
              penalty (c = 1.1, level 5%), iterated sigma. NOTE: with the
              INTERCEPT (alpha) as the target, the second selection step of
              post-double-selection is empty, so this IS the PDS control set.
              With --write-pds the support is saved as pds_results.json in the
              strategy's aen directory, which 07_tables.py picks up for the
              selector-overlap column.
  knockoffs   Model-X second-order Gaussian knockoffs (Candes et al. 2018),
              FDR = 10% — OPTIONAL, runs only if `knockpy` is installed.
              Expected to be low-powered at T ~ 160-240 with clustered designs;
              included so the paper can say it was examined.

WHAT "BETTER" MEANS
-------------------
A method improves on the current primary if, at comparable or smaller n_sel, it
delivers (i) a higher honest split-sample alpha t, (ii) a higher frozen-set OOS
IR, and (iii) a higher half-sample Jaccard. In-sample R2/alpha alone prove
nothing (selection bias) — do not rank on them.

USAGE
-----
    python 08_selector_benchmark.py                 # all strategies, k = K_PRIMARY
    python 08_selector_benchmark.py --write-pds     # also emit pds_results.json
    python 08_selector_benchmark.py --strategies itraxx_combined --k 8

Outputs: console tables + <results>/aen/<suffix>/benchmark/
         selector_benchmark_summary.csv  and  selector_benchmark.json

Institution: EDHEC Business School — PhD Thesis
"""

import argparse
import json
import time
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from sklearn.linear_model import lasso_path, lars_path

warnings.filterwarnings("ignore")

# ── Load 00_config.py exactly like 02s does ────────────────────────────────────
_CFG_PATH = Path(__file__).resolve().parent / "00_config.py"
_spec = importlib.util.spec_from_file_location("aen_config", _CFG_PATH)
aen_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aen_config)

STRATEGIES           = aen_config.STRATEGIES
get_aen_output_dir   = aen_config.get_aen_output_dir
get_strategy_aen_dir = aen_config.get_strategy_aen_dir

HAC_LAGS   = 4          # NW(1994) rule floor(4*(T/100)^(2/9)) = 4 for T in [160, 245]
K_DEFAULT  = 8          # mirror of K_PRIMARY in 02s
SEED       = 42
CV_FOLDS   = 5
CPSS_B     = 50         # complementary pairs -> 100 half-samples
CPSS_Q     = 12         # variables entering per half-sample
CPSS_EV    = 1.0        # tolerated expected false positives
GAMMAS     = (0.0, 0.25, 0.5, 0.75, 1.0)

rng = np.random.default_rng(SEED)


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════════
def load_inputs(strategy: str):
    """y_centered, X_standardized, y_raw (centered + stored mean), factor names."""
    d = get_strategy_aen_dir(strategy)
    y = pd.read_parquet(d / "y_centered.parquet").iloc[:, 0]
    X = pd.read_parquet(d / "X_standardized.parquet")
    y, X = y.align(X, join="inner", axis=0)

    y_mean = None
    for jf in sorted(d.glob("*.json")):
        try:
            blob = json.loads(jf.read_text())
        except Exception:
            continue
        if isinstance(blob, dict) and "y_mean" in blob:
            y_mean = float(blob["y_mean"]); break
    if y_mean is None:
        print(f"   ⚠️  {strategy}: y_mean not found in {d} — alphas computed on centered y "
              f"(levels off by the mean; comparisons across methods remain valid).")
        y_mean = 0.0
    return y.astype(float), X.astype(float), y_mean, list(X.columns)


def load_aen_selection(strategy: str, names):
    f = get_strategy_aen_dir(strategy) / "aen_results.json"
    if not f.exists():
        return None
    blob = json.loads(f.read_text())
    sel = blob.get("selected_factors") or blob.get("selected") or []
    idx = sorted(names.index(s) for s in sel if s in names)
    return idx if idx else None


# ══════════════════════════════════════════════════════════════════════════════
#  OLS / metrics helpers
# ══════════════════════════════════════════════════════════════════════════════
def _ols(Xz: np.ndarray, y: np.ndarray, idx):
    if not idx:
        return np.zeros(0)
    A = Xz[:, idx]
    return np.linalg.lstsq(A, y, rcond=None)[0]


def r2_adj(Xz, y, idx):
    n = len(y)
    if not idx:
        return 0.0
    beta = _ols(Xz, y, idx)
    rss = float(np.sum((y - Xz[:, idx] @ beta) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    k = len(idx)
    return 1.0 - (rss / (n - k - 1)) / (tss / (n - 1))


def alpha_hac(y_raw: np.ndarray, Xz: np.ndarray, idx, lags=HAC_LAGS):
    """Intercept (annualized, % p.a. if y is monthly %) + HAC t of y_raw on X_S."""
    X = sm.add_constant(Xz[:, idx]) if idx else np.ones((len(y_raw), 1))
    res = sm.OLS(y_raw, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return 12.0 * float(res.params[0]), float(res.tvalues[0])


def frozen_oos(y_raw: np.ndarray, Xz: np.ndarray, idx, burnin=60, lags=HAC_LAGS):
    """Frozen-set expanding-window OOS abnormal return (Table IX estimand)."""
    T = len(y_raw)
    if T <= burnin + 12:
        return np.nan, np.nan, np.nan, 0
    e = []
    for t in range(burnin, T):
        Xtr = sm.add_constant(Xz[:t, idx]) if idx else np.ones((t, 1))
        b = np.linalg.lstsq(Xtr, y_raw[:t], rcond=None)[0]
        xt = np.concatenate(([1.0], Xz[t, idx])) if idx else np.array([1.0])
        e.append(y_raw[t] - float(xt @ b))
    e = np.asarray(e)
    res = sm.OLS(e, np.ones((len(e), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    ir = float(e.mean() / e.std(ddof=1) * np.sqrt(12)) if e.std(ddof=1) > 0 else np.nan
    return 12.0 * float(e.mean()), float(res.tvalues[0]), ir, len(e)


def jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A | B) else np.nan


# ══════════════════════════════════════════════════════════════════════════════
#  Selectors  (each: (Xz, y, names, k) -> (sorted idx list, engine label))
# ══════════════════════════════════════════════════════════════════════════════
def sel_forward(Xz, y, names=None, k=K_DEFAULT):
    chosen, p = [], Xz.shape[1]
    while len(chosen) < k:
        best_rss, best_j = np.inf, -1
        for j in range(p):
            if j in chosen:
                continue
            idx = chosen + [j]
            beta = _ols(Xz, y, idx)
            rss = float(np.sum((y - Xz[:, idx] @ beta) ** 2))
            if rss < best_rss:
                best_rss, best_j = rss, j
        chosen.append(best_j)
    return sorted(chosen), "forward-forced"


def sel_ell0(Xz, y, names=None, k=K_DEFAULT):
    try:
        from abess.linear import LinearRegression
        m = LinearRegression(support_size=k, fit_intercept=False)
        m.fit(Xz, y)
        idx = sorted(np.where(np.abs(m.coef_) > 0)[0].tolist())
        if len(idx) == k:
            return idx, "abess(ell0)"
        print(f"   ⚠️  abess returned {len(idx)} != k={k}; falling back to forward.")
    except Exception as e:
        print(f"   ⚠️  abess unavailable ({type(e).__name__}) — l0 NOT verified, using forward.")
    return sel_forward(Xz, y, names, k)[0], "forward-forced (NOT l0)"


def sel_relaxed_lasso(Xz, y, names=None, k=None):
    """Relaxed lasso, (lambda, gamma) by contiguous-block CV. Support = S(lambda*)."""
    n = len(y)
    alphas, coefs, _ = lasso_path(Xz, y, n_alphas=40, eps=1e-3)
    folds = np.array_split(np.arange(n), CV_FOLDS)
    cv = np.zeros((len(alphas), len(GAMMAS)))
    for test in folds:
        train = np.setdiff1d(np.arange(n), test)
        _, c_tr, _ = lasso_path(Xz[train], y[train], alphas=alphas)
        for ai in range(len(alphas)):
            b_l = c_tr[:, ai]
            S = np.where(np.abs(b_l) > 1e-10)[0].tolist()
            b_ols = np.zeros_like(b_l)
            if S:
                b_ols[S] = _ols(Xz[train], y[train], S)
            for gi, g in enumerate(GAMMAS):
                b = g * b_l + (1 - g) * b_ols
                cv[ai, gi] += float(np.mean((y[test] - Xz[test] @ b) ** 2)) / CV_FOLDS
    ai, gi = np.unravel_index(np.argmin(cv), cv.shape)
    S = sorted(np.where(np.abs(coefs[:, ai]) > 1e-10)[0].tolist())
    return S, f"relaxed(lam#{ai},gam={GAMMAS[gi]})"


def sel_cpss(Xz, y, names=None, k=None):
    """Shah–Samworth complementary pairs; first-q LARS order per half."""
    n, p = Xz.shape
    freq = np.zeros(p)
    half = n // 2
    for _ in range(CPSS_B):
        perm = rng.permutation(n)
        for ids in (perm[:half], perm[half:2 * half]):
            _, _, cf = lars_path(Xz[ids], y[ids], method="lasso", max_iter=CPSS_Q)
            order = []
            for step in range(cf.shape[1]):
                for j in np.where(np.abs(cf[:, step]) > 1e-12)[0]:
                    if j not in order:
                        order.append(int(j))
            for j in order[:CPSS_Q]:
                freq[j] += 1.0
    freq /= (2 * CPSS_B)
    pi_thr = min(0.9, 0.5 + CPSS_Q ** 2 / (2.0 * p * CPSS_EV))
    S = sorted(np.where(freq >= pi_thr)[0].tolist())
    return S, f"cpss(q={CPSS_Q},pi={pi_thr:.2f})"


def sel_bch_pds(Xz, y, names=None, k=None, c=1.1, iters=5):
    """Post-lasso with the BCH (2014) plug-in penalty, hdm::rlasso defaults
    (gamma = 0.1/log(p v n), c = 1.1), iterated sigma-hat. Computed on the
    unit-VARIANCE rescaling of the design (the canonical normalization).
    Deliberately conservative: it controls false selections, so lean sets are
    the expected behaviour. If the set comes back empty, falls back to
    LassoCV support and says so in the engine label."""
    n, p = Xz.shape
    Xu = Xz * np.sqrt(n)                      # ||x_j||_2 = 1  ->  unit variance
    gamma = 0.1 / np.log(max(p, n))
    sigma = float(np.std(y, ddof=1))
    S = []
    from sklearn.linear_model import Lasso, LassoCV
    for _ in range(iters):
        # BCH lambda = 2c*sqrt(n)*PHI^{-1}(1-gamma/2p) in their (1/n)RSS scale;
        # sklearn uses (1/2n)RSS, so the matching alpha is c*sigma*PHI^{-1}/sqrt(n).
        lam = c * sigma * norm.ppf(1 - gamma / (2 * p)) / np.sqrt(n)
        m = Lasso(alpha=lam, fit_intercept=False, max_iter=50_000).fit(Xu, y)
        S = sorted(np.where(np.abs(m.coef_) > 1e-10)[0].tolist())
        resid = y - (Xz[:, S] @ _ols(Xz, y, S) if S else 0.0)
        new_sigma = float(np.std(resid, ddof=1))
        if abs(new_sigma - sigma) < 1e-6:
            break
        sigma = new_sigma or sigma
    if S:
        return S, "post-lasso BCH plug-in"
    cv = LassoCV(cv=CV_FOLDS, fit_intercept=False, max_iter=50_000,
                 random_state=SEED).fit(Xu, y)
    S = sorted(np.where(np.abs(cv.coef_) > 1e-10)[0].tolist())
    return S, "BCH empty -> lassoCV support"


def sel_knockoffs(Xz, y, names=None, k=None, fdr=0.10):
    try:
        from knockpy import KnockoffFilter
        kf = KnockoffFilter(ksampler="gaussian", fstat="lasso")
        rej = kf.forward(X=Xz, y=y, fdr=fdr)
        return sorted(np.where(rej > 0)[0].tolist()), f"modelX-knockoffs(fdr={fdr})"
    except Exception as e:
        return None, f"knockoffs unavailable ({type(e).__name__})"


# ══════════════════════════════════════════════════════════════════════════════
#  Benchmark driver
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(name, selector, Xz, y_c, y_raw, names, k, refit_halves=True):
    t0 = time.time()
    out = selector(Xz, y_c, names, k)
    if out[0] is None:
        return {"method": name, "engine": out[1], "n_sel": np.nan}
    S, engine = out
    row = {"method": name, "engine": engine, "n_sel": len(S),
           "factors": [names[j] for j in S]}
    row["R2adj_IS"] = r2_adj(Xz, y_c, S)
    row["alpha_IS"], row["t_IS"] = alpha_hac(y_raw, Xz, S)
    a, t, ir, n_oos = frozen_oos(y_raw, Xz, S)
    row.update({"alpha_OOS": a, "t_OOS": t, "IR_OOS": ir, "N_oos": n_oos})
    if refit_halves:
        h = len(y_c) // 2
        S1, _ = selector(Xz[:h], y_c[:h], names, k)
        S2, _ = selector(Xz[h:], y_c[h:], names, k)
        if S1 is None or S2 is None:
            row["Jaccard"], row["alpha_SPLIT"], row["t_SPLIT"] = np.nan, np.nan, np.nan
        else:
            row["Jaccard"] = jaccard(S1, S2)
            row["alpha_SPLIT"], row["t_SPLIT"] = alpha_hac(y_raw[h:], Xz[h:], S1)
    else:
        row["Jaccard"] = np.nan
        row["alpha_SPLIT"], row["t_SPLIT"] = np.nan, np.nan
    row["sec"] = round(time.time() - t0, 1)
    return row


def run_strategy(strategy: str, k: int, write_pds: bool):
    print(f"\n{'=' * 78}\n  {strategy.upper()}\n{'=' * 78}")
    y_c, X, y_mean, names = load_inputs(strategy)
    Xz = X.values
    yc = y_c.values
    yr = yc + y_mean
    print(f"   T = {len(yc)}, p = {Xz.shape[1]}, k = {k}")

    rows = []
    rows.append(evaluate("ell0",     lambda X_, y_, n_, k_: sel_ell0(X_, y_, n_, k_),    Xz, yc, yr, names, k))
    rows.append(evaluate("forward",  lambda X_, y_, n_, k_: sel_forward(X_, y_, n_, k_), Xz, yc, yr, names, k))

    aen_idx = load_aen_selection(strategy, names)
    if aen_idx is not None:
        rows.append(evaluate("aen(paper)", lambda X_, y_, n_, k_: (aen_idx, "loaded aen_results.json"),
                             Xz, yc, yr, names, k, refit_halves=False))
    else:
        print("   ⚠️  aen_results.json not found — AEN column skipped.")

    rows.append(evaluate("relaxed",  lambda X_, y_, n_, k_: sel_relaxed_lasso(X_, y_),   Xz, yc, yr, names, k))
    rows.append(evaluate("cpss",     lambda X_, y_, n_, k_: sel_cpss(X_, y_),            Xz, yc, yr, names, k))
    rows.append(evaluate("bch_pds",  lambda X_, y_, n_, k_: sel_bch_pds(X_, y_),         Xz, yc, yr, names, k))
    rows.append(evaluate("knockoffs", lambda X_, y_, n_, k_: sel_knockoffs(X_, y_),      Xz, yc, yr, names, k))

    df = pd.DataFrame(rows)
    cols = ["method", "n_sel", "R2adj_IS", "alpha_IS", "t_IS",
            "alpha_SPLIT", "t_SPLIT", "Jaccard", "alpha_OOS", "t_OOS", "IR_OOS", "sec", "engine"]
    df_show = df[[c for c in cols if c in df.columns]]
    with pd.option_context("display.width", 160, "display.float_format", "{:.2f}".format):
        print("\n" + df_show.to_string(index=False))

    if write_pds:
        pds_row = next((r for r in rows if r["method"] == "bch_pds"), None)
        if pds_row and pds_row.get("factors") is not None:
            out = {"selected_factors": pds_row["factors"],
                   "n_selected": pds_row["n_sel"],
                   "method": "post-double-selection (BCH plug-in lasso; empty 2nd step for intercept target)",
                   "engine": pds_row["engine"]}
            f = get_strategy_aen_dir(strategy) / "pds_results.json"
            f.write_text(json.dumps(out, indent=2))
            print(f"   ✅  PDS support written -> {f}")

    return strategy, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="all")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--write-pds", action="store_true")
    args = ap.parse_args()

    todo = list(STRATEGIES) if args.strategies == "all" else args.strategies.split(",")
    all_rows, blob = [], {}
    for s in todo:
        s_name, df = run_strategy(s, args.k, args.write_pds)
        df.insert(0, "strategy", s_name)
        all_rows.append(df)
        blob[s_name] = df.to_dict(orient="records")

    out_dir = get_aen_output_dir() / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.concat(all_rows, ignore_index=True)
    summary.drop(columns=["factors"], errors="ignore").to_csv(out_dir / "selector_benchmark_summary.csv", index=False)
    (out_dir / "selector_benchmark.json").write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n✅  Written: {out_dir / 'selector_benchmark_summary.csv'}")
    print(f"✅  Written: {out_dir / 'selector_benchmark.json'}")
    print("\nRead the table as: better = higher t_SPLIT and IR_OOS and Jaccard at "
          "equal or smaller n_sel. Ignore in-sample columns for ranking.")


if __name__ == "__main__":
    main()

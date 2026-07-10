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
  group_rep   EXACTLY one factor per economic category: constrained l0 solved
              by block-coordinate exchange (best-|corr| init + random restarts).
  exclusive   Exclusive lasso (Zhou-Jin-Hoi 2010; Campbell-Allen 2017):
              lambda * sum_g (sum_{j in g}|beta_j|)^2 — competition WITHIN each
              category, tends to keep at least one representative per group.
  cluster_rep Cluster-representative lasso (Buhlmann et al. 2013): one
              screened representative per category, LassoCV picks which
              categories matter (at most one factor per group).
  aen_dedup   The paper's AEN selection, CORRECTED: at most one factor per
              category (RSS exchange among each category's AEN members;
              AEN factors outside every category are kept as singletons).
              Removes the grouping-effect inflation (31 factors on iTraxx).

Categories for the group methods, in priority order: (1) --groups file.csv
(columns factor,group) if you want a manual override; (2) AUTOMATIC: the
paper's own economic panels, read from FACTOR_INFO in
src/factor_models/06_generate_factor_table.py (same categories as Table
A.XV; safe AST parse, no import side effects); (3) data-driven correlation
clusters (|rho|>0.60) as a last resort. --emit-groups-template still writes
a CSV skeleton if you ever want the manual route.

NOTE ON alpha_OOS: the fitted value EXCLUDES the estimated intercept
(e_t = y_t - x_t' beta_hat), so the abnormal return carries the alpha —
same estimand as Table IX. An empty selection therefore reports the raw
OOS alpha of the strategy, which is the natural no-controls anchor.

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
    """Frozen-set expanding-window OOS abnormal return (Table IX estimand).

    The OLS at each date includes an intercept, but the fitted value used to
    form the abnormal return EXCLUDES it: e_t = y_t - x_t' beta_hat. The alpha
    therefore shows up in e_t instead of being absorbed mechanically. With an
    empty set, e_t = y_t and the row reports the raw OOS alpha (useful anchor)."""
    T = len(y_raw)
    if T <= burnin + 12:
        return np.nan, np.nan, np.nan, 0
    e = []
    for t in range(burnin, T):
        if idx:
            Xtr = sm.add_constant(Xz[:t, idx])
            b = np.linalg.lstsq(Xtr, y_raw[:t], rcond=None)[0]
            e.append(y_raw[t] - float(Xz[t, idx] @ b[1:]))   # intercept excluded
        else:
            e.append(y_raw[t])
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


def _rss(Xz, y, idx):
    if not idx:
        return float(np.sum((y - y.mean()) ** 2))
    beta = _ols(Xz, y, idx)
    return float(np.sum((y - Xz[:, idx] @ beta) ** 2))


def sel_ell0(Xz, y, names=None, k=K_DEFAULT):
    """Exact-size l0 via abess splicing, polished against greedy forward:
    splicing can land on a local optimum, so we keep whichever of the two
    feasible k-subsets has the lower in-sample RSS. The engine label records
    which one won — if forward wins often, tighten abess's search instead of
    trusting the default."""
    fwd = sel_forward(Xz, y, names, k)[0]
    try:
        from abess.linear import LinearRegression
        m = LinearRegression(support_size=k, fit_intercept=False)
        m.fit(Xz, y)
        ab = sorted(np.where(np.abs(m.coef_) > 0)[0].tolist())
        if len(ab) == k:
            if _rss(Xz, y, ab) <= _rss(Xz, y, fwd):
                return ab, "abess(ell0)"
            return fwd, "ell0: greedy beat splicing (best-RSS kept)"
        print(f"   ⚠️  abess returned {len(ab)} != k={k}; using forward.")
    except Exception as e:
        print(f"   ⚠️  abess unavailable ({type(e).__name__}) — l0 NOT verified, using forward.")
    return fwd, "forward-forced (NOT l0)"


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
#  Group-aware selectors ("at least / exactly one factor per economic category")
# ══════════════════════════════════════════════════════════════════════════════
def _load_factor_info():
    """Extract FACTOR_INFO / PANEL_NAMES from src/factor_models/
    06_generate_factor_table.py WITHOUT importing it (AST literal parse:
    no side effects, no path assumptions). Returns (dict, panel_names, path)."""
    import ast
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "factor_models" / "06_generate_factor_table.py",
        Path.cwd() / "src" / "factor_models" / "06_generate_factor_table.py",
    ]
    rd = getattr(aen_config, "RESULTS_DIR", None)
    if rd is not None:
        candidates.append(Path(rd).parent / "src" / "factor_models" / "06_generate_factor_table.py")
    for cand in candidates:
        if not cand.exists():
            continue
        try:
            tree = ast.parse(cand.read_text(encoding="utf-8", errors="ignore"))
            fi, pn = None, {}
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if getattr(tgt, "id", "") == "FACTOR_INFO":
                            fi = ast.literal_eval(node.value)
                        elif getattr(tgt, "id", "") == "PANEL_NAMES":
                            pn = ast.literal_eval(node.value)
            if fi:
                return fi, pn, cand
        except Exception as e:
            print(f"   ⚠️  could not parse {cand.name}: {type(e).__name__}: {e}")
    return None, {}, None


def build_groups(names, Xz, y, groups_path=None):
    """factor -> group. Priority: (1) user CSV via --groups (columns:
    factor,group); (2) the paper's own economic categories, read from
    FACTOR_INFO in src/factor_models/06_generate_factor_table.py (the same
    panels as Table A.XV); (3) data-driven correlation clusters (greedy,
    |rho| > 0.60) as a last resort. Returns (list of index-lists, label)."""
    if groups_path:
        gmap = pd.read_csv(groups_path)
        gmap.columns = [c.strip().lower() for c in gmap.columns]
        lookup = dict(zip(gmap["factor"].astype(str).str.strip(),
                          gmap["group"].astype(str).str.strip()))
        groups, missing = {}, 0
        for j, nm in enumerate(names):
            g = lookup.get(nm)
            if g is None or g == "" or g.lower() == "nan":
                missing += 1
                continue
            groups.setdefault(g, []).append(j)
        if missing:
            print(f"   ⚠️  {missing} factors not in {groups_path} — excluded from group methods.")
        return list(groups.values()), f"csv:{Path(groups_path).name} ({len(groups)} groups)"

    fi, pnames, src06 = _load_factor_info()
    if fi:
        groups, unmatched = {}, []
        for j, nm in enumerate(names):
            meta = fi.get(nm)
            if meta is None:
                unmatched.append(nm)
                continue
            groups.setdefault(str(meta[0]), []).append(j)
        if groups:
            if unmatched:
                print(f"   ⚠️  {len(unmatched)} factors not in FACTOR_INFO — excluded from "
                      f"group methods: {', '.join(unmatched)}")
            keys = sorted(groups)
            label = ", ".join(pnames.get(k2, k2).replace(" Factors", "") for k2 in keys)
            return [groups[k2] for k2 in keys], (
                f"{src06.name} ({len(keys)} categories: {label})")

    # last resort: greedy correlation clustering
    cy = np.abs([np.corrcoef(Xz[:, j], y)[0, 1] for j in range(Xz.shape[1])])
    order = np.argsort(-cy)
    reps, clusters = [], []
    for j in order:
        placed = False
        for gi, r in enumerate(reps):
            if abs(np.corrcoef(Xz[:, j], Xz[:, r])[0, 1]) > 0.60:
                clusters[gi].append(int(j)); placed = True; break
        if not placed:
            reps.append(int(j)); clusters.append([int(j)])
    return clusters, f"auto-corr-clusters ({len(clusters)} groups, |rho|>0.60)"


def _abs_corr_y(Xz, y):
    return np.abs([np.corrcoef(Xz[:, j], y)[0, 1] if np.std(Xz[:, j]) > 0 else 0.0
                   for j in range(Xz.shape[1])])


def _exchange(Xz, y, cand_lists, restarts=3):
    """Pick exactly one index from each candidate list to minimize RSS,
    by block-coordinate exchange with best-|corr(y)| init + random restarts."""
    cy = _abs_corr_y(Xz, y)
    starts = [[max(c, key=lambda j: cy[j]) for c in cand_lists]]
    for _ in range(restarts - 1):
        starts.append([int(rng.choice(c)) for c in cand_lists])
    best_S, best_rss = None, np.inf
    for S in starts:
        S = list(S)
        for _ in range(15):
            changed = False
            for gi, cand in enumerate(cand_lists):
                for j in cand:
                    if j == S[gi]:
                        continue
                    trial = S.copy(); trial[gi] = j
                    if _rss(Xz, y, sorted(trial)) < _rss(Xz, y, sorted(S)) - 1e-10:
                        S[gi] = j; changed = True
            if not changed:
                break
        r = _rss(Xz, y, sorted(S))
        if r < best_rss:
            best_rss, best_S = r, sorted(S)
    return best_S


def sel_group_rep(Xz, y, groups, restarts=3):
    """EXACTLY one factor per group: constrained l0 solved by block
    coordinate exchange (init = best |corr(y)| per group + random restarts)."""
    S = _exchange(Xz, y, groups, restarts)
    return S, f"one-per-group exchange ({len(groups)} groups)"


def sel_exclusive(Xz, y, groups, n_lam=8):
    """Exclusive lasso (Zhou-Jin-Hoi 2010; Campbell-Allen 2017): penalty
    lambda * sum_g (sum_{j in g} |beta_j|)^2 induces competition WITHIN each
    group and tends to keep at least one representative per group. Coordinate
    descent on the unit-variance design; lambda by contiguous-block CV."""
    n, p = Xz.shape
    Xu = Xz * np.sqrt(n)
    in_group = np.full(p, -1)
    for gi, g in enumerate(groups):
        for j in g:
            in_group[j] = gi
    active = np.where(in_group >= 0)[0]

    def fit(lam, Xw, yw):
        b = np.array([Xw[:, j] @ yw / len(yw) for j in range(p)])
        b[in_group < 0] = 0.0
        for _ in range(300):
            delta = 0.0
            Sg = np.zeros(len(groups))
            for gi2, g2 in enumerate(groups):
                Sg[gi2] = np.sum(np.abs(b[g2]))
            r = yw - Xw @ b
            for j in active:
                gi2 = in_group[j]
                rho = Xw[:, j] @ r / len(yw) + b[j]
                thr = 2.0 * lam * (Sg[gi2] - abs(b[j]))
                new = np.sign(rho) * max(abs(rho) - thr, 0.0) / (1.0 + 2.0 * lam)
                if new != b[j]:
                    r -= Xw[:, j] * (new - b[j])
                    Sg[gi2] += abs(new) - abs(b[j])
                    delta = max(delta, abs(new - b[j]))
                    b[j] = new
            if delta < 1e-7:
                break
        return b

    b0 = np.array([Xu[:, j] @ y / n for j in range(p)])
    lam_ref = 1.0 / (2.0 * max(np.sum(np.abs(b0[g])) for g in groups) + 1e-12)
    lams = lam_ref * np.logspace(-2, 0.7, n_lam)
    folds = np.array_split(np.arange(n), CV_FOLDS)
    cv = np.zeros(n_lam)
    for test in folds:
        train = np.setdiff1d(np.arange(n), test)
        for li, lam in enumerate(lams):
            b = fit(lam, Xu[train], y[train])
            cv[li] += float(np.mean((y[test] - Xu[test] @ b) ** 2)) / CV_FOLDS
    li = int(np.argmin(cv))
    b = fit(lams[li], Xu, y)
    thr = 1e-3 * max(np.max(np.abs(b)), 1e-12)
    S = sorted(np.where(np.abs(b) > thr)[0].tolist())
    return S, f"exclusive-lasso (lam#{li}, {len(groups)} groups)"


def sel_cluster_rep(Xz, y, groups):
    """Cluster-representative lasso (Buhlmann et al. 2013): pre-screen ONE
    representative per group (max |corr(y)|), then LassoCV decides which
    groups matter. At most one factor per category, not necessarily all."""
    from sklearn.linear_model import LassoCV
    n = len(y)
    cy = _abs_corr_y(Xz, y)
    reps = [max(g, key=lambda j: cy[j]) for g in groups]
    Xu = Xz[:, reps] * np.sqrt(n)
    cvm = LassoCV(cv=CV_FOLDS, fit_intercept=False, max_iter=50_000,
                  random_state=SEED).fit(Xu, y)
    S = sorted(reps[i] for i in np.where(np.abs(cvm.coef_) > 1e-10)[0])
    return S, f"cluster-representative lasso ({len(groups)} groups)"


def sel_aen_dedup(Xz, y, groups, aen_idx):
    """The user's 'corrected AEN': keep the AEN-selected set, but allow at
    most ONE factor per category. Within each category the AEN members
    compete via the exchange step; AEN factors that fall outside every
    provided category are kept as singletons (nothing is silently dropped).
    Directly removes the grouping-effect inflation (31 factors on iTraxx)."""
    if not aen_idx:
        return None, "aen_dedup unavailable (no aen_results.json)"
    cand_lists, covered = [], set()
    for g in groups:
        members = [j for j in g if j in aen_idx]
        if members:
            cand_lists.append(members)
            covered.update(members)
    leftovers = [j for j in aen_idx if j not in covered]
    cand_lists += [[j] for j in leftovers]
    S = _exchange(Xz, y, cand_lists)
    tag = f"AEN deduped to one-per-category ({len(cand_lists)} slots"
    tag += f", {len(leftovers)} ungrouped kept)" if leftovers else ")"
    return S, tag


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


def run_strategy(strategy: str, k: int, write_pds: bool, groups_path=None):
    print(f"\n{'=' * 78}\n  {strategy.upper()}\n{'=' * 78}")
    y_c, X, y_mean, names = load_inputs(strategy)
    Xz = X.values
    yc = y_c.values
    yr = yc + y_mean
    print(f"   T = {len(yc)}, p = {Xz.shape[1]}, k = {k}")

    groups, gsrc = build_groups(names, Xz, yc, groups_path)
    print(f"   groups: {gsrc}")

    rows = []
    rows.append(evaluate("ell0",     lambda X_, y_, n_, k_: sel_ell0(X_, y_, n_, k_),    Xz, yc, yr, names, k))
    rows.append(evaluate("forward",  lambda X_, y_, n_, k_: sel_forward(X_, y_, n_, k_), Xz, yc, yr, names, k))

    aen_idx = load_aen_selection(strategy, names)
    if aen_idx is not None:
        rows.append(evaluate("aen(paper)", lambda X_, y_, n_, k_: (aen_idx, "loaded aen_results.json"),
                             Xz, yc, yr, names, k, refit_halves=False))
    else:
        print("   ⚠️  aen_results.json not found — AEN column skipped.")

    rows.append(evaluate("relaxed",     lambda X_, y_, n_, k_: sel_relaxed_lasso(X_, y_),      Xz, yc, yr, names, k))
    rows.append(evaluate("cpss",        lambda X_, y_, n_, k_: sel_cpss(X_, y_),               Xz, yc, yr, names, k))
    rows.append(evaluate("bch_pds",     lambda X_, y_, n_, k_: sel_bch_pds(X_, y_),            Xz, yc, yr, names, k))
    rows.append(evaluate("group_rep",   lambda X_, y_, n_, k_: sel_group_rep(X_, y_, groups),  Xz, yc, yr, names, k))
    rows.append(evaluate("exclusive",   lambda X_, y_, n_, k_: sel_exclusive(X_, y_, groups),  Xz, yc, yr, names, k))
    rows.append(evaluate("cluster_rep", lambda X_, y_, n_, k_: sel_cluster_rep(X_, y_, groups),Xz, yc, yr, names, k))
    if aen_idx is not None:
        rows.append(evaluate("aen_dedup",
                             lambda X_, y_, n_, k_: sel_aen_dedup(X_, y_, groups, aen_idx),
                             Xz, yc, yr, names, k, refit_halves=False))
    rows.append(evaluate("knockoffs",   lambda X_, y_, n_, k_: sel_knockoffs(X_, y_),          Xz, yc, yr, names, k))

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


def emit_tex(blob, out_path):
    """Write the paper-ready selector-robustness table (canonical template:
    note above, singlespacing, tabular* full width, HAC lags declared)."""
    DISPLAY = [("ell0",        r"Best subset ($\ell_0$, $k{=}8$)"),
               ("group_rep",   r"One per risk category"),
               ("aen(paper)",  r"Adaptive elastic net"),
               ("aen_dedup",   r"AEN, one per category"),
               ("relaxed",     r"Relaxed lasso"),
               ("bch_pds",     r"Post-lasso (plug-in penalty)"),
               ("cpss",        r"Stability selection (CPSS)")]
    STRATS = [("btp_italia", "BTP Italia"),
              ("cds_bond_basis", "CDS--Bond Basis"),
              ("itraxx_combined", "iTraxx Combined")]

    def _stars_p(t):
        if t is None or (isinstance(t, float) and np.isnan(t)):
            return ""
        a = abs(t)
        return "^{***}" if a > 2.576 else "^{**}" if a > 1.96 else "^{*}" if a > 1.645 else ""

    def cell(rec, a_key, t_key):
        a, t = rec.get(a_key), rec.get(t_key)
        if a is None or (isinstance(a, float) and np.isnan(a)):
            return "--", ""
        return f"${a:+.2f}{_stars_p(t)}$", f"({t:.2f})"

    L = [r"\begin{table}[H]", r"\centering", r"\singlespacing",
         r"\caption{Alpha Across Factor-Selection Methods}",
         r"\label{tab:selector_robustness}",
         r"\begin{minipage}{\textwidth}",
         r"{\footnotesize\noindent For each selection method, $n$ is the number of "
         r"selected factors and two post-selection alphas of the monthly net-of-cost "
         r"strategy excess return are reported. $\alpha^{\text{split}}$: honest "
         r"split-sample alpha --- factors are selected on the first half of the sample "
         r"only, and the alpha is the intercept of an OLS on the second half "
         r"(valid post-selection inference by sample splitting); not available for the "
         r"AEN rows, whose set is estimated on the full sample. $\alpha^{\text{OOS}}$: "
         r"frozen-set out-of-sample alpha --- with the selected set held fixed, loadings "
         r"are re-estimated each month on an expanding window (60-month burn-in) and the "
         r"alpha is the mean abnormal return $y_t - \mathbf{x}_t'\hat{\beta}_{t-1}$, with "
         r"the estimated intercept excluded from the fitted value. Alphas are annualized "
         r"(\% p.a.); $t$-statistics in parentheses use Newey--West HAC standard errors "
         r"(4 lags). The $\ell_0$ problem is solved by the splicing algorithm "
         r"(\texttt{abess}) polished against greedy forward search; risk categories are "
         r"the panels of Table~\ref{tab:factor_list}. Selection with zero factors "
         r"reports the raw strategy alpha. "
         r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$.}",
         r"\end{minipage}",
         r"\par\vspace{6pt}",
         r"\footnotesize",
         r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + " rrr" * len(STRATS) + "}",
         r"\toprule"]
    head = " " + "".join(rf" & \multicolumn{{3}}{{c}}{{{lab}}}" for _, lab in STRATS) + r" \\"
    L.append(head)
    cml, c0 = [], 2
    for _ in STRATS:
        cml.append(rf"\cmidrule(lr){{{c0}-{c0+2}}}"); c0 += 3
    L.append("".join(cml))
    L.append("Method" + r" & $n$ & $\alpha^{\text{split}}$ & $\alpha^{\text{OOS}}$" * len(STRATS) + r" \\")
    L.append(r"\midrule")
    for key, disp in DISPLAY:
        r1, r2 = disp, ""
        ok = False
        for skey, _ in STRATS:
            rec = next((r for r in blob.get(skey, []) if r.get("method") == key), None)
            if rec is None:
                r1 += " & -- & -- & --"; r2 += " & & &"
                continue
            ok = True
            n = rec.get("n_sel")
            n_str = "--" if n is None or (isinstance(n, float) and np.isnan(n)) else f"{int(n)}"
            a_s, t_s = cell(rec, "alpha_SPLIT", "t_SPLIT")
            a_o, t_o = cell(rec, "alpha_OOS", "t_OOS")
            r1 += f" & {n_str} & {a_s} & {a_o}"
            r2 += f" &  & {t_s} & {t_o}"
        if ok:
            L += [r1 + r" \\", r2 + r" \\[2pt]"]
    L += [r"\bottomrule", r"\end{tabular*}", r"\end{table}", ""]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"✅  Paper table written -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", default="all")
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--write-pds", action="store_true")
    ap.add_argument("--groups", default=None,
                    help="CSV with columns factor,group (economic categories). "
                         "If omitted, groups are data-driven correlation clusters.")
    ap.add_argument("--emit-groups-template", action="store_true",
                    help="Write factor_groups_template.csv with all factor names "
                         "and an empty group column, then exit.")
    ap.add_argument("--emit-tex", action="store_true",
                    help="Also write the paper-ready LaTeX table "
                         "selector_robustness_article.tex (canonical template).")
    ap.add_argument("--tex-out", default=None,
                    help="Override output path for the LaTeX table.")
    args = ap.parse_args()

    todo = list(STRATEGIES) if args.strategies == "all" else args.strategies.split(",")

    if args.emit_groups_template:
        seen = []
        for s in todo:
            _, X, _, names = load_inputs(s)
            for nm in names:
                if nm not in seen:
                    seen.append(nm)
        out_dir = get_aen_output_dir() / "benchmark"
        out_dir.mkdir(parents=True, exist_ok=True)
        f = out_dir / "factor_groups_template.csv"
        pd.DataFrame({"factor": seen, "group": ""}).to_csv(f, index=False)
        print(f"✅  Template written -> {f}\n    Fill the 'group' column "
              f"(e.g. Equity, Term, Credit, Volatility, Funding, Macro, FX) "
              f"and re-run with --groups {f}")
        return

    all_rows, blob = [], {}
    for s in todo:
        s_name, df = run_strategy(s, args.k, args.write_pds, args.groups)
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
    if args.emit_tex:
        tex_path = Path(args.tex_out) if args.tex_out else out_dir / "selector_robustness_article.tex"
        emit_tex(blob, tex_path)
    print("\nRead the table as: better = higher t_SPLIT and IR_OOS and Jaccard at "
          "equal or smaller n_sel. Ignore in-sample columns for ranking.")


if __name__ == "__main__":
    main()
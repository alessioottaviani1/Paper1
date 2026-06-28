"""
================================================================================
02s_best_subset.py — Best-Subset Selection (primary selector) + EBIC cross-check
================================================================================

Primary factor-selection step for the spanning regressions. Replaces the AEN
(02_estimation.py) as the *primary* selector; the AEN is retained downstream as
a selection-method robustness check.

References
----------
  Bertsimas, King & Mazumder (2016), "Best Subset Selection via a Modern
      Optimization Lens", Annals of Statistics 44(2):813-852.
      ell0-constrained least squares: the cardinality constraint ||beta||_0 <= k
      directly controls the number of selected factors and, unlike the LASSO,
      brings a factor in without shrinkage — draining its correlated surrogates
      (i.e. among a correlated cluster it keeps the most informative member, not
      all of them).

  Chen & Chen (2008), "Extended Bayesian Information Criteria for Model Selection
      with Large Model Spaces", Biometrika 95(3):759-771.
      EBIC adds a model-space penalty 2*gamma*log C(p,k) to the BIC, correcting
      the tendency of AIC/BIC to over-select in the small-n-large-p regime.
      Selection-consistent in high dimensions. Used here as a *data-driven*
      cross-check on the cardinality.

Design
------
  Reads the SAME preprocessed inputs as 02 (per strategy):
      <results>/<strategy>/aen/hqc/y_centered.parquet
      <results>/<strategy>/aen/hqc/X_standardized.parquet   (||X_j||_2 = 1)
  Selection is performed on the standardized design (scale-invariant for which
  factors enter). The reported point estimates / inference on RAW data are the
  job of 03 (post-selection OLS + HAC) — this module only fixes the support.

  Writes (per strategy), schema-compatible with 02's aen_results.json so that
  03/07 consume it unchanged once pointed at the new filename:
      subset_results.json       {selected_factors, n_selected, selected_coefficients, ...}
      subset_sensitivity.json   best-subset support at k in K_GRID + the EBIC path
  And an aggregate subset_summary.json.

Parsimony level
---------------
  K_PRIMARY is the cardinality of the primary spec, fixed for cross-strategy
  comparability and chosen for interpretability (<= the parsimony cap). Results
  are reported across K_GRID as the robustness defence for the choice of k. The
  EBIC count is reported alongside; note it tends to be sharper than K_PRIMARY
  for the concentrated strategies and validates the parsimony of the dense one.

Institution: EDHEC Business School — PhD Thesis
"""

import json
import importlib.util
from math import lgamma, log
from pathlib import Path

import numpy as np
import pandas as pd

# ── Load configuration (same mechanism as 02; module name starts with a digit) ─
_CFG_PATH = Path(__file__).resolve().parent / "00_config.py"
_spec = importlib.util.spec_from_file_location("aen_config", _CFG_PATH)
aen_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aen_config)

STRATEGIES            = aen_config.STRATEGIES
get_aen_output_dir    = aen_config.get_aen_output_dir
get_strategy_aen_dir  = aen_config.get_strategy_aen_dir
COEF_TOL              = getattr(aen_config, "COEF_TOL", 1e-8)

# ── Parameters ────────────────────────────────────────────────────────────────
K_PRIMARY  = 10          # cardinality of the primary spec (<= interpretability cap)
K_GRID     = (5, 8, 10)  # support sizes reported for the k-sensitivity table
K_PATH_MAX = 20          # max support explored when building the EBIC path


# ── Best-subset of EXACT size k (ell0) ─────────────────────────────────────────
def _forward_forced(Xz, y, k):
    """Greedy forward selection forced to exactly k factors (min RSS at each step).

    Documented fast approximation to the ell0 optimum (Hastie, Tibshirani &
    Tibshirani 2020). Deterministic, dependency-free; used as the default and as
    the fallback when an exact ell0 solver is unavailable.
    """
    chosen, p = [], Xz.shape[1]
    while len(chosen) < k:
        best_rss, best_j = np.inf, -1
        for j in range(p):
            if j in chosen:
                continue
            idx = chosen + [j]
            A = Xz[:, idx]
            beta = np.linalg.lstsq(A, y, rcond=None)[0]
            rss = float(np.sum((y - A @ beta) ** 2))
            if rss < best_rss:
                best_rss, best_j = rss, j
        chosen.append(best_j)
    return sorted(chosen)


def best_subset(Xz, y, k):
    """Exact ell0 best subset of size k via `abess` if available, else forward-forced.

    Returns (sorted index list, engine_label).
    """
    try:
        from abess.linear import LinearRegression
        model = LinearRegression(support_size=k, fit_intercept=False)
        model.fit(Xz, y)
        idx = sorted(np.where(np.abs(model.coef_) > 0)[0].tolist())
        if len(idx) == k:                      # only trust abess if it honours k
            return idx, "abess(ell0)"
    except Exception:
        pass
    return _forward_forced(Xz, y, k), "forward-forced"


# ── EBIC over a forward path (Chen-Chen 2008) ──────────────────────────────────
def _log_choose(p, k):
    if k <= 0 or k > p:
        return 0.0
    return lgamma(p + 1) - lgamma(k + 1) - lgamma(p - k + 1)


def _rss(Xz, y, idx):
    if not idx:
        return float(y @ y)
    A = Xz[:, idx]
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.sum((y - A @ beta) ** 2))


def ebic_select(Xz, y, gamma, kmax):
    """Data-driven cardinality by EBIC along a forward path.

    EBIC(S) = T*log(RSS/T) + |S|*log(T) + 2*gamma*log C(p, |S|).
    Returns (selected index list, ebic_k, [(k, ebic) ...]).
    """
    T, p = Xz.shape
    # build a forward path (order in which factors enter, by min RSS)
    avail, order = list(range(p)), []
    while len(order) < min(kmax, p):
        best_rss, best_j = np.inf, -1
        for j in avail:
            rss = _rss(Xz, y, order + [j])
            if rss < best_rss:
                best_rss, best_j = rss, j
        order.append(best_j)
        avail.remove(best_j)
    # score each prefix of the path
    table, best = [], (np.inf, [])
    for k in range(0, len(order) + 1):
        idx = order[:k]
        eb = T * log(_rss(Xz, y, idx) / T) + k * log(T) + 2.0 * gamma * _log_choose(p, k)
        table.append((k, round(eb, 3)))
        if eb < best[0]:
            best = (eb, list(idx))
    return best[1], len(best[1]), table


# ── OLS coefficients on the standardized design (for the JSON; 03 is authoritative) ─
def ols_betas(Xz, y, idx):
    if not idx:
        return {}
    A = Xz[:, idx]
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return {int(i): float(b) for i, b in zip(idx, beta)}


def r2_adj(Xz, y, idx):
    """In-sample adjusted R^2 on the standardized design (descriptive only)."""
    T = len(y)
    k = len(idx)
    if k == 0 or k >= T - 1:
        return 0.0
    rss = _rss(Xz, y, idx)
    tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss
    return 1.0 - (1.0 - r2) * (T - 1) / (T - k - 1)


# ── Per-strategy driver ─────────────────────────────────────────────────────────
def run_strategy(strategy_name):
    strategy_dir = get_strategy_aen_dir(strategy_name)
    y_path = strategy_dir / "y_centered.parquet"
    X_path = strategy_dir / "X_standardized.parquet"
    if not y_path.exists() or not X_path.exists():
        print(f"   ❌ Preprocessed data not found for {strategy_name}. Run 01 first.")
        return None

    y_df = pd.read_parquet(y_path)
    X_df = pd.read_parquet(X_path)
    y = np.asarray(y_df.iloc[:, 0], dtype=float)
    X = np.asarray(X_df, dtype=float)
    names = list(X_df.columns)
    T, p = X.shape
    gamma = 1.0 - log(T) / (2.0 * log(p))            # Chen-Chen data-driven default

    print(f"\n   {strategy_name}: T={T}, p={p}, EBIC gamma={gamma:.3f}")

    # k-sensitivity (best-subset at each grid size)
    sensitivity = {}
    print(f"   {'support':>10} {'k':>3} {'R2adj(std)':>11}  engine")
    for k in K_GRID:
        idx, engine = best_subset(X, y, k)
        sensitivity[k] = {
            "factors": [names[i] for i in idx],
            "r2_adj_std": round(r2_adj(X, y, idx), 4),
            "engine": engine,
        }
        print(f"   {'k=' + str(k):>10} {len(idx):>3} {sensitivity[k]['r2_adj_std']:>11.4f}  {engine}")

    # EBIC data-driven count (cross-check)
    ebic_idx, ebic_k, ebic_table = ebic_select(X, y, gamma, K_PATH_MAX)
    print(f"   {'EBIC':>10} {ebic_k:>3} {r2_adj(X, y, ebic_idx):>11.4f}  data-driven")

    # primary spec = best-subset at K_PRIMARY
    primary_idx, primary_engine = best_subset(X, y, K_PRIMARY)
    selected = [names[i] for i in primary_idx]
    coeffs_std = {names[i]: b for i, b in ols_betas(X, y, primary_idx).items()}

    # write subset_results.json (schema-compatible with aen_results.json)
    subset_results = {
        "method": "best-subset (Bertsimas-King-Mazumder 2016)",
        "cardinality_k": K_PRIMARY,
        "engine": primary_engine,
        "selected_factors": selected,
        "n_selected": len(selected),
        "selected_coefficients": coeffs_std,
        "coefficient_space": "standardized (||X_j||_2 = 1); raw-data inference in 03",
        "r2_adj_std": round(r2_adj(X, y, primary_idx), 4),
        "ebic_data_driven_k": ebic_k,
        "ebic_gamma": round(gamma, 4),
    }
    with open(strategy_dir / "subset_results.json", "w") as f:
        json.dump(subset_results, f, indent=2)

    with open(strategy_dir / "subset_sensitivity.json", "w") as f:
        json.dump({
            "k_grid": {str(k): sensitivity[k] for k in K_GRID},
            "ebic": {
                "gamma": round(gamma, 4),
                "selected_k": ebic_k,
                "selected_factors": [names[i] for i in ebic_idx],
                "ebic_by_k": ebic_table,
            },
        }, f, indent=2)

    print(f"   💾 subset_results.json ({len(selected)} factors), subset_sensitivity.json")
    return {
        "strategy": strategy_name,
        "n_selected": len(selected),
        "selected_factors": selected,
        "ebic_k": ebic_k,
        "r2_adj_std": subset_results["r2_adj_std"],
    }


def main():
    print("=" * 80)
    print("BEST-SUBSET SELECTION (primary) + EBIC cross-check")
    print("  Bertsimas-King-Mazumder (2016, Ann. Stat.) | Chen-Chen (2008, Biometrika)")
    print(f"  K_PRIMARY = {K_PRIMARY}   K_GRID = {K_GRID}")
    print("=" * 80)

    summary = {}
    for strategy_name in STRATEGIES:
        print("\n" + "-" * 80)
        print(f"STRATEGY: {strategy_name}")
        print("-" * 80)
        row = run_strategy(strategy_name)
        if row is not None:
            summary[strategy_name] = row

    if summary:
        out = get_aen_output_dir() / "subset_summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"   {'Strategy':<20} {'k':>3} {'EBIC k':>7} {'R2adj(std)':>11}")
        for s, r in summary.items():
            print(f"   {s:<20} {r['n_selected']:>3} {r['ebic_k']:>7} {r['r2_adj_std']:>11.4f}")
        print(f"\n   💾 {out}")

    print("\n" + "=" * 80)
    print("✅ BEST-SUBSET SELECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

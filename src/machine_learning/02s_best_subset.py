"""
================================================================================
02s_best_subset.py — Best-Subset Selection (primary selector)
================================================================================

Primary factor-selection step for the spanning regressions. Replaces the AEN
(02_estimation.py) as the *primary* selector; the AEN is retained downstream as
a selection-method robustness check.

Reference
---------
  Bertsimas, King & Mazumder (2016), "Best Subset Selection via a Modern
      Optimization Lens", Annals of Statistics 44(2):813-852.
      ell0-constrained least squares: the cardinality constraint ||beta||_0 <= k
      directly controls the number of selected factors and, unlike the LASSO,
      brings a factor in without shrinkage — draining its correlated surrogates
      (among a correlated cluster it keeps the most informative member, not all).
      All-subsets model selection has asset-pricing precedent (Barillas-Shanken
      2018, J. Finance).

Cardinality
-----------
  k is FIXED at K_PRIMARY for cross-strategy comparability and parsimony, and the
  choice is defended by the k-sensitivity over K_GRID (reported in the paper).
  Data-driven count selection (EBIC, CV) is unstable on these weak-signal,
  alpha-dominated series — the objective curve is near-flat, so the argmin is
  ill-determined — and is therefore NOT used to pick k. This is documented in the
  paper's methodology section (it is a justification for fixing k, not a result),
  hence no information-criterion machinery runs here.

Design
------
  Reads the SAME preprocessed inputs as 02 (per strategy):
      <results>/<strategy>/aen/hqc/y_centered.parquet
      <results>/<strategy>/aen/hqc/X_standardized.parquet   (||X_j||_2 = 1)
  Selection is on the standardized design (scale-invariant for which factors
  enter). Point estimates / inference on RAW data are produced by 03 (post-
  selection OLS + HAC, plus valid post-selection inference).

  Writes (per strategy), schema-compatible with 02's aen_results.json so 03/07
  consume it once pointed at the new filename:
      subset_results.json       {selected_factors, n_selected, selected_coefficients, ...}
      subset_sensitivity.json   best-subset support at each k in K_GRID
  And an aggregate subset_summary.json.

Institution: EDHEC Business School — PhD Thesis
"""

import json
import importlib.util
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

# ── Parameters ────────────────────────────────────────────────────────────────
K_PRIMARY = 8            # cardinality of the primary spec; fixed by parsimony, robustness on K_GRID.
                         # Data-driven count (EBIC/CV) unstable on these weak-signal series — see paper methodology.
K_GRID    = (5, 8, 10)   # support sizes reported for the k-sensitivity table


# ── Best-subset of EXACT size k (ell0) ─────────────────────────────────────────
def _forward_k(Xz, y, k):
    """Greedy forward selection forced to exactly k (min RSS at each step)."""
    chosen, p = [], Xz.shape[1]
    while len(chosen) < k:
        best_rss, best_j = np.inf, -1
        for j in range(p):
            if j in chosen:
                continue
            r = _rss(Xz, y, chosen + [j])
            if r < best_rss:
                best_rss, best_j = r, j
        chosen.append(best_j)
    return sorted(chosen)


def best_subset(Xz, y, k):
    """ell0 best subset of size k: abess splicing (Zhu et al. 2020, PNAS)
    POLISHED against greedy forward search. Splicing can land on a local
    optimum, so the feasible k-subset with the lower in-sample RSS is kept
    and the engine label records which one won. Raises if abess is
    unavailable or does not honour k: the declared solver must run.
    """
    try:
        import abess
        from abess.linear import LinearRegression
    except ImportError as e:
        raise RuntimeError(
            "abess is required for the ell0 best-subset selection "
            "(pip install abess); no silent fallback is provided by design."
        ) from e
    fwd = _forward_k(Xz, y, k)
    model = LinearRegression(support_size=k, fit_intercept=False)
    model.fit(Xz, y)
    ab = sorted(np.where(np.abs(model.coef_) > 0)[0].tolist())
    if len(ab) != k:
        raise RuntimeError(
            f"abess returned {len(ab)} factors for k={k}: the ell0 solver "
            f"did not honour the cardinality constraint."
        )
    ver = getattr(abess, "__version__", "unknown")
    if _rss(Xz, y, ab) <= _rss(Xz, y, fwd):
        return ab, f"abess(ell0) v{ver}"
    return fwd, f"ell0: greedy beat splicing v{ver} (best-RSS kept)"


# ── OLS helpers on the standardized design (the JSON; 03 is authoritative) ─────
def _rss(Xz, y, idx):
    if not idx:
        return float(y @ y)
    A = Xz[:, idx]
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.sum((y - A @ beta) ** 2))


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

    print(f"\n   {strategy_name}: T={T}, p={p}")

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
    }
    with open(strategy_dir / "subset_results.json", "w") as f:
        json.dump(subset_results, f, indent=2)

    with open(strategy_dir / "subset_sensitivity.json", "w") as f:
        json.dump({"k_grid": {str(k): sensitivity[k] for k in K_GRID}}, f, indent=2)

    print(f"   💾 subset_results.json ({len(selected)} factors), subset_sensitivity.json")
    return {
        "strategy": strategy_name,
        "n_selected": len(selected),
        "selected_factors": selected,
        "r2_adj_std": subset_results["r2_adj_std"],
    }


def main():
    print("=" * 80)
    print("BEST-SUBSET SELECTION (primary)")
    print("  Bertsimas-King-Mazumder (2016, Ann. Stat.)")
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
        print(f"   {'Strategy':<20} {'k':>3} {'R2adj(std)':>11}")
        for s, r in summary.items():
            print(f"   {s:<20} {r['n_selected']:>3} {r['r2_adj_std']:>11.4f}")
        print(f"\n   💾 {out}")

    print("\n" + "=" * 80)
    print("✅ BEST-SUBSET SELECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

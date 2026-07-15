"""
================================================================================
04_bootstrap.py — Block-bootstrap inference on the best-subset set
================================================================================

ROLE
----
Inference (NOT selection) on the PRIMARY factor set chosen by 02s
(best-subset, Bertsimas-King-Mazumder 2016). The block bootstrap is recycled
purely to obtain resampling-based standard errors / confidence intervals for
the post-selection alpha and factor loadings — complementary to the analytic
HAC inference (03) and to the double-selection inference.

WHY M&B STABILITY SELECTION WAS REMOVED FROM HERE
-------------------------------------------------
Meinshausen & Bühlmann (2010) stability selection is an FWER-control device
(high precision, low recall): it is built for SPARSE recovery and, by
construction, discards genuinely explanatory factors when the true model is
dense and collinear (here iTraxx: ~10 significant loadings collapse to ~0
under the q-capped λ*). That objective is the OPPOSITE of the paper's RQ2
("which factors best EXPLAIN"). Keeping it as a headline robustness check
would contradict the main result for a known mechanical reason and confuse the
reader. It is therefore not used in this paper; the robustness evidence is the
cross-method bake-off (05) + this bootstrap inference + the OOS test (06).
METHOD
------
Block bootstrap (Politis & Romano 1994, stationary; Künsch 1989, circular) on
the EXACT aligned (y, X) used by 03 (read from regression_inputs.parquet, so
the bootstrap distribution is anchored to 03's point estimate). For each
replicate: resample blocks, refit OLS of y on [1, X_selected], record the
intercept (alpha) and the factor loadings. Report bootstrap SE, 95% percentile
CI, and a two-sided bootstrap p-value (centered null).

References
----------
  Politis & Romano (1994), JASA 89(428), 1303-1313.   (stationary bootstrap)
  Künsch (1989), Ann. Statist. 17(3), 1217-1241.       (block bootstrap)

OUTPUT
------
  bootstrap_stability.json   stable_factors (= best-subset set; consumed by
                             06_aen_oos / 06e / aen_postselection) + bootstrap
                             inference for alpha and each loading.
  bootstrap_summary.json     one-line-per-strategy aggregate.

Institution: EDHEC Business School — PhD Thesis
"""

import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# ── Config ──────────────────────────────────────────────────────────────────
_CFG_PATH = Path(__file__).resolve().parent / "00_config.py"
_spec = importlib.util.spec_from_file_location("aen_config", _CFG_PATH)
aen_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aen_config)

STRATEGIES             = aen_config.STRATEGIES
get_aen_output_dir     = aen_config.get_aen_output_dir
get_strategy_aen_dir   = aen_config.get_strategy_aen_dir
HAC_LAGS               = aen_config.HAC_LAGS
BOOTSTRAP_METHOD       = aen_config.BOOTSTRAP_METHOD
BOOTSTRAP_N_REPS       = aen_config.BOOTSTRAP_N_REPS        # raise to >=999 for publication CIs
BOOTSTRAP_BLOCK_LENGTH = aen_config.BOOTSTRAP_BLOCK_LENGTH

RANDOM_SEED = 12345


# ── Block bootstrap index generators ──────────────────────────────────────────
def circular_block_bootstrap(T, block_length, rng):
    """Circular block bootstrap (Künsch 1989; Politis & Romano 1992)."""
    block_length = int(block_length)
    n_blocks = int(np.ceil(T / block_length))
    starts = rng.integers(0, T, size=n_blocks)
    idx = []
    for s in starts:
        idx.extend([(s + j) % T for j in range(block_length)])
    return np.asarray(idx[:T], dtype=int)


def stationary_bootstrap(T, block_length, rng):
    """Stationary bootstrap (Politis & Romano 1994). E[block length] = block_length."""
    p_new = 1.0 / float(block_length)
    idx = np.empty(T, dtype=int)
    idx[0] = rng.integers(0, T)
    for t in range(1, T):
        if rng.random() < p_new:
            idx[t] = rng.integers(0, T)
        else:
            idx[t] = (idx[t - 1] + 1) % T
    return idx


def boot_indices(T, block_length, method, rng):
    if method == "stationary":
        return stationary_bootstrap(T, block_length, rng)
    return circular_block_bootstrap(T, block_length, rng)


# ── Bootstrap summary of one estimator ─────────────────────────────────────────
def _summarise(point, draws):
    """Bootstrap SE, 95% percentile CI, two-sided centered bootstrap p-value."""
    se = float(np.std(draws, ddof=1))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    centered = draws - point                     # null distribution of the estimator
    p = float(np.mean(np.abs(centered) >= abs(point)))
    return se, float(lo), float(hi), p


# ── Per-strategy driver ─────────────────────────────────────────────────────────
def run_strategy(strategy_name):
    sd = get_strategy_aen_dir(strategy_name)

    sub_path = sd / "subset_results.json"
    ri_path = sd / "regression_inputs.parquet"
    if not sub_path.exists():
        print(f"   ❌ subset_results.json missing for {strategy_name}. Run 02s first.")
        return None
    if not ri_path.exists():
        print(f"   ❌ regression_inputs.parquet missing for {strategy_name}. Run 03 first.")
        return None

    selected = json.load(open(sub_path))["selected_factors"]
    ri = pd.read_parquet(ri_path)
    y = ri["__y__"].to_numpy(dtype=float)
    X = ri.drop(columns="__y__")
    factor_names = list(X.columns)
    Xc = sm.add_constant(X.to_numpy(dtype=float), prepend=True)   # column 0 = intercept
    T = len(y)

    print(f"\n   {strategy_name}: T={T}, k={len(factor_names)}  "
          f"[{BOOTSTRAP_METHOD} bootstrap, B={BOOTSTRAP_N_REPS}, L={BOOTSTRAP_BLOCK_LENGTH}]")

    # point estimate (HAC) — matches 03
    res = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    alpha_hat = float(res.params[0])
    coef_hat = {factor_names[j]: float(res.params[j + 1]) for j in range(len(factor_names))}

    # bootstrap replicates
    rng = np.random.default_rng(RANDOM_SEED)
    alpha_boot = np.empty(BOOTSTRAP_N_REPS)
    coef_boot = {n: np.empty(BOOTSTRAP_N_REPS) for n in factor_names}
    for b in range(BOOTSTRAP_N_REPS):
        idx = boot_indices(T, BOOTSTRAP_BLOCK_LENGTH, BOOTSTRAP_METHOD, rng)
        beta = np.linalg.lstsq(Xc[idx], y[idx], rcond=None)[0]
        alpha_boot[b] = beta[0]
        for j, n in enumerate(factor_names):
            coef_boot[n][b] = beta[j + 1]

    a_se, a_lo, a_hi, a_p = _summarise(alpha_hat, alpha_boot)

    out = {
        "method": "block-bootstrap inference on best-subset set (Politis-Romano 1994)",
        "bootstrap_method": BOOTSTRAP_METHOD,
        "n_reps": BOOTSTRAP_N_REPS,
        "block_length": BOOTSTRAP_BLOCK_LENGTH,
        "stable_factors": selected,          # consumed by 06_aen_oos / 06e / aen_postselection
        "n_stable": len(selected),
        "alpha": {
            "point_monthly": round(alpha_hat, 6),
            "point_annualized_pct": round(alpha_hat * 12, 4),
            "boot_se_monthly": round(a_se, 6),
            "ci95_annualized_pct": [round(a_lo * 12, 4), round(a_hi * 12, 4)],
            "boot_pvalue": round(a_p, 4),
        },
        "factors": {},
    }
    for n in factor_names:
        c_se, c_lo, c_hi, c_p = _summarise(coef_hat[n], coef_boot[n])
        out["factors"][n] = {
            "coefficient": round(coef_hat[n], 6),
            "boot_se": round(c_se, 6),
            "ci95": [round(c_lo, 6), round(c_hi, 6)],
            "boot_pvalue": round(c_p, 4),
        }

    with open(sd / "bootstrap_stability.json", "w") as f:
        json.dump(out, f, indent=2)

    star = "***" if a_p < 0.01 else "**" if a_p < 0.05 else "*" if a_p < 0.10 else ""
    print(f"      α = {alpha_hat * 12:+.2f}%/yr  "
          f"95% CI [{a_lo * 12:+.2f}, {a_hi * 12:+.2f}]  boot-p={a_p:.4f} {star}")
    print(f"      💾 bootstrap_stability.json")
    return {
        "strategy": strategy_name,
        "n_stable": len(selected),
        "alpha_annualized_pct": round(alpha_hat * 12, 4),
        "alpha_ci95_annualized_pct": [round(a_lo * 12, 4), round(a_hi * 12, 4)],
        "alpha_boot_pvalue": round(a_p, 4),
    }


def main():
    print("=" * 80)
    print("BLOCK-BOOTSTRAP INFERENCE ON BEST-SUBSET SET")
    print(f"  {BOOTSTRAP_METHOD} bootstrap (Politis-Romano 1994), B={BOOTSTRAP_N_REPS}, "
          f"L={BOOTSTRAP_BLOCK_LENGTH}")
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
        out = get_aen_output_dir() / "bootstrap_summary.json"
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print("\n" + "=" * 80)
        print("SUMMARY (alpha annualized)")
        print("=" * 80)
        print(f"   {'Strategy':<20} {'α/yr':>8} {'95% CI':>22} {'boot-p':>8}")
        for s, r in summary.items():
            lo, hi = r["alpha_ci95_annualized_pct"]
            print(f"   {s:<20} {r['alpha_annualized_pct']:>+7.2f}%"
                  f"   [{lo:+.2f}, {hi:+.2f}]   {r['alpha_boot_pvalue']:>6.4f}")
        print(f"\n   💾 {out}")

    print("\n" + "=" * 80)
    print("✅ BOOTSTRAP INFERENCE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

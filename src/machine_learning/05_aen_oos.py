# -*- coding: utf-8 -*-
"""
BEST-SUBSET OUT-OF-SAMPLE ALPHA  (alpha_0, the unconditional level)
==================================================================

Implementability test for the unconditional alpha of the primary (best-subset)
set, the direct analogue of the factor-benchmark Panel D (oos.py).  The in-
sample alpha (06e) is a full-sample OLS intercept on the selected set: the
hedge loadings use the whole period, so the implied hedge is formed with
hindsight (Moreira-Muir in-sample-weights critique; Tessaromatis
implementability).  Here we FREEZE the best-subset set (02s) and re-estimate
ONLY the loadings on a PAST-ONLY window, evaluating the realised abnormal
return on the next month:

        e_mod_t = r_t - beta_hat'_{t-1} F_t          (slopes only)

NO penalised selection is run here.  The factors are the best-subset set
(fixed, named series read from bootstrap_stability.json), so there is no per-
window re-selection and no rotation/sign ambiguity (unlike the PCA OOS).  The
estimand is the FROZEN-set loadings-OOS alpha_0 -- the fair counterpart of the
benchmark Panel D and the PCA OOS, computed with the SAME engine
(oos.oos_residuals / oos.summarize) so the three layers sit on identical ground.

  Conservative asymmetry: a past-only beta explains (weakly) LESS than the
  full-sample beta, so the OOS alpha_0 is expected at or slightly ABOVE the
  in-sample level -- the same direction as the benchmark and PCA layers.

  Scope: this is alpha_0 (the level).  The CONDITIONAL stress slope alpha_1 is
  NOT covered here -- the selection is on the unconditional model and does not
  protect alpha_1, which requires valid post-selection inference
  (post-double-selection, Belloni-Chernozhukov-Hansen 2014).

Author: Alessio Ottaviani, EDHEC Business School.
"""

# ============================================================================
#  OOS schemes -- both run in one pass (expanding primary + rolling-60),
#  matching the benchmark Panel D and the PCA OOS.
# ============================================================================
OOS_SCHEMES     = ("expanding", "rolling")
OOS_MIN_TRAIN   = 60              # months before the first OOS evaluation
OOS_ROLL_WINDOW = 60              # rolling-window length (rolling scheme)
COND_MAX        = 1e3             # skip ill-conditioned windows (same guard as Panel D)

from pathlib import Path
import json
import importlib.util
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#  Bootstrap project paths + shared modules (same mechanism as 06e).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_cfg_spec = importlib.util.spec_from_file_location(
    "aen_config", str(PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"))
aen_config = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(aen_config)

_oos_spec = importlib.util.spec_from_file_location(
    "oos", str(PROJECT_ROOT / "src" / "factor_models" / "oos.py"))
oos = importlib.util.module_from_spec(_oos_spec)
_oos_spec.loader.exec_module(oos)

FACTORS_PATH     = aen_config.FACTORS_PATH
FACTORS_END_DATE = aen_config.FACTORS_END_DATE
STRATEGIES       = aen_config.STRATEGIES

RESULTS_OUT = PROJECT_ROOT / "results" / "machine_learning"
RESULTS_OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
#  Data loading -- mirrored EXACTLY from 06e so the sample matches the
#  in-sample AEN alpha (same returns, same factor scale, same alignment).
# ---------------------------------------------------------------------------
def load_strategy_returns(strategy_path):
    daily_df = pd.read_csv(strategy_path, index_col=0, parse_dates=True)
    daily_returns = daily_df["index_return"].dropna()
    monthly = daily_returns.resample("ME").apply(
        lambda x: ((1 + x / 100).prod() - 1) * 100 if len(x) > 0 else np.nan
    )
    return monthly.dropna()


def load_stable_factors(strategy_name):
    """Stable factor list from bootstrap_stability.json (NO LASSO is run here)."""
    strategy_dir = aen_config.get_strategy_aen_dir(strategy_name)
    stability_path = strategy_dir / "bootstrap_stability.json"
    if not stability_path.exists():
        raise FileNotFoundError(f"Missing {stability_path}")
    stability = json.loads(stability_path.read_text(encoding="utf-8"))
    stable_factors = stability.get("stable_factors", [])
    if not stable_factors:
        raise ValueError(f"No stable_factors for {strategy_name}")
    return stable_factors


def prepare_data(strategy_name, strategy_path):
    """Returns (y, X, stable_factors) aligned on common dates, NaN-free."""
    stable_factors = load_stable_factors(strategy_name)
    all_factors = pd.read_parquet(FACTORS_PATH)
    factors_end = pd.Timestamp(FACTORS_END_DATE)
    all_factors = all_factors[(all_factors.index >= pd.Timestamp(aen_config.AEN_START_DATE)) &
                              (all_factors.index <= factors_end)]

    returns = load_strategy_returns(strategy_path)
    returns = returns[returns.index <= factors_end]

    common = returns.index.intersection(all_factors.index)
    y = returns.loc[common]
    X = all_factors.loc[common][stable_factors].copy()

    mask = ~(X.isna().any(axis=1) | y.isna())
    return y[mask], X[mask], stable_factors


# ---------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------
def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def run_scheme(mode):
    """One OOS scheme (expanding | rolling); prints, saves JSON, returns out dict."""
    win_note = (f"rolling {OOS_ROLL_WINDOW}m" if mode == "rolling"
                else f"expanding, min_train={OOS_MIN_TRAIN}")
    print(f"\n  ── {mode.upper()} ──  ({win_note} | hedge = frozen best-subset set, "
          f"slopes only | HAC = NW(1994))")
    print("-" * 92)

    out = {}
    for name, path in STRATEGIES.items():
        y, X, stable = prepare_data(name, path)
        e_mod, e_naive, info = oos.oos_residuals(
            y, X, scheme=mode, min_window=OOS_MIN_TRAIN,
            roll_window=OOS_ROLL_WINDOW, cond_max=COND_MAX)
        s   = oos.summarize(e_mod)
        ins = oos.insample_alpha(y, X)             # full-sample intercept (= 06e level)
        r2  = oos.oos_r2(e_mod, e_naive)

        skip = f", {info['n_skipped']} skipped" if info["n_skipped"] else ""
        print(f"  {name:<18s} alpha_OOS = {s['alpha_ann']:+6.2f}% ann   "
              f"t = {s['t']:5.2f}   p = {s['p']:.4f} {stars(s['p']):<3s}   "
              f"[in-sample {ins['alpha_ann']:+.2f}%, k={len(stable)}, "
              f"n={info['n_oos']}{skip}]")

        out[name] = {
            "alpha_oos_ann": s["alpha_ann"], "t": s["t"], "p": s["p"],
            "ir": s["ir"], "oos_r2": r2,
            "alpha_insample_ann": ins["alpha_ann"], "t_insample": ins["t"],
            "n_oos": info["n_oos"], "n_total": info["n_total"],
            "n_skipped": info["n_skipped"], "k_stable": len(stable),
            "stable_factors": stable, "scheme": mode,
            "min_train": OOS_MIN_TRAIN, "roll_window": OOS_ROLL_WINDOW,
        }

    # ---- Panel D block (paste-ready, parallel to benchmark & PCA) ----------
    print("-" * 60)
    print(f"  Panel D (best-subset, {mode}) — annualised alpha, t in ()")
    print("-" * 60)
    for name in STRATEGIES:
        r = out[name]
        print(f"   {name:<20s} {r['alpha_oos_ann']:+6.2f}{stars(r['p']):<3s}  "
              f"({r['t']:.2f})")

    out_path = RESULTS_OUT / f"aen_oos_alpha_{mode}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"  Saved: {out_path}")
    return out


def main():
    bar = "=" * 92
    print(bar)
    print("BEST-SUBSET OUT-OF-SAMPLE ALPHA  (alpha_0, frozen set, past-only beta)")
    print(bar)
    for mode in OOS_SCHEMES:
        run_scheme(mode)
    print("\nNotes:")
    print("  - Both schemes in one pass: expanding (primary) + rolling-60 (robustness),")
    print("    matching the benchmark Panel D and the PCA OOS.")
    print("  - Same engine as the benchmark (oos.oos_residuals): loadings re-estimated on")
    print("    the past window, slopes-only residual, NW(1994) HAC. NO penalised")
    print("    selection, NO PCA — the best-subset set is frozen.")
    print("  - This is alpha_0 (the level). The conditional stress slope alpha_1 is handled")
    print("    separately (post-double-selection, Belloni-Chernozhukov-Hansen 2014).")


if __name__ == "__main__":
    main()

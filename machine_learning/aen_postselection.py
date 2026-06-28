# -*- coding: utf-8 -*-
"""
aen_postselection.py  (v3)
==========================
DIAGNOSTIC (writes nothing into the paper directly): valid post-selection
inference on the AEN CONDITIONAL stress slope alpha_1, i.e. the coefficient on
the (predetermined) stress variable z in the conditional model

        r_t = alpha_0 + alpha_1 z_{t-1}
              + sum_j (beta_j + delta_j z_{t-1}) X_{j,t} + eps_t .

WHY THIS MODULE EXISTS (and what changed in v3)
-----------------------------------------------
The conservative-selection asymmetry protects alpha_0 (the level): the AEN
selects the controls X_S on the UNCONDITIONAL model (minimising sum (r-b'X)^2,
no z, no interactions), so the controls absorb the LEVEL, and a surviving
alpha_0 is conservative.  That asymmetry does NOT protect alpha_1: selection was
not done to absorb the stress dependence, so alpha_1 needs explicit valid
post-selection inference.

  * alpha_0 (the LEVEL) is no longer handled here.  Its implementability test is
    the loadings-OOS on the frozen stable set, computed in
        src/machine_learning/06_aen_oos.py
    with the SAME expanding/rolling engine as the factor-benchmark Panel D.
    The v1/v2 fresh-LASSO-from-the-full-pool cross-fit is DROPPED: it re-did the
    selection over ~80 candidates and was an apples-to-oranges, harsher object.

  * alpha_1 (the STRESS slope) is covered here by TWO valid procedures that
    AGREE, on z_{t-1} (predetermined, canonical CFG timing):

      (1) PDS  -- post-double-selection LASSO (Belloni-Chernozhukov-Hansen 2014).
          HEADLINE.  Uniform validity (robust to selection mistakes; avoids the
          Leeb-Poetscher 2005/2008 non-uniformity).  Selection debiases the
          CONTROLS that predict r and that predict z; z itself is fixed/exogenous
          (the iTraxx Main level, a conditioning choice, NOT selected).  The
          double-selection LASSO is the STANDARD textbook inner selector, not the
          home-grown one.  Final SE on alpha_1 is HAC + moving-block bootstrap
          (returns are autocorrelated; BCH theory is iid/weakly dependent, so the
          point set is BCH-valid but the SE must be serial-dependence robust).

      (2) RE-SELECTION BOOTSTRAP -- coherent cross-check that reuses the paper's
          OWN selection machine (bootstrap_stability.py), NO new LASSO.  Each
          block resample re-runs the AEN selection (adaptive-weight Stage-1
          weights and lambda2 fixed at their full-sample values exactly as the
          stability-selection bootstrap fixes them; the Stage-2 lambda1 re-tuned
          per resample by the same HQC information criterion), then fits the CFG
          and records alpha_1.  The distribution of alpha_1 across resamples
          integrates selection AND sampling uncertainty (Efron 2014, "bootstrap
          the whole procedure").  CAVEAT: an n-out-of-n bootstrap after model
          selection is NOT uniformly consistent near the null (Leeb-Poetscher) --
          it can under-cover for borderline cases -- which is why it is a
          cross-check, not the headline.  For a clearly-separated signal (CDS) it
          confirms PDS; for the null cases (BTP, iTraxx) it agrees they are ns.

  z TIMING:  z_{t-1} (lagged, predetermined) is the HEADLINE for all three
  alpha_1 measures, matching the conditional-alpha (CFG) sections (03/05b/06e).
  z_t (contemporaneous) is reported as a robustness line for in-sample and PDS.

Refs: Christopherson, Ferson & Glassman (1998, RFS); Belloni, Chernozhukov &
Hansen (2014, ReStud); Chernozhukov et al. (2018, Econ. J.); Leeb & Poetscher
(2005, ET; 2008, JoE); Efron (2014, JASA); Meinshausen & Buehlmann (2010, JRSS-B);
Du, Walter & Ulrich (2026).

Run:  python src/machine_learning/aen_postselection.py
"""
import json
import time
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# -- load AEN config (mirror 06e) ------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_cfg_path = PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"
_spec = importlib.util.spec_from_file_location("aen_config", _cfg_path)
aen_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aen_config)

FACTORS_PATH         = aen_config.FACTORS_PATH
FACTORS_END_DATE     = aen_config.FACTORS_END_DATE
STRATEGIES           = aen_config.STRATEGIES
HAC_LAGS             = aen_config.HAC_LAGS
get_strategy_aen_dir = aen_config.get_strategy_aen_dir
FACTORS_TO_EXCLUDE   = getattr(aen_config, "FACTORS_TO_EXCLUDE", [])

DATA_DIR = PROJECT_ROOT / "data"
TRADABLE_CB_FILE = DATA_DIR / "external" / "factors" / "Tradable_corporate_bond_factors.xlsx"

# -- shared bootstrap module (inference.py) for the CFG alpha_1 bootstrap ---
_inf_path = PROJECT_ROOT / "src" / "factor_models" / "inference.py"
_ispec = importlib.util.spec_from_file_location("inference", _inf_path)
_inf = importlib.util.module_from_spec(_ispec)
_ispec.loader.exec_module(_inf)
cfg_alpha1_bootstrap_p = _inf.cfg_alpha1_bootstrap_p
moving_block_pvalue    = _inf._moving_block_bootstrap_pvalue
STAB_BLOCK = _inf.STABILITY_BLOCK
BOOT_B     = _inf.BOOTSTRAP_B

# -- the paper's OWN AEN machine = 04_bootstrap.py, the SAME engine that ----
#    produces bootstrap_stability.json. Re-used verbatim for the re-selection
#    bootstrap: NO new estimator, NO second copy of the engine.
_bs_path = PROJECT_ROOT / "src" / "machine_learning" / "04_bootstrap.py"
_bspec = importlib.util.spec_from_file_location("bootstrap_engine", _bs_path)
bs = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(bs)
compute_weights_and_grid  = bs._compute_weights_and_grid
weighted_elastic_net_cd   = bs.weighted_elastic_net_cd
build_lambda1_grid        = bs.build_lambda1_grid
compute_ic                = bs.compute_ic
gen_boot_idx              = bs.generate_bootstrap_indices
COEF_TOL                  = bs.COEF_TOL
AEN_LAMBDA2_GRID          = bs.AEN_LAMBDA2_GRID
AEN_LAMBDA1_N_VALUES      = bs.AEN_LAMBDA1_N_VALUES
AEN_GAMMA                 = bs.AEN_GAMMA
AEN_CRITERION             = bs.AEN_TUNING_CRITERION
GIC_ALPHA                 = bs.GIC_ALPHA
BOOT_METHOD               = bs.BOOTSTRAP_METHOD
BOOT_BLOCK                = bs.BOOTSTRAP_BLOCK_LENGTH

# -- configuration ----------------------------------------------------------
CANDIDATE        = "post_sis"   # "post_sis" or "full"  (AEN candidate pool)
CV_FOLDS         = 5            # PDS inner LassoCV folds
PERIODS_PER_YEAR = 12
RNG_SEED         = 0
TOL              = 1e-10        # PDS LASSO coefficient tolerance

# re-selection bootstrap controls (offline diagnostic; a few minutes/strategy)
RESEL_B          = 299          # bootstrap replications (bump to 999 for the final run)
RESEL_N_LAMBDA1  = 50           # lambda1 grid per resample (HQC-tuned); paper's bootstrap uses ~50
RESEL_PROGRESS   = 100          # print a heartbeat every this many reps


def _stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# -- data loaders (mirror 06e) ---------------------------------------------
def load_strategy_returns(strategy_path):
    daily = pd.read_csv(strategy_path, index_col=0, parse_dates=True)["index_return"].dropna()
    monthly = daily.resample("ME").apply(
        lambda x: ((1 + x / 100).prod() - 1) * 100 if len(x) > 0 else np.nan)
    return monthly.dropna()


def load_z_itrx_main():
    raw = pd.read_excel(TRADABLE_CB_FILE, sheet_name="CDS_INDEX",
                        skiprows=14, usecols=[0, 1], header=0)
    raw.columns = ["Date", "value"]
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date"]).set_index("Date")
    daily = pd.to_numeric(raw["value"], errors="coerce").dropna()
    return daily.resample("ME").last().dropna()


def candidate_names(strategy_name, all_factors):
    if CANDIDATE == "post_sis":
        xpath = get_strategy_aen_dir(strategy_name) / "X_standardized.parquet"
        if not xpath.exists():
            raise FileNotFoundError(
                f"{xpath} not found - run 01_preprocessing.py first (or set CANDIDATE='full').")
        cols = pd.read_parquet(xpath).columns.tolist()
        return [c for c in cols if c in all_factors.columns]
    drop = set(FACTORS_TO_EXCLUDE)
    return [c for c in all_factors.columns if c not in drop]


def load_stable_factors(strategy_name):
    p = get_strategy_aen_dir(strategy_name) / "bootstrap_stability.json"
    return json.loads(p.read_text(encoding="utf-8")).get("stable_factors", [])


def prepare(strategy_name, strategy_path):
    """
    Returns y, F (raw candidate factors), z_c (contemporaneous, standardised),
    z_l (z_{t-1}, standardised), Fst (raw stable set), cand, stable -- all aligned
    on the common sample, NaN-free.  Lagging z drops the first observation; both
    z series are standardised on the post-drop sample (unit sd -> alpha_1 is per
    1 sd of z, comparable across timings).
    """
    allf = pd.read_parquet(FACTORS_PATH)
    allf = allf[allf.index <= pd.Timestamp(FACTORS_END_DATE)]
    y = load_strategy_returns(strategy_path)
    y = y[y.index <= pd.Timestamp(FACTORS_END_DATE)]

    cand = candidate_names(strategy_name, allf)
    stable = [f for f in load_stable_factors(strategy_name) if f in allf.columns]
    z_raw = load_z_itrx_main()

    common = y.index.intersection(allf.index)
    y = y.loc[common]
    F = allf.loc[common, cand].copy()
    good = F.columns[F.notna().all()].tolist()
    cand = [c for c in cand if c in good]
    F = F[cand]

    z = z_raw.reindex(y.index, method="nearest").astype(float)   # contemporaneous level
    z_lag = z.shift(1)                                           # z_{t-1}

    df = pd.concat([y.rename("__y__"), F,
                    z.rename("__zc__"), z_lag.rename("__zl__")], axis=1).dropna()
    y = df["__y__"]; F = df[cand]
    zc = df["__zc__"]; zc = (zc - zc.mean()) / zc.std()
    zl = df["__zl__"]; zl = (zl - zl.mean()) / zl.std()
    stable = [f for f in stable if f in df.columns]
    Fst = df[stable]
    return y, F, zc, zl, Fst, cand, stable


# -- helpers ----------------------------------------------------------------
def _lasso_mask(X, yv, seed=RNG_SEED):
    Xs = StandardScaler().fit_transform(X)
    m = LassoCV(cv=CV_FOLDS, alphas=100, max_iter=50000, random_state=seed).fit(Xs, yv)
    return np.abs(m.coef_) > TOL


def _hac(y, X, idx):
    """HAC OLS; return (coef, t, p) of regressor at column `idx`."""
    r = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    return float(r.params[idx]), float(r.tvalues[idx]), float(r.pvalues[idx])


# -- alpha_0 (unconditional level, in-sample reference) --------------------
def unconditional_alpha0(y, Fst):
    yv = y.to_numpy(); Xk = Fst.to_numpy()
    a0, a0_t, a0_p = _hac(yv, sm.add_constant(Xk), 0)
    return {"alpha0_ann": a0 * PERIODS_PER_YEAR, "t": a0_t, "p": a0_p, "k": Xk.shape[1]}


# -- alpha_1 (CFG conditional slope) on the STABLE set, in-sample ----------
def insample_alpha1(y, Fst, z):
    yv = y.to_numpy(); Xk = Fst.to_numpy(); zv = z.to_numpy(); k = Xk.shape[1]
    Xc = np.column_stack([np.ones_like(yv), Xk, zv, Xk * zv[:, None]])
    a1, a1_t, a1_phac = _hac(yv, Xc, 1 + k)
    a1_pboot = cfg_alpha1_bootstrap_p(yv, Xk, zv)
    return {"alpha1": a1, "t": a1_t, "p_hac": a1_phac, "p_boot": a1_pboot}


# -- alpha_1 PDS (BCH 2014 double selection) -------------------------------
def pds_alpha1(y, F, z):
    yv = y.to_numpy(); zv = z.to_numpy(); Fv = F.to_numpy()
    W = np.column_stack([Fv, Fv * zv[:, None]])                 # controls + interactions
    s_y = _lasso_mask(W, yv); s_z = _lasso_mask(W, zv); S = s_y | s_z
    cols = [W[:, j] for j in range(W.shape[1]) if S[j]]
    Xf = (np.column_stack([np.ones_like(yv), zv] + cols)
          if cols else np.column_stack([np.ones_like(yv), zv]))
    a1, a1_t, a1_phac = _hac(yv, Xf, 1)
    a1_pboot = moving_block_pvalue(Xf, yv, z_idx=1, block=STAB_BLOCK, B=BOOT_B, seed=RNG_SEED)
    return {"alpha1": a1, "t": a1_t, "p_hac": a1_phac, "p_boot": a1_pboot,
            "n_sel_y": int(s_y.sum()), "n_sel_z": int(s_z.sum()),
            "n_union": int(S.sum()), "n_controls": int(W.shape[1])}


# -- re-selection bootstrap for alpha_1 (reuses the paper's AEN machine) ----
def _reselect_aen(y_b, F_b, weights_fs, lambda2_fs, lambda1_grid):
    """
    Re-run the AEN selection on one block resample.  Adaptive weights and
    lambda2 are FIXED at their full-sample values (exactly as the stability-
    selection bootstrap fixes them); lambda1 is re-tuned by the HQC criterion
    on this resample.  Returns the selected column indices (into F_b's columns).
    Re-centres y and re-L2-normalises X internally, identical to _aen_boot_cell.
    """
    y_c = y_b - y_b.mean()
    Xc = F_b - F_b.mean(axis=0)
    l2 = np.sqrt((Xc ** 2).sum(axis=0)); l2[l2 < 1e-12] = 1.0
    X_s = Xc / l2
    T = len(y_c)
    best_ic, best_sel = np.inf, np.array([], dtype=int)
    for lam1 in lambda1_grid:
        beta = weighted_elastic_net_cd(X_s, y_c, lam1, lambda2_fs, weights_fs,
                                       max_iter=10000, tol=1e-7)
        df = int(np.sum(np.abs(beta) > COEF_TOL))
        ic, _ = compute_ic(y_c, X_s @ beta, T, df, AEN_CRITERION, GIC_ALPHA)
        if ic < best_ic:
            best_ic = ic
            best_sel = np.where(np.abs(beta) > COEF_TOL)[0]
    return best_sel


def _cfg_alpha1_point(y_b, Xk_b, z_b):
    """OLS alpha_1 (no HAC) on one resample's design [1, Xk, z, Xk*z]."""
    if Xk_b.shape[1] == 0:
        X = np.column_stack([np.ones_like(y_b), z_b])
        beta, *_ = np.linalg.lstsq(X, y_b, rcond=None)
        return float(beta[1])
    X = np.column_stack([np.ones_like(y_b), Xk_b, z_b, Xk_b * z_b[:, None]])
    beta, *_ = np.linalg.lstsq(X, y_b, rcond=None)
    return float(beta[1 + Xk_b.shape[1]])


def reselection_bootstrap_alpha1(y, F, z, cand, alpha1_point,
                                  B=RESEL_B, n_lambda1=RESEL_N_LAMBDA1, seed=RNG_SEED):
    """
    Bootstrap the AEN-select-then-CFG procedure and read off alpha_1 each time.
    alpha1_point is the full-sample procedure value (CFG on the stable set), used
    only as the reported point estimate; the CI/p come from the resample
    distribution (percentile method).
    """
    yv = y.to_numpy(); Fv = F.to_numpy(); zv = z.to_numpy(); T = len(yv)

    # full-sample weights / lambda2 / lambda1 range (the AEN's own tuning)
    y_c = yv - yv.mean()
    Xc = Fv - Fv.mean(axis=0)
    l2 = np.sqrt((Xc ** 2).sum(axis=0)); l2[l2 < 1e-12] = 1.0
    X_full = Xc / l2
    w_fs, lam2_fs, _lam1_opt, _g, lam1_max, _b, fs_sel = compute_weights_and_grid(
        X_full, y_c, lambda2_grid=AEN_LAMBDA2_GRID, n_lambda1=AEN_LAMBDA1_N_VALUES,
        criterion=AEN_CRITERION, gamma=AEN_GAMMA, gic_alpha=GIC_ALPHA)
    lambda1_grid = build_lambda1_grid(lam1_max, n_lambda1)
    fs_sel_names = [cand[j] for j in fs_sel]

    rng = np.random.default_rng(seed)
    a1_boot, sel_sizes, n_empty = [], [], 0
    t0 = time.time()
    for b in range(B):
        idx = gen_boot_idx(T, BOOT_BLOCK, BOOT_METHOD, rng)
        y_b = yv[idx]; F_b = Fv[idx]
        z_b = zv[idx]; z_b = (z_b - z_b.mean()) / (z_b.std() + 1e-12)
        sel = _reselect_aen(y_b, F_b, w_fs, lam2_fs, lambda1_grid)
        sel_sizes.append(len(sel))
        if len(sel) == 0:
            n_empty += 1
        a1_boot.append(_cfg_alpha1_point(y_b, F_b[:, sel], z_b))
        if RESEL_PROGRESS and (b + 1) % RESEL_PROGRESS == 0:
            print(f"        ... {b + 1}/{B} reps ({time.time() - t0:.0f}s)", flush=True)

    arr = np.asarray(a1_boot, float)
    p_perc = min(2.0 * min((arr <= 0).mean(), (arr >= 0).mean()), 1.0)
    return {
        "alpha1_point": float(alpha1_point),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "p_perc": float(p_perc),
        "se_boot": float(arr.std(ddof=1)),
        "median_sel": float(np.median(sel_sizes)),
        "frac_empty": float(n_empty / B),
        "B": int(B),
        "fs_selected": fs_sel_names,
    }


def main():
    print("=" * 104)
    print(f"AEN POST-SELECTION INFERENCE v3  (alpha_1 on z_(t-1); candidate='{CANDIDATE}')")
    print("=" * 104)
    print("  alpha_0 (the level): see src/machine_learning/06_aen_oos.py (loadings-OOS, Panel D engine).")
    print("  This module: valid post-selection inference on alpha_1 (the conditional stress slope).")

    out = {}
    for sn, path in STRATEGIES.items():
        y, F, zc, zl, Fst, cand, stable = prepare(sn, path)

        a0   = unconditional_alpha0(y, Fst)
        # headline: z_{t-1}
        ins_l = insample_alpha1(y, Fst, zl)
        pds_l = pds_alpha1(y, F, zl)
        # robustness: z_t (contemporaneous)
        ins_c = insample_alpha1(y, Fst, zc)
        pds_c = pds_alpha1(y, F, zc)

        print(f"\n{'-'*104}\n{sn}   (T={len(y)},  candidate p={len(cand)},  stable k={len(stable)})\n{'-'*104}")
        print(f"  alpha_0 (in-sample reference): {a0['alpha0_ann']:+.2f}% ann"
              f"   t={a0['t']:.2f}  p(HAC)={a0['p']:.4f}{_stars(a0['p'])}   "
              f"[OOS implementability -> 06_aen_oos.py]")

        print("  alpha_1  (conditional stress slope, coefficient on z)   -- HEADLINE: z_(t-1) --")
        print(f"     in-sample (stable set):       {ins_l['alpha1']:+.4f}"
              f"   t={ins_l['t']:.2f}  p(HAC)={ins_l['p_hac']:.4f}{_stars(ins_l['p_hac'])}"
              f"   p(boot)={ins_l['p_boot']:.4f}{_stars(ins_l['p_boot'])}")
        print(f"     PDS (double-selection):       {pds_l['alpha1']:+.4f}"
              f"   t={pds_l['t']:.2f}  p(HAC)={pds_l['p_hac']:.4f}{_stars(pds_l['p_hac'])}"
              f"   p(boot)={pds_l['p_boot']:.4f}{_stars(pds_l['p_boot'])}"
              f"   [union {pds_l['n_union']}/{pds_l['n_controls']}; sel_y={pds_l['n_sel_y']}, sel_z={pds_l['n_sel_z']}]")

        print(f"     re-selection bootstrap (reuses AEN stability selection, B={RESEL_B})...", flush=True)
        rsl = reselection_bootstrap_alpha1(y, F, zl, cand, alpha1_point=ins_l["alpha1"])
        print(f"        alpha_1 = {rsl['alpha1_point']:+.4f}"
              f"   95% CI [{rsl['ci_lo']:+.4f}, {rsl['ci_hi']:+.4f}]"
              f"   p(perc)={rsl['p_perc']:.4f}{_stars(rsl['p_perc'])}"
              f"   [median sel={rsl['median_sel']:.0f}, empty={rsl['frac_empty']:.0%}]")

        print("  alpha_1  robustness: z_t (contemporaneous)")
        print(f"     in-sample (stable set):       {ins_c['alpha1']:+.4f}"
              f"   t={ins_c['t']:.2f}  p(HAC)={ins_c['p_hac']:.4f}{_stars(ins_c['p_hac'])}"
              f"   p(boot)={ins_c['p_boot']:.4f}{_stars(ins_c['p_boot'])}")
        print(f"     PDS (double-selection):       {pds_c['alpha1']:+.4f}"
              f"   t={pds_c['t']:.2f}  p(HAC)={pds_c['p_hac']:.4f}{_stars(pds_c['p_hac'])}"
              f"   p(boot)={pds_c['p_boot']:.4f}{_stars(pds_c['p_boot'])}")

        out[sn] = {
            "T": int(len(y)), "p_candidate": len(cand), "k_stable": len(stable),
            "alpha0_insample": a0,
            "z_lag":    {"insample": ins_l, "pds": pds_l, "reselection_boot": rsl},
            "z_contemp": {"insample": ins_c, "pds": pds_c},
        }

    outdir = PROJECT_ROOT / "results" / "aen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "aen_postselection.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved: {outdir / 'aen_postselection.json'}")
    print("\nHow to read:")
    print("  HEADLINE alpha_1 = PDS on z_(t-1) (BCH 2014; uniform validity, HAC+block-boot SE).")
    print("  CROSS-CHECK      = re-selection bootstrap (paper's own AEN selection, NO new LASSO;")
    print("                     percentile CI/p; n-out-of-n -> can under-cover near the null,")
    print("                     so it confirms PDS away from the null, not the other way round).")
    print("  alpha_0 level + implementability: 06_aen_oos.py (loadings-OOS, expanding/rolling).")


if __name__ == "__main__":
    main()

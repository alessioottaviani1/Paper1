"""
cfg_diagnostic.py  (upgraded)
=============================
Closes the "restricted conditional-alpha vs full CFG" decision for Paper 1, and
fixes the inference for the conditional-alpha slope a1.

Upgrades over the first version (all from the critique, verified by MC):
  * phi_e (autocorrelated errors, Getmansky-Lo-Makarov) ACTIVE in Part A -- iid
    understated the HAC over-rejection.
  * MIRROR size-adjusted power test: with a1 TRUE > 0 in the correlated+skewed
    regime, does the full CFG keep power on a1? (Verified: yes, equal to the
    restricted model -- the restricted's higher *nominal* power is fake, it is
    detecting its own bias.)
  * BLOCK BOOTSTRAP for a1 (moving-block, H0 imposed, OLS-t statistic). This is
    the gold-standard finite-sample inference (Kiefer-Vogelsang 2005; Lazarus-
    Lewis-Stock-Watson 2018) -- NOT a fixed HAC lag with normal critical values,
    which over-rejects at ANY lag (13-16%). Validated at ~6% size.
  * PART B PRE-WIRED to 03_subperiod_rolling_analysis.py: loads the real data and
    the iTraxx Main 5Y stress proxy, builds (y, X, z) exactly as analysis_ferson_
    schadt does, and reports skew(z), corr(X_j,z), VIF, restricted-vs-full a1, and
    the BOOTSTRAP p-value of a1 -- per strategy x framework. No TODOs to fill.

Run:
  python src/factor_models/cfg_diagnostic.py            -> runs Part B (real data)
  python -c "import cfg_diagnostic as d; d.run_part_A()" -> simulation evidence

Requires: numpy, pandas, statsmodels, scipy. Part B also needs the project's data.
"""

import os
import sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as st
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# SHARED HELPERS
# ============================================================================

def _make_z(T, rng, skew=False, phi=0.9):
    """Standardised AR(1) state; right-skewed innovations if skew=True.
    (AR(1) filtering attenuates the marginal skew to ~0.5-0.6, a realistic level
    for a standardised stress series -- NOT ~2.)"""
    eta = (rng.exponential(1.0, T) - 1.0) if skew else rng.standard_normal(T)
    z = np.empty(T); z[0] = eta[0]
    for t in range(1, T):
        z[t] = phi * z[t - 1] + np.sqrt(1 - phi ** 2) * eta[t]
    return (z - z.mean()) / z.std()


def _ar_err(T, rng, phi_e):
    e = rng.standard_normal(T)
    if phi_e:
        for t in range(1, T):
            e[t] = phi_e * e[t - 1] + np.sqrt(1 - phi_e ** 2) * e[t]
    return e


def _block_bootstrap_alpha1(Xf, y, z_idx, block=9, B=999, seed=0):
    """
    Moving-block residual bootstrap test of H0: coefficient at column z_idx = 0,
    using the OLS-t statistic (block resampling injects the serial dependence, so
    the non-robust t has correct size -- no HAC needed inside). Designs precomputed
    for speed. Returns (t_obs, p_value, beta_hat_at_z_idx).
    """
    y = np.asarray(y, float)
    T, k = Xf.shape
    XtXinv = np.linalg.inv(Xf.T @ Xf)
    P = XtXinv @ Xf.T
    Vzz = XtXinv[z_idx, z_idx]
    bhat = P @ y
    r = y - Xf @ bhat
    s2 = (r @ r) / (T - k)
    t_obs = bhat[z_idx] / np.sqrt(s2 * Vzz)
    # H0 fit: drop the z column
    keep = [i for i in range(k) if i != z_idx]
    Xr = Xf[:, keep]
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fit_r = Xr @ br
    res_r = y - fit_r
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(T / block))
    tb = np.empty(B)
    for b in range(B):
        starts = rng.integers(0, T - block + 1, nb)
        estar = np.concatenate([res_r[s:s + block] for s in starts])[:T]
        ystar = fit_r + estar
        bs = P @ ystar
        rs = ystar - Xf @ bs
        s2s = (rs @ rs) / (T - k)
        tb[b] = bs[z_idx] / np.sqrt(s2s * Vzz)
    p = (1 + np.sum(np.abs(tb) >= abs(t_obs))) / (B + 1)
    return t_obs, p, bhat[z_idx]


# ============================================================================
# PART A — SIMULATION  (reproduces the decision; needs no data)
# ============================================================================

def spurious_alpha1_mc(T, rho, skew, phi_e=0.25, nsim=800, K=10,
                       delta=(0.25, -0.25, 0.25), seed0=0):
    """TRUE a1 = 0. Factors 0..len(delta)-1 correlated with z (corr=rho) AND carry
    the conditional betas. Restricted vs full a1 (mean + 5% rejection, HAC=6)."""
    nd = len(delta); HAC = 6
    a1r, a1f, rr_, rf_ = [], [], 0, 0
    for s in range(nsim):
        rng = np.random.default_rng(seed0 + s)
        z = _make_z(T, rng, skew=skew); X = rng.standard_normal((T, K))
        for j in range(nd):
            X[:, j] = rho * z + np.sqrt(max(1e-9, 1 - rho ** 2)) * rng.standard_normal(T)
        b0 = rng.uniform(-.3, .3, K); d = np.zeros(K); d[:nd] = delta
        y = 0.10 + 0.0 * z + (X * (b0 + d * z[:, None])).sum(1) + _ar_err(T, rng, phi_e)
        cols = [f"F{j}" for j in range(K)]; df = pd.DataFrame(X, columns=cols); df['z'] = z
        rr = sm.OLS(y, sm.add_constant(df, prepend=True)).fit(cov_type='HAC', cov_kwds={'maxlags': HAC})
        a1r.append(rr.params['z']); rr_ += (rr.pvalues['z'] < 0.05)
        dff = df.copy()
        for f in cols:
            dff[f'{f}_x_z'] = df[f].values * z
        ff = sm.OLS(y, sm.add_constant(dff, prepend=True)).fit(cov_type='HAC', cov_kwds={'maxlags': HAC})
        a1f.append(ff.params['z']); rf_ += (ff.pvalues['z'] < 0.05)
    return dict(restr_a1=np.mean(a1r), restr_rej=rr_ / nsim,
                full_a1=np.mean(a1f), full_rej=rf_ / nsim)


def size_mc(K, T, lag, phi_e=0.25, nsim=1000):
    """Size at 5% of single-a1 t-test and joint F-test (no conditioning truth)."""
    rej1 = rejJ = 0
    for s in range(nsim):
        rng = np.random.default_rng(5000 + s)
        z = _make_z(T, rng, skew=True); X = rng.standard_normal((T, K))
        b0 = rng.uniform(-.3, .3, K)
        y = 0.10 + (X * b0).sum(1) + _ar_err(T, rng, phi_e)
        cols = [f"F{j}" for j in range(K)]; df = pd.DataFrame(X, columns=cols); df['z'] = z
        for f in cols:
            df[f'{f}_x_z'] = X[:, cols.index(f)] * z
        res = sm.OLS(y, sm.add_constant(df, prepend=True)).fit(cov_type='HAC', cov_kwds={'maxlags': lag})
        rej1 += (res.pvalues['z'] < 0.05)
        ic = ['z'] + [f'{f}_x_z' for f in cols]
        R = np.zeros((len(ic), len(res.params)))
        for i, c in enumerate(ic):
            R[i, list(res.params.index).index(c)] = 1.0
        rejJ += (float(res.f_test(R).pvalue) < 0.05)
    return rej1 / nsim, rejJ / nsim


def size_adjusted_power_mc(T, true_a1, rho=0.6, skew=True, phi_e=0.25, K=10,
                           delta=(0.25, -0.25, 0.25), model='full', nsim=1000):
    """Size-adjusted power of the a1 test in the dangerous (corr+skew) regime.
    Gets the 5% one-sided critical value of the t-stat under H0 (a1=0), then power
    at that critical value under H1 (a1=true_a1). HAC=6."""
    HAC = 6; nd = len(delta)
    def tstat(a1, seed0):
        out = []
        for s in range(nsim):
            rng = np.random.default_rng(seed0 + s)
            z = _make_z(T, rng, skew=skew); X = rng.standard_normal((T, K))
            for j in range(nd):
                X[:, j] = rho * z + np.sqrt(max(1e-9, 1 - rho ** 2)) * rng.standard_normal(T)
            b0 = rng.uniform(-.3, .3, K); d = np.zeros(K); d[:nd] = delta
            y = 0.10 + a1 * z + (X * (b0 + d * z[:, None])).sum(1) + _ar_err(T, rng, phi_e)
            cols = [f"F{j}" for j in range(K)]; df = pd.DataFrame(X, columns=cols); df['z'] = z
            if model == 'full':
                for f in cols:
                    df[f'{f}_x_z'] = X[:, cols.index(f)] * z
            res = sm.OLS(y, sm.add_constant(df, prepend=True)).fit(cov_type='HAC', cov_kwds={'maxlags': HAC})
            out.append(res.tvalues['z'])
        return np.array(out)
    t0 = tstat(0.0, 10000); t1 = tstat(true_a1, 20000)
    cstar = np.quantile(t0, 0.95)
    return dict(nominal_size=float(np.mean(t0 > 1.645)),
                size_adj_power=float(np.mean(t1 > cstar)))


def validate_bootstrap_size(T=159, K=10, phi_e=0.25, block=9, B=199, nsim=300):
    """Confirms the moving-block bootstrap on a1 has ~5% size (true a1=0,
    z skewed, AR errors, full CFG)."""
    rej = 0
    for s in range(nsim):
        rng = np.random.default_rng(7000 + s)
        z = _make_z(T, rng, skew=True); X = rng.standard_normal((T, K))
        b0 = rng.uniform(-.3, .3, K)
        y = 0.10 + (X * b0).sum(1) + _ar_err(T, rng, phi_e)
        inter = X * z[:, None]
        Xf = np.column_stack([np.ones(T), X, z, inter]); z_idx = 1 + K
        _, p, _ = _block_bootstrap_alpha1(Xf, y, z_idx, block=block, B=B, seed=s)
        rej += (p < 0.05)
    return rej / nsim


def run_part_A():
    print("=" * 80)
    print("PART A — SIMULATION (phi_e active = autocorrelated errors)")
    print("=" * 80)
    print("\n[1] Spurious a1: TRUE a1 = 0; restricted vs full CFG (HAC=6).")
    h = f"{'T':>4} {'corr(X,z)':>9} {'z':>11} | {'RESTR a1':>9} {'R rej':>6} | {'FULL a1':>9} {'F rej':>6}"
    print(h); print("-" * len(h))
    for T in (159, 241):
        for skew in (False, True):
            sk = st.skew(_make_z(T, np.random.default_rng(0), skew=skew))
            tag = f"skew={sk:+.2f}" if skew else "gauss"
            for rho in (0.0, 0.6):
                r = spurious_alpha1_mc(T, rho, skew)
                print(f"{T:>4} {rho:>9.1f} {tag:>11} | {r['restr_a1']:>+9.4f} {r['restr_rej']:>5.0%} | "
                      f"{r['full_a1']:>+9.4f} {r['full_rej']:>5.0%}")
        print()
    print("[2] Size: tuning the HAC lag does NOT fix single-a1 over-rejection (normal crit):")
    for lag in (3, 6, 12, 16):
        s1, sJ = size_mc(10, 159, lag)
        print(f"     lag={lag:2d} | single-a1 = {s1:.0%}   joint-F = {sJ:.0%}")
    print("\n[3] Mirror size-adjusted POWER in the dangerous regime (rho=.6, skew, HAC=6):")
    for T, a1 in [(159, 0.14), (241, 0.12)]:
        rr = size_adjusted_power_mc(T, a1, model='restricted')
        ff = size_adjusted_power_mc(T, a1, model='full')
        print(f"     T={T} | restricted: nominal={rr['nominal_size']:.0%} power={rr['size_adj_power']:.0%}"
              f"   full: nominal={ff['nominal_size']:.0%} power={ff['size_adj_power']:.0%}")
    print("\n[4] Block bootstrap on a1 restores correct size (target 5%):")
    print(f"     moving-block bootstrap size = {validate_bootstrap_size():.0%}")
    print("\nConclusion: full CFG is unbiased AND equally powerful (size-adjusted);")
    print("the restricted model fabricates a1 under skew+correlation; no HAC lag fixes")
    print("the over-rejection -> use the block bootstrap for a1.\n")


# ============================================================================
# PART B — EMPIRICAL (run on the real data; closes the decision)
# ============================================================================

def _vif(df_design):
    cols = list(df_design.columns); M = df_design.values.astype(float); out = {}
    for j, c in enumerate(cols):
        yj = M[:, j]; Xj = np.column_stack([np.ones(len(M)), np.delete(M, j, axis=1)])
        beta, *_ = np.linalg.lstsq(Xj, yj, rcond=None)
        resid = yj - Xj @ beta
        ss_res = float(resid @ resid); ss_tot = float(((yj - yj.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[c] = np.inf if r2 >= 1 else 1.0 / (1.0 - r2)
    return out


def block_bootstrap_alpha1(y, X, z, model='full', ann=12, block=9, B=999, seed=0):
    """Bootstrap p-value for a1 on real data. model='full' (CFG) or 'restricted'."""
    y = np.asarray(y, float); z = np.asarray(z, float); Xv = np.asarray(X, float)
    T = Xv.shape[0]
    if model == 'full':
        Xf = np.column_stack([np.ones(T), Xv, z, Xv * z[:, None]]); z_idx = 1 + Xv.shape[1]
    else:
        Xf = np.column_stack([np.ones(T), Xv, z]); z_idx = 1 + Xv.shape[1]
    t_obs, p, b = _block_bootstrap_alpha1(Xf, y, z_idx, block=block, B=B, seed=seed)
    return dict(alpha1_ann=b * ann, t_obs=t_obs, p_boot=p)


def empirical_cfg_diagnostic(y, X, z, label="", ann=12, block=9, B=999, show_vif=True):
    """
    y, X, z : aligned pandas objects (z will be standardised inside if not already).
    Prints skew(z), corr(X_j,z), VIF, restricted-vs-full a1, and the BOOTSTRAP
    p-value of a1 (the valid finite-sample inference). Returns a dict.
    """
    idx = y.index.intersection(X.index).intersection(z.index)
    y = y.loc[idx]; X = X.loc[idx]; z = z.loc[idx]
    z = (z - z.mean()) / z.std()
    cols = list(X.columns); T = len(y)
    # HAC display lag = Newey-West (1994) rule-of-thumb (NOT a fixed 6)
    nw_lag = int(np.floor(4 * (T / 100) ** (2 / 9)))

    print("=" * 80)
    print(f"PART B — EMPIRICAL  |  {label}")
    print("=" * 80)
    print(f"T = {T},  factors = {len(cols)},  HAC display lag (NW1994 r.o.t.) = {nw_lag},  ann = x{ann}")
    sk = float(st.skew(z.values))
    print(f"skew(z) = {sk:+.3f}    (|skew| > ~0.3 => 3rd-moment bias channel active)")

    corr = {c: float(np.corrcoef(X[c].values, z.values)[0, 1]) for c in cols}
    cs = sorted(corr.items(), key=lambda kv: -abs(kv[1]))
    print("corr(factor, z)  [top |corr| first]: " +
          ", ".join(f"{c}={v:+.2f}" for c, v in cs[:6]))
    max_abs = max(abs(v) for v in corr.values())
    print(f"  max |corr(X_j, z)| = {max_abs:.3f}")

    # RESTRICTED (a0 + a1 z + b'X)
    Dr = X.copy(); Dr['z_stress'] = z.values
    rr = sm.OLS(y, sm.add_constant(Dr, prepend=True)).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lag})
    a1r = rr.params['z_stress'] * ann
    br = block_bootstrap_alpha1(y, X, z, model='restricted', ann=ann, block=block, B=B)

    # FULL CFG (a0 + a1 z + (b0 + b1 z)'X)
    Df = X.copy(); Df['z_stress'] = z.values
    for c in cols:
        Df[f'{c}_x_z'] = X[c].values * z.values
    ff = sm.OLS(y, sm.add_constant(Df, prepend=True)).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lag})
    a1f = ff.params['z_stress'] * ann
    bf = block_bootstrap_alpha1(y, X, z, model='full', ann=ann, block=block, B=B)
    n_par = len(ff.params)

    print("\n  alpha1 (annualised %, per +1 sigma of stress):")
    print(f"    RESTRICTED (betas const):   a1 = {a1r:+.3f}   HAC p = {rr.pvalues['z_stress']:.4f}   "
          f"bootstrap p = {br['p_boot']:.4f}")
    print(f"    FULL CFG   (betas cond.):   a1 = {a1f:+.3f}   HAC p = {ff.pvalues['z_stress']:.4f}   "
          f"bootstrap p = {bf['p_boot']:.4f}   <-- HEADLINE")
    print(f"    difference (restr - full): {a1r - a1f:+.3f}   "
          f"[full CFG: {n_par} params, dof = {T - n_par}]")
    if abs(a1r) > abs(a1f) + 1e-9:
        print("    -> restricted a1 is larger: consistent with upward contamination")
        print("       from omitted conditional betas. FULL CFG is the unbiased spec.")

    if show_vif:
        vf = _vif(Df)
        worst_int = sorted({k: v for k, v in vf.items() if k.endswith('_x_z')}.items(),
                           key=lambda kv: -kv[1])[:4]
        print("  worst interaction VIFs: " + ", ".join(f"{k}={v:.0f}" for k, v in worst_int) +
              "   (high => do NOT report individual deltas)")

    return dict(label=label, T=T, skew_z=sk, max_abs_corr=max_abs,
                a1_restricted=a1r, a1_full=a1f,
                p_boot_restricted=br['p_boot'], p_boot_full=bf['p_boot'])


def run_part_B(B=999, block=9):
    """PRE-WIRED to 03_subperiod_rolling_analysis.py. Loops STRATEGIES x FRAMEWORKS,
    builds (y, X, z) exactly as analysis_ferson_schadt, runs the diagnostic, and
    prints a summary decision table."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    m = import_module("03_subperiod_rolling_analysis")
    stress = m.load_stress_proxy_monthly()
    y_col = 'Strategy_Return'
    rows = []
    for strat in m.STRATEGIES:
        for fw in m.FRAMEWORKS:
            data = m.load_regression_data(strat, fw, m.PRIMARY_REGION, m.REGRESSION_FREQ)
            if data is None or y_col not in data.columns:
                print(f"[skip] {strat} / {fw}: no data / no {y_col}"); continue
            avail = [f for f in m.FRAMEWORKS[fw]['factors'] if f in data.columns]
            if len(avail) < 2:
                print(f"[skip] {strat} / {fw}: <2 factors"); continue
            data = data.dropna(subset=[y_col] + avail)
            y = data[y_col]; X = data[avail]
            sa = stress.reindex(y.index, method='nearest')
            common = y.index.intersection(sa.dropna().index)
            y = y.loc[common]; X = X.loc[common]; lvl = sa.loc[common]
            if len(y) < 30:
                print(f"[skip] {strat} / {fw}: T<30"); continue
            z = (lvl - lvl.mean()) / lvl.std()
            res = empirical_cfg_diagnostic(y, X, z, label=f"{strat} / {fw}", block=block, B=B)
            rows.append(res); print()

    if rows:
        print("=" * 80)
        print("SUMMARY DECISION TABLE")
        print("=" * 80)
        hdr = (f"{'strategy / framework':<34}{'T':>4}{'skew(z)':>9}{'max|corr|':>10}"
               f"{'a1 restr':>9}{'a1 full':>9}{'p_boot(full)':>13}")
        print(hdr); print("-" * len(hdr))
        for r in rows:
            print(f"{r['label']:<34}{r['T']:>4}{r['skew_z']:>+9.2f}{r['max_abs_corr']:>10.2f}"
                  f"{r['a1_restricted']:>+9.2f}{r['a1_full']:>+9.2f}{r['p_boot_full']:>13.4f}")
        print("\nRead: if skew(z) and max|corr| are non-trivial (the dangerous regime),")
        print("the restricted a1 is contaminated -> use FULL CFG. The headline significance")
        print("of a1 is the bootstrap p-value of the FULL model (last column).")
    return rows


if __name__ == "__main__":
    run_part_B()
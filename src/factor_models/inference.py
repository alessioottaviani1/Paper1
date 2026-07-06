"""
inference.py
============
Shared finite-sample inference utilities for the conditional-alpha (CFG) models in
03_subperiod_rolling_analysis.py (benchmark), 05b_pca_conditional_alpha.py (PCA),
and 06e_conditional_alpha.py (AEN).

GOLD-STANDARD INFERENCE for the conditional-alpha slope alpha1
--------------------------------------------------------------
The headline p-value for alpha1 is a MOVING-BLOCK RESIDUAL BOOTSTRAP (H0 imposed),
which is finite-sample valid and -- verified by Monte Carlo on this paper's data-
generating process (right-skewed stress, autocorrelated arbitrage returns, full
CFG, T in 160-240) -- controls size at ~5-6%. A fixed-bandwidth HAC t-test with
normal critical values over-rejects at ANY lag (13-16% at nominal 5%); tuning the
lag does not help (small-b problem). See Kiefer & Vogelsang (2005) and Lazarus,
Lewis, Stock & Watson (2018, JBES). The block length matches the stationary block
bootstrap used elsewhere in the paper for stability selection.

auto_hac_lags() is the Newey-West (1994) rule-of-thumb bandwidth floor(4 (n/100)^(2/9));
for the paper's monthly samples (T in 162-243) it equals 4. It is used for the HAC
standard errors that are still reported (as robustness for alpha1, and for the
unconditional alphas), replacing the previous ad hoc fixed lag of 6.

All bootstrap p-values are deterministic (fixed seed) for exact replication.
"""

import numpy as np

# Block length for the moving-block bootstrap, matching the stability-selection
# block bootstrap used elsewhere in the paper. The MC size validation used this value.
STABILITY_BLOCK = 9
# Bootstrap replications (publication precision: p-value resolution ~1e-4).
BOOTSTRAP_B = 9999


def auto_hac_lags(n):
    """Newey-West (1994) rule-of-thumb truncation lag: floor(4*(n/100)^(2/9)).
    Equals 4 for the paper's monthly samples (T in 162-243)."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _moving_block_bootstrap_pvalue(Xf, y, z_idx, block=STABILITY_BLOCK,
                                   B=BOOTSTRAP_B, seed=0):
    """
    Two-sided bootstrap p-value for H0: coefficient at column `z_idx` = 0, in the
    linear model y = Xf @ b + e.

    Moving-block residual resampling under H0 preserves the serial dependence of the
    errors, so the (non-robust) OLS-t statistic has correct finite-sample size; no
    HAC is needed inside the loop. Projection matrices are precomputed, so the loop
    is a sequence of cheap matrix multiplications.
    """
    y = np.asarray(y, float)
    Xf = np.asarray(Xf, float)
    T, k = Xf.shape
    XtXinv = np.linalg.inv(Xf.T @ Xf)
    P = XtXinv @ Xf.T
    Vzz = XtXinv[z_idx, z_idx]
    bhat = P @ y
    r = y - Xf @ bhat
    t_obs = bhat[z_idx] / np.sqrt((r @ r) / (T - k) * Vzz)
    # Fit under H0 (drop the tested column) -> restricted residuals to resample
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
        tb[b] = bs[z_idx] / np.sqrt((rs @ rs) / (T - k) * Vzz)
    return float((1 + np.sum(np.abs(tb) >= abs(t_obs))) / (B + 1))


def cfg_alpha1_bootstrap_p(y, X, z, block=STABILITY_BLOCK, B=BOOTSTRAP_B, seed=0):
    """
    Block-bootstrap p-value for the conditional-alpha slope alpha1 (the coefficient
    on z) in the full CFG model

        r_t = a0 + a1 z_t + (b0 + b1 z_t)' X_t + e_t.

    Parameters
    ----------
    y : array-like (T,)      strategy excess returns (aligned).
    X : array-like (T x K)   factor matrix (pandas DataFrame or ndarray), aligned.
    z : array-like (T,)      standardised conditioning state (e.g. iTraxx Main 5Y).
    Tests H0: a1 = 0. Returns the two-sided bootstrap p-value.
    """
    y = np.asarray(y, float)
    z = np.asarray(z, float)
    Xv = np.asarray(X, float)
    T = Xv.shape[0]
    # design: [const, X (K cols), z, X*z (K cols)]  -> z is at index 1+K
    Xf = np.column_stack([np.ones(T), Xv, z, Xv * z[:, None]])
    z_idx = 1 + Xv.shape[1]
    return _moving_block_bootstrap_pvalue(Xf, y, z_idx, block=block, B=B, seed=seed)

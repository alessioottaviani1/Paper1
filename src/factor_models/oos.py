# -*- coding: utf-8 -*-
"""
Out-of-sample (OOS) alpha engine for the factor-model benchmarks (Paper 1).

The benchmark alphas in 02a/02b/02c are full-sample OLS intercepts (in-sample):
the loadings use the whole period, so the implied hedge is formed with hindsight.
This module estimates the loadings on a PAST-ONLY window and evaluates the
realised abnormal return on the subsequent observation (Welch-Goyal 2008
recursive scheme; the implementability test Tessaromatis asks for, and the
out-of-sample counterpart to the Moreira-Muir in-sample-weights critique,
Cederburg, O'Doherty, Wang & Yan 2020 JFE).

Clean function separation so each piece is testable:
  _nw_lags, _ill_conditioned, oos_residuals (engine), summarize, oos_r2,
  insample_alpha.

Author: Alessio Ottaviani, EDHEC Business School.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _nw_lags(n):
    """Newey-West (1994) automatic lag for a monthly series of length n."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def _ill_conditioned(Xw, cond_max):
    """
    True if the in-window design is numerically unreliable.
    Xw includes the constant column. We (i) flag rank deficiency and
    (ii) compute the condition index of the WITHIN-WINDOW STANDARDISED factor
    block (scale-free, so it reflects collinearity among factors -- e.g. the
    highly correlated bond returns R2/R5/R10 in Duarte -- not their units).
    """
    if np.linalg.matrix_rank(Xw) < Xw.shape[1]:
        return True
    F = Xw[:, 1:]                                   # drop the constant
    sd = F.std(axis=0, ddof=0)
    if np.any(sd == 0.0):                           # a factor constant in-window
        return True
    Fs = (F - F.mean(axis=0)) / sd
    return np.linalg.cond(Fs) > cond_max            # condition index


def oos_residuals(y, X, scheme="expanding", min_window=60, roll_window=60,
                  cond_max=1e3):
    """
    Recursive out-of-sample residuals.

    For each t >= start, estimate the loadings by OLS of y on [const, X] over the
    past window (expanding [0,t) or rolling [t-w,t)).  The model abnormal return
    uses the SLOPES ONLY (the in-window constant is the alpha we measure, so it is
    NOT carried into the hedge):
        e_mod_t   = y_t - beta_hat'_{t-1} F_t
    The naive benchmark is the expanding historical mean of the strategy
    (Campbell-Thompson / Welch-Goyal):
        e_naive_t = y_t - mean(y_[0:t])
    Windows whose design is rank-deficient or ill-conditioned (cond > cond_max)
    are SKIPPED and counted (n_skipped) -- this is the explicit guard against
    near-singular Duarte rolling windows producing exploding beta_hat / outlier
    e_t.  Returns (e_mod, e_naive, info).
    """
    if scheme not in ("expanding", "rolling"):
        raise ValueError("scheme must be 'expanding' or 'rolling'")
    df = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    cols = list(X.columns)
    yv = df["__y__"].to_numpy()
    Xv = df[cols].to_numpy()
    T, K = Xv.shape
    start = max(min_window, roll_window) if scheme == "rolling" else min_window
    if start >= T:
        raise ValueError(f"Sample too short: T={T} usable months, start={start}, "
                         f"K={K} factors. No out-of-sample observations.")

    e_mod, e_naive, idx = [], [], []
    n_skipped = 0
    for t in range(start, T):
        lo = (t - roll_window) if scheme == "rolling" else 0
        Xw = sm.add_constant(Xv[lo:t])
        if _ill_conditioned(Xw, cond_max):
            n_skipped += 1
            continue
        beta = np.linalg.lstsq(Xw, yv[lo:t], rcond=None)[0]   # [const, slopes...]
        slopes = beta[1:]
        e_mod.append(yv[t] - Xv[t] @ slopes)                  # slopes only
        e_naive.append(yv[t] - yv[0:t].mean())                # expanding hist. mean
        idx.append(df.index[t])

    info = {"n_total": int(T), "n_oos": len(e_mod), "n_skipped": int(n_skipped),
            "K": int(K), "scheme": scheme, "start": int(start)}
    return pd.Series(e_mod, index=idx), pd.Series(e_naive, index=idx), info


def summarize(e_mod, periods_per_year=12, min_obs=12):
    """OOS alpha (annualised), Newey-West HAC t/p on the e_mod series, and IR."""
    n = len(e_mod)
    if n < min_obs:
        return {"alpha_ann": np.nan, "t": np.nan, "p": np.nan,
                "ir": np.nan, "n_oos": n}
    fit = sm.OLS(e_mod.to_numpy(), np.ones(n)).fit(
        cov_type="HAC", cov_kwds={"maxlags": _nw_lags(n)})
    ir = e_mod.mean() / e_mod.std(ddof=1) * np.sqrt(periods_per_year)
    return {"alpha_ann": fit.params[0] * periods_per_year, "t": fit.tvalues[0],
            "p": fit.pvalues[0], "ir": ir, "n_oos": n}


def oos_r2(e_mod, e_naive):
    """OOS explanatory R^2 vs the historical-mean benchmark (a WEAK threshold for
    a near-zero-mean arbitrage: positive only means the factors beat the mean)."""
    num = float((e_mod.to_numpy() ** 2).sum())
    den = float((e_naive.to_numpy() ** 2).sum())
    return np.nan if den == 0 else 1.0 - num / den


def insample_alpha(y, X, restrict_index=None, periods_per_year=12):
    """Full-sample (or OOS-subperiod) OLS intercept with HAC; lags computed on
    THIS sample's length so column 2 and the OOS columns use the same rule."""
    df = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if restrict_index is not None:
        df = df.loc[df.index.intersection(restrict_index)]
    n = len(df)
    Xc = sm.add_constant(df[list(X.columns)].to_numpy())
    fit = sm.OLS(df["__y__"].to_numpy(), Xc).fit(
        cov_type="HAC", cov_kwds={"maxlags": _nw_lags(n)})
    return {"alpha_ann": fit.params[0] * periods_per_year, "t": fit.tvalues[0], "n": n}

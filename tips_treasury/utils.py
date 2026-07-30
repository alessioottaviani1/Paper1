"""Funzioni condivise per il package tips_treasury (Paper 2)."""
import numpy as np, pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from config import *

def load_bbg(path):
    """Export Bloomberg: ticker in riga 4, dati da riga 7."""
    raw = pd.read_excel(path, header=None)
    tk = [str(t).strip() for t in raw.iloc[3].tolist()[1:]]
    d = raw.iloc[6:].copy(); d.columns = ["date"] + tk
    d["date"] = pd.to_datetime(d["date"])
    return d.set_index("date").apply(pd.to_numeric, errors="coerce").sort_index()

def col_of(df, prefix):
    h = [c for c in df.columns if c.startswith(prefix)]
    if not h: raise KeyError(f"ticker '{prefix}' non trovato")
    return df[h[0]]

def nw_mean(x, lags=NW_LAGS):
    x = x.dropna()
    r = sm.OLS(x.values, np.ones((len(x), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return r.params[0], r.tvalues[0], len(x)

def ou_fit(x):
    """OU discreto su serie mensile: ritorna kappa/m, theta, sigma, half-life (mesi)."""
    x = pd.Series(x).dropna()
    dx = x.diff().dropna(); xl = x.shift(1).loc[dx.index]
    r = sm.OLS(dx.values, sm.add_constant(xl.values)).fit()
    k = -r.params[1]; th = r.params[0] / k if k != 0 else np.nan
    return k, th, r.resid.std(), (np.log(2) / k if k > 0 else np.inf)

def one_break(x, trim=0.10):
    x = pd.Series(x).dropna(); n = len(x); t0 = int(trim * n)
    ssr = [((x[:k] - x[:k].mean())**2).sum() + ((x[k:] - x[k:].mean())**2).sum()
           for k in range(t0, n - t0)]
    k = int(np.argmin(ssr)) + t0
    F = (((x - x.mean())**2).sum() - min(ssr)) / (min(ssr) / (n - 2))
    return x.index[k], x[:k].mean(), x[k:].mean(), F, k

def adf_p(x):
    x = pd.Series(x).dropna()
    return adfuller(x, regression="c", autolag="AIC")[1] if len(x) > 24 else np.nan

def deseasonalise(x_monthly):
    """Dummy mensili: serie destagionalizzata, ampiezza, p del test F congiunto (HAC12)."""
    x = pd.Series(x_monthly).dropna()
    D = pd.get_dummies(x.index.month, drop_first=True).astype(float); D.index = x.index
    r = sm.OLS(x.values, sm.add_constant(D.values)).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    fit = pd.Series(r.fittedvalues, index=x.index)
    mm = fit.groupby(fit.index.month).mean()
    return x - fit + x.mean(), float(mm.max() - mm.min()), float(r.f_pvalue)

def save_txt(name, lines):
    p = RES / name
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {p}")

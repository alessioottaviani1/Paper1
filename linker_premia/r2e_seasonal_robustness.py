"""r2e - ROBUSTEZZA stagionale del Test 4. Le agenzie nazionali ed Eurostat pubblicano
il CPI headline solo NSA (verificato: Eurostat esplicitamente 'not seasonally adjusted',
ISTAT/Destatis diffondono l'headline NSA). Il NSA e' peraltro la serie CORRETTA per i
linker (i coupon si indicizzano all'indice grezzo), e Maffei stesso (sez. 6.2.1, citando
Haubrich et al.) nota che la stagionalita' non ha impatto materiale.

Per rendere la conclusione inattaccabile: destagionalizziamo NOI le serie nazionali NSA
con STL (Seasonal-Trend decomposition, standard, puro-Python) e rifacciamo lambda~InfS
SA-vs-NSA affiancati. Se il risultato non cambia -> la conclusione (incl. la reazione
di DE che indebolisce il canale fiscale puro) NON dipende dalla stagionalita'.

Legge le stesse serie di r2d (an_cpi_{IT,FR,DE}_NAT). Output: an_seasonal_robustness.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import rp
from config import CACHE

MATS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20)
REL_DAY = {"IT": 17, "FR": 15, "DE": 15}
NAT_KEY = {"IT": "IT_NAT", "FR": "FR_NAT", "DE": "DE_NAT"}
CRISIS = ("2011-01-01", "2012-12-31")


def _release_grid(mkt, months):
    f = CACHE / f"cpi_release_dates_{mkt}.csv"
    if f.exists():
        rel = pd.to_datetime(pd.read_csv(f).iloc[:, 0]).sort_values()
        out = {}
        for m in months:
            nxt = rel[(rel > m) & (rel <= m + pd.offsets.MonthEnd(2))]
            if len(nxt):
                out[m] = nxt.iloc[0]
        return pd.Series(out)
    d = REL_DAY[mkt]
    return pd.Series({m: (m + pd.offsets.MonthBegin(1) + pd.Timedelta(days=d - 1)
                          + pd.offsets.BusinessDay(0)) for m in months})


def _sample_at(daily, grid):
    daily = daily.sort_index()
    return pd.DataFrame({m: daily.asof(g) for m, g in grid.items()
                         if g >= daily.index.min()}).T.dropna(how="all")


def _cpi_national(mkt):
    path = CACHE / f"an_cpi_{NAT_KEY[mkt]}.parquet"
    if not path.exists():
        return None
    s = pd.read_parquet(path).iloc[:, 0]
    s.index = pd.to_datetime(s.index) + pd.offsets.MonthEnd(0)
    return s[~s.index.duplicated(keep="last")].sort_index()


def stl_sa(level: pd.Series) -> pd.Series:
    """Destagionalizza un indice di livello mensile con STL: SA = livello / stagionale.
    STL lavora in additivo sul LOG (equivalente a moltiplicativo sul livello, adatto a
    un indice prezzi), period=12. Robust=True per attenuare gli outlier (es. 2020)."""
    s = level.asfreq("ME").interpolate(limit_area="inside")
    ln = np.log(s.dropna())
    res = STL(ln, period=12, robust=True).fit()
    sa = np.exp(ln - res.seasonal)                    # log-livello meno stagionale -> SA
    return sa.reindex(level.index)


def _surprise(idx_series):
    pi = rp.yoy(idx_series).dropna()
    return (pi - pi.rolling(120, min_periods=120).mean()).dropna()


def _lambda(mkt):
    bei_d, isr_d = rp.bei_euro(mkt, MATS), rp.isr(mkt, MATS)
    months = pd.date_range(bei_d.index.min().normalize().replace(day=1),
                           bei_d.index.max(), freq="ME")
    grid = _release_grid(mkt, months)
    bei, s = _sample_at(bei_d, grid), _sample_at(isr_d, grid)
    idx = bei.index.intersection(s.index)
    return (s.loc[idx] - bei.loc[idx])


def _reg(dep, x, lags=6):
    rows = []
    for n in dep.columns:
        d = pd.concat([(dep[n] * 100).rename("y"), x.rename("x")], axis=1).dropna()
        if len(d) < 30:
            continue
        b, e, _, r2, X = rp.ols(d["y"].values, d[["x"]].values)
        t = rp.nw_t(e, X, b, lags)
        rows.append([float(n), len(d), b[1], t[1], r2])
    return pd.DataFrame(rows, columns=["mat", "n", "beta", "t", "R2"])


def _fmt(tab):
    return "  ".join(f"{r.mat:g}y {r.beta:+.2f}({r.t:+.1f})" for r in tab.itertuples())


def _avg(tab, lo=3, hi=10):
    m = tab[(tab.mat >= lo) & (tab.mat <= hi)]
    if m.empty:
        return np.nan, 0.0
    return m.beta.mean(), (m.t.abs() >= 1.96).mean()


def run(mkt):
    print(f"\n=== {mkt} | lambda~InfS: NSA (grezzo) vs SA (STL) sulla sorpresa nazionale ===")
    lam = _lambda(mkt)
    nat = _cpi_national(mkt)
    if nat is None:
        print(f"  [!] CPI nazionale assente (an_cpi_{NAT_KEY[mkt]}): lanciare r0.")
        return pd.DataFrame()
    sa = stl_sa(nat)
    # quanto pesa la stagionalita': deviazione std del fattore stagionale implicito
    seas_amp = float((np.log(nat.dropna()) - np.log(sa.reindex(nat.index).dropna())).std() * 100)
    print(f"  ampiezza stagionale (sd del fattore, log-punti) = {seas_amp:.2f}  "
          f"({'trascurabile' if seas_amp < 0.5 else 'piccola' if seas_amp < 1.5 else 'rilevante'})")
    surp_nsa = _surprise(nat).reindex(lam.index)
    surp_sa = _surprise(sa).reindex(lam.index)
    cc = pd.concat([surp_nsa.rename("nsa"), surp_sa.rename("sa")], axis=1).dropna().corr().iloc[0, 1]
    print(f"  corr(sorpresa NSA, sorpresa SA) = {cc:.3f}")
    out = []
    for lab, sp in (("NSA", surp_nsa), ("SA_STL", surp_sa)):
        tab = _reg(lam, sp)
        tab.insert(0, "mkt", mkt); tab.insert(1, "cpi", lab)
        out.append(tab)
        a, f = _avg(tab)
        print(f"  [{lab}]  media beta 3-10y = {a:+.2f} bp | signif. {f*100:.0f}%")
        print(f"    {_fmt(tab)}")
    a_nsa, a_sa = _avg(out[0])[0], _avg(out[1])[0]
    delta = abs(a_sa - a_nsa)
    print(f"  >>> NSA {a_nsa:+.2f} vs SA {a_sa:+.2f} | scarto {delta:.2f} bp "
          f"({'INVARIANTE' if delta < 0.4 else 'differenza non trascurabile'})")
    return pd.concat(out)


if __name__ == "__main__":
    res = []
    for mkt in ("IT", "FR", "DE"):
        t = run(mkt)
        if not t.empty:
            res.append(t)
    print("\n" + "=" * 78)
    print("VERDETTO -- la conclusione dipende dalla destagionalizzazione?")
    print("=" * 78)
    if res:
        allr = pd.concat(res)
        for mkt in ("IT", "FR", "DE"):
            n = allr[(allr.mkt == mkt) & (allr.cpi == "NSA")]
            s = allr[(allr.mkt == mkt) & (allr.cpi == "SA_STL")]
            an = _avg(n)[0] if len(n) else np.nan
            as_ = _avg(s)[0] if len(s) else np.nan
            role = " (controllo)" if mkt == "DE" else ""
            print(f"  {mkt}{role}: NSA {an:+.2f} | SA {as_:+.2f} bp")
        print("\n  se NSA ~ SA per tutti (scarti < 0.4 bp) -> la stagionalita' NON guida i")
        print("  risultati: la conclusione del Test 4 e' robusta. In particolare, se DE")
        print("  reagisce sia in NSA sia in SA -> la sua reazione e' reale, il canale NON e'")
        print("  puramente fiscale, e la teoria fiscale resta indebolita a prescindere.")
        print("  se invece SA cambia il quadro (es. DE muto in SA) -> rivedere: la reazione")
        print("  tedesca era artefatto stagionale e la teoria fiscale torna in gioco.")
    allr.to_csv(CACHE / "an_seasonal_robustness.csv", index=False) if res else None
    print("\nsalvato: an_seasonal_robustness.csv")

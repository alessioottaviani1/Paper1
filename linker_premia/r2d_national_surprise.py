"""r2d - TEST 4 (il test che blinda il canale fiscale): lambda ~ sorpresa NAZIONALE.
r2b/r2c usano la sorpresa d'inflazione AGGREGATA (HICP eurozona, CPTFEMU), comune a
IT/FR/DE. Ma la teoria fiscale dice: inflazione DEL PAESE -> timori fiscali DEL PAESE ->
premio sovrano DEL PAESE. Quindi il test corretto usa il CPI NAZIONALE.

  Se lambda_IT reagisce alla sorpresa ITALIANA (non solo a quella euro) -> la teoria
  fiscale e' blindata: e' l'inflazione del paese a muovere il suo breakeven.
  Se svanisce passando al CPI nazionale -> l'effetto era la componente comune euro,
  e la storia e' diversa (shock d'area, non canale fiscale nazionale).

Confronto AFFIANCATO per ogni mercato: sorpresa aggregata (come r2b) vs nazionale.
DE e' il controllo: rischio sovrano ~0 -> il canale fiscale deve essere MUTO anche
sulla sorpresa nazionale tedesca.

Richiede i CPI nazionali in cache (r0 aggiornato: ITCPIUNR/FRCPXTOB/GRCP2000).
Output: an_national_surprise.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np
import pandas as pd
import rp
from config import CACHE

MATS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20)
REL_DAY = {"IT": 17, "FR": 15, "DE": 15}
NAT_KEY = {"IT": "IT_NAT", "FR": "FR_NAT", "DE": "DE_NAT"}   # chiavi an_cpi_* dei CPI nazionali
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
    """CPI nazionale dal cache (an_cpi_{IT_NAT|FR_NAT|DE_NAT}, da r0)."""
    path = CACHE / f"an_cpi_{NAT_KEY[mkt]}.parquet"
    if not path.exists():
        return None
    s = pd.read_parquet(path).iloc[:, 0]
    s.index = pd.to_datetime(s.index) + pd.offsets.MonthEnd(0)
    return s[~s.index.duplicated(keep="last")].sort_index()


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
    print(f"\n=== {mkt} | lambda ~ sorpresa AGGREGATA (euro) vs NAZIONALE ===")
    lam = _lambda(mkt)
    surp_agg = _surprise(rp.cpi(mkt)).reindex(lam.index)     # HICP euro (come r2b)
    nat = _cpi_national(mkt)
    if nat is None:
        print(f"  [!] CPI nazionale assente (an_cpi_{NAT_KEY[mkt]}): lanciare r0 aggiornato.")
        return pd.DataFrame()
    surp_nat = _surprise(nat).reindex(lam.index)
    # correlazione tra le due sorprese: quanto e' distinta la nazionale dall'aggregata
    cc = pd.concat([surp_agg.rename("agg"), surp_nat.rename("nat")], axis=1).dropna().corr().iloc[0, 1]
    print(f"  corr(sorpresa aggregata, nazionale) = {cc:.2f}  "
          f"({'molto simili' if cc > 0.9 else 'distinte' if cc < 0.7 else 'correlate'})")
    out = []
    for lab, sp in (("aggregata", surp_agg), ("nazionale", surp_nat)):
        tab = _reg(lam, sp)
        tab.insert(0, "mkt", mkt); tab.insert(1, "surprise", lab)
        out.append(tab)
        a, f = _avg(tab)
        print(f"  [{lab}]  media beta 3-10y = {a:+.2f} bp | signif. {f*100:.0f}%")
        print(f"    {_fmt(tab)}")
    # anche ex-crisi sulla nazionale (robustezza del canale fiscale nazionale)
    mask = ~((lam.index >= pd.Timestamp(CRISIS[0])) & (lam.index <= pd.Timestamp(CRISIS[1])))
    tab_ex = _reg(lam.loc[mask], surp_nat.loc[mask])
    tab_ex.insert(0, "mkt", mkt); tab_ex.insert(1, "surprise", "nazionale_ex-crisi")
    out.append(tab_ex)
    a_ex, f_ex = _avg(tab_ex)
    print(f"  [nazionale, ex-crisi 2011-12]  media beta 3-10y = {a_ex:+.2f} bp | signif. {f_ex*100:.0f}%")
    # verdetto per mercato
    a_nat = _avg(out[1])[0]; a_agg = _avg(out[0])[0]
    ratio = a_nat / a_agg if a_agg else np.nan
    print(f"  >>> nazionale/aggregata = {ratio*100:.0f}% | nazionale ex-crisi = {a_ex:+.2f}")
    if mkt == "DE":
        if abs(a_nat) < 0.7 or _avg(out[1])[1] < 0.4:
            print("  >>> DE (controllo): il canale e' MUTO sulla sorpresa nazionale, come atteso")
            print("      per un sovrano a rischio ~0. Coerente con la teoria fiscale.")
        else:
            print("  >>> DE reagisce anche alla sorpresa nazionale: ATTENZIONE, il canale non e'")
            print("      puramente fiscale (la Germania non dovrebbe avere premio sovrano).")
    else:
        if a_nat > 0 and _avg(out[1])[1] >= 0.5 and (np.isnan(ratio) or ratio > 0.5):
            surv = "e resta fuori crisi" if a_ex > 0.5 * a_nat and a_ex > 0 else "ma si attenua fuori crisi"
            print(f"  >>> {mkt}: lambda REAGISCE alla sorpresa NAZIONALE {surv}.")
            print("      -> il canale fiscale e' BLINDATO: e' l'inflazione del paese a muovere")
            print("         il suo breakeven, non solo la componente comune euro.")
        else:
            print(f"  >>> {mkt}: l'effetto SVANISCE o si indebolisce sulla sorpresa nazionale")
            print("      -> era in larga parte la componente comune euro, non il canale nazionale.")
    return pd.concat(out)


if __name__ == "__main__":
    res = []
    for mkt in ("IT", "FR", "DE"):
        t = run(mkt)
        if not t.empty:
            res.append(t)
    print("\n" + "=" * 78)
    print("VERDETTO TEST 4 -- il canale fiscale regge sulla sorpresa NAZIONALE?")
    print("=" * 78)
    if res:
        allr = pd.concat(res)
        for mkt in ("IT", "FR", "DE"):
            n = allr[(allr.mkt == mkt) & (allr.surprise == "nazionale")]
            a = _avg(n)[0] if len(n) else np.nan
            role = " (controllo, atteso ~0)" if mkt == "DE" else ""
            print(f"  {mkt}{role}: lambda~InfS_nazionale, beta 3-10y = {a:+.2f} bp")
        print("\n  lettura per la scelta del paper:")
        print("   - IT/FR reagiscono alla sorpresa nazionale + DE muto = canale fiscale CONFERMATO")
        print("     sull'inflazione del paese -> teoria propria solida, blindata, pubblicabile.")
        print("     A quel punto l'euro-linker ha una teoria testata come lo slow-moving capital")
        print("     del Paper 1, ma su una domanda NUOVA (canale fiscale del breakeven sovrano).")
        print("   - effetto solo sull'aggregata = shock d'AREA, non nazionale: storia piu' debole,")
        print("     i CCT restano preferibili (meccanismo gia' identificato).")
    allr.to_csv(CACHE / "an_national_surprise.csv", index=False) if res else None
    print("\nsalvato: an_national_surprise.csv")

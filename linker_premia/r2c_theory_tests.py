"""r2c - TEST DELLA TEORIA per l'euro-linker: la reazione del breakeven alle sorprese
d'inflazione e' un MECCANISMO STRUTTURALE (canale fiscale: inflazione -> timori
fiscali -> premio sovrano) o solo la correlazione inflazione-spread della crisi 2011-12?
Tre test decisivi. Nessun Bloomberg: legge le stesse serie di r2b (curve, ISR, HICP).

  TEST 1 (il piu' importante): lambda_IT ~ InfS ESCLUDENDO 2011-2012.
    Se +2.98 (t+4.5) sopravvive fuori crisi -> meccanismo strutturale, teoria fiscale
    plausibile e testabile -> l'euro-linker e' un paper. Se sparisce -> era stress,
    nessuna teoria propria.

  TEST 2 (l'esperimento naturale): (lambda_IT - lambda_DE) ~ InfS.
    IT e DE vedono lo STESSO shock (HICP euro) e lo STESSO swap (EUSWI): nella
    differenza inflazione e swap comuni si CANCELLANO, resta il rischio-PAESE. Se il
    differenziale reagisce alle sorprese -> il breakeven differenziale traccia il canale
    fiscale = la teoria propria che ne' Maffei ne' i CCT toccano.

  TEST 3 (la gerarchia): i beta scalano IT > FR > DE (ordine del rischio sovrano)?
    Se si', lambda~InfS e' credito mascherato da inflazione -> la storia e' sovrana
    (gia' quella dei CCT). Se no, e' un fenomeno d'inflazione distinto.

Output: an_theory_tests.csv + verdetto a schermo.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np
import pandas as pd
import rp
from config import CACHE

MATS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20)
REL_DAY = {"IT": 17, "FR": 15, "DE": 15, "US": 12, "UK": 15}
CRISIS = ("2011-01-01", "2012-12-31")          # crisi debito sovrano euro (BTP-Bund > 500bp)
FULL_MIN = "2004-01-01"


# ------------------------------------------------------------------ costruzione lambda alla Maffei
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


def lambda_and_surprise(mkt):
    """Ritorna (lambda EOM-release [frazione], surprise mensile) per un mercato,
    con la STESSA costruzione di r2b (release-date, MA120 piena)."""
    fn = {"US": rp.bei_us, "UK": rp.bei_uk}.get(mkt, lambda m: rp.bei_euro(mkt, m))
    bei_d, isr_d = fn(MATS), rp.isr(mkt, MATS)
    months = pd.date_range(bei_d.index.min().normalize().replace(day=1),
                           bei_d.index.max(), freq="ME")
    grid = _release_grid(mkt, months)
    bei, s = _sample_at(bei_d, grid), _sample_at(isr_d, grid)
    idx = bei.index.intersection(s.index)
    lam = (s.loc[idx] - bei.loc[idx])
    src = "US_SA" if (mkt == "US" and (CACHE / "an_cpi_US_SA.parquet").exists()) else mkt
    pi = rp.yoy(rp.cpi(src)).dropna()
    surp = (pi - pi.rolling(120, min_periods=120).mean()).dropna().reindex(lam.index)
    return lam, surp


def reg_by_tenor(dep, x, lags=6):
    """Regressione by-T: dep(T) [bp] su x. Ritorna DataFrame(mat, n, beta, t, R2)."""
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


def _avg_line(tab, lo=3, hi=10):
    """Media dei beta e frazione significativa sul tratto [lo,hi]y (il cuore della curva)."""
    m = tab[(tab.mat >= lo) & (tab.mat <= hi)]
    if m.empty:
        return np.nan, np.nan, 0
    return m.beta.mean(), (m.t.abs() >= 1.96).mean(), len(m)


# ================================================================== TEST 1: IT fuori crisi
def test1():
    print("\n" + "=" * 78)
    print("TEST 1 -- lambda_IT ~ InfS: il segno regge ESCLUDENDO la crisi 2011-2012?")
    print("=" * 78)
    lam, surp = lambda_and_surprise("IT")
    out = []
    for label, sl in (("full", slice(pd.Timestamp(FULL_MIN), None)),
                      ("ex-crisi 2011-12", None)):
        if label == "full":
            l, s = lam.loc[sl], surp.loc[sl]
        else:
            mask = ~((lam.index >= pd.Timestamp(CRISIS[0])) & (lam.index <= pd.Timestamp(CRISIS[1])))
            l, s = lam.loc[mask], surp.loc[mask]
        tab = reg_by_tenor(l, s)
        tab.insert(0, "test", "T1_IT_infS"); tab.insert(1, "sample", label)
        out.append(tab)
        avg, frac, k = _avg_line(tab)
        print(f"\n  [{label}]  n={int(tab.n.iloc[0])}")
        print(f"    {_fmt(tab)}")
        print(f"    media beta 3-10y = {avg:+.2f} bp | signif. {frac*100:.0f}% dei tenor centrali")
    # verdetto
    a_full = _avg_line(out[0])[0]; a_ex = _avg_line(out[1])[0]
    keep = a_ex / a_full if a_full else np.nan
    print(f"\n  >>> il coefficiente centrale passa da {a_full:+.2f} (full) a {a_ex:+.2f} (ex-crisi): "
          f"resta il {keep*100:.0f}%")
    if keep > 0.6 and a_ex > 0:
        print("  >>> SOPRAVVIVE: la reazione NON e' solo la crisi -> meccanismo strutturale,")
        print("      teoria fiscale (inflazione->premio sovrano) PLAUSIBILE e testabile.")
    elif keep > 0.3:
        print("  >>> ATTENUATO ma presente: parte e' crisi, parte struttura. Da approfondire.")
    else:
        print("  >>> SPARISCE fuori crisi: era la correlazione inflazione-spread del 2011-12,")
        print("      NON una teoria propria -> l'euro-linker perde la storia economica.")
    return pd.concat(out)


# ================================================================== TEST 2: differenziale IT-DE
def test2():
    print("\n" + "=" * 78)
    print("TEST 2 -- (lambda_IT - lambda_DE) ~ InfS: l'esperimento naturale sul rischio-paese")
    print("=" * 78)
    print("  inflazione (HICP) e swap (EUSWI) sono COMUNI: nella differenza si cancellano,")
    print("  resta il rischio-PAESE. Se il differenziale reagisce -> canale fiscale isolato.")
    lam_it, surp = lambda_and_surprise("IT")
    lam_de, _ = lambda_and_surprise("DE")
    idx = lam_it.index.intersection(lam_de.index)
    cols = [c for c in lam_it.columns if c in lam_de.columns]
    diff = lam_it.loc[idx, cols] - lam_de.loc[idx, cols]
    out = []
    for label, dep, s in (("full", diff, surp),
                          ("ex-crisi 2011-12", None, None)):
        if label == "full":
            d_, s_ = dep, surp.reindex(diff.index)
        else:
            mask = ~((diff.index >= pd.Timestamp(CRISIS[0])) & (diff.index <= pd.Timestamp(CRISIS[1])))
            d_, s_ = diff.loc[mask], surp.reindex(diff.index).loc[mask]
        tab = reg_by_tenor(d_, s_)
        tab.insert(0, "test", "T2_IT-DE_infS"); tab.insert(1, "sample", label)
        out.append(tab)
        avg, frac, k = _avg_line(tab)
        print(f"\n  [{label}]  n={int(tab.n.iloc[0]) if len(tab) else 0}")
        print(f"    {_fmt(tab)}")
        print(f"    media beta 3-10y = {avg:+.2f} bp | signif. {frac*100:.0f}%")
    a_full = _avg_line(out[0])[0]; a_ex = _avg_line(out[1])[0]
    print(f"\n  >>> differenziale IT-DE su sorprese: {a_full:+.2f} (full), {a_ex:+.2f} (ex-crisi)")
    if a_full > 0 and _avg_line(out[0])[1] >= 0.4:
        surv = " e SOPRAVVIVE fuori crisi" if (a_ex > 0.5 * a_full and a_ex > 0) else " ma si ATTENUA fuori crisi"
        print(f"  >>> il breakeven differenziale REAGISCE alle sorprese{surv}.")
        print("      -> questo isola il rischio-paese al netto di inflazione e swap comuni:")
        print("         e' la TEORIA PROPRIA (canale fiscale) che ne' Maffei ne' i CCT hanno.")
    else:
        print("  >>> il differenziale NON reagisce in modo netto: il rischio-paese non e'")
        print("      isolabile cosi' -> niente teoria propria per questa via.")
    return pd.concat(out)


# ================================================================== TEST 3: gerarchia IT>FR>DE
def test3():
    print("\n" + "=" * 78)
    print("TEST 3 -- gerarchia: i beta di lambda~InfS scalano col rischio sovrano IT>FR>DE?")
    print("=" * 78)
    print("  se si': lambda~InfS e' CREDITO mascherato da inflazione (storia sovrana, = CCT).")
    print("  se no : e' un fenomeno d'inflazione distinto dal credito.")
    out = []
    summary = {}
    for mkt in ("IT", "FR", "DE"):
        lam, surp = lambda_and_surprise(mkt)
        tab = reg_by_tenor(lam.loc[pd.Timestamp(FULL_MIN):], surp.loc[pd.Timestamp(FULL_MIN):])
        tab.insert(0, "test", "T3_hierarchy"); tab.insert(1, "mkt", mkt)
        out.append(tab)
        avg, frac, _ = _avg_line(tab)
        summary[mkt] = avg
        print(f"\n  {mkt}: media beta 3-10y = {avg:+.2f} bp | signif. {frac*100:.0f}%")
        print(f"     {_fmt(tab)}")
    it, fr, de = summary.get("IT", np.nan), summary.get("FR", np.nan), summary.get("DE", np.nan)
    print(f"\n  >>> gerarchia dei coefficienti: IT {it:+.2f} | FR {fr:+.2f} | DE {de:+.2f}")
    if it > fr > de:
        print("  >>> SCALA col rischio sovrano (IT>FR>DE): lambda~InfS e' in larga parte")
        print("      un premio di CREDITO che reagisce all'inflazione -> la storia converge")
        print("      su quella SOVRANA (gia' il meccanismo dei CCT). Meno 'nuova' del previsto.")
    elif it > de:
        print("  >>> ordine parziale (IT>DE ma FR fuori linea): componente di credito presente")
        print("      ma non e' pura gerarchia sovrana -> storia mista inflazione/credito.")
    else:
        print("  >>> NON scala col rischio sovrano: la reazione all'inflazione e' un fenomeno")
        print("      distinto dal credito -> potenzialmente una storia PROPRIA d'inflazione.")
    return pd.concat(out)


if __name__ == "__main__":
    t1 = test1()
    t2 = test2()
    t3 = test3()
    pd.concat([t1, t2, t3]).to_csv(CACHE / "an_theory_tests.csv", index=False)
    print("\n" + "=" * 78)
    print("VERDETTO COMPLESSIVO -- l'euro-linker ha una TEORIA PROPRIA testabile?")
    print("=" * 78)
    print("  leggi i tre verdetti sopra insieme:")
    print("   - T1 regge fuori crisi  +  T2 differenziale reagisce  = teoria fiscale PROPRIA")
    print("     (breakeven sovrano reagisce all'inflazione via canale fiscale) -> puo' battere")
    print("     i CCT per originalita' (apre una domanda nuova, non replica un puzzle noto).")
    print("   - T3 scala IT>FR>DE = la storia e' SOVRANA (= gia' i CCT) -> i CCT restano meglio")
    print("     perche' li' il meccanismo e' gia' identificato con natural experiment.")
    print("   - T1 sparisce = era la crisi -> nessuna teoria -> CCT vincono nettamente.")
    print("\n  ricorda i tre buchi ancora aperti a prescindere dai test:")
    print("   (a) lambda euro NON identificato (ISR~PCA != 0 per IT/DE): mescola infl./credito;")
    print("   (b) sorpresa AGGREGATA (HICP euro) non nazionale: serve CPI italiano per pulizia;")
    print("   (c) confronto col Paper 1: la' la teoria (slow-moving capital) e' testata a pieno.")
    print("\nsalvato: an_theory_tests.csv")

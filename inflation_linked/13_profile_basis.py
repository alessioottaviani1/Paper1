"""13 - PROFILO DELLA BASE: gli estremi sono economia o sono dati sporchi?

Offline, nessuna chiamata Bloomberg.

LA DOMANDA. Una distribuzione con media e mediana molto distanti (ES: media -0.1bp,
mediana +10.1bp, sd 51bp) ha una coda pesante. Due spiegazioni che si escludono:
  ECONOMICA  gli estremi si concentrano in periodi riconoscibili (crisi sovrana 2011-12,
             COVID marzo 2020, picco inflazione 2022) e durano settimane o mesi.
  TECNICA    gli estremi sono date isolate, sparse, spesso sulle stesse date in cui il
             fit NSS e' degradato (rmse_bp alto) o su un singolo bond.
Questo script le separa: distribuzione per anno, per bond, i peggiori casi con la
diagnostica del fit IN QUELLA data, e il test di sovrapposizione fra date estreme e
date a fit degradato.

Non decide al posto tuo: mette i numeri uno accanto all'altro.

Prima di lanciare: 04_basis_markets.py sul mercato scelto.
"""
import numpy as np
import pandas as pd
from config import CACHE

# ----------------------------------------------------------------- impostazioni
MERCATI = ["ES", "IT", "DE", "FR"]
MISURA  = "basis_zspread"     # o basis_cexact_cc / basis_nearest_cc
CODA    = 0.01                # 1% per lato = "estremo"
N_PEGGIO = 15

# ----------------------------------------------------------------- esecuzione
def _nss(mkt):
    """Diagnostica del fit per data. Assente per DE/UK/US: curva da fonte esterna."""
    p = CACHE / f"nss_{mkt}.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _mat(mkt):
    """Scadenze per ISIN: servono a distinguere un estremo di mercato da uno di fine vita.
    Lo z-spread scala come 1/duration, quindi va letto sempre accanto alla vita residua."""
    try:
        import bbg
        r = bbg.load("ref_linker")
        return pd.to_datetime(r[r["mkt"] == mkt]["MATURITY"])
    except Exception:
        return None

for mkt in MERCATI:
    p = CACHE / f"{MISURA}_{mkt}.parquet"
    if not p.exists():
        print(f"\n### {mkt}: manca {p.name} -> lancia 04 su {mkt}\n")
        continue
    df = pd.read_parquet(p)
    s = df.stack().dropna()
    s.index.names = ["data", "isin"]
    print(f"\n{'='*74}\n### {mkt}   {MISURA}   {len(s)} osservazioni, "
          f"{df.shape[1]} bond, {df.index.min():%Y-%m} -> {df.index.max():%Y-%m}")
    print(f"  media {s.mean():+7.2f}  mediana {s.median():+7.2f}  sd {s.std():6.2f}   "
          f"skew {s.skew():+.2f}")
    print(f"  p1 {s.quantile(.01):+8.2f}   p99 {s.quantile(.99):+8.2f}   "
          f"min {s.min():+8.2f}   max {s.max():+8.2f}")

    # --- per anno: dove sta la coda nel tempo -----------------------------------
    anno = s.index.get_level_values("data").year
    g = s.groupby(anno)
    print(f"\n  {'anno':<6}{'n':>7}{'media':>9}{'mediana':>9}{'sd':>8}{'p1':>9}{'p99':>9}")
    for a, v in g:
        print(f"  {a:<6}{len(v):>7}{v.mean():>9.2f}{v.median():>9.2f}{v.std():>8.2f}"
              f"{v.quantile(.01):>9.2f}{v.quantile(.99):>9.2f}")

    # --- dove cadono gli estremi ------------------------------------------------
    lo, hi = s.quantile(CODA), s.quantile(1 - CODA)
    est = s[(s < lo) | (s > hi)]
    per_anno = est.groupby(est.index.get_level_values("data").year).size()
    quota = (per_anno / g.size()).dropna()
    print(f"\n  estremi (oltre {CODA:.0%}/{1-CODA:.0%}): {len(est)} osservazioni")
    print(f"  quota di estremi per anno (se concentrata -> economia; se piatta -> rumore):")
    for a, q in quota.sort_values(ascending=False).head(6).items():
        print(f"    {a}: {q:.1%}  ({int(per_anno[a])} su {int(g.size()[a])})")

    # --- per bond: e' un titolo solo? -------------------------------------------
    pb = s.groupby(level="isin").agg(["count", "mean", "median", "std", "min", "max"])
    print(f"\n  per bond:")
    print(f"    {'isin':<16}{'n':>7}{'media':>9}{'mediana':>9}{'sd':>8}{'min':>9}{'max':>9}")
    for i, r in pb.sort_values("std", ascending=False).head(8).iterrows():
        print(f"    {i:<16}{int(r['count']):>7}{r['mean']:>9.2f}{r['median']:>9.2f}"
              f"{r['std']:>8.2f}{r['min']:>9.2f}{r['max']:>9.2f}")

    # --- incrocio con la qualita' del fit ---------------------------------------
    nss = _nss(mkt)
    if nss is None or "rmse_bp" not in nss.columns:
        print(f"\n  (nessun nss_{mkt}.parquet: curva da fonte esterna, incrocio non applicabile)")
    else:
        rm = nss["rmse_bp"]
        d_est = pd.Index(est.index.get_level_values("data").unique())
        d_all = pd.Index(s.index.get_level_values("data").unique())
        bad = rm[rm > 12].index
        q_est = len(d_est.intersection(bad)) / max(len(d_est), 1)
        q_all = len(d_all.intersection(bad)) / max(len(d_all), 1)
        print(f"\n  fit NSS: {len(bad)} date con rmse > 12bp su {len(rm)}")
        print(f"    quota di date a fit degradato fra le date ESTREME  : {q_est:.2%}")
        print(f"    quota fra TUTTE le date                            : {q_all:.2%}")
        if q_all > 0:
            r = q_est / q_all
            print(f"    rapporto {r:.1f}x  -> {'gli estremi NASCONO dal fit degradato' if r > 2 else 'nessuna concentrazione: il fit NON spiega gli estremi'}")

    # --- i peggiori, con la diagnostica del fit di quel giorno ------------------
    mat = _mat(mkt)
    # quanta parte degli estremi e' solo fine vita? Con lo z-spread che scala come
    # 1/duration, un bond a 4 mesi dalla scadenza amplifica meccanicamente.
    if mat is not None:
        ttm_all = pd.Series([(mat.get(i, pd.NaT) - d).days
                             for d, i in s.index], index=s.index).dropna()
        ttm_est = ttm_all.reindex(est.index).dropna()
        if len(ttm_est):
            print(f"\n  vita residua: mediana {ttm_all.median():.0f}gg su tutte le "
                  f"osservazioni, {ttm_est.median():.0f}gg sugli ESTREMI")
            for soglia in (180, 365):
                q_e = (ttm_est < soglia).mean(); q_a = (ttm_all < soglia).mean()
                print(f"    sotto {soglia}gg: {q_e:6.1%} degli estremi contro "
                      f"{q_a:5.1%} di tutte   ({q_e/max(q_a,1e-9):.1f}x)")

    print(f"\n  {N_PEGGIO} osservazioni piu' estreme:")
    hdr = f"    {'data':<12}{'isin':<16}{'valore':>10}"
    if mat is not None:
        hdr += f"{'vita res.':>11}"
    if nss is not None and "rmse_bp" in nss.columns:
        hdr += f"{'rmse_fit':>10}{'n_bonds':>9}"
    print(hdr)
    for (d, i), v in s.reindex(s.abs().sort_values(ascending=False).index).head(N_PEGGIO).items():
        line = f"    {d:%Y-%m-%d}  {i:<16}{v:>10.1f}"
        if mat is not None:
            m = mat.get(i, pd.NaT)
            line += f"{((m - d).days if pd.notna(m) else -1):>9}gg" if pd.notna(m) else f"{'n/d':>11}"
        if nss is not None and "rmse_bp" in nss.columns and d in nss.index:
            line += f"{nss.at[d, 'rmse_bp']:>10.1f}{int(nss.at[d, 'n_bonds']):>9}"
        print(line)

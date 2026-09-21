"""11 - COPERTURA DELLE SERIE CPI in cache.

Offline. Dice da quando parte ogni serie CPI scaricata e se e' abbastanza lunga per:
  - la stagionalita' a inizio campione (InflationEngine years_history=10: la prima data
    utile dei pannelli vuole 10 anni di CPI a monte);
  - il base_cpi dei linker piu' vecchi (serve la storia fino a data_base - lag mesi).

Se una serie e' troppo corta: abbassa il floor in bbg.fetch_cpi, CANCELLA il parquet
(_update_wide e' incrementale e altrimenti non riscarica la coda vecchia) e rilancia
02_download.py con le sole fasi CPI.
"""
import pandas as pd
import bbg
from config import MARKETS, CACHE

ANNI_STAGIONALITA = 10

serie = sorted({MARKETS[c].cpi for c in MARKETS})
print(f"{'serie':<12}{'da':<10}{'a':<10}{'punti':>7}   stato")
for tk in serie:
    p = CACHE / f"cpi_{tk}.parquet"
    if not p.exists():
        print(f"{tk:<12}{'-':<10}{'-':<10}{'-':>7}   non scaricata")
        continue
    c = bbg.load(f"cpi_{tk}").iloc[:, 0].dropna()
    c.index = pd.to_datetime(c.index)
    mkts = [k for k, v in MARKETS.items() if v.cpi == tk]
    # prima data utile dei pannelli di quei mercati
    primi = []
    for mk in mkts:
        q = CACHE / f"basis_cexact_{mk}.parquet"
        if q.exists():
            primi.append(pd.read_parquet(q).index.min())
    stato = "ok"
    if primi:
        serve = min(primi) - pd.DateOffset(years=ANNI_STAGIONALITA)
        if c.index.min() > serve:
            manca = (c.index.min().year - serve.year) * 12 + (c.index.min().month - serve.month)
            stato = f"CORTA: per la stagionalita' servirebbe da {serve:%Y-%m} ({manca} mesi in meno)"
    print(f"{tk:<12}{c.index.min():%Y-%m}{'':<3}{c.index.max():%Y-%m}{'':<3}{len(c):>7}   {stato}")
    print(f"{'':12}mercati: {', '.join(mkts)}   file: {p}")

print("\nPer allungare una serie:")
print("  1. bbg.fetch_cpi: abbassa il floor (es. pd.Timestamp('1980-01-01'))")
print("  2. CANCELLA cache/cpi_<TICKER>.parquet   <-- senza questo non riscarica")
print("  3. 02_download.py con FASI = dict(ANAGRAFICA=False, PREZZI=False,")
print("     NOMINALI=False, ILS=False, CPI=True)")

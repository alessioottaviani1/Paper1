"""07 - GATE DI REGRESSIONE: il nuovo codice riproduce la base BTPei dell'originale?

config.py lo dichiara come vincolo di sicurezza della riscrittura ("finche' questo non
passa, non si tocca nulla d'altro") ma REGRESSION_TARGET non era usata da nessuno script:
questo colma il buco.

Confronta basis_nearest_IT.parquet con data/btpei_basis.xlsx sull'INTERSEZIONE di date e
ISIN, tolleranza REGRESSION_TOL_BP (0.01bp). Il confronto e' su basis_nearest, NON sulle
colonne _cc o zspread: quelle sono misure nuove e non hanno un target storico.

Se fallisce, i sospetti in ordine:
  1. FURTHER_IT_HOLIDAYS vuota  -> business date italiane diverse (causa piu' probabile)
  2. ytm_convention             -> l'originale non convertiva; per IT non cambia nulla
  3. PX_FIELD                   -> l'originale usa PX_ASK, il default qui e' px_mid
"""
import numpy as np
import pandas as pd
from config import CACHE, REGRESSION_TARGET, REGRESSION_TOL_BP

# ----------------------------------------------------------------- impostazioni
MERCATO  = "IT"
SHEET    = 0          # foglio di btpei_basis.xlsx; cambialo se non e' il primo
COL_DATA = None       # None = prima colonna come indice; oppure il nome della colonna data

# ----------------------------------------------------------------- esecuzione
p_new = CACHE / f"basis_nearest_{MERCATO}.parquet"
if not p_new.exists():
    raise SystemExit(f"manca {p_new}\n-> lancia prima 04_basis_markets.py con MERCATI = ['{MERCATO}']")
if not REGRESSION_TARGET.exists():
    raise SystemExit(f"manca il target {REGRESSION_TARGET}")

new = pd.read_parquet(p_new)
old = pd.read_excel(REGRESSION_TARGET, sheet_name=SHEET,
                    index_col=(0 if COL_DATA is None else COL_DATA))
old.index = pd.to_datetime(old.index)
new.index = pd.to_datetime(new.index)
old.columns = [str(c).strip() for c in old.columns]
new.columns = [str(c).strip() for c in new.columns]

print(f"target  {old.shape[0]} date x {old.shape[1]} ISIN   {old.index.min():%Y-%m-%d} -> {old.index.max():%Y-%m-%d}")
print(f"nuovo   {new.shape[0]} date x {new.shape[1]} ISIN   {new.index.min():%Y-%m-%d} -> {new.index.max():%Y-%m-%d}")

idx, col = old.index.intersection(new.index), old.columns.intersection(new.columns)
print(f"\nintersezione: {len(idx)} date x {len(col)} ISIN")
if not len(idx) or not len(col):
    print("\nnessuna sovrapposizione: controlla SHEET / COL_DATA e il formato degli ISIN.")
    print(f"  esempio colonne target: {list(old.columns[:4])}")
    print(f"  esempio colonne nuove : {list(new.columns[:4])}")
    raise SystemExit(1)
if len(col) < old.shape[1]:
    print(f"  ISIN nel target ma non nel nuovo: {sorted(set(old.columns) - set(col))}")

d = (new.loc[idx, col] - old.loc[idx, col]).stack().dropna()
n_bad = int((d.abs() > REGRESSION_TOL_BP).sum())
print(f"\nconfronti validi: {len(d)}")
print(f"scarto  max {d.abs().max():.6f}bp   mediana {d.median():+.6f}bp   sd {d.std():.6f}bp")
print(f"fuori tolleranza ({REGRESSION_TOL_BP}bp): {n_bad} ({n_bad/max(len(d),1)*100:.3f}%)")

if n_bad == 0:
    print("\n>>> REGRESSIONE OK: la riscrittura riproduce l'originale. Si puo' procedere.")
else:
    print("\n>>> REGRESSIONE FALLITA.")
    worst = d.abs().sort_values(ascending=False).head(10)
    print("\ndieci scarti peggiori (data, isin, bp):")
    for (dt, isin), v in worst.items():
        print(f"  {dt:%Y-%m-%d}  {isin}  {d.loc[(dt, isin)]:+.4f}")
    per_isin = d.abs().groupby(level=1).max().sort_values(ascending=False)
    print(f"\nscarto max per ISIN (primi 5):\n{per_isin.head(5).to_string()}")
    if per_isin.head(5).min() > 0.01:
        print("\n-> lo scarto e' diffuso su tutti gli ISIN: sospetto calendario")
        print("   (FURTHER_IT_HOLIDAYS), non un bond specifico.")

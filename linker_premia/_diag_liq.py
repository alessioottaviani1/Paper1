import pandas as pd
from pathlib import Path
import sys; sys.path.insert(0, r".\src\inflation_linked")
from config import CACHE
d = pd.read_parquet(CACHE / "an_liq.parquet"); d.index = pd.to_datetime(d.index)
print("copertura per proxy (prima e ultima data non-NaN, n valori):")
for c in d.columns:
    s = d[c].dropna()
    print(f"  {c:8s}: {s.index.min().date()} -> {s.index.max().date()} | n={len(s)} | std={s.std():.3g}")
m = d.resample("ME").last()
print("\nmesi con TUTTE e 4 le proxy presenti:", int(m.dropna().shape[0]), "su", len(m))
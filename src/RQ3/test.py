import sys
from pathlib import Path
import numpy as np
import pandas as pd
import re

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_00_config import RAW_BASIS_FILES

cfg = RAW_BASIS_FILES["itraxx_combined"]

print("="*70)
print("DIAGNOSTICA SCALA SKEW per sub-indice iTraxx")
print("="*70)

otr_by_index = {}
for idx_name, fpath in cfg["paths"].items():
    if not fpath.exists():
        print(f"  ⚠️ {idx_name}: file non trovato")
        continue
    wide = pd.read_parquet(fpath)
    b_cols = [c for c in wide.columns if c.endswith(cfg["suffix"])]
    # on-the-run: serie col numero più alto
    series_nums = {c: int(re.search(r'Ser(\d+)', c).group(1))
                   for c in b_cols if re.search(r'Ser(\d+)', c)}
    b_cols_sorted = sorted(b_cols, key=lambda c: series_nums.get(c, 0), reverse=True)
    otr = wide[b_cols_sorted].bfill(axis=1).iloc[:, 0].dropna()
    otr = otr.groupby(otr.index).mean()
    otr_by_index[idx_name] = otr

print(f"\n{'Index':<10} {'mean|skew|':>12} {'median|skew|':>14} "
      f"{'p90|skew|':>12} {'std skew':>12} {'N':>6}")
print("-"*70)
for idx_name, otr in otr_by_index.items():
    a = otr.abs()
    print(f"{idx_name:<10} {a.mean():>12.2f} {a.median():>14.2f} "
          f"{a.quantile(0.90):>12.2f} {otr.std():>12.2f} {len(otr):>6}")

# Quanto Xover domina la media delle magnitudini?
print("\n--- Peso di ciascun indice nella media |skew| (quota del totale) ---")
mean_abs = {k: v.abs().mean() for k, v in otr_by_index.items()}
tot = sum(mean_abs.values())
for k, val in mean_abs.items():
    print(f"  {k:<10}: {val:>8.2f} bps  →  {100*val/tot:>5.1f}% del totale")

# Correlazione di ogni sub-indice (in livello skew) con gli altri, per capire
# se aggregarli ha senso o se sono troppo eterogenei
print("\n--- Correlazione tra gli skew dei 4 sub-indici (daily, overlap) ---")
df = pd.DataFrame(otr_by_index).dropna()
print(df.corr().round(2).to_string())
print(f"\n(overlap: {len(df)} giorni)")
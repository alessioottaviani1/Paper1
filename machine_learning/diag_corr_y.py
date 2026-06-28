"""corr(f,y) dei selezionati + verdetto sulle 2 coppie ridondanti. Throwaway."""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cfg", ROOT / "src" / "machine_learning" / "00_config.py")
cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)

PAIRS = [("SS5Y", "SS10Y"), ("ATM_IV_ITRX", "ATM_IV_CDX")]
cy_global = {}

for name in cfg.STRATEGIES:
    sd = cfg.get_strategy_aen_dir(name)
    sel = json.load(open(sd / "subset_results.json"))["selected_factors"]
    y = pd.read_parquet(sd / "y_centered.parquet").iloc[:, 0].values
    X = pd.read_parquet(sd / "X_standardized.parquet")
    print("=" * 52)
    print(name)
    for f in sel:
        c = abs(np.corrcoef(X[f].values, y)[0, 1])
        cy_global.setdefault(f, []).append(c)
        print(f"   {f:18s}  |corr y| = {c:.3f}")
    print()

print("=" * 52)
print("VERDETTO coppie ridondanti (tieni il piu' correlato con y):")
for a, b in PAIRS:
    ca = max(cy_global.get(a, [float("nan")]))
    cb = max(cy_global.get(b, [float("nan")]))
    keep, drop = (a, b) if ca >= cb else (b, a)
    print(f"   {a} ({ca:.3f})  vs  {b} ({cb:.3f})   ->   TIENI {keep},  ESCLUDI {drop}")

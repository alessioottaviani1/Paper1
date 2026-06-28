"""Diagnosi: correlazioni tra i fattori selezionati — conferma coppie collineari. Throwaway."""
import warnings; warnings.filterwarnings("ignore")
import json
from pathlib import Path
import importlib.util
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cfg", ROOT / "src" / "machine_learning" / "00_config.py")
cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)

for name in cfg.STRATEGIES:
    sd = cfg.get_strategy_aen_dir(name)
    sel = json.load(open(sd / "subset_results.json"))["selected_factors"]
    X = pd.read_parquet(sd / "X_standardized.parquet")[sel]
    C = X.corr().abs()
    print("=" * 64)
    print(f"{name}  ({len(sel)} fattori selezionati)")
    pairs = []
    for i in range(len(sel)):
        for j in range(i + 1, len(sel)):
            r = float(C.iloc[i, j])
            if r > 0.6:
                pairs.append((sel[i], sel[j], r))
    if pairs:
        print("  coppie correlate tra i selezionati (|ρ| > 0.6):")
        for a, b, r in sorted(pairs, key=lambda x: -x[2]):
            print(f"    {a:18s} ~ {b:18s}  ρ = {r:.3f}")
    else:
        print("  nessuna coppia |ρ| > 0.6 tra i selezionati")
    # per ogni fattore, massima correlazione con un altro selezionato
    print("  max |ρ| di ciascun fattore con gli altri selezionati:")
    for f in sel:
        others = [c for c in sel if c != f]
        mx = C.loc[f, others].max()
        partner = C.loc[f, others].idxmax()
        print(f"    {f:18s}  max ρ = {mx:.3f}  (con {partner})")
    print()

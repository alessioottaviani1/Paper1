# -*- coding: utf-8 -*-
"""
diag_factor_availability.py — quale PCA_START_DATE tiene ~tutti i 75 fattori?

Per la PCA in-sample (full-sample) serve un panel BILANCIATO: un fattore e' usabile
solo se ha dati completi (nessun NaN) su [start, FACTORS_END_DATE]. Questo script,
per una griglia di date d'inizio, conta quanti dei 75 fattori sono completi fino a
fine campione, e mostra i fattori che iniziano piu' tardi (il vincolo sulla start).

Mettilo in src/pca/ accanto a 06_pca_oos.py e lancia:  python src/pca/diag_factor_availability.py
"""
import importlib.util
from pathlib import Path
import pandas as pd

# carica 00_pca_config.py (stesso pattern di 06_pca_oos.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_cfg = PROJECT_ROOT / "src" / "pca" / "00_pca_config.py"
spec = importlib.util.spec_from_file_location("pca_config", _cfg)
cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)

END = pd.Timestamp(cfg.FACTORS_END_DATE)             # 2025-05-31
F = pd.read_parquet(cfg.FACTORS_PATH)
F = F[F.index <= END].sort_index()
N = F.shape[1]
print(f"Parquet: {N} fattori | {F.index.min().date()} -> {F.index.max().date()} | END={END.date()}\n")

# (1) fattori completi su [start, END] per una griglia di start (ogni 3 mesi)
print(f"  {'start':>12}  completi/{N}")
for start in F.index[::3]:
    sub = F.loc[start:]
    if len(sub) < 24:                                # troppa poca storia residua
        break
    n = int(sub.notna().all().sum())
    print(f"  {start.date()!s:>12}  {n:>3}/{N}  {'#' * (n * 45 // N)}")

# (2) i fattori che iniziano piu' tardi = vincolano la start date
fv = F.apply(lambda c: c.first_valid_index()).sort_values(ascending=False)
print("\nFattori che iniziano piu' tardi (vincolano la start date):")
for name, d in fv.head(15).items():
    print(f"   {(str(d.date()) if d is not None else 'ALL-NaN'):>12}  {name}")

# (3) per una data scelta: quanti tieni e quali droppi
CHOSEN = "2009-06-30"                                # <- cambia e rilancia
sub = F.loc[pd.Timestamp(CHOSEN):]
dropped = sorted(sub.columns[~sub.notna().all()])
print(f"\nSe start={CHOSEN}: tieni {N - len(dropped)}/{N}; droppi {len(dropped)}: {dropped}")

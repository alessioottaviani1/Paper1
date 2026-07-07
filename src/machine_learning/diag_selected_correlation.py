"""
diag_selected_correlation.py — Diagnostica di collinearità sui fattori
SELEZIONATI dal best-subset, per ciascuna strategia.

Per ogni strategia:
  - carica selected_factors da subset_results.json (best-subset, 02s)
  - carica i valori standardizzati da X_standardized.parquet
  - stampa la matrice di correlazione tra i soli fattori selezionati
  - stampa il VIF di ciascun fattore selezionato
  - riassume: |corr| massima fuori diagonale, VIF massimo

Lettura: se |corr| max << 0.5 e VIF max < 5, il best-subset ha scelto
fattori distinti (collinearità residua trascurabile) → nessuna nota
necessaria nel paper. Se emergono coppie con |corr| alta o VIF > 5/10,
va menzionato/controllato.

Lancia:  python src/machine_learning/diag_selected_correlation.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

# --- config paths (stessa logica di 00_config) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import importlib.util
_cfg_path = PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"
_spec = importlib.util.spec_from_file_location("aen_config", _cfg_path)
aen_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(aen_config)

STRATEGIES = aen_config.STRATEGIES
get_strategy_aen_dir = aen_config.get_strategy_aen_dir

OUT_DIR = PROJECT_ROOT / "results" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def vif_series(X: pd.DataFrame) -> pd.Series:
    """VIF_j = 1 / (1 - R^2_j), R^2_j da regressione di X_j sugli altri."""
    vifs = {}
    cols = list(X.columns)
    for j, c in enumerate(cols):
        others = [k for k in cols if k != c]
        if not others:
            vifs[c] = 1.0
            continue
        Xo = X[others].to_numpy()
        Xo = np.column_stack([np.ones(len(Xo)), Xo])   # intercetta
        y = X[c].to_numpy()
        beta, *_ = np.linalg.lstsq(Xo, y, rcond=None)
        resid = y - Xo @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vifs[c] = 1.0 / (1.0 - r2) if r2 < 1 else np.inf
    return pd.Series(vifs)


def run():
    summary = {}
    for strat in STRATEGIES:
        sd = get_strategy_aen_dir(strat)
        sub_fp = sd / "subset_results.json"
        X_fp = sd / "X_standardized.parquet"
        if not sub_fp.exists() or not X_fp.exists():
            print(f"\n[{strat}] file mancanti (subset_results.json o X_standardized.parquet) — salto")
            continue

        selected = json.loads(sub_fp.read_text())["selected_factors"]
        X = pd.read_parquet(X_fp)
        missing = [f for f in selected if f not in X.columns]
        if missing:
            print(f"\n[{strat}] ⚠ fattori selezionati assenti nel parquet: {missing}")
        selected = [f for f in selected if f in X.columns]

        Xs = X[selected]
        corr = Xs.corr()
        vif = vif_series(Xs)

        # |corr| max fuori diagonale
        c = corr.to_numpy().copy()
        np.fill_diagonal(c, 0.0)
        abs_c = np.abs(c)
        imax = np.unravel_index(np.argmax(abs_c), abs_c.shape)
        max_corr = abs_c[imax]
        pair = (selected[imax[0]], selected[imax[1]])

        print("\n" + "=" * 78)
        print(f"[{strat}]  {len(selected)} fattori selezionati")
        print("=" * 78)
        print("\nMatrice di correlazione (fattori selezionati):")
        print(corr.round(2).to_string())
        print("\nVIF:")
        print(vif.round(2).to_string())
        print(f"\n  → |corr| massima fuori diagonale: {max_corr:.2f}  ({pair[0]} ~ {pair[1]})")
        print(f"  → VIF massimo: {vif.max():.2f}  ({vif.idxmax()})")

        # verdetto automatico
        if max_corr < 0.5 and vif.max() < 5:
            verdict = "OK: fattori distinti, collinearità residua trascurabile"
        elif vif.max() < 10:
            verdict = "ATTENZIONE: collinearità moderata (VIF 5-10 o corr>=0.5) — menzionare"
        else:
            verdict = "PROBLEMA: collinearità seria (VIF>10) — da controllare"
        print(f"  → {verdict}")

        # salva CSV per strategia
        corr.to_csv(OUT_DIR / f"selected_corr_{strat}.csv")
        vif.to_frame("VIF").to_csv(OUT_DIR / f"selected_vif_{strat}.csv")
        summary[strat] = {"n": len(selected), "max_abs_corr": round(float(max_corr), 3),
                          "max_corr_pair": list(pair), "max_vif": round(float(vif.max()), 2),
                          "max_vif_factor": vif.idxmax(), "verdict": verdict}

    (OUT_DIR / "selected_collinearity_summary.json").write_text(
        json.dumps(summary, indent=2))
    print("\n" + "=" * 78)
    print("RIEPILOGO")
    print("=" * 78)
    for s, d in summary.items():
        print(f"  {s:<18} n={d['n']:<3} |corr|max={d['max_abs_corr']:<5} "
              f"VIFmax={d['max_vif']:<6} {d['verdict']}")
    print(f"\nSalvato in: {OUT_DIR}")


if __name__ == "__main__":
    run()
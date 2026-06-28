"""EBIC sul path best-subset VERO (abess): conteggio data-driven, metodo uniforme. Throwaway."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from math import lgamma, log
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cfg", ROOT / "src" / "machine_learning" / "00_config.py")
cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
from abess.linear import LinearRegression


def log_choose(p, k):
    return (lgamma(p + 1) - lgamma(k + 1) - lgamma(p - k + 1)) if 0 < k <= p else 0.0


def r2adj(Xz, y, idx):
    T = len(y); k = len(idx)
    if k == 0 or k >= T - 1:
        return 0.0
    A = Xz[:, idx]; b = np.linalg.lstsq(A, y, rcond=None)[0]
    rss = float(np.sum((y - A @ b) ** 2)); tss = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - rss / tss
    return 1.0 - (1.0 - r2) * (T - 1) / (T - k - 1)


def best_subset(Xz, y, k):
    m = LinearRegression(support_size=k, fit_intercept=False); m.fit(Xz, y)
    return sorted(np.where(np.abs(m.coef_) > 0)[0].tolist())


BMK = {"btp_italia": 0.130, "cds_bond_basis": 0.173, "itraxx_combined": 0.102}
PCA = {"btp_italia": 0.088, "cds_bond_basis": 0.102, "itraxx_combined": 0.134}
KMAX = 15

for name in cfg.STRATEGIES:
    sd = cfg.get_strategy_aen_dir(name)
    y = pd.read_parquet(sd / "y_centered.parquet").iloc[:, 0].values.astype(float)
    Xdf = pd.read_parquet(sd / "X_standardized.parquet")
    names = list(Xdf.columns); Xz = Xdf.values.astype(float)
    T, p = Xz.shape
    g = 1.0 - log(T) / (2.0 * log(p))
    print("=" * 70)
    print(f"{name}  (Duarte={BMK[name]:.3f}  PCA={PCA[name]:.3f})  EBIC gamma={g:.3f}")

    rows, best = [], (None, np.inf, [])
    for k in range(1, KMAX + 1):
        idx = best_subset(Xz, y, k)
        A = Xz[:, idx]; b = np.linalg.lstsq(A, y, rcond=None)[0]
        rss = float(np.sum((y - A @ b) ** 2))
        eb = T * log(rss / T) + k * log(T) + 2.0 * g * log_choose(p, k)
        rows.append((k, eb, r2adj(Xz, y, idx)))
        if eb < best[1]:
            best = (k, eb, idx)

    kstar = best[0]
    win = "sì" if (r2adj(Xz, y, best[2]) > BMK[name] and r2adj(Xz, y, best[2]) > PCA[name]) else \
          ("solo PCA" if r2adj(Xz, y, best[2]) > PCA[name] else "NO")
    print(f"  EBIC* → k={kstar}  R2adj={r2adj(Xz, y, best[2]):.3f}  vince benchmark+PCA? {win}")
    print(f"  fattori: {[names[i] for i in best[2]]}")
    print(f"  curva EBIC:")
    for k, eb, r in rows:
        mark = "  <-- EBIC*" if k == kstar else ""
        print(f"    k={k:2d}:  EBIC={eb:9.2f}   R2adj={r:.3f}{mark}")
    print()

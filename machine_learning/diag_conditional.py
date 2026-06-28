"""Throwaway: marginale vs condizionale dei fattori attesi, dato il base AEN. Cancellare dopo l'uso."""
import pandas as pd, numpy as np, statsmodels.api as sm
from pathlib import Path
import importlib.util, warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path  = PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"
spec = importlib.util.spec_from_file_location("aen_config", config_path)
aen_config = importlib.util.module_from_spec(spec); spec.loader.exec_module(aen_config)
get_strategy_aen_dir = aen_config.get_strategy_aen_dir

# base = fattori che l'AEN HA selezionato (dal tuo ultimo run) | extra = quelli che TI ASPETTI contino
SETUP = {
    "cds_bond_basis": {"base": ["EMERG_DEBT", "RI_EU"],
                       "extra": ["CRED_SPR_US", "CRED_SPR_EU", "DEF_US", "PB_CDS_1Y_EU", "PB_CDS_5Y_EU", "SS5Y"]},
    "btp_italia":     {"base": ["FI_DEF"],
                       "extra": ["SMB_EU", "UMD_EU", "CDX_IG", "SS10Y", "LIQ_V"]},
}

def fit(y, X, cols):
    return sm.OLS(y, sm.add_constant(X[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 6})

def hqc_ols(y, X, cols):
    if not cols:
        rss = float(np.sum(y.values ** 2)); return len(y) * np.log(rss / len(y))
    A = X[cols].values; b = np.linalg.lstsq(A, y.values, rcond=None)[0]
    rss = float(np.sum((y.values - A @ b) ** 2)); T, k = len(y), len(cols)
    return T * np.log(rss / T) + 2 * np.log(np.log(T)) * k   # df = k (dati centrati, no const)

for sn, cfg in SETUP.items():
    sdir = get_strategy_aen_dir(sn)
    X = pd.read_parquet(sdir / "X_standardized.parquet")
    y = pd.read_parquet(sdir / "y_centered.parquet")["y"]
    y = y.loc[X.index]

    base    = [c for c in cfg["base"]  if c in X.columns]
    extra   = [c for c in cfg["extra"] if c in X.columns]
    missing = [c for c in cfg["base"] + cfg["extra"] if c not in X.columns]

    print("=" * 72); print(f"{sn}  (T={len(y)})")
    if missing: print("  ⚠️  nomi NON trovati in X (correggili):", missing)

    m0, m1 = fit(y, X, base), fit(y, X, base + extra)
    print(f"\n  base adjR² = {m0.rsquared_adj:.4f}   {base}")
    print(f"  full adjR² = {m1.rsquared_adj:.4f}   +{extra}")
    print("\n  fattore           |  MARGINALE (univar)  |  CONDIZIONALE (dato base)")
    print("  " + "-" * 62)
    for c in extra:
        tm = fit(y, X, [c]).tvalues[c]
        tc = m1.tvalues[c]
        print(f"  {c:16s}  |   t = {tm:+6.2f}        |   t = {tc:+6.2f}{'   <-- conta in più' if abs(tc) > 2 else ''}")

    # forward-stepwise per HQC su TUTTI i 75: cosa vuole davvero l'HQC, oltre i 6 che ho scelto io?
    chosen = []
    while len(chosen) < 10:
        h_now = hqc_ols(y, X, chosen)
        h_best, c_best = min((hqc_ols(y, X, chosen + [c]), c) for c in X.columns if c not in chosen)
        if h_best < h_now - 1e-9:
            chosen.append(c_best)
        else:
            break
    print(f"\n  HQC  AEN-base ({len(base)}f)        = {hqc_ols(y, X, base):8.2f}   {base}")
    print(f"  HQC  forward-optimal ({len(chosen)}f) = {hqc_ols(y, X, chosen):8.2f}")
    print(f"       {chosen}")
    print()


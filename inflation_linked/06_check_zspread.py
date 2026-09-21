"""06 - ACCETTAZIONE delle nuove misure (basis_zspread, *_cc).

Offline: legge solo i parquet gia' in cache, nessun Bloomberg.
Lanciare DOPO 04_basis_markets.py sul mercato indicato in MERCATO.

  1. zspread vs cexact_cc  -> devono coincidere entro TOL_BP salvo l'ultimo periodo
     cedolare, dove synthetic_irr passa al rendimento monetario semplice ACT/365 mentre
     lo zspread resta esponenziale (divergenza per costruzione). Il controllo localizza
     la coda per vita residua, cosi' non serve rilanciare 04 con ESCLUDI_CODA per saperlo.
  2. cexact / cexact_cc    -> il rapporto NON e' (1+y/100) ma la MEDIA LOGARITMICA di
     (1+y_tot/100) e (1+y_syn/100):  ce/cc = (a-b)/ln(a/b).  Identita' esatta, quindi il
     residuo atteso e' rumore float (1e-9). Se e' grande, la conversione cc e' rotta.
  3. copertura             -> zspread deve popolare le stesse celle di cexact, e nessun
     ISIN deve avere copertura anomala (segnala i bond che il loop ha perso).
"""
import numpy as np
import pandas as pd
from config import CACHE

# ----------------------------------------------------------------- impostazioni
MERCATO = "ES"
TOL_BP  = 0.10        # scarto MEDIANO accettato fra zspread e cexact_cc
SUF     = ""          # "_lag20" se hai girato con PUB_LAG_DAYS

# ----------------------------------------------------------------- esecuzione
def _load(name):
    p = CACHE / f"{name}_{MERCATO}{SUF}.parquet"
    if not p.exists():
        raise SystemExit(f"manca {p}\n-> lancia prima 04_basis_markets.py con MERCATI = ['{MERCATO}']")
    return pd.read_parquet(p)

ce, cc = _load("basis_cexact"), _load("basis_cexact_cc")
zs, yt = _load("basis_zspread"), _load("totalytm")
nr, nc = _load("basis_nearest"), _load("basis_nearest_cc")

print(f"=== {MERCATO}: {ce.shape[1]} linker x {ce.shape[0]} date ===\n")

# --- 1. zspread vs cexact_cc
d = (zs - cc).stack().dropna()
med = d.median()
print(f"1. zspread - cexact_cc   mediana {med:+.4f}bp   p99 {d.abs().quantile(.99):.4f}bp   "
      f"max {d.abs().max():.4f}bp")
print(f"   {'OK' if abs(med) < TOL_BP else 'ATTENZIONE'}: soglia mediana {TOL_BP}bp")

# dove sta la coda? -> vita residua. Se e' tutta sotto l'anno, e' l'ultimo periodo cedolare.
try:
    import bbg
    mats = bbg.load("ref_linker")["MATURITY"]
    ttm = pd.DataFrame({c: (pd.to_datetime(mats.get(c)) - ce.index).days / 365.25
                        for c in ce.columns if c in mats.index}, index=ce.index)
    big = d[d.abs() > 1.0]
    if len(big):
        t = ttm.stack().reindex(big.index).dropna()
        print(f"   coda |diff|>1bp: {len(big)} celle ({len(big)/len(d)*100:.3f}%), "
              f"vita residua mediana {t.median():.2f}a, p90 {t.quantile(.90):.2f}a")
        msg = ("coda confinata a fine vita: e' il ramo ACT/365, atteso"
               if t.quantile(.90) < 1.5 else
               "ATTENZIONE: la coda NON e' solo a fine vita, da investigare")
        print(f"   -> {msg}")
    else:
        print("   nessuna cella oltre 1bp")
except Exception as e:
    print(f"   (diagnostica vita residua saltata: {e})")

# --- 2. identita' esatta: ce/cc = media logaritmica
a = 1 + yt / 100.0
b = 1 + (yt - ce / 100.0) / 100.0            # y_syn = y_tot - cexact/100
with np.errstate(divide="ignore", invalid="ignore"):
    Lm = (a - b) / np.log(a / b)
r = (ce / cc)
err = (r - Lm).stack().replace([np.inf, -np.inf], np.nan).dropna().abs()
print(f"\n2. ce/cc vs media logaritmica   mediana {err.median():.3e}   max {err.max():.3e}")
print(f"   {'OK' if err.median() < 1e-9 else 'ATTENZIONE'}: atteso rumore float (< 1e-9)")
print(f"   (base max nel campione: {ce.stack().abs().max():.0f}bp)")

# --- 3. copertura
n_ce, n_zs = int(ce.notna().sum().sum()), int(zs.notna().sum().sum())
print(f"\n3. copertura   cexact {n_ce}   zspread {n_zs}   ({n_zs/max(n_ce,1)*100:.2f}%)")
print(f"   {'OK' if n_zs >= 0.99 * n_ce else 'ATTENZIONE'}: atteso >= 99%")
cov = ce.notna().sum().sort_values()
thin = cov[cov < 0.20 * cov.max()]
if len(thin):
    print(f"   ISIN con copertura < 20% del massimo ({len(thin)}): controllali, il loop li ha persi")
    for isin, n in thin.head(8).items():
        print(f"     {isin}: {int(n)} date su {int(cov.max())}")

# --- bonus: quanto pesa l'effetto cedola nel metodo nearest
gap = (nc - zs).stack().dropna()
if len(gap):
    print(f"\nBONUS  nearest_cc - zspread   mediana {gap.median():+.2f}bp   "
          f"IQR [{gap.quantile(.25):+.2f}, {gap.quantile(.75):+.2f}]bp")
    print("   E' l'effetto cedola + il mismatch di scadenza del matched-maturity, misurato.")

print(f"\n=== livelli medi (bp) ===")
for nome, df in [("nearest", nr), ("nearest_cc", nc), ("cexact", ce),
                 ("cexact_cc", cc), ("zspread", zs)]:
    s = df.stack().dropna()
    print(f"  {nome:12} media {s.mean():+8.2f}   mediana {s.median():+8.2f}   sd {s.std():7.2f}")

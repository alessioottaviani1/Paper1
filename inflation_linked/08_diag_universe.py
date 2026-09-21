"""08 - DIAGNOSTICA UNIVERSO: perche' un linker manca o ha poche osservazioni.

Offline: legge solo i parquet in cache.

Risponde a due domande diverse che 04 e 06 lasciano aperte:

  A) QUALI sono i linker scartati da build_market per BASE_CPI mancante. Vengono filtrati
     PRIMA del loop ("esclusi N linker senza BASE_CPI valida"), quindi non compaiono
     nemmeno come colonne nei pannelli: 06 non li puo' vedere. Qui si applica lo stesso
     filtro di build_market e si stampano gli ISIN, con il nome e le date, pronti per DES.

  B) La copertura bassa e' un PROBLEMA o e' un bond GIOVANE? 06 usa una soglia grezza sul
     massimo e quindi segnala anche le emissioni recenti, che hanno poca storia per
     costruzione. Qui la copertura si misura contro le date ATTESE -- l'intersezione fra
     la vita del bond (first settle -> scadenza) e la finestra effettiva dei pannelli --
     e solo chi sta sotto SOGLIA_ATTESA e' davvero perso.
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE, MARKETS

# ----------------------------------------------------------------- impostazioni
MERCATO       = "ES"
SUF           = ""
SOGLIA_ATTESA = 0.80      # copertura minima sulle date attese prima di segnalare

# ----------------------------------------------------------------- esecuzione
ref = bbg.load("ref_linker")
ref = ref[ref["mkt"] == MERCATO]
p = CACHE / f"basis_cexact_{MERCATO}{SUF}.parquet"
if not p.exists():
    raise SystemExit(f"manca {p}\n-> lancia prima 04_basis_markets.py con MERCATI = ['{MERCATO}']")
pan = pd.read_parquet(p)
try:
    px = bbg.load(f"px_mid_{MERCATO}")
except Exception:
    px = None

def _col(df, *names):
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(np.nan, index=df.index)

name = _col(ref, "SECURITY_NAME", "NAME", "TICKER")
mat  = pd.to_datetime(_col(ref, "MATURITY"), errors="coerce")
fs   = pd.to_datetime(_col(ref, "FIRST_SETTLE_DT"), errors="coerce").fillna(
       pd.to_datetime(_col(ref, "ISSUE_DT"), errors="coerce"))
bcpi = _col(ref, "base_cpi_final")

print(f"=== {MERCATO}: {len(ref)} linker in ref_linker, {pan.shape[1]} colonne nei pannelli ===")
print(f"    finestra pannelli: {pan.index.min():%Y-%m-%d} -> {pan.index.max():%Y-%m-%d}\n")

# ---------------------------------------------------------------- A) BASE_CPI mancante
bad = bcpi.isna() | (bcpi <= 0)
print(f"--- A) SCARTATI per BASE_CPI non valida: {int(bad.sum())} ---")
if bad.any():
    print(f"{'ISIN':<16}{'nome':<34}{'first settle':<14}{'scadenza':<12}{'base_cpi'}")
    for isin in ref.index[bad]:
        print(f"{isin:<16}{str(name.get(isin, ''))[:33]:<34}"
              f"{(f'{fs[isin]:%Y-%m-%d}' if pd.notna(fs.get(isin)) else 'n/d'):<14}"
              f"{(f'{mat[isin]:%Y-%m-%d}' if pd.notna(mat.get(isin)) else 'n/d'):<12}"
              f"{bcpi.get(isin)}")
    print("\n  Su DES/Bloomberg controlla il campo BASE CPI (o INFLATION_LINKED_BASE_CPI).")
    print("  Se il bond e' SCADUTO prima di un ribasamento Eurostat, la base e' vecchia:")
    print("  e' il caso che basis.rebase_base_cpi gestisce -- verifica che sia stato applicato.")
    print("  Se il campo e' proprio vuoto su Bloomberg, va inserito a mano dal prospetto.")
else:
    print("  nessuno")

# ---------------------------------------------------------------- B) copertura reale
print(f"\n--- B) COPERTURA sulle date ATTESE (soglia {SOGLIA_ATTESA:.0%}) ---")
rows = []
for isin in pan.columns:
    got = int(pan[isin].notna().sum())
    lo = max(pan.index.min(), fs.get(isin, pd.NaT)) if pd.notna(fs.get(isin)) else pan.index.min()
    hi = min(pan.index.max(), mat.get(isin, pd.NaT)) if pd.notna(mat.get(isin)) else pan.index.max()
    exp = int(((pan.index >= lo) & (pan.index <= hi)).sum())
    n_px = int(px[isin].notna().sum()) if (px is not None and isin in px.columns) else -1
    rows.append((isin, got, exp, got / exp if exp else np.nan, n_px, lo, hi))

d = pd.DataFrame(rows, columns=["isin", "ottenute", "attese", "quota", "prezzi", "da", "a"]
                 ).set_index("isin").sort_values("quota")
sospetti = d[(d["quota"] < SOGLIA_ATTESA) | d["quota"].isna()]
print(f"sotto soglia: {len(sospetti)} su {len(d)}\n")
if len(sospetti):
    print(f"{'ISIN':<16}{'ott.':>7}{'attese':>8}{'quota':>8}{'prezzi':>8}  {'finestra attesa':<25}diagnosi")
    for isin, r in sospetti.iterrows():
        if r["prezzi"] == 0:
            dg = "NESSUN PREZZO in px_mid -> riscaricare"
        elif r["prezzi"] < 0:
            dg = "ISIN assente da px_mid -> non scaricato"
        elif r["attese"] == 0:
            dg = "vita fuori dalla finestra dei pannelli"
        elif r["ottenute"] == 0:
            dg = "prezzi presenti ma 0 basi -> guardare gli errori di 04"
        else:
            dg = "parziale -> curva ILS corta o errori sparsi"
        print(f"{isin:<16}{int(r['ottenute']):>7}{int(r['attese']):>8}"
              f"{(r['quota'] if pd.notna(r['quota']) else 0):>8.1%}"
              f"{int(r['prezzi']):>8}  {r['da']:%Y-%m-%d} -> {r['a']:%Y-%m-%d}  {dg}")

giovani = d[(d["quota"] >= SOGLIA_ATTESA) & (d["attese"] < 0.3 * len(pan.index))]
if len(giovani):
    print(f"\n  ({len(giovani)} bond con poca storia ma copertura piena: emissioni recenti "
          f"o gia' scadute, NON sono un problema)")
    print("   " + ", ".join(giovani.index[:10]))

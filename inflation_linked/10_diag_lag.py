"""10 - LAG IMPLICITO: misura la convenzione di indicizzazione invece di assumerla.

Offline. NON scrive nulla, NON propone valori: e' solo una misura.

PERCHE'. base_cpi e' per definizione l'indice di riferimento alla data base del titolo.
Quindi per ogni linker che HA un base_cpi vale, in teoria,

        reference_index(data_base, lag=L, interp) == base_cpi

con L la convenzione del mercato. Girando L e guardando quale valore soddisfa l'identita'
si MISURA la convenzione sui dati, invece di prenderla da config e sperare.

COME LEGGERE IL RISULTATO
  - Se quasi tutti i bond di un segmento concordano su uno stesso L, quello E' il lag e
    va messo in config. L'accordo fra bond diversi e' l'evidenza: nessun grado di liberta'
    e' stato speso per ottenerlo.
  - Se gli L sono sparpagliati, il problema non e' il lag: e' la DATA BASE (il campo di
    Bloomberg non e' quello giusto, oppure per i titoli riaperti in tranche Bloomberg
    riporta la data della tranche invece di quella originale). In quel caso il base va
    letto dal prospetto, e nessuna derivazione e' lecita.
  - Se un segmento concorda e un altro no, sono due convenzioni diverse: e' il caso
    atteso per UK (old-style lag 8 non interpolato, new-style lag 3 interpolato).

Questo script non risolve i base_cpi mancanti. Serve a sapere se la macchina che calcola
l'indice di riferimento -- quella che entra in OGNI cedola di OGNI linker -- e' tarata.
"""
import numpy as np
import pandas as pd
import bbg
from basis import mef_reference
from config import MARKETS

# ----------------------------------------------------------------- impostazioni
MERCATO   = "UK"
CAMPO     = "START_ACC_DT"          # data base da testare
LAG_RANGE = range(0, 13)            # mesi di lag da provare
TOL_REL   = 1e-4

# ----------------------------------------------------------------- esecuzione
m = MARKETS[MERCATO]
ref = bbg.load("ref_linker")
ref = ref[ref["mkt"] == MERCATO].copy()
cpi = bbg.load(f"cpi_{m.cpi}").iloc[:, 0]
cpi.index = pd.to_datetime(cpi.index)

noti = ref[ref["base_cpi_final"].notna() & (ref["base_cpi_final"] > 0)]
seg = ref["segment"] if "segment" in ref.columns else pd.Series("-", index=ref.index)
print(f"=== {MERCATO}: lag implicito su {len(noti)} linker con base nota ===")
print(f"    campo data base: {CAMPO}   |   config dice index_lag={m.index_lag}\n")

rows = []
for isin in noti.index:
    d = ref.at[isin, CAMPO] if CAMPO in ref.columns else None
    if pd.isna(d):
        rows.append((isin, seg.get(isin, "-"), np.nan, np.nan, np.nan, "data base mancante"))
        continue
    base = float(noti.at[isin, "base_cpi_final"])
    best_L, best_e, best_i = np.nan, np.inf, None
    for L in LAG_RANGE:
        for interp in (True, False):
            try:
                v = mef_reference(cpi, pd.Timestamp(d).date(), lag=L, interpolate=interp)
            except Exception:
                continue
            e = abs(v / base - 1)
            if e < best_e:
                best_L, best_e, best_i = L, e, interp
    nota = "" if best_e < TOL_REL else f"nessun lag riproduce (min err {best_e:.1e})"
    rows.append((isin, seg.get(isin, "-"), best_L, best_i, best_e, nota))

d = pd.DataFrame(rows, columns=["isin", "segmento", "lag", "interp", "err", "nota"]
                 ).set_index("isin")
ok = d[d["err"] < TOL_REL]
print(f"riprodotti entro {TOL_REL:.0e}: {len(ok)}/{len(d)}\n")

for s, g in d.groupby("segmento", dropna=False):
    g_ok = g[g["err"] < TOL_REL]
    print(f"--- segmento '{s}': {len(g)} bond, {len(g_ok)} riprodotti ---")
    if len(g_ok):
        tab = g_ok.groupby(["lag", "interp"]).size().sort_values(ascending=False)
        for (L, it), n in tab.items():
            quota = n / len(g_ok)
            flag = "  <<< CONVENZIONE" if quota >= 0.80 else ""
            print(f"    lag={int(L)} interp={it}: {n} bond ({quota:.0%}){flag}")
        if tab.iloc[0] / len(g_ok) < 0.80:
            print("    -> NESSUN accordo: il problema e' la DATA BASE, non il lag.")
            print("       Leggi il base dal prospetto. Non derivarlo.")
    if len(g) - len(g_ok):
        print(f"    {len(g)-len(g_ok)} non riprodotti da nessun lag 0-12:")
        for isin, r in g[g['err'] >= TOL_REL].head(6).iterrows():
            print(f"      {isin}  err minimo {r['err']:.2e} a lag {r['lag']}")

print("\n--- confronto con config ---")
for s, g in ok.groupby("segmento", dropna=False):
    if not len(g):
        continue
    L = int(g["lag"].mode().iloc[0]); it = g["interp"].mode().iloc[0]
    atteso = 8 if (MERCATO == "UK" and str(s) == "old") else m.index_lag
    verdetto = "COINCIDE" if L == atteso else f"DIVERGE da config ({atteso})"
    print(f"  segmento '{s}': misurato lag={L} interp={it}   config={atteso}   -> {verdetto}")
print("\nSe diverge, NON cambiare config prima di aver controllato la convenzione")
print("sul prospetto: l'accordo fra bond e' evidenza forte, ma la fonte resta il prospetto.")

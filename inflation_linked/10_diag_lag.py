"""10 - LAG IMPLICITO: misura la convenzione di indicizzazione invece di assumerla.

Offline. NON scrive nulla, NON propone valori: e' solo una misura.

PERCHE'. base_cpi e' per definizione l'indice di riferimento alla data base del titolo.
Quindi per ogni linker che HA un base_cpi vale, in teoria,

        reference_index(data_base, lag=L, interp) == base_cpi

con L la convenzione del mercato. Girando L e guardando quale valore soddisfa l'identita'
si MISURA la convenzione sui dati, invece di prenderla da config e sperare.

PAREGGI. In un mese a inflazione piatta piu' lag danno lo STESSO indice, quindi per quel
bond l'identita' non discrimina. Prendere l'argmin sarebbe arbitrario e frammenterebbe la
moda, facendo scattare un falso "nessun accordo". Qui si raccoglie l'INSIEME dei lag
ammissibili per ogni bond, e si riportano due cose diverse:
  - quanti bond sono COMPATIBILI con la convenzione che il codice applica gia';
  - la distribuzione sui soli bond NON AMBIGUI (un solo lag ammissibile), che sono
    l'evidenza vera.

COME LEGGERE IL RISULTATO
  - CONFERMA = la convenzione del codice e' fra gli ammissibili di tutti i bond, e i bond
    non ambigui la indicano. La macchina e' tarata: non c'e' nulla da cambiare.
  - Bond riprodotti ma NON dalla convenzione del codice = vere eccezioni, da verificare
    sul prospetto uno per uno.
  - Bond non ambigui che non concordano fra loro = il sospetto non e' il lag ma la DATA
    BASE (campo Bloomberg sbagliato, o titoli riaperti in tranche per cui Bloomberg
    riporta la data della tranche invece di quella originale). Li' il base va letto dal
    prospetto e nessuna derivazione e' lecita.
  - FUORI SERIE CPI = la data base cade prima dell'inizio della serie scaricata. Non e'
    un problema di convenzione: allunga la storia CPI e il bond diventa testabile.

Questo script non risolve i base_cpi mancanti. Serve a sapere se la macchina che calcola
l'indice di riferimento -- quella che entra in OGNI cedola di OGNI linker -- e' tarata.
Solo se lo e', la derivazione di 09_recover_base_cpi.py e' lecita.
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
print(f"    campo data base: {CAMPO}")
print(f"    serie {m.cpi}: {cpi.index.min():%Y-%m} -> {cpi.index.max():%Y-%m} ({len(cpi)} punti)")
print(f"    il CODICE usa: old-style lag 8 NON interp, tutto il resto lag 3 interp")
print(f"    (basis.LinkerBond.from_ref ~riga 248. config.index_lag NON lo legge nessuno:")
print(f"     e' un campo documentativo, non confrontarcisi.)\n")
CPI_LO = cpi.index.min()

def _usato(sg):
    """La convenzione che il CODICE applica a quel segmento (basis.LinkerBond.from_ref)."""
    return (8, False) if (MERCATO == "UK" and str(sg) == "old") else (3, True)

# NB: non si prende l'argmin. In un mese a inflazione piatta piu' lag danno lo STESSO
# valore, quindi l'argmin ne sceglie uno arbitrario e frammenta la moda, facendo scattare
# un falso "nessun accordo". Si raccoglie invece l'INSIEME dei lag ammissibili per ogni
# bond: l'evidenza sta nei bond che ne ammettono uno solo, e la domanda vera e' se la
# convenzione del codice e' fra gli ammissibili.
rows = []
for isin in noti.index:
    dd = ref.at[isin, CAMPO] if CAMPO in ref.columns else None
    sg = seg.get(isin, "-")
    if pd.isna(dd):
        rows.append((isin, sg, (), np.nan, np.nan, False, "data base mancante"))
        continue
    base = float(noti.at[isin, "base_cpi_final"])
    amm, best_e, best_L = [], np.inf, np.nan
    for L in LAG_RANGE:
        for interp in (True, False):
            try:
                v = mef_reference(cpi, pd.Timestamp(dd).date(), lag=L, interpolate=interp)
            except Exception:
                continue
            e = abs(v / base - 1)
            if e < best_e:
                best_e, best_L = e, L
            if e < TOL_REL:
                amm.append((L, interp))
    need = pd.Timestamp(dd) - pd.DateOffset(months=max(LAG_RANGE))
    if not np.isfinite(best_e):
        nota = f"FUORI SERIE CPI: serve ~{need:%Y-%m}, la serie parte {CPI_LO:%Y-%m}"
    elif not amm:
        nota = f"nessun lag 0-12 riproduce (err minimo {best_e:.1e} a lag {best_L})"
    else:
        nota = ""
    rows.append((isin, sg, tuple(amm), best_e, len(amm), _usato(sg) in amm, nota))

d = pd.DataFrame(rows, columns=["isin", "segmento", "ammissibili", "err",
                                "n_amm", "compat_codice", "nota"]).set_index("isin")
d["lag"] = [a[0][0] if len(a) == 1 else np.nan for a in d["ammissibili"]]
d["interp"] = [a[0][1] if len(a) == 1 else np.nan for a in d["ammissibili"]]
ok = d[d["n_amm"] > 0]
print(f"riprodotti da almeno un lag: {len(ok)}/{len(d)}\n")

for sg, g in d.groupby("segmento", dropna=False):
    g_ok = g[g["n_amm"] > 0]
    L_u, it_u = _usato(sg)
    print(f"--- segmento '{sg}': {len(g)} bond, {len(g_ok)} riprodotti ---")
    if len(g_ok):
        n_comp = int(g_ok["compat_codice"].sum())
        print(f"    compatibili con la convenzione del CODICE (lag={L_u} interp={it_u}): "
              f"{n_comp}/{len(g_ok)} ({n_comp/len(g_ok):.0%})")
        # l'evidenza vera sta nei bond che ammettono UNA SOLA combinazione
        uni = g_ok[g_ok["n_amm"] == 1]
        if len(uni):
            tab = uni.groupby(["lag", "interp"]).size().sort_values(ascending=False)
            print(f"    evidenza non ambigua ({len(uni)} bond con un solo lag ammissibile):")
            for (L, it), n in tab.items():
                flag = "  <<< = quella del codice" if (int(L), bool(it)) == (L_u, it_u) else ""
                print(f"      lag={int(L)} interp={it}: {n} bond ({n/len(uni):.0%}){flag}")
            if tab.iloc[0] / len(uni) < 0.80:
                print("      -> NESSUN accordo fra bond non ambigui: il sospetto non e' il")
                print("         lag ma la DATA BASE (campo sbagliato, o tranche riaperte).")
                print("         Leggi il base dal prospetto. Non derivarlo.")
        else:
            print("    nessun bond con lag unico: troppi pareggi per concludere.")
            print("    (inflazione piatta nel periodo: l'evidenza e' debole per costruzione)")
        contrari = g_ok[~g_ok["compat_codice"]]
        if len(contrari):
            print(f"    {len(contrari)} bond riprodotti ma NON dalla convenzione del codice:")
            for isin, r in contrari.head(8).iterrows():
                print(f"      {isin}  ammette {list(r['ammissibili'])[:4]}")
            print("      -> vere eccezioni: verifica questi sul prospetto.")
    n_ko = len(g) - len(g_ok)
    if n_ko:
        print(f"    {n_ko} non riprodotti da nessun lag 0-12:")
        for isin, r in g[g["n_amm"] == 0].head(8).iterrows():
            print(f"      {isin}  {r['nota']}")
        n_fuori = int(g["nota"].astype(str).str.startswith("FUORI SERIE").sum())
        if n_fuori:
            print(f"    -> {n_fuori} di questi NON sono un problema di convenzione: e' la")
            print(f"       STORIA CPI troppo corta. Abbassa il floor in bbg.fetch_cpi,")
            print(f"       cancella cache/cpi_{m.cpi}.parquet e riscarica il CPI:")
            print(f"       diventano testabili e l'evidenza si allarga.")

print("\n--- verdetto ---")
for sg, g in ok.groupby("segmento", dropna=False):
    if not len(g):
        continue
    L_u, it_u = _usato(sg)
    n_comp = int(g["compat_codice"].sum())
    if n_comp == len(g):
        v = f"CONFERMA su {len(g)} bond"
    elif n_comp / len(g) >= 0.90:
        v = f"CONFERMA su {n_comp}/{len(g)} ({len(g)-n_comp} eccezioni da guardare)"
    else:
        v = f"DIVERGE: solo {n_comp}/{len(g)} compatibili -- da investigare"
    print(f"  segmento '{sg}': codice usa lag={L_u} interp={it_u}  ->  {v}")
print("\nCONFERMA = la macchina dell'indice di riferimento e' tarata su quel segmento.")
print("Non c'e' nulla da cambiare, e i base_cpi mancanti sono derivabili con 09.")
print("DIVERGE = prima di toccare basis.LinkerBond.from_ref, verifica sul prospetto.")

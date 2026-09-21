"""29 - IL BUCO DI 915 GIORNI E' VERO O MANCANO TITOLI? Offline, istantaneo.

LA DOMANDA. Il 26 ha misurato un bracket di 915 giorni fra i due nominali che bracciano
il BTPei 2035, e io l'ho riportato come un fatto senza mai guardare QUALI siano quei due
titoli. Su un emittente che ha decine di BTP in circolazione, due anni e mezzo di vuoto
attorno al 2035 vanno verificati, non accettati: se invece il pool e' incompleto -- titoli
esclusi dai filtri dell'universo, o presenti in anagrafica ma senza prezzo -- allora il
bracket largo e' un nostro difetto e non una proprieta' della curva italiana.

Sono due cose diverse e si distinguono guardando l'elenco:
  - se fra agosto 2034 e febbraio 2037 il Tesoro non aveva emesso nulla, 915 giorni sono
    la realta' e l'interpolazione su quel buco e' un problema vero da trattare;
  - se c'e' un BTP in mezzo che noi non usiamo, il problema e' il pool e si ripara.

E ATTENZIONE ALLA DATA. Un BTP 2035 emesso nel 2020 nel 2009 non esiste: il matching
filtra per data di emissione, e fa bene. Quindi il bracket si guarda ALLA DATA, non oggi.
Un elenco fatto con l'occhio di oggi direbbe che di titoli ce n'e' un sacco, e sarebbe una
risposta alla domanda sbagliata.

Non serve nessun run: legge l'anagrafica e i prezzi che sono gia' in cache.
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE

# ----------------------------------------------------------------- impostazioni
MERCATO  = "IT"
TARGET   = "IT0003745541"                       # il linker sotto esame
QUANDO   = ["2009-06-30", "2012-06-29", "2016-06-30", "2020-06-30", "2026-06-30"]
FINESTRA = 8                                    # anni attorno alla scadenza del target

# ----------------------------------------------------------------- dati
nom_mkt = bbg.NOMINAL_POOL_ALIAS.get(MERCATO, MERCATO)
ref_n = bbg.load("ref_nominal"); ref_n = ref_n[ref_n["mkt"] == nom_mkt]
ref_l = bbg.load("ref_linker")
pxn = bbg.load(f"px_nom_{nom_mkt}"); pxn.index = pd.to_datetime(pxn.index)
mat = pd.to_datetime(ref_n["MATURITY"], errors="coerce")
fs = pd.to_datetime(ref_n.get("FIRST_SETTLE_DT"), errors="coerce")
iss = pd.to_datetime(ref_n.get("ISSUE_DT"), errors="coerce")
nasce = fs.fillna(iss) if fs is not None else iss
m_t = pd.to_datetime(ref_l.at[TARGET, "MATURITY"])

print(f"=== {MERCATO}: il pool dei nominali attorno al {TARGET} ===")
print(f"    scadenza del linker: {m_t:%Y-%m-%d}")
print(f"    anagrafica: {len(ref_n)} nominali, {pxn.shape[1]} con serie prezzi")

# --- 1. l'elenco, ALLA DATA -------------------------------------------------------
print(f"\n--- 1. quali BTP esistevano davvero, attorno al {m_t:%Y}, a ciascuna data ---")
for q in QUANDO:
    d = pd.Timestamp(q)
    vivi = ref_n.index[(mat > d) & (nasce <= d)]
    vicini = [i for i in vivi
              if abs((mat[i] - m_t).days) < FINESTRA * 365.25]
    quot = set(pxn.columns[pxn.loc[d].notna()]) if d in pxn.index else set()
    vicini = sorted(vicini, key=lambda i: mat[i])
    print(f"\n    {d:%Y-%m-%d}  ({len(vicini)} BTP entro {FINESTRA} anni dalla scadenza "
          f"del linker)")
    print(f"      {'isin':<16}{'scadenza':<12}{'emesso':<12}{'prezzo?':<10}"
          f"{'dist dal linker':>16}")
    prec = None
    for i in vicini:
        dist = (mat[i] - m_t).days
        seg = "quotato" if i in quot else "NO PREZZO"
        buco = ""
        if prec is not None:
            g = (mat[i] - mat[prec]).days
            if g > 400:
                buco = f"   <- {g} gg dal precedente"
        print(f"      {i:<16}{mat[i]:%Y-%m-%d}  {nasce[i]:%Y-%m-%d}  {seg:<10}"
              f"{dist:>+16}{buco}")
        prec = i
    # il bracket effettivo: i due piu' vicini sotto e sopra, QUOTATI a quella data
    usabili = [i for i in vivi if i in quot]
    sotto = [i for i in usabili if mat[i] <= m_t]
    sopra = [i for i in usabili if mat[i] > m_t]
    if sotto and sopra:
        a = max(sotto, key=lambda i: mat[i]); b = min(sopra, key=lambda i: mat[i])
        print(f"      -> bracket effettivo: {a} ({mat[a]:%Y-%m-%d}) / "
              f"{b} ({mat[b]:%Y-%m-%d}) = {(mat[b]-mat[a]).days} giorni")
    else:
        print(f"      -> nessun bracket: manca un lato")

# --- 2. c'e' qualcosa in anagrafica che NON usiamo? -------------------------------
print(f"\n--- 2. titoli in anagrafica ma senza prezzo (li perderemmo nel matching) ---")
senza = [i for i in ref_n.index if i not in pxn.columns]
vicini_senza = [i for i in senza
                if pd.notna(mat.get(i)) and abs((mat[i] - m_t).days) < FINESTRA * 365.25]
print(f"    {len(senza)} nominali su {len(ref_n)} non hanno serie prezzi; "
      f"{len(vicini_senza)} stanno entro {FINESTRA} anni dal linker")
for i in vicini_senza:
    print(f"      {i}  scad {mat[i]:%Y-%m-%d}  emesso "
          f"{nasce[i]:%Y-%m-%d}" if pd.notna(nasce.get(i)) else f"      {i}  scad {mat[i]:%Y-%m-%d}")
if vicini_senza:
    print("    Se uno di questi cade DENTRO il buco, il bracket largo e' colpa nostra e")
    print("    si ripara scaricando il suo prezzo. Altrimenti il buco e' del Tesoro.")

# --- 3. e i titoli esclusi dall'universo? -----------------------------------------
print(f"\n--- 3. titoli scartati dai filtri dell'universo, vicini alla scadenza ---")
try:
    uni = bbg.load("universe")
    u = uni[(uni.get("mkt") == nom_mkt) if "mkt" in uni.columns else slice(None)]
    mu = pd.to_datetime(u["maturity"], errors="coerce")
    fuori = u[(u["incl"] == False) if "incl" in u.columns else
              (u["excl_reason"].astype(str) != "")]
    mf = pd.to_datetime(fuori["maturity"], errors="coerce")
    vic = fuori[(mf - m_t).abs().dt.days < FINESTRA * 365.25]
    print(f"    {len(vic)} titoli esclusi con scadenza entro {FINESTRA} anni dal linker")
    for i, r in vic.iterrows():
        rr = r.get("excl_reason", "?")
        print(f"      {i}  scad {pd.to_datetime(r['maturity']):%Y-%m-%d}   motivo: {rr}")
    if len(vic):
        print("    Un'esclusione LEGITTIMA (valuta pre-euro, funged, scaduto) va bene. Una")
        print("    che toglie un BTP a tasso fisso vivo in quel periodo e' da rivedere.")
except Exception as e:
    print(f"    universe.parquet non leggibile ({e}): salto il controllo.")

# --- 4. i buchi della scala, in generale ------------------------------------------
print(f"\n--- 4. i buchi piu' larghi della scala delle scadenze, a meta' campione ---")
for q in ("2010-06-30", "2020-06-30"):
    d = pd.Timestamp(q)
    if d not in pxn.index:
        continue
    quot = pxn.columns[pxn.loc[d].notna()]
    mm = mat.reindex(quot).dropna().sort_values()
    mm = mm[mm > d]
    g = mm.diff().dt.days.dropna()
    print(f"\n    {d:%Y-%m-%d}: {len(mm)} nominali quotati, da "
          f"{mm.iloc[0]:%Y-%m} a {mm.iloc[-1]:%Y-%m}")
    print(f"      buco mediano {g.median():.0f} gg, p95 {g.quantile(.95):.0f} gg, "
          f"massimo {g.max():.0f} gg")
    top = g.sort_values(ascending=False).head(5)
    for isin, gg in top.items():
        j = list(mm.index).index(isin)
        prima = mm.index[j - 1]
        print(f"      {gg:>5.0f} gg   fra {mm[prima]:%Y-%m-%d} e {mm[isin]:%Y-%m-%d}")
print("\n    Se i buchi grandi stanno TUTTI sul lungo, non e' un difetto del pool: e' la")
print("    forma del debito italiano, che oltre i vent'anni emette di rado. E allora il")
print("    bracket largo e' una proprieta' del problema, da trattare, non da riparare.")

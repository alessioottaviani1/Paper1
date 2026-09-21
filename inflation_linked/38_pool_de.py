"""38 - QUANTO E' DAVVERO FITTA LA GRIGLIA NOMINALE TEDESCA. Nessun terminale: legge il
file universo (Excel) e i parquet gia' in cache.

LA DOMANDA DI ALESSIO, ED E' GIUSTA. La Germania emette BKO (2 anni), OBL (5) e DBR (10 e
30) su un calendario regolare: non e' credibile che attorno a un Bund-ei i due nominali
piu' vicini distino dodici anni. Il 37 ha misurato un bracket mediano di 4564 giorni sulle
osservazioni perse, e quel numero non descrive il mercato tedesco.

DESCRIVE IL NOSTRO POOL. In bbg.py:

    CURVE_FROM_FILE = {"US","UK"} | ({"DE"} if DE_NOMINAL_CURVE == "bundesbank" else set())

e il ramo che ne dipende applica _matching_pool, cioe' tiene NOMINAL_POOL_PER_SIDE = 2
nominali per lato entro NOMINAL_POOL_MAX_DAYS = 550 giorni da una scadenza linker --
all'ANAGRAFICA, non solo agli storici. Con DE_NOMINAL_CURVE = 'bundesbank' la Germania e'
finita in quel ramo, e il pannello ha 33 titoli contro i 271 italiani. Il commento a
bbg.py:69 dice l'opposto di quel che succede: "Per IT/FR/DE la curva la fittiamo noi dai
bond, quindi serve lo spettro completo: nessun filtro". Il filtro e' arrivato come effetto
collaterale di una scelta su DOVE PRENDERE LA CURVA, che con il pool non c'entra nulla --
tanto piu' che la nostra base sconta sulla curva SWAP e la curva nominale tedesca non la
usa affatto.

COSA MISURA QUESTO SCRIPT. Il passo grosso (riscaricare ~290 titoli di anagrafica e
prezzi) si fa solo se serve, e per saperlo basta l'Excel dell'universo, che le scadenze le
ha gia' tutte. Per ogni osservazione del campione si calcola il bracket che avremmo con lo
SPETTRO COMPLETO e lo si mette accanto a quello che abbiamo.

ONESTA' SU COSA E' QUESTO NUMERO. L'universo non porta la data di emissione, quindi non si
puo' sapere con certezza quali titoli fossero gia' sul mercato a una certa data. Si scarta
cio' che e' impossibile -- un bond con piu' di TENOR_MAX anni residui non era ancora
emesso, visto che il tenor massimo tedesco e' 30 anni -- ma resta una condizione
NECESSARIA, non sufficiente. Il bracket che esce e' quindi un LIMITE INFERIORE: quello
vero sara' un po' piu' largo, soprattutto a inizio campione. Va letto per l'ordine di
grandezza, ed e' piu' che sufficiente: fra qualche mese e dodici anni non c'e' partita.
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE

MERCATO   = "DE"
TENOR_MAX = 31.0       # anni: oltre, il titolo non poteva essere gia' stato emesso
CAMPIONE  = 400        # date estratte a caso dal campione, per non fare 5205 giri

uni = bbg.build_universe(save=False)
nom = uni[uni["kind"].eq("nominal") & uni["mkt"].eq(MERCATO)].copy()
if "incl" in uni.columns:
    nom_incl = nom[nom["incl"]]
else:
    nom_incl = nom
nom_incl = nom_incl.dropna(subset=["maturity"])

print(f"=== {MERCATO}: lo spettro nominale che abbiamo contro quello che esiste ===")
print(f"\n--- 1. l'universo, per tipo di titolo ---")
print(f"    {'ticker':<10}{'nel file':>10}{'inclusi':>10}{'scaricati':>12}")
ref_n = bbg.load("ref_nominal")
ref_de = ref_n[ref_n["mkt"] == MERCATO]
scaric = set(ref_de.index)
for tk in sorted(set(nom["tick"].dropna())):
    a = int((nom["tick"] == tk).sum())
    b = int((nom_incl["tick"] == tk).sum())
    c = int(nom_incl.loc[nom_incl["tick"] == tk, "isin"].isin(scaric).sum())
    print(f"    {tk:<10}{a:>10}{b:>10}{c:>12}")
print(f"    {'TOTALE':<10}{len(nom):>10}{len(nom_incl):>10}{len(scaric):>12}")
if len(nom_incl) > len(scaric):
    print(f"    -> mancano {len(nom_incl)-len(scaric)} titoli: e' il pool di matching")
    print(f"       (2 per lato entro {bbg.NOMINAL_POOL_MAX_DAYS} gg da una scadenza linker).")

# --- 2. il bracket vero contro il nostro ---------------------------------------------
D = pd.read_parquet(CACHE / f"swapbasis_{MERCATO}.parquet")
D["date"] = pd.to_datetime(D["date"])
V = D[D["ttm"] > 365].copy()
tutte = pd.Series(sorted(V["date"].unique()))
sel = set(tutte.sample(min(CAMPIONE, len(tutte)), random_state=0))
S = V[V["date"].isin(sel)].copy()

mfull = np.sort(pd.to_datetime(nom_incl["maturity"]).values.astype("datetime64[D]").astype(int))
larghi, senza = [], 0
for d, g in S.groupby("date"):
    dnum = np.datetime64(pd.Timestamp(d).date(), "D").astype(int)
    viv = mfull[(mfull > dnum) & (mfull - dnum <= TENOR_MAX * 365.25)]
    for _, r in g.iterrows():
        ml = dnum + int(r["ttm"])
        pos = int(np.searchsorted(viv, ml))
        if pos == 0 or pos >= len(viv):
            senza += 1
            continue
        larghi.append((r["date"], r["isin"], int(viv[pos] - viv[pos - 1]),
                       float(r["bracket"]) if pd.notna(r["bracket"]) else np.nan,
                       int(len(viv))))
B = pd.DataFrame(larghi, columns=["date", "isin", "brk_pieno", "brk_nostro", "n_viv"])

print(f"\n--- 2. bracket su {len(S)} osservazioni ({len(sel)} date estratte) ---")
print(f"    {'':<22}{'mediana':>10}{'p90':>10}{'max':>10}")
print(f"    {'con lo spettro pieno':<22}{B['brk_pieno'].median():>10.0f}"
      f"{B['brk_pieno'].quantile(.9):>10.0f}{B['brk_pieno'].max():>10.0f}")
nn = B["brk_nostro"].dropna()
print(f"    {'col pool di oggi':<22}{nn.median():>10.0f}{nn.quantile(.9):>10.0f}"
      f"{nn.max():>10.0f}")
print(f"    nominali vivi per data: mediana {B['n_viv'].median():.0f} "
      f"(oggi ne abbiamo {len(scaric)} in tutto il campione)")
if senza:
    print(f"    {senza} osservazioni restano senza un lato anche con lo spettro pieno:")
    print("       sono i linker oltre la scadenza nominale piu' lunga, ed e' vero buco.")

sopra = (B["brk_pieno"] > 1095).mean()
print(f"\n    quota che resterebbe sopra MAX_BRACKET (1095 gg): {sopra:.1%}")
print(f"    quota persa oggi: 51%")

# --- 3. il dettaglio per linker ------------------------------------------------------
print(f"\n--- 3. per linker ---")
print(f"    {'isin':<16}{'n oss':>8}{'brk pieno':>12}{'brk nostro':>12}{'rapporto':>10}")
for isin, g in B.groupby("isin"):
    p, q = g["brk_pieno"].median(), g["brk_nostro"].median()
    print(f"    {isin:<16}{len(g):>8}{p:>12.0f}"
          + (f"{q:>12.0f}{q/p:>10.1f}x" if np.isfinite(q) and p > 0 else f"{'-':>12}{'-':>10}"))

print(f"\n--- come si legge ---")
print("    Se il bracket con lo spettro pieno e' di qualche mese mentre il nostro e' di")
print("    anni, il 4564 del 37 e' un artefatto del pool e non dice nulla sulla curva")
print("    tedesca: INTERP sulla Germania non e' impossibile, e' solo affamata di dati.")
print("    Allora la strada e' scaricare lo spettro completo, non cambiare la soglia ne'")
print("    ripiegare su CURVA. Se invece i due numeri si somigliano, il reticolo tedesco")
print("    e' davvero rado attorno ai linker e la conclusione di prima regge.")

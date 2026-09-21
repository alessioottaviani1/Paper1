"""35 - I MERCATI UNO ACCANTO ALL'ALTRO. Offline: legge i swapbasis_*.parquet che ci sono.

PERCHE' ADESSO. Tre mercati girano e i diagnostici li danno puliti. Il valore del capitolo
pero' non e' in nessuna delle tre serie: e' nel CONFRONTO. Stessa misura, stessa curva
swap, stesso indice HICP, stessa finestra -- l'unica cosa che cambia e' l'emittente. Ogni
differenza che resta e' credito e liquidita', non metodo.

E c'e' gia' un fatto da guardare in faccia. La Francia ha il massimo nel 2008-09 e resta
ai suoi livelli normali nel 2011-12; l'Italia fa l'opposto. L'emittente core reagisce allo
shock di liquidita' globale, quello periferico soprattutto allo shock sovrano. Questo
script lo mette in tabella invece che in un ricordo.

COSA FA, E COSA NON FA. Allinea le serie annuali, calcola le correlazioni sulle DATE IN
COMUNE (non sugli anni: una correlazione fra medie annuali su dodici punti non significa
niente), e confronta le finestre di stress. Non fa grafici e non sceglie una storia: mette
i numeri vicini e lascia che si guardino.

ATTENZIONE ALLE FINESTRE DIVERSE. La Spagna ha linker dal maggio 2014, gli altri dal 2004.
Confrontare le medie di campione fra chi ha visto Lehman e chi no non e' un confronto: e'
un artefatto della data di emissione. Quindi ogni confronto di livello si fa anche sulla
FINESTRA COMUNE, ed e' la riga che conta.
"""
import numpy as np
import pandas as pd
from config import CACHE, MARKETS

# ----------------------------------------------------------------- impostazioni
MERCATI  = ["IT", "ES", "FR", "FR_CPI", "DE"]
MISURA   = "interp"
MIN_VITA = 365
STRESS   = {"crisi 2008-09": (2008, 2009), "sovrana 2011-12": (2011, 2012),
            "covid 2020": (2020, 2020), "inflazione 2022-23": (2022, 2023)}

# ----------------------------------------------------------------- dati
S = {}
for m in MERCATI:
    p = CACHE / f"swapbasis_{m}.parquet"
    if not p.exists():
        continue
    d = pd.read_parquet(p)
    d["date"] = pd.to_datetime(d["date"])
    d = d[d[MISURA].notna() & (d["ttm"] > MIN_VITA)]
    if len(d):
        S[m] = d
if not S:
    raise SystemExit("nessun swapbasis_*.parquet in cache: lancia prima il 24.")

print("=== la base bond-level, mercato per mercato ===")
print(f"    misura {MISURA}, vita residua oltre {MIN_VITA} giorni\n")
print(f"    {'mercato':<9}{'titoli':>8}{'osservazioni':>14}{'da':>12}{'a':>12}"
      f"{'media':>9}{'mediana':>9}")
for m, d in S.items():
    # La data si formatta PRIMA e si allinea dopo: '{x:%Y-%m-%d:>12}' non e' una data
    # allineata, e' una data con ':>12' stampato dentro, e il conteggio a fianco ci
    # finisce appiccicato. Due specificatori nello stesso campo non si sommano.
    d0, d1 = f"{d['date'].min():%Y-%m-%d}", f"{d['date'].max():%Y-%m-%d}"
    print(f"    {m:<9}{d['isin'].nunique():>8}{len(d):>14}{d0:>13}{d1:>13}"
          f"{d[MISURA].mean():>9.1f}{d[MISURA].median():>9.1f}")

# --- 1. la tabella annuale, affiancata --------------------------------------------
print(f"\n--- 1. mediana annuale della base (bp) ---")
ann = pd.DataFrame({m: d.groupby(d["date"].dt.year)[MISURA].median() for m, d in S.items()})
n_t = pd.DataFrame({m: d.groupby(d["date"].dt.year)["isin"].nunique() for m, d in S.items()})
print(f"    {'anno':<7}" + "".join(f"{m:>12}" for m in ann.columns))
for y in ann.index:
    cel = "".join(f"{ann.at[y, m]:>8.1f} ({int(n_t.at[y, m])})"
                  if pd.notna(ann.at[y, m]) else f"{'-':>12}" for m in ann.columns)
    print(f"    {y:<7}{cel}")
print("    fra parentesi il numero di titoli vivi in quell'anno: una mediana su due")
print("    titoli non e' una mediana di mercato, e va letta sapendolo.")

# --- 2. le finestre di stress ------------------------------------------------------
print(f"\n--- 2. le finestre di stress: chi reagisce a che cosa ---")
print(f"    {'finestra':<20}" + "".join(f"{m:>10}" for m in ann.columns))
base = {}
for nome, (a, b) in STRESS.items():
    riga = {}
    for m in ann.columns:
        v = ann.loc[[y for y in ann.index if a <= y <= b], m].dropna()
        riga[m] = float(v.mean()) if len(v) else np.nan
    base[nome] = riga
    print(f"    {nome:<20}" + "".join(
        f"{riga[m]:>10.1f}" if pd.notna(riga[m]) else f"{'-':>10}" for m in ann.columns))
print(f"    {'mediana di ogni anno':<20}" + "".join(
    f"{ann[m].median():>10.1f}" if ann[m].notna().any() else f"{'-':>10}" for m in ann.columns))
print("\n    Il confronto che conta non e' il livello -- quello dipende dal credito -- ma")
print("    QUALE finestra fa il massimo di ciascun mercato:")
for m in ann.columns:
    v = {k: r[m] for k, r in base.items() if pd.notna(r[m])}
    if v:
        top = max(v, key=v.get)
        print(f"      {m:<8} massimo in '{top}' ({v[top]:.1f} bp), "
              f"contro una mediana di {ann[m].median():.1f}")

# --- 3. correlazioni sulle DATE in comune ------------------------------------------
print(f"\n--- 3. correlazioni fra mercati, sulle date in comune ---")
print("    Si correlano le mediane GIORNALIERE, non le medie annuali: dodici punti")
print("    annuali darebbero un numero che sembra una correlazione e non lo e'.\n")
gio = pd.DataFrame({m: d.groupby("date")[MISURA].median() for m, d in S.items()})
cols = list(gio.columns)
print(f"    {'':<9}" + "".join(f"{m:>10}" for m in cols) + f"{'  n date':>10}")
# gio[[x, y]] con x == y da' un DataFrame a due colonne omonime, e .corr() ci si rompe
# dentro: la diagonale si scrive, non si calcola. E le due serie si estraggono per
# POSIZIONE, non per nome, cosi' l'omonimia non puo' piu' mordere.
for x in cols:
    riga = ""
    for y in cols:
        if x == y:
            riga += f"{1.00:>10.2f}"
            continue
        cc = gio[[x, y]].dropna()
        riga += (f"{cc.iloc[:, 0].corr(cc.iloc[:, 1]):>10.2f}"
                 if len(cc) > 30 else f"{'-':>10}")
    print(f"    {x:<9}{riga}{int(gio[x].notna().sum()):>10}")
print("\n    Correlazioni alte fra mercati diversi = un fattore comune (liquidita' o")
print("    inflazione), basse = ciascuno segue il proprio credito. Nessuna delle due e'")
print("    un difetto: sono due regimi, e il capitolo dovrebbe dire quando vale quale.")

# --- 4. la finestra comune, che e' l'unico confronto di livello lecito -------------
print(f"\n--- 4. sulla finestra COMUNE a tutti i mercati presenti ---")
com = gio.dropna()
if len(com) > 30:
    print(f"    {com.index.min():%Y-%m-%d} -> {com.index.max():%Y-%m-%d}, {len(com)} date\n")
    print(f"    {'mercato':<9}{'media':>9}{'mediana':>9}{'sd':>9}{'p5':>9}{'p95':>9}")
    for m in cols:
        s = com[m]
        print(f"    {m:<9}{s.mean():>9.1f}{s.median():>9.1f}{s.std():>9.1f}"
              f"{s.quantile(.05):>9.1f}{s.quantile(.95):>9.1f}")
    print("\n    QUESTA e' la riga da mettere nel paper per il confronto di livello. Le")
    print("    medie di campione intero non sono confrontabili: chi comincia nel 2014 non")
    print("    ha visto Lehman, e la sua media e' bassa per la data di emissione, non per")
    print("    il credito dell'emittente.")
else:
    print(f"    finestra comune troppo corta ({len(com)} date): i mercati presenti non si")
    print(f"    sovrappongono abbastanza. Confrontare i livelli qui sarebbe un artefatto.")

# --- 5. cosa manca --------------------------------------------------------------
print(f"\n--- 5. cosa manca al quadro ---")
assenti = [m for m in MERCATI if m not in S]
if assenti:
    print(f"    mercati non ancora calcolati: {', '.join(assenti)}")
for m in S:
    if not (CACHE / f"bbgbasis_interp_{m}.parquet").exists():
        print(f"    {m}: manca il confronto Bloomberg (serve il 15 e poi il 17, col terminale)")

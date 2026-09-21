"""18 - DOVE SI ROMPE LA BASE BLOOMBERG. Offline: legge i file gia' prodotti da 15 e 17.

IL FATTO. La base da Z-spread Bloomberg e' pulita dal 2018 (+25/+30 bp, sd 8-11) e assurda
fra il 2012 e il 2017 (-130/-200 bp di media, code a -1200). Una base sovrana di -1200 bp
non esiste. E la correlazione con la nostra misura sui flussi e' -0.07: non sono due
stime rumorose della stessa cosa, e' una delle due che in quella finestra misura altro.

LA DOMANDA E' UNA SOLA: quale GAMBA. La base e' z(linker) - z(nominale). Se il nominale e'
sano e il linker vola, il problema e' nel campo sul linker. Se volano entrambi, e' il
pool dei nominali. Finche' non si sa questo, ogni ipotesi e' aria.

COME SI RISPONDE. Le due gambe si guardano SEPARATE, in livello. Il nominale ha un
riferimento indipendente: lo Z-spread di un BTP e' stato fra 0 e ~500 bp nella crisi, mai
negativo di centinaia. La diagnosi NON si fa sulla mediana -- se i titoli rotti sono una
minoranza la mediana non si muove di un bp e il guasto resta invisibile -- ma sulla QUOTA
di osservazioni che escono dal corridoio, gamba per gamba.

POI CHI. Se e' il linker, non tutti lo saranno: si ordina per quanto ciascuno contribuisce
alle code e si guarda l'ANAGRAFICA del gruppo cattivo contro quella del gruppo sano --
ticker, calc type, descrizione, date di emissione. Se i cattivi condividono un attributo,
quello e' il colpevole, e non l'abbiamo dedotto: l'abbiamo letto.

L'IPOTESI DA FALSIFICARE. I BTP Italia sono indicizzati al FOI italiano, non all'HICP, e
rimborsano l'accrual di inflazione a ogni cedola invece che a scadenza: struttura diversa,
prezzo diverso, e uno Z-spread calcolato con la convenzione dei BTPei sarebbe senza senso.
Il primo e' stato emesso a MARZO 2012 -- lo stesso mese in cui la base si rompe. Questo
script non assume che sia cosi': stampa le anagrafiche e lascia che siano i dati a dirlo.
Se i cattivi fossero invece sparsi su tutte le anagrafiche, l'ipotesi cade e si guarda
altrove (le date, non i titoli).
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE

# ----------------------------------------------------------------- impostazioni
MERCATO = "IT"
CAMPO   = "Z_SPRD_MID"
ASSURDA = 100.0     # |base| oltre cui l'osservazione non e' economia, e' un difetto
MIN_VITA = 365

# ----------------------------------------------------------------- dati
pm = CACHE / f"bbgmatch_{MERCATO}.parquet"
pz = CACHE / f"zsprd_{MERCATO}_{CAMPO}.parquet"
for p in (pm, pz):
    if not p.exists():
        raise SystemExit(f"manca {p.name}: lancia prima 15 e 17.")

D = pd.read_parquet(pm)
D["date"] = pd.to_datetime(D["date"])
Z = pd.read_parquet(pz); Z.index = pd.to_datetime(Z.index)
ref_l = bbg.load("ref_linker"); ref_l = ref_l[ref_l["mkt"] == MERCATO]

# le due gambe, separate
# .dropna() esplicito: in pandas recente stack() NON scarta i NaN, e la join
# che segue finisce per appaiare celle vuote, azzerando intere annate.
zl = Z.stack().dropna().rename("z_lnk"); zl.index.names = ["date", "isin"]
D = D.join(zl, on=["date", "isin"])
gem = D[["date", "gemello"]].dropna()
zn = Z.stack().dropna().rename("z_nom"); zn.index.names = ["date", "gemello"]
D = D.join(zn, on=["date", "gemello"])
D = D[D["ttm"] > MIN_VITA]

print(f"=== {MERCATO}: dove si rompe la base ({len(D)} osservazioni, ttm > {MIN_VITA}gg) ===\n")

# --- 1. le due gambe in livello, anno per anno -------------------------------------
print("--- 1. le due gambe SEPARATE, in livello (mediana per anno, bp) ---")
print(f"    {'anno':<8}{'n':>7}{'z(linker)':>12}{'z(nominale)':>13}{'base':>10}"
      f"{'  min z(lnk)':>13}{'  min z(nom)':>13}")
for y, g in D.groupby(D["date"].dt.year):
    print(f"    {y:<8}{len(g):>7}{g['z_lnk'].median():>12.1f}{g['z_nom'].median():>13.1f}"
          f"{g['interp'].median():>10.1f}{g['z_lnk'].min():>13.1f}{g['z_nom'].min():>13.1f}")
print("\n    Lo Z-spread di un BTP e' stato fra 0 e ~500 bp, mai negativo di centinaia.")
print("    La colonna che esce da quel corridoio e' la gamba rotta.")

# --- 2. chi contribuisce alle code -------------------------------------------------
D["male"] = D["interp"].abs() > ASSURDA
print(f"\n--- 2. chi produce le osservazioni con |base| > {ASSURDA:.0f} bp ---")
print(f"    {int(D['male'].sum())} osservazioni su {len(D)} ({D['male'].mean():.1%})")
per = D.groupby("isin").agg(n=("interp", "size"), n_male=("male", "sum"),
                            med=("interp", "median"), med_zl=("z_lnk", "median"),
                            med_zn=("z_nom", "median"),
                            da=("date", "min"), a=("date", "max"))
per["quota"] = per["n_male"] / per["n"]
per = per.sort_values("quota", ascending=False)
col = [c for c in ("tick", "calc", "descr", "maturity") if c in ref_l.columns]
print(f"\n    {'isin':<14}{'n':>6}{'%assurde':>10}{'base med':>10}{'z(lnk)':>9}{'z(nom)':>9}"
      f"  {'periodo':<18} anagrafica")
for isin, r in per.head(14).iterrows():
    an = "  ".join(f"{ref_l.at[isin, c]}" for c in col) if isin in ref_l.index else "-"
    print(f"    {isin:<14}{int(r['n']):>6}{r['quota']:>9.0%}{r['med']:>10.1f}"
          f"{r['med_zl']:>9.1f}{r['med_zn']:>9.1f}  "
          f"{r['da']:%Y-%m}/{r['a']:%Y-%m}  {an[:60]}")

# --- 3. anagrafica: cattivi contro sani --------------------------------------------
cattivi = per[per["quota"] > 0.20].index
sani    = per[per["quota"] < 0.02].index
print(f"\n--- 3. anagrafica: {len(cattivi)} cattivi (>20% assurde) vs {len(sani)} sani (<2%) ---")
if len(cattivi) and len(sani) and col:
    for c in col:
        if c == "maturity":
            continue
        vc = ref_l.loc[[i for i in cattivi if i in ref_l.index], c].astype(str).value_counts()
        vs = ref_l.loc[[i for i in sani if i in ref_l.index], c].astype(str).value_counts()
        chiavi = sorted(set(vc.index) | set(vs.index))
        print(f"\n    {c}:")
        for k in chiavi:
            print(f"      {k[:52]:<54} cattivi {int(vc.get(k,0)):>3}   sani {int(vs.get(k,0)):>3}")
    print("\n    Un attributo presente SOLO fra i cattivi e' il colpevole. Se invece i due")
    print("    gruppi hanno le stesse anagrafiche, non e' una questione di titoli: guarda")
    print("    il punto 4, sono le DATE.")
else:
    print("    gruppi troppo piccoli o anagrafica assente: guarda il punto 4.")

# --- 4. e' una questione di date? --------------------------------------------------
print(f"\n--- 4. le assurde nel tempo ---")
mm = D.groupby(D["date"].dt.to_period("Q"))["male"].agg(["size", "sum"])
mm["quota"] = mm["sum"] / mm["size"]
attivi = mm[mm["quota"] > 0.05]
if len(attivi):
    print(f"    trimestri con oltre il 5% di assurde: da {attivi.index.min()} a {attivi.index.max()}")
    print(f"    {'trim':<10}{'n':>7}{'%assurde':>10}")
    att = mm[mm["quota"] > 0]
    for q, r in att.head(10).iterrows():
        print(f"    {str(q):<10}{int(r['size']):>7}{r['quota']:>9.0%}")
    if len(att) > 20:
        print(f"    ... ({len(att) - 20} trimestri intermedi omessi)")
    for q, r in att.tail(min(10, max(0, len(att) - 10))).iterrows():
        print(f"    {str(q):<10}{int(r['size']):>7}{r['quota']:>9.0%}")
else:
    print("    nessun trimestre concentrato.")

# --- 5. verdetto -------------------------------------------------------------------
# La diagnosi di gamba NON si fa sulla mediana: se i titoli rotti sono una minoranza la
# mediana non si muove di un bp e il guasto resta invisibile. Si conta la QUOTA di
# osservazioni fuori da un corridoio che nessuno Z-spread sovrano ha mai lasciato.
LO, HI = -50.0, 800.0
f_l = ((D["z_lnk"] < LO) | (D["z_lnk"] > HI))
f_n = ((D["z_nom"] < LO) | (D["z_nom"] > HI))
print(f"\n--- verdetto ---")
print(f"    osservazioni fuori dal corridoio [{LO:.0f}, {HI:.0f}] bp:")
print(f"      z(linker)   {int(f_l.sum()):>7} / {len(D)}  ({f_l.mean():.1%})")
print(f"      z(nominale) {int(f_n.sum()):>7} / {len(D)}  ({f_n.mean():.1%})")
fuori_l = f_l.mean() > 0.005
fuori_n = f_n.mean() > 0.005
if fuori_l and not fuori_n:
    print("\n    E' LA GAMBA LINKER. I nominali stanno nel corridoio, i linker no: il campo")
    print("    su quei titoli non sta misurando uno Z-spread confrontabile. Guarda il")
    print("    punto 3: se i cattivi condividono un'anagrafica, vanno tolti dall'universo")
    print("    -- non dalla misura, dall'UNIVERSO, perche' se sono titoli di un'altra")
    print("    specie non appartengono al campione a prescindere da Bloomberg.")
elif fuori_n and not fuori_l:
    print("\n    E' IL POOL DEI NOMINALI. Il linker regge, il nominale no: c'e' dentro")
    print("    qualcosa che non e' un BTP a tasso fisso, oppure il gemello scelto e' un")
    print("    titolo che in quel periodo non era quotato davvero.")
elif fuori_l and fuori_n:
    print("\n    ENTRAMBE fuori corridoio: non e' un problema di selezione dei titoli ma")
    print("    del campo stesso in quel periodo. Guarda il punto 4 e confronta con la")
    print("    tabella per anno del 16: se le due finestre coincidono, e' Bloomberg.")
else:
    print("\n    Nessuna delle due gambe esce dal corridoio: il guasto non e' nei livelli.")
    print("    Allora non e' il campo ma il MATCHING -- punto 2, i titoli con quota alta di")
    print("    assurde -- oppure poche date con prezzi stantii su uno dei due lati.")

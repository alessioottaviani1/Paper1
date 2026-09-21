"""21 - LISTA DA VERIFICARE A TERMINALE. Offline: legge, stampa, salva un CSV. Nient'altro.

A COSA SERVE. Tre ipotesi sull'origine delle basi assurde sono state costruite e respinte
dai dati (BTP Italia, floor di rimborso, capitale non rivalutato). L'ultima e' caduta
guardando le serie giorno per giorno: le rotture non sono un regime, sono stampe isolate e
picchi in date di stress. A questo punto continuare a inventare ipotesi sugli aggregati e'
il modo piu' veloce per convincersi di una quarta cosa sbagliata. Si guardano i numeri
sullo schermo, uno per uno.

LE STRATE, E PERCHE' SERVONO TUTTE E QUATTRO.

  ISOLATA   un giorno rotto fra due giorni sani, sullo stesso titolo. Se a schermo il
            valore coincide con quello che abbiamo, e' una stampa sporca di Bloomberg;
            se a schermo il valore e' sano, l'errore e' nello scarico o nel pannello.

  PICCO+    base sopra +100 bp, concentrate nell'agosto 2011. Potrebbero essere economia
            vera: in quei giorni il mercato italiano si e' mosso di decine di bp al
            giorno. Vanno distinte dalle rotture, altrimenti si butta via segnale.

  REGIME-   base sotto -100 bp nel 2013-2016, persistenti. Sono quelle che hanno prodotto
            le medie annuali a -100/-200 bp. Se a schermo coincidono, il campo e' cosi'
            e va dichiarato inutilizzabile li'.

  CONTROLLO osservazioni sane, sugli stessi titoli, in date vicine. SONO LE PIU'
            IMPORTANTI. Se le sane coincidono con lo schermo e le rotte no, il difetto e'
            nostro -- scarico o abbinamento. Se coincidono TUTTE, il nostro dato e'
            fedele e il problema e' di Bloomberg. Senza il controllo non si puo'
            distinguere il caso in cui sbagliamo noi, ed e' l'unico che possiamo aggiustare.

COSA GUARDARE A SCHERMO. Per ogni riga ci sono ticker, data e i due Z-spread che abbiamo
in cache. Su YAS del titolo, alla data, il campo e' INDEX_Z_SPREAD_BP con pricing source
CBBT. In alternativa, in Excel:

    =BDH("<ticker>","INDEX_Z_SPREAD_BP","<data>","<data>")

Il prezzo del linker e' in tabella per riconoscere la data: se il prezzo a schermo non e'
quello, la data o la source non corrispondono e il confronto non vale.
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE, MARKETS

# ----------------------------------------------------------------- impostazioni
MERCATO = "IT"
CAMPO   = "Z_SPRD_MID"
SOURCE  = "BGN"
ASSURDA = 100.0
MIN_VITA = 365
PER_STRATO = 5

# ----------------------------------------------------------------- dati
D = pd.read_parquet(CACHE / f"bbgmatch_{MERCATO}.parquet")
D["date"] = pd.to_datetime(D["date"])
Z = pd.read_parquet(CACHE / f"zsprd_{MERCATO}_{CAMPO}.parquet")
Z.index = pd.to_datetime(Z.index)
px = bbg.load(f"px_mid_{MERCATO}"); px.index = pd.to_datetime(px.index)
ref_l = bbg.load("ref_linker"); ref_n = bbg.load("ref_nominal")
bb = pd.concat([ref_l["bb_id"], ref_n["bb_id"]]).astype(str).str.strip()
bb = bb[~bb.index.duplicated()]
mat = pd.concat([pd.to_datetime(ref_l["maturity"], errors="coerce"),
                 pd.to_datetime(ref_n["maturity"], errors="coerce")])
mat = mat[~mat.index.duplicated()]

# .dropna() esplicito: in pandas recente stack() NON scarta i NaN, e la join
# che segue finisce per appaiare celle vuote, azzerando intere annate.
zl = Z.stack().dropna().rename("z_lnk"); zl.index.names = ["date", "isin"]
D = D.join(zl, on=["date", "isin"])
zn = Z.stack().dropna().rename("z_nom"); zn.index.names = ["date", "gemello"]
D = D.join(zn, on=["date", "gemello"])
pxs = px.stack().dropna().rename("px_lnk"); pxs.index.names = ["date", "isin"]
D = D.join(pxs, on=["date", "isin"])
D = D[(D["ttm"] > MIN_VITA) & D["z_lnk"].notna() & D["z_nom"].notna()].copy()
D["male"] = D["interp"].abs() > ASSURDA
D = D.sort_values(["isin", "date"]).reset_index(drop=True)

# isolate: rotta con il giorno prima E il giorno dopo sani, sullo stesso titolo
g = D.groupby("isin")["male"]
D["prec"] = g.shift(1).fillna(False).astype(bool)
D["succ"] = g.shift(-1).fillna(False).astype(bool)
D["isolata"] = D["male"] & ~D["prec"] & ~D["succ"]

# index ratio: costa nulla e probabilmente e' la chiave. Stessa convenzione della pipeline.
try:
    from basis import mef_reference
    cpi = bbg.load(f"cpi_{MARKETS[MERCATO].cpi}").iloc[:, 0]
    cpi.index = pd.to_datetime(cpi.index)
    bcp = pd.to_numeric(ref_l["base_cpi_final"], errors="coerce")
    rif = {}
    for d in D["date"].drop_duplicates():
        try:
            rif[d] = mef_reference(cpi, d.date(), lag=3, interpolate=True)
        except Exception:
            rif[d] = np.nan
    D["IR"] = D["date"].map(rif) / D["isin"].map(bcp)
except Exception as e:
    print(f"(index ratio non calcolabile: {e})")
    D["IR"] = np.nan
D["anni"] = D["ttm"] / 365.25

# --- UNA strata per riga, in ordine di priorita': senza questo la stessa osservazione
# compariva in piu' strate e la lista si riduceva a poche righe distinte.
D["strato"] = ""
D.loc[D["male"] & (D["interp"] < 0), "strato"] = "REGIME-"
D.loc[D["male"] & (D["interp"] > 0), "strato"] = "PICCO+"
D.loc[D["isolata"], "strato"] = "ISOLATA"
D.loc[~D["male"], "strato"] = "CONTROLLO"

ORD = ["ISOLATA", "REGIME-", "PICCO+", "CONTROLLO"]
righe = []
for nome in ORD:
    sub = D[D["strato"] == nome]
    if not len(sub):
        continue
    if nome == "CONTROLLO":
        # sane sugli stessi titoli rotti, spalmate nel tempo: se fossero tutte recenti
        # non direbbero nulla sugli anni in cui la misura sbanda
        sub = sub[sub["isin"].isin(D.loc[D["male"], "isin"].unique())]
        pick = (sub.assign(_a=sub["date"].dt.year).sort_values("_a")
                   .groupby("_a", group_keys=False).head(1))
    else:
        pick = sub.sort_values("interp", key=lambda x: -x.abs())
    # Si preferiscono titoli E anni diversi -- cinque righe sullo stesso bond a tre giorni
    # di distanza sono un controllo solo. Tre passate su chiavi sempre piu' deboli: prima
    # (titolo, anno) entrambi nuovi, poi solo il titolo nuovo, poi qualunque cosa. Cosi' il
    # numero di righe richiesto esce comunque, invece di svuotarsi quando i titoli finiscono.
    scelte, chiavi_i, chiavi_ia = [], set(), set()
    for passata in (1, 2, 3):
        for _, r in pick.iterrows():
            if len(scelte) >= PER_STRATO:
                break
            k_i, k_ia = r["isin"], (r["isin"], r["date"].year)
            if any(x["isin"] == r["isin"] and x["date"] == r["date"] for x in scelte):
                continue
            if passata == 1 and (k_i in chiavi_i or k_ia in chiavi_ia):
                continue
            if passata == 2 and k_ia in chiavi_ia:
                continue
            chiavi_i.add(k_i); chiavi_ia.add(k_ia); scelte.append(r)
        if len(scelte) >= PER_STRATO:
            break
    for r in scelte:
        righe.append({
            "strato": nome,
            "data": r["date"].date(),
            "linker_isin": r["isin"],
            "linker_ticker": f"{bb.get(r['isin'], '?')}@{SOURCE} Corp",
            "linker_scad": mat.get(r["isin"]).date() if pd.notna(mat.get(r["isin"])) else None,
            "anni": round(float(r["anni"]), 2),
            "index_ratio": round(float(r["IR"]), 4) if pd.notna(r["IR"]) else None,
            "z_linker": round(float(r["z_lnk"]), 2),
            "px_linker": round(float(r["px_lnk"]), 4) if pd.notna(r["px_lnk"]) else None,
            "nom_isin": r["gemello"],
            "nom_ticker": f"{bb.get(r['gemello'], '?')}@{SOURCE} Corp",
            "nom_scad": mat.get(r["gemello"]).date() if pd.notna(mat.get(r["gemello"])) else None,
            "z_nominale": round(float(r["z_nom"]), 2),
            "base_pair": round(float(r["pair"]), 2) if pd.notna(r["pair"]) else None,
            "base_interp": round(float(r["interp"]), 2) if pd.notna(r["interp"]) else None,
            "mismatch_gg": int(r["mismatch"]) if pd.notna(r["mismatch"]) else None,
        })

T = pd.DataFrame(righe).drop_duplicates(subset=["linker_isin", "data"])
if not len(T):
    raise SystemExit("nessuna riga da verificare.")

print(f"=== {MERCATO}: {len(T)} righe da verificare a terminale ===")
print(f"    campo {CAMPO}, pricing source {SOURCE}")
print(f"    px = prezzo del linker in cache: serve a riconoscere la data a schermo\n")
print(f"  {'#':>2} {'strato':<10}{'data':<12}{'linker (ticker)':<22}{'scad':<12}"
      f"{'anni':>6}{'IR':>7}{'z_lnk':>10}{'z_nom':>9}{'base':>10}{'px':>10}")
for i, (_, r) in enumerate(T.iterrows(), 1):
    ir = f"{r['index_ratio']:.3f}" if r["index_ratio"] is not None else "-"
    print(f"  {i:>2} {r['strato']:<10}{str(r['data']):<12}{r['linker_ticker']:<22}"
          f"{str(r['linker_scad']):<12}{r['anni']:>6.2f}{ir:>7}{r['z_linker']:>10.2f}"
          f"{r['z_nominale']:>9.2f}{r['base_pair']:>10.2f}{r['px_linker']:>10.3f}")
print(f"\n  ISIN e gemelli (stessa numerazione):")
for i, (_, r) in enumerate(T.iterrows(), 1):
    chk = r["z_linker"] - r["z_nominale"]
    fl = "" if abs(chk - r["base_pair"]) < 0.02 else "   <<< ARITMETICA MIA SBAGLIATA"
    print(f"  {i:>2} linker {r['linker_isin']}   gemello {r['nom_isin']} "
          f"({r['nom_ticker']}, scad {r['nom_scad']}, mismatch {r['mismatch_gg']}gg){fl}")

out = CACHE / f"controllo_bbg_{MERCATO}.csv"
T.to_csv(out, index=False)
print(f"CSV: {out}")
print("\n--- come leggerlo dopo il controllo ---")
print("  CONTROLLO coincide e le rotte NO  -> il difetto e' NOSTRO: scarico o abbinamento.")
print("     E' l'unico caso che possiamo aggiustare, ed e' per questo che il controllo c'e'.")
print("  coincidono TUTTE                  -> il nostro dato e' fedele: quei valori sono")
print("     proprio cio' che Bloomberg pubblica. Allora il campo va dichiarato")
print("     inutilizzabile dove sbanda, con un criterio scritto prima di guardare i")
print("     risultati, e la misura primaria resta la nostra sui flussi.")
print("  CONTROLLO non coincide            -> il problema non e' nelle code ma ovunque, e")
print("     va rifatto lo scarico prima di qualunque altra cosa.")
print("  PICCO+ coincide ed e' agosto 2011 -> non e' un difetto ma il mercato di quei")
print("     giorni: la soglia a 100 bp sta tagliando segnale, e va alzata o resa")
print("     condizionale alla volatilita' invece che fissa.")

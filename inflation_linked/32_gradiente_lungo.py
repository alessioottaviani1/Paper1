"""32 - IL GRADIENTE STA SUL LUNGO, E NON E' UN TITOLO SOLO. Offline.

L'ERRORE CHE QUESTO SCRIPT RIPARA. Il 27 misurava la pendenza della base contro la
scadenza su TUTTO il campione 1-40 anni, un punto per titolo, e concludeva che togliendo
il 2035 la pendenza spariva. Falso, e per un motivo banale: nel 2010 e nel 2011 la fascia
1-3 anni sta a 4.6 e 5.8 bp contro 25.9 e 35.8 della 8-12. Il corto e' basso, e una retta
tirata da li' fino al lungo esce POSITIVA anche mentre il lungo crolla. Ho letto il
coefficiente invece della tabella che gli stava sopra.

La tabella diceva un'altra cosa. Dal 8-12a al 18a+ la base cala di 47.5, 59.0 e 40.4 bp
nel 2009, 2010 e 2011, contro 0.7, 10.8 e 12.6 negli anni di controllo. E il passaggio
intermedio -- 8-12a verso 12-18a -- cala di 17.6, 14.9 e 12.2 contro +5.6, -0.6 e -5.4.
La fascia 12-18a NON contiene il 2035, che sta a venticinque anni: se cala anche lei,
l'effetto non e' di un titolo, e' della scadenza.

QUINDI A' TORNA VIVA. C'e' un gradiente di scadenza sul lungo, negli anni 2009-2011, che
coinvolge piu' titoli. Il 2035 ne e' il punto estremo, non la causa.

COSA FA QUESTO SCRIPT. Misura la pendenza dove serve -- solo dal ginocchio in su -- un
punto per titolo, con e senza il sospetto, contro anni di controllo. E stampa quali
titoli compongono ogni fascia in ogni anno, cosi' che "e' la scadenza" si veda sui nomi
e non solo su un coefficiente. Piu' un confronto con Bloomberg dove Bloomberg funziona
(dal 2012): se anche li' il gradiente e' lo stesso, la forma non e' nostra.

PERCHE' IL TAGLIO A OTTO ANNI E' SCRITTO PRIMA. Non si sceglie il punto di taglio
guardando dove il risultato viene piu' bello: si prende il ginocchio della tabella del 27
(fino a 8-12a la base e' piatta, da li' scende) e si tiene fisso. Il punto 4 mostra la
sensibilita' a quella scelta, che e' il modo onesto di dichiararla.
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE

# ----------------------------------------------------------------- impostazioni
MERCATO   = "IT"
MISURA    = "interp"
MIN_VITA  = 365
SOSPETTO  = "IT0003745541"
GINOCCHIO = 8.0            # anni: da qui in su si misura la pendenza
CALDI     = [2009, 2010, 2011]
CONTROLLO = [2013, 2014, 2016, 2017, 2020]
MIN_TIT   = 3

# ----------------------------------------------------------------- dati
D = pd.read_parquet(CACHE / f"swapbasis_{MERCATO}.parquet")
D["date"] = pd.to_datetime(D["date"])
D = D[D[MISURA].notna() & (D["ttm"] > MIN_VITA)].copy()
D["anno"] = D["date"].dt.year
D["anni"] = D["ttm"] / 365.25
ref_l = bbg.load("ref_linker"); ref_l = ref_l[ref_l["mkt"] == MERCATO]
mat = pd.to_datetime(ref_l["MATURITY"], errors="coerce")

print(f"=== {MERCATO}: il gradiente sul lungo ===")
print(f"    pendenza misurata solo oltre i {GINOCCHIO:.0f} anni, un punto per titolo")


def pend(sub, y, da=GINOCCHIO):
    g = sub[(sub["anno"] == y) & (sub["anni"] >= da)]
    if not len(g):
        return np.nan, 0
    p = g.groupby("isin").agg(b=(MISURA, "median"), a=("anni", "median"))
    if len(p) < MIN_TIT or p["a"].std() < 1e-9:
        return np.nan, len(p)
    return float(np.polyfit(p["a"], p["b"], 1)[0]), len(p)


# --- 1. la pendenza dove conta, con e senza il sospetto ---------------------------
print(f"\n--- 1. pendenza oltre i {GINOCCHIO:.0f} anni (bp per anno di scadenza) ---")
print(f"    {'anno':<8}{'titoli':>8}{'con il 2035':>14}{'titoli':>8}{'senza':>10}"
      f"{'   resta negativa?':>20}")
S = D[D["isin"] != SOSPETTO]
ris = {}
for y in CALDI + CONTROLLO:
    a, na = pend(D, y)
    b, nb = pend(S, y)
    ris[y] = (a, b)
    sa = f"{a:>14.2f}" if pd.notna(a) else f"{'-':>14}"
    sb = f"{b:>10.2f}" if pd.notna(b) else f"{'-':>10}"
    ok = "si" if (pd.notna(b) and b < 0) else ("no" if pd.notna(b) else "n.d.")
    print(f"    {y:<8}{na:>8}{sa}{nb:>8}{sb}{ok:>20}")
cal = [ris[y][1] for y in CALDI if pd.notna(ris[y][1])]
ctr = [ris[y][1] for y in CONTROLLO if pd.notna(ris[y][1])]
if cal and ctr:
    mc, mk = float(np.median(cal)), float(np.median(ctr))
    print(f"\n    SENZA il sospetto: anni caldi {mc:+.2f} bp/anno, controllo {mk:+.2f}")
    if mc < -0.5 and mc < mk - 0.5:
        print("    Il gradiente sopravvive alla rimozione del 2035: non e' quel titolo, e'")
        print("    la SCADENZA. L'ipotesi A' -- qualcosa che degrada con l'orizzonte nella")
        print("    proiezione dei flussi del linker -- torna in gioco, e il 2035 e' solo il")
        print("    punto piu' esposto.")
    elif mc >= -0.5:
        print("    Senza il sospetto il gradiente non c'e': allora era davvero lui, e il")
        print("    calo del 12-18a visto nelle fasce era composizione, non scadenza.")

# --- 2. i nomi, non solo i coefficienti -------------------------------------------
print(f"\n--- 2. chi c'e' in ogni fascia, e a che livello ---")
bins = [GINOCCHIO, 12, 18, 40]
et = [f"{GINOCCHIO:.0f}-12a", "12-18a", "18a+"]
D["fascia"] = pd.cut(D["anni"], bins=bins, labels=et, right=False)
for y in CALDI + CONTROLLO[:2]:
    g = D[D["anno"] == y]
    if not len(g):
        continue
    print(f"\n    {y}")
    for f in et:
        s = g[g["fascia"] == f]
        if not len(s):
            print(f"      {f:<9} -")
            continue
        per = s.groupby("isin")[MISURA].median().sort_values()
        det = "  ".join(f"{i[-6:]}={v:.0f}" + ("*" if i == SOSPETTO else "")
                        for i, v in per.items())
        print(f"      {f:<9} mediana {s[MISURA].median():>6.1f}   ({len(per)} titoli)  {det}")
print("\n    * = il sospetto. Se nella fascia 12-18a, dove lui NON c'e', i titoli stanno")
print("    sotto quelli della 8-12a, il gradiente e' reale e non e' composizione.")

# --- 3. il controllo con Bloomberg, dove Bloomberg funziona -----------------------
print(f"\n--- 3. lo stesso gradiente nei numeri di Bloomberg (solo dal 2012) ---")
pz = CACHE / f"zsprd_{MERCATO}_Z_SPRD_MID.parquet"
if not pz.exists():
    print("    manca il pannello Bloomberg: salto.")
else:
    Zb = pd.read_parquet(pz); Zb.index = pd.to_datetime(Zb.index)
    zb = Zb.stack().dropna().rename("zb_l"); zb.index.names = ["date", "isin"]
    B = D.join(zb, on=["date", "isin"])
    zn = Zb.stack().dropna().rename("zb_n"); zn.index.names = ["date", "gemello"]
    pm = CACHE / f"bbgmatch_{MERCATO}.parquet"
    if pm.exists():
        M = pd.read_parquet(pm); M["date"] = pd.to_datetime(M["date"])
        B = B.merge(M[["date", "isin", "interp"]].rename(columns={"interp": "base_bbg"}),
                    on=["date", "isin"], how="left")
        print(f"    {'anno':<8}{'nostra pendenza':>18}{'Bloomberg':>12}{'n titoli':>10}")
        for y in [2013, 2014, 2016, 2017, 2020, 2022]:
            g = B[(B["anno"] == y) & (B["anni"] >= GINOCCHIO) & B["base_bbg"].notna()]
            if len(g) < 50:
                continue
            p1 = g.groupby("isin").agg(b=(MISURA, "median"), a=("anni", "median"))
            p2 = g.groupby("isin").agg(b=("base_bbg", "median"), a=("anni", "median"))
            if len(p1) < MIN_TIT:
                continue
            s1 = float(np.polyfit(p1["a"], p1["b"], 1)[0])
            s2 = float(np.polyfit(p2["a"], p2["b"], 1)[0])
            print(f"    {y:<8}{s1:>18.2f}{s2:>12.2f}{len(p1):>10}")
        print("\n    Se le due pendenze coincidono dal 2012, la FORMA della base contro la")
        print("    scadenza non e' un artefatto nostro: due macchine diverse la vedono")
        print("    uguale. Questo non dice nulla sul 2009-2011, dove Bloomberg e' rotto --")
        print("    ma dice che il nostro modo di trattare il lungo non e' storto di suo.")
    else:
        print("    manca bbgmatch: salto il confronto.")

# --- 4. quanto dipende dal ginocchio scelto ---------------------------------------
print(f"\n--- 4. e se il taglio fosse altrove? ---")
print(f"    {'taglio':<10}" + "".join(f"{y:>10}" for y in CALDI)
      + "   |" + "".join(f"{y:>10}" for y in CONTROLLO[:3]))
for da in (5.0, 8.0, 10.0, 12.0):
    riga = "".join(f"{pend(S, y, da)[0]:>10.2f}" if pd.notna(pend(S, y, da)[0])
                   else f"{'-':>10}" for y in CALDI)
    riga2 = "".join(f"{pend(S, y, da)[0]:>10.2f}" if pd.notna(pend(S, y, da)[0])
                    else f"{'-':>10}" for y in CONTROLLO[:3])
    print(f"    {f'>= {da:.0f}a':<10}{riga}   |{riga2}")
print("\n    Tutte le righe sono SENZA il sospetto. Se il segno regge a ogni taglio")
print("    ragionevole, la conclusione non dipende dal punto in cui ho deciso di")
print("    tagliare -- che e' l'unica cosa che rende quella scelta innocua.")

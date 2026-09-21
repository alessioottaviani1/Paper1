"""19 - E' IL FLOOR? Offline: nessuna chiamata a Bloomberg, nessuna scrittura.

L'IPOTESI, IN UNA RIGA. Lo Z-spread Bloomberg dei linker (BX326) sembra non applicare il
floor di rimborso alla pari. Se e' cosi', quando l'index ratio scende verso 1 Bloomberg
proietta flussi piu' bassi di quelli che il mercato prezza -- perche' il mercato il floor
lo incorpora -- il prezzo modello viene troppo basso, e per riagganciare il prezzo
osservato lo Z-spread deve scendere di centinaia di bp. Negativi, non grandi: NEGATIVI.

PERCHE' E' PLAUSIBILE, E PERCHE' NON BASTA CHE LO SIA. I titoli che producono le
osservazioni assurde sono quelli emessi fra il 2007 e il 2011, con base CPI vicina al picco
del 2011-2012, guardati negli anni in cui l'inflazione euro e' andata a zero e sotto. Il
BTPei 2035, emesso nel 2004 con base bassa, ha il 5% di assurde; quelli emessi dal 2023,
zero. E' esattamente l'ordinamento che l'ipotesi prevede -- il che e' un motivo per
testarla, non per crederle. La stessa graduatoria la produrrebbe qualunque effetto legato
all'eta' del titolo o al periodo.

IL TEST CHE DISCRIMINA. Se e' il floor, la rottura dipende dall'INDEX RATIO e non dall'anno
ne' dall'eta'. Quindi:

  1. |base| e quota di assurde per fascia di index ratio. L'ipotesi predice una relazione
     monotona che esplode sotto IR ~ 1.05 e sparisce sopra.
  2. LO STESSO DENTRO OGNI ANNO. E' il punto che separa l'ipotesi dalle rivali: se fosse
     "il periodo 2012-2017 e' rotto", dentro un anno tutti i titoli starebbero male allo
     stesso modo. Se e' il floor, nello STESSO anno i titoli con IR basso stanno male e
     quelli con IR alto no.
  3. Lo stesso per vita residua, per escludere che IR stia solo facendo da procura all'eta'.
  4. Il segno. Il floor non modellato puo' solo spingere lo Z-spread verso il BASSO: il
     prezzo osservato e' piu' alto di quello modello. Se le assurde fossero simmetriche,
     o positive, l'ipotesi e' morta comunque bella.

SE PASSA, non e' un dato da buttare: e' un LIMITE DOCUMENTATO del campo. La misura
Bloomberg resta valida dove il floor e' lontano dai soldi, e va dichiarata inutilizzabile
dove non lo e'. E spiega la correlazione -0.07 con la nostra misura, che il floor lo
modella (MARKET_FLOOR, value_floor).

SE NON PASSA -- se le assurde ci sono a tutti i livelli di IR, o sono simmetriche in segno
-- il floor non c'entra e si torna a guardare senza un'ipotesi in mano, che e' la posizione
onesta.
"""
import numpy as np
import pandas as pd
import bbg
from basis import mef_reference
from config import CACHE, MARKETS

# ----------------------------------------------------------------- impostazioni
MERCATO = "IT"
LAG, INTERP = 3, True      # convenzione euro, la stessa di basis.LinkerBond.from_ref
ASSURDA = 100.0
MIN_VITA = 365

# ----------------------------------------------------------------- dati
pm = CACHE / f"bbgmatch_{MERCATO}.parquet"
if not pm.exists():
    raise SystemExit(f"manca {pm.name}: lancia prima 17.")
D = pd.read_parquet(pm)
D["date"] = pd.to_datetime(D["date"])
D = D[D["ttm"] > MIN_VITA].copy()

m = MARKETS[MERCATO]
cpi = bbg.load(f"cpi_{m.cpi}").iloc[:, 0]
cpi.index = pd.to_datetime(cpi.index)
ref_l = bbg.load("ref_linker"); ref_l = ref_l[ref_l["mkt"] == MERCATO]
base = pd.to_numeric(ref_l["base_cpi_final"], errors="coerce")

# indice di riferimento: dipende solo dalla data (stessa convenzione per tutti i BTPei),
# quindi si calcola una volta per data invece che una volta per osservazione.
date_u = D["date"].drop_duplicates()
rif = {}
for d in date_u:
    try:
        rif[d] = mef_reference(cpi, d.date(), lag=LAG, interpolate=INTERP)
    except Exception:
        rif[d] = np.nan
D["rif"] = D["date"].map(rif)
D["base_cpi"] = D["isin"].map(base)
D["IR"] = D["rif"] / D["base_cpi"]
D = D[D["IR"].notna() & (D["IR"] > 0)]
if not len(D):
    raise SystemExit("nessun index ratio calcolabile: controlla base_cpi_final e la serie CPI.")

D["male"] = D["interp"].abs() > ASSURDA
D["ad"] = D["interp"].abs()
print(f"=== {MERCATO}: l'anomalia dipende dall'index ratio? ===")
print(f"    {len(D)} osservazioni, IR da {D['IR'].min():.3f} a {D['IR'].max():.3f}, "
      f"{int(D['male'].sum())} assurde ({D['male'].mean():.1%})\n")

tagli = [0, 0.98, 1.00, 1.02, 1.05, 1.10, 1.20, 1.40, 99]
et = ["< 0.98", "0.98-1.00", "1.00-1.02", "1.02-1.05", "1.05-1.10",
      "1.10-1.20", "1.20-1.40", "> 1.40"]
D["bIR"] = pd.cut(D["IR"], tagli, labels=et, right=False)
D["bTT"] = pd.cut(D["ttm"] / 365.25, [1, 2, 4, 7, 12, 99],
                  labels=["1-2a", "2-4a", "4-7a", "7-12a", "> 12a"], right=False)


def _riga(g: pd.DataFrame) -> str:
    if not len(g):
        return f"{0:>8}" + " " * 36
    return (f"{len(g):>8}{g['male'].mean():>10.0%}{g['interp'].median():>10.1f}"
            f"{np.percentile(g['interp'], 5):>11.1f}{g['interp'].min():>10.1f}")


H = f"{'n':>8}{'%assurde':>10}{'base med':>10}{'base p5':>11}{'base min':>10}"

# --- 1. per index ratio ------------------------------------------------------------
print("--- 1. per index ratio ---")
print(f"    {'IR':<12}{H}")
for b, g in D.groupby("bIR", observed=True):
    print(f"    {str(b):<12}{_riga(g)}")

# --- 2. per vita residua (IR fa solo da procura all'eta'?) -------------------------
print("\n--- 2. per vita residua ---")
print(f"    {'ttm':<12}{H}")
for b, g in D.groupby("bTT", observed=True):
    print(f"    {str(b):<12}{_riga(g)}")

# --- 3. IR DENTRO l'anno: il test che separa l'ipotesi dalle rivali ----------------
print("\n--- 3. quota di assurde per anno x index ratio ---")
print("    Se fosse 'il periodo e' rotto', dentro una riga sarebbero tutti uguali.")
print("    Se e' il floor, dentro la STESSA riga le colonne basse stanno male e le alte no.\n")
piv = D.pivot_table(index=D["date"].dt.year, columns="bIR", values="male",
                    aggfunc="mean", observed=True)
cnt = D.pivot_table(index=D["date"].dt.year, columns="bIR", values="male",
                    aggfunc="size", observed=True)
cols = [c for c in et if c in piv.columns]
print("    anno  " + "".join(f"{c:>12}" for c in cols))
for y in piv.index:
    cells = []
    for c in cols:
        v, n = piv.at[y, c], cnt.at[y, c]
        cells.append("          ." if pd.isna(v) or n < 20 else f"{v:>11.0%} ")
    print(f"    {y:<6}" + "".join(cells))
print("    ('.' = meno di 20 osservazioni in quella cella)")

# --- 4. il segno ------------------------------------------------------------------
ass = D[D["male"]]
neg = float((ass["interp"] < 0).mean()) if len(ass) else np.nan
print(f"\n--- 4. il segno delle assurde ---")
print(f"    {len(ass)} assurde: {neg:.0%} negative, {1-neg:.0%} positive")
print("    Un floor non modellato puo' spingere lo Z-spread solo verso il BASSO: il prezzo")
print("    osservato incorpora il floor ed e' PIU' ALTO di quello modello. Assurde")
print("    simmetriche, o in prevalenza positive, non sono spiegabili col floor.")

# --- verdetto ---------------------------------------------------------------------
basso = D[D["IR"] < 1.05]
alto  = D[D["IR"] >= 1.10]
q_b = basso["male"].mean() if len(basso) > 50 else np.nan
q_a = alto["male"].mean() if len(alto) > 50 else np.nan
# dentro-anno: si confrontano solo gli anni che hanno ENTRAMBE le fasce, altrimenti il
# confronto e' fra anni diversi e torna a essere l'ipotesi rivale.
anni_e = [y for y in piv.index
          if any(c in piv.columns and cnt.at[y, c] >= 20 and piv.at[y, c] == piv.at[y, c]
                 for c in cols if c in ("< 0.98", "0.98-1.00", "1.00-1.02", "1.02-1.05"))
          and any(c in piv.columns and cnt.at[y, c] >= 20 and piv.at[y, c] == piv.at[y, c]
                  for c in cols if c in ("1.10-1.20", "1.20-1.40", "> 1.40"))]
dentro = []
for y in anni_e:
    b = D[(D["date"].dt.year == y) & (D["IR"] < 1.05)]["male"].mean()
    a = D[(D["date"].dt.year == y) & (D["IR"] >= 1.10)]["male"].mean()
    dentro.append(b - a)

print(f"\n--- verdetto ---")
print(f"    assurde con IR < 1.05:  {q_b:.0%}" if np.isfinite(q_b) else "    IR < 1.05: poche oss.")
print(f"    assurde con IR >= 1.10: {q_a:.0%}" if np.isfinite(q_a) else "    IR >= 1.10: poche oss.")
if dentro:
    print(f"    dentro lo stesso anno, differenza fra le due fasce: "
          f"mediana {np.median(dentro):+.0%} su {len(dentro)} anni confrontabili")
else:
    print("    nessun anno ha entrambe le fasce: il test dentro-anno non e' possibile,")
    print("    e IR resta confuso col periodo. Il verdetto sotto vale meno.")

ok_liv = np.isfinite(q_b) and np.isfinite(q_a) and q_b > 3 * max(q_a, 0.01)
ok_den = bool(dentro) and np.median(dentro) > 0.10
ok_seg = np.isfinite(neg) and neg > 0.90

if ok_liv and ok_den and ok_seg:
    print("\n    IPOTESI FLOOR CONFERMATA su tutti e tre i controlli: dipende dall'index")
    print("    ratio, la dipendenza sopravvive dentro l'anno, e le assurde sono negative.")
    print("    BX326 non applica il floor di rimborso alla pari. Non e' un dato sporco da")
    print("    buttare: e' un limite del campo, da dichiarare. La misura Bloomberg vale")
    print("    dove il floor e' lontano dai soldi e non vale dove non lo e'. Va scelta una")
    print("    soglia di IR, motivata dal floor e non dal risultato, e riportata.")
elif ok_seg and ok_liv and not ok_den:
    print("\n    Dipende da IR e il segno torna, MA dentro l'anno la dipendenza sparisce:")
    print("    IR e periodo sono confusi. Con questo campione non si distingue il floor")
    print("    da 'quegli anni sono rotti'. Non si puo' concludere.")
elif not ok_seg:
    print("\n    IPOTESI FLOOR RESPINTA dal segno: le assurde non sono prevalentemente")
    print("    negative. Un floor non modellato non puo' produrre questo. Si ricomincia")
    print("    senza ipotesi: guarda le singole serie dei titoli peggiori, giorno per")
    print("    giorno, invece di cercarne un'altra a tavolino.")
else:
    print("\n    IPOTESI FLOOR NON CONFERMATA: le assurde non si concentrano dove l'index")
    print("    ratio e' basso. Qualunque sia la causa, non e' il floor.")

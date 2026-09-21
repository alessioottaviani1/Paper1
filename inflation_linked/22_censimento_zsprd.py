"""22 - CENSIMENTO DEL PANNELLO GREZZO. Offline, nessuna chiamata, nessuna scrittura.

QUESTO ANDAVA FATTO PER PRIMO. Finora ogni diagnosi e' partita dalla BASE, cioe' da una
differenza fra due numeri, e ha cercato di risalire a quale dei due fosse sbagliato. E'
il verso difficile. Il pannello grezzo si puo' interrogare da solo, senza il nominale,
senza il matching e senza la vita residua: uno Z-spread sovrano sta fra 0 e qualche
centinaio di bp, e tutto cio' che ne esce e' sospetto a prescindere da cosa ci si fa dopo.

LE QUATTRO DOMANDE.

 1. ZERI ESATTI. Il valore 0.00 preciso e' quasi sempre un dato mancante travestito: le
    librerie e i feed lo usano come riempitivo. Un vero Z-spread che vale esattamente zero
    a due decimali capita, ma non capita spesso, e non capita in blocco. Se gli zeri sono
    tanti, il problema non e' la metodologia di Bloomberg -- sono buchi che si comportano
    da numeri, e ogni statistica costruita sopra e' contaminata.

 2. DATE SPORCHE. Se in un giorno solo saltano MOLTI titoli insieme, e' una fotografia
    sbagliata, non un fenomeno di mercato. Si misura contando, per ogni data, quanti
    titoli escono dal corridoio: una data con il 60% dei titoli fuori non e' il 2011, e'
    uno snapshot rotto. Vanno scartate come date, non come osservazioni sparse.

 3. TITOLI SPORCHI. Simmetrico: titoli che escono dal corridoio quasi sempre.

 4. LE DUE GAMBE SEPARATE. Tutto sopra, per linker e per nominali. Se gli zeri e le date
    sporche stanno solo da una parte, il difetto e' di quel campo su quella classe di
    titoli, e si puo' circoscrivere invece di buttare via il campione.

IL CORRIDOIO. [-50, 800] bp e' largo di proposito. Lo spread dei BTP a 10 anni ha toccato
~550 bp nel novembre 2011: un corridoio stretto scambierebbe la crisi sovrana per un
difetto, che e' esattamente l'errore da non fare -- quella e' la parte piu' interessante
del campione, non la piu' sospetta.
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE

# ----------------------------------------------------------------- impostazioni
MERCATO = "IT"
CAMPO   = "Z_SPRD_MID"
LO, HI  = -50.0, 800.0      # corridoio plausibile per uno Z-spread sovrano
QUOTA_DATA = 0.25           # oltre questa quota di titoli fuori, la DATA e' sospetta

# ----------------------------------------------------------------- dati
p = CACHE / f"zsprd_{MERCATO}_{CAMPO}.parquet"
if not p.exists():
    raise SystemExit(f"manca {p.name}: lancia prima 15.")
Z = pd.read_parquet(p); Z.index = pd.to_datetime(Z.index); Z = Z.sort_index()
ref_l = bbg.load("ref_linker"); ref_l = ref_l[ref_l["mkt"] == MERCATO]
c_l = [c for c in Z.columns if c in ref_l.index]
c_n = [c for c in Z.columns if c not in ref_l.index]
mat = pd.concat([pd.to_datetime(ref_l["maturity"], errors="coerce"),
                 pd.to_datetime(bbg.load("ref_nominal")["maturity"], errors="coerce")])
mat = mat[~mat.index.duplicated()]

print(f"=== {MERCATO}: censimento di {CAMPO} ===")
print(f"    {Z.shape[1]} colonne ({len(c_l)} linker, {len(c_n)} nominali), "
      f"{Z.shape[0]} date, {int(Z.notna().sum().sum())} valori")
print(f"    {Z.index.min():%Y-%m-%d} -> {Z.index.max():%Y-%m-%d}\n")

gambe = {"linker": c_l, "nominali": c_n}


def _q(v: pd.Series) -> str:
    v = v.dropna()
    if not len(v):
        return " " * 48
    q = np.percentile(v, [1, 50, 99])
    return f"{q[0]:>10.1f}{q[1]:>10.1f}{q[2]:>10.1f}{v.min():>12.1f}{v.max():>12.1f}"


# --- 1. zeri esatti ---------------------------------------------------------------
print("--- 1. zeri esatti: dato mancante travestito da numero? ---")
for nome, cols in gambe.items():
    if not cols:
        continue
    V = Z[cols]
    n = int(V.notna().sum().sum())
    z0 = (V == 0.0)
    nz = int(z0.sum().sum())
    print(f"    {nome:<10} {nz:>7} zeri esatti su {n} valori ({nz/max(n,1):.2%})")
    if nz:
        per_anno = z0.groupby(z0.index.year).sum().sum(axis=1)
        per_anno = per_anno[per_anno > 0]
        print("      per anno: " + "  ".join(f"{y}:{int(v)}" for y, v in per_anno.items()))
        per_tit = z0.sum().sort_values(ascending=False)
        per_tit = per_tit[per_tit > 0]
        print(f"      su {len(per_tit)} titoli; i peggiori: " +
              ", ".join(f"{i} ({int(v)})" for i, v in per_tit.head(5).items()))
print("    Uno Z-spread che vale ESATTAMENTE 0.00 e' raro; in blocco non esiste.")

# --- 2. fuori corridoio, per gamba ------------------------------------------------
print(f"\n--- 2. distribuzione e code, corridoio [{LO:.0f}, {HI:.0f}] bp ---")
print(f"    {'':<10}{'n':>9}{'p1':>10}{'mediana':>10}{'p99':>10}{'min':>12}{'max':>12}{'fuori':>9}")
for nome, cols in gambe.items():
    if not cols:
        continue
    # .dropna() esplicito: in pandas recente stack() NON scarta piu' i NaN, e senza
    # questo la quota fuori corridoio verrebbe divisa per le celle vuote del pannello.
    v = Z[cols].stack().dropna()
    fuori = ((v < LO) | (v > HI)).mean()
    print(f"    {nome:<10}{len(v):>9}{_q(v)}{fuori:>8.1%}")

# --- 3. date sporche --------------------------------------------------------------
print(f"\n--- 3. date in cui saltano MOLTI titoli insieme (>{QUOTA_DATA:.0%}) ---")
for nome, cols in gambe.items():
    if not cols:
        continue
    V = Z[cols]
    fuori = ((V < LO) | (V > HI)) & V.notna()
    quota = fuori.sum(axis=1) / V.notna().sum(axis=1).replace(0, np.nan)
    brutte = quota[quota > QUOTA_DATA].dropna().sort_values(ascending=False)
    print(f"    {nome}: {len(brutte)} date su {int(V.notna().any(axis=1).sum())}")
    for d, q in brutte.head(12).items():
        n_out = int(fuori.loc[d].sum()); n_tot = int(V.loc[d].notna().sum())
        est = V.loc[d][fuori.loc[d]]
        print(f"      {d:%Y-%m-%d}  {n_out}/{n_tot} titoli fuori  "
              f"(da {est.min():.0f} a {est.max():.0f} bp)")
    if len(brutte) > 12:
        print(f"      ... e altre {len(brutte)-12}")
print("    Molti titoli fuori nello STESSO giorno = fotografia sbagliata, non mercato.")
print("    Si scartano come DATE: togliere le singole osservazioni lascerebbe dentro")
print("    quelle che quel giorno sono rimaste dentro il corridoio per caso.")

# --- 4. titoli sporchi ------------------------------------------------------------
print(f"\n--- 4. titoli che stanno fuori corridoio piu' spesso ---")
V = Z
fuori = ((V < LO) | (V > HI)) & V.notna()
tab = pd.DataFrame({"n": V.notna().sum(), "fuori": fuori.sum()})
tab = tab[tab["n"] >= 100]
tab["quota"] = tab["fuori"] / tab["n"]
tab["gamba"] = np.where(tab.index.isin(c_l), "linker", "nominale")
tab["scad"] = tab.index.map(mat)
tab = tab[tab["quota"] > 0].sort_values("quota", ascending=False)
print(f"    {len(tab)} titoli con almeno un valore fuori")
print(f"    {'isin':<15}{'gamba':<10}{'scad':<12}{'n':>7}{'fuori':>8}{'quota':>8}")
for i, r in tab.head(12).iterrows():
    sc = f"{r['scad']:%Y-%m-%d}" if pd.notna(r["scad"]) else "-"
    print(f"    {i:<15}{r['gamba']:<10}{sc:<12}{int(r['n']):>7}{int(r['fuori']):>8}"
          f"{r['quota']:>8.1%}")

# --- 5. vita residua: l'amplificazione 1/duration ---------------------------------
print(f"\n--- 5. fuori corridoio per vita residua ---")
tt = pd.DataFrame(index=Z.index, columns=Z.columns, dtype=float)
for c in Z.columns:
    if pd.notna(mat.get(c)):
        tt[c] = (mat[c] - Z.index).days / 365.25
L = pd.concat([Z.stack().rename("z"), tt.stack().rename("anni")], axis=1).dropna(how="any")
L["fuori"] = (L["z"] < LO) | (L["z"] > HI)
L["gamba"] = np.where(L.index.get_level_values(1).isin(c_l), "linker", "nominale")
L["b"] = pd.cut(L["anni"], [0, 0.5, 1, 2, 4, 8, 99],
                labels=["<6m", "6m-1a", "1-2a", "2-4a", "4-8a", ">8a"], right=False)
piv = L.pivot_table(index="b", columns="gamba", values="fuori", aggfunc="mean", observed=True)
cnt = L.pivot_table(index="b", columns="gamba", values="fuori", aggfunc="size", observed=True)
print(f"    {'ttm':<10}" + "".join(f"{c:>22}" for c in piv.columns))
for b in piv.index:
    cells = "".join(f"{piv.at[b,c]:>13.1%} ({int(cnt.at[b,c]):>6})" if pd.notna(piv.at[b, c])
                    else f"{'-':>22}" for c in piv.columns)
    print(f"    {str(b):<10}{cells}")
print("    Se la quota esplode sotto l'anno su ENTRAMBE le gambe, e' l'amplificazione")
print("    1/duration e si risolve con la soglia di vita residua. Se esplode solo sui")
print("    linker, no: li' la duration e' la stessa e il difetto e' del campo.")

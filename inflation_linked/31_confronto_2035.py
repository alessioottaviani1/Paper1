"""31 - IL 2035: e' la nostra macchina o era caro davvero? Offline, nessun run.

DOVE SIAMO. Quattro ipotesi costruttive, quattro morte, tutte con un test che poteva
falsificarle:
  A  proiezione estrapolata      -> l'ILS copre 25 e 30 anni al 100% in quegli anni;
  A' effetto di orizzonte        -> tolto il 2035 la pendenza non cala, CAMBIA SEGNO;
  B  interpolazione su buco largo-> 48 eventi di restringimento del bracket, effetto
                                    mediano -0.3 bp. Stringere non sposta niente;
  C  prezzo stantio              -> zero o un giorno all'anno a prezzo invariato.

A questo punto la tentazione e' inventare la quinta ipotesi. Ieri ne ho inventate quattro
di fila su un campo che misurava un'altra cosa, e sembravano tutte ragionevoli. Quindi no:
prima si usa l'unico confronto indipendente rimasto, che e' gratis e non l'ho ancora
usato qui.

LO Z-SPREAD DI BLOOMBERG SULLO STESSO TITOLO. Z_SPRD_MID e' uno z-spread del linker contro
la curva swap: la STESSA grandezza che calcoliamo noi, per una strada completamente
diversa -- la loro analitica, la loro curva, la loro proiezione. Sul 2012+ abbiamo gia'
visto che le due strade danno la stessa base a 0.18 bp. Allora la domanda diventa netta:

  se anche Bloomberg vede il 2035 con uno z basso nel 2009-2010, il titolo era CARO e
  non c'e' niente da riparare -- e' economia, va raccontata;
  se Bloomberg lo vede in linea con gli altri e solo noi no, il difetto e' nostro e si
  restringe a quel titolo, quindi a prezzo o anagrafica.

CON UNA CAUTELA CHE NON VA DIMENTICATA. Il 18 ha stabilito che nel pre-2012 la gamba
linker di Bloomberg esce dal corridoio nel 7.8% dei casi. Quindi prima di usarlo come
giudice si controlla se il 2035 sia fra i titoli che Bloomberg sbaglia: un giudice che
sbaglia proprio sull'imputato non serve. Il controllo e' nel punto 1 e viene PRIMA del
confronto, non dopo.
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE, MARKETS

# ----------------------------------------------------------------- impostazioni
MERCATO  = "IT"
CAMPO    = "Z_SPRD_MID"
TARGET   = "IT0003745541"
MIN_VITA = 365
LO, HI   = -50.0, 800.0          # corridoio di plausibilita' per uno z sovrano
CALDI    = (2009, 2011)

# ----------------------------------------------------------------- dati
D = pd.read_parquet(CACHE / f"swapbasis_{MERCATO}.parquet")
D["date"] = pd.to_datetime(D["date"])
D = D[D["interp"].notna() & (D["ttm"] > MIN_VITA)].copy()
D["anno"] = D["date"].dt.year

pz = CACHE / f"zsprd_{MERCATO}_{CAMPO}.parquet"
if not pz.exists():
    raise SystemExit(f"manca {pz.name}: e' lo scarico del 15.")
Z = pd.read_parquet(pz); Z.index = pd.to_datetime(Z.index)
zb = Z.stack().dropna().rename("z_bbg"); zb.index.names = ["date", "isin"]
D = D.join(zb, on=["date", "isin"])

ref_l = bbg.load("ref_linker"); ref_l = ref_l[ref_l["mkt"] == MERCATO]
mat = pd.to_datetime(ref_l["MATURITY"], errors="coerce")
px = bbg.load(f"px_mid_{MERCATO}"); px.index = pd.to_datetime(px.index)

print(f"=== {MERCATO}: il 2035 secondo noi e secondo Bloomberg ===")
print(f"    {len(D)} osservazioni, {int(D['z_bbg'].notna().sum())} con z Bloomberg")

# --- 1. Bloomberg e' un giudice attendibile SU QUESTO TITOLO? ----------------------
print(f"\n--- 1. prima di usarlo come giudice: Bloomberg sbaglia sul 2035? ---")
print(f"    quota di z Bloomberg fuori dal corridoio [{LO:.0f}, {HI:.0f}] bp\n")
print(f"    {'titolo':<16}{'scad':<12}{'n':>7}{'fuori':>8}{'quota':>8}   {'periodo':<18}")
for isin, g in D[D["z_bbg"].notna()].groupby("isin"):
    fuori = ((g["z_bbg"] < LO) | (g["z_bbg"] > HI))
    if fuori.mean() > 0.01 or isin == TARGET:
        sc = mat.get(isin)
        print(f"    {isin:<16}{sc:%Y-%m-%d}  {len(g):>7}{int(fuori.sum()):>8}"
              f"{fuori.mean():>8.1%}   {g['date'].min():%Y-%m}/{g['date'].max():%Y-%m}"
              + ("   <- il sospetto" if isin == TARGET else ""))
S = D[D["isin"] == TARGET]
f_t = ((S["z_bbg"] < LO) | (S["z_bbg"] > HI))
print(f"\n    Sul sospetto: {int(f_t.sum())} valori fuori corridoio su "
      f"{int(S['z_bbg'].notna().sum())} ({f_t.mean():.1%})")
if f_t.mean() > 0.05:
    print("    !! Bloomberg sbaglia proprio su questo titolo: come giudice non vale, e il")
    print("       confronto che segue va letto con le pinze.")
else:
    print("    Bloomberg sta nel corridoio su questo titolo: come termine di confronto")
    print("    indipendente regge, almeno come livello.")

# --- 2. i due z(linker) a confronto, sul sospetto ---------------------------------
print(f"\n--- 2. z(linker) nostro contro z(linker) Bloomberg, su {TARGET} ---")
print(f"    {'anno':<7}{'n':>6}{'nostro':>10}{'bloomberg':>12}{'scarto':>9}"
      f"{'  corr giornaliera':>19}")
for y, g in S.groupby("anno"):
    gg = g[g["z_bbg"].notna()]
    if not len(gg):
        print(f"    {y:<7}{len(g):>6}{g['z_lnk'].median():>10.1f}{'-':>12}{'-':>9}{'-':>19}")
        continue
    r = gg["z_lnk"].corr(gg["z_bbg"]) if len(gg) > 10 else np.nan
    rs = f"{r:+.2f}" if pd.notna(r) else "-"
    print(f"    {y:<7}{len(gg):>6}{gg['z_lnk'].median():>10.1f}{gg['z_bbg'].median():>12.1f}"
          f"{(gg['z_lnk'] - gg['z_bbg']).median():>9.1f}{rs:>19}")
print("\n    Se lo scarto e' piccolo e la correlazione alta, i due z sono la stessa cosa:")
print("    allora il livello basso del nostro NON e' un difetto nostro.")

# --- 3. e negli anni caldi, il 2035 e' basso anche per Bloomberg? -----------------
print(f"\n--- 3. il confronto che conta: il 2035 contro gli altri linker, "
      f"nei due sistemi ---")
print("    Non il livello assoluto -- quello dipende dalla curva e dalle convenzioni --")
print("    ma la POSIZIONE RELATIVA: il 2035 sta sotto il gruppo? In tutti e due?\n")
print(f"    {'anno':<7}{'  n altri':>10}{'nostro: 2035':>14}{'altri':>9}{'diff':>8}"
      f"{'   bbg: 2035':>14}{'altri':>9}{'diff':>8}")
for y, g in D.groupby("anno"):
    t = g[g["isin"] == TARGET]
    a = g[g["isin"] != TARGET]
    if not len(t) or a["isin"].nunique() < 2:
        continue
    d_noi = t["z_lnk"].median() - a["z_lnk"].median()
    tb, ab = t["z_bbg"].dropna(), a["z_bbg"].dropna()
    if len(tb) and len(ab):
        d_bbg = tb.median() - ab.median()
        print(f"    {y:<7}{a['isin'].nunique():>10}{t['z_lnk'].median():>14.1f}"
              f"{a['z_lnk'].median():>9.1f}{d_noi:>8.1f}{tb.median():>14.1f}"
              f"{ab.median():>9.1f}{d_bbg:>8.1f}")
    else:
        print(f"    {y:<7}{a['isin'].nunique():>10}{t['z_lnk'].median():>14.1f}"
              f"{a['z_lnk'].median():>9.1f}{d_noi:>8.1f}{'-':>14}{'-':>9}{'-':>8}")
print("\n    ATTENZIONE: 'altri' e' una mediana su titoli piu' CORTI, quindi una differenza")
print("    non nulla e' attesa -- la curva ha una pendenza. Quel che conta e' se la")
print("    differenza si comporta allo STESSO MODO nei due sistemi: se nel 2009-2010")
print("    scende in tutti e due, il movimento e' del mercato.")

# --- 4. anagrafica: c'e' qualcosa di diverso in quel titolo? ----------------------
print(f"\n--- 4. anagrafica del sospetto contro gli altri linker ---")
col = [c for c in ("CPN", "CPN_FREQ", "base_cpi_final", "START_ACC_DT", "ISSUE_DT",
                   "FIRST_CPN_DT", "FIRST_CPN_PERIOD_TYP", "INFLATION_LAG",
                   "MATURITY") if c in ref_l.columns]
print(f"    {'campo':<24}{'il 2035':<24}{'gli altri (range)':<40}")
for c in col:
    v = ref_l.at[TARGET, c] if TARGET in ref_l.index else "?"
    al = ref_l.loc[[i for i in ref_l.index if i != TARGET], c].dropna()
    if pd.api.types.is_numeric_dtype(al):
        rng = f"da {al.min()} a {al.max()}"
    else:
        u = sorted({str(x)[:10] for x in al})
        rng = ", ".join(u[:3]) + (f" (+{len(u)-3})" if len(u) > 3 else "")
    print(f"    {c:<24}{str(v)[:22]:<24}{rng[:38]:<40}")
print("\n    Un valore del sospetto FUORI dal range degli altri e' un indizio; dentro il")
print("    range non prova nulla, ma toglie un sospetto.")

# --- 5. la lista da controllare a terminale ---------------------------------------
print(f"\n--- 5. righe da verificare a schermo, se i punti sopra non bastano ---")
bb = ref_l["bb_id"].astype(str).str.strip()
righe = []
for y in range(CALDI[0], CALDI[1] + 1):
    g = S[S["anno"] == y]
    for _, r in g.iloc[:: max(1, len(g) // 2)].head(2).iterrows():
        righe.append(r)
g = S[S["anno"] == 2014]
if len(g):
    righe += [g.iloc[len(g) // 2]]          # una riga di CONTROLLO, anno sano
print(f"    {'#':>2} {'data':<12}{'ticker':<22}{'nostro z':>10}{'bbg z':>9}"
      f"{'base':>8}{'prezzo':>10}")
for i, r in enumerate(righe, 1):
    p = px.at[r["date"], TARGET] if TARGET in px.columns and r["date"] in px.index else np.nan
    zb_ = f"{r['z_bbg']:.1f}" if pd.notna(r.get("z_bbg")) else "-"
    print(f"    {i:>2} {r['date']:%Y-%m-%d}  {bb.get(TARGET,'?')+'@BGN Corp':<22}"
          f"{r['z_lnk']:>10.1f}{zb_:>9}{r['interp']:>8.1f}{p:>10.3f}")
print(f"\n    Su YAS del titolo, a quella data: il PREZZO serve a riconoscere la data, poi")
print(f"    si guarda lo z-spread. Se a schermo coincide col nostro, la macchina e' a")
print(f"    posto e quel titolo era caro davvero.")

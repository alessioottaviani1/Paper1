"""33 - LA TERZA MISURA: la gamba nominale da una CURVA, non da due punti.

LA DOMANDA DI ALESSIO: se il gemello e' lontano, qual e' il gold standard? Tenere
l'osservazione solo quando escono titoli vicini, o altro?

LA RISPOSTA BREVE: altro. Scartare le osservazioni con buco largo sarebbe curare un male
che abbiamo MISURATO e trovato assente -- il 30 ha fatto 48 esperimenti naturali sui
restringimenti di bracket e l'effetto mediano e' -0.3 bp. Costerebbe il 7-12% del
campione, tutto concentrato negli anni iniziali, cioe' selezione correlata col tempo, per
comprare niente.

Il problema vero del due-punti non e' il buco: e' che BUTTA VIA INFORMAZIONE. Al 2009,
attorno al BTPei 2035, ci sono sette BTP entro otto anni di scadenza -- 2027, 2029, 2031,
2033, 2034, 2037, 2039 -- e noi ne usiamo due. Una retta fra due punti non puo' vedere la
CURVATURA della curva degli spread, che e' esattamente cio' che si perde interpolando su
un tratto lungo. Usarli tutti non richiede di scartare nulla e degrada dolcemente quando
la scala e' rada: e' piu' informazione, non meno dati.

COME. Si stima z_nom(T) con una parabola locale sui nominali entro una finestra attorno
alla scadenza del linker, e la si legge alla scadenza del linker. Parabola e non retta
perche' la curvatura e' il termine che la retta non vede; LOCALE e non globale perche'
una curva unica su tutte le scadenze verrebbe tirata dal breve, dove i titoli sono
tantissimi. Un giro robusto: si stima, si scartano i residui oltre 3 MAD, si ristima --
cosi' un singolo nominale con prezzo sporco non trascina il fit. Lo scarto avviene sui
RESIDUI del fit nominale, mai sul valore della base: e' l'unico modo di essere robusti
senza selezionare sulla variabile dipendente.

COSA NON E'. Non e' la curva sovrana fittata che usavamo in basis_zspread. Li' la curva
serviva a SCONTARE, ed era costruita dai nominali stessi, per cui lo z del nominale
contro di essa era zero per costruzione. Qui lo sconto resta la curva SWAP -- esogena a
entrambe le gambe -- e il fit serve solo a leggere il livello della gamba nominale alla
scadenza giusta. Sono due usi diversi della stessa parola.

TRE MISURE, E SI DICHIARANO TUTTE. PAIR (un gemello), INTERP (due), CURVA (tutti quelli
vicini). Se danno lo stesso numero, la scelta non conta e si dice. Se divergono, divergono
dove la scala e' rada, e allora il paper ha un fatto da raccontare invece di un filtro da
giustificare.
"""
import time
import numpy as np
import pandas as pd
import bbg
from config import CACHE

# ----------------------------------------------------------------- impostazioni
MERCATO   = "DE"
FINESTRA  = 8.0        # anni attorno alla scadenza del linker entro cui prendere i nominali
MIN_BOND  = 5          # sotto, la parabola non si stima: si ricade sulla retta
MIN_RETTA = 3
GRADO     = 2
MAD_K     = 3.0        # soglia di robustezza, sui residui del fit nominale
SOSPETTO  = None       # None = lo sceglie dai dati; oppure un ISIN imposto a mano
SALVA     = True

# ----------------------------------------------------------------- dati
pz = CACHE / f"swapz_nom_{MERCATO}.parquet"
if not pz.exists():
    raise SystemExit(
        f"manca {pz.name}: rilancia il 24, che ora salva il pannello degli z dei\n"
        f"nominali. E' il file su cui questo script lavora.")
Z = pd.read_parquet(pz); Z.index = pd.to_datetime(Z.index)
D = pd.read_parquet(CACHE / f"swapbasis_{MERCATO}.parquet")
D["date"] = pd.to_datetime(D["date"])
ref_n = bbg.load("ref_nominal")
mat_n = pd.to_datetime(ref_n["MATURITY"], errors="coerce").dropna()
mat_n = mat_n[[c for c in Z.columns if c in mat_n.index]]
Z = Z[list(mat_n.index)]
mn = mat_n.values.astype("datetime64[D]").astype(int)

print(f"=== {MERCATO}: la gamba nominale da una curva locale ===")
print(f"    {Z.shape[0]} date x {Z.shape[1]} nominali; finestra +/-{FINESTRA:.0f} anni, "
      f"grado {GRADO}, robustezza {MAD_K:.0f} MAD")


def _fit(t, z, t0):
    """Parabola locale robusta: stima, scarta i residui oltre MAD_K MAD, ristima.
    Ritorna (valore in t0, n usati, sd dei residui)."""
    g = GRADO if len(t) >= MIN_BOND else 1
    if len(t) < MIN_RETTA:
        return np.nan, len(t), np.nan
    x = t - t0                       # centrare sulla scadenza del linker: il termine noto
    c = np.polyfit(x, z, g)          # diventa direttamente il valore cercato, e il fit
    r = z - np.polyval(c, x)         # e' meglio condizionato
    mad = np.median(np.abs(r - np.median(r)))
    if mad > 1e-9:
        tieni = np.abs(r - np.median(r)) <= MAD_K * mad
        if tieni.sum() >= max(MIN_RETTA, g + 1) and tieni.sum() < len(t):
            x, z = x[tieni], z[tieni]
            g = GRADO if len(x) >= MIN_BOND else 1
            c = np.polyfit(x, z, g)
            r = z - np.polyval(c, x)
    return float(c[-1]), len(x), float(np.std(r))


# --- il calcolo ---------------------------------------------------------------------
righe, t0 = [], time.time()
per_data = {d: g for d, g in D.groupby("date")}
for k, (d, g) in enumerate(per_data.items()):
    if d not in Z.index:
        continue
    zz = Z.loc[d].values.astype(float)
    ok = np.isfinite(zz)
    if ok.sum() < MIN_RETTA:
        continue
    t_all = (mn[ok] - np.datetime64(d.date(), "D").astype(int)) / 365.25
    z_all = zz[ok]
    for _, r in g.iterrows():
        t_l = float(r["ttm"]) / 365.25
        sel = np.abs(t_all - t_l) <= FINESTRA
        if sel.sum() < MIN_RETTA:
            continue
        zf, n_u, sd = _fit(t_all[sel], z_all[sel], t_l)
        if not np.isfinite(zf):
            continue
        righe.append((d, r["isin"], float(r["z_lnk"]), float(r["z_lnk"]) - zf,
                      zf, n_u, sd, float(r["ttm"]),
                      float(r["interp"]) if pd.notna(r["interp"]) else np.nan,
                      float(r["pair"]) if pd.notna(r["pair"]) else np.nan,
                      float(r["bracket"]) if pd.notna(r["bracket"]) else np.nan))
    if (k + 1) % 1000 == 0:
        print(f"  {k+1}/{len(per_data)} date ({time.time()-t0:.0f}s)")

C = pd.DataFrame(righe, columns=["date", "isin", "z_lnk", "curva", "z_nom_fit",
                                 "n_fit", "sd_fit", "ttm", "interp", "pair", "bracket"])
if not len(C):
    raise SystemExit("nessuna osservazione prodotta.")
C["anno"] = C["date"].dt.year
C["anni"] = C["ttm"] / 365.25
print(f"\nfatto: {len(C)} osservazioni, {C['isin'].nunique()} linker, "
      f"{time.time()-t0:.0f}s")
print(f"    nominali usati per fit: mediana {C['n_fit'].median():.0f}, "
      f"minimo {C['n_fit'].min():.0f}")
print(f"    residuo del fit nominale: mediana {C['sd_fit'].median():.1f} bp, "
      f"p95 {C['sd_fit'].quantile(.95):.1f}")

# --- 1. le tre misure a confronto ---------------------------------------------------
V = C[C["ttm"] > 365]
print(f"\n--- 1. le tre misure (bp, vita residua > 1 anno) ---")
print(f"    {'misura':<10}{'n':>8}{'media':>9}{'sd':>8}{'mediana':>9}{'p5':>8}{'p95':>8}")
for nome in ("pair", "interp", "curva"):
    s = V[nome].dropna()
    print(f"    {nome:<10}{len(s):>8}{s.mean():>9.1f}{s.std():>8.1f}{s.median():>9.1f}"
          f"{s.quantile(.05):>8.1f}{s.quantile(.95):>8.1f}")
b = V[V["interp"].notna()]
_r = b["interp"].corr(b["curva"]) if len(b) > 3 else np.nan
print(f"\n    corr(interp, curva) = "
      + (f"{_r:+.4f}" if np.isfinite(_r) else "n.d.")
      + f", |differenza| mediana {(b['interp'] - b['curva']).abs().median():.1f} bp")
print(f"    dove il bracket e' largo (> 700 gg): "
      f"{(b.loc[b['bracket'] > 700, 'interp'] - b.loc[b['bracket'] > 700, 'curva']).abs().median():.1f} bp")
print("    Se le due coincidono anche a bracket largo, il due-punti non stava perdendo")
print("    nulla e INTERP resta la misura primaria senza bisogno di difenderla.")

# --- 2. per anno, e il sospetto -----------------------------------------------------
# Il titolo da seguire si SCEGLIE DAI DATI quando non e' imposto: quello la cui base
# mediana si scosta di piu' dalla mediana di mercato. Cablato a un ISIN italiano, su
# Francia e Spagna la colonna usciva tutta "-" senza che si notasse -- e una colonna vuota
# si legge come "niente da vedere", non come "sto guardando il titolo sbagliato".
if SOSPETTO is None or SOSPETTO not in set(V["isin"]):
    _med = V.groupby("isin")["interp"].median().dropna()
    SOSPETTO = (str((_med - float(V["interp"].median())).abs().idxmax())
                if len(_med) else str(V["isin"].iloc[0]))
    print(f"\n    titolo piu' anomalo, scelto dai dati: {SOSPETTO}")

print(f"\n--- 2. per anno: interp contro curva ---")
print(f"    {'anno':<7}{'n':>7}{'interp':>9}{'curva':>9}{'diff':>8}   "
      f"{'{} interp'.format(SOSPETTO[-6:]):>14}{'curva':>9}{'diff':>8}")
for y, g in V.groupby("anno"):
    t = g[g["isin"] == SOSPETTO]
    ti = f"{t['interp'].median():>14.1f}" if len(t) and t["interp"].notna().any() else f"{'-':>14}"
    tc = f"{t['curva'].median():>9.1f}" if len(t) else f"{'-':>9}"
    td = (f"{t['curva'].median() - t['interp'].median():>8.1f}"
          if len(t) and t["interp"].notna().any() else f"{'-':>8}")
    print(f"    {y:<7}{len(g):>7}{g['interp'].median():>9.1f}{g['curva'].median():>9.1f}"
          f"{g['curva'].median() - g['interp'].median():>8.1f}   {ti}{tc}{td}")
print("\n    Se sul sospetto la curva alza sensibilmente la base nel 2009-2011 mentre")
print("    sugli altri non cambia nulla, allora il due-punti su quel titolo PERDEVA")
print("    davvero la curvatura, e la misura giusta e' la curva. Se non cambia, il")
print("    valore negativo e' quel che dicono i dati e va raccontato, non aggiustato.")

# --- 3. la qualita' del fit dove la scala e' rada -----------------------------------
print(f"\n--- 3. dove il fit e' piu' fragile ---")
# La colonna 'n con interp' non e' decorazione. Nella versione precedente, dove in una
# fascia nessuna osservazione aveva interp, |interp-curva| usciva 'nan' e si leggeva come
# un bug di formattazione: era invece il dato piu' importante della tabella -- in quella
# fascia la misura primaria NON ESISTE e solo CURVA la copre. Un confronto vuoto va
# dichiarato vuoto, col suo conteggio accanto, non stampato come numero mancante.
print(f"    {'vita residua':<14}{'n':>9}{'n con interp':>14}{'bond nel fit':>14}"
      f"{'residuo sd':>13}{'|interp-curva|':>16}")
for lo, hi in [(1, 5), (5, 10), (10, 15), (15, 20), (20, 40)]:
    s = V[(V["anni"] >= lo) & (V["anni"] < hi)]
    if not len(s):
        continue
    d_ = (s["interp"] - s["curva"]).abs().dropna()
    print(f"    {f'{lo}-{hi}a':<14}{len(s):>9}{int(s['interp'].notna().sum()):>14}"
          f"{s['n_fit'].median():>14.0f}{s['sd_fit'].median():>13.1f}"
          + (f"{d_.median():>16.1f}" if len(d_) else f"{'-':>16}"))
print("    Il numero di bond nel fit e' l'analogo onesto del bracket: dice quanta")
print("    informazione c'e' davvero dietro la gamba nominale di quell'osservazione.")
_vuote = [f"{lo}-{hi}a" for lo, hi in [(1, 5), (5, 10), (10, 15), (15, 20), (20, 40)]
          if len(V[(V["anni"] >= lo) & (V["anni"] < hi)])
          and not V.loc[(V["anni"] >= lo) & (V["anni"] < hi), "interp"].notna().any()]
if _vuote:
    print(f"    !! fasce senza NESSUN interp: {', '.join(_vuote)}. Li' la misura primaria")
    print("       non copre il campione e va detto nel paper, o allargata la sua regola.")

if SALVA:
    out = CACHE / f"curvabasis_{MERCATO}.parquet"
    C.to_parquet(out)
    print(f"\nsalvato: {out}")
    print("    colonne: curva (la terza misura), z_nom_fit, n_fit, sd_fit, piu' interp e")
    print("    pair per il confronto. Le tre misure vivono insieme: nel paper si mostrano")
    print("    tutte e tre e si dice che coincidono, invece di sceglierne una e spiegare.")

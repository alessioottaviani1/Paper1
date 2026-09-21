"""37 - PERCHE' INTERP PERDE META' DEL CAMPIONE TEDESCO. Nessun terminale: legge i parquet.

IL FATTO. Sulla Germania il 24 chiude con 'pair 17688' e 'interp 8720'. Su Italia, Francia
e Spagna le due colonne sono quasi uguali. Meta' del campione tedesco non ha la misura che
abbiamo dichiarato PRIMARIA, e prima di decidere cosa fare bisogna sapere QUALE dei due
vincoli la sta togliendo, perche' i due hanno statuto opposto:

  (a) NESSUN NOMINALE SU UN LATO. Il linker sta fuori dall'intervallo delle scadenze
      nominali disponibili, o il lato lungo e' vuoto. Non e' un filtro: e' assenza di dato.
      Interpolare si trasformerebbe in estrapolazione, e non si puo' recuperare senza
      cambiare il POOL -- cioe' senza rimettere mano a DE_NOMINAL_CURVE.

  (b) BRACKET OLTRE MAX_BRACKET. I due nominali ci sono entrambi, ma distano piu' di
      MAX_BRACKET giorni e l'osservazione viene buttata da una SOGLIA NOSTRA. Questa e'
      una decisione, non un limite dei dati -- e per di piu' una decisione che abbiamo
      gia' misurato altrove: il 30 ha fatto 48 esperimenti naturali sui restringimenti di
      bracket e l'effetto mediano sulla base e' -0.3 bp. Se sulla Germania la causa
      dominante e' questa, stiamo pagando meta' campione per comprare niente, e per di piu'
      in modo incoerente col resto del capitolo.

I due casi si distinguono senza ricalcolare niente, perche' il 24 salva gia' la colonna
'bracket': dove interp e' NaN, bracket NaN dice (a) e bracket finito dice (b).

POI SI MISURA, non si decide a tavolino. Con swapz_nom_{M}.parquet si puo' ricostruire
interp SENZA nessun limite di bracket e guardare cosa succede al crescere del buco: se la
base recuperata resta sullo stesso livello e continua a coincidere con CURVA -- che usa
tutti i nominali vicini e non ha soglie -- allora MAX_BRACKET non stava proteggendo nulla.
Se invece oltre una certa larghezza interp si stacca da CURVA, quella larghezza e' la
soglia giusta, e la si sceglie sul numero invece che a occhio.

CONTROLLO OBBLIGATORIO. MAX_BRACKET e' globale: toccarlo cambia anche IT, FR, ES. Quindi
questo script va lanciato ANCHE con MERCATO = "IT", dove il pool e' denso e la soglia non
dovrebbe quasi mai mordere. Se allargando il bracket la base italiana si muove, la soglia
sta facendo un lavoro vero e non si tocca.
"""
import re
import numpy as np
import pandas as pd
import bbg
from config import CACHE, ROOT

MERCATO = "IT"
BANDE   = [(0, 183), (183, 366), (366, 731), (731, 1095), (1095, 1826),
           (1826, 2557), (2557, 10_000)]

# La soglia si LEGGE dal 24 invece di riscriverla qui. Due copie della stessa costante in
# due file e' esattamente il modo in cui questo progetto si e' gia' fatto male: basta che
# una delle due cambi e questo script misura un mondo che non esiste.
_src = (ROOT / "24_basis_swap.py").read_text(encoding="utf-8", errors="ignore")
_m = re.search(r"^MAX_BRACKET\s*=\s*(\d+)", _src, re.M)
if not _m:
    raise SystemExit("non trovo MAX_BRACKET nel 24: e' stato rinominato? allinea questo script.")
MAX_BRACKET = int(_m.group(1))

D = pd.read_parquet(CACHE / f"swapbasis_{MERCATO}.parquet")
D["date"] = pd.to_datetime(D["date"])
Z = pd.read_parquet(CACHE / f"swapz_nom_{MERCATO}.parquet")
Z.index = pd.to_datetime(Z.index)
ref_n = bbg.load("ref_nominal")
mat_n = pd.to_datetime(ref_n["MATURITY"], errors="coerce").dropna()
com = [c for c in Z.columns if c in mat_n.index]
Z, mat_n = Z[com], mat_n[com]
mnum_all = mat_n.values.astype("datetime64[D]").astype(int)

print(f"=== {MERCATO}: perche' INTERP perde osservazioni ===")
print(f"    MAX_BRACKET letto dal 24: {MAX_BRACKET} giorni")
print(f"    pannello nominali: {Z.shape[0]} date x {Z.shape[1]} titoli")

V = D[D["ttm"] > 365].copy()
n_tot, n_pair, n_int = len(V), int(V["pair"].notna().sum()), int(V["interp"].notna().sum())
print(f"\n--- 0. il conto (vita residua > 1 anno) ---")
print(f"    osservazioni            {n_tot:>8}")
print(f"    con pair                {n_pair:>8}  ({n_pair/n_tot:>4.0%})")
print(f"    con interp              {n_int:>8}  ({n_int/n_tot:>4.0%})")
print(f"    senza interp            {n_tot-n_int:>8}  ({1-n_int/n_tot:>4.0%})")

# --- 1. le due cause ----------------------------------------------------------------
P = V[V["interp"].isna()]
lato = P["bracket"].isna()
n_a, n_b = int(lato.sum()), int((~lato).sum())
print(f"\n--- 1. perche' si perdono ---")
print(f"    {'causa':<34}{'n':>8}{'quota':>8}")
if n_tot > n_int:
    print(f"    {'nessun nominale su un lato':<34}{n_a:>8}{n_a/(n_tot-n_int):>8.0%}")
    print(f"    {f'bracket oltre {MAX_BRACKET} gg':<34}{n_b:>8}{n_b/(n_tot-n_int):>8.0%}")
if n_b:
    q = P.loc[~lato, "bracket"]
    print(f"    bracket delle perse per soglia: mediana {q.median():.0f} gg, "
          f"p90 {q.quantile(.9):.0f}, max {q.max():.0f}")
if n_a:
    print(f"    linker che perdono un lato: "
          + ", ".join(f"{i[-6:]} ({c})" for i, c in
                      P.loc[lato, "isin"].value_counts().head(5).items()))

# --- 2. interp SENZA limite, ricostruita dal pannello --------------------------------
# Si ricalcola con la stessa identica meccanica del 24 (searchsorted sulle scadenze
# ordinate, peso lineare in giorni) ma senza la soglia, cosi' l'unica differenza fra le
# due colonne e' MAX_BRACKET e nient'altro.
libero, brk_l = {}, {}
for d, g in V.groupby("date"):
    if d not in Z.index:
        continue
    zz = Z.loc[d]
    ok = zz.notna().values
    if ok.sum() < 2:
        continue
    mm, zs = mnum_all[ok], zz.values[ok].astype(float)
    o = np.argsort(mm)
    mm, zs = mm[o], zs[o]
    dnum = np.datetime64(d.date(), "D").astype(int)
    for i, r in g.iterrows():
        ml = dnum + int(r["ttm"])
        pos = int(np.searchsorted(mm, ml))
        a, b_ = pos - 1, pos
        if a < 0 or b_ >= len(mm):
            continue
        brk = int(mm[b_] - mm[a])
        w = 0.5 if brk == 0 else (ml - mm[a]) / brk
        libero[i] = float(r["z_lnk"]) - ((1 - w) * zs[a] + w * zs[b_])
        brk_l[i] = brk
V["interp_free"] = pd.Series(libero)
V["brk_free"] = pd.Series(brk_l)

# CONTROLLO DURO. Dove il 24 aveva prodotto interp, la ricostruzione deve dare lo stesso
# numero. Se non lo da', sto ricostruendo un'altra cosa e tutto cio' che segue e' rumore.
ctl = V[V["interp"].notna() & V["interp_free"].notna()]
print(f"\n--- 2. ricostruzione senza soglia ---")
# Il controllo puo' non avere righe su cui girare: se la soglia ha tolto TUTTO, non esiste
# nessun interp del 24 da confrontare. Non e' un fallimento -- ma non e' nemmeno una
# verifica passata, e le due cose non vanno confuse: si dichiara che il controllo non ha
# potuto girare e si va avanti. Fermarsi qui sarebbe la versione peggiore, perche' il caso
# in cui la ricostruzione serve di piu' e' proprio quello in cui non c'e' nulla da
# confrontare.
if len(ctl):
    err = float((ctl["interp"] - ctl["interp_free"]).abs().max())
    print(f"    controllo sulle {len(ctl)} righe dove il 24 aveva gia' interp: "
          f"scarto max {err:.6f} bp")
    if not err < 1e-6:
        raise SystemExit("    !! la ricostruzione NON riproduce il 24: non leggere oltre, "
                         "e' un altro calcolo.")
else:
    print("    !! nessuna riga di controllo: il 24 non ha prodotto NESSUN interp qui, "
          "quindi la\n       ricostruzione non e' verificabile contro di lui. "
          "Numeri da leggere con questa riserva.")
rec = V["interp_free"].notna() & V["interp"].isna()
print(f"    recuperate {int(rec.sum())} osservazioni "
      f"({(n_int + int(rec.sum()))/n_tot:.0%} del campione contro {n_int/n_tot:.0%})")
print(f"    restano fuori {int(V['interp_free'].isna().sum())}: quelle senza un lato, che")
print("    nessuna soglia puo' recuperare -- solo un pool piu' largo.")

# --- 3. la domanda vera: allargare il bracket peggiora la misura? --------------------
pc = CACHE / f"curvabasis_{MERCATO}.parquet"
C = None
if pc.exists():
    C = pd.read_parquet(pc)
    C["date"] = pd.to_datetime(C["date"])
    V = V.merge(C[["date", "isin", "curva"]], on=["date", "isin"], how="left")
else:
    print(f"\n    (manca {pc.name}: lancia il 33 per avere anche il confronto con CURVA)")

print(f"\n--- 3. interp per larghezza del bracket ---")
print(f"    {'bracket':<14}{'n':>8}{'media':>9}{'mediana':>9}{'sd':>8}"
      + (f"{'|int-curva|':>13}{'corr':>8}" if C is not None else ""))
for lo, hi in BANDE:
    s = V[(V["brk_free"] >= lo) & (V["brk_free"] < hi) & V["interp_free"].notna()]
    if not len(s):
        continue
    eti = f"{lo}-{hi} gg" if hi < 10_000 else f"oltre {lo} gg"
    riga = (f"    {eti:<14}{len(s):>8}{s['interp_free'].mean():>9.1f}"
            f"{s['interp_free'].median():>9.1f}{s['interp_free'].std():>8.1f}")
    if C is not None:
        x = s[["interp_free", "curva"]].dropna()
        riga += (f"{(x['interp_free']-x['curva']).abs().median():>13.1f}"
                 f"{x['interp_free'].corr(x['curva']):>8.3f}") if len(x) > 3 else \
                f"{'-':>13}{'-':>8}"
    print(riga)
print(f"    la riga di soglia e' fra {'/'.join(str(b[1]) for b in BANDE[:4])}: sopra")
print(f"    {MAX_BRACKET} gg il 24 oggi butta tutto.")

# --- 4. il livello cambia? -----------------------------------------------------------
# Il confronto onesto non e' 'stretti contro larghi' -- sono bond e anni diversi. E'
# sullo STESSO campione: quanto si sposta la media di mercato aggiungendo le recuperate.
print(f"\n--- 4. effetto sulla misura aggregata ---")
a_ = V.loc[V["interp"].notna(), "interp"]
b_ = V.loc[V["interp_free"].notna(), "interp_free"]
print(f"    {'campione':<24}{'n':>8}{'media':>9}{'mediana':>9}{'p5':>8}{'p95':>8}")
print(f"    {'oggi (bracket <= soglia)':<24}{len(a_):>8}{a_.mean():>9.1f}"
      f"{a_.median():>9.1f}{a_.quantile(.05):>8.1f}{a_.quantile(.95):>8.1f}")
print(f"    {'senza soglia':<24}{len(b_):>8}{b_.mean():>9.1f}"
      f"{b_.median():>9.1f}{b_.quantile(.05):>8.1f}{b_.quantile(.95):>8.1f}")
if C is not None:
    cc = V["curva"].dropna()
    print(f"    {'curva (nessuna soglia)':<24}{len(cc):>8}{cc.mean():>9.1f}"
          f"{cc.median():>9.1f}{cc.quantile(.05):>8.1f}{cc.quantile(.95):>8.1f}")

print(f"\n--- 5. per anno: quanto campione cambia ---")
V["anno"] = V["date"].dt.year
print(f"    {'anno':<8}{'n oggi':>9}{'n senza':>9}{'media oggi':>13}{'media senza':>13}")
for y, g in V.groupby("anno"):
    o = g["interp"].dropna(); f_ = g["interp_free"].dropna()
    print(f"    {y:<8}{len(o):>9}{len(f_):>9}"
          + (f"{o.mean():>13.1f}" if len(o) else f"{'-':>13}")
          + (f"{f_.mean():>13.1f}" if len(f_) else f"{'-':>13}"))

print(f"\n--- come si legge ---")
print("    Sezione 1 dice se il buco e' DATO MANCANTE (nessun lato) o NOSTRA SOGLIA.")
print("    Sezione 3 dice se la soglia sta proteggendo qualcosa: se |interp-curva| resta")
print("    piatta anche a bracket largo, no. Se esplode oltre una certa banda, quella e'")
print("    la soglia giusta e si scrive quella, motivandola con questa tabella.")
print("    Sezione 4 dice se togliendola la misura di mercato si sposta: se la media si")
print("    muove di meno del suo errore standard, il campione grande e' gratis.")
print("    Poi RILANCIARE questo stesso script con MERCATO = 'IT' come controllo: la")
print("    soglia e' globale, e se muove l'Italia non si tocca per far contenta la")
print("    Germania.")

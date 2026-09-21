"""30 - L'ESPERIMENTO NATURALE: cosa succede alla base quando il bracket si stringe?

LA DOMANDA DI ALESSIO, che e' la domanda giusta. Il gemello non e' fisso: per il BTPei
2035 il bracket e' 915 giorni nel 2009 (BTP 2034-08 / 2037-02), 762 nel 2016 quando esce
il 2036-09, 366 nel 2020 col 2035-03 / 2036-03, 61 nel 2026. La REGOLA e' sempre la
stessa -- i due nominali quotati piu' vicini, uno sotto e uno sopra, ricalcolati ogni
giorno -- ma la QUALITA' della misura cambia nel tempo, e migliora sempre. Che e' un modo
elegante di dire che l'errore di misura e' correlato col calendario, il che puo' fabbricare
una tendenza che non c'e'.

E QUI C'E' UN ESPERIMENTO NATURALE, gratis. Il giorno in cui il Tesoro emette un titolo
dentro il buco, il bracket di quel linker si stringe da un giorno all'altro. Il linker e'
lo stesso, il prezzo e' lo stesso, il mercato e' lo stesso: l'unica cosa cambiata e' COME
costruiamo la sua gamba nominale. Se la base fa un salto in quel momento, il salto e'
nostro -- e' la misura, non il mercato. Se non fa nulla, interpolare su 915 giorni non
costava niente e tutta l'ipotesi B cade.

PERCHE' E' MEGLIO DEL LEAVE-ONE-OUT DEL 28. Il 28 misura l'errore su ALTRI titoli e lo
trasporta per analogia. Qui la misura e' sullo STESSO titolo, nello STESSO giorno, con
l'unica differenza che conta. Non serve trasportare niente. E non serve nemmeno rilanciare
il 24: il bracket e' gia' una colonna del pannello.

IL CONTROLLO CONTROFATTUALE. Fra il mese prima e il mese dopo la base si muove anche per
ragioni di mercato, e quel movimento non c'entra. Quindi non si guarda il salto del
linker colpito, ma la DIFFERENZA fra il suo salto e quello degli altri linker nello stesso
periodo, che l'evento non tocca. Se il mercato si e' mosso, si muove per tutti e sparisce
nella differenza.
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE

# ----------------------------------------------------------------- impostazioni
MERCATO  = "IT"
MISURA   = "interp"
MIN_VITA = 365
TARGET   = "IT0003745541"
FIN      = 20        # giorni di borsa prima e dopo l'evento
SALTO    = 30        # giorni: variazione minima di bracket per contare come evento
CAPS     = [1095, 900, 730, 550, 400]     # soglie di bracket da valutare

# ----------------------------------------------------------------- dati
D = pd.read_parquet(CACHE / f"swapbasis_{MERCATO}.parquet")
D["date"] = pd.to_datetime(D["date"])
D = D[D[MISURA].notna() & (D["ttm"] > MIN_VITA) & D["bracket"].notna()].copy()
D = D.sort_values(["isin", "date"]).reset_index(drop=True)
ref_l = bbg.load("ref_linker"); mat_l = pd.to_datetime(ref_l["MATURITY"], errors="coerce")

print(f"=== {MERCATO}: la base salta quando cambia il bracket? ===")
print(f"    {len(D)} osservazioni, {D['isin'].nunique()} linker")

# --- 1. la regola, e come si vede sui dati ----------------------------------------
print(f"\n--- 1. la regola in atto: i due quotati piu' vicini, ogni giorno ---")
print(f"    storia del bracket per {TARGET} (scad {mat_l.get(TARGET):%Y-%m-%d})\n")
S = D[D["isin"] == TARGET]
cam = S[S["bracket"].diff().abs() > SALTO]
print(f"    {'dal':<12}{'al':<12}{'bracket gg':>12}{'w':>8}{'giorni':>8}{'base mediana':>15}")
tagli = [S["date"].min()] + list(cam["date"]) + [S["date"].max() + pd.Timedelta(days=1)]
for i in range(len(tagli) - 1):
    g = S[(S["date"] >= tagli[i]) & (S["date"] < tagli[i + 1])]
    if not len(g):
        continue
    print(f"    {g['date'].min():%Y-%m-%d}  {g['date'].max():%Y-%m-%d}"
          f"{g['bracket'].median():>12.0f}{g['w'].median():>8.2f}{len(g):>8}"
          f"{g[MISURA].median():>15.1f}")
print("\n    La regola non cambia mai: cambia il mercato sotto di essa. Ogni riga e' un")
print("    regime di misura diverso per lo STESSO titolo, e il confronto fra le righe")
print("    e' quasi un esperimento -- manca solo il controllo per il mercato.")

# --- 2. tutti gli eventi, su tutti i linker ---------------------------------------
print(f"\n--- 2. gli eventi: bracket che cambia di oltre {SALTO} giorni ---")
ev = []
for isin, g in D.groupby("isin"):
    g = g.reset_index(drop=True)
    d = g["bracket"].diff()
    for k in g.index[(d.abs() > SALTO)]:
        if k < FIN or k + FIN >= len(g):
            continue
        pre, post = g.loc[k - FIN:k - 1], g.loc[k:k + FIN - 1]
        ev.append({"isin": isin, "data": g.at[k, "date"],
                   "br_pre": float(pre["bracket"].median()),
                   "br_post": float(post["bracket"].median()),
                   "b_pre": float(pre[MISURA].mean()),
                   "b_post": float(post[MISURA].mean()),
                   "k": k})
E = pd.DataFrame(ev)
if not len(E):
    raise SystemExit("nessun evento: il bracket non cambia mai di tanto.")
E["d_br"] = E["br_post"] - E["br_pre"]
E = E[E["d_br"].abs() > SALTO]
print(f"    {len(E)} eventi su {E['isin'].nunique()} linker, "
      f"da {E['data'].min():%Y-%m} a {E['data'].max():%Y-%m}")
print(f"    di cui {int((E['d_br'] < 0).sum())} restringimenti e "
      f"{int((E['d_br'] > 0).sum())} allargamenti")

# --- 3. il salto, al netto del mercato --------------------------------------------
# Senza controfattuale questo test misurerebbe soprattutto il mercato: fra un mese e
# l'altro la base si muove di decine di bp per conto suo. Il controllo sono gli ALTRI
# linker nella stessa finestra, che l'evento non tocca.
print(f"\n--- 3. il salto della base, meno il salto degli altri linker ---")
eff = []
for _, r in E.iterrows():
    fin_d = D[(D["date"] >= r["data"] - pd.Timedelta(days=FIN * 2)) &
              (D["date"] <= r["data"] + pd.Timedelta(days=FIN * 2))]
    altri = fin_d[fin_d["isin"] != r["isin"]]
    # esclude i linker che hanno un evento loro nella stessa finestra
    tocchi = set(E.loc[(E["data"] - r["data"]).abs() <= pd.Timedelta(days=FIN * 2), "isin"])
    altri = altri[~altri["isin"].isin(tocchi)]
    if altri["isin"].nunique() < 3:
        continue
    a_pre = altri[altri["date"] < r["data"]].groupby("isin")[MISURA].mean()
    a_post = altri[altri["date"] >= r["data"]].groupby("isin")[MISURA].mean()
    com = a_pre.index.intersection(a_post.index)
    if len(com) < 3:
        continue
    mercato = float((a_post[com] - a_pre[com]).median())
    eff.append({**r.to_dict(), "mercato": mercato,
                "effetto": (r["b_post"] - r["b_pre"]) - mercato})
F = pd.DataFrame(eff)
if not len(F):
    raise SystemExit("nessun evento con controllo sufficiente.")
print(f"    {len(F)} eventi con almeno 3 linker di controllo\n")
print(f"    {'variazione bracket':<24}{'n':>7}{'effetto mediano':>18}{'p25':>9}{'p75':>9}")
for et, s in [("si stringe molto (< -300)", F[F["d_br"] < -300]),
              ("si stringe (-300..-30)", F[(F["d_br"] >= -300) & (F["d_br"] < -SALTO)]),
              ("si allarga (> +30)", F[F["d_br"] > SALTO])]:
    if len(s):
        print(f"    {et:<24}{len(s):>7}{s['effetto'].median():>18.1f}"
              f"{s['effetto'].quantile(.25):>9.1f}{s['effetto'].quantile(.75):>9.1f}")
rho = float(F["d_br"].corr(F["effetto"])) if len(F) >= 4 else np.nan
print(f"\n    corr(variazione bracket, effetto) = "
      + (f"{rho:+.3f}" if np.isfinite(rho) else f"n.d. (servono 4 eventi, ce ne sono {len(F)})"))
print("    Atteso se l'interpolazione su buchi larghi SOVRASTIMA la gamba nominale:")
print("    stringendo il bracket la gamba nominale scende e la base SALE, quindi")
print("    variazione negativa del bracket ed effetto positivo -> correlazione NEGATIVA.")

# --- 4. gli eventi del titolo sospetto, uno per uno --------------------------------
print(f"\n--- 4. gli eventi di {TARGET} ---")
T = F[F["isin"] == TARGET]
if len(T):
    print(f"    {'data':<12}{'bracket pre':>13}{'post':>8}{'base pre':>11}{'post':>8}"
          f"{'mercato':>10}{'effetto':>10}")
    for _, r in T.iterrows():
        print(f"    {r['data']:%Y-%m-%d}{r['br_pre']:>13.0f}{r['br_post']:>8.0f}"
              f"{r['b_pre']:>11.1f}{r['b_post']:>8.1f}{r['mercato']:>10.1f}"
              f"{r['effetto']:>10.1f}")
else:
    print(f"    nessun evento con controllo per {TARGET}")

# --- 5. verdetto, e cosa costa una soglia ------------------------------------------
print(f"\n--- verdetto ---")
stretti = F[F["d_br"] < -SALTO]
m_str = float(stretti["effetto"].median()) if len(stretti) else np.nan
print(f"    restringendo il bracket la base si sposta di {m_str:+.1f} bp (mediana su "
      f"{len(stretti)} eventi)")
if pd.notna(m_str) and m_str > 5:
    print("\n    L'INTERPOLAZIONE SU BUCHI LARGHI COSTA. Stringere il bracket alza la base:")
    print("    il buco largo stava sovrastimando la gamba nominale, esattamente come")
    print("    serviva per spiegare il 2035. E' misurato sullo stesso titolo nello stesso")
    print("    giorno, quindi non e' un'analogia.")
elif pd.notna(m_str) and abs(m_str) < 3:
    print("\n    NON COSTA NULLA. Stringere il bracket non sposta la base: interpolare su")
    print("    915 giorni era innocuo, e l'ipotesi B cade. Allora l'anomalia del 2035 e'")
    print("    sua -- prezzo o anagrafica -- e i dati in cache hanno finito di parlare.")
else:
    print("\n    EFFETTO PICCOLO ma non nullo: contribuisce senza spiegare tutto.")

print(f"\n--- 6. cosa costerebbe una soglia sul bracket ---")
print("    Il bracket e' ESOGENO -- dipende dal calendario di emissione del Tesoro, non")
print("    dal prezzo del titolo ne' dal valore della base -- quindi filtrarci sopra e'")
print("    legittimo, come la vita residua. Filtrare sulla base non lo sarebbe mai.\n")
print(f"    {'soglia gg':<12}{'osservazioni':>14}{'persa':>9}{'linker':>9}{'prima data':>14}")
for c in CAPS:
    s = D[D["bracket"] <= c]
    if not len(s):
        print(f"    {c:<12}{'vuoto':>14}")
        continue
    d0 = f"{s['date'].min():%Y-%m-%d}"
    print(f"    {c:<12}{len(s):>14}{1 - len(s)/len(D):>8.1%}"
          f"{s['isin'].nunique():>9}{d0:>14}")
print("\n    Una soglia che tagliasse solo gli anni iniziali sarebbe selezione su una cosa")
print("    correlata col tempo: si dichiara, si mostra la serie con e senza, e si sceglie")
print("    PRIMA di guardare quale delle due piace di piu'.")

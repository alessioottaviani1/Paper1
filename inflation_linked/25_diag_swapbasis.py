"""25 - LA NOSTRA BASE REGGE UN ESAME? Offline: legge solo cio' che ha prodotto il 24.

PERCHE' QUESTO SCRIPT ESISTE. La serie annuale che esce dal 24 e' economicamente
plausibile: bassa e stabile prima del 2008, sale con Lehman, esplode nel 2011-2012, poi
si assesta. E' esattamente la forma che ci si aspetta da un premio di liquidita' sui
BTPei. Ma "plausibile" e' precisamente lo stato in cui si accettano numeri sbagliati:
ieri quattro ipotesi sono state costruite sopra un campo che misurava un'altra cosa, e
sembravano tutte ragionevoli. Quindi prima di estendere a ES/FR/DE si chiede alla serie
di reggere tre domande che possono FALSIFICARLA.

  1. CONCENTRAZIONE. La media annuale e' su coppie bond-giorno. Nel 2011-2012 i BTPei
     vivi erano pochi: se uno solo e' rotto e ha molti giorni, si prende l'anno da solo e
     il picco non e' il mercato, e' quel titolo. Si confronta la media con la MEDIANA
     DELLE MEDIANE PER TITOLO -- se il picco e' del mercato le due coincidono, se e' di
     un titolo divergono. E si stampa la quota del titolo piu' presente.

  2. LE CODE NEGATIVE. Ci sono anni con p5 a -30/-50 bp. Una base negativa esiste
     (il linker puo' stare caro), ma se le negative si ammucchiassero sulla vita residua
     corta sarebbero rumore: a duration piccola un errore di prezzo di un centesimo
     diventa decine di bp di spread. Si guarda DOVE stanno, non solo quante sono.

  3. PAIR CONTRO INTERP. PAIR ha skew -19.9, INTERP -0.19. Sono la stessa grandezza
     misurata in due modi, quindi la differenza non e' economia: e' il disallineamento di
     scadenza fra linker e gemello. Se la differenza cresce col mismatch, la spiegazione
     e' quella e INTERP resta la misura primaria -- e lo si sa, non lo si spera.

COSA NON FA. Non aggiusta niente e non filtra niente. Stampa e basta: i filtri si
decidono dopo aver visto, e con un criterio scritto prima di guardare i risultati.
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE

# ----------------------------------------------------------------- impostazioni
MERCATO   = "DE"
MISURA    = "interp"     # la primaria; PAIR entra solo nel confronto del punto 3
NEG       = -25.0        # bp: sotto questa soglia un'osservazione e' "coda negativa"
MIN_VITA  = 365          # giorni: soglia ESOGENA (vita residua), mai sul valore misurato

# ----------------------------------------------------------------- dati
p = CACHE / f"swapbasis_{MERCATO}.parquet"
if not p.exists():
    raise SystemExit(f"manca {p.name}: lancia prima il 24.")
D = pd.read_parquet(p)
D["date"] = pd.to_datetime(D["date"])
D["anno"] = D["date"].dt.year
D["anni"] = D["ttm"] / 365.25
ref_l = bbg.load("ref_linker"); ref_l = ref_l[ref_l["mkt"] == MERCATO]
mat = pd.to_datetime(ref_l["MATURITY"], errors="coerce")

V = D[D[MISURA].notna() & (D["ttm"] > MIN_VITA)].copy()
print(f"=== {MERCATO}: esame della base ({MISURA}) ===")
print(f"    {len(D)} righe grezze -> {len(V)} con vita residua > {MIN_VITA}gg, "
      f"{V['isin'].nunique()} titoli, {V['date'].nunique()} date")
if "regime" in V.columns:
    print(f"    costruzione curva: {dict(V['regime'].value_counts())}")

# --- 1. concentrazione: il picco e' del mercato o di un titolo? --------------------
print(f"\n--- 1. concentrazione: media contro mediana-delle-mediane-per-titolo ---")
print("    Se le due colonne coincidono, l'anno lo fanno tutti i titoli insieme. Se")
print("    divergono, lo fa uno solo, e la media di quell'anno non e' il mercato.\n")
print(f"    {'anno':<7}{'n':>7}{'titoli':>8}{'media':>9}{'med/tit':>9}{'scarto':>9}"
      f"{'top titolo':>12}{'  quota':>8}")
righe = []
for y, g in V.groupby("anno"):
    per_tit = g.groupby("isin")[MISURA].median()
    quote = g["isin"].value_counts(normalize=True)
    med_med = float(per_tit.median())
    mu = float(g[MISURA].mean())
    top = quote.index[0]
    print(f"    {y:<7}{len(g):>7}{g['isin'].nunique():>8}{mu:>9.1f}{med_med:>9.1f}"
          f"{mu - med_med:>9.1f}{top[-8:]:>12}{quote.iloc[0]:>7.0%}")
    righe.append((y, mu, med_med, mu - med_med, g["isin"].nunique(), quote.iloc[0]))
R = pd.DataFrame(righe, columns=["anno", "media", "med_med", "scarto", "n_tit", "quota"])
sospetti = R[(R["scarto"].abs() > 10) | ((R["quota"] > 0.5) & (R["n_tit"] > 1))]
if len(sospetti):
    print(f"\n    !! {len(sospetti)} anni da guardare: {', '.join(str(int(a)) for a in sospetti['anno'])}")
    print("       scarto oltre 10 bp = la media e' tirata da pochi titoli; quota > 50% =")
    print("       un titolo da solo fa meta' delle osservazioni dell'anno.")
else:
    print("\n    Nessun anno in cui la media diverga dalla mediana-per-titolo di oltre 10 bp:")
    print("    il profilo annuale e' del mercato, non di un titolo.")

# --- 2. gli anni di stress, titolo per titolo -------------------------------------
# La finestra si SCEGLIE DAI DATI e non si scrive a mano: centrata sull'anno di massima
# base mediana, piu' due prima e tre dopo. Fissata al 2009-2014 funzionava per l'Italia e
# sarebbe uscita VUOTA per la Spagna, che ha linker solo dal 2014 -- e una tabella vuota
# non si legge come "finestra sbagliata", si legge come "nessun problema".
_ann = V.groupby("anno")[MISURA].median()
_pic = int(_ann.idxmax()) if len(_ann) else int(V["anno"].min())
_a0, _a1 = max(int(V["anno"].min()), _pic - 2), min(int(V["anno"].max()), _pic + 3)
print(f"\n--- 2. gli anni attorno al massimo ({_pic}): si muovono TUTTI i titoli? ---")
fin = V[V["anno"].between(_a0, _a1)]
tav = fin.pivot_table(index="isin", columns="anno", values=MISURA, aggfunc="median")
cnt = fin.pivot_table(index="isin", columns="anno", values=MISURA, aggfunc="size")
print(f"    mediana per titolo e anno (n fra parentesi); '-' = titolo non vivo\n")
print(f"    {'titolo':<16}{'scad':<12}" + "".join(f"{a:>13}" for a in tav.columns))
for isin in tav.index:
    sc = mat.get(isin)
    cel = "".join(
        f"{f'{tav.at[isin, a]:.0f} ({int(cnt.at[isin, a])})':>13}"
        if pd.notna(tav.at[isin, a]) else f"{'-':>13}" for a in tav.columns)
    print(f"    {isin:<16}{sc:%Y-%m-%d}  {cel}" if pd.notna(sc)
          else f"    {isin:<16}{'?':<12}{cel}")
vivi = tav.notna().sum()
salita = {}
for a in tav.columns:
    if a - 1 in tav.columns:
        d = (tav[a] - tav[a - 1]).dropna()
        salita[a] = (len(d), int((d > 0).sum()))
print(f"\n    titoli che salgono rispetto all'anno prima:")
for a, (n, su) in salita.items():
    print(f"      {a}: {su}/{n}" + ("   <- movimento comune" if n and su >= 0.8 * n else ""))
print("    Se negli anni di stress sale quasi ogni titolo vivo, il picco e' il mercato")
print("    di quegli anni. Se sale uno solo mentre gli altri stanno fermi, no.")

# --- 3. le code negative: dove stanno? --------------------------------------------
print(f"\n--- 3. code negative (sotto {NEG:.0f} bp): rumore o economia? ---")
V["corta"] = V["anni"] < 2.0
neg = V[V[MISURA] < NEG]
print(f"    {len(neg)} osservazioni su {len(V)} ({len(neg)/len(V):.2%})")
if len(neg):
    print(f"\n    {'vita residua':<16}{'n tutte':>10}{'n negative':>12}{'% negative':>12}")
    bins = [1, 2, 3, 5, 10, 100]
    et = ["1-2a", "2-3a", "3-5a", "5-10a", "10a+"]
    V["fascia"] = pd.cut(V["anni"], bins=bins, labels=et, right=False)
    for f in et:
        tot = int((V["fascia"] == f).sum())
        nn = int(((V["fascia"] == f) & (V[MISURA] < NEG)).sum())
        print(f"    {f:<16}{tot:>10}{nn:>12}{(nn/tot if tot else 0):>11.2%}")
    # Il verdetto LEGGE la fascia dominante invece di annunciare quella che mi aspettavo.
    # Scritto come prima -- "se cresce al calare della vita residua" -- lo script nominava
    # solo l'esito che avevo in testa, e sull'Italia l'esito e' stato l'opposto: zero
    # negative fra 3 e 10 anni, l'85% oltre i 10. Un diagnostico che vede una sola delle
    # due risposte e' mezzo cieco.
    quote_f = {f: (float(((V["fascia"] == f) & (V[MISURA] < NEG)).mean() /
                         max(1e-12, (V["fascia"] == f).mean())))
               for f in et if (V["fascia"] == f).any()}
    dom = max(quote_f, key=quote_f.get)
    print()
    if dom == et[0]:
        print("    Si concentrano sulla vita residua PIU' CORTA: e' rumore da duration")
        print("    piccola -- a duration bassa un errore di prezzo fisso vale tanti piu' bp")
        print("    quanto piu' corto e' il titolo. Si alza MIN_VITA: filtro ESOGENO, sulla")
        print("    vita residua, mai sul valore della base.")
    elif dom == et[-1]:
        print("    Si concentrano sul LUNGO, non sul corto. La duration li' e' grande, quindi")
        print("    NON e' rumore di prezzo: un errore che sopravvive a una duration di venti")
        print("    anni e' un errore di costruzione. I sospetti sono la proiezione")
        print("    dell'inflazione (l'ILS lungo e' rado) o l'interpolazione fra gemelli")
        print("    distanti. Alzare MIN_VITA qui non serve a niente: guarda il 26.")
    else:
        print(f"    Si concentrano sulla fascia {dom}, ne' al corto ne' al lungo: non e'")
        print("    duration ne' curva. Guarda quali titoli le producono, qui sotto.")
    top_a = neg["anno"].value_counts().head(5)
    print(f"\n    anni che le concentrano: " +
          ", ".join(f"{a} ({n}, {n/len(neg):.0%})" for a, n in top_a.items()))
    top_i = neg["isin"].value_counts().head(5)
    print(f"    titoli che le producono:")
    for isin, n in top_i.items():
        tot = int((V["isin"] == isin).sum())
        sc = mat.get(isin)
        sc_s = f"{sc:%Y-%m-%d}" if pd.notna(sc) else "?"
        print(f"      {isin}  scad {sc_s}   {n} negative su {tot} sue "
              f"({n/tot:.1%}), {n/len(neg):.0%} del totale")
    if len(top_i) and top_i.iloc[0] / len(neg) > 0.4:
        print(f"\n    !! un solo titolo fa il {top_i.iloc[0]/len(neg):.0%} delle negative:")
        print("       non e' una proprieta' della misura, e' quel titolo in quel periodo.")

# --- 4. PAIR contro INTERP ---------------------------------------------------------
print(f"\n--- 4. PAIR contro INTERP: la differenza e' il disallineamento di scadenza? ---")
B = D[D["pair"].notna() & D["interp"].notna() & (D["ttm"] > MIN_VITA)].copy()
if len(B):
    B["gap"] = B["pair"] - B["interp"]
    rho = float(B["pair"].corr(B["interp"]))
    print(f"    {len(B)} righe con entrambe, correlazione {rho:+.3f}")
    print(f"    skew: pair {B['pair'].skew():+.2f}   interp {B['interp'].skew():+.2f}")
    print(f"\n    {'mismatch (gg)':<18}{'n':>9}{'|pair-interp| mediano':>24}{'p95':>10}")
    for lo, hi in [(0, 30), (30, 90), (90, 183), (183, 10000)]:
        s = B[(B["mismatch"] >= lo) & (B["mismatch"] < hi)]
        et = f"{lo}-{hi}" if hi < 10000 else f"oltre {lo}"
        if len(s):
            print(f"    {et:<18}{len(s):>9}"
                  f"{s['gap'].abs().median():>21.1f} bp{s['gap'].abs().quantile(.95):>9.1f}")
    rho_m = float(B["gap"].abs().corr(B["mismatch"]))
    print(f"\n    corr(|pair - interp|, mismatch) = {rho_m:+.3f}")
    if rho_m > 0.15:
        print("    Positiva: la divergenza cresce col disallineamento, come deve. La coda")
        print("    di PAIR e' il bias di scadenza, non un difetto della misura -- e INTERP,")
        print("    che interpola fra i due nominali che bracciano, non ce l'ha. INTERP")
        print("    primaria, PAIR robustezza.")
    else:
        print("    Debole: allora la coda di PAIR NON viene dal mismatch, e va spiegata")
        print("    altrove prima di usare PAIR anche solo come robustezza.")
else:
    print("    nessuna riga con entrambe le misure.")

print("\n--- cosa decide questo script ---")
print("  punto 1 e 2 puliti  -> il profilo annuale e' il mercato: si puo' estendere a")
print("     gli altri mercati sapendo che questo regge.")
print("  punto 1 sporco      -> la media annuale va sostituita con la mediana per titolo")
print("     (o si pesano i titoli, non le osservazioni) PRIMA di qualunque grafico.")
print("  punto 3 in crescita al corto -> si alza MIN_VITA e si rifa la tabella annuale.")

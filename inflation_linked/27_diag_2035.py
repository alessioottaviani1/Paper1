"""27 - E' IL 2035, O E' LA SCADENZA? Offline: solo file gia' in cache.

DOVE SIAMO. Il 26 ha escluso due ipotesi su tre in modo pulito. Il prezzo del titolo non
era fermo (zero o un giorno all'anno a prezzo invariato). E il bracket delle osservazioni
negative e' 915 giorni contro i 365 delle sane sopra i dieci anni, con corr(bracket, base)
= -0.372: un indizio serio per l'interpolazione fra gemelli distanti.

MA IL MIO TEST SULL'IPOTESI A ERA SBAGLIATO, e va detto prima di costruirci sopra. Ho
guardato la COPERTURA dei tenor ILS lunghi -- 25a e 30a quotati al 100% in ogni anno -- e
ho concluso che la proiezione non e' estrapolata. E' lo stesso errore di ieri, in forma
piu' mite: la copertura non e' validita'. INDEX_Z_SPREAD_BP rispondeva al 100% e misurava
un'altra cosa. Un tenor ILS puo' essere quotato ogni giorno e avere un LIVELLO sbagliato o
fermo, e per una proiezione a venticinque anni quello e' esattamente cio' che conta.

E c'e' un motivo per riaprirla. Se l'inflazione proiettata e' troppo BASSA, i flussi
sintetici del linker sono troppo piccoli, e lo spread che li riporta al prezzo osservato
scende: z(lnk) troppo basso, base negativa. Il 2009-2010 e' il periodo in cui le
aspettative di inflazione a lungo termine si sono dislocate di piu'. L'errore cresce con
l'orizzonte, quindi colpisce il titolo piu' lungo -- che e' proprio quello che abbiamo
trovato.

LA DOMANDA CHE SEPARA LE DUE IPOTESI. Sono due storie diverse e fanno previsioni diverse:

  B (interpolazione fra gemelli distanti): riguarda QUEL titolo, perche' e' l'unico con un
    bracket di 915 giorni. Gli altri linker, che hanno gemelli vicini, non devono mostrare
    niente. La base contro la scadenza deve essere PIATTA una volta tolto il 2035.

  A' (proiezione troppo bassa sul lungo): e' un effetto di ORIZZONTE, non di titolo.
    Allora la base deve calare con la vita residua su TUTTI i linker nel 2009-2010, e il
    2035 e' solo il punto estremo della stessa retta. Togliendolo, la pendenza resta.

Una guarda il titolo, l'altra guarda la scadenza, e il dato le distingue senza aggiungere
niente. Piu' un anno di controllo in cui non deve succedere nulla.
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE, MARKETS

# ----------------------------------------------------------------- impostazioni
MERCATO   = "IT"
MISURA    = "interp"
MIN_VITA  = 365
SOSPETTO  = "IT0003745541"
CALDI     = [2009, 2010, 2011]     # gli anni dell'anomalia
CONTROLLO = [2014, 2016, 2017]     # anni calmi: qui non deve esserci pendenza

# ----------------------------------------------------------------- dati
D = pd.read_parquet(CACHE / f"swapbasis_{MERCATO}.parquet")
D["date"] = pd.to_datetime(D["date"])
D["anno"] = D["date"].dt.year
D["anni"] = D["ttm"] / 365.25
D = D[D[MISURA].notna() & (D["ttm"] > MIN_VITA)].copy()
D["z_nom"] = D["z_lnk"] - D[MISURA]
m = MARKETS[MERCATO]
ils = bbg.load(f"ils_{m.ils}"); ils.index = pd.to_datetime(ils.index)
ils.columns = [float(c) for c in ils.columns]

print(f"=== {MERCATO}: e' il titolo o e' la scadenza? ===")

# --- 1. l'ILS lungo: i LIVELLI, non la copertura ----------------------------------
print("\n--- 1. la curva ILS sul lungo: livelli e mobilita' ---")
print("    Il 26 guardava quante celle sono piene. Qui si guarda cosa c'e' dentro: un")
print("    tenor quotato ogni giorno ma FERMO, o su un livello implausibile, rovina la")
print("    proiezione esattamente come un tenor mancante -- e non si vede dalla copertura.\n")
mostra = [t for t in (5.0, 10.0, 20.0, 25.0, 30.0) if t in ils.columns]
print(f"    {'anno':<7}" + "".join(f"{f'{t:g}a':>9}" for t in mostra)
      + f"{'30a-10a':>10}{'  gg fermi 30a':>16}")
for y, g in ils.groupby(ils.index.year):
    if y < 2006 or y > 2015:
        continue
    liv = "".join(f"{g[t].median():>9.2f}" if t in g and g[t].notna().any()
                  else f"{'-':>9}" for t in mostra)
    sl = (g[30.0].median() - g[10.0].median()) if (30.0 in g and 10.0 in g) else np.nan
    d30 = g[30.0].dropna().diff().dropna() if 30.0 in g else pd.Series(dtype=float)
    fermi = f"{int((d30 == 0).sum())}/{len(d30)}" if len(d30) else "-"
    print(f"    {y:<7}{liv}{sl:>10.2f}{fermi:>16}")
print("\n    Uno swap di inflazione EUR a 30 anni sta fra il 2% e il 2.8% in condizioni")
print("    normali, sopra il 10a di qualche decina di bp. Se nel 2009-2010 il 30a e'")
print("    SOTTO il 10a, o si muove pochissimo, la proiezione lunga e' sospetta.")

# --- 2. la base contro la SCADENZA, anno per anno ---------------------------------
print("\n--- 2. la base contro la vita residua ---")
print("    Se e' un effetto di orizzonte, la base cala con la scadenza su tutti i")
print("    titoli. Se e' il 2035, le altre fasce stanno ferme.\n")
bins = [1, 3, 5, 8, 12, 18, 40]
et = ["1-3a", "3-5a", "5-8a", "8-12a", "12-18a", "18a+"]
D["fascia"] = pd.cut(D["anni"], bins=bins, labels=et, right=False)


MIN_TITOLI = 4      # sotto, una pendenza non significa niente


def pendenza(sub, y):
    """bp di base per ANNO di vita residua, un punto per TITOLO.

    Non si differenziano le mediane delle fasce: in quegli anni la fascia lunga contiene
    due o tre titoli, e la sua mediana non si muove nemmeno se uno e' completamente rotto
    -- il test non vedrebbe proprio il caso che deve distinguere. E non si regredisce sulle
    osservazioni: un titolo con 258 giorni peserebbe 258 volte uno con 20, e la pendenza
    diventerebbe una proprieta' di chi ha piu' giorni. Un punto per titolo, mediana sua."""
    g = sub[sub["anno"] == y]
    if not len(g):
        return np.nan, 0
    p = g.groupby("isin").agg(b=(MISURA, "median"), a=("anni", "median"))
    if len(p) < MIN_TITOLI or p["a"].std() < 1e-9:
        return np.nan, len(p)
    return float(np.polyfit(p["a"], p["b"], 1)[0]), len(p)


def tavola(sub, titolo):
    print(f"    {titolo}")
    print(f"    {'anno':<7}" + "".join(f"{f:>11}" for f in et)
          + f"{'titoli':>8}{'bp/anno':>10}")
    out = {}
    for y in CALDI + CONTROLLO:
        g = sub[sub["anno"] == y]
        if not len(g):
            continue
        med = g.groupby("fascia", observed=False)[MISURA].median()
        cel = "".join(f"{med.get(f):>11.1f}" if pd.notna(med.get(f)) else f"{'-':>11}"
                      for f in et)
        pen, nt = pendenza(sub, y)
        out[y] = pen
        pn = f"{pen:>10.2f}" if pd.notna(pen) else f"{'-':>10}"
        print(f"    {y:<7}{cel}{nt:>8}{pn}")
    return out


pen_tutti = tavola(D, "TUTTI i linker")
print()
pen_senza = tavola(D[D["isin"] != SOSPETTO], f"SENZA {SOSPETTO}")

print(f"\n    {'anno':<9}{'bp/anno con':>14}{'bp/anno senza':>16}{'quanta ne resta':>18}")
for y in CALDI + CONTROLLO:
    a, b = pen_tutti.get(y, np.nan), pen_senza.get(y, np.nan)
    q = f"{b/a:.0%}" if (pd.notna(a) and pd.notna(b) and abs(a) > 1e-9) else "-"
    sa = f"{a:>14.2f}" if pd.notna(a) else f"{'-':>14}"
    sb = f"{b:>16.2f}" if pd.notna(b) else f"{'-':>16}"
    print(f"    {y:<9}{sa}{sb}{q:>18}")

# --- 3. il verdetto, letto dai numeri ---------------------------------------------
caldi_ok = [y for y in CALDI if pd.notna(pen_tutti.get(y, np.nan))]
ctrl_ok = [y for y in CONTROLLO if pd.notna(pen_tutti.get(y, np.nan))]
if caldi_ok and ctrl_ok:
    pc = np.median([pen_tutti[y] for y in caldi_ok])
    pk = np.median([pen_tutti[y] for y in ctrl_ok])
    ps = np.median([pen_senza[y] for y in caldi_ok if y in pen_senza])
    print(f"\n--- verdetto ---")
    print(f"    pendenza mediana negli anni caldi {pc:+.2f} bp/anno, di controllo "
          f"{pk:+.2f} bp/anno")
    print(f"    negli anni caldi SENZA il sospetto: {ps:+.2f} bp/anno")
    print(f"    (su 25 anni di scadenza {pc:+.2f} bp/anno fanno {pc*25:+.0f} bp di base)")
    # Soglia in bp per ANNO di vita residua: mezzo bp all'anno su venticinque anni fa
    # dodici bp di base fra il linker piu' corto e il piu' lungo, ed e' il minimo che
    # valga la pena chiamare pendenza.
    SOGLIA = -0.5
    resta = abs(ps) > 0.4 * abs(pc) if abs(pc) > 1e-9 else False
    if pc < SOGLIA and resta:
        print("\n    E' UN EFFETTO DI ORIZZONTE (A'). La base cala con la scadenza anche")
        print("    togliendo il sospetto: allora non e' quel titolo, e' la proiezione")
        print("    dell'inflazione sul lungo in quegli anni. Si guarda il punto 1: se il")
        print("    30a ILS e' fermo o sotto il 10a, e' li'. Il rimedio NON e' escludere un")
        print("    titolo ma dichiarare il limite, o accorciare il campione sul lungo con")
        print("    una regola scritta sulla scadenza -- esogena.")
    elif pc < SOGLIA and not resta:
        print("\n    E' IL TITOLO (B). Tolto il sospetto la pendenza sparisce: gli altri")
        print("    linker, che hanno gemelli vicini, non mostrano nulla. Resta")
        print("    l'interpolazione su 915 giorni di bracket. Si stringe MAX_BRACKET e si")
        print("    rifa il 24: se la sua base sale verso il gruppo, l'ipotesi e' confermata")
        print("    da un intervento, non da una correlazione.")
    else:
        print("\n    Nessuna pendenza negli anni caldi: il sospetto non e' il punto estremo")
        print("    di un effetto di scadenza, e' un CASO ISOLATO. Allora A' cade e resta B,")
        print("    che il 26 aveva gia' indicato: bracket 915 giorni contro 365 delle sane,")
        print("    corr(bracket, base) -0.372. Il modo di confermarlo e' un intervento, non")
        print("    un'altra correlazione: si stringe MAX_BRACKET nel 24 e si rifa il run. Se")
        print("    la sua base risale verso il gruppo, era l'interpolazione; se non si muove,")
        print("    allora e' il prezzo di quel titolo e va guardato a terminale.")
    if abs(pk) > 0.5 * abs(pc):
        print(f"\n    !! ATTENZIONE: la pendenza c'e' anche negli anni di controllo")
        print(f"       ({pk:+.1f} contro {pc:+.1f}). Allora non e' un'anomalia di periodo ma")
        print("       una proprieta' permanente della misura sul lungo, e va trattata come")
        print("       tale: un controllo per scadenza nelle regressioni, non un filtro.")

# --- 4. quanto pesa davvero, comunque vada ----------------------------------------
print(f"\n--- 4. quanto cambia la serie se il sospetto esce ---")
print(f"    {'anno':<7}{'con':>9}{'senza':>9}{'diff':>8}{'n con':>8}{'n senza':>9}")
for y, g in D.groupby("anno"):
    s = g[g["isin"] != SOSPETTO]
    if not len(s):
        continue
    print(f"    {y:<7}{g[MISURA].mean():>9.1f}{s[MISURA].mean():>9.1f}"
          f"{s[MISURA].mean() - g[MISURA].mean():>8.1f}{len(g):>8}{len(s):>9}")
print("\n    Se la differenza e' di pochi bp, la questione e' di igiene e non cambia il")
print("    paper: si dichiara e si va avanti. Se sposta il profilo, va risolta prima.")

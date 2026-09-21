"""23 - CURVA SWAP EUR (ICVS 45, EUR vs 6M Euribor): lo scarico degli strumenti VERI.

RICHIEDE IL TERMINALE. Costa pochissimo: una trentina di ticker per vent'anni, contro i 145
titoli dello scarico dei bond. Minuti, non ore.

PERCHE'. basis.zspread(cf, asof, zero_row) risolve gia' per lo shift parallelo che riprezza
i flussi al prezzo osservato: gli si passa una riga di curva zero e restituisce lo Z-spread.
Finora gli abbiamo passato la curva sovrana FITTATA sui nominali. Funziona, ma ha un difetto
strutturale: quella curva e' costruita dai nominali stessi, quindi lo Z-spread del nominale
contro di essa e' meccanicamente vicino a zero e assorbe l'errore di fit. La differenza fra
le gambe eredita quell'errore. Una curva swap e' ESOGENA a entrambe le gambe.

QUALI STRUMENTI, E PERCHE' NON QUELLI DI PRIMA. Sullo schermo ICVS 45 sono ACCESI (bianchi)
un solo deposito -- il 6 MO -- otto FRA seriali, e gli swap annuali. E' una curva
MONO-INDICE: tutto e' ancorato all'Euribor 6M. Il deposito 6M fa il primo nodo, la striscia
FRA porta i forward fino a 18 mesi, gli swap prendono da li' in poi. Il mio scarico
precedente prendeva 1M/3M/6M e nessun FRA: mescolava indici diversi e ricostruiva il tratto
0-2 anni con gli strumenti sbagliati -- proprio il tratto che sconta i titoli corti del
campione. Non era incompleto, era sbagliato in modo preciso.

LA VERIFICA E' PER ANCORE, NON PER COPERTURA. La lezione della giornata: INDEX_Z_SPREAD_BP
rispondeva al 100% e misurava un'altra cosa. La copertura non dimostra nulla. Qui ogni
ticker viene confrontato con il numero letto sullo schermo ICVS di oggi: se risponde ma con
un livello diverso, e' un altro strumento e va scartato PRIMA di costruirci sopra. Lo
scarto si misura in bp e si legge in scala: pochi bp sono close-vs-intraday, decine o
centinaia sono un ticker sbagliato.

COSA SALVA. Due file. Il pannello dei tassi grezzi (colonne = TICKER, non tenor: un FRA
1x7 non e' un tasso a 7 mesi e chiamarlo cosi' sarebbe gia' un errore di bootstrap) e una
mappa che dice per ogni ticker che STRUMENTO e', con inizio e fine in anni. Il bootstrap
sta nel 24 e legge la mappa: qui non si interpreta nulla, si scarica e si verifica.
"""
import numpy as np
import pandas as pd
import bbg
from config import CACHE

# ----------------------------------------------------------------- impostazioni
# Deposito: UNO solo, il 6M. E' il nodo iniziale dell'unica curva forward della I45.
CASH = {"EUR006M Index": 0.50}          # ticker -> tenor in anni

FRA = {                                 # ticker -> (inizio, fine) in MESI
    "EUFR0AG Curncy":  (1, 7),
    "EUFR0BH Curncy":  (2, 8),
    "EUFR0CI Curncy":  (3, 9),
    "EUFR0DJ Curncy":  (4, 10),
    "EUFR0EK Curncy":  (5, 11),
    "EUFR0F1 Curncy":  (6, 12),
    "EUFR0I1C Curncy": (9, 15),
    "EUFR011F Curncy": (12, 18),
}

TENOR = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 25, 30, 35, 40, 45, 50]
FORME = ["EUSA{n} Curncy", "EUSA{n} BGN Curncy", "EUSA{n} ICPL Curncy"]
# 35-50 restano nello scarico anche se sullo schermo sembrano spenti: nel campione c'e' un
# BTPei 2051, e senza il lungo la sua curva sarebbe estrapolata. Costano nulla e si possono
# sempre escludere dal fit; non averli costringerebbe a riscaricare.

# ANCORE dallo schermo ICVS del 2026-09-21 (bid, ask). Servono a verificare che il ticker
# risponda CON IL NUMERO GIUSTO: la copertura non basta.
ANCORE = {
    "EUR006M Index":   (2.97800, 2.97800),
    "EUFR0AG Curncy":  (3.07593, 3.09607),
    "EUFR0BH Curncy":  (3.18925, 3.20875),
    "EUFR0CI Curncy":  (3.30718, 3.32282),
    "EUFR0DJ Curncy":  (3.38593, 3.40407),
    "EUFR0EK Curncy":  (3.45888, 3.47912),
    "EUFR0F1 Curncy":  (3.51898, 3.53902),
    "EUFR0I1C Curncy": (3.59880, 3.61920),
    "EUFR011F Curncy": (3.59637, 3.61563),
}
ANCORE_SWAP = {2: (3.48668, 3.49432), 5: (3.48625, 3.49115), 10: (3.50975, 3.51425),
               30: (3.38630, 3.39570), 50: (3.02559, 3.03841)}
# Scala di lettura dello scarto, in bp. Sotto OK_BP e' lo scarto fra un close e uno schermo
# intraday: normale. Sopra MALE_BP non e' un movimento di giornata, e' un altro strumento.
OK_BP, MALE_BP = 5.0, 25.0

NOME = "swap_EUR"

# ----------------------------------------------------------------- esecuzione
try:
    from xbbg import blp  # noqa: F401
except ImportError:
    raise SystemExit("xbbg non disponibile: questo script va lanciato sul terminale.")

oggi = pd.Timestamp.today().normalize()
d0 = bbg.PULL_FLOOR

print("=== curva swap EUR, costruzione ICVS 45 (EUR vs 6M Euribor) ===")
print(f"    finestra {d0:%Y-%m-%d} -> {oggi:%Y-%m-%d}")
print(f"    {len(CASH)} deposito + {len(FRA)} FRA seriali + {len(TENOR)} swap\n")


def _scarica(tickers: list[str], etichetta: str) -> pd.DataFrame:
    """bdh su una lista di ticker. Ritorna largo (date x ticker), o vuoto."""
    try:
        df = bbg._bdh(list(tickers), "PX_LAST", d0.date(), oggi.date())
    except Exception as e:
        print(f"    {etichetta}: errore {str(e)[:60]}")
        return pd.DataFrame()
    if df is None or not len(df):
        print(f"    {etichetta}: risposta vuota")
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c) for c in df.columns]
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _ancora(serie: pd.Series, forchetta) -> tuple:
    """(data, valore, scarto_bp, simbolo). Confronta l'ULTIMO valore con la forchetta
    dello schermo. Lo scarto e' zero dentro la forchetta, altrimenti la distanza dal
    bordo piu' vicino, in bp."""
    s = pd.to_numeric(serie, errors="coerce").dropna()
    if not len(s):
        return (None, np.nan, np.inf, "  --")
    d, v = s.index[-1], float(s.iloc[-1])
    lo, hi = float(min(forchetta)), float(max(forchetta))
    scarto = 0.0 if lo <= v <= hi else (lo - v if v < lo else v - hi) * 100.0
    sym = "  ok" if scarto <= OK_BP else ("   ~" if scarto <= MALE_BP else "  !!")
    return (d, v, scarto, sym)


# --- 1. gli swap: quale forma di ticker, decisa PRIMA sulle ancore -----------------
print("--- 1. swap: quale forma di ticker ---")
print("    Si sceglie per ANCORE, non per copertura: una forma che risponde a tutti i")
print("    tenor ma con livelli diversi dallo schermo non e' la curva della I45.\n")
print(f"    {'forma':<24}{'tenor':>7}{'date':>7}{'storia':>18}{'ancore':>10}{'peggior scarto':>17}")

cand = []
for forma in FORME:
    t2c = {forma.format(n=n): float(n) for n in TENOR}
    df = _scarica(list(t2c), forma)
    if not len(df):
        continue
    df.columns = [t2c.get(c, c) for c in df.columns]
    peggio, nok, ntot = 0.0, 0, 0
    for n, fk in ANCORE_SWAP.items():
        if float(n) not in df.columns:
            continue
        ntot += 1
        _, _, sc, _ = _ancora(df[float(n)], fk)
        if np.isfinite(sc):
            peggio = max(peggio, sc)
            nok += int(sc <= MALE_BP)
    print(f"    {forma:<24}{df.shape[1]:>7}{len(df):>7}"
          f"{f'{df.index.min():%Y-%m}/{df.index.max():%Y-%m}':>18}"
          f"{f'{nok}/{ntot}':>10}{peggio:>14.1f} bp")
    cand.append({"forma": forma, "df": df, "peggio": peggio,
                 "buone": nok, "tot": ntot, "oss": int(df.notna().sum().sum())})

if not cand:
    raise SystemExit("\nNessuna forma risponde: controlla i ticker EUSA su SWPM/FLDS.")

# valide = quelle in cui NESSUNA ancora sbanda. Fra le valide vince la storia.
valide = [c for c in cand if c["tot"] and c["buone"] == c["tot"]]
if valide:
    scelta = max(valide, key=lambda c: c["oss"])
    print(f"\n    scelta: {scelta['forma']}  (ancore tutte dentro, {scelta['oss']} osservazioni)")
else:
    scelta = min(cand, key=lambda c: c["peggio"])
    print(f"\n    !! NESSUNA forma supera le ancore. La migliore e' {scelta['forma']}, che")
    print(f"       comunque sbanda di {scelta['peggio']:.1f} bp. Prima di usarla: apri ICVS 45,")
    print( "       leggi i mid degli swap e aggiorna ANCORE_SWAP -- oppure il ticker e' un altro.")
    print( "       Si prosegue lo scarico, ma il 24 NON va lanciato su questo file finche'")
    print( "       la discrepanza non e' spiegata.")
W_sw = scelta["df"]

# --- 2. il breve: deposito 6M e striscia FRA ---------------------------------------
print("\n--- 2. deposito 6M e striscia FRA seriale ---")
brevi = list(CASH) + list(FRA)
W_br = _scarica(brevi, "breve")
mancanti = [t for t in brevi if t not in W_br.columns]
if mancanti:
    print(f"    !! non rispondono: {', '.join(mancanti)}")

print(f"\n    {'ticker':<20}{'strumento':<12}{'date':>7}{'storia':>18}"
      f"{'ultimo':>10}{'schermo':>18}{'scarto':>11}")
riassunto = []
for tk in brevi:
    if tk not in W_br.columns:
        riassunto.append((tk, np.inf))
        continue
    s = pd.to_numeric(W_br[tk], errors="coerce").dropna()
    strum = f"{CASH[tk]:g}a depo" if tk in CASH else "{}x{} FRA".format(*FRA[tk])
    d, v, sc, sym = _ancora(W_br[tk], ANCORE.get(tk, (-np.inf, np.inf)))
    lo, hi = ANCORE.get(tk, (np.nan, np.nan))
    forc = f"{lo:.4f}/{hi:.4f}" if np.isfinite(lo) else "-"
    print(f"    {tk:<20}{strum:<12}{len(s):>7}"
          f"{f'{s.index.min():%Y-%m}/{s.index.max():%Y-%m}' if len(s) else '-':>18}"
          f"{v:>10.4f}{forc:>18}{sc:>8.1f} bp{sym}")
    riassunto.append((tk, sc))

fuori = [t for t, sc in riassunto if not (sc <= MALE_BP)]
if fuori:
    print(f"\n    !! {len(fuori)} ticker brevi non riconciliano con lo schermo:")
    for t in fuori:
        print(f"       {t}")
    print("       Uno scarto di pochi bp e' close contro intraday. Decine o centinaia no:")
    print("       quello e' un altro strumento, e va risolto prima del bootstrap.")
else:
    print("\n    Tutti i brevi riconciliano con lo schermo entro la tolleranza di giornata.")

# --- 3. FIN DOVE INDIETRO: e' questo che decide la strategia ------------------------
print("\n--- 3. fin dove arriva indietro ciascun pezzo ---")
print("    La domanda vera: possiamo replicare la I45 su TUTTO il campione 2004-2026, o")
print("    la striscia FRA comincia tardi e per gli anni iniziali serve una costruzione")
print("    diversa? Una curva che cambia costruzione a meta' campione introduce uno")
print("    scalino nella base che non e' economia: meglio saperlo adesso.\n")

tutto = W_sw.copy()
tutto.columns = [f"EUSA{c:g}" for c in tutto.columns]
tutto = tutto.join(W_br, how="outer").sort_index()

g_depo = [t for t in CASH if t in tutto.columns]
g_fra = [t for t in FRA if t in tutto.columns]
g_sw = [c for c in tutto.columns if c.startswith("EUSA")]

print(f"    {'anno':<7}{'depo 6M':>9}{'FRA':>9}{'swap':>7}{'  6M':>9}{'2a':>8}{'10a':>8}{'30a':>8}")
for y, gg in tutto.groupby(tutto.index.year):
    n_d = int(gg[g_depo].notna().any(axis=1).sum()) if g_depo else 0
    n_f = int(gg[g_fra].notna().sum(axis=1).max()) if g_fra else 0
    n_s = int(gg[g_sw].notna().sum(axis=1).max()) if g_sw else 0
    def _m(c):
        return f"{gg[c].median():>8.2f}" if c in gg.columns and gg[c].notna().any() else f"{'-':>8}"
    d6 = _m(g_depo[0]) if g_depo else f"{'-':>8}"
    print(f"    {y:<7}{n_d:>9}{f'{n_f}/{len(FRA)}':>9}{f'{n_s}/{len(TENOR)}':>7}"
          f" {d6}{_m('EUSA2')}{_m('EUSA10')}{_m('EUSA30')}")

if g_fra:
    piena = tutto[g_fra].notna().all(axis=1)
    if piena.any():
        print(f"\n    striscia FRA COMPLETA ({len(g_fra)}/{len(FRA)}) dal {tutto.index[piena][0]:%Y-%m-%d}")
    else:
        print(f"\n    la striscia FRA non e' MAI completa nello stesso giorno.")
    parz = tutto[g_fra].notna().any(axis=1)
    print(f"    almeno un FRA quotato dal {tutto.index[parz][0]:%Y-%m-%d}" if parz.any()
          else "    nessun FRA quotato in nessuna data")
    for t in g_fra:
        s = tutto[t].dropna()
        if len(s):
            print(f"      {t:<20} dal {s.index[0]:%Y-%m-%d}  ({len(s)} date)")

print("\n    L'euro a 10 anni: ~4% nel 2007-2008, sotto l'1% nel 2015-2021, ~3% nel 2023.")
print("    Se i numeri non somigliano a questo, il ticker e' un'altra cosa: fermarsi qui.")

# --- 4. plausibilita' e continuita' -------------------------------------------------
print("\n--- 4. plausibilita' e buchi ---")
fuori_r = ((tutto < -2.0) | (tutto > 15.0)) & tutto.notna()
n_f = int(fuori_r.sum().sum())
print(f"    valori fuori da [-2%, 15%]: {n_f} su {int(tutto.notna().sum().sum())}")
for c in fuori_r.columns[fuori_r.any()]:
    est = tutto[c][fuori_r[c]]
    print(f"      {c:<20} {len(est)} valori, da {est.min():.2f} a {est.max():.2f}")

n_t = tutto.notna().sum(axis=1)
print(f"    {len(tutto)} date, {int((n_t == 0).sum())} completamente vuote, "
      f"mediana {int(n_t.median())} strumenti per data")
scarsi = tutto.notna().sum().sort_values()
print("    con meno storia: " + ", ".join(f"{c} ({int(v)})" for c, v in scarsi.head(5).items()))

# --- 5. salvataggio: pannello + mappa degli strumenti -------------------------------
# Le colonne restano TICKER. Un FRA 1x7 non e' un tasso a 7 mesi: dargli un tenor qui
# vorrebbe dire aver gia' fatto (male) il bootstrap. La mappa dice cosa e' ciascuno.
meta = []
for tk in g_depo:
    meta.append({"ticker": tk, "tipo": "depo", "t0": 0.0, "t1": CASH[tk]})
for tk in g_fra:
    m0, m1 = FRA[tk]
    meta.append({"ticker": tk, "tipo": "fra", "t0": m0 / 12.0, "t1": m1 / 12.0})
for c in g_sw:
    meta.append({"ticker": c, "tipo": "swap", "t0": 0.0, "t1": float(c.replace("EUSA", ""))})
M = pd.DataFrame(meta).set_index("ticker")
# Ordine per (tipo, t1). bdh restituisce le colonne in ordine ALFABETICO di ticker, che
# come sequenza di tenor fa 10, 11, 12, 15, 2, 20, ... Chi legge il file poi lo usa come
# ascissa di un np.interp e ottiene numeri senza senso senza nessun errore: e' gia'
# successo, ed e' costato un run. Il file esce ordinato, e a valle si riordina comunque.
M = M.sort_values(["tipo", "t1"], key=lambda c: c.map({"depo": 0, "fra": 1, "swap": 2})
                  if c.name == "tipo" else c)
tutto = tutto[[c for c in M.index if c in tutto.columns]]

p_dati = CACHE / f"{NOME}.parquet"
p_meta = CACHE / f"{NOME}_meta.parquet"
tutto.sort_index().to_parquet(p_dati)
M.to_parquet(p_meta)
print(f"\nsalvato: {p_dati}")
print(f"         {p_meta}")
print("    pannello: indice = data, colonne = TICKER, valori = tasso in PERCENTO.")
print("    mappa   : per ogni ticker -> tipo (depo/fra/swap) e (t0, t1) in ANNI.")
print("    Il bootstrap e' nel 24: depo semplice ACT/360 fino a t1, FRA a comporre i")
print("    forward 6M fino a 18 mesi, swap par annuali da li' in poi.")

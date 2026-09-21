"""34 - COSA MANCA PER ES, FR, DE. Offline, dieci secondi, nessun terminale.

PERCHE' PRIMA DI TUTTO IL RESTO. L'Italia e' chiusa: tre misure che convergono a 2.6 bp,
validazione contro Bloomberg dal 2012 a -0.18 bp sulla media con correlazione +0.747, e
sul BTPei 2035 gli z(linker) delle due macchine coincidono a 1-9 bp con correlazione
1.00. Il passo successivo sono gli altri tre mercati del capitolo.

Ma "lancia il 24 con MERCATO = FR" puo' fallire in quattro modi diversi, e tre si
scoprono solo a meta' di un run da venti minuti: manca l'anagrafica, mancano i prezzi dei
linker, mancano i prezzi dei nominali, manca la curva ILS di quel mercato. Il quarto --
il mercato non e' nemmeno in config -- fallisce subito ma con un KeyError che non spiega
niente.

Questo script guarda la cache e dice, mercato per mercato, che cosa c'e' e che cosa no,
con le finestre temporali. Costa dieci secondi e decide l'ordine del lavoro: si comincia
dal mercato che e' gia' pronto, non da quello che viene primo in ordine alfabetico.

UNA NOTA SULLA SPAGNA. bbg.py mappa gia' SPGBEI/1573 -> ES e sa che e' EUR con calendario
spagnolo, ma il commento dice "fuori perimetro: escluso finche' ES non entra in
config.MARKETS". Quindi per la Spagna non e' questione di scaricare: e' che il mercato
non esiste ancora per la pipeline, e va aggiunto prima. Lo script lo dice invece di
lasciarlo scoprire a un KeyError.

E UNA SULLA FRANCIA. I linker francesi sono DUE famiglie: OATei indicizzati all'HICP
(mercato FR) e OATi indicizzati al CPI francese (FR_CPI, curva ILS FRSWI, lag e
convenzioni proprie). Sono due mercati per la pipeline anche se un solo emittente, e il
capitolo deve decidere se tenerli insieme o separati -- ma la decisione si prende sapendo
quanti titoli ci sono per parte, che e' quello che questo script stampa.
"""
import pandas as pd
import bbg
from config import CACHE, MARKETS

MERCATI = ["IT", "FR", "FR_CPI", "DE", "ES"]
OBBLIGATORI = ("ref_linker", "ref_nominal", "px_mid", "px_nom", "ytm", "ils", "cpi")

print("=== pronti per quali mercati? ===\n")


def _finestra(nome):
    p = CACHE / f"{nome}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        return ("errore", str(e)[:30], 0, 0)
    if isinstance(df.index, pd.DatetimeIndex) or "date" in str(type(df.index)).lower():
        try:
            idx = pd.to_datetime(df.index)
            viva = idx[df.notna().any(axis=1)] if df.shape[1] else idx
            if len(viva):
                return (f"{viva.min():%Y-%m}", f"{viva.max():%Y-%m}", len(df), df.shape[1])
        except Exception:
            pass
    return ("-", "-", len(df), df.shape[1])


try:
    ref_l = bbg.load("ref_linker")
    ref_n = bbg.load("ref_nominal")
except Exception as e:
    raise SystemExit(f"anagrafica non leggibile ({e}): lancia prima 01 e 02.")

print(f"{'mercato':<9}{'in config':<11}{'linker':>8}{'nominali':>10}   "
      f"{'file mancanti':<40}")
stato = {}
for m in MERCATI:
    inc = m in MARKETS
    n_l = int((ref_l["mkt"] == m).sum()) if "mkt" in ref_l.columns else 0
    nom_m = bbg.NOMINAL_POOL_ALIAS.get(m, m)
    n_n = int((ref_n["mkt"] == nom_m).sum()) if "mkt" in ref_n.columns else 0
    manca = []
    if inc:
        mk = MARKETS[m]
        attesi = {"px_mid": f"px_mid_{m}", "px_nom": f"px_nom_{nom_m}",
                  "ytm": f"ytm_{nom_m}", "ils": f"ils_{mk.ils}", "cpi": f"cpi_{mk.cpi}"}
        for et, nome in attesi.items():
            if not (CACHE / f"{nome}.parquet").exists():
                manca.append(nome)
    else:
        manca = ["(non in config.MARKETS)"]
    stato[m] = (inc, n_l, n_n, manca)
    print(f"{m:<9}{'si' if inc else 'NO':<11}{n_l:>8}{n_n:>10}   "
          f"{', '.join(manca)[:40] if manca else 'tutto presente':<40}")

# --- le finestre, per i mercati pronti --------------------------------------------
print(f"\n--- finestre temporali dei file, per mercato ---")
for m in MERCATI:
    inc, n_l, n_n, manca = stato[m]
    if not inc or manca:
        continue
    mk = MARKETS[m]
    nom_m = bbg.NOMINAL_POOL_ALIAS.get(m, m)
    print(f"\n  {m}")
    for et, nome in [("prezzi linker", f"px_mid_{m}"), ("prezzi nominali", f"px_nom_{nom_m}"),
                     ("ytm nominali", f"ytm_{nom_m}"), ("curva ILS", f"ils_{mk.ils}"),
                     ("indice CPI", f"cpi_{mk.cpi}")]:
        f = _finestra(nome)
        if f is None:
            print(f"    {et:<18}{'MANCA':<12}")
        else:
            print(f"    {et:<18}{f[0]:<9}-> {f[1]:<9}{f[2]:>7} date{f[3]:>6} colonne")

# --- la curva swap e' comune -------------------------------------------------------
print(f"\n--- comune a tutti i mercati euro ---")
for nome in ("swap_EUR", "swap_EUR_meta"):
    f = _finestra(nome)
    print(f"    {nome:<18}" + ("MANCA" if f is None else
          f"{f[0]:<9}-> {f[1]:<9}{f[2]:>7} righe{f[3]:>6} colonne"))

# --- cosa fare, in ordine ----------------------------------------------------------
print(f"\n--- l'ordine di lavoro ---")
pronti = [m for m in MERCATI if stato[m][0] and not stato[m][3] and stato[m][1] > 0]
mancanti = [m for m in MERCATI if stato[m][0] and stato[m][3]]
fuori = [m for m in MERCATI if not stato[m][0]]
if pronti:
    print(f"    PRONTI subito ({', '.join(pronti)}): basta cambiare MERCATO in testa al")
    print(f"    24 e rilanciare la catena 24 -> 25 -> 33. Nessuno scarico.")
if mancanti:
    print(f"    DA SCARICARE ({', '.join(mancanti)}): mancano i file elencati sopra. Si")
    print(f"    usa il 02 con SOLO_MERCATO impostato, e le FASI che servono.")
if fuori:
    print(f"    DA CONFIGURARE ({', '.join(fuori)}): il mercato non e' in config.MARKETS,")
    print(f"    quindi l'universo lo esclude a monte. Servono, nell'ordine: la voce in")
    print(f"    MARKETS (ils, cpi, calendario, settle_days), poi 01 per rifare l'universo,")
    print(f"    poi 02 per scaricare. Lo scarico e' l'ultimo passo, non il primo.")
print(f"\n    Si parte dal mercato gia' pronto: ogni mercato chiuso e' una riga in piu'")
print(f"    nella tabella del capitolo, e la Spagna non diventa piu' facile aspettando.")

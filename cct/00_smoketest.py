"""
00 - Diagnostica Bloomberg. Da lanciare UNA VOLTA prima di 02, e ogni volta che 02 fallisce.

Procede a gradini e SI FERMA AL PRIMO CHE FALLISCE, stampando l'errore vero invece di
inghiottirlo: se la connessione non c'e', provare 36 combinazioni di ticker e' inutile.

  gradino 1  import di xbbg
  gradino 2  connessione: un ticker banale e sempre disponibile (EUR006M Index)
  gradino 3  formato ISIN: quattro sintassi diverse su UN solo titolo, con errore esploso
  gradino 4  suffisso di pricing source, solo se il gradino 3 ha trovato una sintassi buona
  gradino 5  campo migliore per i BOT (prezzo vs rendimento)
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

L = []; P = L.append
P("=== 00 DIAGNOSTICA BLOOMBERG ===")

# --- gradino 1: import -------------------------------------------------------
try:
    from xbbg import blp
    import xbbg, bbg
    P(f"[1/5 OK] xbbg importato (versione {getattr(xbbg,'__version__','n/d')})")
except Exception as e:
    P(f"[1/5 STOP] xbbg non importabile: {e}")
    P("  pip install xbbg")
    P("  pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple blpapi")
    save_txt("00_smoketest.txt", L); print("\n".join(L)); raise SystemExit

def try_bdh(tk, fld="PX_LAST", a="2024-01-02", b="2024-01-31"):
    """Ritorna (n_osservazioni, errore|None). Usa bbg.bdh: gestisce il formato narwhals
    LUNGO che la versione installata restituisce -- leggere .iloc[:,0] su quell'oggetto
    darebbe la colonna dei TICKER invece dei valori, in silenzio."""
    try:
        d = bbg.bdh([tk], fld, a, b, verbose=False)
        if d is None or d.empty: return 0, "risposta vuota"
        return int(d.iloc[:, 0].notna().sum()), None
    except Exception as e:
        return -1, f"{type(e).__name__}: {str(e)[:120]}"

# --- gradino 2: connessione --------------------------------------------------
n, err = try_bdh("EUR006M Index")
if n <= 0:
    P(f"[2/5 STOP] connessione fallita su 'EUR006M Index': {err or 'vuoto'}")
    P("")
    P("  La causa quasi certa e' una di queste tre, in ordine di frequenza:")
    P("   a) il TERMINALE BLOOMBERG non e' aperto e loggato sulla STESSA macchina.")
    P("      blpapi parla con il processo locale (porta 8194): senza terminale attivo")
    P("      ogni richiesta fallisce con 'request failed'. Aprire il terminale, fare")
    P("      login completo, lasciarlo aperto, e rilanciare.")
    P("   b) sessione Bloomberg attiva su un'ALTRA macchina: il login e' singolo.")
    P("      Fare logout dall'altra postazione.")
    P("   c) la tua xbbg ha un core Rust (pyo3-xbbg) che gestisce da se' la connessione:")
    P("      blpapi NON serve e non e' nei requirements. Se a) e b) sono a posto e l'errore")
    P("      resta, provare  pip install \"xbbg<1.0\"  per tornare all'implementazione classica.")
    save_txt("00_smoketest.txt", L); print("\n".join(L)); raise SystemExit
P(f"[2/5 OK] connessione attiva: EUR006M Index -> {n} osservazioni")

# --- gradino 3: sintassi ISIN ------------------------------------------------
C = pd.read_csv(PROC/"static_cct.csv", parse_dates=["maturity"])
B = pd.read_csv(PROC/"static_btp.csv", parse_dates=["maturity"])
rb = B[B.maturity > pd.Timestamp("2030-01-01")].iloc[0]
rc = C[C.maturity > pd.Timestamp("2030-01-01")].iloc[0]
test_btp, test_cct = rb["bb_id"], rc["bb_id"]
P(f"   (identificatori: Bloomberg ID {test_btp} / {test_cct}, non ISIN {rb['isin']})")
P(f"\n[3/5] sintassi ISIN, titolo di prova {test_btp} (BTP a lunga, sicuramente quotato):")
SYNTAX = [("{i} Corp", "BloombergID + Corp  <-- metodo inflation_linked"),
          ("{i} Govt", "BloombergID + Govt"),
          ("/isin/{s}", "ISIN con prefisso /isin/"),
          ("{s} Corp", "ISIN + Corp")]
good = None
for tpl, lab in SYNTAX:
    n, err = try_bdh(tpl.format(i=test_btp, s=rb["isin"]), PRICE_FIELD)
    P(f"   {lab:38s} '{tpl.format(i=test_btp, s=rb['isin'])}' -> {n if n>=0 else 'ERRORE'} {err or 'oss.'}")
    if n > 0 and good is None: good = tpl
if good is None:
    P("\n[3/5 STOP] nessuna sintassi funziona con PX_MID. Prove da fare a mano sul terminale:")
    P(f"   1. digitare  {test_btp} <Corp> <GO>  e verificare che il titolo esista;")
    P( "   2. se esiste, controllare i permessi sui dati storici (spesso i govie euro")
    P( "      richiedono l'abbonamento al pricing source: provare senza suffisso);")
    P(f"   3. provare il campo PX_LAST invece di {PRICE_FIELD}.")
    n2, e2 = try_bdh(f"{test_btp} Corp", "PX_LAST")
    P(f"   [prova automatica] PX_LAST su '{test_btp} Corp' -> {n2 if n2>=0 else 'ERRORE'} {e2 or 'oss.'}")
    save_txt("00_smoketest.txt", L); print("\n".join(L)); raise SystemExit
P(f"   -> sintassi funzionante: '{good}'")

# --- gradino 4: pricing source ----------------------------------------------
P(f"\n[4/5] pricing source (su {test_btp} e {test_cct}):")
best, best_n = "", -1
for suf in ["", "@CBBT", "@BGN", "@MILA"]:
    tot = 0; msg = ""
    for i in (test_btp, test_cct):
        tk = good.format(i=i) + (suf + " Corp" if suf and good.endswith("Corp") else suf)
        tk = good.format(i=i).replace(" Corp", f"{suf} Corp") if suf else good.format(i=i)
        n, err = try_bdh(tk, PRICE_FIELD)
        tot += max(n, 0); msg = msg or (err or "")
    P(f"   {(suf or '(nessuno)'):12s} -> {tot} osservazioni totali {('| '+msg) if msg else ''}")
    if tot > best_n: best, best_n = suf, tot
P(f"   -> migliore: '{best or '(nessuno)'}'. In config: PX_SUFFIX = \"{best + ' Corp' if best else ' Corp'}\"")

# --- gradino 5: campo per i BOT ---------------------------------------------
O = pd.read_csv(PROC/"static_bot.csv", parse_dates=["maturity"])
test_bot = O[O.maturity > pd.Timestamp("2026-09-01")]["isin"].iloc[0]
P(f"\n[5/5] campo per i BOT (zero coupon), titolo {test_bot}:")
for fld in [PRICE_FIELD, "PX_LAST", YLD_FIELD_BOT]:
    tk = good.format(i=test_bot).replace(" Corp", f"{best} Corp") if best else good.format(i=test_bot)
    n, err = try_bdh(tk, fld, "2026-01-02", "2026-01-31")
    P(f"   {fld:14s} -> {n if n>=0 else 'ERRORE'} {err or 'oss.'}")

P("\nSe tutti i gradini sono OK: aggiornare PX_SUFFIX in config.py e lanciare 02_pull_prices.py.")
P("02 si ferma da solo al limite giornaliero e riprende dalla cache al rilancio successivo.")
save_txt("00_smoketest.txt", L); print("\n".join(L))

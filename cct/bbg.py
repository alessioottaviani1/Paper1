"""
bbg.py - Strato Bloomberg del package cct.

Ricalca il metodo gia' collaudato in inflation_linked/bbg.py, che risolve tre problemi
che il codice ingenuo non vede:

  1. IDENTIFICATORE. Le richieste vanno fatte con il BLOOMBERG ID + " Corp", non con
     l'ISIN. L'anagrafica BTP_CCT.xlsx ha la colonna 'Bloomberg ID' popolata per tutti
     e 1.499 i titoli: e' quello il ticker da usare.

  2. FORMATO DI RISPOSTA. La versione installata di xbbg NON restituisce un DataFrame
     pandas largo: restituisce un oggetto narwhals in formato LUNGO (ticker/date/value).
     Prendere d.iloc[:,0] su quell'oggetto legge la colonna dei ticker, non i valori --
     un errore silenzioso che produce serie di stringhe invece di prezzi. Qui la
     conversione e' esplicita: narwhals -> pandas -> pivot lungo->largo.

  3. ROBUSTEZZA. Richieste a blocchi con throttle; se un blocco torna anomalo si riprova
     PER SINGOLO TICKER, cosi' un titolo difettoso costa un titolo e non l'intero blocco.

Il limite giornaliero si manifesta come errore -4002 WORKFLOW_REVIEW_NEEDED: viene
riconosciuto e propagato, cosi' il chiamante salva e si ferma invece di ciclare a vuoto.
"""
import time
import numpy as np, pandas as pd

CHUNK_BDP, CHUNK_BDH = 40, 10     # blocchi conservativi (default libreria: 80 / 25)
THROTTLE_SEC = 0.5                # pausa fra richieste
LIMIT_TOKENS = ("4002", "workflow_review", "limit", "quota", "exceed", "daily")
# Errori PERMANENTI: il ticker non esiste. Riprovare e' inutile e costa secondi ogni volta.
DEAD_TOKENS  = ("bad_sec", "invalid security", "securityerror", "unknown/invalid", "not found")

class BbgLimitReached(Exception): pass
class BbgBadSecurity(Exception): pass

def _blp():
    """Import pigro: il modulo resta importabile senza terminale."""
    from xbbg import blp
    return blp

def _is_limit(e) -> bool:
    m = str(e).lower()
    return any(t in m for t in LIMIT_TOKENS)

def _is_dead(e) -> bool:
    """Ticker inesistente: errore permanente, nessun senso riprovare."""
    m = str(e).lower()
    return any(t in m for t in DEAD_TOKENS)

def _to_pandas(obj):
    """narwhals/altro -> DataFrame pandas. None se la risposta e' vuota."""
    if obj is None: return None
    if not isinstance(obj, pd.DataFrame):
        for attr in ("to_pandas", "to_native"):
            if hasattr(obj, attr):
                try:
                    obj = getattr(obj, attr)()
                    if isinstance(obj, pd.DataFrame): break
                except Exception: pass
    if not isinstance(obj, pd.DataFrame):
        try: obj = pd.DataFrame(obj)
        except Exception: return None
    return None if obj.empty else obj

def _wide_from_long(raw: pd.DataFrame) -> pd.DataFrame:
    """Formato lungo ticker/date/value -> largo (indice=date, colonne=ticker)."""
    cols = {str(c).lower(): c for c in raw.columns}
    if {"ticker", "date", "value"}.issubset(cols):
        t, d, v = cols["ticker"], cols["date"], cols["value"]
        long = raw[[t, d, v]].dropna(subset=[v])
        if long.empty: return pd.DataFrame()
        w = long.pivot_table(index=d, columns=t, values=v, aggfunc="first")
        w.index = pd.to_datetime(w.index); w.columns = [str(c) for c in w.columns]
        return w.sort_index()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw.index = pd.to_datetime(raw.index)
    return raw.sort_index()

def bdh(tickers, field, start, end, verbose=True) -> pd.DataFrame:
    """Storici a blocchi, con pivot e fallback per singolo ticker."""
    blp = _blp(); tickers = list(tickers); parts = []
    def _one(chunk):
        try:
            raw = _to_pandas(blp.bdh(chunk, field, start, end))
        finally:
            if THROTTLE_SEC: time.sleep(THROTTLE_SEC)
        return pd.DataFrame() if raw is None else _wide_from_long(raw)
    for i in range(0, len(tickers), CHUNK_BDH):
        ch = tickers[i:i+CHUNK_BDH]
        try:
            df = _one(ch)
            if not df.empty: parts.append(df)
            continue
        except Exception as e:
            if _is_limit(e): raise BbgLimitReached(str(e))
            if _is_dead(e) and len(ch) == 1: raise BbgBadSecurity(str(e))
            if verbose: print(f"    bdh blocco {i}-{i+len(ch)} anomalo ({str(e)[:60]}) -> per ticker")
        for t in ch:
            try:
                df = _one([t])
                if not df.empty: parts.append(df)
            except Exception as e2:
                if _is_limit(e2): raise BbgLimitReached(str(e2))
                if _is_dead(e2):
                    if verbose: print(f"      {t}: ticker inesistente, salto")
                    continue
                if verbose: print(f"      bdh fallito su {t}: {str(e2)[:60]}")
    if not parts: return pd.DataFrame()
    out = pd.concat(parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    out.index = pd.to_datetime(out.index)
    return out.sort_index()

def bdp(tickers, fields, verbose=True) -> pd.DataFrame:
    """Anagrafica a blocchi, con fallback per singolo ticker."""
    blp = _blp(); tickers = list(tickers); parts = []
    def _one(chunk):
        try:
            df = _to_pandas(blp.bdp(chunk, flds=fields))
        finally:
            if THROTTLE_SEC: time.sleep(THROTTLE_SEC)
        if df is None: raise ValueError("risposta bdp vuota")
        return df
    for i in range(0, len(tickers), CHUNK_BDP):
        ch = tickers[i:i+CHUNK_BDP]
        try:
            parts.append(_one(ch)); continue
        except Exception as e:
            if _is_limit(e): raise BbgLimitReached(str(e))
            if verbose: print(f"    bdp blocco {i}-{i+len(ch)} anomalo ({str(e)[:60]}) -> per ticker")
        for t in ch:
            try: parts.append(_one([t]))
            except Exception as e2:
                if _is_limit(e2): raise BbgLimitReached(str(e2))
                if verbose: print(f"      bdp fallito su {t}: {str(e2)[:60]}")
    if not parts: return pd.DataFrame()
    out = pd.concat(parts)
    # Come bdh, anche bdp puo' tornare in formato LUNGO (ticker/field/value): va pivotato,
    # altrimenti il CSV ha tre colonne di testo e nessun campo utilizzabile.
    cols = {str(c).lower(): c for c in out.columns}
    if {"ticker", "field", "value"}.issubset(cols):
        t, f, v = cols["ticker"], cols["field"], cols["value"]
        out = out.pivot_table(index=t, columns=f, values=v, aggfunc="first")
        out.index.name = None
    out = out[~out.index.duplicated(keep="first")]
    out.columns = [str(c).upper() for c in out.columns]
    return out

def ticker(bb_id: str) -> str:
    """Bloomberg ID -> ticker per le richieste. E' il formato del progetto inflation_linked."""
    return f"{str(bb_id).strip()} Corp"

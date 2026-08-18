"""bbg - strato Bloomberg: universo, anagrafica, storici, con cache parquet incrementale.

COSA FA
  1. build_universe(): legge Govt_bonds.xlsx (fogli 'IL' e 'Nominal'), assegna ogni strumento
     al mercato via (Ticker, Calc Type) per i linker e Ticker+valuta per i nominali, applica i
     filtri concordati (funged, valute pre-euro, scaduti prima del floor dati) e salva
     universe.parquet con TUTTE le righe + flag incl/excl_reason: audit trail completo.
  2. enrich_reference(): bdp bulk (chunked) per l'anagrafica che il file universo non ha:
     cedola, frequenza, date (settle/accrual/prima cedola), day count, AMT_OUTSTANDING,
     e per i linker BASE_CPI/INFLATION_LAG aggiornati dal terminale.
  3. fetch_*: storici bdh con CACHE INCREMENTALE -- al secondo run scarica solo i giorni
     mancanti e gli strumenti nuovi, mai l'intera storia. Rispetta la quota del terminale.
       - prezzi linker  : PX_MID (teoria) + PX_BID/PX_ASK se trading=True
       - YTM nominali   : YLD_YTM_MID
       - curve ILS      : {root}{tenor} {src} Curncy per i tenor di config.YEARS_FORWARD
       - indici CPI     : PX_LAST
  4. fetch_all(): orchestrazione completa.

MAPPATURE (dal file universo dell'utente, verificate sui conteggi reali)
  Linker (Ticker, Calc Type) -> mercato:
    BTPS/1143 -> IT          FRTR/1103 -> FR (HICP)     FRTR/864 -> FR_CPI (CPI francese)
    DBRI/1103 -> DE          OBLI/1103 -> DE (Bobl-ei: i linker tedeschi a 5a sono OBLI,
                                              senza di essi il campione DE e' incompleto)
    TII/621 e TII/648 -> US  (due calc type registrati; non separati -- verificare su DES
                              quale sia la differenza prima di qualunque split)
    UKTI/1216 -> UK segment='new' (lag 3m, real-clean)
    UKTI/44 e UKTI/99 -> UK segment='old' (lag 8m, nominal-clean: convenzione di prezzo
                              DIVERSA -- il motore delle basi deve trattarli a parte)
    SPGBEI/1573 -> ES (fuori perimetro: escluso finche' ES non entra in config.MARKETS)
  Nominali Ticker -> mercato (solo valuta corrente):
    BTPS->IT  FRTR->FR  DBR/OBL/BKO->DE (BKO estende lo short end per il fit NSS)
    UKT->UK  T->US
  FR_CPI usa il pool nominale di FR (stesso emittente): NOMINAL_POOL_ALIAS.

FLOOR DATI: PULL_FLOOR = 2003-01-01. Motivo: le curve ILS (EUSWI/USSWIT/BPSWIT/FRSWI) sono
liquide dal ~2004, quindi nessuna base e' calcolabile prima; un anno di margine serve alla
continuita' del fit NSS. Strumenti SCADUTI prima del floor si tengono nell'universo (audit)
ma si escludono dai pull: risparmio di quota enorme sui ~3.300 nominali.

USO
  python bbg.py               # solo universo (offline, nessun terminale richiesto)
  python bbg.py --fetch       # universo + anagrafica + storici (richiede xbbg/terminale)
  UNIVERSE_FILE=... override del percorso del file universo.
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import MARKETS, YEARS_FORWARD, DATA, CACHE

UNIVERSE_FILE = Path(os.environ.get("UNIVERSE_FILE", DATA / "raw" / "Bloomberg" / "Govt_bonds.xlsx"))
PULL_FLOOR = pd.Timestamp("2003-01-01")
NOMINAL_POOL_ALIAS = {"FR_CPI": "FR"}          # FR_CPI abbina contro i nominali FR
CHUNK_BDP, CHUNK_BDH = 80, 25   # dimensione dei blocchi (il runner 02 puo' ridurli)
THROTTLE_SEC = 0.0              # pausa fra richieste API (il runner 02 la imposta)

# Mercati con curva nominale da FILE ISTITUZIONALE (GSW per US, BoE per UK): li' i
# nominali servono SOLO come gemelli del matching ISIN-vs-ISIN -> si scaricano (anagrafica
# E storici) solo i K piu' vicini sotto/sopra ogni scadenza linker. Sull'universo reale:
# US 1313 -> ~394, UK 156 -> ~114 (K=2). Per IT/FR/DE la curva la fittiamo noi dai bond,
# quindi serve lo spettro completo: nessun filtro.
try:
    from config import DE_NOMINAL_CURVE
except ImportError:
    DE_NOMINAL_CURVE = "fit"
CURVE_FROM_FILE = {"US", "UK"} | ({"DE"} if DE_NOMINAL_CURVE == "bundesbank" else set())
NOMINAL_POOL_PER_SIDE = 2
NOMINAL_POOL_MAX_DAYS = 550

NA_STRINGS = ["#N/A Field Not Applicable", "#N/A", "N/A", "#N/A N/A", ""]

LINKER_MAP = {("BTPS", 1143): ("IT", None),
              ("FRTR", 1103): ("FR", None),
              ("FRTR", 864): ("FR_CPI", None),
              ("DBRI", 1103): ("DE", None),
              ("OBLI", 1103): ("DE", None),
              ("SPGBEI", 1573): ("ES", None),
              ("TII", 621): ("US", None),
              ("TII", 648): ("US", None),
              ("UKTI", 1216): ("UK", "new"),
              ("UKTI", 44): ("UK", "old"),
              ("UKTI", 99): ("UK", "old")}
NOMINAL_MAP = {"BTPS": "IT", "FRTR": "FR", "DBR": "DE", "OBL": "DE", "BKO": "DE",
               "UKT": "UK", "T": "US"}
CCY_OK = {"IT": "EUR", "FR": "EUR", "FR_CPI": "EUR", "DE": "EUR",
          "UK": "GBP", "US": "USD", "ES": "EUR"}

REF_FIELDS_COMMON = ["SECURITY_NAME", "CPN", "CPN_FREQ", "FIRST_CPN_PERIOD_TYP",
                     "START_ACC_DT", "FIRST_SETTLE_DT", "FIRST_CPN_DT", "SECOND_CPN_DT",
                     "ISSUE_DT", "MATURITY", "DAY_CNT_DES", "AMT_OUTSTANDING"]
REF_FIELDS_LINKER = REF_FIELDS_COMMON + ["BASE_CPI", "INFLATION_LAG"]


# ------------------------------------------------------------------ util
def _to_date(s: pd.Series) -> pd.Series:
    """Date miste nel file universo: seriali Excel (45589.0) e stringhe dd/mm/yyyy.
    Robusto agli input anomali: i seriali fuori dal range plausibile [1, 80000] NON
    vengono passati a to_datetime (che andrebbe in overflow), ma trattati come stringa."""
    num = pd.to_numeric(s, errors="coerce")
    serial = num.where((num >= 1) & (num <= 80000))
    d_serial = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    mask = serial.notna()
    if mask.any():
        d_serial.loc[mask] = pd.to_datetime(
            serial.loc[mask].astype("int64"), unit="D", origin="1899-12-30", errors="coerce")
    as_str = s.where(d_serial.isna())
    d_string = pd.to_datetime(as_str, dayfirst=True, errors="coerce")
    return d_serial.fillna(d_string)


def _blp():
    """Import pigro: il modulo resta importabile senza terminale."""
    from xbbg import blp
    return blp


def _to_pandas(obj):
    """Converte l'output di xbbg in un DataFrame pandas LARGO (ticker in indice, campi in
    colonna). La versione installata restituisce un narwhals DataFrame in formato LUNGO
    (colonne ticker/field/value): va ripivotato. Gestisce anche il caso gia' pandas/largo
    e le risposte vuote (-> None, cosi' il chiamante fa il fallback per-ticker)."""
    if obj is None:
        return None
    # narwhals -> pandas
    if not isinstance(obj, pd.DataFrame):
        for attr in ("to_pandas", "to_native"):
            if hasattr(obj, attr):
                try:
                    obj = getattr(obj, attr)()
                    break
                except Exception:
                    continue
        if not isinstance(obj, pd.DataFrame):
            try:
                obj = pd.DataFrame(obj)
            except Exception:
                return None
    if obj.empty:
        return None
    cols = {c.lower(): c for c in obj.columns}
    # STORICO (bdh): colonne ticker/date/field/value -> NON pivotare qui, ci pensa _bdh
    # (deve diventare date-in-riga x ticker-in-colonna, non ticker-in-riga).
    if "date" in cols:
        return obj
    # ANAGRAFICA (bdp): ticker/field/value -> pivot a largo (ticker in riga, campi in colonna)
    if {"ticker", "field", "value"}.issubset(cols):
        t, f, v = cols["ticker"], cols["field"], cols["value"]
        long = obj[[t, f, v]].dropna(subset=[f])
        if long.empty:
            return None
        wide = long.pivot_table(index=t, columns=f, values=v, aggfunc="first")
        wide.index.name = None
        wide.columns = [str(c).upper() for c in wide.columns]
        # riconverte a numerico dove possibile (i campi tornano come stringhe)
        for c in wide.columns:
            conv = pd.to_numeric(wide[c], errors="coerce")
            if conv.notna().any():
                wide[c] = conv.where(conv.notna(), wide[c])
        return wide
    return obj


# ------------------------------------------------------------------ universo
def build_universe(path: Path = UNIVERSE_FILE, save: bool = True) -> pd.DataFrame:
    il = pd.read_excel(path, sheet_name="IL", na_values=NA_STRINGS)
    no = pd.read_excel(path, sheet_name="Nominal", na_values=NA_STRINGS)

    def _norm(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        out = pd.DataFrame({
            "isin": df["ISIN"].astype(str).str.strip(),
            "bb_id": df["Bloomberg ID"].astype(str).str.strip(),
            "tick": df["Ticker"].astype(str).str.strip(),
            "calc": pd.to_numeric(df["Calc Type"], errors="coerce").astype("Int64"),
            "ccy": df["Currency"].astype(str).str.strip(),
            "maturity": _to_date(df["Maturity"]),
            "amt_issued": pd.to_numeric(df.get("Amt Issued"), errors="coerce"),
            "infl_lag": pd.to_numeric(df.get("Inflation Lag"), errors="coerce"),
            "il_flag": df.get("Inflation-linked", pd.Series(index=df.index)).astype(str).str.strip(),
            "funged_id": df.get("Funged Sec ID"),
            "descr": df.get("Security Description", pd.Series(index=df.index)).astype(str),
        })
        out["base_cpi"] = pd.to_numeric(df["BASE_CPI"], errors="coerce") if "BASE_CPI" in df else np.nan
        out["kind"] = kind
        return out

    uni = pd.concat([_norm(il, "linker"), _norm(no, "nominal")], ignore_index=True)

    # --- assegnazione mercato e segmento
    mkt, seg = [], []
    for _, r in uni.iterrows():
        if r["kind"] == "linker":
            m, s = LINKER_MAP.get((r["tick"], int(r["calc"]) if pd.notna(r["calc"]) else -1),
                                  (None, None))
        else:
            m, s = NOMINAL_MAP.get(r["tick"]), None
        mkt.append(m)
        seg.append(s)
    uni["mkt"], uni["segment"] = mkt, seg
    # Lag di indicizzazione mancante (Bloomberg non lo riporta su alcune linee funged/
    # vecchie): per l'UK la convenzione e' univoca dal segmento -- old-style = 8 mesi
    # (RPI, nominal-clean, gilt UKTI/44 e /99), new-style = 3 mesi (real-clean, UKTI/1216).
    # Riempiamo il NaN col valore di convenzione; non sovrascriviamo lag gia' presenti.
    _uk = uni["mkt"].eq("UK") & uni["infl_lag"].isna()
    uni.loc[_uk & uni["segment"].eq("old"), "infl_lag"] = 8
    uni.loc[_uk & uni["segment"].eq("new"), "infl_lag"] = 3

    # --- cascata di esclusioni, in ordine dichiarato
    reason = pd.Series("", index=uni.index)
    reason[uni["mkt"].isna()] = "ticker/calc-type non mappato"
    ccy_exp = uni["mkt"].map(CCY_OK)
    bad_ccy = reason.eq("") & ccy_exp.notna() & (uni["ccy"] != ccy_exp)
    reason[bad_ccy] = "valuta pre-euro o estranea (" + uni.loc[bad_ccy, "ccy"] + ")"
    funged = reason.eq("") & uni["funged_id"].notna()
    reason[funged] = "funged in altra linea (si tiene la linea madre)"
    il_mismatch = reason.eq("") & ((uni["kind"] == "linker") & uni["il_flag"].eq("N") |
                                   (uni["kind"] == "nominal") & uni["il_flag"].eq("Y"))
    reason[il_mismatch] = "flag inflation-linked incoerente col foglio"
    matured = reason.eq("") & (uni["maturity"] < PULL_FLOOR)
    reason[matured] = f"scaduto prima del floor dati ({PULL_FLOOR.date()})"
    off_scope = reason.eq("") & ~uni["mkt"].isin(MARKETS.keys())
    reason[off_scope] = "mercato fuori perimetro (non in config.MARKETS)"
    dup = reason.eq("") & uni.duplicated("isin", keep="first")
    reason[dup] = "ISIN duplicato"

    uni["excl_reason"] = reason
    uni["incl"] = reason.eq("")
    # linker con BASE_CPI mancante o zero: non si esclude qui -- bdp la aggiorna;
    # se resta invalida dopo l'enrichment, l'esclusione avviene la' con ragione propria.
    uni["needs_base_cpi"] = uni["incl"] & (uni["kind"] == "linker") & \
                            (uni["base_cpi"].isna() | (uni["base_cpi"] <= 0))
    # incoerenza lag/segmento UK (1216 dovrebbe avere lag 3, old-style 8): solo warning
    uni["lag_warn"] = uni["incl"] & (uni["mkt"] == "UK") & (
        (uni["segment"].eq("new") & uni["infl_lag"].ne(3)) |
        (uni["segment"].eq("old") & uni["infl_lag"].ne(8)))

    uni = uni.sort_values(["kind", "mkt", "maturity"]).reset_index(drop=True)
    if save:
        uni.to_parquet(CACHE / "universe.parquet")
    return uni


def universe_report(uni: pd.DataFrame) -> str:
    L = ["--- universo: inclusi per mercato ---"]
    piv = (uni[uni["incl"]].groupby(["mkt", "kind", "segment"], dropna=False)
           .size().rename("n").reset_index())
    for _, r in piv.iterrows():
        seg = f" [{r['segment']}]" if pd.notna(r["segment"]) else ""
        L.append(f"  {r['mkt']:6s} {r['kind']:7s}{seg:7s}: {r['n']:5d}")
    L.append("--- esclusi per ragione ---")
    for reas, n in uni.loc[~uni["incl"], "excl_reason"].value_counts().items():
        L.append(f"  {n:5d}  {reas}")
    n_bc = int(uni["needs_base_cpi"].sum())
    n_lw = int(uni["lag_warn"].sum())
    if n_bc:
        L.append(f"ATTENZIONE: {n_bc} linker con BASE_CPI mancante/zero nel file -> refresh via bdp")
    if n_lw:
        L.append(f"ATTENZIONE: {n_lw} linker UK con lag incoerente col segmento -> verificare su DES")
    return "\n".join(L)


# ------------------------------------------------------------------ bdp anagrafica
def _bdp(tickers: list[str], fields: list[str]) -> pd.DataFrame:
    """bdp bulk a chunk. Se un chunk torna anomalo (narwhals/vuoto), riprova PER SINGOLO
    ticker: un bond difettoso costa un bond, non l'intero chunk."""
    blp = _blp()

    def _one(chunk: list[str]) -> pd.DataFrame:
        try:
            df = _to_pandas(blp.bdp(chunk, flds=fields))
        finally:
            if THROTTLE_SEC:
                time.sleep(THROTTLE_SEC)
        if df is None or df.empty:
            raise ValueError("risposta bdp non convertibile o vuota")
        return df

    parts = []
    for i in range(0, len(tickers), CHUNK_BDP):
        chunk = tickers[i:i + CHUNK_BDP]
        try:
            parts.append(_one(chunk))
            continue
        except Exception as e:
            print(f"    bdp chunk {i}-{i+len(chunk)} anomalo ({e}) -> riprovo per ticker")
        for t in chunk:
            try:
                parts.append(_one([t]))
            except Exception as e2:
                print(f"      bdp fallito su {t}: {e2}")
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts)
    out = out[~out.index.duplicated(keep="first")]
    out.columns = [str(c).upper() for c in out.columns]
    return out


def _ref_ticker(row: pd.Series) -> str:
    m = MARKETS[row["mkt"]]
    return m.ref_ticker(row["bb_id"]) if row["kind"] == "linker" else f"{row['bb_id']} Corp"


def enrich_reference(uni: pd.DataFrame, kind: str, save: bool = True,
                     mkts: list[str] | None = None,
                     isins: set[str] | None = None) -> pd.DataFrame:
    """bdp bulk INCREMENTALE: se ref_{kind}.parquet esiste, ri-chiede SOLO gli strumenti
    senza anagrafica (CPN mancante): una ripartenza dopo il limite non riconsuma quota.
    mkts limita ai mercati del run; isins restringe al pool (usato per i nominali US/UK)."""
    sub = uni[uni["incl"] & uni["kind"].eq(kind)].copy()
    if mkts is not None:
        sub = sub[sub["mkt"].isin(mkts)]
    if isins is not None:
        sub = sub[sub["isin"].isin(isins)]
    sub["ref_tick"] = sub.apply(_ref_ticker, axis=1)
    fields = REF_FIELDS_LINKER if kind == "linker" else REF_FIELDS_COMMON

    path = CACHE / f"ref_{kind}.parquet"
    old = pd.read_parquet(path) if path.exists() else None
    done = set(old.index[old["CPN"].notna()]) if (old is not None and "CPN" in old.columns) else set()
    todo = sub[~sub["isin"].isin(done)]
    print(f"  anagrafica {kind}: {len(sub)} strumenti nel run, "
          f"{len(done & set(sub['isin']))} gia' in cache, {len(todo)} da chiedere")
    if len(todo):
        ref = _bdp(todo["ref_tick"].tolist(), fields)
        if not ref.empty:
            ref = ref.reindex(todo["ref_tick"].values)
            ref.index = todo["isin"].values
            blk = todo.set_index("isin").join(ref, rsuffix="_BDP")
        else:
            blk = todo.set_index("isin")
        old = blk if old is None else pd.concat([old[~old.index.isin(blk.index)], blk])
    out = old if old is not None else sub.set_index("isin")
    out.index.name = "isin"

    # Il pivot narwhals fa scendere i campi bdp come STRINGHE ('1.0', '2.5', ...).
    # Tipizza i numerici qui, una volta, cosi' il resto del codice riceve numeri veri
    # (evita int('1.0') che esplode). Le date restano stringhe: le parsa chi le usa.
    NUM_FIELDS = ["CPN", "CPN_FREQ", "AMT_OUTSTANDING", "AMT_ISSUED", "INFLATION_LAG",
                  "BASE_CPI", "BASE_CPI_BDP"]
    for c in NUM_FIELDS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if kind == "linker":
        bdp_col = "BASE_CPI_BDP" if "BASE_CPI_BDP" in out.columns else \
                  ("BASE_CPI" if "BASE_CPI" in out.columns else None)
        src_ = pd.to_numeric(out[bdp_col], errors="coerce") if bdp_col \
            else pd.Series(np.nan, index=out.index)
        out["base_cpi_final"] = src_.fillna(pd.to_numeric(out["base_cpi"], errors="coerce"))
        has_ref = out["CPN"].notna() if "CPN" in out.columns else pd.Series(False, out.index)
        bad = has_ref & (out["base_cpi_final"].isna() | (out["base_cpi_final"] <= 0))
        if bad.any():
            print(f"    {int(bad.sum())} linker senza BASE_CPI valida anche dopo bdp -> "
                  f"esclusi dalle basi (vedi ref_{kind}.parquet, base_cpi_final)")
        # Correzione ribasamento IAPC: il BASE_CPI dei bond scaduti prima di un ribasamento
        # Eurostat resta 'congelato' in base vecchia, mentre la serie CPI scaricata oggi e'
        # in base nuova -> CI sfasato. Si riallinea il BASE_CPI alla base corrente (per i
        # soli linker euro il cui CPI e' l'IAPC; per UK/US convenzioni diverse, si salta).
        euro_cpi = {"IT": "CPTFEMU", "FR": "FRCPXTOB", "DE": "CPTFEMU", "FR_CPI": "FRCPXTOB"}
        if "mkt" in out.columns:
            for cpi_tk in set(euro_cpi.get(m) for m in out["mkt"].unique() if m in euro_cpi):
                if cpi_tk is None:
                    continue
                cpi_path = CACHE / f"cpi_{cpi_tk}.parquet"
                if not cpi_path.exists():
                    print(f"    ribasamento: serie {cpi_tk} non in cache -> BASE_CPI non "
                          f"verificato (scarica il CPI e rilancia l'anagrafica)")
                    continue
                cpi_ser = pd.read_parquet(cpi_path).iloc[:, 0]
                mask = out["mkt"].isin([m for m, t in euro_cpi.items() if t == cpi_tk])
                try:
                    from basis import rebase_base_cpi
                    corr, _ = rebase_base_cpi(out[mask], cpi_ser, verbose=True)
                    out.loc[corr.index, "base_cpi_final"] = corr.values
                except Exception as e:
                    print(f"    ribasamento: correzione non applicata ({e})")
    if save:
        out.to_parquet(path)
    return out

# ------------------------------------------------------------------ bdh incrementale
def _bdh(tickers: list[str], field: str, start, end) -> pd.DataFrame:
    """bdh bulk a chunk, con conversione difensiva e fallback per-ticker.
    xbbg restituisce il formato LUNGO ticker/date/field/value: qui lo pivotiamo in LARGO
    (indice = date, colonne = ticker). Serie con storie diverse si allineano per data."""
    blp = _blp()

    def _one(chunk: list[str]) -> pd.DataFrame:
        try:
            raw = _to_pandas(blp.bdh(chunk, field, start, end))
        finally:
            if THROTTLE_SEC:
                time.sleep(THROTTLE_SEC)
        if raw is None:
            return pd.DataFrame()   # risposta vuota (bond scaduto / nessun giorno nuovo): non e' errore
        if isinstance(raw, pd.DataFrame) and raw.empty:
            return pd.DataFrame()
        cols = {c.lower(): c for c in raw.columns}
        # formato lungo ticker/date/(field)/value -> pivot date x ticker
        if {"ticker", "date", "value"}.issubset(cols):
            t, d, v = cols["ticker"], cols["date"], cols["value"]
            long = raw[[t, d, v]].dropna(subset=[v])
            if long.empty:
                return pd.DataFrame()
            wide = long.pivot_table(index=d, columns=t, values=v, aggfunc="first")
            wide.index = pd.to_datetime(wide.index)
            wide.columns = [str(c) for c in wide.columns]
            return wide.sort_index()
        # gia' largo (fallback): appiattisci un eventuale MultiIndex
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw.index = pd.to_datetime(raw.index)
        return raw

    parts = []
    for i in range(0, len(tickers), CHUNK_BDH):
        chunk = tickers[i:i + CHUNK_BDH]
        try:
            df = _one(chunk)
            if not df.empty:
                parts.append(df)
            continue
        except Exception as e:
            print(f"    bdh chunk {i}-{i+len(chunk)} anomalo ({e}) -> riprovo per ticker")
        for t in chunk:
            try:
                df = _one([t])
                if not df.empty:
                    parts.append(df)
            except Exception as e2:
                print(f"      bdh fallito su {t}: {e2}")
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1)
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def _update_wide(name: str, tick2col: dict, field: str,
                 floor: pd.Timestamp = PULL_FLOOR) -> pd.DataFrame:
    """Cache incrementale CON CHECKPOINT: il parquet viene riscritto dopo OGNI blocco di
    strumenti nuovi, cosi' se il limite Bloomberg scatta a meta' non si perde nulla e la
    ripartenza chiede solo i mancanti. Colonne gia' in cache -> solo i giorni nuovi."""
    path = CACHE / f"{name}.parquet"
    today = pd.Timestamp.today().normalize()
    out = pd.read_parquet(path) if path.exists() else None
    have = set(out.columns) if out is not None else set()
    cur = {t: c for t, c in tick2col.items() if c in have}
    new = {t: c for t, c in tick2col.items() if c not in have}

    def _merge_save(df: pd.DataFrame | None) -> None:
        nonlocal out
        if df is None or df.empty:
            return
        out = df if out is None else out.combine_first(df)
        out.sort_index().to_parquet(path)              # checkpoint su disco

    if cur and out is not None and out.index.max() < today:
        df = _bdh(list(cur), field, out.index.max() + pd.Timedelta(days=1), today)
        _merge_save(df.rename(columns=cur) if not df.empty else None)

    items = list(new.items())
    for j in range(0, len(items), CHUNK_BDH):
        blk = dict(items[j:j + CHUNK_BDH])
        df = _bdh(list(blk), field, floor, today)
        _merge_save(df.rename(columns=blk) if not df.empty else None)
        if len(items) > CHUNK_BDH:
            print(f"      [{name}] checkpoint {min(j + CHUNK_BDH, len(items))}/{len(items)}")
    return out if out is not None else pd.DataFrame()


# ------------------------------------------------------------------ fetch specifici
def fetch_linker_prices(mkt: str, ref_link: pd.DataFrame, trading: bool = False) -> None:
    m = MARKETS[mkt]
    sub = ref_link[ref_link["mkt"] == mkt]
    t2c = {m.px_ticker(r["bb_id"]): isin for isin, r in sub.iterrows()}
    if not t2c:
        return
    print(f"  {mkt}: prezzi PX_MID di {len(t2c)} linker")
    _update_wide(f"px_mid_{mkt}", t2c, "PX_MID")
    if trading:
        for f in ("PX_BID", "PX_ASK"):
            _update_wide(f"{f.lower()}_{mkt}", t2c, f)


def _matching_pool(nom: pd.DataFrame, linker_mats: pd.Series) -> pd.DataFrame:
    """Per i mercati con curva nominale da file (US/UK): per ogni scadenza linker si
    tengono i NOMINAL_POOL_PER_SIDE nominali piu' vicini sotto e sopra, entro
    NOMINAL_POOL_MAX_DAYS. E' tutto cio' che serve al matching B (lower/upper/nearest
    piu' una riserva se il piu' vicino non era ancora emesso a inizio campione)."""
    nmat = pd.to_datetime(nom["maturity"]).values
    lmat = pd.to_datetime(linker_mats).values
    D = (nmat[:, None] - lmat[None, :]).astype("timedelta64[D]").astype(int)
    keep: set = set()
    for j in range(D.shape[1]):
        dj = D[:, j]
        lower = np.where(dj < 0)[0]
        upper = np.where(dj >= 0)[0]
        keep |= set(lower[np.argsort(-dj[lower])[:NOMINAL_POOL_PER_SIDE]])
        keep |= set(upper[np.argsort(dj[upper])[:NOMINAL_POOL_PER_SIDE]])
    keep = {i for i in keep if abs(D[i]).min() <= NOMINAL_POOL_MAX_DAYS}
    return nom.iloc[sorted(keep)]


def fetch_nominal_ytm(mkt: str, ref_nom: pd.DataFrame,
                      ref_link: pd.DataFrame | None = None) -> None:
    sub = ref_nom[ref_nom["mkt"] == mkt]
    n_full = len(sub)
    if mkt in CURVE_FROM_FILE and ref_link is not None:
        lm = ref_link.loc[ref_link["mkt"] == mkt, "maturity"]
        if len(lm):
            sub = _matching_pool(sub, lm)
            print(f"  {mkt}: curva nominale da file (GSW/BoE) -> pool matching "
                  f"{n_full} -> {len(sub)} nominali")
    t2c = {f"{r['bb_id']} Corp": isin for isin, r in sub.iterrows()}
    if not t2c:
        return
    print(f"  {mkt}: YLD_YTM_MID di {len(t2c)} nominali")
    _update_wide(f"ytm_{mkt}", t2c, "YLD_YTM_MID")
    if mkt not in CURVE_FROM_FILE:
        # fit NSS GSW sui PREZZI: serve il clean PX_MID dei nominali, ma SOLO dove la
        # curva la fittiamo noi (IT, FR). Per US/UK/DE la curva e' ufficiale (GSW/BoE/
        # Bundesbank): i prezzi dei nominali non servono e non si scaricano.
        print(f"  {mkt}: PX_MID di {len(t2c)} nominali (fit curva sui prezzi, GSW)")
        _update_wide(f"px_nom_{mkt}", t2c, "PX_MID")


def fetch_ils(mkts: list[str] | None = None) -> None:
    seen = set()
    for code, m in MARKETS.items():
        if mkts is not None and code not in mkts:
            continue
        key = (m.ils, m.ils_source)
        if key in seen:
            continue
        seen.add(key)
        t2c = {m.ils_ticker(n): float(n) for n in YEARS_FORWARD}
        print(f"  ILS {m.ils} ({m.ils_source}): {len(t2c)} tenor")
        _update_wide(f"ils_{m.ils}", t2c, "PX_LAST")


def fetch_cpi(mkts: list[str] | None = None) -> None:
    for cpi in sorted({MARKETS[c].cpi for c in (mkts or MARKETS)}):
        print(f"  CPI {cpi}")
        _update_wide(f"cpi_{cpi}", {f"{cpi} Index": cpi}, "PX_LAST",
                     floor=pd.Timestamp("1996-01-01"))   # storia CPI lunga: serve al carry

FETCH_ORDER = ["IT", "FR_CPI", "DE", "FR", "UK", "US"]   # dal piu' piccolo: quota graduale


def fetch_all(trading: bool = False, markets: list[str] | None = None,
              phases: dict | None = None) -> None:
    ph = {"ANAGRAFICA": True, "PREZZI": True, "NOMINALI": True, "ILS": True, "CPI": True}
    if phases:
        ph.update({k.upper(): v for k, v in phases.items()})
    uni = build_universe()
    print(universe_report(uni))
    mkts = [m for m in FETCH_ORDER if m in (markets or MARKETS)]
    print(f"--- run: mercati {mkts} | fasi {[k for k, v in ph.items() if v]} ---")

    if ph["ANAGRAFICA"]:
        print("--- anagrafica bdp (incrementale) ---")
        nom_mkts = sorted({NOMINAL_POOL_ALIAS.get(m, m) for m in mkts})
        # per US/UK anche l'ANAGRAFICA dei nominali si limita al pool di matching
        pool: set[str] | None = None
        if any(m in CURVE_FROM_FILE for m in mkts):
            pool = set()
            for nm_mkt in nom_mkts:
                nm = uni[uni["incl"] & uni["kind"].eq("nominal") & uni["mkt"].eq(nm_mkt)]
                if nm_mkt in CURVE_FROM_FILE:
                    lk = uni[uni["incl"] & uni["kind"].eq("linker") & uni["mkt"].eq(nm_mkt)]
                    nm = _matching_pool(nm, lk["maturity"])
                pool |= set(nm["isin"])
        ref_link = enrich_reference(uni, "linker", mkts=mkts)
        ref_nom = enrich_reference(uni, "nominal", mkts=nom_mkts, isins=pool)
    else:
        ref_link, ref_nom = load("ref_linker"), load("ref_nominal")

    print("--- storici bdh (cache incrementale, checkpoint per blocco) ---")
    for mkt in mkts:
        if ph["PREZZI"]:
            fetch_linker_prices(mkt, ref_link, trading)
        if ph["NOMINALI"] and mkt not in NOMINAL_POOL_ALIAS:
            fetch_nominal_ytm(mkt, ref_nom, ref_link)
    if ph["ILS"]:
        fetch_ils(mkts)
    if ph["CPI"]:
        fetch_cpi(mkts)
    print("fatto.")

# ------------------------------------------------------------------ loader per gli altri moduli
def load(name: str) -> pd.DataFrame:
    return pd.read_parquet(CACHE / f"{name}.parquet")


if __name__ == "__main__":
    uni = build_universe()
    print(universe_report(uni))
    if "--fetch" in sys.argv:
        try:
            import xbbg  # noqa: F401
        except ImportError:
            sys.exit("xbbg non disponibile: lancia sul terminale Bloomberg.")
        fetch_all(trading="--trading" in sys.argv)

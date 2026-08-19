"""r0b - DATE DI PUBBLICAZIONE del CPI/RPI da Bloomberg.

PERCHE'. Maffei campiona yield e swap alla DATA DI RILASCIO del CPI, non a fine mese:
"each step falls on the inflation release dates ... the emphasis on when the market
actually receives new information". Con le date vere la deviazione su questo punto scompare.

IL PROBLEMA DELLA VERSIONE PRECEDENTE. bdh("CPI INDX Index", "ECO_RELEASE_DT", ...) tornava
una tabella con la struttura GIUSTA (ticker/date/field/value) ma VUOTA. Non era un errore di
parsing: la richiesta non restituiva righe. Due cause possibili, e vanno separate:

  (a) FORMATO DELLE DATE. Il codice passava "20000101". Il core Rust di xbbg si aspetta il
      formato ISO "2000-01-01" e con l'altro puo' restituire un intervallo vuoto SENZA
      sollevare eccezione -- il fallimento piu' insidioso, perche' sembra "nessun dato".

  (b) CAMPO NON STORICO. ECO_RELEASE_DT e' un campo di REFERENCE, non di serie storica:
      HistoricalDataRequest serve solo i campi abilitati come storici. Su molte licenze il
      campo risponde a bdp (prossima release) ma non a bdh. E il ticker conta: "CPI INDX
      Index" e' il LIVELLO dell'indice, mentre l'evento del calendario economico e' il
      ticker di RILASCIO ("CPI CHNG Index" per la variazione mensile US).

Questa versione non indovina: prova una GRIGLIA di combinazioni ticker x campo x formato
data, riporta quale risponde, e scrive il file solo con quella. Stesso metodo usato per i
ticker CDS nel progetto CCT, dove aveva risolto un problema identico.

VALIDAZIONE invariata: profilo (una release di inflazione esce fra il ~10 e il ~22 del mese)
e copertura (una data per ogni mese). Se il profilo e' sparso, la fonte e' sbagliata.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import pandas as pd
from config import CACHE

CAMPIONE_START = "2004-01-31"

# griglia di candidati: (ticker, campo). Il primo che risponde vince.
CANDIDATI = {
    # CPI INDX Index e' risultato genuinamente VUOTO su ECO_RELEASE_DT: e' il livello
    # dell'indice, non l'evento del calendario economico. I ticker di RILASCIO rispondono.
    "US": [
        ("CPI CHNG Index",  "ECO_RELEASE_DT"),   # rilascio: variazione mensile SA
        ("CPURNSA Index",   "ECO_RELEASE_DT"),   # CPI-U NSA, livello
        ("CPI YOY Index",   "ECO_RELEASE_DT"),
        ("CPI INDX Index",  "ECO_RELEASE_DT"),
        ("CPI CHNG Index",  "ECO_RELEASE_DATE"),
        ("CPURNSA Index",   "ECO_RELEASE_DATE"),
        ("CPI CHNG Index",  "LATEST_ANNOUNCEMENT_DT"),
    ],
    "UK": [
        ("UKRPI Index",     "ECO_RELEASE_DT"),
        ("UKRPYOY Index",   "ECO_RELEASE_DT"),
        ("UKRPI Index",     "ECO_RELEASE_DATE"),
        ("UKRPI Index",     "LATEST_ANNOUNCEMENT_DT"),
    ],
}
# due formati di data: ISO e compatto. Il core Rust puo' accettarne solo uno.
FORMATI = [("ISO", "2004-01-01", "2026-12-31"), ("compatto", "20040101", "20261231")]


def _blp():
    try:
        from xbbg import blp
    except ImportError:
        raise SystemExit("[!] xbbg non disponibile: r0b va lanciato sul terminale Bloomberg.")
    return blp


def _to_pandas(obj):
    """narwhals/pyarrow -> pandas. La versione installata torna un oggetto narwhals."""
    if obj is None: return None
    if not isinstance(obj, pd.DataFrame):
        for attr in ("to_pandas", "to_native"):
            if hasattr(obj, attr):
                try:
                    obj = getattr(obj, attr)()
                    if hasattr(obj, "to_pandas"): obj = obj.to_pandas()   # pyarrow.Table
                    if isinstance(obj, pd.DataFrame): break
                except Exception: pass
    if not isinstance(obj, pd.DataFrame):
        try: obj = pd.DataFrame(obj)
        except Exception: return None
    return None if obj.empty else obj


def _parse_date_col(v: pd.Series) -> pd.Series:
    """Prova i formati in cui Bloomberg puo' restituire una data dentro 'value'."""
    v = v.dropna().astype(str).str.strip()
    if v.empty: return pd.Series(dtype="datetime64[ns]")
    tentativi = [
        # Bloomberg restituisce YYYYMMDD come FLOAT: la stringa arriva come "20040220.0".
        # Nessun parser di date la accetta, e come seriale cadrebbe nell'anno 56000 (il
        # controllo di plausibilita' la scarterebbe). Va normalizzata a intero PRIMA.
        ("YYYYMMDD float", lambda x: pd.to_datetime(
            pd.to_numeric(x, errors="coerce").astype("Int64").astype(str).replace("<NA>", ""),
            errors="coerce", format="%Y%m%d")),
        ("ISO/auto",   lambda x: pd.to_datetime(x, errors="coerce", format="mixed")),
        ("YYYYMMDD",   lambda x: pd.to_datetime(x, errors="coerce", format="%Y%m%d")),
        ("MM/DD/YYYY", lambda x: pd.to_datetime(x, errors="coerce", format="%m/%d/%Y")),
        ("DD/MM/YYYY", lambda x: pd.to_datetime(x, errors="coerce", format="%d/%m/%Y")),
        # seriale Excel: Bloomberg a volte restituisce il numero, non la stringa
        ("seriale",    lambda x: pd.to_datetime(pd.to_numeric(x, errors="coerce"),
                                                unit="D", origin="1899-12-30", errors="coerce")),
    ]
    # Controllo di PLAUSIBILITA': senza, il parser seriale converte qualunque numero in
    # una data. Un valore CPI come 310.3 diventerebbe l'11 novembre 1900 e passerebbe per
    # buono. Si accettano solo date dentro un intervallo che una release puo' avere.
    LO, HI = pd.Timestamp("1990-01-01"), pd.Timestamp("2035-12-31")
    best, best_n, best_lab = pd.Series(dtype="datetime64[ns]"), 0, None
    for lab, f in tentativi:
        try: d = f(v)
        except Exception: continue
        d = d.where((d >= LO) & (d <= HI))
        n = int(d.notna().sum())
        if n > best_n: best, best_n, best_lab = d, n, lab
    if best_lab and best_n:
        print(f"      formato riconosciuto: {best_lab} ({best_n}/{len(v)} convertite)")
    return best.dropna()


def _estrai_date(raw: pd.DataFrame) -> pd.Series:
    """
    Dalla tabella lunga ticker/date/field/value estrae le DATE DI RILASCIO.

    'date' e' la data di OSSERVAZIONE (il mese di riferimento), 'value' e' la data di
    RILASCIO: serve quest'ultima. NESSUN FALLBACK su 'date': scrivere le date di
    osservazione al posto di quelle di rilascio produrrebbe un file plausibile ma
    SBAGLIATO, e a valle nessuno se ne accorgerebbe -- il tipo di errore peggiore.
    Se 'value' non contiene date, si MOSTRA cosa contiene e ci si ferma su quel candidato.
    """
    cols = {str(c).lower(): c for c in raw.columns}
    if "value" not in cols:
        print(f"      [!] nessuna colonna 'value': {list(raw.columns)}")
        return pd.Series(dtype="datetime64[ns]")
    v = _parse_date_col(raw[cols["value"]])
    if len(v) >= max(10, 0.5 * len(raw)):
        return v
    campione = raw[cols["value"]].dropna().astype(str).head(6).tolist()
    print(f"      [!] 'value' NON contiene date. Primi valori: {campione}")
    print( "          (se sono numeri come 2.4 o 310.3 il campo restituisce il VALORE del")
    print( "           dato, non la data: il campo o il ticker sono sbagliati)")
    return pd.Series(dtype="datetime64[ns]")


def prova_griglia(mkt: str) -> pd.Series:
    blp = _blp()
    print(f"\n[{mkt}] provo la griglia ticker x campo x formato data")
    for tick, campo in CANDIDATI[mkt]:
        for nome_fmt, d0, d1 in FORMATI:
            try:
                raw = _to_pandas(blp.bdh(tick, campo, d0, d1))
            except Exception as e:
                print(f"  {tick:18s} {campo:22s} {nome_fmt:9s} -> errore: {str(e)[:45]}")
                continue
            if raw is None or raw.empty:
                print(f"  {tick:18s} {campo:22s} {nome_fmt:9s} -> vuoto")
                continue
            s = _estrai_date(raw)
            if len(s) >= 24:
                print(f"  {tick:18s} {campo:22s} {nome_fmt:9s} -> OK, {len(s)} date  <== USO QUESTA")
                return s
            print(f"  {tick:18s} {campo:22s} {nome_fmt:9s} -> solo {len(s)} date, scarto")
    return pd.Series(dtype="datetime64[ns]")


def _clean(dates: pd.Series, etichetta: str) -> pd.Series:
    s = pd.Series(pd.to_datetime(dates)).dropna().sort_values().drop_duplicates()
    s = s[s >= pd.Timestamp(CAMPIONE_START)]
    s = s.groupby(s.dt.to_period("M")).first()
    giorni = s.dt.day
    print(f"\n[{etichetta}] {len(s)} date | {s.min().date()} -> {s.max().date()}")
    print(f"  giorno del mese: mediana {giorni.median():.0f}, "
          f"p10 {giorni.quantile(.1):.0f}, p90 {giorni.quantile(.9):.0f}")
    dentro = ((giorni >= 8) & (giorni <= 24)).mean()
    print(f"  dentro la finestra 8-24 del mese: {dentro:.0%}")
    if dentro < 0.80:
        print("  [!] PROFILO ANOMALO: una release di inflazione esce fra il ~10 e il ~22.")
        print("      Se le date sono sparse su tutto il mese la fonte e' sbagliata:")
        print("      probabile che si stia leggendo la data di OSSERVAZIONE, non di rilascio.")
    mesi = pd.period_range(s.index.min(), s.index.max(), freq="M")
    buchi = [str(m) for m in mesi if m not in s.index]
    if buchi:
        print(f"  [!] {len(buchi)} mesi senza data: {buchi[:6]}{' ...' if len(buchi) > 6 else ''}")
        print("      release_grid li escluderebbe in silenzio.")
    return s


if __name__ == "__main__":
    print(">>> r0b release_dates v4 (griglia ticker x campo x formato) <<<")
    solo_us = "--solo-us" in sys.argv
    esiti = {}
    for mkt in (["US"] if solo_us else ["US", "UK"]):
        s = prova_griglia(mkt)
        if s.empty:
            print(f"\n[{mkt}] NESSUNA combinazione ha risposto.")
            print("  Verifica sul terminale, in Excel, in quest'ordine:")
            print('    =BDH("CPI INDX Index";"ECO_RELEASE_DT";"2004-01-01";"2026-12-31")')
            print('    =BDP("CPI INDX Index";"ECO_RELEASE_DT")        <- se questa risponde,')
            print("       il campo esiste ma NON e' storico: serve il calendario ECO <GO>,")
            print("       esportabile in Excel, oppure il ticker di rilascio (CPI CHNG Index).")
            print("  Manda il risultato: dalla differenza fra BDH e BDP si capisce quale")
            print("  delle due cause sia, e la soluzione cambia di conseguenza.")
            print("  NB: nessun file e' stato scritto. Meglio fermarsi che salvare le date di")
            print("      OSSERVAZIONE spacciandole per date di rilascio: a valle r2 e")
            print("      maffei_replica le userebbero senza accorgersi di nulla.")
            continue
        s = _clean(s, mkt)
        esiti[mkt] = s
        out = CACHE / f"cpi_release_dates_{mkt}.csv"
        s.reset_index(drop=True).rename("release_date").to_frame().to_csv(out, index=False)
        print(f"  [saved] {out}")
    if not esiti:
        print("\n[stop] nessun file scritto.")

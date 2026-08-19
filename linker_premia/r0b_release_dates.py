"""r0b - DATE DI PUBBLICAZIONE del CPI/RPI da Bloomberg (campo ECO_RELEASE_DT).

PERCHE'. Maffei campiona yield e swap alla DATA DI RILASCIO del CPI, non a fine mese:
"each step falls on the inflation release dates ... the emphasis on when the market
actually receives new information" (tesi, sez. 4). Con le date vere la deviazione su
questo punto scompare; finora r2/maffei_replica usano una proxy (giorno D del mese dopo).

FONTE: ECO_RELEASE_DT su Bloomberg. E' IL calendario economico che i trader vedono in
tempo reale -> letteralmente "quando il mercato riceve l'informazione", la definizione di
Maffei. Meglio di FRED (release/dates dava date sporche) e dell'API ONS (buchi nel
campione): entrambe scartate dopo verifica del profilo.

SU QUALI INDICI, e perche' NON entrambi NSA. La data di release e' un attributo del
COMUNICATO: nel singolo comunicato BLS escono insieme SA e NSA, stessa data. Ma per far
combaciare data e valore, la data va presa dallo STESSO ticker della serie CPI che entra
nella sorpresa:
  - US  -> CPI INDX Index   (CPI-U SA, CUSR0000SA0): e' la serie della sorpresa di Maffei
                             (an_cpi_US_SA). SA, non NSA, perche' la sorpresa e' su SA.
  - UK  -> UKRPI Index      (RPI): l'ONS lo pubblica solo NSA, ed e' la serie usata
                             (an_cpi_UK). Nessuna scelta possibile, ed e' quella giusta.
Cioe': NON "entrambi NSA", ma ciascuno dalla serie che si usa davvero (US=SA, UK=RPI).

VALIDAZIONE. Dopo lo scarico controlla il PROFILO: una release di inflazione esce tra il
~10 e il ~22 del mese. Se le date sono sparse su tutto il mese, la fonte e' sbagliata e lo
dice (e' cosi' che abbiamo scartato FRED). E la COPERTURA: una data per ogni mese del
campione, altrimenti i mesi mancanti verrebbero esclusi in silenzio da release_grid.

NESSUN FALLBACK: se Bloomberg non risponde o il campo e' sbagliato, si ferma.

Uso (sul terminale Bloomberg, con xbbg installato):
    python .\\src\\linker_premia\\r0b_release_dates.py
    python .\\src\\linker_premia\\r0b_release_dates.py --solo-us

Output: cpi_release_dates_US.csv e cpi_release_dates_UK.csv in cache. r2_surprises e
maffei_replica li usano AUTOMATICAMENTE: stampano "calendario release ESATTO".
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import pandas as pd
from config import CACHE

# la data di release si prende dallo STESSO ticker della serie CPI della sorpresa
RELEASE_TICKER = {"US": "CPI INDX Index",   # SA, come an_cpi_US_SA (sorpresa Maffei)
                  "UK": "UKRPI Index"}       # RPI, come an_cpi_UK
RELEASE_FIELD = "ECO_RELEASE_DT"
CAMPIONE_START = "2004-01-31"


def _blp():
    try:
        from xbbg import blp
    except ImportError:
        raise SystemExit("[!] xbbg non disponibile: r0b va lanciato sul terminale Bloomberg "
                         "(dove xbbg e' installato).")
    return blp


def _clean(dates, etichetta: str) -> pd.Series:
    """Ordina, deduplica, tiene la PRIMA data di ogni mese, valida profilo e stampa."""
    s = pd.to_datetime(pd.Series(list(dates)), errors="coerce").dropna()
    s = s.sort_values()
    s = s[~s.duplicated()].reset_index(drop=True)
    if s.empty:
        raise RuntimeError(f"{etichetta}: nessuna data valida")
    per_month = s.groupby([s.dt.year, s.dt.month]).size()
    if per_month.max() > 1:
        print(f"  [nota] {int((per_month > 1).sum())} mesi con piu' di una data: tengo la "
              f"PRIMA di ogni mese")
        s = s.groupby([s.dt.year, s.dt.month]).min().reset_index(drop=True)
    g = s.dt.day
    print(f"  {etichetta}: {len(s)} date, {s.iloc[0].date()} -> {s.iloc[-1].date()} | "
          f"giorno del mese: min {g.min()}, mediana {int(g.median())}, max {g.max()}")
    fuori = ((g < 8) | (g > 24)).mean() * 100
    if fuori > 10:
        print(f"  [!!] {fuori:.0f}% fuori dalla finestra 8-24: NON sembrano date di "
              f"pubblicazione. Fonte/campo da verificare (FLDS<GO>).")
    else:
        print(f"  {etichetta}: profilo coerente con un comunicato mensile "
              f"({100 - fuori:.0f}% nella finestra 8-24)")
    return s


def _copertura(s: pd.Series, etichetta: str) -> None:
    attesi = pd.date_range(pd.Timestamp(CAMPIONE_START), pd.Timestamp.today(), freq="ME")
    have = {(d.year, d.month) for d in s}
    mancanti = [m for m in attesi
                if ((m + pd.offsets.MonthBegin(1)).year,
                    (m + pd.offsets.MonthBegin(1)).month) not in have]
    if mancanti:
        print(f"  [!] {etichetta}: COPERTURA PARZIALE -- {len(mancanti)} mesi senza data "
              f"(primo {mancanti[0].date()}, ultimo {mancanti[-1].date()}). Quei mesi "
              f"verrebbero esclusi da release_grid. Valuta se restare sulla proxy.")
    else:
        print(f"  {etichetta}: copertura COMPLETA su {len(attesi)} mesi")


def _extract_dates(out) -> pd.Series:
    """Estrae le date da qualunque forma restituita da xbbg.bdh per un campo ECO.
    xbbg puo' dare: (a) DataFrame con colonne MultiIndex (ticker, field) e le date nei
    VALORI; (b) date nell'indice. Si prova prima i valori, poi l'indice."""
    if out is None:
        return pd.Series([], dtype="datetime64[ns]")
    if not isinstance(out, pd.DataFrame):
        return pd.to_datetime(pd.Series(out), errors="coerce").dropna()
    got = []
    # (a) tutte le colonne di valori (una sola, ma robusto se ce ne fossero piu')
    for c in out.columns:
        got.append(pd.to_datetime(out[c], errors="coerce"))
    vals = pd.concat(got) if got else pd.Series([], dtype="datetime64[ns]")
    vals = vals.dropna()
    # (b) se i valori non sono date, prova l'indice
    if len(vals) < 12:
        idx = pd.to_datetime(pd.Series(out.index), errors="coerce").dropna()
        if len(idx) > len(vals):
            vals = idx
    return vals


def release_dates(mkt: str) -> pd.Series:
    tick = RELEASE_TICKER[mkt]
    print(f"[{mkt}] Bloomberg {tick}, campo {RELEASE_FIELD} (come =BDH in Excel)")
    blp = _blp()

    # --- DIAGNOSTICA GREZZA su una sola finestra: stampa la struttura ESATTA che xbbg
    #     restituisce, cosi' si vede dove sono le date invece di indovinare.
    print("  [diag] chiamo bdh su 2000-2005 e ispeziono la struttura...")
    probe = blp.bdh(tick, RELEASE_FIELD, "20000101", "20050101")
    print(f"  [diag] type={type(probe).__name__}")
    if isinstance(probe, pd.DataFrame):
        print(f"  [diag] shape={probe.shape}")
        print(f"  [diag] columns={list(probe.columns)}")
        print(f"  [diag] index[:3]={list(probe.index[:3])}")
        print(f"  [diag] dtypes=\n{probe.dtypes.to_string()}")
        print(f"  [diag] head=\n{probe.head(5).to_string()}")
    else:
        print(f"  [diag] repr={repr(probe)[:400]}")
    print("  [diag] ^^^ manda questo blocco: da qui vedo dove sono le date. Mi fermo.")
    raise SystemExit("  [stop diagnostico -- niente file scritto finche' non vedo la struttura]")


if __name__ == "__main__":
    print(">>> r0b release_dates v3 (Bloomberg ECO_RELEASE_DT) <<<")
    salvati = []
    mkts = ["US"] if "--solo-us" in sys.argv else ["US", "UK"]
    for mkt in mkts:
        s = release_dates(mkt)
        _copertura(s, mkt)
        p = CACHE / f"cpi_release_dates_{mkt}.csv"
        s.dt.strftime("%Y-%m-%d").rename("release_date").to_csv(p, index=False)
        salvati.append(p.name)
        print()
    print("salvati: " + ", ".join(salvati))
    print("ora rilancia r2_surprises e maffei_replica: devono stampare")
    print("  '[US] calendario release ESATTO' e '[UK] calendario release ESATTO'.")
    print("controllo incrociato: apri un CSV e confronta una data col comunicato BLS/ONS.")

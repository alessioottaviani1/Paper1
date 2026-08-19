"""r0 - DATI per l'analisi linker: due serie CPI da Bloomberg (leggerissime) e le
aspettative d'inflazione USA dal modello Cleveland Fed (fonte pubblica, zero quota).

Bloomberg (mensile, PX_LAST, incrementale):
  UKRPI Index    -> an_cpi_UK.parquet    (RPI: e' l'indice dei gilt index-linked)
  CPURNSA Index  -> an_cpi_US.parquet    (CPI-U NSA: l'indice dei TIPS)
Web (come il 03 fa con BoE/GSW):
  Cleveland Fed expected inflation 1..30y -> an_expinf_US.parquet
  (Se il download fallisce: scaricare a mano l'xlsx dal sito e metterlo in
   data/raw/cleveland.xlsx -- lo script lo trova da solo.)
Aspettative UK: nessuna fonte pubblica lunga e pulita -> r1 usa il trend CiP come
proxy dichiarata; per una serie survey (es. Consensus) salvare an_expinf_UK.parquet.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import pandas as pd
from config import CACHE
from curves import RAW

CPI_TICKERS = {"UK": "UKRPI Index", "US": "CPURNSA Index",
               "US_SA": "CPI INDX Index",   # CPI-U SA (BLS CUSR0000SA0): sorpresa Maffei
               # CPI NAZIONALI (test canale fiscale r2d): sorpresa del PAESE, non HICP euro.
               "IT_NAT": "ITCPIUNR Index",  # Italy CPI FOI ex-tabacco (indice dei BTPei)
               "FR_NAT": "FRCPXTOB Index",  # France CPI ex-tabacco
               "DE_NAT": "GRCP2000 Index"}  # Germany CPI all-items (controllo: sovrano ~0)
LIQ_TICKERS = {"VIX": "VIX Index", "MOVE": "MOVE Index",         # proxy liquidita'
               "NOISE": "GVLQUSD Index",                          # (Maffei 6.1.1):
               "ONOFF": "G0111Z 10Y BLC2 Curncy"}                 # 4 variabili -> PC1
CLEV_URL = ("https://www.clevelandfed.org/-/media/files/webcharts/"
            "inflationexpectations/inflation-expectations.xlsx")

# ---------------------------------------------------- Bloomberg via _update_wide (bbg.py)
# Usa la cache incrementale COLLAUDATA della pipeline: gestisce i Timestamp (non stringhe),
# distingue ticker gia' presenti (solo giorni nuovi) da ticker NUOVI (scaricati da floor),
# con checkpoint dopo ogni blocco. Questo evita il bug per cui un ticker aggiunto dopo
# partiva da "ieri" invece che dallo storico completo.
try:
    import xbbg  # noqa
    import bbg
    bbg.THROTTLE_SEC = 0.3
    FLOOR = pd.Timestamp("1970-01-01")
    # CPI: un file per mercato (tick2col = {ticker: nome_colonna})
    for mkt, tick in CPI_TICKERS.items():
        df = bbg._update_wide(f"an_cpi_{mkt}", {tick: mkt}, "PX_LAST", floor=FLOOR)
        if df is None or df.empty:
            print(f"  {mkt}: nessun dato per {tick} -- verificare il ticker a terminale")
        else:
            print(f"  an_cpi_{mkt}: {len(df)} mesi ({tick})")
    # liquidita': 4 proxy in un unico an_liq (ticker nuovi scaricati dallo storico completo)
    liq_map = {tick: name for name, tick in LIQ_TICKERS.items()}
    dl = bbg._update_wide("an_liq", liq_map, "PX_LAST", floor=pd.Timestamp("1990-01-01"))
    if dl is not None and not dl.empty:
        cov = {c: (dl[c].dropna().index.min().date(), int(dl[c].notna().sum())) for c in dl.columns}
        print(f"  an_liq: {len(dl)} righe | copertura per proxy: {cov}")
except ImportError:
    print("xbbg non disponibile: salto il download (lanciare sul terminale Bloomberg).")

# ---------------------------------------------------- Cleveland Fed (pubblico)
def _parse_cleveland(x: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(x)
    sheet = next((s for s in xls.sheet_names if "expected" in s.lower()), xls.sheet_names[0])
    raw = xls.parse(sheet)
    dcol = raw.columns[0]
    raw = raw.rename(columns={dcol: "date"}).dropna(subset=["date"])
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).set_index("date")
    cols = {}
    for c in raw.columns:
        s = str(c).lower().replace("year", "").replace("yr", "").strip()
        try:
            cols[c] = float(s.split()[0])
        except Exception:
            pass
    out = raw[list(cols)].rename(columns=cols)
    return out.apply(pd.to_numeric, errors="coerce").dropna(how="all") * (
        100.0 if out.abs().max().max() < 1 else 1.0)     # in %, robusto a decimali

path = CACHE / "an_expinf_US.parquet"
local = RAW / "cleveland.xlsx"
try:
    if local.exists():
        df = _parse_cleveland(local)
    else:
        import urllib.request, shutil
        tmp = CACHE / "_cleveland_tmp.xlsx"
        with urllib.request.urlopen(CLEV_URL) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)          # handle chiuso prima di leggere
        df = _parse_cleveland(tmp)
        try:
            tmp.unlink()
        except OSError:
            pass                                   # su Windows a volte resta agganciato: ignora
    df.to_parquet(path)
    print(f"  an_expinf_US: {len(df)} mesi, orizzonti {sorted(df.columns)[:3]}..."
          f"{sorted(df.columns)[-1]}y (Cleveland Fed)")
except Exception as e:
    print(f"  Cleveland Fed non scaricato ({e}).")
    print(f"  -> scaricare a mano l'xlsx delle inflation expectations e salvarlo come")
    print(f"     {local}  poi rilanciare r0.")
print("fatto.")

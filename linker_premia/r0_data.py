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

CPI_TICKERS = {"UK": "UKRPI Index", "US": "CPURNSA Index"}
LIQ_TICKERS = {"VIX": "VIX Index", "MOVE": "MOVE Index"}   # proxy liquidita' (Maffei)
CLEV_URL = ("https://www.clevelandfed.org/-/media/files/webcharts/"
            "inflationexpectations/inflation-expectations.xlsx")

# ---------------------------------------------------- CPI via Bloomberg (incrementale)
try:
    import xbbg  # noqa
    import bbg
    bbg.THROTTLE_SEC = 0.3
    for mkt, tick in CPI_TICKERS.items():
        path = CACHE / f"an_cpi_{mkt}.parquet"
        start = "19700101"
        old = None
        if path.exists():
            old = pd.read_parquet(path)
            old.index = pd.to_datetime(old.index)
            start = (old.index.max() + pd.Timedelta(days=1)).strftime("%Y%m%d")
        df = bbg._bdh([tick], "PX_LAST", start, pd.Timestamp.today().strftime("%Y%m%d"))
        if df.empty and old is None:
            print(f"  {mkt}: nessun dato per {tick} -- verificare il ticker a terminale")
            continue
        if not df.empty:
            df.columns = [mkt]
            new = df if old is None else pd.concat([old, df])
            new = new[~new.index.duplicated(keep="last")].sort_index()
            new.to_parquet(path)
        n = len(pd.read_parquet(path))
        print(f"  an_cpi_{mkt}: {n} mesi ({tick})")
    # liquidita': VIX + MOVE in un unico an_liq.parquet (mensile, incrementale)
    lpath = CACHE / "an_liq.parquet"
    lold = None
    lstart = "19900101"
    if lpath.exists():
        lold = pd.read_parquet(lpath); lold.index = pd.to_datetime(lold.index)
        lstart = (lold.index.max() + pd.Timedelta(days=1)).strftime("%Y%m%d")
    liq_cols = {}
    for name, tick in LIQ_TICKERS.items():
        d = bbg._bdh([tick], "PX_LAST", lstart, pd.Timestamp.today().strftime("%Y%m%d"))
        if not d.empty:
            d.columns = [name]; liq_cols[name] = d[name]
    if liq_cols:
        lnew = pd.concat(liq_cols.values(), axis=1)
        lnew = lnew if lold is None else pd.concat([lold, lnew])
        lnew = lnew[~lnew.index.duplicated(keep="last")].sort_index()
        lnew.to_parquet(lpath)
    if lpath.exists():
        print(f"  an_liq: {len(pd.read_parquet(lpath))} mesi (VIX, MOVE)")
except ImportError:
    print("xbbg non disponibile: salto il download CPI (lanciare sul terminale Bloomberg).")

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

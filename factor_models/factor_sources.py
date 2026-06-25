"""
factor_sources.py
==================
Read-only data access + configuration for the Paper 1 factor pipeline.

Everything that depends on WHERE the data lives, WHICH sample window, and the
few fixed modelling constants is collected here.  There is NO factor maths in
this file -- only loading.  (The maths lives in build_all_factors.py.)

Raw-data layout (one folder per source, files kept native/untouched):

    data/raw/
        bloomberg/bbg.xlsx     5 sheets: tr_indices, cds, futures,
                                          swaps_fx_infl, vol_options
        fred/                  FRED csv downloads (DATE, <SERIES>)
        ken_french/            Ken French library files, as downloaded
        aqr/                   AQR data-library files (xlsx)
        hsieh/                 David Hsieh / Duke HFRF (PTFS) data
        stambaugh/             Pastor-Stambaugh liquidity file
        hkm/                   He-Kelly-Manela monthly factors csv
        martin/                Ian Martin SVIX series

Importing this module does NOT touch the disk: it only defines constants and
loader functions.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ===========================================================================
# CONFIG  --  all paths, dates and fixed modelling constants in one place
# ===========================================================================

# This file is at  <root>/src/factor_models/factor_sources.py
ROOT      = Path(__file__).resolve().parents[2]
RAW       = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# --- source locations -------------------------------------------------------
BBG_XLSX   = RAW / "bloomberg" / "bbg.xlsx"
FRED_DIR   = RAW / "fred"
FRENCH_DIR = RAW / "ken_french"
AQR_DIR    = RAW / "AQR"
HSIEH_DIR  = RAW / "Hsieh"
STAMB_DIR  = RAW / "Stambaugh"
HKM_DIR    = RAW / "He_Kelly_Manela"
MARTIN_DIR = RAW / "martin"

# --- output (DOWNSTREAM CONTRACT: do not rename/move) -----------------------
# src/pca/00_pca_config.py and others read exactly this file.
OUT_PARQUET  = PROCESSED / "all_factors_monthly.parquet"
OUT_METADATA = PROCESSED / "all_factors_metadata.csv"

# --- sample window (month-end) ---------------------------------------------
SAMPLE_START = "2005-01"      # adjust to your data
SAMPLE_END   = "2025-12"

# --- CDS instruments (exact tickers in the bbg.xlsx `cds` sheet) -------------
# RPV01 (risky annuity) is NO LONGER a fixed constant: it is the Bloomberg
# SW_CNV_RISK (|.|, available from 2009-04) with an ISDA-model fallback before
# that (build_all_factors._cds_rpv01_monthly).  The index roll is detected from
# the ROLLING_SERIES field (not a fixed calendar month); single names do not roll.
CDS_IDX = {
    "MAIN5":  "ITRX EUR CDSI GEN 5Y Corp",
    "MAIN3":  "ITRX EUR CDSI GEN 3Y Corp",
    "CDXIG":  "CDX IG CDSI GEN 5Y Corp",
    "SNRFIN": "SNRFIN CDSI GEN 5Y Corp",
    "XOVER":  "ITRX XOVER CDSI GEN 5Y Corp",
}
PB_EU_5Y = ["SOCGEN CDS EUR SR 5Y D14 Corp", "DB CDS EUR SLA 5Y D14 Corp",
            "HSBC BK CDS EUR SR 5Y D14 Corp", "BARCLAY CDS EUR SR 5Y D14 Corp",
            "BNP CDS EUR SR 5Y D14 Corp"]
PB_US_5Y = ["MS CDS USD SR 5Y D14 Corp", "CINC CDS USD SR 5Y D14 Corp",
            "JPMCC CDS USD SR 5Y D14 Corp", "BOFA CDS USD SR 5Y D14 Corp"]
PB_EU_1Y = [s.replace(" 5Y ", " 1Y ") for s in PB_EU_5Y]
PB_US_1Y = [s.replace(" 5Y ", " 1Y ") for s in PB_US_5Y]

# --- LIBOR / Eurodollar -> SOFR splice month (disclose in data appendix) ----
ED_SOFR_SPLICE = "2023-06"

# --- money-market position duration ----------------------------------------
# Turns a 3M-rate SPREAD change into the excess return of the implementable
# basis package:  r = -d(spread) * MM_DURATION.
MM_DURATION = 0.25      # 3-month deposit duration, in years

# --- default Bloomberg field ------------------------------------------------
BBG_FIELD = "PX_LAST"


# ===========================================================================
# LOADERS  --  one function per source.  Read-only; return raw DataFrames.
# ===========================================================================
# Convention: each loader returns a DataFrame indexed by a DatetimeIndex,
# with the raw columns as downloaded.  build_all_factors.py picks the columns.

def load_bloomberg(sheet):
    """Read one family-sheet from data/raw/Bloomberg/bbg.xlsx (a BDH export).

    sheet in {'rates_fx','tr_indices','cds','futures','vol_options'}.

    Handles the standard Bloomberg BDH layout automatically:
        Start Date | <date>
        End Date   |
        (blank)
                   | <TICKER 1> | <TICKER 2> | ...   <- ticker row
                   | #N/A Requesting Data...         <- (optional junk)
        Dates      | PX_LAST | PX_LAST | ...         <- field row
        <date>     | <value> | <value> | ...         <- data

    Locates the ticker row and the data row dynamically (no hard-coded
    skiprows) and returns a DataFrame indexed by date with one column PER
    TICKER (e.g. 'EURUSD Curncy', 'EUR001M Index').
    """
    raw = pd.read_excel(BBG_XLSX, sheet_name=sheet, header=None)
    col0 = raw.iloc[:, 0].astype(str).str.strip()
    dates_rows = col0[col0.eq("Dates")].index
    if len(dates_rows) == 0:
        raise ValueError(f"sheet '{sheet}': could not find the 'Dates' header row")
    dr = dates_rows[0]                                   # the 'Dates'/'PX_LAST' row
    suffix = ("Curncy", "Index", "Comdty", "Equity", "Govt", "Corp")
    ticker_row = max(                                   # riga con PIÙ celle-ticker, non la prima:
        range(dr),                                      # la riga descrizioni può finire in 'Index'
        key=lambda i: sum(isinstance(v, str) and v.strip().endswith(suffix)
                          for v in raw.iloc[i, 1:]),
    )
    tcols = [j for j in range(1, raw.shape[1])
             if isinstance(raw.iloc[ticker_row, j], str)
             and raw.iloc[ticker_row, j].strip().endswith(suffix)]
    tickers = [raw.iloc[ticker_row, j].strip() for j in tcols]
    dates = pd.to_datetime(raw.iloc[dr + 1:, 0], errors="coerce")
    data = raw.iloc[dr + 1:, tcols].copy()
    data.columns = tickers
    data.index = dates.values
    data = data[dates.notna().values].sort_index()
    data = data.loc[:, ~data.columns.duplicated()]      # drop dup cols (e.g. LUTLTRUU x2)
    return data.apply(pd.to_numeric, errors="coerce")


def load_futures(sheet="futures"):
    """Extractor del foglio futures a 3 campi/contratto (PX_LAST +
    CONVENTIONAL_CTD_FORWARD_FRSK + FUT_CUR_GEN_TICKER); load_bloomberg gestisce
    1 solo campo. Ritorna {'px','frsk','gen'} (DataFrame index=data, col=contratto).
    px/frsk forward-fillati (quote stantie / DV01 lenta); gen resta grezzo (ffill in _roll_days)."""
    raw = pd.read_excel(BBG_XLSX, sheet_name=sheet, header=None)
    TICK, FLD = 3, 5
    fmap = {"PX_LAST": "px", "CONVENTIONAL_CTD_FORWARD_FRSK": "frsk",
            "FUT_CUR_GEN_TICKER": "gen"}
    def _date_col(j):
        for k in range(j, -1, -1):
            if raw.iloc[FLD, k] == "Dates":
                return k
    out = {"px": {}, "frsk": {}, "gen": {}}
    last = None
    for j in range(raw.shape[1]):
        t = raw.iloc[TICK, j]
        if pd.notna(t):
            last = str(t).strip()
        f = raw.iloc[FLD, j]
        if f in fmap and last:
            d = pd.to_datetime(raw.iloc[6:, _date_col(j)], errors="coerce")
            out[fmap[f]][last] = pd.Series(raw.iloc[6:, j].values, index=d).dropna()
    res = {k: pd.DataFrame(v).sort_index() for k, v in out.items()}
    res["px"]   = res["px"].apply(pd.to_numeric, errors="coerce").ffill()
    res["frsk"] = res["frsk"].apply(pd.to_numeric, errors="coerce").ffill()
    return res


def load_cds_fields(sheet="cds"):
    """Read the CDS sheet, which carries THREE fields per instrument: PX_LAST
    (spread), SW_CNV_RISK (Bloomberg risky annuity RPV01, quoted negative) and
    ROLLING_SERIES (on-the-run series number, for index-roll detection).

    load_bloomberg() returns only PX_LAST; this returns all three as a dict of
    DataFrames indexed by date with one column per instrument ticker:
        {'spread': df, 'rpv01_bbg': df (abs), 'series': df}.
    """
    raw = pd.read_excel(BBG_XLSX, sheet_name=sheet, header=None)
    col0 = raw.iloc[:, 0].astype(str).str.strip()
    dr = col0[col0.eq("Dates")].index[0]
    suffix = ("Curncy", "Index", "Comdty", "Equity", "Govt", "Corp")
    trow = max(range(dr), key=lambda i: sum(
        isinstance(v, str) and v.strip().endswith(suffix) for v in raw.iloc[i, 1:]))
    fields = raw.iloc[dr]
    idx = pd.to_datetime(raw.iloc[dr + 1:, 0], errors="coerce")
    spread, rpv01_bbg, series = {}, {}, {}
    last = None
    for j in range(1, raw.shape[1]):
        tk = raw.iloc[trow, j]
        if isinstance(tk, str) and tk.strip().endswith(suffix):
            last = tk.strip()
        f = str(fields.iloc[j]).strip()
        col = pd.to_numeric(raw.iloc[dr + 1:, j], errors="coerce"); col.index = idx.values
        if   f == "PX_LAST"        and last: spread[last]    = col
        elif f == "SW_CNV_RISK"    and last: rpv01_bbg[last] = col.abs()
        elif f == "ROLLING_SERIES" and last: series[last]    = col
    mk = lambda d: pd.DataFrame(d).loc[idx.notna().values].sort_index()
    return {"spread": mk(spread), "rpv01_bbg": mk(rpv01_bbg), "series": mk(series)}


def load_fred(filename, **kwargs):
    """Read a FRED csv from data/raw/fred/ (e.g. 'BAMLCC0A4BBBTRIV.csv').
    FRED csvs have columns DATE, <SERIES>.
    """
    df = pd.read_csv(FRED_DIR / filename, **kwargs)
    return _index_by_date(df)


def load_french(filename, skiprows=None):
    """Read a Ken French library file from data/raw/ken_french/ (as downloaded).

    Layout (verified): 6 preamble lines, then a header row whose first column
    is blank (the YYYYMM date) followed by the factor names, then the MONTHLY
    block, then an 'Annual Factors' block at the bottom.  We keep only rows
    whose first column is a 6-digit YYYYMM (this drops the annual block and any
    text lines automatically), map -99.99 -> NaN, strip the column names, and
    index by calendar month-end.  Used for Europe_5_Factors.csv (Mkt-RF, SMB,
    HML, RMW, CMA, RF) and Europe_MOM_Factor.csv (WML).
    """
    import re
    path = FRENCH_DIR / filename
    if skiprows is None:
        with open(path) as fh:
            _lines = fh.readlines()
        _data0 = next((i for i, ln in enumerate(_lines) if re.match(r"\s*\d{6}\s*,", ln)), None)
        if _data0 is None:
            raise ValueError(f"load_french: nessuna riga YYYYMM trovata in {filename}")
        skiprows = _data0 - 1          # header = riga subito prima del primo YYYYMM
    df = pd.read_csv(path, skiprows=skiprows)
    first = df.columns[0]
    ym = df[first].astype(str).str.strip()
    keep = ym.str.fullmatch(r"\d{6}")               # YYYYMM only -> drops Annual block
    df = df[keep].copy()
    df.index = pd.to_datetime(ym[keep], format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.drop(columns=[first])
    df.columns = [c.strip() for c in df.columns]     # 'Mkt-RF','SMB',... no spaces
    return df.apply(pd.to_numeric, errors="coerce").replace(-99.99, np.nan)


def load_aqr(filename, sheet_name=0):
    """Read an AQR data-library file from data/raw/AQR/ (monthly, as downloaded).

    AQR factor files have a long preamble; the data header row starts with
    'DATE' and is followed by country/region columns (..., 'USA', 'Global',
    'Europe', ...).  Dates are MM/DD/YYYY (month-end).  Locates the header row
    dynamically and returns a DataFrame indexed by calendar month-end with the
    region columns.  Values are AQR's native units (DECIMAL -> x100 for percent).
    """
    raw = pd.read_excel(AQR_DIR / filename, sheet_name=sheet_name, header=None)
    col0 = raw.iloc[:, 0].astype(str).str.strip().str.upper()
    hdr_rows = col0[col0.eq("DATE")].index
    if len(hdr_rows) == 0:
        raise ValueError(f"AQR file '{filename}': could not find the 'DATE' header row")
    dr = hdr_rows[0]
    hdr = [str(x).strip() for x in raw.iloc[dr].tolist()]
    hdr[0] = "DATE"                       # normalizza il nome della colonna-data (file AQR variano: 'Date'/'DATE')
    data = raw.iloc[dr + 1:].copy()
    data.columns = hdr
    data["DATE"] = pd.to_datetime(data["DATE"], errors="coerce")
    data = data.dropna(subset=["DATE"]).set_index("DATE").sort_index()
    data.index = data.index + pd.offsets.MonthEnd(0)        # canonical month-end
    return data.apply(pd.to_numeric, errors="coerce")


def load_hsieh(filename="TF-Fac.xls"):
    """Read the Fung-Hsieh trend-following (PTFS) factors from data/raw/Hsieh/.

    David Hsieh's 'TF-Fac.xls' has a multi-line header; data rows start with a
    YYYYMM date in column 0, followed by the five PTFS factors in fixed order:
    PTFSBD, PTFSFX, PTFSCOM, PTFSIR, PTFSSTK.  Values are DECIMAL monthly
    returns, expressed in the LOCAL currency of each contract and NOT adjusted
    for FX changes (confirmed by D. Hsieh).  Locates the data block dynamically
    and returns a DataFrame indexed by month-end with those five columns.
    """
    raw = pd.read_excel(HSIEH_DIR / filename, sheet_name=0, header=None)
    s = raw.iloc[:, 0].astype(str).str.strip()
    mask = s.str.fullmatch(r"\d{6}").fillna(False)
    if not mask.any():
        raise ValueError(f"Hsieh file '{filename}': no YYYYMM data rows found")
    data = raw[mask].iloc[:, :6].copy()
    data.columns = ["Date", "PTFSBD", "PTFSFX", "PTFSCOM", "PTFSIR", "PTFSSTK"]
    data["Date"] = pd.to_datetime(data["Date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    data = data.set_index("Date").sort_index()
    return data.apply(pd.to_numeric, errors="coerce")


def load_stambaugh(filename="Stambaugh.xlsx"):
    """Read the Pastor-Stambaugh liquidity file from data/raw/Stambaugh/.

    The xlsx holds the original PS text file pasted into a SINGLE column:
    comment lines start with '%', then whitespace-delimited data rows
        YYYYMM | aggregate-liquidity level | innovation (LIQNT, non-traded) |
        traded factor (LIQ_V, 10-1 portfolio return).
    Missing values are coded -99 / -99.99 -> NaN.  Returns a DataFrame indexed
    by month-end with columns ['AGG_LIQ', 'LIQNT', 'LIQ_V'] (DECIMAL).
    """
    col = pd.read_excel(STAMB_DIR / filename, sheet_name=0, header=None).iloc[:, 0].astype(str)
    recs = []
    for v in col:
        p = v.split()
        if len(p) >= 4 and p[0].strip().isdigit() and len(p[0].strip()) == 6:
            recs.append(p[:4])
    out = pd.DataFrame(recs, columns=["Month", "AGG_LIQ", "LIQNT", "LIQ_V"])
    idx = pd.to_datetime(out["Month"], format="%Y%m") + pd.offsets.MonthEnd(0)
    out = out.drop(columns=["Month"]).set_index(idx).apply(pd.to_numeric, errors="coerce")
    return out.replace([-99, -99.0, -99.99], np.nan)


def load_hkm(filename):
    """Read the He-Kelly-Manela intermediary factors csv from data/raw/He_Kelly_Manela/.

    Columns: yyyymm, intermediary_capital_ratio, intermediary_capital_risk_factor
    (the NON-traded capital-ratio innovation), intermediary_value_weighted_
    investment_return (the TRADED value-weighted equity return of the primary-
    dealer holding companies), intermediary_leverage_ratio_squared.  Returns a
    DataFrame indexed by month-end (values DECIMAL).
    """
    df = pd.read_csv(HKM_DIR / filename)
    idx = pd.to_datetime(df["yyyymm"].astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.drop(columns=["yyyymm"]).set_index(idx).apply(pd.to_numeric, errors="coerce")


def load_martin(filename, **kwargs):
    """Read Ian Martin's SVIX series from data/raw/martin/.
    SVIX is the option-implied risk-neutral vol; SVIX^2 is its variance.
    """
    if str(filename).lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(MARTIN_DIR / filename, **kwargs)
    else:
        df = pd.read_csv(MARTIN_DIR / filename, **kwargs)
    return _index_by_date(df)


# ===========================================================================
# helpers
# ===========================================================================

def _index_by_date(df, date_col=None):
    """Set a sorted DatetimeIndex from a date column (first column by default)."""
    if date_col is None:
        date_col = df.columns[0]
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df.dropna(subset=[date_col]).set_index(date_col).sort_index()

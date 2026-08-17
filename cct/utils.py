"""Funzioni condivise per il package cct (base CCT-BTP)."""
import numpy as np, pandas as pd
from dateutil.relativedelta import relativedelta
from config import *

# --------------------------- date e anagrafica -------------------------------
def excel_or_str_dates(s: pd.Series) -> pd.Series:
    """
    Normalizza una colonna data che puo' arrivare in tre forme diverse dallo stesso
    export Bloomberg: gia' datetime64, seriale Excel numerico, o stringa dd/mm/yyyy.
    Il controllo sul dtype viene PRIMA di tutto: su una colonna gia' datetime64,
    pd.to_numeric restituisce i nanosecondi dall'epoca (~1.7e18) e interpretarli come
    seriali Excel manda in overflow (OutOfBoundsDatetime).
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")
    num = pd.to_numeric(s, errors="coerce")
    # seriale Excel plausibile: 10000 = 1927-05-18, 64000 = 2075-03-15
    is_serial = num.notna() & num.between(10000, 64000)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if is_serial.any():
        out[is_serial] = pd.to_datetime(num[is_serial], unit="D", origin="1899-12-30")
    rest = ~is_serial
    if rest.any():
        out[rest] = pd.to_datetime(s[rest], errors="coerce", dayfirst=True)
    return out

def parse_coupon_desc(desc):
    """
    Estrae la cedola da 'BTPS <cedola> <MM/DD/YY>'. La cedola puo' essere decimale
    ('3.85'), intera ('3'), o mista con frazione ('3 1/4', '1/2'). Ci si ancora alla
    data finale per non confondere la frazione con il giorno/mese.
    """
    import re
    s = str(desc).strip()
    m = re.match(r"^BTPS\s+(.+?)\s+\d{1,2}/\d{1,2}/\d{2,4}\s*$", s)
    if not m: return np.nan
    c = m.group(1).strip()
    mm = re.match(r"^(\d+)\s+(\d+)/(\d+)$", c)
    if mm: return float(mm.group(1)) + float(mm.group(2)) / float(mm.group(3))
    mm = re.match(r"^(\d+)/(\d+)$", c)
    if mm: return float(mm.group(1)) / float(mm.group(2))
    try: return float(c)
    except ValueError: return np.nan

def load_static() -> dict:
    """Legge i tre fogli e restituisce anagrafiche pulite con regime dei CCT."""
    out = {}
    for sheet, isscol in [("CCTS", "Pricing Date"), ("BOTS", "Pricing Date"), ("BTPS", "Issue Date")]:
        d = pd.read_excel(FILE_STATIC, sheet_name=sheet)
        d.columns = [str(c).strip() for c in d.columns]
        # il foglio BTPS ha la prima colonna rinominata 'i' in alcune versioni
        if "Issuer Name" not in d.columns:
            d = d.rename(columns={d.columns[0]: "Issuer Name"})
        d["maturity"] = excel_or_str_dates(d["Maturity"])
        d["issue"]    = excel_or_str_dates(d[isscol])
        d["isin"]     = d["ISIN"].astype(str).str.strip()
        d["amt"]      = pd.to_numeric(d.get("Amt Issued"), errors="coerce")
        # Bloomberg ID: e' l'identificatore da usare nelle richieste, non l'ISIN
        d["bb_id"]    = d["Bloomberg ID"].astype(str).str.strip() if "Bloomberg ID" in d.columns else None
        d["desc"]     = d.get("Security Description")
        d = d.dropna(subset=["maturity", "isin"]).drop_duplicates("isin")
        out[sheet] = d.sort_values("maturity").reset_index(drop=True)

    # --- cedola dei BTP -------------------------------------------------------
    # Fonte primaria: campo CPN di Bloomberg (static_bbg.csv), che e' autorevole.
    # Fallback: parsing della Security Description, che pero' scrive la cedola in
    # FRAZIONE in 252 casi su 365 ("BTPS 3 1/4 07/15/32"). Un regex ingenuo su [\d.]+
    # legge 3.00 invece di 3.25; uno che cerca "n m/k" senza ancoraggio legge
    # "BTPS 3 10/01/29" come 3+10/1=13. Il parser si ancora alla DATA MM/DD/YY e
    # prende come cedola tutto cio' che la precede.
    B = out["BTPS"]
    B["coupon"] = B["desc"].apply(parse_coupon_desc)
    f = FILE_STATIC.parent / "processed" if False else None
    try:
        sb = pd.read_csv(PROC / "static_bbg.csv", index_col=0)
        sb.columns = [str(c).upper() for c in sb.columns]
        if "CPN" in sb.columns:
            cpn = pd.to_numeric(sb["CPN"], errors="coerce")
            B["coupon_bbg"] = B["isin"].map(cpn)
            n_bbg = B["coupon_bbg"].notna().sum()
            B["coupon"] = B["coupon_bbg"].fillna(B["coupon"])
            print(f"  [cedole] {n_bbg}/{len(B)} da Bloomberg CPN, resto da descrizione")
    except Exception:
        pass
    out["BTPS"] = B

    # regime dei CCT
    C = out["CCTS"]
    C["regime"] = np.where(C["issue"] >= pd.Timestamp(CCTEU_CUTOFF), "CCTeu", "CCT-BOT")
    C.loc[C["issue"].isna(), "regime"] = "unknown"
    out["CCTS"] = C
    return out

# --------------------------- flussi di cassa ---------------------------------
def coupon_dates(issue, maturity, freq=CPN_FREQ) -> list:
    """Date cedolari a ritroso dalla scadenza: convenzione dei titoli di stato."""
    step = 12 // freq
    dates, d = [], pd.Timestamp(maturity)
    while d > pd.Timestamp(issue):
        dates.append(d)
        d = d - relativedelta(months=step)
    return sorted(dates)

# --------------------------- curva e convenzioni ------------------------------
# NOTA: le funzioni sottostanti (svensson_zero, fit_svensson, year_fractions) sono la
# versione scritta a mano, tenuta solo come riferimento e per i test senza QuantLib.
# La pipeline usa qlutils.py, che poggia su QuantLib -- gia' nei requirements del
# progetto -- con le convenzioni ufficiali ActualActual(ISMA) e Actual360 invece di
# approssimazioni. Vedi qlutils.accrual_fraction (il fit QuantLib e' stato rimosso: 06_curve).

# --------------------------- output ------------------------------------------
def save_txt(name, lines):
    p = RES / name
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"[saved] {p}")

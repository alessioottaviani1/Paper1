"""r0c - Le date di rilascio coincidono fra i ticker CPI?

PERCHE'. r0b prende la data US da CPI CHNG Index, mentre la specifica di Maffei vuole la
data dello STESSO ticker della serie che entra nella sorpresa: CPI INDX Index (CPI-U SA).
Quel ticker pero' torna VUOTO su ECO_RELEASE_DT, perche' e' il livello dell'indice e non
l'evento del calendario economico. La deviazione e' innocua SE il BLS pubblica SA, NSA e
tendenziale nello stesso comunicato -- il che e' vero per conoscenza istituzionale, ma qui
si verifica sui dati invece di assumerlo.

Il primo run di r0b mostrava le stesse prime sei date su CPI CHNG, CPURNSA e CPI YOY.
Questo script fa il confronto su TUTTE le date e riporta le eventuali divergenze.

Verifica anche i BUCHI: r0b segnala 2 mesi senza rilascio nel 2025 per gli USA. Se sono
reali (sospensione delle pubblicazioni), vanno gestiti a valle e non qui -- ma vanno
identificati con precisione, perche' release_grid escluderebbe quei mesi in silenzio.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import pandas as pd

TICKERS_US = ["CPI CHNG Index", "CPURNSA Index", "CPI YOY Index", "CPI INDX Index"]
TICKERS_UK = ["UKRPI Index", "UKRPYOY Index"]
CAMPO, D0, D1 = "ECO_RELEASE_DT", "2004-01-01", "2026-12-31"

def _blp():
    from xbbg import blp
    return blp

def _to_pandas(o):
    if o is None: return None
    if not isinstance(o, pd.DataFrame):
        for a in ("to_pandas","to_native"):
            if hasattr(o,a):
                try:
                    o = getattr(o,a)()
                    if hasattr(o,"to_pandas"): o = o.to_pandas()
                    if isinstance(o,pd.DataFrame): break
                except Exception: pass
    return o if isinstance(o,pd.DataFrame) and not o.empty else None

def dates_of(tick):
    """Date di rilascio da un ticker. YYYYMMDD arriva come float: '20040220.0'."""
    raw = _to_pandas(_blp().bdh(tick, CAMPO, D0, D1))
    if raw is None: return pd.Series(dtype="datetime64[ns]")
    c = {str(x).lower(): x for x in raw.columns}
    if "value" not in c: return pd.Series(dtype="datetime64[ns]")
    v = pd.to_numeric(raw[c["value"]], errors="coerce").dropna().astype("int64").astype(str)
    d = pd.to_datetime(v, errors="coerce", format="%Y%m%d").dropna()
    return pd.Series(sorted(set(d)))

if __name__ == "__main__":
    print(">>> r0c confronto dei calendari di rilascio <<<")
    for mkt, ticks in [("US", TICKERS_US), ("UK", TICKERS_UK)]:
        print(f"\n{'='*70}\n{mkt}\n{'='*70}")
        serie = {}
        for t in ticks:
            try:
                s = dates_of(t)
                serie[t] = s
                print(f"  {t:20s} {len(s):>4} date  "
                      + (f"{s.min().date()} -> {s.max().date()}" if len(s) else "(vuoto)"))
            except Exception as e:
                print(f"  {t:20s} errore: {str(e)[:50]}")
        pieni = {k: v for k, v in serie.items() if len(v) > 24}
        if len(pieni) < 2:
            print("  meno di due ticker con dati: confronto impossibile"); continue
        base_t = list(pieni)[0]; base = set(pieni[base_t])
        print(f"\n  confronto con {base_t} ({len(base)} date):")
        for t, s in list(pieni.items())[1:]:
            ss = set(s)
            only_b, only_s = sorted(base - ss), sorted(ss - base)
            comuni = len(base & ss)
            print(f"    {t:20s} in comune {comuni:>4} | solo in {base_t.split()[0]}: {len(only_b)}"
                  f" | solo qui: {len(only_s)}")
            if only_b[:3]: print(f"       es. solo base : {[str(x.date()) for x in only_b[:3]]}")
            if only_s[:3]: print(f"       es. solo altro: {[str(x.date()) for x in only_s[:3]]}")
        if all(set(v) == base for v in pieni.values()):
            print(f"\n  --> i calendari COINCIDONO su tutte le {len(base)} date.")
            print("      Prendere la data da CPI CHNG invece che da CPI INDX e' innocuo:")
            print("      il comunicato e' lo stesso. La docstring di r0b va aggiornata.")
        else:
            print("\n  --> i calendari DIVERGONO: la scelta del ticker conta e va motivata.")

        # buchi mensili
        s = pieni[base_t]
        per = s.dt.to_period("M")
        mesi = pd.period_range(per.min(), per.max(), freq="M")
        buchi = [str(m) for m in mesi if m not in set(per)]
        doppi = per.value_counts(); doppi = doppi[doppi > 1]
        print(f"\n  mesi senza rilascio: {len(buchi)} {buchi[:8]}")
        if len(doppi):
            print(f"  mesi con DUE rilasci: {list(doppi.index.astype(str))}")
            print("     [!] r0b tiene solo il PRIMO di ogni mese: il secondo andrebbe perso.")
            print("         Dopo una sospensione i recuperi si accavallano: da gestire.")

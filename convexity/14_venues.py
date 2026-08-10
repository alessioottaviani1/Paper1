"""14 - TRIANGOLAZIONE A TRE VENUE: la curva e' l'outlier, o e' la nostra misura a esserlo?

L'obiezione piu' insidiosa al paper e': ``sigma_BE non co-muove con la vol delle swaption perche'
sigma_BE e' una misura sporca, non perche' i mercati siano segmentati''. Finora rispondevamo con
robustezze (Spearman, ex-crisi, superficie). Qui c'e' la risposta decisiva: una TERZA venue.

La vol implicita ATM 30 giorni sui FUTURE del decennale (RX1 = Bund, TY1 = Treasury) prezza la
stessa volatilita' del tasso lungo, in un mercato diverso sia dalla curva sia dalle swaption
(opzioni su futures, borsa, clientela di dealer e CTA). Dal 2004, dal foglio `futures` di bbg.xlsx.

IL TEST, e perche' e' decisivo:
  - se le DUE VENUE OPZIONALI (swaption e future) co-muovono FORTE tra loro, ma NESSUNA delle due
    co-muove con la curva -> la curva e' l'outlier, e non e' un problema di misura: due misure
    indipendenti di vol delle opzioni concordano tra loro e divergono entrambe dalla curva;
  - se invece la vol dei future non co-muove NEMMENO con le swaption, allora il problema e' la
    misura (o la nostra sigma_BE, o l'appaiamento) e il claim di segmentazione si indebolisce.

NB sull'unita': la vol dei future e' PRICE vol (%), la swaption e' YIELD vol (bp). Il test e' sulle
CORRELAZIONI DELLE VARIAZIONI, che sono invarianti a un riscalamento costante (price vol ~ yield vol
x duration), quindi non serve stimare la duration del future: il confronto e' gia' lecito.

Output: output/convexity/results/14_venues.txt
"""
import pandas as pd, numpy as np, glob, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from config import *
from utils import save_txt, load_vols

print("== 14 triangolazione a tre venue ==")
sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
IV  = load_vols()
L = []; P = L.append
P("=== 14 TRE VENUE: curva (sigma_BE) | swaption (sigma_IV) | future 10Y (IV ATM 30d) ===")

# --- vol implicita sui future dal foglio 'futures' di bbg.xlsx ---
cand = sorted(glob.glob(str(RAW/"**"/"bbg.xlsx"), recursive=True))
if not cand:
    P("bbg.xlsx non trovato sotto raw/ -- pannello saltato (serve il foglio 'futures').")
else:
    f = pd.read_excel(cand[0], sheet_name="futures", header=None, skiprows=6,
                      usecols=[45, 46, 47], names=["Date", "IV_RX_30D", "IV_TY_30D"])
    f["Date"] = pd.to_datetime(f["Date"], errors="coerce")
    f = (f.dropna(subset=["Date"]).set_index("Date")
           .apply(pd.to_numeric, errors="coerce").sort_index())
    f = f[~f.index.duplicated(keep="last")]
    FUT = {"EUR": f["IV_RX_30D"].dropna().resample("ME").last(),   # Bund  -> mercati EUR
           "USD": f["IV_TY_30D"].dropna().resample("ME").last()}   # UST   -> mercati USD
    P(f"future: RX1 (Bund) e TY1 (UST), IV ATM 30d, {f.index.min().date()} -> {f.index.max().date()}")

    def swaption(ccy, exp="3M", ten="10Y"):
        s = IV.get((ccy, exp, ten, "NORM"))
        return None if s is None else s.resample("ME").last()

    # (1) le due venue OPZIONALI si parlano tra loro?
    P("")
    P("[1] LE DUE VENUE OPZIONALI TRA LORO: swaption 3Mx10Y vs future 10Y (stessa valuta)")
    P(f"{'valuta':8}{'corr livelli':>14}{'corr Delta':>12}{'N':>6}")
    optcorr = {}
    for ccy in ("USD", "EUR"):
        sw = swaption(ccy)
        if sw is None or ccy not in FUT: continue
        al = pd.concat([sw, FUT[ccy]], axis=1).dropna()
        if len(al) < 60: continue
        d = al.diff().dropna()
        cl = al.iloc[:,0].corr(al.iloc[:,1]); cd = d.iloc[:,0].corr(d.iloc[:,1])
        optcorr[ccy] = cd
        P(f"{ccy:8}{cl:+14.2f}{cd:+12.2f}{len(al):6d}")

    # (2) ciascuna venue opzionale contro la CURVA, mercato per mercato
    P("")
    P("[2] CIASCUNA VENUE OPZIONALE CONTRO LA CURVA (corr delle variazioni)")
    P(f"{'mercato':9}{'vs swaption':>13}{'vs future':>11}{'N fut':>7}")
    for mkt in sbe.columns:
        ccy = IVMAP.get(mkt)
        if ccy is None: continue
        b = sbe[mkt].dropna()
        sw = swaption(ccy)
        futccy = "USD" if ccy in ("USD",) else ("EUR" if ccy in ("EUR",) else None)
        c_sw = c_fu = np.nan; n = 0
        if sw is not None:
            a = pd.concat([b, sw], axis=1).dropna().diff().dropna()
            if len(a) > 40: c_sw = a.iloc[:,0].corr(a.iloc[:,1])
        if futccy is not None:
            a = pd.concat([b, FUT[futccy]], axis=1).dropna().diff().dropna()
            if len(a) > 40: c_fu = a.iloc[:,0].corr(a.iloc[:,1]); n = len(a)
        P(f"{mkt:9}{c_sw:+13.2f}{c_fu:+11.2f}{n:7d}")

    # (3) il verdetto in una riga per valuta
    P("")
    P("[3] VERDETTO")
    for ccy, mkts in (("USD", ["USDswap", "USTgovt"]), ("EUR", ["EUR", "DEgovt"])):
        if ccy not in optcorr: continue
        sw = swaption(ccy)
        cs = []
        for m in mkts:
            if m not in sbe.columns: continue
            a = pd.concat([sbe[m].dropna(), FUT[ccy]], axis=1).dropna().diff().dropna()
            b2 = pd.concat([sbe[m].dropna(), sw], axis=1).dropna().diff().dropna()
            if len(a) > 40 and len(b2) > 40:
                cs.append((m, b2.iloc[:,0].corr(b2.iloc[:,1]), a.iloc[:,0].corr(a.iloc[:,1])))
        if not cs: continue
        P(f"  {ccy}: le due venue opzionali correlano {optcorr[ccy]:+.2f} TRA LORO;")
        for m, c1, c2 in cs:
            P(f"       la curva {m:8} correla {c1:+.2f} con le swaption e {c2:+.2f} coi future.")
    P("")
    P("  Se il primo numero e' ALTO e i successivi sono ~0, la curva e' l'outlier: due misure")
    P("  indipendenti della vol delle opzioni concordano tra loro e divergono entrambe dalla")
    P("  curva. L'obiezione 'e' la vostra sigma_BE a essere sporca' non regge, perche' una misura")
    P("  sporca non potrebbe divergere in modo SISTEMATICO da due venue che concordano.")

save_txt("14_venues.txt", L); print("\n".join(L))

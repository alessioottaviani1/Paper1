"""15 - I FATTORI DEL PRIMO PAPER: la convessita' e' spannata da term, curva o pendenza?

Il primo paper ha 73 candidati TRADABLE. Cinque famiglie sono direttamente pertinenti a questo
paper, e finora NON erano state testate. Le costruiamo dai grezzi (stesso metodo di
00_build_all_factors.py) e le usiamo come regressori dei rendimenti della strategia.

[A] TERM (il piu' richiesto dai referee). TERM_US = LUTLTRUU (long UST total return) meno RF;
    TERM_EU = I01656EU; GOVT_EU = LETGTREU; GLOBAL_TERM = H00023EU. Sono i fattori term di
    Fama-French in versione tradable. Domanda: la strategia e' solo premio a termine travestito?

[B] CURVATURA (il controllo piu' MIRATO che esista). CURV_2S5S10S = butterfly DV01-neutrale sui
    futures, long belly: TU1/FV1/TY1 (US) e DU1/OE1/RX1 (EU). E' il parente tradable piu' stretto
    del fly 2/10/30: se l'alpha sopravvive a QUESTO, la strategia non e' timing di curvatura.

[C] PENDENZA. SLOPE_2S10S = steepener DV01-matched (TU1/TY1, DU1/RX1); SLOPE_10S30S_EU (RX1/UB1).

[D] LIVELLO. Rendimento del future 10Y (TY1, RX1): la duration pura.

Costruzione dei fattori curva: rendimento in eccesso dei futures pesato DV01, con i DV01 impliciti
nelle duration nominali dei contratti (TU~1.9, FV~4.3, TY~6.5, US/UB~17; DU~1.9, OE~4.6, RX~8.0).
I pesi esatti contano poco: l'obiettivo e' che il regressore SPANNI la variazione di curvatura,
non replicarla al centesimo.

Output: output/convexity/results/15_paper1_factors.txt
"""
import pandas as pd, numpy as np, glob, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from config import *
from utils import nw_t, save_txt, load_govt_extras, load_rf

print("== 15 fattori del primo paper ==")
STRAT = pd.read_csv(PROC/"strat_monthly.csv", index_col=0, parse_dates=True)
sbe   = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
L = []; P = L.append
P("=== 15 FATTORI DEL PRIMO PAPER: term, curvatura, pendenza, livello ===")

cand = sorted(glob.glob(str(RAW/"**"/"bbg.xlsx"), recursive=True))
if not cand:
    P("bbg.xlsx non trovato sotto raw/ -- script saltato")
    save_txt("15_paper1_factors.txt", L); print("\n".join(L)); raise SystemExit

BB = cand[0]
def sheet_series(sh, tickrow=3, datarow=6):
    f = pd.read_excel(BB, sheet_name=sh, header=None)
    hdr = [str(x).strip() for x in f.iloc[tickrow].tolist()]
    dts = pd.to_datetime(f.iloc[datarow:, 0], errors="coerce")
    out = {}
    for j in range(1, f.shape[1]):
        t = hdr[j]
        if t in ("nan", ""): continue
        v = pd.to_numeric(f.iloc[datarow:, j], errors="coerce"); v.index = dts
        v = v.dropna()
        if v.index.duplicated().any(): v = v[~v.index.duplicated(keep="last")]
        out.setdefault(t, v.sort_index())
    return out

TR  = sheet_series("tr_indices")
FUT = sheet_series("futures")
mret = lambda s: s.resample("ME").last().pct_change()*100    # rendimento mensile in %

F = {}
EX = load_govt_extras()
# ---- [A] TERM NATIVI per valuta (indici Long/10+ unhedged in valuta locale), meno RF locale ----
P("")
P("[costruzione TERM] indice Long/10+ in valuta locale, meno RF locale:")
for ccy, tk in TR_TERM.items():
    if tk not in EX: P(f"    {ccy}: {tk} ASSENTE"); continue
    r = mret(EX[tk])
    rf, src = load_rf(ccy, EX)
    if rf is not None: r = (r - rf.reindex(r.index)).dropna()
    F[f"TERM_{ccy}"] = r
    P(f"    TERM_{ccy:4} = {tk:18} - RF({src})   [{r.index.min().date()} -> {r.index.max().date()}]")
for name, tk in [("GOVT_EU","LETGTREU Index"), ("GLOBAL_TERM","H00023EU Index"),
                 ("R2_EU","MLTAGB2E Index"), ("R5_EU","MLTAGB5E Index"), ("R10_EU","MLTAG10E Index")]:
    if tk in TR: F[name] = mret(TR[tk])
for ccy, tk in TR_AGG.items():
    if tk in EX: F[f"GOVT_{ccy}"] = mret(EX[tk])
# ---- [B],[C],[D] curva dai futures ----
DUR = {"TU1 Comdty":1.9, "FV1 Comdty":4.3, "TY1 Comdty":6.5, "UB1 Comdty":17.0,
       "DU1 Comdty":1.9, "OE1 Comdty":4.6, "RX1 Comdty":8.0}
def fret(tk): return mret(FUT[tk]) if tk in FUT else None
def dv01_fly(w_short, belly, l_short, l_long):
    """butterfly DV01-neutrale: long belly, short le ali pesate per duration."""
    b, s, lg = fret(belly), fret(l_short), fret(l_long)
    if any(x is None for x in (b, s, lg)): return None
    ds, db, dl = DUR[l_short], DUR[belly], DUR[l_long]
    ws = 0.5*db/ds; wl = 0.5*db/dl
    return (b - ws*s - wl*lg).dropna()
def dv01_steep(a, b_):
    ra, rb = fret(a), fret(b_)
    if ra is None or rb is None: return None
    return (ra - (DUR[a]/DUR[b_])*rb).dropna()

for nm, v in [("CURV_2S5S10S_US", dv01_fly(None, "FV1 Comdty", "TU1 Comdty", "TY1 Comdty")),
              ("CURV_2S5S10S_EU", dv01_fly(None, "OE1 Comdty", "DU1 Comdty", "RX1 Comdty")),
              ("SLOPE_2S10S_US",  dv01_steep("TU1 Comdty", "TY1 Comdty")),
              ("SLOPE_2S10S_EU",  dv01_steep("DU1 Comdty", "RX1 Comdty")),
              ("SLOPE_10S30S_EU", dv01_steep("RX1 Comdty", "UB1 Comdty")),
              ("LEVEL_US", fret("TY1 Comdty")), ("LEVEL_EU", fret("RX1 Comdty"))]:
    if v is not None: F[nm] = v
for ccy, tk in FUT_LOCAL.items():           # future governativi nativi (gilt, JGB)
    if tk in EX: F[f"LEVEL_{ccy}"] = mret(EX[tk])

P(f"fattori costruiti ({len(F)}): {sorted(F)}")

def ols_nw(y, X, lag=6):
    Xv, yv = np.asarray(X, float), np.asarray(y, float)
    b = np.linalg.lstsq(Xv, yv, rcond=None)[0]; e = yv - Xv@b
    A = np.linalg.inv(Xv.T@Xv); S = (e[:,None]*Xv).T@(e[:,None]*Xv)
    for l in range(1, lag+1):
        w = 1-l/(lag+1); u=(e[l:,None]*Xv[l:]); v=(e[:-l,None]*Xv[:-l]); G=u.T@v; S += w*(G+G.T)
    V = A@S@A
    yhat = Xv@b; r2 = 1 - np.var(yv-yhat)/np.var(yv)
    return b, b/np.sqrt(np.diag(V)), r2

# mappa mercato -> blocco di fattori della sua area
BLOCKS = {   # ogni mercato con i fattori della SUA valuta (term e livello ora NATIVI)
  "USDswap": ["TERM_USD","CURV_2S5S10S_US","SLOPE_2S10S_US","LEVEL_US"],
  "USTgovt": ["TERM_USD","CURV_2S5S10S_US","SLOPE_2S10S_US","LEVEL_US"],
  "EUR":     ["TERM_EUR","CURV_2S5S10S_EU","SLOPE_2S10S_EU","SLOPE_10S30S_EU","LEVEL_EU"],
  "DEgovt":  ["TERM_EUR","CURV_2S5S10S_EU","SLOPE_2S10S_EU","SLOPE_10S30S_EU","LEVEL_EU"],
  "GBP":     ["TERM_GBP","CURV_2S5S10S_EU","SLOPE_2S10S_EU","LEVEL_GBP"],
  "UKgovt":  ["TERM_GBP","CURV_2S5S10S_EU","SLOPE_2S10S_EU","LEVEL_GBP"],
  "JPY":     ["TERM_JPY","CURV_2S5S10S_US","SLOPE_2S10S_US","LEVEL_JPY"],
  "JPgovt":  ["TERM_JPY","CURV_2S5S10S_US","SLOPE_2S10S_US","LEVEL_JPY"],
}

def run(title, picker):
    P(""); P(title)
    P(f"{'mercato':9}{'alpha':>9}{'[t]':>7}{'R2':>7}{'N':>6}   regressori")
    for mkt in STRAT.columns:
        names = [n for n in picker(mkt) if n in F]
        if not names: continue
        y = STRAT[mkt].dropna()
        al = pd.concat([y] + [F[n] for n in names], axis=1).dropna()
        if len(al) < 60: continue
        X = np.column_stack([np.ones(len(al)), al.iloc[:,1:].values])
        b, t, r2 = ols_nw(al.iloc[:,0], X)
        P(f"{mkt:9}{b[0]:9.2f}{t[0]:7.1f}{r2:7.2f}{len(al):6d}   {','.join(names)}")

run("[A] ALPHA controllando SOLO il TERM (la strategia e' premio a termine travestito?)",
    lambda m: [n for n in BLOCKS.get(m, []) if "TERM" in n])
run("[B] ALPHA controllando SOLO la CURVATURA (butterfly su futures: il controllo piu' mirato)",
    lambda m: [n for n in BLOCKS.get(m, []) if n.startswith("CURV")])
run("[C] ALPHA controllando TUTTO il blocco tassi (term + curvatura + pendenza + livello)",
    lambda m: BLOCKS.get(m, []))

# quanto la curvatura tradabile spiega sigma_BE?
P("")
P("[D] Delta-sigma_BE ~ Delta-curvatura tradabile (il fly su futures spiega la nostra misura?)")
P(f"{'mercato':9}{'b_CURV':>10}{'[t]':>7}{'R2':>7}{'N':>6}")
for mkt in sbe.columns:
    nm = [n for n in BLOCKS.get(mkt, []) if n.startswith("CURV")]
    if not nm: continue
    al = pd.concat([sbe[mkt].dropna(), F[nm[0]]], axis=1).dropna()
    if len(al) < 60: continue
    d = pd.concat([al.iloc[:,0].diff(), al.iloc[:,1]], axis=1).dropna()
    X = np.column_stack([np.ones(len(d)), d.iloc[:,1].values])
    b, t, r2 = ols_nw(d.iloc[:,0], X)
    P(f"{mkt:9}{b[1]:10.3f}{t[1]:7.1f}{r2:7.2f}{len(d):6d}")
P("    lettura: R2 basso => la nostra sigma_BE NON e' il butterfly tradabile: misura il PREZZO")
P("    della convessita' (carry-roll), non il suo rendimento realizzato.")

save_txt("15_paper1_factors.txt", L); print("\n".join(L))

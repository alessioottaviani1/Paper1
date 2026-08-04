"""09 - MECCANISMO (C1/S1): lo scollamento di sigma_BE e' pricing VINCOLATO o rumore?

Il test decisivo del paper. La verifica di agosto-2026 ha mostrato che sigma_IV traccia la vol
realizzata (corr +0.32..+0.57) mentre sigma_BE NO (USD +0.03, EUR -0.11, GBP -0.40). Due letture:
  (a) RUMORE: la curvatura e' mossa da premi a termine / domanda-offerta -> il C3 ~ 0 sarebbe banale;
  (b) PRICING VINCOLATO: la curva prezza la convessita' per vincoli di BILANCIO, non per aspettative
      di vol -> lo scollamento E' il fenomeno, e il C3 ~ 0 e' segmentazione.
Discriminante: se (b), lo scollamento deve caricare SISTEMATICAMENTE sugli stati di stress dei dealer
(CDS composite, MOVE) - il rumore no. Tre pannelli:

  A) Delta-sigma_BE su Delta-RV (vol realizzata) e Delta-Stress, congiuntamente.
     (b) predice: coefficiente su Stress significativo e NEGATIVO (stress -> curva cheapens),
     con RV non dominante. (a) predice: nessuno dei due sistematico.
  B) Delta-gap G = s2_TRAIL - s2_BE su Delta-Stress: predizione (b) beta > 0 (stress -> gap si apre).
  C) Confronto di caricamento: lo stesso stress su Delta-sigma_IV (pool opzioni). Se il canale e'
     il bilancio della CURVA, il caricamento deve essere ASIMMETRICO (forte su BE, debole su IV).
     Simmetria -> favorisce il fattore comune (l'alternativa nominata nel proposal).

Output: output/convexity/results/09_mechanism.txt
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import nw_t, save_txt, load_legs_mid, load_vols, load_dealer_cds, load_market_states

print("== 09 mechanism ==")
sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
s2  = pd.read_csv(PROC/"s2be_monthly.csv",  index_col=0, parse_dates=True)
mid = load_legs_mid()
IV  = load_vols()
CDS, _, _ = load_dealer_cds()
MS  = load_market_states()
MOVE = MS["MOVE"].resample("ME").last() if "MOVE" in MS else None

def iv_series(ccy, exp="3M", ten="10Y", fam="NORM"):
    for (c, e, t, f), s in IV.items():
        if c == ccy and e == exp and t == ten and f == fam: return s.resample("ME").last()
    return None

def ols_nw(y, X, L=6):
    """OLS con errori Newey-West; ritorna (coef, t) per ogni regressore (X gia' con costante)."""
    Xv, yv = X.values, y.values
    b = np.linalg.lstsq(Xv, yv, rcond=None)[0]
    e = yv - Xv @ b
    XtX_inv = np.linalg.inv(Xv.T @ Xv)
    S = (e[:, None] * Xv).T @ (e[:, None] * Xv)
    for l in range(1, L+1):
        w = 1 - l/(L+1)
        u = (e[l:, None] * Xv[l:]); v = (e[:-l, None] * Xv[:-l])
        G = u.T @ v
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    return b, b/se

L = []; P = L.append
P("=== 09 MECCANISMO (C1/S1): lo scollamento di sigma_BE e' bilancio o rumore? ===")
P("")
P("[A] Delta-sigma_BE ~ Delta-RV + Delta-Stress   (Stress = CDS dealer regionale, bp)")
P("    lettura: coef Stress < 0 e significativo => la curva cheapens quando il bilancio brucia")
P(f"{'mercato':9}{'b_RV':>9}{'[t]':>7}{'b_STRESS':>11}{'[t]':>7}{'N':>6}")
rowsA = {}
for mkt, (legs, taus) in MK.items():
    if mkt not in sbe.columns: continue
    b = sbe[mkt].dropna()
    dy = (mid[legs[1]]/100.0).diff()
    rv = (dy.rolling(63).std()*np.sqrt(252)).resample("ME").last()*1e4
    st = CDS[CDSREGION[mkt]]
    df = pd.concat([b, rv, st], axis=1).dropna(); df.columns = ["BE", "RV", "ST"]
    d = df.diff().dropna()
    if len(d) < 40: continue
    X = np.column_stack([np.ones(len(d)), d["RV"].values, d["ST"].values])
    coef, t = ols_nw(d["BE"], pd.DataFrame(X))
    rowsA[mkt] = (coef[2], t[2])
    P(f"{mkt:9}{coef[1]:9.3f}{t[1]:7.1f}{coef[2]:11.3f}{t[2]:7.1f}{len(d):6d}")

P("")
P("[B] Delta-gap (G = s2_TRAIL - s2_BE) ~ Delta-Stress   [predizione: beta > 0]")
P(f"{'mercato':9}{'b_STRESS':>11}{'[t]':>7}{'N':>6}")
for mkt, (legs, taus) in MK.items():
    if mkt not in s2.columns: continue
    dy = (mid[legs[1]]/100.0).diff()
    trail = (dy.rolling(63).std()*np.sqrt(252)).resample("ME").last()
    G = (trail**2 - s2[mkt]).dropna()*1e8
    st = CDS[CDSREGION[mkt]]
    df = pd.concat([G, st], axis=1).dropna(); df.columns = ["G", "ST"]
    d = df.diff().dropna()
    if len(d) < 40: continue
    X = pd.DataFrame(np.column_stack([np.ones(len(d)), d["ST"].values]))
    coef, t = ols_nw(d["G"], X)
    P(f"{mkt:9}{coef[1]:11.3f}{t[1]:7.1f}{len(d):6d}")

P("")
P("[C] ASIMMETRIA: stesso Delta-Stress su Delta-sigma_BE (curva) vs Delta-sigma_IV (opzioni)")
P("    lettura: caricamento forte su BE e debole su IV => canale di bilancio della CURVA (H1);")
P("             caricamento simmetrico => favorisce il FATTORE COMUNE (alternativa nominata).")
P(f"{'mercato':9}{'b_ST(BE)':>11}{'[t]':>7}{'b_ST(IV)':>11}{'[t]':>7}{'N':>6}")
for mkt, (legs, taus) in MK.items():
    if mkt not in sbe.columns: continue
    iv = iv_series(IVMAP[mkt])
    if iv is None: continue
    st = CDS[CDSREGION[mkt]]
    df = pd.concat([sbe[mkt], iv, st], axis=1).dropna(); df.columns = ["BE", "IV", "ST"]
    d = df.diff().dropna()
    if len(d) < 40: continue
    X = pd.DataFrame(np.column_stack([np.ones(len(d)), d["ST"].values]))
    cB, tB = ols_nw(d["BE"], X)
    cI, tI = ols_nw(d["IV"], X)
    P(f"{mkt:9}{cB[1]:11.3f}{tB[1]:7.1f}{cI[1]:11.3f}{tI[1]:7.1f}{len(d):6d}")

if MOVE is not None:
    P("")
    P("[D] Robustezza: stesso test [A] con MOVE al posto del CDS dealer (stress di vol/funding)")
    P(f"{'mercato':9}{'b_MOVE':>10}{'[t]':>7}{'N':>6}")
    for mkt, (legs, taus) in MK.items():
        if mkt not in sbe.columns: continue
        dy = (mid[legs[1]]/100.0).diff()
        rv = (dy.rolling(63).std()*np.sqrt(252)).resample("ME").last()*1e4
        df = pd.concat([sbe[mkt], rv, MOVE], axis=1).dropna(); df.columns = ["BE", "RV", "MV"]
        d = df.diff().dropna()
        if len(d) < 40: continue
        X = pd.DataFrame(np.column_stack([np.ones(len(d)), d["RV"].values, d["MV"].values]))
        coef, t = ols_nw(d["BE"], X)
        P(f"{mkt:9}{coef[2]:10.3f}{t[2]:7.1f}{len(d):6d}")

P("")
P("VERDETTO: se [A]/[D] danno coefficienti di stress negativi e significativi in piu' mercati,")
P("e [C] mostra asimmetria BE-vs-IV, allora lo scollamento di sigma_BE e' PRICING VINCOLATO")
P("(la curva prezza il bilancio, non la vol attesa) e il C3 ~ 0 e' segmentazione, non rumore.")
save_txt("09_mechanism.txt", L); print("\n".join(L))

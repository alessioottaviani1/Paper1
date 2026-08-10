"""13 - TERM PREMIUM CROSS-MARKET (fattore CP) + STABILITA' TEMPORALE del C3.

Due buchi che restano dopo 12, ed entrambi sono domande che un referee fa di sicuro.

(A) IL CONTROLLO TERM PREMIUM ESISTE SOLO PER GLI USA. ACM e Kim-Wright sono americani, quindi
    il pannello 12[C]/[D] copre USD e UST ma NON EUR/GBP/JPY. Qui costruiamo un proxy di term
    premium PER OGNI MERCATO alla Cochrane-Piazzesi (2005): il fattore di forward, cioe' il
    valore fittato della regressione del rendimento in eccesso medio dei bond sui forward rate.
    Nessun download: usa la griglia 1-30Y gia' in bbg_paper2 (par ~ zero ai nodi, come il motore).
    NB: il CP usa i forward BREVI (1-5Y), sigma_BE usa il carry-roll del fly 2/10/30 -> non sono
    lo stesso oggetto per costruzione, quindi il controllo non e' circolare.

(B) LA SEGMENTAZIONE E' PERMANENTE O STA SPARENDO? Se il co-movimento salisse negli ultimi anni,
    il paper documenterebbe un fenomeno morente. Rolling a 60 mesi del C3 per mercato: si guarda
    la pendenza e il valore recente.

Output: output/convexity/results/13_tp_stability.txt
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import nw_t, save_txt, load_legs_mid, load_vols, load_dealer_cds, load_bbg_sheet

print("== 13 term premium cross-market + stabilita' ==")
sbe  = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
STRAT= pd.read_csv(PROC/"strat_monthly.csv", index_col=0, parse_dates=True)
L = []; P = L.append
P("=== 13 TERM PREMIUM CROSS-MARKET (CP) + STABILITA' DEL C3 ===")

# ---------- costruzione del fattore CP per mercato ----------
FAMILY = {"USDswap":"USOSFR", "EUR":"EUSA", "GBP":"BPSWS", "JPY":"JYSO"}
CPGRID = [1,2,3,4,5]     # forward brevi alla Cochrane-Piazzesi
RXTEN  = [2,3,5,7,10]    # scadenze su cui misurare il rendimento in eccesso

def par_grid(fam):
    """par rate mensili per i tenor richiesti, dal foglio swap."""
    raw = load_bbg_sheet(SHEET_SWAP)
    out = {}
    for (t, j), v in raw.items():
        tk = t.replace(" Curncy", "")
        if not tk.startswith(fam): continue
        suf = tk[len(fam):]
        if suf.isdigit(): out[int(suf)] = v.resample("ME").last()
    return pd.DataFrame(out).sort_index(axis=1)

def cp_factor(Z):
    """CP: fitted da rx medio (12m) ~ forward 1..5. Z = par rate (%) per tenor.
    La griglia Bloomberg salta il 4Y (1,2,3,5,7,10,...): interpoliamo linearmente sulle
    scadenze intere 1..10, che e' l'approssimazione gia' usata dal motore (roll lineare)."""
    Z = Z[[c for c in Z.columns if c <= 30]]
    tg = list(range(1, 11))
    Zi = pd.DataFrame(index=Z.index, columns=tg, dtype=float)
    cols = sorted(Z.columns)
    for t in tg:
        if t in Z.columns: Zi[t] = Z[t]; continue
        lo = max([c for c in cols if c < t], default=None); hi = min([c for c in cols if c > t], default=None)
        if lo is None or hi is None: continue
        w = (t-lo)/(hi-lo); Zi[t] = (1-w)*Z[lo] + w*Z[hi]
    z = Zi.dropna(how="all", axis=1)/100.0
    need = set(CPGRID) | set(RXTEN) | {1}
    if not need.issubset(set(z.columns)): return None
    # forward impliciti f(n-1,n) = n*z(n) - (n-1)*z(n-1)
    F = pd.DataFrame({n: n*z[n] - (n-1)*z[n-1] for n in CPGRID if n > 1})
    F[1] = z[1]
    F = F[sorted(F.columns)]
    # rendimento in eccesso a 12 mesi su zero a n anni (approx par~zero)
    RX = {}
    for n in RXTEN:
        if (n-1) not in z.columns: continue
        RX[n] = (n*z[n] - (n-1)*z[n-1].shift(-12) - z[1])
    if not RX: return None
    rx = pd.DataFrame(RX).mean(axis=1)
    al = pd.concat([rx, F], axis=1).dropna()
    if len(al) < 80: return None
    y = al.iloc[:,0].values; X = np.column_stack([np.ones(len(al)), al.iloc[:,1:].values])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    Ff = F.dropna()
    return pd.Series(np.column_stack([np.ones(len(Ff)), Ff.values]) @ b, index=Ff.index)

CP = {}
for mkt, fam in FAMILY.items():
    try:
        c = cp_factor(par_grid(fam))
        if c is not None: CP[mkt] = c*100  # in bp
    except Exception as e:
        P(f"[CP] {mkt}: non costruito ({e})")
P("")
P(f"[costruzione CP] mercati con fattore: {sorted(CP)}")

# ---------- (A1) sigma_BE su stress + CP, per mercato ----------
def ols_nw(y, X, lag=6):
    Xv, yv = np.asarray(X, float), np.asarray(y, float)
    b = np.linalg.lstsq(Xv, yv, rcond=None)[0]; e = yv - Xv@b
    A = np.linalg.inv(Xv.T@Xv); S = (e[:,None]*Xv).T@(e[:,None]*Xv)
    for l in range(1, lag+1):
        w = 1-l/(lag+1); u=(e[l:,None]*Xv[l:]); v=(e[:-l,None]*Xv[:-l]); G=u.T@v; S += w*(G+G.T)
    V = A@S@A; return b, b/np.sqrt(np.diag(V))

CDS, _, _ = load_dealer_cds()
P("")
P("[A1] Delta-sigma_BE ~ Delta-Stress + Delta-CP  (term premium LOCALE, per mercato)")
P(f"{'mercato':9}{'b_STRESS':>11}{'[t]':>7}{'b_CP':>9}{'[t]':>7}{'N':>6}")
for mkt in sbe.columns:
    if mkt not in CP or mkt not in CDSREGION: continue
    df = pd.concat([sbe[mkt], CDS[CDSREGION[mkt]], CP[mkt]], axis=1).dropna()
    df.columns = ["BE","ST","CP"]; d = df.diff().dropna()
    if len(d) < 60: continue
    X = np.column_stack([np.ones(len(d)), d["ST"].values, d["CP"].values])
    b, t = ols_nw(d["BE"], X)
    P(f"{mkt:9}{b[1]:11.3f}{t[1]:7.1f}{b[2]:9.3f}{t[2]:7.1f}{len(d):6d}")

# ---------- (A2) doppio sort stress x CP: il test pulito, ora per OGNI mercato ----------
P("")
P("[A2] DOPPIO SORT stress x CP locale (TP neutralizzato per costruzione, tutti i mercati)")
P(f"{'mercato':9}{'HIGH-stress':>13}{'LOW-stress':>12}{'diff':>9}{'[t]':>7}{'N':>6}")
for mkt in STRAT.columns:
    if mkt not in CP: continue
    y = STRAT[mkt].dropna()
    st = CDS[CDSREGION[mkt]].reindex(y.index).ffill()
    cp = CP[mkt].reindex(y.index).ffill()
    df = pd.concat([y, st, cp], axis=1).dropna(); df.columns = ["r","st","cp"]
    if len(df) < 60: continue
    df["b"] = pd.qcut(df["cp"], 3, labels=False, duplicates="drop")
    H = []; Lo = []
    for g, sub in df.groupby("b"):
        q = sub["st"].quantile([1/3, 2/3]).values
        H.append(sub["r"][sub["st"] >= q[1]]); Lo.append(sub["r"][sub["st"] < q[0]])
    H = pd.concat(H); Lo = pd.concat(Lo)
    dd = H.mean() - Lo.mean(); se = np.sqrt(H.var()/len(H) + Lo.var()/len(Lo))
    P(f"{mkt:9}{H.mean():13.2f}{Lo.mean():12.2f}{dd:9.2f}{dd/se:7.1f}{len(df):6d}")
P("    lettura: se la differenza regge a CP fermo in piu' mercati, il meccanismo non e' term")
P("    premium travestito -- e ora il controllo e' LOCALE, non il proxy americano.")

# ---------- (B) stabilita' temporale del C3 ----------
P("")
P("[B] STABILITA': C3 (Delta-corr) su finestra mobile a 60 mesi --- il fenomeno sta sparendo?")
IV = load_vols()
def iv3m10(ccy):
    return IV.get((ccy, "3M", "10Y", "NORM"))
P(f"{'mercato':9}{'primo':>8}{'mediana':>9}{'ultimo':>8}{'min':>7}{'max':>7}{'trend/decade':>14}")
for mkt in sbe.columns:
    s = iv3m10(IVMAP.get(mkt, ""))
    if s is None: continue
    al = pd.concat([sbe[mkt], s.resample("ME").last()], axis=1).dropna()
    if len(al) < 100: continue
    d = al.diff().dropna()
    roll = d.iloc[:,0].rolling(60).corr(d.iloc[:,1]).dropna()
    if len(roll) < 24: continue
    x = np.arange(len(roll)); sl = np.polyfit(x, roll.values, 1)[0]*120   # per decennio
    P(f"{mkt:9}{roll.iloc[0]:+8.2f}{roll.median():+9.2f}{roll.iloc[-1]:+8.2f}{roll.min():+7.2f}{roll.max():+7.2f}{sl:+14.2f}")
P("    lettura: trend ~ 0 e ultimo valore in linea con la mediana => segmentazione PERMANENTE,")
P("    non anomalia in via di chiusura. Un trend positivo forte sarebbe una bandiera rossa.")

save_txt("13_tp_stability.txt", L); print("\n".join(L))

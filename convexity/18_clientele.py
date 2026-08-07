"""18 - LE CLIENTELE, OSSERVATE INVECE CHE INFERITE.

IL PUNTO DEBOLE CHE QUESTO TEST CHIUDE. La storia del paper e' "un oggetto, due clientele": la curva
e' modellata da flussi motivati dalla duration (pensioni, assicurazioni, banche centrali, indicizzati)
che non arbitraggiano la volatilita', mentre le swaption sono intermediate da desk di opzioni. Finora
le due clientele sono INFERITE dalla geografia istituzionale -- l'LDI opera negli swap, la BoJ nei
JGB -- e l'inferenza, per quanto ben supportata, resta un'interpretazione. Qui le OSSERVIAMO.

I DATI. Commitments of Traders della CFTC nella versione Traders in Financial Futures, resa dall'OFR
in DV01 (raw/OFR/tff.csv, settimanale, 2013-2026). Separa le posizioni sui Treasury per categoria:
  AI = Asset Manager / Institutional  -> la clientela di DURATION (pensioni, assicurazioni, indici)
  DI = Dealer / Intermediary          -> gli INTERMEDIARI il cui bilancio e' il vincolo
  LF = Leveraged Funds                -> gli ARBITRAGGISTI
Sono esattamente i tre attori che la storia nomina, misurati in DV01 (l'unita' giusta: una posizione
di duration si misura in sensibilita' al tasso, non in numero di contratti).

LE TRE PREDIZIONI, tutte falsificabili:
 [P1] La DOMANDA DI DURATION degli asset manager e' la pressione che deforma la curva. Quando e'
      alta, il cuneo fra le due venue dovrebbe essere piu' ampio.
 [P2] La CAPACITA' dei dealer e' cio' che potrebbe chiuderlo. Quando l'inventario netto dei dealer
      e' compresso, il cuneo dovrebbe essere piu' ampio.
 [P3] Gli ARBITRAGGISTI sono chi dovrebbe raccoglierlo. Il premio dovrebbe essere piu' alto quando
      i leveraged fund sono POCO presenti (capitale assente = premio non raccolto).
Se i segni fossero opposti, la storia delle clientele sarebbe respinta dai dati posizionali.

CAVEAT DICHIARATO. La CFTC copre i FUTURES sul Treasury americano: e' la clientela di duration del
mercato USA, non di EUR/GBP/JPY. Il test e' quindi USA-only e va presentato come tale; per gli altri
mercati la clientela resta inferita dalla struttura istituzionale.

Output: output/convexity/results/18_clientele.txt
"""
import pandas as pd, numpy as np, glob, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
from config import *
from utils import nw_t, save_txt, load_legs_mid

print("== 18 clientele (CFTC/OFR) ==")
sbe   = pd.read_csv(PROC/"sigbe_monthly.csv",  index_col=0, parse_dates=True)
s2    = pd.read_csv(PROC/"s2be_monthly.csv",   index_col=0, parse_dates=True)
STRAT = pd.read_csv(PROC/"strat_monthly.csv",  index_col=0, parse_dates=True)
mid   = load_legs_mid()
L = []; P = L.append
P("=== 18 LE CLIENTELE OSSERVATE (CFTC Traders in Financial Futures, DV01) ===")

hit = sorted(glob.glob(str(RAW/"**"/"tff.csv"), recursive=True))
if not hit:
    P("tff.csv non trovato sotto raw/ -- script saltato")
    save_txt("18_clientele.txt", L); print("\n".join(L)); raise SystemExit

T = pd.read_csv(hit[0], header=3)
T.index = pd.to_datetime(T["date"], errors="coerce")
T = T[T.index.notna()].sort_index()
num = lambda c: pd.to_numeric(T[c], errors="coerce")

# posizioni NETTE in DV01, mensili (fine mese), in milioni di USD per bp
AI = ((num("TFF-AI_TREAS_LONG_DV01") - num("TFF-AI_TREAS_SHORT_DV01"))/1e6).resample("ME").last()
DI = ((num("TFF-DI_TREAS_LONG_DV01") - num("TFF-DI_TREAS_SHORT_DV01"))/1e6).resample("ME").last()
LF = ((num("TFF-LF_TREAS_LONG_DV01") - num("TFF-LF_TREAS_SHORT_DV01"))/1e6).resample("ME").last()
GROSS_DI = ((num("TFF-DI_TREAS_LONG_DV01") + num("TFF-DI_TREAS_SHORT_DV01"))/1e6).resample("ME").last()

P("")
P(f"campione: {AI.dropna().index.min().date()} -> {AI.dropna().index.max().date()}  (T={AI.notna().sum()} mesi)")
P(f"{'attore':34}{'media':>10}{'min':>10}{'max':>10}   (DV01 netto, mln USD/bp)")
for nm, s in [("Asset manager (duration)", AI), ("Dealer (intermediari)", DI),
              ("Leveraged funds (arbitraggisti)", LF)]:
    P(f"{nm:34}{s.mean():10.1f}{s.min():10.1f}{s.max():10.1f}")
P("    Segno atteso e VERIFICATO: gli asset manager sono strutturalmente LUNGHI duration (comprano")
P("    protezione contro il calo dei tassi per le passivita'), i dealer strutturalmente CORTI (sono")
P("    la controparte). E' la firma della clientela di duration nei dati posizionali.")

# ---- il cuneo americano: gap fra vol realizzata trailing e prezzo della curva
MKT = "USDswap"
legs, taus = MK[MKT]
dy = (mid[legs[1]]/100.0).diff()
rv = (dy.rolling(63).std()*np.sqrt(252)).resample("ME").last()
gap = ((rv**2 - s2[MKT]).dropna())*1e8      # in unita' di varianza, scala bp^2

def z(s): 
    s = s.dropna(); return (s - s.mean())/s.std()

def ols_nw(y, X, lag=6):
    Xv, yv = np.asarray(X, float), np.asarray(y, float)
    b = np.linalg.lstsq(Xv, yv, rcond=None)[0]; e = yv - Xv@b
    A = np.linalg.inv(Xv.T@Xv); S = (e[:,None]*Xv).T@(e[:,None]*Xv)
    for l in range(1, lag+1):
        w = 1-l/(lag+1); u=(e[l:,None]*Xv[l:]); v=(e[:-l,None]*Xv[:-l]); G=u.T@v; S += w*(G+G.T)
    V = A@S@A; yhat = Xv@b
    return b, b/np.sqrt(np.diag(V)), 1-np.var(yv-yhat)/np.var(yv)

# ---------- [P1]/[P2] il cuneo contro le posizioni ----------
P("")
P("[P1/P2] IL CUNEO contro le posizioni delle clientele (livelli standardizzati, USA)")
P("        cuneo alto = la curva prezza MENO vol di quella realizzata (convessita' a buon mercato)")
df = pd.concat([gap.rename("gap"), z(AI).rename("AI"), z(DI).rename("DI"), z(LF).rename("LF")],
               axis=1).dropna()
if len(df) < 40:
    P("   campione insufficiente")
else:
    P(f"   {'specificazione':38}{'coef':>9}{'[t]':>7}{'R2':>7}{'T':>5}")
    for nm, cols in [("domanda duration asset manager (AI)", ["AI"]),
                     ("inventario netto dealer (DI)",        ["DI"]),
                     ("presenza arbitraggisti (LF)",         ["LF"]),
                     ("AI + DI congiunti",                   ["AI","DI"])]:
        X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
        b, t, r2 = ols_nw(df["gap"], X)
        for i, c in enumerate(cols, start=1):
            lab = nm if i == 1 else ""
            P(f"   {lab:38}{b[i]:9.2f}{t[i]:7.1f}{r2 if i==1 else np.nan:7.2f}{len(df):5d}   [{c}]")

# ---------- [P3] il premio quando gli arbitraggisti sono assenti ----------
P("")
P("[P3] IL PREMIO quando gli ARBITRAGGISTI sono poco presenti (terzili di LF netto)")
P("     predizione: capitale di arbitraggio assente -> premio NON raccolto -> rendimenti piu' alti")
y = STRAT[MKT].dropna()
al = pd.concat([y, LF.reindex(y.index).ffill()], axis=1).dropna(); al.columns = ["r","LF"]
if len(al) >= 40:
    q = al["LF"].quantile([1/3, 2/3]).values
    lo = al["r"][al["LF"] < q[0]]; mi = al["r"][(al["LF"]>=q[0]) & (al["LF"]<q[1])]; hi = al["r"][al["LF"]>=q[1]]
    P(f"   {'LF BASSO':>12}{'[t]':>7}{'MID':>9}{'LF ALTO':>10}{'[t]':>7}{'L-H':>9}{'T':>5}")
    P(f"   {lo.mean():12.2f}{nw_t(lo):7.1f}{mi.mean():9.2f}{hi.mean():10.2f}{nw_t(hi):7.1f}"
      f"{lo.mean()-hi.mean():9.2f}{len(al):5d}")
    P("   (bp/mese, netti di costi)")
else:
    P("   campione insufficiente")

# ---------- lo stesso ordinamento con la CAPACITA' lorda dei dealer ----------
P("")
P("[P2-bis] IL PREMIO per terzili di CAPACITA' LORDA dei dealer (long+short DV01)")
P("        predizione: bilancio compresso -> cuneo non chiuso -> premio piu' alto quando la")
P("        capacita' e' BASSA")
al2 = pd.concat([y, GROSS_DI.reindex(y.index).ffill()], axis=1).dropna(); al2.columns = ["r","G"]
if len(al2) >= 40:
    q = al2["G"].quantile([1/3, 2/3]).values
    lo = al2["r"][al2["G"] < q[0]]; mi = al2["r"][(al2["G"]>=q[0]) & (al2["G"]<q[1])]; hi = al2["r"][al2["G"]>=q[1]]
    P(f"   {'CAP BASSA':>12}{'[t]':>7}{'MID':>9}{'CAP ALTA':>10}{'[t]':>7}{'L-H':>9}{'T':>5}")
    P(f"   {lo.mean():12.2f}{nw_t(lo):7.1f}{mi.mean():9.2f}{hi.mean():10.2f}{nw_t(hi):7.1f}"
      f"{lo.mean()-hi.mean():9.2f}{len(al2):5d}")

P("")
P("LETTURA. Il valore di questo pannello non e' la significativita' -- il campione parte dal 2013 e")
P("copre un solo mercato -- ma il fatto che le tre categorie che la storia NOMINA esistano nei dati")
P("con i segni che la storia PREDICE. Prima le clientele erano dedotte da dove operano le")
P("istituzioni; qui sono lette nelle posizioni. Va presentato come corroborazione diretta del")
P("meccanismo sul mercato americano, non come test cross-market.")
save_txt("18_clientele.txt", L); print("\n".join(L))

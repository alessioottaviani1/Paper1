"""29 - GLI STRUMENTI DINAMICI DEL PRIMO PAPER: quando e dove i due prezzi si connettono.

Dalla rilettura integrale del primo paper (sezioni IV e A.10, script rq3_02/05 e ml_06)
emergono quattro strumenti che trasformano affermazioni STATICHE del paper convexity in
affermazioni DINAMICHE o CONDIZIONALI. Nessuno cambia la tesi; ciascuno la affila.

 [1] DCC-GARCH sulla coppia (dSigma_BE, dSigma_IV)   [da rq3_05 sez. A]
     Il C3 e' una correlazione MEDIA. La domanda che un referee fara': "e' zero sempre, o
     zero in media con finestre di connessione?" Il rolling a 60 mesi del 13 e' la versione
     grezza; la correlazione condizionale dinamica e' lo strumento proprio. Se la DCC resta
     piatta anche nel 2008/2020/2022, la disconnessione e' STRUTTURALE e non "media di
     regimi connessi e sconnessi". Se sale nelle crisi, la storia si raffina: le venue si
     connettono quando la volatilita' domina tutto -- coerente con S3, e va scritto.

 [2] REGRESSIONE QUANTILE dSigma_IV ~ dSigma_BE      [da rq3_05 sez. B]
     "Disconnessi al margine" e' un'affermazione sulla MEDIA condizionale. La versione sui
     quantili chiede: nei mesi di coda -- i grandi movimenti -- la curva e l'opzione si
     muovono insieme? Se il beta e' zero al centro e positivo nelle code, l'integrazione
     esiste ma solo per i movimenti grandi (arbitraggio con soglia: coerente coi costi del
     04). Se e' zero ovunque, la disconnessione e' totale. Entrambi gli esiti sono
     informativi e nessuno dei due e' oggi nel paper.

 [3] FERSON-SCHADT (1996) SUL RENDIMENTO DELLA STRATEGIA   [da ml_06, pannello e1]
     La scala del 05/12 ordina i rendimenti per terzili di stress CONTEMPORANEO. La versione
     Ferson-Schadt e' la forma in regressione con strumento PREDETERMINATO:
         r_t = a0 + a1*z_{t-1} + (b + d*z_{t-1})*X_t + e_t
     a1 > 0 dice che l'alpha e' piu' alto quando lo stress era GIA' elevato il mese prima --
     che risponde all'obiezione di look-ahead che l'11 gestisce a fatica con le soglie, e
     usa esattamente la specifica del primo paper (z ritardato e standardizzato).

 [4] PERSISTENZA PER REGIME: phi_HIGH vs phi_LOW      [da rq3_02 sez. D]
     La S5 dice "la reversione e' piu' lenta dove la segmentazione e' piu' profonda" ed e'
     testata CROSS-MARKET nel 17. La versione WITHIN-market: il cuneo e' piu' persistente
     negli stati di stress? Se phi_HIGH > phi_LOW, i limiti dell'arbitraggio mordono quando
     i bilanci sono vincolati -- la S5 condizionale, che il primo paper testa come P6.

Output: results/29_dynamic_tools.txt + figures/figc9_dcc.png
"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats as st
from config import *
from utils import save_txt, load_legs_mid_all, load_vols, load_dealer_cds
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,"grid.alpha":0.3})

sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
STRAT = pd.read_csv(PROC/"strat_monthly.csv", index_col=0, parse_dates=True)
mid = load_legs_mid_all(); IV = load_vols()
L=[]; P=L.append
P("=== 29 STRUMENTI DINAMICI DAL PRIMO PAPER ===")

MK4=[m for m in ["USDswap","EUR","GBP","JPY"] if m in sbe.columns]
LAB={"USDswap":"USD swap","EUR":"EUR","GBP":"GBP","JPY":"JPY"}
REG={"USDswap":"US","EUR":"EU","GBP":"EU","JPY":"US"}

def ivc(m):
    s=IV.get((IVMAP.get(m,""),"3M","10Y","NORM"))
    return None if s is None else s.resample("ME").last()

PAIR={}
for m in MK4:
    iv=ivc(m)
    if iv is None: continue
    al=pd.concat([sbe[m].rename("be"),iv.rename("iv")],axis=1).dropna().diff().dropna()
    if len(al)>=80: PAIR[m]=al

# ============================================================ [1] DCC-GARCH
P("")
P("[1] DCC-GARCH(1,1) SU (dBE, dIV): la disconnessione e' strutturale o media di regimi?")
def garch11(x):
    """GARCH(1,1) via QML, ottimizzazione su griglia grossa + raffinamento Nelder-Mead."""
    x=np.asarray(x,float); x=x-x.mean(); T=len(x)
    def nll(p):
        w,a,b=p
        if w<=0 or a<0 or b<0 or a+b>=0.999: return 1e10
        h=np.empty(T); h[0]=x.var()
        for t in range(1,T): h[t]=w+a*x[t-1]**2+b*h[t-1]
        h=np.maximum(h,1e-12)
        return 0.5*np.sum(np.log(h)+x**2/h)
    from scipy.optimize import minimize
    best=None
    for a0 in (0.05,0.15):
        for b0 in (0.7,0.85):
            r=minimize(nll,[x.var()*(1-a0-b0),a0,b0],method="Nelder-Mead",
                       options={"maxiter":800,"xatol":1e-6,"fatol":1e-6})
            if best is None or r.fun<best.fun: best=r
    w,a,b=best.x
    h=np.empty(T); h[0]=x.var()
    for t in range(1,T): h[t]=w+a*x[t-1]**2+b*h[t-1]
    return x/np.sqrt(np.maximum(h,1e-12))

def dcc(e1,e2):
    """DCC(1,1) di Engle (2002) sugli standardizzati, QML a due stadi."""
    E=np.column_stack([e1,e2]); T=len(E)
    Qbar=np.cov(E.T)
    from scipy.optimize import minimize
    def nll(p):
        a,b=p
        if a<0 or b<0 or a+b>=0.999: return 1e10
        Q=Qbar.copy(); ll=0.0
        for t in range(T):
            R=Q/np.sqrt(np.outer(np.diag(Q),np.diag(Q)))
            r12=np.clip(R[0,1],-0.999,0.999)
            det=1-r12**2
            ll+=0.5*(np.log(det)+(E[t,0]**2+E[t,1]**2-2*r12*E[t,0]*E[t,1])/det)
            Q=(1-a-b)*Qbar+a*np.outer(E[t],E[t])+b*Q
        return ll
    best=None
    for a0 in (0.02,0.05,0.10):
        r=minimize(nll,[a0,0.85],method="Nelder-Mead",
                   options={"maxiter":600,"xatol":1e-5,"fatol":1e-5})
        if best is None or r.fun<best.fun: best=r
    a,b=best.x
    Q=Qbar.copy(); rho=np.empty(T)
    for t in range(T):
        R=Q/np.sqrt(np.outer(np.diag(Q),np.diag(Q)))
        rho[t]=np.clip(R[0,1],-1,1)
        Q=(1-a-b)*Qbar+a*np.outer(E[t],E[t])+b*Q
    return rho,a,b

P(f"    {'mercato':10}{'a_dcc':>8}{'b_dcc':>8}{'rho medio':>11}{'min':>7}{'max':>7}{'rho in crisi':>13}")
CRISES=[("2008-09","2009-06"),("2011-07","2012-08"),("2020-02","2020-05"),("2022-09","2022-12")]
DCCS={}
for m,al in PAIR.items():
    try:
        e1=garch11(al["be"].values); e2=garch11(al["iv"].values)
        rho,a,b=dcc(e1,e2)
        r=pd.Series(rho,index=al.index); DCCS[m]=r
        mask=pd.Series(False,index=r.index)
        for c0,c1 in CRISES: mask.loc[c0:c1]=True
        P(f"    {LAB[m]:10}{a:8.3f}{b:8.3f}{r.mean():11.2f}{r.min():7.2f}{r.max():7.2f}"
          f"{r[mask].mean() if mask.any() else np.nan:13.2f}")
    except Exception as ex:
        P(f"    {LAB[m]:10} non stimato ({ex})")
P("    LETTURA. Se rho resta piatta e vicina alla media anche in crisi, la disconnessione e'")
P("    STRUTTURALE. Se sale nelle finestre di crisi, le venue si connettono quando la")
P("    volatilita' domina -- coerente con S3, e la frase del paper va raffinata di conseguenza.")

if DCCS:
    fig,ax=plt.subplots(figsize=(10,4))
    for m,r in DCCS.items(): ax.plot(r.index,r.rolling(3).mean(),lw=1.1,label=LAB[m])
    for c0,c1 in CRISES: ax.axvspan(pd.Timestamp(c0),pd.Timestamp(c1),color="grey",alpha=.15)
    ax.axhline(0,color="k",lw=.8); ax.legend(fontsize=8,ncol=4); ax.set_ylabel("DCC correlation")
    ax.set_title("Dynamic conditional correlation of the two prices' changes (3m smoothed)",fontsize=10)
    fig.tight_layout(); fig.savefig(FIG/"figc9_dcc.png"); plt.close(fig)
    P(f"    [figura] {FIG/'figc9_dcc.png'}")

# ============================================================ [2] regressione quantile
P("")
P("[2] REGRESSIONE QUANTILE dIV ~ dBE: la connessione vive nelle code?")
def qreg(y,x,q):
    """regressione quantile univariata via programmazione lineare semplice (IRLS pinball)."""
    b=np.polyfit(x,y,1)[0]; a=np.quantile(y-b*x,q)
    for _ in range(200):
        u=y-a-b*x
        w=np.where(u>=0,q,1-q)/np.maximum(np.abs(u),1e-6)
        W=np.sum(w); Wx=np.sum(w*x); Wxx=np.sum(w*x*x)
        Wy=np.sum(w*y); Wxy=np.sum(w*x*y)
        det=W*Wxx-Wx**2
        if abs(det)<1e-12: break
        a_new=(Wxx*Wy-Wx*Wxy)/det; b_new=(W*Wxy-Wx*Wy)/det
        if abs(a_new-a)<1e-8 and abs(b_new-b)<1e-8: a,b=a_new,b_new; break
        a,b=a_new,b_new
    return b

QS=[0.10,0.25,0.50,0.75,0.90]
P(f"    {'mercato':10}"+"".join(f"{'q'+str(int(q*100)):>9}" for q in QS)+f"{'OLS':>9}")
for m,al in PAIR.items():
    x=al["be"].values; y=al["iv"].values
    row="".join(f"{qreg(y,x,q):9.3f}" for q in QS)
    P(f"    {LAB[m]:10}{row}{np.polyfit(x,y,1)[0]:9.3f}")
P("    LETTURA. Beta ~0 al centro e positivo a q10/q90 = integrazione A SOGLIA: i prezzi si")
P("    riallineano solo sui movimenti grandi, coerente con l'arbitraggio in presenza di costi")
P("    (04). Beta ~0 ovunque = disconnessione anche nelle code, l'affermazione piu' forte.")

# ============================================================ [3] Ferson-Schadt
P("")
P("[3] FERSON-SCHADT SUL RENDIMENTO: r_t = a0 + a1*z_{t-1} + (b + d*z_{t-1})*dBE_t")
P("    z = stress dealer STANDARDIZZATO e RITARDATO (predeterminato: niente look-ahead)")
try:
    CDS,_,_=load_dealer_cds()
    P(f"    {'mercato':10}{'a0':>8}{'[t]':>7}{'a1':>8}{'[t]':>7}{'d':>8}{'[t]':>7}{'T':>5}")
    for m in MK4:
        if m not in STRAT.columns: continue
        c=CDS.get(REG[m])
        if c is None: continue
        z=((c-c.rolling(36).mean())/c.rolling(36).std()).resample("ME").last().shift(1)
        al=pd.concat([STRAT[m].rename("r"),z.rename("z"),
                      sbe[m].diff().rename("db")],axis=1).dropna()
        if len(al)<80: continue
        X=np.column_stack([np.ones(len(al)),al["z"],al["db"],al["z"]*al["db"]])
        y=al["r"].values
        b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b
        A=np.linalg.inv(X.T@X); S=(e[:,None]*X).T@(e[:,None]*X)
        for l in range(1,7):
            w=1-l/7; u=e[l:,None]*X[l:]; v=e[:-l,None]*X[:-l]; G=u.T@v; S+=w*(G+G.T)
        V=A@S@A; se=np.sqrt(np.diag(V))
        P(f"    {LAB[m]:10}{b[0]:8.2f}{b[0]/se[0]:7.1f}{b[1]:8.2f}{b[1]/se[1]:7.1f}"
          f"{b[3]:8.2f}{b[3]/se[3]:7.1f}{len(al):5d}")
    P("    LETTURA. a1>0 = l'alpha e' piu' alto quando lo stress era GIA' alto il mese prima:")
    P("    la versione predeterminata della scala del 05, che chiude l'obiezione di look-ahead")
    P("    meglio delle soglie dell'11. E' la specifica (e1) del primo paper.")
except Exception as ex:
    P(f"    non calcolato ({ex})")

# ============================================================ [4] persistenza per regime
P("")
P("[4] PERSISTENZA PER REGIME: phi del cuneo in stress HIGH vs LOW (S5 condizionale)")
try:
    CDS,_,_=load_dealer_cds()
    P(f"    {'mercato':10}{'phi LOW':>9}{'phi HIGH':>10}{'diff':>7}{'HL LOW':>8}{'HL HIGH':>9}{'T':>5}")
    for m in MK4:
        iv=ivc(m)
        if iv is None: continue
        W=(sbe[m]-iv).dropna()
        c=CDS.get(REG[m])
        if c is None: continue
        s=c.resample("ME").last().reindex(W.index).ffill()
        q=s.quantile(2/3)
        hi=s>=q
        al=pd.concat([W.rename("w"),W.shift(1).rename("wl"),hi.rename("h")],axis=1).dropna()
        lo_=al[~al["h"]]; hi_=al[al["h"]]
        if len(lo_)<30 or len(hi_)<30: continue
        pL=np.polyfit(lo_["wl"],lo_["w"],1)[0]; pH=np.polyfit(hi_["wl"],hi_["w"],1)[0]
        hlL=np.log(0.5)/np.log(max(min(pL,0.999),1e-6)); hlH=np.log(0.5)/np.log(max(min(pH,0.999),1e-6))
        P(f"    {LAB[m]:10}{pL:9.2f}{pH:10.2f}{pH-pL:7.2f}{hlL:8.1f}{hlH:9.1f}{len(al):5d}")
    P("    LETTURA. phi_HIGH > phi_LOW = il cuneo e' piu' persistente quando i bilanci sono")
    P("    vincolati: i limiti dell'arbitraggio mordono negli stati di stress, che e' la S5")
    P("    nella versione condizionale (la P6 del primo paper). E' un'evidenza sul meccanismo")
    P("    che non richiede rendimenti, quindi indipendente dalla strategia.")
except Exception as ex:
    P(f"    non calcolato ({ex})")

P("")
P("COSA E' STATO VALUTATO E SCARTATO, dalla rilettura integrale. Threshold clustering (rq3_05C):")
P("richiede segnali di entry multipli, qui la strategia e' una. MPPM (fm_04): confronta gestori,")
P("non pertinente. Selector benchmark (ml_08): utile solo se un referee contesta il best subset.")
P("VAR/GIRF/FEVD (rq3_04): appropriato per l'estensione strutturale multivariata, prematuro ora.")
P("Spanning con interazione (rq3_03D) sui cunei cross-market: interessante per P5 ma il PC1 del")
P("16 gia' cattura il fattore comune; da riconsiderare se P5 diventa centrale. SW/EW (strat_05):")
P("la strategia qui e' deliberatamente non ottimizzata. PCA-K robustness (pca_06): una riga di")
P("robustezza quando il 16 sara' rigenerato col placebo completo.")
save_txt("29_dynamic_tools.txt", L); print("\n".join(L))

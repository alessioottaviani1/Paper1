"""28 - STRUMENTI DAL PRIMO PAPER, APPLICATI AL SECONDO.

Il primo paper ha costruito una cassetta degli attrezzi econometrica che il secondo non usa e
che risolve quattro problemi aperti. Nessuno di questi e' un test nuovo: sono correzioni e
procedure standard che il paper 2 dovrebbe gia' avere e non ha.

 [1] FORBES-RIGOBON (2002) SUL C3 CROSS-MARKET.  IL PROBLEMA PIU' SERIO CHE CHIUDE.
     La predizione S2 ordina i mercati per integrazione: JPY +0.31, EUR +0.24, USD -0.01,
     GBP -0.07. Ma la correlazione campionaria e' DISTORTA VERSO L'ALTO quando la varianza
     del regressore e' alta -- e i quattro mercati hanno volatilita' molto diverse (sigma di
     sigma_BE: USD 115, GBP 93, EUR 78, JPY 67 dal pannello 1 del 17). L'ordinamento potrebbe
     quindi riflettere la DISPERSIONE della curva invece dell'integrazione: il Giappone e' il
     mercato meno volatile e ha la correlazione piu' alta, il che e' l'opposto del bias --
     ma va MOSTRATO, non assunto. La correzione FR normalizza per l'eteroschedasticita':
        rho_FR = rho / sqrt(1 + delta*(1-rho^2)),   delta = sigma_alta^2/sigma_bassa^2 - 1
     Qui applicata in versione SIMMETRICA (media delle due direzioni), come nel primo paper,
     cosi' il risultato non dipende da quale serie si mette al numeratore.

 [2] FDR DI BENJAMINI-HOCHBERG SUL CUBO.  Il paper riporta 128 celle del cubo e 8 geometrie
     della matrice. Con quel numero di test, alcune significativita' sono attese per caso.
     Il primo paper corregge; il secondo no. E' un'obiezione garantita in referaggio.

 [3] BLOCK BOOTSTRAP SULLA DIFFERENZA FRA REGIMI.  Il 25 confronta il premio epurato prima e
     dopo la rottura con t di Newey-West su campioni di 39-49 mesi fortemente autocorrelati
     (AR(1) 0.97). Il primo paper usa block bootstrap per l'inferenza su differenze in questa
     situazione, che e' la procedura corretta.

 [4] MOREIRA-MUIR (2017) SULLA STRATEGIA.  Il paper 2 e' un paper sulla VOLATILITA' e non
     testa il risultato piu' noto della letteratura sul volatility timing: scalare la
     posizione per l'inverso della varianza recente migliora lo Sharpe. Se la strategia di
     convessita' NON beneficia dello scaling, e' un fatto notevole (il premio non e' un
     compenso per varianza); se ne beneficia, va dichiarato prima che lo faccia un referee.

Output: results/28_paper1_tools.txt
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import save_txt, load_legs_mid_all, load_vols
try:
    from statsmodels.stats.multitest import multipletests
    HAS_SM = True
except Exception:
    HAS_SM = False
rng = np.random.default_rng(77)

sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
STRAT = pd.read_csv(PROC/"strat_monthly.csv", index_col=0, parse_dates=True)
mid = load_legs_mid_all(); IV = load_vols()
L=[]; P=L.append
P("=== 28 STRUMENTI DEL PRIMO PAPER APPLICATI AL SECONDO ===")

MK4=[m for m in ["USDswap","EUR","GBP","JPY"] if m in sbe.columns]
LAB={"USDswap":"USD swap","EUR":"EUR","GBP":"GBP","JPY":"JPY"}
EXPS=["3M","6M","1Y","2Y"]; TENS=["2Y","5Y","10Y","30Y"]

def ivc(m,e,t):
    s=IV.get((IVMAP.get(m,""),e,t,"NORM"))
    return None if s is None else s.resample("ME").last()

# ============================================================ [1] Forbes-Rigobon
def fr(rho, s_hi, s_lo):
    if s_lo == 0 or not np.isfinite(rho): return rho
    d = (s_hi**2/s_lo**2) - 1.0
    den = np.sqrt(1 + d*(1-rho**2))
    return rho if den == 0 else rho/den

def fr_sym(rho, sx_hi, sx_lo, sy_hi, sy_lo):
    return 0.5*(fr(rho, sx_hi, sx_lo) + fr(rho, sy_hi, sy_lo))

P("")
P("[1] FORBES-RIGOBON SUL C3: l'ordinamento S2 e' integrazione o dispersione?")
P("    Il mercato con la sigma piu' BASSA e' preso come benchmark; ogni altro mercato viene")
P("    corretto per il rapporto delle varianze rispetto a quello.")
D={}
for m in MK4:
    iv=ivc(m,"3M","10Y")
    if iv is None: continue
    al=pd.concat([sbe[m].rename("be"),iv.rename("iv")],axis=1).dropna().diff().dropna()
    if len(al)<60: continue
    D[m]=al
if D:
    base=min(D, key=lambda k: D[k]["be"].std())
    sb_lo, si_lo = D[base]["be"].std(), D[base]["iv"].std()
    P(f"    benchmark (sigma minima): {LAB[base]}  sd(dBE)={sb_lo:.1f}  sd(dIV)={si_lo:.1f}")
    P(f"    {'mercato':10}{'sd dBE':>9}{'sd dIV':>9}{'rho oss.':>10}{'rho FR':>9}{'variazione':>12}{'T':>5}")
    FRR={}
    for m in D:
        a=D[m]; rho=a["be"].corr(a["iv"])
        r_fr = rho if m==base else fr_sym(rho, a["be"].std(), sb_lo, a["iv"].std(), si_lo)
        FRR[m]=r_fr
        P(f"    {LAB[m]:10}{a['be'].std():9.1f}{a['iv'].std():9.1f}{rho:10.2f}{r_fr:9.2f}"
          f"{r_fr-rho:+12.2f}{len(a):5d}")
    o_raw=sorted(D, key=lambda k: -D[k]["be"].corr(D[k]["iv"]))
    o_fr =sorted(FRR, key=lambda k: -FRR[k])
    P(f"    ordinamento grezzo : {' > '.join(LAB[m] for m in o_raw)}")
    P(f"    ordinamento FR     : {' > '.join(LAB[m] for m in o_fr)}")
    P("    SE L'ORDINAMENTO NON CAMBIA, S2 non e' un artefatto di eteroschedasticita' -- ed e'")
    P("    la risposta all'obiezione 'state ordinando la volatilita', non l'integrazione'.")

# ============================================================ [2] FDR sul cubo
P("")
P("[2] BENJAMINI-HOCHBERG SUL CUBO: quante celle restano significative con FDR al 5%?")
if not HAS_SM:
    P("    statsmodels non disponibile -- pannello saltato")
else:
    rows=[]
    for m in MK4:
        for e in EXPS:
            for t in TENS:
                iv=ivc(m,e,t)
                if iv is None: continue
                al=pd.concat([sbe[m].rename("be"),iv.rename("iv")],axis=1).dropna().diff().dropna()
                n=len(al)
                if n<60: continue
                r=al["be"].corr(al["iv"])
                # p-value asintotico della correlazione
                tt=r*np.sqrt((n-2)/max(1-r**2,1e-12))
                from scipy import stats as st
                pv=2*(1-st.t.cdf(abs(tt),n-2))
                rows.append((m,e,t,r,pv,n))
    if rows:
        pvals=[x[4] for x in rows]
        rej,padj,_,_=multipletests(pvals,alpha=0.05,method="fdr_bh")
        P(f"    celle testate: {len(rows)}")
        P(f"    significative al 5% GREZZO : {sum(p<0.05 for p in pvals)}")
        P(f"    significative al 5% FDR-BH : {int(rej.sum())}")
        P(f"    {'mercato':10}{'grezzo':>9}{'FDR':>7}{'su':>5}")
        for m in MK4:
            idx=[i for i,x in enumerate(rows) if x[0]==m]
            if not idx: continue
            P(f"    {LAB[m]:10}{sum(pvals[i]<0.05 for i in idx):9d}{int(rej[idx].sum()):7d}{len(idx):5d}")
        P("    LETTURA. Il paper afferma che il co-movimento e' ASSENTE: qui la correzione")
        P("    lavora A FAVORE, perche' riduce il numero di celle in cui si potrebbe sostenere")
        P("    che una correlazione esiste. Riportarla e' gratuito e disinnesca l'obiezione.")

# ============================================================ [3] block bootstrap
P("")
P("[3] BLOCK BOOTSTRAP SULLA DIFFERENZA FRA REGIMI (il 25 usa NW su 39-49 mesi con AR(1)=0.97)")
def block_boot_diff(a, b, B=5000, bl=12):
    a=np.asarray(pd.Series(a).dropna()); b=np.asarray(pd.Series(b).dropna())
    if len(a)<24 or len(b)<12: return (np.nan,)*3
    def draw(x):
        n=len(x); nb=int(np.ceil(n/bl))
        st_=rng.integers(0,max(1,n-bl),nb)
        return np.concatenate([x[s:s+bl] for s in st_])[:n]
    d=np.array([draw(a).mean()-draw(b).mean() for _ in range(B)])
    obs=a.mean()-b.mean()
    return obs, np.percentile(d,2.5), np.percentile(d,97.5)

H_M=3; FAM={"USDswap":"USOSFR","EUR":"EUSA","GBP":"BPSWS","JPY":"JYSO"}
def Rser(m):
    dy=(mid[f"{FAM[m]}10"]/100.0).diff()
    rv=(dy.rolling(63).std()*np.sqrt(252)*1e4).resample("ME").last().shift(-H_M)
    al=pd.concat([sbe[m].rename("be"),rv.rename("rv")],axis=1).dropna()
    return (al["be"]-al["rv"]) if len(al)>=60 else None
BREAKS={"USDswap":"2022-03","EUR":"2022-03","GBP":"2021-10","JPY":"2014-05"}
P(f"    {'mercato':10}{'pre':>9}{'post':>9}{'differenza':>12}{'IC95 bootstrap':>22}{'zero dentro?':>14}")
for m in MK4:
    r=Rser(m)
    if r is None or m not in BREAKS: continue
    pre=r.loc[:BREAKS[m]]; post=r.loc[BREAKS[m]:]
    obs,lo,hi=block_boot_diff(pre.values,post.values)
    if np.isnan(obs): continue
    inside = "SI" if lo<=0<=hi else "no"
    P(f"    {LAB[m]:10}{pre.mean():9.0f}{post.mean():9.0f}{obs:12.0f}"
      f"{f'[{lo:7.0f},{hi:7.0f}]':>22}{inside:>14}")
P("    Blocchi di 12 mesi per rispettare l'autocorrelazione. Se lo zero resta fuori, il salto")
P("    fra regimi e' reale e non un artefatto di campioni corti e persistenti.")

# ============================================================ [4] Moreira-Muir
P("")
P("[4] MOREIRA-MUIR: la strategia di convessita' beneficia del volatility scaling?")
P("    posizione scalata per 1/var(rendimenti ultimi 12m), riscalata a pari volatilita'")
P(f"    {'mercato':10}{'Sharpe base':>13}{'Sharpe scalato':>16}{'alpha MM':>10}{'[t]':>7}{'T':>5}")
for m in MK4:
    if m not in STRAT.columns: continue
    y=STRAT[m].dropna()
    if len(y)<60: continue
    v=y.rolling(12).var().shift(1)
    w=(1.0/v).replace([np.inf,-np.inf],np.nan)
    ys=(w*y).dropna()
    yv=y.reindex(ys.index)
    if len(ys)<48 or ys.std()==0: continue
    ys=ys*(yv.std()/ys.std())                      # pari volatilita'
    sh0=yv.mean()/yv.std()*np.sqrt(12)
    sh1=ys.mean()/ys.std()*np.sqrt(12)
    X=np.column_stack([np.ones(len(ys)),yv.values])
    b=np.linalg.lstsq(X,ys.values,rcond=None)[0]; e=ys.values-X@b
    se=np.sqrt(np.linalg.inv(X.T@X)[0,0]*(e@e)/(len(ys)-2))
    P(f"    {LAB[m]:10}{sh0:13.2f}{sh1:16.2f}{b[0]:10.2f}{b[0]/se:7.1f}{len(ys):5d}")
P("    LETTURA. alpha MM positivo e significativo => la strategia beneficia dello scaling e")
P("    parte del premio e' compenso per varianza tempo-variante: va dichiarato. alpha nullo")
P("    => il premio NON e' varianza travestita, che per un paper sulla volatilita' e' un")
P("    risultato da riportare esplicitamente.")

# ============================================================ [5] best subset sull'alpha
P("")
P("[5] BEST-SUBSET (Bertsimas-King-Mazumder) SUL CONTROLLO DELL'ALPHA")
P("    Il pannello [C] del 15 controlla l'alpha con TUTTO il blocco tassi in OLS: ma term,")
P("    pendenza e livello sono collineari, e l'OLS diluisce i coefficienti sul cluster. Il")
P("    primo paper usa la selezione l0 a cardinalita' fissa come primario: entra il membro")
P("    piu' informativo del cluster, senza shrinkage. Qui: enumerazione esaustiva su k=3")
P("    (sensibilita' k=2,4), alpha di post-selezione con t HAC.")
P("    NOTA: richiede il file dei fattori. Aggiungere in coda al 15:")
P("        F.to_csv(PROC/'factors_monthly.csv')")
P("    e rilanciarlo una volta; poi questo pannello gira.")
import itertools as _it
try:
    F = pd.read_csv(PROC/"factors_monthly.csv", index_col=0, parse_dates=True)
    def hac_alpha(y, X, lag=6):
        Xd=np.column_stack([np.ones(len(y)), X]); b=np.linalg.lstsq(Xd,y,rcond=None)[0]
        e=y-Xd@b; A=np.linalg.inv(Xd.T@Xd)
        S=(e[:,None]*Xd).T@(e[:,None]*Xd)
        for l in range(1,lag+1):
            w=1-l/(lag+1); u=e[l:,None]*Xd[l:]; v=e[:-l,None]*Xd[:-l]; G=u.T@v; S+=w*(G+G.T)
        V=A@S@A
        return b[0], b[0]/np.sqrt(V[0,0])
    P(f"    {'mercato':10}{'k':>3}{'alpha':>8}{'[t]':>7}{'R2':>7}{'fattori scelti':>44}")
    for m in MK4:
        if m not in STRAT.columns: continue
        y0=STRAT[m].dropna()
        al=F.join(y0.rename("y"), how="inner").dropna()
        if len(al)<80: continue
        yv=al["y"].values; Xall=al.drop(columns="y")
        cols=list(Xall.columns)
        for k in (2,3,4):
            best=None
            for combo in _it.combinations(range(len(cols)), k):
                X=Xall.values[:,combo]
                b,_res,_r,_sv=np.linalg.lstsq(np.column_stack([np.ones(len(yv)),X]),yv,rcond=None)
                e=yv-np.column_stack([np.ones(len(yv)),X])@b
                rss=e@e
                if best is None or rss<best[0]: best=(rss,combo)
            rss,combo=best
            X=Xall.values[:,combo]
            a,t=hac_alpha(yv,X)
            r2=1-rss/((yv-yv.mean())**2).sum()
            names=",".join(cols[c] for c in combo)
            tag=LAB[m] if k==3 else ""
            P(f"    {tag:10}{k:>3}{a:8.2f}{t:7.1f}{r2:7.2f}{names[:44]:>44}")
    P("    LETTURA. Se l'alpha regge quando il selettore prende i k fattori piu' informativi")
    P("    SENZA shrinkage, la difesa e' piu' forte dell'OLS collineare del 15 [C] -- e usa lo")
    P("    stesso apparato del primo paper, con k fisso difeso dalla sensibilita' (2-4).")
except FileNotFoundError:
    P("    factors_monthly.csv non trovato: aggiungere la riga al 15 e rilanciare.")

P("")
P("PERCHE' QUESTI CINQUE. Nessuno e' un test nuovo: sono procedure che il primo paper gia'")
P("applica e che il secondo dovrebbe avere. [1] chiude l'obiezione piu' seria all'ordinamento")
P("S2. [2] e' gratuito e lavora a favore della tesi. [3] sostituisce un'inferenza fragile con")
P("quella corretta. [4] e' l'unico test di volatility timing che manca a un paper sulla")
P("volatilita', ed e' meglio farlo che vederselo chiedere.")
save_txt("28_paper1_tools.txt", L); print("\n".join(L))

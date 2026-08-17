"""
10 - IL MECCANISMO ITALIANO: fuga verso la liquidita' dentro la curva sovrana.

I FATTI DA SPIEGARE (da 07-09):
  (i)   il CCT e' CARO sul tratto corto in tempi normali;
  (ii)  diventa ECONOMICO nelle crisi (2011-12, 2022-23), con la gobba su 2-4 anni;
  (iii) il premio NON scala con la stabilita' mark-to-market (09: il coefficiente muore
        con gli effetti fissi di titolo), quindi il meccanismo di Fleckenstein-Longstaff
        non si trasferisce;
  (iv)  la variazione e' prevalentemente CROSS-SEZIONALE fra titoli, non temporale entro
        titolo (R2 da 0.44 a 0.67 aggiungendo gli effetti fissi di CCT).

L'IPOTESI. CCT e BTP hanno lo STESSO emittente, quindi il rischio sovrano si cancella nel
confronto -- a meno che non colpisca i due strumenti in modo diverso. E c'e' una ragione
perche' lo faccia: il BTP e' il benchmark liquido, il CCT no. Nello stress gli investitori
si concentrano sullo strumento piu' liquido della stessa curva, il BTP si arricchisce e il
CCT si sconta. E' una fuga verso la liquidita' DENTRO la curva sovrana, non fuori.
Complementare: le banche italiane detengono CCT per il matching a tasso variabile, e nelle
crisi del 2011-12 e 2022-23 sono il settore sotto stress -- chi ha bisogno di liquidita'
vende cio' che ha in bilancio. E' l'analogo italiano del canale MMF di FL, ma con segno
OPPOSTO nelle crisi, perche' qui la clientela e' quella stressata.

TRE PREDIZIONI FALSIFICABILI, tutte testabili con i dati gia' scaricati:
  P1 (serie storica) - la base si sconta quando lo stress sovrano sale. Proxy: spread fra
     il par yield della curva sovrana e il tasso swap di pari scadenza, che e' esattamente
     il premio dell'Italia sopra il settore bancario europeo.
  P2 (serie storica) - e quando sale lo stress di FUNDING bancario. Proxy: Euribor-OIS.
  P3 (sezione trasversale) - lo sconto e' PIU' GRANDE per i CCT meno liquidi. Proxy
     osservabili: ammontare emesso e anzianita' dall'emissione. E' la predizione che
     distingue la fuga verso la liquidita' da una spiegazione puramente macro, perche'
     una spiegazione macro non ha ragione di colpire di piu' i titoli piccoli.
  P4 (interazione) - lo sconto da stress e' AMPLIFICATO sui titoli illiquidi. E' la
     predizione piu' stringente: se P4 non regge, la storia della liquidita' non regge.

Output: results/10_mechanism.txt, PROC/mechanism_panel.csv
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

def nss(tau,p):
    b0,b1,b2,b3,t1,t2=p; tau=np.maximum(np.asarray(tau,float),1e-8)
    x1,x2=tau/t1,tau/t2
    f1=(1-np.exp(-x1))/x1; f2=(1-np.exp(-x2))/x2
    return b0+b1*f1+b2*(f1-np.exp(-x1))+b3*(f2-np.exp(-x2))

def par_yield(p, tau, freq=CPN_FREQ):
    n=max(int(np.ceil(tau*freq-1e-9)),1)
    t=np.array([(i+1)/freq for i in range(n)],float); t=t-(t[-1]-tau); t=t[t>0]
    if len(t)==0: return np.nan
    df=np.exp(-nss(t,p)/100.0*t); ann=float(np.sum(df))/freq
    return np.nan if ann<=0 else float((1.0-df[-1])/ann)*100.0

if __name__ == "__main__":
    print("== 10 meccanismo ==")
    B   = pd.read_csv(PROC/"basis_daily.csv", parse_dates=["date"])
    CRV = pd.read_csv(PROC/"curve_params.csv", index_col=0, parse_dates=True)
    MKT = pd.read_csv(PROC/"curves_market.csv", index_col=0, parse_dates=True)
    # NB: NON chiamare questa variabile "C": nelle formule di statsmodels/patsy C() e'
    # la funzione per le categoriche, e statsmodels valuta la formula nel namespace
    # del chiamante. Un DataFrame di nome C fa fallire ogni regressione con
    # "TypeError: 'DataFrame' object is not callable".
    CSTAT = pd.read_csv(PROC/"static_cct.csv", parse_dates=["maturity","issue"]).set_index("isin")

    # ---- variabili di stress, tutte da dati gia' presenti -------------------
    S = pd.DataFrame(index=CRV.index)
    # P1: premio dell'Italia sopra il settore bancario, a 5 anni
    pars = CRV[["b0","b1","b2","b3","t1","t2"]].values
    S["sov5"] = [par_yield(p,5.0) if np.isfinite(p).all() else np.nan for p in pars]
    S["sov_swap"] = S["sov5"] - MKT["irs5y"].reindex(S.index)
    # P2: stress di funding bancario. Tenor non perfettamente allineati (Euribor 6M contro
    # OIS 1 anno): approssimazione dichiarata, cattura la dinamica non il livello esatto.
    if {"euribor6m","ois1y"}.issubset(MKT.columns):
        S["eur_ois"] = MKT["euribor6m"].reindex(S.index) - MKT["ois1y"].reindex(S.index)
    # volatilita' realizzata del tasso sovrano a 5 anni (controllo)
    S["rate_vol"] = S["sov5"].diff().rolling(21).std()*np.sqrt(252)*100
    S = S.ffill()

    # ---- panel mensile ------------------------------------------------------
    B["ym"] = B.date.dt.to_period("M")
    M = (B.groupby(["CCT_ISIN","regime","ym"])
           .agg(basis_p=("basis3_p","mean"), basis_y=("basis3_y","mean"),
                tau=("tau_cct","mean"), n=("basis3_p","size")).reset_index())
    M = M[M.n>=10]
    Sm = S.resample("ME").mean(); Sm["ym"] = Sm.index.to_period("M")
    M = M.merge(Sm[["sov_swap","eur_ois","rate_vol","ym"]], on="ym", how="left")
    # caratteristiche di liquidita' del singolo CCT
    M["amt"]  = M.CCT_ISIN.map(CSTAT["amt"])
    M["logamt"] = np.log(M.amt.replace(0,np.nan))
    M["iss"]  = M.CCT_ISIN.map(CSTAT["issue"])
    M["age"]  = (M.ym.dt.to_timestamp() - M["iss"]).dt.days/365.25
    M["date"] = M.ym.dt.to_timestamp()
    M = M.dropna(subset=["basis_p","sov_swap"])
    M.to_csv(PROC/"mechanism_panel.csv", index=False)

    L=[]; P=L.append
    P("=== 10 IL MECCANISMO: fuga verso la liquidita' dentro la curva sovrana ===")
    P(f"panel mensile: {len(M):,} osservazioni | {M.CCT_ISIN.nunique()} CCT | "
      f"{M.date.min().date()} -> {M.date.max().date()}")
    P(f"\nvariabili di stress (mediana | min | max):")
    for c,lab in [("sov_swap","Italia sopra swap 5y (%)"),("eur_ois","Euribor-OIS (%)"),
                  ("rate_vol","vol tasso sovrano 5y")]:
        if c in M and M[c].notna().any():
            P(f"  {lab:28s}: {M[c].median():7.3f} | {M[c].min():7.3f} | {M[c].max():7.3f}")
    P(f"  ammontare emesso CCT (mld): mediana {M.amt.median()/1e9:.1f}, "
      f"min {M.amt.min()/1e9:.1f}, max {M.amt.max()/1e9:.1f}")

    try:
        import statsmodels.formula.api as smf
        def run(d, f, lab, keys):
            try:
                r=smf.ols(f,data=d).fit(cov_type="cluster",cov_kwds={"groups":d["CCT_ISIN"]})
                out=[]
                for k in keys:
                    out.append(f"{k}={r.params.get(k,np.nan):+.3f}[{r.tvalues.get(k,np.nan):+.2f}]")
                P(f"  {lab:32s} " + "  ".join(out) + f"   R2adj {r.rsquared_adj:.3f} n {int(r.nobs):,}")
                return r
            except Exception as e:
                P(f"  {lab:32s} fallita ({str(e)[:50]})"); return None

        P("\n" + "="*74)
        P("P1-P2  SERIE STORICA: la base si sconta quando sale lo stress?")
        P("  segno atteso NEGATIVO: piu' stress -> CCT piu' economico (prezzo sotto il fair)")
        P("="*74)
        d=M.copy(); d["mon"]=d.date.dt.month.astype(str)
        run(d, "basis_p ~ sov_swap + tau + I(tau**2) + C(mon)", "(1) stress sovrano", ["sov_swap"])
        if "eur_ois" in d and d.eur_ois.notna().sum()>200:
            run(d.dropna(subset=["eur_ois"]), "basis_p ~ eur_ois + tau + I(tau**2) + C(mon)",
                "(2) stress funding bancario", ["eur_ois"])
            run(d.dropna(subset=["eur_ois"]),
                "basis_p ~ sov_swap + eur_ois + rate_vol + tau + I(tau**2) + C(mon)",
                "(3) entrambi + vol", ["sov_swap","eur_ois"])
        run(d, "basis_p ~ sov_swap + tau + I(tau**2) + C(CCT_ISIN) + C(mon)",
            "(4) con effetti fissi di CCT", ["sov_swap"])

        P("\n" + "="*74)
        P("P3  SEZIONE TRASVERSALE: lo sconto e' piu' grande sui CCT meno liquidi?")
        P("  segno atteso POSITIVO su logamt (titolo piu' grande = piu' liquido = meno sconto)")
        P("="*74)
        run(d.dropna(subset=["logamt"]), "basis_p ~ logamt + age + tau + I(tau**2) + C(mon)",
            "(5) dimensione ed eta'", ["logamt","age"])

        P("\n" + "="*74)
        P("P4  INTERAZIONE: lo stress colpisce DI PIU' i titoli illiquidi?")
        P("  e' la predizione stringente: se non regge, la storia della liquidita' non regge.")
        P("  segno atteso POSITIVO sull'interazione (titolo grande -> stress morde meno)")
        P("="*74)
        dd=d.dropna(subset=["logamt"]).copy()
        dd["logamt_c"]=dd.logamt-dd.logamt.mean(); dd["sov_c"]=dd.sov_swap-dd.sov_swap.mean()
        run(dd, "basis_p ~ sov_c*logamt_c + tau + I(tau**2) + C(mon)",
            "(6) stress x dimensione", ["sov_c","logamt_c","sov_c:logamt_c"])
        run(dd, "basis_p ~ sov_c*logamt_c + tau + I(tau**2) + C(CCT_ISIN) + C(mon)",
            "(7) idem, con eff. fissi CCT", ["sov_c","sov_c:logamt_c"])
    except ImportError:
        P("[statsmodels non disponibile]")

    # --- CONTROLLO: l'ammontare emesso e' liquidita' o e' l'EPOCA di emissione? ---
    P("\n" + "="*74)
    P("CONTROLLO su P3-P4: l'ammontare emesso misura liquidita' o epoca?")
    P("="*74)
    M["iss_yr"] = pd.to_datetime(M["iss"]).dt.year
    cc = M[["logamt","iss_yr"]].dropna().corr().iloc[0,1]
    P(f"  corr(log ammontare, anno di emissione) = {cc:+.3f}")
    P(f"  ammontare mediano per epoca di emissione (mld):")
    for a,b in [(1986,1999),(2000,2009),(2010,2016),(2017,2026)]:
        w = M[(M.iss_yr>=a)&(M.iss_yr<=b)]
        if len(w)>50: P(f"    {a}-{b}: {w.amt.median()/1e9:6.1f}   (n {len(w):,})")
    P("  [se l'ammontare cresce fortemente con l'epoca, logamt e' un proxy del REGIME e")
    P("   non della liquidita': i risultati P3-P4 vanno riletti come effetto di periodo]")
    try:
        import statsmodels.formula.api as smf
        dd2 = M.dropna(subset=["logamt"]).copy()
        dd2["logamt_c"]=dd2.logamt-dd2.logamt.mean(); dd2["sov_c"]=dd2.sov_swap-dd2.sov_swap.mean()
        dd2["mon"]=dd2.date.dt.month.astype(str); dd2["yr"]=dd2.date.dt.year.astype(str)
        P("\n  interazione con EFFETTI FISSI DI ANNO (assorbe l'epoca):")
        try:
            r=smf.ols("basis_p ~ sov_c*logamt_c + tau + I(tau**2) + C(CCT_ISIN) + C(yr) + C(mon)",
                      data=dd2).fit(cov_type="cluster",cov_kwds={"groups":dd2["CCT_ISIN"]})
            P(f"    sov_c:logamt_c = {r.params.get('sov_c:logamt_c',np.nan):+.3f} "
              f"[{r.tvalues.get('sov_c:logamt_c',np.nan):+.2f}]   R2adj {r.rsquared_adj:.3f}")
            P("    [se l'interazione sopravvive agli effetti fissi di anno, non e' epoca]")
        except Exception as e:
            P(f"    fallita ({str(e)[:50]})")
        P("\n  interazione ENTRO REGIME (i due strumenti hanno taglie molto diverse):")
        for reg in ["CCT-BOT","CCTeu"]:
            w = dd2[dd2.regime==reg]
            if len(w)<200 or w.CCT_ISIN.nunique()<5: continue
            w = w.copy(); w["logamt_c"]=w.logamt-w.logamt.mean(); w["sov_c"]=w.sov_swap-w.sov_swap.mean()
            try:
                r=smf.ols("basis_p ~ sov_c*logamt_c + tau + I(tau**2) + C(mon)",
                          data=w).fit(cov_type="cluster",cov_kwds={"groups":w["CCT_ISIN"]})
                P(f"    {reg:8s}: sov_c={r.params.get('sov_c',np.nan):+.3f}"
                  f"[{r.tvalues.get('sov_c',np.nan):+.2f}]  "
                  f"interaz={r.params.get('sov_c:logamt_c',np.nan):+.3f}"
                  f"[{r.tvalues.get('sov_c:logamt_c',np.nan):+.2f}]  n {int(r.nobs):,}")
            except Exception: pass
    except ImportError: pass

    P("\n" + "="*74)
    P("COSA SERVE PER LA VERSIONE COMPLETA (dati non ancora scaricati)")
    P("="*74)
    P("  - CDS delle banche italiane (ITRXESE Index per il settore, o ISPIM/UCG CDS 5y):")
    P("    misura diretta dello stress della clientela, molto piu' pulita di Euribor-OIS.")
    P("  - detenzioni di titoli di Stato per SETTORE (Banca d'Italia, base dati BDS): permette")
    P("    la tabella di ownership analoga alla Tabella 7 di FL, che e' cio' che rende")
    P("    credibile l'attribuzione a una clientela specifica.")
    P("  - volumi o bid-ask su MTS, se accessibili: misura di liquidita' diretta invece dei")
    P("    proxy per dimensione ed eta'.")
    save_txt("10_mechanism.txt", L); print("\n".join(L))

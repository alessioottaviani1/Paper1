"""
07e - DOVE NASCE IL +7 bp: confronto diretto dei due modi di calcolare il residuo.

IL PUNTO. 06b calcola il residuo di un BTP sulla curva usando le DATE CEDOLARI VERE
(generate a ritroso dalla scadenza per mesi di calendario) e trova mediana ~0 in ogni fascia.
07 lo calcola sulla stessa curva ma con una GRIGLIA SINTETICA di mezzi anni esatti, e trova
+0.277 punti di prezzo. Stesso titolo, stessa curva, stesso giorno: la differenza deve stare
nella costruzione dei flussi, oppure nell'aggregazione.

Lo script calcola ENTRAMBI i residui sulle STESSE osservazioni e li scompone:
  A) griglia sintetica (metodo 07)   B) date vere (metodo 06b)   e la differenza A-B.
Riporta anche il residuo in bp usando la duration EFFETTIVA di ciascun titolo, invece di
convertire con una duration assunta -- passaggio in cui si perde facilmente il fattore giusto.

E verifica quanto spesso il tie-break on-the-run di 04 sia davvero vincolante: se decide in
pochi casi, non puo' spiegare un bias sistematico, qualunque sia la sua giustificazione.
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

def nss(tau,p):
    b0,b1,b2,b3,t1,t2=p; tau=np.maximum(np.asarray(tau,float),1e-8)
    x1,x2=tau/t1,tau/t2
    f1=(1-np.exp(-x1))/x1; f2=(1-np.exp(-x2))/x2
    return b0+b1*f1+b2*(f1-np.exp(-x1))+b3*(f2-np.exp(-x2))

def dates_back(issue, maturity, freq=CPN_FREQ):
    from dateutil.relativedelta import relativedelta
    step=12//freq; ds,d=[],pd.Timestamp(maturity)
    lim=pd.Timestamp(issue) if pd.notna(issue) else pd.Timestamp(maturity)-relativedelta(years=50)
    while d>lim: ds.append(d); d-=relativedelta(months=step)
    return np.array(sorted(ds))

def accrued(d, dates, coupon, freq=CPN_FREQ):
    if not coupon: return 0.0
    nxt=dates[dates>d]
    if len(nxt)==0: return 0.0
    nxt=nxt[0]; prv=dates[dates<=d]
    prv=prv[-1] if len(prv) else nxt-pd.DateOffset(months=12//freq)
    per=(nxt-prv).days
    return 0.0 if per<=0 else coupon/freq*((d-prv).days/per)

if __name__ == "__main__":
    print("== 07e isolamento del gap ==")
    SB=pd.read_csv(PROC/"static_btp.csv",parse_dates=["maturity","issue"]).set_index("isin")
    PXB=pd.read_csv(PROC/"px_btp.csv",index_col=0,parse_dates=True)
    CRV=pd.read_csv(PROC/"curve_params.csv",index_col=0,parse_dates=True)
    PAIRS=pd.read_csv(PROC/"pairs_cct_btp.csv")
    twins=list(PAIRS.BTP_ISIN.dropna().unique())

    DT={i:dates_back(SB.loc[i,"issue"],SB.loc[i,"maturity"]) for i in twins if i in SB.index}
    dts=[d for d in CRV.index if pd.Timestamp(START_PRIMARY)<=d<=pd.Timestamp(END_SAMPLE)]
    dts=dts[::max(len(dts)//600,1)]

    rows=[]
    for d in dts:
        if d not in PXB.index: continue
        p=CRV.loc[d,["b0","b1","b2","b3","t1","t2"]].values.astype(float)
        if not np.isfinite(p).all(): continue
        for isin in twins:
            if isin not in PXB.columns or isin not in DT: continue
            px=PXB.loc[d,isin]
            if not np.isfinite(px) or not (20<float(px)<200): continue
            mat=SB.loc[isin,"maturity"]; cb=float(SB.loc[isin,"coupon"] or 0.0)
            tau=(mat-d).days/365.25
            if not (1.0<=tau<=8.0): continue
            dd=DT[isin]; fut=dd[dd>d]
            if len(fut)==0: continue
            pobs=float(px)+accrued(d,dd,cb)

            # (A) metodo 07: griglia sintetica di mezzi anni
            nb=max(int(round(tau*CPN_FREQ)),1)
            tA=np.array([(i+1)/CPN_FREQ for i in range(nb)]); tA=tA-(tA[-1]-tau); tA=tA[tA>0]
            aA=np.full(len(tA),cb/CPN_FREQ); aA[-1]+=100.0
            pA=float(np.sum(aA*np.exp(-nss(tA,p)/100.0*tA)))

            # (B) metodo 06b: date cedolari vere
            tB=np.array([(x-d).days/365.25 for x in fut])
            aB=np.full(len(tB),cb/CPN_FREQ); aB[-1]+=100.0
            pB=float(np.sum(aB*np.exp(-nss(tB,p)/100.0*tB)))

            w=aB*np.exp(-0.03*tB); dur=max(float(np.sum(w*tB)/np.sum(w)),0.05)
            rows.append({"date":d,"isin":isin,"tau":tau,"dur":dur,"pobs":pobs,
                         "resA_p":pobs-pA,"resB_p":pobs-pB,
                         "resA_bp":(pobs-pA)/(dur*pobs)*1e4,
                         "resB_bp":(pobs-pB)/(dur*pobs)*1e4,
                         "n_cf_A":len(tA),"n_cf_B":len(tB)})
    D=pd.DataFrame(rows)

    L=[]; P=L.append
    P("=== 07e DOVE NASCE IL GAP ===")
    P(f"osservazioni (date x BTP gemelli, 1-8 anni): {len(D):,}")
    P("\nresiduo del BTP sulla curva, stessi titoli e stessa curva, due costruzioni:")
    P(f"  {'':28}{'in PREZZO':>14}{'in bp':>12}")
    P(f"  {'(A) griglia sintetica  [07]':28}{D.resA_p.median():>14.4f}{D.resA_bp.median():>12.2f}")
    P(f"  {'(B) date cedolari vere [06b]':28}{D.resB_p.median():>14.4f}{D.resB_bp.median():>12.2f}")
    P(f"  {'differenza A-B':28}{(D.resA_p-D.resB_p).median():>14.4f}"
      f"{(D.resA_bp-D.resB_bp).median():>12.2f}")
    P(f"\n  numero di flussi diverso fra A e B: {(D.n_cf_A!=D.n_cf_B).mean():.1%} delle osservazioni")
    P(f"  duration effettiva mediana: {D.dur.median():.2f} anni | prezzo mediano {D.pobs.median():.2f}")
    P("\n  [se A e B coincidono, la griglia sintetica NON e' il problema e il residuo dei")
    P("   gemelli e' davvero quello: allora il confronto con 06b va rifatto sull'AGGREGAZIONE]")
    P("\nresiduo (B, metodo corretto) per fascia di scadenza:")
    for lo,hi in [(1,2),(2,3),(3,4.5),(4.5,8)]:
        v=D[(D.tau>=lo)&(D.tau<hi)]
        if len(v)>50: P(f"  {f'{lo}-{hi}y':>10}: {v.resB_bp.median():7.2f} bp | n {len(v):,}")

    # quanto pesa davvero il tie-break on-the-run
    P("\n=== il tie-break ON-THE-RUN e' vincolante? ===")
    C=pd.read_csv(PROC/"static_cct.csv",parse_dates=["maturity"])
    B2=SB.reset_index()
    n_tie=0; n_tot=0
    for _,c in C.iterrows():
        if pd.isna(c["maturity"]): continue
        cand=B2[(B2.maturity-c["maturity"]).abs()<=pd.Timedelta(days=MAX_MISMATCH_D)].copy()
        if cand.empty: continue
        cand["mm"]=(cand.maturity-c["maturity"]).dt.days
        best=cand[cand.mm.abs()==cand.mm.abs().min()]
        best=best[(best.mm>0)==(best.mm>0).iloc[0]] if len(best)>1 else best
        n_tot+=1; n_tie+= int(len(best)>1)
    P(f"  CCT per cui il terzo criterio (on-the-run) decide davvero: {n_tie}/{n_tot} ({n_tie/max(n_tot,1):.0%})")
    P("  [se e' una frazione piccola, il tie-break non puo' spiegare un bias sistematico]")
    save_txt("07e_isolate_gap.txt", L); print("\n".join(L))

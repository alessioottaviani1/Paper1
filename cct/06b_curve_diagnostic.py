"""
06b - LA CURVA E' STIMATA BENE? Diagnosi dei residui e ruolo dei BOT.

IL SINTOMO. La curva e' fittata SUI BTP, quindi i loro residui dovrebbero essere centrati
su zero. Invece in 07 il BTP appaiato risulta caro di ~7-9 bp in mediana: un bias
sistematico piu' grande dell'RMSE del fit (5.9 bp). Qualcosa tira la curva.

IL SOSPETTO. Il fit include i BOT, con peso 1/(duration x prezzo). Un BOT a 0.25 anni pesa
venti volte un BTP a 5 anni: con ~16 BOT vivi la somma dei loro pesi domina, e poiche' NSS ha
sei parametri GLOBALI, la distorsione del tratto corto si propaga a tutta la curva. BOT e BTP
hanno inoltre clientele diverse -- i primi sono strumenti di mercato monetario -- e possono
trattare su curve leggermente diverse.

IL PRECEDENTE. Gurkaynak-Sack-Wright (2007) ESCLUDONO i Treasury bill dal loro fit proprio
perche' trattano a rendimenti fuori linea rispetto a note e bond. Qui li ho inclusi per
ancorare il tratto corto: la scelta va verificata, non assunta.

IL TEST. Su un campione di date si rifitta la curva in tre varianti -- con BOT (attuale),
senza BOT, e con BOT ma peso ridotto -- e si confrontano i residui dei soli BTP. Se togliendo
i BOT il bias sui BTP sparisce, la curva va rifatta senza BOT (o con uno spread dedicato).

Output: results/06b_curve_diagnostic.txt
"""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from config import *
from utils import save_txt

N_DATES = 400          # campione di date su cui rifittare (il test non richiede tutte)
NSS_BOUNDS = (np.array([-5.0,-20.0,-50.0,-50.0,0.05,1.0]), np.array([15.0,20.0,50.0,50.0,8.0,30.0]))

def nss(tau,p):
    b0,b1,b2,b3,t1,t2=p; tau=np.maximum(np.asarray(tau,float),1e-8)
    x1,x2=tau/t1,tau/t2
    f1=(1-np.exp(-x1))/x1; f2=(1-np.exp(-x2))/x2
    return b0+b1*f1+b2*(f1-np.exp(-x1))+b3*(f2-np.exp(-x2))

def cashflows(issue, maturity, coupon, freq=CPN_FREQ):
    from dateutil.relativedelta import relativedelta
    step=12//freq; ds,d=[],pd.Timestamp(maturity)
    lim=pd.Timestamp(issue) if pd.notna(issue) else pd.Timestamp(maturity)-relativedelta(years=30)
    while d>lim: ds.append(d); d-=relativedelta(months=step)
    ds=sorted(ds); a=np.full(len(ds),(coupon or 0.0)/freq); a[-1]+=100.0
    return np.array([pd.Timestamp(x) for x in ds]), a

def accrued(d, dates, coupon, freq=CPN_FREQ):
    if not coupon: return 0.0
    nxt=dates[dates>d]
    if len(nxt)==0: return 0.0
    nxt=nxt[0]; prv=dates[dates<=d]
    prv=prv[-1] if len(prv) else nxt-pd.DateOffset(months=12//freq)
    per=(nxt-prv).days
    return 0.0 if per<=0 else coupon/freq*((d-prv).days/per)

def fit(cf_t, cf_a, po, dm, x0=None):
    ft=np.concatenate(cf_t); fa=np.concatenate(cf_a)
    idx=np.concatenate([np.full(len(t),i) for i,t in enumerate(cf_t)]).astype(int)
    nb=len(cf_t); po=np.asarray(po,float); w=1.0/(np.asarray(dm,float)*po)
    def resid(p):
        z=nss(ft,p)
        pm=np.bincount(idx,weights=fa*np.exp(-z/100.0*ft),minlength=nb)
        return (pm-po)*w
    st=([np.asarray(x0,float)] if x0 is not None else [])+[np.array([4.0,-1.0,0.0,0.0,1.5,10.0])]
    best=None
    for s in st:
        try:
            r=least_squares(resid,s,bounds=NSS_BOUNDS,method="trf",max_nfev=3000)
            if best is None or r.cost<best.cost: best=r
        except Exception: pass
    return best

if __name__ == "__main__":
    print("== 06b diagnostica curva ==")
    meta={}
    for lab in ("btp","bot"):
        st=pd.read_csv(PROC/f"static_{lab}.csv",parse_dates=["maturity","issue"]).set_index("isin")
        for isin,r in st.iterrows():
            if pd.isna(r["maturity"]): continue
            c=0.0 if lab=="bot" else float(r.get("coupon") or 0.0)
            try: dd,aa=cashflows(r["issue"],r["maturity"],c)
            except Exception: continue
            meta[isin]={"dates":dd,"amts":aa,"cpn":c,"kind":lab}
    PX={l:pd.read_csv(PROC/f"px_{l}.csv",index_col=0,parse_dates=True) for l in ("btp","bot")}
    dates=sorted(set(PX["btp"].index)|set(PX["bot"].index))
    dates=[d for d in dates if pd.Timestamp(START_PRIMARY)<=d<=pd.Timestamp(END_SAMPLE)]
    sel=dates[::max(len(dates)//N_DATES,1)]

    out=[]
    for d in sel:
        rec={}
        for lab in ("btp","bot"):
            if d not in PX[lab].index: continue
            for isin,p in PX[lab].loc[d].dropna().items():
                m=meta.get(isin)
                if m is None or not (20.0<float(p)<200.0): continue
                fut=m["dates"]>d
                if not fut.any(): continue
                tau=np.array([(x-d).days/365.25 for x in m["dates"][fut]])
                if not (CURVE_EXCL_TAU<=tau[-1]<=CURVE_MAX_TAU): continue
                a=m["amts"][fut]; ww=a*np.exp(-0.03*tau)
                rec[isin]={"t":tau,"a":a,"p":float(p)+accrued(d,m["dates"],m["cpn"]),
                           "dm":max(float(np.sum(ww*tau)/np.sum(ww)),0.05),"k":m["kind"],"tau":tau[-1]}
        if len(rec)<12: continue
        variants={}
        for name, keep in [("con BOT (attuale)", lambda k: True), ("senza BOT", lambda k: k=="btp")]:
            ks=[i for i,v in rec.items() if keep(v["k"])]
            if len(ks)<8: continue
            r=fit([rec[i]["t"] for i in ks],[rec[i]["a"] for i in ks],
                  [rec[i]["p"] for i in ks],[rec[i]["dm"] for i in ks])
            if r is None: continue
            variants[name]=r.x
        if len(variants)<2: continue
        row={"date":d}
        for name,px in variants.items():
            for kind in ("btp","bot"):
                ks=[i for i,v in rec.items() if v["k"]==kind]
                if not ks: continue
                res=[]
                for i in ks:
                    pm=float(np.sum(rec[i]["a"]*np.exp(-nss(rec[i]["t"],px)/100.0*rec[i]["t"])))
                    res.append((rec[i]["p"]-pm)/(rec[i]["dm"]*rec[i]["p"])*1e4)
                row[f"{name}|{kind}"]=float(np.median(res))
        # --- residui dei BTP per FASCIA DI SCADENZA, con la curva attuale (con BOT).
        # E' il test decisivo: 06b misura TUTTI i BTP (1-30 anni) e trova bias ~0, mentre 07
        # misura solo i ~60 appaiati ai CCT, che stanno tutti fra 1 e 7 anni. Se NSS a sei
        # parametri non cattura la curvatura in quella fascia, i residui possono essere
        # centrati nel complesso e sistematicamente diversi da zero LI' DENTRO -- il che
        # contaminerebbe la misura (3) e non la (1).
        px = variants.get("con BOT (attuale)")
        if px is not None:
            for lo,hi in [(0.5,1.5),(1.5,3),(3,4.5),(4.5,7),(7,12),(12,31)]:
                ks=[i for i,v in rec.items() if v["k"]=="btp" and lo<=v["tau"]<hi]
                if len(ks)<2: continue
                res=[]
                for i in ks:
                    pm=float(np.sum(rec[i]["a"]*np.exp(-nss(rec[i]["t"],px)/100.0*rec[i]["t"])))
                    res.append((rec[i]["p"]-pm)/(rec[i]["dm"]*rec[i]["p"])*1e4)
                row[f"bucket|{lo}-{hi}"]=float(np.median(res))
        out.append(row)

    D=pd.DataFrame(out).set_index("date")
    L=[]; P=L.append
    P("=== 06b DIAGNOSTICA DELLA CURVA: i BOT distorcono il fit? ===")
    P(f"date campionate: {len(D):,} fra {D.index.min().date()} e {D.index.max().date()}")
    P("\nresiduo MEDIANO in bp (osservato - modello; positivo = titolo CARO rispetto alla curva)")
    P(f"  {'variante':>22}{'sui BTP':>12}{'sui BOT':>12}")
    for name in ["con BOT (attuale)", "senza BOT"]:
        cb=D.get(f"{name}|btp"); co=D.get(f"{name}|bot")
        P(f"  {name:>22}{(cb.median() if cb is not None else float('nan')):>12.2f}"
          f"{(co.median() if co is not None else float('nan')):>12.2f}")
    a=D.get("con BOT (attuale)|btp"); b=D.get("senza BOT|btp")
    if a is not None and b is not None:
        P(f"\n  bias sui BTP: {a.median():.2f} bp con i BOT  ->  {b.median():.2f} bp senza")
        P(f"  riduzione: {abs(a.median())-abs(b.median()):+.2f} bp")
        P(f"  giorni in cui togliere i BOT riduce |bias|: {(b.abs()<a.abs()).mean():.0%}")
    P("\n" + "="*66)
    P("TEST DECISIVO: residui dei BTP per FASCIA DI SCADENZA (curva attuale)")
    P("="*66)
    P("  I CCT vivono fra 1 e 7 anni: se il bias e' concentrato li', la misura (3)")
    P("  (CCT contro curva) ne eredita l'errore, mentre la (1) (CCT contro BTP con")
    P("  lo stesso bias) lo cancella. Sarebbe l'opposto di quanto assunto finora.")
    P(f"\n  {'fascia':>12}{'residuo mediano':>18}{'p25':>9}{'p75':>9}{'n giorni':>10}")
    for lo,hi in [(0.5,1.5),(1.5,3),(3,4.5),(4.5,7),(7,12),(12,31)]:
        c=D.get(f"bucket|{lo}-{hi}")
        if c is None or c.dropna().empty: continue
        v=c.dropna()
        star = "  <-- fascia CCT" if hi<=7 else ""
        P(f"  {f'{lo}-{hi}y':>12}{v.median():>18.2f}{v.quantile(.25):>9.2f}{v.quantile(.75):>9.2f}"
          f"{len(v):>10,}{star}")
    P("\n  Se le fasce 0.5-7y hanno residuo mediano vicino a zero come le altre, il +7 bp")
    P("  visto in 07 NON viene dalla curva e va cercato nel calcolo del fair price in 07.")
    P("\nLETTURA")
    P("  Se il bias sui BTP sparisce togliendo i BOT, la curva va rifittata SENZA BOT")
    P("  (come fa GSW con i bill) oppure con uno spread dedicato al mercato monetario.")
    P("  Se il bias resta, la causa e' altrove e va cercata nel trimming o nelle convenzioni.")
    save_txt("06b_curve_diagnostic.txt", L); print("\n".join(L))

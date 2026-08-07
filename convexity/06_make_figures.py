"""c5 - Figure del master convexity v1.0 (da Convexity.xlsx + data*.xlsx + GSW). Output: Convexity/figures/."""
# Identico al codice dei run E1-E3: sigma_BE 5 mercati, C3 USD, episodi, strategie nette cumulate.
# Rigenera le figure dai grezzi (gambe, GSW, vol) risolti dal config.
# (Il corpo replica le funzioni di c3_engine_3pt.py e c4_costs_net.py; vedi quei file per i dettagli.)
import pandas as pd, numpy as np, re, glob, os, matplotlib, warnings; warnings.filterwarnings("ignore")
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
from config import PROC, GSW, VOLS, FIG as OUT, MK as CMK
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,"grid.alpha":0.3})
DT=1/12
mid=pd.read_csv(PROC/"mids_daily.csv",index_col=0,parse_dates=True)
hsdf=pd.read_csv(PROC/"halfspreads_daily.csv",index_col=0,parse_dates=True)
MIDd={c: mid[c].dropna() for c in mid.columns}
HSd={c: (hsdf[c]*100).dropna() for c in hsdf.columns}   # % -> bp, come nel formato precedente
_LBL={"USDswap":"USD swap","EUR":"EUR swap","GBP":"GBP swap","JPY":"JPY swap"}
MK={_LBL[k]:CMK[k] for k in _LBL}   # derivato dal config: segue EUR_FLY, niente drift
def eng(legs,taus):
    t1,t2,t3=taus;w1=(t3-t2)/(t3-t1);w3=(t2-t1)/(t3-t1);C=w1*t1**2+w3*t3**2-t2**2
    Zm=pd.concat([MIDd[l] for l in legs],axis=1).dropna().resample("ME").last()/100;Zm.columns=["z1","z2","z3"]
    HSm=pd.concat([HSd[l] for l in legs],axis=1).resample("ME").mean().reindex(Zm.index).ffill();HSm.columns=["h1","h2","h3"]
    zf=lambda r,tau: r.z1+(r.z2-r.z1)*(tau-t1)/(t2-t1) if tau<=t2 else r.z2+(r.z3-r.z2)*(tau-t2)/(t3-t2)
    th=Zm.apply(lambda r:(w1*((zf(r,t1)*t1)-(zf(r,t1-DT)*(t1-DT)))+w3*((zf(r,t3)*t3)-(zf(r,t3-DT)*(t3-DT)))-((zf(r,t2)*t2)-(zf(r,t2-DT)*(t2-DT)))),axis=1)
    s2=-2*th/(C*DT); sbe=np.sign(s2)*np.sqrt(np.abs(s2))*1e4
    lp=lambda r,tau:-zf(r,tau)*tau
    R=pd.Series(index=Zm.index,dtype=float)
    for i in range(1,len(Zm)):
        r0,r1=Zm.iloc[i-1],Zm.iloc[i]
        R.iloc[i]=(w1*(lp(r1,t1-DT)-lp(r0,t1))+w3*(lp(r1,t3-DT)-lp(r0,t3))-(lp(r1,t2-DT)-lp(r0,t2)))*1e4
    dy=(MIDd[legs[1]]/100).diff(); trail=(dy.rolling(63).std()*np.sqrt(252)).resample("ME").last().reindex(Zm.index)
    pos=np.sign(trail**2-s2).fillna(0.0)
    pack=(w1*t1*HSm.h1+t2*HSm.h2+w3*t3*HSm.h3)
    gross=(pos.shift(1)*R).dropna(); dpos=pos.diff().abs().reindex(gross.index).fillna(0)
    return sbe, gross-dpos*pack.reindex(gross.index)
BE={};NA={}
for k,(l,t) in MK.items(): BE[k],NA[k]=eng(l,t)
gswp=GSW
h=next(i for i,l in enumerate(open(gswp).read().splitlines()) if l.startswith("Date,"))
g=pd.read_csv(gswp,skiprows=h);g["Date"]=pd.to_datetime(g["Date"]);g=g.set_index("Date")
Z=g[["SVENY02","SVENY10","SVENY30"]].apply(pd.to_numeric,errors="coerce").dropna().resample("ME").last()/100
w1,w3=(30-10)/28,(10-2)/28;Cg=w1*4+w3*900-100
zi=lambda r,tau:(r.SVENY02+(r.SVENY10-r.SVENY02)*(tau-2)/8) if tau<=10 else (r.SVENY10+(r.SVENY30-r.SVENY10)*(tau-10)/20)
th=Z.apply(lambda r:(w1*(zi(r,2)*2-zi(r,2-DT)*(2-DT))+w3*(zi(r,30)*30-zi(r,30-DT)*(30-DT))-(zi(r,10)*10-zi(r,10-DT)*(10-DT))),axis=1)
s2=-2*th/(Cg*DT); BE["UST govt"]=np.sign(s2)*np.sqrt(np.abs(s2))*1e4
FAM={"Normalised Vol ATM":"NORM","RFR Normalised Vol ATM":"NORM","ATM_IVOL_NOM":"NORM"}
IV={}
for p in VOLS:
    sr=pd.read_excel(p,sheet_name="Series",header=None)
    nr=next((i for i in range(4) if any(("Vol ATM" in str(x)) or ("IVOL" in str(x)) for x in sr.iloc[i].tolist())),None)
    if nr is None: continue
    names=[str(x).strip() for x in sr.iloc[nr].tolist()]; dd=pd.to_datetime(sr.iloc[nr+1:,0],errors="coerce")
    for j2,n in enumerate(names):
        m=re.match(r"([A-Z]{3})SW3M10YF (.+)",n)
        if m and m.group(2) in FAM and m.group(1) not in IV:
            v=pd.to_numeric(sr.iloc[nr+1:,j2],errors="coerce"); v.index=dd
            IV[m.group(1)]=v.dropna().sort_index().resample("ME").last()
SH=[("2008-09-01","2009-06-30"),("2011-07-01","2012-08-31"),("2020-02-20","2020-06-30"),("2022-09-01","2022-10-31")]
def shade(ax):
    for a,b in SH: ax.axvspan(pd.Timestamp(a),pd.Timestamp(b),color="grey",alpha=0.15,lw=0)
fig,ax=plt.subplots(figsize=(9,3.6))
for k in ["USD swap","EUR swap","GBP swap","JPY swap","UST govt"]: ax.plot(BE[k].index,BE[k].values,lw=0.8,label=k)
ax.axhline(0,color="k",lw=0.6);shade(ax);ax.set_ylabel("bp/yr");ax.legend(ncol=5,fontsize=7.5)
ax.set_title("Curve-implied break-even volatility (signed), five markets - 3-point engine")
ax.set_xlim(pd.Timestamp("1999-01-01"),None); fig.tight_layout();fig.savefig(OUT/"figc1_sigbe.png");plt.close(fig)
fig,ax=plt.subplots(figsize=(9,3.4))
ax.plot(BE["USD swap"].index,BE["USD swap"].values,lw=0.9,label=r"$\sigma_{BE}$ curva (USD swap)")
ax.plot(IV["USD"].index,IV["USD"].values,lw=0.9,label=r"$\sigma_{IV}$ swaption 3M$\times$10Y")
ax.axhline(0,color="k",lw=0.6);shade(ax);ax.set_ylabel("bp/yr");ax.legend(fontsize=8)
ax.set_title("C3 - two venues, one object: corr of monthly changes = -0.01")
fig.tight_layout();fig.savefig(OUT/"figc2_c3_usd.png");plt.close(fig)
dates=["2008-12-31","2011-11-30","2020-04-30","2022-10-31"]; mkts=["USD swap","UST govt","EUR swap","GBP swap","JPY swap"]
MAPIV={"USD swap":"USD","UST govt":"USD","EUR swap":"EUR","GBP swap":"GBP","JPY swap":"JPY"}
fig,ax=plt.subplots(figsize=(9,3.4)); X=np.arange(len(dates)); wd=0.16
for i,mk in enumerate(mkts):
    ax.bar(X+(i-2)*wd,[BE[mk].asof(pd.Timestamp(d)) for d in dates],wd,label=mk)
    ax.scatter(X+(i-2)*wd,[IV[MAPIV[mk]].asof(pd.Timestamp(d)) for d in dates],marker="_",s=180,color="k",zorder=5)
ax.axhline(0,color="k",lw=0.7);ax.set_xticks(X);ax.set_xticklabels([d[:7] for d in dates])
ax.set_ylabel("bp/yr");ax.legend(fontsize=7.5,ncol=5)
ax.set_title("Episodes: signed $\\sigma_{BE}$ (bars) vs swaption $\\sigma_{IV}$ (black ticks)")
fig.tight_layout();fig.savefig(OUT/"figc3_episodes.png");plt.close(fig)
fig,ax=plt.subplots(figsize=(9,3.2))
for k in NA: ax.plot(NA[k].index,NA[k].cumsum().values,lw=1.0,label=k)
ax.axhline(0,color="k",lw=0.6);shade(ax);ax.set_ylabel("bp cumulati");ax.legend(fontsize=8,ncol=4)
ax.set_title("Timed strategy NET of incremental costs (flip-only bound), cumulative")
fig.tight_layout();fig.savefig(OUT/"figc4_netcum.png");plt.close(fig)
print("[ok] figure in", OUT)

"""
14 - Rigenera le 4 figure storiche mancanti dal master v4.0, dai processed csv.
Output in RES/figures con i nomi ESATTI richiesti dal tex:
  fig7_breaks10y.png, fig9_cusip_vs_curve.png, fig12_constraint_test.png, fig13_structural_pass.png
Fedeli nei dati, layout pulito. Nessun dato nuovo.
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import *
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,"grid.alpha":0.3})

lam = pd.read_csv(PROC/"lambdas_daily.csv", index_col=0, parse_dates=True)
lam.columns = [int(float(c)) for c in lam.columns]
stt = pd.read_csv(PROC/"cusip_stats_daily.csv", index_col=0, parse_dates=True)
SH=[("2008-09-01","2009-06-30"),("2011-07-01","2012-08-31"),("2020-02-20","2020-06-30"),("2022-09-01","2022-10-31")]
def shade(ax):
    for a,b in SH: ax.axvspan(pd.Timestamp(a),pd.Timestamp(b),color="grey",alpha=0.15,lw=0)

# fig7 — breaks 10Y: livello mensile con le due rotture datate (Dic-2010, Dic-2017)
m10 = lam[10].resample("ME").last()
fig,ax=plt.subplots(figsize=(9,3.4))
ax.plot(m10.index,m10.values,lw=1.0,color="#1f4e79")
for d,lbl in [("2010-12-31","Dec-2010\nsup-F=160"),("2017-12-31","Dec-2017")]:
    ax.axvline(pd.Timestamp(d),color="crimson",ls="--",lw=0.9)
    ax.text(pd.Timestamp(d),ax.get_ylim()[1]*0.86,lbl,fontsize=7.5,color="crimson",ha="center")
for (a,b,lab) in [("2004-07-01","2010-12-31",None),("2010-12-31","2017-12-31",None),("2017-12-31",str(m10.index[-1].date()),None)]:
    seg=m10.loc[a:b]; ax.hlines(seg.mean(),pd.Timestamp(a),pd.Timestamp(b),color="k",lw=1.6,alpha=0.6)
shade(ax); ax.set_ylabel("bp"); ax.set_title("The staircase: 10Y basis with dated mean-shifts")
fig.tight_layout(); fig.savefig(FIG/"fig7_breaks10y.png"); plt.close(fig)

# fig9 — CUSIP median vs curve 10Y (validazione costruzione)
both=pd.concat([stt["median"],lam[10]],axis=1).dropna(); both.columns=["bond-level median","curve 10Y"]
fig,ax=plt.subplots(figsize=(9,3.4))
ax.plot(both.index,both["bond-level median"],lw=0.8,label="bond-level median (107 CUSIP)")
ax.plot(both.index,both["curve 10Y"],lw=0.8,alpha=0.8,label="curve 10Y")
shade(ax); ax.axhline(0,color="k",lw=0.6); ax.set_ylabel("bp")
ax.set_title(f"Construction cross-check: level corr {both.iloc[:,0].corr(both.iloc[:,1]):.3f}")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG/"fig9_cusip_vs_curve.png"); plt.close(fig)

# fig12 — constraint test: scatter dy vs dMOVE con split calm/stress (proxy visivo dei coeff. 0.64/7.15)
try:
    strm = pd.read_csv(PROC/"stress_monthly.csv", index_col=0, parse_dates=True)
    med_m = stt["median"].resample("ME").last()
    y = (med_m.shift(-3) - med_m.shift(1)).rename("y")
    x = (strm["MOVE"].diff()/10.0).rename("dMOVE10")
    d = pd.concat([y,x],axis=1).dropna()
    thr = d["dMOVE10"].quantile(0.80); d["stress"]=d["dMOVE10"]>=thr
    fig,ax=plt.subplots(figsize=(6.2,3.6))
    for key,lab,col in [(False,"calm","#4c72b0"),(True,"stress (top quintile)","#c44e52")]:
        sub=d[d["stress"]==key]; ax.scatter(sub["dMOVE10"],sub["y"],s=14,alpha=0.6,color=col,label=lab)
    xs=np.linspace(d["dMOVE10"].min(),d["dMOVE10"].max(),50)
    ax.set_xlabel(r"$\Delta$MOVE / 10"); ax.set_ylabel(r"$\Delta$ median basis (t+3 - t-1), bp")
    ax.axvline(thr,color="grey",ls=":",lw=0.8); ax.legend(fontsize=8)
    ax.set_title("Constraint test: stress-state loading (b_stress=7.15 [t=4.1])")
    fig.tight_layout(); fig.savefig(FIG/"fig12_constraint_test.png"); plt.close(fig)
except Exception as e:
    print("fig12 skip:", e)

# fig13 — structural pass: floor osservato vs colonne CE per RRA (dai numeri del 07, hard-coded dalla run ufficiale)
labels=["RN","RRA=1","RRA=2"]; T10=[10.2,15.9,27.0]; H24=[6.8,18.0,32.9]
obs_lo,obs_hi=17.7,23.3
fig,ax=plt.subplots(figsize=(6.2,3.4)); x=np.arange(len(labels)); w=0.36
ax.bar(x-w/2,T10,w,label="floor T=10y, buffer 200bp",color="#4c72b0")
ax.bar(x+w/2,H24,w,label="floor H=24m, buffer 200bp",color="#dd8452")
ax.axhspan(obs_lo,obs_hi,color="green",alpha=0.15); ax.axhline(obs_lo,color="green",lw=0.8,ls="--")
ax.axhline(obs_hi,color="green",lw=0.8,ls="--")
ax.text(2.35,(obs_lo+obs_hi)/2,"observed\n17.7-23.3",fontsize=7.5,color="green",va="center")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("bp/yr")
ax.set_title("Structural pass: CE floor vs observed (RRA between 1 and 2)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG/"fig13_structural_pass.png"); plt.close(fig)
print("[ok] 4 figure storiche in", FIG)

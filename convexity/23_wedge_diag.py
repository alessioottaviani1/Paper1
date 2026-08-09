"""23 - IL GRAFICO PER REBONATO: la differenza e' persistente o pochi episodi (dati sporchi)?

Rebonato chiede se "c'e' un effetto da spiegare". Dopo il 21/22 la risposta pulita e' sul RESIDUO
epurato dal VRP, R = sigma_BE - E[RV]: quello e' il cuneo al netto del premio, cioe' la parte che
la segmentazione dovrebbe spiegare. Questo script visualizza R nel tempo per i quattro mercati e
affianca i CONTROLLI che distinguono un fenomeno diffuso da pochi spike o outlier sporchi:

 (a) serie mensile di R con banda +/-1 sd e finestre di crisi ombreggiate;
 (b) media di R con e senza il top/bottom 5% dei mesi (se crolla -> guidato dalle code);
 (c) frazione di mesi con R>0 (se ~50% e media grande -> pochi spike; se molto sbilanciata ->
     fenomeno persistente);
 (d) istogramma di R: una distribuzione centrata lontano da zero e' diffusa; una centrata su zero
     con poche code estreme e' "episodi".

Serve la stessa infrastruttura del 21/22 (sigbe_monthly.csv, swaption NORM, RV forward-friendly).
Salva in output/convexity/figures/figc5_wedge_purged.png e stampa i controlli a schermo.
"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import save_txt, load_legs_mid_all, load_vols
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,"grid.alpha":0.3})

sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
mid = load_legs_mid_all()
IV  = load_vols()
L=[]; P=L.append
P("=== 23 diagnostica del cuneo epurato R = sigma_BE - E[RV] ===")

H_M=3
MK4=[m for m in ["USDswap","EUR","GBP","JPY"] if m in sbe.columns]
FAM={"USDswap":"USOSFR","EUR":"EUSA","GBP":"BPSWS","JPY":"JYSO"}
CRISES=[("2008-09-01","2009-06-30"),("2011-07-01","2012-08-31"),
        ("2020-02-01","2020-05-31"),("2022-09-01","2022-12-31")]
LAB={"USDswap":"USD swap","EUR":"EUR","GBP":"GBP","JPY":"JPY"}

def ivser(m):
    s=IV.get((IVMAP.get(m,""),"3M","10Y","NORM"))
    return None if s is None else s.resample("ME").last()
def rvfwd(m):
    dy=(mid[f"{FAM[m]}10"]/100.0)
    rv=(dy.diff().rolling(63).std()*np.sqrt(252)*1e4).resample("ME").last()
    return rv.shift(-H_M)
def Rseries(m):
    iv=ivser(m); rv=rvfwd(m)
    if iv is None: return None
    al=pd.concat([sbe[m].rename("be"),rv.rename("rv")],axis=1).dropna()
    return (al["be"]-al["rv"]) if len(al)>=60 else None

R={m:Rseries(m) for m in MK4}; R={m:v for m,v in R.items() if v is not None}

# ---- controlli quantitativi
P("")
P("[controlli] e' diffuso o sono episodi/outlier?")
P(f"{'mercato':9}{'media':>8}{'trim5%':>9}{'mediana':>9}{'% R>0':>8}{'|top5%|share':>14}{'T':>5}")
for m,r in R.items():
    r=r.dropna(); x=np.sort(r.values); n=len(x)
    trim=x[int(.05*n):int(.95*n)].mean()
    top=np.abs(x[int(.95*n):]).sum(); tot=np.abs(x).sum()
    P(f"{LAB[m]:9}{r.mean():8.1f}{trim:9.1f}{np.median(r):9.1f}{(r>0).mean():8.0%}"
      f"{top/tot:14.0%}{n:5d}")
P("  lettura: se 'media' e 'trim5%' sono vicine -> NON e' code; se 'mediana' ha lo stesso segno")
P("  della media -> corpo della distribuzione spostato, quindi diffuso; se |top5%|share e' alto")
P("  -> il livello e' dominato da pochi mesi estremi (candidati a episodi o dati sporchi).")

# ---- figura 2x2
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
for ax,m in zip(axes.flat,MK4):
    if m not in R: ax.set_visible(False); continue
    r=R[m].dropna()
    ax.axhline(0,color="k",lw=.8)
    ax.axhspan(r.mean()-r.std(),r.mean()+r.std(),color="tab:blue",alpha=.08)
    ax.plot(r.index,r.values,color="tab:blue",lw=.9)
    ax.axhline(r.mean(),color="tab:red",lw=1.1,ls="--")
    for a,b in CRISES: ax.axvspan(pd.Timestamp(a),pd.Timestamp(b),color="grey",alpha=.15)
    ax.set_title(f"{LAB[m]}   mean R={r.mean():.0f}bp  median={np.median(r):.0f}",fontsize=9)
    ax.set_ylabel("bp/yr")
fig.suptitle(r"Purged wedge $R=\sigma_{BE}-\mathbb{E}[RV]$ (dashed: mean; band: $\pm 1$ sd; shaded: crises)",
             fontsize=10)
fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(FIG/"figc5_wedge_purged.png"); plt.close(fig)
P(""); P(f"[figura] {FIG/'figc5_wedge_purged.png'}")

# ---- istogrammi
fig,axes=plt.subplots(1,len(R),figsize=(3.2*len(R),3),sharey=True)
if len(R)==1: axes=[axes]
for ax,(m,r) in zip(axes,R.items()):
    r=r.dropna()
    ax.hist(r.values,bins=30,color="tab:blue",alpha=.7)
    ax.axvline(0,color="k",lw=.8); ax.axvline(r.mean(),color="tab:red",ls="--",lw=1.1)
    ax.set_title(LAB[m],fontsize=9); ax.set_xlabel("R (bp/yr)")
fig.suptitle("Distribution of the purged wedge (red dashed: mean)",fontsize=10)
fig.tight_layout(rect=[0,0,1,0.94])
fig.savefig(FIG/"figc6_wedge_hist.png"); plt.close(fig)
P(f"[figura] {FIG/'figc6_wedge_hist.png'}")

save_txt("23_wedge_diag.txt", L); print("\n".join(L))

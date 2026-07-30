"""
08 — Replica FLL Figura 2 (media ponderata per notional) + diagnostico per-coppia
vs Tabella III (validation suite). Figure: fll_fig2_replica, fll_fig2_extended.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from utils import save_txt

bl = pd.read_csv(PROC/"cusip_panel_daily.csv", index_col=0, parse_dates=True)
notional = pd.read_csv(PROC/"cusip_notionals.csv", index_col=0).iloc[:,0].reindex(bl.columns)
W = pd.DataFrame(1.0, index=bl.index, columns=bl.columns).where(bl.notna()) * notional
avg  = (bl*W).sum(axis=1)/W.sum(axis=1)
avg_e= bl.mean(axis=1)

L=[]; P=L.append
P("=== 08 FLL FIGURE-2 REPLICATION ===")
w=avg.loc[FLL_A:FLL_B]
P(f"window mean: {w.mean():.1f} weighted | {avg_e.loc[FLL_A:FLL_B].mean():.1f} equal  (paper {FLL_MEAN_BP})")
P(f"peak: {w.max():.1f} on {w.idxmax().date()}  (paper ~{FLL_PEAK_BP:.0f})")

def fll_axes(ax):
    ax.set_ylabel("Basis-Point Mispricing"); ax.grid(False)
    ax.tick_params(direction="out", top=False, right=False)
fig,ax=plt.subplots(figsize=(8.2,4.6))
ax.plot(w.index,w.values,lw=.9,color="black"); fll_axes(ax); ax.set_ylim(bottom=0)
fig.tight_layout(); fig.savefig(FIG/"fll_fig2_replica.png", dpi=150); plt.close(fig)
fig,ax=plt.subplots(figsize=(11,4.8))
ax.plot(avg.index,avg.values,lw=.8,color="black"); fll_axes(ax)
ax.axhline(0,color="black",lw=.5); ax.axvline(pd.Timestamp(FLL_B),color="crimson",ls="--",lw=1.2)
ax.annotate("end of FLL sample (Nov-2009)", xy=(pd.Timestamp(FLL_B), ax.get_ylim()[1]*.93),
            xytext=(8,0), textcoords="offset points", color="crimson", fontsize=9, va="top")
fig.tight_layout(); fig.savefig(FIG/"fll_fig2_extended.png", dpi=150); plt.close(fig)

if FILE_FLLPAIRS.exists():
    pairs=pd.read_csv(FILE_FLLPAIRS, parse_dates=["tips_mat","tsy_mat"])
    last=bl.apply(lambda s: s.last_valid_index())
    out=[]
    for _,p in pairs.iterrows():
        cand=last[(last-p.tips_mat).abs()<=pd.Timedelta(days=45)]
        if len(cand)==0:
            out.append((p.tips_mat.date(),p.mismatch_d,p.fll_bp_mean,p.fll_N,np.nan,np.nan,"no-match")); continue
        colc = cand.index[0] if len(cand)==1 else \
               (bl[cand.index].loc[FLL_A:FLL_B].notna().sum()-p.fll_N).abs().idxmin()
        ww=bl[colc].loc[FLL_A:FLL_B].dropna()
        out.append((p.tips_mat.date(),p.mismatch_d,p.fll_bp_mean,p.fll_N,round(ww.mean(),1),len(ww),colc))
    D=pd.DataFrame(out,columns=["tips_mat","mm_d","FLL_mean","FLL_N","panel_mean","panel_N","col"])
    D["diff"]=D["panel_mean"]-D["FLL_mean"]; D.to_csv(RES/"08_fll_pair_diagnostic.csv",index=False)
    m=D.dropna()
    P(f"\nper-pair diagnostic: matched {len(m)}/29 | diff medio per mismatch: "
      + str(m.groupby('mm_d')['diff'].mean().round(1).to_dict()))
save_txt("08_fll_replication.txt", L); print("\n".join(L))

"""13 - Figure nuove del master v4.0 (da PROC csv). Output in RES/figures e accanto al tex."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import *
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,"grid.alpha":0.3})
ib = pd.read_csv(PROC/"intl_basis_daily.csv", index_col=0, parse_dates=True)
pb = pd.read_csv(PROC/"pair_basis_daily.csv", parse_dates=["date"])
SH=[("2008-09-01","2009-06-30"),("2011-07-01","2012-08-31"),("2020-02-20","2020-06-30"),("2022-09-01","2022-10-31")]
def shade(ax):
    for a,b in SH: ax.axvspan(pd.Timestamp(a),pd.Timestamp(b),color="grey",alpha=0.18,lw=0)
fig,ax=plt.subplots(figsize=(9,3.6))
for c,l in [("FR10","France 10Y"),("IT10","Italy 10Y"),("UK10","UK 10Y"),("DE5","Germany 5Y")]:
    ax.plot(ib.index, ib[c], lw=0.8, label=l)
shade(ax); ax.axhline(0,color="k",lw=0.6); ax.set_ylabel("bp"); ax.legend(ncol=4,fontsize=8)
ax.set_title("Constant-maturity linker bases, unified pipeline (2004-2026)")
fig.tight_layout(); fig.savefig(FIG/"fig14_intl_bases.png"); plt.close(fig)
fig,ax=plt.subplots(figsize=(9,3.4)); z=ib.loc["2010-07":"2013-06"]
for c,l in [("FR10","France 10Y"),("IT10","Italy 10Y"),("DE5","Germany 5Y")]:
    ax.plot(z.index,z[c],lw=1.1,label=l)
ax.plot(z.index,(z["DE5"]-z["FR5"]),lw=1.1,ls="--",label="Bund-OAT 5Y diff.")
ax.axvline(pd.Timestamp("2011-11-15"),color="k",lw=0.6,ls=":"); ax.axhline(0,color="k",lw=0.6)
ax.set_ylabel("bp"); ax.legend(ncol=4,fontsize=8); ax.set_title("The euro episode, 2010-2013")
fig.tight_layout(); fig.savefig(FIG/"fig15_euro_zoom.png"); plt.close(fig)
med = pb.groupby("date")["lam"].median(); sh = pb[(pb.tau>=1.0)&(pb.tau<2.5)].groupby("date")["lam"].median()
fig,ax=plt.subplots(figsize=(9,3.4))
ax.plot(med.index,med.values,lw=0.8,label="pair-panel median (107 TIPS)")
ax.plot(sh.index,sh.values,lw=0.8,alpha=0.85,label=r"seasoned short pairs, $\tau\in[1,2.5)$")
ax.axhline(0,color="k",lw=0.7); shade(ax); ax.set_ylabel("bp"); ax.set_ylim(-20,120); ax.legend(fontsize=8)
ax.set_title("The floor at pair level (2021 min +2.6 bp)")
fig.tight_layout(); fig.savefig(FIG/"fig16_pair_floor.png"); plt.close(fig)
fig,ax=plt.subplots(figsize=(6.5,3.0)); u=ib["UK10"].loc["2022-05":"2023-03"]
ax.plot(u.index,u.values,lw=1.2); ax.axvline(pd.Timestamp("2022-10-13"),color="k",ls=":",lw=0.8)
ax.set_ylabel("bp"); ax.set_title("UK 10Y basis through the LDI event")
fig.tight_layout(); fig.savefig(FIG/"fig17_uk_ldi.png"); plt.close(fig)
print("[ok] figure v4 in", FIG)

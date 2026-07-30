"""
04 — Eventi di stress (gambe esatte) e dispersione cross-sectional.
Regola 'pre' uniforme (E4): media trailing 60bd dell'IQR, stop 5bd prima dell'onset.
Include E7 (mediana marzo-2020 restated) e U6 (IQR within-bucket di scadenza).
Figure: fig_basis_exact, fig_mar2020_bondlevel.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from utils import save_txt

lam = pd.read_csv(PROC/"lambdas_daily.csv", index_col=0, parse_dates=True); lam.columns=lam.columns.astype(int)
st  = pd.read_csv(PROC/"cusip_stats_daily.csv", index_col=0, parse_dates=True)
bl  = pd.read_csv(PROC/"cusip_panel_daily.csv", index_col=0, parse_dates=True)
med_d, iqr_d = st["median"], st["iqr"]

L=[]; P=L.append
P("=== 04 EVENTS & DISPERSION ===")
P("\nevent table (exact legs): pre -> peak (date), half-reversion")
for ep,(on,a,b) in EPISODES.items():
    for m in [2,5,10,30]:
        s=lam[m].loc[a:b].dropna()
        if s.empty: continue
        pre = lam[m].loc[:on].dropna().iloc[-1]
        pk,pkd = s.max(), s.idxmax()
        half = s.loc[pkd:][s.loc[pkd:] <= pre + (pk-pre)/2]
        hv = half.index[0].date() if len(half) else ">win"
        P(f"  {ep:<9} {m:>2}Y: {pre:6.1f} -> {pk:7.1f} ({pkd.date()})  half-rev {hv}")

P("\n(E4) dispersion, uniform pre rule:")
for ep,(on,a,b) in EPISODES.items():
    on=pd.Timestamp(on)
    pre = iqr_d.loc[:on].iloc[-(PRE_RULE_BD+PRE_RULE_GAP+1):-PRE_RULE_GAP].mean()
    pk, pkd = iqr_d.loc[a:b].max(), iqr_d.loc[a:b].idxmax()
    P(f"  {ep:<9}: pre={pre:5.1f} -> peak {pk:6.1f} ({pkd.date()})")
P(f"\n(E7) Mar-2020 median: {med_d.loc[:'2020-02-14'].iloc[-1]:.1f} (14-Feb) -> "
  f"{med_d.loc['2020-03-16']:.1f} (16-Mar, x2) -> max {med_d.loc['2020-02-20':'2020-06-30'].max():.1f} "
  f"({med_d.loc['2020-02-20':'2020-06-30'].idxmax().date()}); IQR x3.5 con picco 16-Mar")

last_obs = bl.apply(lambda s: s.last_valid_index()); end=bl.index[-1]
matured = last_obs[last_obs < end - pd.Timedelta(days=45)]
def biqr(sl, lo, hi):
    out={}
    for t in bl.loc[sl].index:
        rem=(matured-t).dt.days/365.25
        ids=rem[(rem>lo)&(rem<=hi)].index
        v=bl.loc[t,ids].dropna()
        out[t]=(v.quantile(.75)-v.quantile(.25)) if len(v)>=3 else np.nan
    return pd.Series(out)
for lab,(lo,hi) in [("<5y",(0.5,5)),(">=5y",(5,30))]:
    s=biqr(slice("2020-02-20","2020-06-30"),lo,hi)
    pre=biqr(slice("2019-11-01","2020-02-13"),lo,hi).mean()
    if s.notna().any():
        P(f"  (U6) Mar-2020 IQR within {lab}: pre~{pre:4.1f} -> peak {s.max():5.1f} ({s.idxmax().date()})")

lm = lam.resample("ME").last()
plt.figure(figsize=(11,5.5))
for m in [2,5,10,30]: plt.plot(lm.index, lm[m], lw=1.1, label=f"{m}Y")
plt.plot(lm.index, lm[20], lw=1.1, ls=":", label="20Y (2020+)")
plt.axhline(0,c="k",lw=.6); plt.legend(ncol=5); plt.ylabel("bp")
plt.title("TIPS–Treasury basis, exact legs")
plt.tight_layout(); plt.savefig(FIG/"fig_basis_exact.png", dpi=150); plt.close()

w=slice("2020-01-01","2020-12-31")
q25=bl.loc[w].quantile(.25,axis=1); q75=bl.loc[w].quantile(.75,axis=1)
plt.figure(figsize=(10,5))
plt.plot(med_d.loc[w].index, med_d.loc[w], lw=1.4, label="median")
plt.fill_between(q25.index, q25, q75, alpha=.25, label="IQR")
plt.plot(lam[10].loc[w].index, lam[10].loc[w], lw=1.1, ls="--", label="swap-proxy 10Y")
plt.legend(); plt.ylabel("bp"); plt.title("March 2020 at bond level")
plt.tight_layout(); plt.savefig(FIG/"fig_mar2020_bondlevel.png", dpi=150); plt.close()
save_txt("04_events_dispersion.txt", L); print("\n".join(L))

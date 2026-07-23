"""
05 — Constraint test (design CONGELATO 15-07-2026, Pilot Report v3 §4.3):
y = med_{t+3} − med_{t−1}; shock ΔS; threshold sullo stato ritardato del vincolo.
Include: passaggi market-proxy (per completezza), Spec A–D, sup-F (wild/block-wild),
Chow al quintile pre-specificato (U2), composizione + leave-one-episode-out (U5).
Figure: fig_basis_hkm, fig_constraint_loadings.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from config import *
from utils import save_txt
np.random.seed(SEED)

st  = pd.read_csv(PROC/"cusip_stats_daily.csv", index_col=0, parse_dates=True)
med_m = st["median"].resample("ME").last()
cons = pd.read_csv(PROC/"constraints_monthly.csv", index_col=0, parse_dates=True)
strs = pd.read_csv(PROC/"stress_monthly.csv", index_col=0, parse_dates=True)
HKM = cons["HKM"]; PDz=(cons["PD_bn"]-cons["PD_bn"].rolling(36).mean())/cons["PD_bn"].rolling(36).std()
TFFz=(cons["TFF_netDV01"]-cons["TFF_netDV01"].rolling(24).mean())/cons["TFF_netDV01"].rolling(24).std()
y = (med_m.shift(-LAG_H) - med_m.shift(1)).loc["2004-08":]

L=[]; P=L.append
P("=== 05 CONSTRAINT TEST (frozen design) ===")

def threshold(shock, state, side, grid_q, n_boot=N_BOOT, block=BOOT_BLOCK):
    Z=pd.concat([y,shock,state],axis=1).dropna(); Z.columns=["y","dS","S1"]
    grid=np.quantile(Z["S1"], np.linspace(*grid_q))
    rlin=sm.OLS(Z["y"],sm.add_constant(Z["dS"])).fit(cov_type="HAC",cov_kwds={"maxlags":HAC_LAGS})
    def split(th):
        stress=(Z["S1"]<=th) if side=="below" else (Z["S1"]>th)
        X=sm.add_constant(np.column_stack([Z["dS"]*(~stress),Z["dS"]*stress]))
        return sm.OLS(Z["y"],X).fit(cov_type="HAC",cov_kwds={"maxlags":HAC_LAGS})
    best=None
    for th in grid:
        r=split(th); F=(rlin.ssr-r.ssr)/(r.ssr/(len(Z)-3))
        if best is None or F>best[0]: best=(F,th,r)
    F0,th0,r0=best
    e=rlin.resid.values; xb=rlin.fittedvalues.values; n=len(e); cnt=0
    for _ in range(n_boot):
        s=np.repeat(np.random.choice([-1,1],n//block+1),block)[:n]
        zb=Z.copy(); zb["y"]=xb+e*s
        rl=sm.OLS(zb["y"],sm.add_constant(zb["dS"])).fit(); Fb=0
        for th in grid:
            stress=(zb["S1"]<=th) if side=="below" else (zb["S1"]>th)
            X=sm.add_constant(np.column_stack([zb["dS"]*(~stress),zb["dS"]*stress]))
            rr=sm.OLS(zb["y"],X).fit(); Fb=max(Fb,(rl.ssr-rr.ssr)/(rr.ssr/(len(zb)-3)))
        cnt+=Fb>=F0
    return r0,F0,cnt/n_boot,th0,Z

P("\nmarket-proxy pass (lag-aware) — shocks, not constraints:")
for nm,S in [("dealerCDS",strs["dealerCDS"]),("LOIS",strs["LOIS"]),("MOVE",strs["MOVE"])]:
    r,F,p,th,_=threshold(S.diff(), S.shift(1), "above", (0.50,0.90,9))
    P(f"  {nm:<9} th={th:7.1f}  b_c={r.params.iloc[1]:6.3f}[{r.tvalues.iloc[1]:4.1f}] "
      f"b_s={r.params.iloc[2]:6.3f}[{r.tvalues.iloc[2]:4.1f}]  supF={F:5.1f} p={p:.3f}")

P("\nSpec A/B (vincolo = HKM, stress = below):")
specs = {"A: dHKM": HKM.diff(), "B: dMOVE/10": strs["MOVE"].diff()/10}
res={}
for nm,sh in specs.items():
    r,F,p,th,Z = threshold(sh, HKM.shift(1), "below", GRID_Q)
    res[nm]=(r,Z)
    P(f"  {nm:<11} th={th:5.2f}  b_calm={r.params.iloc[1]:7.3f}[{r.tvalues.iloc[1]:4.1f}]  "
      f"b_stress={r.params.iloc[2]:7.3f}[{r.tvalues.iloc[2]:4.1f}]  supF={F:5.1f}  p={p:.3f}")
P("\nSpec C/C2/D (inventari e futures — attesi non informativi):")
for nm,sh,stt,side in [("C: dPD",cons["PD_bn"].diff(),PDz.shift(1),"above"),
                       ("C2: dMOVE|PDz",strs["MOVE"].diff()/10,PDz.shift(1),"above"),
                       ("D: dTFF",cons["TFF_netDV01"].diff(),TFFz.shift(1),"below")]:
    try:
        r,F,p,th,_=threshold(sh,stt,side,(0.50,0.90,9) if side=="above" else GRID_Q, n_boot=299)
        P(f"  {nm:<13} b_c={r.params.iloc[1]:6.3f}[{r.tvalues.iloc[1]:4.1f}] b_s={r.params.iloc[2]:6.3f}[{r.tvalues.iloc[2]:4.1f}] supF={F:4.1f} p={p:.2f}")
    except Exception as e: P(f"  {nm}: dati mancanti ({e})")

P("\n(U2) Chow al quintile pre-specificato + (U5) composizione e LOO:")
def chow(shock, excl=None, n_boot=N_BOOT):
    Z=pd.concat([y,shock,HKM.shift(1)],axis=1).dropna(); Z.columns=["y","dS","S1"]
    if excl: Z=Z[~Z.index.to_period("M").isin(pd.period_range(*excl,freq="M"))]
    th=np.quantile(Z["S1"],Q_STRESS); stq=(Z["S1"]<=th).astype(float)
    X=sm.add_constant(np.column_stack([Z["dS"]*(1-stq),Z["dS"]*stq]))
    rth=sm.OLS(Z["y"],X).fit(cov_type="HAC",cov_kwds={"maxlags":HAC_LAGS})
    rl=sm.OLS(Z["y"],sm.add_constant(Z["dS"])).fit()
    F=(rl.ssr-rth.ssr)/(rth.ssr/(len(Z)-3))
    e=rl.resid.values; xb=rl.fittedvalues.values; n=len(e); cnt=0
    for _ in range(n_boot):
        s=np.repeat(np.random.choice([-1,1],n//BOOT_BLOCK+1),BOOT_BLOCK)[:n]
        zb=Z.copy(); zb["y"]=xb+e*s
        rlb=sm.OLS(zb["y"],sm.add_constant(zb["dS"])).fit()
        rtb=sm.OLS(zb["y"],sm.add_constant(np.column_stack([zb["dS"]*(1-stq),zb["dS"]*stq]))).fit()
        cnt+=((rlb.ssr-rtb.ssr)/(rtb.ssr/(len(zb)-3)))>=F
    return rth,F,cnt/n_boot,th,Z,stq
for nm,sh in specs.items():
    rth,F,p,th,Z,stq = chow(sh)
    P(f"  Spec {nm}: Chow F={F:5.1f}  block-wild p={p:.4f}  b_stress={rth.params.iloc[2]:6.2f}[{rth.tvalues.iloc[2]:4.1f}]")
rth,F,p,th,Z,stq = chow(specs["B: dMOVE/10"])
comp = Z.index[Z["S1"]<=th].to_period("Y").value_counts().sort_index()
P("  composizione stress: " + ", ".join(f"{k}:{v}" for k,v in comp.items()))
for lab,excl in [("ex-GFC",("2007-07","2009-12")),("ex-COVID",("2020-01","2020-12"))]:
    r2,F2,p2,_,_,_ = chow(specs["B: dMOVE/10"], excl=excl, n_boot=499)
    P(f"  LOO {lab:9s}: b_stress={r2.params.iloc[2]:6.2f}[{r2.tvalues.iloc[2]:4.1f}]  F={F2:5.1f} p={p2:.3f}")

fig,ax1=plt.subplots(figsize=(11,5))
ax1.plot(med_m.index, med_m, lw=1.1, label="basis median (bp)")
ax2=ax1.twinx(); ax2.plot(HKM.loc["2004":].index, HKM.loc["2004":], c="crimson", lw=1, label="HKM (%)")
ax2.axhline(np.quantile(pd.concat([y,HKM.shift(1)],axis=1).dropna().iloc[:,1],Q_STRESS), c="crimson", ls="--", lw=.9)
ax1.set_title("Basis vs HKM capital ratio (dashed: pre-specified quintile)")
ax1.legend(loc="upper left"); ax2.legend(loc="upper right")
plt.tight_layout(); plt.savefig(FIG/"fig_basis_hkm.png", dpi=150); plt.close()
save_txt("05_constraint_test.txt", L); print("\n".join(L))

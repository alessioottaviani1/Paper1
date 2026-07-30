"""
07 — Engine strutturale v2 (post defect-review):
monitoraggio DAILY della barriera; due orizzonti (H=24m con legge MTM per i
sopravvissuti; T=scadenza, mixture esatto della nota); ramo stoppato scontato;
condizione di entrata CE esatta (floor vs RRA); charge di capitale DERIVATO (m×VaR).
Trucco: con theta=lambda0 le deviazioni z sono lambda0-free -> una sola simulazione
per (buffer, orizzonte), poi X(lambda0) in forma chiusa e bisezione sui floor.
"""
import numpy as np, pandas as pd
from config import *
from utils import ou_fit, save_txt
np.random.seed(SEED)

med_m = pd.read_csv(PROC/"cusip_stats_daily.csv", index_col=0, parse_dates=True)["median"].resample("ME").last()
kappa, theta, sig, hl = ou_fit(med_m.loc[POST13:])
L=[]; P=L.append
P("=== 07 STRUCTURAL ENGINE v2 ===")
P(f"OU post-2013: kappa={kappa:.4f}/m  theta={theta:.1f}bp  sigma={sig:.2f}bp/m  HL={hl:.1f}m")

AF = lambda t: (1-np.exp(-Y0*t))/Y0            # carry come annuity (baseline §6.8)
LP = lambda t: t*np.exp(-Y0*t)                 # carry lump alla nota (robustness AF-vs-lump)
CARRY = AF
def core(B, Hm, N=N_PATHS, seed=1):
    rng=np.random.default_rng(seed)
    dt=1.0/DPM; kd=kappa*dt; sd=sig*np.sqrt(dt)
    z=np.zeros(N); alive=np.ones(N,bool); stopt=np.full(N,np.nan)
    for s in range(Hm*DPM):
        z = z - kd*z + sd*rng.standard_normal(N)
        hit = alive & (D_DUR*z > B)
        stopt[hit]=(s+1)/(DPM*12.0); alive&=~hit
    return alive, stopt, z
CORE={(Hm,B): core(B,Hm,seed=Hm+B) for Hm in (H_RISK_M,T_MAT_M) for B in BUFFERS_BP}

def X_of(l0,Hm,B):
    alive,tau,zH = CORE[(Hm,B)]
    X=np.empty(len(zH))
    if Hm>=T_MAT_M: X[alive]=l0*CARRY(10.0)
    else:
        Hy=Hm/12.0; X[alive]=l0*CARRY(Hy)+np.exp(-Y0*Hy)*(-D_DUR*zH[alive])
    X[~alive]=l0*CARRY(tau[~alive])-B*np.exp(-Y0*tau[~alive])
    return X
def solve(cond,Hm,B,lo=1.,hi=120.):
    if cond(X_of(hi,Hm,B))<0: return np.nan
    for _ in range(24):
        mid=.5*(lo+hi)
        lo,hi = (lo,mid) if cond(X_of(mid,Hm,B))>=0 else (mid,hi)
    return hi
def ce_floor(B,Hm,R):
    if R==0: return solve(lambda X: X.mean(),Hm,B)
    a=R/B; return solve(lambda X: -np.log(np.mean(np.exp(-a*X)))/a, Hm,B)
def hurdle_floor(B,Hm,h=HURDLE):
    Hy=10.0 if Hm>=T_MAT_M else Hm/12.0
    return solve(lambda X: X.mean()-h*B*Hy, Hm,B)

P("\n(a) stop-out 24m, DAILY (vs replay empirico 43.0% / 2.5%):")
for B in BUFFERS_BP:
    alive,_,_=CORE[(H_RISK_M,B)]
    P(f"  buffer {B}bp: model {1-alive.mean():5.1%}")
P("\n(b) floors (bp/yr): CE>=W0 per RRA, con RN e hurdle come riferimento")
P("    horizon  buffer   RN     RRA=1  RRA=2  RRA=5   hurdle10%")
fmt=lambda v: f"{v:6.1f}" if np.isfinite(v) else "  >120"
for Hm,lab in [(H_RISK_M,"H=24m"),(T_MAT_M,"T=10y")]:
    for B in BUFFERS_BP:
        vals=[ce_floor(B,Hm,r) for r in RRA_GRID]+[hurdle_floor(B,Hm)]
        P(f"    {lab:7s} {B:4d}bp {fmt(vals[0])} {fmt(vals[1])} {fmt(vals[2])} {fmt(vals[3])}  {fmt(vals[4])}")
P("    observed: 17.7 (median) - 23.3 (10Y)  ->  RRA tra 1 e 2 al buffer 2% derivato")

P("\n(b-bis) robustness carry LUMP (nota-esatta), T=10y, buffer 200bp:")
globals()["CARRY"] = LP
vals = [ce_floor(200, T_MAT_M, r) for r in (0, 1, 2)]
P(f"    RN {fmt(vals[0])}  RRA=1 {fmt(vals[1])}  RRA=2 {fmt(vals[2])}"
  f"   (AF(10)={AF(10):.3f} vs lump {LP(10):.3f}, ratio {AF(10)/LP(10):.3f})")
globals()["CARRY"] = AF

P("\n(c) capital charge DERIVATO (I3):")
sdm=D_DUR*sig
for z,lab in [(2.326,"Basel 10d 99%, m=3"),(1.960,"FRTB 97.5%, m=3")]:
    P(f"  {lab}: {3*47.0*np.sqrt(10/21)*z/100:.2f}% of notional")
P(f"  (D*sigma = {sdm:.1f} bp/m; 47 con gamba yield; curtosi 23.4 -> charge più alto)")
save_txt("07_structural_engine.txt", L); print("\n".join(L))

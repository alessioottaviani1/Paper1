"""Frontiera parsimonia: criterio IC → k e R2adj per strategia. Decide se un IC principled dà <=10. Throwaway."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.linear_model import lars_path
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("cfg",ROOT/"src"/"machine_learning"/"00_config.py")
cfg=importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
FP=cfg.FACTORS_PATH; END=pd.Timestamp(cfg.FACTORS_END_DATE); START=pd.Timestamp(cfg.AEN_START_DATE)
HAC=cfg.HAC_LAGS; G=cfg.AEN_GAMMA
def load_ret(p):
    d=pd.read_csv(p,index_col=0,parse_dates=True)['index_return'].dropna()
    return d.resample('ME').apply(lambda x:((1+x/100).prod()-1)*100 if len(x)>0 else np.nan).dropna()
def l2(X): return X/np.sqrt((X**2).sum(0))
def pen(c,T,k):
    return {"BIC":np.log(T)*k,"HQC":2*np.log(np.log(T))*k,"GIC":3.0*k,
            "AICc":2*k+(2*k*(k+1))/max(T-k-1,1),"AIC":2.0*k}[c]
def sel(Xw,y,c):
    _,_,C=lars_path(Xw,y,method='lasso'); T=len(y); best=None
    for j in range(C.shape[1]):
        idx=np.where(C[:,j]!=0)[0]; k=len(idx)
        rss=float(y@y) if k==0 else float(((y-Xw[:,idx]@np.linalg.lstsq(Xw[:,idx],y,rcond=None)[0])**2).sum())
        ic=T*np.log(rss/T)+pen(c,T,k)
        if best is None or ic<best[0]: best=(ic,idx)
    return best[1]
def r2adj(y,Xs):
    if Xs.shape[1]==0: return 0.,0.,0.,0
    m=sm.OLS(y,sm.add_constant(Xs)).fit(cov_type='HAC',cov_kwds={'maxlags':HAC})
    return m.rsquared_adj,m.params[0]*12,m.tvalues[0],Xs.shape[1]
BMK={"btp_italia":0.130,"cds_bond_basis":0.173,"itraxx_combined":0.102}  # Duarte same-sample
PCA={"btp_italia":0.088,"cds_bond_basis":0.102,"itraxx_combined":0.134}
for name,path in cfg.STRATEGIES.items():
    print("="*68); print(f"{name}  (Duarte={BMK[name]:.3f}  PCA={PCA[name]:.3f})")
    y=load_ret(path); y=y[(y.index>=START)&(y.index<=END)]
    F=pd.read_parquet(FP); F=F[(F.index>=START)&(F.index<=END)]
    idx=y.index.intersection(F.index); yv=y.loc[idx].values; F=F.loc[idx]; names=list(F.columns)
    yc=yv-yv.mean(); Xz=l2(F.values-F.values.mean(0))
    bols=np.linalg.lstsq(Xz,yc,rcond=None)[0]; w=(np.abs(bols)+1/len(yv))**(-G); Xw=Xz/w
    print(f"  {'crit':6s}{'k':>4}{'R2adj':>9}{'a/yr':>7}{'t':>6}  vince?")
    for c in ("BIC","GIC","HQC","AICc","AIC"):
        s=sel(Xw,yc,c); S=[names[i] for i in s]
        r2,a,t,k=r2adj(yv, F[S].values if S else np.empty((len(yv),0)))
        v="sì" if r2>BMK[name] and r2>PCA[name] else ("solo PCA" if r2>PCA[name] else "NO")
        print(f"  {c:6s}{k:>4}{r2:>9.3f}{a:>7.2f}{t:>6.2f}  {v}")
    print()
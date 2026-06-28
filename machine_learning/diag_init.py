"""Prova: init pesi adaptive — OLS vs RIDGE (GCV) — effetto su selezione. Throwaway."""
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
def ridge_gcv(X,y,lams=np.logspace(-4,2,40)):
    n,p=X.shape; XtX=X.T@X; Xty=X.T@y; best=None
    for lam in lams:
        Hd=np.trace(X@np.linalg.solve(XtX+lam*np.eye(p),X.T))
        b=np.linalg.solve(XtX+lam*np.eye(p),Xty); rss=float(((y-X@b)**2).sum())
        gcv=rss/(n*(1-Hd/n)**2)
        if best is None or gcv<best[0]: best=(gcv,lam,b)
    return best[2],best[1]
def hqc_pick(Xw,y):
    _,_,coefs=lars_path(Xw,y,method='lasso'); T=len(y); best=None
    for j in range(coefs.shape[1]):
        b=coefs[:,j]; idx=np.where(b!=0)[0]; k=len(idx)
        if k==0: rss=float(y@y)
        else:
            A=Xw[:,idx]; bb=np.linalg.lstsq(A,y,rcond=None)[0]; rss=float(((y-A@bb)**2).sum())
        hq=T*np.log(rss/T)+2*np.log(np.log(T))*k
        if best is None or hq<best[0]: best=(hq,b.copy())
    return best[1]
def r2adj(y,Xs):
    if Xs.shape[1]==0: return 0.,0.,0.,0
    m=sm.OLS(y,sm.add_constant(Xs)).fit(cov_type='HAC',cov_kwds={'maxlags':HAC})
    return m.rsquared_adj,m.params[0]*12,m.tvalues[0],Xs.shape[1]

for name,path in cfg.STRATEGIES.items():
    print("="*78); print(name)
    y=load_ret(path); y=y[(y.index>=START)&(y.index<=END)]
    F=pd.read_parquet(FP); F=F[(F.index>=START)&(F.index<=END)]
    idx=y.index.intersection(F.index); yv=y.loc[idx].values; F=F.loc[idx]; names=list(F.columns)
    yc=yv-yv.mean(); Xz=l2(F.values-F.values.mean(0))
    C=np.corrcoef(Xz.T); np.fill_diagonal(C,0)
    print(f"  N={len(yv)}  cond(X)={np.linalg.cond(Xz):,.0f}  max|corr|={np.abs(C).max():.3f}")
    bols=np.linalg.lstsq(Xz,yc,rcond=None)[0]
    bri,lam=ridge_gcv(Xz,yc)
    for lab,bi,ex in (("OLS-init",bols,""),("RIDGE-init",bri,f"  (λ_GCV={lam:.4g})")):
        w=(np.abs(bi)+1.0/len(yv))**(-G); bt=hqc_pick(Xz/w,yc)
        S=[names[i] for i in np.where(bt!=0)[0]]
        r2,a,t,k=r2adj(yv, F[S].values if S else np.empty((len(yv),0)))
        print(f"  {lab:11s} k={k:3d}  R2adj={r2:.3f}  a%/yr={a:5.2f}  t={t:4.2f}{ex}")
        print(f"      {S}")
    print()
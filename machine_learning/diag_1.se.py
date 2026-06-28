"""1-SE rule (CV, ESL 7.10) sull'AEN: taglia parsimoniosa con anchor. Throwaway."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.linear_model import Lasso
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
def wts(Xtr,ytr):
    b=np.linalg.lstsq(Xtr,ytr,rcond=None)[0]; return (np.abs(b)+1.0/len(ytr))**(-G)
def r2adj(y,Xs):
    if Xs.shape[1]==0: return 0.,0.,0.,0
    m=sm.OLS(y,sm.add_constant(Xs)).fit(cov_type='HAC',cov_kwds={'maxlags':HAC})
    return m.rsquared_adj,m.params[0]*12,m.tvalues[0],Xs.shape[1]
NF=5
BMK={"btp_italia":0.130,"cds_bond_basis":0.173,"itraxx_combined":0.102}
PCA={"btp_italia":0.088,"cds_bond_basis":0.102,"itraxx_combined":0.134}
for name,path in cfg.STRATEGIES.items():
    print("="*64); print(f"{name}  (Duarte={BMK[name]:.3f}  PCA={PCA[name]:.3f})")
    y=load_ret(path); y=y[(y.index>=START)&(y.index<=END)]
    F=pd.read_parquet(FP); F=F[(F.index>=START)&(F.index<=END)]
    idx=y.index.intersection(F.index); yv=y.loc[idx].values; F=F.loc[idx]; names=list(F.columns)
    yc=yv-yv.mean(); Xz=l2(F.values-F.values.mean(0)); T=len(yc)
    wf=wts(Xz,yc); Xwf=Xz/wf
    amax=np.max(np.abs(Xwf.T@yc))/T; alphas=amax*np.logspace(-3,0,40)
    folds=np.array_split(np.arange(T),NF); err=np.full((NF,len(alphas)),np.nan)
    for fi,te in enumerate(folds):
        tr=np.setdiff1d(np.arange(T),te); w=wts(Xz[tr],yc[tr]); Xw=Xz/w
        for ai,al in enumerate(alphas):
            m=Lasso(alpha=al,fit_intercept=False,max_iter=8000).fit(Xw[tr],yc[tr])
            err[fi,ai]=np.mean((yc[te]-m.predict(Xw[te]))**2)
    cv=err.mean(0); se=err.std(0,ddof=1)/np.sqrt(NF)
    amin=int(np.argmin(cv)); thr=cv[amin]+se[amin]
    a1se=int(np.where(cv<=thr)[0].max())
    print(f"  {'regola':8s}{'k':>4}{'R2adj':>9}{'a/yr':>7}{'t':>6}  vince?")
    for lab,ai in (("min-CV",amin),("1-SE",a1se)):
        b=Lasso(alpha=alphas[ai],fit_intercept=False,max_iter=8000).fit(Xwf,yc).coef_
        S=[names[i] for i in np.where(b!=0)[0]]
        r2,a,t,k=r2adj(yv, F[S].values if S else np.empty((T,0)))
        v="sì" if r2>BMK[name] and r2>PCA[name] else ("solo PCA" if r2>PCA[name] else "NO")
        print(f"  {lab:8s}{k:>4}{r2:>9.3f}{a:>7.2f}{t:>6.2f}  {v}")
    bb=Lasso(alpha=alphas[a1se],fit_intercept=False,max_iter=8000).fit(Xwf,yc).coef_
    print(f"  1-SE factors: {[names[i] for i in np.where(bb!=0)[0]]}\n")
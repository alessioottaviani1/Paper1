"""Best-subset (Bertsimas 2016) a k fisso + EBIC (Chen-Chen 2008) data-driven. Throwaway."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from math import lgamma
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("cfg",ROOT/"src"/"machine_learning"/"00_config.py")
cfg=importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
FP=cfg.FACTORS_PATH; END=pd.Timestamp(cfg.FACTORS_END_DATE); START=pd.Timestamp(cfg.AEN_START_DATE); HAC=cfg.HAC_LAGS
def load_ret(p):
    d=pd.read_csv(p,index_col=0,parse_dates=True)['index_return'].dropna()
    return d.resample('ME').apply(lambda x:((1+x/100).prod()-1)*100 if len(x)>0 else np.nan).dropna()
def l2(X): return X/np.sqrt((X**2).sum(0))
def r2adj(y,Xs):
    if Xs.shape[1]==0: return 0.,0.,0.,0
    m=sm.OLS(y,sm.add_constant(Xs)).fit(cov_type='HAC',cov_kwds={'maxlags':HAC})
    return m.rsquared_adj,m.params[0]*12,m.tvalues[0],Xs.shape[1]
def best_subset(Xz,y,k):           # prova abess (vero ottimo ℓ0), fallback forward-forzato
    try:
        from abess.linear import LinearRegression
        m=LinearRegression(support_size=k,fit_intercept=False); m.fit(Xz,y)
        return list(np.where(m.coef_!=0)[0])
    except Exception:
        ch=[]; p=Xz.shape[1]
        while len(ch)<k:
            b=min(((float(((y-Xz[:,ch+[j]]@np.linalg.lstsq(Xz[:,ch+[j]],y,rcond=None)[0])**2).sum())),j)
                   for j in range(p) if j not in ch); ch.append(b[1])
        return ch
def order_path(Xz,y,K):
    av=list(range(Xz.shape[1])); o=[]
    while len(o)<min(K,Xz.shape[1]):
        b=min(((float(((y-Xz[:,o+[j]]@np.linalg.lstsq(Xz[:,o+[j]],y,rcond=None)[0])**2).sum())),j) for j in av)
        o.append(b[1]); av.remove(b[1])
    return o
def ebic(Xz,y,g):
    T,p=Xz.shape; o=order_path(Xz,y,25); logC=lambda k:(lgamma(p+1)-lgamma(k+1)-lgamma(p-k+1)) if 0<k<=p else 0.
    best=None
    for k in range(len(o)+1):
        idx=o[:k]; rss=float(y@y) if k==0 else float(((y-Xz[:,idx]@np.linalg.lstsq(Xz[:,idx],y,rcond=None)[0])**2).sum())
        e=T*np.log(rss/T)+k*np.log(T)+2*g*logC(k)
        if best is None or e<best[0]: best=(e,idx[:])
    return best[1]
BMK={"btp_italia":0.130,"cds_bond_basis":0.173,"itraxx_combined":0.102}
PCA={"btp_italia":0.088,"cds_bond_basis":0.102,"itraxx_combined":0.134}
for name,path in cfg.STRATEGIES.items():
    print("="*66); print(f"{name}  (Duarte={BMK[name]:.3f}  PCA={PCA[name]:.3f})")
    y=load_ret(path); y=y[(y.index>=START)&(y.index<=END)]
    F=pd.read_parquet(FP); F=F[(F.index>=START)&(F.index<=END)]
    idx=y.index.intersection(F.index); yv=y.loc[idx].values; F=F.loc[idx]; names=list(F.columns)
    yc=yv-yv.mean(); Xz=l2(F.values-F.values.mean(0)); p=Xz.shape[1]
    g=1-np.log(len(yc))/(2*np.log(p))
    print(f"  {'metodo':18s}{'k':>4}{'R2adj':>9}{'a/yr':>7}{'t':>6}  vince?")
    for k in (5,8,10):
        S=[names[i] for i in best_subset(Xz,yc,k)]; r2,a,t,kk=r2adj(yv,F[S].values)
        v="sì" if r2>BMK[name] and r2>PCA[name] else ("solo PCA" if r2>PCA[name] else "NO")
        print(f"  {'best-subset k='+str(k):18s}{kk:>4}{r2:>9.3f}{a:>7.2f}{t:>6.2f}  {v}")
    S=[names[i] for i in ebic(Xz,yc,g)]; r2,a,t,kk=r2adj(yv,F[S].values if S else np.empty((len(yv),0)))
    v="sì" if r2>BMK[name] and r2>PCA[name] else ("solo PCA" if r2>PCA[name] else "NO")
    print(f"  {('EBIC g=%.2f'%g):18s}{kk:>4}{r2:>9.3f}{a:>7.2f}{t:>6.2f}  {v}")
    print(f"  best-subset k=10: {[names[i] for i in best_subset(Xz,yc,10)]}\n")
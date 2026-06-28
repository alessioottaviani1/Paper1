"""Scoreboard same-sample: capped-AEN vs PCA vs benchmark, stesso y & stesse date. Throwaway."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cfg", ROOT/"src"/"machine_learning"/"00_config.py")
cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
FP=cfg.FACTORS_PATH; END=pd.Timestamp(cfg.FACTORS_END_DATE); START=pd.Timestamp(cfg.AEN_START_DATE)
HAC=cfg.HAC_LAGS; STRAT=cfg.STRATEGIES
PROC=ROOT/"data"/"processed"; PCS=ROOT/"results"/"pca"/"pc_scores_contemporaneous.parquet"
BENCH={"FungHsieh":"fung_hsieh_","ActiveFI":"active_fi_","Duarte":""}

def load_ret(p):
    d=pd.read_csv(p,index_col=0,parse_dates=True)['index_return'].dropna()
    return d.resample('ME').apply(lambda x:((1+x/100).prod()-1)*100 if len(x)>0 else np.nan).dropna()
def r2adj(y,X):
    df=pd.concat([y.rename('y'),X],axis=1).dropna()
    m=sm.OLS(df['y'].values, sm.add_constant(df.drop(columns='y')).values).fit(cov_type='HAC',cov_kwds={'maxlags':HAC})
    return m.rsquared_adj, m.params[0]*12, m.tvalues[0], X.shape[1], len(df)
def hqc(yc,Xz,idx):
    if not idx: rss=float(yc@yc); return len(yc)*np.log(rss/len(yc))
    A=Xz[:,idx]; b=np.linalg.lstsq(A,yc,rcond=None)[0]; r=yc-A@b
    T,k=len(yc),len(idx); return T*np.log(float(r@r)/T)+2*np.log(np.log(T))*k
def fwd(yc,Xz,kmax):
    ch=[]; p=Xz.shape[1]
    while len(ch)<kmax:
        h0=hqc(yc,Xz,ch); b=min((hqc(yc,Xz,ch+[j]),j) for j in range(p) if j not in ch)
        if b[0]<h0-1e-9: ch.append(b[1])
        else: break
    return ch

for name in STRAT:
    print("="*82); print(name)
    y=load_ret(STRAT[name]); y=y[(y.index>=START)&(y.index<=END)]
    F=pd.read_parquet(FP); F=F[(F.index>=START)&(F.index<=END)]
    try: PC=pd.read_parquet(PCS); pcc=[c for c in PC.columns if c.startswith('PC')][:8]
    except Exception as e: PC=None; print("  ⚠ PCA non caricata:",e)
    bench={}
    for mdl,inf in BENCH.items():
        for reg in ("us","eur"):
            fp=PROC/f"regression_data_{inf}{name}_{reg}_monthly.csv"
            try:
                d=pd.read_csv(fp,index_col=0,parse_dates=True)
                c=corr=np.corrcoef(y.reindex(d.index.intersection(y.index)),
                                   d['Strategy_Return'].reindex(d.index.intersection(y.index)))[0,1]
                bench[f"{mdl}.{reg}"]=(d[[x for x in d.columns if x!='Strategy_Return']], c)
            except Exception: pass
    idx=y.index.intersection(F.index)
    if PC is not None: idx=idx.intersection(PC.index)
    for k,(Xb,_) in bench.items(): idx=idx.intersection(Xb.index)
    idx=idx.sort_values()
    print(f"  common sample N={len(idx)}  [{idx.min().date()} → {idx.max().date()}]")
    yf=y.loc[idx]; Ff=F.loc[idx]; names=list(Ff.columns)
    Xz=((Ff-Ff.mean())/Ff.std()).values; ycen=(yf-yf.mean()).values

    rows=[]
    for cap in (5,8,10):
        S=[names[i] for i in fwd(ycen,Xz,cap)]; rows.append((f"AEN/fwd cap{cap}",)+r2adj(yf,Ff[S]))
    rows.append(("full_OLS(75)",)+r2adj(yf,Ff))
    if PC is not None: rows.append((f"PCA({len(pcc)}PC)",)+r2adj(yf,PC.loc[idx][pcc]))
    best={}
    for key,(Xb,c) in bench.items():
        mdl=key.split('.')[0]; res=r2adj(yf,Xb.loc[idx])
        if mdl not in best or res[0]>best[mdl][1][0]: best[mdl]=(key,res,c)
    for mdl,(key,res,c) in best.items(): rows.append((f"{key}",)+res)

    print(f"  {'model':16s}{'k':>4}{'N':>5}{'R2adj':>9}{'a%/yr':>9}{'t':>7}")
    for nm,r2a,ayr,t,k,N in rows:
        print(f"  {nm:16s}{k:>4}{N:>5}{r2a:>9.3f}{ayr:>9.2f}{t:>7.2f}")
    print(f"  corr(y, bench Strategy_Return): " + ", ".join(f"{k.split('.')[0]}.{k.split('.')[1]}={c:.3f}" for k,(_,c) in bench.items()))
    print(f"  cap5 factors: {[names[i] for i in fwd(ycen,Xz,5)]}")
    print(f"  cap8 factors: {[names[i] for i in fwd(ycen,Xz,8)]}\n")
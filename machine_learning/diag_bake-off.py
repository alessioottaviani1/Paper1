"""Bake-off selezione+inferenza per lo spanning test. Throwaway."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
import importlib.util
from sklearn.linear_model import LassoLarsIC, RidgeCV
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("cfg", PROJECT_ROOT/"src"/"machine_learning"/"00_config.py")
cfg = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
FACTORS_PATH=cfg.FACTORS_PATH; END=pd.Timestamp(cfg.FACTORS_END_DATE)
START=pd.Timestamp(cfg.AEN_START_DATE); HAC=cfg.HAC_LAGS; STRAT=cfg.STRATEGIES

# set noti dai tuoi run, per contrasto
MB_STABLE={"btp_italia":['FI_DEF','R5_EU','UMD_EU','ITRX_MAIN'],
           "cds_bond_basis":['RI_EU','SS5Y','CURV_2S5S10S_EU','GOVT_EU'],
           "itraxx_combined":['CRED_SPR_EU','GLOBAL_AGG']}
AEN02={"btp_italia":['CRED_SPR_US','SMB_EU','UMD_EU','FI_VAL','FI_DEF','ATM_IV_CDX','5Y5Y_INFL_US','R5_EU','GOVT_EU','SS5Y'],
       "cds_bond_basis":['EMERG_DEBT','RI_EU','R5_EU','TERM_US','GOVT_EU','CURV_2S5S10S_EU','SS5Y'],
       "itraxx_combined":['BTP_BUND','BTP_BUND_2Y','ITRX_MAIN','PB_CDS_5Y_EU','PB_CDS_5Y_US','CRED_SPR_EU','DEF_EU','DEF_US','HY_CORP','EMERG_DEBT','GLOBAL_AGG','HKM_IC','SMB_EU','HML_EU','BAB_EU','GMOM','FI_MOM','FI_DEF','FX_CARRY','PTFSFX','PTFSSTK','SVS_RET_1M','ATM_IV_ITRX','IV_BUND','IV_TSY','R10_EU','CURV_2S5S10S_EU','SLOPE_10S30S_EU','SLOPE_2S10S_US','SS5Y','SS10Y']}

def load_returns(p):
    d=pd.read_csv(p,index_col=0,parse_dates=True)['index_return'].dropna()
    m=d.resample('ME').apply(lambda x:((1+x/100).prod()-1)*100 if len(x)>0 else np.nan)
    return m.dropna()
def load_data(name):
    F=pd.read_parquet(FACTORS_PATH); F=F[(F.index>=START)&(F.index<=END)]
    r=load_returns(STRAT[name]); r=r[r.index<=END]
    dates=r.index.intersection(F.index); y=r.loc[dates]; X=F.loc[dates].copy()
    mask=~(X.isna().any(axis=1)|y.isna()); return y[mask], X[mask]
def lasso_sel(Xz,t):
    m=LassoLarsIC(criterion='bic',fit_intercept=False).fit(Xz,t)
    return np.where(np.abs(m.coef_)>1e-10)[0]
def alpha_ols(y,Xf,cols):
    if len(cols)==0:
        m=sm.OLS(y.values,np.ones((len(y),1))).fit(cov_type='HAC',cov_kwds={'maxlags':HAC})
        return m.params[0],m.tvalues[0],m.pvalues[0],0.0,0
    X=sm.add_constant(Xf[cols].values,prepend=True)
    m=sm.OLS(y.values,X).fit(cov_type='HAC',cov_kwds={'maxlags':HAC})
    return m.params[0],m.tvalues[0],m.pvalues[0],m.rsquared_adj,len(cols)
def hqc(yc,Xz,idx):
    if not idx: rss=float(yc@yc); return len(yc)*np.log(rss/len(yc))
    A=Xz[:,idx]; b=np.linalg.lstsq(A,yc,rcond=None)[0]; r=yc-A@b; rss=float(r@r)
    T,k=len(yc),len(idx); return T*np.log(rss/T)+2*np.log(np.log(T))*k
def forward_hqc(yc,Xz,kmax=20):
    ch=[]; p=Xz.shape[1]
    while len(ch)<kmax:
        h0=hqc(yc,Xz,ch); b=min((hqc(yc,Xz,ch+[j]),j) for j in range(p) if j not in ch)
        if b[0]<h0-1e-9: ch.append(b[1])
        else: break
    return ch
def double_selection(yc,Xz,names,a=0.05):
    n,p=Xz.shape; res={}
    for k in range(p):
        others=[j for j in range(p) if j!=k]; Xo=Xz[:,others]
        S=sorted(set(lasso_sel(Xo,yc))|set(lasso_sel(Xo,Xz[:,k])))
        cols=[k]+[others[i] for i in S]
        m=sm.OLS(yc,sm.add_constant(Xz[:,cols])).fit(cov_type='HAC',cov_kwds={'maxlags':HAC})
        res[names[k]]=(m.params[1],m.tvalues[1])
    c=stats.norm.ppf(1-a/2); sig=[nm for nm,(co,t) in res.items() if abs(t)>=c]
    return res,sig
def debiased_lasso(yc,Xz,names,a=0.05):
    n,p=Xz.shape; bh=LassoLarsIC('bic',fit_intercept=False).fit(Xz,yc).coef_; resid=yc-Xz@bh
    Th=np.zeros((p,p))
    for j in range(p):
        o=[c for c in range(p) if c!=j]
        g=LassoLarsIC('bic',fit_intercept=False).fit(Xz[:,o],Xz[:,j]).coef_
        rj=Xz[:,j]-Xz[:,o]@g; tau2=float(rj@Xz[:,j])/n
        row=np.zeros(p); row[j]=1.0; row[o]=-g; Th[j]=row/tau2
    bdb=bh+Th@(Xz.T@resid)/n; t=np.zeros(p)
    for j in range(p):
        s=(Xz@Th[j])*resid; s=s-s.mean(); lrv=float(s@s)/n
        for l in range(1,HAC+1): lrv+=2*(1-l/(HAC+1))*float(s[l:]@s[:-l])/n
        se=np.sqrt(lrv/n); t[j]=bdb[j]/se if se>0 else 0.0
    res={names[j]:(bdb[j],t[j]) for j in range(p)}
    c=stats.norm.ppf(1-a/2); sig=[names[j] for j in range(p) if abs(t[j])>=c]
    pv=2*(1-stats.norm.cdf(np.abs(t))); o=np.argsort(pv); th=0
    for i in range(p):
        if pv[o[i]]<=(i+1)/p*a: th=i+1
    return res,sig,[names[o[i]] for i in range(th)]

for name in STRAT:
    y,X=load_data(name); names=list(X.columns); p=len(names); T=len(y)
    Xz=((X-X.mean())/X.std()).values; yc=(y-y.mean()).values
    print("="*84); print(f"{name}   T={T}  p={p}")
    sets={"forward_HQC":[names[i] for i in forward_hqc(yc,Xz)],
          "AEN_HQC(02)":[c for c in AEN02[name] if c in names],
          "MB_stable(04)":[c for c in MB_STABLE[name] if c in names]}
    dsr,dss=double_selection(yc,Xz,names); sets["double_select"]=dss
    dbr,dbs,dbbh=debiased_lasso(yc,Xz,names); sets["debiased_lasso"]=dbs; sets["debiased_FDR"]=dbbh
    sets["full_OLS(all)"]=names
    print(f"  {'method':16s}{'k':>4}{'a%/mo':>9}{'a%/yr':>9}{'t':>7}{'p':>8}{'R2adj':>8}")
    for mn,S in sets.items():
        al,tt,pp,r2,k=alpha_ols(y,X,S)
        print(f"  {mn:16s}{k:>4}{al:>9.3f}{al*12:>9.2f}{tt:>7.2f}{pp:>8.4f}{r2:>8.3f}")
    rcv=RidgeCV(alphas=np.logspace(-3,3,40)).fit(Xz,yc)
    top=[n for n,_ in sorted(zip(names,np.abs(rcv.coef_)),key=lambda x:-x[1])[:10]]
    print(f"  ridge(KNS): in-sample R²={rcv.score(Xz,yc):.3f}  top10|coef|: {top}")
    print(f"\n  double_select sig: " + ", ".join(f"{nm}({dsr[nm][1]:+.1f})" for nm in sorted(dss,key=lambda n:-abs(dsr[n][1]))))
    print(f"  debiased   sig: " + ", ".join(f"{nm}({dbr[nm][1]:+.1f})" for nm in sorted(dbs,key=lambda n:-abs(dbr[n][1]))))
    print(f"  debiased FDR  : {dbbh}\n")
"""Confronto potere esplicativo: forward-optimal vs 02 full vs 04 stable. Throwaway."""
import pandas as pd, numpy as np, statsmodels.api as sm
from pathlib import Path
import importlib.util, warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path  = PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"
spec = importlib.util.spec_from_file_location("aen_config", config_path)
aen_config = importlib.util.module_from_spec(spec); spec.loader.exec_module(aen_config)
get_strategy_aen_dir = aen_config.get_strategy_aen_dir

SETS = {
  "btp_italia": {
    "full02":  ['CRED_SPR_US','SMB_EU','UMD_EU','FI_VAL','FI_DEF','ATM_IV_CDX','5Y5Y_INFL_US','R5_EU','GOVT_EU','SS5Y'],
    "stable04":['FI_DEF','R5_EU','UMD_EU','ITRX_MAIN']},
  "cds_bond_basis": {
    "full02":  ['EMERG_DEBT','RI_EU','R5_EU','TERM_US','GOVT_EU','CURV_2S5S10S_EU','SS5Y'],
    "stable04":['RI_EU','SS5Y','CURV_2S5S10S_EU','GOVT_EU']},
  "itraxx_combined": {
    "full02":  ['BTP_BUND','BTP_BUND_2Y','ITRX_MAIN','PB_CDS_5Y_EU','PB_CDS_5Y_US','CRED_SPR_EU','DEF_EU','DEF_US','HY_CORP','EMERG_DEBT','GLOBAL_AGG','HKM_IC','SMB_EU','HML_EU','BAB_EU','GMOM','FI_MOM','FI_DEF','FX_CARRY','PTFSFX','PTFSSTK','SVS_RET_1M','ATM_IV_ITRX','IV_BUND','IV_TSY','R10_EU','CURV_2S5S10S_EU','SLOPE_10S30S_EU','SLOPE_2S10S_US','SS5Y','SS10Y'],
    "stable04":['CRED_SPR_EU','GLOBAL_AGG']},
}

def ols(y, X, cols): return sm.OLS(y, sm.add_constant(X[cols])).fit(cov_type="HAC", cov_kwds={"maxlags":6})
def hqc(y, X, cols):
    if not cols:
        rss=float(np.sum(y.values**2)); return len(y)*np.log(rss/len(y))
    A=X[cols].values; b=np.linalg.lstsq(A,y.values,rcond=None)[0]
    rss=float(np.sum((y.values-A@b)**2)); T,k=len(y),len(cols)
    return T*np.log(rss/T)+2*np.log(np.log(T))*k
def forward(y, X, kmax=15):
    ch=[]
    while len(ch)<kmax:
        h0=hqc(y,X,ch); hb,cb=min((hqc(y,X,ch+[c]),c) for c in X.columns if c not in ch)
        if hb<h0-1e-9: ch.append(cb)
        else: break
    return ch
def summ(y, X, cols):
    if not cols: return 0.0, [], []
    m=ols(y,X,cols)
    return m.rsquared_adj, [c for c in cols if abs(m.tvalues[c])>2], [c for c in cols if abs(m.tvalues[c])<=2]

for sn,cfg in SETS.items():
    sdir=get_strategy_aen_dir(sn)
    X=pd.read_parquet(sdir/"X_standardized.parquet"); y=pd.read_parquet(sdir/"y_centered.parquet")["y"]; y=y.loc[X.index]
    full=[c for c in cfg["full02"] if c in X.columns]; stab=[c for c in cfg["stable04"] if c in X.columns]; fwd=forward(y,X)
    print("="*74); print(f"{sn}  (T={len(y)})")
    for name,cols in [("forward-opt",fwd),("02 full",full),("04 stable",stab)]:
        r2,sig,ins=summ(y,X,cols)
        print(f"  {name:11s} k={len(cols):2d}  adjR²={r2:.3f}  #sig={len(sig)}")
        print(f"               {cols}")
        if ins: print(f"               INUTILI (|t|≤2): {ins}")
    _,sigf,_=summ(y,X,fwd)
    print(f"  >>> significativi nel forward-opt MA buttati dal 04-stable: {[c for c in sigf if c not in stab]}")
    print()
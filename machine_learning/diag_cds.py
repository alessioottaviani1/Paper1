"""CDS: i fattori 'significativi' dei benchmark contano oltre il nostro set di 3? Throwaway."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location("cfg",ROOT/"src"/"machine_learning"/"00_config.py")
cfg=importlib.util.module_from_spec(spec); spec.loader.exec_module(cfg)
FP=cfg.FACTORS_PATH; END=pd.Timestamp(cfg.FACTORS_END_DATE); START=pd.Timestamp(cfg.AEN_START_DATE); HAC=cfg.HAC_LAGS
def load_ret(p):
    d=pd.read_csv(p,index_col=0,parse_dates=True)['index_return'].dropna()
    return d.resample('ME').apply(lambda x:((1+x/100).prod()-1)*100 if len(x)>0 else np.nan).dropna()
y=load_ret(cfg.STRATEGIES["cds_bond_basis"]); y=y[(y.index>=START)&(y.index<=END)]
F=pd.read_parquet(FP); F=F[(F.index>=START)&(F.index<=END)]
idx=y.index.intersection(F.index); y=y.loc[idx]; F=F.loc[idx]
base=["EMERG_DEBT","SS5Y","COM_CARRY"]
# analoghi nei 75 dei fattori significativi dei benchmark (correggi i nomi se serve)
print("colonne candidate:",[c for c in F.columns if any(k in c.upper() for k in ("R10","R5_","R2_","RI_","CRED","DEF","BAA","RB_","RS_","TERM","GLOBAL_AGG"))])
extra=["R10_EU","R5_EU","RI_EU","CRED_SPR_EU","CRED_SPR_US","DEF_EU","TERM_US","GLOBAL_AGG","RB_EU"]
extra=[c for c in extra if c in F.columns and c not in base]
def fit(cols): return sm.OLS(y.values, sm.add_constant(F[cols].values)).fit(cov_type='HAC',cov_kwds={'maxlags':HAC})
m0=fit(base); print(f"\nbase(3) R2adj={m0.rsquared_adj:.3f}")
print("\nfattore           |  MARGINALE (univar) |  CONDIZIONALE (dato i 3)  |  ΔR2adj se aggiunto")
for c in extra:
    tm=fit([c]).tvalues[1]
    mc=fit(base+[c]); tc=mc.tvalues[-1]; dR=mc.rsquared_adj-m0.rsquared_adj
    flag="  <-- aggiunge" if abs(tc)>2 and dR>0.005 else ""
    print(f"  {c:16s} | t={tm:+6.2f}            | t={tc:+6.2f}              | {dR:+.3f}{flag}")
# e il modello completo coi candidati
mc=fit(base+extra); print(f"\nbase+tutti i candidati ({len(base+extra)}f): R2adj={mc.rsquared_adj:.3f}")
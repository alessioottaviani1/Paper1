"""
12 — Power analysis del test pooled H7 (design pre-registrato, calibrazione su dati reali).
Domanda: quanta potenza compra il terzo episodio (euro) rispetto al solo campione USA?
Metodo: semi-parametrico. Stati = storie REALI (dMOVE per USA; d dealer-CDS EU per FR/IT),
quindi il conteggio dei cluster e' quello vero. Effetto (b_calm, b_stress) calibrato sulla
regressione USA reale in scala standardizzata; rumore = residui reali per paese,
ricampionati a blocchi (MBB, blocco 4). Statistica: t one-sided sull'interazione
(quintile pre-specificato Q80 dello stato), valore critico = 95mo percentile sotto H0
simulato con lo stesso design (robusto ai pochi cluster). R=500+500 per configurazione.
"""
import numpy as np, pandas as pd, re
from config import *
np.random.seed(SEED)

# ---- ricostruisco il loader multi-blocco (identico a 11) ----
def load_bbg_blocks(path):
    raw = pd.read_excel(path, header=None)
    hdr = [str(t).strip() for t in raw.iloc[3].tolist()]
    ncol = raw.shape[1]; isd=[False]*ncol
    for j in range(ncol):
        if j==0 or hdr[j] in ("nan","","date","Date"):
            dt = pd.to_datetime(raw.iloc[6:,j], errors="coerce")
            if dt.notna().sum() > 500: isd[j]=True
    axes=[j for j in range(ncol) if isd[j]]; frames=[]
    for a,ax in enumerate(axes):
        end = axes[a+1] if a+1<len(axes) else ncol
        cols=[j for j in range(ax+1,end) if hdr[j] not in ("nan","")]
        if not cols: continue
        d = raw.iloc[6:,[ax]+cols].copy(); d.columns=["date"]+[hdr[j] for j in cols]
        d["date"]=pd.to_datetime(d["date"],errors="coerce")
        frames.append(d.dropna(subset=["date"]).set_index("date").apply(pd.to_numeric,errors="coerce").sort_index())
    v=frames[0]
    for f in frames[1:]: v=v.join(f,how="outer")
    return v

print("== 12 power ==")
v = load_bbg_blocks(FILE_TIPS_VARS)
# y USA: mediana del pannello coppie vere (10), mensile
pb = pd.read_csv(PROC/"pair_basis_daily.csv", parse_dates=["date"])
yUS = pb.groupby("date")["lam"].median().resample("ME").last()
ib = pd.read_csv(PROC/"intl_basis_daily.csv", index_col=0, parse_dates=True)
yFR = ib["FR10"].resample("ME").last(); yIT = ib["IT10"].resample("ME").last()

def dstate(col_pred):
    cols=[c for c in v.columns if col_pred(c)]
    s = v[cols].mean(axis=1).resample("ME").last()
    return s.diff()
xUS = (v[[c for c in v.columns if c.startswith("MOVE")][0]].resample("ME").last().diff())/10.0
xEU = dstate(lambda c: ("CDS EUR" in c and any(k in c for k in ["BNP","BARC","UBS"])) or c.startswith("DB CDS"))

def prep(y, x):
    yy = (y.shift(-LAG_H) - y.shift(1)).rename("y")   # design congelato: med_{t+3} - med_{t-1}
    z  = ((x - x.mean())/x.std()).rename("z")
    d  = pd.concat([yy,z],axis=1).dropna()
    d["D"] = (d["z"] >= d["z"].quantile(1-Q_STRESS)).astype(float)
    return d

D_US, D_FR, D_IT = prep(yUS,xUS), prep(yFR,xEU), prep(yIT,xEU)
import statsmodels.api as sm
def fitsplit(d):
    X = sm.add_constant(np.column_stack([d["z"], d["z"]*d["D"]]))
    r = sm.OLS(d["y"].values, X).fit()
    return r.params[1], r.params[1]+r.params[2], r.resid
bc, bs, resUS = fitsplit(D_US)
_,_,resFR = fitsplit(D_FR); _,_,resIT = fitsplit(D_IT)
print(f"calibrazione USA (scala z): b_calm={bc:.2f}  b_stress={bs:.2f}  | sd residui: US {resUS.std():.1f}  FR {resFR.std():.1f}  IT {resIT.std():.1f}")
print(f"mesi stress (Q80): US {int(D_US['D'].sum())} | FR {int(D_FR['D'].sum())} | IT {int(D_IT['D'].sum())}")

def mbb(res, n, rng, block=4):
    res = np.asarray(res); out=np.empty(n); i=0
    while i<n:
        j=rng.integers(0,len(res)-block); k=min(block,n-i)
        out[i:i+k]=res[j:j+k]; i+=k
    return out

def sim_t(dsets, sds, h1, rng):
    ys=[]; Xs=[]
    for k,(d,sd_res) in enumerate(zip(dsets,sds)):
        e = mbb(sd_res, len(d), rng)
        beta_int = (bs-bc) if h1 else 0.0
        y = bc*d["z"].values + beta_int*(d["z"]*d["D"]).values + e
        ys.append(y)
        fe = np.zeros((len(d), len(dsets))); fe[:,k]=1.0
        Xs.append(np.column_stack([fe, d["z"].values, (d["z"]*d["D"]).values]))
    Y=np.concatenate(ys); X=np.vstack(Xs)
    r = sm.OLS(Y,X).fit()
    return r.tvalues[-1]

R=500
configs = {"solo USA":[ (D_US,resUS) ],
           "USA+FR":[ (D_US,resUS),(D_FR,resFR) ],
           "USA+FR+IT":[ (D_US,resUS),(D_FR,resFR),(D_IT,resIT) ]}
print(f"\npotenza del test pre-registrato (interazione one-sided, cv da H0 simulato, R={R}+{R}):")
for name,cfg in configs.items():
    ds=[c[0] for c in cfg]; sds=[c[1] for c in cfg]
    rng=np.random.default_rng(SEED)
    t0=np.array([sim_t(ds,sds,False,rng) for _ in range(R)])
    t1=np.array([sim_t(ds,sds,True ,rng) for _ in range(R)])
    cv=np.quantile(t0,0.95)
    print(f"  {name:10}: potenza {np.mean(t1>cv):5.1%}   (cv95 H0 = {cv:.2f}; size check {np.mean(t0>cv):.2%})")
print("\nnota: rumore = residui reali MBB(4); effetto omogeneo tra paesi (assunzione H7);")
print("      stati = storie osservate -> i cluster di stress sono quelli veri, non ipotizzati.")

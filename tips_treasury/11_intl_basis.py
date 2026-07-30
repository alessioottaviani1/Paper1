"""
11 — Basi internazionali a scadenza costante (pipeline unica).
FR/IT/UK a 10Y, DE a 5Y (+FR 5Y per il differenziale Bund-OAT).
Gamba real: interpolazione per-ISIN dai linker (stesso metodo per tutti i paesi).
Gamba swap: EUSWI / BPSWIT. Gamba nominale: GFRN/GDBR/GBTPGR/GUKG.
Output: PROC/intl_basis_daily.csv ; RES/11_intl_basis.txt
"""
import numpy as np, pandas as pd
from config import *
from utils import load_bbg, col_of, save_txt

def load_bbg_blocks(path):
    """Foglio Bloomberg multi-blocco: ogni blocco ha la SUA colonna-data (header vuoto).
    Rileva le colonne-data (>=90% di date valide), spezza in blocchi, join per data."""
    raw = pd.read_excel(path, header=None)
    hdr = [str(t).strip() for t in raw.iloc[3].tolist()]
    ncol = raw.shape[1]
    is_date=[False]*ncol
    for j in range(ncol):
        if j==0 or hdr[j] in ("nan","","date","Date"):
            dt = pd.to_datetime(raw.iloc[6:,j], errors="coerce")
            if dt.notna().sum() > 500: is_date[j]=True
    axes=[j for j in range(ncol) if is_date[j]]
    frames=[]
    for a,ax in enumerate(axes):
        end = axes[a+1] if a+1<len(axes) else ncol
        cols=[j for j in range(ax+1,end) if hdr[j] not in ("nan","")]
        if not cols: continue
        d = raw.iloc[6:,[ax]+cols].copy()
        d.columns=["date"]+[hdr[j] for j in cols]
        d["date"]=pd.to_datetime(d["date"],errors="coerce")
        d=d.dropna(subset=["date"]).set_index("date").apply(pd.to_numeric,errors="coerce").sort_index()
        frames.append(d)
    v=frames[0]
    for f in frames[1:]: v=v.join(f,how="outer")
    return v

print("== 11 intl ==")
FILE_LINK = {"FR": BBG/"inflation_linked_fra.csv", "DE": BBG/"inflation_linked_ger.csv",
             "IT": BBG/"inflation_linked_ita.csv", "UK": BBG/"inflation_linked_uk.csv"}

def load_long(path):
    d = pd.read_csv(path, sep=";", low_memory=False)
    d.columns=[c.strip() for c in d.columns]
    d = d.rename(columns={d.columns[0]:"date"})
    d["date"]=pd.to_datetime(d["date"], dayfirst=True, errors="coerce")
    d["YL017"]=pd.to_numeric(d["YL017"], errors="coerce")
    return d.pivot_table(index="date", columns="ISIN", values="YL017").sort_index()

il = pd.read_excel(BBG/"inflation linked.xlsx", sheet_name="Bonds")
il["Maturity"]=pd.to_datetime(il["Maturity"]); il["Issue Date"]=pd.to_datetime(il["Issue Date"])
il["ISIN"]=il["ISIN"].astype(str).str.strip()
PFX={"FR":"French","DE":"Deutsche","IT":"Italy","UK":"United Kingdom"}
mats={}
for cc,p in PFX.items():
    a = il[il["Issuer Name"].str.startswith(p)].copy()
    if cc=="UK": a = a[a["Issue Date"]>="2005-09-01"]     # solo new-style (lag 3 mesi)
    mats[cc] = a.set_index("ISIN")["Maturity"].to_dict()

Y = {cc: load_long(f) for cc,f in FILE_LINK.items()}
v = load_bbg_blocks(FILE_TIPS_VARS)
SWP = {("FR",10):"EUSWI10 ", ("FR",5):"EUSWI5 ", ("DE",5):"EUSWI5 ",
       ("IT",10):"EUSWI10 ", ("UK",10):"BPSWIT10 "}
NOM = {("FR",10):"GFRN10 ", ("FR",5):"GFRN5 ", ("DE",5):"GDBR5 ",
       ("IT",10):"GBTPGR10 ", ("UK",10):"GUKG10 "}

def real_at(cc, tau_star):
    W = Y[cc]; mm = mats[cc]
    cols = [c for c in W.columns if c in mm]
    out = pd.Series(index=W.index, dtype=float)
    M = np.array([mm[c].to_datetime64() for c in cols])
    A = W[cols].values
    for i,t in enumerate(W.index.values):
        tau = (M - t).astype("timedelta64[D]").astype(float)/365.25
        ok = np.isfinite(A[i]) & (tau>0.5)
        if not ok.any(): continue
        tt, yy = tau[ok], A[i][ok]
        lo = (tt<=tau_star); hi = (tt>=tau_star)
        if lo.any() and hi.any():
            t1,y1 = tt[lo].max(), yy[lo][np.argmax(tt[lo])]
            t2,y2 = tt[hi].min(), yy[hi][np.argmin(tt[hi])]
            if t2-t1 < 12.0:
                out.iloc[i] = y1 if t2==t1 else y1+(y2-y1)*(tau_star-t1)/(t2-t1)
        else:
            j = np.argmin(np.abs(tt-tau_star))
            if abs(tt[j]-tau_star) <= 2.0: out.iloc[i] = yy[j]
    return out

L=[]; P=L.append
P("=== 11 INTL BASIS (pipeline unica) ===")
bases={}
for cc,tau in [("FR",10),("IT",10),("UK",10),("DE",5),("FR",5)]:
    r = real_at(cc,tau)
    s = col_of(v, SWP[(cc,tau)]); n = col_of(v, NOM[(cc,tau)])
    lam = ((r + s - n)*100).dropna()
    bases[(cc,tau)] = lam
    print(f"  [cov] {cc}{tau}Y: {len(lam)} gg  {lam.index.min().date() if len(lam) else '-'} -> {lam.index.max().date() if len(lam) else '-'}")
    P(f"  {cc}{tau}Y: {len(lam):5d} gg | {lam.index.min().date()} -> {lam.index.max().date()} | media {lam.mean():6.1f} | mediana {lam.median():6.1f}")

df = pd.DataFrame({f"{cc}{t}": b for (cc,t),b in bases.items()})
df.to_csv(PROC/"intl_basis_daily.csv")

P("\n(a) EPISODIO EURO 2011-12 (il terzo cluster):")
for key in [("FR",10),("IT",10),("DE",5),("FR",5)]:
    b = bases[key]
    pre  = b.loc["2011-03-01":"2011-05-31"].mean()
    win  = b.loc["2011-07-01":"2012-08-31"]
    if len(win)==0 or np.isnan(pre): P(f"  {key[0]}{key[1]}Y: dati insufficienti"); continue
    pk = win.resample("ME").mean().max(); pkd = win.resample("ME").mean().idxmax()
    P(f"  {key[0]}{key[1]}Y: pre(mar-mag11) {pre:6.1f} -> picco mensile {pk:6.1f} ({pkd.strftime('%Y-%m')}) | 15-nov-11: {b.asof(pd.Timestamp('2011-11-15')):6.1f}")
dif = (bases[("DE",5)] - bases[("FR",5)]).dropna()
if len(dif.loc["2011":"2012"])>0:
    P(f"  Bund-OAT 5Y: pre {dif.loc['2011-03':'2011-05'].mean():6.1f} -> picco mensile {dif.loc['2011-07':'2012-08'].resample('ME').mean().max():6.1f}")

P("\n(b) UK LDI (ott-2022):")
if len(bases[("UK",10)].loc["2022-06-01":"2022-12-31"])==0:
    P("  [!] UK10Y senza dati nel 2022 - diagnosticare"); b=bases[("UK",10)]

b = bases[("UK",10)]
if len(b.loc["2022-06-01":"2022-12-31"])==0:
    P("  [!] UK 2022 non calcolabile: UK 2022 senza dati nelle gambe")
else:
    P(f"  UK10Y: pre(ago-22) {b.loc['2022-08-01':'2022-08-31'].mean():6.1f} -> picco daily {b.loc['2022-09-01':'2022-10-31'].max():6.1f} ({b.loc['2022-09-01':'2022-10-31'].idxmax().date()}) | fine-22: {b.loc['2022-12-01':'2022-12-31'].mean():6.1f}")

P("\n(c) prima whiff di meccanismo (Delta mensili vs dealer-CDS EU):")
cds_cols=[c for c in v.columns if "CDS EUR" in c and any(k in c for k in ["BNP","BARC","UBS"])]+[c for c in v.columns if c.startswith("DB CDS")]
if cds_cols:
    cds = v[cds_cols].mean(axis=1).resample("ME").last()
    for key in [("FR",10),("IT",10)]:
        m = bases[key].resample("ME").last()
        j = pd.concat([m.diff(),cds.diff()],axis=1).dropna(); j.columns=["dl","dc"]
        full = j["dl"].corr(j["dc"]); euro = j.loc["2010":"2013","dl"].corr(j.loc["2010":"2013","dc"])
        P(f"  {key[0]}10Y: corr(dLambda,dCDS) full {full:+.2f} | 2010-13 {euro:+.2f}")
else:
    P("  [!] colonne dealer-CDS EU non trovate")
save_txt("11_intl_basis.txt", L); print("\n".join(L))

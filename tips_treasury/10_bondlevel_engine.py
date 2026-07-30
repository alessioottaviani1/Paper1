"""
10 — Bond-level engine v1 (yield-space): coppie vere + serie per-ISIN.
Input: pairs_engine_us.csv, inflation_linked_us.csv, treasury_ISINs_expanded.csv,
       USSWIT grid (TIPS_variables), GSW par yields (feds200628).
Output: PROC/pair_basis_daily.csv ; RES/10_bondlevel_engine.txt
(a) replica FLL Tabella III coppia-per-coppia (vs fll_bp_mean, pesi fll_N)
(b) test GSW-vs-gemello (decide i 133 bracket extra)
(c) certificazione corto post-2024 + episodio 2021 dal pannello coppie
"""
import numpy as np, pandas as pd, re
from config import *
from utils import load_bbg, save_txt

print("== 10 bondlevel ==")
FILE_PAIRS   = RAW/"pairs"/"pairs_engine_us.csv"
FILE_LINK_US = BBG/"inflation_linked_us.csv"
FILE_TSY_EXP = BBG/"treasury_ISINs_expanded.csv"
FILE_GSW     = RAW/"Fed Board"/"feds200628.csv"

def load_long(path, isin_col):
    d = pd.read_csv(path, sep=";", low_memory=False)
    d.columns=[c.strip() for c in d.columns]
    d = d.rename(columns={d.columns[0]:"date"})
    d["date"]=pd.to_datetime(d["date"], dayfirst=True, errors="coerce")
    d["YL017"]=pd.to_numeric(d["YL017"], errors="coerce")
    return d.pivot_table(index="date", columns=isin_col, values="YL017").sort_index()

tips_y = load_long(FILE_LINK_US, "ISIN")
tsy_y  = load_long(FILE_TSY_EXP, "TSY_ISIN")
print(f"tips {tips_y.shape} | tsy {tsy_y.shape}")

v = load_bbg(FILE_TIPS_VARS)
sw_cols = sorted([c for c in v.columns if re.match(r"USSWIT\d+ ", c)], key=lambda c:int(re.findall(r"\d+",c)[0]))
tenors = np.array([int(re.findall(r"\d+",c)[0]) for c in sw_cols], float)
SW = v[sw_cols]

h = next(i for i,l in enumerate(open(FILE_GSW).read().splitlines()) if l.startswith("Date,"))
g = pd.read_csv(FILE_GSW, skiprows=h); g["Date"]=pd.to_datetime(g["Date"]); g=g.set_index("Date")
G = g[[f"SVENPY{k:02d}" for k in range(1,31)]].apply(pd.to_numeric, errors="coerce")
gk = np.arange(1,31,dtype=float)

il = pd.read_excel(BBG/"inflation linked.xlsx", sheet_name="Bonds")
il["Maturity"]=pd.to_datetime(il["Maturity"]); il["ISIN"]=il["ISIN"].astype(str).str.strip()
il["Coupon"]=pd.to_numeric(il["Coupon"],errors="coerce")
ust = il[il["Issuer Name"].str.startswith("United States")]
tmat = ust.set_index("ISIN")["Maturity"].to_dict()

def basis_for(tips, tsy, w0, w1):
    if tips not in tips_y.columns or tsy not in tsy_y.columns or tips not in tmat: return None
    idx = tips_y[tips].dropna().index.intersection(tsy_y[tsy].dropna().index).intersection(SW.index)
    idx = idx[idx>=w0]
    if pd.notna(w1): idx = idx[idx<=w1]
    if len(idx)==0: return None
    tau = ((tmat[tips]-idx).days/365.25).values.clip(1,30)
    Sm = SW.reindex(idx).values
    s  = np.array([np.interp(t,tenors[np.isfinite(r)],r[np.isfinite(r)]) if np.isfinite(r).sum()>1 else np.nan for t,r in zip(tau,Sm)])
    Gm = G.reindex(idx).values
    yg = np.array([np.interp(t,gk[np.isfinite(r)],r[np.isfinite(r)]) if np.isfinite(r).sum()>1 else np.nan for t,r in zip(tau,Gm)])
    yt = tips_y[tips].reindex(idx).values; yn = tsy_y[tsy].reindex(idx).values
    return pd.DataFrame({"date":idx,"tips":tips,"tsy":tsy,"tau":tau,
                         "lam":(yt+s-yn)*100.0,"lam_gsw":(yt+s-yg)*100.0})

pairs = pd.read_csv(FILE_PAIRS)
pairs["window_start"]=pd.to_datetime(pairs["window_start"])
pairs["window_end"]=pd.to_datetime(pairs["window_end"], errors="coerce")
segs=[]
for _,r in pairs.iterrows():
    b = basis_for(r["TIPS_ISIN"], r["TSY_ISIN"], r["window_start"], r["window_end"])
    if b is not None: b["role"]=r["role"]; segs.append(b)
panel = pd.concat(segs, ignore_index=True)
prim = panel[panel.role=="primary"].set_index(["date","tips"])["lam"]
brk  = panel[panel.role!="primary"].groupby(["date","tips"])["lam"].mean()
lam_pair = prim.combine_first(brk).rename("lam").reset_index()
gsw = panel.groupby(["date","tips"])["lam_gsw"].mean().reset_index()
tau_m = panel.groupby(["date","tips"])["tau"].first().reset_index()
out = lam_pair.merge(gsw, on=["date","tips"], how="left").merge(tau_m, on=["date","tips"], how="left")
out.to_csv(PROC/"pair_basis_daily.csv", index=False)

L=[]; P=L.append
P("=== 10 BONDLEVEL ENGINE v1 (yield-space) ===")
P(f"pannello coppie: {out['tips'].nunique()} TIPS | {len(out):,} giorni-coppia | {out['date'].min().date()} -> {out['date'].max().date()}")

# (a) replica FLL Tabella III
fp = pd.read_csv(FILE_FLLPAIRS); fp["tips_mat"]=pd.to_datetime(fp["tips_mat"]); fp["tsy_mat"]=pd.to_datetime(fp["tsy_mat"])
tra = pd.read_excel(BBG/"Treasury.xlsx", sheet_name="Bonds")
tra["Maturity"]=pd.to_datetime(tra["Maturity"]); tra["ISIN"]=tra["ISIN"].astype(str).str.strip()
tra["Coupon"]=pd.to_numeric(tra["Coupon"],errors="coerce")
rows=[]
for _,r in fp.iterrows():
    mt = ust[(ust["Maturity"]==r["tips_mat"])&(np.isclose(ust["Coupon"],r["tips_cpn"],atol=1e-6))]
    mn = tra[(tra["Maturity"]==r["tsy_mat"])&(np.isclose(tra["Coupon"],r["tsy_cpn"],atol=1e-6))]
    if len(mt)!=1 or len(mn)!=1: continue
    b = basis_for(mt["ISIN"].iloc[0], mn["ISIN"].iloc[0], pd.Timestamp(FLL_A), pd.Timestamp(FLL_B))
    if b is None or len(b)==0: continue
    rows.append({"mm":int(r["mismatch_d"]),"ours":b["lam"].mean(),"fll":r["fll_bp_mean"],"N":int(r["fll_N"]),"n_ours":len(b)})
rep = pd.DataFrame(rows)
wa  = lambda x,w: float(np.average(x,weights=w))
P(f"\n(a) replica FLL Tab.III: {len(rep)}/29 coppie ricostruite")
P(f"    media pesata-N: NOSTRA {wa(rep['ours'],rep['N']):.1f} bp  vs  FLL {wa(rep['fll'],rep['N']):.1f} bp (target 54.5)")
P(f"    corr coppia-per-coppia: {rep['ours'].corr(rep['fll']):.3f}")
for m in sorted(rep["mm"].unique()):
    sub=rep[rep.mm==m]
    P(f"    mismatch {m:>2}gg: diff media (nostra-FLL) {float((sub['ours']-sub['fll']).mean()):+6.1f} bp  ({len(sub)} coppie)")

# (b) GSW vs gemello
d = out.dropna(subset=["lam","lam_gsw"]).copy()
d["diff"]=d["lam"]-d["lam_gsw"]
P(f"\n(b) GSW-vs-gemello su {len(d):,} giorni-coppia: diff media {d['diff'].mean():+.1f} bp | sd {d['diff'].std():.1f}")
mo = d.set_index("date").groupby("tips").resample("ME")[["lam","lam_gsw"]].last().dropna()
dcor = mo.groupby("tips").apply(lambda x: x["lam"].diff().corr(x["lam_gsw"].diff()))
P(f"    corr dei Delta mensili per TIPS: mediana {dcor.median():.3f} | p10 {dcor.quantile(.10):.3f}")
P("    verdetto 133 bracket: se corr>~0.95 e sd stabile, la gamba GSW basta nei buchi -> NON scaricarli")

# (c) corto post-2024 e 2021
med = out.set_index("date").groupby(level=0)["lam"].median()
sh  = out[out.tau<2.5].set_index("date").groupby(level=0)["lam"].median()
post = sh.loc["2024-10-01":]
P(f"\n(c) corto (tau<2.5y) post-ott-2024: mediana {post.median():.1f} bp su {len(post)} giorni")
y21 = med.loc["2021"]
P(f"    2021 pannello coppie: mediana<0 in {int((y21<0).sum())} gg | min {y21.min():.1f} bp ({y21.idxmin().date()})")
save_txt("10_bondlevel_engine.txt", L); print("\n".join(L))

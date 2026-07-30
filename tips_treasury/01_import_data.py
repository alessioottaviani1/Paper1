"""
01 — Import e costruzione dataset processati (Paper 2, TIPS floor).
Output in data/processed/tips/: lambdas_daily.csv, lambdas_monthly.csv,
cusip_stats_daily.csv, constraints_monthly.csv, stress_monthly.csv.
"""
import numpy as np, pandas as pd, glob, csv
from config import *
from utils import load_bbg, col_of

print("== 01 import ==")
bbg = load_bbg(FILE_TIPS_VARS).join(load_bbg(FILE_CARTEL1), how="outer", rsuffix="_c")
c = lambda t: col_of(bbg, t)

real = {2:"USGGT02Y",5:"USGGT05Y",10:"USGGT10Y",20:"USGGT20Y",30:"USGGT30Y"}
nom  = {2:"USGG2YR",5:"USGG5YR",10:"USGG10YR",20:"USGG20YR",30:"USGG30YR"}
lam = pd.DataFrame({m:(c(f"USSWIT{m} Curncy")+c(real[m])-c(nom[m]))*100 for m in MATS}).loc["2004-07-01":]
lam.to_csv(PROC/"lambdas_daily.csv")
lam.resample("ME").last().to_csv(PROC/"lambdas_monthly.csv")

lamBE = pd.DataFrame({m:(c(f"USSWIT{m} Curncy")-c(f"USGGBE{m:02d}"))*100 for m in MATS}).loc["2004-07-01":]
lamBE.to_csv(PROC/"lambdas_be_daily.csv")

bl = pd.read_excel(FILE_CUSIP, sheet_name=0)
bl = bl.rename(columns={bl.columns[0]:"date"}).set_index("date")
bl.index = pd.to_datetime(bl.index); bl = bl.apply(pd.to_numeric, errors="coerce").sort_index()
bl.to_csv(PROC/"cusip_panel_daily.csv")
stats = pd.DataFrame({"median": bl.median(axis=1),
                      "iqr": bl.quantile(.75,axis=1)-bl.quantile(.25,axis=1),
                      "p90": bl.quantile(.90,axis=1),
                      "n_live": bl.notna().sum(axis=1)})
stats.to_csv(PROC/"cusip_stats_daily.csv")

w = pd.read_excel(FILE_CUSIP, sheet_name=1)
key = [cc for cc in w.columns if w[cc].astype(str).str.startswith("US").all()][0]
w.set_index(key)["Amount Issued"].astype(float).reindex(bl.columns)\
 .to_csv(PROC/"cusip_notionals.csv")

# --- vincoli: HKM, FR2004 (splice), TFF ---
h = pd.read_csv(FILE_HKM)
h["date"] = pd.to_datetime(h["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
HKM = h.set_index("date")["intermediary_capital_ratio"].astype(float)*100
HKM = HKM[~HKM.index.duplicated(keep="last")]

frames=[]
for f in glob.glob(str(DIR_NETPOS/"*.csv")):
    d = pd.read_csv(f); d = d[d["Time Series"].str.contains("TII|TIPS", case=False)]
    d["As Of Date"]=pd.to_datetime(d["As Of Date"])
    d["Value (millions)"]=pd.to_numeric(d["Value (millions)"],errors="coerce")
    frames.append(d)
PD_w = pd.concat(frames).groupby("As Of Date")["Value (millions)"].sum()/1000.0 if frames else pd.Series(dtype=float)

rows = list(csv.reader(open(DIR_OFR/"tff.csv"))) if (DIR_OFR/"tff.csv").exists() else None
TFF = pd.Series(dtype=float)
if rows:
    desc, codes = rows[1], rows[3]
    def find(subs):
        for j,(dd,_) in enumerate(zip(desc,codes)):
            if all(s.lower() in dd.lower() for s in subs): return j
        return None
    jl, js = find(["Leveraged funds","DV01 of long","Treasury futures"]), find(["Leveraged funds","DV01 of short","Treasury futures"])
    t = pd.DataFrame(rows[4:], columns=codes).replace("", np.nan)
    t["date"]=pd.to_datetime(t["date"]); t=t.set_index("date")
    TFF = (pd.to_numeric(t.iloc[:,jl-1],errors="coerce")-pd.to_numeric(t.iloc[:,js-1],errors="coerce"))/1e6

cons = pd.DataFrame({"HKM": HKM,
                     "PD_bn": PD_w.resample("ME").last() if len(PD_w) else np.nan,
                     "TFF_netDV01": TFF.resample("ME").last() if len(TFF) else np.nan})
cons.to_csv(PROC/"constraints_monthly.csv")

cds = pd.concat([c(t) for t in ["GS CDS","MS CDS","JPMCC CDS","BOFA CDS","BNP CDS","DB CDS","BARCLAY CDS","UBS AG CDS"]], axis=1).mean(axis=1)
stress = pd.DataFrame({"MOVE": c("MOVE Index"), "LOIS": (c("US0003M")-c("USSOC"))*100, "dealerCDS": cds})
stress.resample("ME").last().to_csv(PROC/"stress_monthly.csv")
c("CPURNSA").resample("ME").last().to_csv(PROC/"cpi_nsa_monthly.csv")
print("[ok] processed scritti in", PROC)

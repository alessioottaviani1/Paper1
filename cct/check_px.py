import pandas as pd
from pathlib import Path

PROC = Path(r".\data\processed\cct")
px = pd.read_csv(PROC / "px_cct.csv", index_col=0, parse_dates=True)
C  = pd.read_csv(PROC / "static_cct.csv", parse_dates=["issue", "maturity"])

first = px[px.notna().any(axis=1)].index.min()
print("primo giorno con QUALSIASI prezzo CCT:", first.date())

n_pre99 = sum(px.loc[px.index < "1999-01-01", c].notna().any() for c in px.columns)
n_pre96 = sum(px.loc[px.index < "1996-01-01", c].notna().any() for c in px.columns)
print("CCT con almeno un prezzo prima del 1999:", n_pre99)
print("CCT con almeno un prezzo prima del 1996:", n_pre96)
print()

old = C[C.issue < "1995-01-01"].sort_values("issue")
print("CCT emessi PRE-1995 in anagrafica:", len(old))
has_px = [i for i in old["isin"] if i in px.columns and px[i].notna().any()]
print("  di questi, con ALMENO UN prezzo (a qualsiasi data):", len(has_px))
print()

print("primi prezzi dei 8 CCT piu vecchi:")
for _, r in old.head(8).iterrows():
    isin = r["isin"]
    if isin in px.columns and px[isin].notna().any():
        fp = px[isin].dropna().index.min().date()
    else:
        fp = "NESSUNO"
    print(f"  {isin} (emesso {r.issue.date()}, mat {r.maturity.date()}): primo px {fp}")
print()

cb = C[C.regime == "CCT-BOT"].copy()
cb["has_px"] = cb["isin"].apply(lambda i: i in px.columns and px[i].notna().any())
cb["quinq"] = (cb["issue"].dt.year // 5) * 5
print("CCT-BOT con prezzo, per quinquennio di emissione:")
print(cb.groupby("quinq").agg(totali=("isin", "size"), con_prezzo=("has_px", "sum")).to_string())
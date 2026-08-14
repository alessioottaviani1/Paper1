"""Verifica mirata fonte prezzo linker: MILA (vecchio codice) vs CBBT (nuovo), su pochi
ISIN/date. Scarica SOLO i prezzi MILA di alcuni linker su poche date (pochissimi data
point) e li affianca ai CBBT gia' in cache. Dice se lo scarto prezzo spiega il residuo
di ~2 bp della base. Lancia dalla root:  python .\src\inflation_linked\_prova_fonte.py
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from xbbg import blp
import bbg

# pochi linker VIVI, sparsi per scadenza; poche date sparse nel campione
ISINS = ["IT0005387052", "IT0003745541", "IT0005004426", "IT0004545890"]
DATE  = ["2018-06-15", "2020-11-02", "2022-07-29", "2023-03-24"]

ref = bbg.load("ref_linker")
px_cbbt_cache = {}
for f in ("px_ask_IT",):
    try:
        px_cbbt_cache[f] = bbg.load(f)
    except Exception:
        pass
cache = px_cbbt_cache.get("px_ask_IT")

print(f"{'ISIN':16s} {'data':12s} {'MILA ask':>10s} {'CBBT ask':>10s} {'scarto':>10s}")
print("-"*64)
rows = []
for isin in ISINS:
    bb = ref.loc[isin, "bb_id"] if isin in ref.index else None
    if bb is None:
        print(f"{isin}: non in ref"); continue
    # scarico MILA solo per queste date (una bdh stretta per ISIN)
    d0, d1 = min(DATE).replace("-",""), max(DATE).replace("-","")
    mila = bbg._bdh([f"{bb}@MILA Corp"], "PX_ASK", d0, d1)
    for d in DATE:
        ts = pd.Timestamp(d)
        pm = mila.iloc[:,0].reindex([ts]).iloc[0] if len(mila) else float("nan")
        pc = float("nan")
        if cache is not None and isin in cache.columns:
            s = cache[isin]; s.index = pd.to_datetime(s.index)
            pc = s.reindex([ts]).iloc[0]
        if pd.notna(pm) or pd.notna(pc):
            sc = (pm - pc) if (pd.notna(pm) and pd.notna(pc)) else float("nan")
            print(f"{isin:16s} {d:12s} {pm:10.4f} {pc:10.4f} {sc:10.4f}")
            if pd.notna(sc): rows.append(sc)

if rows:
    import statistics
    print(f"\nscarto MILA-CBBT (ask): media {statistics.mean(rows):+.4f}  "
          f"n={len(rows)}  min {min(rows):+.4f}  max {max(rows):+.4f}")
    print("Se lo scarto prezzo e' ~sistematico e dell'ordine di 0.02-0.05, spiega i ~2 bp")
    print("di base (la base e' in yield, ma su questi bond ~1bp di yield ~ pochi cent di prezzo).")

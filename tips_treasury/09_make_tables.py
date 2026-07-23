"""
09 — Genera i frammenti LaTeX (booktabs) per paper/tables_paper2/,
da includere nei capitoli paper2_*.tex con \\input{tables_paper2/...}.
Tabelle: floor means, breaks, dispersion, constraint (A-D), episode-HKM,
floors strutturali. I numeri sono ricalcolati dai processed (fonte unica).
"""
import numpy as np, pandas as pd
from config import *
from utils import nw_mean, one_break

lm = pd.read_csv(PROC/"lambdas_monthly.csv", index_col=0, parse_dates=True); lm.columns=lm.columns.astype(int)

def w(name, s): (TABDIR/name).write_text(s, encoding="utf-8"); print("[tex]", TABDIR/name)

rows=[]
for nm,(a,b) in {"FLL 2004/07--2009/11":(FLL_A[:7],FLL_B[:7]),"2010--2012":("2010-01","2012-12"),
                 "2013--2019":("2013-01","2019-12"),"2020--2026":("2020-01","2026-12"),
                 "Post-2013 (all)":(POST13,"2026-12")}.items():
    cells=[]
    for m in MATS:
        x=lm[m].loc[a:b]
        cells.append(f"{nw_mean(x)[0]:.1f}\\,[{nw_mean(x)[1]:.1f}]" if x.notna().sum()>12 else "---")
    rows.append(nm+" & "+" & ".join(cells)+" \\\\")
w("tab_floor_means.tex",
  "\\begin{tabular}{lccccc}\n\\toprule\nPeriod & 2Y & 5Y & 10Y & 20Y & 30Y \\\\\n\\midrule\n"
  + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

rows=[]
for m in MATS:
    x=lm[m].dropna()
    if len(x)<60: continue
    d1,a1,b1,F1,k=one_break(x); seg=x[:k] if k>=len(x)-k else x[k:]
    d2,a2,b2,F2,_=one_break(seg) if len(seg)>=60 else (None,)*5
    r2 = f"{d2.strftime('%Y-%m')} & ${a2:.1f}\\to{b2:.1f}$ ({F2:.0f})" if d2 is not None else "--- & ---"
    rows.append(f"{m}Y & {d1.strftime('%Y-%m')} & ${a1:.1f}\\to{b1:.1f}$ ({F1:.0f}) & {r2} \\\\")
w("tab_breaks.tex",
  "\\begin{tabular}{lcccc}\n\\toprule\nMaturity & Break 1 & Means (sup-$F$) & Break 2 & Means (sup-$F$) \\\\\n\\midrule\n"
  + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
print("[nota] tabelle constraint/structural: incolla i valori da results/tips/05 e 07 (o estendi qui).")

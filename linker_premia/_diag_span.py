import pandas as pd
from pathlib import Path
import sys; sys.path.insert(0, r".\src\inflation_linked")
from config import CACHE
import rp

for mkt in ("US", "UK"):
    fn = {"US": rp.bei_us, "UK": rp.bei_uk}[mkt]
    mats = (2, 5, 10, 20) if mkt == "US" else (2.5, 5, 10, 20)
    bei = fn(mats); s = rp.isr(mkt, mats)
    idx = bei.index.intersection(s.index)
    print(f"{mkt}: BEI {bei.index.min().date()}->{bei.index.max().date()} | "
          f"ISR {s.index.min().date()}->{s.index.max().date()} | "
          f"lambda (intersez.) {idx.min().date()}->{idx.max().date()} | n={len(idx)}")
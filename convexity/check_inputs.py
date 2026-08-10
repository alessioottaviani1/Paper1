"""check_inputs.py - inventario input PRIMA della pipeline. Esegui da src/convexity/.
Non importa config: cerca da solo (RAW = data/raw oppure raw), riporta cosa trova e dove."""
from pathlib import Path
import glob, re, warnings; warnings.filterwarnings("ignore")
try: import pandas as pd
except ImportError: pd=None
PKG=Path(__file__).resolve().parent; ROOT=PKG.parents[1]
RAW=next((p for p in (ROOT/"data"/"raw", ROOT/"raw") if p.exists()), None)
print(f"ROOT: {ROOT}\nRAW : {RAW if RAW else '*** ne data/raw ne raw esistono ***'}")
if RAW is None: raise SystemExit(1)
BB=RAW/"Bloomberg"; FED=RAW/"Fed Board"
BARC=next((RAW/n for n in ("Barclays Live","Barclays","BarclaysLive") if (RAW/n).exists()), RAW/"Barclays Live")
def find(folder,pats):
    h=[]
    for p in pats: h+=glob.glob(str(folder/"**"/p),recursive=True)
    return sorted({Path(x) for x in h})
def report(nome,folder,pats):
    print(f"\n[{nome}] cerco {pats} in {folder.relative_to(ROOT) if folder.exists() else folder}")
    hits=find(folder,pats)
    if not hits:
        print("   *** NIENTE nella cartella attesa ***")
        broad=find(RAW,pats)
        for h in broad: print(f"   pero' c'e': {h.relative_to(ROOT)}  (cartella diversa)")
        return hits
    for h in hits: print(f"   OK  {h.relative_to(ROOT)}")
    return hits

bbg=report("bbg_paper2 (hub Bloomberg)", BB, ["bbg_paper2.xlsx"])
report("GSW Fed Board", FED, ["feds200628.csv"])
vols=report("Superfici swaption (Barclays)", BARC, ["swaption_vols*.xlsx","data*.xlsx"])
# CDS e stati NON sono file separati: vivono nei fogli cds / rates & vol di bbg_paper2 (letti direttamente)

if pd is not None and bbg:
    xl=pd.ExcelFile(bbg[0]); print(f"\n[fogli di bbg_paper2]: {xl.sheet_names} (attesi: swap, cds, rates & vol)")
    fc=pd.read_excel(bbg[0],sheet_name="cds",header=None); hc=[str(x).strip() for x in fc.iloc[3].tolist()]
    CANON=["MS CDS USD SR 5Y","JPMCC CDS USD SR 5Y","BOFA CDS USD SR 5Y",
           "BNP CDS EUR SR 5Y","BARCLAY CDS EUR SR 5Y","UBS AG CDS EUR SR 5Y"]  # GS escluso (2012+stale)
    print("[CDS canonici nel foglio cds]")
    for n in CANON: print(f"   {'OK ' if any(h.startswith(n) for h in hc) else 'MANCA'}: {n}")
    db_ok=any(h.startswith("DB CDS EUR SR 5Y") or h.startswith("DB CDS EUR SLA 5Y") for h in hc)
    print(f"   {'OK ' if db_ok else 'MANCA'}: DB CDS EUR (SR o SLA) 5Y")
    fr=pd.read_excel(bbg[0],sheet_name="rates & vol",header=None); hr=[str(x).strip() for x in fr.iloc[3].tolist()]
    for n in ["MOVE Index","VIX Index"]: print(f"   {'OK ' if n in hr else 'MANCA'}: {n}")
    # conteggio serie swaption
    seen=set(); cnt={}
    for p in vols:
        sr=pd.read_excel(p,sheet_name=0,header=None,nrows=6)
        for r in range(min(6,len(sr))):
            for x in sr.iloc[r].tolist():
                m=re.match(r"([A-Z]{3})SW\d+[MY]\d+YF ",str(x))
                if m and str(x) not in seen: seen.add(str(x)); cnt[m.group(1)]=cnt.get(m.group(1),0)+1
    print(f"\n[swaption grezzo, upper bound]: {cnt} tot {sum(cnt.values())} (filtrato atteso da 01: ~166)")
print("\n--- Se ogni riga chiave e' OK, la pipeline ha i suoi input. ---")

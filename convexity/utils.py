"""utils - loader e motore sigma_BE a 3 punti del pacchetto convexity."""
import pandas as pd, numpy as np, re, glob, warnings; warnings.filterwarnings("ignore")
from config import *

def load_bbg_sheet(sheet):
    """Parser dei fogli di bbg_paper2: col0=data condivisa, intestazioni riga 3, dati da riga 6."""
    f=pd.read_excel(BBG, sheet_name=sheet, header=None)
    dts=pd.to_datetime(f.iloc[6:,0],errors="coerce")
    hdr=[str(x).strip() for x in f.iloc[3].tolist()]
    out={}
    for j in range(1,f.shape[1]):
        t=hdr[j]
        if t in ("nan",""): continue
        v=pd.to_numeric(f.iloc[6:,j],errors="coerce"); v.index=dts
        v=v.dropna()
        if v.index.duplicated().any():   # robustezza: incolli Bloomberg con date ripetute
            v=v[~v.index.duplicated(keep="last")]
        out[(t,j)]=v.sort_index()
    return out

def load_legs_mid():
    """Mid giornalieri delle gambe del fly: foglio 'swap' + foglio 'govt' (PX_LAST).
    Le curve governative entrano con la stessa costruzione degli swap."""
    raw=load_bbg_sheet(SHEET_SWAP)
    try: raw.update(load_bbg_sheet(SHEET_GOVT))
    except Exception as e: print(f"[utils] foglio govt non letto ({e}); solo swap")
    byname={}
    for (t,j),v in raw.items():
        byname.setdefault(t.replace(" Curncy","").replace(" Index","").strip(), v)
    need=sorted({l for legs,_ in MK.values() for l in legs})
    miss=[l for l in need if l not in byname]
    if miss: raise FileNotFoundError(f"gambe mancanti nel foglio swap: {miss}")
    mid=pd.DataFrame({l: byname[l] for l in need}).sort_index()
    return mid

def load_halfspreads(index):
    """Half-spread per gamba (bp). Se APPLY_COSTS: HS_BASE x m(MOVE_t) (tier Paper 1 A.III).
    Altrimenti: HS_BASE costante (placeholder) - i costi non entrano nei risultati."""
    need=sorted({l for m,(legs,_) in MK.items() if m not in GOVT_MKTS for l in legs})
    if not APPLY_COSTS:
        H=pd.DataFrame({l: float(HS_BASE[l]) for l in need}, index=index)
        return H, "costi OFF (HS_BASE placeholder; risultati GROSS finche' non arrivano le quote dealer)"
    ms=load_market_states(); mv=ms["MOVE"].reindex(index).ffill()
    t1,t2=MOVE_TIERS; m0,m1,m2=COST_MULT
    mult=pd.Series(np.where(mv<t1,m0,np.where(mv<t2,m1,m2)),index=index)
    H=pd.DataFrame({l: float(HS_BASE[l])*COST_SAFETY*mult for l in need})
    return H, f"MOVE-tiered da QUOTE DEALER: x{m0}/x{m1}/x{m2} a MOVE<{t1:.0f}/<{t2:.0f}/>={t2:.0f} (safety x{COST_SAFETY})"


def save_txt(name, lines):
    (RES/name).write_text("\n".join(lines)); print(f"[saved] {RES/name}")

def nw_t(x, L=6):
    x=np.asarray(pd.Series(x).dropna()); n=len(x); m=x.mean(); u=x-m; s=u@u/n
    for l in range(1,L+1): s+=2*(1-l/(L+1))*((u[l:]@u[:-l])/n)
    return m/np.sqrt(s/n)

def clean_vol(s, fam):
    """Regola QC pre-specificata: (i) drop valori <=0; (ii) Hampel(7, 5x1.4826xMAD, e >10% del livello) -> NaN,
    nessuna interpolazione; (iii) famiglie LOG: nessun 'fix' dei blowup da tassi~0 (convenzione, non errore) --
    usate solo via conversione sigma_N ~ sigma_B x F, mai direttamente nei test."""
    s = s[s > 0]
    med = s.rolling(7, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(7, center=True, min_periods=3).median()
    bad = (s - med).abs() > np.maximum(5 * 1.4826 * mad, 0.10 * med.abs())
    return s[~bad]

_FAM={"Normalised Vol ATM":"NORM","RFR Normalised Vol ATM":"NORM","ATM_IVOL_NOM":"NORM",
      "LIBOR Normalised Vol ATM":"LNORM","LIBOR_ATM_IVOL_N":"LNORM",
      "Implied Vol ATM":"LOG","RFR Implied Vol ATM":"LOG","ATM_IVOL":"LOG"}

def load_dealer_cds():
    """Composite regionale {US, EU} dei CDS dealer CANONICI, letta DIRETTAMENTE dal foglio cds
    di bbg_paper2 (config.CDS_CANON_US/EU), dedup per nome. Nessun CSV intermedio."""
    raw=load_bbg_sheet(SHEET_CDS)
    def pick(prefixes):
        cols={}
        for (t,j),v in sorted(raw.items(), key=lambda kv: kv[0][1]):
            if any(t.startswith(p) for p in prefixes) and t not in cols: cols[t]=v
        return cols
    us=pick(CDS_CANON_US); eu=pick(CDS_CANON_EU)
    US=pd.DataFrame(us).mean(axis=1).resample("ME").last() if us else pd.Series(dtype=float)
    EU=pd.DataFrame(eu).mean(axis=1).resample("ME").last() if eu else pd.Series(dtype=float)
    return {"US":US,"EU":EU}, list(us), list(eu)

def load_market_states():
    """MOVE/VIX/funding dal foglio 'rates & vol' + SnrFin/Main dal foglio 'cds' (con FIN_MINUS_MAIN),
    letti DIRETTAMENTE da bbg_paper2. Nessun CSV intermedio."""
    rv=load_bbg_sheet(SHEET_RV)
    df=pd.DataFrame({t.replace(" Index","").replace(" Curncy",""): v for (t,j),v in rv.items()})
    raw=load_bbg_sheet(SHEET_CDS)
    for pref,name in [("SNRFIN CDSI GEN 5Y","SNRFIN"),("ITRX EUR CDSI GEN 5Y","ITRX_MAIN")]:
        s=next((v for (t,j),v in sorted(raw.items(),key=lambda kv: kv[0][1]) if t.startswith(pref)),None)
        if s is not None: df[name]=s
    if "SNRFIN" in df and "ITRX_MAIN" in df: df["FIN_MINUS_MAIN"]=df["SNRFIN"]-df["ITRX_MAIN"]
    return df

def load_govt_extras():
    """Serie NON-curva del foglio 'govt': indici TR, bill, future. Ritorna dict nome->serie."""
    raw=load_bbg_sheet(SHEET_GOVT)
    out={}
    for (t,j),v in raw.items():
        out.setdefault(t.strip(), v)
    return out

def load_rf(ccy, extras=None):
    """Risk-free MENSILE in % per valuta. RF_MODE='ois' (uniforme, default) o 'bill'.
    Il tasso annuo osservato a fine mese precedente e' diviso per 12 (ACT/12 semplificato)."""
    if RF_MODE == "bill":
        tk = RF_BILL.get(ccy)
        if tk is not None:
            ex = extras if extras is not None else load_govt_extras()
            if tk in ex:
                r = ex[tk].resample("ME").last()
                return (r.shift(1)/12.0), f"bill {tk}"
    tk = RF_OIS.get(ccy)
    if tk is None: return None, "n/d"
    mid = load_legs_mid_all()
    if tk not in mid: return None, f"OIS {tk} assente"
    r = mid[tk].resample("ME").last()
    return (r.shift(1)/12.0), f"OIS 1Y {tk}"

def load_legs_mid_all():
    """Tutti i mid del foglio swap (non solo le gambe del fly): serve per i tenor RF e il CP."""
    raw=load_bbg_sheet(SHEET_SWAP)
    out={}
    for (t,j),v in raw.items():
        out.setdefault(t.replace(" Curncy","").replace(" Index","").strip(), v)
    return out

def load_vols(clean=True):
    """tutte le serie swaption dai data*.xlsx (due schemi di nomi) -> dict {(ccy,exp,ten,fam): serie mensile}."""
    IV={}
    for p in VOLS:
        sr=pd.read_excel(p,sheet_name="Series",header=None)
        nr=next((i for i in range(4) if any(("Vol ATM" in str(x)) or ("IVOL" in str(x)) for x in sr.iloc[i].tolist())),None)
        if nr is None: continue
        names=[str(x).strip() for x in sr.iloc[nr].tolist()]; dd=pd.to_datetime(sr.iloc[nr+1:,0],errors="coerce")
        for j,n in enumerate(names):
            m=re.match(r"([A-Z]{3})SW(\d+[MY])(\d+Y)F (.+)",n)
            if not m or m.group(4) not in _FAM: continue
            key=(m.group(1),m.group(2),m.group(3),_FAM[m.group(4)])
            if key in IV: continue
            v=pd.to_numeric(sr.iloc[nr+1:,j],errors="coerce"); v.index=dd
            v=v.dropna().sort_index()
            if clean: v=clean_vol(v, key[3])
            IV[key]=v.resample("ME").last()
    return IV

def load_gsw_nodes():
    h=next(i for i,l in enumerate(open(GSW).read().splitlines()) if l.startswith("Date,"))
    g=pd.read_csv(GSW,skiprows=h); g["Date"]=pd.to_datetime(g["Date"]); g=g.set_index("Date")
    return g[["SVENY02","SVENY10","SVENY30"]].apply(pd.to_numeric,errors="coerce").dropna().resample("ME").last()/100.0

def fly_weights(t1,t2,t3):
    w1=(t3-t2)/(t3-t1); w3=(t2-t1)/(t3-t1)
    return w1,w3,w1*t1**2+w3*t3**2-t2**2

def sigbe_and_returns(Zm, taus):
    """Zm: df mensile z1,z2,z3 (decimali). Ritorna (sigma_BE firmata bp/yr, R pacchetto bp, s2)."""
    t1,t2,t3=taus; w1,w3,C=fly_weights(t1,t2,t3)
    zf=lambda r,tau: r.z1+(r.z2-r.z1)*(tau-t1)/(t2-t1) if tau<=t2 else r.z2+(r.z3-r.z2)*(tau-t2)/(t3-t2)
    th=Zm.apply(lambda r:(w1*((zf(r,t1)*t1)-(zf(r,t1-DT)*(t1-DT)))+w3*((zf(r,t3)*t3)-(zf(r,t3-DT)*(t3-DT)))-((zf(r,t2)*t2)-(zf(r,t2-DT)*(t2-DT)))),axis=1)
    s2=-2*th/(C*DT)
    lp=lambda r,tau:-zf(r,tau)*tau
    R=pd.Series(index=Zm.index,dtype=float)
    for i in range(1,len(Zm)):
        r0,r1=Zm.iloc[i-1],Zm.iloc[i]
        R.iloc[i]=(w1*(lp(r1,t1-DT)-lp(r0,t1))+w3*(lp(r1,t3-DT)-lp(r0,t3))-(lp(r1,t2-DT)-lp(r0,t2)))*1e4
    return np.sign(s2)*np.sqrt(np.abs(s2))*1e4, R, s2


"""25 - IL PREMIO EPURATO E' CONDIZIONALE? Il test che il pannello 1b rende necessario.

COSA HA MOSTRATO IL 24. Sullo stesso campione, previsore ex-post ed ex-ante danno lo stesso
numero (differenza 1-2bp): il previsore NON e' il problema. Tutta la differenza col campione
pieno sta nei mesi scartati, e quei mesi hanno R enorme: +84.6 in USD, +108.9 in GBP, contro
un campione recente a -20.2 e -43.2. Uno scarto di oltre 150bp DENTRO lo stesso mercato.

CONSEGUENZA. In USD e GBP non esiste un "livello" del premio di curva: esiste una struttura per
REGIME. Riportare una media incondizionata su quei mercati e' fuorviante in entrambe le
direzioni -- ed e' anche la ragione per cui il 21 concludeva "il cuneo e' interamente VRP":
la media incondizionata mescola regimi opposti che si cancellano.

IL TEST CHE MANCA. Tutti i pannelli sul meccanismo (05, 12) ordinano i RENDIMENTI per stress.
Nessuno ordina il LIVELLO EPURATO. Ma la storia delle clientele fa una predizione diretta sul
livello: quando il bilancio degli intermediari e' vincolato, la clientela di duration resta
sola a fissare il prezzo della convessita' sulla curva, quindi il premio epurato R deve essere
piu' ALTO. Se cosi' fosse, il "nulla" di USD e GBP non e' assenza di premio: e' un premio
CONDIZIONALE la cui media incondizionata e' zero perche' i regimi si compensano.

QUATTRO PANNELLI:
 [1] R per era, sui livelli epurati (dove sta il salto, e ha lo stesso segno nei due mercati?)
 [2] R per terzile di stress dei dealer -- il test diretto della predizione
 [3] R per terzile di capitale degli intermediari (HKM) -- la stessa cosa con lente indipendente
 [4] regressione con rottura endogena: R ~ costante + shift dopo la data di rottura ottimale
     (Quandt-Andrews su finestra centrale), per stabilire SE la rottura e' unica e datata

Output: results/25_regime.txt + figures/figc8_regime.png
"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import save_txt, load_legs_mid_all, load_vols, load_dealer_cds
import glob
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,"grid.alpha":0.3})

sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
mid = load_legs_mid_all(); IV = load_vols()
L=[]; P=L.append
P("=== 25 IL PREMIO EPURATO E' CONDIZIONALE? ===")

H_M=3
MK4=[m for m in ["USDswap","EUR","GBP","JPY"] if m in sbe.columns]
FAM={"USDswap":"USOSFR","EUR":"EUSA","GBP":"BPSWS","JPY":"JYSO"}
LAB={"USDswap":"USD swap","EUR":"EUR","GBP":"GBP","JPY":"JPY"}
REG={"USDswap":"US","EUR":"EU","GBP":"EU","JPY":"US"}
ERAS=[("pre-GFC","2002","2007-06"),("GFC","2007-07","2009-12"),("ZIRP","2010","2015"),
      ("normal.","2016","2019"),("COVID","2020","2021"),("inflaz.","2022","2026")]

def nw(x,lag=6):
    x=pd.Series(x).dropna().values; n=len(x)
    if n<18: return np.nan
    e=x-x.mean(); s=e@e/n
    for l in range(1,lag+1): s+=2*(1-l/(lag+1))*(e[l:]@e[:-l])/n
    return x.mean()/np.sqrt(s/n)

def Rser(m):
    """premio epurato sul campione PIENO (ex-post: il 24 mostra che il previsore non cambia nulla)"""
    dy=(mid[f"{FAM[m]}10"]/100.0).diff()
    rv=(dy.rolling(63).std()*np.sqrt(252)*1e4).resample("ME").last().shift(-H_M)
    al=pd.concat([sbe[m].rename("be"),rv.rename("rv")],axis=1).dropna()
    return (al["be"]-al["rv"]) if len(al)>=60 else None

R={m:Rser(m) for m in MK4}; R={k:v for k,v in R.items() if v is not None}

# ---------------------------------------------------------------- [1] per era
P("")
P("[1] IL PREMIO EPURATO R PER ERA (bp/yr; media [NW t], T)")
P(f"{'mercato':9}" + "".join(f"{e[0]:>17}" for e in ERAS))
for m,r in R.items():
    row=""
    for _,a,b in ERAS:
        s=r.loc[a:b].dropna()
        row += f"{s.mean():8.0f}[{nw(s) if len(s)>=18 else np.nan:4.1f}]" if len(s)>=6 else f"{'--':>17}"
    P(f"{LAB[m]:9}{row}")
P("    Se il segno del salto e' lo STESSO nei mercati anglosassoni (alto post-crisi, negativo")
P("    nell'era inflazione/QT) e' struttura economica comune; se ogni mercato ha il suo, e'")
P("    idiosincratico e il livello resta non informativo.")

# ---------------------------------------------------------------- [2] terzili di stress
P("")
P("[2] IL TEST DIRETTO: R per terzile di STRESS dei dealer")
P("    predizione della storia: bilancio vincolato -> la clientela di duration resta sola a")
P("    fissare il prezzo sulla curva -> premio epurato PIU' ALTO nel terzile HIGH.")
try:
    CDS,_us,_eu = load_dealer_cds()
    P(f"{'mercato':9}{'LOW':>9}{'MID':>9}{'HIGH':>9}{'[t] HIGH':>10}{'H-L':>9}{'[t] H-L':>9}{'T':>5}")
    for m,r in R.items():
        c=CDS.get(REG[m])
        if c is None: continue
        al=pd.concat([r.rename("R"),c.resample("ME").last().rename("s")],axis=1).dropna()
        if len(al)<60: continue
        q=al["s"].quantile([1/3,2/3]).values
        lo=al["R"][al["s"]<q[0]]; mi=al["R"][(al["s"]>=q[0])&(al["s"]<q[1])]; hi=al["R"][al["s"]>=q[1]]
        dif=hi.mean()-lo.mean()
        # t sulla differenza via regressione su dummy HIGH vs LOW
        sub=pd.concat([hi.rename("y").to_frame().assign(d=1), lo.rename("y").to_frame().assign(d=0)])
        yv=sub["y"].values; dv=sub["d"].values
        X=np.column_stack([np.ones(len(dv)),dv]); b=np.linalg.lstsq(X,yv,rcond=None)[0]
        e=yv-X@b; V=np.linalg.inv(X.T@X)*(e@e)/(len(yv)-2)
        P(f"{LAB[m]:9}{lo.mean():9.0f}{mi.mean():9.0f}{hi.mean():9.0f}{nw(hi):10.1f}"
          f"{dif:9.0f}{b[1]/np.sqrt(V[1,1]):9.1f}{len(al):5d}")
except Exception as e:
    P(f"    non calcolato ({e})")

# ---------------------------------------------------------------- [3] lente HKM
P("")
P("[3] STESSO TEST CON LENTE INDIPENDENTE: terzili di capitale degli intermediari (HKM)")
P("    CAP-BASSO = intermediari vincolati -> il premio epurato deve stare li'.")
hits = sorted(glob.glob(str(RAW/"**"/"He_Kelly_Manela_Factors_monthly*.csv"), recursive=True)) \
     + sorted(glob.glob(str(RAW/"**"/"*Kelly_Manela*.csv"), recursive=True))
if not hits:
    P("    file HKM non trovato sotto raw/ -- pannello saltato")
else:
    H=pd.read_csv(hits[0])
    H=H[pd.to_numeric(H["yyyymm"],errors="coerce").notna()].drop_duplicates(subset="yyyymm",keep="last")
    H.index=pd.to_datetime(H["yyyymm"].astype(int).astype(str),format="%Y%m")+pd.offsets.MonthEnd(0)
    cap=pd.to_numeric(H["intermediary_capital_ratio"],errors="coerce").dropna()
    P(f"{'mercato':9}{'CAP-BASSO':>11}{'MID':>9}{'CAP-ALTO':>10}{'[t] BASSO':>11}{'L-H':>9}{'T':>5}")
    for m,r in R.items():
        al=pd.concat([r.rename("R"),cap.rename("h")],axis=1).dropna()
        if len(al)<60: continue
        q=al["h"].quantile([1/3,2/3]).values
        lo=al["R"][al["h"]<q[0]]; mi=al["R"][(al["h"]>=q[0])&(al["h"]<q[1])]; hi=al["R"][al["h"]>=q[1]]
        P(f"{LAB[m]:9}{lo.mean():11.0f}{mi.mean():9.0f}{hi.mean():10.0f}{nw(lo):11.1f}"
          f"{lo.mean()-hi.mean():9.0f}{len(al):5d}")

# ------------------------------------------------- [3b] il sort HKM e' il regime travestito?
P("")
P("[3b] CONTROLLO DECISIVO: il sort HKM sopravvive DENTRO i regimi?")
P("     Il pannello 4 trova rotture a fine 2021/inizio 2022. Se il capitale HKM e' basso PRIMA")
P("     e alto DOPO (o viceversa), il sort dei terzili sta solo replicando il salto temporale.")
P("     Qui il sort e' rifatto SEPARATAMENTE nei due sotto-periodi definiti dalla rottura.")
if hits:
    P(f"{'mercato':9}{'sottoperiodo':16}{'CAP-BASSO':>11}{'CAP-ALTO':>10}{'L-H':>8}{'[t] L-H':>9}{'T':>5}")
    for m,r in R.items():
        al=pd.concat([r.rename("R"),cap.rename("h")],axis=1).dropna()
        if len(al)<80: continue
        # rottura ricalcolata qui per indipendenza dal pannello 4
        n=len(al); best=None
        for i in range(int(.15*n),int(.85*n)):
            d=(np.arange(n)>=i).astype(float); X=np.column_stack([np.ones(n),d]); y=al["R"].values
            b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b
            F=(((y-y.mean())**2).sum()-e@e)/((e@e)/(n-2))
            if best is None or F>best[0]: best=(F,i)
        bi=best[1]; parts=[("prima",al.iloc[:bi]),("dopo",al.iloc[bi:])]
        for nm,sub in parts:
            if len(sub)<30: P(f"{'':9}{nm:16}{'campione troppo corto':>38}"); continue
            q=sub["h"].quantile([1/3,2/3]).values
            lo=sub["R"][sub["h"]<q[0]]; hi=sub["R"][sub["h"]>=q[1]]
            if len(lo)<8 or len(hi)<8: continue
            yv=np.r_[lo.values,hi.values]; dv=np.r_[np.ones(len(lo)),np.zeros(len(hi))]
            X=np.column_stack([np.ones(len(dv)),dv]); b=np.linalg.lstsq(X,yv,rcond=None)[0]
            e=yv-X@b; V=np.linalg.inv(X.T@X)*(e@e)/max(len(yv)-2,1)
            P(f"{LAB[m] if nm=='prima' else '':9}{nm:16}{lo.mean():11.0f}{hi.mean():10.0f}"
              f"{lo.mean()-hi.mean():8.0f}{b[1]/np.sqrt(V[1,1]):9.1f}{len(sub):5d}")
        P("")
    P("     Se il segno L-H resta positivo in ENTRAMBI i sottoperiodi, il premio condizionale")
    P("     e' reale e non e' la rottura travestita. Se sparisce dentro i regimi, il pannello 3")
    P("     stava solo ordinando il tempo.")
else:
    P("     saltato (file HKM assente)")

# ---------------------------------------------------------------- [4] rottura endogena
P("")
P("[4] ROTTURA ENDOGENA (Quandt-Andrews sul 15-85% centrale): esiste UNA data di rottura?")
P(f"{'mercato':9}{'data rottura':>14}{'F max':>8}{'medio PRIMA':>13}{'medio DOPO':>12}{'T':>5}")
BRK={}
for m,r in R.items():
    r=r.dropna(); n=len(r)
    if n<80: continue
    lo_i,hi_i=int(.15*n),int(.85*n); best=None
    for i in range(lo_i,hi_i):
        d=(np.arange(n)>=i).astype(float)
        X=np.column_stack([np.ones(n),d]); y=r.values
        b=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b
        rss=e@e; rss0=((y-y.mean())**2).sum()
        F=(rss0-rss)/(rss/(n-2))
        if best is None or F>best[0]: best=(F,i,b)
    F,i,b=best; BRK[m]=r.index[i]
    P(f"{LAB[m]:9}{str(r.index[i].date()):>14}{F:8.0f}{b[0]:13.0f}{b[0]+b[1]:12.0f}{n:5d}")
P("    Una rottura netta e datata simile fra mercati suggerisce un evento comune (QT, fine")
P("    del QE); date molto diverse suggeriscono cause locali. NB: la data e' scelta sui dati,")
P("    quindi F non ha la distribuzione standard -- va letto come descrittivo.")

# ---------------------------------------------------------------- figura
fig,ax=plt.subplots(figsize=(10,4.5))
for m,r in R.items():
    ax.plot(r.index, r.rolling(12).mean(), lw=1.3, label=LAB[m])
ax.axhline(0,color="k",lw=.8)
for m,d in BRK.items(): ax.axvline(d,ls=":",lw=.8,color="grey")
ax.set_ylabel("bp/yr"); ax.legend(fontsize=8,ncol=4)
ax.set_title("Purged curve premium R, 12-month moving average (dotted: endogenous break dates)",fontsize=10)
fig.tight_layout(); fig.savefig(FIG/"figc8_regime.png"); plt.close(fig)
P(""); P(f"[figura] {FIG/'figc8_regime.png'}")

P("")
P("COME LEGGERE L'INSIEME. Se [2] e [3] mostrano il premio epurato concentrato negli stati")
P("VINCOLATI anche dove la media incondizionata e' nulla, allora USD e GBP non sono 'mercati")
P("senza premio': sono mercati dove il premio e' CONDIZIONALE e la media incondizionata e' zero")
P("perche' i regimi si compensano. Sarebbe il livello che replica cio' che i rendimenti gia'")
P("mostrano in 05/12, e riporterebbe il livello dentro la storia invece che fuori.")
save_txt("25_regime.txt", L); print("\n".join(L))

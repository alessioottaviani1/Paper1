"""26 - I DUE CANALI SUL LIVELLO EPURATO: perche' il segno si ribalta, e se e' coerente.

COSA HA MOSTRATO IL 25. Il sort del premio epurato R per capitale degli intermediari ha t di
4.8 (USD) e 4.4 (GBP) INCONDIZIONATAMENTE, ma dentro i sottoperiodi il segno SI RIBALTA:
L-H = +70 prima della rottura e -101 dopo in USD; +48 e -130 in GBP. Il sort incondizionato
stava quindi ordinando anche IL TEMPO, non solo lo stato dei bilanci.

L'IPOTESI CHE QUESTO SCRIPT TESTA. Il ribaltamento non e' rumore: e' il meccanismo A DUE CANALI
gia' documentato in 09 (MOVE e CDS dealer con segni opposti) e in 16 (ILLIQ col segno
"sbagliato"). Due tipi di stress agiscono in direzioni opposte sul prezzo di curva:

  CANALE 1 -- FUGA / STRESS DI VOLATILITA' (MOVE, illiquidita'): il capitale si rifugia sul
    lungo, la domanda di convessita' sale, la curva diventa CARA  =>  R positivo.
  CANALE 2 -- DISTRUZIONE DI BILANCIO (CDS dealer, funding, QT/LDI): i copritori diventano
    venditori forzati, la convessita' viene liquidata, la curva diventa ECONOMICA => R negativo.

Se i due canali entrano INSIEME con segni OPPOSTI e ciascuno significativo, il ribaltamento del
25 e' spiegato invece che imbarazzante, e il livello epurato rientra nella storia come oggetto
a due facce. Se invece un solo canale sopravvive, il livello resta non identificato e va
presentato come tale.

PANNELLI
 [1] R ~ MOVE + CDS dealer, in livelli standardizzati, con NW. Il test: segni OPPOSTI.
 [2] Lo stesso con la lente indipendente: R ~ ILLIQ/VIX + HKM.
 [3] Fuori campione temporale: gli stessi coefficienti stimati PRIMA della rottura predicono
     il segno DOPO? (stima pre-rottura, valutazione post-rottura)
 [4] Il contributo dei due canali al ribaltamento: quanto della variazione media di R fra i
     due regimi e' spiegato dalla variazione media dei due stress?

Output: results/26_two_channels.txt
"""
import pandas as pd, numpy as np, glob, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import save_txt, load_legs_mid_all, load_vols, load_dealer_cds, load_market_states

sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
mid = load_legs_mid_all(); IV = load_vols()
L=[]; P=L.append
P("=== 26 I DUE CANALI SUL LIVELLO EPURATO ===")

H_M=3
MK4=[m for m in ["USDswap","EUR","GBP","JPY"] if m in sbe.columns]
FAM={"USDswap":"USOSFR","EUR":"EUSA","GBP":"BPSWS","JPY":"JYSO"}
LAB={"USDswap":"USD swap","EUR":"EUR","GBP":"GBP","JPY":"JPY"}
REG={"USDswap":"US","EUR":"EU","GBP":"EU","JPY":"US"}

def Rser(m):
    dy=(mid[f"{FAM[m]}10"]/100.0).diff()
    rv=(dy.rolling(63).std()*np.sqrt(252)*1e4).resample("ME").last().shift(-H_M)
    al=pd.concat([sbe[m].rename("be"),rv.rename("rv")],axis=1).dropna()
    return (al["be"]-al["rv"]) if len(al)>=60 else None
R={m:Rser(m) for m in MK4}; R={k:v for k,v in R.items() if v is not None}

def z(s): s=s.dropna(); return (s-s.mean())/s.std()
def ols_nw(y,X,lag=6):
    Xv,yv=np.asarray(X,float),np.asarray(y,float)
    b=np.linalg.lstsq(Xv,yv,rcond=None)[0]; e=yv-Xv@b
    A=np.linalg.inv(Xv.T@Xv); S=(e[:,None]*Xv).T@(e[:,None]*Xv)
    for l in range(1,lag+1):
        w=1-l/(lag+1); u=e[l:,None]*Xv[l:]; v=e[:-l,None]*Xv[:-l]; G=u.T@v; S+=w*(G+G.T)
    V=A@S@A
    return b, b/np.sqrt(np.diag(V)), 1-np.var(e)/np.var(yv)

CDS,_,_ = load_dealer_cds()
ST = load_market_states()
def col(df,*names):
    for n in names:
        for c in df.columns:
            if n.lower() in str(c).lower(): return df[c]
    return None
MOVE = col(ST,"MOVE"); VIX = col(ST,"VIX")

hits = sorted(glob.glob(str(RAW/"**"/"He_Kelly_Manela_Factors_monthly*.csv"), recursive=True)) \
     + sorted(glob.glob(str(RAW/"**"/"*Kelly_Manela*.csv"), recursive=True))
HKM=None
if hits:
    H=pd.read_csv(hits[0])
    H=H[pd.to_numeric(H["yyyymm"],errors="coerce").notna()].drop_duplicates(subset="yyyymm",keep="last")
    H.index=pd.to_datetime(H["yyyymm"].astype(int).astype(str),format="%Y%m")+pd.offsets.MonthEnd(0)
    HKM=pd.to_numeric(H["intermediary_capital_ratio"],errors="coerce").dropna()

# ---------------------------------------------------------------- [1] i due canali insieme
P("")
P("[1] R ~ STRESS DI VOLATILITA' + STRESS DI BILANCIO (livelli standardizzati, NW lag 6)")
P("    predizione: b_VOL > 0 (fuga -> curva cara)  e  b_BIL < 0 (deleveraging -> curva economica)")
P(f"{'mercato':9}{'b_MOVE':>9}{'[t]':>7}{'b_CDS':>9}{'[t]':>7}{'R2':>7}{'segni':>10}{'T':>5}")
for m,r in R.items():
    mv=MOVE; cd=CDS.get(REG[m])
    if mv is None or cd is None: continue
    al=pd.concat([r.rename("R"), z(mv.resample("ME").last()).rename("mv"),
                  z(cd.resample("ME").last()).rename("cd")],axis=1).dropna()
    if len(al)<60: continue
    X=np.column_stack([np.ones(len(al)),al["mv"],al["cd"]])
    b,t,r2=ols_nw(al["R"].values,X)
    ok = "attesi" if (b[1]>0 and b[2]<0) else ("opposti" if (b[1]<0 and b[2]>0) else "misti")
    P(f"{LAB[m]:9}{b[1]:9.1f}{t[1]:7.1f}{b[2]:9.1f}{t[2]:7.1f}{r2:7.2f}{ok:>10}{len(al):5d}")

# ---------------------------------------------------------------- [2] lente indipendente
P("")
P("[2] LENTE INDIPENDENTE: R ~ VIX (stress di vol) + HKM (capitale intermediari)")
P("    predizione: b_VIX > 0  e  b_HKM > 0 (capitale ALTO = non vincolati = curva meno economica)")
if VIX is not None and HKM is not None:
    P(f"{'mercato':9}{'b_VIX':>9}{'[t]':>7}{'b_HKM':>9}{'[t]':>7}{'R2':>7}{'T':>5}")
    for m,r in R.items():
        al=pd.concat([r.rename("R"), z(VIX.resample("ME").last()).rename("v"),
                      z(HKM).rename("h")],axis=1).dropna()
        if len(al)<60: continue
        X=np.column_stack([np.ones(len(al)),al["v"],al["h"]])
        b,t,r2=ols_nw(al["R"].values,X)
        P(f"{LAB[m]:9}{b[1]:9.1f}{t[1]:7.1f}{b[2]:9.1f}{t[2]:7.1f}{r2:7.2f}{len(al):5d}")
else:
    P("    VIX o HKM non disponibili -- pannello saltato")

# ---------------------------------------------------------------- [3] out-of-sample temporale
P("")
P("[3] I COEFFICIENTI STIMATI PRIMA DELLA ROTTURA PREDICONO IL SEGNO DOPO?")
P("    stima su pre-rottura, valutazione su post-rottura. E' il test piu' severo: se i due")
P("    canali sono struttura, i coefficienti pre devono spiegare il crollo post.")
P(f"{'mercato':9}{'rottura':>12}{'R medio post':>14}{'R predetto post':>17}{'errore':>9}{'T post':>7}")
for m,r in R.items():
    mv=MOVE; cd=CDS.get(REG[m])
    if mv is None or cd is None: continue
    al=pd.concat([r.rename("R"), z(mv.resample("ME").last()).rename("mv"),
                  z(cd.resample("ME").last()).rename("cd")],axis=1).dropna()
    n=len(al)
    if n<100: continue
    best=None
    for i in range(int(.15*n),int(.85*n)):
        d=(np.arange(n)>=i).astype(float); X=np.column_stack([np.ones(n),d]); y=al["R"].values
        b0=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b0
        F=(((y-y.mean())**2).sum()-e@e)/((e@e)/(n-2))
        if best is None or F>best[0]: best=(F,i)
    bi=best[1]; pre=al.iloc[:bi]; post=al.iloc[bi:]
    if len(pre)<60 or len(post)<12: continue
    Xp=np.column_stack([np.ones(len(pre)),pre["mv"],pre["cd"]])
    bp=np.linalg.lstsq(Xp,pre["R"].values,rcond=None)[0]
    Xq=np.column_stack([np.ones(len(post)),post["mv"],post["cd"]])
    pred=(Xq@bp).mean(); obs=post["R"].mean()
    P(f"{LAB[m]:9}{str(al.index[bi].date()):>12}{obs:14.0f}{pred:17.0f}{pred-obs:9.0f}{len(post):7d}")
P("    Se il predetto ha il SEGNO GIUSTO e ordine di grandezza plausibile, i due canali")
P("    spiegano il ribaltamento. Se predice ~0 mentre l'osservato e' -140, il crollo post-2022")
P("    e' qualcosa che le due variabili di stato NON catturano, e va detto.")

# ---------------------------------------------------------------- [4] decomposizione del salto
P("")
P("[4] QUANTO DEL SALTO FRA REGIMI E' SPIEGATO DAL MOVIMENTO DEI DUE STRESS?")
P(f"{'mercato':9}{'salto R':>10}{'da MOVE':>10}{'da CDS':>10}{'residuo':>10}{'quota spieg.':>14}")
for m,r in R.items():
    mv=MOVE; cd=CDS.get(REG[m])
    if mv is None or cd is None: continue
    al=pd.concat([r.rename("R"), z(mv.resample("ME").last()).rename("mv"),
                  z(cd.resample("ME").last()).rename("cd")],axis=1).dropna()
    n=len(al)
    if n<100: continue
    best=None
    for i in range(int(.15*n),int(.85*n)):
        d=(np.arange(n)>=i).astype(float); X=np.column_stack([np.ones(n),d]); y=al["R"].values
        b0=np.linalg.lstsq(X,y,rcond=None)[0]; e=y-X@b0
        F=(((y-y.mean())**2).sum()-e@e)/((e@e)/(n-2))
        if best is None or F>best[0]: best=(F,i)
    bi=best[1]; pre=al.iloc[:bi]; post=al.iloc[bi:]
    if len(pre)<60 or len(post)<12: continue
    X=np.column_stack([np.ones(n),al["mv"],al["cd"]])
    b,_,_=ols_nw(al["R"].values,X)
    d_mv=b[1]*(post["mv"].mean()-pre["mv"].mean())
    d_cd=b[2]*(post["cd"].mean()-pre["cd"].mean())
    jump=post["R"].mean()-pre["R"].mean()
    res=jump-d_mv-d_cd
    P(f"{LAB[m]:9}{jump:10.0f}{d_mv:10.0f}{d_cd:10.0f}{res:10.0f}{1-abs(res)/max(abs(jump),1e-9):14.0%}")

P("")
P("VERDETTO DA LEGGERE ONESTAMENTE. Il 25 ha mostrato che il sort incondizionato per capitale")
P("degli intermediari NON sopravvive dentro i regimi: il segno si ribalta. Qui si stabilisce se")
P("il ribaltamento e' il meccanismo a DUE CANALI (e allora il livello e' un oggetto a due facce,")
P("coerente con 09 e 16) oppure se il crollo post-2022 e' fuori dalla portata delle variabili di")
P("stato disponibili. Nel secondo caso l'affermazione corretta per il paper e': il livello")
P("epurato e' informativo solo in EUR, e negli altri mercati il contributo passa dal CO-MOVIMENTO")
P("e dai RENDIMENTI condizionali (05, 12), non dal livello.")
save_txt("26_two_channels.txt", L); print("\n".join(L))

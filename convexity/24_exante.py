"""24 - IL CUNEO CON PREVISIONE EX-ANTE: separare il PREMIO dalla SORPRESA di volatilita'.

IL PROBLEMA CHE QUESTO SCRIPT RISOLVE. In 21/22/23 il residuo e' costruito come
    R_expost = sigma_BE(t) - RV(t, t+3m)
con RV EX POST. Ma allora
    R_expost = [sigma_BE - E_t(RV)]  -  [RV_realizzata - E_t(RV)]
             =        PREMIO          -        SORPRESA
e la sorpresa e' rumore ex ante con code enormi: nel 2022 la volatilita' realizzata esplode e
R crolla a -300/-380 in USD/GBP anche se la curva non ha "sbagliato" nulla. Risultato: la MEDIA
di R e' schiacciata verso zero (USD 4.9, GBP 3.5) mentre la MEDIANA e' saldamente positiva
(35.7, 36.6) e il 66-69% dei mesi e' positivo. La media misura premio MENO sorpresa; la
mediana e' robusta alla coda ma non e' una stima del premio.

Nota che questo e' esattamente il difetto che Rebonato-Putyatin (2018, sez. 13) diagnosticano
nel proprio lavoro: le discrepanze fra profitto predetto e realizzato vengono dal fallimento nel
prevedere la volatilita' realizzata istantanea.

LA CORREZIONE. Sostituire RV ex post con una PREVISIONE EX ANTE E_t[RV], costruita con sola
informazione disponibile in t. Tre previsori, tutti out-of-sample:
  (a) HAR-RV: regressione su RV giornaliera/settimanale/mensile trailing, stimata su finestra
      espandente (min 60 mesi) e proiettata a t+3m. E' lo standard della letteratura.
  (b) RV trailing 63g: benchmark naive.
  (c) sigma_IV meno il VRP MEDIO stimato in modo espandente: usa l'opzione come previsore,
      depurata del suo premio medio storico. Indipendente dalla curva per costruzione.
Con questi, R_exante = sigma_BE - E_t[RV] misura il PREMIO DI CURVA, non premio meno sorpresa.

INFERENZA. Per ciascun previsore: media con t di Newey-West, mediana con IC bootstrap (5000
ricampionamenti a blocchi, blocco 12m per l'autocorrelazione), quota di mesi positivi con test
binomiale esatto. La mediana con IC e' la statistica da riportare nel paper accanto alla media.

RICONCILIAZIONE CAMPIONI. 23 usava T=329 per EUR (non serve sigma_IV), 21 usava T=264. Qui tutto
gira sul CAMPIONE COMUNE con sigma_IV disponibile, e la differenza e' riportata esplicitamente.

Output: results/24_exante.txt + figures/figc7_exante.png
"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from scipy import stats
from config import *
from utils import save_txt, load_legs_mid_all, load_vols
plt.rcParams.update({"figure.dpi":150,"font.size":9,"axes.grid":True,"grid.alpha":0.3})
rng = np.random.default_rng(77)

sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
mid = load_legs_mid_all(); IV = load_vols()
L=[]; P=L.append
P("=== 24 CUNEO EX-ANTE: premio di curva separato dalla sorpresa di volatilita' ===")

H_M=3; MIN_TRAIN=60
MK4=[m for m in ["USDswap","EUR","GBP","JPY"] if m in sbe.columns]
FAM={"USDswap":"USOSFR","EUR":"EUSA","GBP":"BPSWS","JPY":"JYSO"}
LAB={"USDswap":"USD swap","EUR":"EUR","GBP":"GBP","JPY":"JPY"}

def ivser(m):
    s=IV.get((IVMAP.get(m,""),"3M","10Y","NORM"))
    return None if s is None else s.resample("ME").last()

def rv_components(m):
    """RV trailing a 3 orizzonti (giorn./sett./mens. aggregate) + RV futura, mensili."""
    dy=(mid[f"{FAM[m]}10"]/100.0).diff()
    ann=lambda w: (dy.rolling(w).std()*np.sqrt(252)*1e4).resample("ME").last()
    d=pd.DataFrame({"rv_d":ann(21),"rv_w":ann(63),"rv_m":ann(252)})
    d["rv_fut"]=ann(63).shift(-H_M)
    return d.dropna(subset=["rv_d","rv_w","rv_m"])

def nw(x,lag=6):
    x=pd.Series(x).dropna().values; n=len(x)
    if n<24: return np.nan
    e=x-x.mean(); s=e@e/n
    for l in range(1,lag+1):
        s+=2*(1-l/(lag+1))*(e[l:]@e[:-l])/n
    return x.mean()/np.sqrt(s/n)

def block_boot_median(x,B=5000,bl=12):
    x=np.asarray(pd.Series(x).dropna()); n=len(x)
    if n<36: return (np.nan,np.nan)
    nb=int(np.ceil(n/bl)); out=np.empty(B)
    for b in range(B):
        st=rng.integers(0,max(1,n-bl),nb)
        out[b]=np.median(np.concatenate([x[s:s+bl] for s in st])[:n])
    return tuple(np.percentile(out,[2.5,97.5]))

def har_forecast(D):
    """HAR-RV out-of-sample espandente: prevede rv_fut da rv_d, rv_w, rv_m."""
    idx=D.index; f=pd.Series(index=idx,dtype=float)
    X=D[["rv_d","rv_w","rv_m"]].values; y=D["rv_fut"].values
    for i in range(MIN_TRAIN,len(D)):
        tr=slice(0,i-H_M)                       # esclude le osservazioni non ancora realizzate
        Xt,yt=X[tr],y[tr]; ok=~np.isnan(yt)
        if ok.sum()<MIN_TRAIN-H_M: continue
        A=np.column_stack([np.ones(ok.sum()),Xt[ok]])
        b=np.linalg.lstsq(A,yt[ok],rcond=None)[0]
        f.iloc[i]=float(b[0]+b[1:]@X[i])
    return f

def iv_minus_vrp(D,iv):
    """sigma_IV meno il VRP medio stimato in modo ESPANDENTE (solo passato)."""
    al=pd.concat([iv.rename("iv"),D["rv_fut"].rename("rvf")],axis=1)
    vrp=(al["iv"]-al["rvf"])
    exp_vrp=vrp.shift(H_M).expanding(MIN_TRAIN).mean()   # shift: usa solo VRP gia' osservati
    return al["iv"]-exp_vrp

# ---------------------------------------------------------------- pannello principale
P("")
P("[1] IL PREMIO DI CURVA CON PREVISORI EX-ANTE (campione comune con sigma_IV)")
P("    R = sigma_BE - E_t[RV].  ex-post = la vecchia costruzione, per confronto.")
P(f"{'mercato':9}{'previsore':16}{'media':>8}{'[NW t]':>8}{'mediana':>9}{'IC95 mediana':>18}{'%>0':>6}{'p bin':>8}{'T':>5}")
STORE={}
for m in MK4:
    iv=ivser(m); D=rv_components(m)
    if iv is None: continue
    D=D.join(iv.rename("iv"),how="inner").join(sbe[m].rename("be"),how="inner")
    if len(D)<MIN_TRAIN+24: continue
    fc={"ex-post (21/22/23)":D["rv_fut"],
        "trailing 63g":D["rv_w"],
        "HAR-RV oos":har_forecast(D),
        "IV - VRP espand.":iv_minus_vrp(D,D["iv"])}
    for nm,f in fc.items():
        R=(D["be"]-f).dropna()
        if len(R)<36: continue
        lo,hi=block_boot_median(R.values)
        pos=int((R>0).sum()); pb=stats.binomtest(pos,len(R),0.5).pvalue
        STORE[(m,nm)]=R
        P(f"{LAB[m] if nm.startswith('ex-post') else '':9}{nm:16}{R.mean():8.1f}{nw(R):8.1f}"
          f"{R.median():9.1f}{f'[{lo:6.1f},{hi:6.1f}]':>18}{(R>0).mean():6.0%}{pb:8.3f}{len(R):5d}")
    P("")
P("    LETTURA. Se con i previsori EX-ANTE la media sale verso la mediana e il t diventa")
P("    informativo, allora il quasi-zero della media ex-post era SORPRESA di volatilita', non")
P("    assenza di premio. La mediana con IC bootstrap e la quota di mesi positivi col test")
P("    binomiale sono le statistiche robuste da riportare nel paper accanto alla media.")

# ---------------------------------------------------------------- [1b] STESSO CAMPIONE
P("")
P("[1b] IL CONFRONTO CORRETTO: ex-post vs ex-ante SULLO STESSO CAMPIONE")
P("     L'HAR richiede 60 mesi di training, quindi scarta i primi anni. Se quelli erano mesi")
P("     con R alto, il calo apparente e' TRONCAMENTO DEL CAMPIONE, non qualita' del previsore.")
P("     Qui ex-post e ex-ante sono valutati sulle STESSE date.")
P(f"{'mercato':9}{'campione':22}{'ex-post':>9}{'[t]':>7}{'ex-ante':>9}{'[t]':>7}{'differenza':>12}{'T':>5}")
for m in MK4:
    iv=ivser(m); D=rv_components(m)
    if iv is None: continue
    D=D.join(iv.rename("iv"),how="inner").join(sbe[m].rename("be"),how="inner")
    if len(D)<MIN_TRAIN+24: continue
    f=har_forecast(D)
    al=pd.concat([D["be"].rename("be"),D["rv_fut"].rename("rvf"),f.rename("f")],axis=1).dropna()
    if len(al)<36: continue
    rex_full=(D["be"]-D["rv_fut"]).dropna()
    rex_sub=(al["be"]-al["rvf"]); rex_ante=(al["be"]-al["f"])
    P(f"{LAB[m]:9}{'pieno (ex-post)':22}{rex_full.mean():9.1f}{nw(rex_full):7.1f}{'--':>9}{'--':>7}"
      f"{'--':>12}{len(rex_full):5d}")
    P(f"{'':9}{'comune HAR':22}{rex_sub.mean():9.1f}{nw(rex_sub):7.1f}{rex_ante.mean():9.1f}"
      f"{nw(rex_ante):7.1f}{rex_ante.mean()-rex_sub.mean():12.1f}{len(al):5d}")
    dropped = len(rex_full)-len(al)
    early = rex_full.iloc[:dropped] if dropped>0 else pd.Series(dtype=float)
    if len(early)>6:
        P(f"{'':9}{'mesi scartati':22}{early.mean():9.1f}{'':7}{'':9}{'':7}{'':12}{len(early):5d}")
    P("")
P("     Se 'comune HAR' ex-post e' gia' vicino a ex-ante, il previsore NON e' il problema: la")
P("     differenza col campione pieno e' interamente il periodo scartato. La riga 'mesi")
P("     scartati' quantifica cosa si perde.")

# ---------------------------------------------------------------- scomposizione premio/sorpresa
P("")
P("[2] SCOMPOSIZIONE ESPLICITA: R_expost = PREMIO - SORPRESA  (previsore HAR)")
P(f"{'mercato':9}{'R ex-post':>11}{'premio':>9}{'[t]':>7}{'-sorpresa':>11}{'quota sorpresa':>16}")
for m in MK4:
    iv=ivser(m); D=rv_components(m)
    if iv is None: continue
    D=D.join(iv.rename("iv"),how="inner").join(sbe[m].rename("be"),how="inner")
    f=har_forecast(D)
    al=pd.concat([D["be"],D["rv_fut"],f.rename("f")],axis=1).dropna()
    if len(al)<36: continue
    rex=al["be"]-al["rv_fut"]; prem=al["be"]-al["f"]; sur=al["rv_fut"]-al["f"]
    P(f"{LAB[m]:9}{rex.mean():11.1f}{prem.mean():9.1f}{nw(prem):7.1f}{-sur.mean():11.1f}"
      f"{abs(sur.mean())/max(abs(prem.mean()),1e-9):16.0%}")
P("    La colonna 'quota sorpresa' dice quanto della media ex-post e' errore di previsione")
P("    invece che premio. E' il difetto che Rebonato-Putyatin (2018, sez.13) diagnosticano")
P("    nel proprio lavoro: le discrepanze vengono dal non prevedere la vol realizzata.")

# ---------------------------------------------------------------- riconciliazione campioni
P("")
P("[3] RICONCILIAZIONE DEI CAMPIONI (la discrepanza 23 vs 21 su EUR)")
for m in MK4:
    iv=ivser(m); D=rv_components(m)
    if iv is None: continue
    full=(sbe[m].rename("be")).to_frame().join(D["rv_fut"],how="inner").dropna()
    comm=full.join(iv.rename("iv"),how="inner").dropna()
    if len(full)<36: continue
    P(f"   {LAB[m]:9} senza sigma_IV: T={len(full):3d}, R medio {(full['be']-full['rv_fut']).mean():6.1f}"
      f"  |  campione comune: T={len(comm):3d}, R medio {(comm['be']-comm['rv_fut']).mean():6.1f}")
P("   Nel paper usare SEMPRE il campione comune: il confronto fra venue richiede entrambe.")

# ---------------------------------------------------------------- figura
fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
for ax,m in zip(axes.flat,MK4):
    k=(m,"HAR-RV oos")
    if k not in STORE: ax.set_visible(False); continue
    r=STORE[k].dropna(); rex=STORE.get((m,"ex-post (21/22/23)"))
    ax.axhline(0,color="k",lw=.8)
    if rex is not None: ax.plot(rex.index,rex.values,color="grey",lw=.7,alpha=.6,label="ex-post")
    ax.plot(r.index,r.values,color="tab:blue",lw=1.0,label="ex-ante (HAR)")
    ax.axhline(r.median(),color="tab:red",ls="--",lw=1.1)
    ax.set_title(f"{LAB[m]}   ex-ante: mean {r.mean():.0f}, median {r.median():.0f} bp",fontsize=9)
    ax.set_ylabel("bp/yr"); ax.legend(fontsize=7,loc="lower left")
fig.suptitle("Curve premium with an ex-ante volatility forecast (grey: ex-post construction)",fontsize=10)
fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig(FIG/"figc7_exante.png"); plt.close(fig)
P(""); P(f"[figura] {FIG/'figc7_exante.png'}")
save_txt("24_exante.txt", L); print("\n".join(L))

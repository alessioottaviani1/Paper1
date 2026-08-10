"""30 - LO SPANNING: sigma_BE e' un prezzo di volatilita' o forma di curva travestita?

L'OBIEZIONE, nella sua forma piu' forte. sigma_BE e' derivata dal carry della farfalla, quindi
e' una funzione DETERMINISTICA delle pendenze locali della curva ai tre nodi. Un referee puo'
scrivere: "il vostro oggetto e' forma di curva, non un prezzo di volatilita'; che non co-muova
con la volatilita' implicita e' il fatto noto della volatilita' NON SPANNATA (Collin-Dufresne e
Goldstein 2002; Andersen e Benzoni 2010), non segmentazione". Tre elementi del pipeline
alimentano quella lettura: 12[C] (Delta-sigma_BE carica sul term premium con t 5.6-7.7),
22[5] (sigma_BE non prevede la RV futura in 3 mercati su 4), 15[D] (bassa R2 sul fly tradabile,
che pero' e' un oggetto DIVERSO dai fattori di curva).

Il test che manca, e che questo script esegue, ha quattro pannelli in sequenza logica:

 [1] QUANTO DI sigma_BE E' SPANNATO dai fattori di curva della SUA stessa curva
     (livello, pendenza, curvatura in variazioni)? R2 alto => l'oggetto e' quasi ridondante e
     la presentazione va cambiata. R2 basso => contiene informazione oltre i fattori standard,
     nonostante la derivazione dalla forma.

 [2] LO STESSO PER sigma_IV. Questo e' il test di Andersen-Benzoni nella forma diretta: se
     nemmeno la volatilita' implicita e' spannata dai fattori di curva, allora il confronto del
     paper non e' "forma contro volatilita'" ma "due oggetti entrambi non spannati", e la
     letteratura USV diventa la CORNICE del paper invece della sua minaccia.

 [3] IL TEST DECISIVO: la componente NON SPANNATA di sigma_BE co-muove con sigma_IV?
     Si purga Delta-sigma_BE dei fattori di curva e si correla il residuo con Delta-sigma_IV.
     Se la disconnessione SOPRAVVIVE alla purga, l'obiezione e' respinta: non e' forma di curva
     a non correlare, e' la parte di sigma_BE che NON e' forma di curva. Se invece la
     correlazione sale molto, il fatto centrale era in parte meccanico e va detto.

 [4] LA §6.6 IN VARIAZIONI. L'identificazione within-currency (swap vs governativo contro la
     stessa superficie) e' presentata in LIVELLI, ma il paper dichiara che i livelli sono
     confusi dal VRP. Qui il contrasto e' calcolato in livelli E in variazioni, affiancati.
     Se sopravvive in variazioni e' la migliore identificazione del progetto; se svanisce, va
     retrocesso a corollario sui livelli. Meglio saperlo adesso che da un referee.

Output: results/30_spanning.txt
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import save_txt, load_legs_mid_all, load_vols

sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
mid = load_legs_mid_all(); IV = load_vols()
L=[]; P=L.append
P("=== 30 SPANNING: sigma_BE e' un prezzo di volatilita' o forma di curva? ===")

FAM={"USDswap":"USOSFR","EUR":"EUSA","GBP":"BPSWS","JPY":"JYSO",
     "USTgovt":"USGG","DEgovt":"GDBR","UKgovt":"GUKG","JPgovt":"GJGB"}
LAB={"USDswap":"USD swap","EUR":"EUR","GBP":"GBP","JPY":"JPY",
     "USTgovt":"UST","DEgovt":"Bund","UKgovt":"gilt","JPgovt":"JGB"}
IVFOR={"USDswap":"USDswap","EUR":"EUR","GBP":"GBP","JPY":"JPY",
       "USTgovt":"USDswap","DEgovt":"EUR","UKgovt":"GBP","JPgovt":"JPY"}
PAIRS=[("EUR","DEgovt"),("GBP","UKgovt"),("JPY","JPgovt"),("USDswap","USTgovt")]

def ivc(m):
    s=IV.get((IVMAP.get(IVFOR.get(m,m),""),"3M","10Y","NORM"))
    return None if s is None else s.resample("ME").last()

def curve_factors(m):
    """livello, pendenza, curvatura della curva del mercato stesso, mensili.
    Le gambe si prendono da MK, non si ricostruiscono per concatenazione: USGG2 non esiste
    (e' USGG2YR) e con la vecchia versione le quattro curve governative cadevano in silenzio."""
    if m not in MK: return None
    legs = MK[m][0]
    if any(l not in mid for l in legs): return None
    y = mid[list(legs)].dropna().resample("ME").last()
    y.columns = ["z2","z10","z30"]
    return pd.DataFrame({"lev":y.z10, "slo":y.z10-y.z2, "cur":2*y.z10-y.z2-y.z30})

def ols(y,X,lag=6):
    Xd=np.column_stack([np.ones(len(y)),X]); b=np.linalg.lstsq(Xd,y,rcond=None)[0]
    e=y-Xd@b; A=np.linalg.inv(Xd.T@Xd); S=(e[:,None]*Xd).T@(e[:,None]*Xd)
    for l in range(1,lag+1):
        w=1-l/(lag+1); u=e[l:,None]*Xd[l:]; v=e[:-l,None]*Xd[:-l]; G=u.T@v; S+=w*(G+G.T)
    V=A@S@A
    r2=1-np.var(e)/np.var(y)
    return b, b/np.sqrt(np.diag(V)), r2, e

# ---------------------------------------------------------------- [1] sigma_BE spannata?
P("")
P("[1] Delta-sigma_BE ~ Delta(livello, pendenza, curvatura) DELLA SUA STESSA CURVA")
P("    R2 alto => sigma_BE e' quasi ridondante coi fattori standard.")
P(f"{'mercato':10}{'b_lev':>8}{'[t]':>6}{'b_slo':>8}{'[t]':>6}{'b_cur':>8}{'[t]':>6}{'R2':>7}{'T':>5}")
RESID={}
for m in FAM:
    if m not in sbe.columns: continue
    CF=curve_factors(m)
    if CF is None: continue
    al=pd.concat([sbe[m].rename("be"),CF],axis=1).dropna().diff().dropna()
    if len(al)<80: continue
    b,t,r2,e=ols(al["be"].values, al[["lev","slo","cur"]].values)
    RESID[m]=pd.Series(e,index=al.index)
    P(f"{LAB[m]:10}{b[1]:8.1f}{t[1]:6.1f}{b[2]:8.1f}{t[2]:6.1f}{b[3]:8.1f}{t[3]:6.1f}{r2:7.2f}{len(al):5d}")

# Il residuo serve al pannello [4] della 32: senza questa riga quel pannello gira su un file
# vecchio e alla prima esecuzione su cartella pulita sparisce in silenzio.
pd.DataFrame(RESID).to_csv(PROC/"sigbe_perp_delta_monthly.csv")
P(f"    residuo salvato per la 32: {len(RESID)} mercati -> sigbe_perp_delta_monthly.csv")

# ---------------------------------------------------------------- [2] sigma_IV spannata?
P("")
P("[2] LO STESSO SU Delta-sigma_IV: il test di Andersen-Benzoni nella forma diretta")
P("    Se nemmeno l'implicita e' spannata, il confronto del paper e' fra DUE oggetti non")
P("    spannati, e la letteratura USV diventa la cornice invece della minaccia.")
P(f"{'mercato':10}{'b_lev':>8}{'[t]':>6}{'b_slo':>8}{'[t]':>6}{'b_cur':>8}{'[t]':>6}{'R2':>7}{'T':>5}")
for m in ["USDswap","EUR","GBP","JPY"]:
    iv=ivc(m); CF=curve_factors(m)
    if iv is None or CF is None: continue
    al=pd.concat([iv.rename("iv"),CF],axis=1).dropna().diff().dropna()
    if len(al)<80: continue
    b,t,r2,_=ols(al["iv"].values, al[["lev","slo","cur"]].values)
    P(f"{LAB[m]:10}{b[1]:8.1f}{t[1]:6.1f}{b[2]:8.1f}{t[2]:6.1f}{b[3]:8.1f}{t[3]:6.1f}{r2:7.2f}{len(al):5d}")

# ---------------------------------------------------------------- [3] il test decisivo
P("")
P("[3] TEST DECISIVO: la componente NON SPANNATA di sigma_BE co-muove con sigma_IV?")
P("    corr(residuo di Delta-sigma_BE dopo i fattori di curva, Delta-sigma_IV)")
P(f"{'mercato':10}{'corr grezza':>13}{'corr non-spannata':>19}{'variazione':>12}{'T':>5}")
for m in ["USDswap","EUR","GBP","JPY"]:
    iv=ivc(m)
    if iv is None or m not in RESID: continue
    div=iv.diff()
    raw=pd.concat([sbe[m].diff().rename("db"),div.rename("di")],axis=1).dropna()
    pur=pd.concat([RESID[m].rename("r"),div.rename("di")],axis=1).dropna()
    if len(pur)<60: continue
    c0=raw["db"].corr(raw["di"]); c1=pur["r"].corr(pur["di"])
    P(f"{LAB[m]:10}{c0:13.2f}{c1:19.2f}{c1-c0:+12.2f}{len(pur):5d}")
P("    LETTURA. Se la correlazione della componente non spannata resta bassa, l'obiezione")
P("    'e' forma di curva' e' RESPINTA: non correla nemmeno la parte di sigma_BE che non e'")
P("    forma di curva. Se sale molto, il fatto centrale era in parte meccanico e va detto.")

# ---------------------------------------------------------------- [4] within-currency
P("")
P("[4] IDENTIFICAZIONE WITHIN-CURRENCY: livelli E variazioni affiancati")
P("    Il paper dichiara i livelli confusi dal VRP, ma presenta questo contrasto in livelli.")
P(f"{'coppia':22}{'corr LIVELLI':>14}{'corr VARIAZ.':>14}{'sopravvive?':>13}{'T':>5}")
for sw,gv in PAIRS:
    iv=ivc(sw)
    if iv is None or sw not in sbe.columns or gv not in sbe.columns: continue
    al=pd.concat([sbe[sw].rename("sw"),sbe[gv].rename("gv"),iv.rename("iv")],axis=1).dropna()
    if len(al)<80: continue
    lsw=al["sw"].corr(al["iv"]); lgv=al["gv"].corr(al["iv"])
    d=al.diff().dropna()
    dsw=d["sw"].corr(d["iv"]); dgv=d["gv"].corr(d["iv"])
    gap_l=lgv-lsw; gap_d=dgv-dsw
    surv = "SI" if (np.sign(gap_l)==np.sign(gap_d) and abs(gap_d)>=0.10) else "no"
    P(f"{LAB[sw]+' vs '+LAB[gv]:22}{lsw:7.2f}/{lgv:<6.2f}{dsw:7.2f}/{dgv:<6.2f}{surv:>13}{len(al):5d}")
P("    formato: swap/governativo. 'sopravvive' = lo scarto ha lo stesso segno in variazioni")
P("    e vale almeno 0.10. Se no, la §6.6 va retrocessa a corollario sui livelli.")

P("")
P("PERCHE' QUESTI QUATTRO PANNELLI DECIDONO. L'obiezione piu' pericolosa al paper non e' il")
P("VRP -- quella e' chiusa -- ma che sigma_BE sia forma di curva e che la sua disconnessione")
P("dall'implicita sia il fatto noto della volatilita' non spannata. [1] e [2] misurano quanto")
P("i due oggetti siano spannati; [3] stabilisce se il fatto centrale sopravvive alla purga;")
P("[4] verifica se l'identificazione bandiera regge nell'unico spazio -- le variazioni -- in cui")
P("il paper dichiara di potersi difendere. Se [3] tiene, la letteratura USV diventa la cornice")
P("e la frase del paper diventa piu' forte: FONDAZIONE limits-to-arbitrage per la volatilita'")
P("non spannata, con il prezzo relativo misurato.")
save_txt("30_spanning.txt", L); print("\n".join(L))

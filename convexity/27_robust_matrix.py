"""27 - LA MATRICE DI ROBUSTEZZA DEL RESIDUO: terne di nodi x celle della superficie.

IL BUCO CHE QUESTO SCRIPT CHIUDE. Tutta la sequenza VRP (21-26) gira su UNA sola geometria
(fly 2/10/30) contro UNA sola cella (3Mx10Y). Ma il residuo R = sigma_BE - E[RV] e' un LIVELLO,
e i livelli di sigma_BE variano molto fra terne: dal pannello 1 del 19, in EUR la 2/5/30 da' 65
e la 2/10/20 da' 101, un intervallo di 36bp -- dello stesso ordine del residuo (38bp) che il
paper rivendica. Non e' ovvio che l'euro sopravviva a una geometria diversa, e finche' non lo
si testa l'affermazione poggia su una scelta di costruzione non giustificata.

IL VINCOLO DI APPAIAMENTO, e perche' NON si testano tutte le terne contro tutte le celle.
Cambiando terna cambia il tratto di curva misurato: la 2/5/30 pesa il segmento corto, la
5/10/30 il lungo. Ma sigma_IV e' ancorata al suo TENOR. Confrontare una fly centrata sul 5Y con
una swaption a coda 10Y misura il disallineamento, non la robustezza. Quindi:
  - terne con CORPO 10Y  (2/10/30, 2/10/20, 5/10/30, 3/10/30)  ->  celle con TENOR 10Y
  - terna  con CORPO  5Y (2/5/30, 2/5/10)                      ->  celle con TENOR  5Y
  - terna  con CORPO 30Y (10/30/50 se disponibile)             ->  celle con TENOR 30Y
Il pannello 3 sfrutta questo come TEST POSITIVO: se il residuo e' economia e non artefatto,
deve seguire il corpo della fly quando si sposta il tenor della swaption in modo appaiato.

PANNELLI
 [1] Matrice principale: terne a corpo 10Y  x  celle {3Mx10Y, 6Mx10Y, 1Yx10Y, 2Yx10Y}.
     Per ciascuna cella: residuo R medio [NW t] per i quattro mercati.
 [2] Sintesi: in quante combinazioni l'euro resta positivo e significativo; in quante gli altri
     tre restano indistinguibili da zero. E' la tabella di robustezza da mettere nel paper.
 [3] Appaiamento corpo-tenor: fly a corpo 5Y contro tenor 5Y, corpo 30Y contro tenor 30Y.
     Se il residuo euro c'e' anche li', non e' un fenomeno del solo decennale.
 [4] Il caso limite: fly a corpo 10Y contro tenor 2Y (deliberatamente DISALLINEATA). Serve da
     controllo negativo: qui il residuo NON deve essere interpretabile.

NB: il motore qui e' quello semplificato del 19 (roll lineare fra nodi della terna), non il
motore di 02. I LIVELLI non sono quindi confrontabili con quelli del 21; cio' che conta e' se
il SEGNO e la SIGNIFICATIVITA' del residuo sono stabili al variare della geometria. Prima della
submission il pannello va rigenerato chiamando il motore di 02 per ciascuna terna.

Output: results/27_robust_matrix.txt
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import save_txt, load_legs_mid_all, load_vols

mid = load_legs_mid_all(); IV = load_vols()
L=[]; P=L.append
P("=== 27 MATRICE DI ROBUSTEZZA: terne di nodi x celle della superficie ===")

H_M = 3
FAM = {"USDswap":"USOSFR","EUR":"EUSA","GBP":"BPSWS","JPY":"JYSO"}
LAB = {"USDswap":"USD swap","EUR":"EUR","GBP":"GBP","JPY":"JPY"}
MK4 = list(FAM)

def nw(x, lag=6):
    x=pd.Series(x).dropna().values; n=len(x)
    if n<24: return np.nan
    e=x-x.mean(); s=e@e/n
    for l in range(1,lag+1): s+=2*(1-l/(lag+1))*(e[l:]@e[:-l])/n
    return x.mean()/np.sqrt(s/n)

def sigbe(fam, taus):
    """sigma_BE mensile per una terna, motore semplificato (come 19)."""
    t1,t2,t3 = taus
    legs=[f"{fam}{t}" for t in taus]
    if any(l not in mid for l in legs): return None
    w1=(t3-t2)/(t3-t1); w3=(t2-t1)/(t3-t1)
    C = w1*t1**2 + w3*t3**2 - t2**2
    if C <= 0: return None
    y=pd.DataFrame({t: mid[f"{fam}{t}"]/100.0 for t in taus}).dropna()
    if len(y)<500: return None
    dt=1/12.0
    def roll(t):
        lo=max([k for k in taus if k<t], default=None)
        if lo is None:
            hi=min([k for k in taus if k>t]); return (y[hi]-y[t])/(hi-t)
        return (y[t]-y[lo])/(t-lo)
    theta=0.0
    for t,w in [(t1,w1),(t2,-1.0),(t3,w3)]:
        theta = theta + w*(-t)*(-roll(t))*dt
    s2=(-2.0*theta)/(C*dt)
    return (np.sign(s2)*np.sqrt(np.abs(s2))*1e4).resample("ME").last()

def rvfwd(m):
    dy=(mid[f"{FAM[m]}10"]/100.0).diff()
    return (dy.rolling(63).std()*np.sqrt(252)*1e4).resample("ME").last().shift(-H_M)

def ivcell(m, exp, ten):
    s=IV.get((IVMAP.get(m,""), exp, ten, "NORM"))
    return None if s is None else s.resample("ME").last()

def parts(m, taus, exp, ten):
    """restituisce (R, W, VRP): R dipende SOLO dalla terna; W e VRP dipendono dalla cella."""
    be=sigbe(FAM[m], taus); iv=ivcell(m, exp, ten)
    if be is None or iv is None: return None
    al=pd.concat([be.rename("be"), rvfwd(m).rename("rv"), iv.rename("iv")], axis=1).dropna()
    if len(al)<60: return None
    return al["be"]-al["rv"], al["be"]-al["iv"], al["iv"]-al["rv"]

def resid(m, taus, exp, ten):
    p=parts(m, taus, exp, ten)
    return None if p is None else p[0]

# ---------------------------------------------------------------- [1] matrice principale
TRI10 = {"2/10/30":(2,10,30), "2/10/20":(2,10,20), "5/10/30":(5,10,30), "3/10/30":(3,10,30)}
CELLS10 = [("3M","10Y"), ("6M","10Y"), ("1Y","10Y"), ("2Y","10Y")]
P("")
P("[1a] IL RESIDUO R DIPENDE SOLO DALLA TERNA (sigma_IV non entra in R)")
P("     Nota di disegno: R = sigma_BE - E[RV] non contiene sigma_IV, quindi la cella della")
P("     swaption entra solo nell'allineamento del campione. Le colonne 'campione' distinguono")
P("     le due finestre disponibili (celle 3M/6M vs 1Y/2Y).")
CNT={m:{"pos_sig":0,"zero":0,"tot":0} for m in MK4}
for exp,ten in [("3M","10Y"),("1Y","10Y")]:
    P("")
    P(f"   campione allineato alla cella {exp}x{ten}")
    P(f"   {'terna':10}" + "".join(f"{LAB[m]:>16}" for m in MK4))
    for nm,taus in TRI10.items():
        row=""
        for m in MK4:
            r=resid(m,taus,exp,ten)
            if r is None: row+=f"{'--':>16}"; continue
            t=nw(r); row+=f"{r.mean():10.0f}[{t:4.1f}]"
            CNT[m]["tot"]+=1
            if r.mean()>0 and t is not None and not np.isnan(t) and t>=2.0: CNT[m]["pos_sig"]+=1
            if t is not None and not np.isnan(t) and abs(t)<2.0: CNT[m]["zero"]+=1
        P(f"   {nm:10}{row}")

P("")
P("[1b] IL CUNEO W = sigma_BE - sigma_IV, QUESTO SI' DIPENDE DALLA CELLA")
P("     E' l'oggetto su cui la dimensione 'celle' e' informativa: se W cambia molto fra")
P("     scadenze dell'opzione a parita' di terna, il confronto e' sensibile all'appaiamento.")
for nm,taus in TRI10.items():
    P("")
    P(f"   terna {nm}")
    P(f"   {'cella':10}" + "".join(f"{LAB[m]:>16}" for m in MK4))
    for exp,ten in CELLS10:
        row=""
        for m in MK4:
            p=parts(m,taus,exp,ten)
            row += f"{p[1].mean():10.0f}[{nw(p[1]):4.1f}]" if p is not None else f"{'--':>16}"
        P(f"   {exp+'x'+ten:10}{row}")

# ---------------------------------------------------------------- [2] sintesi
P("")
P("[2] SINTESI (8 combinazioni: 4 terne x 2 campioni) -- sul RESIDUO")
P(f"{'mercato':10}{'positivo e sig. (t>=2)':>26}{'indistinguibile da 0':>24}{'totale':>9}")
for m in MK4:
    c=CNT[m]
    P(f"{LAB[m]:10}{c['pos_sig']:>18}/{c['tot']:<7}{c['zero']:>16}/{c['tot']:<7}{c['tot']:>9}")
P("    LETTURA. Se l'euro e' positivo e significativo in tutte o quasi tutte le combinazioni,")
P("    il risultato di livello e' blindato e questa e' la tabella di robustezza del paper. Se")
P("    lo e' solo in alcune, il perimetro dell'affermazione va ristretto a quelle e dichiarato.")
P("    Simmetricamente, i tre nulli devono restare nulli: se qualcuno diventa significativo in")
P("    una geometria diversa, il 'nulla' era anch'esso una scelta di costruzione.")

# ---------------------------------------------------------------- [3] appaiamento corpo-tenor
P("")
P("[3] APPAIAMENTO CORPO-TENOR: la fly segue il tenor della swaption?")
P("    corpo 5Y -> cella 3Mx5Y  |  corpo 10Y -> 3Mx10Y  |  corpo 30Y -> 3Mx30Y")
PAIRED = [("2/5/30",(2,5,30),"5Y"), ("2/5/10",(2,5,10),"5Y"),
          ("2/10/30",(2,10,30),"10Y"),
          ("10/30/50",(10,30,50),"30Y"), ("5/30/50",(5,30,50),"30Y")]
P(f"   {'terna':11}{'tenor':7}" + "".join(f"{LAB[m]:>16}" for m in MK4))
for nm, taus, ten in PAIRED:
    row=""
    for m in MK4:
        r=resid(m, taus, "3M", ten)
        row += f"{r.mean():10.0f}[{nw(r):4.1f}]" if r is not None else f"{'--':>16}"
    P(f"   {nm:11}{ten:7}{row}")
P("    Se il residuo euro c'e' anche a corpo 5Y e 30Y contro i rispettivi tenor, non e' un")
P("    fenomeno del solo decennale ma una proprieta' della curva euro nel suo complesso.")

# ---------------------------------------------------------------- [4] controllo negativo
P("")
P("[4] CONTROLLO NEGATIVO, sul CUNEO W (l'unico oggetto su cui ha senso)")
P("    fly a corpo 10Y contro tenor 2Y: oggetti non appaiati. W deve differire nettamente")
P("    dal caso appaiato 10Y; se non differisce, l'appaiamento non conta e va detto.")
P(f"   {'terna':11}{'tenor':7}" + "".join(f"{LAB[m]:>16}" for m in MK4))
for nm,taus in list(TRI10.items())[:2]:
    for ten in ["10Y","2Y"]:
        row=""
        for m in MK4:
            p=parts(m,taus,"3M",ten)
            row += f"{p[1].mean():10.0f}[{nw(p[1]):4.1f}]" if p is not None else f"{'--':>16}"
        P(f"   {nm if ten=='10Y' else '':11}{ten:7}{row}")
    P("")

P("")
P("PROMEMORIA PER LA SUBMISSION. Il motore usato qui e' quello semplificato (roll lineare fra")
P("nodi della terna): i LIVELLI non sono confrontabili con il 21, che usa il motore di 02. Cio'")
P("che questo pannello stabilisce e' la STABILITA' DEL SEGNO E DELLA SIGNIFICATIVITA' al variare")
P("della geometria. Prima della submission va rigenerato invocando il motore di 02 su ciascuna")
P("terna, cosi' che i livelli siano quelli del paper.")
save_txt("27_robust_matrix.txt", L); print("\n".join(L))

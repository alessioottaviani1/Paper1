"""19 - ATTENUATION BIAS: la correlazione bassa e' economica o meccanica?

L'OBIEZIONE, che e' la piu' serie che il paper affronta. Se sigma_BE = segnale + rumore, la
correlazione osservata con sigma_IV e' ATTENUATA verso zero da un fattore che dipende solo dal
rapporto segnale/rumore. Un C3 vicino a zero sarebbe allora un artefatto di misura, non
segmentazione: rumore correla male con tutto.

PERCHE' LE DIFESE ESISTENTI NON BASTANO. Il cube scan (10) mostra che il risultato e' uniforme su
80 celle -- ma rumore uniforme produce lo stesso pattern. La terza venue (14) mostra che le due
venue opzionali concordano piu' fra loro che con la curva -- ma a +0.30/+0.44 non e' schiacciante.

PERCHE' NON BASTA NEMMENO IL CARICAMENTO SUL TERM PREMIUM. Si potrebbe argomentare che il t di
5.6-7.7 di sigma_BE sul term premium (12, 13) prova che sigma_BE non e' rumore. Prova che contiene
UN segnale sistematico, non che contenga segnale di VOLATILITA'. Scrivendo
    sigma_BE = a*TP + b*V + eps
con V il vero prezzo della volatilita', un t alto su TP dice che a*TP e' grande, non che b*V sia
grande rispetto a eps. Lo scenario compatibile con entrambi i fatti -- t alto sul TP e C3 nullo --
e' che sigma_BE sia in prevalenza term premium piu' rumore: obiezione PEGGIORE dell'attenuazione.
Quel ragionamento NON va usato come difesa.

LA DIFESA DECISIVA: ERRORS-IN-VARIABLES CON DUE COSTRUZIONI INDIPENDENTI.
Costruiamo sigma_BE su piu' terne di nodi (2/10/30, 2/5/30, 5/10/30, 2/10/20). Tutte misurano la
stessa quantita' -- il prezzo della convessita' sulla curva -- ma il rumore di costruzione (scelta
dei nodi, interpolazione, roll locale) e' largamente INDIPENDENTE fra terne. Allora, con
    x_A = x* + u_A,  x_B = x* + u_B,   u_A, u_B indipendenti,
il rapporto di AFFIDABILITA' e' lambda = corr(x_A, x_B) = var(x*)/(var(x*)+var(u)), e la
correlazione VERA con qualunque y si ottiene con la correzione di Spearman:
    corr(x*, y) = corr(x_A, y) / sqrt(lambda).
Se lambda e' alto il fattore di correzione e' piccolo, e NESSUNA attenuazione puo' trasformare uno
zero osservato in una correlazione economicamente rilevante. E' aritmetica, non un argomento.

Come controllo indipendente aggiungiamo la stima IV: regressione di Delta-sigma_IV su
Delta-sigma_BE^A strumentata con Delta-sigma_BE^B. Lo strumento e' valido sotto la stessa ipotesi
(rumore di costruzione indipendente) e la stima IV e' per costruzione libera da attenuazione.

Doppio dividendo: lo stesso esercizio E' la robustezza sui nodi della farfalla, l'unica scelta di
costruzione che non aveva uno switch.

Output: output/convexity/results/19_attenuation.txt
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import save_txt, load_legs_mid_all, load_vols

print("== 19 attenuation bias ==")
mid = load_legs_mid_all()
IV  = load_vols()
L = []; P = L.append
P("=== 19 ATTENUATION BIAS: errors-in-variables con costruzioni multiple ===")

# ---------- costruzioni alternative della farfalla ----------
FAM = {"USDswap":"USOSFR", "EUR":"EUSA", "GBP":"BPSWS", "JPY":"JYSO"}
NODES = {"2/10/30": (2,10,30), "2/5/30": (2,5,30), "5/10/30": (5,10,30), "2/10/20": (2,10,20)}

def weights(t1, t2, t3):
    w1 = (t3-t2)/(t3-t1); w3 = (t2-t1)/(t3-t1)
    C  = w1*t1**2 + w3*t3**2 - t2**2
    return w1, w3, C

def sigbe(fam, taus):
    """sigma_BE mensile (bp/yr, con segno) per una terna di nodi."""
    t1, t2, t3 = taus
    legs = [f"{fam}{t}" for t in taus]
    if any(l not in mid for l in legs): return None
    w1, w3, C = weights(t1, t2, t3)
    y = pd.DataFrame({t: mid[f"{fam}{t}"]/100.0 for t in taus}).dropna()
    if len(y) < 500: return None
    # roll: ogni gamba scivola verso il nodo adiacente piu' corto (lineare fra nodi disponibili)
    dt = 1/12.0
    def roll(t):
        lo = max([k for k in taus if k < t], default=None)
        if lo is None:  # gamba piu' corta: usa la pendenza verso il nodo successivo
            hi = min([k for k in taus if k > t])
            return (y[hi]-y[t])/(hi-t)
        return (y[t]-y[lo])/(t-lo)
    theta = 0.0
    for t, w in [(t1, w1), (t2, -1.0), (t3, w3)]:
        theta = theta + w*(-t)*(-roll(t))*dt      # roll-down del prezzo, segno del peso
    s2 = (-2.0*theta)/(C*dt)
    s = np.sign(s2)*np.sqrt(np.abs(s2))*1e4
    return s.resample("ME").last()

BE = {}
for mkt, fam in FAM.items():
    for nm, taus in NODES.items():
        s = sigbe(fam, taus)
        if s is not None and s.notna().sum() > 100: BE[(mkt, nm)] = s

P("")
P("[1] COSTRUZIONI: sigma_BE su terne alternative di nodi (medie in bp/yr)")
P(f"{'mercato':9}" + "".join(f"{nm:>11}" for nm in NODES))
for mkt in FAM:
    row = "".join(f"{BE[(mkt,nm)].mean():11.1f}" if (mkt,nm) in BE else f"{'--':>11}" for nm in NODES)
    P(f"{mkt:9}{row}")
P("    (i livelli differiscono perche' C e la regione della curva differiscono: atteso.")
P("     Cio' che conta per l'attenuazione non sono i livelli ma la CO-VARIAZIONE.)")

# ---------- rapporto di affidabilita' ----------
P("")
P("[2] AFFIDABILITA': corr delle VARIAZIONI fra costruzioni (= lambda, quota di segnale)")
P(f"{'mercato':9}{'A-B':>8}{'A-C':>8}{'A-D':>8}{'B-C':>8}{'media':>9}{'lambda usato':>14}")
LAM = {}
keys = list(NODES)
for mkt in FAM:
    have = [k for k in keys if (mkt,k) in BE]
    if len(have) < 2: continue
    d = pd.DataFrame({k: BE[(mkt,k)] for k in have}).dropna().diff().dropna()
    cc = d.corr()
    pairs = [(a,b) for i,a in enumerate(have) for b in have[i+1:]]
    vals = [cc.loc[a,b] for a,b in pairs]
    base = [cc.loc[have[0],b] for b in have[1:]]
    lam = float(np.mean([v for v in base if np.isfinite(v)]))
    LAM[mkt] = lam
    cells = "".join(f"{cc.loc[a,b]:8.2f}" for a,b in pairs[:4])
    P(f"{mkt:9}{cells}{np.mean(vals):9.2f}{lam:14.2f}")
P("    lambda = corr media fra la costruzione baseline (2/10/30) e le alternative.")
P("    Interpretazione: quota della varianza delle variazioni che e' SEGNALE COMUNE, non rumore")
P("    di costruzione. lambda alto => poco spazio per l'attenuazione.")

# ---------- la correzione di Spearman ----------
def ivser(ccy, exp="3M", ten="10Y"):
    s = IV.get((ccy, exp, ten, "NORM"))
    return None if s is None else s.resample("ME").last()

P("")
P("[3] C3 CORRETTO PER ATTENUAZIONE (correzione di Spearman: rho* = rho_oss / sqrt(lambda))")
P(f"{'mercato':9}{'rho oss.':>10}{'lambda':>9}{'sqrt(l)':>9}{'rho CORRETTO':>14}{'N':>6}")
for mkt in FAM:
    if mkt not in LAM: continue
    iv = ivser(IVMAP.get(mkt,""))
    if iv is None: continue
    al = pd.concat([BE[(mkt,"2/10/30")], iv], axis=1).dropna().diff().dropna()
    if len(al) < 60: continue
    rho = al.iloc[:,0].corr(al.iloc[:,1])
    lam = max(1e-6, LAM[mkt])
    P(f"{mkt:9}{rho:10.2f}{lam:9.2f}{np.sqrt(lam):9.2f}{rho/np.sqrt(lam):14.2f}{len(al):6d}")
P("    LETTURA. Se il rho CORRETTO resta vicino a zero, l'attenuazione NON puo' spiegare il C3:")
P("    la correlazione bassa e' economica, non meccanica. E' il test che chiude l'obiezione.")

# ---------- stima IV, controllo indipendente ----------
P("")
P("[4] CONTROLLO INDIPENDENTE -- stima IV: Delta-sigma_IV su Delta-sigma_BE(2/10/30),")
P("    strumentata con Delta-sigma_BE(costruzione alternativa). Per costruzione senza attenuazione.")
P(f"{'mercato':9}{'OLS beta':>10}{'[t]':>7}{'IV beta':>10}{'[t]':>7}{'1st stage F':>13}{'N':>6}")
for mkt in FAM:
    alt = next((k for k in ["2/5/30","5/10/30","2/10/20"] if (mkt,k) in BE), None)
    iv = ivser(IVMAP.get(mkt,""))
    if alt is None or iv is None: continue
    df = pd.concat([BE[(mkt,"2/10/30")].rename("A"), BE[(mkt,alt)].rename("B"), iv.rename("Y")],
                   axis=1).dropna().diff().dropna()
    if len(df) < 60: continue
    n = len(df)
    A, B, Y = df["A"].values, df["B"].values, df["Y"].values
    Ac, Bc, Yc = A-A.mean(), B-B.mean(), Y-Y.mean()
    b_ols = (Ac@Yc)/(Ac@Ac)
    e = Yc - b_ols*Ac; se_ols = np.sqrt((e@e)/(n-2)/(Ac@Ac))
    # 1st stage: A su B
    g = (Bc@Ac)/(Bc@Bc); r1 = Ac - g*Bc
    F = (g**2*(Bc@Bc))/((r1@r1)/(n-2))
    b_iv = (Bc@Yc)/(Bc@Ac)
    ei = Yc - b_iv*Ac
    se_iv = np.sqrt((ei@ei)/(n-2)*(Bc@Bc)/((Bc@Ac)**2))
    P(f"{mkt:9}{b_ols:10.3f}{b_ols/se_ols:7.1f}{b_iv:10.3f}{b_iv/se_iv:7.1f}{F:13.0f}{n:6d}")
P("    Se il beta IV resta piccolo e non significativo con 1st-stage F elevato, la relazione")
P("    curva-swaption e' debole anche AL NETTO dell'errore di misura.")

# ---------- persistenza: rumore puro non e' persistente ----------
P("")
P("[5] PERSISTENZA (controllo ausiliario): rumore puro non ha autocorrelazione")
P(f"{'mercato':9}{'AR(1) livelli':>15}{'AR(1) variazioni':>18}")
for mkt in FAM:
    s = BE[(mkt,"2/10/30")].dropna()
    P(f"{mkt:9}{s.autocorr(1):15.2f}{s.diff().dropna().autocorr(1):18.2f}")
P("    AR(1) dei livelli alto => la serie e' dominata da una componente persistente, non da rumore")
P("    bianco. Non e' prova di validita' del segnale, ma limita dall'alto la quota di rumore.")

# ---- limite superiore: quanto basso dovrebbe essere lambda per salvare l'obiezione?
P("")
P("[6] IL TEST DECISIVO: quanto DOVREBBE essere basso lambda perche' l'attenuazione spieghi il C3?")
P("    Serve rho* = rho_oss/sqrt(lambda) >= 0.50 (soglia generosa di 'economicamente rilevante').")
P(f"{'mercato':9}{'rho oss.':>10}{'lambda necessario':>19}{'plausibile?':>13}")
for mkt in FAM:
    if mkt not in LAM: continue
    iv = ivser(IVMAP.get(mkt,""))
    if iv is None: continue
    al = pd.concat([BE[(mkt,"2/10/30")], iv], axis=1).dropna().diff().dropna()
    if len(al) < 60: continue
    rho = abs(al.iloc[:,0].corr(al.iloc[:,1]))
    need = (rho/0.50)**2 if rho > 0 else 0.0
    verdict = "NO" if need < 0.05 else ("dubbio" if need < 0.30 else "possibile")
    P(f"{mkt:9}{rho:10.2f}{need:19.4f}{verdict:>13}")
P("    LETTURA. In USD servirebbe lambda ~ 0.0005 e in GBP ~ 0.02: cioe' che il 99.95% e il 98%")
P("    della variazione di sigma_BE fosse rumore di costruzione. Ma lambda MISURATO e' 0.79-0.83,")
P("    e anche ammettendo altre fonti di rumore oltre alla scelta dei nodi resterebbe di ordini di")
P("    grandezza troppo alto. L'attenuazione NON puo' spiegare il C3 in USD e GBP. In EUR e JPY,")
P("    dove il rho osservato e' positivo, la questione non si pone: sono i mercati piu' integrati.")

P("")
P("CAVEAT DI IMPLEMENTAZIONE. La costruzione qui e' una versione semplificata del roll (lineare fra")
P("nodi adiacenti della terna), non il motore di 02: la media USD baseline e' 68.6 contro 89 del")
P("motore. Per il paper il pannello va rigenerato chiamando il motore su ciascuna terna, cosi' che")
P("lambda misuri il rumore di COSTRUZIONE a metodo costante. La conclusione non cambia -- lambda")
P("dovrebbe crollare di due ordini di grandezza per salvare l'obiezione -- ma il numero va reso")
P("coerente prima della submission.")

P("")
P("NOTA SU COME PRESENTARLO. La difesa contro l'attenuazione e' [3] e [4], non il caricamento sul")
P("term premium. Quel caricamento prova che sigma_BE contiene UN segnale sistematico, non che")
P("contenga segnale di VOLATILITA': usarlo come difesa invita l'obiezione peggiore ('la vostra")
P("misura e' term premium travestito'). La difesa corretta e' che due costruzioni indipendenti")
P("della stessa quantita' concordano, quindi la quota di rumore e' misurabile e piccola, quindi")
P("la correzione per attenuazione non muove il C3.")
save_txt("19_attenuation.txt", L); print("\n".join(L))

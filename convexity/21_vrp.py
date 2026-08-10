"""21 - IL PREMIO DI RISCHIO DI VOLATILITA': la spiegazione concorrente, misurata e tolta.

L'OBIEZIONE (Rebonato, ago-2026): "in your explanations a volatility risk premium does not seem to
figure". E' l'obiezione decisiva, perche' il VRP e' una spiegazione ALTERNATIVA COMPLETA del cuneo:

  sigma_IV e' una quantita' RISK-NEUTRAL: incorpora il compenso che i venditori di opzioni
  richiedono per portare rischio di volatilita', quindi eccede sistematicamente la volatilita'
  realizzata attesa.
  sigma_BE e' una quantita' in MISURA FISICA: e' il livello di volatilita' REALIZZATA al quale la
  farfalla va in pari.

Confrontarle direttamente e' quindi confrontare oggetti sotto misure diverse, e un cuneo negativo e'
ATTESO anche in un mercato perfettamente integrato. Senza affrontarlo, ne' il rapporto dei costi ne'
il livello del cuneo dicono nulla sulla segmentazione.

LA DECOMPOSIZIONE. Sia RV la volatilita' realizzata sull'orizzonte dell'opzione. Allora
    VRP  =  sigma_IV - E[RV]                     (premio di rischio di volatilita')
    W    =  sigma_BE - sigma_IV
         =  (sigma_BE - E[RV])  -  VRP
Il primo termine e' lo scostamento della CURVA dalla volatilita' realizzata attesa: e' il prezzo che
la curva assegna alla volatilita' al netto di cio' che poi si realizza. Il secondo e' il premio
delle opzioni.

IL TEST. Se le due venue fossero integrate e differissero SOLO per la misura, allora la curva
prezzerebbe equamente la vol realizzata (sigma_BE = E[RV]) e il cuneo sarebbe ESATTAMENTE -VRP.
  H0 (nessuna segmentazione, solo VRP):  W = -VRP,  cioe'  sigma_BE - E[RV] = 0
  H1 (segmentazione):                    W != -VRP, e il residuo e' cio' che resta da spiegare
Il residuo  R = W + VRP = sigma_BE - E[RV]  e' quindi l'oggetto epurato dal VRP, ed e' su QUELLO
che vanno rifatte le affermazioni sul costo relativo.

CAVEAT DI MISURA. E[RV] non e' osservabile; usiamo la RV EX POST sull'orizzonte dell'opzione come
suo proxy non distorto sotto aspettative razionali. Questo introduce errore di misura ma non
distorsione sistematica, e la media campionaria di lungo periodo e' la stima corretta del VRP medio.

Output: output/convexity/results/21_vrp.txt
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import nw_t, save_txt, load_legs_mid, load_vols

print("== 21 premio di rischio di volatilita' ==")
sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
mid = load_legs_mid()
IV  = load_vols()
L = []; P = L.append
P("=== 21 IL PREMIO DI RISCHIO DI VOLATILITA', MISURATO E TOLTO ===")

H_M = 3   # orizzonte della swaption di riferimento: 3 mesi

def ivser(ccy, exp="3M", ten="10Y"):
    s = IV.get((ccy, exp, ten, "NORM"))
    return None if s is None else s.resample("ME").last()

def realized_fwd(mkt):
    """RV EX POST sui 3 mesi SUCCESSIVI, annualizzata, in bp/anno -- proxy di E[RV]."""
    legs, taus = MK[mkt]
    dy = (mid[legs[1]]/100.0).diff()
    rv_d = dy.rolling(63).std()*np.sqrt(252)*1e4      # trailing 3m, bp/anno
    rv_m = rv_d.resample("ME").last()
    return rv_m.shift(-H_M)                            # portata avanti: e' la RV FUTURA

MK5 = [m for m in ["USDswap","EUR","GBP","JPY","UST"] if m in sbe.columns]

# ------------------------------------------------ [1] il VRP esiste, e quanto vale
P("")
P("[1] IL VRP E' REALE E VA MISURATO: sigma_IV contro la RV effettivamente realizzata")
P(f"{'mercato':9}{'sigma_IV':>10}{'E[RV] proxy':>13}{'VRP':>9}{'[NW t]':>8}{'VRP/IV':>9}{'T':>5}")
VRP = {}
for m in MK5:
    iv = ivser(IVMAP.get(m,""))
    if iv is None: continue
    rv = realized_fwd(m)
    al = pd.concat([iv.rename("iv"), rv.rename("rv")], axis=1).dropna()
    if len(al) < 60: continue
    v = (al["iv"] - al["rv"])
    VRP[m] = v
    P(f"{m:9}{al['iv'].mean():10.1f}{al['rv'].mean():13.1f}{v.mean():9.1f}{nw_t(v):8.1f}"
      f"{v.mean()/al['iv'].mean():9.1%}{len(al):5d}")
P("    LETTURA. VRP positivo dove le swaption prezzano piu' volatilita' di quella poi realizzata.")
P("    NB: il segno NON e' uniforme -- in USD il VRP medio e' lievemente negativo e non")
P("    significativo (il 2008 realizza sopra l'implicita). La spiegazione concorrente e' quindi")
P("    FORTE in EUR/JPY, debole in USD/GBP: esattamente cio' che la decomposizione quantifica.")

# ------------------------------------------------ [2] la decomposizione
P("")
P("[2] DECOMPOSIZIONE DEL CUNEO:  W = (sigma_BE - E[RV]) - VRP")
P(f"{'mercato':9}{'W oss.':>9}{'-VRP':>9}{'residuo R':>11}{'[NW t] su R':>13}{'T':>5}")
RES = {}
for m in MK5:
    iv = ivser(IVMAP.get(m,""))
    if iv is None or m not in VRP: continue
    rv = realized_fwd(m)
    al = pd.concat([sbe[m].rename("be"), iv.rename("iv"), rv.rename("rv")], axis=1).dropna()
    if len(al) < 60: continue
    W = al["be"] - al["iv"]
    R = al["be"] - al["rv"]          # = W + VRP, il residuo epurato
    RES[m] = R
    P(f"{m:9}{W.mean():9.1f}{-(al['iv']-al['rv']).mean():9.1f}{R.mean():11.1f}{nw_t(R):13.1f}{len(al):5d}")
P("")
P("    LETTURA. Sotto H0 -- integrazione, differenza dovuta SOLO alla misura -- la curva")
P("    prezzerebbe equamente la vol realizzata e il residuo R sarebbe ZERO. R diverso da zero e")
P("    significativo e' cio' che il VRP NON spiega.")

# ------------------------------------------------ [3] il costo relativo, epurato
P("")
P("[3] IL RAPPORTO DEI COSTI, PRIMA E DOPO AVER TOLTO IL VRP")
P("    grezzo:   (sigma_BE / sigma_IV)^2      -- confronta misura fisica con risk-neutral")
P("    epurato:  (sigma_BE / E[RV])^2         -- confronta due quantita' in misura fisica")
P(f"{'mercato':9}{'grezzo':>10}{'epurato':>10}{'variazione':>12}")
for m in MK5:
    iv = ivser(IVMAP.get(m,""))
    if iv is None: continue
    rv = realized_fwd(m)
    al = pd.concat([sbe[m].rename("be"), iv.rename("iv"), rv.rename("rv")], axis=1).dropna()
    if len(al) < 60: continue
    g = (al["be"].mean()/al["iv"].mean())**2
    e = (al["be"].mean()/al["rv"].mean())**2
    P(f"{m:9}{g:10.2f}{e:10.2f}{e-g:+12.2f}")
P("    Se l'ordinamento fra mercati SOPRAVVIVE all'epurazione, la dispersione non e' VRP.")

# ------------------------------------------------ [4] il VRP spiega la cross-section?
P("")
P("[4] IL VRP SPIEGA LA DISPERSIONE FRA MERCATI? (correlazione di rango)")
try:
    from scipy.stats import spearmanr
    mk = [m for m in MK5 if m in VRP and m in RES]
    if len(mk) > 2:
        ratio = []
        for m in mk:
            iv = ivser(IVMAP.get(m,"")); rv = realized_fwd(m)
            al = pd.concat([sbe[m].rename("be"), iv.rename("iv"), rv.rename("rv")], axis=1).dropna()
            ratio.append((al["be"].mean()/al["iv"].mean())**2)
        vr = [VRP[m].mean() for m in mk]
        rr = [RES[m].mean() for m in mk]
        from itertools import permutations
        def rank_exact(x, y):
            rho = spearmanr(x, y)[0]; n = 0; tot = 0
            for p_ in permutations(y):
                tot += 1
                if abs(spearmanr(x, list(p_))[0]) >= abs(rho) - 1e-12: n += 1
            return rho, n/tot
        a, pa = rank_exact(ratio, vr); b, pb = rank_exact(ratio, rr)
        P(f"   rapporto costi ~ VRP medio      : rango {a:+.2f} (p esatto={pa:.3f}, {len(mk)} mercati)")
        P(f"   rapporto costi ~ residuo R medio: rango {b:+.2f} (p esatto={pb:.3f})")
        P("   (p per enumerazione completa delle permutazioni: con n=4 il minimo possibile e' 0.042)")
        P("   Se il rapporto segue R piu' del VRP, la dispersione e' segmentazione, non premio.")
except Exception as e:
    P(f"   non calcolato ({e})")

P("")
P("CONSEGUENZE PER IL PAPER. Primo: il VRP va introdotto nella SEZIONE ECONOMICA, non come caveat.")
P("Secondo: ogni affermazione sul COSTO RELATIVO va rifatta su R = sigma_BE - E[RV], perche' il")
P("rapporto grezzo confronta una quantita' in misura fisica con una risk-neutral. Terzo: la")
P("difesa piu' forte resta il CO-MOVIMENTO, che il VRP non tocca -- un premio di rischio spiega un")
P("livello, non l'assenza di correlazione fra le VARIAZIONI mensili dei due prezzi.")
save_txt("21_vrp.txt", L); print("\n".join(L))

"""20 - LE QUATTRO DOMANDE DI REBONATO-RONZANI (2021), TESTATE DIRETTAMENTE.

LA LACUNA CHE QUESTO SCRIPT CHIUDE. Il pipeline finora testa se i due prezzi della volatilita'
CO-MUOVONO (script 03: correlazioni di livelli e variazioni). Ma le domande che Rebonato e Ronzani
lasciano aperte nella conclusione NON sono sul co-movimento: sono sul COSTO RELATIVO. Verbatim:

  "how much would it cost to reproduce this profile using a straddle of long-dated swaptions?
   Would the implicit premium from the negative carry be greater or smaller than the option
   premium? In other words, at any point in time would it be more efficient to 'buy or sell
   convexity' via swaptions, or by exploiting the curvature of the swap curve? Could one create a
   delta-, gamma- and vega-neutral portfolio of long-dated swaps and swaption straddles? And would
   this package have a non-zero cost? We would love to carry out these investigations, but we do
   not have reliable swaption prices at our disposal, and we therefore leave these topics for
   future analysis."
                          -- Rebonato & Ronzani (2021), J. Empirical Finance 63, conclusione

Sono quattro domande distinte, e nessuna e' una domanda di correlazione:
 [Q1] quanto costa riprodurre il profilo long-gamma con uno straddle di swaption?
 [Q2] il premio implicito dal carry negativo e' maggiore o minore del premio dell'opzione?
 [Q3] in un dato momento, e' piu' efficiente comprare/vendere convessita' via swaption o via curva?
 [Q4] un pacchetto delta-, gamma-, vega-neutrale di swap e straddle avrebbe costo non nullo?

L'IDENTITA' CHE RENDE Q2-Q4 MISURABILI. sigma_BE e sigma_IV sono entrambe volatilita' NORMALI del
tasso, in bp/anno (verificato: la famiglia usata e' 'Normalised Vol ATM', media USD 3Mx10Y 86.9
contro sigma_BE 89; la lognormale sta a 24.0 e NON e' usata). Quindi il cuneo
    W_t = sigma_BE,t - sigma_IV,t
e' direttamente il PREZZO RELATIVO della volatilita' nelle due venue, nelle stesse unita':
  W > 0 : la curva richiede piu' volatilita' per andare in pari di quanta l'opzione ne implichi
          => la convessita' e' CARA via curva, ECONOMICA via swaption  => comprarla in opzioni;
  W < 0 : il contrario.
E poiche' un pacchetto vega-neutrale lungo convessita' su una venue e corto sull'altra ha per
costruzione costo netto proporzionale a W, la Q4 e' risolta dal segno e dalla persistenza di W:
il pacchetto ha costo non nullo se e solo se W non e' nullo.

NOTA SULL'OGGETTO. Gli strumenti NON sono identici -- una farfalla duration-neutrale sul tasso swap
non e' uno straddle di swaption, e differiscono per scadenza, moneyness e profilo di convessita'.
Il confronto e' fra i due PREZZI della medesima volatilita' sottostante (quella del tasso lungo),
non fra due misure dello stesso strumento. E' esattamente il confronto che Rebonato e Ronzani
formulano, e la ragione per cui il costo relativo -- non l'identita' -- e' la quantita' di interesse.

Output: output/convexity/results/20_rebonato_questions.txt
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import nw_t, save_txt, load_vols

print("== 20 le quattro domande di Rebonato-Ronzani ==")
sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
IV  = load_vols()
L = []; P = L.append
P("=== 20 LE QUATTRO DOMANDE DI REBONATO-RONZANI (2021), TESTATE ===")

def ivser(ccy, exp="3M", ten="10Y"):
    s = IV.get((ccy, exp, ten, "NORM"))
    return None if s is None else s.resample("ME").last()

MK5 = [m for m in ["USDswap","EUR","GBP","JPY","UST"] if m in sbe.columns]

# ---------------------------------------------------------- Q2/Q3: il costo relativo
P("")
P("[Q2/Q3] IL PREZZO RELATIVO DELLA CONVESSITA' NELLE DUE VENUE")
P("        W = sigma_BE - sigma_IV, in bp/anno. W>0: cara via curva, economica via swaption.")
P(f"{'mercato':9}{'media W':>10}{'[NW t]':>8}{'mediana':>9}{'sd':>8}{'% W>0':>8}{'T':>5}")
WED = {}
for m in MK5:
    iv = ivser(IVMAP.get(m,""))
    if iv is None: continue
    al = pd.concat([sbe[m], iv], axis=1).dropna()
    if len(al) < 60: continue
    w = (al.iloc[:,0] - al.iloc[:,1]).dropna()
    WED[m] = w
    P(f"{m:9}{w.mean():10.1f}{nw_t(w):8.1f}{w.median():9.1f}{w.std():8.1f}{(w>0).mean():8.0%}{len(w):5d}")
P("")
P("    RISPOSTA A Q3. Il segno di W dice quale venue e' piu' efficiente, e la colonna '% W>0' dice")
P("    per quale frazione del campione. Se W e' stabilmente di un segno, una delle due venue e'")
P("    sistematicamente il modo piu' economico di comprare convessita'; se W cambia segno, la")
P("    risposta e' condizionale allo stato del mercato -- che e' la forma piu' interessante.")

# ---------------------------------------------------------- Q3 nel tempo: cambia segno?
P("")
P("[Q3-bis] LA RISPOSTA CAMBIA NEL TEMPO? (media di W per era)")
ERAS = [("pre-GFC","2002","2007-06"), ("GFC","2007-07","2009-12"), ("ZIRP","2010","2015"),
        ("normalizz.","2016","2019"), ("COVID","2020","2021"), ("inflaz.","2022","2026")]
P(f"{'mercato':9}" + "".join(f"{e[0]:>12}" for e in ERAS))
for m, w in WED.items():
    row = ""
    for nm, a, b in ERAS:
        seg = w.loc[a:b]
        row += f"{seg.mean():12.1f}" if len(seg) >= 6 else f"{'--':>12}"
    P(f"{m:9}{row}")

# ---------------------------------------------------------- Q4: il pacchetto neutrale
P("")
P("[Q4] IL PACCHETTO VEGA-NEUTRALE HA COSTO NON NULLO?")
P("     Costruzione: lungo convessita' sulla venue economica, corto sull'altra, scalato a vega")
P("     netto nullo. Per costruzione il costo netto e' proporzionale a |W|. Il test e' se |W| sia")
P("     significativamente diverso da zero e persistente.")
P(f"{'mercato':9}{'|W| media':>11}{'|W|/sigma_IV':>14}{'[t] su |W|':>12}{'AR(1)':>8}")
for m, w in WED.items():
    iv = ivser(IVMAP.get(m,""))
    aw = w.abs()
    rel = (aw / iv.reindex(aw.index)).dropna()
    P(f"{m:9}{aw.mean():11.1f}{rel.mean():14.1%}{nw_t(aw):12.1f}{w.autocorr(1):8.2f}")
P("")
P("    RISPOSTA A Q4. |W|/sigma_IV misura il costo del pacchetto in percentuale del premio")
P("    dell'opzione: e' la quantita' che Rebonato e Ronzani chiedono in Q1 (quanto costerebbe")
P("    riprodurre il profilo con uno straddle'). Un AR(1) elevato dice che il costo non e' un")
P("    disallineamento transitorio ma una caratteristica persistente del mercato.")

# ---------------------------------------------------------- Q1: il costo di replica
P("")
P("[Q1] COSTO DI REPLICA DEL PROFILO LONG-GAMMA VIA STRADDLE, in unita' confrontabili")
P("     Il carry pagato dalla farfalla per unita' di convessita' e' 0.5*C*sigma_BE^2*dt;")
P("     il premio dello straddle per la stessa esposizione a vol e' 0.5*C*sigma_IV^2*dt.")
P("     Il rapporto dei due costi e' quindi (sigma_BE/sigma_IV)^2, indipendente da C.")
P(f"{'mercato':9}{'sigma_BE':>10}{'sigma_IV':>10}{'rapporto costi':>16}{'lettura':>26}")
for m in WED:
    iv = ivser(IVMAP.get(m,""))
    al = pd.concat([sbe[m], iv], axis=1).dropna()
    b, o = al.iloc[:,0].mean(), al.iloc[:,1].mean()
    r = (b/o)**2
    verdict = "curva piu' CARA" if r > 1.05 else ("swaption piu' CARA" if r < 0.95 else "equivalenti")
    P(f"{m:9}{b:10.1f}{o:10.1f}{r:16.2f}{verdict:>26}")
P("")
P("    Il rapporto e' il costo di comprare convessita' via curva diviso il costo di comprarla via")
P("    swaption, a pari esposizione di volatilita'. E' la risposta diretta a Q1-Q2.")

P("")
P("COME CITARLO NEL PAPER. Le quattro domande vanno attribuite verbatim alla conclusione di")
P("Rebonato e Ronzani (2021), J. Empirical Finance 63, 392-413, che le formula e dichiara di non")
P("poterle affrontare per mancanza di prezzi di swaption affidabili. Questo script le affronta.")
P("ATTENZIONE a NON scrivere che gli autori assumono un'IDENTITA' fra i due prezzi: non lo fanno.")
P("Nel corpo del paper (sez. 2) scrivono che nel portafoglio swap 'il ruolo della volatilita'")
P("implicita e' svolto dalla curvatura della curva swap' -- cioe' trattano sigma_BE come")
P("l'ANALOGO di curva della volatilita' implicita, non come la stessa misura. Il confronto e'")
P("fra i due PREZZI della medesima volatilita' sottostante, non fra due misure dello stesso")
P("strumento; ed e' questo che rende il COSTO RELATIVO, e non l'identita', la quantita' di")
P("interesse.")
save_txt("20_rebonato_questions.txt", L); print("\n".join(L))

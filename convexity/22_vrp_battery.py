"""22 - LA BATTERIA VRP: tutti i test da fare PRIMA di rispondere a Rebonato.

Il 21 ha stabilito che in 3 mercati su 4 il cuneo e' interamente VRP e che solo l'euro sopravvive
(R=37.8, t=3.1). Prima di scriverlo al supervisore, ogni anello di quella catena va stressato.
Questo script fa OGNI test che possiamo fare con i dati in casa, in otto pannelli:

 [1] ROBUSTEZZA DELLA RV -- la realized e' calcolata su tassi CMS (par a scadenza costante):
     la variazione giornaliera dello spot approssima quella del forward sottostante la swaption.
     Testiamo: finestre 21/63/126gg; variazioni SETTIMANALI (immuni a microstruttura);
     e la RV del TASSO FORWARD 3M->10Y costruito per interpolazione dalla curva.
 [2] ERE -- dove vive il VRP e dove vive il residuo R: il VRP USD negativo e' il 2008?
     Il residuo EUR e' la crisi sovrana 2011-12 o e' diffuso?
 [3] EX-CRISI -- R con tutte le finestre di crisi rimosse: sopravvive l'euro?
 [4] MEDIANE E TRIMMED -- R e' una media trascinata dalle code?
 [5] MINCER-ZARNOWITZ -- quale venue PREVEDE meglio la vol realizzata futura: sigma_BE, sigma_IV,
     o entrambe? Se sigma_BE non aggiunge nulla a sigma_IV, la curva non contiene informazione
     di volatilita' (coerente con RR2021); se aggiunge, ne contiene.
 [6] IL VRP TEMPO-VARIANTE E IL CO-MOVIMENTO -- il test che decide se la difesa "il co-movimento
     e' intatto" e' scrivibile. Decomponiamo in tempo reale sigma_IV = E_hat[RV] + VRP_hat
     (trailing RV come proxy della componente attesa) e chiediamo: le variazioni di sigma_BE
     seguono la componente FONDAMENTALE (Delta E_hat), il PREMIO (Delta VRP_hat), o nessuna?
     - se sigma_BE non segue nemmeno la componente attesa => segmentazione anche al netto del VRP
     - se segue E_hat ma non VRP_hat => le venue condividono le aspettative ma non il premio
       (storia diversa e PIU' pulita: "one volatility, two prices OF RISK")
 [7] EURO CONTRO BUND -- il residuo euro e' della curva SWAP o della valuta? Stessa
     decomposizione sul sigma_BE del Bund contro la stessa superficie euro: se il Bund non ha
     residuo, il canale e' collaterale/clearing (specifico swap), non "l'Europa".
 [8] PERMUTAZIONE ESATTA -- il rango costi~residuo con n=4: enumerazione completa.

Output: output/convexity/results/22_vrp_battery.txt
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from itertools import permutations
from config import *
from utils import save_txt, load_legs_mid_all, load_vols

print("== 22 batteria VRP ==")
sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
mid = load_legs_mid_all()
try:
    from utils import load_bbg_sheet
    _g = load_bbg_sheet(SHEET_GOVT)
    for (t, j), v in _g.items():
        mid.setdefault(t.replace(" Curncy","").replace(" Index","").strip(), v)
except Exception as _e:
    print(f"[22] foglio govt non letto ({_e}): pannello 7 usera' i proxy")
IV  = load_vols()
L = []; P = L.append
P("=== 22 LA BATTERIA VRP: ogni test prima della risposta ===")

H_M = 3
MK4 = [m for m in ["USDswap","EUR","GBP","JPY"] if m in sbe.columns]
FAM = {"USDswap":"USOSFR", "EUR":"EUSA", "GBP":"BPSWS", "JPY":"JYSO"}
CRISES = [("2008-09-01","2009-06-30"),("2011-07-01","2012-08-31"),
          ("2020-02-01","2020-05-31"),("2022-09-01","2022-12-31")]
ERAS = [("pre-GFC","2002","2007-06"),("GFC","2007-07","2009-12"),("ZIRP","2010","2015"),
        ("normal.","2016","2019"),("COVID","2020","2021"),("inflaz.","2022","2026")]

def nw(x, lag=6):
    x = pd.Series(x).dropna().values
    n = len(x)
    if n < 24: return np.nan
    m = x.mean(); e = x - m
    s = e @ e / n
    for l in range(1, lag+1):
        w = 1 - l/(lag+1); s += 2*w*(e[l:] @ e[:-l])/n
    return m/np.sqrt(s/n)

def ivser(mkt, exp="3M"):
    s = IV.get((IVMAP.get(mkt,""), exp, "10Y", "NORM"))
    return None if s is None else s.resample("ME").last()

def daily10(mkt):
    return (mid[f"{FAM[mkt]}10"]/100.0)

def rv_from(dy_daily, win):
    return (dy_daily.diff().rolling(win).std()*np.sqrt(252)*1e4).resample("ME").last()

def rv_fwd(mkt, win=63, weekly=False, use_forward=False):
    """RV futura sull'orizzonte dell'opzione, varianti di costruzione."""
    if use_forward:
        # forward 3M->10Y per interpolazione lineare della curva sui nodi disponibili
        fam = FAM[mkt]
        nodes = sorted(int(k[len(fam):]) for k in mid if k.startswith(fam) and k[len(fam):].isdigit())
        if not nodes: return None
        y = {t: mid[f"{fam}{t}"]/100.0 for t in nodes}
        def interp(tau):
            lo = max([t for t in nodes if t <= tau], default=nodes[0])
            hi = min([t for t in nodes if t >= tau], default=nodes[-1])
            if lo == hi: return y[lo]
            wl = (hi-tau)/(hi-lo)
            return wl*y[lo] + (1-wl)*y[hi]
        y025  = y[nodes[0]]                     # flat sotto il primo nodo
        y1025 = interp(10.25)
        F = (10.25*y1025 - 0.25*y025)/10.0
        base = F
    else:
        base = daily10(mkt)
    if weekly:
        w = base.resample("W-FRI").last()
        rv = (w.diff().rolling(13).std()*np.sqrt(52)*1e4).resample("ME").last()
    else:
        rv = rv_from(base, win)
    return rv.shift(-H_M)

def decomp(mkt, rv):
    iv = ivser(mkt)
    if iv is None or rv is None: return None
    al = pd.concat([sbe[mkt].rename("be"), iv.rename("iv"), rv.rename("rv")], axis=1).dropna()
    if len(al) < 60: return None
    return al

# ---------------------------------------------------------------- [1] robustezza RV
P("")
P("[1] ROBUSTEZZA DELLA COSTRUZIONE DELLA RV (VRP medio | residuo R medio [NW t, lag=6])")
P(f"{'mercato':9}{'base 63g':>16}{'21g':>16}{'126g':>16}{'settimanale':>16}{'FORWARD 3Mx10Y':>17}")
variants = [("base 63g", dict(win=63)), ("21g", dict(win=21)), ("126g", dict(win=126)),
            ("settimanale", dict(weekly=True)), ("FORWARD", dict(use_forward=True))]
for m in MK4:
    row = f"{m:9}"
    for nm, kw in variants:
        al = decomp(m, rv_fwd(m, **kw))
        if al is None: row += f"{'--':>16}"; continue
        v = (al["iv"]-al["rv"]).mean(); r = (al["be"]-al["rv"])
        row += f"{v:6.1f}|{r.mean():5.1f}[{nw(r):4.1f}]"
    P(row)
P("    Se il residuo R cambia poco fra colonne -- incluso il FORWARD, che e' il sottostante")
P("    corretto della swaption -- la conclusione non dipende dalla convenzione CMS/spot.")

# ---------------------------------------------------------------- [2] ere
P("")
P("[2] DOVE VIVONO VRP E RESIDUO (medie per era; base 63g)")
P(f"{'mercato':9}" + "".join(f"{e[0]:>15}" for e in ERAS))
for m in MK4:
    al = decomp(m, rv_fwd(m))
    if al is None: continue
    v = al["iv"]-al["rv"]; r = al["be"]-al["rv"]
    row_v = "".join((f"{v.loc[a:b].mean():7.1f}|{r.loc[a:b].mean():6.1f}" if len(v.loc[a:b])>=6 else f"{'--':>15}") for _,a,b in ERAS)
    P(f"{m:9}{row_v}   (VRP|R)")
P("    Domande a cui risponde: il VRP USD negativo e' solo il 2008? il residuo EUR e' solo")
P("    la crisi sovrana o e' distribuito? il VRP JPY e' un artefatto del periodo YCC?")

# ---------------------------------------------------------------- [3] ex-crisi
P("")
P("[3] IL RESIDUO SENZA LE CRISI (tutte e 4 le finestre rimosse)")
P(f"{'mercato':9}{'R pieno':>10}{'[t]':>7}{'R ex-crisi':>12}{'[t]':>7}{'T ex':>6}")
for m in MK4:
    al = decomp(m, rv_fwd(m))
    if al is None: continue
    r = al["be"]-al["rv"]
    mask = pd.Series(True, index=r.index)
    for a,b in CRISES: mask.loc[a:b] = False
    rx = r[mask]
    P(f"{m:9}{r.mean():10.1f}{nw(r):7.1f}{rx.mean():12.1f}{nw(rx):7.1f}{len(rx):6d}")

# ---------------------------------------------------------------- [4] mediane
P("")
P("[4] MEDIA, MEDIANA, TRIMMED 10% del residuo R")
P(f"{'mercato':9}{'media':>9}{'mediana':>10}{'trim10%':>10}")
for m in MK4:
    al = decomp(m, rv_fwd(m))
    if al is None: continue
    r = (al["be"]-al["rv"]).dropna().values
    tr = np.sort(r)[int(.1*len(r)):int(.9*len(r))].mean()
    P(f"{m:9}{r.mean():9.1f}{np.median(r):10.1f}{tr:10.1f}")

# ---------------------------------------------------------------- [5] Mincer-Zarnowitz
P("")
P("[5] CHI PREVEDE LA VOL REALIZZATA FUTURA? (Mincer-Zarnowitz, RV_{t+3M} su predittori in t)")
P(f"{'mercato':9}{'solo BE: b':>12}{'[t]':>6}{'R2':>6}{'solo IV: b':>12}{'[t]':>6}{'R2':>6}"
  f"{'joint: bBE':>12}{'[t]':>6}{'bIV':>7}{'[t]':>6}{'R2':>6}")
def ols_nw(y, X, lag=6):
    Xv, yv = np.asarray(X,float), np.asarray(y,float)
    b = np.linalg.lstsq(Xv, yv, rcond=None)[0]; e = yv - Xv@b
    A = np.linalg.inv(Xv.T@Xv); S = (e[:,None]*Xv).T@(e[:,None]*Xv)
    for l in range(1,lag+1):
        w=1-l/(lag+1); u=e[l:,None]*Xv[l:]; v=e[:-l,None]*Xv[:-l]; G=u.T@v; S+=w*(G+G.T)
    V=A@S@A
    return b, b/np.sqrt(np.diag(V)), 1-np.var(e)/np.var(yv)
for m in MK4:
    al = decomp(m, rv_fwd(m))
    if al is None: continue
    y = al["rv"].values; one = np.ones(len(al))
    b1,t1,r1 = ols_nw(y, np.column_stack([one, al["be"]]))
    b2,t2,r2 = ols_nw(y, np.column_stack([one, al["iv"]]))
    b3,t3,r3 = ols_nw(y, np.column_stack([one, al["be"], al["iv"]]))
    P(f"{m:9}{b1[1]:12.2f}{t1[1]:6.1f}{r1:6.2f}{b2[1]:12.2f}{t2[1]:6.1f}{r2:6.2f}"
      f"{b3[1]:12.2f}{t3[1]:6.1f}{b3[2]:7.2f}{t3[2]:6.1f}{r3:6.2f}")
P("    Se nel joint la BE ha t ~ 0, la curva non aggiunge informazione di volatilita' oltre")
P("    l'opzione (coerente con 'troppa o troppo poca curvatura' di RR2021). Se aggiunge,")
P("    il prezzo di curva contiene un segnale proprio.")

# ---------------------------------------------------------------- [6] VRP tempo-variante
P("")
P("[6] IL TEST DECISIVO PER LA DIFESA DEL CO-MOVIMENTO")
P("    Decomposizione in tempo reale: sigma_IV = E_hat[RV] + VRP_hat, con E_hat = RV trailing 63g.")
P("    Domanda: le variazioni di sigma_BE seguono la componente ATTESA, il PREMIO, o nessuna?")
P(f"{'mercato':9}{'corr(dBE,dIV)':>14}{'corr(dBE,dEhat)':>16}{'corr(dBE,dVRPhat)':>18}"
  f"{'parz. dIV|dVRP':>16}{'T':>5}")
for m in MK4:
    iv = ivser(m)
    if iv is None: continue
    ehat = rv_from(daily10(m), 63)                  # trailing, DISPONIBILE in t
    al = pd.concat([sbe[m].rename("be"), iv.rename("iv"), ehat.rename("eh")], axis=1).dropna()
    if len(al) < 60: continue
    al["vh"] = al["iv"] - al["eh"]
    d = al.diff().dropna()
    c_iv, c_eh, c_vh = d["be"].corr(d["iv"]), d["be"].corr(d["eh"]), d["be"].corr(d["vh"])
    # parziale: corr(dBE, dIV | dVRPhat)
    def resid(a, b):
        bb = np.polyfit(b, a, 1); return a - np.polyval(bb, b)
    pc = pd.Series(resid(d["be"].values, d["vh"].values)).corr(
         pd.Series(resid(d["iv"].values, d["vh"].values)))
    P(f"{m:9}{c_iv:14.2f}{c_eh:16.2f}{c_vh:18.2f}{pc:16.2f}{len(d):5d}")
P("    LETTURA. Se corr(dBE,dEhat) e' anch'essa ~0, la curva non segue nemmeno la componente")
P("    fondamentale: il co-movimento nullo NON e' un artefatto del VRP tempo-variante, e la")
P("    difesa e' scrivibile. Se invece dBE segue dEhat ma non dVRPhat, la storia si RAFFINA:")
P("    le venue condividono le aspettative di volatilita' ma non il prezzo del rischio.")

# ---------------------------------------------------------------- [7] euro vs Bund
P("")
P("[7] IL RESIDUO E' DELLA CURVA SWAP O DELLA VALUTA? (le tre coppie within-currency)")
P("    Stessa decomposizione W = (sigma_BE - E[RV]) - VRP sul governativo, contro la STESSA")
P("    superficie di swaption della valuta. Se il residuo vive nello SWAP e non nel governativo,")
P("    il canale e' lo strumento (collaterale/clearing), non il paese.")
PAIRS = [("EUR","DEgovt","GDBR10"), ("GBP","UKgovt","GUKG10"), ("JPY","JPgovt","GJGB10")]
P(f"{'coppia':22}{'W':>8}{'VRP':>8}{'residuo R':>11}{'[NW t]':>8}{'T':>6}")
for sw, gv, gleg in PAIRS:
    rows = []
    al_sw = decomp(sw, rv_fwd(sw))
    if al_sw is not None:
        r = al_sw["be"] - al_sw["rv"]
        rows.append((f"{sw} swap", (al_sw["be"]-al_sw["iv"]).mean(),
                     (al_sw["iv"]-al_sw["rv"]).mean(), r.mean(), nw(r), len(al_sw)))
    if gv in sbe.columns:
        if gleg in mid:
            rvg = rv_from(mid[gleg]/100.0, 63).shift(-H_M)
            src = gleg
        else:
            rvg = rv_fwd(sw)
            src = f"proxy: RV {sw} swap"
        ivg = ivser(gv)
        alg = pd.concat([sbe[gv].rename("be"), ivg.rename("iv"), rvg.rename("rv")],
                        axis=1).dropna() if ivg is not None else None
        if alg is not None and len(alg) >= 60:
            rg = alg["be"] - alg["rv"]
            rows.append((f"{gv} ({src})", (alg["be"]-alg["iv"]).mean(),
                         (alg["iv"]-alg["rv"]).mean(), rg.mean(), nw(rg), len(alg)))
        else:
            rows.append((f"{gv}: dati insufficienti", np.nan, np.nan, np.nan, np.nan, 0))
    else:
        rows.append((f"{gv}: NON in sigbe_monthly.csv -- rilanciare 02", np.nan, np.nan, np.nan, np.nan, 0))
    for nm, w, v, r, t, n in rows:
        if np.isnan(w): P(f"   {nm}")
        else: P(f"   {nm:19}{w:8.1f}{v:8.1f}{r:11.1f}{t:8.1f}{n:6d}")
    # test sulla DIFFERENZA appaiata: R_swap - R_gov, mese per mese
    if al_sw is not None and gv in sbe.columns and 'alg' in dir() and alg is not None and len(alg) >= 60:
        dsw = (al_sw["be"] - al_sw["rv"]).rename("s")
        dgv = (alg["be"] - alg["rv"]).rename("g")
        dd = pd.concat([dsw, dgv], axis=1).dropna()
        if len(dd) >= 60:
            diff = dd["s"] - dd["g"]
            P(f"   {'DIFFERENZA swap-gov':19}{'':8}{'':8}{diff.mean():11.1f}{nw(diff):8.1f}{len(dd):6d}")
    P("")
P("    LETTURA PER COPPIA. EUR/Bund: se il Bund non ha residuo, i 37.8 dell'euro sono dello")
P("    strumento swap. GBP/gilt e JPY/JGB: con residui swap ~0, un residuo governativo non nullo")
P("    ribalterebbe la lettura -- entrambe le direzioni sono informative.")
# ---------------------------------------------------------------- [8] permutazione esatta
P("")
P("[8] LA DISPERSIONE DEI COSTI SEGUE IL RESIDUO? (permutazione ESATTA, n=4 -> 24 permutazioni)")
try:
    from scipy.stats import spearmanr
    ratios, resids, vrps = [], [], []
    for m in MK4:
        al = decomp(m, rv_fwd(m))
        if al is None: continue
        ratios.append((al["be"].mean()/al["iv"].mean())**2)
        resids.append((al["be"]-al["rv"]).mean())
        vrps.append((al["iv"]-al["rv"]).mean())
    def exact(x, y):
        rho = spearmanr(x, y)[0]; n=0; t=0
        for p_ in permutations(y):
            t += 1
            if abs(spearmanr(x, list(p_))[0]) >= abs(rho)-1e-12: n += 1
        return rho, n/t
    a,pa = exact(ratios, vrps); b,pb = exact(ratios, resids)
    P(f"   rapporto grezzo ~ VRP     : rango {a:+.2f}  p esatto {pa:.3f}")
    P(f"   rapporto grezzo ~ residuo : rango {b:+.2f}  p esatto {pb:.3f}   (minimo possibile con n=4: 0.042)")
except Exception as e:
    P(f"   non calcolato ({e})")

P("")
P("COSA GUARDARE, IN ORDINE. (i) Pannello 1, colonna FORWARD: se R non cambia, la questione CMS")
P("e' chiusa. (ii) Pannello 6: decide se la frase 'il co-movimento e' intatto' e' scrivibile a")
P("Rebonato o va qualificata. (iii) Pannello 3: se l'euro regge ex-crisi, il residuo non e' un")
P("artefatto del 2011-12. (iv) Pannello 7: se il Bund non ha residuo, hai la risposta alla")
P("domanda 'e' l'Europa o e' lo swap?'. (v) Pannello 5: stabilisce cosa la curva SA della")
P("volatilita' -- il tassello che decide fra 'due prezzi di un rischio' e 'un prezzo e un rumore'.")
save_txt("22_vrp_battery.txt", L); print("\n".join(L))

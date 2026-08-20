"""r1 - BEI US e UK e premio residuo (richiesta RR n.1), specificazione market-based
di Maffei-Rebonato: projected inflation = zero-coupon inflation swap rate (ISR).
  BEI - projected = BEI - ISR = -lambda   (mispricing / liquidita' dei linker,
                                          il lambda di Fleckenstein et al. e di Maffei)
  BEI - spot      = BEI - YoY corrente     (premio grezzo)
Confronto US vs UK per sottoperiodi + correlazione. Output: an_bei_premia.csv.
Curve: BoE (UK) e GSW (US) + ISR dal cache della pipeline (BPSWIT / USSWIT).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import pandas as pd
import rp
from config import CACHE

print(">>> r1 US/UK-only v3 -- griglia completa <<<")

# La griglia NON e' piu' fissata a (5,10). Il divario US-UK e' statisticamente NULLO sul
# tratto 3-7 anni (t < 1.1) e fortemente significativo sul lungo (t -9.5 a 15y, -11.0 a
# 20y): con due soli nodi si guardava proprio dove non succede nulla. Le scadenze coperte
# si scoprono a runtime perche' interp_cols si ferma -- giustamente -- se un nodo cade
# fuori dalla curva o dagli swap, e le quattro fonti (BEI e ISR, US e UK) non coincidono.
CANDIDATE = (2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)

# Il 30y e' ESCLUSO, e non per prudenza generica: la curva d'inflazione BoE arriva a 30
# anni solo dal gennaio 2016 (2.673 osservazioni contro le >10.000 degli altri nodi --
# verificato con r1c_check30.py). Il campione comune a 30y sarebbe quindi di 127 mesi
# contro 264, tutti nel decennio recente: non e' confrontabile con gli altri nodi e il
# livello anomalo del premio US a quel nodo (-5 bp contro -23 altrove) e' un effetto di
# periodo, non di scadenza. Il vincolo e' la CURVA, non gli swap: nessuna scelta possibile.
MIN_OSS_NODO = 200          # mesi minimi di campione comune perche' un nodo entri

def _coperte(cand):
    ok = []
    for m in cand:
        try:
            for f in (lambda: rp.bei_us((m,)), lambda: rp.bei_uk((m,)),
                      lambda: rp.isr("US", (m,)), lambda: rp.isr("UK", (m,))):
                if f().dropna(how="all").empty: raise ValueError
            ok.append(m)
        except Exception:
            pass
    return tuple(ok)

MATS = _coperte(CANDIDATE)
# secondo filtro: profondita' del campione comune, non solo esistenza del nodo
_prof = []
for _m in MATS:
    _b = (rp.bei_us if True else None)((_m,))[_m].dropna()
    _u = rp.bei_uk((_m,))[_m].dropna()
    _n = len(_b.index.intersection(_u.index).to_period("M").unique())
    if _n >= MIN_OSS_NODO: _prof.append(_m)
    else: print(f"  [escluso] {int(_m)}y: solo {_n} mesi di campione comune (< {MIN_OSS_NODO})")
MATS = tuple(_prof)
print(f"scadenze coperte da tutte le fonti: {[int(m) for m in MATS]}")
print(f"  (candidate {[int(m) for m in CANDIDATE]}; le escluse non sono nella curva o negli swap)")
out = {}
for mkt in ("US", "UK"):        # SOLO i due mercati che RR chiede (curve solide)
    b, s, lam = rp.lam_gamma(mkt, MATS)
    pi = rp.yoy(rp.cpi(mkt))
    for m in MATS:
        k = int(m)
        out[f"BEI_{mkt}_{k}y"] = b[m]
        out[f"ISR_{mkt}_{k}y"] = s[m]
        out[f"BEI-ISR_{mkt}_{k}y"] = -lam[m]           # BEI - projected
        out[f"BEI-spot_{mkt}_{k}y"] = b[m] - pi.reindex(b.index).ffill()
out = pd.DataFrame(out).sort_index()
out.to_csv(CACHE / "an_bei_premia.csv")

MKTS = ("US", "UK")             # RR n.1 riguarda US e UK
PER = {"pre-2009": lambda x: x < "2009", "2009-2019": lambda x: (x >= "2009") & (x < "2020"),
       "2020-oggi": lambda x: x >= "2020"}

# livello medio di lambda per mercato, ognuno sul PROPRIO campione (niente intersezione a 5,
# che la Germania giovane taglierebbe al 2012): ogni linker sfrutta tutta la sua storia.
print("=== livello medio  -lambda = BEI-ISR, per scadenza e periodo [bp] ===")
print(f"{'periodo':10s}" + "".join(f"{f'{int(t)}y {m}':>9s}" for t in MATS for m in MKTS))
for k, mask in PER.items():
    row = f"{k:10s}"
    for t in MATS:
        for m in MKTS:
            d = out[f"BEI-ISR_{m}_{int(t)}y"].dropna(); d = d[mask(d.index)]
            row += f"{d.mean()*100:9.0f}" if len(d) >= 12 else f"{'--':>9s}"
    print(row)
_ref = int(MATS[len(MATS)//2])
sp = {m: (out[f"BEI-ISR_{m}_{_ref}y"].dropna().index.min(),
          out[f"BEI-ISR_{m}_{_ref}y"].dropna().index.max()) for m in MKTS}
print("campioni:", "  ".join(f"{m} {a.year}-{b.year}" for m,(a,b) in sp.items()))

# correlazione BILATERALE con US: ogni coppia sul suo campione comune
print("\n=== correlazione US-UK di -lambda, per scadenza e periodo ===")
print(f"{'periodo':10s}" + "".join(f"{f'{int(t)}y':>9s}" for t in MATS))
for k, mask in PER.items():
    row = f"{k:10s}"
    for t in MATS:
        d = out[[f"BEI-ISR_US_{int(t)}y", f"BEI-ISR_UK_{int(t)}y"]].dropna()
        d = d[mask(d.index)]
        row += f"{d.iloc[:,0].corr(d.iloc[:,1]):9.2f}" if len(d) >= 12 else f"{'--':>9s}"
    print(row)
# ============================================================================ RR domanda 1
# "Calculate US BEI and UK BEI inflation. Subtract projected and spot inflation from both.
#  Compare the residual risk premium for US and UK. Similar? If different, why?"
# Le due sottrazioni sono entrambe richieste: projected = ISR (swap zero-coupon),
# spot = YoY corrente. Qui il confronto DIRETTO US vs UK sui due residui, per scadenza
# e per sottoperiodo, con la differenza e la sua dispersione.
print("\n" + "=" * 78)
print("RR n.1 -- CONFRONTO DIRETTO US vs UK del premio residuo")
print("=" * 78)
print("  due residui, come chiesto: BEI-ISR (projected) e BEI-spot (inflazione corrente)")

def _nw_bandwidth_1994(x):
    """Banda di Newey-West (1994) scelta dai dati (Bartlett). Stessa regola di r1b_why.py:
    niente numero fisso. Rif.: Newey-West (1994) Rev. Econ. Stud. 61."""
    import numpy as _np
    e = _np.asarray(pd.Series(x).dropna(), float); e = e - e.mean(); T = len(e)
    if T < 24: return 12
    def sig(j): return float(e[j:] @ e[:-j]) / T if j > 0 else float(e @ e) / T
    n = max(1, min(int(4 * (T / 100.0) ** (2.0 / 9.0)), T - 1))
    s0 = sig(0) + 2.0 * sum(sig(j) for j in range(1, n + 1))
    s1 = 2.0 * sum(j * sig(j) for j in range(1, n + 1))
    if s0 == 0: return max(1, n)
    gamma = 1.1447 * (abs(s1 / s0)) ** (2.0 / 3.0)
    return max(1, min(int(gamma * T ** (1.0 / 3.0)), T - 1))


def _nw_t_mean(x, lags=None):
    """t di Newey-West sulla media (H0: i due premi coincidono), banda AUTOMATICA
    (NW1994) se lags=None. Sostituisce le soglie arbitrarie della versione precedente."""
    import numpy as _np
    x = _np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if n < 24: return float("nan")
    if lags is None: lags = _nw_bandwidth_1994(x)
    e = x - x.mean(); s = float(e @ e) / n
    for l in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - l / (lags + 1)) * float(e[l:] @ e[:-l]) / n
    return float("nan") if s <= 0 else x.mean() / (s / n) ** 0.5

for tag, lbl in (("BEI-ISR", "BEI - projected (ISR)"), ("BEI-spot", "BEI - spot (YoY)")):
    print(f"\n  {lbl}   -- intero campione comune  [bp]")
    print(f"    {'scad.':>7}{'US':>9}{'UK':>9}{'diff':>9}{'t-NW':>8}{'lag':>5}{'sd':>8}{'US>UK':>8}{'n':>7}")
    for t_ in MATS:
        k_ = int(t_)
        both = out[[f"{tag}_US_{k_}y", f"{tag}_UK_{k_}y"]].dropna()
        if len(both) < 24: continue
        dif = (both.iloc[:, 0] - both.iloc[:, 1]) * 100
        L = _nw_bandwidth_1994(dif)                    # banda scelta per QUESTA scadenza
        print(f"    {k_:>6}y{both.iloc[:,0].mean()*100:>9.0f}{both.iloc[:,1].mean()*100:>9.0f}"
              f"{dif.mean():>9.0f}{_nw_t_mean(dif):>8.2f}{L:>5}{dif.std():>8.0f}"
              f"{(dif>0).mean():>7.0%}{len(both):>7}")
    print("    [t-NW banda automatica (NW1994), colonna 'lag' = banda scelta; H0 = i due premi coincidono]")
    print(f"\n  {lbl}   -- per sottoperiodo, differenza US-UK  [bp]")
    print(f"    {'periodo':10s}" + "".join(f"{f'{int(t_)}y':>9s}" for t_ in MATS))
    for k, mask in PER.items():
        row = f"    {k:10s}"
        for t_ in MATS:
            both = out[[f"{tag}_US_{int(t_)}y", f"{tag}_UK_{int(t_)}y"]].dropna()
            both = both[mask(both.index)]
            row += (f"{(both.iloc[:,0]-both.iloc[:,1]).mean()*100:9.0f}"
                    if len(both) >= 12 else f"{'--':>9s}")
        print(row)

# ------------------------------------------------- diagnostica dell'inferenza
# Il t nelle tabelle sopra e' un test sulla MEDIA della differenza (regressione sulla
# sola costante) con errore standard HAC a 12 lag: H0 = le due basi coincidono in media.
# Su una serie molto persistente 12 lag possono NON bastare: il kernel di Bartlett si
# chiude prima che l'autocorrelazione si esaurisca e il t resta gonfiato. Qui si misura
# invece di assumere: rho AR(1), t a bande crescenti, e la dimensione campionaria
# EFFICACE n*(1-rho)/(1+rho), che dice quante osservazioni indipendenti ci sono davvero.
print("\n" + "=" * 78)
print("DIAGNOSTICA -- quanto e' affidabile il t sulla differenza? (BEI-ISR)")
print("=" * 78)
print(f"    {'scad.':>7}{'AR(1)':>8}{'t(12)':>9}{'t(24)':>9}{'t(36)':>9}{'n':>7}{'n_eff':>8}")
for t_ in MATS:
    k_ = int(t_)
    both = out[[f"BEI-ISR_US_{k_}y", f"BEI-ISR_UK_{k_}y"]].dropna()
    if len(both) < 24:
        continue
    dif = (both.iloc[:, 0] - both.iloc[:, 1]) * 100
    rho = float(dif.autocorr(1))
    neff = len(dif) * (1 - rho) / (1 + rho) if rho < 0.999 else float("nan")
    print(f"    {k_:>6}y{rho:>8.2f}{_nw_t_mean(dif, 12):>9.2f}{_nw_t_mean(dif, 24):>9.2f}"
          f"{_nw_t_mean(dif, 36):>9.2f}{len(dif):>7}{neff:>8.0f}")
print("\n  [se il t CROLLA passando da 12 a 36 lag, la significativita' dipende dalla banda")
print("   e va riportata con la banda piu' conservativa. Se e' STABILE, 12 lag bastano.")
print("   n_eff molto minore di n = poche osservazioni indipendenti: il t va letto con")
print("   prudenza a prescindere, e la quota di mesi con US>UK e' l'evidenza piu' robusta")
print("   perche' non richiede ipotesi sulla struttura di autocorrelazione.]")

# ---------------------------------------------------------------- BEI - spot
# RR chiede di sottrarre projected E spot. Le due sottrazioni NON misurano la stessa
# cosa, e l'identita' lo rende esplicito:
#     BEI - spot  =  (BEI - ISR)  +  (ISR - spot)
#                     [base]         [gap attese-realizzato]
# Il primo termine e' il premio che studiamo; il secondo e' l'errore di previsione
# d'inflazione del mercato, che nel campione e' dominato dal ciclo (2021-22) e ha una
# volatilita' un ordine di grandezza superiore. Stampare la scomposizione evita di
# leggere come "premio" cio' che e' ciclo d'inflazione.
print("\n" + "=" * 78)
print("SCOMPOSIZIONE -- BEI-spot = (BEI-ISR) + (ISR-spot)")
print("=" * 78)
print("  il secondo termine NON e' un premio: e' il divario fra inflazione attesa dal")
print("  mercato (ISR) e inflazione realizzata (YoY). Serve a capire perche' le due")
print("  sottrazioni che chiede RR danno risposte diverse.")
print(f"\n    {'scad.':>7}{'ISR-spot US':>13}{'ISR-spot UK':>13}{'diff':>8}"
      f"{'sd base':>10}{'sd spot':>10}{'rapp.':>8}")
for t_ in MATS:
    k_ = int(t_)
    cols = [f"BEI-ISR_US_{k_}y", f"BEI-ISR_UK_{k_}y",
            f"BEI-spot_US_{k_}y", f"BEI-spot_UK_{k_}y"]
    d = out[cols].dropna()
    if len(d) < 24:
        continue
    gap_us = (d.iloc[:, 2] - d.iloc[:, 0]) * 100      # (BEI-spot) - (BEI-ISR) = ISR-spot
    gap_uk = (d.iloc[:, 3] - d.iloc[:, 1]) * 100
    sd_base = ((d.iloc[:, 0] - d.iloc[:, 1]) * 100).std()
    sd_spot = ((d.iloc[:, 2] - d.iloc[:, 3]) * 100).std()
    print(f"    {k_:>6}y{gap_us.mean():>13.0f}{gap_uk.mean():>13.0f}"
          f"{gap_us.mean()-gap_uk.mean():>8.0f}{sd_base:>10.0f}{sd_spot:>10.0f}"
          f"{sd_spot/sd_base:>7.1f}x")
print("\n  [rapp. = sd della differenza US-UK su BEI-spot / su BEI-ISR. Se e' >>1, la")
print("   misura grezza e' dominata dal ciclo d'inflazione e non discrimina i due mercati.]")
print("\n  NOTA di comparabilita': lo spot US e' CPI, quello UK e' RPI. L'RPI corre")
print("  strutturalmente sopra il CPI (effetto formula, ~0.8-1pp), quindi il confronto")
print("  CROSS-PAESE di BEI-spot e' contaminato dall'indice. In BEI-ISR il problema non")
print("  si pone: dentro ogni paese BEI e ISR sono sullo STESSO indice e il divario si")
print("  cancella. E' la ragione per cui la base e' la misura di riferimento.")

# la differenza US-UK: livello medio, dispersione e stabilita' del segno (10y, BEI-ISR)
print("\n" + "=" * 78)
print("VERDETTO -- 'Similar? If different, why?'")
print("=" * 78)
sim, dif_ = [], []
for t_ in MATS:
    k_ = int(t_)
    both = out[[f"BEI-ISR_US_{k_}y", f"BEI-ISR_UK_{k_}y"]].dropna()
    if len(both) < 24: continue
    dd = (both.iloc[:, 0] - both.iloc[:, 1]) * 100
    (dif_ if abs(_nw_t_mean(dd)) > 2 else sim).append(k_)
print(f"  premi INDISTINGUIBILI (|t| <= 2) alle scadenze: {sim}")
print(f"  premi DIVERSI        (|t| >  2) alle scadenze: {dif_}")
print("  -> la risposta NON e' un si'/no: dipende dalla scadenza. La forma delle due")
print("     curve di premio e' cio' che differisce, non il loro livello medio.")
_prof = {m: [out[f"BEI-ISR_{m}_{int(t_)}y"].dropna().mean()*100 for t_ in MATS] for m in MKTS}
for m in MKTS:
    print(f"  profilo {m}: " + "  ".join(f"{int(t_)}y={v:+.0f}" for t_, v in zip(MATS, _prof[m])))
print("  [il PERCHE' e' in r1b_why.py: struttura a termine, event study giornalieri e i")
print("   tre canali -- floor di deflazione, domanda LDI, liquidita' (ONOFF/NOISE/MOVE)]")

print(f"\nsalvato: an_bei_premia.csv. -lambda<0 = linker cheap al sintetico (FLL).")

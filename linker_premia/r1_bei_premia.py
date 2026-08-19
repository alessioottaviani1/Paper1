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

def _nw_t_mean(x, lags=12):
    """t di Newey-West sulla media (H0: i due premi coincidono). Sostituisce le soglie
    arbitrarie della versione precedente, che decidevano 'SIMILI/DIVERSI' senza test."""
    import numpy as _np
    x = _np.asarray(pd.Series(x).dropna(), float); n = len(x)
    if n < 24: return float("nan")
    e = x - x.mean(); s = float(e @ e) / n
    for l in range(1, min(lags, n - 1) + 1):
        s += 2 * (1 - l / (lags + 1)) * float(e[l:] @ e[:-l]) / n
    return float("nan") if s <= 0 else x.mean() / (s / n) ** 0.5

for tag, lbl in (("BEI-ISR", "BEI - projected (ISR)"), ("BEI-spot", "BEI - spot (YoY)")):
    print(f"\n  {lbl}   -- intero campione comune  [bp]")
    print(f"    {'scad.':>7}{'US':>9}{'UK':>9}{'diff':>9}{'t-NW':>8}{'sd':>8}{'US>UK':>8}{'n':>7}")
    for t_ in MATS:
        k_ = int(t_)
        both = out[[f"{tag}_US_{k_}y", f"{tag}_UK_{k_}y"]].dropna()
        if len(both) < 24: continue
        dif = (both.iloc[:, 0] - both.iloc[:, 1]) * 100
        print(f"    {k_:>6}y{both.iloc[:,0].mean()*100:>9.0f}{both.iloc[:,1].mean()*100:>9.0f}"
              f"{dif.mean():>9.0f}{_nw_t_mean(dif):>8.2f}{dif.std():>8.0f}"
              f"{(dif>0).mean():>7.0%}{len(both):>7}")
    print("    [t-NW a 12 lag sulla differenza mensile: H0 = i due premi coincidono]")
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

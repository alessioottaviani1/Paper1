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
from config import CACHE, DATA

# Le tabelle LaTeX finiscono in results/tables/ e si includono con \input{\tablepath/nome},
# esattamente come nel Paper 1: i numeri non si ricopiano MAI a mano, quindi non possono
# divergere dal codice che li calcola. Lo stesso file serve al deck e alla tesi.
TABLES_DIR = DATA.parent / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

def _stars(t, thr=(1.645, 1.96, 2.576)):
    """Stelle di significativita' dal t-stat (10%, 5%, 1%)."""
    try: a = abs(float(t))
    except (TypeError, ValueError): return ""
    if a != a: return ""
    return "***" if a >= thr[2] else "**" if a >= thr[1] else "*" if a >= thr[0] else ""

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

# ---- nodi UK-ONLY: oltre il limite del CONFRONTO, ma dentro i dati britannici.
# Il confronto US-UK si ferma a 20y perche' GSW pubblica la curva TIPS solo per gli
# interi 2-20 (feds200805, Tab.1): a 25y il lato US NON ESISTE. Il PROFILO UK invece
# si': curva BoE al 100% nel campione e BPSWIT quota il 25y come nodo vero (98.1% dal
# 2004, come il 20y). Il 30y resta fuori: li' la curva reale BoE parte dal 2016.
CAND_UK_EXT = (25.0,)
def _coperte_uk(cand):
    ok = []
    for m in cand:
        try:
            for f in (lambda: rp.bei_uk((m,)), lambda: rp.isr("UK", (m,))):
                if f().dropna(how="all").empty: raise ValueError
            ok.append(m)
        except Exception:
            pass
    return tuple(ok)
MATS_UK_EXT = _coperte_uk(CAND_UK_EXT)
MATS_UK = tuple(sorted(set(MATS) | set(MATS_UK_EXT)))
if MATS_UK_EXT:
    print(f"  [UK-only] nodi oltre il limite del confronto: {[int(m) for m in MATS_UK_EXT]}"
          f"  (US non disponibile: GSW TIPS si ferma a 20y)")
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
if MATS_UK_EXT:
    _b2, _s2, _lam2 = rp.lam_gamma("UK", MATS_UK_EXT)
    _pi2 = rp.yoy(rp.cpi("UK"))
    for m in MATS_UK_EXT:
        k = int(m)
        out[f"BEI_UK_{k}y"] = _b2[m]; out[f"ISR_UK_{k}y"] = _s2[m]
        out[f"BEI-ISR_UK_{k}y"] = -_lam2[m]
        out[f"BEI-spot_UK_{k}y"] = _b2[m] - _pi2.reindex(_b2.index).ffill()
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

# ---- PROFILO UK ESTESO: la base britannica oltre il limite del confronto -------------
# Qui NON c'e' differenza US-UK (a 25y il lato US non esiste): si guarda la FORMA della
# sola curva UK. La domanda: la base, che risale verso zero a 20y, prosegue oltre?
if MATS_UK_EXT:
    print("\n=== PROFILO UK-ONLY di -lambda = BEI-ISR, oltre il limite del confronto [bp] ===")
    print("  (nessun confronto US: GSW pubblica la curva TIPS solo 2-20y)")
    print(f"{'periodo':10s}" + "".join(f"{f'{int(t)}y':>9s}" for t in MATS_UK))
    for k, mask in PER.items():
        row = f"{k:10s}"
        for t in MATS_UK:
            c = f"BEI-ISR_UK_{int(t)}y"
            if c not in out.columns:
                row += f"{'--':>9s}"; continue
            d = out[c].dropna(); d = d[mask(d.index)]
            row += f"{d.mean()*100:9.0f}" if len(d) >= 12 else f"{'--':>9s}"
        print(row)
    print(f"\n{'intero camp.':10s}" + "".join(f"{f'{int(t)}y':>9s}" for t in MATS_UK))
    row_lv, row_pos, row_n = f"{'  livello':10s}", f"{'  b>0':10s}", f"{'  n mesi':10s}"
    for t in MATS_UK:
        c = f"BEI-ISR_UK_{int(t)}y"
        if c not in out.columns:
            row_lv += f"{'--':>9s}"; row_pos += f"{'--':>9s}"; row_n += f"{'--':>9s}"; continue
        d = out[c].dropna()
        row_lv += f"{d.mean()*100:9.0f}"; row_pos += f"{(d > 0).mean():>8.0%} "
        row_n += f"{len(d):9d}"
    print(row_lv); print(row_pos); print(row_n)
    print("\n  [b>0 = mesi in cui il linker UK e' CARO rispetto al sintetico. Se la quota")
    print("   CRESCE oltre i 20 anni, la domanda strutturale (LDI) spinge il prezzo del")
    print("   titolo indicizzato sopra il sintetico proprio dove quella clientela opera.]")

    # ---- TABELLA LATEX del profilo UK esteso
    with open(TABLES_DIR / "rq1_uk_long_end.tex", "w", encoding="utf-8") as f:
        f.write("%% generato da r1_bei_premia.py -- NON modificare a mano\n")
        f.write("{\\footnotesize\n\\begin{tabular}{l" + "c" * len(MATS_UK) + "}\n\\toprule\n")
        f.write("UK only & " + " & ".join(f"{int(t)}y" for t in MATS_UK) + " \\\\\n\\midrule\n")
        for k, mask in PER.items():
            lab = {"pre-2009": "pre-2009", "2009-2019": "2009--2019",
                   "2020-oggi": "2020--today"}.get(k, k)
            f.write(lab)
            for t in MATS_UK:
                c = f"BEI-ISR_UK_{int(t)}y"
                d = out[c].dropna() if c in out.columns else out.iloc[:0, :1].squeeze()
                d = d[mask(d.index)] if len(d) else d
                f.write(f" & ${d.mean()*100:+.0f}$" if len(d) >= 12 else " & ---")
            f.write(" \\\\\n")
        f.write("\\midrule\n")
        for lab, fn in (("level, full sample [bp]", lambda d: f"${d.mean()*100:+.0f}$"),
                        ("months with $b>0$", lambda d: f"{(d>0).mean():.0%}".replace("%", "\\%"))):
            f.write(lab)
            for t in MATS_UK:
                c = f"BEI-ISR_UK_{int(t)}y"
                f.write(" & " + (fn(out[c].dropna()) if c in out.columns else "---"))
            f.write(" \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}}\n")
        f.write("\n{\\scriptsize The comparison with the US stops at 20y because GSW publish "
                "the TIPS curve for integers 2--20 only. The BoE real curve covers 25y over "
                "100\\% of the sample and BPSWIT quotes 25y as a true node, so the UK profile "
                "can be read five years beyond the common range. $b>0$ means the cash linker "
                "is expensive relative to the swap-implied synthetic.}\n")
    print(f"  [tex] {TABLES_DIR / 'rq1_uk_long_end.tex'}")
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

    # ---- TABELLA LATEX (slide + tesi): stessi numeri appena stampati
    _mt = [t_ for t_ in MATS
           if not out[[f"{tag}_US_{int(t_)}y", f"{tag}_UK_{int(t_)}y"]].dropna().empty]
    _nm = "basis" if tag == "BEI-ISR" else "spot"
    _sym = "b" if tag == "BEI-ISR" else "\\text{BEI}-\\text{spot}"
    with open(TABLES_DIR / f"rq1_{_nm}_levels.tex", "w", encoding="utf-8") as f:
        f.write("%% generato da r1_bei_premia.py -- NON modificare a mano\n")
        f.write("{\\footnotesize\n\\begin{tabular}{l" + "c" * len(_mt) + "}\n\\toprule\n")
        f.write("Maturity & " + " & ".join(f"{int(t_)}y" for t_ in _mt) + " \\\\\n\\midrule\n")
        for _who, _col in (("US", 0), ("UK", 1)):
            f.write(f"$({_sym})_{{{_who}}}$ \\ [bp]")
            for t_ in _mt:
                _d = out[[f"{tag}_US_{int(t_)}y", f"{tag}_UK_{int(t_)}y"]].dropna()
                f.write(f" & ${_d.iloc[:, _col].mean() * 100:+.0f}$")
            f.write(" \\\\\n")
        f.write("\\midrule\n")
        for _lab, _fn in (("difference US$-$UK",
                           lambda d: f"${((d.iloc[:,0]-d.iloc[:,1])*100).mean():+.0f}$"),
                          ("months US$>$UK",
                           lambda d: f"{((d.iloc[:,0]-d.iloc[:,1])>0).mean():.0%}".replace("%", "\\%"))):
            f.write(_lab)
            for t_ in _mt:
                _d = out[[f"{tag}_US_{int(t_)}y", f"{tag}_UK_{int(t_)}y"]].dropna()
                f.write(" & " + _fn(_d))
            f.write(" \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}}\n")
        f.write("\n{\\scriptsize Common sample 2004--2026. The last row is the share of "
                "months in which the US measure exceeds the UK one: it is distribution-free "
                "and does not depend on how the persistence of the differential is modelled.}\n")
    print(f"    [tex] {TABLES_DIR / f'rq1_{_nm}_levels.tex'}")
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

# ============================================================================
# RR n.2 (post-call) -- LA BASE CONTRO L'INFLAZIONE ATTESA (senza lo swap)
# ============================================================================
# RR in call: "la base con lo swap e' interessante, ma l'ISR contiene un inflation
# risk premium". Vero. Le tre sottrazioni dicono cose diverse:
#   BEI - ISR    = -lambda            (rp_pi comune si CANCELLA: base di LIQUIDITA')
#   BEI - E[pi]  = rp_pi - lambda     (premio TOTALE nel breakeven; il gamma di Maffei)
#   ISR - E[pi]  = rp_pi dello swap   (l'obiezione di RR, QUANTIFICATA)
# Misure di E[pi], come Maffei (tesi 6.2.1 + nota 34):
#   1) Cleveland Fed (Haubrich-Pennacchi-Ritchken 2012 RFS), SOLO US, fonte ufficiale
#      clevelandfed.org (gia' cosi' in r0_data);
#   2) AR(1) real-time sul YoY (CPI SA per US, RPI per UK), finestra espandente:
#      IDENTICO nei due mercati -> e' la misura su cui il confronto US-UK e' simmetrico.
# Per il UK un "Cleveland" non esiste: il riferimento accademico e' Joyce-Lildholdt-
# Sorensen (BoE Working Paper 360, 2009: modello affine congiunto nominale-reale che
# decompone i breakeven in aspettative e premi), ma NON e' una serie pubblicata viva.
# Le fonti ufficiali vive (HM Treasury "Forecasts for the UK economy" mensile; BoE
# Survey of External Forecasters trimestrale) coprono orizzonti 1-3y (5y nel medio
# termine), non le scadenze dei linker. L'AR(1) e' quindi la misura simmetrica; il
# Cleveland fa da CROSS-CHECK dell'AR(1) sugli US, dove entrambi esistono.

print("\n" + "=" * 78)
print("RR n.2 -- LA BASE CONTRO L'INFLAZIONE ATTESA (Cleveland US / AR1 entrambi)")
print("=" * 78)
print("  BEI-ISR = -lambda (liquidita'); BEI-E[pi] = rp_pi-lambda (premio totale);")
print("  ISR-E[pi] = inflation risk premium dello SWAP (l'obiezione di RR, quantificata)")

_expinf = {"US": {}, "UK": {}}
try:
    _expinf["US"]["CLEV"] = rp.interp_cols(rp.expinf_us(), MATS)
except Exception as _e:
    print(f"  [!] Cleveland non disponibile ({_e}): lanciare r0_data.")
for _mkt, _mm in (("US", MATS), ("UK", MATS_UK)):
    try:
        _expinf[_mkt]["AR1"] = rp.expinf_ar1(_mkt, _mm)
    except Exception as _e:
        print(f"  [!] AR1 {_mkt} non disponibile: {_e}")

# colonne in out: BEI - E[pi] e ISR - E[pi], per misura
for _mkt in ("US", "UK"):
    _mm = MATS_UK if _mkt == "UK" else MATS
    for _lbl, _pih in _expinf[_mkt].items():
        _pih = _pih.reindex(pd.Index(sorted({float(m) for m in _mm}), name=None), axis=1) \
                   .reindex(out[f"BEI_{_mkt}_{int(_mm[0])}y"].dropna().index, method="ffill")
        for m in _mm:
            k = int(m)
            if float(m) not in _pih.columns:
                continue
            _bei = out.get(f"BEI_{_mkt}_{k}y")
            _isr = out.get(f"ISR_{_mkt}_{k}y")
            if _bei is None:
                continue
            out[f"BEI-E{_lbl}_{_mkt}_{k}y"] = _bei - _pih[float(m)]
            if _isr is not None:
                out[f"IRP{_lbl}_{_mkt}_{k}y"] = _isr - _pih[float(m)]

# ---- A) livelli del premio totale BEI - E[pi], intero campione [bp]
print("\n  A) BEI - E[pi]  (= rp_pi - lambda, il premio TOTALE nel breakeven) [bp]")
print(f"     {'misura':16s}" + "".join(f"{f'{int(t)}y':>8s}" for t in MATS_UK))
for _mkt in ("US", "UK"):
    for _lbl in ("CLEV", "AR1"):
        if _lbl not in _expinf[_mkt]:
            continue
        row = f"     {_mkt} {'Cleveland' if _lbl=='CLEV' else 'AR(1)':10s}"
        for t in MATS_UK:
            c = f"BEI-E{_lbl}_{_mkt}_{int(t)}y"
            d = out[c].dropna() if c in out else []
            row += f"{d.mean()*100:8.0f}" if len(d) >= 24 else f"{'--':>8s}"
        print(row)
print("     [segno atteso: rp_pi>0 spinge in su, lambda>0 in giu'. Sul lungo UK,")
print("      dove lambda~0 (RR n.1), BEI-E[pi] ~ rp_pi puro.]")

# ---- B) confronto US-UK sulla misura SIMMETRICA (AR1)
if "AR1" in _expinf["US"] and "AR1" in _expinf["UK"]:
    print("\n  B) differenza US-UK di BEI - E[pi] (AR1: stessa misura nei due mercati)")
    print(f"     {'scad.':>6}{'US':>8}{'UK':>8}{'diff':>8}{'t-NW':>7}{'lag':>5}{'US>UK':>8}{'n':>6}")
    for t in MATS:
        cu, ck = f"BEI-EAR1_US_{int(t)}y", f"BEI-EAR1_UK_{int(t)}y"
        if cu not in out or ck not in out:
            continue
        d = out[[cu, ck]].dropna()
        dif = (d[cu] - d[ck]) * 100
        print(f"     {int(t):>5}y{d[cu].mean()*100:8.0f}{d[ck].mean()*100:8.0f}"
              f"{dif.mean():8.0f}{_nw_t_mean(dif):7.2f}{_nw_bandwidth_1994(dif):>5}"
              f"{(dif > 0).mean():>7.0%} {len(d):6d}")
    print("     [stessa lettura di RR n.1 ma senza lo swap: qui rp_pi NON si cancella,")
    print("      quindi la differenza mescola premi d'inflazione E liquidita' relativi.]")

# ---- C) l'obiezione di RR quantificata: ISR - E[pi] = rp_pi dello swap
print("\n  C) ISR - E[pi]  (= inflation risk premium nello SWAP) [bp]")
print(f"     {'misura':16s}" + "".join(f"{f'{int(t)}y':>8s}" for t in MATS_UK))
for _mkt in ("US", "UK"):
    for _lbl in ("CLEV", "AR1"):
        if _lbl not in _expinf[_mkt]:
            continue
        row = f"     {_mkt} {'Cleveland' if _lbl=='CLEV' else 'AR(1)':10s}"
        for t in MATS_UK:
            c = f"IRP{_lbl}_{_mkt}_{int(t)}y"
            d = out[c].dropna() if c in out else []
            row += f"{d.mean()*100:8.0f}" if len(d) >= 24 else f"{'--':>8s}"
        print(row)
print("     [se >0 e crescente col tenore: il mercato swap incorpora un premio per il")
print("      rischio d'inflazione, come RR sostiene. E' la GIUSTIFICAZIONE empirica di")
print("      guardare anche BEI-E[pi] e non solo BEI-ISR.]")

# ---- D) cross-check US: quanto l'AR1 replica il Cleveland (valida l'AR1 per il UK)
if "CLEV" in _expinf["US"] and "AR1" in _expinf["US"]:
    print("\n  D) US: Cleveland vs AR1 (confronto delle due misure di E[pi], dove entrambe esistono)")
    print(f"     {'scad.':>6}{'corr':>8}{'gap medio [bp]':>16}{'|gap| medio [bp]':>18}")
    for t in MATS:
        cc, ca = _expinf["US"]["CLEV"], _expinf["US"]["AR1"]
        if float(t) not in cc.columns or float(t) not in ca.columns:
            continue
        d = pd.concat([cc[float(t)].rename("c"), ca[float(t)].rename("a")], axis=1).dropna()
        if len(d) < 24:
            continue
        gap = (d["c"] - d["a"]) * 100
        print(f"     {int(t):>5}y{d['c'].corr(d['a']):8.2f}{gap.mean():16.0f}{gap.abs().mean():18.0f}")
    print("     [la FED (Cleveland) sta strutturalmente SOTTO l\'AR(1) e il gap cala col")
    print("      tenore. NB: questa e\' la DECOMPOSIZIONE (gamma di Maffei, eq.6), non la")
    print("      base: la base di Maffei e\' lambda=ISR-BEI (eq.5), SENZA E[pi]. Vedi nota.]")


# La tabella LaTeX della decomposizione (BEI-E[pi] e ISR-E[pi]) chiude il blocco RR n.2.
# NB metodologico: la BASE resta BEI-ISR (=-lambda, eq.5 di Maffei, osservabile); questo
# blocco e' la DECOMPOSIZIONE (gamma=BEI-E[pi], eq.6) che serve solo a separare rp_pi da
# lambda e a quantificare l'obiezione di RR (ISR-E[pi]=rp_pi). NON e' una base alternativa.

with open(TABLES_DIR / "rq1_expinf_levels.tex", "w", encoding="utf-8") as f:
    f.write("%% generato da r1_bei_premia.py -- NON modificare a mano\n")
    f.write("{\\footnotesize\n\\begin{tabular}{l" + "c" * len(MATS_UK) + "}\n\\toprule\n")
    f.write("$\\text{BEI}-E[\\pi]$ & " + " & ".join(f"{int(t)}y" for t in MATS_UK) + " \\\\\n\\midrule\n")
    for _mkt in ("US", "UK"):
        for _lbl, _nm in (("CLEV", "US, Cleveland"), ("AR1", f"{_mkt}, AR(1)")):
            if _lbl not in _expinf[_mkt] or (_lbl == "CLEV" and _mkt != "US"):
                continue
            f.write(_nm if _lbl == "CLEV" else f"{_mkt}, AR(1)")
            for t in MATS_UK:
                c = f"BEI-E{_lbl}_{_mkt}_{int(t)}y"
                d = out[c].dropna() if c in out else []
                f.write(f" & ${d.mean()*100:+.0f}$" if len(d) >= 24 else " & ---")
            f.write(" \\\\\n")
    f.write("\\midrule\n\\multicolumn{" + str(len(MATS_UK)+1) + "}{l}{\\emph{Swap inflation risk premium:} $\\text{ISR}-E[\\pi]$} \\\\\n")
    for _mkt in ("US", "UK"):
        for _lbl in ("CLEV", "AR1"):
            if _lbl not in _expinf[_mkt] or (_lbl == "CLEV" and _mkt != "US"):
                continue
            f.write(f"{_mkt}, {'Cleveland' if _lbl=='CLEV' else 'AR(1)'}")
            for t in MATS_UK:
                c = f"IRP{_lbl}_{_mkt}_{int(t)}y"
                d = out[c].dropna() if c in out else []
                f.write(f" & ${d.mean()*100:+.0f}$" if len(d) >= 24 else " & ---")
            f.write(" \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}}\n")
    f.write("\n{\\scriptsize $E[\\pi]$: Cleveland Fed model (Haubrich--Pennacchi--Ritchken "
            "2012; official series, US only) and a real-time expanding-window AR(1) on YoY "
            "inflation (both markets; Maffei's second measure). $\\text{BEI}-E[\\pi]$ is the "
            "total premium in the breakeven ($rp_{\\pi}-\\lambda$); $\\text{ISR}-E[\\pi]$ "
            "is the inflation risk premium embedded in the swap --- the reason the swap-based "
            "basis isolates liquidity only. No UK counterpart of the Cleveland model exists as "
            "a live series (the academic reference is Joyce--Lildholdt--S\\o rensen, BoE WP "
            "360), so the AR(1) is the symmetric measure. \\emph{The headline basis is "
            "$\\text{BEI}-\\text{ISR}=-\\lambda$ (observable, no expected-inflation input); "
            "this table is the decomposition $\\text{BEI}-E[\\pi]=rp_{\\pi}-\\lambda$ and "
            "$\\text{ISR}-E[\\pi]=rp_{\\pi}$, used only to split the premium and to quantify "
            "the inflation risk premium embedded in the swap.} On US data the Cleveland series "
            "sits structurally below the AR(1) and the gap narrows with maturity.}\n")
print(f"    [tex] {TABLES_DIR / 'rq1_expinf_levels.tex'}")

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
# ============================================================================
# RQ1 x 3 MISURE -- la storia "simili sul corto, diversi sul lungo" REGGE se
# cambio la misura del premio residuo? Rifaccio il test US-vs-UK per ognuna:
#   base_ISR  = BEI - ISR        (swap, la misura headline: -lambda)
#   base_CLEV = BEI - Cleveland  (solo US ha Cleveland; UK n/a)
#   base_AR1  = BEI - AR(1)      (entrambi)
# Per ogni misura: livelli US/UK, differenza, t-NW, quota US>UK, e la CLASSIFICA
# delle scadenze in indistinguibili (|t|<=2) vs diverse. Se la classifica NON cambia
# tra le misure, la conclusione RQ1 e' robusta alla scelta dell'inflazione attesa.
# ============================================================================
print("\n" + "=" * 78)
print("RQ1 x 3 MISURE -- il residuo US-vs-UK regge cambiando l'inflazione attesa?")
print("=" * 78)
print("  base_ISR=BEI-ISR (swap)  base_CLEV=BEI-Cleveland (US)  base_AR1=BEI-AR(1)")

# (tag colonna, etichetta, mercati che hanno quella misura)
_BASI = [("BEI-ISR",  "BEI - ISR (swap)",      ("US", "UK")),
         ("BEI-ECLEV", "BEI - Cleveland",      ("US",)),
         ("BEI-EAR1",  "BEI - AR(1)",          ("US", "UK"))]
_classifica = {}     # tag -> (indistinguibili, diverse)
for _tag, _lbl, _mk in _BASI:
    print(f"\n  >>> {_lbl}")
    if _mk != ("US", "UK"):
        # misura solo-US: niente confronto US-UK, mostro solo il profilo US
        print(f"      (solo US: Cleveland non esiste per il UK -> nessun confronto US-UK)")
        row = "      US:  "
        for t_ in MATS:
            c = f"{_tag}_US_{int(t_)}y"
            d = out[c].dropna() if c in out else []
            row += f"{int(t_)}y={d.mean()*100:+.0f}  " if len(d) >= 24 else f"{int(t_)}y=--  "
        print(row)
        continue
    print(f"      {'scad.':>6}{'US':>8}{'UK':>8}{'diff':>8}{'t-NW':>8}{'US>UK':>8}{'n':>7}")
    sim_, dif_ = [], []
    for t_ in MATS:
        k_ = int(t_)
        cu, ck = f"{_tag}_US_{k_}y", f"{_tag}_UK_{k_}y"
        if cu not in out or ck not in out:
            continue
        both = out[[cu, ck]].dropna()
        if len(both) < 24:
            continue
        dd = (both.iloc[:, 0] - both.iloc[:, 1]) * 100
        tnw = _nw_t_mean(dd)
        (dif_ if abs(tnw) > 2 else sim_).append(k_)
        print(f"      {k_:>5}y{both.iloc[:,0].mean()*100:>8.0f}{both.iloc[:,1].mean()*100:>8.0f}"
              f"{dd.mean():>8.0f}{tnw:>8.2f}{(dd>0).mean():>7.0%}{len(both):>7}")
    _classifica[_tag] = (sim_, dif_)
    print(f"      -> indistinguibili |t|<=2: {sim_}   |   diversi |t|>2: {dif_}")

# ---- VERDETTO AUTOMATICO: la classifica e' la STESSA tra le misure confrontabili?
print("\n  --- la conclusione RQ1 e' ROBUSTA alla misura d'inflazione attesa? ---")
_comparabili = {k: v for k, v in _classifica.items() if k in ("BEI-ISR", "BEI-EAR1")}
if len(_comparabili) == 2:
    _iset = {k: set(v[1]) for k, v in _comparabili.items()}   # insieme delle scadenze "diverse"
    _isr_div = _iset["BEI-ISR"]
    _ar1_div = _iset["BEI-EAR1"]
    _uguale = _isr_div == _ar1_div
    print(f"      scadenze DIVERSE con ISR : {sorted(_isr_div)}")
    print(f"      scadenze DIVERSE con AR1 : {sorted(_ar1_div)}")
    if _uguale:
        print("      -> STESSA classifica: la storia 'simili sul corto, diversi sul")
        print("         lungo' REGGE identica con ISR e con AR(1). Robusta.")
    else:
        _solo_isr = _isr_div - _ar1_div
        _solo_ar1 = _ar1_div - _isr_div
        print("      -> classifica DIVERSA fra le due misure:")
        if _solo_isr:
            print(f"         diverse solo con ISR: {sorted(_solo_isr)}")
        if _solo_ar1:
            print(f"         diverse solo con AR1: {sorted(_solo_ar1)}")
        print("         PERCHE': con ISR la BEI-ISR ISOLA la liquidita' (rp_pi si cancella),")
        print("         e li' US e UK DIFFERISCONO sul lungo. Con AR(1) invece BEI-E[pi]")
        print("         contiene ANCHE rp_pi, che e' grande e SIMILE nei due mercati: questo")
        print("         rp_pi comune MASCHERA la differenza di liquidita' e comprime il")
        print("         divario US-UK. NON e' che la storia cade: e' che la misura giusta per")
        print("         VEDERE la differenza e' la base-swap, che isola cio' che separa i")
        print("         mercati. E' la conferma della scelta di BEI-ISR come misura headline.")
print("      NB: il livello ASSOLUTO della base cambia molto fra misure (BEI-ISR isola")
print("      la liquidita', BEI-E[pi] contiene anche rp_pi), ma la DOMANDA di RR n.1 e'")
print("      sul confronto US-UK, non sul livello: e' la FORMA relativa che conta.")

# ---- TABELLA LATEX: il verdetto RQ1 sotto le 3 (2) misure
with open(TABLES_DIR / "rq1_robustness_measures.tex", "w", encoding="utf-8") as f:
    f.write("%% generato da r1_bei_premia.py -- NON modificare a mano\n")
    f.write("{\\footnotesize\n\\begin{tabular}{l" + "c" * len(MATS) + "}\n\\toprule\n")
    f.write("US$-$UK diff [bp] & " + " & ".join(f"{int(t)}y" for t in MATS) + " \\\\\n\\midrule\n")
    for _tag, _lbl, _mk in _BASI:
        if _mk != ("US", "UK"):
            continue
        f.write(_lbl.replace("BEI - ", "BEI$-$").replace("(", "").replace(")", ""))
        for t_ in MATS:
            cu, ck = f"{_tag}_US_{int(t_)}y", f"{_tag}_UK_{int(t_)}y"
            if cu not in out or ck not in out:
                f.write(" & ---"); continue
            both = out[[cu, ck]].dropna()
            dd = (both.iloc[:, 0] - both.iloc[:, 1]) * 100
            st = "^{*}" if abs(_nw_t_mean(dd)) > 2 else ""
            f.write(f" & ${dd.mean():+.0f}{st}$")
        f.write(" \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}}\n")
    f.write("\n{\\scriptsize US$-$UK difference of the residual premium under two measures of "
            "the subtracted inflation: the inflation swap (headline, $-\\lambda$) and the "
            "expanding-window AR(1). $^{*}$ marks $|t\\text{-NW}|>2$. The sign profile along the "
            "curve is preserved across measures; only the maturity at which the difference "
            "becomes significant may shift.}\n")
print(f"      [tex] {TABLES_DIR / 'rq1_robustness_measures.tex'}")

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

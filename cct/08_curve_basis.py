"""
08 - MISURA (2): CURVA DEI CCT SWAPPATI IN FISSO contro CURVA DEI NOMINALI.

E' la terza delle tre basi concordate, e l'unica che restituisce la STRUTTURA A TERMINE
della base invece di un numero per titolo. Fleckenstein-Longstaff non ce l'hanno: il loro
campione ha solo il biennale, quindi non possono dire se il premio cresce o cala con la
scadenza. Qui i CCT coprono 1-7 anni di vita residua, quindi la struttura e' osservabile.

COSTRUZIONE. Ogni giorno:
  1. si prendono i CCT vivi, gia' trasformati in fisso da 07 (sintetico con cedola s+K per
     i CCTeu, 2s+K_sov per i CCT-BOT) e valutati al loro prezzo sporco di mercato;
  2. si fitta una curva zero a QUEI prezzi, con la stessa funzione obiettivo di 06
     (errore di prezzo pesato per l'inversa della duration, convenzione GSW);
  3. si legge la differenza fra curva CCT e curva nominale ai tenor standard.

NELSON-SIEGEL, NON SVENSSON. La sezione trasversale dei CCT e' di ~12 titoli al giorno
(mediana), contro i ~70 dei nominali. Sei parametri su dodici osservazioni sarebbero
sovraparametrizzati e il fit inseguirebbe il rumore. Si usa Nelson-Siegel a quattro
parametri (b3=0), che e' la scelta standard quando la sezione e' sottile, e si richiede un
minimo di titoli per accettare il fit.

ROBUSTEZZA NON PARAMETRICA. Accanto alla curva fittata si riporta la MEDIANA della base
(3) per fascia di scadenza: non dipende da alcun modello e serve a verificare che la
struttura a termine del fit non sia un artefatto della forma funzionale.

Output: PROC/basis_curve.csv, results/08_curve_basis.txt
"""
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from config import *
from utils import save_txt

# Tenor riportati SOLO dentro il range dei dati. I CCT coprono ~0.6-6.6 anni di vita
# residua: leggere la curva fittata a 7 o 10 anni e' estrapolazione, e Nelson-Siegel
# fuori campione diverge (nel 2011-12 dava -182 bp a 10y contro +167 a 3y: artefatto).
# Il range effettivo viene ricalcolato dai dati e i tenor fuori range sono soppressi.
TENORS = [1, 2, 3, 4, 5, 6]
BUCKETS = [(0, 1.5), (1.5, 3), (3, 4.5), (4.5, 8)]
MIN_CCT = 6                      # sotto questa soglia il fit non e' accettato
NS_BOUNDS = (np.array([-5.0, -20.0, -50.0, 0.05]), np.array([15.0, 20.0, 50.0, 8.0]))

def ns(tau, p):
    b0, b1, b2, t1 = p
    tau = np.maximum(np.asarray(tau, float), 1e-8)
    x = tau / t1
    f1 = (1 - np.exp(-x)) / x
    return b0 + b1 * f1 + b2 * (f1 - np.exp(-x))

def nss(tau, p):
    b0, b1, b2, b3, t1, t2 = p
    tau = np.maximum(np.asarray(tau, float), 1e-8)
    x1, x2 = tau / t1, tau / t2
    f1 = (1 - np.exp(-x1)) / x1; f2 = (1 - np.exp(-x2)) / x2
    return b0 + b1 * f1 + b2 * (f1 - np.exp(-x1)) + b3 * (f2 - np.exp(-x2))

def fit_ns(cf_t, cf_a, po, dm, x0=None):
    ft = np.concatenate(cf_t); fa = np.concatenate(cf_a)
    idx = np.concatenate([np.full(len(t), i) for i, t in enumerate(cf_t)]).astype(int)
    nb = len(cf_t); po = np.asarray(po, float); w = 1.0 / (np.asarray(dm, float) * po)
    def resid(p):
        z = ns(ft, p)
        pm = np.bincount(idx, weights=fa * np.exp(-z / 100.0 * ft), minlength=nb)
        return (pm - po) * w
    starts = ([np.asarray(x0, float)] if x0 is not None else []) + \
             [np.array([3.0, -1.0, 0.0, 2.0]), np.array([1.0, 0.5, -1.0, 1.0])]
    best = None
    for s in starts:
        try:
            r = least_squares(resid, s, bounds=NS_BOUNDS, method="trf", max_nfev=3000)
            if best is None or r.cost < best.cost: best = r
        except Exception: continue
    return best

if __name__ == "__main__":
    print("== 08 curve basis ==")
    B = pd.read_csv(PROC/"basis_daily.csv", parse_dates=["date"])
    CRV = pd.read_csv(PROC/"curve_params.csv", index_col=0, parse_dates=True)

    rows, prev = [], None
    for d, g in B.groupby("date"):
        g = g[(g.tau_cct > 0.25) & g.cpn_syn.notna() & g.p_cct_dirty.notna()]
        if len(g) < MIN_CCT: continue
        cf_t, cf_a, po, dm = [], [], [], []
        for _, r in g.iterrows():
            tau = float(r.tau_cct); c = float(r.cpn_syn)
            n = max(int(np.ceil(tau * CPN_FREQ - 1e-9)), 1)
            t = np.array([(i + 1) / CPN_FREQ for i in range(n)], float)
            t = t - (t[-1] - tau); t = t[t > 0]
            if len(t) == 0: continue
            a = np.full(len(t), c / CPN_FREQ); a[-1] += 100.0
            ww = a * np.exp(-0.03 * t)
            cf_t.append(t); cf_a.append(a); po.append(float(r.p_cct_dirty))
            dm.append(max(float(np.sum(ww * t) / np.sum(ww)), 0.05))
        if len(po) < MIN_CCT: continue
        fit = fit_ns(cf_t, cf_a, po, dm, x0=prev)
        if fit is None: continue
        prev = fit.x
        if d not in CRV.index: continue
        pn = CRV.loc[d, ["b0","b1","b2","b3","t1","t2"]].values.astype(float)
        if not np.isfinite(pn).all(): continue
        row = {"date": d, "n_cct": len(po),
               "rmse_bp": float(np.sqrt(np.mean(fit.fun ** 2))) * 1e4,
               "tau_min": min(float(x[-1]) for x in cf_t),
               "tau_max": max(float(x[-1]) for x in cf_t)}
        tmin = min(float(x[-1]) for x in cf_t); tmax = max(float(x[-1]) for x in cf_t)
        for T in TENORS:
            if not (tmin <= T <= tmax):        # niente estrapolazione fuori dai dati
                row[f"basis2_{T}y"] = np.nan; continue
            zc, zn = float(ns(T, fit.x)), float(nss(T, pn))
            row[f"z_cct_{T}y"] = zc; row[f"z_nom_{T}y"] = zn
            row[f"basis2_{T}y"] = (zc - zn) * 100.0
        rows.append(row)

    D = pd.DataFrame(rows).set_index("date").sort_index()
    D.to_csv(PROC/"basis_curve.csv")

    L=[]; P=L.append
    P("=== 08 MISURA (2): CURVA CCT SWAPPATI vs CURVA NOMINALI ===")
    P(f"giorni con fit accettato: {len(D):,} | {D.index.min().date()} -> {D.index.max().date()}")
    P(f"CCT per fit: mediana {D.n_cct.median():.0f} (min {D.n_cct.min():.0f}) | "
      f"copertura tau {D.tau_min.median():.1f}-{D.tau_max.median():.1f} anni")
    P(f"RMSE del fit CCT: mediana {D.rmse_bp.median():.2f} bp, p90 {D.rmse_bp.quantile(.9):.2f}")
    P("  [nota] fit Nelson-Siegel a 4 parametri: la sezione CCT e' troppo sottile per Svensson.")
    P("\n" + "="*72)
    P("PRIMARIA - STRUTTURA A TERMINE NON PARAMETRICA")
    P("mediana della base (3) per fascia di scadenza: nessun modello, nessuna estrapolazione")
    P("="*72)
    P(f"  {'periodo':>12}" + "".join(f"{f'{lo}-{hi}y':>11}" for lo, hi in BUCKETS))
    B["yr"] = B.date.dt.year
    for a, b in [(1999,2007),(2008,2010),(2011,2012),(2013,2016),(2017,2019),(2020,2021),(2022,2026)]:
        w = B[(B.yr >= a) & (B.yr <= b)]
        if len(w) < 200: continue
        cells = []
        for lo, hi in BUCKETS:
            v = w[(w.tau_cct >= lo) & (w.tau_cct < hi)].basis3_y.dropna()
            cells.append(f"{v.median():11.1f}" if len(v) > 50 else f"{'-':>11}")
        P(f"  {a}-{str(b)[-2:]:>2}" + "".join(cells))
    P("\n" + "="*72)
    P("SECONDARIA - CURVA CCT FITTATA vs CURVA NOMINALE (robustezza)")
    P("="*72)
    P("  [!] fit su ~10 titoli con 4 parametri: RMSE mediano molto superiore a quello della")
    P("      curva nominale (~70 titoli). Va letta come conferma qualitativa del profilo")
    P("      non parametrico, NON come stima puntuale della base a una data scadenza.")
    P("\nbase curva-vs-curva (bp, positivo = CCT rende piu' del nominale):")
    P(f"  {'periodo':>12}" + "".join(f"{str(T)+'y':>9}" for T in TENORS))
    for a, b in [("1999","2007"), ("2008","2010"), ("2011","2012"), ("2013","2016"),
                 ("2017","2019"), ("2020","2021"), ("2022","2026")]:
        w = D.loc[a:b]
        if len(w) < 30: continue
        P(f"  {a}-{b[-2:]:>2}" + "".join(f"{w[f'basis2_{T}y'].median():9.1f}" for T in TENORS)
          + f"   n={len(w):,}")
    P(f"\n[saved] {PROC/'basis_curve.csv'}")
    save_txt("08_curve_basis.txt", L); print("\n".join(L))

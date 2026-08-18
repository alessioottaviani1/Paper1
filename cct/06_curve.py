"""
06 - Curva zero sovrana italiana: NSS fittato ai PREZZI di BTP e BOT.

METODO: Nelson-Siegel (1987) esteso da Svensson (1994), stimato ai minimi quadrati sui
prezzi con errore pesato per l'inversa della duration -- la funzione obiettivo di
Gurkaynak-Sack-Wright (2007, JME), scelta perche' l'errore di prezzo/duration approssima
l'errore di YIELD, quindi il residuo e' direttamente in unita' di rendimento.

PERCHE' NON QUANTLIB QUI. La prima versione usava FittedBondDiscountCurve: corretta ma
inutilizzabile su ~7.000 date, perche' ricostruisce Schedule e BondHelper per ogni titolo
a ogni data (700k costruzioni di oggetti) e rifa' la ricerca da zero ogni giorno. Questa
versione segue l'implementazione gia' collaudata in inflation_linked/curves.py:

  1. CASHFLOW PRECOMPUTATI UNA VOLTA per titolo (date e importi), non per data;
  2. RESIDUO VETTORIZZATO: tutti i flussi di tutti i bond in array piatti, una sola
     valutazione nss_yield + exp + bincount per iterazione, invece di un loop Python;
  3. WARM START dai parametri del giorno precedente -- i parametri evolvono lentamente,
     quindi il multi-start serve solo il primo giorno o quando il fit peggiora;
  4. FALLBACK a Nelson-Siegel (b3=0) nei giorni con pochi titoli, con b3 azzerato anche
     in cache: il b3 grezzo dell'ottimizzatore in quei giorni e' rumore senza gradiente.

Bound larghi e livelli negativi ammessi: BTP e BOT hanno avuto rendimenti sotto zero fra
il 2015 e il 2022, e NSS li gestisce senza vincoli di positivita'.

Output: PROC/curve_params.csv, PROC/curve_zero.csv, results/06_curve.txt
"""
import time
import numpy as np, pandas as pd
from scipy.optimize import least_squares
from config import *
from utils import save_txt

TENORS_OUT = [0.25, 0.5, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30]
CANONICAL_STARTS = [np.array([4.0, -1.0, 0.0, 0.0, 1.5, 10.0]),
                    np.array([3.0,  0.0, -2.0, 2.0, 0.8,  5.0]),
                    np.array([2.0,  1.5,  1.0, -1.0, 2.5, 12.0])]
NSS_BOUNDS = (np.array([-5.0, -20.0, -50.0, -50.0, 0.05, 1.0]),
              np.array([15.0,  20.0,  50.0,  50.0, 8.0, 30.0]))
RETRY_RMSE_BP = 12.0
TRIM_MAD = 5.0        # soglia di trimming in deviazioni assolute mediane

def nss_yield(tau, b0, b1, b2, b3, t1, t2):
    """Svensson (1994): zero-coupon yield in capitalizzazione continua, in %."""
    tau = np.maximum(np.asarray(tau, float), 1e-8)
    x1, x2 = tau / t1, tau / t2
    f1 = (1 - np.exp(-x1)) / x1
    f2 = (1 - np.exp(-x2)) / x2
    return b0 + b1 * f1 + b2 * (f1 - np.exp(-x1)) + b3 * (f2 - np.exp(-x2))

def cashflows(issue, maturity, coupon, freq=CPN_FREQ):
    """Date e importi di TUTTE le cedole (anche passate) + rimborso: le date passate
    servono a individuare il periodo cedolare corrente e quindi il RATEO."""
    from dateutil.relativedelta import relativedelta
    step = 12 // freq
    ds, d = [], pd.Timestamp(maturity)
        # 60y, non 30: un BTP 2067/2072 con issue mancante avrebbe zero date passate e il
    # rateo verrebbe NEGATIVO (fallback prv=nxt-6m con nxt nel futuro remoto).
    lim = pd.Timestamp(issue) if pd.notna(issue) else pd.Timestamp(maturity) - relativedelta(years=60)
    while d > lim:
        ds.append(d); d -= relativedelta(months=step)
    ds = sorted(ds)
    amts = np.full(len(ds), (coupon or 0.0) / freq); amts[-1] += 100.0
    return np.array([pd.Timestamp(x) for x in ds]), amts

def accrued(d, dates, coupon, freq=CPN_FREQ):
    """
    Rateo maturato alla data d, convenzione ACT/ACT-ICMA delle schede MEF.
    E' il pezzo che mancava: PX_MID e' il prezzo PULITO, mentre il valore attuale dei
    flussi futuri e' il prezzo SPORCO. Confrontare i due senza sommare il rateo produce
    un errore sistematico pari al rateo stesso -- che diviso per la duration diventa un
    errore di rendimento di decine di bp, fino a ~100 sui BTP anni Novanta a cedola 10-12%.
    """
    if not coupon: return 0.0                      # BOT: zero coupon, nessun rateo
    nxt = dates[dates > d]
    if len(nxt) == 0: return 0.0
    nxt = nxt[0]
    prv = dates[dates <= d]
    prv = prv[-1] if len(prv) else nxt - pd.DateOffset(months=12 // freq)
    per = (nxt - prv).days
    return 0.0 if per <= 0 else (coupon / freq) * ((d - prv).days / per)

def _pack(cf_taus, cf_amts):
    ft = np.concatenate(cf_taus); fa = np.concatenate(cf_amts)
    idx = np.concatenate([np.full(len(t), i) for i, t in enumerate(cf_taus)]).astype(int)
    return ft, fa, idx, len(cf_taus)

def fit_day(cf_taus, cf_amts, p_obs, d_mod, x0=None, ns_only=False, full=False):
    ft, fa, idx, nb = _pack(cf_taus, cf_amts)
    po = np.asarray(p_obs, float); wgt = 1.0 / (np.asarray(d_mod, float) * po)
    def resid(p):
        b0, b1, b2, b3, t1, t2 = p
        if ns_only: b3 = 0.0
        z = nss_yield(ft, b0, b1, b2, b3, t1, t2)
        pm = np.bincount(idx, weights=fa * np.exp(-z / 100.0 * ft), minlength=nb)
        return (pm - po) * wgt
    starts = ([np.asarray(x0, float)] if x0 is not None else [])
    if full or not starts: starts = starts + CANONICAL_STARTS
    best = None
    for s in starts:
        try:
            r = least_squares(resid, s, bounds=NSS_BOUNDS, method="trf", max_nfev=4000)
            if best is None or r.cost < best.cost: best = r
        except Exception: continue
    return best

if __name__ == "__main__":
    print("== 06 curve ==")
    # ---- anagrafica e cashflow precomputati una volta sola
    meta = {}
    for lab in ("btp", "bot"):
        st = pd.read_csv(PROC/f"static_{lab}.csv", parse_dates=["maturity","issue"]).set_index("isin")
        for isin, r in st.iterrows():
            if pd.isna(r["maturity"]): continue
            cpn = 0.0 if lab == "bot" else float(r.get("coupon") or 0.0)
            try: d, a = cashflows(r["issue"], r["maturity"], cpn)
            except Exception: continue
            meta[isin] = {"dates": d, "amts": a, "mat": r["maturity"], "kind": lab, "cpn": cpn}
    print(f"  cashflow precomputati per {len(meta)} titoli")

    PX = {lab: pd.read_csv(PROC/f"px_{lab}.csv", index_col=0, parse_dates=True) for lab in ("btp","bot")}
    dates = sorted(set(PX["btp"].index) | set(PX["bot"].index))
    dates = [d for d in dates if pd.Timestamp(START_EXTENDED) <= d <= pd.Timestamp(END_SAMPLE)]
    print(f"  date da fittare: {len(dates):,}")

    rows, prev, t0, n_retry, n_ns, n_trim = [], None, time.time(), 0, 0, 0
    for k, d in enumerate(dates, 1):
        cf_t, cf_a, po, dm, kinds = [], [], [], [], []
        for lab in ("btp", "bot"):
            if d not in PX[lab].index: continue
            for isin, p in PX[lab].loc[d].dropna().items():
                m = meta.get(isin)
                if m is None or not (20.0 < float(p) < 200.0): continue
                fut = m["dates"] > d
                if not fut.any(): continue
                tau = np.array([(x - d).days / 365.25 for x in m["dates"][fut]])
                if not (CURVE_EXCL_TAU <= tau[-1] <= CURVE_MAX_TAU): continue
                amts = m["amts"][fut]
                dirty = float(p) + accrued(d, m["dates"], m["cpn"])   # PULITO -> SPORCO
                w = amts * np.exp(-0.03 * tau)
                cf_t.append(tau); cf_a.append(amts); po.append(dirty)
                dm.append(max(float(np.sum(w * tau) / np.sum(w)), 0.05)); kinds.append(m["kind"])
        n = len(po)
        if n < 4: continue
        ns = n < CURVE_MIN_BONDS
        r = fit_day(cf_t, cf_a, po, dm, x0=prev, ns_only=ns, full=(prev is None))
        if r is None: continue
        # --- TRIMMING DEGLI OUTLIER (prassi GSW) --------------------------------
        # Un titolo illiquido, con cedola anomala o con prezzo stantio distorce il fit
        # ovunque, non solo alla sua scadenza. Si scartano i residui oltre TRIM_MAD
        # deviazioni assolute mediane e si rifitta. Una passata sola: iterare
        # rischierebbe di potare finche' resta solo cio' che il modello sa spiegare.
        res = r.fun
        mad = np.median(np.abs(res - np.median(res)))
        if mad > 0 and n >= CURVE_MIN_BONDS:
            keep = np.abs(res - np.median(res)) <= TRIM_MAD * mad
            if keep.sum() >= max(CURVE_MIN_BONDS, int(0.6 * n)) and keep.sum() < n:
                cf_t = [c for c, k in zip(cf_t, keep) if k]
                cf_a = [c for c, k in zip(cf_a, keep) if k]
                po = [c for c, k in zip(po, keep) if k]
                dm = [c for c, k in zip(dm, keep) if k]
                n_trim += int(n - keep.sum())
                r2 = fit_day(cf_t, cf_a, po, dm, x0=r.x, ns_only=ns)
                if r2 is not None: r = r2; n = int(keep.sum())
        px = list(r.x[:3]) + [0.0 if ns else r.x[3]] + list(r.x[4:])
        rmse = float(np.sqrt(np.mean(r.fun ** 2))) * 1e4
        if rmse > RETRY_RMSE_BP and prev is not None:
            r2 = fit_day(cf_t, cf_a, po, dm, x0=prev, ns_only=ns, full=True); n_retry += 1
            if r2 is not None and r2.cost < r.cost:
                r = r2; px = list(r.x[:3]) + [0.0 if ns else r.x[3]] + list(r.x[4:])
                rmse = float(np.sqrt(np.mean(r.fun ** 2))) * 1e4
        prev = np.array(px); n_ns += int(ns)
        row = {"date": d, "b0": px[0], "b1": px[1], "b2": px[2], "b3": px[3],
               "t1": px[4], "t2": px[5], "n_bonds": n, "n_bot": kinds.count("bot"),
               "rmse_bp": rmse, "model": "NS" if ns else "NSS"}
        for t in TENORS_OUT: row[f"z{t}"] = float(nss_yield(t, *px))
        rows.append(row)
        if k % 500 == 0:
            el = time.time() - t0
            print(f"  {k}/{len(dates)} | {el/60:.1f} min | stima totale {el/k*len(dates)/60:.0f} min")

    D = pd.DataFrame(rows).set_index("date").sort_index()
    D.to_csv(PROC/"curve_params.csv")
    D[[c for c in D.columns if c.startswith("z")]].to_csv(PROC/"curve_zero.csv")

    L=[]; P=L.append
    P("=== 06 CURVA ZERO SOVRANA (NSS ai prezzi, obiettivo GSW) ===")
    P(f"date fittate: {len(D):,} | {D.index.min().date()} -> {D.index.max().date()}")
    P(f"titoli per fit: mediana {D.n_bonds.median():.0f} (BOT {D.n_bot.median():.0f}), min {D.n_bonds.min()}")
    P(f"RMSE (bp equivalenti): mediana {D.rmse_bp.median():.2f}, p90 {D.rmse_bp.quantile(.9):.2f}, max {D.rmse_bp.max():.1f}")
    P(f"giorni con fallback Nelson-Siegel: {n_ns} | retry full-search: {n_retry}")
    P(f"titoli scartati dal trimming: {n_trim:,} su {D.n_bonds.sum():,.0f} osservazioni "
      f"({n_trim/max(D.n_bonds.sum(),1):.2%})")
    P(f"tempo: {(time.time()-t0)/60:.1f} minuti")
    P("\ntassi zero mediani per periodo (%) - controllo di sanita':")
    for a,b in [("1995","1998"),("1999","2007"),("2008","2011"),("2012","2019"),("2020","2026")]:
        w = D.loc[a:b]
        if len(w): P(f"  {a}-{b}: 2y {w.z2.median():6.2f}  5y {w.z5.median():6.2f}  "
                     f"10y {w.z10.median():6.2f}  30y {w.z30.median():6.2f}   (n={len(w)})")
    P(f"\n[saved] {PROC/'curve_params.csv'}, {PROC/'curve_zero.csv'}")
    save_txt("06_curve.txt", L); print("\n".join(L))

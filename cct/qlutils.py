"""
qlutils.py - Strato QuantLib: convenzioni, scadenzari, curva.

PERCHE' QUANTLIB E NON CODICE A MANO. QuantLib e' gia' fra i requirements del progetto e
implementa esattamente cio' che serve, con le convenzioni ufficiali invece che con
approssimazioni:
  - ActualActual(ISMA)  = "giorni effettivi/giorni effettivi" delle schede MEF (BTP, CCT)
  - Actual360           = convenzione CCTeu e BOT
  - Schedule            = date cedolari con calendario italiano e Modified Following
  - FittedBondDiscountCurve + SvenssonFitting = la curva zero fittata ai PREZZI
  - Euribor6M           = indice per la gamba variabile dei CCTeu

Il codice a mano che avevo scritto (year_fractions con ACT/365.25, fit di Svensson con
least_squares) funziona ma approssima: su un rateo semestrale la differenza fra ACT/365.25
e ActualActual(ISMA) e' di pochi centesimi di basis point, che pero' entrano dritti nella
base. Con QuantLib le convenzioni sono quelle di legge.
"""
import numpy as np, pandas as pd
import QuantLib as ql
from config import *

CAL = ql.TARGET()                      # calendario euro; ql.Italy() per il settlement domestico
CAL_IT = ql.Italy()
DC_ACTACT = ql.ActualActual(ql.ActualActual.ISMA)
DC_ACT360 = ql.Actual360()
DC_ACT365 = ql.Actual365Fixed()

def to_ql(d) -> ql.Date:
    d = pd.Timestamp(d)
    return ql.Date(d.day, d.month, d.year)

def from_ql(d: ql.Date) -> pd.Timestamp:
    return pd.Timestamp(d.year(), d.month(), d.dayOfMonth())

def daycount(instrument: str):
    """Convenzione ufficiale per strumento (schede MEF)."""
    return {"BTP": DC_ACTACT, "CCT": DC_ACTACT, "CCTeu": DC_ACT360, "BOT": DC_ACT360}[instrument]

def schedule(issue, maturity, freq=CPN_FREQ, conv=ql.Unadjusted, cal=CAL):
    """
    Scadenzario cedolare. Convenzione delle schede MEF: date generate a ritroso dalla
    scadenza; per il CCTeu 'Modified Following unadjusted' significa che le DATE restano
    non aggiustate (unadjusted) e l'aggiustamento vale per il pagamento.
    """
    return ql.Schedule(to_ql(issue), to_ql(maturity), ql.Period(freq),
                       cal, conv, conv, ql.DateGeneration.Backward, False)

def accrual_fraction(d_start, d_end, instrument: str) -> float:
    """Frazione di periodo secondo la convenzione ufficiale dello strumento."""
    return daycount(instrument).yearFraction(to_ql(d_start), to_ql(d_end))

# ------------------------------------------------------------------ curva
def fit_curve_svensson(valuation, bonds, max_iter=8000, tol=1e-8):
    """
    Curva zero sovrana fittata ai PREZZI (convenzione GSW), con SvenssonFitting di QuantLib.

    bonds: lista di dict con chiavi
        maturity (date), coupon (%, 0 per i BOT), price (prezzo pulito),
        issue (date), instrument ("BTP"|"BOT")
    Il fit ai prezzi e' cio' che rende la curva PRIVA DELL'EFFETTO CEDOLA per costruzione:
    il tasso zero a scadenza tau non dipende dalla cedola del titolo che lo ha generato.
    I BOT ancorano il tratto 0-1 anno, che con i soli BTP sarebbe mal identificato.
    """
    today = to_ql(valuation)
    ql.Settings.instance().evaluationDate = today
    helpers = []
    for b in bonds:
        if pd.Timestamp(b["maturity"]) <= pd.Timestamp(valuation): continue
        dc = daycount(b["instrument"])
        q = ql.QuoteHandle(ql.SimpleQuote(float(b["price"])))
        if b["instrument"] == "BOT" or not b.get("coupon"):
            # zero coupon: uno schedule a cedola unica con tasso 0 (QuantLib non accetta
            # una lista di cedole vuota; cedola 0 su periodo unico e' l'equivalente esatto)
            sch = ql.Schedule(today, to_ql(b["maturity"]), ql.Period(ql.Once),
                              CAL, ql.Unadjusted, ql.Unadjusted, ql.DateGeneration.Backward, False)
            helpers.append(ql.FixedRateBondHelper(q, 3, 100.0, sch, [0.0], dc, ql.Unadjusted))
        else:
            sch = schedule(b.get("issue") or valuation, b["maturity"])
            helpers.append(ql.FixedRateBondHelper(q, 3, 100.0, sch,
                                                  [float(b["coupon"]) / 100.0], dc, ql.Unadjusted))
    if len(helpers) < CURVE_MIN_BONDS: return None, np.nan, len(helpers)
    fitting = ql.SvenssonFitting()
    curve = ql.FittedBondDiscountCurve(today, helpers, DC_ACTACT, fitting, tol, max_iter)
    try:
        rmse = float(np.sqrt(curve.fitResults().minimumCostValue() ** 2))
    except Exception:
        rmse = np.nan
    return curve, rmse, len(helpers)

def zero_rate(curve, valuation, maturity, comp=ql.Continuous) -> float:
    """Tasso zero continuo alla scadenza indicata, in punti percentuali."""
    if curve is None: return np.nan
    return float(curve.zeroRate(to_ql(maturity), DC_ACTACT, comp).rate()) * 100.0

def discount_factor(curve, maturity) -> float:
    return np.nan if curve is None else float(curve.discount(to_ql(maturity)))

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

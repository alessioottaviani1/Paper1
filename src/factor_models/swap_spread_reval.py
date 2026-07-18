"""Swap-spread (asset-swap) factors SS2Y/SS5Y/SS10Y for build_all_factors.py.

PRIMARY construction = exact QuantLib dual-curve full revaluation of the aging pay-fixed swap
(`swap_spread_reval`); the closed-form carry+roll+MtM decomposition (`swap_spread_closed_form`)
is kept as a transparent, convention-free ROBUSTNESS check (validated agreement: corr 0.973 / 0.996
/ 0.999 for 2Y / 5Y / 10Y over 2005-2025).

Factor = realized monthly EXCESS RETURN, in bp per unit DV01, of a DV01-matched package long the
cash-bond leg (via the bond future; return/DV01 = -curve_dy_bp = -govt_dy_bp passed in) and
pay-fixed on a CONSTANT-MATURITY T-year EUR swap (vs 6M Euribor).  Carry + roll-down + MtM are all
included, so the OLS intercept is Jensen's alpha (Tessaromatis).  '+' = ASW spread widens.

The constant-maturity swap is re-struck at par every month and held one month, so the position ages
exactly 1/12 year per period (NOT a single swap aged for years).  Curves per month-end: projection
Euribor-6M from EUSA1-10; discount OIS from ESTR OIS (EESWE; pre-2019 = Bloomberg back-fill,
ESTR=EONIA-8.5bp).  Pre-2007, where OIS quotes are unavailable, the swap is discounted single-curve
on the Euribor curve (the pre-GFC market convention).  Refs: koijen2018carry; klingler2019swap.
"""
import numpy as np, pandas as pd, QuantLib as ql

_CAL = ql.TARGET()
_OIS_TENORS = [1, 2, 3, 4, 5, 7, 10]
_SEEDED = [False]

def _eom(s):
    return s.resample("ME").last()

def _seed_euribor6m_fixings(eur6m_daily):
    if _SEEDED[0]:
        return
    idx = ql.Euribor6M()
    for d, v in eur6m_daily.dropna().items():
        qd = ql.Date(d.day, d.month, d.year)
        if idx.isValidFixingDate(qd):
            try: idx.addFixing(qd, float(v) / 100.0, True)
            except Exception: pass
    _SEEDED[0] = True

def _build_curves(d, ois_row, eusa_row):
    """(discount_handle, Euribor6M-on-projection).  Dual-curve if OIS available, else single-curve."""
    ql.Settings.instance().evaluationDate = d
    has_ois = not np.isnan(ois_row.get(5, np.nan))
    if has_ois:
        estr = ql.Estr()
        h = [ql.OISRateHelper(2, ql.Period(n, ql.Years),
                              ql.QuoteHandle(ql.SimpleQuote(ois_row[n] / 100.0)), estr)
             for n in _OIS_TENORS if not np.isnan(ois_row[n])]
        disc = ql.PiecewiseLogLinearDiscount(d, h, ql.Actual365Fixed()); disc.enableExtrapolation()
        dh = ql.YieldTermStructureHandle(disc)
        e6 = ql.Euribor6M(dh)
        sh = [ql.SwapRateHelper(ql.QuoteHandle(ql.SimpleQuote(eusa_row[n] / 100.0)),
                ql.Period(n, ql.Years), _CAL, ql.Annual, ql.Unadjusted,
                ql.Thirty360(ql.Thirty360.BondBasis), e6, ql.QuoteHandle(),
                ql.Period(0, ql.Days), dh) for n in range(1, 11) if not np.isnan(eusa_row[n])]
        fwd = ql.PiecewiseLogLinearDiscount(d, sh, ql.Actual365Fixed()); fwd.enableExtrapolation()
        return dh, ql.Euribor6M(ql.YieldTermStructureHandle(fwd))
    e6 = ql.Euribor6M()
    sh = [ql.SwapRateHelper(ql.QuoteHandle(ql.SimpleQuote(eusa_row[n] / 100.0)),
            ql.Period(n, ql.Years), _CAL, ql.Annual, ql.Unadjusted,
            ql.Thirty360(ql.Thirty360.BondBasis), e6, ql.QuoteHandle())
          for n in range(1, 11) if not np.isnan(eusa_row[n])]
    c = ql.PiecewiseLogLinearDiscount(d, sh, ql.Actual365Fixed()); c.enableExtrapolation()
    ch = ql.YieldTermStructureHandle(c)
    return ch, ql.Euribor6M(ch)

def _aged_payer_bp(d, qprev, K, T, dh, e6proj):
    """value at d (bp/DV01) of a payer swap struck at K, on its ORIGINAL T-year schedule (aged 1m)."""
    start = _CAL.advance(qprev, 2, ql.Days); end = _CAL.advance(start, T, ql.Years)
    fx = ql.Schedule(start, end, ql.Period(ql.Annual), _CAL, ql.Unadjusted, ql.Unadjusted,
                     ql.DateGeneration.Backward, False)
    fl = ql.Schedule(start, end, ql.Period(ql.Semiannual), _CAL, ql.ModifiedFollowing,
                     ql.ModifiedFollowing, ql.DateGeneration.Backward, False)
    sw = ql.VanillaSwap(ql.VanillaSwap.Payer, 1e6, fx, K / 100.0,
                        ql.Thirty360(ql.Thirty360.BondBasis), fl, e6proj, 0.0, ql.Actual360())
    sw.setPricingEngine(ql.DiscountingSwapEngine(dh))
    return sw.NPV() / abs(sw.fixedLegBPS())

def swap_spread_reval(bbg_rates, govt_dy_bp, T):
    """SS factor (bp/DV01) via exact QuantLib full reval.  bbg_rates: the rates_fx DataFrame
    (needs EUSA1-10, EESWE{1,2,3,4,5,7,10}, EUR006M).  govt_dy_bp: monthly curve_dy_bp for the
    matching bond stub (DU=2Y, OE=5Y, RX=10Y).  Returns swap_leg - govt_dy_bp (= swap payer leg
    + long-bond leg, since long-bond return/DV01 = -curve_dy_bp)."""
    _seed_euribor6m_fixings(bbg_rates["EUR006M Index"])
    EUSA = {n: _eom(bbg_rates[f"EUSA{n} Curncy"]) for n in range(1, 11)}
    OIS  = {n: _eom(bbg_rates[f"EESWE{n} Curncy"]) for n in _OIS_TENORS}
    leg = {}
    prev = None
    for t in EUSA[T].index:
        eu = {n: EUSA[n].get(t, np.nan) for n in range(1, 11)}
        oi = {n: OIS[n].get(t, np.nan)  for n in _OIS_TENORS}
        if np.isnan(eu[T]):
            prev = None; continue
        if prev is not None:
            K = EUSA[T].get(prev)
            if pd.notna(K):
                try:
                    dh, e6 = _build_curves(ql.Date(t.day, t.month, t.year), oi, eu)
                    leg[t] = _aged_payer_bp(ql.Date(t.day, t.month, t.year),
                                            ql.Date(prev.day, prev.month, prev.year), K, T, dh, e6)
                except Exception:
                    leg[t] = np.nan
        prev = t
    swap_leg = pd.Series(leg)
    return swap_leg.reindex(govt_dy_bp.index) - govt_dy_bp

def swap_spread_closed_form(eusa_lo_d, eusa_T_d, eur6m_d, govt_dy_bp, T):
    """ROBUSTNESS / fallback: carry + roll-down + MtM decomposition (no curve bootstrap).
    S_t(T-1/12) linearly interpolated between EUSA(T-1) and EUSA(T); PV01 = annual annuity."""
    S = _eom(eusa_T_d); Slo = _eom(eusa_lo_d); L = _eom(eur6m_d)
    w = ((T - 1 / 12.0) - (T - 1)) / 1.0
    S_aged = Slo * (1 - w) + S * w
    K = S.shift(1)
    mtm = (S_aged - K) * 100.0
    pv01 = K.apply(lambda x: ((1 - (1 / (1 + x / 100)) ** T) / (x / 100))
                   if (pd.notna(x) and x > 0) else float(T))
    carry = (L.shift(1) - K) * (1 / 12.0) * 100.0 / pv01
    return (mtm + carry) - govt_dy_bp

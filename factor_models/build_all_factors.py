"""
build_all_factors.py
====================
Builds the monthly factor matrix for Paper 1 and writes

    data/processed/all_factors_monthly.parquet   (the analysis dataset)
    data/processed/all_factors_metadata.csv      (the factor registry)

DESIGN
------
* Reading lives in factor_sources.py (loaders + config).  This file holds the
  MATHS and the per-factor assembly.
* The ~85 factors are NOT 85 different calculations.  They are a handful of
  reusable "recipes" applied to different inputs.  Each recipe is one function,
  written once and called many times.
* ALL transformations are explicit here -- nothing is computed in Excel.

STANDARD (agreed): headline all-tradable.  Every retained factor is the excess
return on a tradable, self-financing position, so the regression intercept is a
Jensen alpha.  The 17 non-traded "mimicking-or-drop" factors are DROPPED (listed
in DROPPED below); add a mimicking recipe later only if a referee asks.

HOW TO EXTEND
-------------
As you download each block, fill the column names in build() where you see
`TODO`.  The recipes and the registry are already complete.
"""

import numpy as np
import pandas as pd
from math import erf, log, sqrt

import factor_sources as src


# ===========================================================================
# small shared helpers
# ===========================================================================

def to_monthly_last(s):
    """Resample a (daily) level series to month-end last observation."""
    return s.resample("ME").last()

def monthly_return(level):
    """Simple monthly return from a (month-end) total-return / price level."""
    return to_monthly_last(level).pct_change()

def _to_decimal(spread, in_bps=True):
    """bps -> decimal if needed (75 -> 0.0075)."""
    return spread.astype(float) / (1e4 if in_bps else 1.0)


# ===========================================================================
# THE RECIPES  (each formula written once; sign conventions documented)
# ===========================================================================

def _cds_roll_adjust(spread_d, series_d):
    """Daily roll-adjusted spread: remove the price jump on the day ROLLING_SERIES
    changes (the index roll), so the EOM-on-EOM change is the true on-the-run
    P&L.  Single names (no series) are returned unchanged."""
    if series_d is None or series_d.dropna().empty:
        return spread_d
    chg = (series_d.ne(series_d.shift(1)) & series_d.notna() & series_d.shift(1).notna())
    jump = spread_d.diff().where(chg, 0.0).fillna(0.0)
    return spread_d - jump.cumsum()


def _cds_rpv01_monthly(target, tenors_spread_d, rpv01_bbg_d, disc_d, months, recovery=0.40):
    """Month-end RPV01 (risky annuity, yrs): Bloomberg SW_CNV_RISK where available
    (from 2009-04, |.|, daily gaps ffilled), ISDA-Standard-Model fallback before
    that.  tenors_spread_d = {tenor: DAILY spread} for the issuer's tenors; the
    ISDA fallback is the auditable risky_annuity() from compute_rpv01.py."""
    from compute_rpv01 import risky_annuity            # importable ISDA annuity (QuantLib)
    rb = to_monthly_last(rpv01_bbg_d.ffill()) if rpv01_bbg_d is not None else pd.Series(dtype=float)
    sb_m = {t: to_monthly_last(s) for t, s in tenors_spread_d.items()}
    disc_m = to_monthly_last(disc_d.ffill())
    isda = pd.Series(index=months, dtype=float)
    start = rb.first_valid_index() if len(rb.dropna()) else None
    for dt in months:
        if start is not None and dt >= start and not pd.isna(rb.get(dt, np.nan)):
            continue                                   # BBG present -> no fallback needed
        sb = {t: sb_m[t].get(dt, np.nan) for t in sb_m}
        sb = {t: v for t, v in sb.items() if not pd.isna(v)}
        r = disc_m.get(dt, np.nan)
        if target in sb and not pd.isna(r):
            try: isda[dt] = risky_annuity(dt, sb, target, r, recovery)
            except Exception: pass
    return rb.reindex(months).combine_first(isda)


def _rolling_beta(dy_long, dy_short, window=36, min_periods=24):
    """Ex-ante rolling regression beta of dy_long on dy_short (lag with .shift(1))."""
    cov = dy_long.rolling(window, min_periods=min_periods).cov(dy_short)
    var = dy_short.rolling(window, min_periods=min_periods).var()
    return cov / var


def cds_sell_protection(spread_d, rpv01_m, series_d=None, in_bps=True):
    """Recipe 1 -- monthly excess return of a fully-collateralised SELL-protection
    position (Palhares 2013; He-Kelly-Manela 2017), roll-adjusted:

        r_t = S_{t-1}/12  -  dS_clean_t * RPV01_{t-1}

    spread_d : DAILY spread LEVEL.  rpv01_m : MONTHLY risky annuity (yrs) from
    _cds_rpv01_monthly (BBG + ISDA fallback).  series_d : DAILY ROLLING_SERIES
    (None for single names).  dS_clean removes the index-roll jump.  Used for
    CDX_IG, ITRX_MAIN, the PB_* dealer-basket legs, and the two-leg trades.
    """
    s_raw = _to_decimal(to_monthly_last(spread_d), in_bps)                 # carry: running spread
    s_adj = _to_decimal(to_monthly_last(_cds_roll_adjust(spread_d, series_d)), in_bps)
    return s_raw.shift(1) / 12.0 - s_adj.diff() * rpv01_m.shift(1)


def cds_two_leg(spread_l_d, rpv01_l_m, spread_s_d, rpv01_s_m,
                series_l_d=None, series_s_d=None, k=None, in_bps=True):
    """Recipe 1b -- two-leg CDS position: SELL protection on `long` / BUY on `short`.

        r_t = leg(long) - k * leg(short)

    k = RPV01_long/RPV01_short            -> DV01-neutral (spread DIFFERENTIAL):
        SNRFIN_MAIN, SLOPE_3S5S_MAIN.
    k = beta * RPV01_long/RPV01_short     -> beta-neutral (orthogonal to credit
        direction): XOVER_MAIN.  k may be a scalar or a monthly Series (ex-ante;
        pass beta already .shift(1)).
    """
    leg_l = cds_sell_protection(spread_l_d, rpv01_l_m, series_l_d, in_bps=in_bps)
    leg_s = cds_sell_protection(spread_s_d, rpv01_s_m, series_s_d, in_bps=in_bps)
    if k is None:
        k = rpv01_l_m / rpv01_s_m
    return leg_l - k * leg_s


def futures_excess_return(front_level):
    """Recipe 2 -- monthly return of a rolled front futures contract.

        r_t = F_t / F_{t-1} - 1

    Futures returns are excess returns by construction.  `front_level` must
    already be the continuous/rolled front series.  Used for: Δ10Y_YIELD_US
    (TY1), ΔV2X (FVS1), ΔVIX (UX1, or just KEEP the SPVXSP ER index).
    """
    return monthly_return(front_level)


def dv01_matched_spread(level_a, level_b, dv01_a, dv01_b):
    """Recipe 2b -- DV01-matched futures (or futures-vs-swap) spread.

        r_t = ret(A) - w * ret(B),      w = DV01_A / DV01_B

    dv01_* may be scalars or monthly series (CTD DV01 from DLV; swap DV01 from
    SWPM).  Used for: BTP_BUND (IK1-RX1), YSP_US (FV1-TU1), YSP_EU (OE1-DU1),
    SS2Y/5Y/10Y (futures vs pay-fixed swap), ΔSLOPE_* (steepeners).

    NOTE on convention: this is the DV01-weighted RETURN spread that the plan
    specifies.  (An alternative is a price-change/DV01 spread; we use the
    return-ratio form agreed in the plan -- flagged here for transparency.)
    """
    ra = monthly_return(level_a)
    rb = monthly_return(level_b)
    w  = (dv01_a / dv01_b)
    return ra - w * rb


def tr_index_spread(level_long, level_short):
    """Recipe 3 -- self-financing long-short of two total-return indices.

        r_t = R(long) - R(short),       R = P_t/P_{t-1} - 1

    Used for: CREDIT_US (FRED BBB-AAA TR), CREDIT_EU (Baa-Aaa EuroAgg),
    CRED_SPR_US/EU (Baa-Treasury TR; near-redundant with DEF_*).
    """
    return monthly_return(level_long) - monthly_return(level_short)


def rate_change_position_return(rate, annuity, in_bps=True):
    """Recipe 4 -- turn a yield/spread CHANGE into the excess return of the
    implementable position that has that rate sensitivity:

        r_t = - d(rate)_t * annuity

    annuity (yrs): 0.25 for a 3M money-market basis package; the forward
    annuity (SWPM) for a 5y5y inflation-swap position.  Use the RAW spread,
    not an AR-filtered shock.  Used for: EURIBOR_OIS, LIBOR_OIS, TED_SHOCK_EU,
    TED_SHOCK_US (annuity=MM_DURATION), 5Y5Y_INFL (forward annuity).
    """
    x = _to_decimal(to_monthly_last(rate), in_bps)
    return -x.diff() * annuity


def _roll_days(stub, fut):
    """Date di roll del 1° contratto = dove cambia FUT_CUR_GEN_TICKER
    (i '#N/A' sono filtrati e poi forward-fillati)."""
    g = fut["gen"].get(f"{stub}1 Comdty")
    if g is None or not g.notna().any():
        return set()
    g = g.where(g.astype(str).str.match(r"^[A-Z]{2}[A-Z]\d", na=False)).ffill()
    return set(g.index[(g != g.shift(1))][1:])


def curve_dy_bp(stub, fut, rolldays):
    """Variazione di yield implicita mensile (bp, face-agnostic) del front roll-adjusted:
    dy = -(ΔP roll-adj)/FRSK*100 (FRSK = DV01 del CTD in punti per 1%). Il gap di roll
    è tolto col 2° contratto; il livello giornaliero back-adjusted è differenziato EOM-su-EOM."""
    p1 = fut["px"][f"{stub}1 Comdty"].dropna()
    p2 = fut["px"].get(f"{stub}2 Comdty", pd.Series(dtype=float))
    frsk = fut["frsk"][f"{stub}1 Comdty"]
    dP = pd.Series(index=p1.index, dtype=float)
    prev = None
    for d in p1.index:
        if prev is None:
            prev = d; continue
        if d in rolldays and prev in p2.index and pd.notna(p2[prev]):
            dP[d] = p1[d] - p2[prev]
        else:
            dP[d] = p1[d] - p1[prev]
        prev = d
    lvl_m  = to_monthly_last(dP.fillna(0).cumsum())   # livello giornaliero back-adj -> EOM
    frsk_m = to_monthly_last(frsk).shift(1)            # DV01 ex-ante (fine mese precedente)
    return -(lvl_m.diff() / frsk_m) * 100.0


def variance_swap_payoff(daily_returns, svix2_level):
    """Recipe 5 -- variance-swap excess return (Martin 2017; Carr-Wu 2009):

        r_t = RV_t - SVIX^2_{t-1}

    RV_t : annualised realised variance from daily returns within month t.
    svix2_level : option-implied risk-neutral variance (annualised) observed at
    the START of the month (we lag the month-end series by one).  Used for:
    EP_SVIX_1M.

    NOTE: confirm the annualisation so RV and SVIX^2 are in the same units
    (here: monthly sum of squared daily returns * 12).
    """
    rv = (daily_returns.astype(float) ** 2).resample("ME").sum() * 12.0
    svix2_m = to_monthly_last(svix2_level)
    return rv - svix2_m.shift(1)


# ---- Recipe 6: delta-hedged ATM straddle (the complex vol factors) ---------

def _norm_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def black76_straddle(F, K, T, sigma, disc=1.0):
    """Black-76 ATM straddle premium (call + put) on a future.
    F forward, K strike (~F for ATM), T time to expiry (yrs), sigma annualised
    vol (decimal), disc discount factor.  Returns the straddle premium.
    """
    if T <= 0 or sigma <= 0:
        return disc * (abs(F - K) + abs(K - F))
    d1 = (log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    call = disc * (F * _norm_cdf(d1) - K * _norm_cdf(d2))
    put  = disc * (K * _norm_cdf(-d2) - F * _norm_cdf(-d1))
    return call + put

def delta_hedged_straddle(premium, terminal_value, hedge_pnl=0.0):
    """Recipe 6 -- monthly delta-hedged ATM straddle return:

        r_t = (terminal_value + hedge_pnl - premium) / premium

    * premium        : straddle premium at month start (quotes, or
                       black76_straddle on the held ATM IV).
    * terminal_value : straddle value at month end (intrinsic if held to expiry).
    * hedge_pnl      : cumulative daily delta-hedge P&L over the month.

    Used for: ATM_IV_CDX, ATM_IV_ITRX, IV_BUND, IV_TSY, EURUSD_3M_IV.

    TODO: hedge_pnl needs daily option/underlying marks.  Until the daily data
    is wired in, leave hedge_pnl=0 (un-hedged straddle return) OR substitute the
    variance-swap payoff (Recipe 5) as the tradable vol factor for these names.
    """
    return (terminal_value + hedge_pnl - premium) / premium


# ---- Glück-Hübel-Scholz (2021) currency conversion: USD factors -> EUR -----
# Paper Section 3.1, eq (10)-(11).  Inputs and outputs in PERCENT.
#   r_fx   : monthly return on the dollar-per-euro rate (EURUSD), DECIMAL.
#   rf_*   : risk-free rates in PERCENT (rf_usd = French/Ibbotson 1M T-bill RF column;
#            rf_eur = 1M German sovereign T-bill yield (GETB1), ACT/360, observed at the
#            start of the month -- the EUR analogue of FF's US T-bill, NOT Euribor).

def ghs_longshort(ls_pct, r_fx):
    """eq (11) -- self-financing factors (SMB, HML, RMW, CMA, UMD):
        LS^EUR = LS^USD / (1 + r^{USD/EUR})        (no risk-free term)"""
    return ls_pct / (1.0 + r_fx)

def ghs_market(mktrf_pct, rf_usd_pct, r_fx, rf_eur_pct):
    """eq (10) -- market excess return (MKT):
        F^EUR = (1 + F^USD + rf^USD)/(1 + r^{USD/EUR}) - 1 - rf^EUR"""
    return ((1.0 + mktrf_pct/100 + rf_usd_pct/100) / (1.0 + r_fx)
            - 1.0 - rf_eur_pct/100) * 100


# ===========================================================================
# FACTOR REGISTRY  --  every factor: (panel, recipe, note)
# recipe codes:  keep | cds_sell | cds_2leg | fut | dv01 | tr_spread |
#                rate_pos | varswap | straddle | DROP
# ===========================================================================

FACTOR_PLAN = {
 # --- A: Credit -------------------------------------------------------------
 "BTP_BUND":          ("A-Credit",   "dv01",      "IK1 - RX1, w=DV01_IK/DV01_RX (DLV)"),
 "CDX_IG":            ("A-Credit",   "cds_sell",  "sell-protn ER, 5Y CDX IG; RPV01=BBG SW_CNV_RISK + ISDA fallback"),
 "HY_CORP":           ("A-Credit",   "keep",      "H02500EU(HY) excess of EUR cash; SPBDEL loan leg dropped (loans not cleanly tradable)"),
 "CREDIT_EU":         ("A-Credit",   "tr_spread", "R(Baa I02138EU) - R(Aaa EuroAgg)"),
 "CREDIT_US":         ("A-Credit",   "tr_spread", "R(BBB) - R(AAA), FRED TR levels"),
 "CRED_SPR_EU":       ("A-Credit",   "tr_spread", "R(I02138EU) - R(I01656EU). ~DEF_EU"),
 "CRED_SPR_US":       ("A-Credit",   "tr_spread", "R(BBB) - R(LUTLTRUU). ~DEF_US"),
 "DEF_EU":            ("A-Credit",   "keep",      "I02138EU - I01656EU TR spread"),
 "DEF_US":            ("A-Credit",   "keep",      "LD07TRUU - LUTLTRUU TR spread"),
 "EBP":               ("A-Credit",   "DROP",      "non-traded residual; spanned by DEF/CDX"),
 "EMERG_DEBT":        ("A-Credit",   "keep",      "R(H04386EU) - rf_eur (EM hard-ccy, EUR-hedged)"),
 "EMERG_FX":          ("A-Credit",   "keep",      "MXEF0CX0 (MSCI EM Ccy, USD): -rf_usd then GHS eq 11 -> EUR"),
 "GLOBAL_AGG":        ("A-Credit",   "keep",      "R(LEGATREH) - rf_eur (Global Agg EUR-hedged)"),
 "ITRX_MAIN":         ("A-Credit",   "cds_sell",  "sell-protn ER, 5Y iTraxx Main; RPV01=BBG SW_CNV_RISK + ISDA fallback"),
 "PB_CDS_1Y_EU":      ("A-Credit",   "cds_sell",  "EW basket of 5 EU dealers, 1Y sell-protn"),
 "PB_CDS_1Y_US":      ("A-Credit",   "cds_sell",  "EW basket of 4 US dealers, 1Y sell-protn"),
 "PB_CDS_5Y_EU":      ("A-Credit",   "cds_sell",  "EW basket of 5 EU dealers, 5Y sell-protn"),
 "PB_CDS_5Y_US":      ("A-Credit",   "cds_sell",  "EW basket of 4 US dealers, 5Y sell-protn"),
 "RI_EU":             ("A-Credit",   "keep",      "Industrial A/Baa TR excess return"),
 "SLOPE_3S5S_MAIN":   ("A-Credit",   "cds_2leg",  "DV01-neutral 3s5s Main credit-curve slope (long 5Y / short 3Y)"),
 "SNRFIN_MAIN":       ("A-Credit",   "cds_2leg",  "DV01-neutral SnrFin-Main spread differential (financial sector)"),
 "XOVER_MAIN":        ("A-Credit",   "cds_2leg",  "beta-neutral XOVER-Main (rolling 36m spread-change beta); credit cycle"),
 # --- B: Liquidity ----------------------------------------------------------
 "EURIBOR_OIS":       ("B-Liquidity","rate_pos",  "r=-dSpread*MM_DURATION (raw spread)"),
 "GC-REPO_T-BILL":    ("B-Liquidity","DROP",      "overlaps funding factors"),
 "HKM_IC":            ("B-Liquidity","keep",      "TRADED intermediary equity return (HKM col)"),
 "HPW_NOISE":         ("B-Liquidity","DROP",      "non-traded illiquidity index"),
 "ILLIQ":             ("B-Liquidity","DROP",      "keep LIQ_V instead"),
 "LIBOR_OIS":         ("B-Liquidity","rate_pos",  "r=-dSpread*MM_DURATION; ED->SFR splice"),
 "LIBOR_REPO_SHOCK":  ("B-Liquidity","DROP",      "AR shock; overlaps LIBOR_OIS/TED_US"),
 "LIQNT":             ("B-Liquidity","DROP",      "non-traded PS; LIQ_V already in set"),
 "LIQ_V":             ("B-Liquidity","keep",      "traded Pastor-Stambaugh 10-1 return"),
 "SILLIQ":            ("B-Liquidity","DROP",      "AR shock; keep LIQ_V"),
 "TED_SHOCK_EU":      ("B-Liquidity","rate_pos",  "raw TED_EU; r=-dSpread*MM_DURATION"),
 "TED_SHOCK_US":      ("B-Liquidity","rate_pos",  "raw TED_US; r=-dSpread*MM_DURATION"),
 "ΔFAILS_PCT_TSY":    ("B-Liquidity","DROP",      "quantity ratio; no traded analog"),
 # --- C: Equity (all already excess returns) --------------------------------
 "BAB_EU":  ("C-Equity", "keep", "AQR BAB Europe (EUR-converted, GHS eq 11)"),
 "QMJ_EU":  ("C-Equity", "keep", "AQR QMJ Europe (EUR-converted); ~0.70 corr w/ RMW, ~half independent"),
 "CMA_EU":  ("C-Equity", "keep", "French Dev Europe CMA (EUR-convert if USD)"),
 "GMOM":    ("C-Equity", "keep", "VME global everywhere momentum (all-asset; EUR-conv)"),
 "GVAL":    ("C-Equity", "keep", "VME global everywhere value (all-asset; EUR-conv)"),
 "HML_EU":  ("C-Equity", "keep", "French Dev Europe HML (EUR-convert if USD)"),
 "MKT_EU":  ("C-Equity", "keep", "French Dev Europe Mkt-RF (EUR-convert if USD)"),
 "RB_EU":   ("C-Equity", "keep", "EU bank-bond excess return"),
 "RMW_EU":  ("C-Equity", "keep", "French Dev Europe RMW (EUR-convert if USD)"),
 "RS_EU":   ("C-Equity", "keep", "EuroStoxx Banks excess return"),
 "SMB_EU":  ("C-Equity", "keep", "French Dev Europe SMB (EUR-convert if USD)"),
 "UMD_EU":  ("C-Equity", "keep", "French Dev Europe WML (EUR-convert if USD)"),
 # --- D: Volatility ---------------------------------------------------------
 "ATM_IV_CDX":  ("D-Volatility","straddle", "CDX delta-hedged straddle (or varswap)"),
 "ATM_IV_ITRX": ("D-Volatility","straddle", "iTraxx delta-hedged straddle (keep 1 if collinear)"),
 "EP_SVIX_1M":  ("D-Volatility","varswap",  "RV - SVIX^2 (Martin)"),
 "EP_SVIX_3M":  ("D-Volatility","DROP",     "redundant with EP_SVIX_1M"),
 "IV_BUND":     ("D-Volatility","straddle", "RX1 delta-hedged straddle (Black-76)"),
 "IV_TSY":      ("D-Volatility","straddle", "TY1 delta-hedged straddle"),
 "MOVE":        ("D-Volatility","DROP",     "redundant if IV_TSY straddle built"),
 "PTFSBD":  ("D-Volatility","keep", "Hsieh trend bond"),
 "PTFSCOM": ("D-Volatility","keep", "Hsieh trend commodity"),
 "PTFSFX":  ("D-Volatility","keep", "Hsieh trend FX"),
 "PTFSIR":  ("D-Volatility","keep", "Hsieh trend short rate"),
 "PTFSSTK": ("D-Volatility","keep", "Hsieh trend equity"),
 "ΔV2X":    ("D-Volatility","fut",  "FVS1 rolled front future return"),
 "ΔVIX":    ("D-Volatility","keep", "SPVXSP ER index (or fut UX1)"),
 # --- E: Macro --------------------------------------------------------------
 "5Y5Y_INFL": ("E-Macro","rate_pos", "r=-d(rate)*forward annuity (SWPM)"),
 "BFCI_EU":   ("E-Macro","DROP", "non-traded conditions index"),
 "EPU_EU":    ("E-Macro","DROP", "non-traded uncertainty index"),
 "EPU_US":    ("E-Macro","DROP", "non-traded uncertainty index"),
 "ΔUF":       ("E-Macro","DROP", "Ludvigson financial uncertainty"),
 "ΔUM":       ("E-Macro","DROP", "Ludvigson macro uncertainty"),
 "ΔUR":       ("E-Macro","DROP", "Ludvigson real uncertainty"),
 # --- F: Rates --------------------------------------------------------------
 "EURUSD_3M_IV": ("F-Rates","straddle", "GK delta-hedged FX straddle"),
 "GLOBAL_TERM":  ("F-Rates","keep", "R(H00023EU) - rf_eur (Global Treasury EUR-hedged)"),
 "GOVT_EU":      ("F-Rates","keep", "R(LETGTREU) - rf_eur (broad German Treasury aggregate; EU analogue of US Term)"),
 "INFL_LINK":    ("F-Rates","keep", "R(LF94TREH) - rf_eur (Global ILB EUR-hedged)"),
 "R10_EU":       ("F-Rates","keep", "EU 10Y govt portfolio TR"),
 "R2_EU":        ("F-Rates","keep", "EU 2Y govt portfolio TR"),
 "R5_EU":        ("F-Rates","keep", "EU 5Y govt portfolio TR"),
 "SLOPE_2S10S_EU":  ("F-Rates", "curve", "DV01-matched DU/RX steepener (Bund 2s10s), futures roll-adj"),
 "SLOPE_2S10S_US":  ("F-Rates", "curve", "DV01-matched TU/TY steepener (UST 2s10s), futures roll-adj"),
 "SLOPE_10S30S_EU": ("F-Rates", "curve", "DV01-matched RX/UB steepener (Bund/Buxl 10s30s); from Oct-2005"),
 "CURV_2S5S10S_EU": ("F-Rates", "curve", "DV01-neutral DU/OE/RX butterfly (long belly)"),
 "CURV_2S5S10S_US": ("F-Rates", "curve", "DV01-neutral TU/FV/TY butterfly (long belly)"),
 "SS10Y":        ("F-Rates",    "dv01", "RX1 vs pay-fixed EUSA10, DV01-matched"),
 "SS2Y":         ("F-Rates","dv01", "DU1 vs EUSA2, DV01-matched"),
 "SS5Y":         ("F-Rates","dv01", "OE1 vs EUSA5, DV01-matched"),
 "TERM_EU":      ("F-Rates","keep", "H01656EU long-Treasury TR (minus RF)"),
 "TERM_US":      ("F-Rates","keep", "LUTLTRUU long-Treasury TR (minus RF)"),
 "YSP_EU":       ("F-Rates","dv01", "OE1 - DU1 steepener, DV01-matched"),
 "YSP_US":       ("F-Rates","dv01", "FV1 - TU1 steepener, DV01-matched"),
 "Δ10Y_YIELD_EU":("F-Rates","DROP", "R10_EU already spans it"),
 "Δ10Y_YIELD_US":("F-Rates","fut",  "TY1 rolled front future return"),
 "ΔSLOPE_EU":    ("F-Rates","dv01", "RX1-ER1; ~TERM_EU (consider drop)"),
 "ΔSLOPE_US":    ("F-Rates","dv01", "TY1-SFR1; ~TERM_US (consider drop)"),
}

DROPPED = [k for k, v in FACTOR_PLAN.items() if v[1] == "DROP"]
KEPT    = [k for k, v in FACTOR_PLAN.items() if v[1] != "DROP"]


# ===========================================================================
# BUILD  --  assemble every retained factor.  Wire column names where TODO.
# ===========================================================================

def build():
    """Assemble the monthly factor matrix.  Fill the TODO column names as you
    download each source.  Returns a DataFrame indexed by month-end."""
    F = {}   # factor_id -> monthly Series

    # ---- load raw sheets/files (only those you've downloaded) --------------
    # All factors are stored in PERCENT (to match the strategy returns y).
    # rates_fx pieces (used by the French EUR conversion and the funding bases):
    bbg_rates = src.load_bloomberg("rates_fx")
    eurusd = to_monthly_last(bbg_rates["EURUSD Curncy"])
    getb1  = to_monthly_last(bbg_rates["GETB1 Index"])             # 1M German T-bill YIELD, %
    r_fx   = eurusd.pct_change()                                   # decimal
    _days  = pd.Series(getb1.index.day, index=getb1.index)         # calendar days/month
    # EUR risk-free = 1M German sovereign bill (Fama-French analogue of the US T-bill;
    # NOT Euribor, which embeds the TED credit/liquidity premium).  FF use the 1M bill
    # rate "observed at the beginning of the month" -> prior month-end yield (.shift(1)),
    # accrued ACT/360.  Same rf_eur feeds every excess return (factors AND strategy
    # returns y) and the EUR leg of the GHS conversion.
    rf_eur = getb1.shift(1) * _days / 360.0                        # euro RF, %, ACT/360
    rf_usd = src.load_french("Europe_5_Factors.csv")["RF"]         # US T-bill, %; USD excess -> EUR (GHS)

    bbg_tr  = src.load_bloomberg("tr_indices")
    # ---- CDS sheet (3 fields/instrument) + monthly RPV01 (BBG + ISDA fallback) ----
    cds = src.load_cds_fields()
    SPR, RPVb, SER = cds["spread"], cds["rpv01_bbg"], cds["series"]
    disc_d = bbg_rates["EUR001M Index"] / 100.0                       # daily EUR short rate (discount)
    cmonths = pd.date_range("2004-01-31", "2026-12-31", freq="ME")
    _ser = lambda t: (SER[t] if t in SER.columns else None)
    _bbg = lambda t: (RPVb[t] if t in RPVb.columns else None)
    def _rpv(target, tickers_by_tenor, bbg_ticker):
        td = {ten: SPR[tk] for ten, tk in tickers_by_tenor.items()}
        return _cds_rpv01_monthly(target, td, _bbg(bbg_ticker), disc_d, cmonths)
    Ix = src.CDS_IDX
    rpv_main5 = _rpv("5Y", {"5Y": Ix["MAIN5"], "3Y": Ix["MAIN3"]}, Ix["MAIN5"])
    rpv_main3 = _rpv("3Y", {"5Y": Ix["MAIN5"], "3Y": Ix["MAIN3"]}, Ix["MAIN3"])
    rpv_cdxig = _rpv("5Y", {"5Y": Ix["CDXIG"]},  Ix["CDXIG"])
    rpv_snr   = _rpv("5Y", {"5Y": Ix["SNRFIN"]}, Ix["SNRFIN"])
    rpv_xover = _rpv("5Y", {"5Y": Ix["XOVER"]},  Ix["XOVER"])
    bbg_fut = src.load_bloomberg("futures")
    fut     = src.load_futures()              # strutturato: px / frsk / gen per contratto
    # bbg_vol = src.load_bloomberg("vol_options")
    # fred_bbb = src.load_fred("BAMLCC0A4BBBTRIV.csv")
    # ... aqr/hsieh/stambaugh/hkm/martin as needed ...

    # ===== Panel A: Credit =================================================
    # F["BTP_BUND"] = dv01_matched_spread(bbg_fut["IK1"], bbg_fut["RX1"],
    #                                     dv01_a=..., dv01_b=...)         # TODO DV01
    # ---- CDS factors (all sell-protection EXCESS returns; tradable, %). ----------
    # Index IG credit risk premium:
    F["CDX_IG"]    = cds_sell_protection(SPR[Ix["CDXIG"]], rpv_cdxig, _ser(Ix["CDXIG"])) * 100
    F["ITRX_MAIN"] = cds_sell_protection(SPR[Ix["MAIN5"]], rpv_main5, _ser(Ix["MAIN5"])) * 100
    # Prime-broker dealer baskets (equal-weight avg of per-name sell-protection rets):
    def _pb(names5, names1, tenor):
        legs = []
        for n5, n1 in zip(names5, names1):
            tk = n5 if tenor == "5Y" else n1
            rp = _cds_rpv01_monthly(tenor, {"5Y": SPR[n5], "1Y": SPR[n1]}, _bbg(tk), disc_d, cmonths)
            legs.append(cds_sell_protection(SPR[tk], rp, _ser(tk)))
        return pd.concat(legs, axis=1).mean(axis=1)
    F["PB_CDS_5Y_EU"] = _pb(src.PB_EU_5Y, src.PB_EU_1Y, "5Y") * 100
    F["PB_CDS_1Y_EU"] = _pb(src.PB_EU_5Y, src.PB_EU_1Y, "1Y") * 100
    F["PB_CDS_5Y_US"] = _pb(src.PB_US_5Y, src.PB_US_1Y, "5Y") * 100
    F["PB_CDS_1Y_US"] = _pb(src.PB_US_5Y, src.PB_US_1Y, "1Y") * 100
    # SnrFin vs Main: DV01-neutral spread DIFFERENTIAL (financial-sector stress):
    F["SNRFIN_MAIN"] = cds_two_leg(SPR[Ix["SNRFIN"]], rpv_snr, SPR[Ix["MAIN5"]], rpv_main5,
                                   _ser(Ix["SNRFIN"]), _ser(Ix["MAIN5"]),
                                   k=(rpv_snr / rpv_main5)) * 100
    # 3s5s Main credit-curve slope: DV01-neutral (long 5Y/short 3Y; 3Y rolls with 5Y):
    F["SLOPE_3S5S_MAIN"] = cds_two_leg(SPR[Ix["MAIN5"]], rpv_main5, SPR[Ix["MAIN3"]], rpv_main3,
                                       _ser(Ix["MAIN5"]), _ser(Ix["MAIN5"]),
                                       k=(rpv_main5 / rpv_main3)) * 100
    # Xover vs Main: beta-neutral (rolling 36m beta of dS_X on dS_M, lagged):
    _dX = _to_decimal(to_monthly_last(_cds_roll_adjust(SPR[Ix["XOVER"]], _ser(Ix["XOVER"]))), True).diff()
    _dM = _to_decimal(to_monthly_last(_cds_roll_adjust(SPR[Ix["MAIN5"]], _ser(Ix["MAIN5"]))), True).diff()
    _beta = _rolling_beta(_dX, _dM, window=36, min_periods=24).shift(1)
    F["XOVER_MAIN"] = cds_two_leg(SPR[Ix["XOVER"]], rpv_xover, SPR[Ix["MAIN5"]], rpv_main5,
                                  _ser(Ix["XOVER"]), _ser(Ix["MAIN5"]),
                                  k=(_beta * rpv_xover / rpv_main5)) * 100
    # Credit factors from bbg tr_indices (total-return levels -> percent return spreads).
    # EU legs stay in EUR; US legs are USD self-financing long-shorts -> EUR via GHS eq (11).
    # NB: in tr_indices the Bund 2/5/10 (MLTAG*) are EXCESS-RETURN indices, all the rest are
    #     TOTAL-RETURN.  In a TR-TR spread the risk-free cancels; CRED_SPR_EU mixes a TR
    #     corporate leg with the ER Bund-10Y leg, so we subtract rf_eur to remove the RF.
    F["CREDIT_EU"]   = tr_index_spread(bbg_tr["I02202EU Index"], bbg_tr["I02199EU Index"]) * 100
    F["CREDIT_US"]   = ghs_longshort(tr_index_spread(bbg_tr["LCB1TRUU Index"], bbg_tr["I08218US Index"]) * 100, r_fx)
    F["CRED_SPR_EU"] = tr_index_spread(bbg_tr["I02202EU Index"], bbg_tr["MLTAG10E Index"]) * 100 - rf_eur
    F["CRED_SPR_US"] = ghs_longshort(tr_index_spread(bbg_tr["LCB1TRUU Index"], bbg_tr["SPBDU1BT Index"]) * 100, r_fx)
    F["DEF_EU"]      = tr_index_spread(bbg_tr["I02138EU Index"], bbg_tr["I01656EU Index"]) * 100
    F["DEF_US"]      = ghs_longshort(tr_index_spread(bbg_tr["LD07TRUU Index"], bbg_tr["LUTLTRUU Index"]) * 100, r_fx)
    # (CDS factors are all built in the block above.)
    # ---- Active-FI / benchmark credit (Brooks-Gould-Richardson 2020), from tr_indices.
    #      EUR / EUR-hedged legs -> excess of EUR cash (-rf_eur), NO FX.
    #      EMERG_FX is the only USD leg (MSCI EM Ccy MXEF0CX0 vs USD): self-finance
    #      (-rf_usd) then convert to EUR via GHS eq (11).
    F["HY_CORP"]    = monthly_return(bbg_tr["H02500EU Index"]) * 100 - rf_eur
    F["EMERG_DEBT"] = monthly_return(bbg_tr["H04386EU Index"]) * 100 - rf_eur
    F["GLOBAL_AGG"] = monthly_return(bbg_tr["LEGATREH Index"]) * 100 - rf_eur
    F["EMERG_FX"]   = ghs_longshort(monthly_return(bbg_tr["MXEF0CX0 Index"]) * 100 - rf_usd, r_fx)
    F["RI_EU"]      = (0.5 * monthly_return(bbg_tr["I02209EU Index"]) * 100
                       + 0.5 * monthly_return(bbg_tr["I02210EU Index"]) * 100) - rf_eur
    # TODO still: DEF_EU/DEF_US (I02138EU-I01656EU / LD07TRUU-LUTLTRUU spreads), RI_EU
    #                      (DEF_EU/DEF_US now built above from tr_indices)

    # ===== Panel B: Liquidity =============================================
    # Pastor-Stambaugh TRADED liquidity (LIQ_V, 10-1 portfolio): USD long-short
    # -> percent (x100) -> EUR (GHS eq 11).  Non-traded LIQNT is dropped.
    stamb = src.load_stambaugh("Stambaugh.xlsx")
    F["LIQ_V"] = ghs_longshort(to_monthly_last(stamb["LIQ_V"]) * 100, r_fx)
    # He-Kelly-Manela TRADED intermediary factor = value-weighted equity return
    # of the primary dealers (USD, gross, long-only) -> EUR excess (GHS eq 10,
    # like the market).  The non-traded capital-ratio innovation is dropped.
    hkm = src.load_hkm("He_Kelly_Manela_Factors_monthly_250627.csv")
    hkm_inv = to_monthly_last(hkm["intermediary_value_weighted_investment_return"])
    F["HKM_IC"] = ((1 + hkm_inv) / (1 + r_fx) - 1 - rf_eur / 100) * 100
    # F["EURIBOR_OIS"] = rate_change_position_return(bbg_sw["EURIBOR_OIS_SPR"], src.MM_DURATION)
    # F["LIBOR_OIS"]   = rate_change_position_return(bbg_sw["LIBOR_OIS_SPR"],   src.MM_DURATION)
    # F["TED_SHOCK_EU"]= rate_change_position_return(bbg_sw["TED_EU_SPR"],      src.MM_DURATION)
    # F["TED_SHOCK_US"]= rate_change_position_return(bbg_sw["TED_US_SPR"],      src.MM_DURATION)
    # KEEP also: HKM_IC (traded col)

    # ===== Panel C: Equity ================================================
    # French factors are reported in USD -> convert to EUR (GHS 2021, eq 10-11).
    # Output is already in PERCENT.
    ff5 = src.load_french("Europe_5_Factors.csv")    # Mkt-RF,SMB,HML,RMW,CMA,RF
    mom = src.load_french("Europe_MOM_Factor.csv")   # WML
    F["MKT_EU"] = ghs_market(ff5["Mkt-RF"], ff5["RF"], r_fx, rf_eur)
    F["SMB_EU"] = ghs_longshort(ff5["SMB"], r_fx)
    F["HML_EU"] = ghs_longshort(ff5["HML"], r_fx)
    F["RMW_EU"] = ghs_longshort(ff5["RMW"], r_fx)
    F["CMA_EU"] = ghs_longshort(ff5["CMA"], r_fx)
    F["UMD_EU"] = ghs_longshort(mom["WML"], r_fx)
    # AQR factors (USD, DECIMAL) -> percent (x100) -> EUR (GHS eq 11).
    aqr_bab = src.load_aqr("Betting Against Beta Equity Factors Monthly.xlsx")
    aqr_qmj = src.load_aqr("Quality Minus Junk Factors Monthly.xlsx")
    F["BAB_EU"] = ghs_longshort(to_monthly_last(aqr_bab["Europe"]) * 100, r_fx)
    F["QMJ_EU"] = ghs_longshort(to_monthly_last(aqr_qmj["Europe"]) * 100, r_fx)
    # GVAL / GMOM = VME global "EVERYWHERE" value/momentum (cols 'VAL'/'MOM').
    aqr_vme = src.load_aqr("Value and Momentum Everywhere Factors Monthly.xlsx")
    F["GVAL"] = ghs_longshort(to_monthly_last(aqr_vme["VAL"]) * 100, r_fx)
    F["GMOM"] = ghs_longshort(to_monthly_last(aqr_vme["MOM"]) * 100, r_fx)
    # bank credit & bank equity (EUR, excess of EUR cash; no FX) -- Tradable_stock_factors
    F["RB_EU"] = (0.5 * monthly_return(bbg_tr["I02205EU Index"]) * 100
                  + 0.5 * monthly_return(bbg_tr["I02206EU Index"]) * 100) - rf_eur
    F["RS_EU"] = monthly_return(bbg_tr["SX7EFSTR Index"]) * 100 - rf_eur

    # ===== Panel D: Volatility ============================================
    # PTFS (Fung-Hsieh trend-following): decimal -> percent; LOCAL currency, NO
    # FX adjustment (per D. Hsieh).  Standardized downstream, so scale is moot.
    hsieh = src.load_hsieh()
    for k in ("PTFSBD", "PTFSFX", "PTFSCOM", "PTFSIR", "PTFSSTK"):
        F[k] = to_monthly_last(hsieh[k]) * 100
    # F["EP_SVIX_1M"] = variance_swap_payoff(spx_daily_ret, svix2_level)       # TODO
    # Volatility factors: long short-term vol futures (constant ~1M maturity), excess return.
    # VXXIDSP is already the VIX short-term futures EXCESS-RETURN index (USD) -> EUR via eq (11).
    # VST1MT is the VSTOXX short-term futures TOTAL-RETURN index (EUR) -> subtract rf_eur for ER.
    F["VIX_FUT"] = ghs_longshort(monthly_return(bbg_fut["VXXIDSP Index"]) * 100, r_fx)
    F["V2X_FUT"] = monthly_return(bbg_fut["VST1MT Index"]) * 100 - rf_eur
    # straddles (ATM_IV_CDX/ITRX, IV_BUND, IV_TSY): premium via black76_straddle
    #   on held IV, then delta_hedged_straddle(...) once daily marks are wired.

    # ===== Panel E: Macro =================================================
    # F["5Y5Y_INFL"] = rate_change_position_return(bbg_sw["FWISEU55"], annuity=FWD_ANNUITY) # TODO annuity

    # ===== Panel F: Rates =================================================
    # F["YSP_US"] = dv01_matched_spread(bbg_fut["FV1"], bbg_fut["TU1"], dv01_a=..., dv01_b=...)
    # F["YSP_EU"] = dv01_matched_spread(bbg_fut["OE1"], bbg_fut["DU1"], dv01_a=..., dv01_b=...)
    # F["SS10Y"]  = dv01_matched_spread(bbg_fut["RX1"], swap10_level, dv01_a=..., dv01_b=...)
    # F["Δ10Y_YIELD_US"] = futures_excess_return(bbg_fut["TY1"])
    # Government term/rate factors from tr_indices.  MLTAG* (Bund 2/5/10) are EXCESS-RETURN
    # indices -> monthly return used directly, NO rf subtraction.  TERM_US is the excess
    # return on the long UST (TR minus US T-bill RF) then converted to EUR via GHS eq (11).
    F["R2_EU"]   = monthly_return(bbg_tr["MLTAGB2E Index"]) * 100
    F["R5_EU"]   = monthly_return(bbg_tr["MLTAGB5E Index"]) * 100
    F["R10_EU"]  = monthly_return(bbg_tr["MLTAG10E Index"]) * 100
    F["TERM_US"] = ghs_longshort(monthly_return(bbg_tr["LUTLTRUU Index"]) * 100 - ff5["RF"], r_fx)
    # ---- Active-FI term/linker + EU broad-govt term (excess of EUR cash, no FX) ----
    F["GLOBAL_TERM"] = monthly_return(bbg_tr["H00023EU Index"]) * 100 - rf_eur
    F["INFL_LINK"]   = monthly_return(bbg_tr["LF94TREH Index"]) * 100 - rf_eur
    F["GOVT_EU"]     = monthly_return(bbg_tr["LETGTREU Index"]) * 100 - rf_eur   # broad German Treasury (EU Term)

    # --- Curve factors DV01-matched dai futures (roll-adjusted; in bp yield-eq) ---
    _rd = {s: _roll_days(s, fut) for s in ["DU", "OE", "RX", "UB", "FV", "TU"]}
    _rd["TY"] = _rd["TU"]                       # TY1 senza ticker -> roll CME (= TU)
    _dy = {s: curve_dy_bp(s, fut, _rd[s]) for s in ["DU","OE","RX","UB","TU","FV","TY"]}
    F["SLOPE_2S10S_EU"]  = _dy["RX"] - _dy["DU"]                 # steepener (+ = irripidimento)
    F["CURV_2S5S10S_EU"] = _dy["DU"] + _dy["RX"] - 2 * _dy["OE"] # long belly (+ = belly richens)
    _s1030 = _dy["UB"] - _dy["RX"]
    _s1030.loc[:"2005-09"] = np.nan             # Euro-Buxl: 2° contratto solo da set-2005
    F["SLOPE_10S30S_EU"] = _s1030
    F["SLOPE_2S10S_US"]  = _dy["TY"] - _dy["TU"]
    F["CURV_2S5S10S_US"] = _dy["TU"] + _dy["TY"] - 2 * _dy["FV"]
    
    F["TERM_EU"]     = monthly_return(bbg_tr["I01656EU Index"]) * 100 - rf_eur   

    # ---- assemble ---------------------------------------------------------
    if not F:
        raise SystemExit("build() is a skeleton: wire the TODO columns first.")
    out = pd.DataFrame(F).sort_index()
    out = out.loc[src.SAMPLE_START:src.SAMPLE_END]
    return out


def write_metadata():
    """Write the factor registry (factor -> panel, recipe, status, note)."""
    rows = [{"factor": k, "panel": v[0], "recipe": v[1],
             "status": "DROP" if v[1] == "DROP" else "KEEP",
             "note": v[2]} for k, v in FACTOR_PLAN.items()]
    md = pd.DataFrame(rows)
    src.OUT_METADATA.parent.mkdir(parents=True, exist_ok=True)
    md.to_csv(src.OUT_METADATA, index=False)
    return md


if __name__ == "__main__":
    print(f"Registry: {len(FACTOR_PLAN)} factors  |  KEEP/BUILD {len(KEPT)}  |  DROP {len(DROPPED)}")
    md = write_metadata()
    print(f"metadata -> {src.OUT_METADATA}")
    factors = build()                       # wire TODOs, then this runs
    factors.to_parquet(src.OUT_PARQUET)
    print(f"factors  -> {src.OUT_PARQUET}  shape={factors.shape}")

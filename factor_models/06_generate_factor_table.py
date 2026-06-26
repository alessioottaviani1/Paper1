"""
================================================================================
06_generate_factor_table.py - Generate Factor List Table for Paper
================================================================================
Reads all_factors_monthly.parquet to get the current factor list,
maps each factor to its description/reference/panel, and generates
a longtable LaTeX file for the thesis appendix (A.3.1).

OUTPUT: results/tables/factor_list.tex

If a new factor is added to 00_import_all_factors.py and appears in the
parquet but is not in the FACTOR_INFO dictionary below, it will appear
in the table with "Description TBD" so you know to add it.

Author: Alessio Ottaviani
Institution: EDHEC Business School - PhD Thesis
================================================================================
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_PATH = DATA_DIR / "all_factors_monthly.parquet"

# ============================================================================
# FACTOR METADATA: {factor_name: (panel, description, reference)}
# ============================================================================
# Panels:
#   A = Credit Risk Factors
#   B = Liquidity Factors
#   C = Equity Factors
#   D = Volatility Factors
#   E = Macro Factors
#   F = Interest Rate Factors
#   G = Active FI / Benchmark Factors
#
# If you add a factor to 00_import_all_factors.py, add an entry here too.
# ============================================================================

FACTOR_INFO = {

    # ========== PANEL A: CREDIT RISK ==========
    "BTP_BUND": ("A", "BTP-Bund spread factor (10-year). Duration-matched excess return of Italian over German government bonds, long the Italian and short the German 7-10Y total-return index.", r"\citet{pelizzon2016sovereign}"),
    "BTP_BUND_2Y": ("A", "BTP-Bund spread factor (2-year). Duration-matched excess return of Italian over German government bonds at the short end, long the Italian and short the German 1-3Y total-return index.", r"\citet{pelizzon2016sovereign}"),
    "CDX_IG": ("A", "Credit risk factor (U.S. investment grade). Excess return on a fully-collateralised sell-protection position in the 5-year CDX North American Investment Grade index.", r"\citet{asvanunt2017credit, bongaerts2011liquidity}"),
    "CREDIT_EU": ("A", "Bond credit risk factor. Return spread between euro-area BBB and AAA corporate bond total-return indices.", r"\citet{fama1993common}"),
    "CREDIT_US": ("A", "Bond credit risk factor. Return spread between BAA and AAA U.S. corporate bond total-return indices.", r"\citet{fama1993common}"),
    "CRED_SPR_EU": ("A", "Bond credit spread factor. Return spread between euro-area BBB corporate bonds and 10-year German Bunds.", r"\citet{fung2001risk}"),
    "CRED_SPR_US": ("A", "Bond credit spread factor. Return spread between BAA U.S. corporate bonds and 10-year U.S. Treasuries.", r"\citet{fung2001risk}"),
    "DEF_EU": ("A", "Bond default risk factor. Total return spread between long-term European corporate bonds and long-term German Bunds.", r"\citet{fama1993common}"),
    "DEF_US": ("A", "Bond default risk factor. Total return spread between long-term U.S. corporate bonds and long-term U.S. government bonds.", r"\citet{fama1993common}"),
    "EBP": ("A", "Excess Bond Premium. Residual component of aggregate corporate bond spreads after netting out expected default risk and bond characteristics via cross-sectional spread regressions; proxies time-varying credit risk compensation tied to intermediary constraints. Used with a one-month lag.", r"\citet{gilchrist2012credit}"),
    "ITRX_MAIN": ("A", "Credit risk factor (European investment grade). Excess return on a fully-collateralised sell-protection position in the 5-year iTraxx Europe Main index.", r"\citet{asvanunt2017credit, bongaerts2011liquidity}"),
    "PB_CDS_1Y_EU": ("A", "Intermediary funding factor (Europe, short tenor). Excess return on an equally weighted basket of 1-year sell-protection positions in the senior CDS of five European prime brokers.", r"\citet{siriwardane2019capital, he2017intermediary}"),
    "PB_CDS_1Y_US": ("A", "Intermediary funding factor (United States, short tenor). Excess return on an equally weighted basket of 1-year sell-protection positions in the senior CDS of four U.S. prime brokers.", r"\citet{siriwardane2019capital, he2017intermediary}"),
    "PB_CDS_5Y_EU": ("A", "Intermediary funding factor (Europe). Excess return on an equally weighted basket of 5-year sell-protection positions in the senior CDS of five European prime brokers.", r"\citet{siriwardane2019capital, he2017intermediary}"),
    "PB_CDS_5Y_US": ("A", "Intermediary funding factor (United States). Excess return on an equally weighted basket of 5-year sell-protection positions in the senior CDS of four U.S. prime brokers.", r"\citet{siriwardane2019capital, he2017intermediary}"),
    "SLOPE_3S5S_MAIN": ("A", "Credit-curve slope factor. Excess return on a DV01-matched position long protection at the 5-year and short protection at the 3-year point of the iTraxx Europe Main curve.", r"\citet{hansubrahmanyamzhou2017, hanzhou2015cds}"),
    "SNRFIN_MAIN": ("A", "Financial-sector stress factor. Excess return on a DV01-matched position long credit via the iTraxx Senior Financials index and short the iTraxx Europe Main index.", r"\citet{he2017intermediary, siriwardane2019capital}"),
    "XOVER_MAIN": ("A", "Credit-cycle factor. Excess return on a beta-neutral position long credit via the 5-year iTraxx Crossover index and short the 5-year iTraxx Europe Main index, the short leg scaled by a rolling 36-month spread-change beta.", r"\citet{asvanunt2017credit}"),

    # ========== PANEL B: LIQUIDITY ==========
    "\u0394FAILS_PCT_TSY": ("B", r"First difference of the ratio of monthly total notional amount of U.S. Treasury securities fails-to-deliver reported by primary dealers, excluding TIPS, to U.S. Total Debt Outstanding.", r"\citet{fleckenstein2014tips}"),
    "EURIBOR_OIS": ("B", "First difference of the Euribor--OIS spread (3-month Euribor minus the 3-month overnight index swap rate based on EONIA or \\euro STR).", r"\citet{nyborg2014money}"),
    "GC-REPO_T-BILL": ("B", "First difference of the spread between the 3-month U.S. Treasury general-collateral repo rate and the 3-month U.S. Treasury bill yield.", r"\citet{bai2019cds}"),
    "HKM_IC": ("B", "Intermediary capital factor (tradable). Value-weighted equity return on the primary-dealer holding companies, in excess of cash and converted to euro; a tradable proxy (0.92 correlation) for the He-Kelly-Manela capital-ratio innovation.", r"\citet{he2017intermediary}"),
    "HPW_NOISE": ("B", "First difference of the Treasury noise measure of market-wide illiquidity (root mean squared deviation between observed U.S. Treasury yields and model-implied yields from a fitted smooth zero-coupon yield curve).", r"\citet{hu2013noise}"),
    "ILLIQ": ("B", "Change in the Amihud illiquidity measure. First difference of the ratio of absolute daily return to daily dollar volume.", r"\citet{amihud2015pricing}"),
    "LIBOR_OIS": ("B", "First difference of the Libor--OIS spread (3-month LIBOR minus the 3-month overnight index swap rate).", r"\citet{nyborg2014money}"),
    "LIBOR_REPO_SHOCK": ("B", "AR(2) residual of the spread between 3-month U.S. interbank LIBOR and the 3-month U.S. Treasury general-collateral repo rate.", r"\citet{asness2013value}"),
    "LIQ_V": ("B", "Value-weighted return on a 10--1 portfolio formed by sorting stocks on historical liquidity betas.", r"\citet{pastor2003liquidity}"),
    "LIQNT": ("B", "Pastor--Stambaugh non-traded liquidity factor. Innovation in aggregate market liquidity constructed from a cross-sectional average of stock-level order-flow-induced return-reversal measures.", r"\citet{pastor2003liquidity}"),
    "SILLIQ": ("B", "Stock illiquidity shock. AR(3) residual of the aggregate Amihud illiquidity measure.", r"\citet{acharya2013liquidity}"),
    "TED_SHOCK_EU": ("B", "AR(2) residual of the euro TED spread (3-month Euribor minus the 3-month German government bill rate).", r"\citet{asness2013value}"),
    "TED_SHOCK_US": ("B", "AR(2) residual of the U.S. TED spread (3-month LIBOR minus the 3-month U.S. government bill rate).", r"\citet{asness2013value}"),

    # ========== PANEL C: EQUITY ==========
    "BAB_EU": ("C", "Betting Against Beta (Europe). Long low-beta and short high-beta European equities, each leg rescaled to unit beta; USD factor converted to euro.", r"\citet{frazzini2014betting}"),
    "QMJ_EU": ("C", "Quality Minus Junk (Europe). Long high-quality and short low-quality (``junk'') European stocks, where quality aggregates profitability, growth, safety, and payout; USD factor converted to euro.", r"\citet{asness2019quality}"),
    "CMA_EU": ("C", "Investment factor (Conservative Minus Aggressive). Return spread between European conservative and aggressive investment portfolios.", r"\citet{fama2017international}"),
    "GMOM": ("C", "Global all-asset momentum factor. Long-short momentum return averaged across the eight Value-and-Momentum-Everywhere strategies (stock selection in the U.S., U.K., Europe and Japan, plus equity-index, government-bond, currency and commodity allocations), converted to euro.", r"\citet{asness2013value}"),
    "GVAL": ("C", "Global all-asset value factor. Long-short value return averaged across the eight Value-and-Momentum-Everywhere strategies (stock selection in the U.S., U.K., Europe and Japan, plus equity-index, government-bond, currency and commodity allocations), converted to euro.", r"\citet{asness2013value}"),
    "FX_CARRY": ("C", "Currency carry factor. Self-financing long-short return, long high-interest-rate and short low-interest-rate developed-market (G10) currencies; converted to euro.", r"\citet{ilmanen2021factor}"),
    "FX_MOM": ("C", "Currency momentum factor. Self-financing long-short return on developed-market (G10) currencies sorted on their past 12-month return; converted to euro.", r"\citet{ilmanen2021factor}"),
    "COM_CARRY": ("C", "Commodity carry factor. Self-financing long-short return on commodity futures sorted on the slope of the futures curve (carry); converted to euro.", r"\citet{ilmanen2021factor}"),
    "COM_MOM": ("C", "Commodity momentum factor. Self-financing long-short return on commodity futures sorted on their past 12-month return; converted to euro.", r"\citet{ilmanen2021factor}"),
    "HML_EU": ("C", "Value factor (High Minus Low). Return spread between European high and low book-to-market portfolios.", r"\citet{fama2017international}"),
    "MKT_EU": ("C", "Market excess return on the European value-weighted equity market portfolio over the 1-month German government bill.", r"\citet{fama2017international}"),
    "RMW_EU": ("C", "Profitability factor (Robust Minus Weak). Return spread between European robust and weak operating profitability portfolios.", r"\citet{fama2017international}"),
    "SMB_EU": ("C", "Size factor (Small Minus Big). Return spread between diversified European small- and large-cap portfolios.", r"\citet{fama2017international}"),
    "UMD_EU": ("C", "Momentum factor (Up Minus Down). Monthly return on a European zero-investment winners-minus-losers momentum portfolio.", r"\citet{carhart1997persistence}"),

    # ========== PANEL D: VOLATILITY ==========
    "V2X_FUT": ("D", "Volatility factor (euro area). Excess return on a constant-maturity (one-month) rolling long position in VSTOXX futures (VSTOXX Short-Term Futures index), in excess of the euro risk-free.", r"\citet{eraker2017explaining}"),
    "VIX_FUT": ("D", r"Volatility factor (U.S.). Excess return on a constant-maturity (one-month) rolling long position in VIX futures (S\&P 500 VIX Short-Term Futures Excess-Return index).", r"\citet{eraker2017explaining}"),
    "ATM_IV_CDX": ("D", "Excess return on a three-month variance swap on the CDX Investment Grade index spread: realized variance of the on-the-run index spread minus its at-the-money (100\\% moneyness) implied variance fixed at the start of the window (equivalently, the delta-hedged ATM payer/receiver straddle return).", r"\citet{carr2009variance}"),
    "ATM_IV_ITRX": ("D", "Excess return on a three-month variance swap on the iTraxx Europe Main index spread: realized variance of the on-the-run index spread minus its at-the-money (100\\% moneyness) implied variance fixed at the start of the window (equivalently, the delta-hedged ATM payer/receiver straddle return).", r"\citet{carr2009variance}"),
    "SVS_RET_1M": ("D", "Excess return on a one-month simple variance swap on the S\\&P~500: realized simple variance minus the option-implied SVIX\\textsuperscript{2} strike fixed at the start of the month.", r"\citet{martin2017expected}"),
    "SVS_RET_3M": ("D", "Excess return on a simple variance swap on the S\\&P~500 over a trailing three-month window (realized simple variance minus the SVIX\\textsuperscript{2} strike).", r"\citet{martin2017expected}"),
    "IV_BUND": ("D", "Excess return on a one-month variance swap on the 10-year German Bund future: realized variance of the future minus its at-the-money implied variance fixed at the start of the month (equivalently, the delta-hedged ATM straddle return).", r"\citet{cremers2021treasury}"),
    "IV_TSY": ("D", "Excess return on a one-month variance swap on the 10-year U.S. Treasury future: realized variance of the future minus its at-the-money implied variance fixed at the start of the month (equivalently, the delta-hedged ATM straddle return).", r"\citet{cremers2021treasury}"),
    "BdOpt": ("D", "Return on the bond trend-following factor, constructed from lookback straddles on government bond futures.", r"\citet{fung2004hedge}"),
    "ComOpt": ("D", "Return on the commodity trend-following factor, constructed from lookback straddles on commodity futures.", r"\citet{fung2004hedge}"),
    "FXOpt": ("D", "Return on the currency trend-following factor, constructed from lookback straddles on major currency futures.", r"\citet{fung2004hedge}"),
    "PTFSIR": ("D", "Return on the short-term interest rate trend-following factor, constructed from lookback straddles on 3-month interest rate futures.", r"\citet{fung2001risk}"),
    "PTFSSTK": ("D", "Return on the equity index trend-following factor, constructed from lookback straddles on stock index futures.", r"\citet{fung2004hedge}"),

    # ========== PANEL F: INTEREST RATES ==========
    "EURUSD_3M_IV": ("F", "First difference of the 3-month at-the-money implied volatility on the EUR/USD exchange rate.", "Market indicator"),
    "SS2Y": ("F", "Swap-spread factor at the 2-year point. Excess return on a DV01-matched position long the German Schatz future and pay-fixed on a 2-year constant-maturity EUR swap.", r"\citet{koijen2018carry, klingler2019swap}"),
    "SS5Y": ("F", "Swap-spread factor at the 5-year point. Excess return on a DV01-matched position long the German Bobl future and pay-fixed on a 5-year constant-maturity EUR swap.", r"\citet{koijen2018carry, klingler2019swap}"),
    "SS10Y": ("F", "Swap-spread factor at the 10-year point. Excess return on a DV01-matched position long the German Bund future and pay-fixed on a 10-year constant-maturity EUR swap.", r"\citet{koijen2018carry, klingler2019swap}"),
    "TERM_EU": ("F", "Bond term structure factor. Excess total return on long-term German Bunds over the 1-month German government bill return.", r"\citet{fama1993common}"),
    "FI_VAL": ("F", "Fixed-income value factor. Self-financing long-short return on a cross-section of developed-market government bonds sorted on a value signal; converted to euro.", r"\citet{ilmanen2021factor}"),
    "FI_MOM": ("F", "Fixed-income momentum factor. Self-financing long-short return on developed-market government bonds sorted on their past 12-month return; converted to euro.", r"\citet{ilmanen2021factor}"),
    "FI_CARRY": ("F", "Fixed-income carry factor. Self-financing long-short return on developed-market government bonds sorted on term-structure carry (yield plus roll-down); converted to euro.", r"\citet{ilmanen2021factor}"),
    "FI_DEF": ("F", "Fixed-income defensive factor. Self-financing long-short return on developed-market government bonds tilted toward low-risk exposures; converted to euro.", r"\citet{ilmanen2021factor}"),
    "TERM_US": ("F", "Bond term structure factor. Excess total return on long-term U.S. government bonds over the 1-month U.S. Treasury bill return.", r"\citet{fama1993common}"),
    "YSP_EU": ("F", "Change in the yield-curve slope factor. First difference of the spread between the 5-year and 1-year German government bond yields.", r"\citet{koijen2017cross}"),
    "YSP_US": ("F", "Change in the yield-curve slope factor. First difference of the spread between the 5-year and 1-year U.S. Treasury yields.", r"\citet{koijen2017cross}"),
    "HY_CORP": ("A", "Sub-investment-grade corporate credit factor: excess return of a EUR-hedged Pan-European high-yield corporate bond index over cash.", r"\citet{asvanunt2017credit}"),
    "EMERG_DEBT": ("A", "Excess return on hard-currency emerging market debt.", r"\citet{brooks2020active}"),
    "EMERG_FX": ("A", "Return on a basket of emerging market currencies versus the USD (MSCI Emerging Markets Currency Index; currency weights from the MSCI EM Index), in excess of USD cash.", r"\citet{brooks2020active}"),
    "GLOBAL_AGG": ("A", "Excess return on the Bloomberg Global Aggregate Bond Index.", r"\citet{brooks2020active}"),
    "GLOBAL_TERM": ("F", "Excess return over cash on the Bloomberg Global Treasury Index (EUR-hedged).", r"\citet{brooks2020active}"),
    "GOVT_EU": ("F", "Euro-area term-premium factor. Excess return over cash on the broad German government bond aggregate (all maturities), the euro-area analogue of the US Term factor.", r"\citet{brooks2020active}"),
    "SLOPE_2S10S_EU": ("F", "Yield-curve slope factor. Excess return on a DV01-matched steepener in German government bond futures, long the 2-year (Schatz) and short the 10-year (Bund).", r"\citet{litterman1991common}"),
    "SLOPE_2S10S_US": ("F", "Yield-curve slope factor. Excess return on a DV01-matched steepener in U.S. Treasury futures, long the 2-year and short the 10-year.", r"\citet{litterman1991common}"),
    "SLOPE_10S30S_EU": ("F", "Yield-curve slope factor. Excess return on a DV01-matched steepener in German government bond futures, long the 10-year (Bund) and short the 30-year (Buxl).", r"\citet{litterman1991common}"),
    "CURV_2S5S10S_EU": ("F", "Yield-curve curvature factor. Excess return on a DV01-neutral butterfly in German government bond futures, long the 5-year belly (Bobl) and short the 2-year (Schatz) and 10-year (Bund) wings.", r"\citet{litterman1991common}"),
    "CURV_2S5S10S_US": ("F", "Yield-curve curvature factor. Excess return on a DV01-neutral butterfly in U.S. Treasury futures, long the 5-year belly and short the 2- and 10-year wings.", r"\citet{litterman1991common}"),
    "INFL_LINK": ("F", "Excess return on global inflation-linked bonds over cash or nominal Treasuries.", r"\citet{brooks2020active}"),
    "R2_EU": ("F", "Excess return on a European 2-year government bond portfolio.", r"\citet{duarte2007risk}"),
    "R5_EU": ("F", "Excess return on a European 5-year government bond portfolio.", r"\citet{duarte2007risk}"),
    "R10_EU": ("F", "Excess return on a European 10-year government bond portfolio.", r"\citet{duarte2007risk}"),
    "RI_EU": ("A", "Excess return on a European A-rated industrial corporate bond index.", r"\citet{duarte2007risk}"),
    "RB_EU": ("C", "Excess return on a European bank bond index.", r"\citet{duarte2007risk}"),
    "RS_EU": ("C", "Excess return on the EuroStoxx Banks index.", r"\citet{duarte2007risk}"),
}

PANEL_NAMES = {
    "A": "Credit Risk Factors",
    "B": "Liquidity Factors",
    "C": "Equity Factors",
    "D": "Volatility Factors",
    "E": "Macro Factors",
    "F": "Interest Rate Factors",

}


# ============================================================================
# LATEX ESCAPING
# ============================================================================

def escape_latex(s: str) -> str:
    """Escape underscores and special chars for LaTeX, preserving commands."""
    # Don't escape if it already contains LaTeX commands
    if "\\" in s or "$" in s:
        return s
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def factor_id_latex(name: str) -> str:
    """Convert factor name to LaTeX display format.
    For names longer than 13 chars, allow line break after underscores."""
    MAX_LEN = 10  # names longer than this get line-break hints

    # Handle delta prefix
    if name.startswith("Δ") or name.startswith("\u0394"):
        rest = name[1:]
        if len(name) > MAX_LEN:
            return r"$\Delta$" + rest.replace("_", r"\_\allowbreak ")
        return r"$\Delta$" + escape_latex(rest)

    if len(name) > MAX_LEN:
        return name.replace("_", r"\_\allowbreak ").replace("&", r"\&").replace("%", r"\%")
    return escape_latex(name)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import pandas as pd

    print("=" * 70)
    print("GENERATING FACTOR LIST TABLE FOR PAPER")
    print("=" * 70)

    # Read parquet to get current factor list
    if not PARQUET_PATH.exists():
        print(f"❌ {PARQUET_PATH} not found. Run 00_import_all_factors.py first.")
        return

    df = pd.read_parquet(PARQUET_PATH)
    factor_names = sorted(df.columns.tolist())
    print(f"✅ {len(factor_names)} factors found in parquet")

    # Check for factors not in FACTOR_INFO
    missing_info = [f for f in factor_names if f not in FACTOR_INFO]
    if missing_info:
        print(f"\n⚠️  {len(missing_info)} factors without description:")
        for m in missing_info:
            print(f"   - {m}")
        print("   → Will appear as 'Description TBD' in the table.\n")

    # Organize by panel
    panels = {}
    for fac in factor_names:
        if fac in FACTOR_INFO:
            panel, desc, ref = FACTOR_INFO[fac]
        else:
            panel = "Z"  # Unknown panel, will sort last
            desc = "Description TBD."
            ref = "---"
        panels.setdefault(panel, []).append((fac, desc, ref))

    # Generate LaTeX
    output_path = OUTPUT_DIR / "factor_list.tex"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("% " + "=" * 74 + "\n")
        f.write("% FACTOR LIST - Auto-generated by 06_generate_factor_table.py\n")
        f.write(f"% {len(factor_names)} factors from all_factors_monthly.parquet\n")
        f.write("% " + "=" * 74 + "\n\n")

        f.write(r"\begin{singlespace}" + "\n")
        f.write(r"\footnotesize" + "\n")
        f.write(r"\setlength{\tabcolsep}{4pt}" + "\n")
        f.write(r"\renewcommand{\arraystretch}{1.1}" + "\n\n")

        f.write(r"\begin{longtable}{@{}p{0.14\textwidth}>{\RaggedRight\arraybackslash}p{0.62\textwidth}p{0.18\textwidth}@{}}" + "\n")
        f.write(r"\caption{Candidate Risk Factors}" + "\n")
        f.write(r"\label{tab:factor_list} \\" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\textbf{Factor ID} & \textbf{Description} & \textbf{Reference} \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\endfirsthead" + "\n\n")

        f.write(r"\multicolumn{3}{c}{{\bfseries Table \thetable\ (continued)}} \\" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"\textbf{Factor ID} & \textbf{Description} & \textbf{Reference} \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\endhead" + "\n\n")

        f.write(r"\midrule" + "\n")
        f.write(r"\multicolumn{3}{@{}p{\textwidth}@{}}{\tiny\textit{Note:} " +
                "Factors are organized by category. Each series is reconstructed "
                "consistently with the original reference. "
                f"Total: {len(factor_names)} factors.}}" + " \\\\\n")
        f.write(r"\endfoot" + "\n\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\endlastfoot" + "\n\n")

        # Write panels in order
        for panel_key in sorted(panels.keys()):
            panel_label = PANEL_NAMES.get(panel_key, "Other Factors")
            factors_in_panel = panels[panel_key]

            f.write(r"\midrule" + "\n")
            f.write(f"\\multicolumn{{3}}{{c}}{{\\textbf{{Panel {panel_key}: {panel_label}}}}} \\\\\n")
            f.write(r"\midrule" + "\n")

            for fac_name, desc, ref in factors_in_panel:
                fid = factor_id_latex(fac_name)
                f.write(f"{fid} & {desc} & {ref} \\\\ \\addlinespace[0.25em]\n")

            f.write("\n")

        f.write(r"\end{longtable}" + "\n")
        f.write(r"\end{singlespace}" + "\n")

    print(f"\n💾 Saved: {output_path}")
    print(f"   {len(factor_names)} factors across {len(panels)} panels")
    print("=" * 70)


if __name__ == "__main__":
    main()
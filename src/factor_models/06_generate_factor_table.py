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
    "BTP_BUND": ("A", "Yield differential between the Italian 10-year BTP and the German Bund.", r"\citet{fausch2018impact}"),
    "CDX_IG": ("A", "First difference of the CDS spread on the CDX Investment Grade index.", r"\citet{klaus2009hedge}"),
    "CREDIT_EU": ("A", "Bond credit risk factor. Yield spread between euro-area BBB and AAA corporate bond indices.", r"\citet{fama1989business}"),
    "CREDIT_US": ("A", "Bond credit risk factor. Yield spread between Moody's BAA and AAA corporate bond indices.", r"\citet{fama1989business}"),
    "CRED_SPR_EU": ("A", "Change in the credit spread between euro-area BAA corporate bond yields and 10-year German Bund yields.", r"\citet{fung2001risk}"),
    "CRED_SPR_US": ("A", "Change in the credit spread between Moody's BAA yields and 10-year U.S. Treasury yields.", r"\citet{fung2001risk}"),
    "DEF_EU": ("A", "Bond default risk factor. Total return spread between long-term European corporate bonds and long-term German Bunds.", r"\citet{fama1993common}"),
    "DEF_US": ("A", "Bond default risk factor. Total return spread between long-term U.S. corporate bonds and long-term U.S. government bonds.", r"\citet{fama1993common}"),
    "EBP": ("A", "Excess Bond Premium. Residual component of aggregate corporate bond spreads after netting out expected default risk and bond characteristics via cross-sectional spread regressions; proxies time-varying credit risk compensation tied to intermediary constraints. Used with a one-month lag.", r"\citet{gilchrist2012credit}"),
    "ITRX_MAIN": ("A", "First difference of the CDS spread on the iTraxx Europe Main index.", r"\citet{klaus2009hedge}"),
    "MAIN_5/3_FLATTENER": ("A", "First difference of the natural logarithm of the ratio between the 5-year and 3-year iTraxx Europe Main CDS spreads.", "Market indicator"),
    "PB_EU_CDS_1Y": ("A", "Average 1-year CDS spread of European prime brokers.", r"\citet{klaus2009hedge}"),
    "PB_EU_CDS_5Y": ("A", "Average 5-year CDS spread of European prime brokers.", r"\citet{klaus2009hedge}"),
    "PB_US_CDS_1Y": ("A", "Average 1-year CDS spread of U.S. prime brokers.", r"\citet{klaus2009hedge}"),
    "PB_US_CDS_5Y": ("A", "Average 5-year CDS spread of U.S. prime brokers.", r"\citet{klaus2009hedge}"),
    "SNR-MAIN": ("A", "First difference of the 5-year CDS spread difference between the iTraxx Senior Financials index and the iTraxx Europe Main index.", "Market indicator"),
    "XOVER/MAIN": ("A", "First difference of the natural logarithm of the ratio between the 5-year iTraxx Crossover and 5-year iTraxx Europe Main CDS spreads.", "Market indicator"),

    # ========== PANEL B: LIQUIDITY ==========
    "\u0394FAILS_PCT_TSY": ("B", r"First difference of the ratio of monthly total notional amount of U.S. Treasury securities fails-to-deliver reported by primary dealers, excluding TIPS, to U.S. Total Debt Outstanding.", r"\citet{fleckenstein2014tips}"),
    "EURIBOR_OIS": ("B", "Euribor--OIS spread. Difference between 3-month Euribor and the 3-month overnight index swap rate based on EONIA or euroSTR.", r"\citet{nyborg2014money}"),
    "GC-REPO_T-BILL": ("B", "Spread between the 3-month U.S. Treasury general-collateral repo rate and the 3-month U.S. Treasury bill yield.", r"\citet{bai2019cds}"),
    "HKM_IC": ("B", "Intermediary capital factor. AR(1) innovation in the market-based capital ratio of primary dealers, scaled by the lagged capital ratio.", r"\citet{he2017intermediary}"),
    "HPW_NOISE": ("B", "Treasury noise measure of market-wide illiquidity. Root mean squared deviation between observed U.S. Treasury yields and model-implied yields from a fitted smooth zero-coupon yield curve.", r"\citet{hu2013noise}"),
    "ILLIQ": ("B", "Change in the Amihud illiquidity measure. First difference of the ratio of absolute daily return to daily dollar volume.", r"\citet{amihud2015pricing}"),
    "LIBOR_OIS": ("B", "Libor--OIS spread. Difference between 3-month LIBOR and the 3-month overnight index swap rate.", r"\citet{nyborg2014money}"),
    "LIBOR_REPO_SHOCK": ("B", "AR(2) residual of the spread between 3-month U.S. interbank LIBOR and the 3-month U.S. Treasury general-collateral repo rate.", r"\citet{asness2013value}"),
    "LIQ_V": ("B", "Value-weighted return on a 10--1 portfolio formed by sorting stocks on historical liquidity betas.", r"\citet{pastor2003liquidity}"),
    "LIQNT": ("B", "Pastor--Stambaugh non-traded liquidity factor. Innovation in aggregate market liquidity constructed from a cross-sectional average of stock-level order-flow-induced return-reversal measures.", r"\citet{pastor2003liquidity}"),
    "SILLIQ": ("B", "Stock illiquidity shock. AR(3) residual of the aggregate Amihud illiquidity measure.", r"\citet{acharya2013liquidity}"),
    "TED_SHOCK_EU": ("B", "AR(2) residual of the euro TED spread (3-month Euribor minus the 3-month German government bill rate).", r"\citet{asness2013value}"),
    "TED_SHOCK_US": ("B", "AR(2) residual of the U.S. TED spread (3-month LIBOR minus the 3-month U.S. government bill rate).", r"\citet{asness2013value}"),

    # ========== PANEL C: EQUITY ==========
    "BAB_EU": ("C", "Betting Against Beta (Europe). Long low-beta and short high-beta European equities, each leg rescaled to unit beta.", r"\citet{frazzini2014betting}"),
    "BAB_US": ("C", "Betting Against Beta (US). Long low-beta and short high-beta U.S. equities, each leg rescaled to unit beta.", r"\citet{frazzini2014betting}"),
    "CMA_EU": ("C", "Investment factor (Conservative Minus Aggressive). Return spread between European conservative and aggressive investment portfolios.", r"\citet{fama2017international}"),
    "GMOM_EU": ("C", "Global momentum factor. Return spread between recent winner and loser equity portfolios based on past 12-month returns, expressed in euros.", r"\citet{asness2013value}"),
    "GVAL_EU": ("C", "Global value factor. Return spread between high and low book-to-market equity portfolios across developed equity regions, expressed in euros.", r"\citet{asness2013value}"),
    "HML_EU": ("C", "Value factor (High Minus Low). Return spread between European high and low book-to-market portfolios.", r"\citet{fama2017international}"),
    "MKT_EU": ("C", "Market excess return on the European value-weighted equity market portfolio over the 1-month German government bill.", r"\citet{fama2017international}"),
    "RMW_EU": ("C", "Profitability factor (Robust Minus Weak). Return spread between European robust and weak operating profitability portfolios.", r"\citet{fama2017international}"),
    "SMB_EU": ("C", "Size factor (Small Minus Big). Return spread between diversified European small- and large-cap portfolios.", r"\citet{fama2017international}"),
    "UMD_EU": ("C", "Momentum factor (Up Minus Down). Monthly return on a European zero-investment winners-minus-losers momentum portfolio.", r"\citet{carhart1997persistence}"),

    # ========== PANEL D: VOLATILITY ==========
    "\u0394V2X": ("D", "Change in V2X. Monthly first difference of the V2X index, measuring 30-day implied volatility on Euro STOXX 50 options.", r"\citet{chung2019volatility}"),
    "\u0394VIX": ("D", r"Change in VIX. Monthly first difference of the VIX index, a model-free measure of 30-day implied volatility on S\&P 500 options.", r"\citet{chung2019volatility}"),
    "ATM_IV_CDX": ("D", "3-month at-the-money implied volatility on options written on the CDX Investment Grade index.", r"\citet{novales2019splitting}"),
    "ATM_IV_ITRX": ("D", "3-month at-the-money implied volatility on options written on the iTraxx Europe Main index.", r"\citet{novales2019splitting}"),
    "EP_SVIX_1M": ("D", "Equity premium proxy from the SVIX methodology. Option-implied risk-neutral variance of the market simple return over the next month.", r"\citet{martin2017expected}"),
    "EP_SVIX_3M": ("D", "Same construction as EP\\_SVIX\\_1M, using options with a 3-month horizon.", r"\citet{martin2017expected}"),
    "IV_BUND": ("D", "3-month at-the-money implied volatility on the 10-year German Bund futures options.", r"\citet{cremers2021treasury}"),
    "IV_TSY": ("D", "3-month at-the-money implied volatility on the 10-year U.S. Treasury futures options.", r"\citet{cremers2021treasury}"),
    "MOVE": ("D", "Option-implied U.S. rates volatility. Yield-curve weighted average of at-the-money implied normal yield volatilities from OTC options on 2-, 5-, 10-, and 30-year constant-maturity interest rate swaps; used lagged by one month.", r"\citet{bansal2022bond}"),
    "MOVE2": ("D", "Squared MOVE; used lagged by one month to capture nonlinear exposure to option-implied U.S. interest rate volatility.", r"\citet{bansal2022bond}"),
    "PTFSBD": ("D", "Return on the bond trend-following factor, constructed from lookback straddles on government bond futures.", r"\citet{fung2001risk}"),
    "PTFSCOM": ("D", "Return on the commodity trend-following factor, constructed from lookback straddles on commodity futures.", r"\citet{fung2001risk}"),
    "PTFSFX": ("D", "Return on the currency trend-following factor, constructed from lookback straddles on major currency futures.", r"\citet{fung2001risk}"),
    "PTFSIR": ("D", "Return on the short-term interest rate trend-following factor, constructed from lookback straddles on 3-month interest rate futures.", r"\citet{fung2001risk}"),
    "PTFSSTK": ("D", "Return on the equity index trend-following factor, constructed from lookback straddles on stock index futures.", r"\citet{fung2001risk}"),

    # ========== PANEL E: MACRO ==========
    "\u0394UF": ("E", r"Change in financial uncertainty. First difference of the financial uncertainty index, constructed from a panel of financial market variables.", r"\citet{ludvigson2021uncertainty}"),
    "\u0394UM": ("E", "Change in macroeconomic uncertainty. First difference of the macro uncertainty index, a model-based measure of expected volatility in the unforecastable component of a large panel of macroeconomic variables.", r"\citet{ludvigson2021uncertainty}"),
    "\u0394UR": ("E", "Change in real uncertainty. First difference of the real-activity uncertainty subindex.", r"\citet{ludvigson2021uncertainty}"),
    "5Y5Y_INFL": ("E", "5-year-5-year inflation swap rate. Forward-starting EUR inflation swap rate capturing long-term inflation expectations.", "Market indicator"),
    "BFCI_EU": ("E", "Bloomberg Euro Area Financial Conditions Index. Composite Z-score of euro-area money, bond, and equity indicators, standardized to the pre-crisis period.", r"\citet{osina2019global}"),
    "EPU_EU": ("E", "Economic Policy Uncertainty index for Europe. Newspaper-based index aggregated across European countries.", r"\citet{baker2016measuring}"),
    "EPU_US": ("E", "Economic Policy Uncertainty index for the United States. Newspaper-based index constructed from normalized frequency of articles referencing economy, uncertainty, and policy.", r"\citet{baker2016measuring}"),

    # ========== PANEL F: INTEREST RATES ==========
    "\u0394SLOPE_EU": ("F", "First difference of the euro yield-curve slope (10-year German Bund yield minus 3-month German government bill yield).", r"\citet{ferson1991variation}"),
    "\u0394SLOPE_US": ("F", "First difference of the U.S. yield-curve slope (10-year Treasury yield minus 3-month Treasury bill yield).", r"\citet{ferson1991variation}"),
    "\u039410Y_YIELD_EU": ("F", "Monthly change in the 10-year German Bund yield.", "Market indicator"),
    "\u039410Y_YIELD_US": ("F", "Monthly change in the 10-year U.S. Treasury yield.", "Market indicator"),
    "EURUSD_3M_IV": ("F", "First difference of the 3-month at-the-money implied volatility on the EUR/USD exchange rate.", "Market indicator"),
    "SS2Y": ("F", "Difference between the 2-year EUR fixed swap par rate and the 2-year German Schatz yield.", r"\citet{collin2001determinants}"),
    "SS5Y": ("F", "Difference between the 5-year EUR fixed swap par rate and the 5-year German Bobl yield.", r"\citet{collin2001determinants}"),
    "SS10Y": ("F", "Difference between the 10-year EUR fixed swap par rate and the 10-year German Bund yield.", r"\citet{collin2001determinants}"),
    "TERM_EU": ("F", "Bond term structure factor. Excess total return on long-term German Bunds over the 1-month German government bill return.", r"\citet{fama1993common}"),
    "TERM_US": ("F", "Bond term structure factor. Excess total return on long-term U.S. government bonds over the 1-month U.S. Treasury bill return.", r"\citet{fama1993common}"),
    "YSP_EU": ("F", "Yield-curve slope factor. Difference between the 5-year and 1-year German government bond yields.", r"\citet{koijen2017cross}"),
    "YSP_US": ("F", "Yield-curve slope factor. Difference between the 5-year and 1-year U.S. Treasury yields.", r"\citet{koijen2017cross}"),

    "CORP_CREDIT": ("A", "Broad corporate credit factor. Excess return on high-yield and leveraged loan indices over duration-matched Treasuries.", r"\citet{brooks2020active}"),
    "EMERG_DEBT": ("A", "Excess return on hard-currency emerging market debt.", r"\citet{brooks2020active}"),
    "EMERG_FX": ("A", "Return on an equal-weighted basket of emerging market currencies versus the USD.", r"\citet{brooks2020active}"),
    "GLOBAL_AGG": ("A", "Excess return on the Bloomberg Global Aggregate Bond Index.", r"\citet{brooks2020active}"),
    "GLOBAL_TERM": ("F", "Excess return on a duration-matched global government bond index.", r"\citet{brooks2020active}"),
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
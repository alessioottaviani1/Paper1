"""
================================================================================
rq3_00_config.py — Configurazione centralizzata per RQ3 (Duffie Analysis)
================================================================================
Research Question 3:
"Are there interdependencies across arbitrage strategies that indicate
capital reallocation or slow-moving arbitrage capital, as suggested by
Duffie (2010)?"

Tutti i parametri per i file rq3_01..rq3_05 sono centralizzati qui.

Author: Alessio Ottaviani
Institution: EDHEC Business School — PhD Thesis
Advisor: Prof. Rebonato
================================================================================
"""

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
#  1. PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"

# --- Input: strategy daily index files ---
STRATEGY_INDEX_FILES = {
    "cds_bond_basis":   RESULTS_DIR / "cds_bond_basis"   / "index_daily.csv",
    "btp_italia":       RESULTS_DIR / "btp_italia"        / "index_daily.csv",
    "itraxx_combined":  RESULTS_DIR / "itraxx_combined"   / "index_daily.csv",
}

# --- Input: trades logs (per threshold clustering e basis reconstruction) ---
STRATEGY_TRADES_FILES = {
    "cds_bond_basis":   RESULTS_DIR / "cds_bond_basis"   / "trades_log.csv",
    "btp_italia":       RESULTS_DIR / "btp_italia"        / "trades_log.csv",
    "itraxx_combined":  RESULTS_DIR / "itraxx_combined"   / "trades_log.csv",
}

# --- Input: raw basis data (per mispricing levels m_{i,t}) ---
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_BASIS_FILES = {
    "btp_italia": {
        "path": PROCESSED_DATA_DIR / "basis_wide.parquet",
        "format": "wide",
        "suffix": "_Basis",
        # Maturity: servono per filtrare bond vicini a scadenza
        "maturity_source": "excel",
        "excel_path": DATA_DIR / "raw" / "BTP_Italia_basis.xlsx",
        "excel_sheet": "Sheet1",
    },
    "cds_bond_basis": {
        "path": PROCESSED_DATA_DIR / "cds_bond_basis_long.parquet",
        "format": "long",
        "basis_col": "Basis",
        "date_col": "date",
        "maturity_col": "Maturity",
    },
    "itraxx_combined": {
        "paths": {
            "Main":   PROCESSED_DATA_DIR / "itraxx_main"   / "itraxx_basis_wide.parquet",
            "SnrFin": PROCESSED_DATA_DIR / "itraxx_snrfin" / "itraxx_basis_wide.parquet",
            "SubFin": PROCESSED_DATA_DIR / "itraxx_subfin" / "itraxx_basis_wide.parquet",
            "Xover":  PROCESSED_DATA_DIR / "itraxx_xover"  / "itraxx_basis_wide.parquet",
        },
        "format": "wide_multi",
        "suffix": "_Basis",
    },
}

# --- Input: pre-computed monthly factors (stationary, first differences) ---
FACTORS_PATH = DATA_DIR / "processed" / "all_factors_monthly.parquet"

# --- Input: raw data files for stress proxy LEVELS ---
FACTORS_EXTERNAL_DIR = DATA_DIR / "external" / "factors"

NONTRADABLE_FILE = FACTORS_EXTERNAL_DIR / "Nontradable_risk_factors.xlsx"
TRADABLE_CB_FILE = FACTORS_EXTERNAL_DIR / "Tradable_corporate_bond_factors.xlsx"

# --- Output ---
RQ3_OUTPUT_DIR = RESULTS_DIR / "rq3_duffie"

RQ3_FIGURES_DIR = RQ3_OUTPUT_DIR / "figures"
RQ3_TABLES_DIR  = RQ3_OUTPUT_DIR / "tables"
RQ3_DATA_DIR    = RQ3_OUTPUT_DIR / "data"

for _d in [RQ3_OUTPUT_DIR, RQ3_FIGURES_DIR, RQ3_TABLES_DIR, RQ3_DATA_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


# ============================================================================
#  2. STRATEGY METADATA
# ============================================================================

STRATEGY_NAMES = ["cds_bond_basis", "btp_italia", "itraxx_combined"]

STRATEGY_LABELS = {
    "cds_bond_basis":  "CDS–Bond Basis",
    "btp_italia":      "BTP Italia",
    "itraxx_combined": "iTraxx Combined",
}

STRATEGY_PAIRS = [
    ("cds_bond_basis", "btp_italia"),
    ("cds_bond_basis", "itraxx_combined"),
    ("btp_italia",     "itraxx_combined"),
]

STRATEGY_COLORS = {
    "cds_bond_basis":  "#1f77b4",
    "btp_italia":      "#d62728",
    "itraxx_combined": "#2ca02c",
}


# ============================================================================
#  3. MISPRICING MAGNITUDE — Definizione di m_{i,t}
# ============================================================================
# Coerente con Duffie (2010): widening ⟺ Δm > 0 per tutte le strategie.

MISPRICING_TRANSFORM = {
    "cds_bond_basis":  "negate_negative_only",
    # m_t = -basis_t  SE basis_t < 0  →  m_t ≥ 0, cresce col mispricing
    #        NaN       SE basis_t ≥ 0  (nessuna opportunità di arbitraggio)

    "btp_italia":      "absolute",
    # m_t = |basis_t|  (si entra long e short → mispricing = distanza da zero)

    "itraxx_combined": "absolute",
    # m_t = |basis_t|  (idem)
}

# Variante per robustness: m = max(-basis, 0) → nessun NaN, Δm sempre definita
# Serve per verificare che i risultati non siano artefatto della sample selection
# (quando basis ≥ 0, m = 0 invece di NaN → nessun mese escluso)
MISPRICING_TRANSFORM_ROBUST = {
    "cds_bond_basis":  "negate_clamp_zero",   # m = max(-basis, 0)
    "btp_italia":      "absolute",            # invariato
    "itraxx_combined": "absolute",            # invariato
}

# ⭐ SCELTA APPROCCIO PER IL CALCOLO DELLA BASIS GIORNALIERA ⭐
#
#   "market"    → Dai dati raw di mercato (configurazione sotto)
#   "portfolio" → Media degli entry_basis dei trade APERTI (dal trades_log)
#
BASIS_APPROACH = "market"

# --- BTP Italia: filtro maturity ---
# Mesi minimi a scadenza per includere un bond nella media cross-sectional.
# Bond vicini a scadenza hanno basis erratiche (convergenza, illiquidità).
BTP_MIN_MONTHS_TO_MATURITY = 6

# --- CDS-Bond Basis: metodo di aggregazione cross-sectional ---
#
#   "top_n_negative"  → Filtra basis < 0, media delle min(N, n_negative) più negative.
#                        Robusto: non dipende da soglia, un singolo outlier non domina.
#                        Se nessuna basis negativa nel giorno → NaN.
#
#   "threshold_only"  → Filtra basis < CDS_BOND_ENTRY_THRESHOLD (es. -40 bps), media.
#                        Coerente con la soglia di entry della strategia.
#                        Se nessun bond sotto soglia → NaN.
#
#   "top_n_all"       → Ordina TUTTE le basis, media delle N più negative (anche positive).
#                        Sempre un valore (mai NaN), ma meno economicamente pulito.
#
CDS_BOND_BASIS_METHOD = "top_n_negative"
CDS_BOND_TOP_N = 20
CDS_BOND_ENTRY_THRESHOLD = -40   # bps, per metodo "threshold_only"


# ============================================================================
#  4. STRESS PROXY — Caricamento livelli per definizione dei regimi
# ============================================================================

# 4a. iTraxx Main 5Y
ITRAXX_MAIN_SHEET    = "CDS_INDEX"
ITRAXX_MAIN_SKIPROWS = 14
ITRAXX_MAIN_USECOLS  = [0, 1]
ITRAXX_MAIN_COLNAMES = ["Date", "ITRX_MAIN"]

# 4b. iTraxx Crossover 5Y — ⚠️ sheet = "ITRX" (Main è su entrambi, Xover solo su ITRX)
ITRAXX_XOVER_SHEET    = "ITRX"
ITRAXX_XOVER_SKIPROWS = 14
ITRAXX_XOVER_USECOLS  = [0, 5]
ITRAXX_XOVER_COLNAMES = ["Date", "ITRX_XOVER"]

# 4c. VIX e V2X
VIX_SHEET    = "VIX"
VIX_SKIPROWS = 14
VIX_USECOLS  = [0, 1, 2]
VIX_COLNAMES = ["Date", "VIX", "V2X"]


# ============================================================================
#  5. REGIME DEFINITION — Soglie di stress
# ============================================================================

MANUAL_THRESHOLDS = {
    "ITRX_MAIN":  {"low_upper":  60.0, "high_lower": 120.0},
    "ITRX_XOVER": {"low_upper": 250.0, "high_lower": 400.0},
    "VIX":         {"low_upper":  15.0, "high_lower":  30.0},
    "V2X":         {"low_upper":  18.0, "high_lower":  35.0},
}

PERCENTILE_THRESHOLDS = {
    "low_upper":  50,
    "high_lower": 90,
}

# Regime mode da usare come DEFAULT nei test
# "manual" oppure "percentile"
DEFAULT_REGIME_MODE = "manual"

# Regime a 2 livelli (NORMAL/HIGH) — alternativa a LOW/MEDIUM/HIGH
# NORMAL = Main <= high_lower, HIGH = Main > high_lower
REGIME_2L = True   # se True, calcola anche la versione a 2 livelli

# Burn-in: skip first N months of each strategy's returns to avoid
# noisy returns from periods with few active trades.
BURN_IN_MONTHS = 0

DEFAULT_STRESS_PROXY = "ITRX_MAIN"
ALL_STRESS_PROXIES = ["ITRX_MAIN", "ITRX_XOVER", "VIX", "V2X"]
PERCENTILE_START_DATE = "2004-01-01"


# ============================================================================
#  6. ENTRY THRESHOLDS — Per threshold clustering test
# ============================================================================

ENTRY_THRESHOLDS = {
    "cds_bond_basis": {"type": "negative_only", "threshold_bps": 40},
    "btp_italia":     {"type": "two_sided", "threshold_long_bps": 40,
                       "threshold_short_bps": 50, "threshold_bps": 40},
    "itraxx_combined": {"type": "two_sided", "threshold_bps": 10},
}


# ============================================================================
#  7. ANALYSIS PARAMETERS
# ============================================================================

ROLLING_WINDOW_MONTHS = 24
ROLLING_MIN_OBS = 18

NW_MAX_LAGS = 3
SPANNING_MAX_LAGS = 2

VAR_MAX_LAGS = 6
VAR_IRF_PERIODS = 6
VAR_IRF_CI = 0.95
VAR_BOOTSTRAP_REPS = 1000

BOOTSTRAP_N_REPS = 5000
BOOTSTRAP_BLOCK_SIZE = 4

TAIL_QUANTILE = 0.90

DCC_GARCH_ORDER = (1, 1)
DCC_ORDER = (1, 1)

QUANTILE_TAUS = [0.10, 0.25, 0.50, 0.75, 0.90]

PCA_ROLLING_WINDOW = 36


# ============================================================================
#  8. RESIDUALIZATION — Fattori comuni per "purged" analysis
# ============================================================================

# LOGICA del PURGING (Duffie 2010):
# L'obiettivo è togliere dai Δm i movimenti dovuti a rischio FONDAMENTALE
# (tassi, credit, sovrano, volatilità), per isolare la componente di
# correlazione residua dovuta a slow-moving capital.
# ⚠️ NON includere fattori di liquidità/funding (EBP, LIBOR_REPO, TED, etc.)
# perché quelli SONO il canale Duffie che stiamo cercando di misurare.

# Set "base": fattori fondamentali significativi per ≥2 strategie
# (da risultati AEN: TERM_EU/US, DEF_US, SS10Y, BTP_BUND, ΔV2X)
COMMON_FACTORS_FOR_PURGING = [
    "Δ10Y_YIELD_EU",      # rate risk (fondamentale per tutte)
    "ΔV2X",               # volatilità/uncertainty
    "SS10Y",              # swap spread (signif. per BTP e CDS-Bond)
]

# Set "extended": aggiunge credit e term structure
COMMON_FACTORS_EXTENDED = [
    "Δ10Y_YIELD_EU",
    "ΔV2X",
    "SS10Y",
    "DEF_US",             # default risk (signif. per CDS-Bond: β=+0.11***)
    "BTP_BUND",           # rischio sovrano (signif. per BTP: β=+0.00**)
]

# Set "no_stress": SENZA variabili usate per definire il regime (ITRX_MAIN)
# e SENZA proxy di mercato creditizio che correlano con ITRX_MAIN
# Risolve l'obiezione di circolarità purge-regime
COMMON_FACTORS_NO_STRESS = [
    "Δ10Y_YIELD_EU",
    "ΔV2X",
    "TERM_EU",            # term structure (signif. per BTP: β=-0.02**)
    "ΔSLOPE_EU",
]

# Set "macro_only": solo rischio fondamentale puro, zero mercato
COMMON_FACTORS_MACRO_ONLY = [
    "Δ10Y_YIELD_EU",
    "ΔSLOPE_EU",
    "5Y5Y_INFL",
]

PURGE_FACTOR_SET = "base"

# Tutti i set disponibili per robustezza
PURGE_SETS = {
    "base":        COMMON_FACTORS_FOR_PURGING,
    "extended":    COMMON_FACTORS_EXTENDED,
    "no_stress":   COMMON_FACTORS_NO_STRESS,
    "macro_only":  COMMON_FACTORS_MACRO_ONLY,
}

# --- Fattori per regressione multivariata PC1 ~ funding stress ---
#
# LOGICA (Duffie 2010): il fattore comune (PC1 delle 3 serie Δm) dovrebbe
# essere spiegato da variabili che catturano lo stato del capitale degli
# intermediari finanziari. Se PC1 ~ funding stress è significativo, abbiamo
# evidenza diretta che il co-movement dei mispricings è guidato da vincoli
# di bilancio dei dealer, non da fattori fondamentali (quelli sono purgati).
#
# I fattori sono organizzati per canale di Duffie:
#   Canale 1 — Intermediary capital (quanto capitale hanno i dealer)
#   Canale 2 — Funding cost (quanto costa prendere a prestito)
#   Canale 3 — Market illiquidity (quanto costa eseguire il trade)
#
# Panel A: Core — un fattore per canale, dalla letteratura di riferimento
# Panel B: Extended — tutti i fattori di funding/liquidity disponibili

# Panel A: Core Intermediary Proxies (one per Duffie channel)
# Panel A: Core Intermediary Proxies (one per Duffie channel)
#   Channel 1 — Intermediary capital:  HKM_IC (He-Kelly-Manela 2017)
#   Channel 1b — Dealer health:        PB_EU_CDS_5Y (European prime broker CDS)
#   Channel 2 — Funding cost:          EURIBOR_OIS (interbank funding stress)
#   Channel 3 — Market illiquidity:    ILLIQ (Amihud illiquidity)
PC1_FUNDING_FACTORS = [
    "HKM_IC",          # Intermediary capital (He-Kelly-Manela 2017)
    "LIBOR_OIS",       # Funding cost (global dealer funding in USD)
    "PB_EU_CDS_5Y",    # Dealer health EUR
    "ILLIQ",           # Market illiquidity (Amihud 2002)
]

# Panel B: Extended Proxy Set (Panel A + additional proxies per channel)
PC1_FUNDING_FACTORS_FULL = [
    "HKM_IC",          # Intermediary capital
    "LIBOR_OIS",       # Funding cost (global)
    "PB_EU_CDS_5Y",    # Dealer health EUR
    "ILLIQ",           # Market illiquidity
    "GC-REPO_T-BILL",  # Repo market stress (Bai-Collin-Dufresne 2019)
    "BFCI_EU",         # Financial conditions EUR (composite)
    "ΔFAILS_PCT_TSY",  # Settlement fails (Fleckenstein et al. 2014)
]


# ============================================================================
#  8b. BOOTSTRAP SETTINGS
# ============================================================================

BOOTSTRAP_METHOD = "both"  # "circular", "stationary", "both"
BOOTSTRAP_BLOCK_SIZES = [2, 4, 6]  # per sensitivity analysis


# ============================================================================
#  9. MACRO EVENTS OVERLAY
# ============================================================================

MACRO_EVENTS = {
    "2011-07-01": "Eurozone\nCrisis",
    "2013-05-22": "Taper\nTantrum",
    "2015-08-24": "China\nDevaluation",
    "2018-02-05": "Volmageddon",
    "2020-03-12": "COVID\nCrash",
    "2022-02-24": "Ukraine\nWar",
    "2022-09-28": "UK LDI\nCrisis",
}


# ============================================================================
# 10. MULTIPLE TESTING CORRECTION
# ============================================================================

MTC_METHOD = "fdr_bh"
MTC_ALPHA  = 0.05

PRIMARY_OUTCOMES = [
    "co_widening_excess_prob_HIGH_vs_LOW",
    "interaction_beta2_stress",
    "granger_causality_F_test",
]


# ============================================================================
# 11. SUB-PERIOD & PAIRWISE LONG-SAMPLE ANALYSIS
# ============================================================================

SUB_PERIODS = {
    "Pre-COVID":    ("2005-01-01", "2020-02-29"),
    "Post-COVID":   ("2020-03-01", "2025-12-31"),
    "ECB Hiking":   ("2022-07-01", "2024-06-30"),
}

# Coppie bivariate con campione più lungo del trivariate overlap
PAIRWISE_LONG = {
    ("cds_bond_basis", "itraxx_combined"): {
        "label": "CDS–Bond vs iTraxx (long sample)",
        "start": "2008-01-01",
    },
}


# ============================================================================
# 12. FIGURE SETTINGS
# ============================================================================

FIGURE_DPI = 150
FIGURE_FORMAT = "pdf"


# ============================================================================
#  LATEX BEAMER TABLE EXPORT HELPERS
# ============================================================================
# Produce .tex files ready for \input{} in beamer slides
# Style matches AEN_Stable_OLS slides (CambridgeUS theme)

def _sig_stars(p):
    """Return significance stars."""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

def _fmt_coeff(val, decimals=2):
    """Format coefficient with sign."""
    return f"{val:+.{decimals}f}"

def export_granger_tex(df, filepath, title="Granger Causality Tests"):
    """Export Granger causality table as beamer slide .tex."""
    lines = []
    lines.append(r"\begin{frame}[t]{" + title + "}")
    lines.append(r"\centering")
    lines.append(r"\vspace{-0.3cm}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    lines.append(r"\begin{tabular}{l l r r l}")
    lines.append(r"\toprule")
    lines.append(r"Causing & Caused & $F$-stat & $p$-value & \\")
    lines.append(r"\midrule")
    
    for _, row in df.iterrows():
        causing = str(row.get('causing', row.get('Causing', '')))
        caused = str(row.get('caused', row.get('Caused', '')))
        f_stat = row.get('F_stat', row.get('f_stat', 0))
        p_val = row.get('p_value', row.get('p', 0))
        stars = _sig_stars(p_val)
        lines.append(f"  {causing} & {caused} & {f_stat:.3f} & {p_val:.4f} & {stars} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{0.3cm}")
    lines.append(r"\footnotesize")
    lines.append(r"\textit{Notes:} VAR estimated on $\Delta m$ (monthly mispricing changes). "
                 r"$F$-statistics from Wald test of zero restrictions on lagged coefficients. "
                 r"*** $p{<}1\%$, ** $p{<}5\%$, * $p{<}10\%$.")
    lines.append(r"\end{frame}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"💾 {filepath.name}")


def export_interaction_tex(df, filepath, title="Interaction Regressions — Stress Amplification"):
    """Export interaction regression table as beamer slide .tex."""
    lines = []
    lines.append(r"\begin{frame}[t,shrink=10]{" + title + "}")
    lines.append(r"\centering")
    lines.append(r"\vspace{-0.3cm}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    lines.append(r"\begin{tabular}{l l r r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Dep. & Indep. & $\beta_1$ & $t_1$ & "
                 r"$\beta_2$ & $t_2$ & $\beta_3$ & $t_3$ & $R^2$ \\")
    lines.append(r"\midrule")
    
    for _, row in df.iterrows():
        dep = str(row.get('dependent', ''))
        indep = str(row.get('independent', ''))
        b1 = row.get('beta1', 0)
        t1 = row.get('beta1_t', 0)
        b2 = row.get('beta2', 0)
        t2 = row.get('beta2_t', 0)
        b3 = row.get('beta3', 0)
        t3 = row.get('beta3_t', 0)
        r2 = row.get('R2', 0)
        
        stars2 = _sig_stars(row.get('beta2_p', 1))
        
        lines.append(f"  {dep} & {indep} & {_fmt_coeff(b1)} & {t1:+.2f} & "
                      f"{_fmt_coeff(b2, 4)}{stars2} & {t2:+.2f} & "
                      f"{_fmt_coeff(b3)} & {t3:+.2f} & {r2:.3f} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{0.25cm}")
    lines.append(r"\footnotesize")
    lines.append(r"$\Delta m_i = \alpha + \beta_1 \Delta m_j + "
                 r"\beta_2 (\Delta m_j \times \mathrm{Stress}) + "
                 r"\beta_3 \mathrm{Stress} + \varepsilon$. "
                 r"Stress is standardized (zero mean, unit variance). "
                 r"HAC standard errors (Newey--West). "
                 r"*** $p{<}1\%$, ** $p{<}5\%$, * $p{<}10\%$.")
    lines.append(r"\end{frame}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"💾 {filepath.name}")


def export_pc1_funding_tex(results_by_set, filepath,
                            title="Common Factor (PC1) and Funding Stress"):
    """Export PC1 ~ funding regression results as beamer slide .tex.
    results_by_set: dict of {set_name: {factors: [...], betas, tvals, pvals, R2, R2_adj, T}}
    """
    lines = []
    lines.append(r"\begin{frame}[t,shrink=12]{" + title + "}")
    lines.append(r"\centering")
    lines.append(r"\vspace{-0.3cm}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.05}")
    
    for set_name, res in results_by_set.items():
        lines.append(r"\vspace{0.2cm}")
        lines.append(r"\textbf{Set: " + set_name + r"}" + 
                      f" ($T={res['T']}$, $R^2_{{adj}}={res['R2_adj']:.3f}$)")
        lines.append(r"\vspace{0.1cm}")
        lines.append(r"\begin{tabular}{l r r r l}")
        lines.append(r"\toprule")
        lines.append(r"Factor & $\beta$ & $t$-stat & $p$-value & \\")
        lines.append(r"\midrule")
        
        for i, factor in enumerate(res['factors']):
            b = res['betas'][i]
            t = res['tvals'][i]
            p = res['pvals'][i]
            stars = _sig_stars(p)
            factor_tex = factor.replace("_", r"\_")
            lines.append(f"  $\\mathrm{{{factor_tex}}}$ & {_fmt_coeff(b, 4)} & "
                          f"{t:+.2f} & {p:.4f} & {stars} \\\\")
        
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
    
    lines.append(r"\vspace{0.25cm}")
    lines.append(r"\footnotesize")
    lines.append(r"\textit{Notes:} PC1 extracted from $\Delta m$ of the three strategies "
                 r"(44.8\% of variance explained). OLS with Newey--West HAC standard errors. "
                 r"*** $p{<}1\%$, ** $p{<}5\%$, * $p{<}10\%$.")
    lines.append(r"\end{frame}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"💾 {filepath.name}")


def export_cowidening_proxy_tex(df, filepath,
                                 title="Co-Widening Across Stress Proxies"):
    """Export co-widening by proxy × pair as beamer slide .tex."""
    lines = []
    lines.append(r"\begin{frame}[t,shrink=15]{" + title + "}")
    lines.append(r"\centering")
    lines.append(r"\vspace{-0.3cm}")
    lines.append(r"\tiny")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.00}")
    lines.append(r"\begin{tabular}{l l l r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"Proxy & Mode & Pair & CW$_{\mathrm{LOW}}$ & CW$_{\mathrm{HIGH}}$ & "
                 r"$\Delta$CW & $\rho_{\mathrm{LOW}}$ & $\rho_{\mathrm{HIGH}}$ & $\Delta\rho$ \\")
    lines.append(r"\midrule")
    
    last_proxy = ""
    for _, row in df.iterrows():
        proxy = str(row['proxy'])
        mode = str(row['mode'])
        pair = str(row['pair'])
        cw_l = row['P_cowiden_LOW']
        cw_h = row['P_cowiden_HIGH']
        d_cw = row['diff_cowiden']
        rho_l = row.get('rho_returns_LOW', float('nan'))
        rho_h = row.get('rho_returns_HIGH', float('nan'))
        d_rho = row.get('diff_rho', float('nan'))
        
        # Add midrule between proxies
        if proxy != last_proxy and last_proxy != "":
            lines.append(r"\addlinespace")
        last_proxy = proxy
        
        # Short pair names
        pair_short = pair.replace("CDS–Bond Basis", "CDS-Bond") \
                         .replace("iTraxx Combined", "iTraxx") \
                         .replace("BTP Italia", "BTP")
        
        import math
        rho_l_s = f"{rho_l:+.3f}" if not math.isnan(rho_l) else "--"
        rho_h_s = f"{rho_h:+.3f}" if not math.isnan(rho_h) else "--"
        d_rho_s = f"{d_rho:+.3f}" if not math.isnan(d_rho) else "--"
        
        lines.append(f"  {proxy} & {mode} & {pair_short} & "
                      f"{cw_l:.3f} & {cw_h:.3f} & {d_cw:+.3f} & "
                      f"{rho_l_s} & {rho_h_s} & {d_rho_s} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{0.2cm}")
    lines.append(r"\footnotesize")
    lines.append(r"CW = P(both $\Delta m > 0$). $\rho$ = Pearson correlation of returns. "
                 r"Regimes: LOW/HIGH defined by manual thresholds or sample percentiles.")
    lines.append(r"\end{frame}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"💾 {filepath.name}")


def export_scorecard_tex(df, filepath, title="Duffie (2010) Scorecard"):
    """Export Duffie scorecard as beamer slide .tex."""
    lines = []
    lines.append(r"\begin{frame}[t,shrink=10]{" + title + "}")
    lines.append(r"\centering")
    lines.append(r"\vspace{-0.2cm}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.10}")
    lines.append(r"\begin{tabular}{l r c c c c c c r}")
    lines.append(r"\toprule")
    lines.append(r"Period & $T$ & P1 & P2 & P3 & P4 & P5 & P6 & Score \\")
    lines.append(r" & & Corr. & Co-wid. & Span. & Granger & PCA & Persist. & \\")
    lines.append(r"\midrule")
    
    for _, row in df.iterrows():
        period = str(row.get('period', ''))
        T = int(row.get('T', 0))
        score = int(row.get('score', 0))
        
        def check(val):
            if isinstance(val, bool):
                return r"\checkmark" if val else r"$\times$"
            if isinstance(val, str):
                return r"\checkmark" if val.lower() == 'true' else r"$\times$"
            try:
                import math
                if math.isnan(float(val)):
                    return "--"
            except:
                pass
            return r"\checkmark" if val else r"$\times$"
        
        p1 = check(row.get('P1_correlation', False))
        p2 = check(row.get('P2_cowidening', False))
        p3 = check(row.get('P3_spanning', False))
        p4 = check(row.get('P4_granger', False))
        p5 = check(row.get('P5_pca', False))
        p6 = check(row.get('P6_persistence', False))
        
        lines.append(f"  {period} & {T} & {p1} & {p2} & {p3} & {p4} & {p5} & {p6} & "
                      f"\\textbf{{{score}}} \\\\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{0.3cm}")
    lines.append(r"\footnotesize")
    lines.append(r"\textit{Notes:} P1 = significant cross-market correlation. "
                 r"P2 = higher co-widening in HIGH stress. "
                 r"P3 = spanning regressions significant. "
                 r"P4 = Granger causality detected. "
                 r"P5 = PC1 $>$ 50\% of variance. "
                 r"P6 = higher persistence in HIGH stress.")
    lines.append(r"\end{frame}")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"💾 {filepath.name}")
FIGURE_STYLE = "seaborn-v0_8-whitegrid"

REGIME_COLORS = {
    "LOW":    "#2ca02c",
    "MEDIUM": "#ff7f0e",
    "HIGH":   "#d62728",
}

REGIME_ALPHAS = {
    "LOW":    0.15,
    "MEDIUM": 0.10,
    "HIGH":   0.20,
}


# ============================================================================
# 13. PRINT CONFIG SUMMARY
# ============================================================================

def print_config_summary():
    """Stampa riepilogo configurazione RQ3."""
    print("=" * 72)
    print("RQ3 DUFFIE ANALYSIS — CONFIGURATION SUMMARY")
    print("=" * 72)
    print(f"  PROJECT_ROOT:          {PROJECT_ROOT}")
    print(f"  RQ3_OUTPUT_DIR:        {RQ3_OUTPUT_DIR}")
    print(f"  Strategies:            {', '.join(STRATEGY_LABELS.values())}")
    print(f"  Basis approach:        {BASIS_APPROACH}")
    print(f"  CDS-Bond method:       {CDS_BOND_BASIS_METHOD} (N={CDS_BOND_TOP_N})")
    print(f"  BTP maturity filter:   >= {BTP_MIN_MONTHS_TO_MATURITY} months")
    print(f"  Default stress proxy:  {DEFAULT_STRESS_PROXY}")
    print(f"  Default regime mode:   {DEFAULT_REGIME_MODE}")
    print(f"  Manual thresholds:     {MANUAL_THRESHOLDS[DEFAULT_STRESS_PROXY]}")
    print(f"  Percentile thresholds: {PERCENTILE_THRESHOLDS}")
    print(f"  Rolling window:        {ROLLING_WINDOW_MONTHS} months")
    print(f"  Newey-West lags:       {NW_MAX_LAGS}")
    print(f"  VAR max lags:          {VAR_MAX_LAGS}")
    print(f"  VAR bootstrap reps:    {VAR_BOOTSTRAP_REPS}")
    print(f"  Bootstrap reps:        {BOOTSTRAP_N_REPS}")
    print(f"  Tail quantile:         {TAIL_QUANTILE}")
    print(f"  Common factors purge:  {COMMON_FACTORS_FOR_PURGING}")
    print(f"  PC1 funding factors:   {PC1_FUNDING_FACTORS}")
    print(f"  MTC method:            {MTC_METHOD} (α={MTC_ALPHA})")
    print("=" * 72)


if __name__ == "__main__":
    print_config_summary()
    print()
    for name, path in STRATEGY_INDEX_FILES.items():
        print(f"  {name}: {path}  (exists: {path.exists()})")
    print(f"\n  FACTORS_PATH: {FACTORS_PATH}  (exists: {FACTORS_PATH.exists()})")
    print(f"  NONTRADABLE_FILE: {NONTRADABLE_FILE}  (exists: {NONTRADABLE_FILE.exists()})")
    print(f"  TRADABLE_CB_FILE: {TRADABLE_CB_FILE}  (exists: {TRADABLE_CB_FILE.exists()})")
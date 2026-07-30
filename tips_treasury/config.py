"""
================================================================================
config.py — Configurazione centralizzata per il Paper 2 (TIPS–Treasury Floor)
================================================================================
"The TIPS–Treasury Basis as a Limits-to-Arbitrage Equilibrium Floor"

Tutti i parametri per i file 01..09 sono centralizzati qui.
Convenzioni identiche al Paper 1 (pathlib, run dalla cartella del package).

Author: Alessio Ottaviani
Institution: EDHEC Business School — PhD Thesis
Advisor: Prof. Rebonato
================================================================================
"""
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ----------------------------- paths -----------------------------------------
ROOT   = Path(__file__).resolve().parents[2]          # .../THESIS
RAW    = ROOT / "data" / "raw"
BBG    = RAW / "Bloomberg"
PROC   = ROOT / "data" / "processed" / "tips"
RES    = ROOT / "results" / "tips"
FIG    = RES / "figures"
TABDIR = ROOT / "paper" / "tables_paper2"

FILE_TIPS_VARS = BBG / "TIPS_variables.xlsx"          # panel 193 serie (curve+stress, USA/EUR/UK)
FILE_CARTEL1   = BBG / "TIPS_nominals.xlsx"           # nominali+BE+USSP+CPURNSA (ex Cartel1)
FILE_CUSIP     = BBG / "tips_cusip.xlsx"              # LEGACY: pannello precalcolato (solo 08 storico)
FILE_HKM       = RAW / "He_Kelly_Manela" / "He_Kelly_Manela_Factors_monthly_250627.csv"
DIR_NETPOS     = RAW / "Net_position"                 # 6 export FR2004
DIR_OFR        = RAW / "OFR"                          # data(1).csv, data(2).csv, tff.csv
FILE_FLLPAIRS  = RAW / "FLL" / "fll_pairs_tableIII.csv"
# --- bond-level engine (10) ---
FILE_PAIRS     = RAW / "pairs" / "pairs_engine_us.csv"
FILE_LINK_US   = BBG / "inflation_linked_us.csv"      # TIPS per-ISIN (long: Date;ISIN;YL017)
FILE_TSY_EXP   = BBG / "treasury_ISINs_expanded.csv"  # nominali per-ISIN (long, 6 colonne)
FILE_GSW       = RAW / "Fed Board" / "feds200628.csv" # curva fitted GSW (SVENPY par yields)
# --- linker internazionali (11, prossimo modulo) ---
FILE_LINK_FR   = BBG / "inflation_linked_fra.csv"
FILE_LINK_DE   = BBG / "inflation_linked_ger.csv"
FILE_LINK_IT   = BBG / "inflation_linked_ita.csv"
FILE_LINK_UK   = BBG / "inflation_linked_uk.csv"

for d in (PROC, RES, FIG, TABDIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------- parametri -------------------------------------
MATS        = [2, 5, 10, 20, 30]
FLL_A, FLL_B = "2004-07-23", "2009-11-19"             # finestra FLL (2014)
FLL_MEAN_BP, FLL_PEAK_BP = 54.5, 175.0

POST13      = "2013-01"
NW_LAGS     = 6

# episodi: (onset, window_start, window_end) — onset per la regola pre uniforme E4
EPISODES = {
    "GFC":      ("2008-09-15", "2008-09-01", "2009-06-30"),
    "Mar-2020": ("2020-02-20", "2020-02-20", "2020-06-30"),
    "2022":     ("2022-03-16", "2022-02-15", "2023-03-31"),
    "SVB-2023": ("2023-03-09", "2023-03-05", "2023-09-30"),
}
PRE_RULE_BD, PRE_RULE_GAP = 60, 5                     # trailing 60bd, stop 5bd prima dell'onset

# design del constraint test — CONGELATO (Pilot Report v3, §4.3, 15-07-2026)
LAG_H       = 3                                        # y = med_{t+3} − med_{t−1}
Q_STRESS    = 0.20                                     # quintile inferiore HKM (pre-specificato, H3)
GRID_Q      = (0.10, 0.50, 9)                          # griglia sup-F
N_BOOT      = 999
BOOT_BLOCK  = 4
HAC_LAGS    = 4

# engine strutturale
D_DUR, Y0, DPM = 8.0, 0.04, 21
H_RISK_M, T_MAT_M = 24, 120
BUFFERS_BP  = (100, 200)
RRA_GRID    = (0, 1, 2, 5)
HURDLE      = 0.10
N_PATHS     = 60_000
SEED        = 77

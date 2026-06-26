"""
================================================================================
00_pca_config.py - Configurazione Pipeline PCA
================================================================================
Parametri centralizzati per la pipeline PCA rolling.
Riferimento: Ludvigson & Ng (2009), "Macro Factors in Bond Risk Premia"

Author: Alessio Ottaviani
Institution: EDHEC Business School - PhD Thesis
================================================================================
"""

from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================================
# INPUT FILES
# ============================================================================

FACTORS_PATH = DATA_DIR / "processed" / "all_factors_monthly.parquet"

# Strategie (stesso formato di ml_config)
STRATEGIES = {
    "btp_italia": RESULTS_DIR / "btp_italia" / "index_daily.csv",
    "cds_bond_basis": RESULTS_DIR / "cds_bond_basis" / "index_daily.csv",
    "itraxx_combined": RESULTS_DIR / "itraxx_combined" / "index_daily.csv"
}

# ============================================================================
# DATA AVAILABILITY
# ============================================================================

FACTORS_END_DATE = "2025-05-31"  # Data massima per i fattori

# ============================================================================
# ⭐ PCA START DATE - MODIFICA QUI PER CAMBIARE DATA INIZIO ⭐
# ============================================================================
# Questa è la data da cui inizia il calcolo dei PC scores.
# La prima rolling window sarà [PCA_START_DATE - PCA_WINDOW_LENGTH, PCA_START_DATE - 1]
#
# Date testate:
#   - 2008-01-31: 69 fattori (mancano ATM_IV_ITRX, ATM_IV_CDX)
#   - 2010-01-31: 71 fattori (tutti disponibili)
#   - 2005-09-30: 19 fattori (solo per match con inizio itraxx_combined)

PCA_START_DATE = "2008-01-31"  # ← MODIFICA QUESTA RIGA PER CAMBIARE DATA INIZIO

# ============================================================================
# PCA ROLLING WINDOW PARAMETERS
# ============================================================================

# Lunghezza rolling window in mesi
# Rebonato suggerisce minimo 24 mesi (2 anni)
PCA_WINDOW_LENGTH = 24  # ← MODIFICA PER CAMBIARE LUNGHEZZA WINDOW

# ============================================================================
# ⭐ NUMERO DI PRINCIPAL COMPONENTS ⭐
# ============================================================================
# IMPORTANTE: Per avere un alpha interpretabile, il numero di PC deve essere
# FISSO per tutta la regressione.
#
# Ludvigson & Ng (2009) usano 5-6 fattori.
# Dalla nostra analisi, ~11 PC spiegano 80% della varianza dei 69 fattori.

PCA_N_COMPONENTS = 8  # ← MODIFICA PER CAMBIARE NUMERO PC

# ============================================================================
# PCA VARIANCE THRESHOLD (solo per diagnostica)
# ============================================================================

# Soglia varianza spiegata cumulativa - usata solo per diagnostica/reporting
# Es: 0.80 = tieni PC che spiegano almeno 80% della varianza
PCA_VARIANCE_THRESHOLD = 0.80  # ← MODIFICA PER CAMBIARE SOGLIA

# ============================================================================
# PCA TIMING CONVENTION
# ============================================================================

# "predictive":      PC_t usati per spiegare R_{t+1} (come Ludvigson & Ng)
# "contemporaneous": PC_t usati per spiegare R_t

PCA_TIMING = "contemporaneous"  # ← MODIFICA PER CAMBIARE TIMING

# ============================================================================
# OUTPUT DIRECTORIES
# ============================================================================

def get_pca_output_dir() -> Path:
    """Ritorna directory output per PCA comune a tutte le strategie."""
    output_dir = RESULTS_DIR / "pca"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_strategy_pca_dir(strategy_name: str) -> Path:
    """Ritorna directory output PCA per singola strategia."""
    output_dir = RESULTS_DIR / strategy_name / "pca"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# HELPER: PRINT CONFIG SUMMARY
# ============================================================================

def print_config_summary():
    """Stampa riepilogo configurazione corrente."""
    print("=" * 70)
    print("PCA CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"  PCA_START_DATE:        {PCA_START_DATE}")
    print(f"  PCA_WINDOW_LENGTH:     {PCA_WINDOW_LENGTH} months")
    print(f"  PCA_N_COMPONENTS:      {PCA_N_COMPONENTS}")
    print(f"  PCA_VARIANCE_THRESHOLD:{PCA_VARIANCE_THRESHOLD:.0%} (for diagnostics)")
    print(f"  PCA_TIMING:            {PCA_TIMING}")
    print(f"  FACTORS_END_DATE:      {FACTORS_END_DATE}")
    print("=" * 70)


if __name__ == "__main__":
    print_config_summary()
    print(f"\nPROJECT_ROOT: {PROJECT_ROOT}")
    print(f"FACTORS_PATH: {FACTORS_PATH}")
    print(f"FACTORS_PATH exists: {FACTORS_PATH.exists()}")
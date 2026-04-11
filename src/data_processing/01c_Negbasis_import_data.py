"""
================================================================================
Script 1: Importazione Dati CDS-Bond Basis
================================================================================
Questo script carica i dati dal file Excel e li prepara per l'analisi.

STRUTTURA DATI:
- Colonna A: date (trade date)
- Colonna B: ISIN
- Colonna C: Ticker (emittente)
- Colonna F: Basis (in bps, es. -60 = -60 bps)
- Colonna G: DV01
- Colonna I: Maturity

COSA FA:
1. Carica il file Excel (formato long: 1 riga per ogni date-ISIN)
2. Mostra informazioni sul dataset (date, ISIN, Ticker, missing values)
3. Calcola statistiche per ISIN e per Ticker
4. Salva in formato Parquet (più veloce per Python)

Author: Alessio Ottaviani
Date: November 2025
Institution: EDHEC Business School - PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETRI - MODIFICA SOLO QUESTI
# ============================================================================

# Nome del file Excel (deve essere in data/raw/)
EXCEL_FILE = "CDS-Bond_Basis.xlsx"

# Nome del foglio Excel
SHEET_NAME = "Sheet1"  # <-- Cambia se il foglio ha altro nome

# Nomi colonne nel file Excel (in ordine: A, B, C, D, E, F, G, H, I)
COL_NAMES = {
    'date': 0,      # Colonna A
    'ISIN': 1,      # Colonna B  
    'Ticker': 2,    # Colonna C
    'Basis': 5,     # Colonna F (index 5 perché 0-based)
    'DV01': 6,      # Colonna G
    'Maturity': 8   # Colonna I
}

# ============================================================================
# PATHS
# ============================================================================

# Trova la cartella del progetto (2 livelli sopra perché siamo in src/data_processing/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Crea cartelle se non esistono
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Path completo del file
excel_path = RAW_DATA_DIR / EXCEL_FILE

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title, char="="):
    """Print formatted section header"""
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")

def print_metric(label, value, unit="", decimals=0):
    """Print formatted metric"""
    if isinstance(value, (int, np.integer)):
        print(f"   {label}: {value:,}{unit}")
    elif isinstance(value, (float, np.floating)):
        if decimals == 0:
            print(f"   {label}: {value:,.0f}{unit}")
        else:
            print(f"   {label}: {value:,.{decimals}f}{unit}")
    else:
        print(f"   {label}: {value}{unit}")

# ============================================================================
# STEP 1: CARICA DATI
# ============================================================================

print_header("CDS-BOND BASIS DATA IMPORT")
print_header("STEP 1: Caricamento dati da Excel")

print(f"📂 File: {excel_path.name}")
print(f"   Location: {RAW_DATA_DIR}")

# Controlla che il file esista
if not excel_path.exists():
    print(f"\n❌ ERRORE: File non trovato!")
    print(f"   Assicurati che il file '{EXCEL_FILE}' sia in: {RAW_DATA_DIR}")
    exit()

# Carica Excel (tutte le colonne, poi selezioniamo quelle che servono)
print("\n⏳ Caricamento in corso...")
print(f"   (questo può richiedere 30-60 secondi per ~1M righe)")

df_raw = pd.read_excel(excel_path, sheet_name=SHEET_NAME)

print(f"\n✅ Caricato!")
print_metric("Righe totali", len(df_raw))
print_metric("Colonne totali", len(df_raw.columns))

# ============================================================================
# STEP 2: SELEZIONA E RINOMINA COLONNE
# ============================================================================

print_header("STEP 2: Selezione colonne necessarie")

# Ottieni i nomi delle colonne dal file
actual_columns = list(df_raw.columns)
print(f"\n📋 Colonne trovate nel file:")
for i, col in enumerate(actual_columns[:10]):  # Mostra prime 10
    print(f"   [{i}] {col}")
if len(actual_columns) > 10:
    print(f"   ... (altre {len(actual_columns)-10} colonne)")

# Seleziona colonne per indice
print(f"\n📊 Selezione colonne necessarie:")

try:
    df = pd.DataFrame({
        'date': df_raw.iloc[:, COL_NAMES['date']],
        'ISIN': df_raw.iloc[:, COL_NAMES['ISIN']],
        'Ticker': df_raw.iloc[:, COL_NAMES['Ticker']],
        'Basis': df_raw.iloc[:, COL_NAMES['Basis']],
        'DV01': df_raw.iloc[:, COL_NAMES['DV01']],
        'Maturity': df_raw.iloc[:, COL_NAMES['Maturity']]
    })
    
    print(f"✅ Colonne selezionate:")
    for col_name, col_idx in COL_NAMES.items():
        print(f"   • {col_name:12s} ← Colonna {chr(65+col_idx)} (index {col_idx})")
    
except IndexError as e:
    print(f"\n❌ ERRORE: Problema con gli indici delle colonne!")
    print(f"   {e}")
    print(f"\n   Verifica che i nomi delle colonne in COL_NAMES siano corretti.")
    exit()

# Converti date
print(f"\n⏳ Conversione date in formato datetime...")
df['date'] = pd.to_datetime(df['date'])
df['Maturity'] = pd.to_datetime(df['Maturity'])

print(f"✅ Date convertite")

# ============================================================================
# STEP 3: INFORMAZIONI SUL DATASET
# ============================================================================

print_header("STEP 3: Informazioni sul dataset")

# Date range
print(f"\n📅 Periodo dati:")
print_metric("Data inizio", df['date'].min().strftime('%Y-%m-%d'))
print_metric("Data fine", df['date'].max().strftime('%Y-%m-%d'))
print_metric("Giorni unici", df['date'].nunique())
print_metric("Righe totali", len(df))

# ISIN count
print(f"\n🔢 ISIN:")
print_metric("ISIN unici", df['ISIN'].nunique())

# Ticker count
print(f"\n🏦 Emittenti (Ticker):")
print_metric("Ticker unici", df['Ticker'].nunique())

# Mostra primi ticker
top_tickers = df['Ticker'].value_counts().head(10)
print(f"\n   Top 10 emittenti per numero osservazioni:")
for ticker, count in top_tickers.items():
    print(f"      {ticker:12s}: {count:,} osservazioni")

# Sample data
print(f"\n📊 Sample data (prime 5 righe):")
print(df.head().to_string(index=False))

# ============================================================================
# STEP 4: ANALISI MISSING VALUES
# ============================================================================

print_header("STEP 4: Analisi dati mancanti (NaN)")

# Missing per colonna
print(f"\n📊 Missing values per colonna:")
for col in ['date', 'ISIN', 'Ticker', 'Basis', 'DV01', 'Maturity']:
    missing = df[col].isna().sum()
    pct = (missing / len(df)) * 100
    print(f"   {col:12s}: {missing:,} ({pct:.2f}%)")

# Missing Basis per Ticker
print(f"\n📊 Copertura Basis per emittente (top 10):")
print(f"   {'Ticker':<12s} {'Obs Totali':>12s} {'Obs Valide':>12s} {'% Copertura':>12s}")
print(f"   {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

ticker_stats = df.groupby('Ticker').agg({
    'Basis': ['count', lambda x: x.notna().sum()]
}).reset_index()
ticker_stats.columns = ['Ticker', 'Total', 'Valid']
ticker_stats['Coverage'] = (ticker_stats['Valid'] / ticker_stats['Total']) * 100
ticker_stats = ticker_stats.sort_values('Total', ascending=False).head(10)

for _, row in ticker_stats.iterrows():
    print(f"   {row['Ticker']:<12s} {row['Total']:>12,} {row['Valid']:>12,} {row['Coverage']:>11.1f}%")

# Missing per ISIN (mostra solo ISIN con >10% missing)
print(f"\n⚠️  ISIN con bassa copertura Basis (<90%):")
isin_coverage = df.groupby('ISIN').agg({
    'Basis': lambda x: x.notna().sum() / len(x) * 100
}).reset_index()
isin_coverage.columns = ['ISIN', 'Coverage']
low_coverage = isin_coverage[isin_coverage['Coverage'] < 90].sort_values('Coverage')

if len(low_coverage) > 0:
    print(f"   Trovati {len(low_coverage)} ISIN con copertura <90%:")
    for _, row in low_coverage.head(10).iterrows():
        print(f"      {row['ISIN']}: {row['Coverage']:.1f}%")
    if len(low_coverage) > 10:
        print(f"      ... (altri {len(low_coverage)-10} ISIN)")
else:
    print(f"   ✅ Nessun ISIN con copertura <90%")

# ============================================================================
# STEP 5: STATISTICHE DESCRITTIVE - BASIS
# ============================================================================

print_header("STEP 5: Statistiche Basis")

# Statistiche aggregate (tutte le osservazioni valide)
basis_valid = df['Basis'].dropna()

print(f"\n📊 Statistiche complessive (tutte le basis valide):")
print_metric("Osservazioni valide", len(basis_valid))
print_metric("Minimo", basis_valid.min(), " bps", 2)
print_metric("Q1 (25%)", basis_valid.quantile(0.25), " bps", 2)
print_metric("Mediana", basis_valid.median(), " bps", 2)
print_metric("Media", basis_valid.mean(), " bps", 2)
print_metric("Q3 (75%)", basis_valid.quantile(0.75), " bps", 2)
print_metric("Massimo", basis_valid.max(), " bps", 2)
print_metric("Std Dev", basis_valid.std(), " bps", 2)

# % negative
pct_negative = (basis_valid < 0).sum() / len(basis_valid) * 100
print_metric("% Negative", pct_negative, "%", 2)

# Statistiche per Ticker (top 10)
print(f"\n📊 Statistiche Basis per emittente (top 10 per numero osservazioni):")
print(f"   {'Ticker':<12s} {'N_Obs':>10s} {'Mean':>10s} {'Median':>10s} {'Min':>10s} {'Max':>10s} {'%_Neg':>8s}")
print(f"   {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

ticker_basis_stats = df.groupby('Ticker')['Basis'].agg([
    ('N_Obs', 'count'),
    ('Mean', 'mean'),
    ('Median', 'median'),
    ('Min', 'min'),
    ('Max', 'max'),
    ('Pct_Neg', lambda x: (x < 0).sum() / len(x) * 100)
]).reset_index()

ticker_basis_stats = ticker_basis_stats.sort_values('N_Obs', ascending=False).head(10)

for _, row in ticker_basis_stats.iterrows():
    print(f"   {row['Ticker']:<12s} {row['N_Obs']:>10,} "
          f"{row['Mean']:>10.2f} {row['Median']:>10.2f} "
          f"{row['Min']:>10.2f} {row['Max']:>10.2f} "
          f"{row['Pct_Neg']:>7.1f}%")

# ============================================================================
# STEP 6: STATISTICHE DESCRITTIVE - DV01
# ============================================================================

print_header("STEP 6: Statistiche DV01")

dv01_valid = df['DV01'].dropna()

print(f"\n📊 Statistiche complessive DV01:")
print_metric("Osservazioni valide", len(dv01_valid))
print_metric("Minimo", dv01_valid.min(), "", 4)
print_metric("Q1 (25%)", dv01_valid.quantile(0.25), "", 4)
print_metric("Mediana", dv01_valid.median(), "", 4)
print_metric("Media", dv01_valid.mean(), "", 4)
print_metric("Q3 (75%)", dv01_valid.quantile(0.75), "", 4)
print_metric("Massimo", dv01_valid.max(), "", 4)

# ============================================================================
# STEP 7: STATISTICHE PER ISIN (summary tables)
# ============================================================================

print_header("STEP 7: Creazione summary tables")

print(f"\n⏳ Calcolo statistiche per ISIN...")

# Summary per ISIN
isin_summary = df.groupby('ISIN').agg({
    'date': ['min', 'max', 'count'],
    'Ticker': 'first',
    'Basis': ['count', 'mean', 'median', 'std', 'min', 'max', lambda x: (x < 0).sum()],
    'DV01': ['mean', 'median'],
    'Maturity': 'first'
}).reset_index()

# Flatten column names
isin_summary.columns = [
    'ISIN', 
    'First_Date', 'Last_Date', 'N_Days',
    'Ticker',
    'N_Obs_Basis', 'Mean_Basis', 'Median_Basis', 'Std_Basis', 'Min_Basis', 'Max_Basis', 'Count_Negative',
    'Mean_DV01', 'Median_DV01',
    'Maturity'
]

# Add % negative
isin_summary['Pct_Negative'] = (isin_summary['Count_Negative'] / isin_summary['N_Obs_Basis']) * 100

# Add Range
isin_summary['Range_Basis'] = isin_summary['Max_Basis'] - isin_summary['Min_Basis']

print(f"✅ Summary per ISIN completata")
print_metric("ISIN processati", len(isin_summary))

# Summary per Ticker
print(f"\n⏳ Calcolo statistiche per Ticker...")

ticker_summary = df.groupby('Ticker').agg({
    'ISIN': 'nunique',
    'date': ['min', 'max'],
    'Basis': ['count', 'mean', 'median', 'std', 'min', 'max', lambda x: (x < 0).sum()],
    'DV01': ['mean', 'median']
}).reset_index()

# Flatten column names
ticker_summary.columns = [
    'Ticker',
    'N_ISIN',
    'First_Date', 'Last_Date',
    'N_Obs_Basis', 'Mean_Basis', 'Median_Basis', 'Std_Basis', 'Min_Basis', 'Max_Basis', 'Count_Negative',
    'Mean_DV01', 'Median_DV01'
]

# Add % negative
ticker_summary['Pct_Negative'] = (ticker_summary['Count_Negative'] / ticker_summary['N_Obs_Basis']) * 100

print(f"✅ Summary per Ticker completata")
print_metric("Ticker processati", len(ticker_summary))

# Mostra sample
print(f"\n📊 Sample ISIN summary (primi 5):")
print(isin_summary.head().to_string(index=False))

print(f"\n📊 Sample Ticker summary (primi 5):")
print(ticker_summary.head().to_string(index=False))

# ============================================================================
# STEP 8: SALVA DATI PROCESSATI
# ============================================================================

print_header("STEP 8: Salvataggio dati processati")

# Salva in formato Parquet (molto più veloce di Excel)
main_data_path = PROCESSED_DATA_DIR / "cds_bond_basis_long.parquet"
isin_summary_path = PROCESSED_DATA_DIR / "summary_by_isin.csv"
ticker_summary_path = PROCESSED_DATA_DIR / "summary_by_ticker.csv"

print("💾 Salvataggio in corso...")

# Main data (long format)
df.to_parquet(main_data_path, index=False)
print(f"✅ Salvato: {main_data_path.name}")
print_metric("   Righe", len(df))
print_metric("   Dimensione file", f"{main_data_path.stat().st_size / 1024 / 1024:.1f} MB")

# Summary tables (CSV per facilità di lettura)
isin_summary.to_csv(isin_summary_path, index=False)
print(f"\n✅ Salvato: {isin_summary_path.name}")
print_metric("   ISIN", len(isin_summary))

ticker_summary.to_csv(ticker_summary_path, index=False)
print(f"\n✅ Salvato: {ticker_summary_path.name}")
print_metric("   Ticker", len(ticker_summary))

# ============================================================================
# STEP 9: RIEPILOGO FINALE
# ============================================================================

print_header("✅ IMPORTAZIONE COMPLETATA CON SUCCESSO!")

print("\n📋 Riepilogo:")
print_metric("Osservazioni totali", len(df))
print_metric("ISIN unici", df['ISIN'].nunique())
print_metric("Ticker unici", df['Ticker'].nunique())
print_metric("Periodo", f"{df['date'].min().strftime('%Y-%m-%d')} → {df['date'].max().strftime('%Y-%m-%d')}")
print_metric("Giorni unici", df['date'].nunique())

print(f"\n📊 Basis statistics:")
print_metric("Mean", basis_valid.mean(), " bps", 2)
print_metric("Median", basis_valid.median(), " bps", 2)
print_metric("% Negative", pct_negative, "%", 1)

print(f"\n📁 File salvati in: {PROCESSED_DATA_DIR}/")
print(f"   • cds_bond_basis_long.parquet (main data)")
print(f"   • summary_by_isin.csv (statistiche per ISIN)")
print(f"   • summary_by_ticker.csv (statistiche per emittente)")

print("\n🎯 Prossimi step:")
print("   1. Esegui script di data cleaning (filtra bond problematici)")
print("   2. Carica lista Main composition (quali CDS sono nel Main per ogni data)")
print("   3. Esegui script di trading strategy simulation")

print("\n" + "=" * 80)
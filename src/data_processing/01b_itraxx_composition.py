"""
================================================================================
Script 2: Importazione iTraxx Main Composition
================================================================================
Questo script carica la composizione dell'indice iTraxx Main per identificare
quali ticker sono eleggibili per la strategia in ogni periodo.

STRUTTURA FILE EXCEL:
- Riga 1: Start Date di ogni series
- Riga 2: Maturity Date di ogni series  
- Riga 3: Nome series (es. S9, S10, S11, ...)
- Riga 4+: Ticker dei constituents (uno per riga)

LOGICA STRATEGIA:
- Per ogni trade_date, identifichiamo quali series sono "vive"
  (start_date <= trade_date <= maturity_date)
- Un ticker è tradabile se appare in ALMENO UNA series viva
- Possiamo entrare solo su bond con maturity < maturity della serie più lunga
  ma > 6 mesi dalla trade_date

OUTPUT:
- CSV con metadata delle series (start, maturity, lista ticker)
- Formato compatto: ~50 righe (una per series)

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
EXCEL_FILE = "CDS_Index_Components.xlsx"

# Nome del foglio Excel
SHEET_NAME = "Sheet1"  # <-- Cambia se necessario

# Righe con metadata (0-indexed)
ROW_START_DATE = 0    # Riga 1 in Excel = index 0
ROW_MATURITY = 1      # Riga 2 in Excel = index 1
ROW_SERIES = 2        # Riga 3 in Excel = index 2
ROW_FIRST_TICKER = 3  # Riga 4 in Excel = index 3 (prima riga con ticker)

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

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

print_header("ITRAXX MAIN COMPOSITION IMPORT")
print_header("STEP 1: Caricamento dati da Excel")

print(f"📂 File: {excel_path.name}")
print(f"   Location: {RAW_DATA_DIR}")

# Controlla che il file esista
if not excel_path.exists():
    print(f"\n❌ ERRORE: File non trovato!")
    print(f"   Assicurati che il file '{EXCEL_FILE}' sia in: {RAW_DATA_DIR}")
    exit()

# Carica Excel senza header (leggiamo tutto come raw data)
print("\n⏳ Caricamento in corso...")

df_raw = pd.read_excel(excel_path, sheet_name=SHEET_NAME, header=None)

print(f"\n✅ Caricato!")
print_metric("Righe totali", len(df_raw))
print_metric("Colonne totali (series)", len(df_raw.columns))

# ============================================================================
# STEP 2: ESTRAI METADATA DELLE SERIES
# ============================================================================

print_header("STEP 2: Estrazione metadata series")

# Le colonne sono le series (S9, S10, S11, ...)
# Le righe contengono: start_date, maturity_date, series_name, ticker1, ticker2, ...

# Estrai i metadata dalle prime 3 righe
start_dates = df_raw.iloc[ROW_START_DATE, :].values
maturity_dates = df_raw.iloc[ROW_MATURITY, :].values
series_names = df_raw.iloc[ROW_SERIES, :].values

print(f"\n📊 Series trovate: {len(series_names)}")
print(f"\n   Prime 5 series:")
for i in range(min(5, len(series_names))):
    print(f"      {series_names[i]}: {start_dates[i]} → {maturity_dates[i]}")

if len(series_names) > 5:
    print(f"      ... (altre {len(series_names)-5} series)")

# ============================================================================
# STEP 3: ESTRAI TICKER PER OGNI SERIES
# ============================================================================

print_header("STEP 3: Estrazione ticker per ogni series")

# Ogni colonna è una series, e contiene i ticker dalla riga 4 in poi
series_data = []

for col_idx in range(len(df_raw.columns)):
    series_name = series_names[col_idx]
    start_date = start_dates[col_idx]
    maturity_date = maturity_dates[col_idx]
    
    # Skip se series name è NaN o vuoto
    if pd.isna(series_name) or str(series_name).strip() == '':
        continue
    
    # Estrai ticker da riga 4 in poi per questa colonna
    tickers_raw = df_raw.iloc[ROW_FIRST_TICKER:, col_idx].values
    
    # Rimuovi NaN e stringhe vuote
    tickers = [
        str(ticker).strip() 
        for ticker in tickers_raw 
        if pd.notna(ticker) and str(ticker).strip() != ''
    ]
    
    # Converti date in datetime se necessario
    if not isinstance(start_date, pd.Timestamp):
        start_date = pd.to_datetime(start_date)
    if not isinstance(maturity_date, pd.Timestamp):
        maturity_date = pd.to_datetime(maturity_date)
    
    series_data.append({
        'series': series_name,
        'start_date': start_date,
        'maturity_date': maturity_date,
        'n_tickers': len(tickers),
        'tickers': tickers  # Lista Python
    })
    
    # Stampa solo summary (non tutti i ticker)
    if col_idx < 3:  # Solo prime 3 series
        print(f"\n   Series {series_name}:")
        print(f"      Start: {start_date.strftime('%Y-%m-%d')}")
        print(f"      Maturity: {maturity_date.strftime('%Y-%m-%d')}")
        print(f"      Tickers: {len(tickers)}")

if len(df_raw.columns) > 3:
    print(f"\n   ... (altre {len(df_raw.columns)-3} series)")


# ============================================================================
# STEP 4: CREA DATAFRAME SERIES
# ============================================================================

print_header("STEP 4: Creazione DataFrame series")

# Crea DataFrame con metadata
df_series = pd.DataFrame(series_data)

# Ordina per start_date
df_series = df_series.sort_values('start_date').reset_index(drop=True)

print(f"\n✅ DataFrame creato:")
print_metric("Series totali", len(df_series))
print_metric("Periodo", f"{df_series['start_date'].min().strftime('%Y-%m-%d')} → {df_series['maturity_date'].max().strftime('%Y-%m-%d')}")

# Mostra DataFrame (senza colonna tickers che è troppo lunga)
print(f"\n📊 Preview (senza colonna tickers):")
preview_df = df_series[['series', 'start_date', 'maturity_date', 'n_tickers']].copy()
print(preview_df.to_string(index=False))

# ============================================================================
# STEP 5: ANALISI COMPOSIZIONE
# ============================================================================

print_header("STEP 5: Analisi composizione")

# Conta ticker unici attraverso tutte le series
all_tickers = set()
for tickers_list in df_series['tickers']:
    all_tickers.update(tickers_list)

print(f"\n📊 Statistiche composizione:")
print_metric("Series totali", len(df_series))
print_metric("Ticker unici (totali)", len(all_tickers))
print_metric("Media ticker per series", df_series['n_tickers'].mean(), "", 1)

# ============================================================================
# STEP 6: SALVA DATI
# ============================================================================

print_header("STEP 6: Salvataggio dati processati")

# Per salvare in CSV, dobbiamo convertire la lista di ticker in stringa
df_series_to_save = df_series.copy()
df_series_to_save['tickers'] = df_series_to_save['tickers'].apply(lambda x: '|'.join(x))

# Salva
series_path = PROCESSED_DATA_DIR / "itraxx_main_series.csv"
df_series_to_save.to_csv(series_path, index=False)

print(f"\n💾 Salvato: {series_path.name}")
print_metric("   Series", len(df_series_to_save))
print_metric("   Dimensione file", f"{series_path.stat().st_size / 1024:.1f} KB")

print(f"\n💡 Formato colonna 'tickers':")
print(f"   I ticker sono separati da '|' (pipe)")
print(f"   Esempio: 'ACFP|AIRFP|ELTLX|...'")
print(f"   In Python usa: tickers.split('|') per ottenere la lista")

# Salva anche una versione "expanded" per reference (opzionale)
# Crea una riga per ogni (series, ticker) combination
expanded_data = []
for _, row in df_series.iterrows():
    for ticker in row['tickers']:
        expanded_data.append({
            'series': row['series'],
            'start_date': row['start_date'],
            'maturity_date': row['maturity_date'],
            'ticker': ticker
        })

df_expanded = pd.DataFrame(expanded_data)
expanded_path = PROCESSED_DATA_DIR / "itraxx_main_expanded.csv"
df_expanded.to_csv(expanded_path, index=False)

print(f"\n💾 Salvato (expanded): {expanded_path.name}")
print_metric("   Righe", len(df_expanded))
print(f"   (Una riga per ogni combinazione series-ticker)")

# ============================================================================
# STEP 7: RIEPILOGO FINALE
# ============================================================================

print_header("✅ IMPORTAZIONE COMPLETATA CON SUCCESSO!")

print("\n📋 Riepilogo:")
print_metric("Series iTraxx Main", len(df_series))
print_metric("Ticker unici totali", len(all_tickers))
print_metric("Periodo coperto", f"{df_series['start_date'].min().strftime('%Y-%m-%d')} → {df_series['maturity_date'].max().strftime('%Y-%m-%d')}")

print(f"\n📁 File salvati in: {PROCESSED_DATA_DIR}/")
print(f"   • itraxx_main_series.csv (metadata compatto)")
print(f"   • itraxx_main_expanded.csv (expanded reference)")

print("\n" + "=" * 80)
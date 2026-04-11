"""
Script 1: Importazione Dati iTraxx - Parametrizzato
====================================================
Questo script carica i dati iTraxx dal file CSV e li prepara per l'analisi.
Funziona per Main, SnrFin, Xover cambiando solo INDEX_NAME.

COSA FA:
1. Carica il file CSV con le basis e Duration
2. Rinomina colonne in formato standardizzato (Ser44_Basis, Ser44_Duration)
3. Estrae maturity dates dai nomi delle colonne
4. Mostra informazioni sul dataset (date, Serie, missing values)
5. Salva in formato Parquet (più veloce per Python)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime

# ============================================================================
# PARAMETRI - MODIFICA SOLO QUESTO
# ============================================================================

INDEX_NAME = "Xover"  # "Main", "SnrFin", "SubFin", "Xover"

# ============================================================================
# FILE NAMES (automatici basati su INDEX_NAME)
# ============================================================================

# Mapping nome indice -> nome file CSV
CSV_FILES = {
    'Main': 'Main_Skew_JPM.csv',
    'SnrFin': 'SnrFin_Skew_JPM.csv',
    'SubFin': 'SubFin_Skew_JPM.csv',
    'Xover': 'Xover_Skew_JPM.csv'
}

if INDEX_NAME not in CSV_FILES:
    raise ValueError(f"INDEX_NAME non valido: {INDEX_NAME}. Usa 'Main', 'SnrFin', 'SubFin', o 'Xover'")

CSV_FILE = CSV_FILES[INDEX_NAME]

# ============================================================================
# PATHS
# ============================================================================

# Trova la cartella del progetto (2 livelli sopra perché siamo in src/data_processing/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / f"itraxx_{INDEX_NAME.lower()}"

# Crea cartelle se non esistono
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Path completo del file
csv_path = RAW_DATA_DIR / CSV_FILE

# ============================================================================
# FUNZIONE HELPER: PARSE MATURITY DATE
# ============================================================================

def parse_maturity_from_column(col_name):
    """
    Estrae la data di maturity dal nome della colonna.
    
    Esempio: 'iTraxx Europe | Main | 5y Ser44 (20-Dec-30) | ...' 
             -> pd.Timestamp('2030-12-20')
    """
    # Cerca pattern tipo (20-Dec-30) o (20-Jun-25)
    match = re.search(r'\((\d{2})-([A-Za-z]{3})-(\d{2})\)', col_name)
    if not match:
        return None
    
    day, month_str, year_short = match.groups()
    
    # Converti mese da testo a numero
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    month = month_map.get(month_str.capitalize())
    if not month:
        return None
    
    # Anno: 09 -> 2009, 30 -> 2030
    year = 2000 + int(year_short)
    
    return pd.Timestamp(year=year, month=month, day=int(day))


def extract_series_number(col_name):
    """
    Estrae il numero di serie dal nome colonna.
    
    Esempio: 'iTraxx Europe | Main | 5y Ser44 (20-Dec-30) | ...' -> 44
    """
    match = re.search(r'Ser(\d+)', col_name)
    if match:
        return int(match.group(1))
    return None

# ============================================================================
# STEP 1: CARICA DATI
# ============================================================================

print("=" * 70)
print(f"STEP 1: Caricamento dati da CSV - iTraxx {INDEX_NAME}")
print("=" * 70)
print(f"📂 File: {csv_path}")

# Controlla che il file esista
if not csv_path.exists():
    print(f"\n❌ ERRORE: File non trovato!")
    print(f"   Assicurati che il file '{CSV_FILE}' sia in: {RAW_DATA_DIR}")
    exit()

# Carica CSV con separatore semicolon
print("⏳ Caricamento in corso...")
df = pd.read_csv(
    csv_path,
    sep=';',
    index_col=0,  # Prima colonna (date) come index
    parse_dates=True,  # Converte le date in formato datetime
    encoding='utf-8-sig'  # Gestisce BOM UTF-8
)

print(f"✅ Caricato! Dimensione: {df.shape[0]} righe × {df.shape[1]} colonne")

# ============================================================================
# STEP 2: RINOMINA COLONNE
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: Rinomina colonne in formato standardizzato")
print("=" * 70)

# Dizionario per mappare vecchi nomi -> nuovi nomi
column_mapping = {}
maturity_dates = {}

# Pattern per identificare Basis vs Duration
basis_keywords = ['Basis to theoretical', 'Basis']
duration_keywords = ['Modified Duration', 'Duration']

for col in df.columns:
    series_num = extract_series_number(col)
    if series_num is None:
        print(f"⚠️  WARNING: Impossibile estrarre serie da: {col}")
        continue
    
    # Determina se è Basis o Duration
    if any(kw in col for kw in basis_keywords):
        new_name = f"Ser{series_num}_Basis"
        column_mapping[col] = new_name
        
        # Estrai maturity date (solo per colonne Basis)
        maturity = parse_maturity_from_column(col)
        if maturity:
            maturity_dates[f"Ser{series_num}"] = maturity
    
    elif any(kw in col for kw in duration_keywords):
        new_name = f"Ser{series_num}_Duration"
        column_mapping[col] = new_name

# Rinomina colonne
df = df.rename(columns=column_mapping)

print(f"✅ Colonne rinominate: {len(column_mapping)}")
print(f"✅ Maturity dates estratte: {len(maturity_dates)}")

# Mostra prime 3 maturity
print(f"\n   Prime 3 serie con maturity:")
for i, (serie, mat_date) in enumerate(sorted(maturity_dates.items(), 
                                               key=lambda x: int(x[0].replace('Ser', '')), 
                                               reverse=True)[:3]):
    print(f"      {serie} → {mat_date.strftime('%Y-%m-%d')}")

# ============================================================================
# STEP 3: INFORMAZIONI SUL DATASET
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: Informazioni sul dataset")
print("=" * 70)

# Date range
print(f"\n📅 Periodo dati:")
print(f"   Da: {df.index.min().strftime('%Y-%m-%d')}")
print(f"   A:  {df.index.max().strftime('%Y-%m-%d')}")
print(f"   Giorni totali: {len(df)}")

# Colonne
print(f"\n📊 Colonne trovate: {len(df.columns)}")
print(f"   Prime 5 colonne: {list(df.columns[:5])}")

# Conta Serie
basis_cols = [col for col in df.columns if 'Basis' in col]
duration_cols = [col for col in df.columns if 'Duration' in col]

print(f"\n🔢 Serie trovate:")
print(f"   Colonne Basis:    {len(basis_cols)}")
print(f"   Colonne Duration: {len(duration_cols)}")

if len(basis_cols) != len(duration_cols):
    print("\n⚠️  ATTENZIONE: Numero diverso di colonne Basis e Duration!")

# Mostra prime Serie
print(f"\n   Prime 3 Serie:")
for col in basis_cols[:3]:
    serie = col.replace('_Basis', '')
    print(f"      - {serie}")

# ============================================================================
# STEP 4: SEPARA BASIS E DURATION
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: Separazione Basis e Duration")
print("=" * 70)

# Crea DataFrame separati
basis_df = df[basis_cols].copy()
duration_df = df[duration_cols].copy()

print(f"✅ Basis DataFrame:    {basis_df.shape}")
print(f"✅ Duration DataFrame: {duration_df.shape}")

# ============================================================================
# STEP 5: ANALISI MISSING VALUES
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: Analisi dati mancanti (NaN)")
print("=" * 70)

# Per ogni Serie, conta quanti valori validi ha
print("\n📈 Copertura dati per Serie (% giorni con dati validi):\n")

for i, col in enumerate(basis_cols[:5]):  # Mostra primi 5
    serie = col.replace('_Basis', '')
    valid_basis = basis_df[col].notna().sum()
    valid_duration = duration_df[col.replace('Basis', 'Duration')].notna().sum()
    pct_basis = (valid_basis / len(df)) * 100
    pct_duration = (valid_duration / len(df)) * 100
    
    print(f"   {serie:10s} → Basis: {pct_basis:5.1f}% | Duration: {pct_duration:5.1f}%")

if len(basis_cols) > 5:
    print(f"   ... (altre {len(basis_cols)-5} serie)")

# ============================================================================
# STEP 6: STATISTICHE DESCRITTIVE
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: Statistiche Basis")
print("=" * 70)

# Statistiche aggregate su tutte le basis
all_basis = basis_df.values.flatten()
all_basis_valid = all_basis[~np.isnan(all_basis)]

print(f"\n📊 Statistiche complessive (tutte le basis valide):")
print(f"   Minimo:  {all_basis_valid.min():8.2f} bps")
print(f"   Q1:      {np.percentile(all_basis_valid, 25):8.2f} bps")
print(f"   Mediana: {np.percentile(all_basis_valid, 50):8.2f} bps")
print(f"   Media:   {all_basis_valid.mean():8.2f} bps")
print(f"   Q3:      {np.percentile(all_basis_valid, 75):8.2f} bps")
print(f"   Massimo: {all_basis_valid.max():8.2f} bps")

# ============================================================================
# STEP 7: SALVA DATI PROCESSATI
# ============================================================================

print("\n" + "=" * 70)
print("STEP 7: Salvataggio dati processati")
print("=" * 70)

# Salva in formato Parquet (molto più veloce di CSV)
basis_path = PROCESSED_DATA_DIR / "itraxx_basis_wide.parquet"
duration_path = PROCESSED_DATA_DIR / "itraxx_duration_wide.parquet"
full_path = PROCESSED_DATA_DIR / "itraxx_full_data.parquet"

# Salva anche maturity dates come CSV
maturity_df = pd.DataFrame(list(maturity_dates.items()), columns=['Serie', 'Maturity'])
maturity_df = maturity_df.sort_values('Serie')
maturity_path = PROCESSED_DATA_DIR / "itraxx_maturity_dates.csv"

print("💾 Salvataggio in corso...")

basis_df.to_parquet(basis_path)
duration_df.to_parquet(duration_path)
df.to_parquet(full_path)
maturity_df.to_csv(maturity_path, index=False)

print(f"✅ Salvato in:")
print(f"   📁 {basis_path.name}")
print(f"   📁 {duration_path.name}")
print(f"   📁 {full_path.name}")
print(f"   📁 {maturity_path.name}")

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================

print("\n" + "=" * 70)
print("✅ IMPORTAZIONE COMPLETATA CON SUCCESSO!")
print("=" * 70)

print("\n📋 Riepilogo:")
print(f"   • Indice: {INDEX_NAME}")
print(f"   • {len(basis_cols)} Serie trovate")
print(f"   • {len(df)} giorni di dati")
print(f"   • Dal {df.index.min().strftime('%Y-%m-%d')} al {df.index.max().strftime('%Y-%m-%d')}")
print(f"   • Dati salvati in: {PROCESSED_DATA_DIR}")

print("\n🎯 Prossimo step:")
print(f"   Esegui lo script di simulazione trading per {INDEX_NAME}!")
print("\n")
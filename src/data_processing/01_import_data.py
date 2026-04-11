"""
Script 1: Importazione Dati BTP-Italia Basis
============================================
Questo script carica i dati dal file Excel e li prepara per l'analisi.

COSA FA:
1. Carica il file Excel con le basis e DV01
2. Mostra informazioni sul dataset (date, ISIN, missing values)
3. Salva in formato Parquet (più veloce per Python)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# PARAMETRI - MODIFICA SOLO QUESTI
# ============================================================================

# Nome del file Excel (deve essere in data/raw/)
EXCEL_FILE = "BTP_Italia_basis.xlsx"

# Nome del foglio Excel
SHEET_NAME = "Sheet1"  # <-- Cambia se il foglio ha altro nome

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
# STEP 1: CARICA DATI
# ============================================================================

print("=" * 70)
print("STEP 1: Caricamento dati da Excel")
print("=" * 70)
print(f"📂 File: {excel_path}")

# Controlla che il file esista
if not excel_path.exists():
    print(f"\n❌ ERRORE: File non trovato!")
    print(f"   Assicurati che il file '{EXCEL_FILE}' sia in: {RAW_DATA_DIR}")
    exit()

# Carica Excel
print("⏳ Caricamento in corso...")
df = pd.read_excel(
    excel_path,
    sheet_name=SHEET_NAME,
    index_col=0,  # Prima colonna (Date) come index
    parse_dates=True  # Converte le date in formato datetime
)

print(f"✅ Caricato! Dimensione: {df.shape[0]} righe × {df.shape[1]} colonne")

# ============================================================================
# STEP 2: INFORMAZIONI SUL DATASET
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: Informazioni sul dataset")
print("=" * 70)

# Date range
print(f"\n📅 Periodo dati:")
print(f"   Da: {df.index.min().strftime('%Y-%m-%d')}")
print(f"   A:  {df.index.max().strftime('%Y-%m-%d')}")
print(f"   Giorni totali: {len(df)}")

# Colonne
print(f"\n📊 Colonne trovate: {len(df.columns)}")
print(f"   Prime 5 colonne: {list(df.columns[:5])}")

# Conta ISIN
basis_cols = [col for col in df.columns if 'Basis' in col]
dv01_cols = [col for col in df.columns if 'DV01' in col]

print(f"\n🔢 ISIN trovati:")
print(f"   Colonne Basis: {len(basis_cols)}")
print(f"   Colonne DV01:  {len(dv01_cols)}")

if len(basis_cols) != len(dv01_cols):
    print("\n⚠️  ATTENZIONE: Numero diverso di colonne Basis e DV01!")

# Mostra primi ISIN
print(f"\n   Primi 3 ISIN:")
for col in basis_cols[:3]:
    isin = col.replace('_Basis', '')
    print(f"      - {isin}")

# ============================================================================
# STEP 3: SEPARA BASIS E DV01
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: Separazione Basis e DV01")
print("=" * 70)

# Crea DataFrame separati
basis_df = df[basis_cols].copy()
dv01_df = df[dv01_cols].copy()

print(f"✅ Basis DataFrame: {basis_df.shape}")
print(f"✅ DV01 DataFrame:  {dv01_df.shape}")

# ============================================================================
# STEP 4: ANALISI MISSING VALUES
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: Analisi dati mancanti (NaN)")
print("=" * 70)

# Per ogni ISIN, conta quanti valori validi ha
print("\n📈 Copertura dati per ISIN (% giorni con dati validi):\n")

for i, col in enumerate(basis_cols[:5]):  # Mostra primi 5
    isin = col.replace('_Basis', '')
    valid_basis = basis_df[col].notna().sum()
    valid_dv01 = dv01_df[col.replace('Basis', 'DV01')].notna().sum()
    pct_basis = (valid_basis / len(df)) * 100
    pct_dv01 = (valid_dv01 / len(df)) * 100
    
    print(f"   {isin[:12]}... → Basis: {pct_basis:5.1f}% | DV01: {pct_dv01:5.1f}%")

if len(basis_cols) > 5:
    print(f"   ... (altri {len(basis_cols)-5} ISIN)")

# ============================================================================
# STEP 5: STATISTICHE DESCRITTIVE
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: Statistiche Basis")
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
# STEP 6: SALVA DATI PROCESSATI
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: Salvataggio dati processati")
print("=" * 70)

# Salva in formato Parquet (molto più veloce di Excel)
basis_path = PROCESSED_DATA_DIR / "basis_wide.parquet"
dv01_path = PROCESSED_DATA_DIR / "dv01_wide.parquet"
full_path = PROCESSED_DATA_DIR / "full_data.parquet"

print("💾 Salvataggio in corso...")

basis_df.to_parquet(basis_path)
dv01_df.to_parquet(dv01_path)
df.to_parquet(full_path)

print(f"✅ Salvato in:")
print(f"   📁 {basis_path.name}")
print(f"   📁 {dv01_path.name}")
print(f"   📁 {full_path.name}")

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================

print("\n" + "=" * 70)
print("✅ IMPORTAZIONE COMPLETATA CON SUCCESSO!")
print("=" * 70)

print("\n📋 Riepilogo:")
print(f"   • {len(basis_cols)} ISIN trovati")
print(f"   • {len(df)} giorni di dati")
print(f"   • Dal {df.index.min().strftime('%Y-%m-%d')} al {df.index.max().strftime('%Y-%m-%d')}")
print(f"   • Dati salvati in: {PROCESSED_DATA_DIR}")

print("\n🎯 Prossimo step:")
print("   Esegui lo script di simulazione trading!")
print("\n")
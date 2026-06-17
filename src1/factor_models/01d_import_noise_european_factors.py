"""
Script 1d: Import Hu, Pan & Wang (2013) Noise Factors - US & EUR
=================================================================
Importa i 2 fattori del modello Hu, Pan & Wang (2013) per US e EUR.

FATTORI IMPORTATI:
US:
1. Mkt-RF: Excess market return (da Fama-French US)
2. HPW_NOISE: Noise index US
   - Col E fino al 29/12/2023 (incluso)
   - Col B dal 30/12/2023 in poi
   - Fattore = First difference

EUR:
1. Mkt-RF: Excess market return (da file all_risk_factors_eur_daily.csv)
2. HPW_NOISE: Noise index Bund
   - Col I (dal 07/08/2007)
   - No switch
   - Fattore = First difference

NOTE:
- HPW Noise è espresso in bps (es. 3.09 = 3.09 basis points)
- NO risk-free adjustment per Noise (è un indice di livello)
- Frequenza: daily, weekly, monthly
- Switch noise US: attenzione a non calcolare diff tra serie vecchia/nuova
- Mkt-RF EUR: riutilizza calcoli da script 01 (no ricalcolo conversioni)

INPUT FILES (in data/external/):
- F-F_Research_Data_Factors_daily.csv (per Mkt-RF US)
- HPW_Noise_Index.xlsx (colonne E, B per US; H, I per EUR)

INPUT FILES (in data/processed/):
- all_risk_factors_eur_daily.csv (per Mkt-RF EUR già convertito)

OUTPUT FILES (in data/processed/):
- noise_factors_us_{freq}.csv
- noise_factors_eur_{freq}.csv
- regression_data_noise_us_{freq}.csv
- regression_data_noise_eur_{freq}.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETRI
# ============================================================================

FACTOR_FREQ = "monthly"  # "daily", "weekly", "monthly"

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================================
# STEP 1: INFORMAZIONI
# ============================================================================

print("=" * 80)
print("IMPORT HU, PAN & WANG (2013) NOISE FACTORS - US & EUR")
print("=" * 80)

print(f"\n📊 Frequenza: {FACTOR_FREQ.upper()}")

print("\n🇺🇸 US FACTORS:")
print("   1. Mkt-RF: Excess market return (Fama-French US)")
print("   2. HPW_NOISE: Noise index US (first difference)")
print("      - Col E fino al 29/12/2023")
print("      - Col B dal 30/12/2023")

print("\n🇪🇺 EUR FACTORS:")
print("   1. Mkt-RF: Excess market return (da all_risk_factors_eur_daily.csv)")
print("   2. HPW_NOISE: Noise index Bund (first difference)")
print("      - Col I dal 07/08/2007")
print()

# ============================================================================
# PART A: US FACTORS
# ============================================================================

print("=" * 80)
print("PART A: US FACTORS")
print("=" * 80)

# ============================================================================
# STEP 2: IMPORTA MKT-RF US DA FAMA-FRENCH
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Importa Mkt-RF US (Fama-French)")
print("=" * 80)

try:
    ff3_us_raw = pd.read_excel(EXTERNAL_DATA_DIR / "Duarte_factors.xlsx", 
                               sheet_name="F-F_Research_Data_Factors", 
                               skiprows=3, header=None)
    ff3_us_raw.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
    ff3_us_raw = ff3_us_raw[ff3_us_raw['Date'].notna()]
    ff3_us_raw = ff3_us_raw[ff3_us_raw['Date'].astype(str).str.strip().str.isdigit()]
    ff3_us_raw['Date'] = pd.to_datetime(ff3_us_raw['Date'].astype(str), format='%Y%m%d')
    ff3_us_raw = ff3_us_raw.set_index('Date')
    
    # Estrai solo Mkt-RF
    mkt_rf_us = ff3_us_raw['Mkt-RF'].astype(float)
    
    print(f"✅ Mkt-RF US importato: {len(mkt_rf_us)} giorni")
    print(f"📅 Periodo: {mkt_rf_us.index.min()} to {mkt_rf_us.index.max()}")
    print(f"📊 Mkt-RF US medio: {mkt_rf_us.mean():.4f}%")
    
except Exception as e:
    print(f"❌ ERRORE import Mkt-RF US: {e}")
    import traceback
    traceback.print_exc()
    exit()

# ============================================================================
# STEP 3: IMPORTA HPW NOISE INDEX US
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Importa HPW Noise Index US")
print("=" * 80)

noise_path = EXTERNAL_DATA_DIR / "HPW_Noise_Index.xlsx"

if not noise_path.exists():
    print(f"❌ ERRORE: File non trovato: {noise_path}")
    exit()

try:
    # Leggi Excel
    noise_raw = pd.read_excel(noise_path)
    
    print(f"📊 Shape iniziale: {noise_raw.shape}")
    
    # Identifica colonne US
    date_col_us = noise_raw.columns[0]  # Col A = Date
    noise_new_col_us = noise_raw.columns[1]  # Col B = Nuovo indice (proxy)
    noise_old_col_us = noise_raw.columns[4]  # Col E = Vecchio indice originale
    
    # Converti date
    noise_raw[date_col_us] = pd.to_datetime(noise_raw[date_col_us], errors='coerce')
    noise_raw_clean = noise_raw.dropna(subset=[date_col_us])
    noise_data_us = noise_raw_clean.set_index(date_col_us)
    
    print(f"✅ Noise US data: {len(noise_data_us)} osservazioni")
    print(f"📅 Periodo: {noise_data_us.index.min()} to {noise_data_us.index.max()}")
    
    # Switch date: 29/12/2023 (incluso)
    switch_date_us = pd.Timestamp('2023-12-29')
    
    # ⭐ CALCOLA RETURNS SEPARATAMENTE PER LE DUE SERIE ⭐
    
    # Serie vecchia (Col E)
    noise_old_us = noise_data_us[noise_old_col_us].astype(float)
    noise_old_diff_us = noise_old_us.diff()
    
    # Serie nuova (Col B)
    noise_new_us = noise_data_us[noise_new_col_us].astype(float)
    noise_new_diff_us = noise_new_us.diff()
    
    # Unisci le due serie di DIFFERENZE (non gli indici!)
    noise_change_us = pd.Series(index=noise_data_us.index, dtype=float)
    
    for date in noise_data_us.index:
        if date <= switch_date_us:
            noise_change_us.loc[date] = noise_old_diff_us.loc[date]
        else:
            noise_change_us.loc[date] = noise_new_diff_us.loc[date]
    
    print(f"\n📊 HPW Noise US Change:")
    print(f"   Mean: {noise_change_us.mean():.4f} bps")
    print(f"   Std: {noise_change_us.std():.4f} bps")
    print(f"   Min: {noise_change_us.min():.2f}, Max: {noise_change_us.max():.2f} bps")
    print(f"   NaN count: {noise_change_us.isna().sum()}")
    print(f"   📊 Pre-switch (Col E): {len(noise_change_us[noise_change_us.index <= switch_date_us])} obs")
    print(f"   📊 Post-switch (Col B): {len(noise_change_us[noise_change_us.index > switch_date_us])} obs")
    
except Exception as e:
    print(f"❌ ERRORE nell'import Noise US: {e}")
    import traceback
    traceback.print_exc()
    exit()

# ============================================================================
# STEP 4: MERGE FATTORI US
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Merge US factors (Mkt-RF + Noise)")
print("=" * 80)

# Crea DataFrame US
mkt_rf_us_df = pd.DataFrame({'Mkt-RF': mkt_rf_us})
noise_us_df = pd.DataFrame({'HPW_NOISE': noise_change_us})

# Merge
factors_us_daily = mkt_rf_us_df.join(noise_us_df, how='inner')
factors_us_daily = factors_us_daily.dropna()

print(f"✅ US Dataset daily: {len(factors_us_daily)} giorni")
print(f"📅 Periodo: {factors_us_daily.index.min()} to {factors_us_daily.index.max()}")
print(f"📊 Colonne: {list(factors_us_daily.columns)}")

# ============================================================================
# PART B: EUR FACTORS
# ============================================================================

print("\n" + "=" * 80)
print("PART B: EUR FACTORS")
print("=" * 80)

# ============================================================================
# STEP 5: IMPORTA MKT-RF EUR (DA FILE GIÀ PROCESSATO)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Importa Mkt-RF EUR (da all_risk_factors_eur)")
print("=" * 80)

# Invece di ricalcolare, usa i fattori EUR già pronti
factors_eur_processed_path = PROCESSED_DATA_DIR / "all_risk_factors_eur_monthly.csv"

if not factors_eur_processed_path.exists():
    print(f"❌ ERRORE: File non trovato: {factors_eur_processed_path}")
    print("⚠️  Devi prima eseguire lo script 01_import_all_risk_factors.py")
    print("⚠️  Continuo solo con fattori US")
    factors_eur_daily = None
else:
    try:
        # Carica fattori EUR già processati
        factors_eur_all = pd.read_csv(factors_eur_processed_path, index_col=0, parse_dates=True)
        
        # Estrai solo Mkt-RF
        mkt_rf_eur = factors_eur_all['Mkt-RF']
        
        print(f"✅ Mkt-RF EUR importato: {len(mkt_rf_eur)} giorni")
        print(f"📅 Periodo: {mkt_rf_eur.index.min()} to {mkt_rf_eur.index.max()}")
        print(f"📊 Mkt-RF EUR medio: {mkt_rf_eur.mean():.4f}%")
        
        # ================================================================
        # STEP 6: IMPORTA HPW NOISE INDEX EUR (BUND)
        # ================================================================

        print("\n" + "=" * 80)
        print("STEP 6: Importa HPW Noise Index EUR (Bund)")
        print("=" * 80)

        # Usa lo stesso file Excel
        # Identifica colonne EUR
        date_col_eur = noise_raw.columns[7]  # Col H = Date EUR
        noise_col_eur = noise_raw.columns[8]  # Col I = Noise Bund
        
        # Filtra righe con date EUR valide
        noise_eur_subset = noise_raw[[date_col_eur, noise_col_eur]].copy()
        noise_eur_subset[date_col_eur] = pd.to_datetime(noise_eur_subset[date_col_eur], errors='coerce')
        noise_eur_subset = noise_eur_subset.dropna(subset=[date_col_eur])
        noise_data_eur = noise_eur_subset.set_index(date_col_eur)
        
        print(f"✅ Noise EUR data: {len(noise_data_eur)} osservazioni")
        print(f"📅 Periodo: {noise_data_eur.index.min()} to {noise_data_eur.index.max()}")
        
        # Calcola first difference (no switch per EUR)
        noise_eur = noise_data_eur[noise_col_eur].astype(float)
        noise_change_eur = noise_eur.diff()
        
        print(f"\n📊 HPW Noise EUR Change:")
        print(f"   Mean: {noise_change_eur.mean():.4f} bps")
        print(f"   Std: {noise_change_eur.std():.4f} bps")
        print(f"   Min: {noise_change_eur.min():.2f}, Max: {noise_change_eur.max():.2f} bps")
        print(f"   NaN count: {noise_change_eur.isna().sum()}")
        
        # ================================================================
        # STEP 7: MERGE FATTORI EUR
        # ================================================================

        print("\n" + "=" * 80)
        print("STEP 7: Merge EUR factors (Mkt-RF + Noise)")
        print("=" * 80)

        # Crea DataFrame EUR
        mkt_rf_eur_df = pd.DataFrame({'Mkt-RF': mkt_rf_eur})
        noise_eur_df = pd.DataFrame({'HPW_NOISE': noise_change_eur})

        # Merge
        factors_eur_daily = mkt_rf_eur_df.join(noise_eur_df, how='inner')
        factors_eur_daily = factors_eur_daily.dropna()

        print(f"✅ EUR Dataset daily: {len(factors_eur_daily)} giorni")
        print(f"📅 Periodo: {factors_eur_daily.index.min()} to {factors_eur_daily.index.max()}")
        print(f"📊 Colonne: {list(factors_eur_daily.columns)}")
        
    except Exception as e:
        print(f"❌ ERRORE nel processing EUR: {e}")
        import traceback
        traceback.print_exc()
        factors_eur_daily = None

# ============================================================================
# STEP 8: RESAMPLE US FACTORS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Resample US factors")
print("=" * 80)

if FACTOR_FREQ == "daily":
    factors_us_final = factors_us_daily.copy()
    print("✅ Frequenza US: daily")
    
elif FACTOR_FREQ == "weekly":
    # Mkt-RF: compound returns
    mkt_rf_us_weekly = factors_us_daily[['Mkt-RF']].resample('W-FRI').apply(
        lambda x: ((1 + x/100).prod() - 1) * 100
    )
    
    # Noise: somma differenze (è un change, non un return)
    noise_us_weekly = factors_us_daily[['HPW_NOISE']].resample('W-FRI').sum()
    
    factors_us_final = mkt_rf_us_weekly.join(noise_us_weekly)
    print("✅ Frequenza US: weekly")
    
elif FACTOR_FREQ == "monthly":
    # Mkt-RF: compound returns
    mkt_rf_us_monthly = factors_us_daily[['Mkt-RF']].resample('M').apply(
        lambda x: ((1 + x/100).prod() - 1) * 100
    )
    
    # Noise: somma differenze (è un change, non un return)
    noise_us_monthly = factors_us_daily[['HPW_NOISE']].resample('M').sum()
    
    factors_us_final = mkt_rf_us_monthly.join(noise_us_monthly)
    print("✅ Frequenza US: monthly")

print(f"📊 US Osservazioni: {len(factors_us_final)}")

# ============================================================================
# STEP 9: RESAMPLE EUR FACTORS
# ============================================================================

if factors_eur_daily is not None:
    print("\n" + "=" * 80)
    print("STEP 9: Resample EUR factors")
    print("=" * 80)

    if FACTOR_FREQ == "daily":
        factors_eur_final = factors_eur_daily.copy()
        print("✅ Frequenza EUR: daily")
        
    elif FACTOR_FREQ == "weekly":
        # Mkt-RF: già weekly dal file Duarte (NON resample)
        # Noise: resample da daily a weekly
        noise_eur_weekly = factors_eur_daily[['HPW_NOISE']].resample('W-FRI').sum()
        
        # mkt_rf_eur è già DataFrame con colonna 'Mkt-RF' alla frequenza corretta
        factors_eur_final = mkt_rf_eur_df.join(noise_eur_weekly)
        print("✅ Frequenza EUR: weekly")
        
    elif FACTOR_FREQ == "monthly":
        # Mkt-RF: già monthly dal file Duarte (NON resample)
        # Noise: resample da daily a monthly
        noise_eur_monthly = factors_eur_daily[['HPW_NOISE']].resample('M').sum()
        
        # mkt_rf_eur è già DataFrame con colonna 'Mkt-RF' alla frequenza corretta
        factors_eur_final = mkt_rf_eur_df.join(noise_eur_monthly)
        print("✅ Frequenza EUR: monthly (Mkt-RF già monthly da Duarte)")

    print(f"📊 EUR Osservazioni: {len(factors_eur_final)}")
else:
    factors_eur_final = None
    print("\n⚠️  EUR factors non disponibili")

# ============================================================================
# STEP 10: SALVA FATTORI
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Salva fattori HPW Noise")
print("=" * 80)

# Salva US factors
factors_us_path = PROCESSED_DATA_DIR / f"noise_factors_us_{FACTOR_FREQ}.csv"
factors_us_final.to_csv(factors_us_path)

print(f"💾 US Salvato: {factors_us_path.name}")
print(f"📊 US Fattori: {list(factors_us_final.columns)}")

# Salva EUR factors (se disponibile)
if factors_eur_final is not None:
    factors_eur_path = PROCESSED_DATA_DIR / f"noise_factors_eur_{FACTOR_FREQ}.csv"
    factors_eur_final.to_csv(factors_eur_path)
    
    print(f"\n💾 EUR Salvato: {factors_eur_path.name}")
    print(f"📊 EUR Fattori: {list(factors_eur_final.columns)}")

# ============================================================================
# STEP 11: MERGE CON TUTTE LE STRATEGIE (LOOP)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 11: Merge con TUTTE le strategie")
print("=" * 80)

STRATEGIES = {
    'btp_italia': 'btp_italia/index_daily.csv',
    'itraxx_main': 'itraxx_main/index_daily.csv',
    'itraxx_snrfin': 'itraxx_snrfin/index_daily.csv',
    'itraxx_subfin': 'itraxx_subfin/index_daily.csv',
    'itraxx_xover': 'itraxx_xover/index_daily.csv',
    'itraxx_combined': 'itraxx_combined/index_daily.csv',
    'CDS_Bond_Basis': 'cds_bond_basis/index_daily.csv'
}

regression_files_created = {'US': [], 'EUR': []}

for strategy_name, strategy_path in STRATEGIES.items():
    
    print(f"\n{'='*80}")
    print(f"Processing: {strategy_name}")
    print(f"{'='*80}")
    
    index_path = RESULTS_DIR / strategy_path
    
    if not index_path.exists():
        print(f"❌ File non trovato: {index_path}")
        print(f"   Skipping {strategy_name}...")
        continue
    
    try:
        # Carica index_daily
        index_df = pd.read_csv(index_path, index_col=0, parse_dates=True)
        
        # Resample strategy
        if FACTOR_FREQ == "weekly":
            strategy = index_df[['index_return']].resample('W-FRI').apply(
                lambda x: ((1 + x/100).prod() - 1) * 100)
        elif FACTOR_FREQ == "monthly":
            strategy = index_df[['index_return']].resample('M').apply(
                lambda x: ((1 + x/100).prod() - 1) * 100)
        else:
            strategy = index_df[['index_return']].copy()

        strategy = strategy.rename(columns={'index_return': 'Strategy_Return'})
        
        # Merge US
        regression_us_data = strategy.join(factors_us_final, how='inner')
        regression_us_data = regression_us_data.dropna()
        
        print(f"✅ US regression data: {len(regression_us_data)} osservazioni")
        
        regression_us_path = PROCESSED_DATA_DIR / f"regression_data_noise_{strategy_name}_us_{FACTOR_FREQ}.csv"
        regression_us_data.to_csv(regression_us_path)
        print(f"💾 Salvato: {regression_us_path.name}")
        
        regression_files_created['US'].append(regression_us_path.name)
        
        # Merge EUR (se disponibile)
        if factors_eur_final is not None and len(factors_eur_final) > 0:
            regression_eur_data = strategy.join(factors_eur_final, how='inner')
            regression_eur_data = regression_eur_data.dropna()
            
            print(f"✅ EUR regression data: {len(regression_eur_data)} osservazioni")
            
            regression_eur_path = PROCESSED_DATA_DIR / f"regression_data_noise_{strategy_name}_eur_{FACTOR_FREQ}.csv"
            regression_eur_data.to_csv(regression_eur_path)
            print(f"💾 Salvato: {regression_eur_path.name}")
            
            regression_files_created['EUR'].append(regression_eur_path.name)
            
    except Exception as e:
        print(f"❌ ERRORE processing {strategy_name}: {e}")
        import traceback
        traceback.print_exc()
        
# ============================================================================
# STEP 12: STATISTICHE
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICHE FINALI")
print("=" * 80)

print("\n🇺🇸 US FACTORS:")
print("\n📊 Statistiche Descrittive:")
print(factors_us_final.describe().round(3))

print("\n📊 Correlazioni tra Fattori:")
print(factors_us_final.corr().round(3))

print("\n📊 Correlazioni con Strategy Return:")
print(regression_us_data.corr()['Strategy_Return'].round(3))

if factors_eur_final is not None and len(factors_eur_final) > 0:
    print("\n" + "=" * 80)
    print("🇪🇺 EUR FACTORS:")
    print("\n📊 Statistiche Descrittive:")
    print(factors_eur_final.describe().round(3))
    
    print("\n📊 Correlazioni tra Fattori:")
    print(factors_eur_final.corr().round(3))
    
    print("\n📊 Correlazioni con Strategy Return:")
    print(regression_eur_data.corr()['Strategy_Return'].round(3))

print("\n" + "=" * 80)
print("✅ COMPLETATO!")
print("=" * 80)

print(f"\n📁 File US generati:")
print(f"   • {factors_us_path.name}")
print(f"   • {regression_us_path.name}")

if factors_eur_final is not None and len(factors_eur_final) > 0:
    print(f"\n📁 File EUR generati:")
    print(f"   • {factors_eur_path.name}")
    print(f"   • {regression_eur_path.name}")

print(f"\n💡 Nota: HPW Noise espresso in bps (basis points)")
print(f"💡 Per cambiare frequenza, modifica FACTOR_FREQ = 'daily', 'weekly' o 'monthly'")
print(f"💡 Mkt-RF EUR riutilizza conversioni da script 01 (no ricalcolo)")
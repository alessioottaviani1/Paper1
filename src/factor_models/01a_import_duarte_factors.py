"""
Script 1: Import All Risk Factors - US & EUROPEAN VERSION - MULTI STRATEGY
===========================================================================
Importa TUTTI i fattori di rischio per le regressioni, sia in versione US che EUR.
Crea dataset di regressione per TUTTE LE STRATEGIE:
1. BTP-Italia
2. iTraxx Main
3. iTraxx SnrFin
4. iTraxx SubFin
5. iTraxx Xover
6. iTraxx Combined

FATTORI IMPORTATI (US):
1-4. Fama-French: Mkt-RF, SMB, HML, UMD (da Kenneth French)
5. RS: S&P Bank Stock Index (da Bloomberg)
6. RI: Industrial Bonds A/BBB (da FRED - composite 50-50)
7. RB: Corporate Bonds A/BBB (composite 50-50)
8-10. R2, R5, R10: Treasury portfolios 2Y/5Y/10Y (da Bloomberg)

FATTORI IMPORTATI (EUR):
1-4. Fama-French EUR: Mkt-RF, SMB, HML, UMD (European markets, convertiti in EUR)
5-10. RS, RI, RB, R2, R5, R10: fattori EUR specifici o fallback US

CONVERSIONI EUR:
- SMB, HML, UMD: LS_t^EUR = LS_t^USD / (1 + r_t^{USD/EUR})
- Mkt-RF: formula specifica con conversione FX

FIX APPLICATO:
- RI_RB.xlsx: skiprows=2 per saltare header rows
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
# DEFINIZIONE STRATEGIE
# ============================================================================

STRATEGIES = {
    'BTP_Italia': 'btp_italia/index_daily.csv',
    'iTraxx_Main': 'itraxx_main/index_daily.csv',
    'iTraxx_SnrFin': 'itraxx_snrfin/index_daily.csv',
    'iTraxx_SubFin': 'itraxx_subfin/index_daily.csv',
    'iTraxx_Xover': 'itraxx_xover/index_daily.csv',
    'iTraxx_Combined': 'itraxx_combined/index_daily.csv',
    'CDS_Bond_Basis': 'cds_bond_basis/index_daily.csv'
}

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

FACTORS_FILE = EXTERNAL_DATA_DIR / "Duarte_factors.xlsx"

if not FACTORS_FILE.exists():
    print(f"❌ ERRORE: File non trovato: {FACTORS_FILE}")
    exit()

print(f"✅ File trovato: {FACTORS_FILE.name}")

# ============================================================================
# STEP 1: INFORMAZIONI
# ============================================================================

print("=" * 80)
print("IMPORT FATTORI DI RISCHIO - US & EUROPEAN VERSION - MULTI STRATEGY")
print("=" * 80)

print("📊 Importazione fattori:")
print("\n🇺🇸 US FACTORS:")
print("   • Fama-French: Mkt-RF, SMB, HML, UMD")
print("   • S&P: RS (Bank Stock Index)")
print("   • Bonds: RI (Industrial A/BBB), RB (Corporate A/BBB)")
print("   • Bloomberg: R2, R5, R10 (Treasury Portfolios)")

print("\n🇪🇺 EUR FACTORS:")
print("   • Fama-French EUR: Mkt-RF, SMB, HML, UMD (converted)")
print("   • RS, RI, RB, R2, R5, R10: EUR specific or US fallback")

print(f"\n📅 Frequenza: {FACTOR_FREQ}")

print(f"\n🎯 Strategie da processare ({len(STRATEGIES)}):")
for i, (name, path) in enumerate(STRATEGIES.items(), 1):
    print(f"   {i}. {name}")

print()

# ============================================================================
# PART A: US FACTORS (UNCHANGED)
# ============================================================================

print("=" * 80)
print("PART A: US FACTORS")
print("=" * 80)

# ============================================================================
# STEP 2: IMPORTA FAMA-FRENCH 3 FACTORS + RF (US)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Importa Fama-French 3 Factors + RF (US)")
print("=" * 80)

try:
    ff3_df = pd.read_excel(FACTORS_FILE, sheet_name="F-F_Research_Data_Factors", 
                           skiprows=3, header=None)
    ff3_df.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
    ff3_df = ff3_df[ff3_df['Date'].notna()]
    ff3_df = ff3_df[ff3_df['Date'].astype(str).str.strip().str.isdigit()]
    ff3_df['Date'] = pd.to_datetime(ff3_df['Date'].astype(str), format='%Y%m%d')
    ff3_df = ff3_df.set_index('Date')
    ff3_df = ff3_df.astype(float)
    
    print(f"✅ FF 3 Factors (US): {len(ff3_df)} giorni")
    
except Exception as e:
    print(f"❌ ERRORE: {e}")
    exit()

# ============================================================================
# STEP 3: IMPORTA MOMENTUM (US)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Importa Momentum (UMD) (US)")
print("=" * 80)

try:
    mom_df = pd.read_excel(FACTORS_FILE, sheet_name="F-F_Momentum_Factor", 
                          skiprows=14, header=None)
    if len(mom_df.columns) == 2:
        mom_df.columns = ['Date', 'Mom']
    else:
        mom_df = mom_df.iloc[:, :2]
        mom_df.columns = ['Date', 'Mom']
    
    mom_df = mom_df[mom_df['Date'].notna()]
    mom_df['Date'] = mom_df['Date'].astype(str).str.strip()
    mom_df['Mom'] = mom_df['Mom'].astype(str).str.strip()
    mom_df['Mom'] = mom_df['Mom'].str.replace(r'[^\d\.\-]', '', regex=True)
    mom_df = mom_df[mom_df['Date'].str.match(r'^\d{8}$', na=False)]
    mom_df['Date'] = pd.to_datetime(mom_df['Date'], format='%Y%m%d')
    mom_df = mom_df.set_index('Date')
    mom_df['Mom'] = pd.to_numeric(mom_df['Mom'], errors='coerce')
    mom_df = mom_df.dropna()
    mom_df = mom_df.rename(columns={'Mom': 'UMD'})
    
    print(f"✅ Momentum (US): {len(mom_df)} giorni")
    
except Exception as e:
    print(f"❌ ERRORE: {e}")
    exit()

# ============================================================================
# STEP 4: IMPORTA RS (S&P BANKS)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Importa RS (S&P Bank Stock Index)")
print("=" * 80)


try:
    rs_raw = pd.read_excel(FACTORS_FILE, sheet_name="S&P_Banks")
    date_col = rs_raw.columns[0]
    value_col = rs_raw.columns[1]
    rs_raw[date_col] = pd.to_datetime(rs_raw[date_col], format='%d/%m/%Y')
    rs_data = rs_raw.set_index(date_col)[value_col]
    rs_return = rs_data.pct_change() * 100
    rs_df = pd.DataFrame({'RS': rs_return})
       
    print(f"✅ S&P Banks: {len(rs_df)} giorni")
        
except Exception as e:
    print(f"❌ ERRORE: {e}")
    rs_df = None

# ============================================================================
# STEP 5: IMPORTA RI e RB (INDUSTRIAL & CORPORATE BONDS) - US
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Importa RI e RB (Industrial & Corporate Bonds) - US")
print("=" * 80)

try:
    ri_rb_raw = pd.read_excel(FACTORS_FILE, sheet_name="RI_RB", skiprows=2)    
    ri_rb_raw.columns = ['Date', 'BBB_Ind_US', 'A_Ind_US', 'BBB_Bond_US', 'A_Bond_US', 
                             'BBB_Ind_EU', 'A_Ind_EU', 'BBB_Bond_EU', 'A_Bond_EU']
    ri_rb_raw['Date'] = pd.to_datetime(ri_rb_raw['Date'], format='%d/%m/%Y', errors='coerce')
    ri_rb_raw = ri_rb_raw.dropna(subset=['Date'])
    ri_rb_data = ri_rb_raw.set_index('Date')
        
    ri_rb_returns = ri_rb_data.pct_change() * 100
        
    ri_gross_us = 0.5 * ri_rb_returns['BBB_Ind_US'] + 0.5 * ri_rb_returns['A_Ind_US']
    ri_df = pd.DataFrame({'RI_gross': ri_gross_us})
        
    rb_gross_us = 0.5 * ri_rb_returns['BBB_Bond_US'] + 0.5 * ri_rb_returns['A_Bond_US']
    rb_df = pd.DataFrame({'RB_gross': rb_gross_us})
        
    print(f"✅ RI US composite: {len(ri_df)} giorni")
    print(f"📊 RI US gross return medio: {ri_df['RI_gross'].mean():.4f}%")
    print(f"✅ RB US composite: {len(rb_df)} giorni")
    print(f"📊 RB US gross return medio: {rb_df['RB_gross'].mean():.4f}%")
        
except Exception as e:
    print(f"❌ ERRORE: {e}")
    import traceback
    traceback.print_exc()
    ri_df = None
    rb_df = None

# ============================================================================
# STEP 6: IMPORTA R2, R5, R10 (TREASURY PORTFOLIOS)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Importa R2, R5, R10 (Treasury Portfolios)")
print("=" * 80)

treasury_path = EXTERNAL_DATA_DIR / "Treasury_Bond_Portfolios.xlsx"

try:
    treasury_raw = pd.read_excel(FACTORS_FILE, sheet_name="Treasury_Bond_Portfolios", 
                                skiprows=2)
    treasury_raw.columns = ['Date', 'R10_Index', 'R5_Index', 'R2_Index']
    treasury_raw['Date'] = pd.to_datetime(treasury_raw['Date'], format='%d/%m/%Y')
    treasury_data = treasury_raw.set_index('Date')
        
    print(f"✅ Treasury portfolios: {len(treasury_data)} giorni")
        
    r10_return = treasury_data['R10_Index'].pct_change() * 100
    r5_return = treasury_data['R5_Index'].pct_change() * 100
    r2_return = treasury_data['R2_Index'].pct_change() * 100
        
    r10_df = pd.DataFrame({'R10': r10_return})
    r5_df = pd.DataFrame({'R5': r5_return})
    r2_df = pd.DataFrame({'R2': r2_return})
        
except Exception as e:
    print(f"❌ ERRORE nella lettura dei Treasury portfolios: {e}")
    r2_df = None
    r5_df = None
    r10_df = None

# ============================================================================
# STEP 7: MERGE TUTTI I FATTORI (DAILY) - US
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Merge fattori (daily) - US")
print("=" * 80)

factors_us_df = ff3_df.join(mom_df, how='inner')

if rs_df is not None:
    factors_us_df = factors_us_df.join(rs_df, how='inner')

if ri_df is not None:
    factors_us_df = factors_us_df.join(ri_df, how='inner')

if rb_df is not None:
    factors_us_df = factors_us_df.join(rb_df, how='inner')

if r2_df is not None:
    factors_us_df = factors_us_df.join(r2_df, how='inner')

if r5_df is not None:
    factors_us_df = factors_us_df.join(r5_df, how='inner')

if r10_df is not None:
    factors_us_df = factors_us_df.join(r10_df, how='inner')

factors_us_df = factors_us_df.dropna()

print(f"✅ Dataset fattori US (daily): {len(factors_us_df)} giorni")
print(f"📊 Colonne: {list(factors_us_df.columns)}")

# ============================================================================
# PART B: EUROPEAN FACTORS
# ============================================================================

print("\n" + "=" * 80)
print("PART B: EUROPEAN FACTORS")
print("=" * 80)

# ============================================================================
# STEP 8: IMPORTA FAMA-FRENCH EUROPE (in USD)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Importa Fama-French Europe (in USD)")
print("=" * 80)

factors_eur_df = None
ff3_eur_df = None
mom_eur_df = None
euribor_data = None

try:
    ff3_eur_df = pd.read_excel(FACTORS_FILE, 
                               sheet_name="Europe_F-F_Research_Data_Factor", 
                               skiprows=7, header=None)
    ff3_eur_df.columns = ['Date', 'Mkt-RF_USD', 'SMB_USD', 'HML_USD', 'RF_USD']
    ff3_eur_df = ff3_eur_df[ff3_eur_df['Date'].notna()]
    ff3_eur_df = ff3_eur_df[ff3_eur_df['Date'].astype(str).str.strip().str.isdigit()]
    ff3_eur_df['Date'] = pd.to_datetime(ff3_eur_df['Date'].astype(str), format='%Y%m%d')
    ff3_eur_df = ff3_eur_df.set_index('Date')
    ff3_eur_df = ff3_eur_df.astype(float)
        
    print(f"✅ FF 3 Factors Europe (USD): {len(ff3_eur_df)} giorni")
        
except Exception as e:
    print(f"❌ ERRORE: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# STEP 9: IMPORTA MOMENTUM EUROPE (in USD)
# ============================================================================

if ff3_eur_df is not None:
    print("\n" + "=" * 80)
    print("STEP 9: Importa Momentum Europe (in USD)")
    print("=" * 80)

    try:
        mom_eur_df = pd.read_excel(FACTORS_FILE, sheet_name="Europe_F-F_Momentum_Factor", 
                                   skiprows=7, header=None)
        if len(mom_eur_df.columns) == 2:
            mom_eur_df.columns = ['Date', 'Mom_USD']
        else:
            mom_eur_df = mom_eur_df.iloc[:, :2]
            mom_eur_df.columns = ['Date', 'Mom_USD']
        
        mom_eur_df = mom_eur_df[mom_eur_df['Date'].notna()]
        mom_eur_df['Date'] = mom_eur_df['Date'].astype(str).str.strip()
        mom_eur_df['Mom_USD'] = mom_eur_df['Mom_USD'].astype(str).str.strip()
        mom_eur_df['Mom_USD'] = mom_eur_df['Mom_USD'].str.replace(r'[^\d\.\-]', '', regex=True)
        mom_eur_df = mom_eur_df[mom_eur_df['Date'].str.match(r'^\d{8}$', na=False)]
        mom_eur_df['Date'] = pd.to_datetime(mom_eur_df['Date'], format='%Y%m%d')
        mom_eur_df = mom_eur_df.set_index('Date')
        mom_eur_df['Mom_USD'] = pd.to_numeric(mom_eur_df['Mom_USD'], errors='coerce')
        mom_eur_df = mom_eur_df.dropna()
        mom_eur_df = mom_eur_df.rename(columns={'Mom_USD': 'UMD_USD'})
        
        print(f"✅ Momentum Europe (USD): {len(mom_eur_df)} giorni")
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        mom_eur_df = None

# ============================================================================
# STEP 10: IMPORTA EURIBOR 1M + FX RATE
# ============================================================================

if ff3_eur_df is not None and mom_eur_df is not None:
    print("\n" + "=" * 80)
    print("STEP 10: Importa Euribor 1M + FX Rate (USD/EUR)")
    print("=" * 80)

    try:
        euribor_raw = pd.read_excel(FACTORS_FILE, sheet_name="Euribor1m", skiprows=7)
        euribor_raw.columns = ['Date', 'Euribor1m', 'Col_C', 'FX_USD_EUR']
        euribor_raw['Date'] = pd.to_datetime(euribor_raw['Date'])
        euribor_data = euribor_raw.set_index('Date')[['Euribor1m', 'FX_USD_EUR']]
        
        euribor_data['RF_EUR'] = euribor_data['Euribor1m'] / 360
        euribor_data['r_FX'] = euribor_data['FX_USD_EUR'].pct_change() * 100
        
        print(f"✅ Euribor + FX: {len(euribor_data)} giorni")
        print(f"📊 RF_EUR medio: {euribor_data['RF_EUR'].mean():.4f}%")
        print(f"📊 FX return medio: {euribor_data['r_FX'].mean():.4f}%")
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        euribor_data = None

# ============================================================================
# STEP 10B: IMPORTA EUROPEAN TREASURY PORTFOLIOS (R2, R5, R10)
# ============================================================================

r2_eur_df = None
r5_eur_df = None
r10_eur_df = None

if euribor_data is not None:
    print("\n" + "=" * 80)
    print("STEP 10B: Importa European Treasury Portfolios (German Bonds)")
    print("=" * 80)

    try:
        treasury_eur_raw = pd.read_excel(FACTORS_FILE, sheet_name="Europe_Treasury_Bond_Portfolios", 
                                        skiprows=6)
        treasury_eur_raw.columns = ['Date', 'R10_EUR_Index', 'R5_EUR_Index', 'R2_EUR_Index']
        treasury_eur_raw['Date'] = pd.to_datetime(treasury_eur_raw['Date'])
        treasury_eur_data = treasury_eur_raw.set_index('Date')
        
        print(f"✅ European Treasury portfolios: {len(treasury_eur_data)} giorni")
        print(f"📅 Date: {treasury_eur_data.index.min()} to {treasury_eur_data.index.max()}")
        
        r10_eur_return = treasury_eur_data['R10_EUR_Index'].pct_change() * 100
        r5_eur_return = treasury_eur_data['R5_EUR_Index'].pct_change() * 100
        r2_eur_return = treasury_eur_data['R2_EUR_Index'].pct_change() * 100
        
        r10_eur_df = pd.DataFrame({'R10_gross': r10_eur_return})
        r5_eur_df = pd.DataFrame({'R5_gross': r5_eur_return})
        r2_eur_df = pd.DataFrame({'R2_gross': r2_eur_return})
        
        print(f"📊 R10 EUR gross return medio: {r10_eur_df['R10_gross'].mean():.4f}%")
        print(f"📊 R5 EUR gross return medio: {r5_eur_df['R5_gross'].mean():.4f}%")
        print(f"📊 R2 EUR gross return medio: {r2_eur_df['R2_gross'].mean():.4f}%")
        
    except Exception as e:
        print(f"❌ ERRORE nella lettura dei European Treasury portfolios: {e}")
        import traceback
        traceback.print_exc()
        r2_eur_df = None
        r5_eur_df = None
        r10_eur_df = None

# ============================================================================
# STEP 10C: IMPORTA EUROPEAN BANK STOCK INDEX (RS)
# ============================================================================

rs_eur_df = None

if euribor_data is not None:
    print("\n" + "=" * 80)
    print("STEP 10C: Importa European Bank Stock Index (RS)")
    print("=" * 80)

    try:
        rs_raw = pd.read_excel(FACTORS_FILE, sheet_name="S&P_Banks")
        
        if len(rs_raw.columns) >= 5:
            date_col_eur = rs_raw.columns[3]
            value_col_eur = rs_raw.columns[4]
            
            rs_eur_data = rs_raw[[date_col_eur, value_col_eur]].copy()
            rs_eur_data.columns = ['Date', 'Index']
            rs_eur_data['Date'] = pd.to_datetime(rs_eur_data['Date'], format='%d/%m/%Y', errors='coerce')
            rs_eur_data = rs_eur_data.dropna(subset=['Date'])
            rs_eur_data = rs_eur_data.set_index('Date')
            
            rs_eur_return = rs_eur_data['Index'].pct_change() * 100
            rs_eur_df = pd.DataFrame({'RS_gross': rs_eur_return})
            
            print(f"✅ European Banks: {len(rs_eur_df)} giorni")
            print(f"📊 RS EUR gross return medio: {rs_eur_df['RS_gross'].mean():.4f}%")
        else:
            print(f"⚠️ File non contiene colonne EUR. Continuo senza RS europeo.")
            
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        rs_eur_df = None

# ============================================================================
# STEP 10D: IMPORTA RI e RB EUR (INDUSTRIAL & CORPORATE BONDS)
# ============================================================================

ri_eur_df = None
rb_eur_df = None

if euribor_data is not None:
    print("\n" + "=" * 80)
    print("STEP 10D: Importa RI e RB EUR (Industrial & Corporate Bonds)")
    print("=" * 80)

    try:
        ri_rb_raw = pd.read_excel(FACTORS_FILE, sheet_name="RI_RB", skiprows=2)
        
        ri_rb_raw.columns = ['Date', 'BBB_Ind_US', 'A_Ind_US', 'BBB_Bond_US', 'A_Bond_US', 
                             'BBB_Ind_EU', 'A_Ind_EU', 'BBB_Bond_EU', 'A_Bond_EU']
        ri_rb_raw['Date'] = pd.to_datetime(ri_rb_raw['Date'], format='%d/%m/%Y', errors='coerce')
        ri_rb_raw = ri_rb_raw.dropna(subset=['Date'])
        ri_rb_data = ri_rb_raw.set_index('Date')
        
        ri_rb_returns = ri_rb_data.pct_change() * 100
        
        ri_gross_eur = 0.5 * ri_rb_returns['BBB_Ind_EU'] + 0.5 * ri_rb_returns['A_Ind_EU']
        ri_eur_df = pd.DataFrame({'RI_gross': ri_gross_eur})
        
        rb_gross_eur = 0.5 * ri_rb_returns['BBB_Bond_EU'] + 0.5 * ri_rb_returns['A_Bond_EU']
        rb_eur_df = pd.DataFrame({'RB_gross': rb_gross_eur})
        
        print(f"✅ RI EUR composite: {len(ri_eur_df)} giorni")
        print(f"📊 RI EUR gross return medio: {ri_eur_df['RI_gross'].mean():.4f}%")
        print(f"✅ RB EUR composite: {len(rb_eur_df)} giorni")
        print(f"📊 RB EUR gross return medio: {rb_eur_df['RB_gross'].mean():.4f}%")
        
    except Exception as e:
        print(f"❌ ERRORE: {e}")
        import traceback
        traceback.print_exc()
        ri_eur_df = None
        rb_eur_df = None

# ============================================================================
# STEP 11: CONVERTI FATTORI IN EUR
# ============================================================================

factors_eur_usd_global = None

if ff3_eur_df is not None and mom_eur_df is not None and euribor_data is not None:
    print("\n" + "=" * 80)
    print("STEP 11: Converti fattori da USD a EUR")
    print("=" * 80)

    try:
        factors_eur_usd = ff3_eur_df.join(mom_eur_df, how='inner')
        factors_eur_usd = factors_eur_usd.join(euribor_data, how='inner')
        factors_eur_usd = factors_eur_usd.dropna()
        
        factors_eur_usd_global = factors_eur_usd.copy()
        
        print(f"✅ Dataset merged (USD): {len(factors_eur_usd)} giorni")
        
        factors_eur_usd['SMB_EUR'] = factors_eur_usd['SMB_USD'] / (1 + factors_eur_usd['r_FX']/100)
        factors_eur_usd['HML_EUR'] = factors_eur_usd['HML_USD'] / (1 + factors_eur_usd['r_FX']/100)
        factors_eur_usd['UMD_EUR'] = factors_eur_usd['UMD_USD'] / (1 + factors_eur_usd['r_FX']/100)
        
        print("✅ SMB, HML, UMD convertiti in EUR")
        
        factors_eur_usd['Mkt-RF_EUR'] = (
            (1 / (1 + factors_eur_usd['r_FX']/100)) * 
            (1 + factors_eur_usd['Mkt-RF_USD']/100 + factors_eur_usd['RF_USD']/100) - 
            1 - 
            factors_eur_usd['RF_EUR']/100
        ) * 100
        
        print("✅ Mkt-RF convertito in EUR")
        
        factors_eur_converted = factors_eur_usd[['Mkt-RF_EUR', 'SMB_EUR', 'HML_EUR', 'UMD_EUR']].copy()
        factors_eur_converted.columns = ['Mkt-RF', 'SMB', 'HML', 'UMD']
        
        factors_eur_converted = factors_eur_converted.join(factors_eur_usd[['RF_EUR']], how='inner')
        
        if rs_eur_df is not None:
            factors_eur_converted = factors_eur_converted.join(rs_eur_df, how='inner')
            factors_eur_converted['RS'] = factors_eur_converted['RS_gross'] - factors_eur_converted['RF_EUR']
            print(f"✅ RS EUR aggiunto al dataset")
        elif rs_df is not None:
            factors_eur_converted = factors_eur_converted.join(rs_df, how='inner')
            print(f"⚠️  RS EUR non disponibile, uso RS US")
        
        if ri_eur_df is not None:
            factors_eur_converted = factors_eur_converted.join(ri_eur_df, how='inner')
            factors_eur_converted['RI'] = factors_eur_converted['RI_gross'] - factors_eur_converted['RF_EUR']
            print(f"✅ RI EUR aggiunto al dataset")
        elif ri_df is not None:
            factors_eur_converted = factors_eur_converted.join(ri_df, how='inner')
            print(f"⚠️  RI EUR non disponibile, uso RI US")
        
        if rb_eur_df is not None:
            factors_eur_converted = factors_eur_converted.join(rb_eur_df, how='inner')
            factors_eur_converted['RB'] = factors_eur_converted['RB_gross'] - factors_eur_converted['RF_EUR']
            print(f"✅ RB EUR aggiunto al dataset")
        elif rb_df is not None:
            factors_eur_converted = factors_eur_converted.join(rb_df, how='inner')
            print(f"⚠️  RB EUR non disponibile, uso RB US")
        
        if r2_eur_df is not None:
            factors_eur_converted = factors_eur_converted.join(r2_eur_df, how='inner')
            factors_eur_converted['R2'] = factors_eur_converted['R2_gross'] - factors_eur_converted['RF_EUR']
            print(f"✅ R2 EUR aggiunto al dataset")
        elif r2_df is not None:
            factors_eur_converted = factors_eur_converted.join(r2_df, how='inner')
            print(f"⚠️  R2 EUR non disponibile, uso R2 US")
        
        if r5_eur_df is not None:
            factors_eur_converted = factors_eur_converted.join(r5_eur_df, how='inner')
            factors_eur_converted['R5'] = factors_eur_converted['R5_gross'] - factors_eur_converted['RF_EUR']
            print(f"✅ R5 EUR aggiunto al dataset")
        elif r5_df is not None:
            factors_eur_converted = factors_eur_converted.join(r5_df, how='inner')
            print(f"⚠️  R5 EUR non disponibile, uso R5 US")
        
        if r10_eur_df is not None:
            factors_eur_converted = factors_eur_converted.join(r10_eur_df, how='inner')
            factors_eur_converted['R10'] = factors_eur_converted['R10_gross'] - factors_eur_converted['RF_EUR']
            print(f"✅ R10 EUR aggiunto al dataset")
        elif r10_df is not None:
            factors_eur_converted = factors_eur_converted.join(r10_df, how='inner')
            print(f"⚠️  R10 EUR non disponibile, uso R10 US")
        
        final_cols = ['Mkt-RF', 'SMB', 'HML', 'UMD']
        if 'RS' in factors_eur_converted.columns:
            final_cols.append('RS')
        if 'RI' in factors_eur_converted.columns:
            final_cols.append('RI')
        if 'RB' in factors_eur_converted.columns:
            final_cols.append('RB')
        if 'R2' in factors_eur_converted.columns:
            final_cols.append('R2')
        if 'R5' in factors_eur_converted.columns:
            final_cols.append('R5')
        if 'R10' in factors_eur_converted.columns:
            final_cols.append('R10')
        
        factors_eur_converted = factors_eur_converted[final_cols]
        factors_eur_converted = factors_eur_converted.dropna()
        
        print(f"✅ Dataset fattori EUR (daily): {len(factors_eur_converted)} giorni")
        print(f"📊 Colonne: {list(factors_eur_converted.columns)}")
        
        print("\n📊 Statistiche conversione:")
        print(f"   Mkt-RF: USD={factors_eur_usd['Mkt-RF_USD'].mean():.3f}% → EUR={factors_eur_converted['Mkt-RF'].mean():.3f}%")
        print(f"   SMB:    USD={factors_eur_usd['SMB_USD'].mean():.3f}% → EUR={factors_eur_converted['SMB'].mean():.3f}%")
        print(f"   HML:    USD={factors_eur_usd['HML_USD'].mean():.3f}% → EUR={factors_eur_converted['HML'].mean():.3f}%")
        print(f"   UMD:    USD={factors_eur_usd['UMD_USD'].mean():.3f}% → EUR={factors_eur_converted['UMD'].mean():.3f}%")
        
        factors_eur_df = factors_eur_converted
        
    except Exception as e:
        print(f"❌ ERRORE nella conversione: {e}")
        import traceback
        traceback.print_exc()
        factors_eur_df = None

# ============================================================================
# STEP 12: RESAMPLE US FACTORS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 12: Resample US Factors")
print("=" * 80)

if FACTOR_FREQ == "daily":
    factors_us_final = factors_us_df.copy()
    
    factors_to_adjust = []
    
    if 'RS' in factors_us_final.columns:
        factors_us_final['RS'] = factors_us_final['RS'] - factors_us_final['RF']
        factors_to_adjust.append('RS')
    
    if 'RI_gross' in factors_us_final.columns:
        factors_us_final['RI'] = factors_us_final['RI_gross'] - factors_us_final['RF']
        factors_to_adjust.append('RI')
    
    if 'RB_gross' in factors_us_final.columns:
        factors_us_final['RB'] = factors_us_final['RB_gross'] - factors_us_final['RF']
        factors_to_adjust.append('RB')
    
    if 'R2' in factors_us_final.columns:
        factors_us_final['R2'] = factors_us_final['R2'] - factors_us_final['RF']
        factors_to_adjust.append('R2')
    
    if 'R5' in factors_us_final.columns:
        factors_us_final['R5'] = factors_us_final['R5'] - factors_us_final['RF']
        factors_to_adjust.append('R5')
    
    if 'R10' in factors_us_final.columns:
        factors_us_final['R10'] = factors_us_final['R10'] - factors_us_final['RF']
        factors_to_adjust.append('R10')
    
    cols_to_drop = ['RF']
    if 'RI_gross' in factors_us_final.columns:
        cols_to_drop.append('RI_gross')
    if 'RB_gross' in factors_us_final.columns:
        cols_to_drop.append('RB_gross')
    
    factors_us_final = factors_us_final.drop(columns=cols_to_drop)
    
    print("✅ Frequenza: daily")
    print(f"   Excess returns calcolati per: {', '.join(factors_to_adjust)}")

elif FACTOR_FREQ in ["weekly", "monthly"]:
    
    freq_label = 'W-FRI' if FACTOR_FREQ == "weekly" else 'M'
    
    print(f"🔧 Applicando resample per frequenza {FACTOR_FREQ}...")
    
    mkt_rf_compound = factors_us_df[['Mkt-RF']].resample(freq_label).apply(
        lambda x: ((1 + x/100).prod() - 1) * 100
    )
    
    ff_factors = factors_us_df[['SMB', 'HML', 'UMD']].resample(freq_label).apply(
        lambda x: ((1 + x/100).prod() - 1) * 100
    )
    
    print(f"✅ Mkt-RF, SMB, HML, UMD: compound diretto (metodo Fama-French)")
    
    rf_compound = factors_us_df[['RF']].resample(freq_label).apply(
        lambda x: ((1 + x/100).prod() - 1) * 100
    )
    
    gross_factors = []
    
    if 'RS' in factors_us_df.columns:
        rs_compound = factors_us_df[['RS']].resample(freq_label).apply(
            lambda x: ((1 + x/100).prod() - 1) * 100
        )
        rs_excess = ((1 + rs_compound['RS']/100) / (1 + rf_compound['RF']/100) - 1) * 100
        gross_factors.append(pd.DataFrame({'RS': rs_excess}))
    
    if 'RI_gross' in factors_us_df.columns:
        ri_compound = factors_us_df[['RI_gross']].resample(freq_label).apply(
            lambda x: ((1 + x/100).prod() - 1) * 100
        )
        ri_excess = ((1 + ri_compound['RI_gross']/100) / (1 + rf_compound['RF']/100) - 1) * 100
        gross_factors.append(pd.DataFrame({'RI': ri_excess}))
    
    if 'RB_gross' in factors_us_df.columns:
        rb_compound = factors_us_df[['RB_gross']].resample(freq_label).apply(
            lambda x: ((1 + x/100).prod() - 1) * 100
        )
        rb_excess = ((1 + rb_compound['RB_gross']/100) / (1 + rf_compound['RF']/100) - 1) * 100
        gross_factors.append(pd.DataFrame({'RB': rb_excess}))
    
    if 'R2' in factors_us_df.columns:
        r2_compound = factors_us_df[['R2']].resample(freq_label).apply(
            lambda x: ((1 + x/100).prod() - 1) * 100
        )
        r2_excess = ((1 + r2_compound['R2']/100) / (1 + rf_compound['RF']/100) - 1) * 100
        gross_factors.append(pd.DataFrame({'R2': r2_excess}))
    
    if 'R5' in factors_us_df.columns:
        r5_compound = factors_us_df[['R5']].resample(freq_label).apply(
            lambda x: ((1 + x/100).prod() - 1) * 100
        )
        r5_excess = ((1 + r5_compound['R5']/100) / (1 + rf_compound['RF']/100) - 1) * 100
        gross_factors.append(pd.DataFrame({'R5': r5_excess}))
    
    if 'R10' in factors_us_df.columns:
        r10_compound = factors_us_df[['R10']].resample(freq_label).apply(
            lambda x: ((1 + x/100).prod() - 1) * 100
        )
        r10_excess = ((1 + r10_compound['R10']/100) / (1 + rf_compound['RF']/100) - 1) * 100
        gross_factors.append(pd.DataFrame({'R10': r10_excess}))
    
    print(f"✅ RS, RI, RB, R2, R5, R10: compound + excess via ratio")
    
    factors_us_final = pd.concat([mkt_rf_compound, ff_factors] + gross_factors, axis=1)
    
    print(f"✅ Frequenza: {FACTOR_FREQ}")
    print(f"✅ Metodo Fama-French applicato correttamente")

print(f"📊 US factors finale: {len(factors_us_final)} osservazioni")

# ============================================================================
# STEP 13: RESAMPLE EUR FACTORS
# ============================================================================

if factors_eur_df is not None:
    print("\n" + "=" * 80)
    print("STEP 13: Resample EUR Factors")
    print("=" * 80)

    if FACTOR_FREQ == "daily":
        factors_eur_final = factors_eur_df.copy()
        
        print("✅ Frequenza: daily")
        print("   Fattori EUR già in excess return format")

    elif FACTOR_FREQ in ["weekly", "monthly"]:
        
        freq_label = 'W-FRI' if FACTOR_FREQ == "weekly" else 'M'
        
        print(f"🔧 METODO CORRETTO: Compound USD factors, POI converti in EUR")
        print(f"   (NON compound di fattori già convertiti)")
        
        if factors_eur_usd_global is None:
            print("❌ ERRORE: Dati USD non disponibili per resample")
            factors_eur_final = None
        else:
            factors_eur_usd_temp = factors_eur_usd_global.copy()
            
            ff_usd_compound = factors_eur_usd_temp[['SMB_USD', 'HML_USD', 'UMD_USD']].resample(freq_label).apply(
                lambda x: ((1 + x/100).prod() - 1) * 100
            )
            
            mkt_usd_compound = factors_eur_usd_temp[['Mkt-RF_USD']].resample(freq_label).apply(
                lambda x: ((1 + x/100).prod() - 1) * 100
            )
            
            rf_usd_compound = factors_eur_usd_temp[['RF_USD']].resample(freq_label).apply(
                lambda x: ((1 + x/100).prod() - 1) * 100
            )
            
            rf_eur_compound = factors_eur_usd_temp[['RF_EUR']].resample(freq_label).apply(
                lambda x: ((1 + x/100).prod() - 1) * 100
            )
            
            fx_compound = factors_eur_usd_temp[['r_FX']].resample(freq_label).apply(
                lambda x: ((1 + x/100).prod() - 1) * 100
            )
            
            compound_data = pd.concat([
                mkt_usd_compound, 
                ff_usd_compound, 
                rf_usd_compound, 
                rf_eur_compound, 
                fx_compound
            ], axis=1)
            
            compound_data['SMB_EUR'] = compound_data['SMB_USD'] / (1 + compound_data['r_FX']/100)
            compound_data['HML_EUR'] = compound_data['HML_USD'] / (1 + compound_data['r_FX']/100)
            compound_data['UMD_EUR'] = compound_data['UMD_USD'] / (1 + compound_data['r_FX']/100)
            
            compound_data['Mkt-RF_EUR'] = (
                (1 / (1 + compound_data['r_FX']/100)) * 
                (1 + compound_data['Mkt-RF_USD']/100 + compound_data['RF_USD']/100) - 
                1 - 
                compound_data['RF_EUR']/100
            ) * 100
            
            ff_eur_factors = compound_data[['Mkt-RF_EUR', 'SMB_EUR', 'HML_EUR', 'UMD_EUR']].copy()
            ff_eur_factors.columns = ['Mkt-RF', 'SMB', 'HML', 'UMD']
            
            print(f"✅ Fattori FF convertiti a livello {FACTOR_FREQ}")
            
            gross_eur_factors = []
            
            rf_eur_compound = factors_eur_usd_temp[['RF_EUR']].resample(freq_label).apply(
                lambda x: ((1 + x/100).prod() - 1) * 100
            )
            
            if rs_eur_df is not None:
                rs_with_rf = rs_eur_df.join(factors_eur_usd_temp[['RF_EUR']], how='inner')
                
                rs_gross_compound = rs_with_rf[['RS_gross']].resample(freq_label).apply(
                    lambda x: ((1 + x/100).prod() - 1) * 100
                )
                rs_excess = ((1 + rs_gross_compound['RS_gross']/100) / (1 + rf_eur_compound['RF_EUR']/100) - 1) * 100
                gross_eur_factors.append(pd.DataFrame({'RS': rs_excess}))
                print(f"✅ RS EUR: compound gross + excess via ratio")
            elif 'RS' in factors_eur_df.columns:
                rs_compound = factors_eur_df[['RS']].resample(freq_label).apply(
                    lambda x: ((1 + x/100).prod() - 1) * 100
                )
                gross_eur_factors.append(rs_compound)
            
            if ri_eur_df is not None:
                ri_with_rf = ri_eur_df.join(factors_eur_usd_temp[['RF_EUR']], how='inner')
                
                ri_gross_compound = ri_with_rf[['RI_gross']].resample(freq_label).apply(
                    lambda x: ((1 + x/100).prod() - 1) * 100
                )
                ri_excess = ((1 + ri_gross_compound['RI_gross']/100) / (1 + rf_eur_compound['RF_EUR']/100) - 1) * 100
                gross_eur_factors.append(pd.DataFrame({'RI': ri_excess}))
                print(f"✅ RI EUR: compound gross + excess via ratio")
            elif 'RI' in factors_eur_df.columns:
                ri_compound = factors_eur_df[['RI']].resample(freq_label).apply(
                    lambda x: ((1 + x/100).prod() - 1) * 100
                )
                gross_eur_factors.append(ri_compound)
            
            if rb_eur_df is not None:
                rb_with_rf = rb_eur_df.join(factors_eur_usd_temp[['RF_EUR']], how='inner')
                
                rb_gross_compound = rb_with_rf[['RB_gross']].resample(freq_label).apply(
                    lambda x: ((1 + x/100).prod() - 1) * 100
                )
                rb_excess = ((1 + rb_gross_compound['RB_gross']/100) / (1 + rf_eur_compound['RF_EUR']/100) - 1) * 100
                gross_eur_factors.append(pd.DataFrame({'RB': rb_excess}))
                print(f"✅ RB EUR: compound gross + excess via ratio")
            elif 'RB' in factors_eur_df.columns:
                rb_compound = factors_eur_df[['RB']].resample(freq_label).apply(
                    lambda x: ((1 + x/100).prod() - 1) * 100
                )
                gross_eur_factors.append(rb_compound)
            
            if r2_eur_df is not None:
                r2_with_rf = r2_eur_df.join(factors_eur_usd_temp[['RF_EUR']], how='inner')
                
                r2_gross_compound = r2_with_rf[['R2_gross']].resample(freq_label).apply(
                    lambda x: ((1 + x/100).prod() - 1) * 100
                )
                r2_excess = ((1 + r2_gross_compound['R2_gross']/100) / (1 + rf_eur_compound['RF_EUR']/100) - 1) * 100
                gross_eur_factors.append(pd.DataFrame({'R2': r2_excess}))
                print(f"✅ R2 EUR: compound gross + excess via ratio")
            elif 'R2' in factors_eur_df.columns:
                r2_compound = factors_eur_df[['R2']].resample(freq_label).apply(
                    lambda x: ((1 + x/100).prod() - 1) * 100
                )
                gross_eur_factors.append(r2_compound)
            
            if r5_eur_df is not None:
                r5_with_rf = r5_eur_df.join(factors_eur_usd_temp[['RF_EUR']], how='inner')
                
                r5_gross_compound = r5_with_rf[['R5_gross']].resample(freq_label).apply(
                    lambda x: ((1 + x/100).prod() - 1) * 100
                )
                r5_excess = ((1 + r5_gross_compound['R5_gross']/100) / (1 + rf_eur_compound['RF_EUR']/100) - 1) * 100
                gross_eur_factors.append(pd.DataFrame({'R5': r5_excess}))
                print(f"✅ R5 EUR: compound gross + excess via ratio")
            elif 'R5' in factors_eur_df.columns:
                r5_compound = factors_eur_df[['R5']].resample(freq_label).apply(
                    lambda x: ((1 + x/100).prod() - 1) * 100
                )
                gross_eur_factors.append(r5_compound)
            
            if r10_eur_df is not None:
                r10_with_rf = r10_eur_df.join(factors_eur_usd_temp[['RF_EUR']], how='inner')
                
                r10_gross_compound = r10_with_rf[['R10_gross']].resample(freq_label).apply(
                    lambda x: ((1 + x/100).prod() - 1) * 100
                )
                r10_excess = ((1 + r10_gross_compound['R10_gross']/100) / (1 + rf_eur_compound['RF_EUR']/100) - 1) * 100
                gross_eur_factors.append(pd.DataFrame({'R10': r10_excess}))
                print(f"✅ R10 EUR: compound gross + excess via ratio")
            elif 'R10' in factors_eur_df.columns:
                r10_compound = factors_eur_df[['R10']].resample(freq_label).apply(
                    lambda x: ((1 + x/100).prod() - 1) * 100
                )
                gross_eur_factors.append(r10_compound)
            
            factors_eur_final = pd.concat([ff_eur_factors] + gross_eur_factors, axis=1)
            
            print(f"✅ Frequenza: {FACTOR_FREQ}")
            print(f"✅ EUR factors: compound USD first, then convert")
    
    print(f"📊 EUR factors finale: {len(factors_eur_final)} osservazioni")

else:
    factors_eur_final = None
    print("\n⚠️  EUR factors non disponibili")

# ============================================================================
# STEP 14: SALVA DATASET FATTORI (GLOBALI)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 14: Salva dataset fattori (globali)")
print("=" * 80)

factors_us_path = PROCESSED_DATA_DIR / f"all_risk_factors_us_{FACTOR_FREQ}.csv"
factors_us_final.to_csv(factors_us_path)

print(f"💾 US factors salvati: {factors_us_path.name}")
print(f"   📊 Fattori: {list(factors_us_final.columns)}")
print(f"   📅 Periodo: {factors_us_final.index.min().strftime('%Y-%m-%d')} to {factors_us_final.index.max().strftime('%Y-%m-%d')}")
print(f"   📊 Osservazioni: {len(factors_us_final)}")

if factors_eur_final is not None:
    factors_eur_path = PROCESSED_DATA_DIR / f"all_risk_factors_eur_{FACTOR_FREQ}.csv"
    factors_eur_final.to_csv(factors_eur_path)
    
    print(f"\n💾 EUR factors salvati: {factors_eur_path.name}")
    print(f"   📊 Fattori: {list(factors_eur_final.columns)}")
    print(f"   📅 Periodo: {factors_eur_final.index.min().strftime('%Y-%m-%d')} to {factors_eur_final.index.max().strftime('%Y-%m-%d')}")
    print(f"   📊 Osservazioni: {len(factors_eur_final)}")

# ============================================================================
# STEP 15: LOOP SU TUTTE LE STRATEGIE - CREA DATASET REGRESSIONE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 15: Loop su tutte le strategie - Crea dataset regressione")
print("=" * 80)

regression_files_created = {'US': [], 'EUR': []}

for strategy_name, strategy_rel_path in STRATEGIES.items():
    
    print(f"\n{'='*80}")
    print(f"STRATEGIA: {strategy_name}")
    print(f"{'='*80}")
    
    # Carica strategy returns
    strategy_path = RESULTS_DIR / strategy_rel_path
    
    if not strategy_path.exists():
        print(f"⚠️  File non trovato: {strategy_path}")
        print(f"   Skipping {strategy_name}...")
        continue
    
    try:
        index_df = pd.read_csv(strategy_path, index_col=0, parse_dates=True)
        
        # Resample strategy returns
        if FACTOR_FREQ == "weekly":
            index_resampled = index_df[['index_return']].resample('W-FRI').apply(
                lambda x: ((1 + x/100).prod() - 1) * 100)
        elif FACTOR_FREQ == "monthly":
            index_resampled = index_df[['index_return']].resample('M').apply(
                lambda x: ((1 + x/100).prod() - 1) * 100)
        else:
            index_resampled = index_df[['index_return']].copy()
        
        index_resampled = index_resampled.rename(columns={'index_return': 'Strategy_Return'})
        
        print(f"✅ Strategy returns caricati: {len(index_resampled)} osservazioni")
        print(f"📅 Periodo: {index_resampled.index.min().strftime('%Y-%m-%d')} to {index_resampled.index.max().strftime('%Y-%m-%d')}")
        
        # === US REGRESSION DATA ===
        regression_us_data = index_resampled.join(factors_us_final, how='inner')
        regression_us_data = regression_us_data.dropna()
        
        if len(regression_us_data) > 0:
            print(f"\n🇺🇸 US regression data: {len(regression_us_data)} osservazioni")
            
            regression_us_path = PROCESSED_DATA_DIR / f"regression_data_{strategy_name.lower()}_us_{FACTOR_FREQ}.csv"
            regression_us_data.to_csv(regression_us_path)
            print(f"💾 Salvato: {regression_us_path.name}")
            
            regression_files_created['US'].append(strategy_name)
        else:
            print(f"\n⚠️  Nessun overlap US per {strategy_name}")
        
        # === EUR REGRESSION DATA ===
        if factors_eur_final is not None:
            regression_eur_data = index_resampled.join(factors_eur_final, how='inner')
            regression_eur_data = regression_eur_data.dropna()
            
            if len(regression_eur_data) > 0:
                print(f"\n🇪🇺 EUR regression data: {len(regression_eur_data)} osservazioni")
                
                regression_eur_path = PROCESSED_DATA_DIR / f"regression_data_{strategy_name.lower()}_eur_{FACTOR_FREQ}.csv"
                regression_eur_data.to_csv(regression_eur_path)
                print(f"💾 Salvato: {regression_eur_path.name}")
                
                regression_files_created['EUR'].append(strategy_name)
            else:
                print(f"\n⚠️  Nessun overlap EUR per {strategy_name}")
        
    except Exception as e:
        print(f"❌ ERRORE processing {strategy_name}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# STEP 16: STATISTICHE FINALI
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICHE FINALI")
print("=" * 80)

print("\n🇺🇸 US FACTORS:")
print("\n📊 Statistiche Descrittive:")
print(factors_us_final.describe().round(3))

print("\n📊 Correlazioni tra Fattori:")
print(factors_us_final.corr().round(3))

# Print correlazioni con strategy returns per ogni strategia processata
if regression_files_created['US']:
    print("\n📊 Correlazioni con Strategy Returns (US):")
    for strategy_name in regression_files_created['US']:
        regression_path = PROCESSED_DATA_DIR / f"regression_data_{strategy_name.lower()}_us_{FACTOR_FREQ}.csv"
        if regression_path.exists():
            reg_data = pd.read_csv(regression_path, index_col=0, parse_dates=True)
            print(f"\n   {strategy_name}:")
            corr_series = reg_data.corr()['Strategy_Return'].drop('Strategy_Return')
            for factor, corr_val in corr_series.items():
                print(f"      {factor:10s}: {corr_val:6.3f}")

if factors_eur_final is not None:
    print("\n" + "=" * 80)
    print("🇪🇺 EUR FACTORS:")
    print("\n📊 Statistiche Descrittive:")
    print(factors_eur_final.describe().round(3))
    
    print("\n📊 Correlazioni tra Fattori:")
    print(factors_eur_final.corr().round(3))
    
    # Print correlazioni con strategy returns per ogni strategia processata
    if regression_files_created['EUR']:
        print("\n📊 Correlazioni con Strategy Returns (EUR):")
        for strategy_name in regression_files_created['EUR']:
            regression_path = PROCESSED_DATA_DIR / f"regression_data_{strategy_name.lower()}_eur_{FACTOR_FREQ}.csv"
            if regression_path.exists():
                reg_data = pd.read_csv(regression_path, index_col=0, parse_dates=True)
                print(f"\n   {strategy_name}:")
                corr_series = reg_data.corr()['Strategy_Return'].drop('Strategy_Return')
                for factor, corr_val in corr_series.items():
                    print(f"      {factor:10s}: {corr_val:6.3f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✅ COMPLETATO!")
print("=" * 80)

print(f"\n📁 File US generati:")
print(f"   • {factors_us_path.name}")
for strategy in regression_files_created['US']:
    print(f"   • regression_data_{strategy.lower()}_us_{FACTOR_FREQ}.csv")

if factors_eur_final is not None:
    print(f"\n📁 File EUR generati:")
    print(f"   • {factors_eur_path.name}")
    for strategy in regression_files_created['EUR']:
        print(f"   • regression_data_{strategy.lower()}_eur_{FACTOR_FREQ}.csv")

print(f"\n📊 Frequenza: {FACTOR_FREQ}")
print(f"📊 Fattori US disponibili: {len(factors_us_final.columns)}")
if factors_eur_final is not None:
    print(f"📊 Fattori EUR disponibili: {len(factors_eur_final.columns)}")

print(f"\n📊 Strategie processate:")
print(f"   US:  {len(regression_files_created['US'])}")
if factors_eur_final is not None:
    print(f"   EUR: {len(regression_files_created['EUR'])}")

print("\n✅ FIX APPLICATO: RI_RB.xlsx skiprows=2")
print("✅ Pronto per le regressioni multi-strategy con fattori US e EUR!")
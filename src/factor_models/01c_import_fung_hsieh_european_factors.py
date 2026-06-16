"""
Script 1c: Import Fung & Hsieh (2004) Factors - US & EUR - MULTI STRATEGY
==========================================================================
Importa i 7 fattori del modello Fung & Hsieh per hedge fund strategies.

STRATEGIE ANALIZZATE:
1. BTP-Italia
2. iTraxx Main
3. iTraxx SnrFin
4. iTraxx SubFin
5. iTraxx Xover
6. iTraxx Combined

FATTORI IMPORTATI:
US:
1. SNP: S&P 500 Return (monthly from EOM)
2. SIZE: Russell - S&P 500 (difference of monthly returns)
3. TERM: 10Y Yield Change (EOM diff)
4. CREDIT: Credit Spread Change (EOM diff)
5-7. PTFSBD, PTFSFX, PTFSCOM: Trend Following (monthly)

EUR:
1. SNP: Mkt-RF EUR + RF EUR (ricostruito da Duarte factors)
2. SIZE: SMB EUR (riutilizzato da Duarte factors)
3. TERM: 10Y Bund Yield Change (EOM diff)
4. CREDIT: Credit Spread EUR Change (EOM diff)
5-7. PTFSBD, PTFSFX, PTFSCOM: Stessi di US (currency-independent)

NOTE:
- NO risk-free adjustment per TF factors (già excess returns)
- Frequenza: SOLO monthly (TF factors disponibili solo mensili)
- EUR: riutilizza calcoli Duarte per SNP e SIZE
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETRI
# ============================================================================

FACTOR_FREQ = "monthly"  # Fixed - TF factors only monthly

# ============================================================================
# STRATEGIE DA ANALIZZARE
# ============================================================================

STRATEGIES = {
    'btp_italia': 'btp_italia/index_daily.csv',
    'itraxx_main': 'itraxx_main/index_daily.csv',
    'itraxx_snrfin': 'itraxx_snrfin/index_daily.csv',
    'itraxx_subfin': 'itraxx_subfin/index_daily.csv',
    'itraxx_xover': 'itraxx_xover/index_daily.csv',
    'itraxx_combined': 'itraxx_combined/index_daily.csv',
    'CDS_Bond_Basis': 'cds_bond_basis/index_daily.csv'
}

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FUNG_HSIEH_FILE = EXTERNAL_DATA_DIR / "Fung_Hsieh_factors.xlsx"

# Check esistenza file
if not FUNG_HSIEH_FILE.exists():
    print(f"❌ ERRORE: File non trovato: {FUNG_HSIEH_FILE}")
    exit()

print(f"✅ File trovato: {FUNG_HSIEH_FILE.name}\n")
# ============================================================================
# STEP 1: INFORMAZIONI
# ============================================================================

print("=" * 80)
print("IMPORT FUNG & HSIEH (2004) FACTORS - US & EUR - MULTI STRATEGY")
print("=" * 80)

print(f"\n📊 Frequenza: {FACTOR_FREQ.upper()}")

print("\n🎯 STRATEGIE:")
for i, strategy_name in enumerate(STRATEGIES.keys(), 1):
    print(f"   {i}. {strategy_name}")

print("\n🇺🇸 US FACTORS:")
print("   1. SNP: S&P 500 Return (monthly from EOM)")
print("   2. SIZE: Russell - S&P (difference of monthly returns)")
print("   3. TERM: 10Y Yield Change (EOM diff)")
print("   4. CREDIT: Credit Spread Change (EOM diff)")
print("   5-7. PTFSBD, PTFSFX, PTFSCOM: Trend Following")

print("\n🇪🇺 EUR FACTORS:")
print("   1. SNP: Mkt-RF EUR + RF EUR (from Duarte factors)")
print("   2. SIZE: SMB EUR (from Duarte factors)")
print("   3. TERM: 10Y Bund Yield Change (EOM diff)")
print("   4. CREDIT: Credit Spread EUR Change (EOM diff)")
print("   5-7. PTFSBD, PTFSFX, PTFSCOM: Same as US")
print()

# ============================================================================
# PART A: US FACTORS
# ============================================================================

print("=" * 80)
print("PART A: US FACTORS")
print("=" * 80)

# ============================================================================
# STEP 2: IMPORTA S&P 500 E RUSSELL (EQUITY_RF.XLSX)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Importa S&P 500 e Russell (US)")
print("=" * 80)

try:
    equity_raw = pd.read_excel(FUNG_HSIEH_FILE, sheet_name="Equity_RF", skiprows=6)
    
    date_col = equity_raw.columns[0]
    sp500_col = equity_raw.columns[1]
    russell_col = equity_raw.columns[2]
    
    print(f"📊 Colonne: {list(equity_raw.columns[:3])}")
    
    # Converti date
    equity_raw[date_col] = pd.to_datetime(equity_raw[date_col], errors='coerce')
    equity_raw = equity_raw.dropna(subset=[date_col])
    equity_data = equity_raw.set_index(date_col)
    
    # Indici (livelli)
    sp500_idx = equity_data[sp500_col].astype(float)
    russell_idx = equity_data[russell_col].astype(float)
    
    print(f"✅ S&P 500 index: {len(sp500_idx)} osservazioni")
    print(f"✅ Russell index: {len(russell_idx)} osservazioni")
    print(f"📅 Periodo: {sp500_idx.index.min()} to {sp500_idx.index.max()}")
    
    # Prendi livelli EOM
    sp500_eom = sp500_idx.resample('M').last()
    russell_eom = russell_idx.resample('M').last()
    
    print(f"\n✅ S&P 500 EOM: {len(sp500_eom)} mesi")
    print(f"✅ Russell EOM: {len(russell_eom)} mesi")
    
    # Calcola monthly returns DA EOM
    sp500_ret_monthly = sp500_eom.pct_change() * 100
    russell_ret_monthly = russell_eom.pct_change() * 100
    
    print(f"\n📊 S&P 500 return medio (monthly): {sp500_ret_monthly.mean():.4f}%")
    print(f"📊 Russell return medio (monthly): {russell_ret_monthly.mean():.4f}%")
    
    # SIZE = Differenza di MONTHLY returns
    size_monthly = russell_ret_monthly - sp500_ret_monthly
    
    print(f"📊 SIZE medio (monthly): {size_monthly.mean():.4f}%")
    
    equity_us_df = pd.DataFrame({
        'SNP': sp500_ret_monthly,
        'SIZE': size_monthly
    })
    
except Exception as e:
    print(f"❌ ERRORE nell'import Equity: {e}")
    import traceback
    traceback.print_exc()
    exit()

# ============================================================================
# STEP 3: IMPORTA 10Y TREASURY YIELD (DGS10.CSV) - US
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Importa 10Y Treasury Yield (US)")
print("=" * 80)

try:
    dgs10_raw = pd.read_excel(FUNG_HSIEH_FILE, sheet_name="DGS10")
    dgs10_raw.columns = ['DATE', 'DGS10']
    
    # Converti date
    dgs10_raw['DATE'] = pd.to_datetime(dgs10_raw['DATE'], errors='coerce')
    dgs10_raw = dgs10_raw.dropna(subset=['DATE'])
    dgs10_data = dgs10_raw.set_index('DATE')
    
    # Yield in %
    dgs10_yield = dgs10_data['DGS10'].astype(float)
    
    print(f"✅ 10Y Yield: {len(dgs10_yield)} osservazioni")
    print(f"📅 Periodo: {dgs10_yield.index.min()} to {dgs10_yield.index.max()}")
    print(f"📊 Yield medio: {dgs10_yield.mean():.2f}%")
    
    # Prendi livelli EOM
    dgs10_eom = dgs10_yield.resample('M').last()
    
    print(f"\n✅ 10Y Yield EOM: {len(dgs10_eom)} mesi")
    print(f"📊 EOM medio: {dgs10_eom.mean():.2f}%")
    
    # TERM = Differenza EOM(t) - EOM(t-1)
    term_change = dgs10_eom.diff()
    
    print(f"📊 TERM change medio (monthly): {term_change.mean():.4f}")
    
    dgs10_us_df = pd.DataFrame({
        'TERM': term_change,
        'DGS10_EOM': dgs10_eom
    })
    
except Exception as e:
    print(f"❌ ERRORE nell'import DGS10: {e}")
    import traceback
    traceback.print_exc()
    exit()

# ============================================================================
# STEP 4: IMPORTA CREDIT SPREAD (DBAA - DGS10) - US
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Importa Credit Spread (Baa - 10Y) - US")
print("=" * 80)

try:
    dbaa_raw = pd.read_excel(FUNG_HSIEH_FILE, sheet_name="DBAA")
    dbaa_raw.columns = ['DATE', 'DBAA']
    
    # Converti date
    dbaa_raw['DATE'] = pd.to_datetime(dbaa_raw['DATE'], errors='coerce')
    dbaa_raw = dbaa_raw.dropna(subset=['DATE'])
    dbaa_data = dbaa_raw.set_index('DATE')
    
    # Baa yield in %
    dbaa_yield = dbaa_data['DBAA'].astype(float)
    
    print(f"✅ Baa Yield: {len(dbaa_yield)} osservazioni")
    print(f"📅 Periodo: {dbaa_yield.index.min()} to {dbaa_yield.index.max()}")
    print(f"📊 Baa yield medio: {dbaa_yield.mean():.2f}%")
    
    # Prendi livelli EOM
    dbaa_eom = dbaa_yield.resample('M').last()
    
    print(f"\n✅ Baa Yield EOM: {len(dbaa_eom)} mesi")
    print(f"📊 EOM medio: {dbaa_eom.mean():.2f}%")
    
    # Merge con 10Y EOM per calcolare spread
    spread_data = pd.DataFrame({
        'Baa_EOM': dbaa_eom,
        'Treasury10Y_EOM': dgs10_us_df['DGS10_EOM']
    }).dropna()
    
    # Credit spread EOM = Baa_EOM - 10Y_EOM
    credit_spread_eom = spread_data['Baa_EOM'] - spread_data['Treasury10Y_EOM']
    
    print(f"📊 Credit spread medio (EOM): {credit_spread_eom.mean():.2f}%")
    
    # CREDIT = Differenza EOM(t) - EOM(t-1)
    credit_change = credit_spread_eom.diff()
    
    print(f"📊 CREDIT change medio (monthly): {credit_change.mean():.4f}")
    
    credit_us_df = pd.DataFrame({'CREDIT': credit_change})
    
except Exception as e:
    print(f"❌ ERRORE nell'import Credit Spread: {e}")
    import traceback
    traceback.print_exc()
    exit()

# ============================================================================
# STEP 5: IMPORTA TREND FOLLOWING FACTORS (TF-FAC.XLS)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Importa Trend Following Factors")
print("=" * 80)

try:
    tf_raw = pd.read_excel(FUNG_HSIEH_FILE, sheet_name="TF_Factors", skiprows=1)
    
    print(f"📊 Colonne TF file: {list(tf_raw.columns[:4])}")
    
    date_col = tf_raw.columns[0]
    ptfsbd_col = tf_raw.columns[1]
    ptfsfx_col = tf_raw.columns[2]
    ptfscom_col = tf_raw.columns[3]
    
    # Filtra solo righe con date valide (6 cifre YYYYMM)
    tf_raw = tf_raw[tf_raw[date_col].notna()]
    tf_raw = tf_raw[tf_raw[date_col].astype(str).str.match(r'^\d{6}$', na=False)]
    
    print(f"📊 Righe valide: {len(tf_raw)}")
    
    # Converti YYYYMM in date (ultimo giorno del mese - EOM)
    tf_raw[date_col] = pd.to_datetime(
        tf_raw[date_col].astype(int).astype(str), 
        format='%Y%m'
    ) + pd.offsets.MonthEnd(0)
    
    tf_data = tf_raw.set_index(date_col)
    
    # Estrai fattori (già in excess returns %)
    ptfsbd = tf_data[ptfsbd_col].astype(float)
    ptfsfx = tf_data[ptfsfx_col].astype(float)
    ptfscom = tf_data[ptfscom_col].astype(float)
    
    print(f"✅ PTFSBD (Bond TF): {len(ptfsbd)} mesi")
    print(f"✅ PTFSFX (FX TF): {len(ptfsfx)} mesi")
    print(f"✅ PTFSCOM (Commodity TF): {len(ptfscom)} mesi")
    print(f"📅 Periodo: {ptfsbd.index.min()} to {ptfsbd.index.max()}")
    
    print(f"📊 PTFSBD medio: {ptfsbd.mean():.4f}%")
    print(f"📊 PTFSFX medio: {ptfsfx.mean():.4f}%")
    print(f"📊 PTFSCOM medio: {ptfscom.mean():.4f}%")
    
    tf_df = pd.DataFrame({
        'PTFSBD': ptfsbd,
        'PTFSFX': ptfsfx,
        'PTFSCOM': ptfscom
    })
    
except Exception as e:
    print(f"❌ ERRORE nell'import TF factors: {e}")
    import traceback
    traceback.print_exc()
    exit()

# ============================================================================
# STEP 6: MERGE TUTTI I FATTORI US
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Merge tutti i fattori US")
print("=" * 80)

# Merge equity + yield/spread
factors_us_df = equity_us_df.copy()

# Join TERM (drop colonna DGS10_EOM che non serve)
factors_us_df = factors_us_df.join(dgs10_us_df[['TERM']], how='inner')

# Join CREDIT
factors_us_df = factors_us_df.join(credit_us_df, how='inner')

print(f"📊 Dopo merge equity + yield/spread: {len(factors_us_df)} mesi")
print(f"📅 Periodo: {factors_us_df.index.min()} to {factors_us_df.index.max()}")

# Merge con TF factors
factors_us_df = factors_us_df.join(tf_df, how='inner')

print(f"📊 Dopo merge con TF factors: {len(factors_us_df)} mesi")
print(f"📅 Periodo finale: {factors_us_df.index.min()} to {factors_us_df.index.max()}")

# Rimuovi NaN
factors_us_df = factors_us_df.dropna()
print(f"✅ Dopo pulizia NaN: {len(factors_us_df)} mesi")

print(f"\n📊 Fattori finali: {list(factors_us_df.columns)}")

# Verifica ordine colonne
expected_order = ['SNP', 'SIZE', 'TERM', 'CREDIT', 'PTFSBD', 'PTFSFX', 'PTFSCOM']
factors_us_df = factors_us_df[expected_order]

# ============================================================================
# PART B: EUR FACTORS
# ============================================================================

print("\n" + "=" * 80)
print("PART B: EUR FACTORS")
print("=" * 80)

# ============================================================================
# STEP 7: IMPORTA SNP E SIZE DA DUARTE FACTORS EUR
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Importa SNP e SIZE EUR (da Duarte factors)")
print("=" * 80)

# Carica fattori Duarte EUR già processati (monthly)
duarte_eur_path = PROCESSED_DATA_DIR / f"all_risk_factors_eur_{FACTOR_FREQ}.csv"

factors_eur_df = None

if not duarte_eur_path.exists():
    print(f"❌ ERRORE: File non trovato: {duarte_eur_path}")
    print("⚠️  Devi prima eseguire lo script 01_import_all_risk_factors.py")
    print("⚠️  Continuo solo con fattori US")
else:
    try:
        # Carica fattori Duarte EUR
        duarte_eur_factors = pd.read_csv(duarte_eur_path, index_col=0, parse_dates=True)
        
        print(f"✅ Duarte EUR factors caricati: {len(duarte_eur_factors)} mesi")
        print(f"📅 Periodo: {duarte_eur_factors.index.min()} to {duarte_eur_factors.index.max()}")
        print(f"📊 Colonne disponibili: {list(duarte_eur_factors.columns)}")
        
        # Carica anche Euribor per RF EUR
        try:
            euribor_raw = pd.read_excel(FUNG_HSIEH_FILE, sheet_name="Euribor1m", skiprows=7)
            euribor_raw.columns = ['Date', 'Euribor1m', 'Col_C', 'FX_USD_EUR']
            euribor_raw['Date'] = pd.to_datetime(euribor_raw['Date'])
            euribor_data = euribor_raw.set_index('Date')[['Euribor1m']]
            
            # RF daily = Euribor1m / 360
            euribor_data['RF_EUR'] = euribor_data['Euribor1m'] / 360
            
            # Resample a monthly
            rf_eur_monthly = euribor_data['RF_EUR'].resample('M').apply(
                lambda x: ((1 + x/100).prod() - 1) * 100
            )
            
            print(f"✅ RF EUR monthly: {len(rf_eur_monthly)} mesi")
            print(f"📊 RF EUR medio: {rf_eur_monthly.mean():.4f}%")
            
            # ================================================================
            # SNP EUR = Mkt-RF EUR + RF EUR (ricostruisco total return)
            # SIZE EUR = SMB EUR (già pronto)
            # ================================================================
            
            # Merge Mkt-RF con RF_EUR
            snp_size_data = duarte_eur_factors[['Mkt-RF', 'SMB']].copy()
            snp_size_data = snp_size_data.join(rf_eur_monthly, how='inner')
            
            # SNP = Mkt-RF + RF
            snp_size_data['SNP'] = snp_size_data['Mkt-RF'] + snp_size_data['RF_EUR']
            snp_size_data['SIZE'] = snp_size_data['SMB']
            
            equity_eur_df = snp_size_data[['SNP', 'SIZE']].copy()
            
            print(f"\n✅ SNP EUR (Mkt total return): mean={equity_eur_df['SNP'].mean():.4f}%")
            print(f"✅ SIZE EUR (SMB): mean={equity_eur_df['SIZE'].mean():.4f}%")
        except Exception as e:
            print(f"❌ ERRORE import Euribor: {e}")
            import traceback
            traceback.print_exc()
            factors_eur_df = None
            
        # ================================================================
        # STEP 8: IMPORTA TERM E CREDIT EUR (EUROPE_FUNG&HSIEH.XLSX)
        # ================================================================

        print("\n" + "=" * 80)
        print("STEP 8: Importa TERM e CREDIT EUR (10Y Bund + Baa EUR)")
        print("=" * 80)


        try:
            fh_eur_raw = pd.read_excel(FUNG_HSIEH_FILE, sheet_name="Europe_Yields", skiprows=6)
            
            # Colonna A: Date, B: 10Y Bund, C: Baa EUR
            fh_eur_raw.columns = ['Date', 'Bund10Y', 'Baa_EUR']
            fh_eur_raw['Date'] = pd.to_datetime(fh_eur_raw['Date'], errors='coerce')
            fh_eur_raw = fh_eur_raw.dropna(subset=['Date'])
            fh_eur_data = fh_eur_raw.set_index('Date')
            
            # Yields in %
            bund10y_yield = fh_eur_data['Bund10Y'].astype(float)
            baa_eur_yield = fh_eur_data['Baa_EUR'].astype(float)
            
            print(f"✅ 10Y Bund Yield: {len(bund10y_yield)} osservazioni")
            print(f"✅ Baa EUR Yield: {len(baa_eur_yield)} osservazioni")
            print(f"📅 Periodo: {bund10y_yield.index.min()} to {bund10y_yield.index.max()}")
            
            # Prendi livelli EOM
            bund10y_eom = bund10y_yield.resample('M').last()
            baa_eur_eom = baa_eur_yield.resample('M').last()
            
            print(f"\n📊 Bund 10Y EOM medio: {bund10y_eom.mean():.2f}%")
            print(f"📊 Baa EUR EOM medio: {baa_eur_eom.mean():.2f}%")
            
            # TERM EUR = Differenza EOM(t) - EOM(t-1)
            term_eur_change = bund10y_eom.diff()
            
            # Credit spread EUR = Baa - Bund
            credit_spread_eur_eom = baa_eur_eom - bund10y_eom
            
            print(f"📊 Credit spread EUR medio: {credit_spread_eur_eom.mean():.2f}%")
            
            # CREDIT EUR = Differenza EOM(t) - EOM(t-1)
            credit_eur_change = credit_spread_eur_eom.diff()
            
            print(f"\n📊 TERM EUR change medio: {term_eur_change.mean():.4f}")
            print(f"📊 CREDIT EUR change medio: {credit_eur_change.mean():.4f}")
            
            term_credit_eur_df = pd.DataFrame({
                'TERM': term_eur_change,
                'CREDIT': credit_eur_change
            })
            
            # ================================================================
            # STEP 9: MERGE TUTTI I FATTORI EUR
            # ================================================================

            print("\n" + "=" * 80)
            print("STEP 9: Merge tutti i fattori EUR")
            print("=" * 80)

            # Merge equity + term/credit
            factors_eur_df = equity_eur_df.copy()
            factors_eur_df = factors_eur_df.join(term_credit_eur_df, how='inner')
            
            print(f"📊 Dopo merge equity + yield/spread: {len(factors_eur_df)} mesi")
            
            # Merge con TF factors (stessi di US - currency independent)
            factors_eur_df = factors_eur_df.join(tf_df, how='inner')
            
            print(f"📊 Dopo merge con TF factors: {len(factors_eur_df)} mesi")
            print(f"📅 Periodo finale: {factors_eur_df.index.min()} to {factors_eur_df.index.max()}")
            
            # Rimuovi NaN
            factors_eur_df = factors_eur_df.dropna()
            print(f"✅ Dopo pulizia NaN: {len(factors_eur_df)} mesi")
            
            print(f"\n📊 Fattori finali EUR: {list(factors_eur_df.columns)}")
            
            # Verifica ordine colonne
            factors_eur_df = factors_eur_df[expected_order]
            
        except Exception as e:
            print(f"❌ ERRORE nell'import EUR yields: {e}")
            import traceback
            traceback.print_exc()
            factors_eur_df = None
                    
    except Exception as e:
        print(f"❌ ERRORE nel processing EUR: {e}")
        import traceback
        traceback.print_exc()
        factors_eur_df = None

# ============================================================================
# STEP 10: SALVA FATTORI GLOBALI
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Salva fattori Fung & Hsieh globali")
print("=" * 80)

# Salva US factors
factors_us_path = PROCESSED_DATA_DIR / f"fung_hsieh_factors_us_{FACTOR_FREQ}.csv"
factors_us_df.to_csv(factors_us_path)

print(f"💾 US Salvato: {factors_us_path.name}")
print(f"📊 US Fattori: {list(factors_us_df.columns)}")

# Salva EUR factors (se disponibile)
if factors_eur_df is not None:
    factors_eur_path = PROCESSED_DATA_DIR / f"fung_hsieh_factors_eur_{FACTOR_FREQ}.csv"
    factors_eur_df.to_csv(factors_eur_path)
    
    print(f"\n💾 EUR Salvato: {factors_eur_path.name}")
    print(f"📊 EUR Fattori: {list(factors_eur_df.columns)}")

# ============================================================================
# STEP 11: MERGE CON TUTTE LE STRATEGIE (LOOP)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 11: Merge con TUTTE le strategie")
print("=" * 80)

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
        
        # Resample strategy a monthly (Fung & Hsieh solo mensile)
        strategy = index_df[['index_return']].resample('M').apply(
            lambda x: ((1 + x/100).prod() - 1) * 100)
        
        strategy = strategy.rename(columns={'index_return': 'Strategy_Return'})
        
        # Merge US
        regression_us_data = strategy.join(factors_us_df, how='inner')
        regression_us_data = regression_us_data.dropna()
        
        print(f"✅ US regression data: {len(regression_us_data)} osservazioni")
        
        regression_us_path = PROCESSED_DATA_DIR / f"regression_data_fung_hsieh_{strategy_name}_us_{FACTOR_FREQ}.csv"
        regression_us_data.to_csv(regression_us_path)
        print(f"💾 Salvato: {regression_us_path.name}")
        
        regression_files_created['US'].append(regression_us_path.name)
        
        # Merge EUR (se disponibile)
        if factors_eur_df is not None:
            regression_eur_data = strategy.join(factors_eur_df, how='inner')
            regression_eur_data = regression_eur_data.dropna()
            
            print(f"✅ EUR regression data: {len(regression_eur_data)} osservazioni")
            
            regression_eur_path = PROCESSED_DATA_DIR / f"regression_data_fung_hsieh_{strategy_name}_eur_{FACTOR_FREQ}.csv"
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
print(factors_us_df.describe().round(3))

print("\n📊 Correlazioni tra Fattori:")
print(factors_us_df.corr().round(3))

# Correlazioni con Strategy Return per OGNI strategia US
print("\n📊 Correlazioni con Strategy Return (US):")
for fname in regression_files_created['US']:
    # Estrai nome strategia dal filename
    strategy_name = fname.replace('regression_data_fung_hsieh_', '').replace(f'_us_{FACTOR_FREQ}.csv', '')
    
    # Carica regression data
    reg_data_path = PROCESSED_DATA_DIR / fname
    if reg_data_path.exists():
        reg_data = pd.read_csv(reg_data_path, index_col=0, parse_dates=True)
        
        print(f"\n   {strategy_name}:")
        corr_with_strategy = reg_data.corr()['Strategy_Return'].drop('Strategy_Return')
        for factor, corr_val in corr_with_strategy.items():
            print(f"      {factor:20s}: {corr_val:7.3f}")

if factors_eur_df is not None:
    print("\n" + "=" * 80)
    print("🇪🇺 EUR FACTORS:")
    print("\n📊 Statistiche Descrittive:")
    print(factors_eur_df.describe().round(3))
    
    print("\n📊 Correlazioni tra Fattori:")
    print(factors_eur_df.corr().round(3))
    
    # Correlazioni con Strategy Return per OGNI strategia EUR
    if len(regression_files_created['EUR']) > 0:
        print("\n📊 Correlazioni con Strategy Return (EUR):")
        for fname in regression_files_created['EUR']:
            # Estrai nome strategia dal filename
            strategy_name = fname.replace('regression_data_fung_hsieh_', '').replace(f'_eur_{FACTOR_FREQ}.csv', '')
            
            # Carica regression data
            reg_data_path = PROCESSED_DATA_DIR / fname
            if reg_data_path.exists():
                reg_data = pd.read_csv(reg_data_path, index_col=0, parse_dates=True)
                
                print(f"\n   {strategy_name}:")
                corr_with_strategy = reg_data.corr()['Strategy_Return'].drop('Strategy_Return')
                for factor, corr_val in corr_with_strategy.items():
                    print(f"      {factor:20s}: {corr_val:7.3f}")

print("\n" + "=" * 80)
print("✅ COMPLETATO!")
print("=" * 80)

print(f"\n📁 File GLOBALI generati:")
print(f"   • fung_hsieh_factors_us_{FACTOR_FREQ}.csv")
if factors_eur_df is not None:
    print(f"   • fung_hsieh_factors_eur_{FACTOR_FREQ}.csv")

print(f"\n📁 File REGRESSION US generati ({len(regression_files_created['US'])}):")
for fname in regression_files_created['US']:
    print(f"   • {fname}")

if factors_eur_df is not None and len(regression_files_created['EUR']) > 0:
    print(f"\n📁 File REGRESSION EUR generati ({len(regression_files_created['EUR'])}):")
    for fname in regression_files_created['EUR']:
        print(f"   • {fname}")

print("\n⚠️  FIX APPLICATI:")
print("   ✅ US: SNP e SIZE da EOM, TERM/CREDIT da EOM diff")
print("   ✅ EUR: SNP=Mkt-RF+RF, SIZE=SMB (da Duarte)")
print("   ✅ EUR: TERM/CREDIT da Bund e Baa EUR")
print("   ✅ TF factors: currency-independent (stessi per US e EUR)")
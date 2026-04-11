"""
Import Active Fixed Income Illusion Factors - MULTI STRATEGY - US & EUR
=========================================================================
Importa i fattori Active FI Illusion sia in versione US che EUR.
Crea regression datasets per TUTTE LE STRATEGIE.

FATTORI (8 totali):
1. US/EU Term - Total Return Index → calc returns → subtract RF
2. Global Term - Total Return Index → calc returns → subtract RF
3. Global Aggregate - Total Return Index → calc returns → subtract RF
4. Inflation-Linkers - Total Return Index → calc returns → subtract RF
5. Corporate Credit - Hybrid: 100% G pre-split, 50-50 post-split
6. Emerging Debt - Already excess returns (daily)
7. Emerging Currency - Total Return Index, switch date → subtract RF
8. UST Implied Volatility - First-order difference

⚠️ CRITICAL FIXES:
- Corporate Credit G: NON riaggiungere RF (è già excess)
- Corporate Credit weekly/monthly: compound corp_credit_daily direttamente
- TRI factors: compound gross → excess via RATIO
- Split dates: US=30/03/2007, EUR=02/05/2013
- Switch dates (EmCurr): entrambi 30/06/2008

INPUT: 
- Active_Fixed_Income_Illusion_Factors.xlsx (US)
- Europe_Active_Fixed_Income_Illusion_Factors.xlsx (EUR)
- index_daily.csv da ogni strategia

OUTPUT (per ogni strategia): 
- regression_data_active_fi_{strategy}_us_{freq}.csv
- regression_data_active_fi_{strategy}_eur_{freq}.csv

OUTPUT GLOBALE:
- active_fi_factors_us_{freq}.csv
- active_fi_factors_eur_{freq}.csv
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
# STRATEGIE DA PROCESSARE
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

ACTIVE_FI_FILE = EXTERNAL_DATA_DIR / "Active_FI_factors.xlsx"

# Check esistenza file
if not ACTIVE_FI_FILE.exists():
    print(f"❌ ERRORE: File non trovato: {ACTIVE_FI_FILE}")
    exit()

print(f"✅ File trovato: {ACTIVE_FI_FILE.name}\n")

# ============================================================================
# INFORMAZIONI
# ============================================================================

print("=" * 80)
print("IMPORT ACTIVE FI ILLUSION FACTORS - MULTI STRATEGY - US & EUR")
print("=" * 80)

print("\n🇺🇸 US FACTORS:")
print("   • File: Active_Fixed_Income_Illusion_Factors.xlsx")
print("   • Corporate Credit split: 30/03/2007")
print("   • Emerging Currency switch: 30/06/2008")

print("\n🇪🇺 EUR FACTORS:")
print("   • File: Europe_Active_Fixed_Income_Illusion_Factors.xlsx")
print("   • Corporate Credit split: 02/05/2013")
print("   • Emerging Currency switch: 30/06/2008")

print(f"\n📅 Frequenza: {FACTOR_FREQ}")

print(f"\n🎯 Strategie da processare ({len(STRATEGIES)}):")
for i, strategy in enumerate(STRATEGIES.keys(), 1):
    print(f"   {i}. {strategy}")

print()

# ============================================================================
# PART A: US FACTORS
# ============================================================================

print("=" * 80)
print("PART A: US FACTORS")
print("=" * 80)

# ============================================================================
# STEP 1: IMPORTA US ACTIVE FI FACTORS DA EXCEL
# ============================================================================

print("\n" + "=" * 80)
print("STEP 1: Importa US Active FI Factors")
print("=" * 80)

try:
    factors_us_raw = pd.read_excel(ACTIVE_FI_FILE, sheet_name="US_Factors", skiprows=17)
        
    print(f"📊 Shape iniziale: {factors_us_raw.shape}")
    
    # Colonna A = Date
    date_col = factors_us_raw.columns[0]
    factors_us_raw[date_col] = pd.to_datetime(factors_us_raw[date_col], format='%d/%m/%Y', errors='coerce')
    factors_us_raw = factors_us_raw.dropna(subset=[date_col])
    factors_us_raw = factors_us_raw.set_index(date_col)
    
    print(f"✅ Date parsate: {len(factors_us_raw)} osservazioni")
    print(f"📅 Periodo: {factors_us_raw.index.min()} to {factors_us_raw.index.max()}")
    
    # Estrai colonne US
    us_term_idx_us = factors_us_raw.iloc[:, 0]      # Col B
    global_term_idx_us = factors_us_raw.iloc[:, 1]  # Col C
    global_agg_idx_us = factors_us_raw.iloc[:, 2]   # Col D
    infl_link_idx_us = factors_us_raw.iloc[:, 3]    # Col E
    corp_credit_f_us = factors_us_raw.iloc[:, 4]    # Col F
    corp_credit_g_us = factors_us_raw.iloc[:, 5]    # Col G
    em_debt_us = factors_us_raw.iloc[:, 6]          # Col H
    em_curr_i_us = factors_us_raw.iloc[:, 7]        # Col I
    em_curr_j_us = factors_us_raw.iloc[:, 8]        # Col J
    ust_vol_us = factors_us_raw.iloc[:, 9]          # Col K
    
    print(f"✅ Colonne US estratte")
    
except Exception as e:
    print(f"❌ ERRORE nell'import Excel US: {e}")
    import traceback
    traceback.print_exc()
    exit()

# ============================================================================
# STEP 2: IMPORTA RISK-FREE RATE US (FAMA-FRENCH)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Importa Risk-Free Rate US")
print("=" * 80)


try:
    ff3_raw = pd.read_excel(ACTIVE_FI_FILE, sheet_name="F-F_Research_Data_Factors", 
                           skiprows=3, header=None)
    ff3_raw.columns = ['Date', 'Mkt-RF', 'SMB', 'HML', 'RF']
    ff3_raw = ff3_raw[ff3_raw['Date'].notna()]
    ff3_raw = ff3_raw[ff3_raw['Date'].astype(str).str.strip().str.isdigit()]
    ff3_raw['Date'] = pd.to_datetime(ff3_raw['Date'].astype(str), format='%Y%m%d')
    ff3_raw = ff3_raw.set_index('Date')
    rf_daily_us = ff3_raw['RF'].astype(float)
    
    # Allinea RF con date Active FI US
    rf_aligned_us = rf_daily_us.reindex(factors_us_raw.index, method='ffill')
    
    print(f"✅ RF US importato: {len(rf_aligned_us)} giorni")
    print(f"📊 RF US medio: {rf_aligned_us.mean():.4f}%")
    
except Exception as e:
    print(f"❌ ERRORE import RF US: {e}")
    exit()

# ============================================================================
# STEP 3: CALCOLA US RETURNS PER TRI FACTORS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Calcola US returns da Total Return Indices")
print("=" * 80)

# 1. US Term
us_term_ret_us = us_term_idx_us.pct_change() * 100
us_term_excess_us = us_term_ret_us - rf_aligned_us
print(f"✅ US Term: mean gross={us_term_ret_us.mean():.4f}%, mean excess={us_term_excess_us.mean():.4f}%")

# 2. Global Term
global_term_ret_us = global_term_idx_us.pct_change() * 100
global_term_excess_us = global_term_ret_us - rf_aligned_us
print(f"✅ Global Term: mean gross={global_term_ret_us.mean():.4f}%, mean excess={global_term_excess_us.mean():.4f}%")

# 3. Global Aggregate
global_agg_ret_us = global_agg_idx_us.pct_change() * 100
global_agg_excess_us = global_agg_ret_us - rf_aligned_us
print(f"✅ Global Aggregate: mean gross={global_agg_ret_us.mean():.4f}%, mean excess={global_agg_excess_us.mean():.4f}%")

# 4. Inflation-Linkers
infl_link_ret_us = infl_link_idx_us.pct_change() * 100
infl_link_excess_us = infl_link_ret_us - rf_aligned_us
print(f"✅ Inflation-Linkers: mean gross={infl_link_ret_us.mean():.4f}%, mean excess={infl_link_excess_us.mean():.4f}%")

# ============================================================================
# STEP 4: US CORPORATE CREDIT (HYBRID) - DAILY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: US Corporate Credit (Hybrid) - DAILY")
print("=" * 80)

# Split date: 30/03/2007
split_date_us = pd.Timestamp('2007-03-30')

# Colonna F: Total Return Index → calc return (gross)
corp_credit_f_ret_us = corp_credit_f_us.pct_change() * 100

# Colonna G: Già excess return %
corp_credit_g_excess_us = corp_credit_g_us.copy()

print(f"📊 US Col F - Gross return: mean={corp_credit_f_ret_us.mean():.4f}%")
print(f"📊 US Col G - Excess return: mean={corp_credit_g_excess_us.mean():.4f}%")

# Costruisci serie ibrida daily
corp_credit_daily_us = pd.Series(index=factors_us_raw.index, dtype=float)

for date in factors_us_raw.index:
    if date <= split_date_us:
        # Pre-2007: Solo G (già excess)
        corp_credit_daily_us.loc[date] = corp_credit_g_excess_us.loc[date]
    else:
        # Post-2007: 50-50 blend
        f_excess = corp_credit_f_ret_us.loc[date] - rf_aligned_us.loc[date]
        g_excess = corp_credit_g_excess_us.loc[date]
        
        if pd.notna(f_excess) and pd.notna(g_excess):
            corp_credit_daily_us.loc[date] = 0.5 * f_excess + 0.5 * g_excess
        elif pd.notna(g_excess):
            corp_credit_daily_us.loc[date] = g_excess
        else:
            corp_credit_daily_us.loc[date] = np.nan

print(f"✅ US Corporate Credit (daily blend):")
print(f"   Mean: {corp_credit_daily_us.mean():.4f}%")
print(f"   Pre-split obs: {len(corp_credit_daily_us[corp_credit_daily_us.index < split_date_us])}")
print(f"   Post-split obs: {len(corp_credit_daily_us[corp_credit_daily_us.index >= split_date_us])}")

# ============================================================================
# STEP 5: US EMERGING DEBT
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: US Emerging Debt")
print("=" * 80)

em_debt_excess_us = em_debt_us.copy()
print(f"✅ US Emerging Debt: mean={em_debt_excess_us.mean():.4f}%")

# ============================================================================
# STEP 6: US EMERGING CURRENCY (SWITCH J→I)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: US Emerging Currency")
print("=" * 80)

# Switch date: 30/06/2008
switch_date = pd.Timestamp('2008-06-30')

# Calcola returns da J e I
em_curr_j_ret_us = em_curr_j_us.pct_change() * 100
em_curr_i_ret_us = em_curr_i_us.pct_change() * 100

# Unisci le due serie di RETURNS
em_curr_ret_us = pd.Series(index=factors_us_raw.index, dtype=float)

for date in factors_us_raw.index:
    if date <= switch_date:
        em_curr_ret_us.loc[date] = em_curr_j_ret_us.loc[date]
    else:
        em_curr_ret_us.loc[date] = em_curr_i_ret_us.loc[date]

# Sottrai RF
em_curr_excess_us = em_curr_ret_us - rf_aligned_us

print(f"✅ US Emerging Currency:")
print(f"   Pre-switch: {len(em_curr_ret_us[em_curr_ret_us.index <= switch_date])} obs")
print(f"   Post-switch: {len(em_curr_ret_us[em_curr_ret_us.index > switch_date])} obs")
print(f"   Mean gross: {em_curr_ret_us.mean():.4f}%")
print(f"   Mean excess: {em_curr_excess_us.mean():.4f}%")

# ============================================================================
# STEP 7: US UST VOLATILITY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: US UST Implied Volatility")
print("=" * 80)

ust_vol_change_us = ust_vol_us.diff()
print(f"✅ US UST Vol Change: mean={ust_vol_change_us.mean():.4f}, std={ust_vol_change_us.std():.4f}")

# ============================================================================
# STEP 8: ASSEMBLA DATAFRAME US DAILY
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Assembla DataFrame US DAILY")
print("=" * 80)

factors_us_daily = pd.DataFrame({
    'US_Term': us_term_excess_us,
    'Global_Term': global_term_excess_us,
    'Global_Aggregate': global_agg_excess_us,
    'Inflation_Linkers': infl_link_excess_us,
    'Corporate_Credit': corp_credit_daily_us,
    'Emerging_Debt': em_debt_excess_us,
    'Emerging_Currency': em_curr_excess_us,
    'UST_Volatility': ust_vol_change_us
})

factors_us_daily = factors_us_daily.dropna()

print(f"✅ US Dataset daily: {len(factors_us_daily)} giorni")
print(f"📅 Periodo: {factors_us_daily.index.min()} to {factors_us_daily.index.max()}")

# ============================================================================
# PART B: EUR FACTORS
# ============================================================================

print("\n" + "=" * 80)
print("PART B: EUR FACTORS")
print("=" * 80)

# ============================================================================
# STEP 9: IMPORTA EUR ACTIVE FI FACTORS DA EXCEL
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Importa EUR Active FI Factors")
print("=" * 80)


# Inizializza variabile EUR factors
factors_eur_daily = None

try:
    factors_eur_raw = pd.read_excel(ACTIVE_FI_FILE, sheet_name="EUR_Factors", skiprows=17)
    
    print(f"📊 Shape iniziale: {factors_eur_raw.shape}")
    
    # Colonna A = Date
    date_col = factors_eur_raw.columns[0]
    factors_eur_raw[date_col] = pd.to_datetime(factors_eur_raw[date_col], format='%d/%m/%Y', errors='coerce')
    factors_eur_raw = factors_eur_raw.dropna(subset=[date_col])
    factors_eur_raw = factors_eur_raw.set_index(date_col)
    
    print(f"✅ Date parsate: {len(factors_eur_raw)} osservazioni")
    print(f"📅 Periodo: {factors_eur_raw.index.min()} to {factors_eur_raw.index.max()}")
    
    # Estrai colonne EUR
    eu_term_idx_eur = factors_eur_raw.iloc[:, 0]      # Col B
    global_term_idx_eur = factors_eur_raw.iloc[:, 1]  # Col C
    global_agg_idx_eur = factors_eur_raw.iloc[:, 2]   # Col D
    infl_link_idx_eur = factors_eur_raw.iloc[:, 3]    # Col E
    corp_credit_f_eur = factors_eur_raw.iloc[:, 4]    # Col F
    corp_credit_g_eur = factors_eur_raw.iloc[:, 5]    # Col G
    em_debt_eur = factors_eur_raw.iloc[:, 6]          # Col H
    em_curr_i_eur = factors_eur_raw.iloc[:, 7]        # Col I
    em_curr_j_eur = factors_eur_raw.iloc[:, 8]        # Col J
    ust_vol_eur = factors_eur_raw.iloc[:, 9]          # Col K
    
    print(f"✅ Colonne EUR estratte")
    
    # ====================================================================
    # STEP 10: IMPORTA RISK-FREE RATE EUR (EURIBOR)
    # ====================================================================

    print("\n" + "=" * 80)
    print("STEP 10: Importa Risk-Free Rate EUR (Euribor)")
    print("=" * 80)

    euribor_raw = pd.read_excel(ACTIVE_FI_FILE, sheet_name="Euribor1m", skiprows=7)
    euribor_raw.columns = ['Date', 'Euribor1m', 'Col_C', 'FX_USD_EUR']
    euribor_raw['Date'] = pd.to_datetime(euribor_raw['Date'])
    euribor_data = euribor_raw.set_index('Date')[['Euribor1m']]
    
    # RF daily = Euribor1m / 360
    euribor_data['RF_EUR'] = euribor_data['Euribor1m'] / 360
    
    # Allinea RF con date Active FI EUR
    rf_aligned_eur = euribor_data['RF_EUR'].reindex(factors_eur_raw.index, method='ffill')
    
    print(f"✅ RF EUR importato: {len(rf_aligned_eur)} giorni")
    print(f"📊 RF EUR medio: {rf_aligned_eur.mean():.4f}%")
    
    # ================================================================
    # STEP 11-15: EUR FACTORS PROCESSING (identico a US)
    # ================================================================

    print("\n" + "=" * 80)
    print("STEP 11: Calcola EUR returns da Total Return Indices")
    print("=" * 80)

    # 1. EU Term
    eu_term_ret_eur = eu_term_idx_eur.pct_change() * 100
    eu_term_excess_eur = eu_term_ret_eur - rf_aligned_eur
    print(f"✅ EU Term: mean gross={eu_term_ret_eur.mean():.4f}%, mean excess={eu_term_excess_eur.mean():.4f}%")

    # 2. Global Term
    global_term_ret_eur = global_term_idx_eur.pct_change() * 100
    global_term_excess_eur = global_term_ret_eur - rf_aligned_eur
    print(f"✅ Global Term: mean gross={global_term_ret_eur.mean():.4f}%, mean excess={global_term_excess_eur.mean():.4f}%")

    # 3. Global Aggregate
    global_agg_ret_eur = global_agg_idx_eur.pct_change() * 100
    global_agg_excess_eur = global_agg_ret_eur - rf_aligned_eur
    print(f"✅ Global Aggregate: mean gross={global_agg_ret_eur.mean():.4f}%, mean excess={global_agg_excess_eur.mean():.4f}%")

    # 4. Inflation-Linkers
    infl_link_ret_eur = infl_link_idx_eur.pct_change() * 100
    infl_link_excess_eur = infl_link_ret_eur - rf_aligned_eur
    print(f"✅ Inflation-Linkers: mean gross={infl_link_ret_eur.mean():.4f}%, mean excess={infl_link_excess_eur.mean():.4f}%")

    # ================================================================
    # STEP 12: EUR CORPORATE CREDIT (HYBRID) - DAILY
    # ================================================================

    print("\n" + "=" * 80)
    print("STEP 12: EUR Corporate Credit (Hybrid) - DAILY")
    print("=" * 80)

    # Split date EUR: 02/05/2013
    split_date_eur = pd.Timestamp('2013-05-02')

    # Colonna F: Total Return Index → calc return (gross)
    corp_credit_f_ret_eur = corp_credit_f_eur.pct_change() * 100

    # Colonna G: Già excess return %
    corp_credit_g_excess_eur = corp_credit_g_eur.copy()

    print(f"📊 EUR Col F - Gross return: mean={corp_credit_f_ret_eur.mean():.4f}%")
    print(f"📊 EUR Col G - Excess return: mean={corp_credit_g_excess_eur.mean():.4f}%")

    # Costruisci serie ibrida daily
    corp_credit_daily_eur = pd.Series(index=factors_eur_raw.index, dtype=float)

    for date in factors_eur_raw.index:
        if date < split_date_eur:
            # Pre-2013: Solo G (già excess)
            corp_credit_daily_eur.loc[date] = corp_credit_g_excess_eur.loc[date]
        else:
            # Post-2013: 50-50 blend
            f_excess = corp_credit_f_ret_eur.loc[date] - rf_aligned_eur.loc[date]
            g_excess = corp_credit_g_excess_eur.loc[date]
            
            if pd.notna(f_excess) and pd.notna(g_excess):
                corp_credit_daily_eur.loc[date] = 0.5 * f_excess + 0.5 * g_excess
            elif pd.notna(g_excess):
                corp_credit_daily_eur.loc[date] = g_excess
            else:
                corp_credit_daily_eur.loc[date] = np.nan

    print(f"✅ EUR Corporate Credit (daily blend):")
    print(f"   Mean: {corp_credit_daily_eur.mean():.4f}%")
    print(f"   Pre-split obs: {len(corp_credit_daily_eur[corp_credit_daily_eur.index < split_date_eur])}")
    print(f"   Post-split obs: {len(corp_credit_daily_eur[corp_credit_daily_eur.index >= split_date_eur])}")

    # ================================================================
    # STEP 13: EUR EMERGING DEBT
    # ================================================================

    print("\n" + "=" * 80)
    print("STEP 13: EUR Emerging Debt")
    print("=" * 80)

    em_debt_excess_eur = em_debt_eur.copy()
    print(f"✅ EUR Emerging Debt: mean={em_debt_excess_eur.mean():.4f}%")

    # ================================================================
    # STEP 14: EUR EMERGING CURRENCY (SWITCH J→I)
    # ================================================================

    print("\n" + "=" * 80)
    print("STEP 14: EUR Emerging Currency")
    print("=" * 80)

    # Switch date: 30/06/2008 (stesso per EUR)
    # Calcola returns da J e I
    em_curr_j_ret_eur = em_curr_j_eur.pct_change() * 100
    em_curr_i_ret_eur = em_curr_i_eur.pct_change() * 100

    # Unisci le due serie di RETURNS
    em_curr_ret_eur = pd.Series(index=factors_eur_raw.index, dtype=float)

    for date in factors_eur_raw.index:
        if date <= switch_date:
            em_curr_ret_eur.loc[date] = em_curr_j_ret_eur.loc[date]
        else:
            em_curr_ret_eur.loc[date] = em_curr_i_ret_eur.loc[date]

    # Sottrai RF EUR
    em_curr_excess_eur = em_curr_ret_eur - rf_aligned_eur

    print(f"✅ EUR Emerging Currency:")
    print(f"   Pre-switch: {len(em_curr_ret_eur[em_curr_ret_eur.index <= switch_date])} obs")
    print(f"   Post-switch: {len(em_curr_ret_eur[em_curr_ret_eur.index > switch_date])} obs")
    print(f"   Mean gross: {em_curr_ret_eur.mean():.4f}%")
    print(f"   Mean excess: {em_curr_excess_eur.mean():.4f}%")

    # ================================================================
    # STEP 15: EUR UST VOLATILITY
    # ================================================================

    print("\n" + "=" * 80)
    print("STEP 15: EUR UST Implied Volatility")
    print("=" * 80)

    ust_vol_change_eur = ust_vol_eur.diff()
    print(f"✅ EUR UST Vol Change: mean={ust_vol_change_eur.mean():.4f}, std={ust_vol_change_eur.std():.4f}")

    # ================================================================
    # STEP 16: ASSEMBLA DATAFRAME EUR DAILY
    # ================================================================

    print("\n" + "=" * 80)
    print("STEP 16: Assembla DataFrame EUR DAILY")
    print("=" * 80)

    factors_eur_daily = pd.DataFrame({
        'EU_Term': eu_term_excess_eur,
        'Global_Term': global_term_excess_eur,
        'Global_Aggregate': global_agg_excess_eur,
        'Inflation_Linkers': infl_link_excess_eur,
        'Corporate_Credit': corp_credit_daily_eur,
        'Emerging_Debt': em_debt_excess_eur,
        'Emerging_Currency': em_curr_excess_eur,
        'UST_Volatility': ust_vol_change_eur
    })

    factors_eur_daily = factors_eur_daily.dropna()

    print(f"✅ EUR Dataset daily: {len(factors_eur_daily)} giorni")
    print(f"📅 Periodo: {factors_eur_daily.index.min()} to {factors_eur_daily.index.max()}")
    
except Exception as e:
    print(f"❌ ERRORE nel processing EUR: {e}")
    import traceback
    traceback.print_exc()
    factors_eur_daily = None

# ============================================================================
# STEP 17: RESAMPLE US FACTORS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 17: Resample US Factors")
print("=" * 80)

if FACTOR_FREQ == "daily":
    factors_us_final = factors_us_daily.copy()
    print("✅ Frequenza US: daily")
    
elif FACTOR_FREQ in ["weekly", "monthly"]:
    
    freq_label = 'W-FRI' if FACTOR_FREQ == "weekly" else 'M'
    
    print(f"🔧 Applicando FIX FINALE per frequenza {FACTOR_FREQ}...")
    print(f"   - US/Global/Infl/EmCurr: compound gross → excess via RATIO")
    print(f"   - Corporate Credit: compound corp_credit_daily direttamente")
    print(f"   - Emerging Debt: compound diretto (già excess)")
    print(f"   - UST Volatility: somma differenze")
    print()
    
    # Compound RF
    rf_aligned_filtered_us = rf_aligned_us.loc[factors_us_daily.index]
    rf_compound_us = rf_aligned_filtered_us.resample(freq_label).apply(
        lambda x: ((1 + x/100).prod() - 1) * 100
    )
    
    print(f"✅ RF US compound: {len(rf_compound_us)} osservazioni")
    
    # --- PARTE 1: Simple TRI factors (US/Global/Infl/EmCurr) ---
    us_term_gross_us = us_term_excess_us + rf_aligned_us
    global_term_gross_us = global_term_excess_us + rf_aligned_us
    global_agg_gross_us = global_agg_excess_us + rf_aligned_us
    infl_link_gross_us = infl_link_excess_us + rf_aligned_us
    em_curr_gross_us = em_curr_excess_us + rf_aligned_us
    
    simple_tri_gross_us = pd.DataFrame({
        'US_Term': us_term_gross_us,
        'Global_Term': global_term_gross_us,
        'Global_Aggregate': global_agg_gross_us,
        'Inflation_Linkers': infl_link_gross_us,
        'Emerging_Currency': em_curr_gross_us
    })
    
    simple_tri_gross_aligned_us = simple_tri_gross_us.loc[factors_us_daily.index]
    
    # Compound gross
    simple_tri_compound_gross_us = simple_tri_gross_aligned_us.resample(freq_label).apply(
        lambda x: ((1 + x/100).prod() - 1) * 100
    )
    
    # Excess via RATIO
    simple_tri_excess_us = pd.DataFrame(index=simple_tri_compound_gross_us.index)
    
    for col in simple_tri_compound_gross_us.columns:
        simple_tri_excess_us[col] = ((1 + simple_tri_compound_gross_us[col]/100) / 
                                      (1 + rf_compound_us/100) - 1) * 100
    
    print(f"✅ US Simple TRI factors (excess via ratio)")
    
    # --- PARTE 2: Corporate Credit - Compound direttamente ---
    corp_credit_aligned_us = corp_credit_daily_us.loc[factors_us_daily.index]
    corp_credit_compound_us = corp_credit_aligned_us.resample(freq_label).apply(
        lambda x: ((1 + x/100).prod() - 1) * 100
    )
    corp_credit_compound_df_us = pd.DataFrame({'Corporate_Credit': corp_credit_compound_us})
    
    print(f"✅ US Corporate Credit compound (diretto)")
    
    # --- PARTE 3: Emerging Debt - Compound diretto ---
    em_debt_aligned_us = em_debt_excess_us.loc[factors_us_daily.index]
    em_debt_compound_us = em_debt_aligned_us.resample(freq_label).apply(
        lambda x: ((1 + x/100).prod() - 1) * 100
    )
    em_debt_compound_df_us = pd.DataFrame({'Emerging_Debt': em_debt_compound_us})
    
    print(f"✅ US Emerging Debt compound")
    
    # --- PARTE 4: UST Volatility - Somma differenze ---
    vol_aligned_us = ust_vol_change_us.loc[factors_us_daily.index]
    vol_compound_us = vol_aligned_us.resample(freq_label).sum()
    vol_compound_df_us = pd.DataFrame({'UST_Volatility': vol_compound_us})
    
    print(f"✅ US UST Volatility summed")
    
    # --- PARTE 5: Merge tutto ---
    factors_us_final = pd.concat([
        simple_tri_excess_us,
        corp_credit_compound_df_us,
        em_debt_compound_df_us,
        vol_compound_df_us
    ], axis=1)
    
    print(f"✅ Frequenza US: {FACTOR_FREQ}")
    print(f"✅ US Tutti i fattori processati correttamente")

print(f"📊 US Osservazioni: {len(factors_us_final)}")

# Riordina colonne
factors_us_final = factors_us_final[[
    'US_Term', 'Global_Term', 'Global_Aggregate', 'Inflation_Linkers',
    'Corporate_Credit', 'Emerging_Debt', 'Emerging_Currency', 'UST_Volatility'
]]

# ============================================================================
# STEP 18: RESAMPLE EUR FACTORS
# ============================================================================

if factors_eur_daily is not None:
    print("\n" + "=" * 80)
    print("STEP 18: Resample EUR Factors")
    print("=" * 80)

    if FACTOR_FREQ == "daily":
        factors_eur_final = factors_eur_daily.copy()
        print("✅ Frequenza EUR: daily")
        
    elif FACTOR_FREQ in ["weekly", "monthly"]:
        
        freq_label = 'W-FRI' if FACTOR_FREQ == "weekly" else 'M'
        
        print(f"🔧 Applicando FIX FINALE per frequenza {FACTOR_FREQ}...")
        
        # Compound RF EUR
        rf_aligned_filtered_eur = rf_aligned_eur.loc[factors_eur_daily.index]
        rf_compound_eur = rf_aligned_filtered_eur.resample(freq_label).apply(
            lambda x: ((1 + x/100).prod() - 1) * 100
        )
        
        print(f"✅ RF EUR compound: {len(rf_compound_eur)} osservazioni")
        
        # --- PARTE 1: Simple TRI factors (EU/Global/Infl/EmCurr) ---
        eu_term_gross_eur = eu_term_excess_eur + rf_aligned_eur
        global_term_gross_eur = global_term_excess_eur + rf_aligned_eur
        global_agg_gross_eur = global_agg_excess_eur + rf_aligned_eur
        infl_link_gross_eur = infl_link_excess_eur + rf_aligned_eur
        em_curr_gross_eur = em_curr_excess_eur + rf_aligned_eur
        
        simple_tri_gross_eur = pd.DataFrame({
            'EU_Term': eu_term_gross_eur,
            'Global_Term': global_term_gross_eur,
            'Global_Aggregate': global_agg_gross_eur,
            'Inflation_Linkers': infl_link_gross_eur,
            'Emerging_Currency': em_curr_gross_eur
        })
        
        simple_tri_gross_aligned_eur = simple_tri_gross_eur.loc[factors_eur_daily.index]
        
        # Compound gross
        simple_tri_compound_gross_eur = simple_tri_gross_aligned_eur.resample(freq_label).apply(
            lambda x: ((1 + x/100).prod() - 1) * 100
        )
        
        # Excess via RATIO
        simple_tri_excess_eur = pd.DataFrame(index=simple_tri_compound_gross_eur.index)
        
        for col in simple_tri_compound_gross_eur.columns:
            simple_tri_excess_eur[col] = ((1 + simple_tri_compound_gross_eur[col]/100) / 
                                          (1 + rf_compound_eur/100) - 1) * 100
        
        print(f"✅ EUR Simple TRI factors (excess via ratio)")
        
        # --- PARTE 2: Corporate Credit - Compound direttamente ---
        corp_credit_aligned_eur = corp_credit_daily_eur.loc[factors_eur_daily.index]
        corp_credit_compound_eur = corp_credit_aligned_eur.resample(freq_label).apply(
            lambda x: ((1 + x/100).prod() - 1) * 100
        )
        corp_credit_compound_df_eur = pd.DataFrame({'Corporate_Credit': corp_credit_compound_eur})
        
        print(f"✅ EUR Corporate Credit compound (diretto)")
        
        # --- PARTE 3: Emerging Debt - Compound diretto ---
        em_debt_aligned_eur = em_debt_excess_eur.loc[factors_eur_daily.index]
        em_debt_compound_eur = em_debt_aligned_eur.resample(freq_label).apply(
            lambda x: ((1 + x/100).prod() - 1) * 100
        )
        em_debt_compound_df_eur = pd.DataFrame({'Emerging_Debt': em_debt_compound_eur})
        
        print(f"✅ EUR Emerging Debt compound")
        
        # --- PARTE 4: UST Volatility - Somma differenze ---
        vol_aligned_eur = ust_vol_change_eur.loc[factors_eur_daily.index]
        vol_compound_eur = vol_aligned_eur.resample(freq_label).sum()
        vol_compound_df_eur = pd.DataFrame({'UST_Volatility': vol_compound_eur})
        
        print(f"✅ EUR UST Volatility summed")
        
        # --- PARTE 5: Merge tutto ---
        factors_eur_final = pd.concat([
            simple_tri_excess_eur,
            corp_credit_compound_df_eur,
            em_debt_compound_df_eur,
            vol_compound_df_eur
        ], axis=1)
        
        print(f"✅ Frequenza EUR: {FACTOR_FREQ}")
        print(f"✅ EUR Tutti i fattori processati correttamente")

    print(f"📊 EUR Osservazioni: {len(factors_eur_final)}")

    # Riordina colonne
    factors_eur_final = factors_eur_final[[
        'EU_Term', 'Global_Term', 'Global_Aggregate', 'Inflation_Linkers',
        'Corporate_Credit', 'Emerging_Debt', 'Emerging_Currency', 'UST_Volatility'
    ]]
else:
    factors_eur_final = None
    print("\n⚠️  EUR factors non disponibili")

# ============================================================================
# STEP 19: SALVA FATTORI GLOBALI
# ============================================================================

print("\n" + "=" * 80)
print("STEP 19: Salva fattori Active FI globali")
print("=" * 80)

# Salva US factors
factors_us_path = PROCESSED_DATA_DIR / f"active_fi_factors_us_{FACTOR_FREQ}.csv"
factors_us_final.to_csv(factors_us_path)

print(f"💾 US Salvato: {factors_us_path.name}")
print(f"📊 US Fattori: {list(factors_us_final.columns)}")

# Salva EUR factors (se disponibile)
if factors_eur_final is not None:
    factors_eur_path = PROCESSED_DATA_DIR / f"active_fi_factors_eur_{FACTOR_FREQ}.csv"
    factors_eur_final.to_csv(factors_eur_path)
    
    print(f"\n💾 EUR Salvato: {factors_eur_path.name}")
    print(f"📊 EUR Fattori: {list(factors_eur_final.columns)}")

# ============================================================================
# STEP 20: MERGE CON TUTTE LE STRATEGIE (LOOP)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 20: Merge con TUTTE le strategie")
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
        
        regression_us_path = PROCESSED_DATA_DIR / f"regression_data_active_fi_{strategy_name}_us_{FACTOR_FREQ}.csv"
        regression_us_data.to_csv(regression_us_path)
        print(f"💾 Salvato: {regression_us_path.name}")
        
        regression_files_created['US'].append(regression_us_path.name)
        
        # Merge EUR (se disponibile)
        if factors_eur_final is not None:
            regression_eur_data = strategy.join(factors_eur_final, how='inner')
            regression_eur_data = regression_eur_data.dropna()
            
            print(f"✅ EUR regression data: {len(regression_eur_data)} osservazioni")
            
            regression_eur_path = PROCESSED_DATA_DIR / f"regression_data_active_fi_{strategy_name}_eur_{FACTOR_FREQ}.csv"
            regression_eur_data.to_csv(regression_eur_path)
            print(f"💾 Salvato: {regression_eur_path.name}")
            
            regression_files_created['EUR'].append(regression_eur_path.name)
            
    except Exception as e:
        print(f"❌ ERRORE processing {strategy_name}: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# STEP 21: STATISTICHE
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICHE FINALI")
print("=" * 80)

print("\n🇺🇸 US FACTORS:")
print("\n📊 Statistiche Descrittive:")
print(factors_us_final.describe().round(3))

print("\n📊 Correlazioni tra Fattori:")
print(factors_us_final.corr().round(3))

# Correlazioni con Strategy Return per OGNI strategia US
print("\n📊 Correlazioni con Strategy Return (US):")
for fname in regression_files_created['US']:
    # Estrai nome strategia dal filename
    strategy_name = fname.replace('regression_data_active_fi_', '').replace(f'_us_{FACTOR_FREQ}.csv', '')
    
    # Carica regression data
    reg_data_path = PROCESSED_DATA_DIR / fname
    if reg_data_path.exists():
        reg_data = pd.read_csv(reg_data_path, index_col=0, parse_dates=True)
        
        print(f"\n   {strategy_name}:")
        corr_with_strategy = reg_data.corr()['Strategy_Return'].drop('Strategy_Return')
        for factor, corr_val in corr_with_strategy.items():
            print(f"      {factor:20s}: {corr_val:7.3f}")

if factors_eur_final is not None:
    print("\n" + "=" * 80)
    print("🇪🇺 EUR FACTORS:")
    print("\n📊 Statistiche Descrittive:")
    print(factors_eur_final.describe().round(3))
    
    print("\n📊 Correlazioni tra Fattori:")
    print(factors_eur_final.corr().round(3))
    
    # Correlazioni con Strategy Return per OGNI strategia EUR
    if len(regression_files_created['EUR']) > 0:
        print("\n📊 Correlazioni con Strategy Return (EUR):")
        for fname in regression_files_created['EUR']:
            # Estrai nome strategia dal filename
            strategy_name = fname.replace('regression_data_active_fi_', '').replace(f'_eur_{FACTOR_FREQ}.csv', '')
            
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
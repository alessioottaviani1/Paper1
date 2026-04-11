"""
Script 2c: Run Factor Model Regressions - Fung & Hsieh (2004) - MULTI STRATEGY - US & EUR
==========================================================================================
Testa l'alpha di TUTTE LE STRATEGIE usando Fung & Hsieh factors.

Reference: Fung & Hsieh (2004) "Hedge Fund Benchmarks: A Risk-Based Approach"
           Financial Analysts Journal

STRATEGIE:
1. BTP-Italia
2. iTraxx Main
3. iTraxx SnrFin
4. iTraxx SubFin
5. iTraxx Xover
6. iTraxx Combined
7. CDS-Bond Basis

OUTPUT:
- 2 FILE .tex SOLO (identici a struttura Duarte e Active FI):
  1. FungHsieh_summary_combined_<freq>.tex (3 tabelle)
  2. FungHsieh_summary_separated_<freq>.tex (per strategia: Alpha + Full + VIF)

LOGICA DUARTE:
- Lista fissa di NOMI di fattori (nomi reali nei CSV)
- File US e EUR separati (SNP in US = S&P 500, SNP in EUR = EuroStoxx)
- Tabelle ciclano su lista fissa

Author: Alessio Ottaviani
Date: December 2025
Institution: EDHEC Business School - PhD Thesis
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETRI
# ============================================================================

REGRESSION_FREQ = "monthly"
HAC_LAGS = None
ALPHA_LEVEL = 0.05
INCLUDE_COMBINED_IN_SUMMARY = True

# ============================================================================
# STRATEGIE
# ============================================================================

STRATEGIES = [
    'BTP_Italia',
    'iTraxx_Main',
    'iTraxx_SnrFin',
    'iTraxx_SubFin',
    'iTraxx_Xover',
    'iTraxx_Combined',
    'CDS_Bond_Basis'
]

# ============================================================================
# FATTORI FUNG & HSIEH (FISSI) - NOMI REALI NEI CSV
# ============================================================================

# Lista fissa dei 7 fattori Fung & Hsieh (nomi REALI nei CSV)
# Nota: SNP, SIZE, TERM, CREDIT sono diversi tra US e EUR ma hanno stesso nome
# perché sono in file separati (data_us vs data_eur)
FUNG_HSIEH_FACTORS = [
    'SNP',      # S&P 500 (US) / EuroStoxx (EUR)
    'SIZE',     # SC-LC (US) / Small-Large (EUR)
    'PTFSBD',   # Bond trend-following (uguale US/EUR)
    'PTFSFX',   # FX trend-following (uguale US/EUR)
    'PTFSCOM',  # Commodity trend-following (uguale US/EUR)
    'TERM',     # Change 10Y (US) / Change 10Y Bund (EUR)
    'CREDIT'    # Change Credit Spread (US) / EUR equivalent
]

# Mapping per LaTeX (nomi display nelle tabelle)
FACTOR_NAMES_LATEX = {
    'SNP': 'S\\&P',
    'SIZE': 'SC-LC',
    'PTFSBD': 'BdOpt',
    'PTFSFX': 'FXOpt',
    'PTFSCOM': 'ComOpt',
    'TERM': '10Y',
    'CREDIT': 'CredSpr'
}


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STORAGE
# ============================================================================

all_results_us = []
all_results_eur = []
vif_storage = {}

# ============================================================================
# LOOP SU TUTTE LE STRATEGIE
# ============================================================================

for strategy_name in STRATEGIES:
    
    print("\n" + "=" * 80)
    print(f"STRATEGIA: {strategy_name}")
    print("=" * 80)
    
    strategy_lower = strategy_name.lower()
    
    # ========================================================================
    # STEP 1: CARICA DATI
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(f"STEP 1: Caricamento dati - {strategy_name}")
    print("=" * 80)
    
    print(f"\n📊 Frequenza: {REGRESSION_FREQ.upper()}")
    
    data_us_path = PROCESSED_DATA_DIR / f"regression_data_fung_hsieh_{strategy_lower}_us_{REGRESSION_FREQ}.csv"
    
    if not data_us_path.exists():
        print(f"\n❌ File US non trovato: {data_us_path.name}")
        print(f"   Skipping {strategy_name}...")
        continue
    
    data_us = pd.read_csv(data_us_path, index_col=0, parse_dates=True)
    
    print(f"\n🇺🇸 DATASET US:")
    print(f"✅ Dataset caricato: {len(data_us)} osservazioni")
    print(f"📅 Periodo: {data_us.index.min().strftime('%Y-%m-%d')} to {data_us.index.max().strftime('%Y-%m-%d')}")
    print(f"📊 Colonne: {list(data_us.columns)}")
    
    data_us = data_us.dropna()
    print(f"✅ Dopo pulizia: {len(data_us)} osservazioni")
    
    data_eur_path = PROCESSED_DATA_DIR / f"regression_data_fung_hsieh_{strategy_lower}_eur_{REGRESSION_FREQ}.csv"
    
    if not data_eur_path.exists():
        print(f"\n❌ File EUR non trovato: {data_eur_path.name}")
        print(f"   Skipping EUR per {strategy_name}...")
        data_eur = None
    else:
        data_eur = pd.read_csv(data_eur_path, index_col=0, parse_dates=True)
        
        print(f"\n🇪🇺 DATASET EUR:")
        print(f"✅ Dataset caricato: {len(data_eur)} osservazioni")
        print(f"📅 Periodo: {data_eur.index.min().strftime('%Y-%m-%d')} to {data_eur.index.max().strftime('%Y-%m-%d')}")
        print(f"📊 Colonne: {list(data_eur.columns)}")
        
        data_eur = data_eur.dropna()
        print(f"✅ Dopo pulizia: {len(data_eur)} osservazioni")
    
    # ========================================================================
    # STEP 2: DEFINISCI FATTORI (verifica disponibilità)
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(f"STEP 2: Verifica fattori disponibili - {strategy_name}")
    print("=" * 80)
    
    available_us = [f for f in FUNG_HSIEH_FACTORS if f in data_us.columns]
    
    print(f"\n🇺🇸 US FACTORS disponibili:")
    print(f"✅ {len(available_us)}/{len(FUNG_HSIEH_FACTORS)} fattori: {available_us}")
    
    if len(available_us) < len(FUNG_HSIEH_FACTORS):
        missing_us = [f for f in FUNG_HSIEH_FACTORS if f not in data_us.columns]
        print(f"⚠️  Fattori mancanti US: {missing_us}")
    
    if data_eur is not None:
        available_eur = [f for f in FUNG_HSIEH_FACTORS if f in data_eur.columns]
        
        print(f"\n🇪🇺 EUR FACTORS disponibili:")
        print(f"✅ {len(available_eur)}/{len(FUNG_HSIEH_FACTORS)} fattori: {available_eur}")
        
        if len(available_eur) < len(FUNG_HSIEH_FACTORS):
            missing_eur = [f for f in FUNG_HSIEH_FACTORS if f not in data_eur.columns]
            print(f"⚠️  Fattori mancanti EUR: {missing_eur}")
    
    # ========================================================================
    # STEP 3: CALCOLA HAC LAGS
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(f"STEP 3: Calcola Newey-West HAC lags - {strategy_name}")
    print("=" * 80)
    
    if HAC_LAGS is None:
        T_us = len(data_us)
        
        if REGRESSION_FREQ == "daily":
            HAC_LAGS_US = int(np.sqrt(T_us))
        elif REGRESSION_FREQ == "weekly":
            HAC_LAGS_US = int(T_us**(1/3))
        elif REGRESSION_FREQ == "monthly":
            HAC_LAGS_US = int(T_us**(1/4))
        
        print(f"🇺🇸 US: Osservazioni = {T_us}, HAC lags (auto) = {HAC_LAGS_US}")
        
        if data_eur is not None:
            T_eur = len(data_eur)
            
            if REGRESSION_FREQ == "daily":
                HAC_LAGS_EUR = int(np.sqrt(T_eur))
            elif REGRESSION_FREQ == "weekly":
                HAC_LAGS_EUR = int(T_eur**(1/3))
            elif REGRESSION_FREQ == "monthly":
                HAC_LAGS_EUR = int(T_eur**(1/4))
            
            print(f"🇪🇺 EUR: Osservazioni = {T_eur}, HAC lags (auto) = {HAC_LAGS_EUR}")
    else:
        HAC_LAGS_US = HAC_LAGS
        HAC_LAGS_EUR = HAC_LAGS
        print(f"📊 HAC lags (manual): {HAC_LAGS}")
    
    # ========================================================================
    # STEP 4A: REGRESSIONI US
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(f"STEP 4A: Stima Full Model US con Newey-West HAC - {strategy_name}")
    print("=" * 80)
    
    print(f"\n{'='*60}")
    print(f"📊 Full Model US - {len(available_us)} Fung & Hsieh Factors")
    print(f"{'='*60}")
    
    y_us = data_us['Strategy_Return']
    X_us = data_us[available_us].copy()
    X_us = sm.add_constant(X_us)
    
    model_us = sm.OLS(y_us, X_us)
    result_us = model_us.fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS_US})
    
    alpha_us = result_us.params['const']
    alpha_tstat_us = result_us.tvalues['const']
    alpha_pval_us = result_us.pvalues['const']
    
    if REGRESSION_FREQ == "daily":
        alpha_annual_us = alpha_us * 252
    elif REGRESSION_FREQ == "weekly":
        alpha_annual_us = alpha_us * 52
    elif REGRESSION_FREQ == "monthly":
        alpha_annual_us = alpha_us * 12
    
    rsq_us = result_us.rsquared
    rsq_adj_us = result_us.rsquared_adj
    nobs_us = result_us.nobs
    dw_us = sm.stats.stattools.durbin_watson(result_us.resid)
    result_us_ols = model_us.fit()
    fstat_us = result_us_ols.fvalue
    fpval_us = result_us_ols.f_pvalue
    
    print(f"\n🎯 ALPHA US:")
    print(f"   {REGRESSION_FREQ.capitalize()}: {alpha_us:.4f}%")
    print(f"   Annualizzato: {alpha_annual_us:.4f}%")
    print(f"   t-stat: {alpha_tstat_us:.4f}")
    print(f"   p-value: {alpha_pval_us:.4f} {'***' if alpha_pval_us < 0.01 else '**' if alpha_pval_us < 0.05 else '*' if alpha_pval_us < 0.10 else ''}")
    
    print(f"\n📊 Modello US:")
    print(f"   R²: {rsq_us:.4f}")
    print(f"   R² adj: {rsq_adj_us:.4f}")
    print(f"   N obs: {int(nobs_us)}")
    
    print(f"\n📈 Betas significativi US (p < 0.10):")
    sig_betas_us = result_us.pvalues[result_us.pvalues < 0.10]
    sig_betas_us = sig_betas_us[sig_betas_us.index != 'const']
    
    if len(sig_betas_us) > 0:
        for factor in sig_betas_us.index:
            beta = result_us.params[factor]
            tstat = result_us.tvalues[factor]
            pval = result_us.pvalues[factor]
            stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*'
            print(f"   {factor}: {beta:.4f} (t={tstat:.2f}){stars}")
    else:
        print(f"   Nessun beta significativo oltre l'alpha")
    
    # ====================================================================
    # SALVA RISULTATI US (usa lista fissa per consistenza)
    # ====================================================================
    
    result_row_us = {
        'Strategy': strategy_name,
        'Region': 'US',
        'Alpha_period': alpha_us,
        'Alpha_annual': alpha_annual_us,
        't_stat': alpha_tstat_us,
        'p_value': alpha_pval_us,
        'Significance': '***' if alpha_pval_us < 0.01 else '**' if alpha_pval_us < 0.05 else '*' if alpha_pval_us < 0.10 else '',
        'R_squared': rsq_us,
        'R_squared_adj': rsq_adj_us,
        'N_obs': int(nobs_us),
        'DW': dw_us,
        'F_stat': fstat_us,
        'F_pval': fpval_us
    }
    
    # SALVA BETA dalla lista fissa (potrebbero essere NaN se non disponibili)
    for factor in FUNG_HSIEH_FACTORS:
        if factor in result_us.params.index:
            result_row_us[f'Beta_{factor}'] = result_us.params[factor]
            result_row_us[f't_{factor}'] = result_us.tvalues[factor]
        else:
            result_row_us[f'Beta_{factor}'] = np.nan
            result_row_us[f't_{factor}'] = np.nan
    
    all_results_us.append(result_row_us)
    
    # ========================================================================
    # STEP 4B: REGRESSIONI EUR
    # ========================================================================
    
    if data_eur is not None:
        
        print("\n" + "=" * 80)
        print(f"STEP 4B: Stima Full Model EUR con Newey-West HAC - {strategy_name}")
        print("=" * 80)
        
        print(f"\n{'='*60}")
        print(f"📊 Full Model EUR - {len(available_eur)} Fung & Hsieh Factors")
        print(f"{'='*60}")
        
        y_eur = data_eur['Strategy_Return']
        X_eur = data_eur[available_eur].copy()
        X_eur = sm.add_constant(X_eur)
        
        model_eur = sm.OLS(y_eur, X_eur)
        result_eur = model_eur.fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS_EUR})
        
        alpha_eur = result_eur.params['const']
        alpha_tstat_eur = result_eur.tvalues['const']
        alpha_pval_eur = result_eur.pvalues['const']
        
        if REGRESSION_FREQ == "daily":
            alpha_annual_eur = alpha_eur * 252
        elif REGRESSION_FREQ == "weekly":
            alpha_annual_eur = alpha_eur * 52
        elif REGRESSION_FREQ == "monthly":
            alpha_annual_eur = alpha_eur * 12
        
        rsq_eur = result_eur.rsquared
        rsq_adj_eur = result_eur.rsquared_adj
        nobs_eur = result_eur.nobs
        dw_eur = sm.stats.stattools.durbin_watson(result_eur.resid)
        result_eur_ols = model_eur.fit()
        fstat_eur = result_eur_ols.fvalue
        fpval_eur = result_eur_ols.f_pvalue
        
        print(f"\n🎯 ALPHA EUR:")
        print(f"   {REGRESSION_FREQ.capitalize()}: {alpha_eur:.4f}%")
        print(f"   Annualizzato: {alpha_annual_eur:.4f}%")
        print(f"   t-stat: {alpha_tstat_eur:.4f}")
        print(f"   p-value: {alpha_pval_eur:.4f} {'***' if alpha_pval_eur < 0.01 else '**' if alpha_pval_eur < 0.05 else '*' if alpha_pval_eur < 0.10 else ''}")
        
        print(f"\n📊 Modello EUR:")
        print(f"   R²: {rsq_eur:.4f}")
        print(f"   R² adj: {rsq_adj_eur:.4f}")
        print(f"   N obs: {int(nobs_eur)}")
        
        print(f"\n📈 Betas significativi EUR (p < 0.10):")
        sig_betas_eur = result_eur.pvalues[result_eur.pvalues < 0.10]
        sig_betas_eur = sig_betas_eur[sig_betas_eur.index != 'const']
        
        if len(sig_betas_eur) > 0:
            for factor in sig_betas_eur.index:
                beta = result_eur.params[factor]
                tstat = result_eur.tvalues[factor]
                pval = result_eur.pvalues[factor]
                stars = '***' if pval < 0.01 else '**' if pval < 0.05 else '*'
                print(f"   {factor}: {beta:.4f} (t={tstat:.2f}){stars}")
        else:
            print(f"   Nessun beta significativo oltre l'alpha")
        
        # ================================================================
        # SALVA RISULTATI EUR (usa lista fissa per consistenza)
        # ================================================================
        
        result_row_eur = {
            'Strategy': strategy_name,
            'Region': 'EUR',
            'Alpha_period': alpha_eur,
            'Alpha_annual': alpha_annual_eur,
            't_stat': alpha_tstat_eur,
            'p_value': alpha_pval_eur,
            'Significance': '***' if alpha_pval_eur < 0.01 else '**' if alpha_pval_eur < 0.05 else '*' if alpha_pval_eur < 0.10 else '',
            'R_squared': rsq_eur,
            'R_squared_adj': rsq_adj_eur,
            'N_obs': int(nobs_eur),
            'DW': dw_eur,
            'F_stat': fstat_eur,
            'F_pval': fpval_eur
        }
        
        # SALVA BETA dalla lista fissa (potrebbero essere NaN se non disponibili)
        for factor in FUNG_HSIEH_FACTORS:
            if factor in result_eur.params.index:
                result_row_eur[f'Beta_{factor}'] = result_eur.params[factor]
                result_row_eur[f't_{factor}'] = result_eur.tvalues[factor]
            else:
                result_row_eur[f'Beta_{factor}'] = np.nan
                result_row_eur[f't_{factor}'] = np.nan
        
        all_results_eur.append(result_row_eur)
    
    # ========================================================================
    # STEP 5: CONFRONTO
    # ========================================================================
    
    if data_eur is not None:
        
        print("\n" + "=" * 80)
        print(f"STEP 5: Confronto US vs EUR - {strategy_name}")
        print("=" * 80)
        
        comparison = [
            {
                'Model': 'Fung & Hsieh US',
                'Alpha_period': alpha_us,
                'Alpha_annual': alpha_annual_us,
                't-stat': alpha_tstat_us,
                'p-value': alpha_pval_us,
                'R²': rsq_us,
                'R² adj': rsq_adj_us,
                'N obs': int(nobs_us)
            },
            {
                'Model': 'Fung & Hsieh EUR',
                'Alpha_period': alpha_eur,
                'Alpha_annual': alpha_annual_eur,
                't-stat': alpha_tstat_eur,
                'p-value': alpha_pval_eur,
                'R²': rsq_eur,
                'R² adj': rsq_adj_eur,
                'N obs': int(nobs_eur)
            }
        ]
        
        comparison_df = pd.DataFrame(comparison)
        print(f"\n{comparison_df.to_string(index=False)}")
    
    # ========================================================================
    # STEP 6A: VIF US
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(f"STEP 6A: VIF Test US (Multicollinearità) - {strategy_name}")
    print("=" * 80)
    
    if len(available_us) > 1:
        X_vif_us = data_us[available_us].copy()
        
        vif_data_us = []
        for i, col in enumerate(X_vif_us.columns):
            vif = variance_inflation_factor(X_vif_us.values, i)
            vif_data_us.append({'Factor': col, 'VIF': vif})
        
        vif_df_us = pd.DataFrame(vif_data_us)
        vif_df_us = vif_df_us.sort_values('VIF', ascending=False)
        
        if strategy_name not in vif_storage:
            vif_storage[strategy_name] = {}
        vif_storage[strategy_name]['US'] = vif_df_us
        
        print(f"\n📊 VIF Test (Fung & Hsieh US):")
        print(vif_df_us.to_string(index=False))
        
        max_vif_us = vif_df_us['VIF'].max()
        if max_vif_us > 10:
            print(f"\n⚠️ VIF massimo: {max_vif_us:.2f} > 10 → PROBLEMA SERIO di multicollinearità!")
        elif max_vif_us > 5:
            print(f"\n⚠️ VIF massimo: {max_vif_us:.2f} > 5 → Multicollinearità moderata, monitorare")
        else:
            print(f"\n✅ VIF massimo: {max_vif_us:.2f} < 5 → Multicollinearità accettabile")
    
    # ========================================================================
    # STEP 6B: VIF EUR
    # ========================================================================
    
    if data_eur is not None:
        
        print("\n" + "=" * 80)
        print(f"STEP 6B: VIF Test EUR (Multicollinearità) - {strategy_name}")
        print("=" * 80)
        
        if len(available_eur) > 1:
            X_vif_eur = data_eur[available_eur].copy()
            
            vif_data_eur = []
            for i, col in enumerate(X_vif_eur.columns):
                vif = variance_inflation_factor(X_vif_eur.values, i)
                vif_data_eur.append({'Factor': col, 'VIF': vif})
            
            vif_df_eur = pd.DataFrame(vif_data_eur)
            vif_df_eur = vif_df_eur.sort_values('VIF', ascending=False)
            
            if strategy_name not in vif_storage:
                vif_storage[strategy_name] = {}
            vif_storage[strategy_name]['EUR'] = vif_df_eur
            
            print(f"\n📊 VIF Test (Fung & Hsieh EUR):")
            print(vif_df_eur.to_string(index=False))
            
            max_vif_eur = vif_df_eur['VIF'].max()
            if max_vif_eur > 10:
                print(f"\n⚠️ VIF massimo: {max_vif_eur:.2f} > 10 → PROBLEMA SERIO di multicollinearità!")
            elif max_vif_eur > 5:
                print(f"\n⚠️ VIF massimo: {max_vif_eur:.2f} > 5 → Multicollinearità moderata, monitorare")
            else:
                print(f"\n✅ VIF massimo: {max_vif_eur:.2f} < 5 → Multicollinearità accettabile")
    
    print(f"\n✅ Analisi completata per {strategy_name}")

# ============================================================================
# STEP 9: CREA TABELLA AGGREGATA
# ============================================================================

print("\n" + "=" * 80)
print("=" * 80)
print(f"STEP 9: Tabella aggregata - TUTTE LE STRATEGIE ({REGRESSION_FREQ.upper()})")
print("=" * 80)
print("=" * 80)

if all_results_us or all_results_eur:
    
    combined_results = []
    
    for result in all_results_us:
        if not INCLUDE_COMBINED_IN_SUMMARY and result['Strategy'] == 'iTraxx_Combined':
            continue
        combined_results.append(result)
    
    for result in all_results_eur:
        if not INCLUDE_COMBINED_IN_SUMMARY and result['Strategy'] == 'iTraxx_Combined':
            continue
        combined_results.append(result)
    
    if combined_results:
        
        summary_df = pd.DataFrame(combined_results)
        summary_df = summary_df.sort_values(['Strategy', 'Region'])
        
        print("\n" + "=" * 80)
        print("TABELLA AGGREGATA - ALPHA E BETA")
        print("=" * 80)
        
        print(f"\n{summary_df[['Strategy', 'Region', 'Alpha_annual', 't_stat', 'Significance', 'R_squared_adj', 'N_obs']].to_string(index=False)}")
        
        # ================================================================
        # STEP 9.1: FILE COMBINED
        # ================================================================
        
        print("\n" + "=" * 80)
        print("STEP 9.1: Genera file LaTeX combined")
        print("=" * 80)

        latex_combined_path = TABLES_DIR / f"FungHsieh_summary_combined_{REGRESSION_FREQ}.tex"
        # NUOVO PERCORSO PER LA SLIDE SINGOLA
        latex_presentation_path = TABLES_DIR / f"FungHsieh_Presentation_Slide_{REGRESSION_FREQ}.tex"

        strategies_list = summary_df['Strategy'].unique()
        
        # USA LA LISTA FISSA
        all_factors = FUNG_HSIEH_FACTORS.copy()
        
        with open(latex_combined_path, 'w') as f:
            
            f.write(f"% {'='*76}\n")
            f.write(f"% FUNG & HSIEH FACTOR MODEL - ALL STRATEGIES COMBINED\n")
            f.write(f"% Reference: Fung & Hsieh (2004)\n")
            f.write(f"% Financial Analysts Journal\n")
            f.write(f"% Frequency: {REGRESSION_FREQ.capitalize()}\n")
            f.write(f"% IMPORTANT: Add to LaTeX preamble:\n")
            f.write(f"% \\usepackage{{booktabs}}\n")
            f.write(f"% \\usepackage{{threeparttable}}\n")
            f.write(f"% {'='*76}\n\n")
            
            # ============================================================
            # TABLE 1: ALPHA COMPARISON
            # ============================================================
            
            f.write(f"% {'='*76}\n")
            f.write(f"% TABLE 1: ALPHA COMPARISON\n")
            f.write(f"% {'='*76}\n\n")
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Fung \\& Hsieh Factor Model - Alpha Comparison ({REGRESSION_FREQ.capitalize()} Frequency)}}\n")
            f.write(f"\\label{{tab:funghsieh_alpha_comparison_{REGRESSION_FREQ}}}\n")
            f.write("\\begin{threeparttable}\n")
            f.write("\\begin{tabular}{lcccccc}\n")
            f.write("\\toprule\n")
            f.write(" & \\multicolumn{3}{c}{US Factors} & \\multicolumn{3}{c}{EUR Factors} \\\\\n")
            f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n")
            f.write(" & $\\alpha$ (\\%) & $R^2$ adj & N & $\\alpha$ (\\%) & $R^2$ adj & N \\\\\n")
            f.write("\\midrule\n")
            
            for strategy in strategies_list:
                us_data = summary_df[(summary_df['Strategy'] == strategy) & (summary_df['Region'] == 'US')]
                eur_data = summary_df[(summary_df['Strategy'] == strategy) & (summary_df['Region'] == 'EUR')]
                
                strategy_display = strategy.replace('_', ' ')
                f.write(f"\\textit{{{strategy_display}}} ")
                
                if len(us_data) > 0:
                    us_row = us_data.iloc[0]
                    alpha_us = us_row['Alpha_annual']
                    tstat_us = us_row['t_stat']
                    if abs(tstat_us) > 2.576:
                        sig = '***'
                    elif abs(tstat_us) > 1.96:
                        sig = '**'
                    elif abs(tstat_us) > 1.645:
                        sig = '*'
                    else:
                        sig = ''
                    f.write(f"& {alpha_us:.2f}{sig} & {us_row['R_squared_adj']:.3f} & {int(us_row['N_obs'])} ")
                else:
                    f.write("& -- & -- & -- ")
                
                if len(eur_data) > 0:
                    eur_row = eur_data.iloc[0]
                    alpha_eur = eur_row['Alpha_annual']
                    tstat_eur = eur_row['t_stat']
                    if abs(tstat_eur) > 2.576:
                        sig = '***'
                    elif abs(tstat_eur) > 1.96:
                        sig = '**'
                    elif abs(tstat_eur) > 1.645:
                        sig = '*'
                    else:
                        sig = ''
                    f.write(f"& {alpha_eur:.2f}{sig} & {eur_row['R_squared_adj']:.3f} & {int(eur_row['N_obs'])} ")
                else:
                    f.write("& -- & -- & -- ")
                
                f.write("\\\\\n")
                
                f.write(" ")
                if len(us_data) > 0:
                    f.write(f"& ({tstat_us:.2f}) & & ")
                else:
                    f.write("& & & ")
                
                if len(eur_data) > 0:
                    f.write(f"& ({tstat_eur:.2f}) & & ")
                else:
                    f.write("& & & ")
                
                f.write("\\\\\n")
                f.write("\\addlinespace\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\begin{tablenotes}\n")
            f.write("\\footnotesize\n")
            f.write("\\item Alpha is annualized monthly excess return. t-statistics in parentheses based on Newey-West HAC standard errors. *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{threeparttable}\n")
            f.write("\\end{table}\n\n")
            f.write("\\clearpage\n\n")
            
            # ============================================================
            # TABLE 2: FULL MODEL RESULTS
            # ============================================================
            
            f.write(f"% {'='*76}\n")
            f.write(f"% TABLE 2: FULL MODEL RESULTS\n")
            f.write(f"% {'='*76}\n\n")
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Fung \\& Hsieh Factor Model - Full Model Results ({REGRESSION_FREQ.capitalize()} Frequency)}}\n")
            f.write(f"\\label{{tab:funghsieh_full_results_{REGRESSION_FREQ}}}\n")
            f.write("\\footnotesize\n")
            f.write("\\begin{threeparttable}\n")
            
            ncols = 1 + len(all_factors) + 2
            f.write(f"\\begin{{tabular}}{{l{'c'*ncols}}}\n")
            f.write("\\toprule\n")
            
            # Header
            f.write(" & $\\alpha$ (\\%) ")
            for factor in all_factors:
                latex_name = FACTOR_NAMES_LATEX.get(factor, factor)
                f.write(f"& $\\beta_{{{latex_name}}}$ ")
            f.write("& $R^2$ adj & N \\\\\n")
            f.write("\\midrule\n")
            
            # Panel A: US
            f.write("\\multicolumn{" + str(ncols+1) + "}{l}{\\textbf{Panel A: US Factors}} \\\\\n")
            f.write("\\addlinespace\n")
            
            for strategy in strategies_list:
                us_data = summary_df[(summary_df['Strategy'] == strategy) & (summary_df['Region'] == 'US')]
                
                if len(us_data) == 0:
                    continue
                
                us_row = us_data.iloc[0]
                strategy_display = strategy.replace('_', ' ')
                
                f.write(f"\\textit{{{strategy_display}}}")
                
                # Alpha
                alpha = us_row['Alpha_annual']
                alpha_tstat = us_row['t_stat']
                if abs(alpha_tstat) > 2.576:
                    sig = '***'
                elif abs(alpha_tstat) > 1.96:
                    sig = '**'
                elif abs(alpha_tstat) > 1.645:
                    sig = '*'
                else:
                    sig = ''
                f.write(f" & {alpha:.2f}{sig}")
                
                # All factors
                for factor in all_factors:
                    beta = us_row[f'Beta_{factor}']
                    tstat = us_row[f't_{factor}']
                    
                    if pd.notna(beta):
                        if abs(tstat) > 2.576:
                            sig = '***'
                        elif abs(tstat) > 1.96:
                            sig = '**'
                        elif abs(tstat) > 1.645:
                            sig = '*'
                        else:
                            sig = ''
                        f.write(f" & {beta:.3f}{sig}")
                    else:
                        f.write(" & --")
                
                f.write(f" & {us_row['R_squared_adj']:.3f} & {int(us_row['N_obs'])}")
                f.write(" \\\\\n")
                
                # t-stats row
                f.write(" ")
                f.write(f" & ({alpha_tstat:.2f})")
                
                for factor in all_factors:
                    tstat = us_row[f't_{factor}']
                    if pd.notna(tstat):
                        f.write(f" & ({tstat:.2f})")
                    else:
                        f.write(" & ")
                
                f.write(" & & ")
                f.write("\\\\\n")
                f.write("\\addlinespace\n")
            
            # Panel B: EUR
            f.write("\\midrule\n")
            f.write("\\multicolumn{" + str(ncols+1) + "}{l}{\\textbf{Panel B: EUR Factors}} \\\\\n")
            f.write("\\addlinespace\n")
            
            for strategy in strategies_list:
                eur_data = summary_df[(summary_df['Strategy'] == strategy) & (summary_df['Region'] == 'EUR')]
                
                if len(eur_data) == 0:
                    continue
                
                eur_row = eur_data.iloc[0]
                strategy_display = strategy.replace('_', ' ')
                
                f.write(f"\\textit{{{strategy_display}}}")
                
                # Alpha
                alpha = eur_row['Alpha_annual']
                alpha_tstat = eur_row['t_stat']
                if abs(alpha_tstat) > 2.576:
                    sig = '***'
                elif abs(alpha_tstat) > 1.96:
                    sig = '**'
                elif abs(alpha_tstat) > 1.645:
                    sig = '*'
                else:
                    sig = ''
                f.write(f" & {alpha:.2f}{sig}")
                
                # All factors
                for factor in all_factors:
                    beta = eur_row[f'Beta_{factor}']
                    tstat = eur_row[f't_{factor}']
                    
                    if pd.notna(beta):
                        if abs(tstat) > 2.576:
                            sig = '***'
                        elif abs(tstat) > 1.96:
                            sig = '**'
                        elif abs(tstat) > 1.645:
                            sig = '*'
                        else:
                            sig = ''
                        f.write(f" & {beta:.3f}{sig}")
                    else:
                        f.write(" & --")
                
                f.write(f" & {eur_row['R_squared_adj']:.3f} & {int(eur_row['N_obs'])}")
                f.write(" \\\\\n")
                
                # t-stats row
                f.write(" ")
                f.write(f" & ({alpha_tstat:.2f})")
                
                for factor in all_factors:
                    tstat = eur_row[f't_{factor}']
                    if pd.notna(tstat):
                        f.write(f" & ({tstat:.2f})")
                    else:
                        f.write(" & ")
                
                f.write(" & & ")
                f.write("\\\\\n")
                f.write("\\addlinespace\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\begin{tablenotes}\n")
            f.write("\\footnotesize\n")
            
            # NOTE SENZA "Factor Descriptions:" (stile Duarte corretto)
            f.write("\\item This table reports summary statistics for the regression of monthly excess returns on the excess returns of the indicated equity and fixed income portfolios and trend-following risk factors. Results for each strategy are reported separately. Alpha is reported in annualized percentage terms. t-statistics in parentheses are based on Newey-West HAC standard errors with automatic lag selection.\n\n")
            f.write("\\item S\\&P is the Standard \\& Poor's 500 stock return. SC-LC is the return difference between the Wilshire Small Cap 1750 and the Wilshire Large Cap 750. 10Y is the month-end to month-end change in the U.S. Federal Reserve 10-year constant-maturity yield. CredSpr is the month-end to month-end change in the difference between Moody's Baa yield and the 10-year Treasury yield. BdOpt is the return of a portfolio of lookback straddles on bond futures. FXOpt is the return of a portfolio of lookback straddles on currency (foreign exchange) futures. ComOpt is the return of a portfolio of lookback straddles on commodity futures.\n\n")
            f.write("\\item US factors are from Kenneth French Data Library, Federal Reserve, and Fung-Hsieh Data Library. EUR factors use European equivalents from ECB Statistical Data Warehouse, Eurostoxx, and European futures markets.\n\n")
            
            f.write("\\item *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n\n")
            f.write("\\vspace{0.3cm}\n")
            f.write("\\begin{center}\n")
            f.write("$R_{it} = \\alpha + \\beta_1 S\\&P_t + \\beta_2 SC\\text{-}LC_t + \\beta_3 BdOpt_t + \\beta_4 FXOpt_t + \\beta_5 ComOpt_t + \\beta_6 10Y_t + \\beta_7 CredSpr_t + \\varepsilon_t$\n")
            f.write("\\end{center}\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{threeparttable}\n")
            f.write("\\end{table}\n\n")
            f.write("\\clearpage\n\n")
            
            # ============================================================
            # TABLE 3: SIGNIFICANT FACTORS
            # ============================================================
            
            f.write(f"% {'='*76}\n")
            f.write(f"% TABLE 3: SIGNIFICANT FACTORS ONLY\n")
            f.write(f"% {'='*76}\n\n")
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Fung \\& Hsieh Factor Model - Significant Factors Only ({REGRESSION_FREQ.capitalize()} Frequency)}}\n")
            f.write(f"\\label{{tab:funghsieh_significant_{REGRESSION_FREQ}}}\n")
            f.write("\\begin{threeparttable}\n")
            f.write("\\begin{tabular}{lllcc}\n")
            f.write("\\toprule\n")
            f.write("Strategy & Region & Factor & Coefficient & t-stat \\\\\n")
            f.write("\\midrule\n")
            
            for strategy in strategies_list:
                strategy_display = strategy.replace('_', ' ')
                
                # US
                us_data = summary_df[(summary_df['Strategy'] == strategy) & (summary_df['Region'] == 'US')]
                if len(us_data) > 0:
                    us_row = us_data.iloc[0]
                    
                    if abs(us_row['t_stat']) > 1.645:
                        alpha = us_row['Alpha_annual']
                        tstat = us_row['t_stat']
                        if abs(tstat) > 2.576:
                            sig = '***'
                        elif abs(tstat) > 1.96:
                            sig = '**'
                        else:
                            sig = '*'
                        f.write(f"{strategy_display} & US & Alpha & {alpha:.2f}{sig} & {tstat:.2f} \\\\\n")
                    
                    for factor in all_factors:
                        tstat = us_row[f't_{factor}']
                        if pd.notna(tstat) and abs(tstat) > 1.645:
                            beta = us_row[f'Beta_{factor}']
                            if abs(tstat) > 2.576:
                                sig = '***'
                            elif abs(tstat) > 1.96:
                                sig = '**'
                            else:
                                sig = '*'
                            factor_display = FACTOR_NAMES_LATEX.get(factor, factor)
                            f.write(f"{strategy_display} & US & {factor_display} & {beta:.3f}{sig} & {tstat:.2f} \\\\\n")
                            
                # EUR
                eur_data = summary_df[(summary_df['Strategy'] == strategy) & (summary_df['Region'] == 'EUR')]
                if len(eur_data) > 0:
                    eur_row = eur_data.iloc[0]
                    
                    if abs(eur_row['t_stat']) > 1.645:
                        alpha = eur_row['Alpha_annual']
                        tstat = eur_row['t_stat']
                        if abs(tstat) > 2.576:
                            sig = '***'
                        elif abs(tstat) > 1.96:
                            sig = '**'
                        else:
                            sig = '*'
                        f.write(f"{strategy_display} & EUR & Alpha & {alpha:.2f}{sig} & {tstat:.2f} \\\\\n")
                    
                    for factor in all_factors:
                        tstat = eur_row[f't_{factor}']
                        if pd.notna(tstat) and abs(tstat) > 1.645:
                            beta = eur_row[f'Beta_{factor}']
                            if abs(tstat) > 2.576:
                                sig = '***'
                            elif abs(tstat) > 1.96:
                                sig = '**'
                            else:
                                sig = '*'
                            factor_display = FACTOR_NAMES_LATEX.get(factor, factor)  
                            f.write(f"{strategy_display} & EUR & {factor_display} & {beta:.3f}{sig} & {tstat:.2f} \\\\\n")  
                            

                
                f.write("\\addlinespace\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\begin{tablenotes}\n")
            f.write("\\footnotesize\n")
            f.write("\\item Only factors with |t-stat| $>$ 1.645 (p $<$ 0.10) are shown. *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{threeparttable}\n")
            f.write("\\end{table}\n\n")
        
        print(f"💾 File combined salvato: {latex_combined_path.name}")
        
        # ================================================================
        # STEP 9.2: FILE SEPARATED
        # ================================================================
        
        print("\n" + "=" * 80)
        print("STEP 9.2: Genera file LaTeX separated")
        print("=" * 80)
        
        latex_separated_path = TABLES_DIR / f"FungHsieh_summary_separated_{REGRESSION_FREQ}.tex"
        
        with open(latex_separated_path, 'w') as f:
            
            f.write(f"% {'='*76}\n")
            f.write(f"% FUNG & HSIEH FACTOR MODEL - STRATEGY-BY-STRATEGY\n")
            f.write(f"% Reference: Fung & Hsieh (2004)\n")
            f.write(f"% Format: VERTICAL layout (US | EUR columns)\n")
            f.write(f"% {'='*76}\n\n")
            
            section_num = 1
            
            for strategy in strategies_list:
                
                us_data = summary_df[(summary_df['Strategy'] == strategy) & (summary_df['Region'] == 'US')]
                eur_data = summary_df[(summary_df['Strategy'] == strategy) & (summary_df['Region'] == 'EUR')]
                
                if len(us_data) == 0 and len(eur_data) == 0:
                    continue
                
                strategy_display = strategy.replace('_', ' ')
                
                f.write(f"% {'='*76}\n")
                f.write(f"% SECTION {section_num}: {strategy_display.upper()}\n")
                f.write(f"% {'='*76}\n\n")
                
                us_row = us_data.iloc[0] if len(us_data) > 0 else None
                eur_row = eur_data.iloc[0] if len(eur_data) > 0 else None
                
                # TABLE 1: ALPHA COMPARISON
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write(f"\\caption{{{strategy_display} - Alpha Comparison}}\n")
                f.write(f"\\label{{tab:funghsieh_{strategy.lower().replace(' ', '')}alpha_{REGRESSION_FREQ}}}\n")
                f.write("\\begin{threeparttable}\n")
                f.write("\\begin{tabular}{lcc}\n")
                f.write("\\toprule\n")
                f.write(" & US Factors & EUR Factors \\\\\n")
                f.write("\\midrule\n")
                
                f.write("$\\alpha$ (\\% p.a.) ")
                if us_row is not None:
                    alpha_us = us_row['Alpha_annual']
                    tstat_us = us_row['t_stat']
                    if abs(tstat_us) > 2.576:
                        sig = '***'
                    elif abs(tstat_us) > 1.96:
                        sig = '**'
                    elif abs(tstat_us) > 1.645:
                        sig = '*'
                    else:
                        sig = ''
                    f.write(f"& {alpha_us:.2f}{sig} ")
                else:
                    f.write("& -- ")
                
                if eur_row is not None:
                    alpha_eur = eur_row['Alpha_annual']
                    tstat_eur = eur_row['t_stat']
                    if abs(tstat_eur) > 2.576:
                        sig = '***'
                    elif abs(tstat_eur) > 1.96:
                        sig = '**'
                    elif abs(tstat_eur) > 1.645:
                        sig = '*'
                    else:
                        sig = ''
                    f.write(f"& {alpha_eur:.2f}{sig} ")
                else:
                    f.write("& -- ")
                
                f.write("\\\\\n")
                
                f.write(" ")
                if us_row is not None:
                    f.write(f"& ({tstat_us:.2f}) ")
                else:
                    f.write("& ")
                
                if eur_row is not None:
                    f.write(f"& ({tstat_eur:.2f}) ")
                else:
                    f.write("& ")
                
                f.write("\\\\\n")
                f.write("\\addlinespace\n")
                
                f.write("$R^2$ adj ")
                if us_row is not None:
                    f.write(f"& {us_row['R_squared_adj']:.3f} ")
                else:
                    f.write("& -- ")
                
                if eur_row is not None:
                    f.write(f"& {eur_row['R_squared_adj']:.3f} ")
                else:
                    f.write("& -- ")
                
                f.write("\\\\\n")
                
                f.write("N ")
                if us_row is not None:
                    f.write(f"& {int(us_row['N_obs'])} ")
                else:
                    f.write("& -- ")
                
                if eur_row is not None:
                    f.write(f"& {int(eur_row['N_obs'])} ")
                else:
                    f.write("& -- ")
                
                f.write("\\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\begin{tablenotes}\n")
                f.write("\\footnotesize\n")
                f.write("\\item Alpha is annualized. t-statistics in parentheses. *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n")
                f.write("\\end{tablenotes}\n")
                f.write("\\end{threeparttable}\n")
                f.write("\\end{table}\n\n")
                
                # TABLE 2: FULL RESULTS
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write(f"\\caption{{{strategy_display} - Full Model Results}}\n")
                f.write(f"\\label{{tab:funghsieh_{strategy.lower().replace(' ', '')}full_{REGRESSION_FREQ}}}\n")
                f.write("\\begin{threeparttable}\n")
                f.write("\\begin{tabular}{lcc}\n")
                f.write("\\toprule\n")
                f.write("Factor & US Factors & EUR Factors \\\\\n")
                f.write("\\midrule\n")
                
                f.write("$\\alpha$ (\\%) ")
                if us_row is not None:
                    alpha_us = us_row['Alpha_annual']
                    tstat_us = us_row['t_stat']
                    if abs(tstat_us) > 2.576:
                        sig = '***'
                    elif abs(tstat_us) > 1.96:
                        sig = '**'
                    elif abs(tstat_us) > 1.645:
                        sig = '*'
                    else:
                        sig = ''
                    f.write(f"& {alpha_us:.2f}{sig} ")
                else:
                    f.write("& -- ")
                
                if eur_row is not None:
                    alpha_eur = eur_row['Alpha_annual']
                    tstat_eur = eur_row['t_stat']
                    if abs(tstat_eur) > 2.576:
                        sig = '***'
                    elif abs(tstat_eur) > 1.96:
                        sig = '**'
                    elif abs(tstat_eur) > 1.645:
                        sig = '*'
                    else:
                        sig = ''
                    f.write(f"& {alpha_eur:.2f}{sig} ")
                else:
                    f.write("& -- ")
                
                f.write("\\\\\n")
                
                f.write(" ")
                if us_row is not None:
                    f.write(f"& ({tstat_us:.2f}) ")
                else:
                    f.write("& ")
                
                if eur_row is not None:
                    f.write(f"& ({tstat_eur:.2f}) ")
                else:
                    f.write("& ")
                
                f.write("\\\\\n")
                f.write("\\addlinespace\n")
                
                # Beta rows
                for factor in all_factors:
                    factor_display = FACTOR_NAMES_LATEX.get(factor, factor)
                    f.write(f"{factor_display} ")
                    
                    if us_row is not None:
                        beta_us = us_row[f'Beta_{factor}']
                        tstat_us_beta = us_row[f't_{factor}']
                        
                        if pd.notna(beta_us):
                            if abs(tstat_us_beta) > 2.576:
                                sig = '***'
                            elif abs(tstat_us_beta) > 1.96:
                                sig = '**'
                            elif abs(tstat_us_beta) > 1.645:
                                sig = '*'
                            else:
                                sig = ''
                            f.write(f"& {beta_us:.3f}{sig} ")
                        else:
                            f.write("& -- ")
                    else:
                        f.write("& -- ")
                    
                    if eur_row is not None:
                        beta_eur = eur_row[f'Beta_{factor}']
                        tstat_eur_beta = eur_row[f't_{factor}']
                        
                        if pd.notna(beta_eur):
                            if abs(tstat_eur_beta) > 2.576:
                                sig = '***'
                            elif abs(tstat_eur_beta) > 1.96:
                                sig = '**'
                            elif abs(tstat_eur_beta) > 1.645:
                                sig = '*'
                            else:
                                sig = ''
                            f.write(f"& {beta_eur:.3f}{sig} ")
                        else:
                            f.write("& -- ")
                    else:
                        f.write("& -- ")
                    
                    f.write("\\\\\n")
                    
                    f.write(" ")
                    if us_row is not None and pd.notna(us_row[f't_{factor}']):
                        f.write(f"& ({us_row[f't_{factor}']:.2f}) ")
                    else:
                        f.write("& ")
                    
                    if eur_row is not None and pd.notna(eur_row[f't_{factor}']):
                        f.write(f"& ({eur_row[f't_{factor}']:.2f}) ")
                    else:
                        f.write("& ")
                    
                    f.write("\\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\begin{tablenotes}\n")
                f.write("\\footnotesize\n")
                f.write("\\item Coefficients with significance stars. t-statistics in parentheses. *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n")
                f.write("\\end{tablenotes}\n")
                f.write("\\end{threeparttable}\n")
                f.write("\\end{table}\n\n")
                
                # TABLE 3: VIF
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write(f"\\caption{{{strategy_display} - VIF Test}}\n")
                f.write(f"\\label{{tab:funghsieh_{strategy.lower().replace(' ', '')}vif_{REGRESSION_FREQ}}}\n")
                f.write("\\begin{threeparttable}\n")
                f.write("\\begin{tabular}{lcc}\n")
                f.write("\\toprule\n")
                f.write("Factor & US Factors & EUR Factors \\\\\n")
                f.write("\\midrule\n")
                
                for factor in all_factors:
                    factor_display = FACTOR_NAMES_LATEX.get(factor, factor)
                    f.write(f"{factor_display} ")
                    
                    if strategy in vif_storage and 'US' in vif_storage[strategy]:
                        vif_df_us = vif_storage[strategy]['US']
                        vif_row_us = vif_df_us[vif_df_us['Factor'] == factor]
                        if len(vif_row_us) > 0:
                            f.write(f"& {vif_row_us.iloc[0]['VIF']:.2f} ")
                        else:
                            f.write("& -- ")
                    else:
                        f.write("& -- ")
                    
                    if strategy in vif_storage and 'EUR' in vif_storage[strategy]:
                        vif_df_eur = vif_storage[strategy]['EUR']
                        vif_row_eur = vif_df_eur[vif_df_eur['Factor'] == factor]
                        if len(vif_row_eur) > 0:
                            f.write(f"& {vif_row_eur.iloc[0]['VIF']:.2f} ")
                        else:
                            f.write("& -- ")
                    else:
                        f.write("& -- ")
                    
                    f.write("\\\\\n")
                
                f.write("\\bottomrule\n")
                f.write("\\end{tabular}\n")
                f.write("\\begin{tablenotes}\n")
                f.write("\\footnotesize\n")
                f.write("\\item VIF $>$ 10 indicates serious multicollinearity. VIF $>$ 5 indicates moderate concerns.\n")
                f.write("\\end{tablenotes}\n")
                f.write("\\end{threeparttable}\n")
                f.write("\\end{table}\n\n")
                
                if section_num < len(strategies_list):
                    f.write("\\clearpage\n\n")
                
                section_num += 1
        
        print(f"💾 File separated salvato: {latex_separated_path.name}")
        
        # ================================================================
        # FINAL SUMMARY
        # ================================================================
        
        print("\n" + "=" * 80)
        print("=" * 80)
        print("✅ ANALISI COMPLETATA!")
        print("=" * 80)
        print("=" * 80)
        
        print(f"\n📁 File generati in {TABLES_DIR}:")
        
        print(f"\n⭐ FILE LATEX GENERATI (2 FILE):")
        print(f"\n   1. FungHsieh_summary_combined_{REGRESSION_FREQ}.tex")
        print(f"      → Tabella 1: Alpha Comparison (side-by-side US | EUR)")
        print(f"      → Tabella 2: Full Model Results")
        print(f"        • Alpha + TUTTI i {len(all_factors)} fattori Fung & Hsieh")
        print(f"        • Panel A: US, Panel B: EUR")
        print(f"        • Note SENZA 'Factor Descriptions:' (stile Duarte)")
        print(f"        • Formula regressione completa centrata")
        print(f"      → Tabella 3: Significant Factors Only")
        print(f"\n   2. FungHsieh_summary_separated_{REGRESSION_FREQ}.tex")
        print(f"      → {section_num-1} sezioni (una per strategia)")
        print(f"      → Layout VERTICALE (2 colonne: US | EUR)")
        print(f"      → Per ogni strategia:")
        print(f"        • Alpha Comparison")
        print(f"        • Full Results (tutti i {len(all_factors)} fattori)")
        print(f"        • VIF Test (diagnostica multicollinearità)")
        
        print(f"\n🎯 FATTORI USATI:")
        for i, factor in enumerate(all_factors, 1):
            latex_name = FACTOR_NAMES_LATEX.get(factor, factor)
            print(f"   β{i} ({latex_name}): {factor}")
        
        print(f"\n✅ IDENTICO A DUARTE E ACTIVE FI:")
        print(f"   • Panel A (US) e Panel B (EUR)")
        print(f"   • Lista fissa di fattori")
        print(f"   • Note discorsive senza 'Factor Descriptions:'")
        print(f"   • VIF con fattori dalla lista fissa")
        print(f"   • PUBLICATION-READY per tesi PhD")

# ================================================================
        # FILE 3: SOLO SLIDE PRESENTAZIONE (FUNG & HSIEH) - LOGICA ORIGINALE
        # ================================================================
        with open(latex_presentation_path, 'w', encoding='utf-8') as f:
            f.write("%------------------------------------------------------------\n")
            f.write("% FUNG & HSIEH FACTOR MODEL - PRESENTATION SLIDE ONLY\n")
            f.write("%------------------------------------------------------------\n")
            f.write("\\begin{frame}[t,shrink=15]{Fung \\& Hsieh Factor Model}\n")
            f.write("\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\n")
            f.write("\\begin{threeparttable}\n")
            f.write("\\begin{tabular}{lcccccccccc}\n\\toprule\n")
            f.write(" & $\\alpha$ (\\%) & $\\beta_{S\\&P}$ & $\\beta_{SC-LC}$ & $\\beta_{BdOpt}$ & $\\beta_{FXOpt}$ & $\\beta_{ComOpt}$ & $\\beta_{10Y}$ & $\\beta_{CredSpr}$ & $R^2$ adj & N \\\\\n")
            f.write("\\midrule\n")

            strategy_map = {'BTP_Italia': 'BTP Italia', 'CDS_Bond_Basis': 'CDS Bond Basis', 'iTraxx_Combined': 'iTraxx Indices Skew'}
            slide_strategies = ['BTP_Italia', 'CDS_Bond_Basis', 'iTraxx_Combined']
            
            # Usiamo esattamente la lista definita nel tuo script (SNP, SC_LC, BD_OPT, etc.)
            # Nota: all_factors è già definita globalmente nel tuo file
            factors_to_print = all_factors 

            for label, region in [("Panel A: US Factors", "US"), ("Panel B: EUR Factors", "EUR")]:
                f.write(f"\\multicolumn{{11}}{{l}}{{\\textbf{{{label}}}}} \\\\\n\\addlinespace\n")
                for strat in slide_strategies:
                    res = summary_df[(summary_df['Strategy'] == strat) & (summary_df['Region'] == region)]
                    if res.empty: continue
                    row = res.iloc[0]
                    
                    # Riga Coefficienti + Alpha
                    f.write(f"\\textit{{{strategy_map[strat]}}} & {row['Alpha_annual']:.2f}{row['Significance']}")
                    
                    for factor in factors_to_print:
                        beta_val = row.get(f'Beta_{factor}', np.nan)
                        t_stat_beta = row.get(f't_{factor}', np.nan)
                        
                        if pd.notna(beta_val):
                            # Stelle di significatività manuali per coerenza con il resto dello script
                            sig = '***' if abs(t_stat_beta) > 2.576 else '**' if abs(t_stat_beta) > 1.96 else '*' if abs(t_stat_beta) > 1.645 else ''
                            f.write(f" & {beta_val:.3f}{sig}")
                        else:
                            f.write(" & --")
                            
                    f.write(f" & {row['R_squared_adj']:.3f} & {int(row['N_obs'])} \\\\\n")
                    
                    # Riga t-stat
                    f.write(f"  & ({row['t_stat']:.2f})")
                    for factor in factors_to_print:
                        t_val = row.get(f't_{factor}', np.nan)
                        f.write(f" & ({t_val:.2f})" if pd.notna(t_val) else " & ")
                    f.write(" & & \\\\\n\\addlinespace\n")
                if region == "US": f.write("\\midrule\n")

            f.write("\\bottomrule\n\\end{tabular}\n")
            f.write("\\begin{tablenotes}\n\\tiny\n\\item Monthly excess returns regressed on equity, spread and trend-following factors. Alpha in annualized percentage terms. Newey-West HAC t-statistics in parentheses. S\\&P is the S\\&P 500 index. SC-LC is the Wilshire small-minus-large equity return. 10Y is the change in the 10y constant-maturity yield. CredSpr is the change in Moody's Baa -- 10y Treasury spread. BdOpt, FXOpt, ComOpt are lookback-straddle portfolios on bond, FX and commodity futures.\n")
            f.write("\\item *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n")
            f.write("\\end{tablenotes}\n\\end{threeparttable}\n\n\\vspace{0.05cm}\n\\tiny\n")
            f.write("$R_{it} = \\alpha + \\beta_1 S\\&P_t + \\beta_2 SC\\text{-}LC_t + \\beta_3 BdOpt_t + \\beta_4 FXOpt_t + \\beta_5 ComOpt_t + \\beta_6 10Y_t + \\beta_7 CredSpr_t + \\varepsilon_t$\n\n")
            f.write("\\end{frame}\n")
# ================================================================
        # FILE 4: ARTICLE TABLE (paper & skeleton)
        # ================================================================
        latex_article_path = TABLES_DIR / f"FungHsieh_article_{REGRESSION_FREQ}.tex"

        with open(latex_article_path, 'w', encoding='utf-8') as f:
            f.write("% " + "=" * 74 + "\n")
            f.write("% FUNG & HSIEH FACTOR MODEL — ARTICLE TABLE\n")
            f.write(f"% Frequency: {REGRESSION_FREQ.capitalize()}\n")
            f.write("% " + "=" * 74 + "\n\n")

            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Fung \\& Hsieh Factor Model Regressions}\n")
            f.write("\\label{tab:fung_hsieh}\n")
            f.write("\\begin{threeparttable}\n")
            f.write("\\begin{singlespace}\n")
            f.write("\\small\n")
            f.write("\\setlength{\\tabcolsep}{1.5pt}\n\n")

            n_fac = len(all_factors)
            n_cols = 1 + 1 + n_fac + 1
            f.write(f"\\begin{{tabular}}{{l{'c' * (n_cols - 1)}}}\n")
            f.write("\\toprule\n")
            f.write(" & $\\alpha$ (\\%)")
            for factor in all_factors:
                latex_name = FACTOR_NAMES_LATEX.get(factor, factor)
                f.write(f" & $\\beta_{{\\text{{{latex_name}}}}}$")
            f.write(" & $\\bar{R}^2$\\\\\n")
            f.write("\\midrule\n")

            strategy_map = {
                'BTP_Italia': 'BTP Italia',
                'CDS_Bond_Basis': 'CDS--Bond',
                'iTraxx_Combined': 'Index Skew'
            }
            article_strategies = ['BTP_Italia', 'CDS_Bond_Basis', 'iTraxx_Combined']

            dw_vals = []
            fstat_reject_all = True
            total_cols = n_cols

            for label, region in [("Panel A: US Factors", "US"),
                                  ("Panel B: EUR Factors", "EUR")]:
                f.write(f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{{label}}}}} \\\\\n")
                f.write("\\addlinespace\n")

                for strat in article_strategies:
                    res = summary_df[(summary_df['Strategy'] == strat)
                                     & (summary_df['Region'] == region)]
                    if res.empty:
                        continue
                    row = res.iloc[0]

                    if 'DW' in row and pd.notna(row.get('DW')):
                        dw_vals.append(row['DW'])
                    if 'F_pval' in row and pd.notna(row.get('F_pval')):
                        if row['F_pval'] > 0.01:
                            fstat_reject_all = False

                    display = strategy_map.get(strat, strat)
                    f.write(f"\\textit{{{display}}}")

                    sig_a = ('***' if abs(row['t_stat']) > 2.576
                             else '**' if abs(row['t_stat']) > 1.96
                             else '*' if abs(row['t_stat']) > 1.645 else '')
                    if sig_a:
                        f.write(f" & ${row['Alpha_annual']:.2f}^{{{sig_a}}}$")
                    else:
                        f.write(f" & {row['Alpha_annual']:.2f}")

                    for factor in all_factors:
                        val = row.get(f'Beta_{factor}', np.nan)
                        t = row.get(f't_{factor}', np.nan)
                        if pd.notna(val):
                            sig_b = ('***' if abs(t) > 2.576
                                     else '**' if abs(t) > 1.96
                                     else '*' if abs(t) > 1.645 else '')
                            if sig_b:
                                f.write(f" & ${val:.2f}^{{{sig_b}}}$")
                            else:
                                f.write(f" & {val:.2f}")
                        else:
                            f.write(" & --")
                    f.write(f" & {row['R_squared_adj']:.2f}")
                    f.write(" \\\\\n")

                    f.write(f"  & ({row['t_stat']:.2f})")
                    for factor in all_factors:
                        t = row.get(f't_{factor}', np.nan)
                        if pd.notna(t):
                            f.write(f" & ({t:.2f})")
                        else:
                            f.write(" & ")
                    f.write(" & \\\\\n")
                    f.write("\\addlinespace\n")

                if region == "US":
                    f.write("\\midrule\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n\n")
            # Collect N per strategy for notes
            n_info = {}
            for strat in article_strategies:
                res = summary_df[(summary_df['Strategy'] == strat)
                                 & (summary_df['Region'] == 'EUR')]
                if not res.empty:
                    n_info[strat] = int(res.iloc[0]['N_obs'])

            # --- Tablenotes ---
            f.write("\\begin{tablenotes}[para,flushleft]\n")
            f.write("\\footnotesize\n")
            f.write("\\item \\textit{Note:} ")
            f.write("Monthly excess returns regressed on the seven Fung \\& Hsieh (2004) ")
            f.write("hedge fund risk factors. ")
            f.write("$\\alpha$ is annualized (multiplied by 12). ")
            f.write("$t$-statistics in parentheses (Newey--West HAC). ")

            if dw_vals:
                f.write(f"Durbin--Watson statistics range from "
                        f"{min(dw_vals):.2f} to {max(dw_vals):.2f}. ")

            if fstat_reject_all:
                f.write("The joint $F$-test rejects the null that all slope "
                        "coefficients are zero at the 1\\% level for every specification. ")
            else:
                f.write("The joint $F$-test rejects the null that all slope "
                        "coefficients are zero at the 5\\% level for most specifications. ")

            f.write("S\\&P: S\\&P 500 excess return. ")
            f.write("SC--LC: Wilshire small-minus-large-cap equity spread. ")
            f.write("BdOpt, FXOpt, ComOpt: lookback-straddle returns on bond, ")
            f.write("currency, and commodity futures. ")
            f.write("10Y: change in 10-year constant-maturity yield. ")
            f.write("CredSpr: change in Moody's Baa minus 10-year Treasury spread. ")
            sample_str = ", ".join(
                f"{strategy_map.get(s, s)} {n_info[s]} months"
                for s in article_strategies if s in n_info)
            f.write(f"Sample: {sample_str}. ")
            f.write("$^{***}\\,p<0.01$, $^{**}\\,p<0.05$, $^{*}\\,p<0.10$.\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{singlespace}\n")
            f.write("\\end{threeparttable}\n")
            f.write("\\end{table}\n")

        print(f"💾 File article table salvato: {latex_article_path.name}")
        
        # ================================================================
        # VIF ARTICLE TABLE (standalone, for Appendix A.9)
        # ================================================================
        article_strategies = ['BTP_Italia', 'CDS_Bond_Basis', 'iTraxx_Combined']
        strategy_map_vif = {
            'BTP_Italia': 'BTP Italia',
            'CDS_Bond_Basis': 'CDS--Bond Basis',
            'iTraxx_Combined': 'iTraxx Combined',
        }

        vif_article_path = TABLES_DIR / f"FungHsieh_VIF_article_{REGRESSION_FREQ}.tex"
        with open(vif_article_path, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Variance Inflation Factors --- Fung \\& Hsieh (2004), EUR Factors}\n")
            f.write("\\label{tab:vif_funghsieh}\n")
            f.write("\\begin{threeparttable}\n")
            f.write("\\begin{singlespace}\n")
            f.write("\\small\n")
            n_s = len(article_strategies)
            f.write("\\begin{tabular}{l" + " r" * n_s + "}\n")
            f.write("\\toprule\n")
            header = "Factor"
            for s in article_strategies:
                header += f" & {strategy_map_vif.get(s, s)}"
            header += " \\\\\n"
            f.write(header)
            f.write("\\midrule\n")
            for factor in FUNG_HSIEH_FACTORS:
                row = factor.replace('_', r'\_')
                for s in article_strategies:
                    if s in vif_storage and 'EUR' in vif_storage[s]:
                        vif_df = vif_storage[s]['EUR']
                        vif_row = vif_df[vif_df['Factor'] == factor]
                        if len(vif_row) > 0:
                            v = vif_row.iloc[0]['VIF']
                            row += f" & {v:.2f}"
                        else:
                            row += " & --"
                    else:
                        row += " & --"
                row += " \\\\\n"
                f.write(row)
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\begin{tablenotes}[para,flushleft]\n")
            f.write("\\footnotesize\n")
            f.write("\\item \\textit{Note:} VIF computed on EUR factor matrix. ")
            f.write("VIF $> 10$ indicates serious multicollinearity; VIF $> 5$ moderate concerns.\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{singlespace}\n")
            f.write("\\end{threeparttable}\n")
            f.write("\\end{table}\n")
        print(f"💾 VIF article table: {vif_article_path.name}")

# ============================================================================
# SUMMARY FINALE
# ============================================================================

print(f"\n⭐ Article table: FungHsieh_article_{REGRESSION_FREQ}.tex  ← PAPER & SKELETON")
print("\n" + "=" * 80)
print("✅ SCRIPT COMPLETATO!")
print("=" * 80)
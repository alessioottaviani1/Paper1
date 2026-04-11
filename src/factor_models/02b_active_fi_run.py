"""
Script 2b: Run Factor Model Regressions - Active FI - MULTI STRATEGY - US & EUR
================================================================================
Testa l'alpha di TUTTE LE STRATEGIE usando Active FI Illusion factors.

Reference: Brooks, Gould, Richardson (2020) "Active Fixed Income Illusions"
           Journal of Fixed Income

STRATEGIE:
1. BTP-Italia
2. iTraxx Main
3. iTraxx SnrFin
4. iTraxx SubFin
5. iTraxx Xover
6. iTraxx Combined
7. CDS-Bond Basis

OUTPUT:
- 2 FILE .tex SOLO (identici a struttura Duarte):
  1. ActiveFI_summary_combined_<freq>.tex (3 tabelle)
  2. ActiveFI_summary_separated_<freq>.tex (per strategia: Alpha + Full + VIF)

LOGICA DUARTE:
- Lista fissa di NOMI di fattori (display names)
- Normalizzazione: US_Term → Term, EU_Term → Term
- DataFrame usa nomi normalizzati
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
# FATTORI ACTIVE FI - LISTA FISSA DEI NOMI (LOGICA DUARTE)
# ============================================================================
# Questi sono i NOMI che appariranno nelle tabelle
# I VALORI di US_Term e EU_Term sono diversi, ma usiamo nome generico "Term"
# Ordine: segue paper Brooks et al. (2020)

ACTIVE_FI_FACTORS = [
    'Term',              # US_Term per US, EU_Term per EUR
    'Global_Term',
    'Global_Aggregate',
    'Inflation_Linkers',
    'Corporate_Credit',
    'Emerging_Debt',
    'Emerging_Currency',
    'UST_Volatility'
]

# Mapping per LaTeX (nomi display nelle tabelle)
FACTOR_NAMES_LATEX = {
    'Term': 'Term',
    'Global_Term': 'Global Term',
    'Global_Aggregate': 'Global Agg',
    'Inflation_Linkers': 'Infl. Linkers',
    'Corporate_Credit': 'Corp Credit',
    'Emerging_Debt': 'Emerg Debt',
    'Emerging_Currency': 'Emerg Curr',
    'UST_Volatility': 'UST Vol'
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
# AUTO-DETECT FREQUENCIES
# ============================================================================

print("=" * 80)
print("ACTIVE FI FACTOR MODEL - AUTO-DETECT FREQUENCIES")
print("=" * 80)

frequencies_available = []

for freq in ['monthly', 'weekly']:
    found = False
    for strategy in STRATEGIES:
        strategy_lower = strategy.lower()
        data_us_path = PROCESSED_DATA_DIR / f"regression_data_active_fi_{strategy_lower}_us_{freq}.csv"
        
        if data_us_path.exists():
            found = True
            break
    
    if found:
        frequencies_available.append(freq)

if not frequencies_available:
    print("\n❌ ERRORE: Nessun file di dati Active FI trovato!")
    exit()

print(f"\n📊 Frequenze disponibili: {', '.join([f.upper() for f in frequencies_available])}")

# ============================================================================
# LOOP PER OGNI FREQUENZA
# ============================================================================

for REGRESSION_FREQ in frequencies_available:
    
    print("\n" + "=" * 80)
    print("=" * 80)
    print(f"PROCESSING FREQUENCY: {REGRESSION_FREQ.upper()}")
    print("=" * 80)
    print("=" * 80)
    
    # Reset storage per ogni frequenza
    all_results_us = []
    all_results_eur = []
    vif_storage = {}
    
    # ========================================================================
    # LOOP SU TUTTE LE STRATEGIE
    # ========================================================================
    
    for strategy_name in STRATEGIES:
        
        print("\n" + "=" * 80)
        print(f"STRATEGIA: {strategy_name}")
        print("=" * 80)
        
        strategy_lower = strategy_name.lower()
        
        # ====================================================================
        # STEP 1: CARICA DATI
        # ====================================================================
        
        print("\n" + "=" * 80)
        print(f"STEP 1: Caricamento dati - {strategy_name}")
        print("=" * 80)
        
        print(f"\n📊 Frequenza: {REGRESSION_FREQ.upper()}")
        
        data_us_path = PROCESSED_DATA_DIR / f"regression_data_active_fi_{strategy_lower}_us_{REGRESSION_FREQ}.csv"
        
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
        
        data_eur_path = PROCESSED_DATA_DIR / f"regression_data_active_fi_{strategy_lower}_eur_{REGRESSION_FREQ}.csv"
        
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
        
        # ====================================================================
        # STEP 2: DEFINISCI FATTORI (con nomi reali nei CSV)
        # ====================================================================
        
        print("\n" + "=" * 80)
        print(f"STEP 2: Definizione Full Model - {strategy_name}")
        print("=" * 80)
        
        # US: usa US_Term (nome reale nel CSV)
        available_us = [col for col in data_us.columns if col != 'Strategy_Return']
        
        print(f"\n🇺🇸 US FACTORS (nomi reali):")
        print(f"✅ Tutti gli {len(available_us)} fattori Active FI disponibili: {available_us}")
        
        if data_eur is not None:
            # EUR: usa EU_Term (nome reale nel CSV)
            available_eur = [col for col in data_eur.columns if col != 'Strategy_Return']
            
            print(f"\n🇪🇺 EUR FACTORS (nomi reali):")
            print(f"✅ Tutti gli {len(available_eur)} fattori Active FI disponibili: {available_eur}")
        
        # ====================================================================
        # STEP 3: CALCOLA HAC LAGS
        # ====================================================================
        
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
        
        # ====================================================================
        # STEP 4A: REGRESSIONI US (con nomi reali)
        # ====================================================================
        
        print("\n" + "=" * 80)
        print(f"STEP 4A: Stima Full Model US con Newey-West HAC - {strategy_name}")
        print("=" * 80)
        
        print(f"\n{'='*60}")
        print(f"📊 Full Model US - {len(available_us)} Active FI Factors")
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
                print(f"   {factor}: {beta:.3f} (t={tstat:.2f}){stars}")
        else:
            print(f"   Nessun beta significativo oltre l'alpha")
        
        # ====================================================================
        # SALVA RISULTATI US CON NOMI NORMALIZZATI (LOGICA DUARTE)
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
        
        # NORMALIZZA I NOMI: US_Term → Term (come in Duarte)
        for factor in ACTIVE_FI_FACTORS:
            # Trova il nome reale nel CSV
            if factor == 'Term':
                real_factor_name = 'US_Term'
            else:
                real_factor_name = factor
            
            if real_factor_name in result_us.params.index:
                result_row_us[f'Beta_{factor}'] = result_us.params[real_factor_name]
                result_row_us[f't_{factor}'] = result_us.tvalues[real_factor_name]
            else:
                result_row_us[f'Beta_{factor}'] = np.nan
                result_row_us[f't_{factor}'] = np.nan
        
        all_results_us.append(result_row_us)
        
        # ====================================================================
        # STEP 4B: REGRESSIONI EUR (con nomi reali)
        # ====================================================================
        
        if data_eur is not None:
            
            print("\n" + "=" * 80)
            print(f"STEP 4B: Stima Full Model EUR con Newey-West HAC - {strategy_name}")
            print("=" * 80)
            
            print(f"\n{'='*60}")
            print(f"📊 Full Model EUR - {len(available_eur)} Active FI Factors")
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
                    print(f"   {factor}: {beta:.3f} (t={tstat:.2f}){stars}")
            else:
                print(f"   Nessun beta significativo oltre l'alpha")
            
            # ================================================================
            # SALVA RISULTATI EUR CON NOMI NORMALIZZATI (LOGICA DUARTE)
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
            
            # NORMALIZZA I NOMI: EU_Term → Term (come in Duarte)
            for factor in ACTIVE_FI_FACTORS:
                # Trova il nome reale nel CSV
                if factor == 'Term':
                    real_factor_name = 'EU_Term'
                else:
                    real_factor_name = factor
                
                if real_factor_name in result_eur.params.index:
                    result_row_eur[f'Beta_{factor}'] = result_eur.params[real_factor_name]
                    result_row_eur[f't_{factor}'] = result_eur.tvalues[real_factor_name]
                else:
                    result_row_eur[f'Beta_{factor}'] = np.nan
                    result_row_eur[f't_{factor}'] = np.nan
            
            all_results_eur.append(result_row_eur)
        
        # ====================================================================
        # STEP 5: CONFRONTO
        # ====================================================================
        
        if data_eur is not None:
            
            print("\n" + "=" * 80)
            print(f"STEP 5: Confronto US vs EUR - {strategy_name}")
            print("=" * 80)
            
            comparison = [
                {
                    'Model': 'Active FI US',
                    'Alpha_period': alpha_us,
                    'Alpha_annual': alpha_annual_us,
                    't-stat': alpha_tstat_us,
                    'p-value': alpha_pval_us,
                    'R²': rsq_us,
                    'R² adj': rsq_adj_us,
                    'N obs': int(nobs_us)
                },
                {
                    'Model': 'Active FI EUR',
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
        
        # ====================================================================
        # STEP 6A: VIF US (con nomi reali, salva con nomi normalizzati)
        # ====================================================================
        
        print("\n" + "=" * 80)
        print(f"STEP 6A: VIF Test US (Multicollinearità) - {strategy_name}")
        print("=" * 80)
        
        if len(available_us) > 1:
            X_vif_us = data_us[available_us].copy()
            
            vif_data_us = []
            for i, col in enumerate(X_vif_us.columns):
                vif = variance_inflation_factor(X_vif_us.values, i)
                
                # NORMALIZZA IL NOME: US_Term → Term (per VIF storage)
                normalized_name = col.replace('US_Term', 'Term')
                
                vif_data_us.append({'Factor': normalized_name, 'VIF': vif})
            
            vif_df_us = pd.DataFrame(vif_data_us)
            vif_df_us = vif_df_us.sort_values('VIF', ascending=False)
            
            if strategy_name not in vif_storage:
                vif_storage[strategy_name] = {}
            vif_storage[strategy_name]['US'] = vif_df_us
            
            print(f"\n📊 VIF Test (Active FI US):")
            print(vif_df_us.to_string(index=False))
            
            max_vif_us = vif_df_us['VIF'].max()
            if max_vif_us > 10:
                print(f"\n⚠️ VIF massimo: {max_vif_us:.2f} > 10 → PROBLEMA SERIO di multicollinearità!")
            elif max_vif_us > 5:
                print(f"\n⚠️ VIF massimo: {max_vif_us:.2f} > 5 → Multicollinearità moderata, monitorare")
            else:
                print(f"\n✅ VIF massimo: {max_vif_us:.2f} < 5 → Multicollinearità accettabile")
        
        # ====================================================================
        # STEP 6B: VIF EUR (con nomi reali, salva con nomi normalizzati)
        # ====================================================================
        
        if data_eur is not None:
            
            print("\n" + "=" * 80)
            print(f"STEP 6B: VIF Test EUR (Multicollinearità) - {strategy_name}")
            print("=" * 80)
            
            if len(available_eur) > 1:
                X_vif_eur = data_eur[available_eur].copy()
                
                vif_data_eur = []
                for i, col in enumerate(X_vif_eur.columns):
                    vif = variance_inflation_factor(X_vif_eur.values, i)
                    
                    # NORMALIZZA IL NOME: EU_Term → Term (per VIF storage)
                    normalized_name = col.replace('EU_Term', 'Term')
                    
                    vif_data_eur.append({'Factor': normalized_name, 'VIF': vif})
                
                vif_df_eur = pd.DataFrame(vif_data_eur)
                vif_df_eur = vif_df_eur.sort_values('VIF', ascending=False)
                
                if strategy_name not in vif_storage:
                    vif_storage[strategy_name] = {}
                vif_storage[strategy_name]['EUR'] = vif_df_eur
                
                print(f"\n📊 VIF Test (Active FI EUR):")
                print(vif_df_eur.to_string(index=False))
                
                max_vif_eur = vif_df_eur['VIF'].max()
                if max_vif_eur > 10:
                    print(f"\n⚠️ VIF massimo: {max_vif_eur:.2f} > 10 → PROBLEMA SERIO di multicollinearità!")
                elif max_vif_eur > 5:
                    print(f"\n⚠️ VIF massimo: {max_vif_eur:.2f} > 5 → Multicollinearità moderata, monitorare")
                else:
                    print(f"\n✅ VIF massimo: {max_vif_eur:.2f} < 5 → Multicollinearità accettabile")
        
        print(f"\n✅ Analisi completata per {strategy_name}")
    
    # ========================================================================
    # STEP 9: CREA TABELLA AGGREGATA
    # ========================================================================
    
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
            # STEP 9.1: FILE COMBINED (LOGICA DUARTE)
            # ================================================================
            
            print("\n" + "=" * 80)
            print("STEP 9.1: Genera file LaTeX combined (LOGICA DUARTE)")
            print("=" * 80)

            latex_combined_path = TABLES_DIR / f"ActiveFI_summary_combined_{REGRESSION_FREQ}.tex"
            # PERCORSO PER LA SLIDE
            latex_presentation_path = TABLES_DIR / f"ActiveFI_Presentation_Slide_{REGRESSION_FREQ}.tex"

            strategies_list = summary_df['Strategy'].unique()
            
            # USA LA LISTA FISSA DEFINITA ALL'INIZIO (LOGICA DUARTE)
            all_factors = ACTIVE_FI_FACTORS.copy()
            
            with open(latex_combined_path, 'w') as f:
                
                f.write(f"% {'='*76}\n")
                f.write(f"% ACTIVE FI FACTOR MODEL - ALL STRATEGIES COMBINED\n")
                f.write(f"% Reference: Brooks, Gould, Richardson (2020)\n")
                f.write(f"% Journal of Fixed Income\n")
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
                f.write(f"\\caption{{Active Fixed Income Factor Model - Alpha Comparison ({REGRESSION_FREQ.capitalize()} Frequency)}}\n")
                f.write(f"\\label{{tab:activefi_alpha_comparison_{REGRESSION_FREQ}}}\n")
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
                # TABLE 2: FULL MODEL RESULTS (IDENTICO A DUARTE)
                # ============================================================
                
                f.write(f"% {'='*76}\n")
                f.write(f"% TABLE 2: FULL MODEL RESULTS\n")
                f.write(f"% {'='*76}\n\n")
                
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write(f"\\caption{{Active Fixed Income Factor Model - Full Model Results ({REGRESSION_FREQ.capitalize()} Frequency)}}\n")
                f.write(f"\\label{{tab:activefi_full_results_{REGRESSION_FREQ}}}\n")
                f.write("\\footnotesize\n")
                f.write("\\begin{threeparttable}\n")
                
                ncols = 1 + len(all_factors) + 2  # alpha + factors + R2 adj + N
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
                    
                    # All factors (usa nomi normalizzati)
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
                    
                    # All factors (usa nomi normalizzati)
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
                
                f.write("\\item This table reports summary statistics for the regression of monthly excess returns on the excess returns of the indicated fixed income portfolios and risk factors. Results for each strategy are reported separately. Alpha is reported in annualized percentage terms. t-statistics in parentheses are based on Newey-West HAC standard errors with automatic lag selection.\n\n")
                
                f.write("\\item Term is Bloomberg Barclays Treasury excess returns, where U.S. Term uses U.S. Treasuries and EU Term uses Eurozone government bonds. Global Term is Bloomberg Barclays Global Treasury Hedged excess returns. Global Aggregate is Bloomberg Barclays Global Aggregate Hedged excess returns. Inflation-Linkers is Bloomberg Barclays Global Aggregate Treasury Inflation-Linked Hedged excess returns. Corporate Credit is an equal-weighted combination of Barclays U.S. High Yield Corporate Bond Index return in excess of Duration-Matched Treasuries and S\\&P Leverage Loan Index in excess of 3m-Libor. Emerging Debt is Barclays Emerging Market Debt duration-adjusted excess returns. Emerging Currency is equal-weighted emerging market currencies. UST Volatility is delta-hedged straddles on 10y Treasury futures. US factors are from Bloomberg, Kenneth French Data Library, and Federal Reserve. EUR factors use European equivalents from Bloomberg, ECB Statistical Data Warehouse, and Eurostoxx.\n")
                
                f.write("\\item US factors are from Bloomberg, Kenneth French Data Library, and Federal Reserve. EUR factors use European equivalents from Bloomberg, ECB Statistical Data Warehouse, and Eurostoxx.\n\n")
                
                f.write("\\item *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n\n")
                
                f.write("\\vspace{0.3cm}\n")
                f.write("\\begin{center}\n")
                f.write("$R_{it} = \\alpha + \\beta_1 R_{Term,t} + \\beta_2 R_{GlobalTerm,t} + \\beta_3 R_{GlobalAgg,t} + \\beta_4 R_{InflLinkers,t} + \\beta_5 R_{CorpCredit,t} + \\beta_6 R_{EmergDebt,t} + \\beta_7 R_{EmergCurr,t} + \\beta_8 \\Delta USTVol_t + \\varepsilon_t$\n")
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
                f.write(f"\\caption{{Active Fixed Income Factor Model - Significant Factors Only ({REGRESSION_FREQ.capitalize()} Frequency)}}\n")
                f.write(f"\\label{{tab:activefi_significant_{REGRESSION_FREQ}}}\n")
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
            # FILE 3: SOLO SLIDE PRESENTAZIONE (ACTIVE FI)
            # ================================================================
            with open(latex_presentation_path, 'w', encoding='utf-8') as f:
                f.write("%------------------------------------------------------------\n")
                f.write("% ACTIVE FIXED INCOME FACTOR MODEL - PRESENTATION SLIDE ONLY\n")
                f.write("%------------------------------------------------------------\n")
                f.write("\\begin{frame}[t,shrink=15]{Active Fixed Income Factor Model}\n")
                f.write("\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\n")
                f.write("\\begin{threeparttable}\n")
                # 12 colonne: Strategia, Alpha, 8 Fattori, R2, N
                f.write("\\begin{tabular}{lccccccccccc}\n\\toprule\n")
                f.write(" & $\\alpha$ (\\%) & $\\beta_{Term}$ & $\\beta_{Global Term}$ & $\\beta_{Global Agg}$ & $\\beta_{Infl. Linkers}$ & $\\beta_{Corp Credit}$ & $\\beta_{Emerg Debt}$ & $\\beta_{Emerg Curr}$ & $\\beta_{UST Vol}$ & $R^2$ adj & N \\\\\n")
                f.write("\\midrule\n")

                strategy_map = {'BTP_Italia': 'BTP Italia', 'CDS_Bond_Basis': 'CDS Bond Basis', 'iTraxx_Combined': 'iTraxx Indices Skew'}
                slide_strategies = ['BTP_Italia', 'CDS_Bond_Basis', 'iTraxx_Combined']
                
                # Usiamo all_factors definita nel tuo script (i nomi normalizzati)
                factors_to_print = all_factors 

                for label, region in [("Panel A: US Factors", "US"), ("Panel B: EUR Factors", "EUR")]:
                    f.write(f"\\multicolumn{{12}}{{l}}{{\\textbf{{{label}}}}} \\\\\n\\addlinespace\n")
                    for strat in slide_strategies:
                        res = summary_df[(summary_df['Strategy'] == strat) & (summary_df['Region'] == region)]
                        if res.empty: continue
                        row = res.iloc[0]
                        
                        # Riga Coeff
                        f.write(f"\\textit{{{strategy_map[strat]}}} & {row['Alpha_annual']:.2f}{row['Significance']}")
                        
                        for factor in factors_to_print:
                            val = row.get(f'Beta_{factor}', np.nan)
                            t_val = row.get(f't_{factor}', np.nan)
                            
                            if pd.notna(val):
                                sig = '***' if abs(t_val) > 2.576 else '**' if abs(t_val) > 1.96 else '*' if abs(t_val) > 1.645 else ''
                                f.write(f" & {val:.3f}{sig}")
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
                f.write("\\begin{tablenotes}\n\\tiny\n\\item Monthly excess returns regressed on traditional fixed income premia (term, credit, EM, inflation-linked) and volatility. Alpha in annualized percentage terms. Newey-West HAC t-statistics in parentheses. Term: Bloomberg Barclays Treasury excess returns (US or Eurozone). Global Term/Agg: BBG Global Treasury/Aggregate (hedged). Inflation-Linkers: BBG Global Inflation-Linked. Corporate Credit: HY + leveraged loans in excess of duration-matched Treasuries. Emerging Debt/Currency: EM bond and FX baskets. UST Vol: delta-hedged straddles on 10y Treasury futures.\n")
                f.write("\\item *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n")
                f.write("\\end{tablenotes}\n\\end{threeparttable}\n\n\\vspace{0.05cm}\n\\tiny\n")
                f.write("$R_{it} = \\alpha + \\beta_1 R_{Term,t} + \\beta_2 R_{GlobalTerm,t} + \\beta_3 R_{GlobalAgg,t} + \\beta_4 R_{InflLinkers,t} + \\beta_5 R_{CorpCredit,t} + \\beta_6 R_{EmergDebt,t} + \\beta_7 R_{EmergCurr,t} + \\beta_8 \\Delta USTVol_t + \\varepsilon_t$\n\n")
                f.write("\\end{frame}\n")
            
            print(f"💾 File Presentation Slide (Active FI) salvato: {latex_presentation_path.name}")

# ================================================================
            # FILE 4: ARTICLE TABLE (paper & skeleton)
            # ================================================================
            latex_article_path = TABLES_DIR / f"ActiveFI_article_{REGRESSION_FREQ}.tex"

            with open(latex_article_path, 'w', encoding='utf-8') as f:
                f.write("% " + "=" * 74 + "\n")
                f.write("% ACTIVE FI FACTOR MODEL — ARTICLE TABLE\n")
                f.write(f"% Frequency: {REGRESSION_FREQ.capitalize()}\n")
                f.write("% " + "=" * 74 + "\n\n")

                f.write("\\begin{table}[H]\n")
                f.write("\\centering\n")
                f.write("\\caption{Active Fixed Income Factor Model Regressions}\n")
                f.write("\\label{tab:active_fi}\n")
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
                f.write("Monthly excess returns regressed on the eight Active Fixed Income ")
                f.write("premia of Brooks et al.\\ (2020). ")
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

                f.write("Term: Treasury excess returns (US or Eurozone). ")
                f.write("Global Term/Agg: Bloomberg Global Treasury/Aggregate (hedged). ")
                f.write("Infl.\\ Linkers: Bloomberg Global Inflation-Linked. ")
                f.write("Corp Credit: HY and leveraged loans in excess of duration-matched Treasuries. ")
                f.write("Emerg Debt/Curr: EM bond and FX baskets. ")
                f.write("UST Vol: delta-hedged straddles on 10Y Treasury futures. ")
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

            vif_article_path = TABLES_DIR / f"ActiveFI_VIF_article_{REGRESSION_FREQ}.tex"
            with open(vif_article_path, 'w', encoding='utf-8') as f:
                f.write("\\begin{table}[H]\n")
                f.write("\\centering\n")
                f.write("\\caption{Variance Inflation Factors --- Brooks et al.\\ (2020), EUR Factors}\n")
                f.write("\\label{tab:vif_activefi}\n")
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
                for factor in ACTIVE_FI_FACTORS:
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


            # ================================================================
            # STEP 9.2: FILE SEPARATED (LOGICA DUARTE)
            # ================================================================
            
            print("\n" + "=" * 80)
            print("STEP 9.2: Genera file LaTeX separated (LOGICA DUARTE)")
            print("=" * 80)
            
            latex_separated_path = TABLES_DIR / f"ActiveFI_summary_separated_{REGRESSION_FREQ}.tex"
            
            with open(latex_separated_path, 'w') as f:
                
                f.write(f"% {'='*76}\n")
                f.write(f"% ACTIVE FI FACTOR MODEL - STRATEGY-BY-STRATEGY\n")
                f.write(f"% Reference: Brooks, Gould, Richardson (2020)\n")
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
                    f.write(f"\\label{{tab:activefi_{strategy.lower().replace(' ', '')}alpha_{REGRESSION_FREQ}}}\n")
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
                    f.write(f"\\label{{tab:activefi_{strategy.lower().replace(' ', '')}full_{REGRESSION_FREQ}}}\n")
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
                    
                    # Beta rows (usa nomi normalizzati dalla lista fissa)
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
                    f.write(f"\\label{{tab:activefi_{strategy.lower().replace(' ', '')}vif_{REGRESSION_FREQ}}}\n")
                    f.write("\\begin{threeparttable}\n")
                    f.write("\\begin{tabular}{lcc}\n")
                    f.write("\\toprule\n")
                    f.write("Factor & US Factors & EUR Factors \\\\\n")
                    f.write("\\midrule\n")
                    
                    # VIF usa nomi normalizzati dalla lista fissa
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
            
            print(f"\n⭐ FILE LATEX GENERATI:")
            print(f"\n   1. ActiveFI_summary_combined_{REGRESSION_FREQ}.tex")
            print(f"   2. ActiveFI_summary_separated_{REGRESSION_FREQ}.tex")
            print(f"   3. ActiveFI_Presentation_Slide_{REGRESSION_FREQ}.tex")
            print(f"   4. ActiveFI_article_{REGRESSION_FREQ}.tex  ← PAPER & SKELETON")
            print(f"      → Tabella 1: Alpha Comparison (side-by-side US | EUR)")
            print(f"      → Tabella 2: Full Model Results")
            print(f"        • Alpha + TUTTI gli {len(all_factors)} fattori Active FI")
            print(f"        • Panel A: US, Panel B: EUR")
            print(f"        • Note con descrizione fattori Brooks et al. (2020)")
            print(f"        • Formula regressione completa centrata")
            print(f"      → Tabella 3: Significant Factors Only")
            print(f"\n   2. ActiveFI_summary_separated_{REGRESSION_FREQ}.tex")
            print(f"      → {section_num-1} sezioni (una per strategia)")
            print(f"      → Layout VERTICALE (2 colonne: US | EUR)")
            print(f"      → Per ogni strategia:")
            print(f"        • Alpha Comparison")
            print(f"        • Full Results (tutti gli {len(all_factors)} fattori)")
            print(f"        • VIF Test (diagnostica multicollinearità)")
            
            print(f"\n🎯 PROSSIMI STEP:")
            print(f"   1. Apri ActiveFI_summary_combined_{REGRESSION_FREQ}.tex in Overleaf")
            print(f"   2. Aggiungi nel preambolo:")
            print(f"      \\usepackage{{booktabs}}")
            print(f"      \\usepackage{{threeparttable}}")
            print(f"   3. Confronta risultati con Duarte factors")
            print(f"   4. Copia tabelle nel LaTeX della tesi")
            
            print(f"\n✅ LAYOUT IDENTICO A DUARTE:")
            print(f"   • Nomi normalizzati (Term invece di US_Term/EU_Term)")
            print(f"   • Panel A (US) e Panel B (EUR)")
            print(f"   • Lista fissa di fattori")
            print(f"   • Nessuna colonna extra")
            print(f"   • VIF con fattori normalizzati")
            print(f"   • PUBLICATION-READY per tesi PhD")

print("\n" + "=" * 80)
print("✅ SCRIPT COMPLETATO!")
print("=" * 80)



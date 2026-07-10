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
- File .tex per paper/slide (monthly):
  1. ActiveFI_article_<freq>.tex  +  ActiveFI_VIF_article_<freq>.tex
  2. ActiveFI_Presentation_Slide_<freq>.tex  +  ActiveFI_summary_<freq>.json

DATI (nuovo database), via 01b (come 01a/01c):
- Lato EUR dal parquet all_factors_monthly.parquet; lato US dal raw bbg
  (ticker USD/USD-hedged: LUATTRUU, BTSYTRUH, LEGATRUH, LF94TRUH, LF98TRUU,
  EMUSTRUU), eurizzato via GHS. UST_Volatility = IV_TSY (varswap) in entrambi.
- Corporate Credit = sola gamba HY (leveraged loan SPBDEL scartata: non tradabile).
  US: LF98TRUU; EUR: HY_CORP (H02500EU). HY in eccesso sul cash.

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
from inference import auto_hac_lags
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

for freq in ['monthly']:
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
            
            HAC_LAGS_US = auto_hac_lags(T_us)
            
            print(f"🇺🇸 US: Osservazioni = {T_us}, HAC lags (auto) = {HAC_LAGS_US}")
            
            if data_eur is not None:
                T_eur = len(data_eur)
                
                HAC_LAGS_EUR = auto_hac_lags(T_eur)
                
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

            # PERCORSO PER LA SLIDE
            latex_presentation_path = TABLES_DIR / f"ActiveFI_Presentation_Slide_{REGRESSION_FREQ}.tex"

            strategies_list = summary_df['Strategy'].unique()
            
            # USA LA LISTA FISSA DEFINITA ALL'INIZIO (LOGICA DUARTE)
            all_factors = ACTIVE_FI_FACTORS.copy()
            
            # FILE 3: SOLO SLIDE PRESENTAZIONE (ACTIVE FI)
            # ================================================================
            with open(latex_presentation_path, 'w', encoding='utf-8') as f:
                f.write("%------------------------------------------------------------\n")
                f.write("% ACTIVE FIXED INCOME FACTOR MODEL - PRESENTATION SLIDE ONLY\n")
                f.write("%------------------------------------------------------------\n")
                f.write("\\begin{frame}[t,shrink=15]{Brooks, Gould \\& Richardson (2020) --- Active FI Premia}\n")
                f.write("\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\n")
                f.write("\\begin{threeparttable}\n")
                # 12 colonne: Strategia, Alpha, 8 Fattori, R2, N
                f.write("\\begin{tabular}{lccccccccccc}\n\\toprule\n")
                f.write(" & $\\alpha$ (\\% p.a.) & $\\beta_{Term}$ & $\\beta_{Global Term}$ & $\\beta_{Global Agg}$ & $\\beta_{Infl. Linkers}$ & $\\beta_{Corp Credit}$ & $\\beta_{Emerg Debt}$ & $\\beta_{Emerg Curr}$ & $\\beta_{UST Vol}$ & $R^2$ adj & N \\\\\n")
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
                        f.write(f"\\textit{{{strategy_map[strat]}}} & \\hlt{{{row['Alpha_annual']:.2f}{row['Significance']}}}")                       
                        for factor in factors_to_print:
                            val = row.get(f'Beta_{factor}', np.nan)
                            t_val = row.get(f't_{factor}', np.nan)
                            
                            if pd.notna(val):
                                sig = '***' if abs(t_val) > 2.576 else '**' if abs(t_val) > 1.96 else '*' if abs(t_val) > 1.645 else ''
                                f.write(f" & {val:.2f}{sig}")
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
                f.write("\\end{threeparttable}\n")
                f.write("\\end{frame}\n")
            
            print(f"💾 File Presentation Slide (Active FI) salvato: {latex_presentation_path.name}")

# ================================================================
            # FILE 4: ARTICLE TABLE (paper & skeleton)
            # ================================================================
            def _write_afi_side(path, region, tbl_label, caption_suffix):
                strategy_map = {'BTP_Italia': 'BTP Italia', 'CDS_Bond_Basis': 'CDS--Bond Basis',
                                'iTraxx_Combined': 'iTraxx Combined'}
                strategy_line2 = {'iTraxx_Combined': 'Combined'}
                article_strategies = ['BTP_Italia', 'CDS_Bond_Basis', 'iTraxx_Combined']
                n_cols = 3 + len(all_factors)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write("\\begin{table}[H]\n\\centering\n\\singlespacing\n")
                    f.write(f"\\caption{{Active Fixed Income Factor Model Regressions{caption_suffix}}}\n")
                    f.write(f"\\label{{{tbl_label}}}\n")
                    f.write("\\begin{minipage}{\\textwidth}\n")
                    f.write("{\\footnotesize\\noindent This table reports estimates of the regression "
                            "$r_{i,t}=\\alpha_i+\\beta_1\\,\\textit{Term}_t+\\beta_2\\,\\textit{Global Term}_t"
                            "+\\beta_3\\,\\textit{Global Agg}_t+\\beta_4\\,\\textit{Infl.\\ Linkers}_t"
                            "+\\beta_5\\,\\textit{Corp Credit}_t+\\beta_6\\,\\textit{Emerg Debt}_t"
                            "+\\beta_7\\,\\textit{Emerg Curr}_t+\\beta_8\\,\\textit{UST Vol}_t"
                            "+\\varepsilon_{i,t}$, "
                            "where $r_{i,t}$ is the monthly excess return of strategy $i$ and the regressors are the eight "
                            "Active Fixed Income risk premia of Brooks et al.\\ (2020). ")
                    f.write("$\\alpha$ is annualized (\\% p.a.). $t$-statistics in parentheses (Newey--West HAC). ")
                    if region == "EUR":
                        f.write("Term: euro-area government bond excess return over cash. ")
                    else:
                        f.write("Term: U.S. Treasury excess return over cash. ")
                    f.write("Global Term: Bloomberg Global Treasury Index (EUR-hedged), in excess of cash. ")
                    f.write("Global Agg: excess return on the Bloomberg Global Aggregate Bond Index. ")
                    f.write("Infl.\\ Linkers: global inflation-linked bonds in excess of cash. ")
                    if region == "EUR":
                        f.write("Corp Credit: EUR-hedged pan-European high-yield corporate bond index, in excess of cash. ")
                    else:
                        f.write("Corp Credit: U.S. high-yield corporate bond index, in excess of cash. ")
                    f.write("Emerg Debt: excess return on hard-currency emerging-market debt. ")
                    f.write("Emerg Curr: emerging-market currency basket versus the U.S. dollar, in excess of USD cash. ")
                    f.write("UST Vol: return on a one-month variance swap (delta-hedged ATM straddle) on the 10-year U.S. Treasury future. ")
                    f.write("$\\bar{R}^2$ is the adjusted $R^2$. ")
                    n_info = {s: int(summary_df[(summary_df['Strategy'] == s) & (summary_df['Region'] == region)].iloc[0]['N_obs'])
                              for s in article_strategies
                              if not summary_df[(summary_df['Strategy'] == s) & (summary_df['Region'] == region)].empty}
                    f.write("Sample: " + ", ".join(f"{strategy_map.get(s, s)} {n_info[s]} months"
                                                   for s in article_strategies if s in n_info) + ". ")
                    f.write("$^{***}\\,p<0.01$, $^{**}\\,p<0.05$, $^{*}\\,p<0.10$.}\n")
                    f.write("\\end{minipage}\n\\par\\vspace{6pt}\n\\footnotesize\n")
                    f.write("\\setlength{\\tabcolsep}{3pt}\n")
                    f.write("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * (n_cols - 1) + "}\n\\toprule\n")
                    f.write(" & $\\alpha$ (\\% p.a.)")
                    for fac in all_factors:
                        f.write(f" & $\\beta_{{\\text{{{FACTOR_NAMES_LATEX.get(fac, fac)}}}}}$")
                    f.write(" & $\\bar{R}^2$ \\\\\n\\midrule\n")
                    for strat in article_strategies:
                        res = summary_df[(summary_df['Strategy'] == strat) & (summary_df['Region'] == region)]
                        if res.empty:
                            continue
                        row = res.iloc[0]
                        f.write(f"\\textit{{{strategy_map.get(strat, strat)}}}")
                        sa = ('***' if abs(row['t_stat']) > 2.576 else '**' if abs(row['t_stat']) > 1.96
                              else '*' if abs(row['t_stat']) > 1.645 else '')
                        f.write(f" & ${row['Alpha_annual']:.2f}^{{{sa}}}$" if sa else f" & {row['Alpha_annual']:.2f}")
                        for fac in all_factors:
                            b = row.get(f'Beta_{fac}', np.nan); t = row.get(f't_{fac}', np.nan)
                            if pd.notna(b):
                                sb = ('***' if abs(t) > 2.576 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.645 else '')
                                f.write(f" & ${b:.2f}^{{{sb}}}$" if sb else f" & {b:.2f}")
                            else:
                                f.write(" & --")
                        f.write(f" & {row['R_squared_adj']:.2f} \\\\\n"
                                f"\\textit{{{strategy_line2.get(strat, '')}}} & ({row['t_stat']:.2f})")
                        for fac in all_factors:
                            t = row.get(f't_{fac}', np.nan)
                            f.write(f" & ({t:.2f})" if pd.notna(t) else " & ")
                        f.write(" & \\\\\n\\addlinespace\n")
                    f.write("\\bottomrule\n\\end{tabular*}\n\\end{table}\n")

            _write_afi_side(TABLES_DIR / f"ActiveFI_article_{REGRESSION_FREQ}.tex", "EUR", "tab:active_fi", "")
            _write_afi_side(TABLES_DIR / f"ActiveFI_US_article_{REGRESSION_FREQ}.tex", "US", "tab:active_fi_us", " (US Factors)")

            print("💾 Active-FI article tables salvate (EUR main + US appendice)")

            # ================================================================
            # JSON EXPORT (for cross-method comparison slide in 07_tables.py)
            # ================================================================
            import json
            json_path = TABLES_DIR / f"ActiveFI_summary_{REGRESSION_FREQ}.json"
            json_records = summary_df.to_dict(orient='records')
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(json_records, jf, indent=2, default=str)
            print(f"💾 JSON salvato (cross-method): {json_path.name}")

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
                f.write("\\begin{minipage}{\\textwidth}\n")
                f.write("{\\footnotesize\\noindent Variance inflation factors of the eight Brooks et al.\\ (2020) ")
                f.write("EUR factors, computed on the factor matrix over each strategy's estimation sample. ")
                f.write("VIF $> 10$ indicates serious multicollinearity; VIF $> 5$ moderate concerns. ")
                f.write("Elevated VIF for Global Term and Global Aggregate reflects the strong correlation ")
                f.write("between global bond benchmarks and does not affect inference on alpha.}\n")
                f.write("\\end{minipage}\n\\par\\vspace{6pt}\n")
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
                f.write("\\end{singlespace}\n")
                f.write("\\end{table}\n")
            print(f"💾 VIF article table: {vif_article_path.name}")

            # JSON export per la tabella VIF combinata (assemblata da 03_subperiod_rolling_analysis.py)
            import json
            vif_json = {s: vif_storage[s]['EUR'].set_index('Factor')['VIF'].to_dict()
                        for s in article_strategies
                        if s in vif_storage and 'EUR' in vif_storage[s]}
            with open(TABLES_DIR / f"ActiveFI_VIF_{REGRESSION_FREQ}.json", 'w', encoding='utf-8') as jf:
                json.dump(vif_json, jf, indent=2)
            print(f"💾 VIF JSON: ActiveFI_VIF_{REGRESSION_FREQ}.json")


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
            print(f"\n   1. ActiveFI_article_{REGRESSION_FREQ}.tex  ← PAPER")
            print(f"      → Alpha + {len(all_factors)} fattori; Panel A: US, Panel B: EUR")
            print(f"   2. ActiveFI_VIF_article_{REGRESSION_FREQ}.tex  ← PAPER (VIF)")
            print(f"   3. ActiveFI_Presentation_Slide_{REGRESSION_FREQ}.tex  ← PRESENTAZIONE")
            print(f"   4. ActiveFI_summary_{REGRESSION_FREQ}.json")

print("\n" + "=" * 80)
print("✅ SCRIPT COMPLETATO!")
print("=" * 80)



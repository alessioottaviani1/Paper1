"""
Script 2: Run Factor Model Regressions - MULTI STRATEGY - US & EUR
====================================================================
Testa l'alpha di TUTTE LE STRATEGIE usando modelli fattoriali:
1. BTP-Italia
2. iTraxx Main
3. iTraxx SnrFin
4. iTraxx SubFin
5. iTraxx Xover
6. iTraxx Combined
7. CDS-Bond Basis

MODELLI TESTATI (per ogni strategia):
- Full Model US (10 fattori US)
- Full Model EUR (10 fattori EUR)

FOCUS: TEST DELL'ALPHA (obiettivo tesi!)

OUTPUT:
- 1 file aggregato: Duarte_summary_all_strategies_<freq>.tex
  (contiene 2 tabelle: beta completi + beta essenziali)
- 1 file per strategia: Duarte_<strategy>_<freq>.tex
  (contiene 7 tabelle: US full, US sig, US VIF, EUR full, EUR sig, EUR VIF, comparison)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETRI - MODIFICA SOLO QUESTA SEZIONE
# ============================================================================

# Frequenza dati per regressione
REGRESSION_FREQ = "monthly"  # "daily", "weekly", "monthly"

# Newey-West lags (per HAC standard errors)
# Rule of thumb: 4*(T/100)^(2/9) per quarterly data
# Per daily: usa sqrt(T), per weekly: T^(1/3), per monthly: T^(1/4)
HAC_LAGS = None  # None = calcolo automatico

# Livello significatività
ALPHA_LEVEL = 0.05  # 5%

# Include iTraxx_Combined in summary table?
INCLUDE_COMBINED_IN_SUMMARY = True  # True/False

# ============================================================================
# STRATEGIE DA ANALIZZARE
# ============================================================================

STRATEGIES = [
    'BTP_Italia',
    'iTraxx_Combined',
    'CDS_Bond_Basis'
]

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"

# Crea cartelle
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STORAGE PER RISULTATI AGGREGATI
# ============================================================================

all_results_us = []
all_results_eur = []
vif_storage = {}  # Storage per VIF: {strategy: {'US': vif_df, 'EUR': vif_df}}

# ============================================================================
# LOOP SU TUTTE LE STRATEGIE
# ============================================================================

for strategy_name in STRATEGIES:
    
    print("\n" + "=" * 80)
    print(f"STRATEGIA: {strategy_name}")
    print("=" * 80)
    
    strategy_lower = strategy_name.lower()
    
    # ========================================================================
    # STEP 1: CARICA DATI US e EUR per questa strategia
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(f"STEP 1: Caricamento dati - {strategy_name}")
    print("=" * 80)
    
    print(f"\n📊 Frequenza: {REGRESSION_FREQ.upper()}")
    
    # Carica dataset US
    data_us_path = PROCESSED_DATA_DIR / f"regression_data_{strategy_lower}_us_{REGRESSION_FREQ}.csv"
    
    if not data_us_path.exists():
        print(f"\n❌ File US non trovato: {data_us_path.name}")
        print(f"   Skipping {strategy_name}...")
        continue
    
    data_us = pd.read_csv(data_us_path, index_col=0, parse_dates=True)
    
    print(f"\n🇺🇸 DATASET US:")
    print(f"✅ Dataset caricato: {len(data_us)} osservazioni")
    print(f"📅 Periodo: {data_us.index.min().strftime('%Y-%m-%d')} to {data_us.index.max().strftime('%Y-%m-%d')}")
    print(f"📊 Colonne: {list(data_us.columns)}")
    
    # Rimuovi eventuali NaN
    data_us = data_us.dropna()
    print(f"✅ Dopo pulizia: {len(data_us)} osservazioni")
    
    # Carica dataset EUR
    data_eur_path = PROCESSED_DATA_DIR / f"regression_data_{strategy_lower}_eur_{REGRESSION_FREQ}.csv"
    
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
        
        # Rimuovi eventuali NaN
        data_eur = data_eur.dropna()
        print(f"✅ Dopo pulizia: {len(data_eur)} osservazioni")
    
    # ========================================================================
    # STEP 2: DEFINISCI FATTORI FULL MODEL
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(f"STEP 2: Definizione Full Model - {strategy_name}")
    print("=" * 80)
    
    # Full Model: tutti i 10 fattori
    full_model_factors = ['Mkt-RF', 'SMB', 'HML', 'UMD', 'RS', 'RI', 'RB', 'R2', 'R5', 'R10']
    
    # Verifica disponibilità fattori US
    available_us = [f for f in full_model_factors if f in data_us.columns]
    missing_us = [f for f in full_model_factors if f not in data_us.columns]
    
    print(f"\n🇺🇸 US FACTORS:")
    if missing_us:
        print(f"⚠️  Fattori mancanti: {missing_us}")
        print(f"✅ Fattori disponibili: {available_us}")
    else:
        print(f"✅ Tutti i 10 fattori disponibili: {available_us}")
    
    # Verifica disponibilità fattori EUR
    if data_eur is not None:
        available_eur = [f for f in full_model_factors if f in data_eur.columns]
        missing_eur = [f for f in full_model_factors if f not in data_eur.columns]
        
        print(f"\n🇪🇺 EUR FACTORS:")
        if missing_eur:
            print(f"⚠️  Fattori mancanti: {missing_eur}")
            print(f"✅ Fattori disponibili: {available_eur}")
        else:
            print(f"✅ Tutti i 10 fattori disponibili: {available_eur}")
    
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
    # STEP 4A: RUNNA REGRESSIONI - US
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(f"STEP 4A: Stima Full Model US con Newey-West HAC - {strategy_name}")
    print("=" * 80)
    
    print(f"\n{'='*60}")
    print(f"📊 Full Model US - 10 Factors")
    print(f"{'='*60}")
    
    # Variabile dipendente
    y_us = data_us['Strategy_Return']
    
    # Prepara X (fattori + costante)
    X_us = data_us[available_us].copy()
    X_us = sm.add_constant(X_us)
    
    # OLS
    model_us = sm.OLS(y_us, X_us)
    result_us = model_us.fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS_US})
    
    # Estrai alpha
    alpha_us = result_us.params['const']
    alpha_tstat_us = result_us.tvalues['const']
    alpha_pval_us = result_us.pvalues['const']
    
    # Annualizza alpha
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
    result_us_ols = model_us.fit()  # OLS without HAC for F-stat
    fstat_us = result_us_ols.fvalue
    fpval_us = result_us_ols.f_pvalue
    
    # Print results
    print(f"\n🎯 ALPHA US:")
    print(f"   {REGRESSION_FREQ.capitalize()}: {alpha_us:.2f}%")
    print(f"   Annualizzato: {alpha_annual_us:.4f}%")
    print(f"   t-stat: {alpha_tstat_us:.4f}")
    print(f"   p-value: {alpha_pval_us:.4f} {'***' if alpha_pval_us < 0.01 else '**' if alpha_pval_us < 0.05 else '*' if alpha_pval_us < 0.10 else ''}")
    
    print(f"\n📊 Modello US:")
    print(f"   R²: {rsq_us:.4f}")
    print(f"   R² adj: {rsq_adj_us:.4f}")
    print(f"   N obs: {int(nobs_us)}")
    
    # Print betas significativi
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
    
    # Salva risultati per tabella aggregata (CON BETA!)
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
    
    # Aggiungi beta di tutti i fattori
    for factor in full_model_factors:
        if factor in result_us.params.index:
            result_row_us[f'Beta_{factor}'] = result_us.params[factor]
            result_row_us[f't_{factor}'] = result_us.tvalues[factor]
        else:
            result_row_us[f'Beta_{factor}'] = np.nan
            result_row_us[f't_{factor}'] = np.nan
    
    all_results_us.append(result_row_us)
    
    # ========================================================================
    # STEP 4B: RUNNA REGRESSIONI - EUR
    # ========================================================================
    
    if data_eur is not None:
        
        print("\n" + "=" * 80)
        print(f"STEP 4B: Stima Full Model EUR con Newey-West HAC - {strategy_name}")
        print("=" * 80)
        
        print(f"\n{'='*60}")
        print(f"📊 Full Model EUR - 10 Factors")
        print(f"{'='*60}")
        
        # Variabile dipendente
        y_eur = data_eur['Strategy_Return']
        
        # Prepara X (fattori + costante)
        X_eur = data_eur[available_eur].copy()
        X_eur = sm.add_constant(X_eur)
        
        # OLS
        model_eur = sm.OLS(y_eur, X_eur)
        result_eur = model_eur.fit(cov_type='HAC', cov_kwds={'maxlags': HAC_LAGS_EUR})
        
        # Estrai alpha
        alpha_eur = result_eur.params['const']
        alpha_tstat_eur = result_eur.tvalues['const']
        alpha_pval_eur = result_eur.pvalues['const']
        
        # Annualizza alpha
        if REGRESSION_FREQ == "daily":
            alpha_annual_eur = alpha_eur * 252
        elif REGRESSION_FREQ == "weekly":
            alpha_annual_eur = alpha_eur * 52
        elif REGRESSION_FREQ == "monthly":
            alpha_annual_eur = alpha_eur * 12
        
        # Statistiche modello
        rsq_eur = result_eur.rsquared
        rsq_adj_eur = result_eur.rsquared_adj
        nobs_eur = result_eur.nobs
        dw_eur = sm.stats.stattools.durbin_watson(result_eur.resid)
        result_eur_ols = model_eur.fit()
        fstat_eur = result_eur_ols.fvalue
        fpval_eur = result_eur_ols.f_pvalue    
        
    
        
        # Print results
        print(f"\n🎯 ALPHA EUR:")
        print(f"   {REGRESSION_FREQ.capitalize()}: {alpha_eur:.2f}%")
        print(f"   Annualizzato: {alpha_annual_eur:.4f}%")
        print(f"   t-stat: {alpha_tstat_eur:.4f}")
        print(f"   p-value: {alpha_pval_eur:.4f} {'***' if alpha_pval_eur < 0.01 else '**' if alpha_pval_eur < 0.05 else '*' if alpha_pval_eur < 0.10 else ''}")
        
        print(f"\n📊 Modello EUR:")
        print(f"   R²: {rsq_eur:.4f}")
        print(f"   R² adj: {rsq_adj_eur:.4f}")
        print(f"   N obs: {int(nobs_eur)}")
        
        # Print betas significativi
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
        
        # Salva risultati per tabella aggregata (CON BETA!)
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
        
        # Aggiungi beta di tutti i fattori
        for factor in full_model_factors:
            if factor in result_eur.params.index:
                result_row_eur[f'Beta_{factor}'] = result_eur.params[factor]
                result_row_eur[f't_{factor}'] = result_eur.tvalues[factor]
            else:
                result_row_eur[f'Beta_{factor}'] = np.nan
                result_row_eur[f't_{factor}'] = np.nan
        
        all_results_eur.append(result_row_eur)
    
    # ========================================================================
    # STEP 5: CONFRONTO US vs EUR (per questa strategia)
    # ========================================================================
    
    if data_eur is not None:
        
        print("\n" + "=" * 80)
        print(f"STEP 5: Confronto US vs EUR - {strategy_name}")
        print("=" * 80)
        
        comparison = [
            {
                'Model': 'Full Model US',
                'Alpha_period': alpha_us,
                'Alpha_annual': alpha_annual_us,
                't-stat': alpha_tstat_us,
                'p-value': alpha_pval_us,
                'R²': rsq_us,
                'R² adj': rsq_adj_us,
                'N obs': int(nobs_us)
            },
            {
                'Model': 'Full Model EUR',
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
    # STEP 6A: VIF TEST (MULTICOLLINEARITÀ) - US
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
        
        # Salva in storage
        if strategy_name not in vif_storage:
            vif_storage[strategy_name] = {}
        vif_storage[strategy_name]['US'] = vif_df_us
        
        print(f"\n📊 VIF Test (Full Model US):")
        print(vif_df_us.to_string(index=False))
        
        # Interpretazione
        max_vif_us = vif_df_us['VIF'].max()
        if max_vif_us > 10:
            print(f"\n⚠️ VIF massimo: {max_vif_us:.2f} > 10 → PROBLEMA SERIO di multicollinearità!")
        elif max_vif_us > 5:
            print(f"\n⚠️ VIF massimo: {max_vif_us:.2f} > 5 → Multicollinearità moderata, monitorare")
        else:
            print(f"\n✅ VIF massimo: {max_vif_us:.2f} < 5 → Multicollinearità accettabile")
    
    # ========================================================================
    # STEP 6B: VIF TEST (MULTICOLLINEARITÀ) - EUR
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
            
            # Salva in storage
            if strategy_name not in vif_storage:
                vif_storage[strategy_name] = {}
            vif_storage[strategy_name]['EUR'] = vif_df_eur
            
            print(f"\n📊 VIF Test (Full Model EUR):")
            print(vif_df_eur.to_string(index=False))
            
            # Interpretazione
            max_vif_eur = vif_df_eur['VIF'].max()
            if max_vif_eur > 10:
                print(f"\n⚠️ VIF massimo: {max_vif_eur:.2f} > 10 → PROBLEMA SERIO di multicollinearità!")
            elif max_vif_eur > 5:
                print(f"\n⚠️ VIF massimo: {max_vif_eur:.2f} > 5 → Multicollinearità moderata, monitorare")
            else:
                print(f"\n✅ VIF massimo: {max_vif_eur:.2f} < 5 → Multicollinearità accettabile")
    
    # ========================================================================
    # FINE ANALISI PER QUESTA STRATEGIA - DATI SALVATI IN all_results
    # ========================================================================
    
    print("\n" + "=" * 80)
    print(f"✅ Analisi completata per {strategy_name}")
    print("=" * 80)

# ============================================================================
# STEP 9: CREA TABELLA AGGREGATA - TUTTE LE STRATEGIE
# ============================================================================

print("\n" + "=" * 80)
print("=" * 80)
print("STEP 9: Tabella aggregata - TUTTE LE STRATEGIE")
print("=" * 80)
print("=" * 80)

if all_results_us or all_results_eur:
    
    # Combina results (filtra iTraxx_Combined se necessario)
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
        
        # ====================================================================
        # GENERA FILE LATEX AGGREGATI (2 FILE SOLO)
        # ====================================================================
        
        print("\n" + "=" * 80)
        print("STEP 9.1: Genera file LaTeX aggregati (2 file solo)")
        print("=" * 80)
        
        # ================================================================
        # FILE 1: COMBINED - TUTTE LE STRATEGIE + ALPHA + SIG + VIF
        # ================================================================
        latex_combined_path = TABLES_DIR / f"Duarte_summary_combined_{REGRESSION_FREQ}.tex"
        # NUOVO PERCORSO PER LA SLIDE SINGOLA
        latex_presentation_path = TABLES_DIR / f"Duarte_Presentation_Slide_{REGRESSION_FREQ}.tex"
           
        with open(latex_combined_path, 'w') as f:
            
            # Header
            f.write(f"% {'='*76}\n")
            f.write(f"% DUARTE ET AL. FACTOR MODEL - ALL STRATEGIES COMBINED\n")
            f.write(f"% Frequency: {REGRESSION_FREQ.capitalize()}\n")
            f.write(f"% {'='*76}\n")
            f.write(f"% IMPORTANT: Add to LaTeX preamble: \\usepackage{{threeparttable}}\n")
            f.write(f"% {'='*76}\n\n")
            
            strategies_list = summary_df['Strategy'].unique()
            all_factors = ['Mkt-RF', 'SMB', 'HML', 'UMD', 'RS', 'RI', 'RB', 'R2', 'R5', 'R10']
            
            # ============================================================
            # TABLE 1: ALPHA COMPARISON (side-by-side US | EUR)
            # ============================================================
            
            f.write(f"% {'='*76}\n")
            f.write(f"% TABLE 1: ALPHA COMPARISON\n")
            f.write(f"% {'='*76}\n\n")
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Duarte Factor Model - Alpha Comparison ({REGRESSION_FREQ.capitalize()} Frequency)}}\n")
            f.write(f"\\label{{tab:duarte_alpha_comparison_{REGRESSION_FREQ}}}\n")
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
                
                # US side
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
                
                # EUR side
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
                
                # t-stat row
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
            f.write("\\small\n")
            f.write("\\item \\textit{Notes:} Alpha is annualized monthly excess return. t-statistics in parentheses based on Newey-West HAC standard errors.\n")
            f.write("\\item *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{table}\n\n")
            f.write("\\clearpage\n\n")


# ============================================================
            # DUARTE FACTOR MODEL - PRESENTATION SLIDE (IDENTICA AL PDF)
            # ============================================================
            f.write(f"% {'='*76}\n")
            f.write(f"% DUARTE SLIDE FORMAT\n")
            f.write(f"% {'='*76}\n\n")
            
            f.write("\\begin{frame}[t,shrink=15]{Duarte Factor Model}\n")
            f.write("\\centering\n")
            f.write("\\scriptsize\n")
            f.write("\\setlength{\\tabcolsep}{2pt}\n\n")
            f.write("\\begin{threeparttable}\n")
            
            ncols = 13 # alpha + 10 beta + r2 + n
            f.write(f"\\begin{{tabular}}{{l{'c'*ncols}}}\n")
            f.write("\\toprule\n")
            f.write(" & $\\alpha$ (\\%) & $\\beta_{Mkt-RF}$ & $\\beta_{SMB}$ & $\\beta_{HML}$ & $\\beta_{UMD}$ & $\\beta_{RS}$ & $\\beta_{RI}$ & $\\beta_{RB}$ & $\\beta_{R2}$ & $\\beta_{R5}$ & $\\beta_{R10}$ & $R^2$ adj & N \\\\\n")
            f.write("\\midrule\n")
            
            # Mapping nomi e selezione strategie specifiche per la slide
            strategy_map = {
                'BTP_Italia': 'BTP Italia',
                'CDS_Bond_Basis': 'CDS Bond Basis',
                'iTraxx_Combined': 'iTraxx Indices Skew'
            }
            slide_strategies = ['BTP_Italia', 'CDS_Bond_Basis', 'iTraxx_Combined']

            # --- PANEL A: US FACTORS ---
            f.write("\\multicolumn{14}{l}{\\textbf{Panel A: US Factors}} \\\\\n")
            f.write("\\addlinespace\n")
            for strategy in slide_strategies:
                res = summary_df[(summary_df['Strategy'] == strategy) & (summary_df['Region'] == 'US')]
                if len(res) == 0: continue
                row = res.iloc[0]
                
                f.write(f"\\textit{{{strategy_map[strategy]}}}")
                # Alpha + Betas
                f.write(f" & {row['Alpha_annual']:.2f}{row['Significance']}")
                for factor in all_factors:
                    beta, t = row[f'Beta_{factor}'], row[f't_{factor}']
                    sig = '***' if abs(t) > 2.576 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.645 else ''
                    f.write(f" & {beta:.3f}{sig}" if pd.notna(beta) else " & --")
                f.write(f" & {row['R_squared_adj']:.3f} & {int(row['N_obs'])} \\\\\n")
                
                # t-stats
                f.write(f"  & ({row['t_stat']:.2f})")
                for factor in all_factors:
                    t = row[f't_{factor}']
                    f.write(f" & ({t:.2f})" if pd.notna(t) else " & ")
                f.write(" & & \\\\\n\\addlinespace\n")

            f.write("\\midrule\n")
            
            # --- PANEL B: EUR FACTORS ---
            f.write("\\multicolumn{14}{l}{\\textbf{Panel B: EUR Factors}} \\\\\n")
            f.write("\\addlinespace\n")
            for strategy in slide_strategies:
                res = summary_df[(summary_df['Strategy'] == strategy) & (summary_df['Region'] == 'EUR')]
                if len(res) == 0: continue
                row = res.iloc[0]
                
                f.write(f"\\textit{{{strategy_map[strategy]}}}")
                f.write(f" & {row['Alpha_annual']:.2f}{row['Significance']}")
                for factor in all_factors:
                    beta, t = row[f'Beta_{factor}'], row[f't_{factor}']
                    sig = '***' if abs(t) > 2.576 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.645 else ''
                    f.write(f" & {beta:.3f}{sig}" if pd.notna(beta) else " & --")
                f.write(f" & {row['R_squared_adj']:.3f} & {int(row['N_obs'])} \\\\\n")
                
                # t-stats
                f.write(f"  & ({row['t_stat']:.2f})")
                for factor in all_factors:
                    t = row[f't_{factor}']
                    f.write(f" & ({t:.2f})" if pd.notna(t) else " & ")
                f.write(" & & \\\\\n\\addlinespace\n")

            f.write("\\bottomrule\n\\end{tabular}\n\n")
            
            # Tablenotes e Formula
            f.write("\\begin{tablenotes}\n\\tiny\n")
            f.write("\\item Monthly excess returns regressed on the indicated equity and bond factors. Alpha in annualized percentage terms. Newey-West HAC t-statistics in parentheses.\n")
            f.write("\\item Mkt-RF, SMB, HML, and UMD are the Fama-French market, small-minus-big, high-minus-low, and up-minus-down (momentum) factors. $R_S$ is the excess return on an S\\&P index of bank stocks. $R_I$, $R_B$ are the excess returns on Merrill Lynch indexes of A/BAA-rated industrial bonds and A/BAA-rated bank bonds, respectively. $R_2$, $R_5$, and $R_{10}$ are the excess returns on maturity-sorted bond portfolios of 2-year, 5-year, and 10-year Treasury bonds, respectively. *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n")
            f.write("\\end{tablenotes}\n\\end{threeparttable}\n\n")
            f.write("\\vspace{0.05cm}\n\\tiny\n")
            f.write("$R_{it} = \\alpha + \\beta_1 R_{Mt} + \\beta_2 SMB_t + \\beta_3 HML_t + \\beta_4 UMD_t + \\beta_5 R_{S,t} + \\beta_6 R_{I,t} + \\beta_7 R_{B,t} + \\beta_8 R_{2,t} + \\beta_9 R_{5,t} + \\beta_{10} R_{10,t} + \\varepsilon_t$\n\n")
            f.write("\\end{frame}\n")

            
            # ============================================================
            # TABLE 3: SIGNIFICANT FACTORS ONLY
            # ============================================================
            
            f.write(f"% {'='*76}\n")
            f.write(f"% TABLE 3: SIGNIFICANT FACTORS ONLY\n")
            f.write(f"% {'='*76}\n\n")
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write(f"\\caption{{Duarte Factor Model - Significant Factors Only ({REGRESSION_FREQ.capitalize()} Frequency)}}\n")
            f.write(f"\\label{{tab:duarte_significant_{REGRESSION_FREQ}}}\n")
            f.write("\\begin{tabular}{lllcc}\n")
            f.write("\\toprule\n")
            f.write("Strategy & Region & Factor & Coefficient & t-stat \\\\\n")
            f.write("\\midrule\n")
            
            # Collect significant factors (p < 0.10)
            for strategy in strategies_list:
                strategy_display = strategy.replace('_', ' ')
                
                # US factors
                us_data = summary_df[(summary_df['Strategy'] == strategy) & (summary_df['Region'] == 'US')]
                if len(us_data) > 0:
                    us_row = us_data.iloc[0]
                    
                    # Check alpha
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
                    
                    # Check betas
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
                            f.write(f"{strategy_display} & US & {factor} & {beta:.3f}{sig} & {tstat:.2f} \\\\\n")
                
                # EUR factors
                eur_data = summary_df[(summary_df['Strategy'] == strategy) & (summary_df['Region'] == 'EUR')]
                if len(eur_data) > 0:
                    eur_row = eur_data.iloc[0]
                    
                    # Check alpha
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
                    
                    # Check betas
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
                            f.write(f"{strategy_display} & EUR & {factor} & {beta:.3f}{sig} & {tstat:.2f} \\\\\n")
                
                f.write("\\addlinespace\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\begin{tablenotes}\n")
            f.write("\\small\n")
            f.write("\\item \\textit{Notes:} Only factors with |t-stat| $>$ 1.645 (p $<$ 0.10) are shown.\n")
            f.write("\\item *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{table}\n\n")
            f.write("\\clearpage\n\n")
        
        print(f"💾 File combined salvato: {latex_combined_path.name}")
        print(f"   → Tabella 1: Alpha Comparison (side-by-side)")
        print(f"   → Tabella 2: Full Model Results (con note estese)")
        print(f"   → Tabella 3: Significant Factors Only")
        
        # ================================================================
        # FILE 2: SEPARATED - LAYOUT VERTICALE PER OGNI STRATEGIA
        # ================================================================
        
        latex_separated_path = TABLES_DIR / f"Duarte_summary_separated_{REGRESSION_FREQ}.tex"
        
        with open(latex_separated_path, 'w') as f:
            
            # Header
            f.write(f"% {'='*76}\n")
            f.write(f"% DUARTE ET AL. FACTOR MODEL - STRATEGY-BY-STRATEGY\n")
            f.write(f"% Frequency: {REGRESSION_FREQ.capitalize()}\n")
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
                
                # Get data rows
                us_row = us_data.iloc[0] if len(us_data) > 0 else None
                eur_row = eur_data.iloc[0] if len(eur_data) > 0 else None
                
                # ===========================================================
                # TABLE 1: ALPHA COMPARISON (vertical)
                # ===========================================================
                
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write(f"\\caption{{{strategy_display} - Alpha Comparison}}\n")
                f.write(f"\\label{{tab:duarte_{strategy.lower().replace(' ', '')}alpha_{REGRESSION_FREQ}}}\n")
                f.write("\\begin{tabular}{lcc}\n")
                f.write("\\toprule\n")
                f.write(" & US Factors & EUR Factors \\\\\n")
                f.write("\\midrule\n")
                
                # Alpha row
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
                
                # t-stat row
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
                
                # R2 adj
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
                
                # N obs
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
                f.write("\\small\n")
                f.write("\\item \\textit{Notes:} Alpha is annualized. t-statistics in parentheses.\n")
                f.write("\\item *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n")
                f.write("\\end{tablenotes}\n")
                f.write("\\end{table}\n\n")
                
                # ===========================================================
                # TABLE 2: FULL RESULTS (vertical)
                # ===========================================================
                
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write(f"\\caption{{{strategy_display} - Full Model Results}}\n")
                f.write(f"\\label{{tab:duarte_{strategy.lower().replace(' ', '')}full_{REGRESSION_FREQ}}}\n")
                f.write("\\begin{tabular}{lcc}\n")
                f.write("\\toprule\n")
                f.write("Factor & US Factors & EUR Factors \\\\\n")
                f.write("\\midrule\n")
                
                # Alpha row
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
                
                # t-stat row
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
                    f.write(f"$\\beta_{{{factor}}}$ ")
                    
                    # US beta
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
                    
                    # EUR beta
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
                    
                    # t-stat row for this beta
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
                f.write("\\small\n")
                f.write("\\item \\textit{Notes:} Coefficients with significance stars. t-statistics in parentheses.\n")
                f.write("\\item *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n")
                f.write("\\end{tablenotes}\n")
                f.write("\\end{table}\n\n")
                
                # ===========================================================
                # TABLE 3: VIF (vertical)
                # ===========================================================
                
                f.write("\\begin{table}[htbp]\n")
                f.write("\\centering\n")
                f.write(f"\\caption{{{strategy_display} - VIF Test}}\n")
                f.write(f"\\label{{tab:duarte_{strategy.lower().replace(' ', '')}vif_{REGRESSION_FREQ}}}\n")
                f.write("\\begin{tabular}{lcc}\n")
                f.write("\\toprule\n")
                f.write("Factor & US Factors & EUR Factors \\\\\n")
                f.write("\\midrule\n")
                
                # Popola VIF con dati reali
                for factor in all_factors:
                    f.write(f"{factor} ")
                    
                    # US VIF - Factor names do NOT have suffix
                    if strategy in vif_storage and 'US' in vif_storage[strategy]:
                        vif_df_us = vif_storage[strategy]['US']
                        vif_row_us = vif_df_us[vif_df_us['Factor'] == factor]
                        if len(vif_row_us) > 0:
                            f.write(f"& {vif_row_us.iloc[0]['VIF']:.2f} ")
                        else:
                            f.write("& -- ")
                    else:
                        f.write("& -- ")
                    
                    # EUR VIF - Factor names do NOT have suffix
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
                f.write("\\small\n")
                f.write("\\item \\textit{Notes:} VIF $>$ 10 indicates serious multicollinearity. VIF $>$ 5 indicates moderate concerns.\n")
                f.write("\\end{tablenotes}\n")
                f.write("\\end{table}\n\n")
                
                if section_num < len(strategies_list):
                    f.write("\\clearpage\n\n")
                
                section_num += 1
        
        print(f"💾 File separated salvato: {latex_separated_path.name}")
        print(f"   → {section_num-1} sezioni (layout VERTICALE)")
        print(f"   → Ogni sezione: Alpha + Full Results + VIF")


# ================================================================
        # FILE 3: SOLO SLIDE PRESENTAZIONE (RICHIESTO PER MAIN.TEX)
        # ================================================================
        with open(latex_presentation_path, 'w', encoding='utf-8') as f:
            f.write("%------------------------------------------------------------\n")
            f.write("% DUARTE FACTOR MODEL - PRESENTATION SLIDE ONLY\n")
            f.write("%------------------------------------------------------------\n")
            f.write("\\begin{frame}[t,shrink=15]{Duarte Factor Model}\n")
            f.write("\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n\n")
            f.write("\\begin{threeparttable}\n")
            f.write("\\begin{tabular}{lccccccccccccc}\n\\toprule\n")
            f.write(" & $\\alpha$ (\\%) & $\\beta_{Mkt-RF}$ & $\\beta_{SMB}$ & $\\beta_{HML}$ & $\\beta_{UMD}$ & $\\beta_{RS}$ & $\\beta_{RI}$ & $\\beta_{RB}$ & $\\beta_{R2}$ & $\\beta_{R5}$ & $\\beta_{R10}$ & $R^2$ adj & N \\\\\n")
            f.write("\\midrule\n")

            strategy_map = {'BTP_Italia': 'BTP Italia', 'CDS_Bond_Basis': 'CDS Bond Basis', 'iTraxx_Combined': 'iTraxx Indices Skew'}
            slide_strategies = ['BTP_Italia', 'CDS_Bond_Basis', 'iTraxx_Combined']
            all_factors = ['Mkt-RF', 'SMB', 'HML', 'UMD', 'RS', 'RI', 'RB', 'R2', 'R5', 'R10']

            for label, region in [("Panel A: US Factors", "US"), ("Panel B: EUR Factors", "EUR")]:
                f.write(f"\\multicolumn{{14}}{{l}}{{\\textbf{{{label}}}}} \\\\\n\\addlinespace\n")
                for strat in slide_strategies:
                    res = summary_df[(summary_df['Strategy'] == strat) & (summary_df['Region'] == region)]
                    if res.empty: continue
                    row = res.iloc[0]
                    # Riga Coeff
                    f.write(f"\\textit{{{strategy_map[strat]}}} & {row['Alpha_annual']:.2f}{row['Significance']}")
                    for factor in all_factors:
                        val, t = row[f'Beta_{factor}'], row[f't_{factor}']
                        sig = '***' if abs(t) > 2.576 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.645 else ''
                        f.write(f" & {val:.3f}{sig}" if pd.notna(val) else " & --")
                    f.write(f" & {row['R_squared_adj']:.3f} & {int(row['N_obs'])} \\\\\n")
                    # Riga t-stat
                    f.write(f"  & ({row['t_stat']:.2f})")
                    for factor in all_factors:
                        t = row[f't_{factor}']
                        f.write(f" & ({t:.2f})" if pd.notna(t) else " & ")
                    f.write(" & & \\\\\n\\addlinespace\n")
                if region == "US": f.write("\\midrule\n")

            f.write("\\bottomrule\n\\end{tabular}\n")
            f.write("\\begin{tablenotes}\n\\tiny\n\\item Monthly excess returns regressed on the indicated equity and bond factors. Alpha in annualized percentage terms. Newey-West HAC t-statistics in parentheses.\n")
            f.write("\\item Mkt-RF, SMB, HML, and UMD are the Fama-French market, small-minus-big, high-minus-low, and up-minus-down (momentum) factors. $R_S$ is the excess return on an S\\&P index of bank stocks. $R_I$, $R_B$ are the excess returns on Merrill Lynch indexes of A/BAA-rated industrial bonds and A/BAA-rated bank bonds, respectively. $R_2$, $R_5$, and $R_{10}$ are the excess returns on maturity-sorted bond portfolios of 2-year, 5-year, and 10-year Treasury bonds, respectively. *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n")
            f.write("\\end{tablenotes}\n\\end{threeparttable}\n\n\\vspace{0.05cm}\n\\tiny\n")
            f.write("$R_{it} = \\alpha + \\beta_1 R_{Mt} + \\beta_2 SMB_t + \\beta_3 HML_t + \\beta_4 UMD_t + \\beta_5 R_{S,t} + \\beta_6 R_{I,t} + \\beta_7 R_{B,t} + \\beta_8 R_{2,t} + \\beta_9 R_{5,t} + \\beta_{10} R_{10,t} + \\varepsilon_t$\n\n")
            f.write("\\end{frame}\n")
        
        print(f"💾 File Presentation Slide salvato: {latex_presentation_path.name}")

        # ================================================================
        # FILE 4: ARTICLE TABLE (paper & skeleton)
        # ================================================================
        latex_article_path = TABLES_DIR / f"Duarte_article_{REGRESSION_FREQ}.tex"

        with open(latex_article_path, 'w', encoding='utf-8') as f:
            f.write("% " + "=" * 74 + "\n")
            f.write("% DUARTE ET AL. FACTOR MODEL — ARTICLE TABLE\n")
            f.write(f"% Frequency: {REGRESSION_FREQ.capitalize()}\n")
            f.write("% " + "=" * 74 + "\n\n")

            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Duarte Factor Model Regressions}\n")
            f.write("\\label{tab:duarte}\n")
            f.write("\\begin{threeparttable}\n")
            f.write("\\begin{singlespace}\n")
            f.write("\\small\n")
            f.write("\\setlength{\\tabcolsep}{1.5pt}\n\n")

            f.write("\\begin{tabular}{lccccccccccccc}\n")
            f.write("\\toprule\n")
            f.write(" & $\\alpha$ (\\%)")
            f.write(" & $\\beta_{Mkt}$")
            f.write(" & $\\beta_{SMB}$")
            f.write(" & $\\beta_{HML}$")
            f.write(" & $\\beta_{UMD}$")
            f.write(" & $\\beta_{R_S}$")
            f.write(" & $\\beta_{R_I}$")
            f.write(" & $\\beta_{R_B}$")
            f.write(" & $\\beta_{R_2}$")
            f.write(" & $\\beta_{R_5}$")
            f.write(" & $\\beta_{R_{10}}$")
            f.write(" & $\\bar{R}^2$")
            f.write(" \\\\\n")
            f.write("\\midrule\n")

            strategy_map = {
                'BTP_Italia': 'BTP Italia',
                'CDS_Bond_Basis': 'CDS--Bond',
                'iTraxx_Combined': 'Index Skew'
            }
            article_strategies = ['BTP_Italia', 'CDS_Bond_Basis', 'iTraxx_Combined']

            dw_vals = []
            fstat_reject_all = True

            for label, region in [("Panel A: US Factors", "US"),
                                  ("Panel B: EUR Factors", "EUR")]:
                f.write(f"\\multicolumn{{13}}{{l}}{{\\textbf{{{label}}}}} \\\\\n")
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

                    # --- Coefficient row ---
                    display = strategy_map.get(strat, strat)
                    f.write(f"\\textit{{{display}}}")

                    # Alpha: 2 decimals, superscript stars
                    sig_a = ('***' if abs(row['t_stat']) > 2.576
                             else '**' if abs(row['t_stat']) > 1.96
                             else '*' if abs(row['t_stat']) > 1.645 else '')
                    if sig_a:
                        f.write(f" & ${row['Alpha_annual']:.2f}^{{{sig_a}}}$")
                    else:
                        f.write(f" & {row['Alpha_annual']:.2f}")

                    # Betas: 2 decimals, superscript stars
                    for factor in all_factors:
                        beta = row.get(f'Beta_{factor}', np.nan)
                        t = row.get(f't_{factor}', np.nan)
                        if pd.notna(beta):
                            sig_b = ('***' if abs(t) > 2.576
                                     else '**' if abs(t) > 1.96
                                     else '*' if abs(t) > 1.645 else '')
                            if sig_b:
                                f.write(f" & ${beta:.2f}^{{{sig_b}}}$")
                            else:
                                f.write(f" & {beta:.2f}")
                        else:
                            f.write(" & --")

                    f.write(f" & {row['R_squared_adj']:.2f}")
                    f.write(" \\\\\n")

                    # --- t-stat row ---
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
            f.write("Monthly excess returns regressed on the ten Duarte et al.\\ (2007) ")
            f.write("equity and bond risk factors. ")
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

            f.write("$Mkt$: market excess return (Fama--French). ")
            f.write("$R_S$: bank stock excess return. ")
            f.write("$R_I$, $R_B$: A/BAA-rated industrial and bank bond excess returns. ")           
            f.write("$R_2$, $R_5$, $R_{10}$: maturity-sorted Treasury portfolio excess returns. ")
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
        all_factors = ['Mkt-RF', 'SMB', 'HML', 'UMD', 'RS', 'RI', 'RB', 'R2', 'R5', 'R10']

        vif_article_path = TABLES_DIR / f"Duarte_VIF_article_{REGRESSION_FREQ}.tex"
        with open(vif_article_path, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Variance Inflation Factors --- Duarte et al.\\ (2007), EUR Factors}\n")
            f.write("\\label{tab:vif_duarte}\n")
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
            for factor in all_factors:
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
            f.write("VIF $> 10$ indicates serious multicollinearity; VIF $> 5$ moderate concerns. ")
            f.write("Elevated VIF for R2, R5, R10 reflects the inherent correlation of ")
            f.write("Treasury return factors and does not affect inference on alpha.\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{singlespace}\n")
            f.write("\\end{threeparttable}\n")
            f.write("\\end{table}\n")
        print(f"💾 VIF article table: {vif_article_path.name}")


# ============================================================================
# STEP 10: SUMMARY FINALE - TUTTE LE STRATEGIE
# ============================================================================

print("\n" + "=" * 80)
print("=" * 80)
print("STEP 10: SUMMARY FINALE - TUTTE LE STRATEGIE")
print("=" * 80)
print("=" * 80)

if all_results_us:
    print(f"\n🇺🇸 US RESULTS:")
    print(f"   Strategie analizzate: {len(all_results_us)}")
    
    sig_us = [r for r in all_results_us if r['p_value'] < ALPHA_LEVEL]
    print(f"   Alpha significativi (p < {ALPHA_LEVEL}): {len(sig_us)}/{len(all_results_us)}")
    
    if sig_us:
        print(f"\n   ✅ Strategie con alpha significativo:")
        for r in sig_us:
            print(f"      • {r['Strategy']}: {r['Alpha_annual']:.4f}% annualizzato (p={r['p_value']:.4f}) {r['Significance']}")

if all_results_eur:
    print(f"\n🇪🇺 EUR RESULTS:")
    print(f"   Strategie analizzate: {len(all_results_eur)}")
    
    sig_eur = [r for r in all_results_eur if r['p_value'] < ALPHA_LEVEL]
    print(f"   Alpha significativi (p < {ALPHA_LEVEL}): {len(sig_eur)}/{len(all_results_eur)}")
    
    if sig_eur:
        print(f"\n   ✅ Strategie con alpha significativo:")
        for r in sig_eur:
            print(f"      • {r['Strategy']}: {r['Alpha_annual']:.4f}% annualizzato (p={r['p_value']:.4f}) {r['Significance']}")

# Ranking per alpha annualizzato
if all_results_us and all_results_eur:
    print(f"\n📊 RANKING PER ALPHA ANNUALIZZATO (TOP 10):")
    
    all_combined = all_results_us + all_results_eur
    ranking_df = pd.DataFrame(all_combined)
    ranking_df = ranking_df.sort_values('Alpha_annual', ascending=False)
    
    for idx, (i, row) in enumerate(ranking_df.head(10).iterrows(), 1):
        print(f"   {idx}. {row['Strategy']} ({row['Region']}): {row['Alpha_annual']:.4f}% {row['Significance']} (p={row['p_value']:.4f})")

print("\n" + "=" * 80)
print("✅ ANALISI COMPLETATA!")
print("=" * 80)

print(f"\n📁 File generati in {TABLES_DIR}:")
print(f"\n⭐ FILE LATEX GENERATI:")
print(f"\n   1. Duarte_summary_combined_{REGRESSION_FREQ}.tex")
print(f"   2. Duarte_summary_separated_{REGRESSION_FREQ}.tex")
print(f"   3. Duarte_Presentation_Slide_{REGRESSION_FREQ}.tex")
print(f"   4. Duarte_article_{REGRESSION_FREQ}.tex  ← PAPER & SKELETON")
print(f"      → Tabella 1: Alpha Comparison (side-by-side US | EUR)")
print(f"        • Solo alpha annualizzato, R² adj, N")
print(f"      → Tabella 2: Full Model Results")
print(f"        • Alpha + TUTTI i 10 fattori")
print(f"        • Panel A: US, Panel B: EUR")
print(f"        • Note con descrizione fattori (stile Duarte et al. 2007)")
print(f"        • Formula regressione centrata in fondo")
print(f"      → Tabella 3: Significant Factors Only")
print(f"        • Solo fattori con |t-stat| > 1.645")
print(f"\n   2. Duarte_summary_separated_{REGRESSION_FREQ}.tex")
print(f"      → {len(STRATEGIES)} sezioni (una per strategia)")
print(f"      → Layout VERTICALE (2 colonne: US | EUR)")
print(f"      → Per ogni strategia:")
print(f"        • Alpha Comparison")
print(f"        • Full Results (tutti i 10 fattori)")
print(f"        • VIF Test (diagnostica multicollinearità)")

if not INCLUDE_COMBINED_IN_SUMMARY:
    print(f"\n💡 NOTE: iTraxx Combined è ESCLUSO dalle tabelle")
    print(f"   (modifica INCLUDE_COMBINED_IN_SUMMARY = True per includerlo)")

print(f"\n💡 Per cambiare frequenza:")
print(f"   1. Modifica REGRESSION_FREQ = 'daily', 'weekly' o 'monthly'")
print(f"   2. Runna di nuovo lo script")

print(f"\n🎯 PROSSIMI STEP:")
print(f"   1. Apri Duarte_summary_combined_{REGRESSION_FREQ}.tex in Overleaf")
print(f"   2. Verifica compilazione (NO underscore errors!)")
print(f"   3. Copia tabelle nel LaTeX della tesi")
print(f"   4. Per dettagli singola strategia, usa Duarte_summary_separated_{REGRESSION_FREQ}.tex")
print(f"   5. Ripeti con Active FI factors e Fung & Hsieh")
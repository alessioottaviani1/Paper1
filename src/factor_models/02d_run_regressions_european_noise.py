"""
Script 2d: Run Factor Model Regressions - Hu, Pan & Wang (2013) - US & EUR
===========================================================================
Testa l'alpha della strategia BTP-Italia basis usando HPW Noise factors.

MODELLI TESTATI:
1. Full Model US (2 fattori HPW US)
2. Full Model EUR (2 fattori HPW EUR)

FOCUS: TEST DELL'ALPHA (confronto con altri modelli)

OUTPUT (separati per US e EUR):
- Tabelle regressione con alpha, beta, t-stat, R²
- Newey-West HAC standard errors
- VIF test multicollinearità
- Tabelle LaTeX per tesi
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
HAC_LAGS = None  # None = calcolo automatico

# Livello significatività
ALPHA_LEVEL = 0.05  # 5%

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
# STEP 1: CARICA DATI US e EUR
# ============================================================================

print("=" * 80)
print("HU, PAN & WANG (2013) - FACTOR MODEL REGRESSION - US & EUR")
print("=" * 80)

print(f"\n📊 Frequenza: {REGRESSION_FREQ.upper()}")

# Carica dataset HPW Noise US
data_us_path = PROCESSED_DATA_DIR / f"regression_data_noise_us_{REGRESSION_FREQ}.csv"

if not data_us_path.exists():
    print(f"\n❌ ERRORE: File US non trovato: {data_us_path}")
    print("\n💡 Runna prima 01d_import_noise_factors.py con la frequenza corretta")
    exit()

data_us = pd.read_csv(data_us_path, index_col=0, parse_dates=True)

print(f"\n🇺🇸 DATASET US:")
print(f"✅ Dataset caricato: {len(data_us)} osservazioni")
print(f"📅 Periodo: {data_us.index.min().strftime('%Y-%m-%d')} to {data_us.index.max().strftime('%Y-%m-%d')}")
print(f"📊 Colonne: {list(data_us.columns)}")

# Rimuovi eventuali NaN
data_us = data_us.dropna()
print(f"✅ Dopo pulizia: {len(data_us)} osservazioni")

# Carica dataset HPW Noise EUR
data_eur_path = PROCESSED_DATA_DIR / f"regression_data_noise_eur_{REGRESSION_FREQ}.csv"

if not data_eur_path.exists():
    print(f"\n❌ ERRORE: File EUR non trovato: {data_eur_path}")
    print("\n💡 Runna prima 01d_import_noise_factors.py con la frequenza corretta")
    exit()

data_eur = pd.read_csv(data_eur_path, index_col=0, parse_dates=True)

print(f"\n🇪🇺 DATASET EUR:")
print(f"✅ Dataset caricato: {len(data_eur)} osservazioni")
print(f"📅 Periodo: {data_eur.index.min().strftime('%Y-%m-%d')} to {data_eur.index.max().strftime('%Y-%m-%d')}")
print(f"📊 Colonne: {list(data_eur.columns)}")

# Rimuovi eventuali NaN
data_eur = data_eur.dropna()
print(f"✅ Dopo pulizia: {len(data_eur)} osservazioni")

# ============================================================================
# STEP 2: DEFINISCI FATTORI FULL MODEL
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Definizione Full Model")
print("=" * 80)

# Fattori US (tutti tranne Strategy_Return)
available_us = [col for col in data_us.columns if col != 'Strategy_Return']

print(f"\n🇺🇸 US FACTORS:")
print(f"✅ Tutti i {len(available_us)} fattori HPW disponibili: {available_us}")

# Fattori EUR (tutti tranne Strategy_Return)
available_eur = [col for col in data_eur.columns if col != 'Strategy_Return']

print(f"\n🇪🇺 EUR FACTORS:")
print(f"✅ Tutti i {len(available_eur)} fattori HPW disponibili: {available_eur}")

# ============================================================================
# STEP 3: CALCOLA HAC LAGS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Calcola Newey-West HAC lags")
print("=" * 80)

if HAC_LAGS is None:
    T_us = len(data_us)
    T_eur = len(data_eur)
    
    if REGRESSION_FREQ == "daily":
        HAC_LAGS_US = int(np.sqrt(T_us))
        HAC_LAGS_EUR = int(np.sqrt(T_eur))
    elif REGRESSION_FREQ == "weekly":
        HAC_LAGS_US = int(T_us**(1/3))
        HAC_LAGS_EUR = int(T_eur**(1/3))
    elif REGRESSION_FREQ == "monthly":
        HAC_LAGS_US = int(T_us**(1/4))
        HAC_LAGS_EUR = int(T_eur**(1/4))
    
    print(f"🇺🇸 US: Osservazioni = {T_us}, HAC lags (auto) = {HAC_LAGS_US}")
    print(f"🇪🇺 EUR: Osservazioni = {T_eur}, HAC lags (auto) = {HAC_LAGS_EUR}")
else:
    HAC_LAGS_US = HAC_LAGS
    HAC_LAGS_EUR = HAC_LAGS
    print(f"📊 HAC lags (manual): {HAC_LAGS}")

# ============================================================================
# STEP 4: RUNNA REGRESSIONI - US
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4A: Stima Full Model US con Newey-West HAC")
print("=" * 80)

print(f"\n{'='*60}")
print(f"📊 Full Model US - {len(available_us)} HPW Factors")
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

# Statistiche modello
rsq_us = result_us.rsquared
rsq_adj_us = result_us.rsquared_adj
nobs_us = result_us.nobs

# Print results
print(f"\n🎯 ALPHA US:")
print(f"   {REGRESSION_FREQ.capitalize()}: {alpha_us:.4f}%")
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
        print(f"   {factor}: {beta:.4f} (t={tstat:.2f}){stars}")
else:
    print(f"   Nessun beta significativo")

# ============================================================================
# STEP 5: RUNNA REGRESSIONI - EUR
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4B: Stima Full Model EUR con Newey-West HAC")
print("=" * 80)

print(f"\n{'='*60}")
print(f"📊 Full Model EUR - {len(available_eur)} HPW Factors")
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

# Print results
print(f"\n🎯 ALPHA EUR:")
print(f"   {REGRESSION_FREQ.capitalize()}: {alpha_eur:.4f}%")
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
        print(f"   {factor}: {beta:.4f} (t={tstat:.2f}){stars}")
else:
    print(f"   Nessun beta significativo")

# ============================================================================
# STEP 6: TABELLA COMPARATIVA US vs EUR
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Tabella comparativa US vs EUR")
print("=" * 80)

# Crea tabella comparativa
comparison = []

comparison.append({
    'Model': 'HPW Noise US',
    f'Alpha ({REGRESSION_FREQ})': f"{alpha_us:.4f}%",
    'Alpha (annual)': f"{alpha_annual_us:.4f}%",
    't-stat': f"{alpha_tstat_us:.4f}",
    'p-value': f"{alpha_pval_us:.4f}",
    'Sig': '***' if alpha_pval_us < 0.01 else '**' if alpha_pval_us < 0.05 else '*' if alpha_pval_us < 0.10 else '',
    'R²': f"{rsq_us:.4f}",
    'R² adj': f"{rsq_adj_us:.4f}",
    'N': int(nobs_us)
})

comparison.append({
    'Model': 'HPW Noise EUR',
    f'Alpha ({REGRESSION_FREQ})': f"{alpha_eur:.4f}%",
    'Alpha (annual)': f"{alpha_annual_eur:.4f}%",
    't-stat': f"{alpha_tstat_eur:.4f}",
    'p-value': f"{alpha_pval_eur:.4f}",
    'Sig': '***' if alpha_pval_eur < 0.01 else '**' if alpha_pval_eur < 0.05 else '*' if alpha_pval_eur < 0.10 else '',
    'R²': f"{rsq_eur:.4f}",
    'R² adj': f"{rsq_adj_eur:.4f}",
    'N': int(nobs_eur)
})

comparison_df = pd.DataFrame(comparison)
print(f"\n{comparison_df.to_string(index=False)}")

# Salva
comparison_path = TABLES_DIR / f"alpha_comparison_noise_us_eur_{REGRESSION_FREQ}.csv"
comparison_df.to_csv(comparison_path, index=False)
print(f"\n💾 Salvato: {comparison_path.name}")

# ============================================================================
# STEP 7: VIF TEST (MULTICOLLINEARITÀ) - US
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6A: VIF Test US (Multicollinearità)")
print("=" * 80)

if len(available_us) > 1:
    X_vif_us = data_us[available_us].copy()
    
    vif_data_us = []
    for i, col in enumerate(X_vif_us.columns):
        vif = variance_inflation_factor(X_vif_us.values, i)
        vif_data_us.append({'Factor': col, 'VIF': vif})
    
    vif_df_us = pd.DataFrame(vif_data_us)
    vif_df_us = vif_df_us.sort_values('VIF', ascending=False)
    
    print(f"\n📊 VIF Test (HPW US):")
    print(vif_df_us.to_string(index=False))
    
    # Interpretazione
    max_vif_us = vif_df_us['VIF'].max()
    if max_vif_us > 10:
        print(f"\n⚠️ VIF massimo: {max_vif_us:.2f} > 10 → PROBLEMA SERIO di multicollinearità!")
    elif max_vif_us > 5:
        print(f"\n⚠️ VIF massimo: {max_vif_us:.2f} > 5 → Multicollinearità moderata, monitorare")
    else:
        print(f"\n✅ VIF massimo: {max_vif_us:.2f} < 5 → Multicollinearità accettabile")
    
    # Salva
    vif_us_path = TABLES_DIR / f"vif_test_noise_us_{REGRESSION_FREQ}.csv"
    vif_df_us.to_csv(vif_us_path, index=False)
    print(f"💾 Salvato: {vif_us_path.name}")

# ============================================================================
# STEP 8: VIF TEST (MULTICOLLINEARITÀ) - EUR
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6B: VIF Test EUR (Multicollinearità)")
print("=" * 80)

if len(available_eur) > 1:
    X_vif_eur = data_eur[available_eur].copy()
    
    vif_data_eur = []
    for i, col in enumerate(X_vif_eur.columns):
        vif = variance_inflation_factor(X_vif_eur.values, i)
        vif_data_eur.append({'Factor': col, 'VIF': vif})
    
    vif_df_eur = pd.DataFrame(vif_data_eur)
    vif_df_eur = vif_df_eur.sort_values('VIF', ascending=False)
    
    print(f"\n📊 VIF Test (HPW EUR):")
    print(vif_df_eur.to_string(index=False))
    
    # Interpretazione
    max_vif_eur = vif_df_eur['VIF'].max()
    if max_vif_eur > 10:
        print(f"\n⚠️ VIF massimo: {max_vif_eur:.2f} > 10 → PROBLEMA SERIO di multicollinearità!")
    elif max_vif_eur > 5:
        print(f"\n⚠️ VIF massimo: {max_vif_eur:.2f} > 5 → Multicollinearità moderata, monitorare")
    else:
        print(f"\n✅ VIF massimo: {max_vif_eur:.2f} < 5 → Multicollinearità accettabile")
    
    # Salva
    vif_eur_path = TABLES_DIR / f"vif_test_noise_eur_{REGRESSION_FREQ}.csv"
    vif_df_eur.to_csv(vif_eur_path, index=False)
    print(f"💾 Salvato: {vif_eur_path.name}")

# ============================================================================
# STEP 9: TABELLE LATEX PER TESI - US
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7A: Genera tabelle LaTeX per tesi - US")
print("=" * 80)

# Tabella coefficienti US
coef_table_us = pd.DataFrame({
    'Factor': result_us.params.index,
    'Coefficient': result_us.params.values,
    't-stat': result_us.tvalues.values,
    'p-value': result_us.pvalues.values
})

# Aggiungi significatività
coef_table_us['Sig'] = coef_table_us['p-value'].apply(
    lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
)

latex_coef_us = coef_table_us.to_latex(
    index=False,
    float_format="%.4f",
    caption=f"HPW Noise US Coefficients - {REGRESSION_FREQ.capitalize()} Frequency",
    label=f"tab:noise_us_{REGRESSION_FREQ}"
)

latex_coef_us_path = TABLES_DIR / f"noise_coefficients_us_{REGRESSION_FREQ}.tex"
with open(latex_coef_us_path, 'w') as f:
    f.write(latex_coef_us)
print(f"💾 Salvato: {latex_coef_us_path.name}")

# ============================================================================
# STEP 10: TABELLE LATEX PER TESI - EUR
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7B: Genera tabelle LaTeX per tesi - EUR")
print("=" * 80)

# Tabella coefficienti EUR
coef_table_eur = pd.DataFrame({
    'Factor': result_eur.params.index,
    'Coefficient': result_eur.params.values,
    't-stat': result_eur.tvalues.values,
    'p-value': result_eur.pvalues.values
})

# Aggiungi significatività
coef_table_eur['Sig'] = coef_table_eur['p-value'].apply(
    lambda p: '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else ''
)

latex_coef_eur = coef_table_eur.to_latex(
    index=False,
    float_format="%.4f",
    caption=f"HPW Noise EUR Coefficients - {REGRESSION_FREQ.capitalize()} Frequency",
    label=f"tab:noise_eur_{REGRESSION_FREQ}"
)

latex_coef_eur_path = TABLES_DIR / f"noise_coefficients_eur_{REGRESSION_FREQ}.tex"
with open(latex_coef_eur_path, 'w') as f:
    f.write(latex_coef_eur)
print(f"💾 Salvato: {latex_coef_eur_path.name}")

# Tabella comparativa LaTeX
latex_comparison = comparison_df.to_latex(
    index=False,
    float_format="%.4f",
    caption=f"HPW Noise - Alpha Comparison US vs EUR - {REGRESSION_FREQ.capitalize()} Frequency",
    label=f"tab:alpha_comparison_noise_us_eur_{REGRESSION_FREQ}"
)

latex_comparison_path = TABLES_DIR / f"alpha_comparison_noise_us_eur_{REGRESSION_FREQ}.tex"
with open(latex_comparison_path, 'w') as f:
    f.write(latex_comparison)
print(f"💾 Salvato: {latex_comparison_path.name}")

# ============================================================================
# STEP 11: SUMMARY FINALE
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY FINALE")
print("=" * 80)

print(f"\n🇺🇸 HPW NOISE US:")
print(f"   Alpha annualizzato: {alpha_annual_us:.4f}%")
print(f"   t-stat: {alpha_tstat_us:.4f}")
print(f"   p-value: {alpha_pval_us:.4f}")

if alpha_pval_us < ALPHA_LEVEL:
    print(f"   ✅ ALPHA STATISTICAMENTE SIGNIFICATIVO (p < {ALPHA_LEVEL})!")
else:
    print(f"   ❌ Alpha NON significativo (p >= {ALPHA_LEVEL})")

print(f"   R² adjusted: {rsq_adj_us:.4f} ({rsq_adj_us*100:.2f}% varianza spiegata)")

print(f"\n🇪🇺 HPW NOISE EUR:")
print(f"   Alpha annualizzato: {alpha_annual_eur:.4f}%")
print(f"   t-stat: {alpha_tstat_eur:.4f}")
print(f"   p-value: {alpha_pval_eur:.4f}")

if alpha_pval_eur < ALPHA_LEVEL:
    print(f"   ✅ ALPHA STATISTICAMENTE SIGNIFICATIVO (p < {ALPHA_LEVEL})!")
else:
    print(f"   ❌ Alpha NON significativo (p >= {ALPHA_LEVEL})")

print(f"   R² adjusted: {rsq_adj_eur:.4f} ({rsq_adj_eur*100:.2f}% varianza spiegata)")

# Confronto
print(f"\n📊 CONFRONTO US vs EUR:")
diff_alpha = alpha_annual_eur - alpha_annual_us
print(f"   Differenza alpha annualizzato: {diff_alpha:+.4f}%")

if abs(alpha_pval_us - alpha_pval_eur) > 0.05:
    if alpha_pval_us < alpha_pval_eur:
        print(f"   → Fattori HPW US spiegano meglio la strategia (p-value più basso)")
    else:
        print(f"   → Fattori HPW EUR spiegano meglio la strategia (p-value più basso)")
else:
    print(f"   → Fattori HPW US e EUR hanno potere esplicativo simile")

print("\n" + "=" * 80)
print("✅ ANALISI COMPLETATA!")
print("=" * 80)

print(f"\n📁 File generati in {TABLES_DIR}:")
print(f"\n🇺🇸 US:")
print(f"   • noise_coefficients_us_{REGRESSION_FREQ}.tex")
print(f"   • vif_test_noise_us_{REGRESSION_FREQ}.csv")

print(f"\n🇪🇺 EUR:")
print(f"   • noise_coefficients_eur_{REGRESSION_FREQ}.tex")
print(f"   • vif_test_noise_eur_{REGRESSION_FREQ}.csv")

print(f"\n📊 Comparativo:")
print(f"   • alpha_comparison_noise_us_eur_{REGRESSION_FREQ}.csv")
print(f"   • alpha_comparison_noise_us_eur_{REGRESSION_FREQ}.tex")

print(f"\n💡 Per cambiare frequenza:")
print(f"   1. Modifica REGRESSION_FREQ = 'daily', 'weekly' o 'monthly'")
print(f"   2. Runna di nuovo lo script")
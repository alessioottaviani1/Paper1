# -*- coding: utf-8 -*-
"""
MPPM Analysis - Goetzmann et al. (2007)
Calculates Manipulation-Proof Performance Measure for fixed income strategies.
Author: Alessio Ottaviani, EDHEC Business School, December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# PARAMETRI
FREQUENCY = "monthly"
RHO_VALUES = [2, 3, 4]

STRATEGIES = {
    'BTP_Italia': 'btp_italia/index_daily.csv',
    'iTraxx_Main': 'itraxx_main/index_daily.csv',
    'iTraxx_SnrFin': 'itraxx_snrfin/index_daily.csv',
    'iTraxx_SubFin': 'itraxx_subfin/index_daily.csv',
    'iTraxx_Xover': 'itraxx_xover/index_daily.csv',
    'iTraxx_Combined': 'itraxx_combined/index_daily.csv',
    'CDS_Bond_Basis': 'cds_bond_basis/index_daily.csv'
}

STRATEGY_NAMES_LATEX = {
    'BTP_Italia': r'BTP\textit{-}Italia',
    'iTraxx_Main': r'iTraxx Main',
    'iTraxx_SnrFin': r'iTraxx SnrFin',
    'iTraxx_SubFin': r'iTraxx SubFin',
    'iTraxx_Xover': r'iTraxx Xover',
    'iTraxx_Combined': r'iTraxx \textit{combined}',
    'CDS_Bond_Basis': r'CDS\textit{-}Bond Basis'
}

EURIBOR_FILE = "Euribor1m.xlsx"
INCLUDE_ALPHA_COMPARISON = True

REGRESSION_FILES = {
    'FamaFrench': 'fama_french_regressions_monthly.csv',
    'Duarte': 'duarte_regressions_monthly.csv',
    'ActiveFI': 'active_fi_illusion_regressions_monthly.csv',
    'FungHsieh': 'fung_hsieh_regressions_monthly.csv'
}

# PATHS
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# CARICA EURIBOR
print("=" * 80)
print("MPPM ANALYSIS - MANIPULATION-PROOF PERFORMANCE MEASURE")
print("Reference: Goetzmann et al. (2007), Review of Financial Studies")
print("=" * 80)
print(f"\nFrequenza: {FREQUENCY.upper()}")
print(f"rho values: {RHO_VALUES}")
print("\n" + "=" * 80)
print("STEP 1: Caricamento Euribor 1M (risk-free rate)")
print("=" * 80)

euribor_path = EXTERNAL_DATA_DIR / EURIBOR_FILE
if not euribor_path.exists():
    print(f"ERRORE: File non trovato: {euribor_path}")
    exit()

euribor_df = pd.read_excel(euribor_path, skiprows=7, header=None, usecols=[0, 1], 
                            names=['Date', 'Euribor1M'], parse_dates=['Date'])
euribor_df.set_index('Date', inplace=True)
euribor_df = euribor_df.dropna()

print(f"Euribor caricato: {len(euribor_df)} osservazioni")
print(f"   Periodo: {euribor_df.index.min()} -> {euribor_df.index.max()}")
print(f"   Euribor medio: {euribor_df['Euribor1M'].mean():.2f}%")

euribor_df['rf_daily_pct'] = euribor_df['Euribor1M'] * (1/360)
print(f"Risk-free daily: {euribor_df['rf_daily_pct'].mean():.6f}%")

# FUNZIONI
def load_alpha_from_regressions(strategy_name):
    alphas = {}
    for model_name, filename in REGRESSION_FILES.items():
        filepath = TABLES_DIR / filename
        if not filepath.exists():
            continue
        try:
            df = pd.read_csv(filepath)
            strategy_variants = [strategy_name, strategy_name.replace('_', '-'), 
                               strategy_name.replace('_', ' '), strategy_name.lower(), 
                               strategy_name.upper()]
            strategy_row = None
            for variant in strategy_variants:
                mask = df['Strategy'].str.contains(variant, case=False, na=False)
                if mask.any():
                    strategy_row = df[mask].iloc[0]
                    break
            if strategy_row is None:
                continue
            if 'Alpha' in strategy_row.index:
                alpha = strategy_row['Alpha']
            elif 'alpha' in strategy_row.index:
                alpha = strategy_row['alpha']
            else:
                continue
            alphas[model_name] = alpha
        except Exception as e:
            print(f"      Could not load {model_name} alpha: {e}")
            continue
    return alphas if alphas else None

def calculate_mppm(returns_dec, rf_dec, rho, delta_t):
    T = len(returns_dec)
    total_return = returns_dec + rf_dec
    ratio = (1 + total_return) / (1 + rf_dec)
    power_term = ratio ** (1 - rho)
    mean_power = power_term.mean()
    theta_hat = (1 / ((1 - rho) * delta_t)) * np.log(mean_power)
    theta_hat_pct = theta_hat * 100
    return theta_hat_pct

# CALCOLI
all_results = []
print("\n" + "=" * 80)
print("STEP 2: Calcolo MPPM per tutte le strategie")
print("=" * 80)

for strategy_name, strategy_path in STRATEGIES.items():
    print(f"\n{strategy_name}...", end=" ")
    full_strategy_path = RESULTS_DIR / strategy_path
    if not full_strategy_path.exists():
        print(f"File non trovato - skip")
        continue
    
    strategy_df = pd.read_csv(full_strategy_path, index_col=0, parse_dates=True)
    if 'index_return' not in strategy_df.columns:
        print(f"Colonna 'index_return' non trovata - skip")
        continue
    
    strategy_returns_daily = strategy_df['index_return'].copy().dropna()
    data_daily = pd.DataFrame({'strategy_return': strategy_returns_daily, 
                               'rf': euribor_df['rf_daily_pct']}).dropna()
    
    if len(data_daily) == 0:
        print(f"Nessuna data in comune - skip")
        continue
    
    data_daily['strategy_return_dec'] = data_daily['strategy_return'] / 100
    data_daily['rf_dec'] = data_daily['rf'] / 100
    
    if FREQUENCY == "weekly":
        data_resampled = pd.DataFrame({
            'strategy_return_dec': (1 + data_daily['strategy_return_dec']).resample('W-FRI').prod() - 1,
            'rf_dec': (1 + data_daily['rf_dec']).resample('W-FRI').prod() - 1})
        Delta_t = 1 / 52
        freq_label = "weekly"
        periods_per_year = 52
    elif FREQUENCY == "monthly":
        data_resampled = pd.DataFrame({
            'strategy_return_dec': (1 + data_daily['strategy_return_dec']).resample('M').prod() - 1,
            'rf_dec': (1 + data_daily['rf_dec']).resample('M').prod() - 1})
        Delta_t = 1 / 12
        freq_label = "monthly"
        periods_per_year = 12
    else:
        data_resampled = data_daily[['strategy_return_dec', 'rf_dec']].copy()
        Delta_t = 1 / 252
        freq_label = "daily"
        periods_per_year = 252
    
    data_resampled = data_resampled.dropna()
    data_resampled['strategy_return_pct'] = data_resampled['strategy_return_dec'] * 100
    data_resampled['rf_pct'] = data_resampled['rf_dec'] * 100
    
    avg_return_period = data_resampled['strategy_return_pct'].mean()
    avg_return_annual = avg_return_period * periods_per_year
    vol_period = data_resampled['strategy_return_pct'].std()
    vol_annual = vol_period * np.sqrt(periods_per_year)
    sharpe_annual = avg_return_annual / vol_annual if vol_annual > 0 else 0
    
    alphas_dict = None
    if INCLUDE_ALPHA_COMPARISON:
        alphas_dict = load_alpha_from_regressions(strategy_name)
        if alphas_dict:
            print(f"   [Alpha: {', '.join([f'{k}={v:.2f}' for k, v in alphas_dict.items()])}]", end=" ")
    
    for rho in RHO_VALUES:
        theta = calculate_mppm(data_resampled['strategy_return_dec'], 
                              data_resampled['rf_dec'], rho, Delta_t)
        result_dict = {'Strategy': strategy_name, 
                      'Strategy_LaTeX': STRATEGY_NAMES_LATEX.get(strategy_name, strategy_name),
                      'rho': rho, 'MPPM': theta, 'Avg_Return': avg_return_annual,
                      'Volatility': vol_annual, 'Sharpe': sharpe_annual, 
                      'N_obs': len(data_resampled)}
        if alphas_dict:
            for model_name, alpha_value in alphas_dict.items():
                result_dict[f'Alpha_{model_name}'] = alpha_value
        all_results.append(result_dict)
    print(f"OK")

# TABELLE
print("\n" + "=" * 80)
print("STEP 3: Generazione tabelle LaTeX (publication-ready)")
print("=" * 80)

if not all_results:
    print("\nNessun risultato disponibile!")
    exit()

results_df = pd.DataFrame(all_results)
print(f"\nRisultati calcolati: {len(results_df['Strategy'].unique())} strategie x {len(RHO_VALUES)} rho")

latex_path = TABLES_DIR / f"MPPM_analysis_{freq_label}_all.tex"

with open(latex_path, 'w', encoding='utf-8') as f:
    f.write("% MPPM (MANIPULATION-PROOF PERFORMANCE MEASURE)\n")
    f.write("% Goetzmann et al. (2007), Review of Financial Studies\n")
    f.write(f"% Frequency: {freq_label.capitalize()}\n")
    f.write(f"% rho values: {RHO_VALUES}\n")
    f.write("% TABLES: (1) MPPM by rho with Sharpe | (2) MPPM vs Alpha (optional)\n")
    f.write("% REQUIRED PACKAGES: \\usepackage{booktabs} \\usepackage{threeparttable}\n\n")
    
    # TABLE 1
    pivot_data = []
    for strategy in results_df['Strategy'].unique():
        strategy_data = results_df[results_df['Strategy'] == strategy]
        strategy_latex = strategy_data['Strategy_LaTeX'].iloc[0]
        row = {'Strategy': strategy_latex}
        for rho in RHO_VALUES:
            rho_data = strategy_data[strategy_data['rho'] == rho].iloc[0]
            row[f'MPPM_rho{rho}'] = rho_data['MPPM']
        rho3_data = strategy_data[strategy_data['rho'] == 3].iloc[0]
        row['Avg_Return'] = rho3_data['Avg_Return']
        row['Volatility'] = rho3_data['Volatility']
        row['Sharpe'] = rho3_data['Sharpe']
        row['N_obs'] = rho3_data['N_obs']
        pivot_data.append(row)
    
    pivot_df = pd.DataFrame(pivot_data)
    pivot_df = pivot_df.sort_values('MPPM_rho3', ascending=False)
    
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{MPPM Index by Risk Aversion Parameter (" + freq_label.capitalize() + " Frequency)}\n")
    f.write("\\label{tab:mppm_by_rho_" + freq_label + "}\n")
    f.write("\\begin{threeparttable}\n")
    f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
    f.write("Strategy & \\multicolumn{3}{c}{MPPM (\\% p.a.)} & Avg Return & Vol & Sharpe & N \\\\\n")
    f.write("\\cmidrule(lr){2-4}\n")
    f.write("& $\\rho=2$ & $\\rho=3$ & $\\rho=4$ & (\\% p.a.) & (\\% p.a.) & & \\\\\n\\midrule\n")
    
    for _, row in pivot_df.iterrows():
        f.write(f"{row['Strategy']} & {row['MPPM_rho2']:.2f} & {row['MPPM_rho3']:.2f} & ")
        f.write(f"{row['MPPM_rho4']:.2f} & {row['Avg_Return']:.2f} & {row['Volatility']:.2f} & ")
        f.write(f"{row['Sharpe']:.2f} & {int(row['N_obs'])} \\\\\n")
    
    f.write("\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}\n\\footnotesize\n")
    f.write("\\item \\textit{Note:} This table reports the Manipulation-Proof Performance Measure (MPPM) ")
    f.write("as defined in Goetzmann et al. (2007) for fixed income arbitrage strategies. ")
    f.write("The MPPM is calculated using " + freq_label + " returns and reported as annualized percentage. ")
    f.write("All metrics are annualized. Sample period varies by strategy.\n")
    f.write("\\end{tablenotes}\n\\end{threeparttable}\n\\end{table}\n\n")
    
    # TABLE 2 (Alpha comparison - optional)
    if INCLUDE_ALPHA_COMPARISON:
        alpha_cols = [col for col in results_df.columns if col.startswith('Alpha_')]
        if alpha_cols:
            comparison_df = results_df[results_df['rho'] == 3].copy().sort_values('MPPM', ascending=False)
            table_cols = ['Strategy_LaTeX', 'MPPM', 'Sharpe'] + alpha_cols
            table_df = comparison_df[table_cols].copy()
            
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{MPPM vs Factor Model Alphas ($\\rho=3$, " + freq_label.capitalize() + " Frequency)}\n")
            f.write("\\label{tab:mppm_alpha_comparison_" + freq_label + "}\n")
            f.write("\\begin{threeparttable}\n")
            
            n_alpha_models = len(alpha_cols)
            col_format = 'l' + 'r' * (2 + n_alpha_models)
            f.write(f"\\begin{{tabular}}{{{col_format}}}\n\\toprule\nStrategy & MPPM & Sharpe")
            
            for col in alpha_cols:
                model_name = col.replace('Alpha_', '')
                if model_name == 'FamaFrench':
                    model_name = 'FF'
                elif model_name == 'FungHsieh':
                    model_name = 'FH'
                elif model_name == 'ActiveFI':
                    model_name = 'AFI'
                f.write(f" & $\\alpha^{{{model_name}}}$")
            f.write(" \\\\\n & (\\% p.a.) &")
            for _ in range(n_alpha_models):
                f.write(" & (\\% p.a.)")
            f.write(" \\\\\n\\midrule\n")
            
            for _, row in table_df.iterrows():
                f.write(f"{row['Strategy_LaTeX']} & {row['MPPM']:.2f} & {row['Sharpe']:.2f}")
                for col in alpha_cols:
                    if pd.notna(row[col]):
                        f.write(f" & {row[col]:.2f}")
                    else:
                        f.write(" & --")
                f.write(" \\\\\n")
            
            f.write("\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}\n\\footnotesize\n")
            f.write("\\item \\textit{Note:} This table compares MPPM with alphas from factor model regressions. ")
            f.write("All alphas are annualized. -- indicates alpha not available.\n")
            f.write("\\end{tablenotes}\n\\end{threeparttable}\n\\end{table}\n")
            print("Table 2: MPPM vs Alpha comparison generated")

print(f"File salvato: {latex_path.name}")

# SUMMARY
print("\n" + "=" * 80)
print("MPPM RESULTS SUMMARY (ALL METRICS ANNUALIZED)")
print("=" * 80)
print(f"\nMPPM by Risk Aversion (rho):")
print("=" * 80)
print(f"{'Strategy':<20} {'rho=2':>8} {'rho=3':>8} {'rho=4':>8} {'Avg Ret':>8} {'Vol':>8} {'Sharpe':>8} {'N':>6}")
print("=" * 80)

for strategy in pivot_df['Strategy'].unique():
    row = pivot_df[pivot_df['Strategy'] == strategy].iloc[0]
    print(f"{strategy:<20} {row['MPPM_rho2']:>8.2f} {row['MPPM_rho3']:>8.2f} {row['MPPM_rho4']:>8.2f} "
          f"{row['Avg_Return']:>8.2f} {row['Volatility']:>8.2f} {row['Sharpe']:>8.2f} {int(row['N_obs']):>6}")

# Use pivot_df for best/worst (already sorted by MPPM rho=3)
best_idx = 0
worst_idx = len(pivot_df) - 1
best = pivot_df.iloc[best_idx]
worst = pivot_df.iloc[worst_idx]
print(f"\nBEST: {best['Strategy']}: MPPM = {best['MPPM_rho3']:.2f}% p.a.")
print(f"WORST: {worst['Strategy']}: MPPM = {worst['MPPM_rho3']:.2f}% p.a.")
print(f"\nMPPM RANGE: {results_df['MPPM'].min():.2f}% to {results_df['MPPM'].max():.2f}% p.a.")

print("\n" + "=" * 80)
print("MPPM ANALYSIS COMPLETED")
print("=" * 80)
print(f"\nOutput: {latex_path.relative_to(PROJECT_ROOT)}")
print(f"Tables: MPPM by rho (with Sharpe) | MPPM vs Alpha (optional)")
print(f"\nLaTeX requirements: \\usepackage{{booktabs}} \\usepackage{{threeparttable}}")
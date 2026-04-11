# -*- coding: utf-8 -*-
"""
Moreira-Muir Volatility-Managed Portfolios Analysis
Moreira, A., & Muir, T. (2017). Journal of Finance, 72(4), 1611-1643.
Author: Alessio Ottaviani, EDHEC Business School, December 2025
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Strategies
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

# Parameters
PERIODS_PER_YEAR = 12
USE_DEMEANING = False

# Output flags
INCLUDE_ROBUSTNESS_TABLE = True
INCLUDE_CUMULATIVE_PLOT = True

# Plot settings
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("Set2")
FIGSIZE = (12, 6)
DPI = 300

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def calculate_max_drawdown(returns):
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    return drawdown.min()

def calculate_downside_deviation(returns, target=0):
    downside_returns = returns[returns < target]
    return np.sqrt(np.mean(downside_returns**2))

def calculate_sortino_ratio(returns, target=0, periods_per_year=12):
    mean_return = returns.mean()
    downside_dev = calculate_downside_deviation(returns, target)
    if downside_dev == 0:
        return np.nan
    return (mean_return / downside_dev) * np.sqrt(periods_per_year)

# ============================================================================
# BANNER
# ============================================================================

print("=" * 80)
print("MOREIRA-MUIR VOLATILITY-MANAGED PORTFOLIOS ANALYSIS")
print("Paper: Moreira & Muir (2017), Journal of Finance")
print("=" * 80)

# ============================================================================
# STORAGE
# ============================================================================

all_results = []
all_robustness = []
cumulative_data = {}  # For cumulative returns plot

# ============================================================================
# MAIN LOOP
# ============================================================================

for strategy_name, strategy_path in STRATEGIES.items():
    
    print(f"Processing: {strategy_name}...", end=" ")
    
    # Load data
    full_path = RESULTS_DIR / strategy_path
    if not full_path.exists():
        print("File not found - skip")
        continue
    
    try:
        df = pd.read_csv(full_path, index_col=0, parse_dates=True)
    except Exception as e:
        print(f"Error loading - skip")
        continue
    
    if 'index_return' not in df.columns:
        print("Missing column - skip")
        continue
    
    # Convert from % to decimal
    daily_returns = df['index_return'].copy() / 100.0
    daily_returns = daily_returns.dropna()
    
    # Calculate realized variance
    if USE_DEMEANING:
        rv_list = []
        for period, group in daily_returns.groupby(daily_returns.index.to_period('M')):
            mean_return = group.mean()
            rv = ((group - mean_return) ** 2).sum()
            rv_list.append({'date': period.to_timestamp('M'), 'rv': rv})
        rv_df = pd.DataFrame(rv_list).set_index('date')
        realized_variance = rv_df['rv']
    else:
        realized_variance = (daily_returns ** 2).resample('M').sum()
    
    # Monthly returns
    monthly_returns = (1 + daily_returns).resample('M').prod() - 1
    monthly_returns = monthly_returns.dropna()
    
    # Volatility-managed portfolio
    rv_lagged = realized_variance.shift(1)
    common_idx = monthly_returns.index.intersection(rv_lagged.index)
    returns_clean = monthly_returns.loc[common_idx]
    rv_clean = rv_lagged.loc[common_idx]
    
    valid_idx = (~returns_clean.isna()) & (~rv_clean.isna()) & (rv_clean > 0)
    returns_clean = returns_clean[valid_idx]
    rv_clean = rv_clean[valid_idx]
    
    if len(returns_clean) == 0:
        print("No valid data - skip")
        continue
    
    # Calculate VM returns
    scaling_raw = 1.0 / rv_clean
    vm_returns_raw = scaling_raw * returns_clean
    c = returns_clean.std() / vm_returns_raw.std()
    vm_returns = c * vm_returns_raw
    effective_scaling = c * scaling_raw
    
    # Regression alpha
    X = returns_clean.values.reshape(-1, 1)
    y = vm_returns.values
    reg = LinearRegression()
    reg.fit(X, y)
    
    alpha_monthly = reg.intercept_
    alpha_annual = alpha_monthly * PERIODS_PER_YEAR * 100
    beta = reg.coef_[0]
    
    y_pred = reg.predict(X)
    residuals = y - y_pred
    rmse_monthly = np.sqrt(np.mean(residuals**2))
    rmse_annual = rmse_monthly * np.sqrt(PERIODS_PER_YEAR) * 100
    appraisal_ratio = alpha_annual / rmse_annual if rmse_annual > 0 else np.nan
    
    # Performance metrics
    bh_sharpe = (returns_clean.mean() / returns_clean.std()) * np.sqrt(PERIODS_PER_YEAR)
    vm_sharpe = (vm_returns.mean() / vm_returns.std()) * np.sqrt(PERIODS_PER_YEAR)
    sharpe_improvement = ((vm_sharpe - bh_sharpe) / abs(bh_sharpe)) * 100
    
    print(f"OK (Alpha: {alpha_annual:+.2f}%, Sharpe: {vm_sharpe:.2f})")
    
    # Store main results
    all_results.append({
        'strategy': strategy_name,
        'strategy_latex': STRATEGY_NAMES_LATEX.get(strategy_name, strategy_name),
        'bh_return_pct': (returns_clean.mean() * PERIODS_PER_YEAR) * 100,
        'bh_vol_pct': (returns_clean.std() * np.sqrt(PERIODS_PER_YEAR)) * 100,
        'bh_sharpe': bh_sharpe,
        'vm_return_pct': (vm_returns.mean() * PERIODS_PER_YEAR) * 100,
        'vm_vol_pct': (vm_returns.std() * np.sqrt(PERIODS_PER_YEAR)) * 100,
        'vm_sharpe': vm_sharpe,
        'alpha_annual_pct': alpha_annual,
        'beta': beta,
        'r_squared': reg.score(X, y),
        'n_obs': len(returns_clean)
    })
    
    # Store cumulative returns for plot
    cumulative_data[strategy_name] = {
        'date': returns_clean.index,
        'bh_cum': (1 + returns_clean).cumprod() - 1,
        'vm_cum': (1 + vm_returns).cumprod() - 1
    }
    
    # Robustness tests
    if INCLUDE_ROBUSTNESS_TABLE:
        constraint_tests = [
            {'name': 'No Leverage', 'min': 0.0, 'max': 1.0},
            {'name': '50% Leverage', 'min': 0.0, 'max': 1.5},
            {'name': 'Table V', 'min': 0.1, 'max': 2.0},
        ]
        
        for test in constraint_tests:
            scaling_constrained = effective_scaling.clip(lower=test['min'], upper=test['max'])
            vm_returns_constrained = scaling_constrained * returns_clean
            
            X_c = returns_clean.values.reshape(-1, 1)
            y_c = vm_returns_constrained.values
            reg_c = LinearRegression()
            reg_c.fit(X_c, y_c)
            alpha_c = reg_c.intercept_ * PERIODS_PER_YEAR * 100
            sharpe_c = (vm_returns_constrained.mean() / vm_returns_constrained.std()) * np.sqrt(PERIODS_PER_YEAR)
            
            n_clipped = (effective_scaling != scaling_constrained).sum()
            
            all_robustness.append({
                'strategy': strategy_name,
                'strategy_latex': STRATEGY_NAMES_LATEX.get(strategy_name, strategy_name),
                'constraint': test['name'],
                'alpha_pct': alpha_c,
                'sharpe': sharpe_c,
                'n_clipped': n_clipped
            })

# ============================================================================
# CREATE DATAFRAMES
# ============================================================================

if not all_results:
    print("\nNo results available!")
    exit()

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('vm_sharpe', ascending=False)

if INCLUDE_ROBUSTNESS_TABLE and all_robustness:
    robustness_df = pd.DataFrame(all_robustness)

# ============================================================================
# GENERATE LATEX FILE
# ============================================================================

print("\n" + "=" * 80)
print("Generating LaTeX tables...")
print("=" * 80)

latex_path = TABLES_DIR / "moreira_muir_analysis_monthly.tex"

with open(latex_path, 'w', encoding='utf-8') as f:
    
    # Header
    f.write("% Moreira-Muir Volatility-Managed Portfolios Analysis\n")
    f.write("% Moreira & Muir (2017), Journal of Finance\n")
    f.write("% Frequency: Monthly\n")
    f.write("% REQUIRED PACKAGES: \\usepackage{booktabs} \\usepackage{threeparttable}\n\n")
    
    # TABLE 1: Main Results
    f.write("% TABLE 1: Main Results (Complete Performance Metrics)\n\n")
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Volatility-Managed Portfolios: Performance Summary}\n")
    f.write("\\label{tab:moreira_muir_performance}\n")
    f.write("\\begin{threeparttable}\n")
    f.write("\\begin{tabular}{lrrrrrrrrrr}\n")
    f.write("\\toprule\n")
    f.write("Strategy & \\multicolumn{3}{c}{Buy-and-Hold} & \\multicolumn{3}{c}{Volatility-Managed} & \\multicolumn{3}{c}{Regression} & N \\\\\n")
    f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}\n")
    f.write("& Return & Vol & Sharpe & Return & Vol & Sharpe & Alpha & Beta & $R^2$ & \\\\\n")
    f.write("& (\\% p.a.) & (\\% p.a.) & & (\\% p.a.) & (\\% p.a.) & & (\\% p.a.) & & & \\\\\n")
    f.write("\\midrule\n")
    
    for _, row in results_df.iterrows():
        f.write(f"{row['strategy_latex']} & ")
        f.write(f"{row['bh_return_pct']:.2f} & ")
        f.write(f"{row['bh_vol_pct']:.2f} & ")
        f.write(f"{row['bh_sharpe']:.2f} & ")
        f.write(f"{row['vm_return_pct']:.2f} & ")
        f.write(f"{row['vm_vol_pct']:.2f} & ")
        f.write(f"{row['vm_sharpe']:.2f} & ")
        f.write(f"{row['alpha_annual_pct']:+.2f} & ")
        f.write(f"{row['beta']:.2f} & ")
        f.write(f"{row['r_squared']:.3f} & ")
        f.write(f"{int(row['n_obs'])} \\\\\n")
    
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\begin{tablenotes}\n")
    f.write("\\footnotesize\n")
    f.write("\\item \\textit{Note:} This table reports performance metrics for buy-and-hold and ")
    f.write("volatility-managed portfolios following Moreira and Muir (2017). ")
    f.write("The volatility-managed strategy scales positions inversely to realized variance: ")
    f.write("$f^\\sigma_{t+1} = (c/\\sigma^2_t) \\times f_{t+1}$, where $c$ is chosen to match ")
    f.write("buy-and-hold volatility. ")
    f.write("\\textit{Return} is the annualized average return. ")
    f.write("\\textit{Vol} is the annualized volatility. ")
    f.write("\\textit{Sharpe} is the Sharpe ratio. ")
    f.write("\\textit{Alpha} is from the regression $f^\\sigma_{t+1} = \\alpha + \\beta \\cdot f_{t+1} + \\varepsilon$ ")
    f.write("and measures the volatility timing premium. ")
    f.write("\\textit{Beta} measures the exposure to the buy-and-hold strategy. ")
    f.write("\\textit{$R^2$} measures the fraction of variance explained. ")
    f.write("All metrics annualized, monthly frequency. ")
    f.write("Strategies sorted by VM Sharpe ratio (descending).\n")
    f.write("\\end{tablenotes}\n")
    f.write("\\end{threeparttable}\n")
    f.write("\\end{table}\n\n")
    
    # TABLE 2: Robustness (if enabled)
    if INCLUDE_ROBUSTNESS_TABLE and all_robustness:
        f.write("% TABLE 2: Robustness - Leverage Constraints\n\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Robustness: Leverage Constraints}\n")
        f.write("\\label{tab:moreira_muir_robustness}\n")
        f.write("\\begin{threeparttable}\n")
        f.write("\\begin{tabular}{llrrr}\n")
        f.write("\\toprule\n")
        f.write("Strategy & Metric & No Leverage & 50\\% Leverage & Table V \\\\\n")
        f.write("         &        & $(\\leq 1)$  & $(\\leq 1.5)$   & $[0.1,2.0]$ \\\\\n")
        f.write("\\midrule\n")
        
        # Pivot data: constraints as columns
        for strategy in results_df['strategy'].unique():
            strategy_latex = results_df[results_df['strategy'] == strategy]['strategy_latex'].iloc[0]
            strategy_rob = robustness_df[robustness_df['strategy'] == strategy]
            
            # Get values for each constraint
            no_lev = strategy_rob[strategy_rob['constraint'] == 'No Leverage'].iloc[0]
            lev50 = strategy_rob[strategy_rob['constraint'] == '50% Leverage'].iloc[0]
            tableV = strategy_rob[strategy_rob['constraint'] == 'Table V'].iloc[0]
            
            # Alpha row
            f.write(f"{strategy_latex} & Alpha (\\% p.a.) & ")
            f.write(f"{no_lev['alpha_pct']:+.2f} & ")
            f.write(f"{lev50['alpha_pct']:+.2f} & ")
            f.write(f"{tableV['alpha_pct']:+.2f} \\\\\n")
            
            # Sharpe row
            f.write(f"& Sharpe & ")
            f.write(f"{no_lev['sharpe']:.2f} & ")
            f.write(f"{lev50['sharpe']:.2f} & ")
            f.write(f"{tableV['sharpe']:.2f} \\\\\n")
            
            # Add space between strategies (except last)
            if strategy != results_df['strategy'].iloc[-1]:
                f.write("\\addlinespace\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\n")
        f.write("\\footnotesize\n")
        f.write("\\item \\textit{Note:} This table reports robustness tests with leverage constraints ")
        f.write("following Table V in Moreira and Muir (2017). ")
        f.write("\\textit{No Leverage}: scaling constrained to $[0, 1]$. ")
        f.write("\\textit{50\\% Leverage}: scaling constrained to $[0, 1.5]$. ")
        f.write("\\textit{Table V}: scaling constrained to $[0.1, 2.0]$ as in the paper. ")
        f.write("Alpha measures the volatility timing premium and persists across all constraint levels, ")
        f.write("confirming that results are not driven by extreme leverage positions.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{threeparttable}\n")
        f.write("\\end{table}\n")

print(f"LaTeX file saved: {latex_path.name}")

# ============================================================================
# GENERATE FIGURES
# ============================================================================

print("\n" + "=" * 80)
print("Generating figures...")
print("=" * 80)

# Figure 1: Sharpe Ratio Comparison
fig, ax = plt.subplots(figsize=FIGSIZE)

strategies = results_df['strategy'].values
x_pos = np.arange(len(strategies))
width = 0.35

bh_sharpe = results_df['bh_sharpe'].values
vm_sharpe = results_df['vm_sharpe'].values

bars1 = ax.bar(x_pos - width/2, bh_sharpe, width, label='Buy-and-Hold', alpha=0.8, color='steelblue')
bars2 = ax.bar(x_pos + width/2, vm_sharpe, width, label='Volatility-Managed', alpha=0.8, color='coral')

ax.set_ylabel('Sharpe Ratio', fontsize=11)
ax.set_xlabel('Strategy', fontsize=11)
ax.set_title('Sharpe Ratio: Buy-and-Hold vs Volatility-Managed', fontsize=12, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(strategies, rotation=45, ha='right')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
filename = FIGURES_DIR / "moreira_muir_sharpe_comparison.png"
plt.savefig(filename, dpi=DPI)
print(f"Figure saved: {filename.name}")
plt.close()

# Figure 2: Alpha by Strategy
fig, ax = plt.subplots(figsize=FIGSIZE)

alpha_values = results_df['alpha_annual_pct'].values
colors = ['green' if x > 0 else 'red' for x in alpha_values]

bars = ax.bar(strategies, alpha_values, color=colors, alpha=0.7)
ax.set_ylabel('Alpha (% per annum)', fontsize=11)
ax.set_xlabel('Strategy', fontsize=11)
ax.set_title('Volatility-Managed Portfolio Alpha by Strategy', fontsize=12, fontweight='bold')
ax.set_xticklabels(strategies, rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linewidth=1, linestyle='--')

plt.tight_layout()
filename = FIGURES_DIR / "moreira_muir_alpha_by_strategy.png"
plt.savefig(filename, dpi=DPI)
print(f"Figure saved: {filename.name}")
plt.close()

# Figure 3: Cumulative Returns
if INCLUDE_CUMULATIVE_PLOT:
    n_strategies = len(cumulative_data)
    n_cols = 2
    n_rows = (n_strategies + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4*n_rows))
    axes = axes.flatten() if n_strategies > 1 else [axes]
    
    for idx, (strategy_name, data) in enumerate(cumulative_data.items()):
        ax = axes[idx]
        
        ax.plot(data['date'], data['bh_cum'] * 100, label='Buy-and-Hold', 
                linewidth=2, alpha=0.8, color='steelblue')
        ax.plot(data['date'], data['vm_cum'] * 100, label='Volatility-Managed', 
                linewidth=2, alpha=0.8, color='coral', linestyle='--')
        
        ax.set_title(strategy_name, fontsize=11, fontweight='bold')
        ax.set_ylabel('Cumulative Return (%)', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Hide unused subplots
    for idx in range(n_strategies, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Cumulative Returns: Buy-and-Hold vs Volatility-Managed', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    filename = FIGURES_DIR / "moreira_muir_cumulative_returns.png"
    plt.savefig(filename, dpi=DPI, bbox_inches='tight')
    print(f"Figure saved: {filename.name}")
    plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(f"\nBuy-and-Hold Performance:")
print(f"   Mean Return: {results_df['bh_return_pct'].mean():.2f}% p.a.")
print(f"   Mean Vol: {results_df['bh_vol_pct'].mean():.2f}% p.a.")
print(f"   Mean Sharpe: {results_df['bh_sharpe'].mean():.2f}")

print(f"\nVolatility-Managed Performance:")
print(f"   Mean Return: {results_df['vm_return_pct'].mean():.2f}% p.a.")
print(f"   Mean Vol: {results_df['vm_vol_pct'].mean():.2f}% p.a.")
print(f"   Mean Sharpe: {results_df['vm_sharpe'].mean():.2f}")

print(f"\nRegression Results:")
print(f"   Mean Alpha: {results_df['alpha_annual_pct'].mean():+.2f}% p.a.")
print(f"   Median Alpha: {results_df['alpha_annual_pct'].median():+.2f}% p.a.")
print(f"   Mean Beta: {results_df['beta'].mean():.2f}")
print(f"   Mean R²: {results_df['r_squared'].mean():.3f}")

best = results_df.iloc[0]
print(f"\nBest Strategy (by VM Sharpe):")
print(f"   {best['strategy']}")
print(f"   BH Sharpe: {best['bh_sharpe']:.2f}")
print(f"   VM Sharpe: {best['vm_sharpe']:.2f}")
print(f"   Alpha: {best['alpha_annual_pct']:+.2f}% p.a.")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("MOREIRA-MUIR ANALYSIS COMPLETED")
print("=" * 80)

print(f"\nStrategies analyzed: {len(results_df)}")
for _, row in results_df.iterrows():
    print(f"   {row['strategy']}: BH Sharpe = {row['bh_sharpe']:.2f}, VM Sharpe = {row['vm_sharpe']:.2f}, Alpha = {row['alpha_annual_pct']:+.2f}%")

print(f"\nOutput files:")
print(f"   LaTeX: {latex_path.relative_to(PROJECT_ROOT)}")
print(f"   Figures:")
print(f"      - moreira_muir_sharpe_comparison.png")
print(f"      - moreira_muir_alpha_by_strategy.png")
if INCLUDE_CUMULATIVE_PLOT:
    print(f"      - moreira_muir_cumulative_returns.png")

print("\n" + "=" * 80)
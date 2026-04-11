"""
================================================================================
Script 4: Visualizzazione Risultati CDS-Bond Basis Trading
================================================================================
Questo script crea grafici e analisi dei risultati della simulazione.

GRAFICI:
1. Equity curve
2. Distribuzione P&L per trade
3. Durata trade
4. Timeline dei trade aperti
5. Exit reasons
6. Rolling statistics
7. Drawdown analysis
8. P&L components (Carry vs Capital Gain)
9. Summary statistics
10. Top Ticker performance (NEW)
11. Entry basis distribution (NEW)
12. Basis compression analysis (NEW)

Author: Alessio Ottaviani
Date: November 2025
Institution: EDHEC Business School - PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D

# Configura stile grafici
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "cds_bond_basis"

# ============================================================================
# STEP 1: CARICA RISULTATI
# ============================================================================

print("=" * 80)
print("STEP 1: Caricamento risultati")
print("=" * 80)

# Carica trades log
trades_df = pd.read_csv(RESULTS_DIR / "trades_log.csv", parse_dates=['entry_date', 'exit_date', 'maturity_date'])
print(f"✅ Trades: {len(trades_df)}")

# Carica indice
index_df = pd.read_csv(RESULTS_DIR / "index_daily.csv", index_col=0, parse_dates=True)
print(f"✅ Indice: {len(index_df)} giorni")

# ============================================================================
# STEP 2: EQUITY CURVE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 2: Grafico Equity Curve")
print("=" * 80)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Interpolate NaN gaps for continuous line
index_value_clean = index_df['index_value'].interpolate(method='time')

# Subplot 1: Index Value
ax1.plot(index_value_clean.index, index_value_clean, linewidth=2, color='darkblue', label='Index Value')
ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('CDS-Bond Basis Strategy - Equity Curve', fontsize=16, fontweight='bold')
ax1.set_ylabel('Index Value (Base 100)', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# Subplot 2: Drawdown
running_max = index_value_clean.cummax()
drawdown_pct = (index_value_clean - running_max) / running_max * 100
ax2.fill_between(drawdown_pct.index, drawdown_pct, 0, color='red', alpha=0.3)
ax2.plot(drawdown_pct.index, drawdown_pct, linewidth=1, color='darkred')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_title('Drawdown', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Drawdown (%)', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / '01_equity_curve.pdf', bbox_inches='tight')
print(f"💾 Salvato: 01_equity_curve.pdf")
plt.close()

# ============================================================================
# STEP 3: DISTRIBUZIONE P&L
# ============================================================================

print("\n" + "=" * 80)
print("STEP 3: Distribuzione P&L")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Histogram P&L
ax = axes[0, 0]
trades_df['cumulative_pnl'].hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.axvline(x=trades_df['cumulative_pnl'].mean(), color='green', linestyle='--', linewidth=2, 
           label=f'Mean: {trades_df["cumulative_pnl"].mean():.2f}%')
ax.set_title('Distribuzione P&L per Trade', fontsize=12, fontweight='bold')
ax.set_xlabel('P&L (%)', fontsize=10)
ax.set_ylabel('Frequenza', fontsize=10)
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 2: Boxplot P&L (solo 1 box, non abbiamo Long/Short)
ax = axes[0, 1]
bp = ax.boxplot([trades_df['cumulative_pnl']], positions=[1], widths=0.6, patch_artist=True,
                showmeans=True, meanline=True)
bp['boxes'][0].set_facecolor('lightblue')
ax.set_xticklabels(['All Trades'])
ax.set_title('Distribuzione P&L', fontsize=12, fontweight='bold')
ax.set_ylabel('P&L (%)', fontsize=10)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

# Subplot 3: Durata Trade
ax = axes[1, 0]
trades_df['duration_days'].hist(bins=50, ax=ax, color='coral', edgecolor='black', alpha=0.7)
ax.axvline(x=trades_df['duration_days'].mean(), color='darkred', linestyle='--', linewidth=2, 
           label=f'Mean: {trades_df["duration_days"].mean():.0f} giorni')
ax.set_title('Distribuzione Durata Trade', fontsize=12, fontweight='bold')
ax.set_xlabel('Durata (giorni)', fontsize=10)
ax.set_ylabel('Frequenza', fontsize=10)
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 4: P&L vs Durata
ax = axes[1, 1]
scatter = ax.scatter(trades_df['duration_days'], trades_df['cumulative_pnl'], 
                     c='steelblue', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('P&L vs Durata Trade', fontsize=12, fontweight='bold')
ax.set_xlabel('Durata (giorni)', fontsize=10)
ax.set_ylabel('P&L (%)', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / '02_pnl_distribution.pdf', bbox_inches='tight')
print(f"💾 Salvato: 02_pnl_distribution.pdf")
plt.close()

# ============================================================================
# STEP 4: TIMELINE TRADE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 4: Timeline Trade")
print("=" * 80)

# Conta numero trade aperti per ogni giorno
all_dates = pd.date_range(trades_df['entry_date'].min(), trades_df['exit_date'].max(), freq='D')
open_count = pd.Series(0, index=all_dates)

for _, trade in trades_df.iterrows():
    trade_dates = pd.date_range(trade['entry_date'], trade['exit_date'], freq='D')
    open_count[trade_dates] += 1

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(open_count.index, open_count.values, linewidth=1.5, color='darkblue')
ax.fill_between(open_count.index, 0, open_count.values, alpha=0.3, color='lightblue')
ax.set_title('Numero Trade Aperti nel Tempo', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Trade Aperti', fontsize=12)
ax.grid(True, alpha=0.3)

# Aggiungi statistiche
avg_open = open_count.mean()
max_open = open_count.max()
ax.axhline(y=avg_open, color='green', linestyle='--', linewidth=1.5, 
           label=f'Media: {avg_open:.1f} trade', alpha=0.7)
ax.axhline(y=max_open, color='red', linestyle='--', linewidth=1.5, 
           label=f'Max: {max_open:.0f} trade', alpha=0.7)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '03_trade_timeline.pdf', bbox_inches='tight')
print(f"💾 Salvato: 03_trade_timeline.pdf")
plt.close()

# ============================================================================
# STEP 5: EXIT REASONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Analisi Exit Reasons")
print("=" * 80)

exit_counts = trades_df['exit_reason'].value_counts()
print("\n📊 Exit Reasons:")
print(exit_counts)

fig, ax = plt.subplots(figsize=(10, 6))
colors_map = {'TARGET_HIT': 'lightgreen', 'MATURITY': 'lightcoral', 'END_OF_SAMPLE': 'lightyellow'}
colors = [colors_map.get(reason, 'lightgray') for reason in exit_counts.index]
wedges, texts, autotexts = ax.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%',
                                    startangle=90, colors=colors)
ax.set_title('Distribuzione Exit Reasons', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '04_exit_reasons.pdf', bbox_inches='tight')
print(f"💾 Salvato: 04_exit_reasons.pdf")
plt.close()

# ============================================================================
# STEP 6: ROLLING STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Rolling Statistics")
print("=" * 80)

# Calcola rolling Sharpe (252 giorni)
window = 252  # 1 anno
rolling_mean = index_df['index_return'].rolling(window).mean()
rolling_std = index_df['index_return'].rolling(window).std()
rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(252)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Subplot 1: Rolling Return
ax1.plot(rolling_mean.index, rolling_mean * 252, linewidth=2, color='darkblue', label='Rolling Avg Return (annualized)')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('Rolling 1-Year Average Return', fontsize=14, fontweight='bold')
ax1.set_ylabel('Return (%)', fontsize=12)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Subplot 2: Rolling Sharpe
ax2.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2, color='darkgreen', label='Rolling 1-Year Sharpe')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='Sharpe = 1')
ax2.set_title('Rolling 1-Year Sharpe Ratio', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Sharpe Ratio', fontsize=12)
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / '05_rolling_stats.pdf', bbox_inches='tight')
print(f"💾 Salvato: 05_rolling_stats.pdf")
plt.close()

# ============================================================================
# STEP 7: DRAWDOWN ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Drawdown Analysis")
print("=" * 80)

# Calcola running maximum
running_max = index_df['index_value'].cummax()
drawdown = (index_df['index_value'] - running_max) / running_max * 100

# Max drawdown
max_dd = drawdown.min()
max_dd_date = drawdown.idxmin()

print(f"📉 Max Drawdown: {max_dd:.2f}% (data: {max_dd_date.strftime('%Y-%m-%d')})")

fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(drawdown.index, 0, drawdown, color='red', alpha=0.3, label='Drawdown')
ax.plot(drawdown.index, drawdown, linewidth=1, color='darkred')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=max_dd, color='red', linestyle='--', linewidth=2, label=f'Max DD: {max_dd:.2f}%')
ax.set_title('Strategy Drawdown', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Drawdown (%)', fontsize=12)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / '06_drawdown.pdf', bbox_inches='tight')
print(f"💾 Salvato: 06_drawdown.pdf")
plt.close()

# ============================================================================
# STEP 8: COMPONENTI P&L (CARRY VS CAPITAL GAIN)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Componenti P&L (Carry vs Capital Gain)")
print("=" * 80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Scatter Carry vs Capital Gain
ax1.scatter(trades_df['cumulative_capital_gain'], trades_df['cumulative_carry'], 
            c='steelblue', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('Carry vs Capital Gain', fontsize=12, fontweight='bold')
ax1.set_xlabel('Capital Gain (%)', fontsize=10)
ax1.set_ylabel('Carry (%)', fontsize=10)
ax1.grid(True, alpha=0.3)

# Subplot 2: Stacked Bar componenti medie
avg_cg = trades_df['cumulative_capital_gain'].mean()
avg_carry = trades_df['cumulative_carry'].mean()

x = [0]
width = 0.5

ax2.bar(x, [avg_cg], width, label='Capital Gain', color='steelblue')
ax2.bar(x, [avg_carry], width, bottom=[avg_cg], label='Carry', color='coral')
ax2.set_xticks(x)
ax2.set_xticklabels(['All Trades'])
ax2.set_ylabel('P&L (%)', fontsize=10)
ax2.set_title('Componenti P&L Medie', fontsize=12, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '07_pnl_components.pdf', bbox_inches='tight')
print(f"💾 Salvato: 07_pnl_components.pdf")
plt.close()

# ============================================================================
# STEP 9: SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Summary Statistics")
print("=" * 80)

# Calcola metriche chiave
total_return = index_df['cumulative_return'].iloc[-1]
annual_return = (index_df['index_return'].mean() * 252)
annual_vol = (index_df['index_return'].std() * np.sqrt(252))
sharpe = annual_return / annual_vol if annual_vol > 0 else 0

# Win rate
winning_trades = len(trades_df[trades_df['cumulative_pnl'] > 0])
win_rate = winning_trades / len(trades_df) * 100

# Avg Win/Loss
avg_win = trades_df[trades_df['cumulative_pnl'] > 0]['cumulative_pnl'].mean()
avg_loss = trades_df[trades_df['cumulative_pnl'] < 0]['cumulative_pnl'].mean()
profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf

# Calmar ratio (return / max drawdown)
calmar = abs(annual_return / max_dd) if max_dd != 0 else np.inf

summary = {
    'Total Return (%)': total_return,
    'Annualized Return (%)': annual_return,
    'Annualized Volatility (%)': annual_vol,
    'Sharpe Ratio': sharpe,
    'Calmar Ratio': calmar,
    'Max Drawdown (%)': max_dd,
    'Total Trades': len(trades_df),
    'Win Rate (%)': win_rate,
    'Avg Win (%)': avg_win,
    'Avg Loss (%)': avg_loss,
    'Profit Factor': profit_factor,
    'Avg Duration (days)': trades_df['duration_days'].mean()
}

summary_df = pd.DataFrame(summary, index=[0]).T
summary_df.columns = ['Value']

print("\n" + "=" * 80)
print("STRATEGY SUMMARY STATISTICS")
print("=" * 80)
print(summary_df.to_string())

# Salva summary
summary_df.to_csv(RESULTS_DIR / 'strategy_summary.csv')
print(f"\n💾 Salvato: strategy_summary.csv")

# ============================================================================
# STEP 10: TOP TICKER PERFORMANCE (NEW)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Top Ticker Performance")
print("=" * 80)

# Aggrega per Ticker
ticker_stats = trades_df.groupby('ticker').agg({
    'trade_id': 'count',
    'cumulative_pnl': ['mean', 'sum', 'std'],
    'duration_days': 'mean',
    'entry_basis': 'mean',
    'exit_basis': 'mean'
}).round(2)

ticker_stats.columns = ['Count', 'Avg_PnL', 'Total_PnL_raw', 'Std_PnL', 'Avg_Duration', 'Avg_Entry_Basis', 'Avg_Exit_Basis']

# Normalizza: contributo % al P&L totale della strategia
total_strategy_pnl = ticker_stats['Total_PnL_raw'].sum()
ticker_stats['Total_PnL'] = (ticker_stats['Total_PnL_raw'] / total_strategy_pnl * 100).round(2)
ticker_stats = ticker_stats.sort_values('Total_PnL', ascending=False)

print("\n📊 Top 10 Ticker per Contributo al P&L:")
print(ticker_stats.head(10).to_string())

# Salva statistiche
ticker_stats_path = RESULTS_DIR / "ticker_statistics.csv"
ticker_stats.to_csv(ticker_stats_path)
print(f"\n💾 Salvato: ticker_statistics.csv")

# Grafico Top Ticker (contributo %)
fig, ax = plt.subplots(figsize=(12, 8))
top_tickers = ticker_stats.head(15)
colors = ['green' if x > 0 else 'red' for x in top_tickers['Total_PnL']]
ax.barh(range(len(top_tickers)), top_tickers['Total_PnL'], color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(top_tickers)))
ax.set_yticklabels(top_tickers.index)
ax.set_xlabel('Contributo al P&L Totale (%)', fontsize=12)
ax.set_title('Top 15 Ticker per Contributo al P&L', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '08_top_tickers.pdf', bbox_inches='tight')
print(f"💾 Salvato: 08_top_tickers.pdf")
plt.close()

# --- Top 20 ISIN per Contributo al P&L ---
# Detect ISIN column name
isin_col = None
for col_candidate in ['isin', 'ISIN', 'bond_isin', 'Bond_ISIN', 'bond_id']:
    if col_candidate in trades_df.columns:
        isin_col = col_candidate
        break

if isin_col is not None:
    print(f"\n📊 Top 20 ISIN (colonna: '{isin_col}')")
    
    isin_stats = trades_df.groupby(isin_col).agg({
        'trade_id': 'count',
        'cumulative_pnl': ['mean', 'sum', 'std'],
        'duration_days': 'mean',
        'entry_basis': 'mean',
        'exit_basis': 'mean'
    }).round(2)
    
    isin_stats.columns = ['Count', 'Avg_PnL', 'Total_PnL_raw', 'Std_PnL', 
                           'Avg_Duration', 'Avg_Entry_Basis', 'Avg_Exit_Basis']
    
    total_pnl_isin = isin_stats['Total_PnL_raw'].sum()
    isin_stats['Total_PnL'] = (isin_stats['Total_PnL_raw'] / total_pnl_isin * 100).round(2)
    isin_stats = isin_stats.sort_values('Total_PnL', ascending=False)
    
    # Lookup ticker per ISIN
    ticker_lookup = {}
    if 'ticker' in trades_df.columns:
        for isin_val, grp in trades_df.groupby(isin_col):
            ticker_lookup[isin_val] = grp['ticker'].mode().iloc[0] if len(grp['ticker'].mode()) > 0 else ''
    
    print(isin_stats.head(20).to_string())
    
    # Salva
    isin_stats.to_csv(RESULTS_DIR / "isin_statistics.csv")
    print(f"💾 Salvato: isin_statistics.csv")
    
    # Grafico Top 20 ISIN
    fig, ax = plt.subplots(figsize=(14, 10))
    top_isins = isin_stats.head(20)
    colors = ['green' if x > 0 else 'red' for x in top_isins['Total_PnL']]
    ax.barh(range(len(top_isins)), top_isins['Total_PnL'], color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(top_isins)))
    
    # Labels: "ISIN (TICKER) [n trades]"
    labels = []
    for isin_val in top_isins.index:
        n_trades = int(top_isins.loc[isin_val, 'Count'])
        tkr = ticker_lookup.get(isin_val, '')
        if tkr:
            labels.append(f"{isin_val}  ({tkr})  [{n_trades} trades]")
        else:
            labels.append(f"{isin_val}  [{n_trades} trades]")
    
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Contributo al P&L Totale (%)', fontsize=12)
    ax.set_title('Top 20 ISIN per Contributo al P&L', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / '08b_top_isins.pdf', bbox_inches='tight')
    print(f"💾 Salvato: 08b_top_isins.pdf")
    plt.close()
else:
    print("\n⚠️ Colonna ISIN non trovata nel trades_log. Colonne disponibili:")
    print(f"   {list(trades_df.columns)}")

# ============================================================================
# STEP 11: ENTRY BASIS DISTRIBUTION (NEW)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 11: Entry Basis Distribution")
print("=" * 80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Histogram Entry Basis
ax1.hist(trades_df['entry_basis'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(x=trades_df['entry_basis'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {trades_df["entry_basis"].mean():.2f} bps')
ax1.axvline(x=trades_df['entry_basis'].median(), color='green', linestyle='--', linewidth=2, 
            label=f'Median: {trades_df["entry_basis"].median():.2f} bps')
ax1.set_title('Distribuzione Entry Basis', fontsize=12, fontweight='bold')
ax1.set_xlabel('Entry Basis (bps)', fontsize=10)
ax1.set_ylabel('Frequenza', fontsize=10)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Entry Basis nel tempo
ax2.scatter(trades_df['entry_date'], trades_df['entry_basis'], 
            c='steelblue', alpha=0.5, s=30, edgecolors='black', linewidth=0.3)
ax2.set_title('Entry Basis nel Tempo', fontsize=12, fontweight='bold')
ax2.set_xlabel('Entry Date', fontsize=10)
ax2.set_ylabel('Entry Basis (bps)', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / '09_entry_basis.pdf', bbox_inches='tight')
print(f"💾 Salvato: 09_entry_basis.pdf")
plt.close()

# ============================================================================
# STEP 12: BASIS COMPRESSION ANALYSIS (NEW)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 12: Basis Compression Analysis")
print("=" * 80)

# Calcola compression (quanto si chiude la basis)
trades_df['basis_change'] = trades_df['exit_basis'] - trades_df['entry_basis']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Scatter Entry vs Exit Basis
ax1.scatter(trades_df['entry_basis'], trades_df['exit_basis'], 
            c='steelblue', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
# Linea y=x (nessuna compressione)
min_val = min(trades_df['entry_basis'].min(), trades_df['exit_basis'].min())
max_val = max(trades_df['entry_basis'].max(), trades_df['exit_basis'].max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='No compression', alpha=0.7)
ax1.set_title('Entry Basis vs Exit Basis', fontsize=12, fontweight='bold')
ax1.set_xlabel('Entry Basis (bps)', fontsize=10)
ax1.set_ylabel('Exit Basis (bps)', fontsize=10)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Subplot 2: Histogram Basis Change
ax2.hist(trades_df['basis_change'], bins=50, color='coral', edgecolor='black', alpha=0.7)
ax2.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
ax2.axvline(x=trades_df['basis_change'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {trades_df["basis_change"].mean():.2f} bps')
ax2.set_title('Distribuzione Basis Change (Exit - Entry)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Basis Change (bps)', fontsize=10)
ax2.set_ylabel('Frequenza', fontsize=10)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / '10_basis_compression.pdf', bbox_inches='tight')
print(f"💾 Salvato: 10_basis_compression.pdf")
plt.close()

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================

print("\n" + "=" * 80)
print("RIEPILOGO GRAFICI CREATI")
print("=" * 80)

print(f"\n📁 File salvati in: {RESULTS_DIR}/\n")
print("   📊 GRAFICI:")
print("   1. 01_equity_curve.pdf       - Curva equity e return cumulativo")
print("   2. 02_pnl_distribution.pdf   - Distribuzione P&L e durata")
print("   3. 03_trade_timeline.pdf     - Trade aperti nel tempo")
print("   4. 04_exit_reasons.pdf       - Distribuzione motivi chiusura")
print("   5. 05_rolling_stats.pdf      - Statistiche rolling 1-anno")
print("   6. 06_drawdown.pdf           - Drawdown analysis")
print("   7. 07_pnl_components.pdf     - Carry vs Capital Gain")
print("   8. 08_top_tickers.pdf        - Top 15 Ticker per performance")
print("   9. 09_entry_basis.pdf        - Distribuzione entry basis")
print("  10. 10_basis_compression.pdf  - Analisi compressione basis")
print("\n   📄 CSV:")
print("   1. trades_log.csv            - Dettaglio completo di tutti i trade")
print("   2. ticker_statistics.csv     - Statistiche aggregate per Ticker")
print("   3. strategy_summary.csv      - Metriche riassuntive strategia")
print("   4. index_daily.csv           - Time series dell'indice")

print("\n✅ VISUALIZZAZIONE COMPLETATA!")
print("=" * 80)
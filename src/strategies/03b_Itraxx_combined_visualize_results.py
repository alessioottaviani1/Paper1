"""
Script 3: Visualizzazione Risultati Trading iTraxx - Portfolio Combinato
=========================================================================
Questo script crea grafici e analisi del portfolio combinato (Main+SnrFin+SubFin+Xover).

GRAFICI:
1. Equity curve
2. Distribuzione P&L per trade
3. Durata trade
4. Timeline dei trade aperti
5. Statistiche per Serie
6. Statistiche per Indice (NUOVO)
7. Exit reasons
8. Rolling statistics
9. Drawdown analysis
10. Summary statistics
11. Componenti P&L (Carry vs Capital Gain)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Configura stile grafici
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "itraxx_combined"

# ============================================================================
# STEP 1: CARICA RISULTATI
# ============================================================================

print("=" * 80)
print("STEP 1: Caricamento risultati - iTraxx Combined Portfolio")
print("=" * 80)

# Carica trades log
trades_df = pd.read_csv(RESULTS_DIR / "trades_log.csv", parse_dates=['entry_date', 'exit_date'])
print(f"✅ Trades: {len(trades_df)}")

# Carica indice
index_files = list(RESULTS_DIR.glob("index_*.csv"))
if len(index_files) == 0:
    print("❌ ERRORE: File indice non trovato!")
    exit()

index_file = index_files[0]
index_df = pd.read_csv(index_file, index_col=0, parse_dates=True)
print(f"✅ Indice: {index_file.name}")

# Carica daily returns
daily_returns = pd.read_csv(RESULTS_DIR / "daily_returns_full.csv", index_col=0, parse_dates=True)
print(f"✅ Daily returns: {len(daily_returns)} giorni")

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
ax1.set_title('iTraxx Combined Portfolio - Equity Curve', fontsize=16, fontweight='bold')
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

# Subplot 2: P&L by Direction
ax = axes[0, 1]
long_pnl = trades_df[trades_df['direction'] == 'LONG']['cumulative_pnl']
short_pnl = trades_df[trades_df['direction'] == 'SHORT']['cumulative_pnl']
positions = [1, 2]
data_to_plot = [long_pnl, short_pnl]
bp = ax.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True,
                showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
    patch.set_facecolor(color)
ax.set_xticklabels(['Long', 'Short'])
ax.set_title('P&L per Direzione', fontsize=12, fontweight='bold')
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
                     c=trades_df['direction'].map({'LONG': 'blue', 'SHORT': 'red'}),
                     alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_title('P&L vs Durata Trade', fontsize=12, fontweight='bold')
ax.set_xlabel('Durata (giorni)', fontsize=10)
ax.set_ylabel('P&L (%)', fontsize=10)
ax.grid(True, alpha=0.3)

# Legenda personalizzata
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Long'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Short')]
ax.legend(handles=legend_elements, loc='upper right')

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
# STEP 5: STATISTICHE PER SERIE
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Statistiche per Serie")
print("=" * 80)

# Aggrega per Serie
serie_stats = trades_df.groupby('serie').agg({
    'trade_id': 'count',
    'cumulative_pnl': ['mean', 'sum', 'std'],
    'duration_days': 'mean',
    'entry_basis': 'mean',
    'exit_basis': 'mean'
}).round(2)

serie_stats.columns = ['Count', 'Avg_PnL', 'Total_PnL', 'Std_PnL', 'Avg_Duration', 'Avg_Entry_Basis', 'Avg_Exit_Basis']
serie_stats = serie_stats.sort_values('Total_PnL', ascending=False)

print("\n📊 Top 10 Serie per P&L Totale:")
print(serie_stats.head(10).to_string())

# Salva statistiche
stats_path = RESULTS_DIR / "serie_statistics.csv"
serie_stats.to_csv(stats_path)
print(f"\n💾 Salvato: serie_statistics.csv")

# Grafico Top Serie
fig, ax = plt.subplots(figsize=(12, 8))
top_series = serie_stats.head(15)
colors = ['green' if x > 0 else 'red' for x in top_series['Total_PnL']]
ax.barh(range(len(top_series)), top_series['Total_PnL'], color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(top_series)))
ax.set_yticklabels(top_series.index)
ax.set_xlabel('P&L Totale (%)', fontsize=12)
ax.set_title('Top 15 Serie per P&L Totale', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '04_top_series.pdf', bbox_inches='tight')
print(f"💾 Salvato: 04_top_series.pdf")
plt.close()

# ============================================================================
# STEP 6: STATISTICHE PER INDICE (NUOVO - SPECIFIC TO COMBINED)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Statistiche per Indice")
print("=" * 80)

# Aggrega per Indice
index_stats = trades_df.groupby('index').agg({
    'trade_id': 'count',
    'cumulative_pnl': ['mean', 'sum', 'std'],
    'duration_days': 'mean'
}).round(2)

index_stats.columns = ['Count', 'Avg_PnL', 'Total_PnL', 'Std_PnL', 'Avg_Duration']
index_stats = index_stats.sort_values('Total_PnL', ascending=False)

print("\n📊 Performance per Indice:")
print(index_stats.to_string())

# Salva statistiche
index_stats_path = RESULTS_DIR / "index_statistics.csv"
index_stats.to_csv(index_stats_path)
print(f"\n💾 Salvato: index_statistics.csv")

# Grafico Performance per Indice
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Numero Trade per Indice
ax1.bar(index_stats.index, index_stats['Count'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], 
        alpha=0.7, edgecolor='black')
ax1.set_title('Numero Trade per Indice', fontsize=12, fontweight='bold')
ax1.set_ylabel('Trade', fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Subplot 2: P&L Totale per Indice
colors = ['green' if x > 0 else 'red' for x in index_stats['Total_PnL']]
ax2.bar(index_stats.index, index_stats['Total_PnL'], color=colors, alpha=0.7, edgecolor='black')
ax2.set_title('P&L Totale per Indice', fontsize=12, fontweight='bold')
ax2.set_ylabel('P&L (%)', fontsize=10)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '05_index_breakdown.pdf', bbox_inches='tight')
print(f"💾 Salvato: 05_index_breakdown.pdf")
plt.close()

# ============================================================================
# STEP 7: EXIT REASONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Analisi Exit Reasons")
print("=" * 80)

exit_counts = trades_df['exit_reason'].value_counts()
print("\n📊 Exit Reasons:")
print(exit_counts)

fig, ax = plt.subplots(figsize=(10, 6))
wedges, texts, autotexts = ax.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%',
                                    startangle=90, colors=['lightgreen', 'lightcoral', 'lightyellow'])
ax.set_title('Distribuzione Exit Reasons', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '06_exit_reasons.pdf', bbox_inches='tight')
print(f"💾 Salvato: 06_exit_reasons.pdf")
plt.close()

# ============================================================================
# STEP 8: ROLLING STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Rolling Statistics")
print("=" * 80)

# Calcola rolling Sharpe (252 giorni)
window = 252  # 1 anno
rolling_mean = daily_returns['index_return'].rolling(window).mean()
rolling_std = daily_returns['index_return'].rolling(window).std()
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
plt.savefig(RESULTS_DIR / '07_rolling_stats.pdf', bbox_inches='tight')
print(f"💾 Salvato: 07_rolling_stats.pdf")
plt.close()

# ============================================================================
# STEP 9: DRAWDOWN ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Drawdown Analysis")
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
plt.savefig(RESULTS_DIR / '08_drawdown.pdf', bbox_inches='tight')
print(f"💾 Salvato: 08_drawdown.pdf")
plt.close()

# ============================================================================
# STEP 10: SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Summary Statistics")
print("=" * 80)

# Calcola metriche chiave
total_return = index_df['cumulative_return'].iloc[-1]
annual_return = (daily_returns['index_return'].mean() * 252)
annual_vol = (daily_returns['index_return'].std() * np.sqrt(252))
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
print("STRATEGY SUMMARY STATISTICS - iTraxx Combined Portfolio")
print("=" * 80)
print(summary_df.to_string())

# Salva summary
summary_df.to_csv(RESULTS_DIR / 'strategy_summary.csv')
print(f"\n💾 Salvato: strategy_summary.csv")

# ============================================================================
# STEP 11: COMPONENTI P&L (CARRY VS CAPITAL GAIN)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 11: Componenti P&L (Carry vs Capital Gain)")
print("=" * 80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Subplot 1: Scatter Carry vs Capital Gain
ax1.scatter(trades_df['cumulative_capital_gain'], trades_df['cumulative_carry'], 
            c=trades_df['direction'].map({'LONG': 'blue', 'SHORT': 'red'}),
            alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('Carry vs Capital Gain', fontsize=12, fontweight='bold')
ax1.set_xlabel('Capital Gain (%)', fontsize=10)
ax1.set_ylabel('Carry (%)', fontsize=10)
ax1.grid(True, alpha=0.3)

# Legenda
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Long'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Short')]
ax1.legend(handles=legend_elements, loc='upper right')

# Subplot 2: Stacked Bar per direzione
long_trades = trades_df[trades_df['direction'] == 'LONG']
short_trades = trades_df[trades_df['direction'] == 'SHORT']

avg_cg_long = long_trades['cumulative_capital_gain'].mean()
avg_carry_long = long_trades['cumulative_carry'].mean()
avg_cg_short = short_trades['cumulative_capital_gain'].mean()
avg_carry_short = short_trades['cumulative_carry'].mean()

x = np.arange(2)
width = 0.35

ax2.bar(x, [avg_cg_long, avg_cg_short], width, label='Capital Gain', color='steelblue')
ax2.bar(x, [avg_carry_long, avg_carry_short], width, bottom=[avg_cg_long, avg_cg_short], 
        label='Carry', color='coral')
ax2.set_xticks(x)
ax2.set_xticklabels(['Long', 'Short'])
ax2.set_ylabel('P&L (%)', fontsize=10)
ax2.set_title('Componenti P&L Medie per Direzione', fontsize=12, fontweight='bold')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '09_pnl_components.pdf', bbox_inches='tight')
print(f"💾 Salvato: 09_pnl_components.pdf")
plt.close()

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================

print("\n" + "=" * 80)
print("RIEPILOGO GRAFICI CREATI")
print("=" * 80)

print(f"\n📁 File salvati in: {RESULTS_DIR}\n")
print("   📊 GRAFICI:")
print("   1. 01_equity_curve.pdf       - Curva equity e return cumulativo")
print("   2. 02_pnl_distribution.pdf   - Distribuzione P&L e durata")
print("   3. 03_trade_timeline.pdf     - Trade aperti nel tempo")
print("   4. 04_top_series.pdf         - Top Serie per performance")
print("   5. 05_index_breakdown.pdf    - Breakdown per Indice (Main/SnrFin/SubFin/Xover)")
print("   6. 06_exit_reasons.pdf       - Distribuzione motivi chiusura")
print("   7. 07_rolling_stats.pdf      - Statistiche rolling 1-anno")
print("   8. 08_drawdown.pdf           - Drawdown analysis")
print("   9. 09_pnl_components.pdf     - Carry vs Capital Gain")
print("\n   📄 CSV:")
print("   1. trades_log.csv            - Dettaglio completo di tutti i trade")
print("   2. serie_statistics.csv      - Statistiche aggregate per Serie")
print("   3. index_statistics.csv      - Statistiche aggregate per Indice")
print("   4. strategy_summary.csv      - Metriche riassuntive strategia")
print("   5. index_daily.csv           - Time series dell'indice")
print("   6. daily_returns_full.csv    - Return giornalieri completi")

print("\n✅ VISUALIZZAZIONE COMPLETATA!")
print()
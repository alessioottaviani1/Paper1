"""
Script 3: Visualizzazione Risultati Trading
============================================
Questo script crea grafici e analisi dei risultati della simulazione.

GRAFICI:
1. Equity curve
2. Distribuzione P&L per trade
3. Durata trade
4. Timeline dei trade aperti
5. Statistiche per ISIN
6. Exit reasons
7. Rolling statistics
8. Drawdown analysis
9. Summary statistics
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
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================================
# STEP 1: CARICA RISULTATI
# ============================================================================

print("=" * 80)
print("STEP 1: Caricamento risultati")
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

# Subplot 1: Index Value
ax1.plot(index_df.index, index_df['index_value'], linewidth=2, color='darkblue', label='Index Value')
ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('BTP-Italia Basis Strategy - Equity Curve', fontsize=16, fontweight='bold')
ax1.set_ylabel('Index Value (Base 100)', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

# Subplot 2: Cumulative Return (%)
ax2.plot(index_df.index, index_df['cumulative_return'], linewidth=2, color='darkgreen', label='Cumulative Return (%)')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_title('Cumulative Return', fontsize=14, fontweight='bold')
ax2.set_xlabel('Date', fontsize=12)
ax2.set_ylabel('Return (%)', fontsize=12)
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / '01_equity_curve.png', dpi=300, bbox_inches='tight')
print(f"💾 Salvato: 01_equity_curve.png")
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
plt.savefig(RESULTS_DIR / '02_pnl_distribution.png', dpi=300, bbox_inches='tight')
print(f"💾 Salvato: 02_pnl_distribution.png")
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
plt.savefig(RESULTS_DIR / '03_trade_timeline.png', dpi=300, bbox_inches='tight')
print(f"💾 Salvato: 03_trade_timeline.png")
plt.close()

# ============================================================================
# STEP 5: STATISTICHE PER ISIN
# ============================================================================

print("\n" + "=" * 80)
print("STEP 5: Statistiche per ISIN")
print("=" * 80)

# Aggrega per ISIN
isin_stats = trades_df.groupby('isin').agg({
    'trade_id': 'count',
    'cumulative_pnl': ['mean', 'sum', 'std'],
    'duration_days': 'mean',
    'entry_basis': 'mean',
    'exit_basis': 'mean'
}).round(2)

isin_stats.columns = ['Count', 'Avg_PnL', 'Total_PnL', 'Std_PnL', 'Avg_Duration', 'Avg_Entry_Basis', 'Avg_Exit_Basis']
isin_stats = isin_stats.sort_values('Total_PnL', ascending=False)

print("\n📊 Top 10 ISIN per P&L Totale:")
print(isin_stats.head(10).to_string())

# Salva statistiche
stats_path = RESULTS_DIR / "isin_statistics.csv"
isin_stats.to_csv(stats_path)
print(f"\n💾 Salvato: isin_statistics.csv")

# Grafico Top ISIN
fig, ax = plt.subplots(figsize=(12, 8))
top_isins = isin_stats.head(15)
colors = ['green' if x > 0 else 'red' for x in top_isins['Total_PnL']]
ax.barh(range(len(top_isins)), top_isins['Total_PnL'], color=colors, alpha=0.7, edgecolor='black')
ax.set_yticks(range(len(top_isins)))
ax.set_yticklabels(top_isins.index)
ax.set_xlabel('P&L Totale (%)', fontsize=12)
ax.set_title('Top 15 ISIN per P&L Totale', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '04_top_isins.png', dpi=300, bbox_inches='tight')
print(f"💾 Salvato: 04_top_isins.png")
plt.close()

# ============================================================================
# STEP 6: EXIT REASONS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 6: Analisi Exit Reasons")
print("=" * 80)

exit_counts = trades_df['exit_reason'].value_counts()
print("\n📊 Exit Reasons:")
print(exit_counts)

fig, ax = plt.subplots(figsize=(10, 6))
wedges, texts, autotexts = ax.pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%',
                                    startangle=90, colors=['lightgreen', 'lightcoral', 'lightyellow'])
ax.set_title('Distribuzione Exit Reasons', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '05_exit_reasons.png', dpi=300, bbox_inches='tight')
print(f"💾 Salvato: 05_exit_reasons.png")
plt.close()

# ============================================================================
# STEP 7: ROLLING STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 7: Rolling Statistics")
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
plt.savefig(RESULTS_DIR / '06_rolling_stats.png', dpi=300, bbox_inches='tight')
print(f"💾 Salvato: 06_rolling_stats.png")
plt.close()

# ============================================================================
# STEP 8: DRAWDOWN ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 8: Drawdown Analysis")
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
plt.savefig(RESULTS_DIR / '07_drawdown.png', dpi=300, bbox_inches='tight')
print(f"💾 Salvato: 07_drawdown.png")
plt.close()

# ============================================================================
# STEP 9: SUMMARY STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("STEP 9: Summary Statistics")
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
print("STRATEGY SUMMARY STATISTICS")
print("=" * 80)
print(summary_df.to_string())

# Salva summary
summary_df.to_csv(RESULTS_DIR / 'strategy_summary.csv')
print(f"\n💾 Salvato: strategy_summary.csv")

# ============================================================================
# STEP 10: COMPONENTI P&L (CARRY VS CAPITAL GAIN)
# ============================================================================

print("\n" + "=" * 80)
print("STEP 10: Componenti P&L (Carry vs Capital Gain)")
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
plt.savefig(RESULTS_DIR / '08_pnl_components.png', dpi=300, bbox_inches='tight')
print(f"💾 Salvato: 08_pnl_components.png")
plt.close()

# ============================================================================
# RIEPILOGO FINALE
# ============================================================================

print("\n" + "=" * 80)
print("RIEPILOGO GRAFICI CREATI")
print("=" * 80)

print(f"\n📁 File salvati in: {RESULTS_DIR}\n")
print("   📊 GRAFICI:")
print("   1. 01_equity_curve.png       - Curva equity e return cumulativo")
print("   2. 02_pnl_distribution.png   - Distribuzione P&L e durata")
print("   3. 03_trade_timeline.png     - Trade aperti nel tempo")
print("   4. 04_top_isins.png          - Top ISIN per performance")
print("   5. 05_exit_reasons.png       - Distribuzione motivi chiusura")
print("   6. 06_rolling_stats.png      - Statistiche rolling 1-anno")
print("   7. 07_drawdown.png           - Drawdown analysis")
print("   8. 08_pnl_components.png     - Carry vs Capital Gain")
print("\n   📄 CSV:")
print("   1. trades_log.csv            - Dettaglio completo di tutti i trade")
print("   2. isin_statistics.csv       - Statistiche aggregate per ISIN")
print("   3. strategy_summary.csv      - Metriche riassuntive strategia")
print("   4. index_daily.csv           - Time series dell'indice")
print("   5. daily_returns_full.csv    - Return giornalieri completi")

print("\n✅ VISUALIZZAZIONE COMPLETATA!")
print()
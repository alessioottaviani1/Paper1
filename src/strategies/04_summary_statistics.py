"""
================================================================================
Script 04: Summary Statistics Table - Multi Strategy
================================================================================
Genera tabella con statistiche descrittive per TUTTE le strategie:
- BTP Italia
- iTraxx Main
- iTraxx SnrFin
- iTraxx SubFin
- iTraxx Xover
- iTraxx Combined (opzionale)
- CDS-Bond Basis

STATISTICHE CALCOLATE:
- n (numero osservazioni)
- Mean (return medio mensile/weekly %)
- Standard deviation (volatilità %)
- Minimum / Maximum
- Skewness
- Kurtosis
- Ratio negative (% return negativi)
- Serial correlation (autocorrelazione AR(1))
- Sharpe ratio (annualizzato)

OUTPUT:
- 1 tabella LaTeX per monthly (se esiste)
- 1 tabella LaTeX per weekly (se esiste)

FIX APPLICATO:
- Usa index_daily.csv e resample a monthly/weekly
- Usa EURIBOR 1M dinamico da Euribor1m.xlsx (media del periodo)

Author: Alessio Ottaviani
Date: December 2025
Institution: EDHEC Business School - PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PARAMETRI
# ============================================================================

# Include iTraxx_Combined in summary?
INCLUDE_COMBINED = True  # True/False

# Basis data — loaded from rq3_01 output
BTP_MIN_MONTHS_TO_MATURITY = 6  # fallback if pickle not available

# ============================================================================
# STRATEGIE DA ANALIZZARE

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
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"

TABLES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# NOTE: NO RISK-FREE RATE NEEDED
# ============================================================================
# Le strategie sono SELF-FINANCING (long/short arbitrage)
# Returns sono già excess returns (arbitrage profit)
# Sharpe ratio = Mean Return / Volatility (NO sottrazione RF)
# Questo è lo standard per strategie arbitrage (Duarte et al., Fung-Hsieh)
# ============================================================================

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title, char="="):
    """Print formatted section header"""
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")

def calculate_summary_stats(returns, freq='monthly'):
    """
    Calcola tutte le statistiche summary per una serie di return
    
    Parameters:
    -----------
    returns : pd.Series
        Serie di return (in %)
    freq : str
        'monthly' o 'weekly' per annualizzazione corretta
    
    Returns:
    --------
    dict con tutte le statistiche
    """
    
    # Rimuovi NaN
    returns_clean = returns.dropna()
    
    if len(returns_clean) == 0:
        return None
    
    # Basic statistics
    n = len(returns_clean)
    mean = returns_clean.mean()
    std = returns_clean.std()
    
  
    
    # Min/Max
    minimum = returns_clean.min()
    maximum = returns_clean.max()
    
    # Skewness & Kurtosis
    skewness = returns_clean.skew()
    kurtosis = returns_clean.kurtosis()  # Excess kurtosis (pandas default)
    
    # Ratio negative
    ratio_negative = (returns_clean < 0).sum() / n
    
    # Serial correlation (AR(1))
    serial_corr = returns_clean.autocorr(lag=1)
    
    # Sharpe ratio (annualizzato)
    # NO risk-free subtraction for self-financing strategies
    if freq == 'monthly':
        periods_per_year = 12
    elif freq == 'weekly':
        periods_per_year = 52
    else:  # daily
        periods_per_year = 252
    
    mean_annual = mean * periods_per_year
    std_annual  = std * np.sqrt(periods_per_year)

    sharpe = mean_annual / std_annual if std_annual > 0 else 0

    return {
        'n': n,
        'mean': mean_annual,
        'std': std_annual,
        'minimum': minimum,
        'maximum': maximum,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'ratio_negative': ratio_negative,
        'serial_corr': serial_corr,
        'sharpe': sharpe
    }

# ============================================================================
# STEP 2: CHECK FREQUENZE DISPONIBILI
# ============================================================================

print_header("STEP 2: Check frequenze disponibili")

# Check se almeno una strategia ha index_daily.csv
frequencies_available = []
strategies_with_daily = []

for strategy in STRATEGIES:
    strategy_dir = RESULTS_DIR / strategy.lower()
    index_file = strategy_dir / "index_daily.csv"
    
    if index_file.exists():
        strategies_with_daily.append(strategy)

if strategies_with_daily:
    frequencies_available = ['daily']
    print(f"📊 Frequenza: daily (from index_return)")

else:
    print("\n❌ ERRORE: Nessun file index_daily.csv trovato!")
    print("   Runna prima gli script di trading simulation.")
    exit()

# ============================================================================
# MAIN: GENERA SUMMARY TABLES
# ============================================================================

# ============================================================================
# LOOP PER OGNI FREQUENZA
# ============================================================================

for freq in frequencies_available:
    
    print_header(f"PROCESSING: {freq.upper()} DATA")
    
    # Storage per risultati
    summary_results = []
    
    # ========================================================================
    # LOOP PER OGNI STRATEGIA
    # ========================================================================
    
    for strategy_name in STRATEGIES:
        
        # Skip iTraxx_Combined se non incluso
        if strategy_name == 'iTraxx_Combined' and not INCLUDE_COMBINED:
            continue
        
        print(f"\n📊 Processing: {strategy_name}")
        
        strategy_dir = RESULTS_DIR / strategy_name.lower()
        index_file = strategy_dir / "index_daily.csv"
        
        # Check se file esiste
        if not index_file.exists():
            print(f"   ⚠️  File non trovato: {index_file.name} - skip")
            continue
        
        # Carica dati
        try:
            data = pd.read_csv(index_file, index_col=0, parse_dates=True)
            
            # Colonna return
            if 'index_return' in data.columns:
                daily_returns = data['index_return']
            elif 'Strategy_Return' in data.columns:
                daily_returns = data['Strategy_Return']
            elif 'return' in data.columns:
                daily_returns = data['return']
            else:
                print(f"   ❌ Colonna return non trovata in {index_file.name}")
                continue
            
            print(f"   ✅ Loaded: {len(daily_returns)} daily observations")
            
            returns = daily_returns.dropna()
            print(f"   🔄 Using daily returns: {len(returns)} observations")

            # Calcola statistiche
            stats = calculate_summary_stats(returns, freq=freq)

            
            if stats is None:
                print(f"   ❌ Impossibile calcolare statistiche (tutti NaN?)")
                continue
            
            # Aggiungi nome strategia
            stats['Strategy'] = strategy_name
            
            # Salva
            summary_results.append(stats)
            
            print(f"   ✅ Statistics calculated")
            print(f"      Mean: {stats['mean']:.4f}%, Std: {stats['std']:.4f}%, Sharpe: {stats['sharpe']:.3f}")
        
        except Exception as e:
            print(f"   ❌ Errore: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # ========================================================================
    # CREA DATAFRAME SUMMARY
    # ========================================================================
    
    if not summary_results:
        print(f"\n❌ Nessuna statistica calcolata per {freq}")
        continue
    
    summary_df = pd.DataFrame(summary_results)
    
    # Riordina colonne
    col_order = ['Strategy', 'n', 'mean', 'std', 'minimum', 'maximum',
                 'skewness', 'kurtosis', 'ratio_negative', 'serial_corr', 'sharpe']
    
    summary_df = summary_df[col_order]
    
    # Sort per nome strategia
    summary_df = summary_df.sort_values('Strategy')
    
    print(f"\n" + "="*80)
    print(f"SUMMARY TABLE - {freq.upper()}")
    print("="*80)
    print(f"\n{summary_df.to_string(index=False)}")
    

# ========================================================================
    # LOAD BASIS DATA FROM rq3_01 OUTPUT
    # ========================================================================

    print(f"\n" + "="*80)
    print(f"LOADING BASIS DATA FOR PANEL A")
    print("="*80)

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from dateutil.relativedelta import relativedelta

    FIGURES_DIR = RESULTS_DIR / "figures"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    RQ3_DATA_DIR = RESULTS_DIR / "rq3_duffie" / "data"
    basis_pickle_path = RQ3_DATA_DIR / "basis_daily.pkl"

    basis_daily = {}
    basis_loaded_from_pickle = False

    if basis_pickle_path.exists():
        basis_daily_raw = pd.read_pickle(basis_pickle_path)
        # Map rq3_01 keys to 04 keys
        key_map = {
            "btp_italia": "BTP_Italia",
            "itraxx_combined": "iTraxx_Combined",
            "cds_bond_basis_median": "CDS_Bond_Basis",
        }
        for src_key, dst_key in key_map.items():
            if src_key in basis_daily_raw:
                basis_daily[dst_key] = basis_daily_raw[src_key]
                print(f"   ✅ {dst_key}: {len(basis_daily_raw[src_key])} days (from pickle)")
        basis_loaded_from_pickle = True
        print(f"   Loaded from: {basis_pickle_path}")
    else:
        print(f"   ⚠️ {basis_pickle_path} not found!")
        print(f"   Run rq3_01_mispricing_construction.py first.")
        print(f"   Skipping Panel A and basis plot.")

    # ------------------------------------------------------------------
    # Compute basis summary statistics
    # ------------------------------------------------------------------

    STRATEGY_LATEX_DISPLAY = {
        "BTP_Italia":      r"BTP Italia",
        "iTraxx_Combined": r"iTraxx Combined",
        "CDS_Bond_Basis":  r"CDS-Bond Basis",
    }

    basis_summary_results = []
    if basis_loaded_from_pickle:
        for strat_name in ["BTP_Italia", "CDS_Bond_Basis", "iTraxx_Combined"]:
            if strat_name not in basis_daily:
                continue
            s = basis_daily[strat_name].dropna()
            if len(s) == 0:
                continue
            stats_b = {
                "Strategy": strat_name,
                "n": len(s),
                "mean": s.mean(),
                "median": s.median(),
                "std": s.std(),
                "minimum": s.min(),
                "maximum": s.max(),
                "skewness": s.skew(),
                "kurtosis": s.kurtosis(),
                "serial_corr": s.autocorr(lag=1),
                "p10": s.quantile(0.10),
                "p90": s.quantile(0.90),
            }
            basis_summary_results.append(stats_b)
            print(f"   {strat_name}: n={stats_b['n']}, mean={stats_b['mean']:.2f}, "
                  f"median={stats_b['median']:.2f}, std={stats_b['std']:.2f}")

    basis_summary_df = pd.DataFrame(basis_summary_results)

    # ========================================================================
    # GENERA TABELLA LATEX CON PANEL A (Basis) + PANEL B (Returns)
    # ========================================================================

    print(f"\n" + "="*80)
    print(f"GENERATING LATEX TABLES - {freq.upper()} (ARTICLE + BEAMER)")
    print("="*80)

    latex_article_path = TABLES_DIR / f"summary_statistics_{freq}_article.tex"
    latex_beamer_path  = TABLES_DIR / f"summary_statistics_{freq}_beamer.tex"

    has_panel_a = len(basis_summary_df) > 0

    # --- Helper: Panel A body (Basis Levels) ---
    def write_panel_a_body(f):
        f.write("\\addlinespace[3pt]\n")
        f.write("\\multicolumn{11}{l}{\\textbf{Panel A: Basis Levels (bps)}} \\\\\n")
        f.write("\\addlinespace[2pt]\n")
        f.write("\\midrule\n")
        f.write("Strategy & Mean & Median & Std Dev & Min & Max "
                "& Skew & Kurt & AC(1) & $P_{10}$ & $P_{90}$ \\\\\n")
        f.write("\\midrule\n")
        for _, row in basis_summary_df.iterrows():
            display = STRATEGY_LATEX_DISPLAY.get(row['Strategy'],
                                                  row['Strategy'])
            f.write(f"\\textit{{{display}}} & ")
            f.write(f"{row['mean']:.2f} & ")
            f.write(f"{row['median']:.2f} & ")
            f.write(f"{row['std']:.2f} & ")
            f.write(f"{row['minimum']:.1f} & ")
            f.write(f"{row['maximum']:.1f} & ")
            f.write(f"{row['skewness']:.2f} & ")
            f.write(f"{row['kurtosis']:.2f} & ")
            f.write(f"{row['serial_corr']:.3f} & ")
            f.write(f"{row['p10']:.1f} & ")
            f.write(f"{row['p90']:.1f} \\\\\n")

    # --- Helper: Panel B body (Strategy Returns) ---
    def write_panel_b_body(f):
        f.write("\\addlinespace[6pt]\n")
        f.write("\\multicolumn{11}{l}{\\textbf{Panel B: Strategy Returns (\\%)}} \\\\\n")
        f.write("\\addlinespace[2pt]\n")
        f.write("\\midrule\n")
        f.write("Strategy & $n$ & Mean & Std Dev & Min & Max "
                "& Skew & Kurt & \\% Neg & AC(1) & Sharpe \\\\\n")
        f.write("         &     & (p.a.) & (p.a.) &     &     "
                "&      &      &       &       &        \\\\\n")
        f.write("\\midrule\n")
        for _, row in summary_df.iterrows():
            strategy_display = row['Strategy'].replace('_', ' ')
            f.write(f"\\textit{{{strategy_display}}} & ")
            f.write(f"{int(row['n'])} & ")
            f.write(f"{row['mean']:.2f} & ")
            f.write(f"{row['std']:.2f} & ")
            f.write(f"{row['minimum']:.2f} & ")
            f.write(f"{row['maximum']:.2f} & ")
            f.write(f"{row['skewness']:.2f} & ")
            f.write(f"{row['kurtosis']:.2f} & ")
            f.write(f"{row['ratio_negative']*100:.1f} & ")
            f.write(f"{row['serial_corr']:.3f} & ")
            f.write(f"{row['sharpe']:.2f} \\\\\n")

    # --- Helper: combined notes ---
    def write_panel_notes(f):
        f.write("\\begin{tablenotes}[para,flushleft]\n")
        f.write("\\footnotesize\n")
        f.write("\\item \\textit{Note:} ")
        if has_panel_a:
            f.write("Panel~A reports daily basis levels in basis points "
                    "(Jan~2005--May~2025 for CDS--Bond, "
                    "Oct~2008--May~2025 for iTraxx, "
                    "Nov~2013--May~2025 for BTP~Italia). ")
            f.write("For BTP~Italia, the basis is the cross-sectional mean of all "
                    "outstanding issues with at least six months to maturity. ")
            f.write("For iTraxx~Combined, it is the mean of the four on-the-run "
                    "sub-index skews (Main, SeniorFin, SubFin, Crossover). ")
            f.write("For CDS--Bond~Basis, it is the cross-sectional median of all "
                    "single-name bases in the iTraxx~Main universe. ")
            f.write("$P_{10}$ and $P_{90}$ are the 10th and 90th percentiles of "
                    "the daily time series. ")
        f.write("Panel~B reports daily percentage returns, with \\textit{Mean} "
                "and \\textit{Std~Dev} annualized ($\\times 252$ and "
                "$\\times\\sqrt{252}$). ")
        f.write("\\textit{Skew} and \\textit{Kurt} denote skewness and excess "
                "kurtosis. ")
        f.write("\\textit{AC(1)} is the first-order autocorrelation. ")
        f.write("\\textit{Sharpe} is annualized. ")
        f.write("Sample periods vary by strategy.\n")
        f.write("\\end{tablenotes}\n")

    # --- Helper: returns-only body (fallback if no Panel A) ---
    def write_returns_only_body(f):
        f.write("\\begin{tabular}{lrrrrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Strategy & $n$ & Mean & Std Dev & Min & Max "
                "& Skew & Kurt & \\% Neg & AC(1) & Sharpe \\\\\n")
        f.write("         &     & (\\% p.a.) & (\\% p.a.) &  (\\%) & (\\%) "
                "&      &      &       &       &        \\\\\n")
        f.write("\\midrule\n")
        for _, row in summary_df.iterrows():
            strategy_display = row['Strategy'].replace('_', ' ')
            f.write(f"\\textit{{{strategy_display}}} & ")
            f.write(f"{int(row['n'])} & ")
            f.write(f"{row['mean']:.2f} & ")
            f.write(f"{row['std']:.2f} & ")
            f.write(f"{row['minimum']:.2f} & ")
            f.write(f"{row['maximum']:.2f} & ")
            f.write(f"{row['skewness']:.2f} & ")
            f.write(f"{row['kurtosis']:.2f} & ")
            f.write(f"{row['ratio_negative']*100:.1f} & ")
            f.write(f"{row['serial_corr']:.3f} & ")
            f.write(f"{row['sharpe']:.2f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    def write_returns_only_notes(f):
        f.write("\\begin{tablenotes}[para,flushleft]\n")
        f.write("\\footnotesize\n")
        f.write("\\item \\textit{Note:} This table reports summary statistics "
                "for daily percentage returns. ")
        f.write("\\textit{Mean} and \\textit{Std Dev} are annualized (252 trading days). ")
        f.write("\\textit{Skew} and \\textit{Kurt} denote skewness and excess kurtosis. ")
        f.write("\\textit{\\% Neg} is the proportion of negative returns. ")
        f.write("\\textit{AC(1)} is the first-order autocorrelation. ")
        f.write("\\textit{Sharpe} is annualized. ")
        f.write("Sample period varies by strategy.\n")
        f.write("\\end{tablenotes}\n")

    # -------------------------
    # 1) ARTICLE STYLE
    # -------------------------
    with open(latex_article_path, "w") as f:
        f.write("% " + "="*76 + "\n")
        f.write(f"% SUMMARY STATISTICS — PANEL A (BASIS) + PANEL B (RETURNS)\n")
        f.write("% " + "="*76 + "\n\n")

        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Summary Statistics for Fixed Income Arbitrage Strategies}\n")
        f.write(f"\\label{{tab:summary_stats_{freq}}}\n")
        f.write("\\begin{threeparttable}\n")
        f.write("\\begin{singlespace}\n")
        f.write("\\small\n")

        if has_panel_a:
            f.write("\\begin{tabular}{lrrrrrrrrrr}\n")
            f.write("\\toprule\n")
            write_panel_a_body(f)
            write_panel_b_body(f)
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            write_panel_notes(f)
        else:
            write_returns_only_body(f)
            write_returns_only_notes(f)

        f.write("\\end{singlespace}\n")
        f.write("\\end{threeparttable}\n")
        f.write("\\end{table}\n")

    print(f"💾 LaTeX (article): {latex_article_path.name}")

    # -------------------------
    # 2) BEAMER STYLE
    # -------------------------
    with open(latex_beamer_path, "w") as f:
        f.write("% " + "="*76 + "\n")
        f.write(f"% SUMMARY STATISTICS — PANEL A (BASIS) + PANEL B (RETURNS) — BEAMER\n")
        f.write("% " + "="*76 + "\n\n")

        f.write("\\begin{threeparttable}\n")
        f.write("\\centering\n")
        f.write("\\small\n")

        if has_panel_a:
            f.write("\\begin{tabular}{lrrrrrrrrrr}\n")
            f.write("\\toprule\n")
            write_panel_a_body(f)
            write_panel_b_body(f)
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            write_panel_notes(f)
        else:
            write_returns_only_body(f)
            write_returns_only_notes(f)

        f.write("\\end{threeparttable}\n")

    print(f"💾 LaTeX (beamer): {latex_beamer_path.name}")

    # ========================================================================
    # PRINT LATEX PREVIEW
    # ========================================================================

    print(f"\n📋 LaTeX Preview (primi 25 righe) — ARTICLE:")
    print("=" * 80)
    with open(latex_article_path, 'r') as f:
        for line in f.readlines()[:25]:
            print(line.rstrip())
    print("...")
    print("=" * 80)

    # ========================================================================
    # BASIS TIME SERIES COMPARISON PLOT (median for CDS-Bond)
    # ========================================================================

    if basis_loaded_from_pickle:
        print(f"\n" + "="*80)
        print(f"GENERATING BASIS TIME SERIES PLOT (comparison)")
        print("="*80)

        plot_config = {
            "BTP_Italia": {
                "label": "BTP Italia Inflation-Linked Basis",
                "color": "#1f77b4",
                "thresholds": [40, -50],
                "threshold_labels": ["Long entry (+40 bps)",
                                     "Short entry (\u221250 bps)"],
                "ylabel": "Basis (bps)",
            },
            "CDS_Bond_Basis": {
                "label": "CDS-Bond Basis (cross-sectional median)",
                "color": "#d62728",
                "thresholds": [-40],
                "threshold_labels": ["Entry (\u221240 bps)"],
                "ylabel": "Basis (bps)",
            },
            "iTraxx_Combined": {
                "label": "CDS Index Skew (iTraxx Combined)",
                "color": "#2ca02c",
                "thresholds": [10, -10],
                "threshold_labels": ["Entry (+10 bps)",
                                     "Entry (\u221210 bps)"],
                "ylabel": "Skew (bps)",
            },
        }

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

        for idx, name in enumerate(["BTP_Italia", "CDS_Bond_Basis",
                                     "iTraxx_Combined"]):
            ax = axes[idx]
            cfg = plot_config[name]

            if name in basis_daily:
                series = basis_daily[name].dropna()
                ax.plot(series.index, series.values, color=cfg["color"],
                        linewidth=0.7, alpha=0.85)
                ax.axhline(0, color='grey', linewidth=0.5, linestyle='-',
                           alpha=0.5)

                threshold_colors = ['#ff7f0e', '#9467bd']
                for t_idx, (thresh, t_label) in enumerate(
                        zip(cfg["thresholds"], cfg["threshold_labels"])):
                    ax.axhline(thresh, color=threshold_colors[t_idx % 2],
                               linewidth=1.0, linestyle='--', alpha=0.7,
                               label=t_label)

                ax.set_ylabel(cfg["ylabel"], fontsize=10)
                ax.set_title(cfg["label"], fontsize=11, fontweight='bold')
                ax.legend(loc='best', fontsize=8, framealpha=0.8)
                ax.grid(True, alpha=0.2)
                ax.text(0.01, 0.95,
                        f"{series.index.min().strftime('%b %Y')} \u2014 "
                        f"{series.index.max().strftime('%b %Y')}",
                        transform=ax.transAxes, fontsize=8,
                        verticalalignment='top', fontstyle='italic',
                        color='grey')
            else:
                ax.text(0.5, 0.5, f"No data for {name}",
                        transform=ax.transAxes, ha='center', fontsize=12,
                        color='red')

        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "basis_time_series_median.pdf",
                    bbox_inches='tight')
        plt.close()

        print(f"📊 basis_time_series_median.pdf → results/figures/")


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print_header("✅ SUMMARY STATISTICS GENERATION COMPLETED")

print(f"\n📁 File generato in {TABLES_DIR}:")
print(f"   • summary_statistics_monthly_article.tex")
print(f"   • summary_statistics_monthly_beamer.tex")

print(f"\n💡 Sharpe Ratio Calculation:")
print(f"   • Formula: Sharpe = Mean Return / Volatility")
print(f"   • NO risk-free rate subtraction (self-financing strategies)")
print(f"   • Standard for arbitrage strategies (Duarte et al., Fung-Hsieh)")

print(f"\n💡 Per includere nella tesi:")
print(f"   1. Apri summary_statistics_monthly.tex")
print(f"   2. Copia tutto il contenuto")
print(f"   3. Incolla nel tuo documento LaTeX")
print(f"   4. Assicurati di avere nel preambolo:")
print(f"      \\usepackage{{booktabs}}")
print(f"      \\usepackage{{threeparttable}}")
print(f"   5. Compila!")

if not INCLUDE_COMBINED:
    print(f"\n💡 NOTE: iTraxx_Combined è ESCLUSO dalla tabella")
    print(f"   (modifica INCLUDE_COMBINED = True per includerlo)")

print(f"\n🎯 INTERPRETAZIONE:")
print(f"   • Sharpe ratio confrontabile tra strategie")
print(f"   • Skewness > 0 indica asimmetria positiva")
print(f"   • Kurtosis > 0 indica code pesanti (excess kurtosis)")
print(f"   • Serial corr indica persistenza nei returns")

print("\n✅ OUTPUT:")
print("   • Solo LaTeX (no CSV)")
print("   • Solo monthly frequency")
print("   • Sharpe = Return / Vol (self-financing strategies)")

# ============================================================================
# STEP 3: TRADE STATISTICS TABLE (Appendix A.2)
# ============================================================================

print_header("TRADE STATISTICS TABLE (Appendix A.2)")

TRADES_FILES = {
    "BTP Italia": RESULTS_DIR / "btp_italia" / "trades_log.csv",
    "iTraxx Combined": RESULTS_DIR / "itraxx_combined" / "trades_log.csv",
    "CDS-Bond Basis": RESULTS_DIR / "cds_bond_basis" / "trades_log.csv",
}

STRATEGY_LATEX_NAMES = {
    "BTP Italia": r"BTP\textit{-}Italia",
    "iTraxx Combined": r"iTraxx \textit{combined}",
    "CDS-Bond Basis": r"CDS\textit{-}Bond Basis",
}


def compute_trade_stats(trades_path):
    """Compute trade-level statistics from trades_log.csv."""
    df = pd.read_csv(trades_path, parse_dates=["entry_date", "exit_date"])

    n_total = len(df)
    if "direction" in df.columns:
        n_long = len(df[df["direction"] == "LONG"])
        n_short = len(df[df["direction"] == "SHORT"])
    else:
        n_long = n_total  # all trades same direction
        n_short = 0
    if "duration_days" not in df.columns:
        df["duration_days"] = (df["exit_date"] - df["entry_date"]).dt.days

    avg_duration = df["duration_days"].mean()
    med_duration = df["duration_days"].median()

    if "pnl_bps" in df.columns:
        pnl_col = "pnl_bps"
    elif "pnl_pct" in df.columns:
        pnl_col = "pnl_pct"
    else:
        pnl_col = "cumulative_pnl"
    pnl = df[pnl_col].dropna()
    avg_pnl = pnl.mean()
    med_pnl = pnl.median()
    win_rate = (pnl > 0).mean() * 100

    exit_counts = df["exit_reason"].value_counts(normalize=True) * 100

    return {
        "n_total": n_total,
        "n_long": n_long,
        "n_short": n_short,
        "avg_duration": avg_duration,
        "med_duration": med_duration,
        "avg_pnl": avg_pnl,
        "med_pnl": med_pnl,
        "win_rate": win_rate,
        "pct_target": exit_counts.get("TARGET_HIT", 0),
        "pct_maturity": exit_counts.get("MATURITY", 0),
        "pct_end": exit_counts.get("END_OF_SAMPLE", 0),
    }


all_trade_stats = {}

for name, path in TRADES_FILES.items():
    if not path.exists():
        print(f"⚠️  {name}: trades_log.csv not found at {path}")
        continue
    print(f"\n📂 {name}...")
    stats = compute_trade_stats(path)
    all_trade_stats[name] = stats
    print(f"   Trades: {stats['n_total']} (L:{stats['n_long']}, S:{stats['n_short']})")
    print(f"   Duration: {stats['avg_duration']:.0f} days avg, {stats['med_duration']:.0f} median")
    print(f"   P&L: {stats['avg_pnl']:.3f}% avg, {stats['med_pnl']:.3f}% median")


if all_trade_stats:
    trade_tex_path = TABLES_DIR / "trade_statistics_article.tex"

    with open(trade_tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Trade-Level Statistics by Strategy}\n")
        f.write("\\label{tab:trade_stats}\n")
        f.write("\\begin{threeparttable}\n")
        f.write("\\begin{singlespace}\n")
        f.write("\\begin{tabular}{lrrr}\n")
        f.write("\\toprule\n")

        header = " & ".join([STRATEGY_LATEX_NAMES.get(s, s) for s in all_trade_stats.keys()])
        f.write(f" & {header} \\\\\n")
        f.write("\\midrule\n")

        rows = [
            ("Total trades", "n_total", "{:.0f}"),
            ("\\quad Long", "n_long", "{:.0f}"),
            ("\\quad Short", "n_short", "{:.0f}"),
            ("Avg duration (days)", "avg_duration", "{:.0f}"),
            ("Median duration (days)", "med_duration", "{:.0f}"),
            ("Avg P\\&L per trade (\\%)", "avg_pnl", "{:.3f}"),
            ("Median P\\&L per trade (\\%)", "med_pnl", "{:.3f}"),
            ("\\addlinespace\n\\multicolumn{4}{l}{\\textit{Exit reasons (\\%)}}", None, None),
            ("\\quad Target hit", "pct_target", "{:.1f}"),
            ("\\quad Maturity", "pct_maturity", "{:.1f}"),
            ("\\quad End of sample", "pct_end", "{:.1f}"),
        ]

        strat_names = list(all_trade_stats.keys())
        for label, key, fmt in rows:
            if key is None:
                f.write(f"{label} \\\\\n")
                continue
            vals = []
            for s in strat_names:
                v = all_trade_stats[s].get(key, np.nan)
                if pd.notna(v):
                    vals.append(fmt.format(v))
                else:
                    vals.append("--")
            f.write(f"{label} & {' & '.join(vals)} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}[para,flushleft]\n")
        f.write("\\footnotesize\n")
        f.write("\\item \\textit{Note:} "
                "This table reports trade-level statistics for each arbitrage strategy. "
                "P\\&L is the cumulative return per trade in percentage points. "
                "Duration is in calendar days. "
                "Exit reasons may not sum to 100\\% due to rounding.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{singlespace}\n")
        f.write("\\end{threeparttable}\n")
        f.write("\\end{table}\n")

    print(f"\n💾 Trade statistics: {trade_tex_path.name}")
else:
    print("\n❌ No trades data found — trade statistics table not generated.")

# ============================================================================
# STEP 4: COMBINED EQUITY CURVES WITH DRAWDOWN (for paper Section 3)
# ============================================================================
# 3 vertical panels, each with equity curve (top) and drawdown (bottom).
# All strategies on separate panels (different sample periods).
# Output: results/figures/equity_curves_combined.pdf
# ============================================================================

print_header("COMBINED EQUITY CURVES WITH DRAWDOWN (for paper)")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EQUITY_STRATEGIES = {
    "BTP Italia": RESULTS_DIR / "btp_italia" / "index_daily.csv",
    "iTraxx Combined": RESULTS_DIR / "itraxx_combined" / "index_daily.csv",
    "CDS-Bond Basis": RESULTS_DIR / "cds_bond_basis" / "index_daily.csv",
}

COLORS = {
    "BTP Italia": "#1f77b4",
    "iTraxx Combined": "#2ca02c",
    "CDS-Bond Basis": "#d62728",
}

fig, axes = plt.subplots(3, 2, figsize=(12, 11),
                         gridspec_kw={"width_ratios": [1, 1],
                                      "hspace": 0.35, "wspace": 0.05})

for idx, (name, path) in enumerate(EQUITY_STRATEGIES.items()):
    if not path.exists():
        print(f"⚠️  {name}: index_daily.csv not found")
        continue

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if "index_return" in df.columns:
        returns = df["index_return"].dropna() / 100
    elif "Strategy_Return" in df.columns:
        returns = df["Strategy_Return"].dropna() / 100
    else:
        print(f"⚠️  {name}: return column not found")
        continue

    # Cumulative return index (start at 1)
    cum_index = (1 + returns).cumprod()

    # Drawdown
    running_max = cum_index.cummax()
    drawdown = (cum_index - running_max) / running_max * 100

    color = COLORS.get(name, "black")

    # --- Equity curve (left panel = axes[idx, 0]) ---
    ax_eq = axes[idx, 0]
    ax_eq.plot(cum_index.index, cum_index.values, color=color, linewidth=0.8)
    ax_eq.set_ylabel("Cumulative Return", fontsize=9)
    ax_eq.set_title(name, fontsize=11, fontweight="bold")
    ax_eq.grid(True, alpha=0.2)
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Annotate sample period
    start = cum_index.index.min().strftime("%b %Y")
    end = cum_index.index.max().strftime("%b %Y")
    ax_eq.text(0.02, 0.95, f"{start} — {end}",
               transform=ax_eq.transAxes, fontsize=7,
               verticalalignment="top", fontstyle="italic", color="grey")

    # --- Drawdown (right panel = axes[idx, 1]) ---
    ax_dd = axes[idx, 1]
    ax_dd.fill_between(drawdown.index, drawdown.values, 0,
                       color=color, alpha=0.3)
    ax_dd.plot(drawdown.index, drawdown.values, color=color,
               linewidth=0.6, alpha=0.7)
    ax_dd.set_ylabel("Drawdown (%)", fontsize=9)
    ax_dd.set_title(f"{name} — Drawdown", fontsize=11, fontweight="bold")
    ax_dd.grid(True, alpha=0.2)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Max drawdown annotation
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax_dd.axhline(max_dd, color=color, linestyle="--", linewidth=0.8, alpha=0.5)
    ax_dd.text(0.98, 0.05, f"Max DD: {max_dd:.1f}%",
               transform=ax_dd.transAxes, fontsize=7,
               ha="right", color=color, fontweight="bold")

    print(f"   {name}: {len(returns)} days, max DD = {max_dd:.1f}%")

fig.savefig(FIGURES_DIR / "equity_curves_combined.pdf", bbox_inches="tight")
plt.close()

print(f"\n💾 equity_curves_combined.pdf → results/figures/")


# ============================================================================
# FEE SCHEDULE TABLE (reads from 02a, 02b, 02c config sections)
# ============================================================================

print("\n" + "=" * 72)
print("GENERATING FEE SCHEDULE TABLE")
print("=" * 72)

import importlib.util, sys as _sys

def _load_fee_constants(filepath):
    """Import a 02*.py file and extract FEE_* constants without executing it."""
    # Parse file for FEE constants (avoid running the whole simulation)
    constants = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_s = line.strip()
            # Match lines like: FEE_LOW_BPS  = 3.0
            if line_s.startswith('FEE_') and '=' in line_s and not line_s.startswith('#'):
                parts = line_s.split('=', 1)
                key = parts[0].strip()
                try:
                    val = float(parts[1].split('#')[0].strip())
                    constants[key] = val
                except ValueError:
                    pass
            # Match ITRAXX thresholds
            if line_s.startswith('ITRAXX_MAIN_LOW_THRESHOLD') and '=' in line_s:
                parts = line_s.split('=', 1)
                try:
                    constants['ITRAXX_MAIN_LOW_THRESHOLD'] = float(parts[1].split('#')[0].strip())
                except ValueError:
                    pass
            if line_s.startswith('ITRAXX_MAIN_HIGH_THRESHOLD') and '=' in line_s:
                parts = line_s.split('=', 1)
                try:
                    constants['ITRAXX_MAIN_HIGH_THRESHOLD'] = float(parts[1].split('#')[0].strip())
                except ValueError:
                    pass
    return constants

# Paths to strategy files
SCRIPTS_DIR = Path(__file__).parent
btp_cfg   = _load_fee_constants(SCRIPTS_DIR / "02a_BTP-Ita.py")
itrx_cfg  = _load_fee_constants(SCRIPTS_DIR / "02b_itraxx_combined.py")
cds_cfg   = _load_fee_constants(SCRIPTS_DIR / "02c_CDS-Bond_basis.py")

# Extract thresholds (same across all strategies)
thr_low  = btp_cfg.get('ITRAXX_MAIN_LOW_THRESHOLD', 60.0)
thr_high = btp_cfg.get('ITRAXX_MAIN_HIGH_THRESHOLD', 120.0)

# Build rows: (label, low, mid, high)
fee_rows = [
    ("BTP Italia",
     btp_cfg.get('FEE_LOW_BPS', 0), btp_cfg.get('FEE_MID_BPS', 0), btp_cfg.get('FEE_HIGH_BPS', 0)),
    ("CDS--Bond Basis",
     cds_cfg.get('FEE_LOW_BPS', 0), cds_cfg.get('FEE_MID_BPS', 0), cds_cfg.get('FEE_HIGH_BPS', 0)),
    ("iTraxx Main",
     itrx_cfg.get('FEE_MAIN_LOW_BPS', 0), itrx_cfg.get('FEE_MAIN_MID_BPS', 0), itrx_cfg.get('FEE_MAIN_HIGH_BPS', 0)),
    ("iTraxx SnrFin",
     itrx_cfg.get('FEE_SNRFIN_LOW_BPS', 0), itrx_cfg.get('FEE_SNRFIN_MID_BPS', 0), itrx_cfg.get('FEE_SNRFIN_HIGH_BPS', 0)),
    ("iTraxx SubFin",
     itrx_cfg.get('FEE_SUBFIN_LOW_BPS', 0), itrx_cfg.get('FEE_SUBFIN_MID_BPS', 0), itrx_cfg.get('FEE_SUBFIN_HIGH_BPS', 0)),
    ("iTraxx Xover",
     itrx_cfg.get('FEE_XOVER_LOW_BPS', 0), itrx_cfg.get('FEE_XOVER_MID_BPS', 0), itrx_cfg.get('FEE_XOVER_HIGH_BPS', 0)),
]

tex_path = TABLES_DIR / "fee_schedule_article.tex"

with open(tex_path, 'w', encoding='utf-8') as f:
    f.write("\\begin{table}[H]\n")
    f.write("\\centering\n")
    f.write("\\caption{Transaction Fee Schedule (bps per unit DV01 or duration)}\n")
    f.write("\\label{tab:fee_schedule}\n")
    f.write("\\begin{threeparttable}\n")
    f.write("\\begin{singlespace}\n")
    f.write("\\small\n")
    f.write("\\begin{tabular}{lccc}\n")
    f.write("\\toprule\n")
    f.write(f" & LOW & MID & HIGH \\\\\n")
    f.write(f" & (Main $<$ {thr_low:.0f}) & ({thr_low:.0f}--{thr_high:.0f}) & (Main $>$ {thr_high:.0f}) \\\\\n")
    f.write("\\midrule\n")
    for label, low, mid, high in fee_rows:
        f.write(f"{label} & {low:.0f} & {mid:.0f} & {high:.0f} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\begin{tablenotes}[para,flushleft]\n")
    f.write("\\footnotesize\n")
    f.write("Fee applied as: entry DV01 $\\times$ fee\\_bps at entry $+$ exit DV01 $\\times$ fee\\_bps at exit. "
            "Exit fee waived for trades held to maturity. "
            f"Regime determined by iTraxx Main 5Y level at trade date: "
            f"LOW ($<${thr_low:.0f} bps), MID ({thr_low:.0f}--{thr_high:.0f} bps), HIGH ($>${thr_high:.0f} bps).\n")
    f.write("\\end{tablenotes}\n")
    f.write("\\end{singlespace}\n")
    f.write("\\end{threeparttable}\n")
    f.write("\\end{table}\n")

print(f"\n💾 {tex_path.name} → results/tables/")
for label, low, mid, high in fee_rows:
    print(f"   {label:20s}  LOW={low:.0f}  MID={mid:.0f}  HIGH={high:.0f}")

# ============================================================================
# ENTRY/EXIT THRESHOLDS TABLE (reads from 02a, 02b, 02c config sections)
# ============================================================================

print("\n" + "=" * 72)
print("GENERATING ENTRY/EXIT THRESHOLDS TABLE")
print("=" * 72)

def _load_thresholds(filepath):
    """Parse a 02*.py file and extract ENTRY/EXIT threshold constants."""
    constants = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_s = line.strip()
            if '=' in line_s and not line_s.startswith('#') and not line_s.startswith('def '):
                for key in ['ENTRY_LONG_THRESHOLD', 'ENTRY_SHORT_THRESHOLD',
                            'EXIT_LONG_THRESHOLD', 'EXIT_SHORT_THRESHOLD',
                            'ENTRY_THRESHOLD', 'EXIT_THRESHOLD',
                            'MIN_MONTHS_TO_MATURITY_ENTRY', 'MIN_OPEN_TRADES',
                            'MIN_TRADE_DURATION_DAYS']:
                    if line_s.startswith(key):
                        parts = line_s.split('=', 1)
                        try:
                            val = parts[1].split('#')[0].strip()
                            if val == 'None':
                                constants[key] = None
                            else:
                                constants[key] = float(val)
                        except (ValueError, IndexError):
                            pass
    return constants

def _load_index_params(filepath):
    """Parse 02b iTraxx file to extract per-index thresholds from INDEX_PARAMS dict."""
    params = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    import ast
    # Find INDEX_PARAMS block
    start = content.find('INDEX_PARAMS = {')
    if start == -1:
        return params
    # Find the matching closing brace
    brace_count = 0
    end = start
    for i, ch in enumerate(content[start:], start):
        if ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i + 1
                break
    try:
        params = ast.literal_eval(content[start + len('INDEX_PARAMS = '):end])
    except (SyntaxError, ValueError):
        pass
    return params

btp_thr = _load_thresholds(SCRIPTS_DIR / "02a_BTP-Ita.py")
cds_thr = _load_thresholds(SCRIPTS_DIR / "02c_CDS-Bond_basis.py")
itrx_params = _load_index_params(SCRIPTS_DIR / "02b_itraxx_combined.py")

# Build rows
threshold_rows = []

# BTP Italia — bidirectional
threshold_rows.append((
    "BTP Italia",
    f"$>{btp_thr.get('ENTRY_LONG_THRESHOLD', 40):.0f}$",
    f"$<{btp_thr.get('ENTRY_SHORT_THRESHOLD', -50):.0f}$",
    f"$<{btp_thr.get('EXIT_LONG_THRESHOLD', 10):.0f}$",
    f"$>{btp_thr.get('EXIT_SHORT_THRESHOLD', -10):.0f}$",
))

# CDS-Bond — unidirectional (negative basis only)
threshold_rows.append((
    "CDS--Bond Basis",
    f"$<{cds_thr.get('ENTRY_THRESHOLD', -40):.0f}$",
    "--",
    f"$>{cds_thr.get('EXIT_THRESHOLD', 0):.0f}$",
    "--",
))

# iTraxx — per index
for idx_name in ['Main', 'SnrFin', 'SubFin', 'Xover']:
    p = itrx_params.get(idx_name, {})
    threshold_rows.append((
        f"iTraxx {idx_name}",
        f"$>{p.get('entry_long', 0):.0f}$",
        f"$<{p.get('entry_short', 0):.0f}$",
        f"$<{p.get('exit_long', 0):.0f}$",
        f"$>{p.get('exit_short', 0):.0f}$",
    ))

min_maturity = btp_thr.get('MIN_MONTHS_TO_MATURITY_ENTRY', 6)
min_duration = btp_thr.get('MIN_TRADE_DURATION_DAYS', 3)

tex_path = TABLES_DIR / "entry_exit_thresholds_article.tex"

with open(tex_path, 'w', encoding='utf-8') as f:
    f.write("\\begin{table}[H]\n")
    f.write("\\centering\n")
    f.write("\\caption{Entry and Exit Thresholds by Strategy}\n")
    f.write("\\label{tab:entry_exit_rules}\n")
    f.write("\\begin{threeparttable}\n")
    f.write("\\begin{singlespace}\n")
    f.write("\\small\n")
    f.write("\\begin{tabular}{lrrrr}\n")
    f.write("\\toprule\n")
    f.write("Strategy & Entry Long & Entry Short & Exit Long & Exit Short \\\\\n")
    f.write("\\midrule\n")
    
    for row in threshold_rows:
        f.write(f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & {row[4]} \\\\\n")
    
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\begin{tablenotes}[para,flushleft]\n")
    f.write("\\footnotesize\n")
    f.write(f"All thresholds in bps. CDS--Bond Basis is a one-directional negative "
            f"basis trade (long bond, buy CDS protection). "
            f"Minimum {min_maturity:.0f} months to maturity at entry for all strategies. "
            f"Trades shorter than {min_duration:.0f} trading days excluded ex-post.\n")
    f.write("\\end{tablenotes}\n")
    f.write("\\end{singlespace}\n")
    f.write("\\end{threeparttable}\n")
    f.write("\\end{table}\n")

print(f"\n💾 {tex_path.name} → results/tables/")
for row in threshold_rows:
    if row[2] is None:
        print(f"   {row[0]:20s}  Entry: {row[1]}  Exit: {row[3]}")
    else:
        print(f"   {row[0]:20s}  Entry: {row[1]} / {row[2]}  Exit: {row[3]} / {row[4]}")
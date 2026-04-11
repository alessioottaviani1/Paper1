"""
================================================================================
03_pca_results_tables.py - PCA Results: Tables and Plots
================================================================================
Genera output per PCA rolling:

1. TABELLE LATEX (stile Duarte et al. 2007):
   - Alpha + Beta su tutti i PC
   - t-stat in parentesi
   - R² adj, N obs
   - Se esistono sia predictive che contemporaneous: due Panel

2. GRAFICI DIAGNOSTICI PCA (solo PNG per LaTeX):
   - Scree plot (varianza spiegata per PC)
   - Time series della varianza spiegata totale

Author: Alessio Ottaviani
Institution: EDHEC Business School - PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IMPORT CONFIG
# ============================================================================

import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Try to load config
config_paths = [
    PROJECT_ROOT / "src" / "pca" / "00_pca_config.py",
    PROJECT_ROOT / "src" / "pca" / "00_pca_config_fix.py",
]

pca_config = None
for config_path in config_paths:
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("pca_config", config_path)
        pca_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pca_config)
        break

if pca_config is None:
    raise FileNotFoundError("PCA config file not found!")

# Export variables
RESULTS_DIR = pca_config.RESULTS_DIR
STRATEGIES = pca_config.STRATEGIES
PCA_N_COMPONENTS = pca_config.PCA_N_COMPONENTS
PCA_WINDOW_LENGTH = pca_config.PCA_WINDOW_LENGTH
get_pca_output_dir = pca_config.get_pca_output_dir
get_strategy_pca_dir = pca_config.get_strategy_pca_dir

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directories
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures" / "pca"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Scree plot configuration
SCREE_PLOT_N_COMPONENTS = 15  # Numero di PC da mostrare nello scree plot (più di quelli usati)

# Plot style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'tertiary': '#F18F01',
    'quaternary': '#C73E1D',
    'light': '#E8E8E8'
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title, char="="):
    """Print formatted section header."""
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")


def detect_available_timings() -> list:
    """Detect which timing conventions have been run."""
    pca_dir = get_pca_output_dir()
    available = []
    
    for timing in ['predictive', 'contemporaneous']:
        # Check for PC scores file with timing suffix
        if (pca_dir / f"pc_scores_{timing}.parquet").exists():
            available.append(timing)
        # Also check old naming (without suffix) - assume predictive
        elif timing == 'predictive' and (pca_dir / "pc_scores.parquet").exists():
            available.append('predictive_legacy')
    
    return available


def load_pca_results(timing: str) -> dict:
    """
    Load PCA results for a given timing convention.
    
    Returns:
        dict with 'pc_scores', 'diagnostics', 'summary', 'strategy_results'
    """
    pca_dir = get_pca_output_dir()
    
    # Handle legacy naming
    if timing == 'predictive_legacy':
        pc_scores_path = pca_dir / "pc_scores.parquet"
        diagnostics_path = pca_dir / "pca_diagnostics.csv"
        summary_path = pca_dir / "pca_summary.json"
        timing_label = 'predictive'
    else:
        pc_scores_path = pca_dir / f"pc_scores_{timing}.parquet"
        diagnostics_path = pca_dir / f"pca_diagnostics_{timing}.csv"
        summary_path = pca_dir / f"pca_summary_{timing}.json"
        timing_label = timing
    
    results = {'timing': timing_label}
    
    # Load PC scores
    if pc_scores_path.exists():
        results['pc_scores'] = pd.read_parquet(pc_scores_path)
    else:
        print(f"   ⚠️ PC scores not found: {pc_scores_path}")
        results['pc_scores'] = None
    
    # Load diagnostics
    if diagnostics_path.exists():
        results['diagnostics'] = pd.read_csv(diagnostics_path)
    else:
        print(f"   ⚠️ Diagnostics not found: {diagnostics_path}")
        results['diagnostics'] = None
    
    # Load summary
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            results['summary'] = json.load(f)
    else:
        print(f"   ⚠️ Summary not found: {summary_path}")
        results['summary'] = None
    
    # Load strategy-specific results
    results['strategy_results'] = {}
    
    for strategy_name in STRATEGIES.keys():
        strategy_dir = get_strategy_pca_dir(strategy_name)
        
        if timing == 'predictive_legacy':
            reg_path = strategy_dir / "spanning_regression_results.json"
        else:
            reg_path = strategy_dir / f"spanning_regression_results_{timing}.json"
        
        if reg_path.exists():
            with open(reg_path, 'r') as f:
                results['strategy_results'][strategy_name] = json.load(f)
    
    return results


# ============================================================================
# LATEX TABLE GENERATION
# ============================================================================

def generate_latex_tables(all_results: dict):
    """
    Generate LaTeX tables in Duarte et al. (2007) style.
    
    Args:
        all_results: dict with timing as key, results dict as value
    """
    print_header("GENERATING LATEX TABLES")
    
    timings = list(all_results.keys())
    n_timings = len(timings)
    
    print(f"   Timings available: {timings}")
    
    # Determine number of PCs from first available result
    first_timing = timings[0]
    first_strategy = list(all_results[first_timing]['strategy_results'].keys())[0]
    first_result = all_results[first_timing]['strategy_results'][first_strategy]
    n_components = first_result.get('n_components', PCA_N_COMPONENTS)
    
    pc_names = [f'PC{i+1}' for i in range(n_components)]
    
    # ========================================================================
    # FILE 1: COMBINED TABLE (Alpha comparison + Full results)
    # ========================================================================
    
    latex_path = TABLES_DIR / "PCA_spanning_regression_results.tex"
    
    with open(latex_path, 'w') as f:
        
        # Header
        f.write(f"% {'='*76}\n")
        f.write(f"% PCA SPANNING REGRESSION RESULTS\n")
        f.write(f"% Rolling Window: {PCA_WINDOW_LENGTH} months\n")
        f.write(f"% Number of PCs: {n_components}\n")
        f.write(f"% Timings: {', '.join(timings)}\n")
        f.write(f"% {'='*76}\n\n")
        
        # ====================================================================
        # TABLE 1: ALPHA COMPARISON (compact)
        # ====================================================================
        
        f.write(f"% {'='*76}\n")
        f.write(f"% TABLE 1: ALPHA COMPARISON\n")
        f.write(f"% {'='*76}\n\n")
        
        f.write("\\centering\n")
        f.write("\\begin{threeparttable}\n")

        
        if n_timings == 1:
            # Single timing
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\toprule\n")
            f.write("Strategy & $\\alpha$ (\\% p.a.) & $R^2$ adj & N \\\\\n")
            f.write("\\midrule\n")
            
            timing = timings[0]
            for strategy_name, result in all_results[timing]['strategy_results'].items():
                strategy_display = strategy_name.replace('_', ' ').title()
                
                alpha = result['alpha'] * 12  # Annualize
                tstat = result['alpha_tstat']
                rsq_adj = result['r_squared_adj']
                nobs = result['n_obs']
                
                # Significance stars
                pval = result['alpha_pvalue']
                if pval < 0.01:
                    sig = '***'
                elif pval < 0.05:
                    sig = '**'
                elif pval < 0.10:
                    sig = '*'
                else:
                    sig = ''
                
                f.write(f"\\textit{{{strategy_display}}} & {alpha:.2f}{sig} & {rsq_adj:.3f} & {int(nobs)} \\\\\n")
                f.write(f" & ({tstat:.2f}) & & \\\\\n")
                f.write("\\addlinespace\n")
            
        else:
            # Multiple timings (side by side)
            f.write("\\begin{tabular}{lcccccc}\n")
            f.write("\\toprule\n")
            f.write(" & \\multicolumn{3}{c}{Predictive ($PC_t \\to R_{t+1}$)} & \\multicolumn{3}{c}{Contemporaneous ($PC_t \\to R_t$)} \\\\\n")
            f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n")
            f.write("Strategy & $\\alpha$ (\\%) & $R^2$ adj & N & $\\alpha$ (\\%) & $R^2$ adj & N \\\\\n")
            f.write("\\midrule\n")
            
            # Get all strategies
            all_strategies = set()
            for timing in timings:
                all_strategies.update(all_results[timing]['strategy_results'].keys())
            
            for strategy_name in sorted(all_strategies):
                strategy_display = strategy_name.replace('_', ' ').title()
                f.write(f"\\textit{{{strategy_display}}} ")
                
                # Predictive
                if 'predictive' in all_results and strategy_name in all_results['predictive']['strategy_results']:
                    result = all_results['predictive']['strategy_results'][strategy_name]
                    alpha = result['alpha'] * 12
                    tstat = result['alpha_tstat']
                    pval = result['alpha_pvalue']
                    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                    f.write(f"& {alpha:.2f}{sig} & {result['r_squared_adj']:.3f} & {int(result['n_obs'])} ")
                else:
                    f.write("& -- & -- & -- ")
                
                # Contemporaneous
                if 'contemporaneous' in all_results and strategy_name in all_results['contemporaneous']['strategy_results']:
                    result = all_results['contemporaneous']['strategy_results'][strategy_name]
                    alpha = result['alpha'] * 12
                    tstat = result['alpha_tstat']
                    pval = result['alpha_pvalue']
                    sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                    f.write(f"& {alpha:.2f}{sig} & {result['r_squared_adj']:.3f} & {int(result['n_obs'])} ")
                else:
                    f.write("& -- & -- & -- ")
                
                f.write("\\\\\n")
                
                # t-stat row
                f.write(" ")
                if 'predictive' in all_results and strategy_name in all_results['predictive']['strategy_results']:
                    tstat = all_results['predictive']['strategy_results'][strategy_name]['alpha_tstat']
                    f.write(f"& ({tstat:.2f}) & & ")
                else:
                    f.write("& & & ")
                
                if 'contemporaneous' in all_results and strategy_name in all_results['contemporaneous']['strategy_results']:
                    tstat = all_results['contemporaneous']['strategy_results'][strategy_name]['alpha_tstat']
                    f.write(f"& ({tstat:.2f}) & & ")
                else:
                    f.write("& & & ")
                
                f.write("\\\\\n")
                f.write("\\addlinespace\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        # --- compute PCA variance explained diagnostics (use first available timing) ---
        diag_timing = timings[0]
        avg_var = min_var = max_var = None
        try:
            if all_results.get(diag_timing, {}).get('summary') is not None:
                diag = all_results[diag_timing]['summary'].get('pca_diagnostics', {})
                avg_var = diag.get('avg_variance_explained', None)
                min_var = diag.get('min_variance_explained', None)
                max_var = diag.get('max_variance_explained', None)
        except Exception:
            pass

        f.write("\\begin{tablenotes}\n")
        f.write("\\small\n")

        # Notes line 1
        f.write(f"\\item \\textit{{Notes:}} Alpha is annualized. ")
        f.write(f"PC1--PC{n_components} are the first {n_components} principal components extracted from {PCA_WINDOW_LENGTH}-month rolling windows. ")

        # Notes line 2 (PCA diagnostics: mean/min/max) — only if available
        if avg_var is not None:
            avg_pct = 100 * float(avg_var)
            # min/max are optional: include only if present
            if (min_var is not None) and (max_var is not None):
                min_pct = 100 * float(min_var)
                max_pct = 100 * float(max_var)
                f.write(f"The first {n_components} PCs explain on average {avg_pct:.1f}\\% of factor variance (min {min_pct:.1f}\\%, max {max_pct:.1f}\\%). ")
            else:
                f.write(f"The first {n_components} PCs explain on average {avg_pct:.1f}\\% of factor variance. ")

        # Notes line 3
        f.write("t-statistics in parentheses based on Newey-West HAC standard errors.\n")
        f.write("\\item *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{threeparttable}\n\n")
        
        # ====================================================================
        # TABLE 2: FULL RESULTS WITH ALL BETAS
        # ====================================================================
        
        f.write(f"% {'='*76}\n")
        f.write(f"% TABLE 2: FULL MODEL RESULTS (ALPHA + ALL BETAS)\n")
        f.write(f"% {'='*76}\n\n")
        
        f.write("\\centering\n")
        f.write("\\begin{threeparttable}\n")
        f.write("\\setlength{\\tabcolsep}{2pt}\n")          
        f.write("\\renewcommand{\\arraystretch}{1.05}\n")

        # Number of columns: Strategy + Alpha + n_components betas + R2 + N
        n_c_cols = 1 + n_components + 2   # alpha + betas + (R2,N)  -> tutte "c"
        f.write(f"\\begin{{tabular}}{{l{'c'*n_c_cols}}}\n")
        total_cols = 1 + n_c_cols         # totale colonne tabella (1 l + c...)
        f.write("\\toprule\n")
        
        # Header row
        f.write("Strategy & $\\alpha$ (\\%)")
        for pc in pc_names:
            f.write(f" & $\\beta_{{{pc}}}$")
        f.write(" & $R^2$ adj & N \\\\\n")
        f.write("\\midrule\n")
        
        # Generate panel for each timing
        for panel_idx, timing in enumerate(timings):
            timing_label = timing.replace('_', ' ').title()
            
            if timing == 'predictive':
                panel_title = "Panel A: Predictive ($PC_t \\to R_{t+1}$)"
            elif timing == 'contemporaneous':
                panel_title = "Panel B: Contemporaneous ($PC_t \\to R_t$)"
            else:
                panel_title = f"Panel {chr(65+panel_idx)}: {timing_label}"
            f.write(f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{{panel_title}}}}} \\\\\n")
            f.write("\\addlinespace\n")
            
            for strategy_name, result in all_results[timing]['strategy_results'].items():
                strategy_display = strategy_name.replace('_', ' ').title()
                
                # Alpha
                alpha = result['alpha'] * 12  # Annualize
                alpha_tstat = result['alpha_tstat']
                alpha_pval = result['alpha_pvalue']
                alpha_sig = '***' if alpha_pval < 0.01 else '**' if alpha_pval < 0.05 else '*' if alpha_pval < 0.10 else ''
                
                # First row: coefficients
                f.write(f"\\textit{{{strategy_display}}} & {alpha:.2f}{alpha_sig}")
                
                # Betas
                betas = result.get('betas', {})
                betas_tstat = result.get('betas_tstat', {})
                betas_pval = result.get('betas_pvalue', {})
                
                for pc in pc_names:
                    if pc in betas:
                        beta = betas[pc]
                        tstat = betas_tstat.get(pc, 0)
                        pval = betas_pval.get(pc, 1)
                        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
                        f.write(f" & {beta:.3f}{sig}")
                    else:
                        f.write(" & --")
                
                f.write(f" & {result['r_squared_adj']:.3f} & {int(result['n_obs'])}")
                f.write(" \\\\\n")
                
                # Second row: t-statistics
                f.write(f" & ({alpha_tstat:.2f})")
                
                for pc in pc_names:
                    if pc in betas_tstat:
                        f.write(f" & ({betas_tstat[pc]:.2f})")
                    else:
                        f.write(" & ")
                
                f.write(" & & \\\\\n")
                f.write("\\addlinespace\n")
            
            if panel_idx < len(timings) - 1:
                f.write("\\midrule\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\begin{tablenotes}\n")
        f.write("\\tiny\n")
        f.write(f"\\item This table reports spanning regression results of monthly strategy excess returns on the first {n_components} principal components. ")
        f.write(f"Principal components are extracted from a rolling window of {PCA_WINDOW_LENGTH} months. ")
        f.write(
            f"Alpha is reported in annualized percentage terms. "
            f"t-statistics in parentheses are based on Newey-West HAC standard errors. "
            f"The first {n_components} PCs explain on average {avg_pct:.1f}\\% of factor variance "
            f"(min {min_pct:.1f}\\%, max {max_pct:.1f}\\%). "
        )
        #f.write("Alpha is reported in annualized percentage terms. t-statistics in parentheses are based on Newey-West HAC standard errors. The first {n_components} PCs explain on average {avg_pct:.1f}\\% of factor variance (min {min_pct:.1f}\\%, max {max_pct:.1f}\\%). ")
        f.write("\\item *** p$<$0.01, ** p$<$0.05, * p$<$0.10.\n") 
        f.write("\\end{tablenotes}\n")
        
        # Regression equation
        eq_betas = " + ".join([f"\\beta_{{{i+1}}} PC{i+1}_t" for i in range(min(3, n_components))])
        if n_components > 3:
            eq_betas += " + \\ldots"
        
        f.write("\\vspace{0.05cm}\n")
        f.write("\\tiny\n")
        f.write("\\end{threeparttable}\n\n")
        
            
    print(f"   💾 Saved: {latex_path}")
    
    return latex_path


# ============================================================================
# DIAGNOSTIC PLOTS (SOLO PDF)
# ============================================================================

def generate_scree_plot(all_results: dict):
    """
    Generate single scree plot showing variance explained by each PC.
    Shows more PCs than used, highlights the K used in regression.
    Output: PNG (for LaTeX)
    """
    print_header("GENERATING SCREE PLOT", "-")
    
    # Use first available timing (variance is identical across timings)
    timings = list(all_results.keys())
    timing = timings[0]
    
    # Get summary data
    summary = all_results[timing].get('summary', {})
    pca_diag = summary.get('pca_diagnostics', {})
    
    # Get per-PC variance if available
    avg_variance_per_pc = pca_diag.get('avg_variance_per_pc', [])
    n_components_used = pca_diag.get('n_components', PCA_N_COMPONENTS)
    
    # Extend to show more PCs if we only have data for K used
    n_to_show = min(SCREE_PLOT_N_COMPONENTS, len(avg_variance_per_pc)) if avg_variance_per_pc else SCREE_PLOT_N_COMPONENTS
    
    if not avg_variance_per_pc:
        print("   ⚠️  No per-PC variance data available. Skipping scree plot.")
        return None
    
    # Show only PCs for which we have real data — never fabricate values
    n_to_show = min(SCREE_PLOT_N_COMPONENTS, len(avg_variance_per_pc))
    avg_variance_per_pc = avg_variance_per_pc[:n_to_show]

    cumulative_vars = np.cumsum(avg_variance_per_pc)
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pc_labels = [f'PC{i+1}' for i in range(len(avg_variance_per_pc))]
    x = np.arange(len(avg_variance_per_pc))
    
    # Bar colors: highlight the PCs actually used in regression
    bar_colors = [COLORS['primary'] if i < n_components_used else COLORS['light'] 
                  for i in range(len(avg_variance_per_pc))]
    
    # Bar plot for individual variance
    bars = ax.bar(x, [v*100 for v in avg_variance_per_pc], 
                  color=bar_colors, alpha=0.8, edgecolor='white', linewidth=0.5)
    
    # Line plot for cumulative variance (secondary axis)
    ax2 = ax.twinx()
    ax2.plot(x, [v*100 for v in cumulative_vars], 
             color=COLORS['secondary'], marker='o', linewidth=2.5, 
             markersize=7, label='Cumulative', zorder=5)
    
    
    # Add vertical line at cutoff (K used)
    ax.axvline(n_components_used - 0.5, color=COLORS['quaternary'], linestyle='--', 
               linewidth=2, alpha=0.8, label=f'K={n_components_used} used')
    
    # Formatting
    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Individual Variance Explained (%)', fontsize=12)
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(pc_labels, fontsize=9)
    
    max_individual = max([v*100 for v in avg_variance_per_pc]) if avg_variance_per_pc else 50
    ax.set_ylim(0, max_individual * 1.25)
    ax2.set_ylim(0, 105)
    
    # Add value labels on bars (all PCs)
    for i, (bar, val) in enumerate(zip(bars, avg_variance_per_pc)):
        if val > 0:
            # Bold for used PCs, normal for others
            weight = 'bold' if i < n_components_used else 'normal'
            color = 'black' if i < n_components_used else 'gray'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                   f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8, 
                   fontweight=weight, color=color)
           
       
    # Add cumulative % text at cutoff (no arrow)
    cum_at_cutoff = cumulative_vars[n_components_used - 1] * 100
    ax2.text(n_components_used - 1, cum_at_cutoff + 3, f'{cum_at_cutoff:.1f}%',
             fontsize=10, fontweight='bold', color=COLORS['secondary'],
             ha='center', va='bottom')
    
    
    # Legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    # Title
    ax.set_title(f'PCA Variance Explained by Principal Component\n(Rolling Window: {PCA_WINDOW_LENGTH} months, K={n_components_used} used in regressions)', 
                 fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    # Save PNG
    plot_path = FIGURES_DIR / "pca_scree_plot.pdf"
    plt.savefig(plot_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   🖼️  Saved: {plot_path}")
    print(f"   📊 Showing {n_to_show} PCs, K={n_components_used} used in regressions")
    print(f"   📊 Cumulative variance at K={n_components_used}: {cum_at_cutoff:.1f}%")
    
    return plot_path


def generate_variance_timeseries(all_results: dict):
    """
    Generate time series plot of variance explained over time.
    Output: PDF only (for LaTeX)
    """
    print_header("GENERATING VARIANCE TIME SERIES", "-")
    
    timings = list(all_results.keys())
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for timing in timings:
        diagnostics = all_results[timing].get('diagnostics')
        
        if diagnostics is None or 'variance_explained' not in diagnostics.columns:
            continue
        
        # Convert date column
        if 'date' in diagnostics.columns:
            diagnostics['date'] = pd.to_datetime(diagnostics['date'])
            x = diagnostics['date']
        else:
            x = range(len(diagnostics))
        
        y = diagnostics['variance_explained'] * 100
        
        timing_label = 'Predictive' if timing == 'predictive' else 'Contemporaneous'
        color = COLORS['primary'] if timing == 'predictive' else COLORS['secondary']
        
        ax.plot(x, y, label=timing_label, color=color, linewidth=1.5)
        
        # Add mean line
        mean_var = y.mean()
        ax.axhline(mean_var, color=color, linestyle='--', alpha=0.5, linewidth=1)
        ax.text(x.iloc[-1], mean_var + 1, f'Mean: {mean_var:.1f}%', 
               fontsize=9, color=color, va='bottom')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Variance Explained (%)', fontsize=11)
    ax.set_title(f'Total Variance Explained by First {PCA_N_COMPONENTS} PCs Over Time\n'
                f'(Rolling Window: {PCA_WINDOW_LENGTH} months)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 100)
    
    # Add shading for crisis periods
    crisis_periods = [
        ('2008-09-01', '2009-06-01', 'GFC'),
        ('2011-07-01', '2012-07-01', 'Euro Crisis'),
        ('2020-02-01', '2020-06-01', 'COVID'),
    ]
    
    for start, end, label in crisis_periods:
        try:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), 
                      alpha=0.2, color='gray')
        except:
            pass
    
    plt.tight_layout()
    
    # Save PNG only
    plot_path = FIGURES_DIR / "pca_variance_timeseries.pdf"
    plt.savefig(plot_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   🖼️  Saved: {plot_path}")
    
    return plot_path

# ============================================================================
# TOP LOADINGS TABLE (sezione 5.1.2 - Interpretazione PC)
# ============================================================================

def generate_top_loadings_table(all_results: dict, n_pcs: int = 3, n_top: int = 6):
    """
    Generate LaTeX table showing top factor loadings for first n_pcs PCs.
    Reads pca_avg_loadings_{timing}.csv produced by 02_pca_rolling.py.
    """
    print_header("GENERATING TOP LOADINGS TABLE", "-")

    # Use first available timing (loadings are essentially the same)
    timings = list(all_results.keys())
    timing = timings[0]

    pca_dir = get_pca_output_dir()
    loadings_path = pca_dir / f"pca_avg_loadings_{timing}.csv"

    if not loadings_path.exists():
        print(f"   ⚠️  Loadings file not found: {loadings_path}")
        print(f"   Re-run 02_pca_rolling.py to generate it.")
        return None

    # Load: rows = PC1..PCK, columns = factor names
    loadings_df = pd.read_csv(loadings_path, index_col=0)
    print(f"   ✅ Loaded loadings: {loadings_df.shape[0]} PCs × {loadings_df.shape[1]} factors")

    # For each PC, get top n_top factors by absolute loading
    top_loadings = {}
    for i in range(min(n_pcs, len(loadings_df))):
        pc_name = f'PC{i+1}'
        row = loadings_df.loc[pc_name]
        sorted_abs = row.abs().sort_values(ascending=False).head(n_top)
        # Keep original sign
        top_loadings[pc_name] = [(factor, row[factor]) for factor in sorted_abs.index]

    # --- Generate LaTeX ---
    tex_path = TABLES_DIR / "PCA_top_loadings.tex"

    with open(tex_path, 'w') as f:
        f.write(f"% {'='*76}\n")
        f.write(f"% PCA TOP FACTOR LOADINGS (average across rolling windows)\n")
        f.write(f"% {'='*76}\n\n")

        f.write("\\centering\n")
        f.write("\\begin{threeparttable}\n")
        f.write("\\setlength{\\tabcolsep}{4pt}\n")
        f.write("\\renewcommand{\\arraystretch}{1.05}\n")

        # Column spec: pairs of (Factor, Loading) for each PC
        col_spec = "l" + "lr" * n_pcs
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")

        # Header row 1: PC names spanning 2 columns each
        headers = []
        for i in range(n_pcs):
            headers.append(f"\\multicolumn{{2}}{{c}}{{PC{i+1}}}")
        f.write(" & ".join(headers) + " \\\\\n")

        # Header row 2: Factor / Loading
        sub_headers = []
        for _ in range(n_pcs):
            sub_headers.append("Factor & Loading")
        f.write(" & ".join(sub_headers) + " \\\\\n")
        f.write("\\midrule\n")

        # Body rows
        for row_idx in range(n_top):
            cells = []
            for i in range(n_pcs):
                pc_name = f'PC{i+1}'
                if row_idx < len(top_loadings[pc_name]):
                    factor, loading = top_loadings[pc_name][row_idx]
                    # Clean factor name for LaTeX (replace underscores)
                    factor_clean = factor.replace('Δ', '$\\Delta$').replace('_', '\\_')
                    cells.append(f"\\texttt{{{factor_clean}}} & {loading:+.3f}")
                else:
                    cells.append("& ")
            f.write(" & ".join(cells) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

        f.write("\\begin{tablenotes}\n")
        f.write("\\small\n")
        f.write(f"\\item \\textit{{Notes:}} Loadings are time-series averages of the eigenvector "
                f"entries across all {PCA_WINDOW_LENGTH}-month rolling windows. "
                f"For each principal component, the {n_top} factors with the largest "
                f"absolute loading are reported. Sign indicates the direction of the relationship.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{threeparttable}\n")

    print(f"   💾 Saved: {tex_path}")

    # Print to console for quick inspection
    print(f"\n   📊 Top {n_top} loadings per PC:")
    for pc_name, factors in top_loadings.items():
        print(f"\n   {pc_name}:")
        for factor, loading in factors:
            print(f"      {factor:<25s} {loading:+.4f}")

    return tex_path

# ============================================================================
# ARTICLE TABLES (for thesis/paper — \begin{table}[H] wrapper)
# ============================================================================

def generate_article_tables(all_results: dict):
    """
    Generate article-format LaTeX tables (thesis/paper).
    Produces PCA_article.tex with:
      - Table 1: Spanning regression full model (Panel A: Predictive, Panel B: Contemporaneous)
      - Table 2: Top factor loadings
    """
    print_header("GENERATING ARTICLE TABLES", "-")

    timings = list(all_results.keys())

    # Determine n_components
    first_timing = timings[0]
    first_strategy = list(all_results[first_timing]['strategy_results'].keys())[0]
    first_result = all_results[first_timing]['strategy_results'][first_strategy]
    n_components = first_result.get('n_components', PCA_N_COMPONENTS)
    pc_names = [f'PC{i+1}' for i in range(n_components)]

    # PCA diagnostics
    diag_timing = timings[0]
    avg_var = min_var = max_var = None
    try:
        if all_results.get(diag_timing, {}).get('summary') is not None:
            diag = all_results[diag_timing]['summary'].get('pca_diagnostics', {})
            avg_var = diag.get('avg_variance_explained', None)
            min_var = diag.get('min_variance_explained', None)
            max_var = diag.get('max_variance_explained', None)
    except Exception:
        pass

    # Collect N per strategy for notes
    n_info = {}
    for timing in timings:
        for sname, sres in all_results[timing]['strategy_results'].items():
            if sname not in n_info:
                n_info[sname] = int(sres['n_obs'])

    strategy_map = {
        'btp_italia': 'BTP Italia',
        'cds_bond_basis': 'CDS--Bond',
        'itraxx_combined': 'Index Skew',
    }

    def sig_stars_super(pval):
        """Stars in superscript format for article tables."""
        if pval < 0.01:
            return '***'
        elif pval < 0.05:
            return '**'
        elif pval < 0.10:
            return '*'
        return ''

    tex_path = TABLES_DIR / "PCA_article.tex"

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("% " + "=" * 74 + "\n")
        f.write("% PCA SPANNING REGRESSION — ARTICLE TABLES\n")
        f.write(f"% Rolling Window: {PCA_WINDOW_LENGTH} months, K = {n_components}\n")
        f.write("% " + "=" * 74 + "\n\n")

        # ==================================================================
        # TABLE 1: FULL MODEL (Alpha + all Betas)
        # ==================================================================
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{PCA Spanning Regressions}\n")
        f.write("\\label{tab:pca_spanning}\n")
        f.write("\\begin{threeparttable}\n")
        f.write("\\begin{singlespace}\n")
        f.write("\\small\n")
        f.write("\\setlength{\\tabcolsep}{1.5pt}\n\n")

        # Column spec: l + alpha + n_components betas + R2 + N
        n_data_cols = 1 + n_components + 2
        f.write(f"\\begin{{tabular}}{{l{'c' * n_data_cols}}}\n")
        f.write("\\toprule\n")

        # Header
        f.write(" & $\\alpha$ (\\%)")
        for pc in pc_names:
            f.write(f" & $\\beta_{{{pc}}}$")
        f.write(" & $\\bar{R}^2$ & $N$")
        f.write(" \\\\\n")
        f.write("\\midrule\n")

        # Panels
        for panel_idx, timing in enumerate(timings):
            if timing == 'predictive':
                panel_title = "Panel A: Predictive ($PC_t \\to R_{t+1}$)"
            elif timing == 'contemporaneous':
                panel_title = "Panel B: Contemporaneous ($PC_t \\to R_t$)"
            else:
                panel_title = f"Panel {chr(65 + panel_idx)}: {timing.title()}"

            total_cols = 1 + n_data_cols
            f.write(f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{{panel_title}}}}} \\\\\n")
            f.write("\\addlinespace\n")

            for strategy_name, result in all_results[timing]['strategy_results'].items():
                display = strategy_map.get(strategy_name, strategy_name.replace('_', ' ').title())

                # Alpha
                alpha = result['alpha'] * 12
                alpha_tstat = result['alpha_tstat']
                alpha_pval = result['alpha_pvalue']
                stars = sig_stars_super(alpha_pval)

                # Coefficient row
                f.write(f"\\textit{{{display}}}")
                if stars:
                    f.write(f" & ${alpha:.2f}^{{{stars}}}$")
                else:
                    f.write(f" & {alpha:.2f}")

                # Betas
                betas = result.get('betas', {})
                betas_pval = result.get('betas_pvalue', {})
                for pc in pc_names:
                    if pc in betas:
                        beta = betas[pc]
                        pval = betas_pval.get(pc, 1)
                        bstars = sig_stars_super(pval)
                        if bstars:
                            f.write(f" & ${beta:.3f}^{{{bstars}}}$")
                        else:
                            f.write(f" & {beta:.3f}")
                    else:
                        f.write(" & --")

                f.write(f" & {result['r_squared_adj']:.3f}")
                f.write(f" & {int(result['n_obs'])}")
                f.write(" \\\\\n")

                # t-stat row
                f.write(f"  & ({alpha_tstat:.2f})")
                betas_tstat = result.get('betas_tstat', {})
                for pc in pc_names:
                    if pc in betas_tstat:
                        f.write(f" & ({betas_tstat[pc]:.2f})")
                    else:
                        f.write(" & ")
                f.write(" & & \\\\\n")
                f.write("\\addlinespace\n")

            if panel_idx < len(timings) - 1:
                f.write("\\midrule\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n\n")

        # Notes
        f.write("\\begin{tablenotes}[para,flushleft]\n")
        f.write("\\footnotesize\n")
        f.write("\\item \\textit{Note:} ")
        f.write(f"Monthly strategy excess returns regressed on the first {n_components} "
                f"principal components extracted from {PCA_WINDOW_LENGTH}-month rolling windows. ")
        f.write("$\\alpha$ is annualized (multiplied by 12). ")
        f.write("$t$-statistics in parentheses (Newey--West HAC). ")

        if avg_var is not None:
            avg_pct = 100 * float(avg_var)
            if min_var is not None and max_var is not None:
                min_pct = 100 * float(min_var)
                max_pct = 100 * float(max_var)
                f.write(f"The first {n_components} PCs explain on average {avg_pct:.1f}\\% "
                        f"of factor variance (min {min_pct:.1f}\\%, max {max_pct:.1f}\\%). ")
            else:
                f.write(f"The first {n_components} PCs explain on average {avg_pct:.1f}\\% "
                        f"of factor variance. ")

        sample_str = ", ".join(
            f"{strategy_map.get(s, s)} {n_info[s]} months"
            for s in all_results[timings[0]]['strategy_results'].keys()
            if s in n_info)
        f.write(f"Sample: {sample_str}. ")
        f.write("$^{***}\\,p<0.01$, $^{**}\\,p<0.05$, $^{*}\\,p<0.10$.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{singlespace}\n")
        f.write("\\end{threeparttable}\n")
        f.write("\\end{table}\n\n")

        # ==================================================================
        # TABLE 2: TOP LOADINGS
        # ==================================================================
        n_pcs_load = 3
        n_top = 6

        pca_dir = get_pca_output_dir()
        loadings_path = pca_dir / f"pca_avg_loadings_{timings[0]}.csv"

        if loadings_path.exists():
            loadings_df = pd.read_csv(loadings_path, index_col=0)

            top_loadings = {}
            for i in range(min(n_pcs_load, len(loadings_df))):
                pc_name = f'PC{i+1}'
                row = loadings_df.loc[pc_name]
                sorted_abs = row.abs().sort_values(ascending=False).head(n_top)
                top_loadings[pc_name] = [(factor, row[factor]) for factor in sorted_abs.index]

            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\caption{Top Factor Loadings by Principal Component}\n")
            f.write("\\label{tab:pca_loadings}\n")
            f.write("\\begin{threeparttable}\n")
            f.write("\\begin{singlespace}\n")
            f.write("\\small\n\n")

            col_spec = "lr" * n_pcs_load
            f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
            f.write("\\toprule\n")

            # Header row 1
            headers = []
            for i in range(n_pcs_load):
                headers.append(f"\\multicolumn{{2}}{{c}}{{PC{i+1}}}")
            f.write(" & ".join(headers) + " \\\\\n")

            # Cmidrules
            for i in range(n_pcs_load):
                st = 1 + i * 2
                f.write(f"\\cmidrule(lr){{{st}-{st + 1}}}")
            f.write("\n")

            # Header row 2
            sub_headers = []
            for _ in range(n_pcs_load):
                sub_headers.append("Factor & Loading")
            f.write(" & ".join(sub_headers) + " \\\\\n")
            f.write("\\midrule\n")

            # Body
            for row_idx in range(n_top):
                cells = []
                for i in range(n_pcs_load):
                    pc_name = f'PC{i+1}'
                    if row_idx < len(top_loadings[pc_name]):
                        factor, loading = top_loadings[pc_name][row_idx]
                        factor_clean = factor.replace('Δ', '$\\Delta$').replace('_', '\\_')
                        cells.append(f"\\texttt{{{factor_clean}}} & {loading:+.3f}")
                    else:
                        cells.append("& ")
                f.write(" & ".join(cells) + " \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n\n")

            f.write("\\begin{tablenotes}[para,flushleft]\n")
            f.write("\\footnotesize\n")
            f.write("\\item \\textit{Note:} ")
            f.write(f"Loadings are time-series averages of the eigenvector entries across "
                    f"all {PCA_WINDOW_LENGTH}-month rolling windows. "
                    f"For each principal component, the {n_top} factors with the largest "
                    f"absolute loading are reported. Sign indicates the direction of the relationship.\n")
            f.write("\\end{tablenotes}\n")
            f.write("\\end{singlespace}\n")
            f.write("\\end{threeparttable}\n")
            f.write("\\end{table}\n")
        else:
            print(f"   ⚠️  Loadings file not found, skipping Table 2")

    print(f"   💾 Saved: {tex_path}")
    return tex_path


# ============================================================================
# LOADINGS STABILITY PLOT (appendice A.5)
# ============================================================================

def generate_loadings_stability_plot(all_results: dict):
    """
    Generate time series plot of loadings stability (abs correlation
    between consecutive rolling windows) for PC1, PC2, PC3.
    Reads pca_loadings_stability_{timing}.csv produced by 02_pca_rolling.py.
    """
    print_header("GENERATING LOADINGS STABILITY PLOT", "-")

    timings = list(all_results.keys())
    timing = timings[0]

    pca_dir = get_pca_output_dir()
    stability_path = pca_dir / f"pca_loadings_stability_{timing}.csv"

    if not stability_path.exists():
        print(f"   ⚠️  Stability file not found: {stability_path}")
        print(f"   Re-run 02_pca_rolling.py to generate it.")
        return None

    stab_df = pd.read_csv(stability_path)
    stab_df['date'] = pd.to_datetime(stab_df['date'])

    # Detect PC columns
    pc_cols = [c for c in stab_df.columns if c.startswith('abs_corr_PC')]
    n_pcs = len(pc_cols)

    print(f"   ✅ Loaded stability data: {len(stab_df)} windows, {n_pcs} PCs")

    fig, ax = plt.subplots(figsize=(12, 5))

    colors_list = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary']]

    for i, col in enumerate(pc_cols):
        pc_label = col.replace('abs_corr_', '')
        color = colors_list[i % len(colors_list)]
        ax.plot(stab_df['date'], stab_df[col], label=pc_label,
                color=color, linewidth=1.2, alpha=0.85)

        # Mean line
        mean_val = stab_df[col].mean()
        ax.axhline(mean_val, color=color, linestyle='--', alpha=0.4, linewidth=0.8)

    # Reference line at 0.90
    ax.axhline(0.90, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    ax.text(stab_df['date'].iloc[-1], 0.905, '0.90', fontsize=8, color='gray', va='bottom')

    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('|Correlation| with Previous Window', fontsize=11)
    ax.set_title(f'Loadings Stability: Absolute Correlation Between Consecutive Rolling Windows\n'
                 f'(Rolling Window: {PCA_WINDOW_LENGTH} months)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.set_ylim(0.5, 1.02)

    plt.tight_layout()

    plot_path = FIGURES_DIR / "pca_loadings_stability.pdf"
    fig.savefig(plot_path, bbox_inches='tight')
    plt.close()

    print(f"   🖼️  Saved: {plot_path}")

    # Print summary stats
    print(f"\n   📊 Summary:")
    for col in pc_cols:
        pc_label = col.replace('abs_corr_', '')
        mean_val = stab_df[col].mean()
        min_val = stab_df[col].min()
        print(f"      {pc_label}: mean={mean_val:.4f}, min={min_val:.4f}")

    return plot_path

# ============================================================================
# ALPHA SYNTHESIS: PCA + AEN — HIGH vs NORMAL (same format as Table 7)
# ============================================================================

def generate_alpha_regime_synthesis():
    """
    Generates a synthesis table identical in structure to Table 7 Panel B
    (alpha_synthesis_across_models) but for PCA and AEN.
    
    Columns: PCA (HIGH, NORMAL) | AEN (HIGH, NORMAL)
    t-stats in parentheses on second row.
    
    Reads:
      - PCA: results/<strategy>/pca/regime_results_<timing>.json
      - AEN: results/<strategy>/aen/hqc/conditional_subsample.json
    """
    print_header("GENERATING PCA+AEN REGIME SYNTHESIS TABLE", "-")

    import importlib.util as _ilu

    # ── Locate AEN config ──
    aen_config_path = PROJECT_ROOT / "src" / "machine_learning" / "00_config.py"
    if not aen_config_path.exists():
        print(f"   ⚠️  AEN config not found: {aen_config_path}")
        return None

    spec = _ilu.spec_from_file_location("aen_config", aen_config_path)
    aen_cfg = _ilu.module_from_spec(spec)
    spec.loader.exec_module(aen_cfg)

    strategy_map = {
        'btp_italia': 'BTP Italia',
        'cds_bond_basis': 'CDS--Bond Basis',
        'itraxx_combined': 'iTraxx Combined',
    }

    def sig_super(pval):
        if pval is None or (isinstance(pval, float) and np.isnan(pval)):
            return ''
        if pval < 0.01: return '***'
        if pval < 0.05: return '**'
        if pval < 0.10: return '*'
        return ''

    strategies = list(STRATEGIES.keys())

    # ── Load PCA regime results ──
    pca_regime_data = {}
    for s in strategies:
        s_dir = get_strategy_pca_dir(s)
        for timing in ['contemporaneous', 'predictive']:
            regime_path = s_dir / f"regime_results_{timing}.json"
            if regime_path.exists():
                with open(regime_path, 'r') as f:
                    pca_regime_data[(s, timing)] = json.load(f)
                print(f"   ✅ PCA regime data: {s} ({timing})")
                break  # use first available timing

    # ── Load AEN regime results ──
    aen_regime_data = {}
    for s in strategies:
        aen_dir = aen_cfg.get_strategy_aen_dir(s)
        cs_path = aen_dir / "conditional_subsample.json"
        if cs_path.exists():
            with open(cs_path, 'r') as f:
                aen_regime_data[s] = json.load(f)
            print(f"   ✅ AEN regime data: {s}")
        else:
            print(f"   ⚠️  AEN regime data not found: {cs_path}")

    if not aen_regime_data and not pca_regime_data:
        print("   ⚠️  No regime data found, skipping synthesis.")
        return None

    # ── Build table ──
    tex_path = TABLES_DIR / "alpha_regime_synthesis_pca_aen.tex"

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("% " + "=" * 74 + "\n")
        f.write("% ALPHA BY REGIME — PCA + AEN SYNTHESIS\n")
        f.write("% " + "=" * 74 + "\n\n")

        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Alpha by Stress Regime: PCA and AEN}\n")
        f.write("\\label{tab:alpha_regime_pca_aen}\n")
        f.write("\\begin{threeparttable}\n")
        f.write("\\begin{singlespace}\n")
        f.write("\\small\n\n")

        f.write("\\begin{tabular}{l r r r r}\n")
        f.write("\\toprule\n")
        f.write(" & \\multicolumn{2}{c}{PCA} & \\multicolumn{2}{c}{AEN} \\\\\n")
        f.write("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n")
        f.write("Strategy & HIGH & NORMAL & HIGH & NORMAL \\\\\n")
        f.write("\\midrule\n")

        for s in strategies:
            display = strategy_map.get(s, s)
            f.write(f"\\textit{{{display}}}")

            # PCA
            pca_found = False
            for timing in ['contemporaneous', 'predictive']:
                key = (s, timing)
                if key in pca_regime_data:
                    rd = pca_regime_data[key]
                    for period in ['Stress', 'Normal']:
                        p_data = rd.get(period, {})
                        alpha = p_data.get('alpha_annual', np.nan)
                        pval = p_data.get('alpha_pvalue', np.nan)
                        stars = sig_super(pval)
                        if not np.isnan(alpha):
                            if stars:
                                f.write(f" & ${alpha:+.2f}^{{{stars}}}$")
                            else:
                                f.write(f" & {alpha:+.2f}")
                        else:
                            f.write(" & --")
                    pca_found = True
                    break
            if not pca_found:
                f.write(" & -- & --")

            # AEN
            aen = aen_regime_data.get(s, {}).get('regimes', {})
            h = aen.get('HIGH', {})
            n = aen.get('NORMAL', {})
            if not h.get('skip') and not n.get('skip'):
                for regime_data in [h, n]:
                    alpha = regime_data.get('alpha_annualized', np.nan)
                    pval = regime_data.get('alpha_pval', np.nan)
                    stars = sig_super(pval)
                    if stars:
                        f.write(f" & ${alpha:+.2f}^{{{stars}}}$")
                    else:
                        f.write(f" & {alpha:+.2f}")
            else:
                f.write(" & -- & --")

            f.write(" \\\\\n")

            # t-stat row
            f.write(" ")

            # PCA t-stats
            pca_found = False
            for timing in ['contemporaneous', 'predictive']:
                key = (s, timing)
                if key in pca_regime_data:
                    rd = pca_regime_data[key]
                    for period in ['Stress', 'Normal']:
                        t = rd.get(period, {}).get('alpha_tstat', np.nan)
                        if not np.isnan(t):
                            f.write(f" & ({t:.2f})")
                        else:
                            f.write(" & ")
                    pca_found = True
                    break
            if not pca_found:
                f.write(" & & ")

            # AEN t-stats
            if not h.get('skip') and not n.get('skip'):
                for regime_data in [h, n]:
                    t = regime_data.get('alpha_tstat', np.nan)
                    f.write(f" & ({t:.2f})")
            else:
                f.write(" & & ")

            f.write(" \\\\\n")
            f.write("\\addlinespace\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n\n")

        # ── Notes ──
        f.write("\\begin{tablenotes}[para,flushleft]\n")
        f.write("\\footnotesize\n")
        f.write("\\item \\textit{Note:} ")
        f.write("Annualized alpha (\\% p.a.) from monthly OLS regressions "
                "with Newey--West HAC standard errors. "
                "PCA: spanning regression on 8 rolling principal components. "
                "AEN: post-selection OLS on bootstrap-stable factors. "
                "HIGH denotes months in which the 5-year iTraxx Europe Main "
                "spread exceeds 100 bps. "
                "$t$-statistics in parentheses. "
                "$^{***}\\,p<0.01$, $^{**}\\,p<0.05$, $^{*}\\,p<0.10$.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{singlespace}\n")
        f.write("\\end{threeparttable}\n")
        f.write("\\end{table}\n")

    print(f"   💾 Saved: {tex_path}")
    return tex_path

# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("PCA RESULTS - TABLES AND PLOTS")
    
    # ========================================================================
    # STEP 1: Detect available timings
    # ========================================================================
    
    print_header("STEP 1: Detecting Available Results", "-")
    
    available_timings = detect_available_timings()
    
    if not available_timings:
        print("❌ No PCA results found!")
        print("   Please run 02_pca_rolling.py first.")
        return
    
    print(f"   Found timings: {available_timings}")
    
    # ========================================================================
    # STEP 2: Load all results
    # ========================================================================
    
    print_header("STEP 2: Loading PCA Results", "-")
    
    all_results = {}
    
    for timing in available_timings:
        print(f"\n   Loading {timing}...")
        results = load_pca_results(timing)
        
        # Use clean timing name
        clean_timing = results['timing']
        all_results[clean_timing] = results
        
        n_strategies = len(results['strategy_results'])
        print(f"   ✅ Loaded {n_strategies} strategy results")
    
    if not all_results:
        print("❌ No valid results loaded!")
        return
    
    # ========================================================================
    # STEP 3: Generate LaTeX tables
    # ========================================================================
    
    latex_path = generate_latex_tables(all_results)
    
    # ========================================================================
    # STEP 3a: Generate article tables (thesis/paper)
    # ========================================================================
    
    try:
        generate_article_tables(all_results)
    except Exception as e:
        print(f"   ⚠️ Article tables error: {e}")
    
    # ========================================================================
    # STEP 3d: Generate PCA+AEN regime synthesis table
    # ========================================================================
    
    try:
        generate_alpha_regime_synthesis()
    except Exception as e:
        print(f"   ⚠️ Regime synthesis table error: {e}")
    
    # ========================================================================
    # STEP 3b: Generate top loadings table
    # ========================================================================
    
    try:
        generate_top_loadings_table(all_results, n_pcs=3, n_top=6)
    except Exception as e:
        print(f"   ⚠️ Top loadings table error: {e}")
    
    # ========================================================================
    # STEP 3c: Generate loadings stability plot (Appendix A.5)
    # ========================================================================
    
    try:
        generate_loadings_stability_plot(all_results)
    except Exception as e:
        print(f"   ⚠️ Loadings stability plot error: {e}")
    
    # ========================================================================
    # STEP 4: Generate diagnostic plots (PDF only)
    # ========================================================================
    
    print_header("STEP 4: Generating Diagnostic Plots (PDF)", "-")
    
    # Scree plot
    try:
        generate_scree_plot(all_results)
    except Exception as e:
        print(f"   ⚠️ Scree plot error: {e}")
    
    # Variance time series
    try:
        generate_variance_timeseries(all_results)
    except Exception as e:
        print(f"   ⚠️ Variance timeseries error: {e}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print_header("SUMMARY")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"\n   LaTeX Tables:")
    print(f"   └── {TABLES_DIR / 'PCA_spanning_regression_results.tex'}")
    print(f"   └── {TABLES_DIR / 'PCA_top_loadings.tex'}")
    print(f"   └── {TABLES_DIR / 'PCA_article.tex'}  ← PAPER & SKELETON")
    
    print(f"\n   Figures (PNG):")
    for fig_file in sorted(FIGURES_DIR.glob("*.pdf")):
        print(f"   └── {fig_file}")
    
    print(f"\n📊 RESULTS SUMMARY:")
    
    for timing, results in all_results.items():
        timing_label = timing.title()
        print(f"\n   {timing_label}:")
        print(f"   {'─' * 50}")
        print(f"   {'Strategy':<20} {'Alpha (%)':<12} {'t-stat':<10} {'p-value':<10}")
        print(f"   {'-' * 50}")
        
        for strategy_name, result in results['strategy_results'].items():
            alpha = result['alpha'] * 12
            tstat = result['alpha_tstat']
            pval = result['alpha_pvalue']
            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.10 else ''
            
            strategy_display = strategy_name.replace('_', ' ').title()
            print(f"   {strategy_display:<20} {alpha:>8.2f}{sig:<3} {tstat:>10.2f} {pval:>10.4f}")
    
    print(f"\n{'=' * 80}")
    print("✅ PCA RESULTS GENERATION COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
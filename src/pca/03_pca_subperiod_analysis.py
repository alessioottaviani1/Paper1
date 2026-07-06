"""
================================================================================
03_pca_subperiod_analysis.py - PCA Subperiod & Regime Analysis
================================================================================
Robustness analysis:

1. Half-sample split (temporal stability)
2. iTraxx Main regime analysis (LOW/MEDIUM/HIGH stress regimes)
3. Rolling 36-month alpha on the FIXED full-sample PCs (in-sample diagnostic)

Regime definition:
  iTraxx Main 5Y spread (bps), cutoffs LOW < 60 <= MEDIUM < 100 <= HIGH.
  Same conditioning variable and cutoffs as the benchmark pipeline
  (factor_models/03_subperiod_rolling_analysis.py), ensuring cross-pipeline
  consistency.

OUTPUT:
- PCA_subperiod_results_article.tex : temporal subperiod alphas (paper)
- PCA_regime_summary_article.tex    : LOW/MEDIUM/HIGH regime alphas (paper)
- pca_rolling_alpha.pdf             : rolling 36m alpha plot, HIGH-stress shading

Author: Alessio Ottaviani
Institution: EDHEC Business School - PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION - MODIFY HERE
# ============================================================================

# iTraxx Main 5Y stress-regime cutoffs (bps): LOW < LOW_CUT <= MEDIUM < HIGH_CUT <= HIGH (= factor_models/03)
LOW_CUT = 60
HIGH_CUT = 100
DEFAULT_THRESHOLD = HIGH_CUT   # usato dallo shading rolling-alpha

# Rolling window length (months)
ROLLING_WINDOW = 36         # Months for rolling alpha estimation

# HAC lags for Newey-West standard errors
HAC_LAGS = 6

# Timing to analyze (will auto-detect available timings)
# Set to None to analyze all available, or specify: "predictive" or "contemporaneous"
TIMING_TO_ANALYZE = None    # None = all available

# ============================================================================
# PATHS
# ============================================================================

import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load PCA config
config_path = PROJECT_ROOT / "src" / "pca" / "00_pca_config.py"
spec = importlib.util.spec_from_file_location("pca_config", config_path)
pca_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pca_config)

# Export config variables
RESULTS_DIR = pca_config.RESULTS_DIR
STRATEGIES = pca_config.STRATEGIES
PCA_N_COMPONENTS = pca_config.PCA_N_COMPONENTS
get_pca_output_dir = pca_config.get_pca_output_dir
get_strategy_pca_dir = pca_config.get_strategy_pca_dir

# Output directories
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures" / "pca"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / "data"

# ============================================================================
# PLOTTING STYLE
# ============================================================================

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'stress': '#C73E1D',
    'normal': '#2E86AB',
    'confidence': '#87CEEB'
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title, char="="):
    """Print formatted section header."""
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")


def load_itrx_main_monthly():
    """iTraxx Europe Main 5Y level (bps), monthly last — dal foglio 'cds' di
    bbg.xlsx (come factor_models/03), al posto del ritirato
    Tradable_corporate_bond_factors.xlsx."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "factor_models"))
    import factor_sources as src
    s = src.load_bloomberg("cds")["ITRX EUR CDSI GEN 5Y Corp"].dropna()
    monthly = s.resample("ME").last().dropna()
    monthly.name = "ITRX_MAIN"
    print(f"   Loaded iTraxx Main 5Y: {len(monthly)} months, "
          f"{monthly.index.min().date()} → {monthly.index.max().date()} "
          f"(mean {monthly.mean():.0f} bps)")
    return monthly
    

def detect_available_timings():
    """Detect which timing conventions have been run."""
    pca_dir = get_pca_output_dir()
    available = []
    
    for timing in ['predictive', 'contemporaneous']:
        if (pca_dir / f"pc_scores_{timing}.parquet").exists():
            available.append(timing)
        elif timing == 'predictive' and (pca_dir / "pc_scores.parquet").exists():
            available.append('predictive_legacy')
    
    return available


def load_pca_data(timing: str):
    """
    Load PC scores and strategy returns for a given timing.
    
    Returns:
        dict with 'pc_scores' and 'strategy_returns'
    """
    pca_dir = get_pca_output_dir()
    
    # Handle legacy naming
    if timing == 'predictive_legacy':
        pc_path = pca_dir / "pc_scores.parquet"
    else:
        pc_path = pca_dir / f"pc_scores_{timing}.parquet"
    
    if not pc_path.exists():
        return None
    
    pc_scores = pd.read_parquet(pc_path)
    
    # Load strategy returns
    strategy_returns = {}
    for strategy_name in STRATEGIES.keys():
        strategy_dir = get_strategy_pca_dir(strategy_name)
        returns_path = strategy_dir / "y_returns_pca.parquet"
        
        if returns_path.exists():
            returns = pd.read_parquet(returns_path)['Strategy_Return']
            strategy_returns[strategy_name] = returns
    
    return {
        'pc_scores': pc_scores,
        'strategy_returns': strategy_returns
    }


def run_regression(y, X, hac_lags=6):
    """
    Run OLS regression with HAC standard errors.
    
    Returns:
        dict with regression results
    """
    if len(y) < 20:  # Minimum observations
        return None
    
    # Align data
    common_idx = y.index.intersection(X.index)
    y_aligned = y.loc[common_idx]
    X_aligned = X.loc[common_idx]
    
    # Remove NaN
    mask = ~(y_aligned.isna() | X_aligned.isna().any(axis=1))
    y_clean = y_aligned[mask]
    X_clean = X_aligned[mask]
    
    if len(y_clean) < 20:
        return None
    
    # Add constant
    X_with_const = sm.add_constant(X_clean)
    
    # Fit OLS with HAC
    model = sm.OLS(y_clean, X_with_const)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
    
    # Extract results
    output = {
        'n_obs': int(len(y_clean)),
        'alpha': float(results.params['const']),
        'alpha_se': float(results.bse['const']),
        'alpha_tstat': float(results.tvalues['const']),
        'alpha_pvalue': float(results.pvalues['const']),
        'r_squared': float(results.rsquared),
        'r_squared_adj': float(results.rsquared_adj),
    }
    
    # Add betas
    pc_cols = [c for c in X_clean.columns]
    for pc in pc_cols:
        output[f'beta_{pc}'] = float(results.params[pc])
        output[f'tstat_{pc}'] = float(results.tvalues[pc])
        output[f'pval_{pc}'] = float(results.pvalues[pc])
    
    return output


def compute_rolling_alpha(y, X, window, hac_lags=6):
    """
    Compute rolling window alpha estimates.
    
    Returns:
        DataFrame with columns ['alpha', 'alpha_se', 'alpha_tstat']
    """
    # Align data
    common_idx = y.index.intersection(X.index)
    y_aligned = y.loc[common_idx].sort_index()
    X_aligned = X.loc[common_idx].sort_index()
    
    results = []
    dates = []
    
    for i in range(window, len(y_aligned) + 1):
        y_window = y_aligned.iloc[i-window:i]
        X_window = X_aligned.iloc[i-window:i]
        
        # Remove NaN
        mask = ~(y_window.isna() | X_window.isna().any(axis=1))
        y_clean = y_window[mask]
        X_clean = X_window[mask]
        
        if len(y_clean) < window * 0.8:  # Require at least 80% of window
            results.append({'alpha': np.nan, 'alpha_se': np.nan})
            dates.append(y_aligned.index[i-1])
            continue
        
        # Add constant and fit
        X_with_const = sm.add_constant(X_clean)
        try:
            model = sm.OLS(y_clean, X_with_const)
            res = model.fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
            
            results.append({
                'alpha': res.params['const'],
                'alpha_se': res.bse['const']
            })
        except:
            results.append({'alpha': np.nan, 'alpha_se': np.nan})
        
        dates.append(y_aligned.index[i-1])
    
    rolling_df = pd.DataFrame(results, index=dates)
    
    # Annualize
    rolling_df['alpha_annual'] = rolling_df['alpha'] * 12
    rolling_df['alpha_se_annual'] = rolling_df['alpha_se'] * 12
    
    return rolling_df


def get_significance_stars(pval):
    """Return significance stars based on p-value."""
    if pval < 0.01:
        return '***'
    elif pval < 0.05:
        return '**'
    elif pval < 0.10:
        return '*'
    return ''

# ============================================================================
# ARTICLE TABLES (thesis/paper)
# ============================================================================

def write_subperiod_article(all_results, timings_to_process, pc_names,
                            vol_threshold, out_path):
    """
    Article-format subperiod table using longtable (spans multiple pages).
    """
    strategy_map = {
        'btp_italia': 'BTP Italia',
        'cds_bond_basis': 'CDS--Bond',
        'itraxx_combined': 'Index Skew',
    }

    def sig_super(pval):
        if pval < 0.01:
            return '***'
        if pval < 0.05:
            return '**'
        if pval < 0.10:
            return '*'
        return ''

    periods_order = ['Full Sample', 'First Half', 'Second Half']

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("% " + "=" * 74 + "\n")
        f.write("% PCA SUBPERIOD ANALYSIS — ARTICLE (longtable)\n")
        f.write("% Requires: \\usepackage{longtable,booktabs,threeparttablex}\n")
        f.write("% " + "=" * 74 + "\n\n")

        for timing in timings_to_process:
            if timing not in all_results:
                continue

            timing_label = timing.replace('_', ' ').title()
            ncols = len(periods_order)

            f.write("\\setlength{\\tabcolsep}{3pt}\n")
            f.write("\\small\n")
            f.write("\\renewcommand{\\arraystretch}{0.85}\n")
            f.write("\\setstretch{1.0}\n")
            f.write(f"\\begin{{longtable}}{{l{'c' * ncols}}}\n")

            # Caption and label
            f.write(f"\\caption{{PCA Subperiod Analysis ({timing_label})}}\n")
            f.write(f"\\label{{tab:pca_subperiod_{timing}_article}} \\\\\n")

            # First header
            f.write("\\toprule\n")
            f.write(" ")
            for period in periods_order:
                f.write(f" & {period}")
            f.write(" \\\\\n")
            f.write("\\midrule\n")
            f.write("\\endfirsthead\n\n")

            # Continuation header
            f.write(f"\\multicolumn{{{ncols + 1}}}{{l}}{{\\small\\textit{{(continued)}}}} \\\\\n")
            f.write("\\toprule\n")
            f.write(" ")
            for period in periods_order:
                f.write(f" & {period}")
            f.write(" \\\\\n")
            f.write("\\midrule\n")
            f.write("\\endhead\n\n")

            # Footer on continuation pages
            f.write(f"\\midrule\n")
            f.write(f"\\multicolumn{{{ncols + 1}}}{{r}}{{\\small\\textit{{continued on next page}}}} \\\\\n")
            f.write("\\endfoot\n\n")

            # Final footer
            f.write("\\bottomrule\n")
            f.write("\\endlastfoot\n\n")

            # Body: each strategy
            for strategy_name in all_results[timing].keys():
                display = strategy_map.get(strategy_name,
                                           strategy_name.replace('_', ' ').title())
                strategy_results = all_results[timing][strategy_name]

                total_cols = ncols + 1
                f.write(f"\\multicolumn{{{total_cols}}}{{l}}{{\\textbf{{{display}}}}} \\\\\n")
                f.write("\\addlinespace\n")

                # Alpha row
                f.write("$\\alpha$ (\\% p.a.)")
                for period in periods_order:
                    if period in strategy_results:
                        res = strategy_results[period]
                        alpha = res['alpha_annual']
                        stars = sig_super(res['alpha_pvalue'])
                        if stars:
                            f.write(f" & ${alpha:.2f}^{{{stars}}}$")
                        else:
                            f.write(f" & {alpha:.2f}")
                    else:
                        f.write(" & --")
                f.write(" \\\\\n")

                # Alpha t-stat row
                f.write(" ")
                for period in periods_order:
                    if period in strategy_results:
                        f.write(f" & ({strategy_results[period]['alpha_tstat']:.2f})")
                    else:
                        f.write(" & ")
                f.write(" \\\\\n")
                f.write("\\addlinespace\n")

                # Beta rows
                for pc in pc_names:
                    f.write(f"$\\beta_{{{pc}}}$")
                    for period in periods_order:
                        if period in strategy_results:
                            res = strategy_results[period]
                            beta_key = f'beta_{pc}'
                            if beta_key in res:
                                beta = res[beta_key]
                                pval = res[f'pval_{pc}']
                                stars = sig_super(pval)
                                if stars:
                                    f.write(f" & ${beta:.3f}^{{{stars}}}$")
                                else:
                                    f.write(f" & {beta:.3f}")
                            else:
                                f.write(" & --")
                        else:
                            f.write(" & --")
                    f.write(" \\\\\n")

                    # t-stat row
                    f.write(" ")
                    for period in periods_order:
                        if period in strategy_results:
                            res = strategy_results[period]
                            tstat_key = f'tstat_{pc}'
                            if tstat_key in res:
                                f.write(f" & ({res[tstat_key]:.2f})")
                            else:
                                f.write(" & ")
                        else:
                            f.write(" & ")
                    f.write(" \\\\\n")

                f.write("\\addlinespace\n")

                # R² adj row
                f.write("$\\bar{R}^2$")
                for period in periods_order:
                    if period in strategy_results:
                        f.write(f" & {strategy_results[period]['r_squared_adj']:.3f}")
                    else:
                        f.write(" & --")
                f.write(" \\\\\n")

                # N obs row
                f.write("$N$")
                for period in periods_order:
                    if period in strategy_results:
                        f.write(f" & {strategy_results[period]['n_obs']}")
                    else:
                        f.write(" & --")
                f.write(" \\\\\n")
                f.write("\\addlinespace\n")
                f.write("\\midrule\n")

            f.write("\\end{longtable}\n")

            # Notes (outside longtable)
            f.write("\\begin{minipage}{\\textwidth}\n")
            f.write("\\footnotesize\n")
            f.write("\\textit{Note:} ")
            f.write("First/Second Half: temporal split at sample midpoint; stress regimes "
                    "(LOW/MEDIUM/HIGH) are reported in Table~\\ref{tab:pca_regime_summary_article}. ")
            f.write("$\\alpha$ is annualized (\\% p.a.). ")
            f.write("$t$-statistics in parentheses (Newey--West HAC). ")
            f.write("$^{***}\\,p<0.01$, $^{**}\\,p<0.05$, $^{*}\\,p<0.10$.\n")
            f.write("\\end{minipage}\n\n")

    print(f"   💾 Saved: {out_path}")


def write_regime_summary_article(all_results, timings_to_process,
                                  vol_threshold, out_path):
    """
    Article-format regime summary table.
    """
    strategy_map = {
        'btp_italia': 'BTP Italia',
        'cds_bond_basis': 'CDS--Bond',
        'itraxx_combined': 'Index Skew',
    }

    def sig_super(pval):
        if pval < 0.01:
            return '***'
        if pval < 0.05:
            return '**'
        if pval < 0.10:
            return '*'
        return ''

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("% " + "=" * 74 + "\n")
        f.write("% PCA REGIME SUMMARY — ARTICLE TABLE\n")
        f.write("% " + "=" * 74 + "\n\n")

        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{PCA Alpha by Regime}\n")
        f.write("\\label{tab:pca_regime_summary_article}\n")
        f.write("\\begin{threeparttable}\n")
        f.write("\\begin{singlespace}\n")
        f.write("\\small\n\n")

        if len(timings_to_process) == 1:
            f.write("\\begin{tabular}{lcccc}\n")
            f.write("\\toprule\n")
            f.write("Strategy & Full Sample & LOW & MEDIUM & HIGH \\\\\n")
        f.write("\\midrule\n")

        all_strategies = set()
        for timing in timings_to_process:
            if timing in all_results:
                all_strategies.update(all_results[timing].keys())

        for strategy_name in sorted(all_strategies):
            display = strategy_map.get(strategy_name,
                                       strategy_name.replace('_', ' ').title())
            f.write(f"\\textit{{{display}}} ")

            if len(timings_to_process) == 1:
                timing = timings_to_process[0]
                if timing in all_results and strategy_name in all_results[timing]:
                    res = all_results[timing][strategy_name]
                    for period in ['Full Sample', 'LOW', 'MEDIUM', 'HIGH']:
                        if period in res:
                            alpha = res[period]['alpha_annual']
                            stars = sig_super(res[period]['alpha_pvalue'])
                            if stars:
                                f.write(f"& ${alpha:.2f}^{{{stars}}}$")
                            else:
                                f.write(f"& {alpha:.2f}")
                            f.write(" ")
                        else:
                            f.write("& -- ")
                    pass  # Delta column removed
            f.write("\\\\\n")

            # t-stat row
            f.write(" ")
            if len(timings_to_process) == 1:
                timing = timings_to_process[0]
                if timing in all_results and strategy_name in all_results[timing]:
                    res = all_results[timing][strategy_name]
                    for period in ['Full Sample', 'LOW', 'MEDIUM', 'HIGH']:
                        if period in res:
                            f.write(f"& ({res[period]['alpha_tstat']:.2f}) ")
                        else:
                            f.write("& ")
                    pass  # Delta column removed
            f.write("\\\\\n")
            f.write("\\addlinespace\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n\n")

        f.write("\\begin{tablenotes}[para,flushleft]\n")
        f.write("\\footnotesize\n")
        f.write("\\item \\textit{Note:} ")
        f.write(f"Regimes on iTraxx Main 5Y: LOW $<$ {LOW_CUT}, MEDIUM $\\in$ [{LOW_CUT},{HIGH_CUT}), HIGH $\\geq$ {HIGH_CUT} bps. ")
        f.write("$\\alpha$ is annualized (\\% p.a.). ")
        f.write("$t$-statistics in parentheses (Newey--West HAC). ")
        f.write("$^{***}\\,p<0.01$, $^{**}\\,p<0.05$, $^{*}\\,p<0.10$.\n")
        f.write("\\end{tablenotes}\n")
        f.write("\\end{singlespace}\n")
        f.write("\\end{threeparttable}\n")
        f.write("\\end{table}\n")

    print(f"   💾 Saved: {out_path}")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print_header("PCA SUBPERIOD & REGIME ANALYSIS")
    
    # Print configuration
    print(f"  regime=iTraxx Main 5Y  cutoffs={LOW_CUT}/{HIGH_CUT} bps  roll={ROLLING_WINDOW}m  hac={HAC_LAGS}")
    print_header("STEP 1: Load iTraxx Main Data", "-")
    
    itrx_monthly = load_itrx_main_monthly()
    
    # Show regime counts for each threshold
    vol_threshold = DEFAULT_THRESHOLD
    n_low  = (itrx_monthly < LOW_CUT).sum()
    n_med  = ((itrx_monthly >= LOW_CUT) & (itrx_monthly < HIGH_CUT)).sum()
    n_high = (itrx_monthly >= HIGH_CUT).sum()
    print(f"   Regimi: LOW(<{LOW_CUT})={n_low}, MEDIUM[{LOW_CUT},{HIGH_CUT})={n_med}, HIGH(>={HIGH_CUT})={n_high}")
    
    # ========================================================================
    # STEP 2: Detect available timings and load PCA data
    # ========================================================================
    
    print_header("STEP 2: Load PCA Data", "-")
    
    available_timings = detect_available_timings()
    print(f"   Available timings: {available_timings}")
    
    if TIMING_TO_ANALYZE:
        timings_to_process = [TIMING_TO_ANALYZE]
    else:
        timings_to_process = [t.replace('_legacy', '') for t in available_timings]
        timings_to_process = list(set(timings_to_process))  # Remove duplicates
    
    print(f"   Processing timings: {timings_to_process}")
    
    # Storage for all results
    all_results = {}
    rolling_results = {}
    
    # ========================================================================
    # STEP 3: Run analysis for each timing
    # ========================================================================
    
    for timing in timings_to_process:
        print_header(f"STEP 3: Analysis for {timing.upper()}", "-")
        
        # Load data
        timing_key = 'predictive_legacy' if timing == 'predictive' and 'predictive_legacy' in available_timings else timing
        data = load_pca_data(timing_key)
        
        if data is None:
            print(f"   ⚠️ No data found for {timing}")
            continue
        
        pc_scores = data['pc_scores']
        strategy_returns = data['strategy_returns']
        
        print(f"   ✅ Loaded PC scores: {len(pc_scores)} months, {len(pc_scores.columns)} components")
        print(f"   ✅ Loaded {len(strategy_returns)} strategies")
        
        all_results[timing] = {}
        rolling_results[timing] = {}
        
        # Process each strategy
        for strategy_name, returns in strategy_returns.items():
            print(f"\n   📊 {strategy_name}:")
            
            # Align all data
            common_idx = returns.index.intersection(pc_scores.index).intersection(itrx_monthly.index)
            common_idx = common_idx.sort_values()

            X = pc_scores.loc[common_idx]
            itrx = itrx_monthly.loc[common_idx]
            stress = itrx > vol_threshold

            # --- TIMING ALIGNMENT (match 02_pca_rolling.py) ---
            if timing == "predictive":
                # PC_t -> R_{t+1}
                y = returns.loc[common_idx].shift(-1)

                # drop last obs where R_{t+1} is missing
                valid_idx = y.index[y.notna()]
                y = y.loc[valid_idx]
                X = X.loc[valid_idx]
                itrx = itrx.loc[valid_idx]
                stress = stress.loc[valid_idx]
            else:
                # PC_t -> R_t
                y = returns.loc[common_idx]

            print(f"      Aligned observations: {len(y)}")
           
            # Define subperiods on the *effective* regression sample
            idx = y.index.sort_values()
            mid_idx = len(idx) // 2

            itrx_idx = itrx.loc[idx]
            subperiods = {
                'Full Sample': idx,
                'First Half': idx[:mid_idx],
                'Second Half': idx[mid_idx:],
                'LOW':    idx[(itrx_idx < LOW_CUT).values],
                'MEDIUM': idx[((itrx_idx >= LOW_CUT) & (itrx_idx < HIGH_CUT)).values],
                'HIGH':   idx[(itrx_idx >= HIGH_CUT).values],
            }
            
            strategy_results = {}
            
            for period_name, period_idx in subperiods.items():
                if len(period_idx) < 20:
                    print(f"      {period_name}: Too few obs ({len(period_idx)}), skipping")
                    continue
                
                y_period = y.loc[period_idx]
                X_period = X.loc[period_idx]
                
                result = run_regression(y_period, X_period, HAC_LAGS)
                
                if result:
                    result['alpha_annual'] = result['alpha'] * 12
                    strategy_results[period_name] = result
                    
                    alpha_ann = result['alpha_annual']
                    tstat = result['alpha_tstat']
                    pval = result['alpha_pvalue']
                    stars = get_significance_stars(pval)
                    
                    print(f"      {period_name:12s}: α={alpha_ann:6.2f}%{stars:3s} (t={tstat:5.2f})  N={result['n_obs']}")
            
            all_results[timing][strategy_name] = strategy_results
            
            # Save regime results as JSON for cross-pipeline synthesis
            regime_json = {}
            for period_name in ['Full Sample', 'LOW', 'MEDIUM', 'HIGH']:
                if period_name in strategy_results:
                    r = strategy_results[period_name]
                    regime_json[period_name] = {
                        'alpha_annual': r['alpha_annual'],
                        'alpha_tstat': r['alpha_tstat'],
                        'alpha_pvalue': r['alpha_pvalue'],
                        'r_squared_adj': r['r_squared_adj'],
                        'n_obs': r['n_obs'],
                    }
            s_pca_dir = get_strategy_pca_dir(strategy_name)
            with open(s_pca_dir / f"regime_results_{timing}.json", 'w') as fj:
                json.dump(regime_json, fj, indent=2)
            
            
            # Rolling alpha

            print(f"      Computing rolling alpha ({ROLLING_WINDOW}m window)...")
            rolling_df = compute_rolling_alpha(y, X, ROLLING_WINDOW, HAC_LAGS)
            rolling_df['stress'] = stress.reindex(rolling_df.index).fillna(False) if len(rolling_df) > 0 else False
            rolling_results[timing][strategy_name] = rolling_df
    
    # ========================================================================
    # STEP 4: Generate LaTeX Tables
    # ========================================================================
    
    print_header("STEP 4: Generate LaTeX Tables", "-")
    
    pc_names = [f'PC{i+1}' for i in range(PCA_N_COMPONENTS)]
    
    # --- Article versions ---
    article_subperiod_path = TABLES_DIR / "PCA_subperiod_results_article.tex"
    write_subperiod_article(all_results, timings_to_process, pc_names,
                            vol_threshold, article_subperiod_path)

    article_regime_path = TABLES_DIR / "PCA_regime_summary_article.tex"
    write_regime_summary_article(all_results, timings_to_process,
                                 vol_threshold, article_regime_path)
    
    # ========================================================================
    # STEP 5: Generate Plots
    # ========================================================================
    
    print_header("STEP 5: Generate Plots", "-")
    
    # ---------- PLOT 1: Rolling Alpha ----------
    
    n_strategies = len(list(rolling_results.values())[0]) if rolling_results else 0
    n_timings = len(timings_to_process)
    
    fig, axes = plt.subplots(n_strategies, n_timings, 
                             figsize=(8*n_timings, 4*n_strategies),
                             squeeze=False)
    
    for t_idx, timing in enumerate(timings_to_process):
        if timing not in rolling_results:
            continue
        
        for s_idx, (strategy_name, rolling_df) in enumerate(rolling_results[timing].items()):
            ax = axes[s_idx, t_idx]
            
            if rolling_df is None or len(rolling_df) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue
            
            # Plot alpha
            ax.plot(rolling_df.index, rolling_df['alpha_annual'], 
                   color=COLORS['primary'], linewidth=2, label='Alpha (annualized)')
            
            # Confidence bands
            lower = rolling_df['alpha_annual'] - 1.96 * rolling_df['alpha_se_annual']
            upper = rolling_df['alpha_annual'] + 1.96 * rolling_df['alpha_se_annual']
            ax.fill_between(rolling_df.index, lower, upper,
                           alpha=0.25, color=COLORS['confidence'], label='95% CI')
            
            # Zero line
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
            
            # Shade stress periods
            stress_periods = rolling_df['stress']
            in_stress = False
            stress_start = None
            
            for date, is_stress in stress_periods.items():
                if is_stress and not in_stress:
                    stress_start = date
                    in_stress = True
                elif not is_stress and in_stress:
                    ax.axvspan(stress_start, date, alpha=0.15, color=COLORS['stress'])
                    in_stress = False
            
            if in_stress:  # Close final stress period
                ax.axvspan(stress_start, rolling_df.index[-1], alpha=0.15, color=COLORS['stress'])
            
            # Labels
            strategy_display = strategy_name.replace('_', ' ').title()
            timing_display = timing.replace('_', ' ').title()
            
            ax.set_title(f'{strategy_display} - {timing_display}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Alpha (% p.a.)', fontsize=10)
            ax.set_xlabel('Date', fontsize=10)
            
            if s_idx == 0 and t_idx == 0:
                ax.legend(loc='upper left', fontsize=9)
    
    plt.suptitle(f'Rolling {ROLLING_WINDOW}-Month Alpha\n(Shaded = iTraxx Main > {vol_threshold:.0f} bps)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    rolling_plot_path = FIGURES_DIR / "pca_rolling_alpha.pdf"
    plt.savefig(rolling_plot_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"   💾 Saved: {rolling_plot_path}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    # POST
    print_header("SUMMARY")
    print(f"\n   Regime alpha (% p.a.) — cutoffs {LOW_CUT}/{HIGH_CUT} bps su iTraxx Main 5Y")

    def _a(res, p):
        v = res.get(p, {}).get('alpha_annual', np.nan)
        return f"{v:.2f}" if not np.isnan(v) else "--"

    for timing in timings_to_process:
        if timing not in all_results:
            continue
        print(f"\n   {timing.title()}:")
        print(f"   {'Strategy':<20} {'Full':>8} {'LOW':>8} {'MEDIUM':>8} {'HIGH':>8}")
        print(f"   {'-' * 60}")
        for strategy_name in sorted(all_results[timing].keys()):
            res = all_results[timing][strategy_name]
            disp = strategy_name.replace('_', ' ').title()
            print(f"   {disp:<20} {_a(res,'Full Sample'):>8} {_a(res,'LOW'):>8} {_a(res,'MEDIUM'):>8} {_a(res,'HIGH'):>8}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"   Tables:")
    print(f"   └── {article_subperiod_path.name}  ← PAPER & SKELETON")
    print(f"   └── {article_regime_path.name}  ← PAPER & SKELETON")
    print(f"   Figures:")
    print(f"   └── {rolling_plot_path.name}")
    
    print(f"\n{'=' * 80}")
    print("✅ PCA SUBPERIOD ANALYSIS COMPLETE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
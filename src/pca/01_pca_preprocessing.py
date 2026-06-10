"""
================================================================================
01_pca_preprocessing.py - Preprocessing per PCA Rolling
================================================================================
Prepara i dati per la PCA rolling:
1. Carica fattori e strategy returns
2. Filtra fattori disponibili alla PCA_START_DATE (considerando window)
3. Allinea tutto e salva dataset pronti per PCA

Output:
- factors_for_pca.parquet: fattori filtrati e allineati
- pca_preprocessing_report.json: report dettagliato

Author: Alessio Ottaviani
Institution: EDHEC Business School - PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IMPORT CONFIG
# ============================================================================

import importlib.util

# Trova root progetto
PROJECT_ROOT = Path(__file__).resolve().parents[2]
config_path = PROJECT_ROOT / "src" / "pca" / "00_pca_config.py"

spec = importlib.util.spec_from_file_location("pca_config", config_path)
pca_config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pca_config)

# Esporta variabili
FACTORS_PATH = pca_config.FACTORS_PATH
STRATEGIES = pca_config.STRATEGIES
FACTORS_END_DATE = pca_config.FACTORS_END_DATE
PCA_START_DATE = pca_config.PCA_START_DATE
PCA_WINDOW_LENGTH = pca_config.PCA_WINDOW_LENGTH
PCA_VARIANCE_THRESHOLD = pca_config.PCA_VARIANCE_THRESHOLD
PCA_TIMING = pca_config.PCA_TIMING
get_pca_output_dir = pca_config.get_pca_output_dir
get_strategy_pca_dir = pca_config.get_strategy_pca_dir

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title, char="="):
    """Print formatted section header."""
    print(f"\n{char * 80}")
    print(title)
    print(f"{char * 80}")


def load_strategy_returns(strategy_path: Path) -> pd.Series:
    """
    Carica return giornalieri e resample a monthly (compound).
    
    Returns:
        pd.Series con index DatetimeIndex (month-end) e values = monthly returns
    """
    daily_df = pd.read_csv(strategy_path, index_col=0, parse_dates=True)
    
    if 'index_return' not in daily_df.columns:
        raise ValueError(f"Colonna 'index_return' non trovata in {strategy_path}")
    
    daily_returns = daily_df['index_return'].dropna()
    
    # Compound a monthly
    monthly_returns = daily_returns.resample('ME').apply(
        lambda x: ((1 + x/100).prod() - 1) * 100 if len(x) > 0 else np.nan
    )
    
    monthly_returns = monthly_returns.dropna()
    monthly_returns.name = 'Strategy_Return'
    
    return monthly_returns


def get_available_factors(
    factors_df: pd.DataFrame,
    pca_start_date: pd.Timestamp,
    window_length: int
) -> dict:
    """
    Determina quali fattori hanno dati sufficienti per la prima rolling window.
    
    Per calcolare PC a pca_start_date, serve window [pca_start_date - W, pca_start_date - 1].
    
    Returns:
        dict con 'available', 'excluded', 'details', 'window_start_required'
    """
    window_start_required = pca_start_date - pd.DateOffset(months=window_length)
    
    available = []
    excluded = []
    details = []
    
    for col in factors_df.columns:
        series = factors_df[col].dropna()
        
        if len(series) == 0:
            excluded.append(col)
            details.append({
                'factor': col,
                'status': 'excluded',
                'reason': 'all_nan',
                'first_valid': None
            })
            continue
        
        first_valid = series.index.min()
        
        if first_valid <= window_start_required:
            available.append(col)
            details.append({
                'factor': col,
                'status': 'available',
                'first_valid': first_valid.strftime('%Y-%m-%d')
            })
        else:
            excluded.append(col)
            details.append({
                'factor': col,
                'status': 'excluded',
                'reason': f'starts_after_window_start',
                'first_valid': first_valid.strftime('%Y-%m-%d'),
                'required_start': window_start_required.strftime('%Y-%m-%d')
            })
    
    return {
        'available': sorted(available),
        'excluded': sorted(excluded),
        'n_available': len(available),
        'n_excluded': len(excluded),
        'window_start_required': window_start_required,
        'details': details
    }


# ============================================================================
# MAIN PREPROCESSING
# ============================================================================

def main():
    print_header("PCA PREPROCESSING PIPELINE")
    
    # Print config
    print(f"\n📋 CONFIGURATION:")
    print(f"   PCA_START_DATE:        {PCA_START_DATE}")
    print(f"   PCA_WINDOW_LENGTH:     {PCA_WINDOW_LENGTH} months")
    print(f"   PCA_VARIANCE_THRESHOLD:{PCA_VARIANCE_THRESHOLD:.0%}")
    print(f"   PCA_TIMING:            {PCA_TIMING}")
    print(f"   FACTORS_END_DATE:      {FACTORS_END_DATE}")
    
    # ========================================================================
    # STEP 1: Load factors
    # ========================================================================
    
    print_header("STEP 1: Load Factors", "-")
    
    if not FACTORS_PATH.exists():
        print(f"❌ ERROR: Factors file not found: {FACTORS_PATH}")
        return
    
    all_factors = pd.read_parquet(FACTORS_PATH)
    
    # Applica FACTORS_END_DATE
    factors_end = pd.Timestamp(FACTORS_END_DATE)
    all_factors = all_factors[all_factors.index <= factors_end]
    
    print(f"   ✅ Loaded: {len(all_factors.columns)} factors, {len(all_factors)} months")
    print(f"   📅 Range: {all_factors.index.min().strftime('%Y-%m-%d')} → {all_factors.index.max().strftime('%Y-%m-%d')}")
    
    # ========================================================================
    # STEP 2: Filter factors by availability
    # ========================================================================
    
    print_header("STEP 2: Filter Factors by Availability", "-")
    
    pca_start = pd.Timestamp(PCA_START_DATE)
    
    availability = get_available_factors(
        all_factors,
        pca_start,
        PCA_WINDOW_LENGTH
    )
    
    print(f"\n   PCA start date: {PCA_START_DATE}")
    print(f"   Window required: [{availability['window_start_required'].strftime('%Y-%m-%d')}, {(pca_start - pd.DateOffset(months=1)).strftime('%Y-%m-%d')}]")
    print(f"\n   ✅ Factors available: {availability['n_available']}")
    print(f"   ❌ Factors excluded:  {availability['n_excluded']}")
    
    if availability['excluded']:
        print(f"\n   Excluded factors:")
        for detail in availability['details']:
            if detail['status'] == 'excluded':
                if detail.get('first_valid'):
                    print(f"      - {detail['factor']}: starts {detail['first_valid']}")
                else:
                    print(f"      - {detail['factor']}: all NaN")
    
    # Filter factors
    # Filter factors (available based on first_valid <= window_start_required)
    factors_filtered = all_factors[availability['available']].copy()

    # ------------------------------------------------------------------------
    # ENFORCE STRICT BALANCED PANEL OVER THE PCA PERIOD
    # Drop any factor that has even 1 NaN between window_start_required and FACTORS_END_DATE.
    # This keeps the factor universe stable across all rolling windows.
    # ------------------------------------------------------------------------
    pca_period_start = availability['window_start_required']
    pca_period_end = factors_end  # already defined above from FACTORS_END_DATE

    check_block = factors_filtered.loc[pca_period_start:pca_period_end]
    nan_counts = check_block.isna().sum()

    factors_with_nan = nan_counts[nan_counts > 0].index.tolist()

    if len(factors_with_nan) > 0:
        print("\n   ⚠️  Removing factors with internal NaN in PCA period "
            f"[{pca_period_start.strftime('%Y-%m-%d')} → {pca_period_end.strftime('%Y-%m-%d')}]:")
        for f in factors_with_nan:
            print(f"      - {f}: NaNs={int(nan_counts[f])}")
        factors_filtered = factors_filtered.drop(columns=factors_with_nan)

    print(f"\n   📊 Factors for PCA (strict balanced): {len(factors_filtered.columns)}")
   
    # ========================================================================
    # STEP 3: Load and align strategy returns
    # ========================================================================
    
    print_header("STEP 3: Load Strategy Returns", "-")
    
    strategy_returns = {}
    strategy_info = {}
    
    for strategy_name, strategy_path in STRATEGIES.items():
        print(f"\n   📂 {strategy_name}:")
        
        if not strategy_path.exists():
            print(f"      ❌ File not found: {strategy_path}")
            continue
        
        try:
            returns = load_strategy_returns(strategy_path)
            
            # Taglia alla data massima fattori
            returns = returns[returns.index <= factors_end]
            
            # Taglia alla data minima PCA (per strategie che iniziano prima)
            # I PC scores saranno disponibili solo da PCA_START_DATE
            
            orig_start = returns.index.min()
            orig_end = returns.index.max()
            orig_len = len(returns)
            
            strategy_returns[strategy_name] = returns
            strategy_info[strategy_name] = {
                'original_start': orig_start.strftime('%Y-%m-%d'),
                'original_end': orig_end.strftime('%Y-%m-%d'),
                'original_n_months': orig_len,
                'file_path': str(strategy_path)
            }
            
            print(f"      ✅ Loaded: {orig_len} months")
            print(f"      📅 Range: {orig_start.strftime('%Y-%m-%d')} → {orig_end.strftime('%Y-%m-%d')}")
            
            # Nota se la strategia inizia prima di PCA_START_DATE
            if orig_start < pca_start:
                months_lost = len(returns[returns.index < pca_start])
                print(f"      ⚠️  Strategy starts before PCA_START_DATE: {months_lost} months will not have PC scores")
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    # ========================================================================
    # STEP 4: Create aligned datasets
    # ========================================================================
    
    print_header("STEP 4: Create Aligned Datasets", "-")
    
    # Per ogni strategia, crea dataset allineato con fattori
    for strategy_name, returns in strategy_returns.items():
        print(f"\n   📊 {strategy_name}:")
        
        # Determina periodo di analisi
        # PC scores disponibili da PCA_START_DATE
        # Returns disponibili dal loro inizio
        
        analysis_start = max(pca_start, returns.index.min())
        analysis_end = min(factors_end, returns.index.max())
        
        # Filtra returns per periodo analisi
        returns_analysis = returns[(returns.index >= analysis_start) & 
                                   (returns.index <= analysis_end)]
        
        # Allinea fattori con returns
        common_dates = returns_analysis.index.intersection(factors_filtered.index)
        
        y = returns_analysis.loc[common_dates]
        X = factors_filtered.loc[common_dates]
        
        print(f"      Analysis period: {analysis_start.strftime('%Y-%m-%d')} → {analysis_end.strftime('%Y-%m-%d')}")
        print(f"      Aligned observations: {len(common_dates)}")
        
        # Aggiorna info
        strategy_info[strategy_name]['analysis_start'] = analysis_start.strftime('%Y-%m-%d')
        strategy_info[strategy_name]['analysis_end'] = analysis_end.strftime('%Y-%m-%d')
        strategy_info[strategy_name]['n_aligned_obs'] = len(common_dates)
        
        # Salva per strategia
        output_dir = get_strategy_pca_dir(strategy_name)
        
        y.to_frame().to_parquet(output_dir / "y_returns_pca.parquet")
        print(f"      💾 Saved: y_returns_pca.parquet")
    
    # ========================================================================
    # STEP 5: Save common factors and report
    # ========================================================================
    
    print_header("STEP 5: Save Outputs", "-")
    
    output_dir = get_pca_output_dir()
    
    # Salva fattori filtrati (comuni a tutte le strategie)
    # Questi sono i fattori che entrano nel PCA
    factors_filtered.to_parquet(output_dir / "factors_for_pca.parquet")
    print(f"   💾 {output_dir / 'factors_for_pca.parquet'}")
    
    # Salva lista fattori
    factor_list = {
        'n_factors': len(availability['available']),
        'factors': availability['available'],
        'excluded': availability['excluded'],
        'n_excluded': len(availability['excluded'])
    }
    
    with open(output_dir / "factor_list.json", 'w') as f:
        json.dump(factor_list, f, indent=2)
    print(f"   💾 {output_dir / 'factor_list.json'}")
    
    # Salva report completo
    report = {
        'config': {
            'pca_start_date': PCA_START_DATE,
            'pca_window_length': PCA_WINDOW_LENGTH,
            'pca_variance_threshold': PCA_VARIANCE_THRESHOLD,
            'pca_timing': PCA_TIMING,
            'factors_end_date': FACTORS_END_DATE
        },
        'factors': {
            'total_available': len(all_factors.columns),
            'selected_for_pca': availability['n_available'],
            'excluded': availability['n_excluded'],
            'window_start_required': availability['window_start_required'].strftime('%Y-%m-%d'),
            'excluded_factors': [d for d in availability['details'] if d['status'] == 'excluded']
        },
        'strategies': strategy_info
    }
    
    with open(output_dir / "pca_preprocessing_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"   💾 {output_dir / 'pca_preprocessing_report.json'}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print_header("SUMMARY")
    
    print(f"\n   PCA Configuration:")
    print(f"   {'─' * 50}")
    print(f"   Start date:          {PCA_START_DATE}")
    print(f"   Window length:       {PCA_WINDOW_LENGTH} months")
    print(f"   Variance threshold:  {PCA_VARIANCE_THRESHOLD:.0%}")
    print(f"   Timing:              {PCA_TIMING}")
    
    print(f"\n   Factors:")
    print(f"   {'─' * 50}")
    print(f"   Available:           {availability['n_available']}")
    print(f"   Excluded:            {availability['n_excluded']}")
    
    print(f"\n   Strategies:")
    print(f"   {'─' * 50}")
    print(f"   {'Strategy':<20} {'Period':<30} {'Obs':<10}")
    print(f"   {'-' * 60}")
    
    for name, info in strategy_info.items():
        period = f"{info.get('analysis_start', 'N/A')} → {info.get('analysis_end', 'N/A')}"
        n_obs = info.get('n_aligned_obs', 'N/A')
        print(f"   {name:<20} {period:<30} {n_obs:<10}")
    
    print(f"\n{'=' * 80}")
    print("✅ PCA PREPROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"\n📁 Outputs saved to: {output_dir}")
    print(f"🎯 Next step: python src/pca/02_pca_rolling.py")


if __name__ == "__main__":
    main()
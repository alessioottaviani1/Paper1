"""
================================================================================
02_pca_rolling.py - Rolling PCA Estimation (UPDATED)
================================================================================
Implementa PCA rolling seguendo Ludvigson & Ng (2009):

1. Per ogni mese t (da PCA_START_DATE in poi):
   - Stima loadings usando dati [t-W, t-1] (W = window length)
   - Standardizza fattori DENTRO la window (media e std su [t-W, t-1])
   - Calcola PC scores al tempo t usando quei loadings
   
2. Usa numero FISSO di PC (da config)

3. Esegue spanning regressions per ogni strategia:
   - Se PCA_TIMING = "predictive":   R_{t+1} = α + β'PC_t + ε
   - Se PCA_TIMING = "contemporaneous": R_t = α + β'PC_t + ε

4. Salva risultati e diagnostics CON TIMING NEL NOME FILE
   (così predictive e contemporaneous non si sovrascrivono)

Author: Alessio Ottaviani
Institution: EDHEC Business School - PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# IMPORT CONFIG
# ============================================================================

import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Try multiple config paths
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
        print(f"✅ Loaded config from: {config_path}")
        break

if pca_config is None:
    raise FileNotFoundError("PCA config file not found!")

# Esporta variabili
FACTORS_PATH = pca_config.FACTORS_PATH
STRATEGIES = pca_config.STRATEGIES
FACTORS_END_DATE = pca_config.FACTORS_END_DATE
PCA_START_DATE = pca_config.PCA_START_DATE
PCA_WINDOW_LENGTH = pca_config.PCA_WINDOW_LENGTH
PCA_N_COMPONENTS = pca_config.PCA_N_COMPONENTS
PCA_VARIANCE_THRESHOLD = pca_config.PCA_VARIANCE_THRESHOLD
PCA_TIMING = pca_config.PCA_TIMING

# Numero di PC per cui salvare varianza (per scree plot) - sempre >= PCA_N_COMPONENTS
PCA_N_COMPONENTS_FOR_SCREE = 15
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


# ============================================================================
# ROLLING PCA CLASS
# ============================================================================

class RollingPCA:
    """
    Implementa PCA rolling con standardizzazione in-window.
    
    Attributes:
        window_length: lunghezza rolling window in mesi
        n_components: numero FISSO di PC da usare
        variance_threshold: soglia varianza (solo per diagnostica)
    """
    
    def __init__(self, window_length: int, n_components: int, variance_threshold: float = 0.80):
        self.window_length = window_length
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        
        # Storage per risultati
        self.pc_scores = None
        self.variance_explained = []
        self.individual_variance = []
        self.diagnostics = []
        self.all_loadings = []
    
    def fit_transform(self, factors_df: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
        """
        Esegue PCA rolling da start_date in poi.
        
        Per ogni mese t >= start_date:
        1. Prende finestra [t-W, t-1]
        2. Standardizza i fattori usando media/std della finestra
        3. Fitta PCA sulla finestra
        4. Trasforma fattori al tempo t usando loadings stimati (primi n_components)
        
        Args:
            factors_df: DataFrame con fattori (index=date, columns=factors)
            start_date: prima data per cui calcolare PC scores
            
        Returns:
            DataFrame con PC scores (index=date, columns=PC1, PC2, ..., PC{n_components})
        """
        dates = factors_df.index
        dates = dates[dates >= start_date]
        
        all_scores = []
        all_dates = []
        
        print(f"\n   Rolling PCA from {start_date.strftime('%Y-%m-%d')}...")
        print(f"   Window length: {self.window_length} months")
        print(f"   Fixed number of components: {self.n_components}")
        
        for t_idx, t in enumerate(dates):
            # Trova indice di t nel dataframe originale
            t_loc = factors_df.index.get_loc(t)
            
            # Window: [t - W, t - 1]
            window_start_loc = t_loc - self.window_length
            window_end_loc = t_loc - 1  # esclude t
            
            if window_start_loc < 0:
                # Non abbastanza dati per la finestra
                continue
            
            # Estrai dati finestra
            window_data = factors_df.iloc[window_start_loc:window_end_loc + 1].copy()
            
            # Estrai dati al tempo t
            t_data = factors_df.iloc[t_loc:t_loc + 1].copy()
            
            # Check NaN nella finestra
            if window_data.isna().any().any():
                # Rimuovi colonne con NaN
                valid_cols = window_data.columns[~window_data.isna().any()]
                window_data = window_data[valid_cols]
                t_data = t_data[valid_cols]
            
            if len(window_data.columns) < self.n_components:
                # Troppo pochi fattori per il numero di componenti richiesto
                print(f"      ⚠️  {t.strftime('%Y-%m-%d')}: only {len(window_data.columns)} factors, need {self.n_components}, skipping")
                continue
            
            # ================================================================
            # STANDARDIZZAZIONE IN-WINDOW (come Ludvigson & Ng)
            # ================================================================
            
            # Calcola media e std sulla finestra
            window_mean = window_data.mean()
            window_std = window_data.std()
            
            # Evita divisione per zero
            window_std = window_std.replace(0, np.nan)
            
            # Standardizza finestra
            window_standardized = (window_data - window_mean) / window_std
            
            # Standardizza t usando stessi parametri
            t_standardized = (t_data - window_mean) / window_std
            
            # Rimuovi colonne con NaN dopo standardizzazione
            valid_cols = ~(window_standardized.isna().any() | t_standardized.isna().any())
            window_standardized = window_standardized.loc[:, valid_cols]
            t_standardized = t_standardized.loc[:, valid_cols]
            
            if len(window_standardized.columns) < self.n_components:
                continue
            
            # ================================================================
            # FIT PCA SULLA FINESTRA
            # ================================================================
            
            # Fit PCA con più componenti (per scree plot) ma usa solo n_components per scores
            n_components_to_fit = min(PCA_N_COMPONENTS_FOR_SCREE, len(window_standardized.columns))
            pca = PCA(n_components=n_components_to_fit)
            pca.fit(window_standardized.values)
            
            # ================================================================
            # TRASFORMA DATI AL TEMPO T
            # ================================================================
            
            # Prendi solo i primi n_components scores (quelli usati nella regressione)
            scores_t_full = pca.transform(t_standardized.values)[0]
            scores_t = scores_t_full[:self.n_components]
            
            all_scores.append(scores_t)
            all_dates.append(t)
            
            # Varianza spiegata dai primi n_components (per regressione)
            var_explained = np.sum(pca.explained_variance_ratio_[:self.n_components])
            self.variance_explained.append(var_explained)
            # Salva varianza di TUTTI i PC fittati (per scree plot)
            self.individual_variance.append(pca.explained_variance_ratio_.tolist())
            self.all_loadings.append(pca.components_[:self.n_components, :].copy())

            
            self.diagnostics.append({
                'date': t,
                'n_factors': len(window_standardized.columns),
                'n_components': self.n_components,
                'variance_explained': var_explained,
                'var_pc1': pca.explained_variance_ratio_[0] if len(pca.explained_variance_ratio_) > 0 else 0,
                'var_pc2': pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0,
                'var_pc3': pca.explained_variance_ratio_[2] if len(pca.explained_variance_ratio_) > 2 else 0,
                'window_start': factors_df.index[window_start_loc],
                'window_end': factors_df.index[window_end_loc]
            })
            
            # Progress
            if (t_idx + 1) % 50 == 0:
                print(f"      Processed {t_idx + 1}/{len(dates)} dates...")
        
        # Crea DataFrame con scores
        columns = [f'PC{i+1}' for i in range(self.n_components)]
        self.pc_scores = pd.DataFrame(
            all_scores,
            index=all_dates,
            columns=columns
        )
        
        print(f"   ✅ Generated PC scores for {len(all_dates)} dates")
        print(f"   📊 Components used: {self.n_components} (fixed)")
        print(f"   📊 Avg variance explained: {np.mean(self.variance_explained):.1%}")
        print(f"   📊 Min variance explained: {np.min(self.variance_explained):.1%}")
        print(f"   📊 Max variance explained: {np.max(self.variance_explained):.1%}")
        
        # Print average variance per PC
        if self.individual_variance:
            avg_individual = np.mean(self.individual_variance, axis=0)
            print(f"\n   📊 Average variance per PC:")
            for i, var in enumerate(avg_individual):
                print(f"      PC{i+1}: {var:.1%}")
        
        return self.pc_scores
    
    def get_diagnostics_df(self) -> pd.DataFrame:
        """Ritorna diagnostics come DataFrame."""
        return pd.DataFrame(self.diagnostics)
    
    def get_average_variance_per_pc(self) -> list:
        """Ritorna varianza media spiegata da ogni PC."""
        if self.individual_variance:
            return np.mean(self.individual_variance, axis=0).tolist()
        return []


# ============================================================================
# SPANNING REGRESSION
# ============================================================================

def run_spanning_regression(
    returns: pd.Series,
    pc_scores: pd.DataFrame,
    timing: str = "predictive",
    n_components: int = None,
    hac_lags: int = 6
) -> dict:
    """
    Esegue spanning regression dei returns sui PC.
    
    Args:
        returns: Serie dei rendimenti strategia
        pc_scores: DataFrame con PC scores
        timing: "predictive" (PC_t → R_{t+1}) o "contemporaneous" (PC_t → R_t)
        n_components: numero di PC da usare (None = tutti)
        hac_lags: lag per HAC standard errors
        
    Returns:
        dict con risultati regressione
    """
    # Allinea date
    common_dates = returns.index.intersection(pc_scores.index)
    
    if len(common_dates) < 30:
        return {'error': f'Insufficient observations: {len(common_dates)}'}
    
    # Prepara X e y in base al timing
    if timing == "predictive":
        # PC_t spiega R_{t+1}
        # Quindi PC deve essere shiftato indietro di 1 (o R avanti di 1)
        pc_aligned = pc_scores.loc[common_dates].iloc[:-1]  # PC_t
        ret_aligned = returns.loc[common_dates].iloc[1:]    # R_{t+1}
        
        # Riallinea indici
        pc_aligned.index = ret_aligned.index
        
    else:  # contemporaneous
        # PC_t spiega R_t
        pc_aligned = pc_scores.loc[common_dates]
        ret_aligned = returns.loc[common_dates]
    
    # Seleziona numero componenti
    if n_components is not None:
        pc_cols = [f'PC{i+1}' for i in range(n_components)]
        pc_cols = [c for c in pc_cols if c in pc_aligned.columns]
        X = pc_aligned[pc_cols]
    else:
        X = pc_aligned
    
    y = ret_aligned
    
    # Rimuovi righe con NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
    if len(y) < 30:
        return {'error': f'Insufficient observations after cleaning: {len(y)}'}
    
    # Aggiungi costante
    X = sm.add_constant(X)
    
    # Fit OLS con HAC standard errors
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': hac_lags})
    
    # Estrai risultati
    output = {
        'n_obs': int(len(y)),
        'n_components': X.shape[1] - 1,  # escludi costante
        'alpha': float(results.params['const']),
        'alpha_se': float(results.bse['const']),
        'alpha_tstat': float(results.tvalues['const']),
        'alpha_pvalue': float(results.pvalues['const']),
        'r_squared': float(results.rsquared),
        'r_squared_adj': float(results.rsquared_adj),
        'betas': {k: float(v) for k, v in results.params.drop('const').items()},
        'betas_se': {k: float(v) for k, v in results.bse.drop('const').items()},
        'betas_tstat': {k: float(v) for k, v in results.tvalues.drop('const').items()},
        'betas_pvalue': {k: float(v) for k, v in results.pvalues.drop('const').items()},
        'timing': timing,
        'hac_lags': hac_lags
    }
    
    return output


# ============================================================================
# MAIN
# ============================================================================

def main():
    print_header("PCA ROLLING ESTIMATION")
    
    # Print config
    print(f"\n📋 CONFIGURATION:")
    print(f"   PCA_START_DATE:        {PCA_START_DATE}")
    print(f"   PCA_WINDOW_LENGTH:     {PCA_WINDOW_LENGTH} months")
    print(f"   PCA_N_COMPONENTS:      {PCA_N_COMPONENTS}")
    print(f"   PCA_VARIANCE_THRESHOLD:{PCA_VARIANCE_THRESHOLD:.0%} (for diagnostics)")
    print(f"   PCA_TIMING:            {PCA_TIMING}")
    
    # ========================================================================
    # STEP 1: Load preprocessed factors
    # ========================================================================
    
    print_header("STEP 1: Load Data", "-")
    
    pca_output_dir = get_pca_output_dir()
    factors_path = pca_output_dir / "factors_for_pca.parquet"
    
    if not factors_path.exists():
        print(f"❌ ERROR: Preprocessed factors not found: {factors_path}")
        print("   Run 01_pca_preprocessing.py first!")
        return
    
    factors_df = pd.read_parquet(factors_path)
    print(f"   ✅ Loaded factors: {len(factors_df.columns)} factors, {len(factors_df)} months")
    
    # ========================================================================
    # STEP 2: Run Rolling PCA
    # ========================================================================
    
    print_header("STEP 2: Rolling PCA Estimation", "-")
    
    pca_start = pd.Timestamp(PCA_START_DATE)
    
    rolling_pca = RollingPCA(
        window_length=PCA_WINDOW_LENGTH,
        n_components=PCA_N_COMPONENTS,
        variance_threshold=PCA_VARIANCE_THRESHOLD
    )
    
    pc_scores = rolling_pca.fit_transform(factors_df, pca_start)
    
    # ========================================================================
    # SAVE PC SCORES (con timing nel nome!)
    # ========================================================================
    
    pc_scores.to_parquet(pca_output_dir / f"pc_scores_{PCA_TIMING}.parquet")
    print(f"\n   💾 Saved: pc_scores_{PCA_TIMING}.parquet")
    
    # Salva diagnostics (con timing nel nome!)
    diagnostics_df = rolling_pca.get_diagnostics_df()
    diagnostics_df.to_csv(pca_output_dir / f"pca_diagnostics_{PCA_TIMING}.csv", index=False)
    # Save average loadings across rolling windows
    if rolling_pca.all_loadings:
        avg_loadings = np.mean(rolling_pca.all_loadings, axis=0)
        factor_names = factors_df.columns.tolist()
        loadings_df = pd.DataFrame(
            avg_loadings,
            index=[f'PC{i+1}' for i in range(avg_loadings.shape[0])],
            columns=factor_names
        )
        loadings_df.to_csv(pca_output_dir / f"pca_avg_loadings_{PCA_TIMING}.csv")
        print(f"   💾 Saved: pca_avg_loadings_{PCA_TIMING}.csv")

        # ================================================================
        # LOADINGS STABILITY: correlation between consecutive windows
        # ================================================================
        # For each pair of consecutive windows, compute the absolute
        # correlation between the loading vectors of PC1, PC2, PC3.
        # High correlation (>0.90) means the PC interpretation is stable.
        # We use abs() because PCA eigenvectors have sign ambiguity:
        # flipping all signs of a PC is equally valid.
        # ================================================================
        n_windows = len(rolling_pca.all_loadings)
        dates_for_loadings = rolling_pca.pc_scores.index.tolist()
        n_pcs_stability = min(3, rolling_pca.all_loadings[0].shape[0])

        stability_records = []
        for w in range(1, n_windows):
            L_prev = rolling_pca.all_loadings[w - 1]  # (n_components, n_factors)
            L_curr = rolling_pca.all_loadings[w]

            record = {'date': dates_for_loadings[w]}
            for pc_idx in range(n_pcs_stability):
                corr = np.corrcoef(L_prev[pc_idx, :], L_curr[pc_idx, :])[0, 1]
                record[f'abs_corr_PC{pc_idx+1}'] = abs(corr)
            stability_records.append(record)

        stability_df = pd.DataFrame(stability_records)
        stability_df.to_csv(pca_output_dir / f"pca_loadings_stability_{PCA_TIMING}.csv", index=False)
        print(f"   💾 Saved: pca_loadings_stability_{PCA_TIMING}.csv")

        # Print summary
        print(f"\n   📊 Loadings stability (abs correlation between consecutive windows):")
        for pc_idx in range(n_pcs_stability):
            col = f'abs_corr_PC{pc_idx+1}'
            mean_corr = stability_df[col].mean()
            min_corr = stability_df[col].min()
            print(f"      PC{pc_idx+1}: mean={mean_corr:.4f}, min={min_corr:.4f}")

    print(f"   💾 Saved: pca_diagnostics_{PCA_TIMING}.csv")
    
    # ========================================================================
    # STEP 3: Spanning Regressions per ogni strategia
    # ========================================================================
    
    print_header("STEP 3: Spanning Regressions", "-")
    
    all_results = {}
    
    for strategy_name in STRATEGIES.keys():
        print(f"\n   📊 {strategy_name}:")
        
        strategy_pca_dir = get_strategy_pca_dir(strategy_name)
        returns_path = strategy_pca_dir / "y_returns_pca.parquet"
        
        if not returns_path.exists():
            print(f"      ❌ Returns not found: {returns_path}")
            continue
        
        returns = pd.read_parquet(returns_path)['Strategy_Return']
        
        # Run spanning regression
        results = run_spanning_regression(
            returns=returns,
            pc_scores=pc_scores,
            timing=PCA_TIMING,
            n_components=None,  # usa tutti
            hac_lags=6
        )
        
        if 'error' in results:
            print(f"      ❌ Error: {results['error']}")
            continue
        
        all_results[strategy_name] = results
        
        # Print risultati
        print(f"      Observations:      {results['n_obs']}")
        print(f"      PC components:     {results['n_components']}")
        print(f"      R²:                {results['r_squared']:.4f}")
        print(f"      R² adjusted:       {results['r_squared_adj']:.4f}")
        print(f"      Alpha:             {results['alpha']:.4f}")
        print(f"      Alpha t-stat:      {results['alpha_tstat']:.2f}")
        print(f"      Alpha p-value:     {results['alpha_pvalue']:.4f}")
        
        # Significatività alpha
        if results['alpha_pvalue'] < 0.01:
            sig = "***"
        elif results['alpha_pvalue'] < 0.05:
            sig = "**"
        elif results['alpha_pvalue'] < 0.10:
            sig = "*"
        else:
            sig = ""
        
        print(f"      Alpha significance: {sig}")
        
        # Salva risultati per strategia (con timing nel nome!)
        with open(strategy_pca_dir / f"spanning_regression_results_{PCA_TIMING}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"      💾 Saved: spanning_regression_results_{PCA_TIMING}.json")
    
    # ========================================================================
    # STEP 4: Summary Report
    # ========================================================================
    
    print_header("SUMMARY RESULTS")
    
    print(f"\n   {'Strategy':<20} {'N':<6} {'R²':<8} {'Alpha':<10} {'t-stat':<8} {'p-val':<8} {'Sig':<5}")
    print(f"   {'-' * 70}")
    
    for strategy_name, results in all_results.items():
        alpha = results['alpha']
        tstat = results['alpha_tstat']
        pval = results['alpha_pvalue']
        
        if pval < 0.01:
            sig = "***"
        elif pval < 0.05:
            sig = "**"
        elif pval < 0.10:
            sig = "*"
        else:
            sig = ""
        
        print(f"   {strategy_name:<20} {results['n_obs']:<6} {results['r_squared']:<8.4f} "
              f"{alpha:<10.4f} {tstat:<8.2f} {pval:<8.4f} {sig:<5}")
    
    print(f"\n   Significance: *** p<0.01, ** p<0.05, * p<0.10")
    
    # Salva summary globale (con timing nel nome!)
    summary = {
        'config': {
            'pca_start_date': PCA_START_DATE,
            'pca_window_length': PCA_WINDOW_LENGTH,
            'pca_n_components': PCA_N_COMPONENTS,
            'pca_variance_threshold': PCA_VARIANCE_THRESHOLD,
            'pca_timing': PCA_TIMING
        },
        'pca_diagnostics': {
            'n_dates': len(pc_scores),
            'n_components': PCA_N_COMPONENTS,
            'avg_variance_explained': float(np.mean(rolling_pca.variance_explained)),
            'min_variance_explained': float(np.min(rolling_pca.variance_explained)),
            'max_variance_explained': float(np.max(rolling_pca.variance_explained)),
            'avg_variance_per_pc': rolling_pca.get_average_variance_per_pc()
        },
        'spanning_results': all_results
    }
    
    with open(pca_output_dir / f"pca_summary_{PCA_TIMING}.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n   💾 Saved: pca_summary_{PCA_TIMING}.json")
    
    print(f"\n{'=' * 80}")
    print(f"✅ PCA ROLLING ESTIMATION COMPLETE (Timing: {PCA_TIMING})")
    print(f"{'=' * 80}")
    print(f"\n📁 Outputs saved to: {pca_output_dir}")
    print(f"\n💡 To run with different timing:")
    print(f"   1. Edit 00_pca_config.py: PCA_TIMING = 'contemporaneous'")
    print(f"   2. Re-run this script")
    print(f"   Results will be saved separately (no overwrite)")
    print(f"\n🎯 Next step: python src/pca/03_pca_results_tables.py")


if __name__ == "__main__":
    main()
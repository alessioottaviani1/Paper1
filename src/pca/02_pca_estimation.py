"""
================================================================================
02_pca_estimation.py - PCA Factor Estimation (full-sample) + Spanning
================================================================================
In-sample: stima K componenti principali UNA volta sul panel bilanciato
full-sample [PCA_START_DATE, end] e le usa come fattori in una spanning
regression contemporanea (Connor-Korajczyk 1986/1988; Giglio-Xiu 2021).
E' l'analogo in-sample dello spanning benchmark; la versione anti-look-ahead
e' l'OOS ricorsivo in 06_pca_oos.py.

Step:
1. Carica il panel bilanciato dei fattori (da 01).
2. PCA full-sample -> K PC scores (K da config).
3. Spanning per strategia, timing da PCA_TIMING:
   - "contemporaneous": R_t     = alpha + beta' PC_t + eps   (baseline)
   - "predictive":      R_{t+1} = alpha + beta' PC_t + eps   (non usato)
4. Salva scores, diagnostics e risultati con il timing nel nome file.

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
    print(f"\n== {title}")


# ============================================================================
# ROLLING PCA CLASS
# ============================================================================

class FullSamplePCA:
    """
    Stima K componenti principali UNA volta sul panel full-sample
    (standardizzazione full-sample). Vedi fit_full_sample.

    Attributes:
        n_components: numero FISSO di PC da usare
        variance_threshold: soglia varianza (solo per diagnostica)
    """
    
    def __init__(self, n_components: int, variance_threshold: float = 0.80):
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        
        # Storage per risultati
        self.pc_scores = None
        self.variance_explained = []
        self.individual_variance = []
        self.diagnostics = []
        self.all_loadings = []
    
    
    def fit_full_sample(self, factors_df: pd.DataFrame, start_date: pd.Timestamp) -> pd.DataFrame:
        """
        PCA FULL-SAMPLE: stima gli n_components PC UNA volta sull'intero panel
        bilanciato [start_date, end] e li usa come fattori (in-sample, come gli
        spanning benchmark). Sostituisce gli score rolling per i test in-sample;
        la versione anti-look-ahead e' l'OOS in 06_pca_oos.py.
        """
        panel = factors_df.loc[start_date:].copy()
        panel = panel.loc[:, ~panel.isna().any()]          # balanced su [start, end]
        if panel.shape[1] < self.n_components:
            raise ValueError(f"Solo {panel.shape[1]} fattori completi su "
                             f"[{start_date.date()}, end], servono >= {self.n_components}")
        mu, sd = panel.mean(), panel.std().replace(0, np.nan)
        Z = ((panel - mu) / sd).dropna(axis=1)             # standardizzazione full-sample
        n_fit = min(PCA_N_COMPONENTS_FOR_SCREE, Z.shape[1])
        pca = PCA(n_components=n_fit)
        scores = pca.fit_transform(Z.values)[:, :self.n_components]
        cols = [f'PC{i+1}' for i in range(self.n_components)]
        self.pc_scores = pd.DataFrame(scores, index=Z.index, columns=cols)
        # diagnostica (stima unica) — stesse strutture che main() si aspetta
        self.variance_explained = [float(np.sum(pca.explained_variance_ratio_[:self.n_components]))]
        self.individual_variance = [pca.explained_variance_ratio_.tolist()]
        self.all_loadings = [pca.components_[:self.n_components, :].copy()]
        self.diagnostics = [{
            'date': Z.index.min(), 'n_factors': int(Z.shape[1]),
            'n_components': self.n_components,
            'variance_explained': self.variance_explained[0],
            'var_pc1': float(pca.explained_variance_ratio_[0]),
            'var_pc2': float(pca.explained_variance_ratio_[1]) if n_fit > 1 else 0.0,
            'var_pc3': float(pca.explained_variance_ratio_[2]) if n_fit > 2 else 0.0,
            'window_start': Z.index.min(), 'window_end': Z.index.max(),
        }]
        print(f"\n   Full-sample PCA su [{Z.index.min().date()}, {Z.index.max().date()}]: "
              f"{Z.shape[1]} fattori, {len(Z)} mesi -> {self.n_components} PC "
              f"({self.variance_explained[0]:.1%} varianza)")
        # Bai-Ng (2002) IC: numero di fattori suggerito dal panel (ancora la scelta di K)
        _s = np.linalg.svd(Z.values, full_matrices=False)[1]
        _eig, (_T, _N) = _s ** 2, Z.shape
        _kmax = min(15, _N - 1)
        _g1 = ((_N + _T) / (_N * _T)) * np.log((_N * _T) / (_N + _T))
        _g2 = ((_N + _T) / (_N * _T)) * np.log(min(_N, _T))
        _ic1 = [np.log(_eig[k:].sum() / (_N * _T)) + k * _g1 for k in range(1, _kmax + 1)]
        _ic2 = [np.log(_eig[k:].sum() / (_N * _T)) + k * _g2 for k in range(1, _kmax + 1)]
        _k1, _k2 = int(np.argmin(_ic1)) + 1, int(np.argmin(_ic2)) + 1
        print(f"   Bai-Ng IC: ICp1 -> k={_k1}, ICp2 -> k={_k2}  (K usato = {self.n_components})")
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
    hac_lags: int = 4
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
    print_header("PCA ESTIMATION")
    
    # Print config
    print(f"  start={PCA_START_DATE}  K={PCA_N_COMPONENTS}  timing={PCA_TIMING}")
    
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
    
    print_header("STEP 2: Full-sample PCA", "-")
    
    pca_start = pd.Timestamp(PCA_START_DATE)
    
    rolling_pca = FullSamplePCA(
        n_components=PCA_N_COMPONENTS,
        variance_threshold=PCA_VARIANCE_THRESHOLD
    )
    
    pc_scores = rolling_pca.fit_full_sample(factors_df, pca_start)
    
    # ========================================================================
    # SAVE PC SCORES (con timing nel nome!)
    # ========================================================================
    
    pc_scores.to_parquet(pca_output_dir / f"pc_scores_{PCA_TIMING}.parquet")
    print(f"\n   💾 Saved: pc_scores_{PCA_TIMING}.parquet")
    
    # Diagnostics + loadings (full-sample = stima unica)
    rolling_pca.get_diagnostics_df().to_csv(
        pca_output_dir / f"pca_diagnostics_{PCA_TIMING}.csv", index=False)
    if rolling_pca.all_loadings:
        L = rolling_pca.all_loadings[0]
        pd.DataFrame(
            L,
            index=[f"PC{i+1}" for i in range(L.shape[0])],
            columns=factors_df.columns.tolist(),
        ).to_csv(pca_output_dir / f"pca_avg_loadings_{PCA_TIMING}.csv")
    print(f"   Saved: pca_diagnostics_{PCA_TIMING}.csv, pca_avg_loadings_{PCA_TIMING}.csv")
    
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
            hac_lags=4
        )
        
        if 'error' in results:
            print(f"      ❌ Error: {results['error']}")
            continue
        
        all_results[strategy_name] = results
             
        # Salva risultati per strategia (con timing nel nome!)
        with open(strategy_pca_dir / f"spanning_regression_results_{PCA_TIMING}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"      💾 Saved: spanning_regression_results_{PCA_TIMING}.json")
    
    # ========================================================================
    # STEP 4: Summary Report
    # ========================================================================
    
    print_header("SUMMARY RESULTS")
    
    print(f"\n   {'Strategy':<20} {'N':<6} {'R²':<8} {'Alpha(ann%)':<12} {'t-stat':<8} {'p-val':<8} {'Sig':<5}")
    print(f"   {'-' * 70}")
    
    for strategy_name, results in all_results.items():
        alpha = results['alpha'] * 12
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
              f"{alpha:<12.2f} {tstat:<8.2f} {pval:<8.4f} {sig:<5}")
    
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
    
    print(f"\n== Done (timing={PCA_TIMING}) -> {pca_output_dir}")


if __name__ == "__main__":
    main()
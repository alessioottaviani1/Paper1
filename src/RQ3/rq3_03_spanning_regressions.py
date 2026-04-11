"""
================================================================================
rq3_03_spanning_regressions.py — Spanning Regressions & PCA
================================================================================
RQ3: Slow-moving capital (Duffie 2010)

Questo file esegue:
  A. Spanning regressions su returns (contemporanee + lagged, HAC)
     → Analog di Fleckenstein et al. (2014) Table VI
  B. Spanning regressions su Δm (mispricing changes, HAC)
  C. Spanning regressions sui residui (purged da common factors)
  D. Interaction regressions: β₂ (Δm_j × Stress) per testare
     se la relazione si rafforza in stress
  E. PCA su strategy returns: common factor structure
  F. PCA rolling stability (out-of-sample)
  G. Multiple testing correction

Prerequisiti: eseguire prima rq3_01 e rq3_02

Author: Alessio Ottaviani
Institution: EDHEC Business School — PhD Thesis
================================================================================
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_00_config import *

print("=" * 72)
print("RQ3 — FILE 3: SPANNING REGRESSIONS & PCA")
print("=" * 72)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📂 Caricamento dati...")

master = pd.read_pickle(RQ3_DATA_DIR / "rq3_master_data.pkl")
df_returns = master['returns']
df_m       = master['mispricing']
df_dm      = master['delta_m']
regime     = master['regime']  # full coverage 2004-2025
n_missing = regime.reindex(df_returns.index).isna().sum()
if n_missing > 0:
    print(f"   ⚠️ {n_missing} return months without regime!")

# Fattori per purging
df_purge = pd.read_csv(RQ3_DATA_DIR / "purge_factors_monthly.csv",
                        index_col=0, parse_dates=True)

# Stress proxy mensile per interaction regressions
df_stress = pd.read_csv(RQ3_DATA_DIR / "stress_proxy_monthly.csv",
                         index_col=0, parse_dates=True)

df_dm_clean = df_dm.dropna()  # trivariate overlap

# Load full pairwise data
full_data_path = RQ3_DATA_DIR / "rq3_full_data.pkl"
if full_data_path.exists():
    _full = pd.read_pickle(full_data_path)
    df_dm_full = _full['delta_m_all']
else:
    df_dm_full = df_dm

T = len(df_returns)
print(f"   Returns: T={T}, Δm: T={len(df_dm_clean)}")

all_pvalues = {}
plt.style.use(FIGURE_STYLE)


# ============================================================================
# HELPER: SPANNING REGRESSION
# ============================================================================

def run_spanning_regression(y_series, x_dict, max_lags=SPANNING_MAX_LAGS,
                             nw_lags=NW_MAX_LAGS, label=""):
    """
    Spanning regression: y_t = α + Σ_j Σ_l β_{j,l} x_{j,t-l} + ε_t
    
    y_series: pd.Series (dependent)
    x_dict: dict {name: pd.Series} (regressors)
    max_lags: include contemporaneous (0) through max_lags
    
    Returns dict with coefficients, t-stats, p-values, R².
    """
    # Costruisci matrice regressori con lag
    regressors = {}
    for name, x in x_dict.items():
        for lag in range(0, max_lags + 1):
            col_name = f"{name}_L{lag}"
            regressors[col_name] = x.shift(lag)
    
    df_reg = pd.DataFrame(regressors)
    df_reg['y'] = y_series
    df_reg = df_reg.dropna()
    
    if len(df_reg) < 15:
        return None
    
    y = df_reg['y'].values
    X_cols = [c for c in df_reg.columns if c != 'y']
    X = sm.add_constant(df_reg[X_cols].values)
    
    try:
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
    except Exception as e:
        print(f"   ⚠️ Errore in {label}: {e}")
        return None
    
    result = {
        'T': len(df_reg),
        'R2': model.rsquared,
        'R2_adj': model.rsquared_adj,
        'F_stat': model.fvalue,
        'F_pval': model.f_pvalue,
        'coefficients': {},
    }
    
    # Coefficienti (skip costante)
    for i, col_name in enumerate(X_cols):
        result['coefficients'][col_name] = {
            'beta': model.params[i + 1],
            't_stat': model.tvalues[i + 1],
            'p_value': model.pvalues[i + 1],
        }
    
    return result


def print_spanning_result(result, dep_label):
    """Pretty print di un risultato spanning."""
    if result is None:
        print(f"   {dep_label}: non stimabile")
        return
    
    print(f"\n   {dep_label}:")
    print(f"   T={result['T']}, R²={result['R2']:.4f}, "
          f"R²_adj={result['R2_adj']:.4f}, F={result['F_stat']:.2f} (p={result['F_pval']:.4f})")
    
    for name, coef in result['coefficients'].items():
        sig = ""
        if coef['p_value'] < 0.01:
            sig = "***"
        elif coef['p_value'] < 0.05:
            sig = "**"
        elif coef['p_value'] < 0.10:
            sig = "*"
        print(f"      {name:25s}: β={coef['beta']:+.4f}, t={coef['t_stat']:+.2f}, "
              f"p={coef['p_value']:.4f} {sig}")


# ============================================================================
# SECTION A: SPANNING REGRESSIONS — RETURNS
# ============================================================================

print("\n" + "=" * 72)
print("SECTION A: Spanning Regressions — Returns")
print("=" * 72)

spanning_ret_results = []

for dep_name in STRATEGY_NAMES:
    y = df_returns[dep_name]
    x_dict = {STRATEGY_LABELS[s]: df_returns[s]
              for s in STRATEGY_NAMES if s != dep_name}
    
    result = run_spanning_regression(
        y, x_dict, max_lags=SPANNING_MAX_LAGS,
        label=f"Returns: {STRATEGY_LABELS[dep_name]}"
    )
    
    print_spanning_result(result, f"Dep = {STRATEGY_LABELS[dep_name]}")
    
    if result:
        row = {
            'dependent': STRATEGY_LABELS[dep_name],
            'series': 'Returns',
            'T': result['T'], 'R2': result['R2'], 'R2_adj': result['R2_adj'],
            'F_stat': result['F_stat'], 'F_pval': result['F_pval'],
        }
        for name, coef in result['coefficients'].items():
            row[f"{name}_beta"] = coef['beta']
            row[f"{name}_tstat"] = coef['t_stat']
            row[f"{name}_pval"] = coef['p_value']
            all_pvalues[f"span_ret_{dep_name}_{name}"] = coef['p_value']
        
        spanning_ret_results.append(row)

df_span_ret = pd.DataFrame(spanning_ret_results)
df_span_ret.to_csv(RQ3_TABLES_DIR / "T3a_spanning_returns.csv", index=False)
print(f"\n💾 T3a_spanning_returns.csv")


# ============================================================================
# SECTION B: SPANNING REGRESSIONS — Δm
# ============================================================================

print("\n" + "=" * 72)
print("SECTION B: Spanning Regressions — Δm (mispricing changes)")
print("=" * 72)

spanning_dm_results = []

for dep_name in STRATEGY_NAMES:
    # Pairwise: use full data, align per regression
    cols_needed = [dep_name] + [s for s in STRATEGY_NAMES if s != dep_name]
    dm_reg = df_dm_full[cols_needed].dropna()
    y = dm_reg[dep_name]
    x_dict = {STRATEGY_LABELS[s]: dm_reg[s]
              for s in STRATEGY_NAMES if s != dep_name}
    
    result = run_spanning_regression(
        y, x_dict, max_lags=SPANNING_MAX_LAGS,
        label=f"Δm: {STRATEGY_LABELS[dep_name]}"
    )
    
    print_spanning_result(result, f"Dep = {STRATEGY_LABELS[dep_name]}")
    
    if result:
        row = {
            'dependent': STRATEGY_LABELS[dep_name],
            'series': 'Δm',
            'T': result['T'], 'R2': result['R2'], 'R2_adj': result['R2_adj'],
            'F_stat': result['F_stat'], 'F_pval': result['F_pval'],
        }
        for name, coef in result['coefficients'].items():
            row[f"{name}_beta"] = coef['beta']
            row[f"{name}_tstat"] = coef['t_stat']
            row[f"{name}_pval"] = coef['p_value']
            all_pvalues[f"span_dm_{dep_name}_{name}"] = coef['p_value']
        
        spanning_dm_results.append(row)

df_span_dm = pd.DataFrame(spanning_dm_results)
df_span_dm.to_csv(RQ3_TABLES_DIR / "T3b_spanning_delta_m.csv", index=False)
print(f"\n💾 T3b_spanning_delta_m.csv")


# ============================================================================
# SECTION C: SPANNING REGRESSIONS — PURGED (residui)
# ============================================================================

print("\n" + "=" * 72)
print("SECTION C: Spanning Regressions — Purged (residui da common factors)")
print("=" * 72)

purge_factors = COMMON_FACTORS_FOR_PURGING if PURGE_FACTOR_SET == "base" else COMMON_FACTORS_EXTENDED
print(f"   Fattori purge: {purge_factors}")

# Residualizza Δm
purge_aligned = df_purge[purge_factors].reindex(df_dm_clean.index).dropna()
residuals_dm = {}

for name in STRATEGY_NAMES:
    y = df_dm_clean[name].reindex(purge_aligned.index).dropna()
    X = sm.add_constant(purge_aligned.reindex(y.index))
    model = sm.OLS(y, X, missing='drop').fit()
    residuals_dm[name] = model.resid

df_resid_dm = pd.DataFrame(residuals_dm)

spanning_purged_results = []

for dep_name in STRATEGY_NAMES:
    y = df_resid_dm[dep_name]
    x_dict = {STRATEGY_LABELS[s]: df_resid_dm[s]
              for s in STRATEGY_NAMES if s != dep_name}
    
    result = run_spanning_regression(
        y, x_dict, max_lags=SPANNING_MAX_LAGS,
        label=f"Purged Δm: {STRATEGY_LABELS[dep_name]}"
    )
    
    print_spanning_result(result, f"Dep = {STRATEGY_LABELS[dep_name]} (purged)")
    
    if result:
        row = {
            'dependent': STRATEGY_LABELS[dep_name],
            'series': 'Δm_purged',
            'T': result['T'], 'R2': result['R2'], 'R2_adj': result['R2_adj'],
            'F_stat': result['F_stat'], 'F_pval': result['F_pval'],
        }
        for name, coef in result['coefficients'].items():
            row[f"{name}_beta"] = coef['beta']
            row[f"{name}_tstat"] = coef['t_stat']
            row[f"{name}_pval"] = coef['p_value']
            all_pvalues[f"span_purged_{dep_name}_{name}"] = coef['p_value']
        
        spanning_purged_results.append(row)

df_span_purged = pd.DataFrame(spanning_purged_results)
df_span_purged.to_csv(RQ3_TABLES_DIR / "T3c_spanning_purged.csv", index=False)
print(f"\n💾 T3c_spanning_purged.csv")


# ============================================================================
# SECTION D: INTERACTION REGRESSIONS (β₂ stress)
# ============================================================================

print("\n" + "=" * 72)
print("SECTION D: Interaction Regressions (Δm_i = α + β₁Δm_j + β₂(Δm_j×Stress) + β₃Stress)")
print("=" * 72)

# Usa la proxy di stress default (livello mensile), standardizzata (z-score)
# per interpretabilità: β₂ = effetto di 1 SD di stress in più
stress_raw = df_stress[DEFAULT_STRESS_PROXY].reindex(df_dm_clean.index, method='ffill')
stress_mean = stress_raw.mean()
stress_std = stress_raw.std()
stress_level = (stress_raw - stress_mean) / stress_std
print(f"   Stress proxy: {DEFAULT_STRESS_PROXY} (standardizzato: μ={stress_mean:.1f}, σ={stress_std:.1f})")

interaction_results = []

for s1, s2 in STRATEGY_PAIRS:
    for dep, indep in [(s1, s2), (s2, s1)]:
        dep_label = STRATEGY_LABELS[dep]
        indep_label = STRATEGY_LABELS[indep]
        
        # Allinea dati
        stress_full = df_stress[DEFAULT_STRESS_PROXY].reindex(df_dm_full.index, method='ffill')
        stress_full_z = (stress_full - stress_full.mean()) / stress_full.std()
        df_int = pd.DataFrame({
            'y': df_dm_full[dep],
            'x': df_dm_full[indep],
            'stress': stress_full_z,
        }).dropna()
        
        if len(df_int) < 15:
            continue
        
        # x × stress
        df_int['x_stress'] = df_int['x'] * df_int['stress']
        
        y = df_int['y'].values
        X = sm.add_constant(df_int[['x', 'x_stress', 'stress']].values)
        
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': NW_MAX_LAGS})
        
        beta1 = model.params[1]  # x
        beta2 = model.params[2]  # x × stress (interaction)
        beta3 = model.params[3]  # stress
        
        print(f"\n   Dep={dep_label}, Indep={indep_label}:")
        print(f"      β₁(Δm_j)          = {beta1:+.4f}, t={model.tvalues[1]:+.2f}, "
              f"p={model.pvalues[1]:.4f}")
        print(f"      β₂(Δm_j × Stress) = {beta2:+.6f}, t={model.tvalues[2]:+.2f}, "
              f"p={model.pvalues[2]:.4f}")
        print(f"      β₃(Stress)         = {beta3:+.4f}, t={model.tvalues[3]:+.2f}, "
              f"p={model.pvalues[3]:.4f}")
        print(f"      R² = {model.rsquared:.4f}, T = {len(df_int)}")
        
        interaction_results.append({
            'dependent': dep_label, 'independent': indep_label,
            'stress_proxy': DEFAULT_STRESS_PROXY,
            'beta1': beta1, 'beta1_t': model.tvalues[1], 'beta1_p': model.pvalues[1],
            'beta2': beta2, 'beta2_t': model.tvalues[2], 'beta2_p': model.pvalues[2],
            'beta3': beta3, 'beta3_t': model.tvalues[3], 'beta3_p': model.pvalues[3],
            'R2': model.rsquared, 'T': len(df_int),
        })
        
        all_pvalues[f"interaction_{dep}_{indep}_beta2"] = model.pvalues[2]

df_interaction = pd.DataFrame(interaction_results)
df_interaction.to_csv(RQ3_TABLES_DIR / "T3d_interaction_regressions.csv", index=False)
export_interaction_tex(df_interaction, RQ3_TABLES_DIR / "RQ3_interaction_regressions_slide.tex")
print(f"\n💾 T3d_interaction_regressions.csv")

# --- D2: Interaction con dummy HIGH (β₂ direttamente interpretabile) ---
print(f"\n--- Interaction con Dummy HIGH ---")

regime_dm_full = master['regime'].reindex(df_dm_full.index, method='ffill')
d_high_full = (regime_dm_full == "HIGH").astype(float)

interaction_dummy_results = []

for s1, s2 in STRATEGY_PAIRS:
    for dep, indep in [(s1, s2), (s2, s1)]:
        dep_label = STRATEGY_LABELS[dep]
        indep_label = STRATEGY_LABELS[indep]
        
        df_int2 = pd.DataFrame({
            'y': df_dm_full[dep],
            'x': df_dm_full[indep],
            'D_high': d_high_full,
        }).dropna()
        
        if len(df_int2) < 15 or df_int2['D_high'].sum() < 3:
            continue
        
        df_int2['x_Dhigh'] = df_int2['x'] * df_int2['D_high']
        
        y = df_int2['y'].values
        X = sm.add_constant(df_int2[['x', 'x_Dhigh', 'D_high']].values)
        
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': NW_MAX_LAGS})
        
        print(f"\n   Dep={dep_label}, Indep={indep_label}:")
        print(f"      β₁(Δm_j)             = {model.params[1]:+.4f}, t={model.tvalues[1]:+.2f}, "
              f"p={model.pvalues[1]:.4f}")
        print(f"      β₂(Δm_j × D_HIGH)    = {model.params[2]:+.4f}, t={model.tvalues[2]:+.2f}, "
              f"p={model.pvalues[2]:.4f}")
        print(f"      β₃(D_HIGH)            = {model.params[3]:+.4f}, t={model.tvalues[3]:+.2f}, "
              f"p={model.pvalues[3]:.4f}")
        
        interaction_dummy_results.append({
            'dependent': dep_label, 'independent': indep_label,
            'beta1': model.params[1], 'beta1_t': model.tvalues[1], 'beta1_p': model.pvalues[1],
            'beta2_dummy': model.params[2], 'beta2_t': model.tvalues[2], 'beta2_p': model.pvalues[2],
            'beta3': model.params[3], 'beta3_t': model.tvalues[3], 'beta3_p': model.pvalues[3],
            'R2': model.rsquared, 'T': len(df_int2), 'n_HIGH': int(df_int2['D_high'].sum()),
        })
        
        all_pvalues[f"interaction_dummy_{dep}_{indep}_beta2"] = model.pvalues[2]

df_int_dummy = pd.DataFrame(interaction_dummy_results)
df_int_dummy.to_csv(RQ3_TABLES_DIR / "T3d2_interaction_dummy.csv", index=False)
export_interaction_tex(df_int_dummy, RQ3_TABLES_DIR / "RQ3_interaction_dummy_slide.tex",
                       title="Interaction Regressions — HIGH Dummy")
print(f"\n💾 T3d2_interaction_dummy.csv")


# ============================================================================
# SECTION E: PCA — Common Factor Structure
# ============================================================================

print("\n" + "=" * 72)
print("SECTION E: PCA — Common Factor Structure")
print("=" * 72)

# --- E1: PCA su returns ---
print("\n--- PCA su Returns ---")

returns_clean = df_returns.dropna()
scaler = StandardScaler()
returns_std = scaler.fit_transform(returns_clean)

pca = PCA()
pca.fit(returns_std)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"   T = {len(returns_clean)}")
for i in range(len(STRATEGY_NAMES)):
    print(f"   PC{i+1}: varianza spiegata = {explained_var[i]:.4f} "
          f"({explained_var[i]*100:.1f}%), cumulativa = {cumulative_var[i]*100:.1f}%")

# Loadings
loadings = pd.DataFrame(
    pca.components_.T,
    index=STRATEGY_NAMES,
    columns=[f"PC{i+1}" for i in range(len(STRATEGY_NAMES))]
)
print(f"\n   Loadings:")
print(loadings.round(3).to_string())

# Salva PC scores
pc_scores = pd.DataFrame(
    pca.transform(returns_std),
    index=returns_clean.index,
    columns=[f"PC{i+1}" for i in range(len(STRATEGY_NAMES))]
)

# Correlazione PC1 con stress proxy
stress_aligned = df_stress.reindex(pc_scores.index, method='ffill')
print(f"\n   Correlazione PC1 con stress proxy:")
for proxy_name in ALL_STRESS_PROXIES:
    if proxy_name in stress_aligned.columns:
        corr = pc_scores['PC1'].corr(stress_aligned[proxy_name])
        print(f"      PC1 vs {proxy_name}: ρ = {corr:+.3f}")

# --- E1b: Regressione multivariata PC1 ~ funding stress factors ---
# Testa multiple configurazioni per trovare le migliori proxy di
# intermediary capital / funding liquidity (Duffie 2010)
print(f"\n--- Regressione multivariata PC1 ~ funding stress ---")

factors_monthly = pd.read_parquet(FACTORS_PATH)

# Definiamo i set di proxy da testare
funding_sets = {
    "Core":     PC1_FUNDING_FACTORS,
    "Extended": PC1_FUNDING_FACTORS_FULL,
}

pc1_funding_results_all = []

for set_name, factor_list in funding_sets.items():
    # Filtra solo fattori disponibili
    available = [f for f in factor_list if f in factors_monthly.columns]
    if not available:
        print(f"\n   Set '{set_name}': ⚠️ nessun fattore disponibile, skip")
        continue
    
    print(f"\n   Set '{set_name}': {available}")
    
    pc1_reg_factors = factors_monthly[available].reindex(pc_scores.index).dropna()
    pc1_aligned = pc_scores['PC1'].reindex(pc1_reg_factors.index)
    common_pc1 = pc1_aligned.dropna().index.intersection(pc1_reg_factors.dropna().index)
    
    if len(common_pc1) < 15:
        print(f"   ⚠️ Troppo pochi dati ({len(common_pc1)}), skip")
        continue
    
    y_pc1 = pc1_aligned[common_pc1].values
    X_pc1 = sm.add_constant(pc1_reg_factors.loc[common_pc1].values)
    
    model_pc1 = sm.OLS(y_pc1, X_pc1).fit(cov_type='HAC', cov_kwds={'maxlags': NW_MAX_LAGS})
    
    print(f"   T = {len(common_pc1)}, R² = {model_pc1.rsquared:.4f}, "
          f"R²_adj = {model_pc1.rsquared_adj:.4f}, F p = {model_pc1.f_pvalue:.4f}")
    
    for i, fname in enumerate(available):
        beta = model_pc1.params[i + 1]
        tval = model_pc1.tvalues[i + 1]
        pval = model_pc1.pvalues[i + 1]
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
        print(f"   {fname:20s}: β={beta:+.4f}, t={tval:+.2f}, p={pval:.4f} {sig}")
        
        pc1_funding_results_all.append({
            'set': set_name, 'variable': fname, 'beta': beta,
            't_stat': tval, 'p_value': pval,
            'R2': model_pc1.rsquared, 'R2_adj': model_pc1.rsquared_adj,
            'T': len(common_pc1),
        })
        all_pvalues[f"pc1_funding_{set_name}_{fname}"] = pval

if pc1_funding_results_all:
    pd.DataFrame(pc1_funding_results_all).to_csv(
        RQ3_TABLES_DIR / "T3e2_pc1_funding_regression.csv", index=False)
    print(f"\n💾 T3e2_pc1_funding_regression.csv")
    
    # Build dict for tex export
    _pc1_tex_data = {}
    for r in pc1_funding_results_all:
        sn = r['set']
        if sn not in _pc1_tex_data:
            _pc1_tex_data[sn] = {'factors': [], 'betas': [], 'tvals': [],
                                  'pvals': [], 'R2': r['R2'],
                                  'R2_adj': r['R2_adj'], 'T': r['T']}
        _pc1_tex_data[sn]['factors'].append(r['variable'])
        _pc1_tex_data[sn]['betas'].append(r['beta'])
        _pc1_tex_data[sn]['tvals'].append(r['t_stat'])
        _pc1_tex_data[sn]['pvals'].append(r['p_value'])
    
    export_pc1_funding_tex(_pc1_tex_data,
                            RQ3_TABLES_DIR / "RQ3_pc1_funding_slide.tex")

    # Report: qual è il set migliore?
    best_set = max(
        {s: r['R2_adj'] for r in pc1_funding_results_all for s in [r['set']]}.items(),
        key=lambda x: x[1]
    )
    print(f"\n   ⭐ Best set per R²_adj: '{best_set[0]}' (R²_adj={best_set[1]:.4f})")

# --- E2: PCA su Δm ---
print("\n--- PCA su Δm ---")

dm_clean_pca = df_dm_clean.dropna()
dm_std = scaler.fit_transform(dm_clean_pca)

pca_dm = PCA()
pca_dm.fit(dm_std)

explained_var_dm = pca_dm.explained_variance_ratio_
cumulative_var_dm = np.cumsum(explained_var_dm)

for i in range(len(STRATEGY_NAMES)):
    print(f"   PC{i+1}: varianza spiegata = {explained_var_dm[i]:.4f} "
          f"({explained_var_dm[i]*100:.1f}%), cumulativa = {cumulative_var_dm[i]*100:.1f}%")

loadings_dm = pd.DataFrame(
    pca_dm.components_.T,
    index=STRATEGY_NAMES,
    columns=[f"PC{i+1}" for i in range(len(STRATEGY_NAMES))]
)
print(f"\n   Loadings Δm:")
print(loadings_dm.round(3).to_string())

# Salva tabella PCA
pca_summary = pd.DataFrame({
    'Component': [f"PC{i+1}" for i in range(len(STRATEGY_NAMES))],
    'Var_Explained_Returns': explained_var,
    'Cumulative_Returns': cumulative_var,
    'Var_Explained_Dm': explained_var_dm,
    'Cumulative_Dm': cumulative_var_dm,
})
pca_summary.to_csv(RQ3_TABLES_DIR / "T3e_pca_summary.csv", index=False)
loadings.to_csv(RQ3_TABLES_DIR / "T3e_pca_loadings_returns.csv")
loadings_dm.to_csv(RQ3_TABLES_DIR / "T3e_pca_loadings_dm.csv")
print(f"\n💾 T3e_pca_*.csv")


# ============================================================================
# SECTION F: PCA ROLLING STABILITY
# ============================================================================

print("\n" + "=" * 72)
print("SECTION F: PCA Rolling Stability (Out-of-Sample)")
print("=" * 72)

rolling_loadings = []
rolling_var_explained = []

for t in range(PCA_ROLLING_WINDOW, len(returns_clean)):
    # Out-of-sample: fit su [t-W, t-1], valuta loadings al tempo t
    train_window = returns_clean.iloc[t - PCA_ROLLING_WINDOW:t]
    
    if train_window.isna().any().any() or len(train_window) < PCA_ROLLING_WINDOW:
        continue
    
    scaler_roll = StandardScaler()
    window_std = scaler_roll.fit_transform(train_window)
    pca_roll = PCA(n_components=1)
    pca_roll.fit(window_std)
    
    rolling_loadings.append({
        'date': returns_clean.index[t],
        **{f"loading_{name}": pca_roll.components_[0][i]
           for i, name in enumerate(STRATEGY_NAMES)}
    })
    
    rolling_var_explained.append({
        'date': returns_clean.index[t],
        'PC1_var_explained': pca_roll.explained_variance_ratio_[0],
    })

df_roll_loadings = pd.DataFrame(rolling_loadings).set_index('date')
df_roll_var = pd.DataFrame(rolling_var_explained).set_index('date')

# Plot: rolling loadings
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle(f"PCA Rolling Stability (window={PCA_ROLLING_WINDOW} months)",
             fontsize=13, fontweight='bold')

# Panel 1: loadings
ax = axes[0]
for name in STRATEGY_NAMES:
    col = f"loading_{name}"
    # Loadings possono avere sign flip → prendiamo il valore assoluto
    # o correggiamo il segno per consistenza con il full-sample
    full_sample_sign = np.sign(loadings.loc[name, 'PC1'])
    adjusted = df_roll_loadings[col] * full_sample_sign
    ax.plot(adjusted.index, adjusted.values,
            color=STRATEGY_COLORS[name], linewidth=1.0,
            label=STRATEGY_LABELS[name])

ax.set_ylabel("PC1 Loading (sign-adjusted)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='grey', linewidth=0.5)

# Panel 2: variance explained
ax = axes[1]
ax.plot(df_roll_var.index, df_roll_var['PC1_var_explained'].values * 100,
        color='black', linewidth=1.0)
ax.axhline(50, color='red', linewidth=0.8, linestyle='--',
           label='50% threshold')
ax.set_ylabel("PC1 Variance Explained (%)", fontsize=10)
ax.set_xlabel("Date")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))

fig.tight_layout()
fig.savefig(RQ3_FIGURES_DIR / f"D1_pca_rolling_stability.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches='tight')
print(f"📊 D1_pca_rolling_stability.{FIGURE_FORMAT}")
plt.close()

df_roll_loadings.to_csv(RQ3_TABLES_DIR / "T3f_pca_rolling_loadings.csv")
df_roll_var.to_csv(RQ3_TABLES_DIR / "T3f_pca_rolling_variance.csv")
print(f"💾 T3f_pca_rolling_*.csv")


# ============================================================================
# SECTION G: MULTIPLE TESTING CORRECTION
# ============================================================================

print("\n" + "=" * 72)
print("SECTION G: Multiple Testing Correction")
print("=" * 72)

pval_names = list(all_pvalues.keys())
pval_values = [all_pvalues[k] for k in pval_names]

valid_mask = [isinstance(v, (int, float)) and not np.isnan(v) for v in pval_values]
pval_names_valid = [n for n, m in zip(pval_names, valid_mask) if m]
pval_values_valid = [v for v, m in zip(pval_values, valid_mask) if m]

if pval_values_valid:
    reject, pvals_corrected, _, _ = multipletests(
        pval_values_valid, alpha=MTC_ALPHA, method=MTC_METHOD
    )
    
    mtc_results = pd.DataFrame({
        'test': pval_names_valid,
        'p_raw': pval_values_valid,
        'p_corrected': pvals_corrected,
        'reject_H0': reject,
    }).sort_values('p_raw')
    
    mtc_results.to_csv(RQ3_TABLES_DIR / "T3g_multiple_testing.csv", index=False)
    
    n_reject = reject.sum()
    print(f"   Test totali: {len(reject)}")
    print(f"   Reject H0 (FDR {MTC_ALPHA}): {n_reject}")
    print(f"\n   Top 10:")
    print(mtc_results.head(10).to_string(index=False))
    print(f"\n💾 T3g_multiple_testing.csv")


# ============================================================================
# SUMMARY PLOT: R² spanning regressions comparison
# ============================================================================

print("\n" + "=" * 72)
print("Plot riassuntivo")
print("=" * 72)

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Spanning Regressions — R² Comparison", fontsize=13, fontweight='bold')

# Combina i 3 set di risultati
all_spans = []
for df_s, label in [(df_span_ret, "Returns"), (df_span_dm, "Δm"),
                     (df_span_purged, "Δm purged")]:
    if len(df_s) > 0:
        temp = df_s[['dependent', 'R2']].copy()
        temp['type'] = label
        all_spans.append(temp)

if all_spans:
    df_all_spans = pd.concat(all_spans)
    
    x_labels = df_all_spans['dependent'].unique()
    types = df_all_spans['type'].unique()
    x_pos = np.arange(len(x_labels))
    width = 0.25
    colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, t in enumerate(types):
        subset = df_all_spans[df_all_spans['type'] == t]
        r2_vals = [subset[subset['dependent'] == d]['R2'].values[0]
                   if d in subset['dependent'].values else 0 for d in x_labels]
        ax.bar(x_pos + i * width, r2_vals, width, label=t,
               color=colors_bar[i % len(colors_bar)], alpha=0.8, edgecolor='black')
    
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_ylabel("R²", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig(RQ3_FIGURES_DIR / f"D2_spanning_R2_comparison.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches='tight')
print(f"📊 D2_spanning_R2_comparison.{FIGURE_FORMAT}")
plt.close()


# ============================================================================
# DONE
# ============================================================================

print("\n" + "=" * 72)
print("✅ FILE 3 COMPLETATO")
print("=" * 72)
print(f"\n   Tabelle: {len(list(RQ3_TABLES_DIR.glob('T3*')))} file")
print(f"   Figure:  {len(list(RQ3_FIGURES_DIR.glob('D*')))} file")
print(f"\n   Prossimo step → rq3_04_var_analysis.py")
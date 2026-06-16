"""
================================================================================
rq3_06_subperiod_duffie_scorecard.py — Sub-Period Duffie Scorecard
================================================================================
RQ3: Slow-moving capital (Duffie 2010)

Per ogni sotto-periodo, replica le analisi principali e produce una
"Duffie scorecard" — quante predizioni teoriche sono confermate?

Predizioni Duffie testate:
  P1. Correlazione positiva cross-strategy nei Δm
  P2. Co-widening in eccesso (P_joint > P_indep)
  P3. Spanning: R² > 0 nelle regressioni Δm_i ~ Δm_j
  P4. Granger causality bidirezionale
  P5. PCA: PC1 spiega > 50% della varianza
  P6. Persistenza più alta in stress (φ_HIGH > φ_LOW)

Sotto-periodi:
  - Full sample (overlap trivariate)
  - Pre-COVID (inizio → 2020-02)
  - Post-COVID (2020-03 → fine)
  - ECB Hiking (2022-07 → 2024-06)
  - Pairwise long sample (per coppie con più storia)

Author: Alessio Ottaviani
Institution: EDHEC Business School — PhD Thesis
================================================================================
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_00_config import *

print("=" * 72)
print("RQ3 — FILE 6: SUB-PERIOD DUFFIE SCORECARD")
print("=" * 72)


# ============================================================================
# LOAD DATA
# ============================================================================

print("\n📂 Caricamento dati...")

master = pd.read_pickle(RQ3_DATA_DIR / "rq3_master_data.pkl")
df_returns = master['returns']
df_m       = master['mispricing']
df_dm      = master['delta_m']
regime     = master['regime']

df_stress = pd.read_csv(RQ3_DATA_DIR / "stress_proxy_monthly.csv",
                         index_col=0, parse_dates=True)

# Full data per pairwise long
full_data_path = RQ3_DATA_DIR / "rq3_full_data.pkl"
if full_data_path.exists():
    full_data = pd.read_pickle(full_data_path)
    dm_all = full_data['delta_m_all']
    ret_all = full_data['returns_all']
else:
    dm_all = df_dm
    ret_all = df_returns

df_dm_clean = df_dm.dropna()
T_full = len(df_dm_clean)

MIN_OBS_REGIME_TEST = 12   # coerente con File 02

print(f"   Full sample: T={T_full}")

plt.style.use(FIGURE_STYLE)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def test_correlation(dm1, dm2, nw_lags=NW_MAX_LAGS):
    """Pearson con HAC e Spearman."""
    common = dm1.dropna().index.intersection(dm2.dropna().index)
    if len(common) < 10:
        return {'rho_pearson': np.nan, 'p_pearson': np.nan,
                'rho_spearman': np.nan, 'p_spearman': np.nan, 'T': len(common)}
    x, y = dm1[common].values, dm2[common].values
    
    # HAC Pearson
    X = sm.add_constant(x)
    try:
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
        rho = np.corrcoef(x, y)[0, 1]
        p_hac = model.pvalues[1]
    except:
        rho, p_hac = np.nan, np.nan
    
    # Spearman
    rho_sp, p_sp = stats.spearmanr(x, y)
    
    return {'rho_pearson': rho, 'p_pearson': p_hac,
            'rho_spearman': rho_sp, 'p_spearman': p_sp, 'T': len(common)}


def test_cowidening(dm1, dm2):
    """Co-widening excess probability + Fisher exact."""
    common = dm1.dropna().index.intersection(dm2.dropna().index)
    if len(common) < 10:
        return {'P_joint': np.nan, 'P_indep': np.nan, 'excess': np.nan,
                'fisher_p': np.nan, 'T': len(common)}
    
    w1 = (dm1[common] > 0).astype(int)
    w2 = (dm2[common] > 0).astype(int)
    
    p1, p2 = w1.mean(), w2.mean()
    p_joint = ((w1 == 1) & (w2 == 1)).mean()
    p_indep = p1 * p2
    excess = p_joint - p_indep
    
    a = ((w1 == 1) & (w2 == 1)).sum()
    b = ((w1 == 1) & (w2 == 0)).sum()
    c = ((w1 == 0) & (w2 == 1)).sum()
    d = ((w1 == 0) & (w2 == 0)).sum()
    _, p_fisher = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
    
    return {'P_joint': p_joint, 'P_indep': p_indep, 'excess': excess,
            'fisher_p': p_fisher, 'T': len(common)}


def test_spanning(dm_dep, dm_indep, nw_lags=NW_MAX_LAGS):
    """Spanning regression: Δm_i = α + β₀Δm_j + β₁Δm_j(-1) + ε."""
    df_reg = pd.DataFrame({'y': dm_dep, 'x': dm_indep, 'x_L1': dm_indep.shift(1)}).dropna()
    if len(df_reg) < 15:
        return {'R2': np.nan, 'beta0': np.nan, 'p_beta0': np.nan, 'T': len(df_reg)}
    
    y = df_reg['y'].values
    X = sm.add_constant(df_reg[['x', 'x_L1']].values)
    try:
        model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
        return {'R2': model.rsquared, 'beta0': model.params[1],
                'p_beta0': model.pvalues[1], 'T': len(df_reg)}
    except:
        return {'R2': np.nan, 'beta0': np.nan, 'p_beta0': np.nan, 'T': len(df_reg)}


def test_granger(dm_df, var_col_names, max_lag=2):
    """Granger causality per tutte le coppie."""
    if len(dm_df) < 20:
        return []
    
    results = []
    try:
        model = VAR(dm_df)
        opt_lag = max(1, min(max_lag, len(dm_df) // 10))
        result = model.fit(maxlags=opt_lag, ic=None)
        
        for caused in var_col_names:
            for causing in var_col_names:
                if caused == causing:
                    continue
                try:
                    test = result.test_causality(caused, [causing], kind='f')
                    results.append({
                        'causing': causing, 'caused': caused,
                        'F_stat': test.test_statistic, 'p_value': test.pvalue,
                    })
                except:
                    pass
    except:
        pass
    return results


def test_pca(dm_df):
    """PCA: varianza spiegata da PC1."""
    clean = dm_df.dropna()
    if len(clean) < 15 or clean.shape[1] < 2:
        return {'PC1_var': np.nan, 'T': len(clean)}
    
    scaler = StandardScaler()
    std_data = scaler.fit_transform(clean)
    pca = PCA()
    pca.fit(std_data)
    return {'PC1_var': pca.explained_variance_ratio_[0], 'T': len(clean)}


def test_persistence_by_regime(m_series, regime_series):
    """AR(1) per regime con HAC SE (coerente con resto dell'analisi)."""
    results = {}
    for rl in ["LOW", "HIGH"]:
        mask = regime_series.reindex(m_series.index, method='ffill') == rl
        m_reg = m_series[mask].dropna()
        
        if len(m_reg) < MIN_OBS_REGIME_TEST:
            results[f"phi_{rl}"] = np.nan
            continue
        
        y = m_reg.iloc[1:].values
        x = m_reg.iloc[:-1].values
        X = sm.add_constant(x)
        try:
            nw_lags = max(1, int(len(y) ** (1/3)))
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': nw_lags})
            results[f"phi_{rl}"] = model.params[1]
        except:
            results[f"phi_{rl}"] = np.nan
    
    return results


# ============================================================================
# DEFINE SUB-PERIODS
# ============================================================================

# Sub-periodi trivariali
overlap_start = df_dm_clean.index.min()
overlap_end = df_dm_clean.index.max()

periods = {
    "Full Sample": (overlap_start, overlap_end),
}

for period_name, (start, end) in SUB_PERIODS.items():
    start_dt = max(pd.Timestamp(start), overlap_start)
    end_dt = min(pd.Timestamp(end), overlap_end)
    if start_dt < end_dt:
        periods[period_name] = (start_dt, end_dt)

print(f"\n📋 Sotto-periodi da testare:")
for name, (s, e) in periods.items():
    print(f"   {name:20s}: {s.strftime('%Y-%m')} → {e.strftime('%Y-%m')}")


# ============================================================================
# RUN SCORECARD PER OGNI SOTTO-PERIODO
# ============================================================================

print("\n" + "=" * 72)
print("DUFFIE SCORECARD — Tutte le predizioni per sotto-periodo")
print("=" * 72)

all_scorecard = []

for period_name, (start_dt, end_dt) in periods.items():
    print(f"\n{'='*60}")
    print(f"📊 {period_name} ({start_dt.strftime('%Y-%m')} → {end_dt.strftime('%Y-%m')})")
    print(f"{'='*60}")
    
    mask = (df_dm_clean.index >= start_dt) & (df_dm_clean.index <= end_dt)
    dm_sub = df_dm_clean[mask]
    m_sub = df_m.reindex(dm_sub.index)
    ret_sub = df_returns.reindex(dm_sub.index)
    
    T_sub = len(dm_sub)
    print(f"   T = {T_sub}")
    
    if T_sub < 15:
        print(f"   ⚠️ Troppo pochi dati, skip")
        continue
    
    # --- P1: Correlazioni cross-strategy ---
    print(f"\n   P1 — Correlazione Δm:")
    n_sig_corr = 0
    for s1, s2 in STRATEGY_PAIRS:
        pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
        res = test_correlation(dm_sub[s1], dm_sub[s2])
        sig = res['p_pearson'] < 0.05 if not np.isnan(res['p_pearson']) else False
        n_sig_corr += int(sig)
        print(f"      {pair_label}: ρ={res['rho_pearson']:+.3f}, p={res['p_pearson']:.4f} "
              f"{'✅' if sig else '❌'}")
    
    p1_pass = n_sig_corr > 0
    
    # --- P2: Co-widening excess ---
    print(f"\n   P2 — Co-widening excess:")
    n_excess_positive = 0
    for s1, s2 in STRATEGY_PAIRS:
        pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
        res = test_cowidening(dm_sub[s1], dm_sub[s2])
        positive = res['excess'] > 0 if not np.isnan(res['excess']) else False
        sig = res['fisher_p'] < 0.10 if not np.isnan(res['fisher_p']) else False
        n_excess_positive += int(positive and sig)
        print(f"      {pair_label}: excess={res['excess']:+.3f}, Fisher p={res['fisher_p']:.4f} "
              f"{'✅' if positive and sig else '❌'}")
    
    p2_pass = n_excess_positive > 0
    
    # --- P3: Spanning R² > 0 ---
    print(f"\n   P3 — Spanning regressions Δm:")
    n_sig_span = 0
    avg_r2 = []
    for s1, s2 in STRATEGY_PAIRS:
        for dep, indep in [(s1, s2), (s2, s1)]:
            res = test_spanning(dm_sub[dep], dm_sub[indep])
            sig = res['p_beta0'] < 0.05 if not np.isnan(res['p_beta0']) else False
            n_sig_span += int(sig)
            if not np.isnan(res['R2']):
                avg_r2.append(res['R2'])
            print(f"      {STRATEGY_LABELS[indep]:20s} → {STRATEGY_LABELS[dep]:20s}: "
                  f"R²={res['R2']:.4f}, β₀ p={res['p_beta0']:.4f} "
                  f"{'✅' if sig else '❌'}")
    
    p3_pass = n_sig_span >= 2  # almeno 2 su 6
    
    # --- P4: Granger causality ---
    print(f"\n   P4 — Granger causality:")
    dm_var = dm_sub.rename(columns=STRATEGY_LABELS)
    granger_res = test_granger(dm_var, list(dm_var.columns))
    n_sig_granger = 0
    for gr in granger_res:
        sig = gr['p_value'] < 0.10
        n_sig_granger += int(sig)
        print(f"      {gr['causing']:20s} → {gr['caused']:20s}: "
              f"F={gr['F_stat']:.3f}, p={gr['p_value']:.4f} "
              f"{'✅' if sig else '❌'}")
    
    p4_pass = n_sig_granger >= 1
    
    # --- P5: PCA PC1 > 50% ---
    print(f"\n   P5 — PCA:")
    pca_res = test_pca(dm_sub)
    p5_pass = pca_res['PC1_var'] > 0.50 if not np.isnan(pca_res['PC1_var']) else False
    print(f"      PC1 varianza = {pca_res['PC1_var']*100:.1f}% "
          f"{'✅ > 50%' if p5_pass else '❌ ≤ 50%'}")
    
    # PCA su returns
    pca_ret = test_pca(ret_sub)
    print(f"      PC1 returns  = {pca_ret['PC1_var']*100:.1f}%")
    
    # --- P6: Persistenza φ_HIGH > φ_LOW ---
    print(f"\n   P6 — Persistenza per regime:")
    n_phi_higher = 0
    for name in STRATEGY_NAMES:
        m_series = m_sub[name].dropna()
        phi_res = test_persistence_by_regime(m_series, regime)
        phi_h = phi_res.get('phi_HIGH', np.nan)
        phi_l = phi_res.get('phi_LOW', np.nan)
        higher = phi_h > phi_l if not np.isnan(phi_h) and not np.isnan(phi_l) else False
        n_phi_higher += int(higher)
        print(f"      {STRATEGY_LABELS[name]:20s}: φ_LOW={phi_l:.3f}, φ_HIGH={phi_h:.3f} "
              f"{'✅' if higher else '❌'}")
    
    p6_pass = n_phi_higher >= 2
    
    # --- Scorecard ---
    score = sum([p1_pass, p2_pass, p3_pass, p4_pass, p5_pass, p6_pass])
    
    print(f"\n   {'='*40}")
    print(f"   SCORECARD: {score}/6 predizioni Duffie confermate")
    print(f"   P1 Correlazione:  {'✅' if p1_pass else '❌'}")
    print(f"   P2 Co-widening:   {'✅' if p2_pass else '❌'}")
    print(f"   P3 Spanning:      {'✅' if p3_pass else '❌'}")
    print(f"   P4 Granger:       {'✅' if p4_pass else '❌'}")
    print(f"   P5 PCA > 50%:     {'✅' if p5_pass else '❌'}")
    print(f"   P6 Persistence:   {'✅' if p6_pass else '❌'}")
    print(f"   {'='*40}")
    
    all_scorecard.append({
        'period': period_name,
        'start': start_dt.strftime('%Y-%m'), 'end': end_dt.strftime('%Y-%m'),
        'T': T_sub,
        'P1_correlation': p1_pass, 'P2_cowidening': p2_pass,
        'P3_spanning': p3_pass, 'P4_granger': p4_pass,
        'P5_pca': p5_pass, 'P6_persistence': p6_pass,
        'score': score,
        'n_sig_corr': n_sig_corr, 'n_excess_cw': n_excess_positive,
        'n_sig_spanning': n_sig_span, 'n_sig_granger': n_sig_granger,
        'PC1_var_dm': pca_res['PC1_var'], 'PC1_var_ret': pca_ret['PC1_var'],
        'avg_R2_spanning': np.mean(avg_r2) if avg_r2 else np.nan,
    })


# ============================================================================
# PAIRWISE LONG-SAMPLE SCORECARD
# ============================================================================

print("\n" + "=" * 72)
print("PAIRWISE LONG-SAMPLE SCORECARD")
print("=" * 72)

for (s1, s2), cfg_pw in PAIRWISE_LONG.items():
    start_dt = pd.Timestamp(cfg_pw["start"])
    label = cfg_pw["label"]
    
    dm1 = dm_all[s1].dropna()
    dm2 = dm_all[s2].dropna()
    common = dm1.index.intersection(dm2.index)
    common = common[common >= start_dt]
    
    if len(common) < 24:
        print(f"   {label}: T={len(common)} — skip")
        continue
    
    dm_pair = pd.DataFrame({s1: dm1[common], s2: dm2[common]}).dropna()
    T_pw = len(dm_pair)
    
    print(f"\n{'='*60}")
    print(f"📊 {label}: T={T_pw}")
    print(f"{'='*60}")
    
    # P1
    res_corr = test_correlation(dm_pair[s1], dm_pair[s2])
    p1 = res_corr['p_pearson'] < 0.05 if not np.isnan(res_corr['p_pearson']) else False
    print(f"   P1 Correlazione: ρ={res_corr['rho_pearson']:+.3f}, p={res_corr['p_pearson']:.4f} "
          f"{'✅' if p1 else '❌'}")
    
    # P2
    res_cw = test_cowidening(dm_pair[s1], dm_pair[s2])
    p2 = res_cw['excess'] > 0 and res_cw['fisher_p'] < 0.10
    print(f"   P2 Co-widening: excess={res_cw['excess']:+.3f}, p={res_cw['fisher_p']:.4f} "
          f"{'✅' if p2 else '❌'}")
    
    # P3
    res_span = test_spanning(dm_pair[s1], dm_pair[s2])
    p3 = res_span['p_beta0'] < 0.05 if not np.isnan(res_span['p_beta0']) else False
    print(f"   P3 Spanning: R²={res_span['R2']:.4f}, β₀ p={res_span['p_beta0']:.4f} "
          f"{'✅' if p3 else '❌'}")
    
    # P4 Granger
    dm_var_pw = dm_pair.rename(columns=STRATEGY_LABELS)
    gr_res = test_granger(dm_var_pw, list(dm_var_pw.columns), max_lag=3)
    n_gr = sum(1 for g in gr_res if g['p_value'] < 0.10)
    p4 = n_gr >= 1
    for gr in gr_res:
        print(f"   P4 Granger {gr['causing']} → {gr['caused']}: "
              f"F={gr['F_stat']:.3f}, p={gr['p_value']:.4f}")
    print(f"   P4 overall: {'✅' if p4 else '❌'}")
    
    # P5 PCA (bivariate: PC1 should explain >50%)
    pca_pw = test_pca(dm_pair)
    p5 = pca_pw['PC1_var'] > 0.50
    print(f"   P5 PCA: PC1={pca_pw['PC1_var']*100:.1f}% {'✅' if p5 else '❌'}")
    
    score_pw = sum([p1, p2, p3, p4, p5])
    print(f"\n   SCORECARD (pairwise): {score_pw}/5")
    
    all_scorecard.append({
        'period': label, 'start': start_dt.strftime('%Y-%m'),
        'end': common.max().strftime('%Y-%m'), 'T': T_pw,
        'P1_correlation': p1, 'P2_cowidening': p2,
        'P3_spanning': p3, 'P4_granger': p4,
        'P5_pca': p5, 'P6_persistence': np.nan,
        'score': score_pw,
        'n_sig_corr': int(p1), 'n_excess_cw': int(p2),
        'n_sig_spanning': int(p3), 'n_sig_granger': n_gr,
        'PC1_var_dm': pca_pw['PC1_var'], 'PC1_var_ret': np.nan,
        'avg_R2_spanning': res_span['R2'],
    })


# ============================================================================
# SAVE & PLOT
# ============================================================================

df_scorecard = pd.DataFrame(all_scorecard)
df_scorecard.to_csv(RQ3_TABLES_DIR / "T6_duffie_scorecard.csv", index=False)
export_scorecard_tex(df_scorecard, RQ3_TABLES_DIR / "RQ3_duffie_scorecard_slide.tex")

print(f"\n\n{'='*72}")
print("RIEPILOGO DUFFIE SCORECARD")
print("=" * 72)
print(df_scorecard[['period', 'T', 'score', 'P1_correlation', 'P2_cowidening',
                     'P3_spanning', 'P4_granger', 'P5_pca', 'P6_persistence']].to_string(index=False))

# Heatmap-style plot
fig, ax = plt.subplots(figsize=(14, max(4, len(df_scorecard) * 0.9)))
fig.suptitle("Duffie (2010) Prediction Scorecard by Sub-Period",
             fontsize=14, fontweight='bold')

pred_cols = ['P1_correlation', 'P2_cowidening', 'P3_spanning',
             'P4_granger', 'P5_pca', 'P6_persistence']
pred_labels = ['P1\nCorrelation', 'P2\nCo-widening', 'P3\nSpanning',
               'P4\nGranger', 'P5\nPCA>50%', 'P6\nPersistence']

data_matrix = df_scorecard[pred_cols].fillna(-1).values.astype(float)

# Color: green=True, red=False, grey=NA
colors = np.zeros((*data_matrix.shape, 3))
for i in range(data_matrix.shape[0]):
    for j in range(data_matrix.shape[1]):
        if data_matrix[i, j] < 0:   # NA
            colors[i, j] = [0.85, 0.85, 0.85]
        elif data_matrix[i, j] > 0:  # True
            colors[i, j] = [0.2, 0.7, 0.2]
        else:                         # False
            colors[i, j] = [0.8, 0.2, 0.2]

ax.imshow(colors, aspect='auto')

# Labels
ax.set_xticks(range(len(pred_labels)))
ax.set_xticklabels(pred_labels, fontsize=9)
ax.set_yticks(range(len(df_scorecard)))
period_labels = [f"{row['period'].replace(' (long sample)', '')}\n(T={row['T']})"
                 for _, row in df_scorecard.iterrows()]
ax.set_yticklabels(period_labels, fontsize=9)

# Score text
for i, row in df_scorecard.iterrows():
    max_score = 5 if pd.isna(row.get('P6_persistence')) else 6
    ax.text(len(pred_cols) + 0.3, i, f"{row['score']:.0f}/{max_score}",
            fontsize=11, fontweight='bold', va='center')
    for j, col in enumerate(pred_cols):
        val = row[col]
        symbol = "✓" if val == True else "✗" if val == False else "—"
        ax.text(j, i, symbol, ha='center', va='center', fontsize=14,
                fontweight='bold', color='white')

ax.set_xlim(-0.5, len(pred_cols) + 0.5)

fig.tight_layout()
fig.savefig(RQ3_FIGURES_DIR / f"G1_duffie_scorecard.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches='tight')
print(f"\n📊 G1_duffie_scorecard.{FIGURE_FORMAT}")
plt.close()

print(f"💾 T6_duffie_scorecard.csv")

print("\n" + "=" * 72)
print("✅ FILE 6 COMPLETATO")
print("=" * 72)
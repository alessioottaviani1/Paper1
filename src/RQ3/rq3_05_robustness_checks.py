"""
================================================================================
rq3_05_robustness.py — Robustness Checks
================================================================================
RQ3: Slow-moving capital (Duffie 2010)

Questo file esegue:
  A. DCC-GARCH bivariate (3 coppie) — correlazione condizionale dinamica
  B. Quantile Regression: Δm_i su Δm_j a diversi quantili
  C. Threshold Clustering: sincronia dei segnali di entry
  D. Alternative Stress Proxies: ripete test chiave con VIX, V2X, ITRX_XOVER
  E. Summary table con tutti i risultati di robustezza

Prerequisiti: eseguire prima rq3_01 ... rq3_04

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
from scipy import stats
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_00_config import *

print("=" * 72)
print("RQ3 — FILE 5: ROBUSTNESS CHECKS")
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

df_stress = pd.read_csv(RQ3_DATA_DIR / "stress_proxy_monthly.csv",
                         index_col=0, parse_dates=True)

df_dm_clean = df_dm.dropna()

# Load full pairwise data
full_data_path = RQ3_DATA_DIR / "rq3_full_data.pkl"
if full_data_path.exists():
    _full = pd.read_pickle(full_data_path)
    df_dm_full = _full['delta_m_all']
    print(f"   ✅ Full pairwise data loaded")
else:
    df_dm_full = df_dm
    print(f"   ⚠️ Full data not found, using overlap")

T = len(df_dm_clean)
print(f"   T={T} mesi (trivariate), {len(STRATEGY_PAIRS)} coppie")

plt.style.use(FIGURE_STYLE)


# ============================================================================
# SECTION A: DCC-GARCH BIVARIATE
# ============================================================================

print("\n" + "=" * 72)
print("SECTION A: DCC-GARCH Bivariate (3 coppie)")
print("=" * 72)

# Con T≈130 mensili, DCC trivariate è fragile.
# Usiamo bivariate (3 coppie) come robustezza.
# Proviamo arch package; se non disponibile, skip con avviso.

dcc_available = True
try:
    from arch import arch_model
    from arch.univariate import GARCH
except ImportError:
    print("   ⚠️ Package 'arch' non installato. Installa con: pip install arch")
    print("      Sezione DCC-GARCH saltata.")
    dcc_available = False

dcc_results = []

if dcc_available:
    for s1, s2 in STRATEGY_PAIRS:
        pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
        print(f"\n--- {pair_label} ---")
        
        # Dati bivariali — pairwise dropna (not trivariate)
        dm_pair = df_dm_full[[s1, s2]].dropna()
        y1 = dm_pair[s1].values
        y2 = dm_pair[s2].values
        dates = dm_pair.index
        
        try:
            # Step 1: GARCH(1,1) marginali per ottenere residui standardizzati
            am1 = arch_model(y1, vol='Garch', p=DCC_GARCH_ORDER[0],
                            q=DCC_GARCH_ORDER[1], mean='Constant', rescale=True)
            res1 = am1.fit(disp='off')
            
            am2 = arch_model(y2, vol='Garch', p=DCC_GARCH_ORDER[0],
                            q=DCC_GARCH_ORDER[1], mean='Constant', rescale=True)
            res2 = am2.fit(disp='off')
            
            # Residui standardizzati
            std_resid1 = res1.std_resid
            std_resid2 = res2.std_resid
            
            # Step 2: DCC estimation (semplificata — Engle 2002)
            # Q_t = (1-a-b)*Qbar + a*(e_{t-1} e_{t-1}') + b*Q_{t-1}
            # Usiamo grid search su a, b ∈ [0.01, 0.05, ..., 0.20]
            
            e = np.column_stack([std_resid1, std_resid2])
            Qbar = np.corrcoef(e.T)  # unconditional correlation of std resids
            
            best_ll = -np.inf
            best_a, best_b = 0.05, 0.90
            
            for a_try in [0.01, 0.03, 0.05, 0.07, 0.10, 0.15]:
                for b_try in [0.80, 0.85, 0.90, 0.93, 0.95]:
                    if a_try + b_try >= 1:
                        continue
                    
                    # Simulate Q_t and R_t
                    n = len(e)
                    Q = np.zeros((n, 2, 2))
                    R = np.zeros((n, 2, 2))
                    Q[0] = Qbar.copy()
                    
                    ll = 0
                    for t in range(1, n):
                        outer = np.outer(e[t-1], e[t-1])
                        Q[t] = (1 - a_try - b_try) * Qbar + a_try * outer + b_try * Q[t-1]
                        
                        # Normalizza Q per ottenere R (matrice di correlazione)
                        d = np.sqrt(np.diag(Q[t]))
                        if d[0] > 0 and d[1] > 0:
                            R[t] = Q[t] / np.outer(d, d)
                        else:
                            R[t] = np.eye(2)
                        
                        # Log-likelihood contribution
                        rho_t = R[t, 0, 1]
                        rho_t = np.clip(rho_t, -0.999, 0.999)
                        det = 1 - rho_t**2
                        if det > 0:
                            ll += -0.5 * (np.log(det) + 
                                         (e[t, 0]**2 - 2*rho_t*e[t, 0]*e[t, 1] + e[t, 1]**2) / det)
                    
                    if ll > best_ll:
                        best_ll = ll
                        best_a, best_b = a_try, b_try
            
            # Ricomputa con parametri ottimali
            n = len(e)
            Q_final = np.zeros((n, 2, 2))
            rho_t_series = np.zeros(n)
            Q_final[0] = Qbar.copy()
            rho_t_series[0] = Qbar[0, 1]
            
            for t in range(1, n):
                outer = np.outer(e[t-1], e[t-1])
                Q_final[t] = (1 - best_a - best_b) * Qbar + best_a * outer + best_b * Q_final[t-1]
                d = np.sqrt(np.diag(Q_final[t]))
                if d[0] > 0 and d[1] > 0:
                    rho_t_series[t] = Q_final[t, 0, 1] / (d[0] * d[1])
                else:
                    rho_t_series[t] = 0
            
            rho_t_series = np.clip(rho_t_series, -1, 1)
            
            # Statistiche
            rho_mean = np.mean(rho_t_series)
            rho_std = np.std(rho_t_series)
            
            print(f"   DCC params: a={best_a:.3f}, b={best_b:.3f}")
            print(f"   ρ_t: media={rho_mean:+.3f}, std={rho_std:.3f}, "
                  f"min={np.min(rho_t_series):+.3f}, max={np.max(rho_t_series):+.3f}")
            
            # Correlazione media per regime
            for regime_label in ["LOW", "MEDIUM", "HIGH"]:
                mask = regime.reindex(dates, method='ffill') == regime_label
                mask_vals = mask.values[:n]
                if mask_vals.sum() > 0:
                    rho_regime = np.mean(rho_t_series[mask_vals])
                    print(f"   ρ_t|{regime_label}: {rho_regime:+.3f} (n={mask_vals.sum()})")
            
            dcc_results.append({
                'pair': pair_label, 'a': best_a, 'b': best_b,
                'rho_mean': rho_mean, 'rho_std': rho_std,
                'rho_min': np.min(rho_t_series), 'rho_max': np.max(rho_t_series),
                'T': n,
                'rho_series': pd.Series(rho_t_series, index=dates[:n]),
            })
            
        except Exception as ex:
            print(f"   ⚠️ Errore DCC: {ex}")
    
    # Plot DCC correlazioni dinamiche
    if dcc_results:
        fig, axes = plt.subplots(len(dcc_results), 1,
                                  figsize=(14, 5 * len(dcc_results)), sharex=True)
        if len(dcc_results) == 1:
            axes = [axes]
        
        fig.suptitle("DCC-GARCH — Dynamic Conditional Correlations ($\\Delta m$)",
                     fontsize=14, fontweight='bold')
        
        # Override regime shading: use 100 bps threshold for consistency
        # with Section 4 (Table 7 Panel B) and Ferson-Schadt body tables
        stress_monthly = df_stress['ITRX_MAIN']
        regime_plot = pd.Series("MEDIUM", index=stress_monthly.index)
        regime_plot[stress_monthly < 60.0] = "LOW"
        regime_plot[stress_monthly >= 100.0] = "HIGH"

        for idx, res in enumerate(dcc_results):
            ax = axes[idx]
            rho_s = res['rho_series']
            ax.plot(rho_s.index, rho_s.values, color='black', linewidth=0.8)
            ax.axhline(0, color='grey', linewidth=0.5)
            ax.axhline(res['rho_mean'], color='blue', linewidth=0.8,
                       linestyle='--', alpha=0.6, label=f"mean={res['rho_mean']:+.2f}")
            
            # Shading regimi
            for regime_label, color in REGIME_COLORS.items():
                mask = regime.reindex(rho_s.index, method='ffill') == regime_label
                if mask.any():
                    blocks = mask.astype(int).diff().fillna(0)
                    starts = rho_s.index[blocks == 1]
                    ends = rho_s.index[blocks == -1]
                    if mask.iloc[0]:
                        starts = starts.insert(0, rho_s.index[0])
                    if mask.iloc[-1]:
                        ends = ends.append(pd.DatetimeIndex([rho_s.index[-1]]))
                    for s, e in zip(starts[:len(ends)], ends[:len(starts)]):
                        ax.axvspan(s, e, alpha=REGIME_ALPHAS[regime_label],
                                  color=color, zorder=0)
            
            ax.set_ylabel("$\\rho_t$", fontsize=12)
            ax.set_title(res['pair'], fontsize=12, fontweight='bold')
            rho_min, rho_max = rho_s.min(), rho_s.max()
            margin = 0.15 * (rho_max - rho_min)
            ax.set_ylim(rho_min - margin, rho_max + margin)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Legenda regimi (solo sull'ultimo asse)
        from matplotlib.patches import Patch
        regime_legend = [
            Patch(facecolor=REGIME_COLORS['HIGH'], alpha=REGIME_ALPHAS['HIGH'], label='HIGH (Main $\\geq$ 100 bps)'),
            Patch(facecolor=REGIME_COLORS['MEDIUM'], alpha=REGIME_ALPHAS['MEDIUM'], label='MEDIUM (60–100 bps)'),
            Patch(facecolor=REGIME_COLORS['LOW'], alpha=REGIME_ALPHAS['LOW'], label='LOW ($<$ 60 bps)'),
        ]
        axes[-1].legend(handles=regime_legend, loc='lower left', fontsize=8, framealpha=0.9)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        fig.tight_layout()
        fig.savefig(RQ3_FIGURES_DIR / f"F1_dcc_garch.{FIGURE_FORMAT}",
                    dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"\n📊 F1_dcc_garch.{FIGURE_FORMAT}")
        plt.close()
    
    # Salva tabella (senza la serie)
    dcc_table = [{k: v for k, v in r.items() if k != 'rho_series'} for r in dcc_results]
    pd.DataFrame(dcc_table).to_csv(RQ3_TABLES_DIR / "T5a_dcc_garch.csv", index=False)
    print(f"💾 T5a_dcc_garch.csv")


# ============================================================================
# SECTION B: QUANTILE REGRESSION
# ============================================================================

print("\n" + "=" * 72)
print("SECTION B: Quantile Regression (Δm_i su Δm_j)")
print("=" * 72)

try:
    import statsmodels.formula.api as smf
    quantreg_available = True
except ImportError:
    quantreg_available = False

qreg_results = []

if quantreg_available:
    for s1, s2 in STRATEGY_PAIRS:
        for dep, indep in [(s1, s2), (s2, s1)]:
            dep_label = STRATEGY_LABELS[dep]
            indep_label = STRATEGY_LABELS[indep]
            
            df_qr = pd.DataFrame({
                'y': df_dm_full[dep],
                'x': df_dm_full[indep],
            }).dropna()
            
            if len(df_qr) < 20:
                continue
            
            print(f"\n   {indep_label} → {dep_label}:")
            
            # OLS benchmark
            ols = sm.OLS(df_qr['y'], sm.add_constant(df_qr['x'])).fit(
                cov_type='HAC', cov_kwds={'maxlags': NW_MAX_LAGS}
            )
            print(f"      OLS: β={ols.params[1]:+.4f}, t={ols.tvalues[1]:+.2f}, p={ols.pvalues[1]:.4f}")
            
            for tau in QUANTILE_TAUS:
                try:
                    qr = smf.quantreg('y ~ x', data=df_qr).fit(q=tau)
                    beta_qr = qr.params['x']
                    p_qr = qr.pvalues['x']
                    
                    sig = ""
                    if p_qr < 0.01: sig = "***"
                    elif p_qr < 0.05: sig = "**"
                    elif p_qr < 0.10: sig = "*"
                    
                    print(f"      τ={tau:.2f}: β={beta_qr:+.4f}, p={p_qr:.4f} {sig}")
                    
                    qreg_results.append({
                        'dependent': dep_label, 'independent': indep_label,
                        'tau': tau,
                        'beta': beta_qr, 'p_value': p_qr,
                        'beta_ols': ols.params[1], 'p_ols': ols.pvalues[1],
                        'T': len(df_qr),
                    })
                except Exception as e:
                    print(f"      τ={tau:.2f}: errore — {e}")

    df_qreg = pd.DataFrame(qreg_results)
    df_qreg.to_csv(RQ3_TABLES_DIR / "T5b_quantile_regression.csv", index=False)
    print(f"\n💾 T5b_quantile_regression.csv")
    
    # Plot: coefficienti per quantile
    if len(qreg_results) > 0:
        unique_pairs = df_qreg.groupby(['dependent', 'independent']).ngroups
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()
        
        for idx, ((dep, indep), group) in enumerate(df_qreg.groupby(['dependent', 'independent'])):
            if idx >= len(axes):
                break
            ax = axes[idx]
            
            ax.plot(group['tau'], group['beta'], 'o-', color='black', linewidth=1.2)
            ax.axhline(group['beta_ols'].iloc[0], color='red', linewidth=1.0,
                       linestyle='--', label=f"OLS={group['beta_ols'].iloc[0]:.3f}")
            ax.axhline(0, color='grey', linewidth=0.5)
            
            # CI approssimate: colora significativi
            for _, row in group.iterrows():
                color = 'green' if row['p_value'] < 0.05 else 'grey'
                ax.scatter(row['tau'], row['beta'], color=color, s=40, zorder=5)
            
            ax.set_title(f"{indep} → {dep}", fontsize=9)
            ax.set_xlabel("τ")
            ax.set_ylabel("β(τ)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        # Nascondi assi vuoti
        for idx in range(len(df_qreg.groupby(['dependent', 'independent'])), len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle("Quantile Regression Coefficients (Δm)", fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.savefig(RQ3_FIGURES_DIR / f"F2_quantile_regression.{FIGURE_FORMAT}",
                    dpi=FIGURE_DPI, bbox_inches='tight')
        print(f"📊 F2_quantile_regression.{FIGURE_FORMAT}")
        plt.close()
else:
    print("   ⚠️ statsmodels.formula.api non disponibile, skip")


# ============================================================================
# SECTION C: THRESHOLD CLUSTERING (sincronia segnali entry)
# ============================================================================

print("\n" + "=" * 72)
print("SECTION C: Threshold Clustering — Sincronia Entry Signals")
print("=" * 72)

# Per ogni mese, indica se la mispricing m > soglia di entry
# (cioè se la strategia avrebbe un segnale di entry attivo)

entry_signals = {}

for name in STRATEGY_NAMES:
    thresh = ENTRY_THRESHOLDS[name]["threshold_bps"]
    m_series = df_m[name]
    entry_signals[name] = (m_series > thresh).astype(int)

df_signals = pd.DataFrame(entry_signals)
df_signals_clean = df_signals.dropna()

print(f"   T = {len(df_signals_clean)}")
for name in STRATEGY_NAMES:
    pct = df_signals_clean[name].mean() * 100
    print(f"   {STRATEGY_LABELS[name]}: segnale attivo {pct:.1f}% dei mesi")

# Test di sincronia per coppie
cluster_results = []

for s1, s2 in STRATEGY_PAIRS:
    pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
    
    sig1 = df_signals_clean[s1]
    sig2 = df_signals_clean[s2]
    
    # Probabilità
    p1 = sig1.mean()
    p2 = sig2.mean()
    p_joint = ((sig1 == 1) & (sig2 == 1)).mean()
    p_indep = p1 * p2
    excess = p_joint - p_indep
    
    # Fisher exact
    a = ((sig1 == 1) & (sig2 == 1)).sum()
    b = ((sig1 == 1) & (sig2 == 0)).sum()
    c = ((sig1 == 0) & (sig2 == 1)).sum()
    d = ((sig1 == 0) & (sig2 == 0)).sum()
    
    _, p_fisher = stats.fisher_exact([[a, b], [c, d]])
    
    print(f"\n   {pair_label}:")
    print(f"      P(both active)={p_joint:.3f}, P_indep={p_indep:.3f}, "
          f"excess={excess:+.3f}, Fisher p={p_fisher:.4f}")
    
    # Per regime
    for regime_label in ["LOW", "HIGH"]:
        mask = regime.reindex(df_signals_clean.index, method='ffill') == regime_label
        if mask.sum() < 5:
            continue
        
        sig1_r = sig1[mask]
        sig2_r = sig2[mask]
        p_joint_r = ((sig1_r == 1) & (sig2_r == 1)).mean()
        p_indep_r = sig1_r.mean() * sig2_r.mean()
        excess_r = p_joint_r - p_indep_r
        
        print(f"      {regime_label}: P(both)={p_joint_r:.3f}, "
              f"excess={excess_r:+.3f}, n={mask.sum()}")
    
    cluster_results.append({
        'pair': pair_label,
        'P_active_1': p1, 'P_active_2': p2,
        'P_joint': p_joint, 'P_indep': p_indep,
        'excess_prob': excess, 'fisher_p': p_fisher, 'T': len(df_signals_clean),
    })

df_cluster = pd.DataFrame(cluster_results)
df_cluster.to_csv(RQ3_TABLES_DIR / "T5c_threshold_clustering.csv", index=False)
print(f"\n💾 T5c_threshold_clustering.csv")


# ============================================================================
# SECTION D: ALTERNATIVE STRESS PROXIES
# ============================================================================

print("\n" + "=" * 72)
print("SECTION D: Robustezza — Alternative Stress Proxies")
print("=" * 72)

# Ripete i test chiave (co-widening, interaction, regime correlation)
# per tutte le stress proxy e entrambe le regime modes

alt_proxy_results = []

for proxy_name in ALL_STRESS_PROXIES:
    for mode in ["manual", "percentile"]:
        
        # Carica regime
        regime_path = RQ3_DATA_DIR / f"regime_{proxy_name}_{mode}.csv"
        if not regime_path.exists():
            continue
        
        regime_alt = pd.read_csv(regime_path, index_col=0, parse_dates=True)["regime"]
        regime_alt = regime_alt.reindex(df_dm_full.index, method='ffill')
        
        n_low = (regime_alt == "LOW").sum()
        n_high = (regime_alt == "HIGH").sum()
        
        if n_low < 5 or n_high < 5:
            continue
        
        # Per ogni coppia: test co-widening HIGH vs LOW
        for s1, s2 in STRATEGY_PAIRS:
            pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
            
            dm1 = df_dm_full[s1]
            dm2 = df_dm_full[s2]
            
            # Co-widening in HIGH
            mask_h = regime_alt == "HIGH"
            common_h = dm1[mask_h].dropna().index.intersection(dm2[mask_h].dropna().index)
            if len(common_h) < 3:
                continue
            cw_high = ((dm1[common_h] > 0) & (dm2[common_h] > 0)).mean()
            
            # Co-widening in LOW
            mask_l = regime_alt == "LOW"
            common_l = dm1[mask_l].dropna().index.intersection(dm2[mask_l].dropna().index)
            if len(common_l) < 3:
                continue
            cw_low = ((dm1[common_l] > 0) & (dm2[common_l] > 0)).mean()
            
            diff = cw_high - cw_low
            
            # Correlazione in HIGH vs LOW
            common_h2 = df_returns[s1].reindex(common_h).dropna().index.intersection(
                df_returns[s2].reindex(common_h).dropna().index)
            common_l2 = df_returns[s1].reindex(common_l).dropna().index.intersection(
                df_returns[s2].reindex(common_l).dropna().index)
            
            rho_high = df_returns[s1][common_h2].corr(df_returns[s2][common_h2]) if len(common_h2) > 3 else np.nan
            rho_low = df_returns[s1][common_l2].corr(df_returns[s2][common_l2]) if len(common_l2) > 3 else np.nan
            
            alt_proxy_results.append({
                'proxy': proxy_name, 'mode': mode, 'pair': pair_label,
                'n_LOW': n_low, 'n_HIGH': n_high,
                'P_cowiden_HIGH': cw_high, 'P_cowiden_LOW': cw_low,
                'diff_cowiden': diff,
                'rho_returns_HIGH': rho_high, 'rho_returns_LOW': rho_low,
                'diff_rho': rho_high - rho_low if not np.isnan(rho_high) and not np.isnan(rho_low) else np.nan,
            })

df_alt_proxy = pd.DataFrame(alt_proxy_results)
df_alt_proxy.to_csv(RQ3_TABLES_DIR / "T5d_alternative_proxies.csv", index=False)
if len(df_alt_proxy) > 0:
    export_cowidening_proxy_tex(df_alt_proxy,
                                 RQ3_TABLES_DIR / "RQ3_cowidening_proxies_slide.tex")

# Stampa riepilogo
print(f"\n   Combinazioni testate: {len(df_alt_proxy)}")
if len(df_alt_proxy) > 0:
    # Conta quante hanno diff_cowiden > 0 (predizione Duffie: più co-widening in HIGH)
    n_positive = (df_alt_proxy['diff_cowiden'] > 0).sum()
    n_total = len(df_alt_proxy)
    print(f"   P(co-widen|HIGH) > P(co-widen|LOW): {n_positive}/{n_total} "
          f"({n_positive/n_total*100:.0f}%)")
    
    # Riepilogo per proxy
    for proxy_name in ALL_STRESS_PROXIES:
        sub = df_alt_proxy[df_alt_proxy['proxy'] == proxy_name]
        if len(sub) > 0:
            avg_diff = sub['diff_cowiden'].mean()
            avg_diff_rho = sub['diff_rho'].mean()
            print(f"   {proxy_name}: Δ co-widen medio={avg_diff:+.3f}, "
                  f"Δ ρ medio={avg_diff_rho:+.3f}")
    
    # ===== DETTAGLIO COPPIA × PROXY × MODE =====
    print(f"\n   {'─'*80}")
    print(f"   DETTAGLIO: Co-widening e Correlazioni per COPPIA × PROXY × MODE")
    print(f"   {'─'*80}")
    
    for proxy_name in ALL_STRESS_PROXIES:
        sub_proxy = df_alt_proxy[df_alt_proxy['proxy'] == proxy_name]
        if len(sub_proxy) == 0:
            continue
        
        for mode in sub_proxy['mode'].unique():
            sub = sub_proxy[sub_proxy['mode'] == mode]
            if len(sub) == 0:
                continue
            
            n_h = sub.iloc[0]['n_HIGH']
            n_l = sub.iloc[0]['n_LOW']
            print(f"\n   📊 {proxy_name} ({mode}) — HIGH: n={n_h}, LOW: n={n_l}")
            
            for _, row in sub.iterrows():
                pair = row['pair']
                cw_h = row['P_cowiden_HIGH']
                cw_l = row['P_cowiden_LOW']
                diff_cw = row['diff_cowiden']
                rho_h = row['rho_returns_HIGH']
                rho_l = row['rho_returns_LOW']
                diff_rho = row['diff_rho']
                
                # Segno per interpretazione rapida
                cw_sign = "⬆" if diff_cw > 0.02 else ("⬇" if diff_cw < -0.02 else "→")
                rho_sign = "⬆" if diff_rho > 0.05 else ("⬇" if diff_rho < -0.05 else "→")
                
                print(f"      {pair:45s}  "
                      f"CW: {cw_l:.3f}→{cw_h:.3f} ({diff_cw:+.3f}) {cw_sign}  "
                      f"ρ: {rho_l:+.3f}→{rho_h:+.3f} ({diff_rho:+.3f}) {rho_sign}"
                      if not np.isnan(diff_rho) else
                      f"      {pair:45s}  "
                      f"CW: {cw_l:.3f}→{cw_h:.3f} ({diff_cw:+.3f}) {cw_sign}  "
                      f"ρ: n/a")
    
    print(f"\n   {'─'*80}")

print(f"\n💾 T5d_alternative_proxies.csv")


# ============================================================================
# SECTION E: SUB-PERIOD ROBUSTNESS
# ============================================================================

print("\n" + "=" * 72)
print("SECTION E: Sub-Period Robustness")
print("=" * 72)

subperiod_results = []

for period_name, (start, end) in SUB_PERIODS.items():
    start_dt, end_dt = pd.Timestamp(start), pd.Timestamp(end)
    
    mask_period = (df_dm_full.index >= start_dt) & (df_dm_full.index <= end_dt)
    dm_sub = df_dm_full[mask_period]
    ret_sub = df_returns.loc[(df_returns.index >= start_dt) & (df_returns.index <= end_dt)]
    
    if len(dm_sub) < 12:
        print(f"\n   {period_name}: T={len(dm_sub)} — troppo pochi, skip")
        continue
    
    print(f"\n   --- {period_name} ({start[:7]} → {end[:7]}): T={len(dm_sub)} ---")
    
    for s1, s2 in STRATEGY_PAIRS:
        pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
        
        # Correlazione Δm
        dm1 = dm_sub[s1].dropna()
        dm2 = dm_sub[s2].dropna()
        common = dm1.index.intersection(dm2.index)
        
        if len(common) < 5:
            continue
        
        rho_dm = dm1[common].corr(dm2[common])
        
        # Co-widening
        p_cowiden = ((dm1[common] > 0) & (dm2[common] > 0)).mean()
        p_indep = (dm1[common] > 0).mean() * (dm2[common] > 0).mean()
        excess = p_cowiden - p_indep
        
        print(f"      {pair_label}: ρ(Δm)={rho_dm:+.3f}, "
              f"co-widen={p_cowiden:.3f}, excess={excess:+.3f}")
        
        subperiod_results.append({
            'period': period_name, 'pair': pair_label,
            'T': len(common), 'rho_dm': rho_dm,
            'P_cowiden': p_cowiden, 'P_indep': p_indep,
            'excess': excess,
        })

df_subperiod = pd.DataFrame(subperiod_results)
df_subperiod.to_csv(RQ3_TABLES_DIR / "T5e_subperiod_robustness.csv", index=False)
print(f"\n💾 T5e_subperiod_robustness.csv")


# ============================================================================
# SECTION F: PAIRWISE LONG-SAMPLE ANALYSIS
# ============================================================================

print("\n" + "=" * 72)
print("SECTION F: Pairwise Long-Sample Analysis")
print("=" * 72)

# Carica dati completi (non solo overlap trivariate)
full_data_path = RQ3_DATA_DIR / "rq3_full_data.pkl"
pairwise_results = []

if full_data_path.exists():
    full_data = pd.read_pickle(full_data_path)
    ret_all = full_data['returns_all']
    dm_all = full_data['delta_m_all']
    
    for (s1, s2), cfg_pw in PAIRWISE_LONG.items():
        start_dt = pd.Timestamp(cfg_pw["start"])
        label = cfg_pw["label"]
        
        # Δm overlap bivariate
        dm1 = dm_all[s1].dropna()
        dm2 = dm_all[s2].dropna()
        common = dm1.index.intersection(dm2.index)
        common = common[common >= start_dt]
        
        if len(common) < 24:
            print(f"   {label}: T={len(common)} — troppo pochi, skip")
            continue
        
        dm1_sub = dm1[common]
        dm2_sub = dm2[common]
        
        T_long = len(common)
        T_trivariate = len(df_dm_clean)
        
        print(f"\n   {label}:")
        print(f"   T (long) = {T_long} vs T (trivariate) = {T_trivariate}")
        
        # Correlazione
        rho_long = dm1_sub.corr(dm2_sub)
        rho_short = df_dm_clean[s1].corr(df_dm_clean[s2])
        
        # Co-widening
        cw_long = ((dm1_sub > 0) & (dm2_sub > 0)).mean()
        cw_short = ((df_dm_clean[s1] > 0) & (df_dm_clean[s2] > 0)).mean()
        
        print(f"   ρ(Δm) long: {rho_long:+.3f} vs short: {rho_short:+.3f}")
        print(f"   Co-widen long: {cw_long:.3f} vs short: {cw_short:.3f}")
        
        # Granger causality sul campione lungo
        from statsmodels.tsa.api import VAR as VAR_long
        df_var_long = pd.DataFrame({
            STRATEGY_LABELS[s1]: dm1_sub,
            STRATEGY_LABELS[s2]: dm2_sub,
        }).dropna()
        
        if len(df_var_long) > 20:
            try:
                model_long = VAR_long(df_var_long)
                result_long = model_long.fit(maxlags=2, ic='bic')
                
                for caused_l, causing_l in [(STRATEGY_LABELS[s1], STRATEGY_LABELS[s2]),
                                             (STRATEGY_LABELS[s2], STRATEGY_LABELS[s1])]:
                    test_l = result_long.test_causality(caused_l, [causing_l], kind='f')
                    print(f"   Granger {causing_l} → {caused_l}: "
                          f"F={test_l.test_statistic:.3f}, p={test_l.pvalue:.4f}")
            except Exception as e:
                print(f"   ⚠️ VAR errore: {e}")
        
        pairwise_results.append({
            'pair': label, 'T_long': T_long, 'T_trivariate': T_trivariate,
            'rho_dm_long': rho_long, 'rho_dm_trivariate': rho_short,
            'cowiden_long': cw_long, 'cowiden_trivariate': cw_short,
        })

    df_pairwise = pd.DataFrame(pairwise_results)
    df_pairwise.to_csv(RQ3_TABLES_DIR / "T5f_pairwise_long_sample.csv", index=False)
    print(f"\n💾 T5f_pairwise_long_sample.csv")
else:
    print("   ⚠️ rq3_full_data.pkl non trovato — rieseguire File 1")


# ============================================================================
# SECTION G: SUMMARY TABLE — Tutte le robustezze
# ============================================================================

print("\n" + "=" * 72)
print("SECTION G: Summary Robustness")
print("=" * 72)

# Costruisci tabella riassuntiva dei risultati chiave per coppia
summary_rows = []

for s1, s2 in STRATEGY_PAIRS:
    pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
    
    row = {'pair': pair_label}
    
    # 1. Correlazione unconditional (da File 2)
    t2a_path = RQ3_TABLES_DIR / "T2a_unconditional_correlations.csv"
    if t2a_path.exists():
        t2a = pd.read_csv(t2a_path)
        match = t2a[(t2a['pair'] == pair_label) & (t2a['series'] == 'Δm')]
        if len(match) > 0:
            row['pearson_Dm'] = match.iloc[0]['pearson_r']
            row['pearson_p_Dm'] = match.iloc[0]['pearson_p_hac']
    
    # 2. Correlazione purged (da File 2)
    t2h_path = RQ3_TABLES_DIR / "T2h_purged_correlations.csv"
    if t2h_path.exists():
        t2h = pd.read_csv(t2h_path)
        match = t2h[t2h['pair'] == pair_label]
        if len(match) > 0:
            row['pearson_purged'] = match.iloc[0]['pearson_r']
            row['pearson_p_purged'] = match.iloc[0]['pearson_p_hac']
    
    # 3. Co-widening excess (da File 2)
    t2e_path = RQ3_TABLES_DIR / "T2e_cowidening.csv"
    if t2e_path.exists():
        t2e = pd.read_csv(t2e_path)
        match_h = t2e[(t2e['pair'] == pair_label) & (t2e['regime'] == 'HIGH')]
        match_l = t2e[(t2e['pair'] == pair_label) & (t2e['regime'] == 'LOW')]
        if len(match_h) > 0:
            row['cowiden_excess_HIGH'] = match_h.iloc[0]['excess_prob']
        if len(match_l) > 0:
            row['cowiden_excess_LOW'] = match_l.iloc[0]['excess_prob']
    
    # 4. DCC mean rho (da questo file)
    dcc_match = [r for r in dcc_results if r['pair'] == pair_label]
    if dcc_match:
        row['dcc_rho_mean'] = dcc_match[0]['rho_mean']
    
    # 5. Quantile regression: β a τ=0.90 (da questo file)
    if len(qreg_results) > 0:
        qr_match = df_qreg[
            (df_qreg['dependent'] == STRATEGY_LABELS[s1]) &
            (df_qreg['independent'] == STRATEGY_LABELS[s2]) &
            (df_qreg['tau'] == 0.90)
        ]
        if len(qr_match) > 0:
            row['qreg_beta_90'] = qr_match.iloc[0]['beta']
            row['qreg_p_90'] = qr_match.iloc[0]['p_value']
    
    # 6. Threshold clustering
    cl_match = df_cluster[df_cluster['pair'] == pair_label]
    if len(cl_match) > 0:
        row['cluster_excess'] = cl_match.iloc[0]['excess_prob']
        row['cluster_fisher_p'] = cl_match.iloc[0]['fisher_p']
    
    # 7. Robustezza proxy: % che confermano
    if len(df_alt_proxy) > 0:
        proxy_match = df_alt_proxy[df_alt_proxy['pair'] == pair_label]
        if len(proxy_match) > 0:
            row['pct_proxies_confirming'] = (proxy_match['diff_cowiden'] > 0).mean() * 100
    
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(RQ3_TABLES_DIR / "T5g_robustness_summary.csv", index=False)

print(f"\n   Righe: {len(df_summary)}")
print(df_summary.round(4).to_string(index=False))
print(f"\n💾 T5e_robustness_summary.csv")


# ============================================================================
# DONE
# ============================================================================

print("\n" + "=" * 72)
print("✅ FILE 5 COMPLETATO — ANALISI RQ3 COMPLETA")
print("=" * 72)
print(f"\n   Tabelle: {len(list(RQ3_TABLES_DIR.glob('T5*')))} file")
print(f"   Figure:  {len(list(RQ3_FIGURES_DIR.glob('F*')))} file")
print(f"\n   Tutti i risultati salvati in: {RQ3_OUTPUT_DIR}")
print(f"\n   File prodotti nell'analisi RQ3:")
print(f"   ├── figures/  (A1..A4, B1..B2, C1, D1..D2, E1..E5, F1..F2)")
print(f"   ├── tables/   (T1..T5)")
print(f"   └── data/     (intermediate)")
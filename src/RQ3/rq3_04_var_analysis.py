"""
================================================================================
rq3_04_var_analysis.py — VAR, Granger Causality, IRF, Threshold VAR
================================================================================
RQ3: Slow-moving capital (Duffie 2010)

Questo file esegue:
  A. VAR su Δm: selezione lag (BIC), stima
  B. Granger Causality tests
  C. GIRF — Generalized Impulse Response Functions (Pesaran–Shin 1998)
  D. Tutte le 6 permutazioni Cholesky (orthogonalized IRF)
  E. Forecast Error Variance Decomposition (FEVD)
  F. Threshold VAR: VAR separati in HIGH vs LOW, confronto IRF
  G. VAR Augmented (VARX): proxy di stress come variabile esogena

Prerequisiti: eseguire prima rq3_01, rq3_02, rq3_03

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
from itertools import permutations
from statsmodels.tsa.api import VAR
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_00_config import *

print("=" * 72)
print("RQ3 — FILE 4: VAR ANALYSIS")
print("=" * 72)


# ============================================================================
# HELPER: Compute true GIRF for any VAR result
# ============================================================================

def compute_girf(var_result, periods):
    """
    Compute Generalized IRF (Pesaran–Shin 1998) for a VAR result.
    GIRF(h, j) = σ_jj^{-1/2} × Ψ_h × Σ × e_j
    Returns: array (periods+1, K, K)
    """
    K = var_result.neqs
    ma_coefs = var_result.ma_rep(maxn=periods)
    Sigma = np.array(var_result.sigma_u)
    
    girf_vals = np.zeros((periods + 1, K, K))
    for j in range(K):
        e_j = np.zeros(K)
        e_j[j] = 1.0
        sigma_jj = Sigma[j, j]
        if sigma_jj > 0:
            for h in range(periods + 1):
                girf_vals[h, :, j] = (1.0 / np.sqrt(sigma_jj)) * ma_coefs[h] @ Sigma @ e_j
    return girf_vals

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

# Stress proxy mensile
df_stress = pd.read_csv(RQ3_DATA_DIR / "stress_proxy_monthly.csv",
                         index_col=0, parse_dates=True)

df_dm_clean = df_dm.dropna()
T = len(df_dm_clean)
print(f"   Δm: T={T} mesi, variabili: {list(df_dm_clean.columns)}")

# Labels corti per VAR
var_labels = {name: STRATEGY_LABELS[name] for name in STRATEGY_NAMES}

plt.style.use(FIGURE_STYLE)


# ============================================================================
# SECTION A: VAR — Selezione Lag e Stima
# ============================================================================

print("\n" + "=" * 72)
print("SECTION A: VAR — Selezione Lag e Stima")
print("=" * 72)

# Rinomina colonne per leggibilità nei risultati
df_var = df_dm_clean.rename(columns=STRATEGY_LABELS)

model_var = VAR(df_var)

# Selezione lag order
print("\n   Selezione lag order (max={})...".format(VAR_MAX_LAGS))
lag_selection = model_var.select_order(maxlags=VAR_MAX_LAGS)
print(lag_selection.summary())

# Usa BIC per parsimonia (campione piccolo)
optimal_lag = lag_selection.hqic
# Almeno 1 lag
if optimal_lag == 0:
    optimal_lag = 1
print(f"\n   ✅ Lag ottimale (BIC): {optimal_lag}")

# Stima VAR
var_result = model_var.fit(maxlags=optimal_lag, ic=None)
print(f"\n   VAR({optimal_lag}) stimato:")
print(f"   T effettivo = {var_result.nobs}")
print(f"   K = {var_result.neqs} variabili endogene")
print(f"   Parametri per equazione = {var_result.k_ar * var_result.neqs + 1}")

# Salva summary
with open(RQ3_TABLES_DIR / "T4a_var_summary.txt", 'w') as f:
    f.write(str(var_result.summary()))
print(f"\n💾 T4a_var_summary.txt")

# --- VAR Residual Diagnostics ---
print(f"\n--- VAR Residual Diagnostics ---")

# Stabilità
is_stable = var_result.is_stable()
print(f"   Stabile (radici < 1): {'✅ Sì' if is_stable else '⚠️ No'}")

# Portmanteau (autocorrelazione residui)
try:
    whiteness = var_result.test_whiteness(nlags=12)
    print(f"   Portmanteau (12 lags): statistic={whiteness.test_statistic:.2f}, "
          f"p={whiteness.pvalue:.4f} {'✅' if whiteness.pvalue > 0.05 else '⚠️'}")
except Exception as e:
    print(f"   Portmanteau: errore — {e}")

# Normalità (Jarque-Bera)
try:
    normality = var_result.test_normality()
    print(f"   Jarque-Bera: statistic={normality.test_statistic:.2f}, "
          f"p={normality.pvalue:.4f} {'✅' if normality.pvalue > 0.05 else '⚠️'}")
except Exception as e:
    print(f"   Normalità: errore — {e}")


# ============================================================================
# SECTION B: GRANGER CAUSALITY
# ============================================================================

print("\n" + "=" * 72)
print("SECTION B: Granger Causality Tests")
print("=" * 72)

granger_results = []

var_col_names = list(df_var.columns)

# B1: Granger al lag ottimale BIC
print(f"\n--- Lag ottimale (BIC = {optimal_lag}) ---")
for caused in var_col_names:
    for causing in var_col_names:
        if caused == causing:
            continue
        
        try:
            test = var_result.test_causality(caused, [causing], kind='f')
            f_stat = test.test_statistic
            p_val = test.pvalue
            
            sig = ""
            if p_val < 0.01: sig = "***"
            elif p_val < 0.05: sig = "**"
            elif p_val < 0.10: sig = "*"
            
            print(f"   {causing:20s} → {caused:20s}: "
                  f"F={f_stat:.3f}, p={p_val:.4f} {sig}")
            
            granger_results.append({
                'causing': causing, 'caused': caused,
                'F_stat': f_stat, 'p_value': p_val,
                'significant_5pct': p_val < 0.05,
                'lags': optimal_lag,
            })
        except Exception as e:
            print(f"   {causing} → {caused}: errore — {e}")

df_granger = pd.DataFrame(granger_results)
df_granger.to_csv(RQ3_TABLES_DIR / "T4b_granger_causality.csv", index=False)
export_granger_tex(df_granger, RQ3_TABLES_DIR / "RQ3_granger_causality_slide.tex")
print(f"\n💾 T4b_granger_causality.csv")

# B2: Granger a lag multipli (robustezza)
print(f"\n--- Granger Causality a lag multipli ---")
granger_multi_results = []

for test_lag in [1, 2, 3]:
    if test_lag > len(df_var) // 5:
        continue
    try:
        model_lag = VAR(df_var).fit(maxlags=test_lag, ic=None)
        for caused in var_col_names:
            for causing in var_col_names:
                if caused == causing:
                    continue
                try:
                    test = model_lag.test_causality(caused, [causing], kind='f')
                    granger_multi_results.append({
                        'lag': test_lag, 'causing': causing, 'caused': caused,
                        'F_stat': test.test_statistic, 'p_value': test.pvalue,
                    })
                except:
                    pass
    except:
        pass

if granger_multi_results:
    df_granger_multi = pd.DataFrame(granger_multi_results)
    df_granger_multi.to_csv(RQ3_TABLES_DIR / "T4b2_granger_multilag.csv", index=False)
    
    # Print summary
    for (causing, caused), group in df_granger_multi.groupby(['causing', 'caused']):
        pvals = ", ".join([f"L{r['lag']}:p={r['p_value']:.3f}" for _, r in group.iterrows()])
        print(f"   {causing:20s} → {caused:20s}: {pvals}")
    print(f"💾 T4b2_granger_multilag.csv")


# ============================================================================
# SECTION C: GIRF — Generalized IRF (Pesaran–Shin 1998)
# ============================================================================

print("\n" + "=" * 72)
print("SECTION C: Generalized Impulse Response Functions (GIRF)")
print("=" * 72)

# True GIRF (Pesaran–Shin 1998):
#   GIRF(h, δ_j, Ω_{t-1}) = σ_jj^{-1/2} × Ψ_h × Σ × e_j
# dove Ψ_h = MA(h) coefficients, Σ = cov(residui), e_j = selezione vettore j

K = var_result.neqs
H = VAR_IRF_PERIODS

# MA representation
ma_coefs = var_result.ma_rep(maxn=H)    # shape: (H+1, K, K)

# Covariance matrix of residuals (convert to numpy if DataFrame)
Sigma = np.array(var_result.sigma_u)

# Compute true GIRF
girf_values = np.zeros((H + 1, K, K))    # (horizon, response_i, impulse_j)

for j in range(K):
    e_j = np.zeros(K)
    e_j[j] = 1.0
    sigma_jj = Sigma[j, j]
    
    for h in range(H + 1):
        girf_values[h, :, j] = (1.0 / np.sqrt(sigma_jj)) * ma_coefs[h] @ Sigma @ e_j

print(f"   GIRF calcolata: {K} variabili, {H} orizzonti")

# --- Bootstrap confidence bands (residual bootstrap) ---
print(f"   Bootstrap confidence bands ({VAR_BOOTSTRAP_REPS} reps)...")

resids = np.array(var_result.resid)            # (T_eff, K)
fitted = np.array(var_result.fittedvalues)     # (T_eff, K)
Y_orig = df_var.values                         # (T, K)
p = optimal_lag

girf_boot = np.zeros((VAR_BOOTSTRAP_REPS, H + 1, K, K))

for b in range(VAR_BOOTSTRAP_REPS):
    # Resample residui con replacement
    boot_idx = np.random.choice(len(resids), size=len(resids), replace=True)
    boot_resid = resids[boot_idx]
    
    # Ricostruisci serie bootstrappata
    T_eff = len(resids)
    Y_boot = np.zeros((T_eff + p, K))
    Y_boot[:p] = Y_orig[:p]  # condizioni iniziali
    
    # Ricostruisci usando coefficienti stimati
    coefs = np.array(var_result.coefs)               # (p, K, K)
    intercept = np.array(var_result.coefs_exog[:, 0]) if var_result.coefs_exog is not None else np.zeros(K)
    
    for t in range(p, T_eff + p):
        Y_boot[t] = intercept.copy()
        for lag in range(p):
            Y_boot[t] += coefs[lag] @ Y_boot[t - lag - 1]
        Y_boot[t] += boot_resid[t - p]
    
    # Stima VAR sulla serie bootstrappata
    try:
        df_boot = pd.DataFrame(Y_boot, columns=var_col_names)
        model_boot = VAR(df_boot)
        result_boot = model_boot.fit(maxlags=p, ic=None, trend='c')
        
        ma_boot = result_boot.ma_rep(maxn=H)
        Sigma_boot = np.array(result_boot.sigma_u)
        
        for j in range(K):
            e_j = np.zeros(K)
            e_j[j] = 1.0
            sigma_jj_b = Sigma_boot[j, j]
            if sigma_jj_b > 0:
                for h in range(H + 1):
                    girf_boot[b, h, :, j] = (1.0 / np.sqrt(sigma_jj_b)) * ma_boot[h] @ Sigma_boot @ e_j
    except:
        girf_boot[b] = girf_values  # fallback: copia il punto-stima

alpha = 1 - VAR_IRF_CI
girf_lower = np.percentile(girf_boot, 100 * alpha / 2, axis=0)
girf_upper = np.percentile(girf_boot, 100 * (1 - alpha / 2), axis=0)

print(f"   ✅ Bootstrap completato")

# --- Plot GIRF con CI ---
fig, axes = plt.subplots(K, K, figsize=(5 * K, 4 * K))
fig.suptitle(f"Generalized IRF (Pesaran–Shin 1998) — VAR({optimal_lag}) on Δm\n"
             f"{int(VAR_IRF_CI*100)}% bootstrap CI ({VAR_BOOTSTRAP_REPS} reps)",
             fontsize=14, fontweight='bold')

for i, response in enumerate(var_col_names):
    for j, impulse in enumerate(var_col_names):
        ax = axes[i][j]
        periods = np.arange(H + 1)
        
        ax.plot(periods, girf_values[:, i, j], color='black', linewidth=1.2)
        ax.fill_between(periods, girf_lower[:, i, j], girf_upper[:, i, j],
                        color='blue', alpha=0.15)
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.set_title(f"{impulse} → {response}", fontsize=9)
        ax.grid(True, alpha=0.3)
        
        if i == K - 1:
            ax.set_xlabel("Months")
        if j == 0:
            ax.set_ylabel("Response")

fig.tight_layout()
fig.savefig(RQ3_FIGURES_DIR / f"E1_girf.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches='tight')
print(f"📊 E1_girf.{FIGURE_FORMAT}")
plt.close()

# Salva IRF values
irf_data = []
for h in range(H + 1):
    for i, response in enumerate(var_col_names):
        for j, impulse in enumerate(var_col_names):
            irf_data.append({
                'horizon': h, 'impulse': impulse, 'response': response,
                'irf_value': girf_values[h, i, j],
                'ci_lower': girf_lower[h, i, j],
                'ci_upper': girf_upper[h, i, j],
            })

df_irf = pd.DataFrame(irf_data)
df_irf.to_csv(RQ3_TABLES_DIR / "T4c_girf_values.csv", index=False)
print(f"💾 T4c_girf_values.csv")

# Keep girf_values accessible for Section D comparison
class _GIRFHolder:
    def __init__(self, irfs):
        self.irfs = irfs
girf = _GIRFHolder(girf_values)


# ============================================================================
# SECTION D: TUTTE LE 6 PERMUTAZIONI CHOLESKY
# ============================================================================

print("\n" + "=" * 72)
print("SECTION D: Orthogonalized IRF — 6 Permutazioni Cholesky")
print("=" * 72)

all_perms = list(permutations(range(len(var_col_names))))
print(f"   {len(all_perms)} permutazioni per {len(var_col_names)} variabili")

# Per ogni permutazione, stima IRF ortogonalizzate
# e salva i risultati per confronto
cholesky_results = {}

fig_chol, axes_chol = plt.subplots(
    len(var_col_names), len(var_col_names),
    figsize=(5 * len(var_col_names), 4 * len(var_col_names))
)
fig_chol.suptitle(f"Orthogonalized IRF — All {len(all_perms)} Cholesky Orderings",
                  fontsize=14, fontweight='bold')

colors_perm = plt.cm.Set2(np.linspace(0, 1, len(all_perms)))

for perm_idx, perm in enumerate(all_perms):
    # Riordina le variabili secondo la permutazione
    reordered_cols = [var_col_names[p] for p in perm]
    df_var_reordered = df_var[reordered_cols]
    
    # Stima VAR con questo ordinamento
    model_perm = VAR(df_var_reordered)
    result_perm = model_perm.fit(maxlags=optimal_lag, ic=None)
    irf_perm = result_perm.irf(periods=VAR_IRF_PERIODS)
    
    perm_label = "→".join([c[:5] for c in reordered_cols])
    cholesky_results[perm_label] = irf_perm
    
    # Plot ogni combinazione impulse-response
    for i_orig, response_orig in enumerate(var_col_names):
        for j_orig, impulse_orig in enumerate(var_col_names):
            ax = axes_chol[i_orig][j_orig]
            
            # Trova indici nella permutazione corrente
            try:
                i_perm = reordered_cols.index(response_orig)
                j_perm = reordered_cols.index(impulse_orig)
            except ValueError:
                continue
            
            irf_vals = irf_perm.orth_irfs[:, i_perm, j_perm]
            periods = np.arange(len(irf_vals))
            
            ax.plot(periods, irf_vals, color=colors_perm[perm_idx],
                    linewidth=0.7, alpha=0.6)

# Aggiungi GIRF come riferimento (linea nera spessa)
for i, response in enumerate(var_col_names):
    for j, impulse in enumerate(var_col_names):
        ax = axes_chol[i][j]
        irf_vals = girf.irfs[:, i, j]
        ax.plot(np.arange(len(irf_vals)), irf_vals,
                color='black', linewidth=2.0, label='GIRF' if i == 0 and j == 0 else '')
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.set_title(f"{impulse} → {response}", fontsize=9)
        ax.grid(True, alpha=0.3)

# Legenda
axes_chol[0][0].legend(fontsize=8, loc='upper right')

fig_chol.tight_layout()
fig_chol.savefig(RQ3_FIGURES_DIR / f"E2_cholesky_permutations.{FIGURE_FORMAT}",
                 dpi=FIGURE_DPI, bbox_inches='tight')
print(f"📊 E2_cholesky_permutations.{FIGURE_FORMAT}")
plt.close()


# ============================================================================
# SECTION E: FORECAST ERROR VARIANCE DECOMPOSITION
# ============================================================================

print("\n" + "=" * 72)
print("SECTION E: Forecast Error Variance Decomposition (FEVD)")
print("=" * 72)

fevd = var_result.fevd(periods=VAR_IRF_PERIODS)
print(fevd.summary())

# Salva FEVD
fevd_data = []
for h in range(VAR_IRF_PERIODS):
    for i, response in enumerate(var_col_names):
        for j, shock in enumerate(var_col_names):
            fevd_data.append({
                'horizon': h + 1, 'response': response, 'shock': shock,
                'variance_share': fevd.decomp[i][h, j],
            })

df_fevd = pd.DataFrame(fevd_data)
df_fevd.to_csv(RQ3_TABLES_DIR / "T4e_fevd.csv", index=False)

# Plot FEVD stacked bar
fig, axes = plt.subplots(1, len(var_col_names),
                          figsize=(5 * len(var_col_names), 5))
fig.suptitle(f"Forecast Error Variance Decomposition — VAR({optimal_lag})",
             fontsize=13, fontweight='bold')

bar_colors = list(STRATEGY_COLORS.values())

for i, response in enumerate(var_col_names):
    ax = axes[i]
    horizons = np.arange(1, VAR_IRF_PERIODS + 1)
    
    bottom = np.zeros(VAR_IRF_PERIODS)
    for j, shock in enumerate(var_col_names):
        vals = fevd.decomp[i][:, j]
        ax.bar(horizons, vals, bottom=bottom, color=bar_colors[j],
               label=shock if i == 0 else '', alpha=0.8)
        bottom += vals
    
    ax.set_title(f"Response: {response}", fontsize=10)
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Variance Share" if i == 0 else "")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

axes[0].legend(fontsize=8, loc='center right')

fig.tight_layout()
fig.savefig(RQ3_FIGURES_DIR / f"E3_fevd.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches='tight')
print(f"📊 E3_fevd.{FIGURE_FORMAT}")
print(f"💾 T4e_fevd.csv")
plt.close()


# ============================================================================
# SECTION F: THRESHOLD VAR (HIGH vs LOW regime)
# ============================================================================

print("\n" + "=" * 72)
print("SECTION F: Threshold VAR — HIGH vs LOW Regime")
print("=" * 72)
regime_dm = master['regime'].reindex(df_dm_clean.index, method='ffill')
print(f"   Regime on Δm: {regime_dm.value_counts().to_dict()}")
print(f"   df_dm_clean dates: {df_dm_clean.index[:3].tolist()} ... {df_dm_clean.index[-3:].tolist()}")
print(f"   master regime dates: {master['regime'].index[:3].tolist()} ... {master['regime'].index[-3:].tolist()}")

# Se il regime default ha troppo pochi obs in HIGH, prova con percentile
MIN_OBS_VAR_REGIME = 15

mask_high_default = regime_dm == "HIGH"
if mask_high_default.sum() < MIN_OBS_VAR_REGIME:
    print(f"   ⚠️ Regime default ({DEFAULT_STRESS_PROXY}, {DEFAULT_REGIME_MODE}) ha solo "
          f"{mask_high_default.sum()} obs in HIGH — provo con percentile...")
    
    # Carica regime percentile
    regime_pct_path = RQ3_DATA_DIR / f"regime_{DEFAULT_STRESS_PROXY}_percentile.csv"
    if regime_pct_path.exists():
        regime_dm_alt = pd.read_csv(regime_pct_path, index_col=0, parse_dates=True)["regime"]
        regime_dm = regime_dm_alt.reindex(df_dm_clean.index, method='ffill')
        mask_high_alt = regime_dm == "HIGH"
        print(f"   Percentile regime: HIGH={mask_high_alt.sum()}, LOW={(regime_dm == 'LOW').sum()}")
    
    # Se ancora troppo pochi, usa split mediana (2 regimi: sopra/sotto mediana)
    if (regime_dm == "HIGH").sum() < MIN_OBS_VAR_REGIME:
        print(f"   ⚠️ Ancora troppo pochi — fallback a split mediana della stress proxy")
        stress_level = df_stress[DEFAULT_STRESS_PROXY].reindex(df_dm_clean.index, method='ffill')
        median_val = stress_level.median()
        regime_dm = pd.Series("LOW", index=df_dm_clean.index)
        regime_dm[stress_level >= median_val] = "HIGH"
        print(f"   Mediana split: HIGH={(regime_dm == 'HIGH').sum()}, "
              f"LOW={(regime_dm == 'LOW').sum()}")

threshold_irf = {}

for regime_label in ["LOW", "HIGH"]:
    mask = regime_dm == regime_label
    df_regime_sub = df_var[mask]
    n_obs = len(df_regime_sub)
    
    print(f"\n   Regime {regime_label}: T={n_obs}")
    
    if n_obs < MIN_OBS_VAR_REGIME:
        print(f"   ⚠️ Troppo pochi dati per VAR (min={MIN_OBS_VAR_REGIME}), skip")
        continue
    
    # Determina lag (massimo 2 per campione piccolo)
    max_lag_regime = min(2, VAR_MAX_LAGS, n_obs // 10)
    if max_lag_regime < 1:
        max_lag_regime = 1
    
    try:
        model_regime = VAR(df_regime_sub)
        
        # Selezione lag
        try:
            lag_sel = model_regime.select_order(maxlags=max_lag_regime)
            opt_lag = max(1, lag_sel.bic)
        except:
            opt_lag = 1
        
        result_regime = model_regime.fit(maxlags=opt_lag, ic=None)
        girf_regime = compute_girf(result_regime, VAR_IRF_PERIODS)
        
        threshold_irf[regime_label] = {
            'girf': girf_regime,
            'result': result_regime,
            'lag': opt_lag,
            'T': result_regime.nobs,
        }
        
        print(f"   VAR({opt_lag}), T_eff={result_regime.nobs}")
        
    except Exception as e:
        print(f"   ⚠️ Errore: {e}")

# Plot confronto IRF: HIGH vs LOW
if len(threshold_irf) == 2:
    fig, axes = plt.subplots(len(var_col_names), len(var_col_names),
                              figsize=(5 * len(var_col_names), 4 * len(var_col_names)))
    fig.suptitle("Threshold VAR — GIRF: HIGH (red) vs LOW (green)",
                 fontsize=14, fontweight='bold')
    
    for i, response in enumerate(var_col_names):
        for j, impulse in enumerate(var_col_names):
            ax = axes[i][j]
            
            for regime_label, color in [("LOW", REGIME_COLORS["LOW"]),
                                         ("HIGH", REGIME_COLORS["HIGH"])]:
                if regime_label in threshold_irf:
                    irf_vals = threshold_irf[regime_label]['girf'][:, i, j]
                    periods = np.arange(len(irf_vals))
                    ax.plot(periods, irf_vals, color=color,
                            linewidth=1.5, label=regime_label)
            
            ax.axhline(0, color='grey', linewidth=0.5)
            ax.set_title(f"{impulse} → {response}", fontsize=9)
            ax.grid(True, alpha=0.3)
            
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    
    fig.tight_layout()
    fig.savefig(RQ3_FIGURES_DIR / f"E4_threshold_var_irf.{FIGURE_FORMAT}",
                dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"\n📊 E4_threshold_var_irf.{FIGURE_FORMAT}")
    plt.close()

    # Salva IRF numeriche per confronto
    threshold_irf_data = []
    for regime_label in ["LOW", "HIGH"]:
        if regime_label not in threshold_irf:
            continue
        girf_obj = threshold_irf[regime_label]['girf']
        for h in range(VAR_IRF_PERIODS + 1):
            for i, response in enumerate(var_col_names):
                for j, impulse in enumerate(var_col_names):
                    threshold_irf_data.append({
                        'regime': regime_label, 'horizon': h,
                        'impulse': impulse, 'response': response,
                        'irf_value': girf_obj[h, i, j],
                    })
    
    df_threshold_irf = pd.DataFrame(threshold_irf_data)
    df_threshold_irf.to_csv(RQ3_TABLES_DIR / "T4f_threshold_var_irf.csv", index=False)
    print(f"💾 T4f_threshold_var_irf.csv")
else:
    print("   ⚠️ Non abbastanza regimi per confronto, skip plot.")


# ============================================================================
# SECTION G: VAR AUGMENTED (VARX) — Stress come variabile esogena
# ============================================================================

print("\n" + "=" * 72)
print("SECTION G: VAR Augmented — Proxy di stress come esogena")
print("=" * 72)

# Allinea stress proxy
stress_var = df_stress[DEFAULT_STRESS_PROXY].reindex(df_dm_clean.index, method='ffill')
# Prima differenza dello stress per stazionarietà del livello
stress_diff = stress_var.diff().dropna()

# Allinea tutti
common_idx = df_var.index.intersection(stress_diff.index)
df_var_x = df_var.loc[common_idx]
exog = stress_diff.loc[common_idx].values.reshape(-1, 1)

print(f"   Proxy esogena: Δ{DEFAULT_STRESS_PROXY}")
print(f"   T = {len(common_idx)}")

try:
    model_varx = VAR(df_var_x, exog=exog)
    
    # Usa stesso lag del VAR base
    result_varx = model_varx.fit(maxlags=optimal_lag, ic=None)
    
    print(f"\n   VARX({optimal_lag}) stimato:")
    print(f"   T_eff = {result_varx.nobs}")
    
    # Salva summary
    with open(RQ3_TABLES_DIR / "T4g_varx_summary.txt", 'w') as f:
        f.write(str(result_varx.summary()))
    print(f"💾 T4g_varx_summary.txt")
    
    # Granger causality nel VARX
    print(f"\n   Granger Causality (VARX):")
    granger_varx_results = []
    
    for caused in var_col_names:
        for causing in var_col_names:
            if caused == causing:
                continue
            try:
                test = result_varx.test_causality(caused, [causing], kind='f')
                f_stat = test.test_statistic
                p_val = test.pvalue
                
                sig = ""
                if p_val < 0.01: sig = "***"
                elif p_val < 0.05: sig = "**"
                elif p_val < 0.10: sig = "*"
                
                print(f"   {causing:20s} → {caused:20s}: "
                      f"F={f_stat:.3f}, p={p_val:.4f} {sig}")
                
                granger_varx_results.append({
                    'causing': causing, 'caused': caused,
                    'F_stat': f_stat, 'p_value': p_val,
                    'exog': f"Δ{DEFAULT_STRESS_PROXY}",
                })
            except Exception as e:
                print(f"   {causing} → {caused}: errore — {e}")
    
    df_granger_varx = pd.DataFrame(granger_varx_results)
    df_granger_varx.to_csv(RQ3_TABLES_DIR / "T4g_granger_varx.csv", index=False)
    print(f"💾 T4g_granger_varx.csv")

    # GIRF del VARX (coerente con VAR base)
    girf_varx = compute_girf(result_varx, VAR_IRF_PERIODS)
    
    # Plot confronto GIRF: VAR vs VARX
    fig, axes = plt.subplots(len(var_col_names), len(var_col_names),
                              figsize=(5 * len(var_col_names), 4 * len(var_col_names)))
    fig.suptitle(f"GIRF Comparison: VAR({optimal_lag}) vs VARX({optimal_lag}, exog=Δ{DEFAULT_STRESS_PROXY})",
                 fontsize=13, fontweight='bold')
    
    for i, response in enumerate(var_col_names):
        for j, impulse in enumerate(var_col_names):
            ax = axes[i][j]
            
            # VAR base (GIRF)
            irf_base = girf.irfs[:, i, j]
            ax.plot(np.arange(len(irf_base)), irf_base,
                    color='black', linewidth=1.5, label='VAR (GIRF)')
            
            # VARX (GIRF)
            irf_x = girf_varx[:, i, j]
            ax.plot(np.arange(len(irf_x)), irf_x,
                    color='blue', linewidth=1.5, linestyle='--', label='VARX (GIRF)')
            
            ax.axhline(0, color='grey', linewidth=0.5)
            ax.set_title(f"{impulse} → {response}", fontsize=9)
            ax.grid(True, alpha=0.3)
            
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    
    fig.tight_layout()
    fig.savefig(RQ3_FIGURES_DIR / f"E5_var_vs_varx_irf.{FIGURE_FORMAT}",
                dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"📊 E5_var_vs_varx_irf.{FIGURE_FORMAT}")
    plt.close()

except Exception as e:
    print(f"   ⚠️ Errore nella stima VARX: {e}")
    print("   Il VARX potrebbe non convergere con questo campione.")


# ============================================================================
# MULTIPLE TESTING CORRECTION — Granger p-values
# ============================================================================

print("\n" + "=" * 72)
print("Multiple Testing Correction — Granger causality")
print("=" * 72)

from statsmodels.stats.multitest import multipletests

# Collect all Granger p-values
granger_pvals = {}
if granger_results:
    for r in granger_results:
        key = f"granger_{r['causing']}_{r['caused']}"
        granger_pvals[key] = r['p_value']

if granger_multi_results:
    for r in granger_multi_results:
        key = f"granger_L{r['lag']}_{r['causing']}_{r['caused']}"
        granger_pvals[key] = r['p_value']

if granger_pvals:
    labels = list(granger_pvals.keys())
    pvals = np.array(list(granger_pvals.values()))
    
    # BH-FDR
    reject_bh, pvals_bh, _, _ = multipletests(pvals, alpha=MTC_ALPHA, method='fdr_bh')
    # Bonferroni-Holm
    reject_holm, pvals_holm, _, _ = multipletests(pvals, alpha=MTC_ALPHA, method='holm')
    
    print(f"   {len(pvals)} test totali")
    print(f"   BH-FDR:    {reject_bh.sum()} significativi (α={MTC_ALPHA})")
    print(f"   Holm-Bonf: {reject_holm.sum()} significativi (α={MTC_ALPHA})")
    
    mtc_df = pd.DataFrame({
        'test': labels, 'p_raw': pvals,
        'p_BH': pvals_bh, 'reject_BH': reject_bh,
        'p_Holm': pvals_holm, 'reject_Holm': reject_holm,
    })
    mtc_df.to_csv(RQ3_TABLES_DIR / "T4_granger_MTC.csv", index=False)
    print(f"💾 T4_granger_MTC.csv")


# ============================================================================
# DONE
# ============================================================================

print("\n" + "=" * 72)
print("✅ FILE 4 COMPLETATO")
print("=" * 72)
print(f"\n   Tabelle: {len(list(RQ3_TABLES_DIR.glob('T4*')))} file")
print(f"   Figure:  {len(list(RQ3_FIGURES_DIR.glob('E*')))} file")
print(f"\n   Prossimo step → rq3_05_robustness.py")
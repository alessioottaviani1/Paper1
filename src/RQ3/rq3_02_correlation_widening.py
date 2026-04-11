"""
================================================================================
rq3_02_correlation_cowidening.py — Correlazioni, Co-Widening, Forbes–Rigobon
================================================================================
RQ3: Slow-moving capital (Duffie 2010)

Questo file esegue:
  A. Correlazioni unconditional (HAC) su returns, Δm, e livelli m
  B. Rolling correlations (returns + livelli m)
  C. Correlazioni per regime (LOW/MED/HIGH) + Forbes–Rigobon correction
  D. Persistence analysis: φ per regime, half-life
  E. Co-widening test su Δm (Fisher/χ², per regime)
  F. Tail co-exceedance test
  G. Difference-in-co-widening (block bootstrap HIGH vs LOW)
  H. Versione "purged" (residui da fattori comuni): correlazioni + co-widening
  I. Multiple testing correction (Benjamini–Hochberg FDR)

Prerequisiti: eseguire prima rq3_01_mispricing_construction.py

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
from itertools import combinations
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq3_00_config import *

print("=" * 72)
print("RQ3 — FILE 2: CORRELATION & CO-WIDENING ANALYSIS")
print("=" * 72)

# ============================================================================
# LOAD DATA FROM FILE 1
# ============================================================================

print("\n📂 Caricamento dati da File 1...")

master = pd.read_pickle(RQ3_DATA_DIR / "rq3_master_data.pkl")
df_returns   = master['returns']
df_m         = master['mispricing']
df_dm        = master['delta_m']
regime_default = master['regime']
regime_2l = master.get('regime_2l', None)

# Carica anche i regimi per tutte le proxy
regimes_all = {}
for proxy_name in ALL_STRESS_PROXIES:
    regimes_all[proxy_name] = {}
    for mode in ["manual", "percentile"]:
        path = RQ3_DATA_DIR / f"regime_{proxy_name}_{mode}.csv"
        if path.exists():
            s = pd.read_csv(path, index_col=0, parse_dates=True)["regime"]
            regimes_all[proxy_name][mode] = s

# Fattori per purging
df_purge = pd.read_csv(RQ3_DATA_DIR / "purge_factors_monthly.csv",
                        index_col=0, parse_dates=True)

# Use full regime (covers 2004–2025), no ffill needed
regime = regime_default
# Verify alignment with returns
n_missing = regime.reindex(df_returns.index).isna().sum()
if n_missing > 0:
    print(f"   ⚠️ {n_missing} return months without regime — check date alignment!")
    # Diagnostic: show which dates are missing
    missing_dates = df_returns.index[regime.reindex(df_returns.index).isna()]
    print(f"   Missing dates: {missing_dates.tolist()}")

# Drop NaN nel Δm (primo mese perso per diff)
df_dm_clean = df_dm.dropna()  # trivariate overlap

# Load full pairwise data
full_data_path = RQ3_DATA_DIR / "rq3_full_data.pkl"
if full_data_path.exists():
    _full = pd.read_pickle(full_data_path)
    df_dm_full = _full['delta_m_all']
else:
    df_dm_full = df_dm

T = len(df_returns)
T_dm = len(df_dm_clean)
print(f"   Returns: T={T} mesi")
print(f"   Δm trivariate: T={T_dm} mesi")
print(f"   Regime:  {regime.value_counts().to_dict()}")

# Collector per p-values (per MTC finale)
all_pvalues = {}

plt.style.use(FIGURE_STYLE)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def hac_correlation_test(x, y, n_lags=NW_MAX_LAGS):
    """
    Testa ρ=0 via regressione OLS con errori Newey-West.
    Ritorna: (pearson_r, beta_ols, t_stat_hac, p_value_hac).
    """
    mask = x.notna() & y.notna()
    x_c, y_c = x[mask].values, y[mask].values
    pearson_r = np.corrcoef(x_c, y_c)[0, 1]
    
    X = sm.add_constant(x_c)
    model = sm.OLS(y_c, X).fit(cov_type='HAC', cov_kwds={'maxlags': n_lags})
    beta = model.params[1]
    t_stat = model.tvalues[1]
    p_val = model.pvalues[1]
    
    return pearson_r, beta, t_stat, p_val


def spearman_with_hac(x, y, n_lags=NW_MAX_LAGS):
    """Spearman via ranks + HAC."""
    mask = x.notna() & y.notna()
    rx = stats.rankdata(x[mask].values)
    ry = stats.rankdata(y[mask].values)
    rx_s = pd.Series(rx)
    ry_s = pd.Series(ry)
    rho_s = rx_s.corr(ry_s)
    
    X = sm.add_constant(rx)
    model = sm.OLS(ry, X).fit(cov_type='HAC', cov_kwds={'maxlags': n_lags})
    p_val = model.pvalues[1]
    
    return rho_s, model.tvalues[1], p_val


def forbes_rigobon_correction(rho_high, sigma_high, sigma_low):
    """
    Correzione Forbes–Rigobon (2002) per bias da volatilità.
    rho_FR = rho_HIGH / sqrt(1 + delta * (1 - rho_HIGH^2))
    dove delta = (sigma_HIGH^2 / sigma_LOW^2) - 1
    """
    if sigma_low == 0:
        return rho_high
    delta = (sigma_high ** 2 / sigma_low ** 2) - 1
    denominator = np.sqrt(1 + delta * (1 - rho_high ** 2))
    if denominator == 0:
        return rho_high
    return rho_high / denominator


def forbes_rigobon_symmetric(rho_high, sigma_x_high, sigma_x_low,
                              sigma_y_high, sigma_y_low):
    """
    Forbes–Rigobon simmetrica: fa la correzione due volte (una con σ_x, una
    con σ_y) e ritorna la media. Così il risultato non dipende dall'ordine
    della coppia.
    
    Returns: (rho_fr_x, rho_fr_y, rho_fr_avg)
    """
    rho_fr_x = forbes_rigobon_correction(rho_high, sigma_x_high, sigma_x_low)
    rho_fr_y = forbes_rigobon_correction(rho_high, sigma_y_high, sigma_y_low)
    rho_fr_avg = (rho_fr_x + rho_fr_y) / 2.0
    return rho_fr_x, rho_fr_y, rho_fr_avg


# --- Bootstrap: circular block + permutation ---

MIN_OBS_REGIME_TEST = 12   # Minimo osservazioni per test formale HIGH vs LOW

def circular_block_bootstrap_indices(T, block_length, rng):
    """
    Circular block bootstrap (Politis & Romano 1992).
    Draws ceil(T/L) blocks of length L from circular starting points.
    """
    block_length = int(max(1, block_length))
    n_blocks = int(np.ceil(T / block_length))
    starts = rng.integers(0, T, size=n_blocks)
    indices = []
    for s in starts:
        block = [(s + j) % T for j in range(block_length)]
        indices.extend(block)
    return np.array(indices[:T])


def stationary_bootstrap_indices(T, block_length, rng):
    """
    Stationary bootstrap (Politis & Romano 1994).
    E[block length] = block_length, Geometric(1/block_length).
    """
    p = 1.0 / max(1, block_length)
    indices = np.empty(T, dtype=np.intp)
    indices[0] = rng.integers(0, T)
    for t in range(1, T):
        if rng.random() < p:
            indices[t] = rng.integers(0, T)
        else:
            indices[t] = (indices[t - 1] + 1) % T
    return indices


def circular_bootstrap_diff_cowidening(dm1_full, dm2_full, regime_full,
                                        n_reps=BOOTSTRAP_N_REPS,
                                        block_size=BOOTSTRAP_BLOCK_SIZE):
    """
    Circular block bootstrap sull'intera serie cronologica per testare
    H0: P(co-widen|HIGH) = P(co-widen|LOW).
    
    Per ogni replica:
    1. Bootstrap indici sulla serie completa (preserva struttura temporale)
    2. Applica gli STESSI indici sia a cowiden sia a regime (mantiene allineamento)
    3. Ricrea le maschere HIGH/LOW sul regime_boot
    4. Ricalcola la statistica diff
    
    Returns: (obs_diff, p_circular, ci_lo, ci_hi)
    """
    T = len(dm1_full)
    rng = np.random.default_rng(42)
    
    cowiden = ((dm1_full > 0) & (dm2_full > 0)).astype(float).values
    regime_vals = regime_full.values if hasattr(regime_full, 'values') else np.array(regime_full)
    
    mask_h = regime_vals == "HIGH"
    mask_l = regime_vals == "LOW"
    
    if mask_h.sum() < 3 or mask_l.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan
    
    obs_diff = cowiden[mask_h].mean() - cowiden[mask_l].mean()
    
    boot_diffs = np.zeros(n_reps)
    for b in range(n_reps):
        idx = circular_block_bootstrap_indices(T, block_size, rng)
        cw_boot = cowiden[idx]
        regime_boot = regime_vals[idx]  # STESSO idx → allineamento preservato
        
        mask_h_boot = regime_boot == "HIGH"
        mask_l_boot = regime_boot == "LOW"
        
        if mask_h_boot.sum() > 0 and mask_l_boot.sum() > 0:
            boot_diffs[b] = cw_boot[mask_h_boot].mean() - cw_boot[mask_l_boot].mean()
        else:
            boot_diffs[b] = 0.0
    
    p_value = np.mean(np.abs(boot_diffs) >= np.abs(obs_diff))
    ci_lo = np.percentile(boot_diffs, 2.5)
    ci_hi = np.percentile(boot_diffs, 97.5)
    
    return obs_diff, p_value, ci_lo, ci_hi


def stationary_bootstrap_diff_cowidening(dm1_full, dm2_full, regime_full,
                                          n_reps=BOOTSTRAP_N_REPS,
                                          block_size=BOOTSTRAP_BLOCK_SIZE):
    """
    Stationary bootstrap (Politis & Romano 1994) sull'intera serie.
    Come circular ma con blocchi di lunghezza geometrica casuale.
    Bootstrappa coppie (cowiden, regime) insieme per preservare allineamento.
    """
    T = len(dm1_full)
    rng = np.random.default_rng(42)
    
    cowiden = ((dm1_full > 0) & (dm2_full > 0)).astype(float).values
    regime_vals = regime_full.values if hasattr(regime_full, 'values') else np.array(regime_full)
    
    mask_h = regime_vals == "HIGH"
    mask_l = regime_vals == "LOW"
    
    if mask_h.sum() < 3 or mask_l.sum() < 3:
        return np.nan, np.nan, np.nan, np.nan
    
    obs_diff = cowiden[mask_h].mean() - cowiden[mask_l].mean()
    
    boot_diffs = np.zeros(n_reps)
    for b in range(n_reps):
        idx = stationary_bootstrap_indices(T, block_size, rng)
        cw_boot = cowiden[idx]
        regime_boot = regime_vals[idx]  # STESSO idx
        
        mask_h_boot = regime_boot == "HIGH"
        mask_l_boot = regime_boot == "LOW"
        
        if mask_h_boot.sum() > 0 and mask_l_boot.sum() > 0:
            boot_diffs[b] = cw_boot[mask_h_boot].mean() - cw_boot[mask_l_boot].mean()
        else:
            boot_diffs[b] = 0.0
    
    p_value = np.mean(np.abs(boot_diffs) >= np.abs(obs_diff))
    ci_lo = np.percentile(boot_diffs, 2.5)
    ci_hi = np.percentile(boot_diffs, 97.5)
    
    return obs_diff, p_value, ci_lo, ci_hi


def permutation_test_diff_cowidening(dm1_full, dm2_full, regime_full,
                                      n_reps=BOOTSTRAP_N_REPS):
    """
    Permutation test: permuta le etichette HIGH/LOW (tra HIGH e LOW solo),
    ricalcola la statistica diff.
    H0: etichette HIGH/LOW sono scambiabili.
    
    Returns: (obs_diff, p_perm)
    """
    cowiden = ((dm1_full > 0) & (dm2_full > 0)).astype(float).values
    regime_vals = regime_full.values if hasattr(regime_full, 'values') else np.array(regime_full)
    
    mask_hl = (regime_vals == "HIGH") | (regime_vals == "LOW")
    cw_hl = cowiden[mask_hl]
    labels_hl = regime_vals[mask_hl]
    
    n_high_orig = (labels_hl == "HIGH").sum()
    obs_diff = cw_hl[labels_hl == "HIGH"].mean() - cw_hl[labels_hl == "LOW"].mean()
    
    rng = np.random.default_rng(123)
    count_extreme = 0
    for b in range(n_reps):
        perm = rng.permutation(len(labels_hl))
        labels_perm = labels_hl[perm]
        diff_b = cw_hl[labels_perm == "HIGH"].mean() - cw_hl[labels_perm == "LOW"].mean()
        if abs(diff_b) >= abs(obs_diff):
            count_extreme += 1
    
    p_perm = count_extreme / n_reps
    return obs_diff, p_perm


# ============================================================================
# SECTION A: UNCONDITIONAL CORRELATIONS (HAC)
# ============================================================================

print("\n" + "=" * 72)
print("SECTION A: Unconditional Correlations (HAC)")
print("=" * 72)

corr_results = []

for series_type, df_data, label in [
    ("returns", df_returns, "Returns"),
    ("delta_m", df_dm_full, "Δm"),
    ("levels_m", df_m, "Levels m"),
]:
    print(f"\n--- {label} ---")
    
    for s1, s2 in STRATEGY_PAIRS:
        x, y = df_data[s1], df_data[s2]
        
        # Pearson + HAC
        pr, beta, t_hac, p_hac = hac_correlation_test(x, y)
        
        # Spearman + HAC
        sr, t_sp, p_sp = spearman_with_hac(x, y)
        
        pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
        print(f"   {pair_label}:")
        print(f"      Pearson:  ρ={pr:+.3f}, t_HAC={t_hac:+.2f}, p={p_hac:.4f}")
        print(f"      Spearman: ρ={sr:+.3f}, t_HAC={t_sp:+.2f}, p={p_sp:.4f}")
        
        corr_results.append({
            'series': label, 'pair': pair_label,
            'pearson_r': pr, 'pearson_t_hac': t_hac, 'pearson_p_hac': p_hac,
            'spearman_r': sr, 'spearman_t_hac': t_sp, 'spearman_p_hac': p_sp,
            'N': len(x.dropna()),
        })
        
        # Salva p-values per MTC
        all_pvalues[f"corr_{series_type}_{s1}_{s2}_pearson"] = p_hac
        all_pvalues[f"corr_{series_type}_{s1}_{s2}_spearman"] = p_sp

df_corr = pd.DataFrame(corr_results)
df_corr.to_csv(RQ3_TABLES_DIR / "T2a_unconditional_correlations.csv", index=False)
print(f"\n💾 T2a_unconditional_correlations.csv")


# ============================================================================
# SECTION B: ROLLING CORRELATIONS
# ============================================================================

print("\n" + "=" * 72)
print("SECTION B: Rolling Correlations")
print("=" * 72)

fig_ret, axes_ret = plt.subplots(len(STRATEGY_PAIRS), 1,
                                  figsize=(14, 4 * len(STRATEGY_PAIRS)), sharex=True)
fig_m, axes_m = plt.subplots(len(STRATEGY_PAIRS), 1,
                              figsize=(14, 4 * len(STRATEGY_PAIRS)), sharex=True)

for idx, (s1, s2) in enumerate(STRATEGY_PAIRS):
    pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
    
    for fig, axes, df_data, title_prefix in [
        (fig_ret, axes_ret, df_returns, "Returns"),
        (fig_m, axes_m, df_m, "Levels m"),
    ]:
        ax = axes[idx]
        
        # Rolling Pearson
        rolling_corr = df_data[s1].rolling(
            window=ROLLING_WINDOW_MONTHS, min_periods=ROLLING_MIN_OBS
        ).corr(df_data[s2])
        
        ax.plot(rolling_corr.index, rolling_corr.values,
                color='black', linewidth=1.0)
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')
        
        # Shading regimi
        for regime_label, color in REGIME_COLORS.items():
            regime_aligned = regime.reindex(df_data.index, method='ffill')
            mask = regime_aligned == regime_label
            if mask.any():
                blocks = mask.astype(int).diff().fillna(0)
                starts = df_data.index[blocks == 1]
                ends = df_data.index[blocks == -1]
                if mask.iloc[0]:
                    starts = starts.insert(0, df_data.index[0])
                if mask.iloc[-1]:
                    ends = pd.DatetimeIndex(list(ends) + [df_data.index[-1]])
                for s, e in zip(starts[:len(ends)], ends[:len(starts)]):
                    ax.axvspan(s, e, alpha=REGIME_ALPHAS[regime_label],
                              color=color, zorder=0)
        
        ax.set_ylabel("ρ rolling", fontsize=10)
        ax.set_title(f"{title_prefix}: {pair_label}", fontsize=11)
        ax.set_ylim(-1, 1)
        ax.grid(True, alpha=0.3)

for fig, fname in [(fig_ret, "B1_rolling_corr_returns"), (fig_m, "B2_rolling_corr_levels_m")]:
    fig.tight_layout()
    fig.savefig(RQ3_FIGURES_DIR / f"{fname}.{FIGURE_FORMAT}",
                dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"📊 {fname}.{FIGURE_FORMAT}")
    plt.close(fig)


# ============================================================================
# SECTION C: CORRELATIONS BY REGIME + FORBES–RIGOBON
# ============================================================================

print("\n" + "=" * 72)
print("SECTION C: Correlations by Regime + Forbes–Rigobon Correction")
print("=" * 72)

regime_corr_results = []

for series_type, df_data, label in [
    ("returns", df_returns, "Returns"),
    ("delta_m", df_dm_full, "Δm"),
    ("levels_m", df_m, "Levels m"),
]:
    print(f"\n--- {label} ---")
    
    for s1, s2 in STRATEGY_PAIRS:
        pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
        
        corr_by_regime = {}
        sigma_x_by_regime = {}
        sigma_y_by_regime = {}
        n_common_by_regime = {}
        
        for regime_label in ["LOW", "MEDIUM", "HIGH"]:
            mask = regime.reindex(df_data.index, method='ffill') == regime_label
            x_reg = df_data.loc[mask, s1].dropna()
            y_reg = df_data.loc[mask, s2].dropna()
            common = x_reg.index.intersection(y_reg.index)
            x_reg, y_reg = x_reg[common], y_reg[common]
            
            n_common_by_regime[regime_label] = len(common)
            
            if len(common) >= MIN_OBS_REGIME_TEST:
                rho = x_reg.corr(y_reg)
                sigma_x = x_reg.std()
                sigma_y = y_reg.std()
            else:
                rho = np.nan
                sigma_x = np.nan
                sigma_y = np.nan
                if len(common) >= 5:
                    # Calcola correlazione illustrativa ma segnala
                    rho = x_reg.corr(y_reg)
                    sigma_x = x_reg.std()
                    sigma_y = y_reg.std()
                    print(f"      ⚠️ {regime_label}: n={len(common)} < {MIN_OBS_REGIME_TEST}, "
                          f"risultato illustrativo")
            
            corr_by_regime[regime_label] = rho
            sigma_x_by_regime[regime_label] = sigma_x
            sigma_y_by_regime[regime_label] = sigma_y
        
        # Forbes–Rigobon correction — SIMMETRICA
        # Non calcolare se n troppo piccolo (varianza instabile)
        rho_fr_x, rho_fr_y, rho_fr_avg = np.nan, np.nan, np.nan
        can_fr = (n_common_by_regime.get("HIGH", 0) >= MIN_OBS_REGIME_TEST and
                  n_common_by_regime.get("LOW", 0) >= MIN_OBS_REGIME_TEST and
                  not np.isnan(corr_by_regime.get("HIGH", np.nan)) and
                  not np.isnan(sigma_x_by_regime.get("HIGH", np.nan)) and
                  not np.isnan(sigma_x_by_regime.get("LOW", np.nan)) and
                  sigma_x_by_regime["LOW"] > 0 and sigma_y_by_regime["LOW"] > 0)
        
        if can_fr:
            rho_fr_x, rho_fr_y, rho_fr_avg = forbes_rigobon_symmetric(
                corr_by_regime["HIGH"],
                sigma_x_by_regime["HIGH"], sigma_x_by_regime["LOW"],
                sigma_y_by_regime["HIGH"], sigma_y_by_regime["LOW"],
            )
        
        print(f"   {pair_label}:")
        for rl in ["LOW", "MEDIUM", "HIGH"]:
            print(f"      {rl}: ρ={corr_by_regime.get(rl, np.nan):+.3f} (n={n_common_by_regime[rl]})")
        print(f"      FR-corrected HIGH: via σ(x)={rho_fr_x:+.3f}, "
              f"via σ(y)={rho_fr_y:+.3f}, avg={rho_fr_avg:+.3f}")
        
        # Fisher z-test: HIGH vs LOW (usando n del common sample!)
        n_high = n_common_by_regime["HIGH"]
        n_low = n_common_by_regime["LOW"]
        
        z_high = np.arctanh(np.clip(corr_by_regime.get("HIGH", 0), -0.999, 0.999))
        z_low = np.arctanh(np.clip(corr_by_regime.get("LOW", 0), -0.999, 0.999))
        
        z_diff, p_fisher = np.nan, np.nan
        z_diff_fr, p_fisher_fr = np.nan, np.nan
        
        if n_high > 3 and n_low > 3:
            # Test naive (ρ_HIGH vs ρ_LOW)
            z_diff = (z_high - z_low) / np.sqrt(1/(n_high-3) + 1/(n_low-3))
            p_fisher = 2 * (1 - stats.norm.cdf(abs(z_diff)))
            
            # Test Forbes-Rigobon (ρ_FR_avg vs ρ_LOW)
            if not np.isnan(rho_fr_avg):
                z_fr = np.arctanh(np.clip(rho_fr_avg, -0.999, 0.999))
                z_diff_fr = (z_fr - z_low) / np.sqrt(1/(n_high-3) + 1/(n_low-3))
                p_fisher_fr = 2 * (1 - stats.norm.cdf(abs(z_diff_fr)))
        
        formal = "FORMAL" if n_high >= MIN_OBS_REGIME_TEST and n_low >= MIN_OBS_REGIME_TEST else "illustr."
        print(f"      Fisher z naive HIGH vs LOW:  z={z_diff:+.2f}, p={p_fisher:.4f} [{formal}]")
        print(f"      Fisher z FR-avg vs LOW:      z={z_diff_fr:+.2f}, p={p_fisher_fr:.4f} [{formal}]")
        
        regime_corr_results.append({
            'series': label, 'pair': pair_label,
            'rho_LOW': corr_by_regime.get("LOW"),
            'rho_MED': corr_by_regime.get("MEDIUM"),
            'rho_HIGH': corr_by_regime.get("HIGH"),
            'rho_FR_x': rho_fr_x, 'rho_FR_y': rho_fr_y, 'rho_FR_avg': rho_fr_avg,
            'fisher_z_naive': z_diff, 'fisher_p_naive': p_fisher,
            'fisher_z_FR': z_diff_fr, 'fisher_p_FR': p_fisher_fr,
            'n_LOW': n_low, 'n_MED': n_common_by_regime.get("MEDIUM", 0),
            'n_HIGH': n_high,
            'test_quality': formal,
        })
        
        all_pvalues[f"regime_corr_{series_type}_{s1}_{s2}_fisher"] = p_fisher
        all_pvalues[f"regime_corr_{series_type}_{s1}_{s2}_fisher_FR"] = p_fisher_fr

df_regime_corr = pd.DataFrame(regime_corr_results)
df_regime_corr.to_csv(RQ3_TABLES_DIR / "T2c_regime_correlations.csv", index=False)
print(f"\n💾 T2c_regime_correlations.csv")

# ============================================================================
# SECTION C2: Correlations by Regime — 2-LEVEL (NORMAL/HIGH)
# ============================================================================

if regime_2l is not None:
    print("\n" + "=" * 72)
    print("SECTION C2: Correlations by Regime — 2-LEVEL (NORMAL/HIGH)")
    print("=" * 72)
    
    regime_2l_aligned = regime_2l
    regime_corr_2l_results = []
    
    for series_type, df_data, label in [
        ("returns", df_returns, "Returns"),
        ("delta_m", df_dm_full, "Δm"),
    ]:
        print(f"\n--- {label} ---")
        
        for s1, s2 in STRATEGY_PAIRS:
            pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
            
            corr_by_regime = {}
            n_common_by_regime = {}
            
            for regime_label in ["NORMAL", "HIGH"]:
                mask = regime_2l.reindex(df_data.index, method='ffill') == regime_label
                x_reg = df_data.loc[mask, s1].dropna()
                y_reg = df_data.loc[mask, s2].dropna()
                common = x_reg.index.intersection(y_reg.index)
                x_reg, y_reg = x_reg[common], y_reg[common]
                
                n_common_by_regime[regime_label] = len(common)
                
                if len(common) >= 5:
                    rho = x_reg.corr(y_reg)
                else:
                    rho = np.nan
                
                corr_by_regime[regime_label] = rho
            
            print(f"   {pair_label}:")
            for rl in ["NORMAL", "HIGH"]:
                print(f"      {rl}: ρ={corr_by_regime.get(rl, np.nan):+.3f} "
                      f"(n={n_common_by_regime[rl]})")
            
            regime_corr_2l_results.append({
                'series': label, 'pair': pair_label,
                'rho_NORMAL': corr_by_regime.get("NORMAL"),
                'rho_HIGH': corr_by_regime.get("HIGH"),
                'n_NORMAL': n_common_by_regime.get("NORMAL", 0),
                'n_HIGH': n_common_by_regime.get("HIGH", 0),
            })
    
    df_regime_corr_2l = pd.DataFrame(regime_corr_2l_results)
    df_regime_corr_2l.to_csv(RQ3_TABLES_DIR / "T2c_regime_correlations_2l.csv", index=False)
    print(f"\n💾 T2c_regime_correlations_2l.csv")

# ============================================================================
# SECTION D: PERSISTENCE ANALYSIS (φ per regime, half-life)
# ============================================================================

print("\n" + "=" * 72)
print("SECTION D: Persistence Analysis (φ per regime)")
print("=" * 72)

persistence_results = []

for name in STRATEGY_NAMES:
    m_series = df_m[name].dropna()
    
    print(f"\n--- {STRATEGY_LABELS[name]} ---")
    
    for regime_label in ["LOW", "MEDIUM", "HIGH", "ALL"]:
        if regime_label == "ALL":
            m_reg = m_series
        else:
            mask = regime.reindex(m_series.index, method='ffill') == regime_label
            m_reg = m_series[mask]
        
        if len(m_reg) < 10:
            print(f"   {regime_label}: T={len(m_reg)} — troppo pochi, skip")
            continue
        
        # AR(1): m_t = φ * m_{t-1} + ε
        y = m_reg.iloc[1:].values
        x = m_reg.iloc[:-1].values
        X = sm.add_constant(x)
        
        try:
            model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': NW_MAX_LAGS})
            phi = model.params[1]
            phi_se = model.bse[1]
            phi_t = model.tvalues[1]
            phi_p = model.pvalues[1]
            
            # Half-life
            if 0 < phi < 1:
                half_life = np.log(0.5) / np.log(phi)
            else:
                half_life = np.nan
            
            print(f"   {regime_label:>6}: φ={phi:.3f} (t={phi_t:.2f}, p={phi_p:.4f}), "
                  f"half-life={half_life:.1f} mesi, T={len(m_reg)}")
            
            persistence_results.append({
                'strategy': STRATEGY_LABELS[name],
                'regime': regime_label,
                'phi': phi, 'phi_se': phi_se, 'phi_t': phi_t, 'phi_p': phi_p,
                'half_life_months': half_life, 'T': len(m_reg),
            })
        except Exception as e:
            print(f"   {regime_label}: errore — {e}")

df_persistence = pd.DataFrame(persistence_results)
df_persistence.to_csv(RQ3_TABLES_DIR / "T2d_persistence.csv", index=False)
print(f"\n💾 T2d_persistence.csv")


# ============================================================================
# SECTION E: CO-WIDENING TEST su Δm (per regime)
# ============================================================================

print("\n" + "=" * 72)
print("SECTION E: Co-Widening Test (Δm > 0)")
print("=" * 72)

cowidening_results = []

for s1, s2 in STRATEGY_PAIRS:
    pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
    print(f"\n--- {pair_label} ---")
    
    for regime_label in ["LOW", "MEDIUM", "HIGH", "ALL"]:
        if regime_label == "ALL":
            dm_pair = df_dm_full[[s1, s2]].dropna()
            dm1 = dm_pair[s1]
            dm2 = dm_pair[s2]
        else:
            dm_pair = df_dm_full[[s1, s2]].dropna()
            mask = regime.reindex(dm_pair.index, method='ffill') == regime_label
            dm1 = dm_pair.loc[mask, s1]
            dm2 = dm_pair.loc[mask, s2]
        
        common = dm1.dropna().index.intersection(dm2.dropna().index)
        dm1_c, dm2_c = dm1[common], dm2[common]
        n = len(common)
        
        if n < 5:
            print(f"   {regime_label}: T={n} — troppo pochi, skip")
            continue
        
        # Widening indicators
        w1 = (dm1_c > 0).astype(int)
        w2 = (dm2_c > 0).astype(int)
        
        # Probabilità
        p1 = w1.mean()
        p2 = w2.mean()
        p_joint = ((w1 == 1) & (w2 == 1)).mean()
        p_indep = p1 * p2
        excess_prob = p_joint - p_indep
        
        # Fisher exact test (tabella di contingenza 2x2)
        a = ((w1 == 1) & (w2 == 1)).sum()  # both widen
        b = ((w1 == 1) & (w2 == 0)).sum()
        c = ((w1 == 0) & (w2 == 1)).sum()
        d = ((w1 == 0) & (w2 == 0)).sum()
        
        _, p_fisher = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
        
        # Chi-square test
        chi2, p_chi2, _, _ = stats.chi2_contingency([[a, b], [c, d]], correction=True)
        
        print(f"   {regime_label:>6}: P(both)={p_joint:.3f}, P_indep={p_indep:.3f}, "
              f"excess={excess_prob:+.3f}, Fisher p={p_fisher:.4f}, T={n}")
        
        cowidening_results.append({
            'pair': pair_label, 'regime': regime_label,
            'T': n, 'P_widen_1': p1, 'P_widen_2': p2,
            'P_joint': p_joint, 'P_indep': p_indep,
            'excess_prob': excess_prob,
            'fisher_p': p_fisher, 'chi2': chi2, 'chi2_p': p_chi2,
        })
        
        all_pvalues[f"cowidening_{s1}_{s2}_{regime_label}_fisher"] = p_fisher

df_cowidening = pd.DataFrame(cowidening_results)
df_cowidening.to_csv(RQ3_TABLES_DIR / "T2e_cowidening.csv", index=False)
print(f"\n💾 T2e_cowidening.csv")


# ============================================================================
# SECTION F: TAIL CO-EXCEEDANCE
# ============================================================================

print("\n" + "=" * 72)
print("SECTION F: Tail Co-Exceedance")
print("=" * 72)

tail_results = []

for s1, s2 in STRATEGY_PAIRS:
    pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
    
    dm_pair = df_dm_full[[s1, s2]].dropna()
    dm1 = dm_pair[s1]
    dm2 = dm_pair[s2]
    common = dm1.index.intersection(dm2.index)
    dm1_c, dm2_c = dm1[common], dm2[common]
    
    # Soglia al TAIL_QUANTILE-esimo percentile di Δm (widening estremo)
    q1 = dm1_c.quantile(TAIL_QUANTILE)
    q2 = dm2_c.quantile(TAIL_QUANTILE)
    
    # Eventi estremi
    ext1 = (dm1_c > q1).astype(int)
    ext2 = (dm2_c > q2).astype(int)
    
    n = len(common)
    p1 = ext1.mean()
    p2 = ext2.mean()
    p_joint = ((ext1 == 1) & (ext2 == 1)).mean()
    p_indep = p1 * p2
    excess = p_joint - p_indep
    
    # Test
    a = ((ext1 == 1) & (ext2 == 1)).sum()
    b = ((ext1 == 1) & (ext2 == 0)).sum()
    c = ((ext1 == 0) & (ext2 == 1)).sum()
    d = ((ext1 == 0) & (ext2 == 0)).sum()
    
    _, p_fisher = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
    
    print(f"   {pair_label}:")
    print(f"      q{TAIL_QUANTILE:.0%}: s1>{q1:.2f}, s2>{q2:.2f}")
    print(f"      P(both extreme)={p_joint:.3f}, P_indep={p_indep:.3f}, "
          f"excess={excess:+.3f}, Fisher p={p_fisher:.4f}")
    
    tail_results.append({
        'pair': pair_label, 'quantile': TAIL_QUANTILE,
        'threshold_1': q1, 'threshold_2': q2,
        'P_extreme_1': p1, 'P_extreme_2': p2,
        'P_joint': p_joint, 'P_indep': p_indep,
        'excess_prob': excess, 'fisher_p': p_fisher,
        'n_both_extreme': a, 'T': n,
    })
    
    all_pvalues[f"tail_coexceed_{s1}_{s2}_fisher"] = p_fisher

df_tail = pd.DataFrame(tail_results)
df_tail.to_csv(RQ3_TABLES_DIR / "T2f_tail_coexceedance.csv", index=False)
print(f"\n💾 T2f_tail_coexceedance.csv")


# ============================================================================
# SECTION F2: PROBIT CO-WIDENING (con interazione stress)
# ============================================================================

print("\n" + "=" * 72)
print("SECTION F2: Probit Co-Widening con Interazione Stress")
print("   P(Δm_i > 0) = Φ(α + β₁·1{Δm_j > 0} + β₂·Stress + β₃·1{Δm_j > 0}×Stress)")
print("=" * 72)

from statsmodels.discrete.discrete_model import Probit

stress_level = df_purge[DEFAULT_STRESS_PROXY].reindex(df_dm_full.index, method='ffill') \
    if DEFAULT_STRESS_PROXY in df_purge.columns else \
    pd.read_parquet(FACTORS_PATH)[DEFAULT_STRESS_PROXY].reindex(df_dm_full.index, method='ffill')

probit_results = []

for s1, s2 in STRATEGY_PAIRS:
    for dep, indep in [(s1, s2), (s2, s1)]:
        dep_label = STRATEGY_LABELS[dep]
        indep_label = STRATEGY_LABELS[indep]
        
        df_probit = pd.DataFrame({
            'y': (df_dm_full[dep] > 0).astype(int),
            'x_widen': (df_dm_full[indep] > 0).astype(int),
            'stress': stress_level,
        }).dropna()
        
        if len(df_probit) < 20 or df_probit['y'].std() == 0:
            continue
        
        df_probit['x_stress'] = df_probit['x_widen'] * df_probit['stress']
        
        y = df_probit['y'].values
        X = sm.add_constant(df_probit[['x_widen', 'stress', 'x_stress']].values)
        
        try:
            model = Probit(y, X).fit(disp=0)
            
            # Effetto marginale dell'interazione: quanto aumenta P(widen_i)
            # quando widen_j=1 E stress è alto
            marg_effects = model.get_margeff(at='overall')
            
            print(f"\n   P({dep_label} widen) ~ 1{{{indep_label} widen}} × Stress:")
            print(f"      β₁(widen_j)            = {model.params[1]:+.4f}, z={model.tvalues[1]:+.2f}, "
                  f"p={model.pvalues[1]:.4f}")
            print(f"      β₂(stress)             = {model.params[2]:+.6f}, z={model.tvalues[2]:+.2f}, "
                  f"p={model.pvalues[2]:.4f}")
            print(f"      β₃(widen_j × stress)   = {model.params[3]:+.6f}, z={model.tvalues[3]:+.2f}, "
                  f"p={model.pvalues[3]:.4f}")
            print(f"      Pseudo-R² = {model.prsquared:.4f}, T = {len(df_probit)}")
            
            probit_results.append({
                'dependent': dep_label, 'independent': indep_label,
                'beta1_widen': model.params[1], 'p_widen': model.pvalues[1],
                'beta2_stress': model.params[2], 'p_stress': model.pvalues[2],
                'beta3_interaction': model.params[3], 'p_interaction': model.pvalues[3],
                'pseudo_R2': model.prsquared, 'T': len(df_probit),
            })
            
            all_pvalues[f"probit_cowiden_{dep}_{indep}_interaction"] = model.pvalues[3]
        
        except Exception as e:
            print(f"   ⚠️ {dep_label} ~ {indep_label}: errore — {e}")

if probit_results:
    df_probit_res = pd.DataFrame(probit_results)
    df_probit_res.to_csv(RQ3_TABLES_DIR / "T2f2_probit_cowidening.csv", index=False)
    print(f"\n💾 T2f2_probit_cowidening.csv")


# ============================================================================
# SECTION G: DIFFERENCE-IN-CO-WIDENING (Circular Bootstrap + Permutation)
# ============================================================================

print("\n" + "=" * 72)
print("SECTION G: Difference-in-Co-Widening")
print("   Method 1: Circular block bootstrap (Politis & Romano 1992)")
print("   Method 2: Permutation test (label shuffling)")
print("=" * 72)

diff_cw_results = []

for s1, s2 in STRATEGY_PAIRS:
    pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
    print(f"\n--- {pair_label} ---")
    
    dm_pair = df_dm_full[[s1, s2]].dropna()
    dm1 = dm_pair[s1]
    dm2 = dm_pair[s2]
    common = dm1.dropna().index.intersection(dm2.dropna().index)
    
    regime_cw = regime.reindex(common, method='ffill')
    
    mask_high = regime_cw == "HIGH"
    mask_low = regime_cw == "LOW"
    
    n_high = mask_high.sum()
    n_low = mask_low.sum()
    
    if n_high < 5 or n_low < 5:
        print(f"   ⚠️ Pochi dati in HIGH ({n_high}) o LOW ({n_low}), skip")
        continue
    
    formal = n_high >= MIN_OBS_REGIME_TEST and n_low >= MIN_OBS_REGIME_TEST
    
    cowiden = ((dm1[common] > 0) & (dm2[common] > 0)).astype(float)
    cw_high = cowiden[mask_high].values
    cw_low = cowiden[mask_low].values
    
    # Excess probability: regime-specific marginals
    p1_high = (dm1[common][mask_high] > 0).mean()
    p2_high = (dm2[common][mask_high] > 0).mean()
    p1_low = (dm1[common][mask_low] > 0).mean()
    p2_low = (dm2[common][mask_low] > 0).mean()
    
    excess_high = cw_high.mean() - p1_high * p2_high
    excess_low = cw_low.mean() - p1_low * p2_low
    obs_diff = cw_high.mean() - cw_low.mean()
    
    print(f"   P(co-widen|HIGH)={cw_high.mean():.3f} (n={n_high})")
    print(f"   P(co-widen|LOW) ={cw_low.mean():.3f} (n={n_low})")
    print(f"   Excess HIGH={excess_high:+.3f}, excess LOW={excess_low:+.3f}")
    print(f"   Diff = {obs_diff:+.3f}")
    
    # Method 1: Circular block bootstrap sull'intera serie
    _, p_circular, ci_lo, ci_hi = circular_bootstrap_diff_cowidening(
        dm1[common], dm2[common], regime_cw,
        n_reps=BOOTSTRAP_N_REPS, block_size=BOOTSTRAP_BLOCK_SIZE
    )
    print(f"   Circular bootstrap (L={BOOTSTRAP_BLOCK_SIZE}): p={p_circular:.4f}, "
          f"95% CI=[{ci_lo:+.3f}, {ci_hi:+.3f}]")
    
    # Method 2: Stationary bootstrap (Politis & Romano 1994)
    _, p_stationary, ci_lo_s, ci_hi_s = stationary_bootstrap_diff_cowidening(
        dm1[common], dm2[common], regime_cw,
        n_reps=BOOTSTRAP_N_REPS, block_size=BOOTSTRAP_BLOCK_SIZE
    )
    print(f"   Stationary bootstrap (E[L]={BOOTSTRAP_BLOCK_SIZE}): p={p_stationary:.4f}, "
          f"95% CI=[{ci_lo_s:+.3f}, {ci_hi_s:+.3f}]")
    
    # Method 3: Permutation test
    _, p_perm = permutation_test_diff_cowidening(
        dm1[common], dm2[common], regime_cw, n_reps=BOOTSTRAP_N_REPS
    )
    print(f"   Permutation test:   p={p_perm:.4f}")
    
    # Block size sensitivity (circular)
    sensitivity = {}
    for bs in BOOTSTRAP_BLOCK_SIZES:
        _, p_bs, _, _ = circular_bootstrap_diff_cowidening(
            dm1[common], dm2[common], regime_cw,
            n_reps=min(2000, BOOTSTRAP_N_REPS), block_size=bs
        )
        sensitivity[bs] = p_bs
    sens_str = ", ".join([f"L={bs}:p={p:.3f}" for bs, p in sensitivity.items()])
    print(f"   Block size sensitivity: {sens_str}")
    
    quality = "FORMAL" if formal else "illustr."
    print(f"   [{quality}]")
    
    diff_cw_results.append({
        'pair': pair_label,
        'P_cowiden_HIGH': cw_high.mean(), 'n_HIGH': n_high,
        'P_cowiden_LOW': cw_low.mean(), 'n_LOW': n_low,
        'diff': obs_diff,
        'p_circular_bootstrap': p_circular,
        'p_stationary_bootstrap': p_stationary,
        'ci_low': ci_lo, 'ci_high': ci_hi,
        'p_permutation': p_perm,
        'excess_HIGH': excess_high, 'excess_LOW': excess_low,
        'test_quality': quality,
    })
    
    all_pvalues[f"diff_cowidening_{s1}_{s2}_circular"] = p_circular
    all_pvalues[f"diff_cowidening_{s1}_{s2}_stationary"] = p_stationary
    all_pvalues[f"diff_cowidening_{s1}_{s2}_perm"] = p_perm

df_diff_cw = pd.DataFrame(diff_cw_results)
df_diff_cw.to_csv(RQ3_TABLES_DIR / "T2g_diff_cowidening_bootstrap.csv", index=False)
print(f"\n💾 T2g_diff_cowidening_bootstrap.csv")


# ============================================================================
# SECTION H: PURGED ANALYSIS (residui da fattori comuni)
# ============================================================================

print("\n" + "=" * 72)
print("SECTION H: Purged Analysis (residui da common factors)")
print("   Set di purge multipli per robustezza")
print("=" * 72)

# Purge principale (per compatibilità con File 03)
purge_factors = PURGE_SETS.get(PURGE_FACTOR_SET, COMMON_FACTORS_FOR_PURGING)
print(f"   Set principale: {PURGE_FACTOR_SET} → {purge_factors}")

# Allinea fattori al periodo overlap
purge_aligned = df_purge[purge_factors].reindex(df_dm_clean.index)
purge_common = purge_aligned.dropna()

# Residualizza Δm
residuals_dm = {}
for name in STRATEGY_NAMES:
    y = df_dm_clean[name].reindex(purge_common.index).dropna()
    X = purge_common.reindex(y.index)
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X, missing='drop').fit()
    residuals_dm[name] = model.resid
    print(f"   {STRATEGY_LABELS[name]}: R²={model.rsquared:.3f} "
          f"(varianza spiegata dai common factors)")

df_resid_dm = pd.DataFrame(residuals_dm)

# Residualizza anche livelli m
residuals_m = {}
purge_m_aligned = df_purge[purge_factors].reindex(df_m.index)
purge_m_common = purge_m_aligned.dropna()

for name in STRATEGY_NAMES:
    y = df_m[name].reindex(purge_m_common.index).dropna()
    X = purge_m_common.reindex(y.index)
    X = sm.add_constant(X)
    
    model = sm.OLS(y, X, missing='drop').fit()
    residuals_m[name] = model.resid

df_resid_m = pd.DataFrame(residuals_m)

# --- H0: Confronto tutti i set di purge ---
print(f"\n--- Confronto correlazioni residue per set di purge ---")
purge_comparison = []

# Carica TUTTI i fattori disponibili (non solo quelli del set base)
all_factors_for_purge = pd.read_parquet(FACTORS_PATH)

for set_name, set_factors in PURGE_SETS.items():
    # Check che i fattori esistano nel dataset completo
    available = [f for f in set_factors if f in all_factors_for_purge.columns]
    if not available:
        print(f"   {set_name}: ⚠️ nessun fattore disponibile, skip")
        continue
    
    p_aligned = all_factors_for_purge[available].reindex(df_dm_clean.index).dropna()
    if len(p_aligned) < 20:
        continue
    
    for s1, s2 in STRATEGY_PAIRS:
        pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
        
        y1 = df_dm_clean[s1].reindex(p_aligned.index).dropna()
        y2 = df_dm_clean[s2].reindex(p_aligned.index).dropna()
        common = y1.index.intersection(y2.index).intersection(p_aligned.index)
        
        if len(common) < 15:
            continue
        
        # Residualizza entrambe
        X = sm.add_constant(p_aligned.reindex(common))
        r1 = sm.OLS(y1[common], X, missing='drop').fit().resid
        r2 = sm.OLS(y2[common], X, missing='drop').fit().resid
        
        rho_purged = np.corrcoef(r1, r2)[0, 1]
        
        purge_comparison.append({
            'purge_set': set_name, 'factors': ", ".join(available),
            'pair': pair_label, 'rho_purged': rho_purged, 'T': len(common),
        })

if purge_comparison:
    df_purge_comp = pd.DataFrame(purge_comparison)
    
    # Pivot per leggibilità
    for pair in df_purge_comp['pair'].unique():
        print(f"\n   {pair}:")
        sub = df_purge_comp[df_purge_comp['pair'] == pair]
        for _, row in sub.iterrows():
            print(f"      {row['purge_set']:15s}: ρ_purged={row['rho_purged']:+.3f} (T={row['T']})")
    
    df_purge_comp.to_csv(RQ3_TABLES_DIR / "T2h0_purge_set_comparison.csv", index=False)
    print(f"\n💾 T2h0_purge_set_comparison.csv")

# --- H1: Correlazioni purged ---
print(f"\n--- Correlazioni purged (Δm residui) ---")
purged_corr_results = []

for s1, s2 in STRATEGY_PAIRS:
    pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
    
    x, y = df_resid_dm[s1], df_resid_dm[s2]
    common = x.dropna().index.intersection(y.dropna().index)
    x, y = x[common], y[common]
    
    if len(common) >= 10:
        pr, beta, t_hac, p_hac = hac_correlation_test(x, y)
        print(f"   {pair_label}: Pearson ρ={pr:+.3f}, t_HAC={t_hac:+.2f}, p={p_hac:.4f}")
        
        purged_corr_results.append({
            'pair': pair_label, 'pearson_r': pr,
            'pearson_t_hac': t_hac, 'pearson_p_hac': p_hac, 'N': len(common),
        })
        
        all_pvalues[f"purged_corr_{s1}_{s2}"] = p_hac

df_purged_corr = pd.DataFrame(purged_corr_results)
df_purged_corr.to_csv(RQ3_TABLES_DIR / "T2h_purged_correlations.csv", index=False)

# --- H2: Co-widening purged ---
print(f"\n--- Co-widening purged (Δm residui) ---")
purged_cw_results = []

for s1, s2 in STRATEGY_PAIRS:
    pair_label = f"{STRATEGY_LABELS[s1]} vs {STRATEGY_LABELS[s2]}"
    
    dm1 = df_resid_dm[s1].dropna()
    dm2 = df_resid_dm[s2].dropna()
    common = dm1.index.intersection(dm2.index)
    dm1_c, dm2_c = dm1[common], dm2[common]
    
    if len(common) < 10:
        continue
    
    w1 = (dm1_c > 0).astype(int)
    w2 = (dm2_c > 0).astype(int)
    
    p_joint = ((w1 == 1) & (w2 == 1)).mean()
    p_indep = w1.mean() * w2.mean()
    excess = p_joint - p_indep
    
    a = ((w1 == 1) & (w2 == 1)).sum()
    b = ((w1 == 1) & (w2 == 0)).sum()
    c = ((w1 == 0) & (w2 == 1)).sum()
    d = ((w1 == 0) & (w2 == 0)).sum()
    _, p_fisher = stats.fisher_exact([[a, b], [c, d]], alternative='greater')
    
    print(f"   {pair_label}: P(both)={p_joint:.3f}, excess={excess:+.3f}, "
          f"Fisher p={p_fisher:.4f}")
    
    purged_cw_results.append({
        'pair': pair_label, 'P_joint': p_joint,
        'P_indep': p_indep, 'excess_prob': excess,
        'fisher_p': p_fisher, 'T': len(common),
    })

df_purged_cw = pd.DataFrame(purged_cw_results)
df_purged_cw.to_csv(RQ3_TABLES_DIR / "T2h_purged_cowidening.csv", index=False)
print(f"\n💾 T2h_purged_correlations.csv, T2h_purged_cowidening.csv")


# ============================================================================
# SECTION I: MULTIPLE TESTING CORRECTION
# ============================================================================

print("\n" + "=" * 72)
print("SECTION I: Multiple Testing Correction (Benjamini–Hochberg)")
print("=" * 72)

# Raccogli tutti i p-values (escludendo NaN)
pval_names = list(all_pvalues.keys())
pval_values = [all_pvalues[k] for k in pval_names]

# Rimuovi NaN
valid_mask = [not (np.isnan(v) if isinstance(v, float) else False) for v in pval_values]
pval_names_valid = [n for n, m in zip(pval_names, valid_mask) if m]
pval_values_valid = [v for v, m in zip(pval_values, valid_mask) if m]

if pval_values_valid:
    # BH-FDR (default)
    reject, pvals_corrected, _, _ = multipletests(
        pval_values_valid, alpha=MTC_ALPHA, method=MTC_METHOD
    )
    
    # Holm-Bonferroni (robustezza)
    reject_holm, pvals_holm, _, _ = multipletests(
        pval_values_valid, alpha=MTC_ALPHA, method="holm"
    )
    
    mtc_results = pd.DataFrame({
        'test': pval_names_valid,
        'p_raw': pval_values_valid,
        'p_BH_FDR': pvals_corrected,
        'reject_BH': reject,
        'p_Holm': pvals_holm,
        'reject_Holm': reject_holm,
    }).sort_values('p_raw')
    
    mtc_results.to_csv(RQ3_TABLES_DIR / "T2i_multiple_testing.csv", index=False)
    
    n_total = len(reject)
    n_reject_bh = reject.sum()
    n_reject_holm = reject_holm.sum()
    print(f"   Test totali: {n_total}")
    print(f"   Reject H0 (BH-FDR {MTC_ALPHA}):        {n_reject_bh} ({n_reject_bh/n_total*100:.0f}%)")
    print(f"   Reject H0 (Holm-Bonferroni {MTC_ALPHA}): {n_reject_holm} ({n_reject_holm/n_total*100:.0f}%)")
    print(f"\n   Top 10 test più significativi:")
    print(mtc_results.head(10).to_string(index=False))
    print(f"\n💾 T2i_multiple_testing.csv")
else:
    print("   ⚠️ Nessun p-value valido raccolto.")


# ============================================================================
# SUMMARY PLOT: Correlazioni per regime (bar chart)
# ============================================================================

print("\n" + "=" * 72)
print("Plot riassuntivo")
print("=" * 72)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle("Correlations by Regime ($\\Delta m$)", fontsize=13, fontweight='bold')

for idx, (s1, s2) in enumerate(STRATEGY_PAIRS):
    ax = axes[idx]
    pair_label = f"{STRATEGY_LABELS[s1]}\nvs {STRATEGY_LABELS[s2]}"
    
    # Trova risultati per questa coppia (returns)
    row = df_regime_corr[
        (df_regime_corr['pair'].str.contains(STRATEGY_LABELS[s1])) &
        (df_regime_corr['pair'].str.contains(STRATEGY_LABELS[s2])) &
        (df_regime_corr['series'] == 'Δm')
    ]
    
    if len(row) > 0:
        row = row.iloc[0]
        regimes_list = ["LOW", "MEDIUM", "HIGH"]
        values = [row['rho_LOW'], row['rho_MED'], row['rho_HIGH']]
        colors = [REGIME_COLORS[r] for r in regimes_list]
        
        bars = ax.bar(regimes_list, values, color=colors, alpha=0.8, edgecolor='black')
        
        # FR-corrected come linea tratteggiata
        if not np.isnan(row['rho_FR_avg']):
            ax.axhline(row['rho_FR_avg'], color='purple', linewidth=1.5,
                       linestyle='--', label=f"FR-corr: {row['rho_FR_avg']:.2f}")
            ax.legend(fontsize=8)
    
    ax.set_title(pair_label, fontsize=10)
    ax.set_ylabel("Pearson ρ" if idx == 0 else "")
    ax.axhline(0, color='grey', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')

fig.tight_layout()
fig.savefig(RQ3_FIGURES_DIR / f"C1_regime_correlations_bar.{FIGURE_FORMAT}",
            dpi=FIGURE_DPI, bbox_inches='tight')
print(f"📊 C1_regime_correlations_bar.{FIGURE_FORMAT}")
plt.close()

# ============================================================================
# SUMMARY PLOT: Correlazioni per regime — 2-LEVEL (NORMAL/HIGH)
# ============================================================================

if regime_2l is not None and len(regime_corr_2l_results) > 0:
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig2.suptitle("Correlations by Regime — NORMAL/HIGH ($\\Delta m$)",
                  fontsize=13, fontweight='bold')

    df_regime_corr_2l_df = pd.DataFrame(regime_corr_2l_results)
    df_regime_corr_2l_dm = df_regime_corr_2l_df[
        df_regime_corr_2l_df['series'] == 'Δm'
    ]

    for idx, (s1, s2) in enumerate(STRATEGY_PAIRS):
        ax = axes2[idx]
        pair_label = f"{STRATEGY_LABELS[s1]}\nvs {STRATEGY_LABELS[s2]}"

        row = df_regime_corr_2l_dm[
            (df_regime_corr_2l_dm['pair'].str.contains(STRATEGY_LABELS[s1])) &
            (df_regime_corr_2l_dm['pair'].str.contains(STRATEGY_LABELS[s2]))
        ]

        if len(row) > 0:
            row = row.iloc[0]
            regimes_list = ["NORMAL", "HIGH"]
            values = [row['rho_NORMAL'], row['rho_HIGH']]
            colors = ['#2ca02c', '#d62728']

            bars = ax.bar(regimes_list, values, color=colors, alpha=0.8,
                         edgecolor='black')

        ax.set_title(pair_label, fontsize=10)
        ax.set_ylabel("Pearson ρ" if idx == 0 else "")
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')

    fig2.tight_layout()
    fig2.savefig(RQ3_FIGURES_DIR / f"C2_regime_correlations_bar_2l.{FIGURE_FORMAT}",
                dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"📊 C2_regime_correlations_bar_2l.{FIGURE_FORMAT}")
    plt.close()

# ============================================================================
# DONE
# ============================================================================

print("\n" + "=" * 72)
print("✅ FILE 2 COMPLETATO")
print("=" * 72)
print(f"\n   Tabelle salvate: {len(list(RQ3_TABLES_DIR.glob('T2*')))} file")
print(f"   Figure salvate:  {len(list(RQ3_FIGURES_DIR.glob('[BC]*')))} file")
print(f"\n   Prossimo step → rq3_03_spanning_regressions.py")
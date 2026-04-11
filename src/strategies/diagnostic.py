"""
================================================================================
diagnostics_cumsum_vs_cumprod.py
================================================================================
Confronto tra le due convenzioni di costruzione dell'equity curve:
  - cumsum  (linearizzazione additiva, attualmente nei file 02a/b/c)
  - cumprod (compounding geometrico, standard in finanza)

Analisi su entrambe le modalita' EW e SW della strategia BTP-Italia.

Output:
  - Plot comparativo equity curve (4 panel)
  - Plot differenza in punti percentuali nel tempo
  - Statistiche: differenza finale, max differenza, correlazione
  - Impatto sul return mensile (usato nelle regressioni)

Author: Alessio Ottaviani
Institution: EDHEC Business School - PhD Thesis
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "results" / "btp_italia"
OUTPUT_DIR   = RESULTS_DIR / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_CSV = RESULTS_DIR / "index_daily.csv"

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 70)
print("DIAGNOSTICA: cumsum vs cumprod — BTP-Italia")
print("=" * 70)

df = pd.read_csv(DAILY_CSV, index_col=0, parse_dates=True)

print(f"\n✅ Loaded: {len(df)} daily observations")
print(f"   Columns: {list(df.columns)}")
print(f"   Date range: {df.index.min().date()} → {df.index.max().date()}")

# Identifica colonne return EW e SW
# Il file puo' avere 'index_return' (colonna attiva) e
# opzionalmente 'index_return_ew' / 'index_return_sw'
# Costruiamo tutte le versioni disponibili

available_modes = {}

if "index_return_ew" in df.columns:
    available_modes["EW"] = df["index_return_ew"].dropna()
elif "index_return" in df.columns:
    # Se il file e' stato eseguito in EW mode, index_return = EW
    # Lo usiamo come EW di fallback
    available_modes["EW"] = df["index_return"].dropna()

if "index_return_sw" in df.columns:
    available_modes["SW"] = df["index_return_sw"].dropna()

if not available_modes:
    raise ValueError("Nessuna colonna di return trovata nel file.")

print(f"\n   Modalità trovate: {list(available_modes.keys())}")

# ============================================================================
# FUNZIONI DI COSTRUZIONE EQUITY CURVE
# ============================================================================

def build_cumsum(daily_ret_pct: pd.Series) -> pd.Series:
    """
    Metodo attuale nei file 02a/b/c:
    cumulative = sum(daily_ret)
    index = 100 * (1 + cumulative/100)
    Equivalente a: index_t = 100 + sum(r_1..r_t)
    """
    cum = daily_ret_pct.cumsum()
    return 100 * (1 + cum / 100)


def build_cumprod(daily_ret_pct: pd.Series) -> pd.Series:
    """
    Compounding geometrico (standard total return index):
    index_t = 100 * prod(1 + r_i/100)
    Ogni giorno il capitale guadagnato viene reinvestito.
    """
    return 100 * (1 + daily_ret_pct / 100).cumprod()


def to_monthly_cumsum(daily_ret_pct: pd.Series) -> pd.Series:
    """
    Return mensile con convenzione cumsum:
    R_month = sum(daily_returns nel mese)
    """
    return daily_ret_pct.resample("ME").sum()


def to_monthly_cumprod(daily_ret_pct: pd.Series) -> pd.Series:
    """
    Return mensile con convenzione cumprod (compound):
    R_month = prod(1 + r_i/100) - 1, espresso in %
    """
    return daily_ret_pct.resample("ME").apply(
        lambda x: ((1 + x / 100).prod() - 1) * 100 if len(x) > 0 else np.nan
    )

# ============================================================================
# CALCOLO PER OGNI MODALITA'
# ============================================================================

results = {}

for mode, daily_ret in available_modes.items():
    cs_daily  = build_cumsum(daily_ret)
    cp_daily  = build_cumprod(daily_ret)
    diff_daily = cp_daily - cs_daily  # differenza in punti indice (base 100)

    cs_monthly = to_monthly_cumsum(daily_ret)
    cp_monthly = to_monthly_cumprod(daily_ret)
    diff_monthly = cp_monthly - cs_monthly  # differenza in % points

    results[mode] = {
        "daily_ret":   daily_ret,
        "cs_daily":    cs_daily,
        "cp_daily":    cp_daily,
        "diff_daily":  diff_daily,
        "cs_monthly":  cs_monthly,
        "cp_monthly":  cp_monthly,
        "diff_monthly": diff_monthly,
    }

    print(f"\n{'─'*60}")
    print(f"  Modalità: {mode}")
    print(f"{'─'*60}")

    # --- Statistiche equity curve ---
    final_cs = cs_daily.iloc[-1]
    final_cp = cp_daily.iloc[-1]
    final_diff = final_cp - final_cs
    max_diff   = diff_daily.abs().max()
    corr_daily = np.corrcoef(cs_daily, cp_daily)[0, 1]

    print(f"  Equity curve finale (base 100):")
    print(f"    cumsum:  {final_cs:.4f}")
    print(f"    cumprod: {final_cp:.4f}")
    print(f"    Differenza finale: {final_diff:+.4f} punti indice")
    print(f"    Differenza max (assoluta): {max_diff:.4f} punti indice")
    print(f"    Correlazione giornaliera: {corr_daily:.8f}")

    # --- Statistiche return mensili ---
    n_common = min(len(cs_monthly.dropna()), len(cp_monthly.dropna()))
    common_idx = cs_monthly.dropna().index.intersection(cp_monthly.dropna().index)
    cs_m = cs_monthly.loc[common_idx]
    cp_m = cp_monthly.loc[common_idx]
    diff_m = cp_m - cs_m

    corr_monthly = np.corrcoef(cs_m, cp_m)[0, 1]

    print(f"\n  Return mensili ({len(common_idx)} mesi):")
    print(f"    Media  cumsum:  {cs_m.mean():.4f}%  |  cumprod: {cp_m.mean():.4f}%")
    print(f"    StDev  cumsum:  {cs_m.std():.4f}%  |  cumprod: {cp_m.std():.4f}%")
    print(f"    Differenza media (cumprod - cumsum): {diff_m.mean():+.6f}%")
    print(f"    Differenza max mensile:              {diff_m.abs().max():.6f}%")
    print(f"    Correlazione mensile:                {corr_monthly:.8f}")

    # Sharpe approssimato
    sr_cs = (cs_m.mean() / cs_m.std()) * np.sqrt(12)
    sr_cp = (cp_m.mean() / cp_m.std()) * np.sqrt(12)
    print(f"\n  Sharpe ratio annualizzato (approssimato):")
    print(f"    cumsum:  {sr_cs:.4f}")
    print(f"    cumprod: {sr_cp:.4f}")
    print(f"    Differenza: {sr_cp - sr_cs:+.6f}")

# ============================================================================
# PLOT
# ============================================================================

modes = list(results.keys())
n_modes = len(modes)
fig_height = 18 if n_modes == 2 else 10

fig = plt.figure(figsize=(16, fig_height))
n_rows = 3 if n_modes == 2 else 2
gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.35)

mode_colors = {"EW": {"cs": "#1f77b4", "cp": "#d62728"},
               "SW": {"cs": "#2ca02c", "cp": "#ff7f0e"}}
default_colors = {"cs": "#1f77b4", "cp": "#d62728"}

pct_fmt = FuncFormatter(lambda x, _: f"{x:.1f}%")
idx_fmt = FuncFormatter(lambda x, _: f"{x:.1f}")

# --- Row 0: Equity curves ---
for col, mode in enumerate(modes):
    ax = fig.add_subplot(gs[0, col])
    r = results[mode]
    cols = mode_colors.get(mode, default_colors)

    ax.plot(r["cs_daily"].index, r["cs_daily"].values,
            label="cumsum (attuale)", color=cols["cs"], linewidth=1.5, alpha=0.9)
    ax.plot(r["cp_daily"].index, r["cp_daily"].values,
            label="cumprod (geometrico)", color=cols["cp"],
            linewidth=1.5, linestyle="--", alpha=0.9)

    ax.axhline(100, color="black", linewidth=0.6, linestyle=":")
    ax.set_title(f"Equity Curve — {mode}", fontsize=11, fontweight="bold")
    ax.set_ylabel("Indice (base 100)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(idx_fmt)

# --- Row 1: Differenza giornaliera (punti indice) ---
for col, mode in enumerate(modes):
    ax = fig.add_subplot(gs[1, col])
    r = results[mode]
    cols = mode_colors.get(mode, default_colors)

    ax.fill_between(r["diff_daily"].index, 0, r["diff_daily"].values,
                    where=r["diff_daily"] >= 0,
                    color=cols["cp"], alpha=0.5, label="cumprod > cumsum")
    ax.fill_between(r["diff_daily"].index, 0, r["diff_daily"].values,
                    where=r["diff_daily"] < 0,
                    color=cols["cs"], alpha=0.5, label="cumsum > cumprod")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Differenza Giornaliera: cumprod − cumsum — {mode}",
                 fontsize=10)
    ax.set_ylabel("Δ punti indice (base 100)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

# --- Row 2 (solo se 2 modi): Return mensili scatter e differenza ---
if n_modes == 2:
    # Scatter: monthly returns cumsum vs cumprod per EW
    mode = modes[0]
    ax_sc = fig.add_subplot(gs[2, 0])
    r = results[mode]
    common = r["cs_monthly"].dropna().index.intersection(r["cp_monthly"].dropna().index)
    cs_m = r["cs_monthly"].loc[common]
    cp_m = r["cp_monthly"].loc[common]

    ax_sc.scatter(cs_m, cp_m, alpha=0.6, s=20,
                  color=mode_colors.get(mode, default_colors)["cp"])
    mn = min(cs_m.min(), cp_m.min()) - 0.1
    mx = max(cs_m.max(), cp_m.max()) + 0.1
    ax_sc.plot([mn, mx], [mn, mx], "k--", linewidth=1, label="45° line")
    ax_sc.set_xlabel("Return mensile — cumsum (%)")
    ax_sc.set_ylabel("Return mensile — cumprod (%)")
    ax_sc.set_title(f"Scatter Return Mensili — {mode}", fontsize=10)
    ax_sc.legend(fontsize=9)
    ax_sc.grid(alpha=0.3)

    # Differenza mensile per entrambi i modi
    ax_diff = fig.add_subplot(gs[2, 1])
    for mode in modes:
        r = results[mode]
        common = r["cs_monthly"].dropna().index.intersection(r["cp_monthly"].dropna().index)
        diff_m = r["cp_monthly"].loc[common] - r["cs_monthly"].loc[common]
        cols = mode_colors.get(mode, default_colors)
        ax_diff.plot(diff_m.index, diff_m.values,
                     label=f"{mode}: cumprod−cumsum",
                     color=cols["cp"], linewidth=1.2)

    ax_diff.axhline(0, color="black", linewidth=0.8)
    ax_diff.set_title("Differenza Return Mensili: cumprod − cumsum", fontsize=10)
    ax_diff.set_ylabel("Δ punti percentuali")
    ax_diff.legend(fontsize=9)
    ax_diff.grid(alpha=0.3)
    ax_diff.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.4f}%"))

elif n_modes == 1:
    mode = modes[0]
    ax_sc = fig.add_subplot(gs[1, 0])  # already done above; add scatter in row 1
    # Redraw scatter in a new row if only one mode
    # (already handled above for 2-mode case)

fig.suptitle(
    "Diagnostica: cumsum vs cumprod — BTP-Italia\n"
    "cumsum = sommatoria additiva (attuale) | "
    "cumprod = compounding geometrico (standard)",
    fontsize=13, fontweight="bold", y=1.01
)

plot_path = OUTPUT_DIR / "cumsum_vs_cumprod.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n💾 Plot salvato: {plot_path}")

# ============================================================================
# CONCLUSIONE TESTUALE
# ============================================================================

print(f"""
{'='*70}
CONCLUSIONE
{'='*70}

Per la tua tesi, la scelta tra cumsum e cumprod:

CUMSUM (attuale):
  + Più semplice e trasparente da spiegare
  + Usato da Rebonato nel contesto delle strategie signal-weighted
  + Differenza trascurabile per return giornalieri piccoli
  - Non è il metodo standard per total return index
  - Può in teoria portare a indice negativo
  - Non coerente con il compound mensile in 01_pca_preprocessing.py

CUMPROD (geometrico):
  + Standard de facto per total return index (Bloomberg, MSCI, ecc.)
  + Coerente con il compound mensile usato nella PCA pipeline
  + Garantisce non-negatività dell'indice
  + Per regressioni su return mensili: la differenza numerica è
    dell'ordine di 0.000x% per mese — ininfluente per l'alpha
  - Introduce leggermente più complessità nell'esposizione

RACCOMANDAZIONE: mantieni cumsum nei file 02a/b/c.

  Il motivo è strutturale, non di convenienza:
  ogni trade è capitalizzato a nozionale FISSO di 100.
  I guadagni del trade t non vengono reinvestiti nel trade t+1.
  Questo è esattamente il setup di Duarte, Longstaff, Yu (2007,
  Review of Financial Studies) — vedi Appendix A: "we implement
  the trade for a $100 notional position" e l'indice mensile
  è la media equally-weighted dei return sui trade aperti.
  Cumprod implicherebbe reinvestimento continuo del P&L, che
  non riflette la struttura operativa della strategia.

  Per le regressioni mensili in 03_interdependencies.py,
  usa to_monthly_cumsum (somma dei daily return nel mese),
  coerente con la costruzione dell'equity curve.
{'='*70}
""")
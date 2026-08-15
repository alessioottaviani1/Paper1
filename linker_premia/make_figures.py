"""make_figures - genera le figure per le slide dai CSV prodotti da r1/r2/r3
(data/cache/an_*.csv). Nessun Bloomberg, nessun ricalcolo: solo lettura e grafica.
Output: data/cache/fig_1_bei_isr.png, fig_2_lambda_beta.png, fig_3_predict_r2.png,
        fig_4_robustness.png. Lancia dalla root:  python .\\src\\linker_premia\\make_figures.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import CACHE, DATA

# le figure vanno DOVE il .tex le cerca (graphicspath: ../results/figures/linker_premia/),
# cosi' basta lanciare questo script e poi compilare: nessuno spostamento manuale.
OUT = DATA.parent / "results" / "figures" / "linker_premia"
OUT.mkdir(parents=True, exist_ok=True)

# PDF vettoriale per \includegraphics in Beamer (nitido a ogni zoom); PNG solo di ripiego.
FMT = "pdf"
C_US, C_UK, C_GREY = "#1E2761", "#B85042", "#8A8A8A"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
                     "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 12,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "pdf.fonttype": 42,
                     "figure.autolayout": False})

# ---------------------------------------------------------------- fig 1: BEI - ISR (= -lambda)
b = pd.read_csv(CACHE / "an_bei_premia.csv", index_col=0, parse_dates=True)
fig, ax = plt.subplots(figsize=(11, 4.6))
for mkt, col in (("US", C_US), ("UK", C_UK)):
    s = (b[f"BEI-ISR_{mkt}_10y"] * 100).rolling(3, min_periods=1).mean()
    ax.plot(s.index, s, color=col, lw=1.8, label=f"{mkt} 10y")
ax.axhline(0, color="black", lw=0.8)
for x0, x1, lab in (("2008-09-01", "2009-06-30", "GFC"), ("2020-03-01", "2020-06-30", "Covid"),
                    ("2021-06-01", "2022-12-31", "inflation surge")):
    ax.axvspan(pd.Timestamp(x0), pd.Timestamp(x1), color=C_GREY, alpha=0.15)
    ax.text(pd.Timestamp(x0), ax.get_ylim()[1] * 0.92, lab, fontsize=9, color=C_GREY)
ax.set_ylabel("bp"); ax.set_title("BEI − ISR (= −λ): linker cheapness to the synthetic, 10y, 3m MA")
c = b[["BEI-ISR_US_10y", "BEI-ISR_UK_10y"]].dropna().corr().iloc[0, 1]
ax.legend(loc="lower left", frameon=False, title=f"corr = {c:.2f}")
fig.tight_layout(); fig.savefig(OUT / "fig_1_bei_isr.{}".format(FMT), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig 2: beta di lambda e dy per scadenza
r = pd.read_csv(CACHE / "an_surprise_regs.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
for ax, var, ttl in ((axes[0], "lambda", "λ(T) = ISR − BEI  [bp per unit surprise]"),
                     (axes[1], "dy_real", "Δ real yield(T)  [bp per unit surprise]")):
    for mkt, col, off in (("US", C_US, -0.15), ("UK", C_UK, 0.15)):
        d = r[(r["mkt"] == mkt) & (r["var"] == var)].sort_values("mat")
        x = np.arange(len(d)) + off
        ax.bar(x, d["beta_bp"], width=0.3, color=col, alpha=0.9, label=mkt)
        for xi, (bb, tt) in zip(x, zip(d["beta_bp"], d["t_NW"])):
            ax.text(xi, bb + (0.25 if bb >= 0 else -0.6), f"t={tt:+.1f}", ha="center", fontsize=8,
                    color=col if abs(tt) >= 1.96 else C_GREY)
        ax.set_xticks(np.arange(len(d))); ax.set_xticklabels([f"{int(m)}y" for m in d["mat"]])
    ax.axhline(0, color="black", lw=0.8); ax.set_title(ttl, fontsize=11)
axes[0].legend(frameon=False)
fig.suptitle("Reaction to inflation surprises (YoY − 10y MA), by maturity — segmentation", y=1.02)
fig.tight_layout(); fig.savefig(OUT / "fig_2_lambda_beta.{}".format(FMT), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig 3: R2 per fattore su rx medio
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
order = ["SLOPE", "CP", "CiP", "COINT"]
for ax, mkt, col in ((axes[0], "UK", C_UK), (axes[1], "US", C_US)):
    t = pd.read_csv(CACHE / f"an_predictability_{mkt}.csv")
    t = t[t["target"] == "rx(media)"].set_index("fattore").reindex(order)
    bars = ax.bar(order, t["R2"], color=[C_GREY if f == "SLOPE" else col for f in order], alpha=0.9)
    for bar, tt in zip(bars, t["t_NW"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"t={tt:+.1f}",
                ha="center", fontsize=9)
    ax.set_title(f"{mkt}: 12m real excess return (avg maturity)"); ax.set_ylabel("R²")
fig.suptitle("Return-predicting factors on linkers: SLOPE vs CP vs Cieslak–Povala vs cointegration", y=1.02)
fig.tight_layout(); fig.savefig(OUT / "fig_3_predict_r2.{}".format(FMT), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig 4: robustezza griglia (dati incollati dal run r3)
rob = {
    "UK": {"(4,5,6,7)": (0.193, 0.168), "(4..8)": (0.240, 0.219), "(5,7,10)": (0.238, 0.216),
           "(4,6..12)": (0.258, 0.242), "(5,10,15)": (0.242, 0.226)},
    "US": {"(3..7)": (0.219, 0.102), "(4..8)": (0.216, 0.099), "(5,7,10)": (0.133, 0.040),
           "(4,6..12)": (0.228, 0.102), "(5,10,15)": (0.134, 0.029)},
}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
for ax, mkt, col in ((axes[0], "UK", C_UK), (axes[1], "US", C_US)):
    g = rob[mkt]; x = np.arange(len(g))
    ax.bar(x - 0.18, [v[0] for v in g.values()], width=0.36, color=C_GREY, label="CP")
    ax.bar(x + 0.18, [v[1] for v in g.values()], width=0.36, color=col, label="COINT")
    ax.axhline({"UK": 0.098, "US": 0.339}[mkt], color="black", ls="--", lw=1, label="CiP (ref.)")
    ax.set_xticks(x); ax.set_xticklabels(list(g), fontsize=9); ax.set_title(f"{mkt}: forward grid")
    ax.set_ylabel("R² on avg rx")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle("Robustness: CP and cointegration predictor across forward grids", y=1.02)
fig.tight_layout(); fig.savefig(OUT / "fig_4_robustness.{}".format(FMT), bbox_inches="tight"); plt.close(fig)

print("figure salvate in", OUT, ":", [f"fig_{i}_*.{FMT}" for i in (1, 2, 3, 4)])

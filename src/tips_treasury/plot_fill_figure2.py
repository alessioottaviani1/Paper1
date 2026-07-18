"""
Replication and extension of Fleckenstein-Longstaff-Lustig (2014), FIGURE 2:
cross-sectional average TIPS-Treasury mispricing in BASIS POINTS, daily.

FLL construction: for each date, the average mispricing across all live
TIPS-Treasury pairs, weighted by the TIPS notional. Sample Jul-2004 - Nov-2009.
Benchmarks from the paper: overall average 54.5 bp; peak ~175 bp at the
Lehman bankruptcy (Fall 2008).

Our input: the 107-CUSIP bond-level mispricing panel (bp per pair, daily).
One stated deviation: pair notionals are not in the panel, so the average is
TIPS-notional-weighted, exactly as in the paper, using the Amount Issued of
each TIPS (second sheet of the panel file). Equal-weighted printed as check.

Style matched to the paper's figures: white background, black 0.8 series line,
Times serif, vector PDF. Per Cochrane: axes labelled, sensible units, no
invisible dotted lines, no dashes on the volatile series itself. The extended
panel marks the end of the FLL sample with a single dashed red vertical rule.

Outputs (vector PDF), written to <THESIS>/results/tips_treasury/figures/ :
  fll_fig2_replica.pdf    - exact FLL window (Jul-2004 .. Nov-2009)
  fll_fig2_extended.pdf   - full sample to 2026, FLL sample end marked

Usage (from the THESIS root):  python src\tips_treasury\plot_fll_figure2.py
"""
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ------------------------- house style (matches the paper's figures) ----------
matplotlib.rcParams.update({
    "pdf.fonttype":   42,            # TrueType embedding (no Type-3 in print)
    "ps.fonttype":    42,
    "font.family":    "serif",       # Times-like, as the manuscript body
    "font.serif":     ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":      11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "black",
})

# ------------------------- paths ----------------------------------------------
# Resolve everything from the file location so the script runs from any cwd,
# exactly like the other modules in src/.
ROOT     = Path(__file__).resolve().parents[2]                       # ...\THESIS
PANEL    = ROOT / "data" / "raw" / "Bloomberg" / "tips_cusip.xlsx"   # proprietary Bloomberg panel
FIG_DIR  = ROOT / "results" / "figures"          # where the paper's \figpath points
FIG_DIR.mkdir(parents=True, exist_ok=True)

FLL_A, FLL_B = "2004-07-23", "2009-11-19"                            # FLL sample window
OUT_EXACT = FIG_DIR / "fll_fig2_replica.pdf"     # exact-window replica (not used in the paper)
OUT_EXT   = FIG_DIR / "TIPS_basis.pdf"           # the file paper1_figures_final.tex includes

# ------------------------- load panel -----------------------------------------
bl = pd.read_excel(PANEL, sheet_name=0)
bl = bl.rename(columns={bl.columns[0]: "date"}).set_index("date")
bl.index = pd.to_datetime(bl.index)
bl = bl.apply(pd.to_numeric, errors="coerce").sort_index()

# ---- notional weights (Amount Issued per TIPS), exactly as in FLL ----
wsheet = pd.read_excel(PANEL, sheet_name=1)
key_col = [c for c in wsheet.columns if wsheet[c].astype(str).str.startswith("US").all()][0]
notional = wsheet.set_index(key_col)["Amount Issued"].astype(float).reindex(bl.columns)
print(f"notional match  : {notional.notna().sum()}/{bl.shape[1]} CUSIPs")

W = pd.DataFrame(1.0, index=bl.index, columns=bl.columns).where(bl.notna()) * notional
avg = (bl * W).sum(axis=1) / W.sum(axis=1)    # TIPS-notional-weighted mean, daily
avg_eq = bl.mean(axis=1)                       # equal-weighted, printed for comparison

# ------------------------- validation -----------------------------------------
w = avg.loc[FLL_A:FLL_B]
print(f"FLL-window mean : {w.mean():6.1f} bp  weighted   |  {avg_eq.loc[FLL_A:FLL_B].mean():5.1f} equal-w   (paper: 54.5)")
print(f"FLL-window peak : {w.max():6.1f} bp on {w.idxmax().date()}   (paper: ~175, Fall 2008)")


# ------------------------- style helper ---------------------------------------
def style_axes(ax):
    ax.set_ylabel("Mispricing (basis points)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.8)
    ax.tick_params(direction="out", top=False, right=False, length=3)
    ax.axhline(0, color="black", linewidth=0.5)


# ------------------------- Figure A: exact replica ----------------------------
fig, ax = plt.subplots(figsize=(7.5, 3.4))
ax.plot(w.index, w.values, lw=0.8, color="black")
style_axes(ax)
ax.set_xlim(pd.Timestamp("2004-07-01"), pd.Timestamp("2009-12-31"))
ax.set_ylim(bottom=0)
fig.tight_layout()
fig.savefig(OUT_EXACT, bbox_inches="tight")
plt.close(fig)

# ------------------------- Figure B: extended, same style ---------------------
fig, ax = plt.subplots(figsize=(9.5, 3.2))
ax.plot(avg.index, avg.values, lw=0.8, color="black")
# Single event marker: end of the FLL sample, dashed red, set off from the data.
ax.axvline(pd.Timestamp(FLL_B), color="crimson", lw=1.0, ls="--")
ax.annotate("End of FLL sample",
            xy=(pd.Timestamp(FLL_B), ax.get_ylim()[1] * 0.94),
            xytext=(6, 0), textcoords="offset points",
            fontsize=9, va="top", color="crimson")
style_axes(ax)
fig.tight_layout()
fig.savefig(OUT_EXT, bbox_inches="tight")
plt.close(fig)
print(f"saved -> {OUT_EXACT}")
print(f"saved -> {OUT_EXT}")
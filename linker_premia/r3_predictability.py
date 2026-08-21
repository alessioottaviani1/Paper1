"""r3 - Predicibilita' degli excess return REALI (richiesta RR n.3): slope vs CP vs
Cieslak-Povala vs predittore di COINTEGRAZIONE diretto tra i forward (Rebonato-Nyholm
JEF 2025: il potere di CP sta nell'identificare la combinazione stazionaria dei forward
quasi-unit-root; la cointegrazione fornisce predittori direttamente).
Excess return da curve zero (cc), holding 12m overlapping, funding = reale piu' corto
disponibile (dichiarato); inferenza Hansen-Hodrick a 12 lag, la convenzione di Cochrane-Piazzesi (2005, p.143)
per i coefficienti singoli in regressioni previsive con dati sovrapposti; Newey-West a
18 lag resta calcolato come riserva per i casi in cui la matrice HH non e' PSD. Output
per mercato: tabella beta/t/R2 per maturita' e per rx medio + t-ADF del residuo di
cointegrazione.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import pandas as pd
import rp
from config import CACHE, DATA

# Tabelle LaTeX per slide/tesi: \input{\tablepath/nome}. Le scrive lo stesso script che
# calcola i numeri (come nel Paper 1), cosi' non possono divergere.
TABLES_DIR = DATA.parent / "results" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

def _stars(t, thr=(1.645, 1.96, 2.576)):
    try: a = abs(float(t))
    except (TypeError, ValueError): return ""
    if a != a: return ""
    return "***" if a >= thr[2] else "**" if a >= thr[1] else "*" if a >= thr[0] else ""

_ROWS_FACT, _ROWS_GRID = [], []          # accumulatori per le tabelle finali

# tenor allineati a Maffei (2-20yr) per r2; in r3 il target piu' corto e' 3y perche'
# l'excess return 12m richiede il rendimento a (target-1)=2y, affidabile; a 2y servirebbe
# l'1y reale, che e' rado. La curva US include i punti bassi per il fit dei forward.
#
# SOLO US e UK, per la domanda RR n.3. I mercati euro sono esclusi: il censimento delle
# curve (r5) mostra che le curve reali IT/FR/DE sono FRAGILI o NON utilizzabili ai nodi
# usati qui (DE: 4 parametri su 4-5 Bund€i, gradi di liberta' ~0, l'rmse va a 0 per
# costruzione; FR: 2y non utilizzabile, 5y/10y fragili; IT: 10y fragile, 20y no).
# I risultati euro erano artefatti del fit (es. DE CiP R2=0.69 con t+16, non credibile).
# US (108 linker) e UK (46) hanno curve solide su tutti i nodi: sono i mercati che
# Rebonato chiede e gli unici su cui la risposta e' difendibile.
CONF = {"UK": dict(y=lambda: rp.boe("real"), mats=(5., 7., 10., 15., 20.), cpi="UK"),
        "US": dict(y=lambda: rp.gsw("tips", (2., 3., 4., 5., 7., 10., 15., 20.)),
                   mats=(5., 7., 10., 15., 20.), cpi="US")}

print(">>> r3 US/UK-only v2 <<<  (se non vedi questa riga, gira un file diverso)")
for mkt, c in CONF.items():
    y = c["y"]()
    rx, fund = rp.rx_panel(y, c["mats"])
    rxb = rx.mean(axis=1)
    F = rp.forwards(y, mats=(3, 4, 5, 6, 7))     # forward 1y da 3y a 7y: coperti da entrambe le curve reali (min 2-2.5y)
    short = max(2.0, float(min(y.columns)))       # punto corto REALE della curva (BoE 2.5, GSW 2.0): niente estrapolazione
    ye = rp.interp_cols(rp.eom(y), (short, 10.0))
    slope = (ye[10.0] - ye[short]).rename("SLOPE")
    cp, b_cp, r2_cp = rp.cp_factor(rxb, F)
    trend = rp.ewma_trend(rp.yoy(rp.cpi(c["cpi"])))
    cip, cycles, r2_cip = rp.cip_factor(y, trend, rxb, c["mats"])
    coin, r_j, adf = rp.coint_johansen_predictor(F, rxb)
    tab = rp.predict_table(rx, {"SLOPE": slope, "CP": cp, "CiP": cip, "COINT": coin})
    tab.insert(0, "mkt", mkt)
    tab.to_csv(CACHE / f"an_predictability_{mkt}.csv", index=False)
    print(f"\n=== {mkt}: excess return reali 12m (funding = reale {fund:.1f}y, slope = {short:.1f}y-10y) ===")
    print(f"  cointegrazione forward (Johansen): rango r = {r_j} | t-ADF primo ECM = "
          f"{adf:.2f} ({'stazionario' if adf < -2.86 else 'radice unitaria non rigettata'})")
    piv = tab.pivot_table(index="fattore", columns="target", values="R2")
    print("  R2 per fattore (colonna rx(media) = il confronto chiave):")
    print(piv.round(3).to_string())
    tm = tab[tab.target == "rx(media)"].set_index("fattore")
    print("  t-HH(12) su rx(media): " + "  ".join(f"{i}={v:+.1f}" for i, v in tm.t_HH.items()))
    if tm.t_HH.isna().any():
        # HH senza smorzamento non garantisce una matrice semidefinita positiva: dove la
        # varianza esce negativa il t non esiste e si legge quello di Newey-West.
        bad = list(tm.index[tm.t_HH.isna()])
        print(f"    [!] HH non calcolabile per {bad} (matrice non PSD): uso NW-18")
        print("    t-NW(18) di riserva: " + "  ".join(f"{i}={v:+.1f}" for i, v in tm.t_NW.items()))

    # accumulo per la tabella LaTeX dei fattori
    for _f in ("SLOPE", "CP", "CiP", "COINT"):
        if _f in tm.index:
            _t = tm.loc[_f, "t_HH"]
            if pd.isna(_t): _t = tm.loc[_f, "t_NW"]
            _ROWS_FACT.append((mkt, _f, float(tm.loc[_f, "R2"]), float(_t), r_j, adf))

    # ---- ROBUSTEZZA: il risultato COINT/CP dipende dalla griglia dei forward? ----
    # Griglie alternative (tutte coperte dalle curve reali); per ognuna R2 e t di
    # CP, COINT (e CiP, che non dipende dai forward, come riferimento) su rx medio.
    print("  robustezza griglia forward (R2 | t-HH su rx(media)):")
    print(f"    {'griglia':14s} {'CP':>14s} {'COINT':>14s} {'CiP':>14s}  rango")
    for grid in [(3,4,5,6,7), (4,5,6,7,8), (5,7,10), (4,6,8,10,12), (5,10,15)]:
        Fg = rp.forwards(y, mats=grid)
        if Fg.shape[1] < 2:
            continue
        cpg, _, _ = rp.cp_factor(rxb, Fg)
        cog, rg, _ = rp.coint_johansen_predictor(Fg, rxb)
        tg = rp.predict_table(rx, {"CP": cpg, "COINT": cog, "CiP": cip})
        tg = tg[tg.target == "rx(media)"].set_index("fattore")
        # stessa inferenza della tabella principale: HH-12, con NW-18 solo se HH manca
        def cell(f):
            if f not in tg.index: return "  n/a"
            t = tg.loc[f, "t_HH"]
            if pd.isna(t): t = tg.loc[f, "t_NW"]
            return f"{tg.loc[f,'R2']:.3f}|{t:+.1f}"
        used = tuple(int(c) for c in Fg.columns)
        print(f"    {str(used):14s} {cell('CP'):>14s} {cell('COINT'):>14s} {cell('CiP'):>14s}  r={rg}")
        def _rt(f):
            if f not in tg.index: return (float("nan"), float("nan"))
            t = tg.loc[f, "t_HH"]
            if pd.isna(t): t = tg.loc[f, "t_NW"]
            return (float(tg.loc[f, "R2"]), float(t))
        _ROWS_GRID.append((mkt, used, _rt("CP"), _rt("COINT"), _rt("CiP"), rg))
# ============================================================ TABELLE LATEX (slide + tesi)
if _ROWS_FACT:
    _mkts = sorted({r[0] for r in _ROWS_FACT}, reverse=True)      # US prima, poi UK
    _facts = ["SLOPE", "CP", "CiP", "COINT"]
    with open(TABLES_DIR / "rq3_factors.tex", "w", encoding="utf-8") as f:
        f.write("%% generato da r3_predictability.py -- NON modificare a mano\n")
        f.write("{\\footnotesize\n\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write(" & SLOPE & CP & Cieslak--Povala & COINT & Cointegration \\\\\n\\midrule\n")
        for m in _mkts:
            f.write(m)
            for fa in _facts:
                hit = [r for r in _ROWS_FACT if r[0] == m and r[1] == fa]
                if not hit:
                    f.write(" & ---"); continue
                _, _, r2, t, _, _ = hit[0]
                st = _stars(t)
                f.write(f" & ${r2:.2f}^{{{st}}}\\,[{t:.1f}]$" if st else f" & ${r2:.2f}\\,[{t:.1f}]$")
            _any = [r for r in _ROWS_FACT if r[0] == m][0]
            f.write(f" & $r{{=}}{_any[4]}$, ADF ${_any[5]:.1f}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}}\n")
        f.write("\n{\\scriptsize $R^2$ of predictive regressions of the 12-month real excess "
                "return (averaged across maturities) on each factor, with $t$-HH(12) in "
                "brackets (Hansen--Hodrick, the inference Cochrane--Piazzesi use with "
                "overlapping data). The last column reports the Johansen rank on 1-year real "
                "forwards and the ADF $t$ of the first error-correction term. "
                "$^{*}$ 10\\%, $^{**}$ 5\\%, $^{***}$ 1\\%.}\n")
    print(f"  [tex] {TABLES_DIR / 'rq3_factors.tex'}")

if _ROWS_GRID:
    _mk = sorted({r[0] for r in _ROWS_GRID}, reverse=True)
    _grids = []
    for r in _ROWS_GRID:
        if r[1] not in _grids: _grids.append(r[1])
    with open(TABLES_DIR / "rq3_grid_robustness.tex", "w", encoding="utf-8") as f:
        f.write("%% generato da r3_predictability.py -- NON modificare a mano\n")
        f.write("{\\scriptsize\n\\begin{tabular}{l" + "cc c" * len(_mk) + "}\n\\toprule\n")
        f.write("Forward grid" + "".join(f" & \\multicolumn{{3}}{{c}}{{{m}}}" for m in _mk) + " \\\\\n")
        f.write("" + "".join(" & CP & COINT & $r$" for _ in _mk) + " \\\\\n\\midrule\n")
        for g in _grids:
            f.write("$" + ",".join(str(x) for x in g) + "$")
            for m in _mk:
                hit = [r for r in _ROWS_GRID if r[0] == m and r[1] == g]
                if not hit:
                    f.write(" & --- & --- & ---"); continue
                _, _, cp, co, _cip, rg = hit[0]
                for (r2, t) in (cp, co):
                    st = _stars(t)
                    f.write(f" & ${r2:.2f}^{{{st}}}[{t:.1f}]$" if st else f" & ${r2:.2f}\\,[{t:.1f}]$")
                f.write(f" & {rg}")
            f.write(" \\\\\n")
        f.write("\\midrule\n\\emph{CiP (reference)}")
        for m in _mk:
            hit = [r for r in _ROWS_GRID if r[0] == m]
            if hit:
                r2, t = hit[0][4]
                f.write(f" & \\multicolumn{{3}}{{c}}{{\\emph{{{r2:.2f}\\,[{t:.1f}]}}}}")
            else:
                f.write(" & \\multicolumn{3}{c}{---}")
        f.write(" \\\\\n\\bottomrule\n\\end{tabular}}\n")
        f.write("\n{\\scriptsize CP and the cointegration predictor are both built from a "
                "vector of 1-year forward rates, so which forwards enter is a free choice: the "
                "table varies it. Cieslak--Povala is built from the inflation trend and cycle, "
                "does not use forwards, and is unchanged by construction. $r$ is the Johansen "
                "rank, which moves with the grid because the dimension of the system moves.}\n")
    print(f"  [tex] {TABLES_DIR / 'rq3_grid_robustness.tex'}")

print("\nsalvato: an_predictability_{UK,US}.csv")
print("Lettura per RR: se COINT ~ CP (R2 simili) sui linker come sui nominali, la")
print("storia Rebonato-Nyholm si estende ai reali; se CiP domina, contano le sorprese")
print("d'inflazione anche nel real space (coerente col loro risultato sui TIPS).")

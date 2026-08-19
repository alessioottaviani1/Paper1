"""r3 - Predicibilita' degli excess return REALI (richiesta RR n.3): slope vs CP vs
Cieslak-Povala vs predittore di COINTEGRAZIONE diretto tra i forward (Rebonato-Nyholm
JEF 2025: il potere di CP sta nell'identificare la combinazione stazionaria dei forward
quasi-unit-root; la cointegrazione fornisce predittori direttamente).
Excess return da curve zero (cc), holding 12m overlapping, funding = reale piu' corto
disponibile (dichiarato); inferenza Newey-West (18 lag). Output per mercato: tabella
beta/t/R2 per maturita' e per rx medio + t-ADF del residuo di cointegrazione.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import pandas as pd
import rp
from config import CACHE

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
    print("  t-NW su rx(media): " + "  ".join(f"{i}={v:+.1f}" for i, v in tm.t_NW.items()))

    # ---- ROBUSTEZZA: il risultato COINT/CP dipende dalla griglia dei forward? ----
    # Griglie alternative (tutte coperte dalle curve reali); per ognuna R2 e t di
    # CP, COINT (e CiP, che non dipende dai forward, come riferimento) su rx medio.
    print("  robustezza griglia forward (R2 | t-NW su rx(media)):")
    print(f"    {'griglia':14s} {'CP':>14s} {'COINT':>14s} {'CiP':>14s}  rango")
    for grid in [(3,4,5,6,7), (4,5,6,7,8), (5,7,10), (4,6,8,10,12), (5,10,15)]:
        Fg = rp.forwards(y, mats=grid)
        if Fg.shape[1] < 2:
            continue
        cpg, _, _ = rp.cp_factor(rxb, Fg)
        cog, rg, _ = rp.coint_johansen_predictor(Fg, rxb)
        tg = rp.predict_table(rx, {"CP": cpg, "COINT": cog, "CiP": cip})
        tg = tg[tg.target == "rx(media)"].set_index("fattore")
        cell = lambda f: (f"{tg.loc[f,'R2']:.3f}|{tg.loc[f,'t_NW']:+.1f}" if f in tg.index else "  n/a")
        used = tuple(int(c) for c in Fg.columns)
        print(f"    {str(used):14s} {cell('CP'):>14s} {cell('COINT'):>14s} {cell('CiP'):>14s}  r={rg}")
print("\nsalvato: an_predictability_{UK,US}.csv")
print("Lettura per RR: se COINT ~ CP (R2 simili) sui linker come sui nominali, la")
print("storia Rebonato-Nyholm si estende ai reali; se CiP domina, contano le sorprese")
print("d'inflazione anche nel real space (coerente col loro risultato sui TIPS).")

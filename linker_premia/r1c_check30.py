"""r1c - Perche' manca il 30y pre-2009? Diagnostica delle quattro fonti, nodo per nodo.

Nella tabella di r1 il 30y ha 127 osservazioni contro le 264 delle altre scadenze, il
premio US salta da -23 a -5 bp, e nel sottoperiodo pre-2009 il 30y UK e' vuoto. Prima di
interpretare quel nodo -- o di escluderlo -- va stabilito QUALE delle quattro fonti si
interrompe: la curva reale, quella nominale, o gli swap d'inflazione. Le tre cause hanno
implicazioni diverse: se mancano gli swap il premio non e' calcolabile per definizione,
se manca la curva lunga il nodo e' estrapolato e va escluso comunque.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inflation_linked"))
import pandas as pd
import rp

MATS = (3., 5., 7., 10., 15., 20., 30.)

def cop(s: pd.Series):
    s = s.dropna()
    return (len(s), s.index.min(), s.index.max()) if len(s) else (0, None, None)

if __name__ == "__main__":
    print(">>> r1c copertura per fonte e per nodo <<<\n")
    fonti = {
        "BEI US (GSW nom-tips)": lambda m: rp.bei_us((m,))[m],
        "BEI UK (BoE infl)":     lambda m: rp.bei_uk((m,))[m],
        "ISR US (USSWIT)":       lambda m: rp.isr("US", (m,))[m],
        "ISR UK (BPSWIT)":       lambda m: rp.isr("UK", (m,))[m],
    }
    print(f"  {'fonte':24s}" + "".join(f"{int(m):>7}y" for m in MATS))
    inizio = {}
    for nome, f in fonti.items():
        riga = f"  {nome:24s}"
        for m in MATS:
            try:
                n, a, b = cop(f(m))
                riga += f"{(a.year if a else 0):>8}"
                inizio.setdefault(nome, {})[m] = a
            except Exception:
                riga += f"{'--':>8}"
                inizio.setdefault(nome, {})[m] = None
        print(riga)
    print("  (anno di INIZIO della serie; -- = nodo non nella curva)")

    print(f"\n  {'fonte':24s}" + "".join(f"{int(m):>7}y" for m in MATS) + "   [n osservazioni]")
    for nome, f in fonti.items():
        riga = f"  {nome:24s}"
        for m in MATS:
            try: riga += f"{cop(f(m))[0]:>8,}"
            except Exception: riga += f"{'--':>8}"
        print(riga)

    print("\n=== il vincolo: chi taglia il campione comune, nodo per nodo ===")
    for m in MATS:
        try:
            date = {}
            for nome, f in fonti.items():
                s = f(m).dropna()
                if len(s): date[nome] = s.index.min()
            if not date: print(f"  {int(m):>3}y: nessuna fonte"); continue
            tardi = max(date, key=date.get)
            comune = max(date.values())
            print(f"  {int(m):>3}y: campione comune parte dal {comune.date()}  "
                  f"-- vincolo: {tardi}")
        except Exception as e:
            print(f"  {int(m):>3}y: errore {str(e)[:40]}")
    print("\n  Se il vincolo sul 30y e' uno SWAP, il premio li' non e' calcolabile prima")
    print("  di quella data: e' un limite del dato, non una scelta. Se e' una CURVA, il")
    print("  nodo lungo va escluso perche' sarebbe estrapolato.")

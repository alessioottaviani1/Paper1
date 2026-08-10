"""10 - LA SUPERFICIE: il C3 su TUTTO il cubo swaption (16 celle x mercato), non solo 3Mx10Y.

Perche' conta. Il C3 finora usa due celle (3Mx10Y, 1Yx10Y). Ma il cubo ha 4 expiry x 4 tenor
per valuta, gia' scaricato: 8 volte piu' evidenza a costo zero. Due risultati che solo lo scan
completo puo' dare:

  (1) UCCIDE l'obiezione ``oggetti diversi''. sigma_BE del fly 2/10/30 e' dominata dal 10Y:
      quindi le celle con TAIL 10Y sono le MEGLIO APPAIATE. Se persino quelle non co-muovono,
      il quasi-zero non e' un artefatto di appaiamento sbagliato. Il test e' pre-specificato:
      se il co-movimento fosse un problema di matching, dovrebbe SALIRE sul tail 10Y.
  (2) TERM STRUCTURE della segmentazione: il co-movimento varia per expiry/tenor? Se la
      segmentazione e' un fatto di clientele (non di misura), il pattern deve essere
      sistematico per mercato, non casuale cella per cella.

Output: matrice 4x4 per mercato (Delta-corr), piu' i marginali per expiry e per tenor.
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import save_txt, load_vols

print("== 10 surface ==")
sbe = pd.read_csv(PROC/"sigbe_monthly.csv", index_col=0, parse_dates=True)
IV = load_vols()
EXPS = ["3M", "6M", "1Y", "2Y"]
TENS = ["2Y", "5Y", "10Y", "30Y"]
FAM = "NORM"   # famiglia normale: unita' confrontabili (bp/yr), niente esplosioni a tassi zero

L = []; P = L.append
P("=== 10 SUPERFICIE: C3 (Delta-corr) su tutto il cubo swaption ===")
P(f"famiglia {FAM} | expiry {EXPS} | tenor {TENS}")
P("")

allcells = {}
for mkt in sbe.columns:
    ccy = IVMAP.get(mkt)
    if ccy is None: continue
    b = sbe[mkt].dropna()
    M = pd.DataFrame(index=EXPS, columns=TENS, dtype=float)
    N = pd.DataFrame(index=EXPS, columns=TENS, dtype=float)
    for e in EXPS:
        for t in TENS:
            s = IV.get((ccy, e, t, FAM))
            if s is None: continue
            iv = s.resample("ME").last()
            al = pd.concat([b, iv], axis=1).dropna()
            if len(al) < 60: continue
            d = al.diff().dropna()
            M.loc[e, t] = d.iloc[:, 0].corr(d.iloc[:, 1])
            N.loc[e, t] = len(d)
    allcells[mkt] = M
    P(f"--- {mkt} ---")
    P(f"{'expiry':8}" + "".join(f"{t:>9}" for t in TENS) + f"{'media':>9}")
    for e in EXPS:
        row = "".join((f"{M.loc[e,t]:+9.2f}" if pd.notna(M.loc[e,t]) else f"{'--':>9}") for t in TENS)
        P(f"{e:8}{row}{M.loc[e].mean(skipna=True):+9.2f}")
    P(f"{'media':8}" + "".join(f"{M[t].mean(skipna=True):+9.2f}" for t in TENS)
      + f"{np.nanmean(M.values.astype(float)):+9.2f}")
    P("")

P("=== TEST DECISIVO: le celle col TAIL 10Y sono le meglio appaiate a sigma_BE (fly 2/10/30).")
P("Se il quasi-zero fosse un artefatto di appaiamento, il co-movimento dovrebbe SALIRE li'. ===")
P(f"{'mercato':9}{'tail 2Y':>10}{'tail 5Y':>10}{'tail 10Y':>10}{'tail 30Y':>10}{'|max-min|':>11}")
for mkt, M in allcells.items():
    v = [M[t].mean(skipna=True) for t in TENS]
    P(f"{mkt:9}" + "".join(f"{x:+10.2f}" for x in v) + f"{np.nanmax(v)-np.nanmin(v):11.2f}")
P("")
P("Lettura: se la colonna tail-10Y non e' sistematicamente piu' alta delle altre, l'obiezione")
P("'state confrontando oggetti diversi' e' esclusa: nemmeno l'appaiamento migliore co-muove.")

P("")
P("=== ampiezza dell'evidenza: quante celle su quante hanno |corr| < 0.20? ===")
tot = 0; small = 0
for mkt, M in allcells.items():
    v = M.values.astype(float); v = v[~np.isnan(v)]
    tot += len(v); small += int((np.abs(v) < 0.20).sum())
    P(f"{mkt:9}{int((np.abs(v)<0.20).sum()):3d}/{len(v):<3d} celle con |corr|<0.20 | mediana {np.median(v):+.2f} | max |corr| {np.max(np.abs(v)):.2f}")
P(f"{'TOTALE':9}{small:3d}/{tot:<3d} celle ({100*small/tot:.0f}%)")
save_txt("10_surface.txt", L); print("\n".join(L))

"""11 - STRATEGIA CONDIZIONATA ALLO STATO DI BILANCIO (implementabile, senza look-ahead).

Motivazione, che e' TEORICA e non di data mining. H4 dice che il premio e' la rendita del capitale
paziente negli stati in cui il bilancio e' scarso: il ladder (05) lo conferma (celle HIGH +20..+24
bp/m, 4/4 significative, 3/4 ex-crisi). Ma la strategia di 04 e' incondizionata --- prende posizione
sempre, anche negli stati LOW dove la teoria dice che NON si e' compensati. Questa e' la versione
che la teoria prescrive: prendere il segnale solo quando lo stato di stress dei dealer e' alto.

DISCIPLINA ANTI-LOOK-AHEAD (essenziale: il ladder di 05 usa terzili in-sample, quindi NON e'
implementabile; qui invece):
  - soglia di stress = quantile calcolato su FINESTRA ESPANDENTE con soli dati passati
    (minimo 36 mesi di storia prima di poter operare);
  - stato noto a t-1, posizione applicata a t.
Quindi ogni numero qui e' realizzabile in tempo reale. Riportiamo entrambe le versioni --- la
incondizionata (04) e la condizionata --- e l'inferenza a shift circolari esatti su entrambe.

Output: output/convexity/results/11_conditional.txt
"""
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from config import *
from utils import nw_t, save_txt, load_dealer_cds

print("== 11 conditional strategy ==")
STRAT = pd.read_csv(PROC/"strat_monthly.csv", index_col=0, parse_dates=True)
CDS, _, _ = load_dealer_cds()
MINH, Q = 36, 2/3          # storia minima (mesi) e quantile di stress
# DUE specifiche di soglia, riportate SEMPRE entrambe (la scelta e' un grado di liberta':
# esporlo e' l'unico modo onesto di trattarlo). ESPANDENTE = stress rispetto a TUTTA la storia
# (la barra la fissa il 2008 e non scende piu'); ROLLING = stress rispetto al REGIME RECENTE,
# piu' fedele all'idea che il vincolo dei dealer sia relativo al normale corrente.
SPECS = [("espandente", None), ("rolling-60m", 60)]

L = []; P = L.append
P("=== 11 STRATEGIA CONDIZIONATA (stress dealer alto, soglia espandente: NO look-ahead) ===")
P(f"soglia = quantile {Q:.2f} su finestra espandente (min {MINH} mesi di soli dati passati)")
P("")
for lab, W in SPECS:
    P(f"--- soglia {lab} ---")
    P(f"{'mercato':9}{'N_on':>6}{'quota':>7}{'incond.':>9}{'[t]':>6}   |{'COND.(attivi)':>14}{'[t]':>6}{'p-sh':>7}{'gain':>8}")
    for mkt in STRAT.columns:
        x = STRAT[mkt].dropna()
        st = CDS[CDSREGION[mkt]].reindex(x.index).ffill()
        thr = (st.expanding(min_periods=MINH) if W is None else st.rolling(W, min_periods=MINH)).quantile(Q).shift(1)
        on = (st.shift(1) >= thr)
        m = thr.notna(); xl = x[m]; onl = on[m]
        if len(xl) < 60: continue
        rv = xl.values; msk = onl.values.astype(float)
        if msk.sum() < 8: 
            P(f"{mkt:9}{int(msk.sum()):6d}{msk.mean():7.0%}   (troppi pochi mesi attivi)"); continue
        obs = (msk*rv).mean(); K = len(rv)
        psh = sum(((np.roll(msk, k)*rv).mean() >= obs) for k in range(1, K))/(K-1)
        act = xl[onl]
        P(f"{mkt:9}{int(msk.sum()):6d}{msk.mean():7.0%}{xl.mean():9.2f}{nw_t(xl):6.1f}   |"
          f"{act.mean():14.2f}{nw_t(act):6.1f}{psh:7.3f}{act.mean()-xl.mean():+8.2f}")
    P("")
P("")
P("LETTURA ONESTA. Con soglia ROLLING i rendimenti condizionati sono 2-4x gli incondizionati in")
P("4/4 mercati: il premio vive dove la teoria dice. Ma con soglia ESPANDENTE l'effetto svanisce,")
P("perche' la barra la fissa il 2008 e i mesi attivi diventano pochissimi. Il risultato e' quindi")
P("SENSIBILE ALLA SPECIFICA e i p-shift non scendono sotto 0.05: e' un exhibit di supporto, NON")
P("un headline, e va presentato con entrambe le specifiche in vista. Corollario importante:")
P("il ladder di 05 (terzili in-sample) NON e' implementabile in tempo reale --- e' una descrizione")
P("condizionale di dove vive il premio (evidenza sul MECCANISMO), non una strategia tradabile.")
save_txt("11_conditional.txt", L); print("\n".join(L))

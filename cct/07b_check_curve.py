"""
07b - CONTROLLO: la base e' dislocazione vera o errore di fit della curva?

PERCHE'. La base collassa nel 2011-12, che e' anche il periodo in cui il fit della curva e'
piu' a rischio: mercato dislocato, prezzi rumorosi, RMSE con p90 a 12 bp e massimo 56. Se i
giorni di crisi fossero anche quelli col fit peggiore, parte della "base" sarebbe errore di
misura invece che dislocazione. Il test e' diretto: correlare la base giornaliera con l'RMSE
della curva dello stesso giorno e confrontare il regime di crisi col resto del campione.

LETTURA. La correlazione da sola NON discrimina: base e RMSE hanno una causa comune, perche'
in un mercato dislocato i titoli si allontanano da qualsiasi curva liscia (RMSE sale) E il CCT
si allontana dal nominale (base si allarga). Una correlazione anche forte e' quindi attesa.
Il test che discrimina e' il secondo: se la base fosse errore di curva, restringendo ai giorni
con fit MIGLIORE dovrebbe collassare verso zero. Se invece resta stabile, la dislocazione e'
reale. Attenzione alla numerosita' residua: ai quantili bassi il campione di crisi si assottiglia
e il test perde potenza -- le righe informative sono p90 e p75, non p50.
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

print("== 07b check curva ==")
B = pd.read_csv(PROC/"basis_daily.csv", parse_dates=["date"])
C = pd.read_csv(PROC/"curve_params.csv", index_col=0, parse_dates=True)

L=[]; P=L.append
P("=== 07b LA BASE E' DISLOCAZIONE O ERRORE DI CURVA? ===")
for reg in ["CCT-BOT", "CCTeu", "TUTTI"]:
    g = B if reg == "TUTTI" else B[B.regime == reg]
    if g.empty: continue
    d = (g.groupby("date")[["basis3_p", "basis3_y", "basis1_y"]].median()
          .join(C[["rmse_bp", "n_bonds"]], how="inner").dropna(subset=["rmse_bp"]))
    if len(d) < 100: continue
    P(f"\n--- {reg} ({len(d):,} giorni) ---")
    for c in ["basis3_p", "basis3_y", "basis1_y"]:
        r = d[c].corr(d["rmse_bp"])
        P(f"  corr({c:9s}, RMSE curva) = {r:+.3f}")
    crisis = d.loc["2011":"2012"]
    rest = d.drop(crisis.index, errors="ignore")
    if len(crisis) and len(rest):
        P(f"  RMSE mediano  2011-12: {crisis.rmse_bp.median():6.2f} bp   |  resto: {rest.rmse_bp.median():6.2f} bp"
          f"   rapporto {crisis.rmse_bp.median()/max(rest.rmse_bp.median(),1e-9):.2f}x")
        P(f"  titoli nel fit 2011-12: {crisis.n_bonds.median():.0f}  |  resto: {rest.n_bonds.median():.0f}")
        P(f"  base mediana   2011-12: {crisis.basis3_p.median():+7.3f}  |  resto: {rest.basis3_p.median():+7.3f}")

# la base sopravvive escludendo i giorni con fit scadente?
P("\n=== ROBUSTEZZA: base ricalcolata escludendo i giorni con fit peggiore ===")
d = (B.groupby("date")[["basis3_p"]].median().join(C[["rmse_bp"]], how="inner").dropna())
for q in [1.00, 0.90, 0.75, 0.50]:
    thr = d.rmse_bp.quantile(q)
    w = d[d.rmse_bp <= thr]
    c11 = w.loc["2011":"2012", "basis3_p"]
    P(f"  tieni RMSE <= p{int(q*100):3d} ({thr:5.2f} bp): n {len(w):,} | mediana globale "
      f"{w.basis3_p.median():+.3f} | 2011-12 {c11.median() if len(c11) else float('nan'):+.3f} (n {len(c11):,})")
P("\n  Se la base del 2011-12 resta ampiamente negativa anche tenendo solo i giorni con")
P("  fit migliore (p50), la dislocazione e' reale e non un artefatto di stima.")
save_txt("07b_check_curve.txt", L); print("\n".join(L))

"""
05b - Verifica della formula cedolare dei CCT-BOT contro il dato Bloomberg.

LA DOMANDA. La scheda MEF dice: "si considera il rendimento lordo semplice annuo registrato
sui BOT a sei mesi nell'ultima asta che precede il godimento della cedola, si moltiplica per
0,5 e si somma lo spread". Letteralmente:  cedola_semestrale = 0.5*y_BOT + s,  quindi in
termini ANNUI  y_BOT + 2s: lo spread pesa doppio. L'alternativa  0.5*(y_BOT + s)  darebbe
annuo  y_BOT + s. La differenza e' lo spread stesso (15, 30 o 50 bp secondo l'epoca): non e'
un dettaglio, entra dritta nella cedola del sintetico e quindi nella base.

IL TEST. Bloomberg riporta in CPN la cedola CORRENTE del floater, su base ANNUA. Si confronta
quel valore con le due formule calcolate sul rendimento d'asta BOT vigente alla data, e si
vede quale delle due lo riproduce. E' una verifica diretta, non interpretativa.
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

L=[]; P=L.append
P("=== 05b VERIFICA DELLA FORMULA CEDOLARE CCT-BOT ===")
S  = pd.read_csv(PROC/"coupon_schedule.csv", parse_dates=["accr_start","pay_date","fixing_date"])
C  = pd.read_csv(PROC/"static_cct.csv", parse_dates=["maturity","issue"])
try:
    SB = pd.read_csv(PROC/"static_bbg.csv", index_col=0); SB.columns=[c.upper() for c in SB.columns]
except Exception:
    P("[STOP] static_bbg.csv assente"); save_txt("05b_verify.txt", L); raise SystemExit

if "CPN" not in SB.columns:
    P("[STOP] colonna CPN assente in static_bbg.csv"); save_txt("05b_verify.txt", L); raise SystemExit

cpn_bbg = pd.to_numeric(SB["CPN"], errors="coerce")
bot = C[C.regime == "CCT-BOT"]["isin"].tolist()
S = S[(S.rule == "B") & S.param_ann.notna()]        # regola post-1995, parametro noto

rows = []
for isin in bot:
    if isin not in cpn_bbg.index or not np.isfinite(cpn_bbg.get(isin, np.nan)): continue
    g = S[S["isin"] == isin]
    if g.empty: continue
    # Bloomberg riporta la cedola CORRENTE: si confronta con l'ULTIMA cedola determinata
    r = g.sort_values("accr_start").iloc[-1]
    y, sp = float(r.param_ann), float(r.spread_pct)
    fA = 2 * (0.5 * y + sp)        # MEF letterale -> annuo = y + 2s
    fB = 2 * (0.5 * (y + sp))      # alternativa   -> annuo = y + s
    rows.append({"isin": isin, "y_BOT": y, "spread": sp, "cpn_bbg": float(cpn_bbg[isin]),
                 "formulaA_y+2s": fA, "formulaB_y+s": fB,
                 "errA": abs(fA - float(cpn_bbg[isin])), "errB": abs(fB - float(cpn_bbg[isin]))})
D = pd.DataFrame(rows)
if D.empty:
    P("[STOP] nessun CCT-BOT con CPN Bloomberg e parametro d'asta disponibili")
else:
    P(f"CCT-BOT verificabili: {len(D)}")
    P(f"\n  errore assoluto mediano  formula A (y + 2s): {D.errA.median():.4f} punti pct")
    P(f"  errore assoluto mediano  formula B (y +  s): {D.errB.median():.4f} punti pct")
    win = "A (MEF letterale: 0.5*y + s)" if D.errA.median() < D.errB.median() else "B (0.5*(y+s))"
    P(f"\n  --> riproduce meglio il dato Bloomberg: FORMULA {win}")
    P(f"  vince A in {int((D.errA < D.errB).sum())}/{len(D)} titoli")
    P("\n  primi 12 titoli:")
    P(f"  {'isin':>14}{'y_BOT':>8}{'spr':>6}{'CPN bbg':>9}{'A=y+2s':>9}{'B=y+s':>8}")
    for _, r in D.head(12).iterrows():
        P(f"  {r['isin']:>14}{r.y_BOT:>8.3f}{r.spread:>6.2f}{r.cpn_bbg:>9.3f}"
          f"{r['formulaA_y+2s']:>9.3f}{r['formulaB_y+s']:>8.3f}")
    P("\n  [attenzione] il confronto e' valido solo se la cedola Bloomberg si riferisce")
    P("  allo stesso periodo dell'ultima riga dello scadenzario. Titoli gia' scaduti")
    P("  possono riportare l'ultima cedola pagata: guardare il verdetto sulla MEDIANA,")
    P("  non sul singolo titolo.")
save_txt("05b_verify.txt", L); print("\n".join(L))

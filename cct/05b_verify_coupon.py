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

def round_step(x, step=CCT_ROUND_STEP):
    return np.round(np.asarray(x, float) / step) * step

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
Cx = C.set_index("isin")
today = pd.Timestamp(END_SAMPLE)
bot = C[C.regime == "CCT-BOT"]["isin"].tolist()
S = S[(S.rule == "B") & S.param_ann.notna()]        # regola post-1995, parametro noto

rows = []
for isin in bot:
    if isin not in cpn_bbg.index or not np.isfinite(cpn_bbg.get(isin, np.nan)): continue
    g = S[S["isin"] == isin]
    if g.empty: continue
    # Bloomberg riporta la cedola CORRENTE. Per un titolo VIVO l'ultima riga dello
    # scadenzario con fixing <= oggi e' quella giusta; per uno SCADUTO il CPN Bloomberg
    # e' l'ultima cedola pagata (periodo diverso da param_ann) -> confronto inaffidabile.
    mat = Cx.loc[isin, "maturity"] if isin in Cx.index else pd.NaT
    alive = bool(pd.notna(mat) and mat > today)
    gg = g[g.fixing_date <= today] if alive else g
    if gg.empty: gg = g
    r = gg.sort_values("accr_start").iloc[-1]
    y, sp = float(r.param_ann), float(r.spread_pct)
    cpn_semi_A = round_step(0.5 * y + sp)     # cedola SEMESTRALE arrotondata (come in produzione)
    cpn_semi_B = round_step(0.5 * (y + sp))
    fA, fB = 2 * cpn_semi_A, 2 * cpn_semi_B   # annualizzata per il confronto col CPN Bloomberg
    fA_raw = 2 * (0.5 * y + sp)               # senza arrotondamento (per vedere se il gap e' lo step)
    cb = float(cpn_bbg[isin])
    rows.append({"isin": isin, "alive": alive, "y_BOT": y, "spread": sp, "cpn_bbg": cb,
                 "formulaA_y+2s": fA, "formulaB_y+s": fB, "A_noround": fA_raw,
                 "resA": fA - cb,                     # FIRMATO: segno del bias
                 "errA": abs(fA - cb), "errB": abs(fB - cb),
                 "err_noround": abs(fA_raw - cb)})
D = pd.DataFrame(rows)
if D.empty:
    P("[STOP] nessun CCT-BOT con CPN Bloomberg e parametro d'asta disponibili")
else:
    P(f"CCT-BOT verificabili: {len(D)}  (vivi {int(D.alive.sum())}, scaduti {int((~D.alive).sum())})")
    P(f"\n  errore assoluto mediano  formula A (y + 2s): {D.errA.median():.4f} punti pct")
    P(f"  errore assoluto mediano  formula B (y +  s): {D.errB.median():.4f} punti pct")
    win = "A (MEF letterale: 0.5*y + s)" if D.errA.median() < D.errB.median() else "B (0.5*(y+s))"
    P(f"\n  --> riproduce meglio il dato Bloomberg: FORMULA {win}")
    P(f"  vince A in {int((D.errA < D.errB).sum())}/{len(D)} titoli")
    # --- il test decisivo: SOLO I VIVI, dove il CPN Bloomberg e' la cedola corrente ---
    Dv = D[D.alive]
    if len(Dv):
        P(f"\n  [SOLO VIVI, n={len(Dv)}] errore mediano A: {Dv.errA.median():.4f}  "
          f"(se < 0.025 = mezzo step, la formula B e' validata)")
        P(f"  [SOLO VIVI] residuo FIRMATO A (formula - bbg): mediana {Dv.resA.median():+.4f}, "
          f"media {Dv.resA.mean():+.4f}")
        P(f"  [SOLO VIVI] errore mediano SENZA arrotondamento: {Dv.err_noround.median():.4f}")
        P("    (se l'errore con arrotondamento << senza, il gap era solo lo step 0.05;")
        P("     se il residuo firmato e' sistematico e non nullo, c'e' un bias nel y_BOT)")
    # scaduti separati: confronto inaffidabile, mostrato solo per completezza
    Dd = D[~D.alive]
    if len(Dd):
        P(f"\n  [SCADUTI, n={len(Dd)}] errore mediano A: {Dd.errA.median():.4f}  "
          f"(inaffidabile: il CPN e' l'ultima cedola pagata, periodo != param_ann)")
    P("\n  primi 12 titoli:")
    P(f"  {'isin':>14}{'live':>5}{'y_BOT':>8}{'spr':>6}{'CPN bbg':>9}{'A=y+2s':>9}{'resA':>8}")
    for _, r in D.sort_values("alive", ascending=False).head(12).iterrows():
        P(f"  {r['isin']:>14}{('vivo' if r.alive else 'scad'):>5}{r.y_BOT:>8.3f}{r.spread:>6.2f}"
          f"{r.cpn_bbg:>9.3f}{r['formulaA_y+2s']:>9.3f}{r['resA']:>+8.3f}")
    P("\n  [attenzione] il confronto e' valido solo se la cedola Bloomberg si riferisce")
    P("  allo stesso periodo dell'ultima riga dello scadenzario. Titoli gia' scaduti")
    P("  possono riportare l'ultima cedola pagata: guardare il verdetto sulla MEDIANA,")
    P("  non sul singolo titolo.")

# ===================================================================================
# BLOCCO CCTeu: il test decisivo, perche' i CCTeu sono VIVI e il CPN Bloomberg e' la
# cedola CORRENTE. Formula C: annuo = round(Euribor6M, 3dec) + spread; cedola semestrale
# = annuo * (giorni/360). Bloomberg CPN e' il TASSO ANNUO corrente del floater.
# ===================================================================================
P("\n" + "=" * 72)
P("=== 05b-C VERIFICA CCTeu (titoli VIVI: test affidabile) ===")
P("=" * 72)
Se = pd.read_csv(PROC/"coupon_schedule.csv", parse_dates=["accr_start","pay_date","fixing_date"])
Se = Se[(Se.rule == "C") & Se.param_ann.notna() & Se.spread_pct.notna()]
today = pd.Timestamp(END_SAMPLE)
Cx = C.set_index("isin")
rowsC = []
for isin, g in Se.groupby("isin"):
    if isin not in cpn_bbg.index or not np.isfinite(cpn_bbg.get(isin, np.nan)): continue
    mat = Cx.loc[isin, "maturity"] if isin in Cx.index else pd.NaT
    if pd.isna(mat) or mat <= today: continue           # SOLO VIVI
    # la cedola corrente e' quella il cui periodo contiene 'today' (fixing <= today < pay)
    gg = g[g.fixing_date <= today].sort_values("accr_start")
    if gg.empty: continue
    r = gg.iloc[-1]
    eur, sp = float(r.param_ann), float(r.spread_pct)
    rate_ann = round(eur, 3) + sp                       # tasso ANNUO del floater (cio' che CPN riporta)
    cb = float(cpn_bbg[isin])
    rowsC.append({"isin": isin, "euribor6m": eur, "spread": sp,
                  "rate_ann": rate_ann, "cpn_bbg": cb, "res": rate_ann - cb})
DC = pd.DataFrame(rowsC)
if DC.empty:
    P("  [nota] nessun CCTeu vivo con CPN e parametro disponibili "
      "(serve euribor6m in curves_market.csv e FLT_SPREAD in static_bbg.csv)")
else:
    P(f"  CCTeu vivi verificabili: {len(DC)}")
    P(f"  errore assoluto mediano (round(Eur6M,3)+spread vs CPN): {DC.res.abs().median():.4f} punti pct")
    P(f"  residuo FIRMATO: mediana {DC.res.median():+.4f}, media {DC.res.mean():+.4f}")
    P(f"  |errore| < 0.010 in {int((DC.res.abs()<0.010).sum())}/{len(DC)} titoli "
      f"(entro il 3o decimale dell'arrotondamento Euribor)")
    P(f"\n  {'isin':>14}{'Eur6M':>9}{'spr':>7}{'rate':>8}{'CPN bbg':>9}{'res':>8}")
    for _, r in DC.head(15).iterrows():
        P(f"  {r['isin']:>14}{r.euribor6m:>9.3f}{r.spread:>7.3f}{r.rate_ann:>8.3f}"
          f"{r.cpn_bbg:>9.3f}{r.res:>+8.3f}")
    P("\n  [lettura] residuo ~0 = engine CCTeu validato sui titoli vivi. Un residuo pari")
    P("   allo SPREAD segnalerebbe che il CPN Bloomberg e' gia' comprensivo/escluso di spread;")
    P("   un residuo pari a Eur6M(oggi)-Eur6M(fixing) = fixing letto alla data sbagliata.")

save_txt("05b_verify.txt", L); print("\n".join(L))
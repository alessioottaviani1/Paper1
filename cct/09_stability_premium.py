"""
09 - IL TEST DEL MECCANISMO, importato da Fleckenstein-Longstaff (JFE 2020).

QUALE TEST E PERCHE'. Il loro contributo non e' misurare il premio: e' spiegarlo. La Sezione
9.1 e la Tabella 10 contengono il test che lega il premio alla sua causa presunta -- la
STABILITA' MARK-TO-MARKET. Il ragionamento e' semplice e forte: se un floater vale di piu'
perche' il suo prezzo oscilla poco, allora il premio deve essere tanto piu' grande quanto
piu' il floater e' stabile RISPETTO al titolo a tasso fisso con cui lo si confronta. Non e'
una correlazione qualsiasi: e' la quantita' che il meccanismo predice, e la predizione e'
falsificabile.

Loro trovano coefficiente 5.32 (t=4.47) contro i bill e 1.87 (t=4.52) contro le note, con
R2 aggiustato 0.56 e 0.51. Effetti fissi di mese e anno, errori standard clusterizzati per
titolo. Qui si replica la stessa specifica sui CCT.

DUE BLOCCHI, come nel paper:
  (A) Tabella 6 -- descrittivo: deviazione standard delle variazioni giornaliere di prezzo,
      CCT contro BTP appaiato, stratificata per vita residua. Serve a stabilire che il CCT
      E' effettivamente piu' stabile, e di quanto.
  (B) Tabella 10 -- il test: regressione panel della base mensile sulla DIFFERENZA fra la
      volatilita' del BTP e quella del CCT. Coefficiente positivo e significativo = il
      premio paga la stabilita'.

ATTENZIONE AL SEGNO. La base in PREZZO (basis3_p) e' positiva quando il CCT costa piu' del
suo fair value, cioe' quando e' CARO: e' l'analogo esatto del "premium" di FL. La base in bp
ha segno opposto. Qui si usa il prezzo, come loro.

IL CONFONDIMENTO DA NEUTRALIZZARE. Nel nostro campione sd_diff correla 0.99 con la vita
residua: e' quasi una misura della scadenza travestita da misura di volatilita'. Poiche' la
base ha una struttura a termine marcata (CCT caro sul corto, neutro sul lungo), una
regressione senza controlli restituisce un coefficiente negativo che riflette solo quella
struttura. FL non hanno il problema perche' i loro FRN sono TUTTI biennali all'emissione: la
scadenza varia poco. Il nostro campione va da 3 mesi a 8 anni, quindi la specifica va
irrobustita in tre modi, riportati tutti:
  (B1) grezza, come FL          -- confrontabile con il loro numero, ma confusa;
  (B2) con tau e tau^2          -- toglie la struttura a termine media;
  (B3) con EFFETTI FISSI di CCT -- identifica SOLO dalla variazione temporale entro titolo,
                                   che e' l'unica variazione non contaminata dalla scadenza;
  (B4) entro fascia di scadenza -- non parametrico, nessuna forma funzionale imposta.
Se il coefficiente resta negativo in B3 e B4, il risultato e' economico. Se si annulla o
cambia segno, era struttura a termine.

UN'AVVERTENZA CHE VA DETTA. Nei nostri dati il CCT risulta caro soprattutto nel primo
periodo e ECONOMICO nelle crisi: se il premio di stabilita' fosse il meccanismo dominante ci
aspetteremmo il contrario, perche' e' proprio nello stress che la stabilita' vale. Il test
serve quindi anche a stabilire se il meccanismo americano vale in Italia o se qui domina
altro (rischio sovrano, liquidita' del gemello). Un esito negativo e' informativo quanto uno
positivo, e va scritto come tale.

Output: results/09_stability_premium.txt, PROC/stability_panel.csv
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

BUCKETS = [(0.25,0.5),(0.5,1),(1,1.5),(1.5,2),(2,3),(3,4),(4,5),(5,8)]

if __name__ == "__main__":
    print("== 09 premio per la stabilita' ==")
    B = pd.read_csv(PROC/"basis_daily.csv", parse_dates=["date"])
    PXC = pd.read_csv(PROC/"px_cct.csv", index_col=0, parse_dates=True)
    PXB = pd.read_csv(PROC/"px_btp.csv", index_col=0, parse_dates=True)

    # variazioni giornaliere di prezzo pulito, come FL (Tabella 6: "based on clean prices")
    dC, dB = PXC.diff(), PXB.diff()

    rows = []
    for (cct, btp, reg), g in B.groupby(["CCT_ISIN","BTP_ISIN","regime"]):
        if cct not in dC.columns or btp not in dB.columns: continue
        g = g.set_index("date").sort_index()
        m = g.resample("ME").agg(basis_p=("basis3_p","mean"), basis_y=("basis3_y","mean"),
                                 tau=("tau_cct","mean"), n=("basis3_p","size"))
        sc = dC[cct].resample("ME").std()
        sb = dB[btp].resample("ME").std()
        m = m.join(sc.rename("sd_cct")).join(sb.rename("sd_btp")).dropna()
        m = m[m.n >= 10]
        if m.empty: continue
        m["cct"], m["btp"], m["regime"] = cct, btp, reg
        rows.append(m.reset_index())
    P0 = pd.concat(rows, ignore_index=True)
    P0["sd_diff"] = P0.sd_btp - P0.sd_cct          # positivo = il BTP oscilla di piu'
    P0["ym"] = P0.date.dt.to_period("M")
    P0.to_csv(PROC/"stability_panel.csv", index=False)

    L=[]; P=L.append
    P("=== 09 IL PREMIO PAGA LA STABILITA'? (test di Fleckenstein-Longstaff 2020) ===")
    P(f"panel mensile: {len(P0):,} osservazioni | {P0.cct.nunique()} CCT | "
      f"{P0.date.min().date()} -> {P0.date.max().date()}")

    P("\n" + "="*72)
    P("(A) TABELLA 6 -- il CCT e' davvero piu' stabile del BTP appaiato?")
    P("    deviazione standard delle variazioni giornaliere di prezzo, per vita residua")
    P("="*72)
    P(f"  {'vita residua':>14}{'sd CCT':>10}{'sd BTP':>10}{'rapporto':>10}{'n mesi':>9}")
    for lo,hi in BUCKETS:
        w = P0[(P0.tau>=lo)&(P0.tau<hi)]
        if len(w) < 30: continue
        P(f"  {f'{lo}-{hi}y':>14}{w.sd_cct.median():>10.3f}{w.sd_btp.median():>10.3f}"
          f"{w.sd_btp.median()/max(w.sd_cct.median(),1e-9):>10.2f}{len(w):>9,}")
    P("  [FL trovano il floater 2-3x piu' stabile del replicante, con il rapporto che")
    P("   cresce con la scadenza: e' la premessa del meccanismo.]")

    P("\n" + "="*72)
    P("(B) TABELLA 10 -- la base cresce dove il CCT e' PIU' stabile del BTP?")
    P("    base mensile in prezzo ~ (sd BTP - sd CCT), effetti fissi mese e anno,")
    P("    errori clusterizzati per CCT. Coefficiente POSITIVO = il premio paga la stabilita'.")
    P("="*72)
    P(f"  ATTENZIONE: corr(sd_diff, tau) = {P0.sd_diff.corr(P0.tau):.3f} nel nostro panel.")
    P( "  sd_diff e' quasi una misura della scadenza: senza controlli la regressione")
    P( "  restituisce la struttura a termine della base, non il premio di stabilita'.")
    try:
        import statsmodels.formula.api as smf
        def run(d, formula, lab):
            try:
                r = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["cct"]})
                b, t = r.params.get("sd_diff", np.nan), r.tvalues.get("sd_diff", np.nan)
                P(f"  {lab:34s} coef {b:+8.4f}  t {t:+6.2f}  R2adj {r.rsquared_adj:5.3f}  n {int(r.nobs):,}")
                return b, t
            except Exception as e:
                P(f"  {lab:34s} fallita ({str(e)[:50]})"); return np.nan, np.nan
        for lab, sub in [("TUTTI", P0), ("CCT-BOT", P0[P0.regime=="CCT-BOT"]),
                         ("CCTeu", P0[P0.regime=="CCTeu"])]:
            if len(sub) < 100: continue
            d = sub.copy()
            d["mon"]=d.date.dt.month.astype(str); d["yr"]=d.date.dt.year.astype(str)
            P(f"\n  --- {lab} ---")
            run(d, "basis_p ~ sd_diff + C(mon) + C(yr)", "(B1) grezza, come FL")
            run(d, "basis_p ~ sd_diff + tau + I(tau**2) + C(mon) + C(yr)", "(B2) + tau, tau^2")
            if d.cct.nunique() > 3:
                run(d, "basis_p ~ sd_diff + C(cct) + C(mon) + C(yr)", "(B3) + effetti fissi di CCT")
        P("\n  [FL: coef +5.32 (t 4.47) vs bill, +1.87 (t 4.52) vs note, senza controlli di")
        P("   scadenza perche' i loro FRN sono tutti biennali]")

        P("\n  (B4) ENTRO FASCIA DI SCADENZA -- nessuna forma funzionale imposta:")
        P(f"  {'fascia':>12}{'coef':>10}{'t':>8}{'n':>8}")
        for lo,hi in [(0.5,1.5),(1.5,3),(3,5),(5,8)]:
            d = P0[(P0.tau>=lo)&(P0.tau<hi)].copy()
            if len(d) < 150 or d.cct.nunique() < 4: continue
            d["mon"]=d.date.dt.month.astype(str); d["yr"]=d.date.dt.year.astype(str)
            try:
                r = smf.ols("basis_p ~ sd_diff + C(mon) + C(yr)", data=d).fit(
                        cov_type="cluster", cov_kwds={"groups": d["cct"]})
                P(f"  {f'{lo}-{hi}y':>12}{r.params.get('sd_diff',np.nan):>10.4f}"
                  f"{r.tvalues.get('sd_diff',np.nan):>8.2f}{int(r.nobs):>8,}")
            except Exception: pass
        P("  [entro fascia la scadenza varia poco: se il coefficiente resta negativo QUI,")
        P("   il risultato e' economico e non struttura a termine]")
    except ImportError:
        P("  [statsmodels non disponibile]")

    P("\n" + "="*72)
    P("(C) controllo: la relazione regge anche ESCLUDENDO le crisi?")
    P("="*72)
    try:
        import statsmodels.formula.api as smf
        d = P0[~P0.date.dt.year.isin([2008,2009,2010,2011,2012,2020,2022,2023])].copy()
        if len(d) > 100:
            d["mon"]=d.date.dt.month.astype(str); d["yr"]=d.date.dt.year.astype(str)
            r = smf.ols("basis_p ~ sd_diff + C(mon) + C(yr)", data=d).fit(
                    cov_type="cluster", cov_kwds={"groups": d["cct"]})
            P(f"  ex-crisi: coef {r.params.get('sd_diff',np.nan):+8.4f}  "
              f"t {r.tvalues.get('sd_diff',np.nan):+6.2f}  n {int(r.nobs):,}")
            P("  [se il coefficiente sopravvive fuori dalle crisi, il meccanismo e' strutturale;")
            P("   se sparisce, la base e' un fenomeno di stress e il premio di stabilita' non e'")
            P("   la spiegazione -- esito negativo ma informativo, da scrivere come tale.]")
    except Exception:
        pass
    P(f"\n[saved] {PROC/'stability_panel.csv'}")
    save_txt("09_stability_premium.txt", L); print("\n".join(L))

"""
10b - IL MECCANISMO CON I CDS: dal proxy alla misura diretta.

COSA CAMBIA RISPETTO A 10. In 10 lo stress della clientela era catturato da Euribor-OIS,
funding interbancario europeo: un proxy indiretto e per di piu' non italiano. Qui entra il
CDS delle banche italiane, che e' la misura diretta. E soprattutto entra il CDS SOVRANO,
che permette la scomposizione decisiva:

    premio Italia-sopra-swap  =  credito sovrano (CDS)  +  RESIDUO non-credito

Se la base si muove con il residuo e non con il CDS sovrano, allora il fenomeno NON e'
rischio di credito -- che comunque si cancellerebbe fra CCT e BTP, stesso emittente -- ma
frizione di mercato. E' la differenza fra un risultato banale e il risultato del paper.

INDICE DI STRESS BANCARIO. Mediana cross-sectional dei cinque CDS disponibili, invece di un
singolo nome: fusioni, aumenti di capitale e il caso MPS producono salti idiosincratici che
renderebbero fragile qualunque scelta singola.

TRE TEST:
  T1 - la base risponde allo stress BANCARIO, controllando per quello sovrano?
  T2 - risponde al residuo NON-CREDITO del premio Italia-sopra-swap?
  T3 - l'interazione con la dimensione (il canale clientela di 10) regge con i CDS al posto
       dei proxy? E' la replica del risultato chiave su una misura migliore.

Output: results/10b_mechanism_cds.txt
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

if __name__ == "__main__":
    print("== 10b meccanismo con CDS ==")
    L=[]; P=L.append
    P("=== 10b IL MECCANISMO CON I CDS ===")
    f = PROC/"extra_series.csv"
    if not f.exists():
        P("[STOP] manca extra_series.csv: lanciare prima 02b_pull_extra.py")
        save_txt("10b_mechanism_cds.txt", L); print("\n".join(L)); raise SystemExit
    X = pd.read_csv(f, index_col=0, parse_dates=True)
    M = pd.read_csv(PROC/"mechanism_panel.csv", parse_dates=["date"])

    bankcols = [c for c in X.columns if c.startswith("cds_") and c != "cds_italy"]
    P(f"CDS bancari disponibili: {bankcols}")
    if bankcols:
        X["cds_banks"] = X[bankcols].median(axis=1)      # mediana di settore
        P(f"  indice di settore (mediana di {len(bankcols)}): "
          f"{X.cds_banks.first_valid_index().date()} -> {X.cds_banks.last_valid_index().date()}, "
          f"mediana {X.cds_banks.median():.0f} bp, max {X.cds_banks.max():.0f}")
    if "cds_italy" in X:
        P(f"  CDS sovrano Italia: mediana {X.cds_italy.median():.0f} bp, max {X.cds_italy.max():.0f} bp")

    Xm = X.resample("ME").mean(); Xm["ym"] = Xm.index.to_period("M")
    M["ym"] = M.date.dt.to_period("M")
    keep = [c for c in ["cds_banks","cds_italy","bund5","bund10"] if c in Xm.columns] + ["ym"]
    M = M.merge(Xm[keep], on="ym", how="left")

    # scomposizione: il premio Italia-sopra-swap AL NETTO del credito sovrano
    if "cds_italy" in M.columns:
        d0 = M.dropna(subset=["sov_swap","cds_italy"])
        if len(d0) > 100:
            b = np.polyfit(d0.cds_italy/100.0, d0.sov_swap, 1)
            M["sov_resid"] = M.sov_swap - np.polyval(b, M.cds_italy/100.0)
            r2 = np.corrcoef(d0.cds_italy, d0.sov_swap)[0,1]**2
            P(f"\nscomposizione: sov_swap = {b[0]:.3f} x CDS + {b[1]:+.3f}   (R2 {r2:.3f})")
            P("  il residuo e' la parte del premio Italia-sopra-swap NON spiegata dal credito")

    try:
        import statsmodels.formula.api as smf
        def run(d, f, lab, keys):
            try:
                r=smf.ols(f,data=d).fit(cov_type="cluster",cov_kwds={"groups":d["CCT_ISIN"]})
                P(f"  {lab:38s} " + "  ".join(
                    f"{k}={r.params.get(k,np.nan):+.4f}[{r.tvalues.get(k,np.nan):+.2f}]" for k in keys)
                  + f"   n {int(r.nobs):,}")
            except Exception as e:
                P(f"  {lab:38s} fallita ({str(e)[:45]})")
        d = M.copy(); d["mon"]=d.date.dt.month.astype(str); d["yr"]=d.date.dt.year.astype(str)
        for c in ["cds_banks","cds_italy"]:
            if c in d: d[c+"_pp"] = d[c]/100.0        # da bp a punti percentuali

        P("\n" + "="*74)
        P("T1  lo stress BANCARIO muove la base, al netto di quello sovrano?")
        P("    segno atteso NEGATIVO su cds_banks_pp")
        P("="*74)
        if "cds_banks_pp" in d:
            w = d.dropna(subset=["cds_banks_pp"])
            run(w, "basis_p ~ cds_banks_pp + tau + I(tau**2) + C(mon)", "(1) solo CDS bancari", ["cds_banks_pp"])
            if "cds_italy_pp" in w:
                w2 = w.dropna(subset=["cds_italy_pp"])
                run(w2, "basis_p ~ cds_banks_pp + cds_italy_pp + tau + I(tau**2) + C(mon)",
                    "(2) bancari + sovrano", ["cds_banks_pp","cds_italy_pp"])
                run(w2, "basis_p ~ cds_banks_pp + cds_italy_pp + tau + I(tau**2) + C(CCT_ISIN) + C(mon)",
                    "(3) idem, eff. fissi CCT", ["cds_banks_pp","cds_italy_pp"])

        P("\n" + "="*74)
        P("T2  e' credito sovrano o FRIZIONE? il residuo non-credito e' quello che conta?")
        P("="*74)
        if "sov_resid" in d:
            w = d.dropna(subset=["sov_resid","cds_italy_pp"])
            run(w, "basis_p ~ sov_resid + cds_italy_pp + tau + I(tau**2) + C(mon)",
                "(4) residuo vs credito puro", ["sov_resid","cds_italy_pp"])
            P("  [se il residuo domina e il CDS sovrano no, la base e' frizione di mercato,")
            P("   non credito -- come dev'essere, dato che CCT e BTP hanno lo stesso emittente]")

        P("\n" + "="*74)
        P("T3  il canale clientela regge con i CDS al posto dei proxy?")
        P("    replica del risultato chiave di 10 su una misura migliore")
        P("="*74)
        if "cds_banks_pp" in d:
            w = d.dropna(subset=["cds_banks_pp","logamt"]).copy()
            if len(w) > 200:
                w["cds_c"]=w.cds_banks_pp-w.cds_banks_pp.mean()
                w["logamt_c"]=w.logamt-w.logamt.mean()
                run(w, "basis_p ~ cds_c*logamt_c + tau + I(tau**2) + C(mon)",
                    "(5) stress bancario x dimensione", ["cds_c","cds_c:logamt_c"])
                run(w, "basis_p ~ cds_c*logamt_c + tau + I(tau**2) + C(CCT_ISIN) + C(yr) + C(mon)",
                    "(6) idem, eff. fissi CCT e anno", ["cds_c","cds_c:logamt_c"])
                P("  [in 10 con i proxy: interazione -2.23 (t -2.49) con effetti fissi di anno]")
    except ImportError:
        P("[statsmodels non disponibile]")
    save_txt("10b_mechanism_cds.txt", L); print("\n".join(L))

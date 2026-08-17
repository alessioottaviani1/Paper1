"""
10c - Il meccanismo, ripulito da tre difetti emersi in 10b.

DIFETTO 1: COMPOSIZIONE VARIABILE DELL'INDICE BANCARIO. In 10b la mediana era su quattro
nomi, ma MPS entra solo nel dicembre 2012 -- in mezzo alla crisi che stiamo studiando. Una
mediana a composizione variabile ha salti artificiali quando un nome entra o esce, e qui il
salto cade nel punto peggiore. Qui l'indice principale usa i TRE nomi a storia lunga --
UniCredit, Intesa, Mediobanca, distinti e tutti dal 2001-02 -- con composizione costante;
MPS e Banco BPM restano come robustezza.

DIFETTO 2: COLLINEARITA'. In 10b il CDS bancario passa da -0.85 [t -3.84] da solo a +0.58
[t +2.84] con il sovrano accanto. Un'inversione di segno all'aggiunta di un regressore
correlato e' la firma della multicollinearita': in Italia CDS bancari e sovrano sono legati
dal doom loop. I due coefficienti non sono interpretabili separatamente. Qui il CDS bancario
viene ORTOGONALIZZATO sul sovrano: la componente residua e' lo stress bancario NON spiegato
dal sovrano, che e' l'unica parte su cui si possa dire qualcosa.

DIFETTO 3, IL PIU' SERIO: il CDS SOVRANO non dovrebbe muovere la base, perche' CCT e BTP
hanno lo STESSO emittente e il rischio di default si cancella. Che invece domini (-0.84,
t -3.81) ha due spiegazioni possibili, e sono distinguibili:
   (a) economica  - un canale che non abbiamo identificato;
   (b) MECCANICA  - per i CCTeu il sintetico paga s + K_eur, dove K_eur e' il tasso SWAP,
       ma lo sconto avviene sulla curva SOVRANA. Nel 2011-12 le due curve divergevano di
       centinaia di bp, e quella divergenza e' quasi per definizione il CDS sovrano.
Il test che discrimina e' gratuito: i CCT-BOT NON hanno il problema, perche' il loro
sintetico usa il par yield sovrano -- una curva sola, nessuna divergenza. Se l'effetto c'e'
anche li', e' economico. Se vive solo nei CCTeu, e' costruzione e va corretto.

Output: results/10c_mechanism_clean.txt
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

LONG = ["cds_unicredit", "cds_intesa", "cds_bpm"]   # UniCredit, Intesa, Mediobanca (BACRED)

if __name__ == "__main__":
    print("== 10c meccanismo ripulito ==")
    L=[]; P=L.append
    P("=== 10c IL MECCANISMO, RIPULITO ===")
    X = pd.read_csv(PROC/"extra_series.csv", index_col=0, parse_dates=True)
    M = pd.read_csv(PROC/"mechanism_panel.csv", parse_dates=["date"])

    have = [c for c in LONG if c in X.columns]
    P(f"indice bancario a COMPOSIZIONE COSTANTE su {len(have)} nomi: {have}")
    X["cds_b3"] = X[have].median(axis=1)
    cov = X[have].notna().all(axis=1)
    P(f"  finestra con tutti e {len(have)} i nomi presenti: {X.index[cov].min().date()} -> "
      f"{X.index[cov].max().date()}  ({cov.sum():,} giorni)")
    X.loc[~cov, "cds_b3"] = np.nan          # niente mediana a composizione ridotta
    allb = [c for c in X.columns if c.startswith("cds_") and c!="cds_italy"]
    X["cds_all"] = X[allb].median(axis=1)   # versione a 4 nomi, per robustezza
    P(f"  mediana 3 nomi: {X.cds_b3.median():.0f} bp (max {X.cds_b3.max():.0f}) | "
      f"4 nomi: {X.cds_all.median():.0f} bp")

    Xm = X.resample("ME").mean(); Xm["ym"] = Xm.index.to_period("M")
    M["ym"] = M.date.dt.to_period("M")
    M = M.merge(Xm[["cds_b3","cds_all","cds_italy","ym"]], on="ym", how="left")
    for c in ["cds_b3","cds_all","cds_italy"]:
        M[c+"_pp"] = M[c]/100.0

    # --- ortogonalizzazione: stress bancario NON spiegato dal sovrano
    d0 = M.dropna(subset=["cds_b3_pp","cds_italy_pp"])
    if len(d0) > 100:
        rho = d0.cds_b3_pp.corr(d0.cds_italy_pp)
        b = np.polyfit(d0.cds_italy_pp, d0.cds_b3_pp, 1)
        M["cds_bank_orth"] = M.cds_b3_pp - np.polyval(b, M.cds_italy_pp)
        P(f"\ncorr(CDS bancari, CDS sovrano) = {rho:+.3f}   [doom loop: collinearita' attesa]")
        P(f"  ortogonalizzazione: cds_bank = {b[0]:.3f} x sovrano {b[1]:+.3f}")
        P( "  il residuo e' lo stress bancario IDIOSINCRATICO, l'unico interpretabile a parte")

    try:
        import statsmodels.formula.api as smf
        def run(d, f, lab, keys):
            try:
                r=smf.ols(f,data=d).fit(cov_type="cluster",cov_kwds={"groups":d["CCT_ISIN"]})
                P(f"  {lab:36s} " + "  ".join(
                    f"{k}={r.params.get(k,np.nan):+.4f}[{r.tvalues.get(k,np.nan):+.2f}]" for k in keys)
                  + f"   n {int(r.nobs):,}")
            except Exception as e:
                P(f"  {lab:36s} fallita ({str(e)[:45]})")
        d = M.copy(); d["mon"]=d.date.dt.month.astype(str); d["yr"]=d.date.dt.year.astype(str)

        P("\n" + "="*76)
        P("TEST DECISIVO -- il CDS sovrano muove la base anche dove NON puo' farlo per")
        P("costruzione? I CCT-BOT usano una curva sola: li' l'effetto, se c'e', e' economico.")
        P("="*76)
        for reg in ["CCT-BOT", "CCTeu"]:
            w = d[(d.regime==reg)].dropna(subset=["cds_italy_pp"])
            if len(w) < 150: P(f"  {reg}: troppe poche osservazioni ({len(w)})"); continue
            nota = " [una curva: effetto NON meccanico]" if reg=="CCT-BOT" else " [due curve: possibile effetto meccanico]"
            P(f"\n  --- {reg}{nota}")
            run(w, "basis_p ~ cds_italy_pp + tau + I(tau**2) + C(mon)", "    solo CDS sovrano", ["cds_italy_pp"])
            run(w, "basis_p ~ cds_italy_pp + tau + I(tau**2) + C(CCT_ISIN) + C(mon)",
                "    + eff. fissi CCT", ["cds_italy_pp"])
        P("\n  [se il coefficiente e' simile nei due regimi -> economico;")
        P("   se e' molto piu' forte nei CCTeu -> in buona parte meccanico, e la misura")
        P("   CCTeu va ricostruita scontando i flussi sulla curva swap o rifacendo il")
        P("   sintetico con il par yield sovrano come per i CCT-BOT]")

        P("\n" + "="*76)
        P("STRESS BANCARIO IDIOSINCRATICO (ortogonale al sovrano)")
        P("  segno atteso NEGATIVO: banche sotto stress -> vendono CCT -> CCT si sconta")
        P("="*76)
        if "cds_bank_orth" in d:
            w = d.dropna(subset=["cds_bank_orth","cds_italy_pp"])
            run(w, "basis_p ~ cds_bank_orth + cds_italy_pp + tau + I(tau**2) + C(mon)",
                "  ortogonale + sovrano", ["cds_bank_orth","cds_italy_pp"])
            run(w, "basis_p ~ cds_bank_orth + cds_italy_pp + tau + I(tau**2) + C(CCT_ISIN) + C(yr) + C(mon)",
                "  + eff. fissi CCT e anno", ["cds_bank_orth","cds_italy_pp"])

        P("\n" + "="*76)
        P("CANALE CLIENTELA -- replica con l'indice a composizione costante")
        P("  e' il risultato che ha retto a tutto finora: -1.79 [t -2.37] in 10b")
        P("="*76)
        for lab, col in [("indice 3 nomi (costante)","cds_b3_pp"), ("indice 4 nomi","cds_all_pp")]:
            w = d.dropna(subset=[col,"logamt"]).copy()
            if len(w) < 200: continue
            w["cds_c"]=w[col]-w[col].mean(); w["logamt_c"]=w.logamt-w.logamt.mean()
            run(w, "basis_p ~ cds_c*logamt_c + tau + I(tau**2) + C(CCT_ISIN) + C(yr) + C(mon)",
                f"  {lab}", ["cds_c","cds_c:logamt_c"])
        P("\n  [se l'interazione regge con la composizione costante, il canale clientela e'")
        P("   il risultato solido del paper, indipendente dalle scelte di costruzione]")
    except ImportError:
        P("[statsmodels non disponibile]")
    save_txt("10c_mechanism_clean.txt", L); print("\n".join(L))

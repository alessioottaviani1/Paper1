"""15 - TEST DELLA TEORIA (habitat bancario + vincolo di arbitraggio).

La teoria: le banche italiane sono simultaneamente (i) la CLIENTELA DI HABITAT dei CCT
(domanda anelastica per l'asset a tasso variabile che immunizza il passivo a vista) e
(ii) l'ARBITRAGGISTA VINCOLATO (quando il loro capitale/collaterale si stringe, vendono).
Un solo agente, due regimi: in calma la domanda di habitat -> CCT cari; sotto vincolo ->
vendita -> CCT economici. L'asimmetria di segno (P1) emerge dal passaggio del vincolo da
inattivo ad attivo. Questo distingue la storia da FL (convenience yield di stabilita',
sempre positivo) e da uno slow-moving capital generico (solo allargamento sotto stress).

Due predizioni testabili SENZA dati nuovi (CDS in extra_series + HKM in raw/):

  TEST A -- l'asimmetria e' guidata dal VINCOLO BANCARIO, non dal calendario.
    Ordino i mesi per lo stato del vincolo (CDS bancari IT = vincolo DOMESTICO; HKM
    intermediary capital = vincolo GLOBALE) e verifico che l'effetto della dimensione
    passi da POSITIVO (vincolo lasco) a NEGATIVO (vincolo stretto). E -- il test che
    identifica il CANALE -- se conta di piu' il vincolo DOMESTICO o GLOBALE: se domina il
    domestico, la clientela vincolata e' italiana (le banche), non un intermediario generico.

  TEST E -- il DOOM LOOP. La base si sconta quando il nesso banca-sovrano si intensifica.
    Costruisco l'intensita' del doom loop come co-movimento condizionale fra CDS bancari e
    CDS sovrano (rolling corr) e come il LIVELLO congiunto; la base deve scontarsi quando il
    nesso e' forte, AL NETTO del solo stress sovrano. Collega il micro-fatto (base CCT) alla
    letteratura sovereign-bank nexus (Acharya-Steffen, Brunnermeier et al.).

Richiede: mechanism_panel.csv (10), extra_series.csv (02b), e He_Kelly_Manela_*.csv (in
raw/He_Kelly_Manela, copiato in PROC/hkm.csv -- vedi sotto).
Output: results/15_theory_tests.txt
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

LONG = ["cds_unicredit", "cds_intesa", "cds_bpm"]   # composizione costante (come 10c)
ONLY_CCTEU = True                                    # coerente col resto: solo l'arbitrabile


def _load_hkm():
    """HKM intermediary capital ratio, mensile. Cerca PROC/hkm.csv (copiare da raw/)."""
    p = PROC / "hkm.csv"
    if not p.exists():
        return None
    h = pd.read_csv(p)
    col = "yyyymm" if "yyyymm" in h.columns else h.columns[0]
    h["date"] = pd.to_datetime(h[col].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    return h.set_index("date")["intermediary_capital_ratio"]


if __name__ == "__main__":
    print("== 15 test della teoria (habitat bancario + vincolo) ==")
    L = []; P = L.append
    P("=== 15 TEST DELLA TEORIA: HABITAT BANCARIO + VINCOLO DI ARBITRAGGIO ===")
    M = pd.read_csv(PROC/"mechanism_panel.csv", parse_dates=["date"])
    X = pd.read_csv(PROC/"extra_series.csv", index_col=0, parse_dates=True)
    if ONLY_CCTEU:
        n0 = len(M); M = M[M.regime == "CCTeu"].copy()
        P(f"[ONLY_CCTEU] {n0:,} -> {len(M):,} oss. (solo l'arbitrabile)")

    # --- vincolo DOMESTICO: indice CDS bancari a composizione costante (come 10c) ---
    have = [c for c in LONG if c in X.columns]
    X["cds_bank"] = X[have].median(axis=1)
    cov = X[have].notna().all(axis=1); X.loc[~cov, "cds_bank"] = np.nan
    P(f"vincolo DOMESTICO = CDS bancari IT ({len(have)} nomi): mediana {X.cds_bank.median():.0f} bp")

    # --- vincolo GLOBALE: HKM intermediary capital ratio (basso = vincolo stretto) ---
    hkm = _load_hkm()
    if hkm is None:
        P("[!] PROC/hkm.csv assente. Copiare raw/He_Kelly_Manela/He_Kelly_Manela_Factors_"
          "monthly_*.csv in PROC/hkm.csv. Il TEST A_globale e' saltato; A_domestico procede.")

    # aggancio mensile
    Xm = X.resample("ME").mean(); Xm["ym"] = Xm.index.to_period("M")
    M["ym"] = M.date.dt.to_period("M")
    M = M.merge(Xm[["cds_bank", "cds_italy"]].assign(ym=Xm["ym"]), on="ym", how="left")
    if hkm is not None:
        hm = hkm.to_frame("hkm"); hm["ym"] = hm.index.to_period("M")
        M = M.merge(hm, on="ym", how="left")
    M["mon"] = M.date.dt.month.astype(str); M["yr"] = M.date.dt.year.astype(str)

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        P("[statsmodels assente]"); save_txt("15_theory_tests.txt", L); raise SystemExit

    def run(d, f, keys, lab):
        try:
            r = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["CCT_ISIN"]})
            P(f"  {lab:40s} " + "  ".join(
                f"{k}={r.params.get(k,np.nan):+.4f}[{r.tvalues.get(k,np.nan):+.2f}]" for k in keys)
              + f"   n {int(r.nobs):,}")
            return {k: (r.params.get(k, np.nan), r.tvalues.get(k, np.nan)) for k in keys}
        except Exception as e:
            P(f"  {lab:40s} fallita ({str(e)[:45]})"); return {}

    # ========================================================= TEST A
    P("\n" + "=" * 76)
    P("TEST A -- l'asimmetria di segno e' guidata dal VINCOLO BANCARIO?")
    P("  la teoria: dimensione POSITIVA quando il vincolo e' lasco (habitat: banche")
    P("  detengono i titoli grandi -> domanda -> cari), NEGATIVA quando e' stretto (le")
    P("  stesse banche vendono). Ordino i mesi per il vincolo, non per il calendario.")
    P("=" * 76)

    # A.1 -- terzili del vincolo DOMESTICO (CDS bancari)
    d = M.dropna(subset=["basis_p", "cds_bank", "logamt"]).copy()
    lo, hi = d.cds_bank.quantile(.33), d.cds_bank.quantile(.67)
    P(f"\n  [A.1 vincolo DOMESTICO -- CDS bancari IT] terzili: lasco <= {lo:.0f} bp, "
      f"stretto >= {hi:.0f} bp")
    resA = {}
    for lab, sub in [("vincolo LASCO  (CDS basso)", d[d.cds_bank <= lo]),
                     ("intermedio", d[(d.cds_bank > lo) & (d.cds_bank < hi)]),
                     ("vincolo STRETTO (CDS alto)", d[d.cds_bank >= hi])]:
        if len(sub) < 150: P(f"    {lab}: poche oss. ({len(sub)})"); continue
        w = sub.copy(); w["logamt_c"] = w.logamt - w.logamt.mean()
        resA[lab] = run(w, "basis_p ~ logamt_c + tau + I(tau**2) + C(mon)", ["logamt_c"], f"    {lab}")
    if "vincolo LASCO  (CDS basso)" in resA and "vincolo STRETTO (CDS alto)" in resA:
        bl = resA["vincolo LASCO  (CDS basso)"]["logamt_c"]
        bs = resA["vincolo STRETTO (CDS alto)"]["logamt_c"]
        ok = bl[0] > 0 and bs[0] < 0
        P(f"    >>> lasco {bl[0]:+.2f} (t{bl[1]:+.1f}) -> stretto {bs[0]:+.2f} (t{bs[1]:+.1f}): "
          f"{'ASIMMETRIA guidata dal vincolo domestico' if ok else 'segni non coerenti'}")

    # A.2 -- terzili del vincolo GLOBALE (HKM)
    if hkm is not None and "hkm" in M.columns:
        d = M.dropna(subset=["basis_p", "hkm", "logamt"]).copy()
        # capital ratio BASSO = vincolo stretto -> inverto per leggere come 'stress'
        lo, hi = d.hkm.quantile(.33), d.hkm.quantile(.67)
        P(f"\n  [A.2 vincolo GLOBALE -- HKM capital ratio] terzili: stretto (ratio basso) "
          f"<= {lo:.3f}, lasco (ratio alto) >= {hi:.3f}")
        resG = {}
        for lab, sub in [("vincolo STRETTO (ratio basso)", d[d.hkm <= lo]),
                         ("vincolo LASCO  (ratio alto)", d[d.hkm >= hi])]:
            if len(sub) < 150: continue
            w = sub.copy(); w["logamt_c"] = w.logamt - w.logamt.mean()
            resG[lab] = run(w, "basis_p ~ logamt_c + tau + I(tau**2) + C(mon)", ["logamt_c"], f"    {lab}")

    # A.3 -- il test del CANALE: domestico vs globale nella stessa interazione
    P("\n  [A.3 CANALE: domestico o globale?] interazione dimensione x vincolo, entrambi insieme")
    d = M.dropna(subset=["basis_p", "cds_bank", "logamt"]).copy()
    d["cdsb_c"] = d.cds_bank - d.cds_bank.mean(); d["logamt_c"] = d.logamt - d.logamt.mean()
    run(d, "basis_p ~ cdsb_c*logamt_c + tau + I(tau**2) + C(CCT_ISIN) + C(mon)",
        ["cdsb_c:logamt_c"], "    domestico (CDS banche IT) x size")
    if hkm is not None and "hkm" in M.columns:
        d2 = M.dropna(subset=["basis_p", "cds_bank", "hkm", "logamt"]).copy()
        d2["cdsb_c"] = d2.cds_bank - d2.cds_bank.mean()
        d2["hkm_c"] = d2.hkm - d2.hkm.mean(); d2["logamt_c"] = d2.logamt - d2.logamt.mean()
        run(d2, "basis_p ~ cdsb_c*logamt_c + hkm_c*logamt_c + tau + I(tau**2) + C(CCT_ISIN) + C(mon)",
            ["cdsb_c:logamt_c", "logamt_c:hkm_c"], "    domestico + globale, orizzonte comune")
        P("    [se l'interazione DOMESTICA sopravvive accanto alla GLOBALE, il vincolo che")
        P("     conta e' quello delle banche ITALIANE -> canale clientela domestico identificato]")

    # ========================================================= TEST E
    P("\n" + "=" * 76)
    P("TEST E -- il DOOM LOOP: la base si sconta quando il nesso banca-sovrano si intensifica?")
    P("  intensita' = co-movimento fra CDS bancari e CDS sovrano; la base deve scontarsi")
    P("  quando il nesso e' forte, AL NETTO del solo stress sovrano (sov_swap).")
    P("=" * 76)
    # costruisco l'intensita' del doom loop sui CDS giornalieri -> mensile
    dl = X[["cds_bank", "cds_italy"]].dropna()
    roll = dl["cds_bank"].rolling(63).corr(dl["cds_italy"])          # ~3 mesi
    lvl = (dl["cds_bank"] * dl["cds_italy"]) ** 0.5                    # livello congiunto (media geom.)
    DL = pd.DataFrame({"dl_corr": roll, "dl_level": lvl}).resample("ME").mean()
    DL["ym"] = DL.index.to_period("M")
    d = M.merge(DL, on="ym", how="left").dropna(subset=["basis_p", "dl_corr", "sov_swap"])
    P(f"  doom-loop corr(CDS banca, CDS sovrano): mediana {d.dl_corr.median():+.2f} | "
      f"livello congiunto mediana {d.dl_level.median():.0f} bp | n {len(d):,}")
    d["mon"] = d.date.dt.month.astype(str)
    run(d, "basis_p ~ dl_corr + tau + I(tau**2) + C(mon)", ["dl_corr"], "  (1) solo doom-loop corr")
    run(d, "basis_p ~ dl_corr + sov_swap + tau + I(tau**2) + C(mon)", ["dl_corr", "sov_swap"],
        "  (2) + stress sovrano (controllo)")
    run(d, "basis_p ~ dl_level + sov_swap + tau + I(tau**2) + C(mon)", ["dl_level", "sov_swap"],
        "  (3) livello congiunto + stress")
    run(d, "basis_p ~ dl_corr + sov_swap + tau + I(tau**2) + C(CCT_ISIN) + C(mon)",
        ["dl_corr"], "  (4) doom-loop, con eff. fissi CCT")
    P("  [segno atteso NEGATIVO su dl_corr/dl_level: nesso banca-sovrano forte -> le banche")
    P("   sono piu' vincolate e piu' esposte al sovrano -> vendono CCT -> base si sconta.")
    P("   Se sopravvive AL NETTO di sov_swap, il doom loop e' un canale distinto dal puro")
    P("   stress sovrano -> collega il micro-fatto al sovereign-bank nexus]")

    save_txt("15_theory_tests.txt", L); print("\n".join(L))

"""16 - TEST C: il canale COLLATERALE (haircut). La fonte di identificazione che i CDS
non danno, perche' l'haircut e' ESOGENO allo stress sovrano (e' una regola regolamentare).

LA TEORIA (meta' 'calma' del meccanismo). I CCT sono cari in regime normale perche' come
COLLATERALE valgono di piu': un floater prende l'haircut della fascia 0-1 anno dell'Eurosistema
A QUALSIASI scadenza (regola BCE, verbatim: 'haircuts applicable to marketable debt instruments
in liquidity categories I-IV with variable rate coupons will be those applicable to the 0-1 year
maturity bucket of fixed coupon instruments'), mentre il BTP prende l'haircut CRESCENTE della
sua scadenza. Il differenziale di haircut CCT-vs-BTP e' quindi POSITIVO e CRESCE con la
scadenza: un CCT lungo e' molto piu' efficiente come collaterale del BTP pari scadenza.

TAVOLA HAIRCUT BCE (Categoria I = titoli di Stato centrali, cedola fissa), da
ecb.europa.eu (Annex, verificato). Due schemi per qualita' di credito -- e l'Italia CAMBIA
step nel campione (A fino al 2011, BBB dalla crisi sovrana), quindi il differenziale varia
anche NEL TEMPO col rating, una seconda fonte di variazione oltre alla scadenza.

  fascia     Step 1-2 (AAA-A)   Step 3 (BBB)
  0-1y            0.5               5.5
  1-3y            1.5               6.5
  3-5y            2.5               7.5
  5-7y            3.0               8.0
  7-10y           4.0               9.0
  >10y            5.5              10.5

DUE TEST:
  C1 -- CROSS-SECTION: la ricchezza in CALMA (base negativa) e' proporzionale al
        differenziale di haircut CCT-vs-BTP? Segno atteso: differenziale alto -> CCT piu'
        caro -> base piu' NEGATIVA (in prezzo, positiva) -> coef NEGATIVO su hc_diff (bp).
        E' il test P3 reso QUANTITATIVO: non 'cresce con la scadenza' ma 'cresce QUANTO il
        differenziale di haircut', con la non-linearita' esatta dello schedule BCE.

  C2 -- ESPERIMENTO NATURALE: la BCE ha deciso di EQUALIZZARE gli haircut floater=fisso
        (riforma del collateral framework, in vigore da ~fine 2025/2026). Se la ricchezza
        in calma dei CCT dipende dal vantaggio di haircut, deve SVANIRE dopo la riforma.
        E' identificazione ESOGENA -- la riforma non ha nulla a che vedere con lo stress
        sovrano italiano. ATTENZIONE: verificare quanti mesi di post-riforma ci sono nel
        campione; se la riforma non e' ancora in vigore, C2 non e' testabile e si dichiara.

Richiede solo basis_daily.csv (07) + le date rating Italia (sotto, modificabili).
Output: results/16_haircut_test.txt
"""
import numpy as np, pandas as pd
from config import *
from utils import save_txt

# --- schedule haircut BCE, Categoria I, cedola fissa (bordi fascia in anni) ---
HC_STEP12 = [(0, 1, 0.5), (1, 3, 1.5), (3, 5, 2.5), (5, 7, 3.0), (7, 10, 4.0), (10, 99, 5.5)]
HC_STEP3  = [(0, 1, 5.5), (1, 3, 6.5), (3, 5, 7.5), (5, 7, 8.0), (7, 10, 9.0), (10, 99, 10.5)]
HC_FLOAT_STEP12 = 0.5    # floater = fascia 0-1y (pre-riforma)
HC_FLOAT_STEP3  = 5.5

# rating Italia (S&P/Moody's): A/step1-2 fino a fine 2011, poi BBB/step3 (declassamenti
# sovrani 2011-12; Italia resta BBB-area da allora). Modificabile se serve piu' granularita'.
IT_STEP3_FROM = pd.Timestamp("2011-10-01")   # da qui in poi step 3 (BBB)

# riforma haircut floater=fisso: in vigore da (BCE: 'no earlier than' Q4 2025 / Nov 2026).
# Metto una data prudenziale; il test riporta quante osservazioni cadono dopo.
HC_REFORM = pd.Timestamp("2025-10-01")

ONLY_CCTEU = True


def _hc_fixed(tau, step3):
    tbl = HC_STEP3 if step3 else HC_STEP12
    for lo, hi, h in tbl:
        if lo <= tau < hi:
            return h
    return tbl[-1][2]


def _hc_float(step3, post_reform, tau):
    if post_reform:                      # riforma: floater prende l'haircut del FISSO pari scadenza
        return _hc_fixed(tau, step3)
    return HC_FLOAT_STEP3 if step3 else HC_FLOAT_STEP12


if __name__ == "__main__":
    print("== 16 test C: canale collaterale (haircut) ==")
    L = []; P = L.append
    P("=== 16 TEST C -- IL CANALE COLLATERALE (HAIRCUT ECB) ===")
    B = pd.read_csv(PROC/"basis_daily.csv", parse_dates=["date"])
    if ONLY_CCTEU:
        n0 = len(B); B = B[B.regime == "CCTeu"].copy()
        P(f"[ONLY_CCTEU] {n0:,} -> {len(B):,} oss.")

    # scadenza del CCT e del BTP appaiato
    tau_c = "tau_cct" if "tau_cct" in B.columns else "tau"
    if tau_c not in B.columns:
        P("[!] manca la colonna scadenza (tau_cct/tau) in basis_daily. STOP.")
        save_txt("16_haircut_test.txt", L); raise SystemExit
    # per il BTP: se non c'e' tau_btp, uso quella del CCT (coppie a scadenza ravvicinata)
    tau_b = "tau_btp" if "tau_btp" in B.columns else tau_c

    B["step3"] = B.date >= IT_STEP3_FROM
    B["post_reform"] = B.date >= HC_REFORM
    B["hc_cct"] = [ _hc_float(s3, pr, t) for s3, pr, t in zip(B.step3, B.post_reform, B[tau_c]) ]
    B["hc_btp"] = [ _hc_fixed(t, s3) for t, s3 in zip(B[tau_b], B.step3) ]
    B["hc_diff"] = B.hc_btp - B.hc_cct                     # >0: CCT piu' efficiente come collaterale
    # in bp per allinearsi alla base
    B["hc_diff_bp"] = B.hc_diff * 100

    P(f"\ndifferenziale haircut CCT-vs-BTP (punti %): mediana {B.hc_diff.median():.2f} | "
      f"IQR [{B.hc_diff.quantile(.25):.2f}, {B.hc_diff.quantile(.75):.2f}] | max {B.hc_diff.max():.2f}")
    P("  per costruzione cresce con la scadenza e salta quando l'Italia passa a BBB (2011)")
    P(f"  osservazioni step-3 (BBB): {int(B.step3.sum()):,} | post-riforma: {int(B.post_reform.sum()):,}")

    # regime di stress per isolare la CALMA (uso il segno della base aggregata come proxy
    # o, se c'e', sov_swap dal mechanism_panel). Qui uso i terzili di anno-stress via base.
    try:
        M = pd.read_csv(PROC/"mechanism_panel.csv", parse_dates=["date"])
        M["ym"] = M.date.dt.to_period("M")
        B["ym"] = B.date.dt.to_period("M")
        sov = M.groupby("ym")["sov_swap"].mean()
        B = B.merge(sov.rename("sov_swap"), on="ym", how="left")
        lo, hi = B.sov_swap.quantile(.33), B.sov_swap.quantile(.67)
        P(f"\n  regime CALMA = terzile basso di stress sovrano (sov_swap <= {lo:.3f}%)")
        calm = B[B.sov_swap <= lo]
    except Exception:
        P("\n  [mechanism_panel assente: uso tutto il campione come proxy, meno pulito]")
        calm = B; lo = np.nan

    try:
        import statsmodels.formula.api as smf
    except ImportError:
        P("[statsmodels assente]"); save_txt("16_haircut_test.txt", L); raise SystemExit

    def run(d, f, keys, lab):
        try:
            r = smf.ols(f, data=d).fit(cov_type="cluster", cov_kwds={"groups": d["CCT_ISIN"]})
            P(f"  {lab:44s} " + "  ".join(
                f"{k}={r.params.get(k,np.nan):+.4f}[{r.tvalues.get(k,np.nan):+.2f}]" for k in keys)
              + f"   R2adj {r.rsquared_adj:.3f}  n {int(r.nobs):,}")
            return {k: (r.params.get(k, np.nan), r.tvalues.get(k, np.nan)) for k in keys}
        except Exception as e:
            P(f"  {lab:44s} fallita ({str(e)[:45]})"); return {}

    # ========================================================= C1 cross-section
    P("\n" + "=" * 76)
    P("C1 -- la ricchezza in CALMA e' proporzionale al differenziale di haircut?")
    P("  base3 in PREZZO (negativa = CCT caro). Segno atteso NEGATIVO su hc_diff:")
    P("  piu' vantaggio di haircut -> CCT piu' caro -> base in prezzo piu' negativa.")
    P("=" * 76)
    d = calm.dropna(subset=["basis3_p", "hc_diff_bp"]).copy()
    d["mon"] = d.date.dt.month.astype(str); d["yr"] = d.date.dt.year.astype(str)
    run(d, "basis3_p ~ hc_diff_bp + C(mon)", ["hc_diff_bp"], "  (1) grezza")
    run(d, "basis3_p ~ hc_diff_bp + C(yr) + C(mon)", ["hc_diff_bp"], "  (2) + effetti fissi di anno")
    # controllo: l'haircut diff e' quasi una misura della scadenza -> aggiungo tau per separare
    run(d, f"basis3_p ~ hc_diff_bp + {tau_c} + I({tau_c}**2) + C(yr) + C(mon)",
        ["hc_diff_bp"], "  (3) + scadenza (separa haircut da tenor)")
    P("  [se hc_diff resta significativo CON la scadenza dentro, non e' solo un effetto")
    P("   di maturita': e' la NON-LINEARITA' dello schedule BCE (i salti di fascia) a")
    P("   identificare, piu' il salto di rating del 2011. Se svanisce con tau, e' tenor]")

    # controllo rating: la variazione TEMPORALE (salto BBB 2011) identifica separatamente
    P("\n  [C1b] identificazione dal SALTO DI RATING (2011: A->BBB alza tutti gli haircut):")
    run(d, "basis3_p ~ hc_diff_bp + C(step3) + C(mon)", ["hc_diff_bp"],
        "  + dummy step3 (assorbe il livello del salto)")

    # ========================================================= C2 natural experiment
    P("\n" + "=" * 76)
    P("C2 -- ESPERIMENTO NATURALE: la ricchezza svanisce dopo la riforma haircut?")
    P("  la BCE equalizza floater=fisso -> il vantaggio di collaterale sparisce.")
    P("  identificazione ESOGENA allo stress sovrano.")
    P("=" * 76)
    n_post = int(B.post_reform.sum())
    if n_post < 200:
        P(f"  [!] solo {n_post} osservazioni post-riforma ({HC_REFORM.date()}): la riforma")
        P("      non e' ancora sufficientemente in vigore nel campione. C2 NON e' testabile")
        P("      ora -- va rifatto quando il campione copre >~1 anno post-riforma. Si dichiara.")
        P("      [nota: e' comunque il test da pre-registrare -- l'identificazione piu' pulita]")
    else:
        d2 = B.dropna(subset=["basis3_p"]).copy()
        d2["mon"] = d2.date.dt.month.astype(str)
        # confronto la ricchezza-per-haircut PRIMA e DOPO: interazione hc_diff x post_reform
        # ma post-riforma hc_diff->0 per costruzione, quindi testo la ricchezza CALMA pre vs post
        pre = d2[(~d2.post_reform) & (d2.sov_swap <= lo)]
        post = d2[(d2.post_reform) & (d2.sov_swap <= lo)] if "sov_swap" in d2 else d2[d2.post_reform]
        P(f"  ricchezza calma PRE-riforma:  base3 prezzo mediana {pre.basis3_p.median():+.3f} (n {len(pre):,})")
        if len(post):
            P(f"  ricchezza calma POST-riforma: base3 prezzo mediana {post.basis3_p.median():+.3f} (n {len(post):,})")
            P("  [se la ricchezza (prezzo negativo) si ATTENUA verso zero dopo la riforma,")
            P("   il canale haircut e' confermato dall'esperimento naturale]")

    save_txt("16_haircut_test.txt", L); print("\n".join(L))
